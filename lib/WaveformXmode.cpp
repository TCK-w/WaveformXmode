//==============================================================
// A pattern is composed of a series of characters. 
// We defines a waveform as consisting of a 
// sequence of 2 to 16 characters. `WaveformXmode` 
// can find all the waveform combinations needed to 
// represent a pattern and offload the search 
// operation to the GPU.
// 
// Env:
// Intel oneAPI Base Toolkit Version 2025.0.1.47_offline
// Intel Graphics Driver 32.0.101.6647 (WHQL Certified)
// 
// Author: TCK
//
// SPDX-License-Identifier: MIT
// =============================================================

#include "WaveformXmode.h"
#include<math.h>
#include<string.h>
#include<set>


// int prefer_my_device(const sycl::device& d) {
//     return d.get_info<info::device::name>() == "Intel(R) UHD Graphics 770";
// }

class WaveformXmode {
private:
    bool init = false;
    bool safe = false;
    std::vector<std::vector<char>*> wave;
    std::vector<int> xmode;
    std::vector<std::vector<char>*> res;
    void wave_search(queue& q, std::vector<char>& wave, int x, std::vector<char>& re);
public:
    WaveformXmode(std::vector<std::vector<char>*> wave);
    WaveformXmode();
    ~WaveformXmode();
    void set_w(std::vector<std::vector<char>*>& wave);
    void set_xmode(const std::vector<int>& num_list, int xmode);
    void execute();
    std::vector<std::string> get_result(int num);
    std::string get_w(const std::vector<int>& num_list);
};

WaveformXmode::WaveformXmode(std::vector<std::vector<char>*> wave) : xmode(wave.size(), 0), res(wave.size(), nullptr) {
    this->wave = wave;
    this->init = true;
}
WaveformXmode::WaveformXmode() {}
WaveformXmode::~WaveformXmode() {
    for (std::vector<std::vector<char>*>::iterator it = this->wave.begin();it != this->wave.end();it++) {
        std::vector<char>* tmp = *it;
        *it = nullptr;
        delete tmp;
    }
    for (std::vector<std::vector<char>*>::iterator it = this->res.begin();it != this->res.end();it++) {
        std::vector<char>* tmp = *it;
        *it = nullptr;
        delete tmp;
    }
}

void WaveformXmode::set_w(std::vector<std::vector<char>*>& wave) {
    if (this->wave.empty()) {
        this->wave = wave;
        std::vector<int> xmode(wave.size(), 0);
        std::vector<std::vector<char>*> res(wave.size(), nullptr);
        this->xmode = xmode;
        this->res = res;
        this->init = true;
        wave.clear();
    }
    else {
        if (this->wave.size() != wave.size()) {
            std::cerr << "set_w Err: The size of additional input wave not equal to original wave.\n";
            terminate();
        }
        for (int i = 0;i < this->wave.size();i++) {
            std::vector<char>* tmp = wave.at(i);
            this->wave.at(i)->insert(this->wave.at(i)->end(), tmp->begin(), tmp->end());
            wave.at(i) = nullptr;
            delete tmp;
        }
        wave.clear();
        this->safe = false;
    }
}

void WaveformXmode::set_xmode(const std::vector<int>& num_list, int xmode) {
    if (!this->init) {
        std::cerr << "set_xmode Err: No initial data for this object. Initialize it using method set_w().\n";
        terminate();
    }
    if (xmode < 2 || xmode > 16) {
        std::cerr << "set_xmode Err: Unacceptable xmode number. Please set 1<xmode<17.\n";
        terminate();
    }

    for (std::vector<int>::const_iterator it = num_list.begin();it != num_list.end();it++) {
        if (*it < 0 || *it >= this->xmode.size()) {
            std::cerr << "set_xmode Err: Numbers in num_list should smaller than pattern size.\n";
            terminate();
        }
        this->xmode.at(*it) = xmode;
    }
    this->safe = false;
}

void WaveformXmode::wave_search(queue& q, std::vector<char>& wave, int x, std::vector<char>& re) {
    unsigned int local_s = 256 * x;
    constexpr unsigned int thrd = 64;
    constexpr unsigned int arr_s = 16;
    unsigned int w_s = (wave.size() / x) * x;
    unsigned int re_s = re.size();
    unsigned int w_com = (w_s / (arr_s * x)) * (arr_s * x);
    unsigned int w_rest = w_s % (arr_s * x);

    char* w_host = wave.data();
    char* re_host = re.data();

    char* w_dev = malloc_device<char>(w_s, q);
    char* buf_dev = malloc_device<char>(local_s * thrd, q);
    char* re_dev = malloc_device<char>(re_s, q);
    unsigned int* len_dev = malloc_device<unsigned int>(thrd, q);
    q.memcpy(w_dev, w_host, sizeof(char) * w_s).wait();

    q.submit([&](handler& h) {

        local_accessor<char, 1> local_re(sycl::range(local_s), h);
        //sycl::stream out(65536, 256, h);
        h.parallel_for(sycl::nd_range(sycl::range{ x * thrd }, sycl::range{ x }), [=](sycl::nd_item<1> it) {
            unsigned int groupId = it.get_group(0);
            unsigned int globalId = it.get_global_linear_id();
            auto sg = it.get_sub_group();
            unsigned int sgSize = sg.get_local_range()[0];
            unsigned int sgGroupId = sg.get_group_id()[0];
            unsigned int sgId = sg.get_local_id()[0];

            unsigned int local_idx = 0;
            char arr[arr_s];
            bool check_tot;
            bool check_once;
            bool check_cur;
            for (unsigned int i = groupId * x * arr_s;i < w_com;i += thrd * x * arr_s) {
#pragma unroll
                for (unsigned int j = 0; j < arr_s; j++) {
                    arr[j] = w_dev[i + sgId + j * x];
                }
                it.barrier(sycl::access::fence_space::local_space);
#pragma unroll
                for (unsigned int j = 0; j < arr_s; j++) {
                    check_tot = false;
                    char cur = arr[j];
                    for (unsigned int k = sgId;k < local_idx;k += x) {
                        check_once = true;
                        check_cur = (local_re[k] == cur);
                        it.barrier(sycl::access::fence_space::local_space);
                        for (size_t ki = 0;ki < x;ki++) {
                            check_once = check_once & sycl::select_from_group(sg, check_cur, ki);
                        }
                        check_tot = check_tot | check_once;
                        if (check_tot) {
                            break;
                        }
                    }
                    if (!check_tot) {
                        local_re[local_idx + sgId] = cur;
                        local_idx += x;
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                }
            }
            for (unsigned int k = sgId;k < local_idx;k += x) {
                buf_dev[groupId * local_s + k] = local_re[k];
            }
            for (unsigned int k = local_idx + sgId;k < local_s;k += x) {
                buf_dev[groupId * local_s + k] = 0;
            }
            if (sgId == 0) {
                len_dev[groupId] = local_idx;
            }
            it.barrier(sycl::access::fence_space::local_space);

            //out << "globalId = " << sycl::setw(2) << globalId
            //    << " groupId = " << groupId
            //    << " sgGroupId = " << sgGroupId << " sgId = " << sgId
            //    << " sgSize = " << sycl::setw(2) << sgSize
            //    << sycl::endl;
            });

        });
    q.wait();

    q.submit([&](handler& h) {

        local_accessor<char, 1> local_re(sycl::range(local_s), h);
        //sycl::stream out(65536, 256, h);
        h.parallel_for(sycl::nd_range(sycl::range{ x }, sycl::range{ x }), [=](sycl::nd_item<1> it) {
            unsigned int groupId = it.get_group(0);
            unsigned int globalId = it.get_global_linear_id();
            auto sg = it.get_sub_group();
            unsigned int sgSize = sg.get_local_range()[0];
            unsigned int sgGroupId = sg.get_group_id()[0];
            unsigned int sgId = sg.get_local_id()[0];

            unsigned int local_idx = len_dev[0];
            for (unsigned int j = 0; j < local_s; j+=x) {
                local_re[sgId + j] = buf_dev[sgId + j];
            }
            it.barrier(sycl::access::fence_space::local_space);

            bool check_tot;
            bool check_once;
            bool check_cur;
#pragma unroll
            for (unsigned int i = 0; i < thrd; i++) {
                if (i == 0) {
                    for (unsigned int j = 0; j < w_rest; j+=x) {
                        check_tot = false;
                        char cur = w_dev[w_com + sgId + j];
                        for (unsigned int k = sgId;k < local_idx;k += x) {
                            check_once = true;
                            check_cur = (local_re[k] == cur);
                            it.barrier(sycl::access::fence_space::local_space);
                            for (size_t ki = 0;ki < x;ki++) {
                                check_once = check_once & sycl::select_from_group(sg, check_cur, ki);
                            }
                            check_tot = check_tot | check_once;
                            if (check_tot) {
                                break;
                            }
                        }
                        if (!check_tot) {
                            local_re[local_idx + sgId] = cur;
                            local_idx += x;
                        }
                        it.barrier(sycl::access::fence_space::local_space);
                    }
                }
                else {
                    for (unsigned int j = 0; j < len_dev[i]; j+=x) {
                        check_tot = false;
                        char cur = buf_dev[local_s * i + sgId + j];
                        for (unsigned int k = sgId;k < local_idx;k += x) {
                            check_once = true;
                            check_cur = (local_re[k] == cur);
                            it.barrier(sycl::access::fence_space::local_space);
                            for (size_t ki = 0;ki < x;ki++) {
                                check_once = check_once & sycl::select_from_group(sg, check_cur, ki);
                            }
                            check_tot = check_tot | check_once;
                            if (check_tot) {
                                break;
                            }
                        }
                        if (!check_tot) {
                            local_re[local_idx + sgId] = cur;
                            local_idx += x;
                        }
                        it.barrier(sycl::access::fence_space::local_space);
                    }
                }
            }

            for (unsigned int j = 0; j < local_s ; j+=x) {
                re_dev[sgId + j] = local_re[sgId + j];
            }
            it.barrier(sycl::access::fence_space::local_space);

            //out << "globalId = " << sycl::setw(2) << globalId
            //    << " groupId = " << groupId
            //    << " sgGroupId = " << sgGroupId << " sgId = " << sgId
            //    << " sgSize = " << sycl::setw(2) << sgSize
            //    << sycl::endl;
            });

        });
    q.wait();
    
    q.memcpy(re_host, re_dev, sizeof(char) * re_s).wait();

    free(w_dev, q);
    free(buf_dev, q);
    free(len_dev, q);
    free(re_dev, q);
}

void WaveformXmode::execute() {
    static auto exception_handler = [](sycl::exception_list e_list) {
        for (std::exception_ptr const& e : e_list) {
            try {
                rethrow_exception(e);
            }
            catch (std::exception const& e) {
                std::cerr << "Throw Failure\n";
                terminate();
            }
        }
    };

    if (!this->init) {
        std::cerr << "No initial data for this object. Initialize it using method set_w().\n";
        terminate();
    }

    //sycl::device preferred_device{ prefer_my_device };
    auto preferred_device = default_selector_v;
    queue q(preferred_device, exception_handler);
    // Print out the device information used for the kernel code.
    std::cout << "Running on device: "
        << q.get_device().get_info<info::device::name>() << "\n";

    // execute sort
    try {
        int i = 0;
        for (std::vector<int>::iterator it = this->xmode.begin(); it != this->xmode.end(); it++) {
            if (*it != 0) {
                if (this->res.at(i) != nullptr) {
                    std::vector<char>* tmp = this->res.at(i);
                    this->res.at(i) = nullptr;
                    delete tmp;
                }
                std::vector<char>* re = new std::vector<char>(256 * (*it), 0);
                this->res.at(i) = re;
                wave_search(q, *(this->wave.at(i)), *it, *re);
            }
            i++;
        }
        q.wait();
    }
    catch (std::exception const& e) {
        std::cerr << "An exception is caught for execute.\n";
        std::cerr << "exception caught: " << e.what() << '\n';
        terminate();
    }

    this->safe = true;
}

std::vector<std::string> WaveformXmode::get_result(int num) {
    if (!this->init) {
        std::cerr << "get_result Err: No initial data for this object. Initialize it using method set_w().\n";
        terminate();
    }
    if (!this->safe) {
        std::cerr << "get_result Err: Method 'get_result' only avalible after execute().\n";
        terminate();
    }
    if (num < 0 || num >= this->res.size()) {
        std::cerr << "get_result Err: Variable 'num' should smaller than pattern size.\n";
        terminate();
    }
    std::vector<std::string> res;
    std::vector<char>* res_i = this->res.at(num);
    if (res_i == nullptr) { return res; }

    size_t xmode_i = this->xmode.at(num);
    std::string tmp;
    for (std::vector<char>::iterator it = res_i->begin();it != res_i->end();it++) {
        if (*it == 0) {
            break;
        }
        tmp.push_back(*it);
        if (tmp.size() == xmode_i) {
            res.push_back(tmp);
            tmp.clear();
        }
    }

    return res;
}

std::string WaveformXmode::get_w(const std::vector<int>& num_list) {
    if (!this->init) {
        std::cerr << "get_w Err: No initial data for this object. Initialize it using method set_w().\n";
        terminate();
    }
    for (std::vector<int>::const_iterator it = num_list.begin();it != num_list.end();it++) {
        if (*it < 0 || *it >= this->xmode.size()) {
            std::cerr << "get_w Err: Numbers in num_list should smaller than pattern size.\n";
            terminate();
        }
    }
    std::string re;
    for (size_t s = 0; s < this->wave.at(num_list.at(0))->size();s++) {
        for (std::vector<int>::const_iterator it = num_list.begin();it != num_list.end();it++) {
            re.push_back(this->wave.at(*it)->at(s));
        }
        re.push_back('\n');
    }
    return re;
}


void* init() {
    return new WaveformXmode;
}

void pattern_byline(void* p, const char* pattern, const char* end_key, const char* stop_key) {
    std::string pattern_s(pattern);
    std::vector<std::vector<char>*> pat;
    // remove words behind stop_key
    size_t stop_i = pattern_s.rfind(stop_key);
    while (stop_i != std::string::npos) {
        pattern_s.erase(stop_i, pattern_s.size() - stop_i);
        stop_i = pattern_s.rfind(stop_key);
    }
    // get first line pattern
    size_t start = 0;
    size_t end = pattern_s.find(end_key);
    if (end == std::string::npos) {
        std::cerr << "Pattern first line not found!\n";
        terminate();
    }
    for (size_t idx = start;idx < end;idx++) {
        std::vector<char>* tmp = new std::vector<char>;
        tmp->push_back(pattern_s.at(idx));
        pat.push_back(tmp);
    }
    start = pattern_s.find('\n', end);
    if (start != std::string::npos) {
        end = pattern_s.find(end_key, start);
    }
    while (start != std::string::npos && end != std::string::npos) {
        if (end - start - 1 != pat.size()) {
            std::cerr << "Pattern number isn't a constant. Some of the line sizes do not equal the first line!\n";
            terminate();
        }
        for (size_t idx = start + 1;idx < end;idx++) {
            pat.at(idx - start - 1)->push_back(pattern_s.at(idx));
        }
        start = pattern_s.find('\n', end);
        if (start != std::string::npos) {
            end = pattern_s.find(end_key, start);
        }
    }
    try {
        ((WaveformXmode*)p)->set_w(pat);
    }
    catch (std::exception const& e) {
        std::cerr << "An exception is caught for set_w.\n";
        std::cerr << "exception caught: " << e.what() << '\n';
        terminate();
    }
}

void xmode(void* p, const char* num_list, int xmode) {
    // Copy
    size_t cs = strlen(num_list);
    char* buf = new char[cs+1];
    strcpy(buf, num_list);

    std::vector<int> num_v;
    char* n;
    n = strtok(buf, ",");
    while (n != NULL) {
        num_v.push_back(std::stoi(n));
        n = strtok(NULL, ",");
    }

    try {
        ((WaveformXmode*)p)->set_xmode(num_v, xmode);
    }
    catch (std::exception const& e) {
        std::cerr << "An exception is caught for set_xmode.\n";
        std::cerr << "exception caught: " << e.what() << '\n';
        terminate();
    }
}

void execute(void* p) {
    try {
        ((WaveformXmode*)p)->execute();
    }
    catch (std::exception const& e) {
        std::cerr << "An exception is caught for execute.\n";
        std::cerr << "exception caught: " << e.what() << '\n';
        terminate();
    }
}

char* get_used(void* p, int num) {
    std::vector<std::string> vs;
    try {
        vs = ((WaveformXmode*)p)->get_result(num);
    }
    catch (std::exception const& e) {
        std::cerr << "An exception is caught for get_result.\n";
        std::cerr << "exception caught: " << e.what() << '\n';
        terminate();
    }
    if (vs.size() < 1) {
        char* c = new char;
        return c;
    }
    char* c = new char[vs.size() * (vs.at(0).size() + 1) + 1];
    int i = 0;
    for (std::vector<std::string>::iterator it = vs.begin(); it != vs.end(); it++) {
        for (std::string::iterator its = (*it).begin();its != (*it).end();its++) {
            c[i] = *its;
            i++;
        }
        c[i] = ',';
        i++;
    }
    c[i - 1] = '\0';
    return c;
}

char* get_pattern(void* p, const char* num_list) {
    size_t cs = strlen(num_list);
    char* buf = new char[cs+1];
    strcpy(buf, num_list);

    std::vector<int> num_v;
    char* n;
    n = strtok(buf, ",");
    while (n != NULL) {
        num_v.push_back(std::stoi(n));
        n = strtok(NULL, ",");
    }

    std::string s;
    try {
        s = ((WaveformXmode*)p)->get_w(num_v);
    }
    catch (std::exception const& e) {
        std::cerr << "An exception is caught for get_w.\n";
        std::cerr << "exception caught: " << e.what() << '\n';
        terminate();
    }
    char* re = new char[s.size() + 1];
    int i = 0;
    for (std::string::iterator it = s.begin(); it != s.end(); it++) {
        re[i] = *it;
        i++;
    }
    re[s.size()] = '\0';

    return re;
}

void end(void* p) {
    delete static_cast<WaveformXmode*>(p);
}

void recycle(char* c) {
    delete c;
}