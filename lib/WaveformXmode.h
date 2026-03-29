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

#pragma once

#if defined(_WIN32)
    #ifdef WaveformXmode_EXPORTS
        #define DPCPP_DLL_API __declspec(dllexport)
    #else
        #define DPCPP_DLL_API __declspec(dllimport)
    #endif
#else
    #define DPCPP_DLL_API __attribute__ ((visibility ("default")))
#endif

#include <sycl/sycl.hpp>
#include <vector>
#include <string>
#include <iostream>

using namespace sycl;


//************************************
// Initialize an object.
//************************************
extern "C" DPCPP_DLL_API void* init();

//************************************
// Input pattern function. 
// Read pattern by line until finding `end_key`. 
// Read process stop when finding `stop_key`.
// `p` is the object from `init()` function.
//************************************
extern "C" DPCPP_DLL_API void pattern_byline(void* p, const char* pattern, const char* end_key, const char* stop_key);

//************************************
// Set the number of characters for the waveform.
// `num_list` used to choose the column by number. Example: "1,3,4".
// `xmode` can be set from `2` to `16`. If not, an exception is thrown.
//************************************
extern "C" DPCPP_DLL_API void xmode(void* p, const char* num_list, int xmode);

//************************************
// Execute the computation.
//************************************
extern "C" DPCPP_DLL_API void execute(void* p);

//************************************
// Execute the calculation in parallel mode. (Overlap data transfer and computation)
// Note: Might take more RAM. 
//************************************
extern "C" DPCPP_DLL_API void execute_p(void* p);

//************************************
// Get the used waveforms. `num` is the number of columns in pattern.
//************************************
extern "C" DPCPP_DLL_API char* get_used(void* p, int num);

//************************************
// Get the patterns by column numbers. `num_list` is column numbers. Example: "1,3,4".
//************************************
extern "C" DPCPP_DLL_API char* get_pattern(void* p, const char* num_list);

//************************************
// Recycle to prevent the memory leak. Need user to send it back manually.
// `p` is the object from `init()` function.
//************************************
extern "C" DPCPP_DLL_API void end(void* p);

//************************************
// Recycle the output characters to prevent the memory leak.
// Need user to send the char* back manually.
//************************************
extern "C" DPCPP_DLL_API void recycle(char* c);
