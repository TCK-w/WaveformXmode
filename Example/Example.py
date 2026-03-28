"""
ENV:
    Python 3.11.9
"""

import time
import ctypes

dllpath = "./../build/WaveformXmode.dll"
dll = ctypes.CDLL(dllpath, winmode=0)
dll.init.restype = ctypes.c_void_p
dll.pattern_byline.argtypes = [ctypes.c_void_p,ctypes.c_char_p,ctypes.c_char_p,ctypes.c_char_p]
dll.xmode.argtypes = [ctypes.c_void_p,ctypes.c_char_p,ctypes.c_int]
dll.execute.argtypes = [ctypes.c_void_p]
dll.get_used.argtypes = [ctypes.c_void_p,ctypes.c_int]
dll.get_used.restype = ctypes.c_void_p
dll.get_pattern.argtypes = [ctypes.c_void_p,ctypes.c_char_p]
dll.get_pattern.restype = ctypes.c_void_p
dll.end.argtypes = [ctypes.c_void_p]
dll.recycle.argtypes = [ctypes.c_void_p]

f = open("Input_example.txt","r")
pat = f.read()
f.close()
stt = time.time()

try:
    p = dll.init()
    dll.pattern_byline(p, pat.encode("utf-8"), "#".encode("utf-8"), "END".encode("utf-8"))
    dll.xmode(p, "0,2,4,6,8,10,12,14,16,18".encode("utf-8"),4)
    dll.xmode(p, "1,3,5,7,9,11,13,15,17,19".encode("utf-8"),3)
    dll.execute(p)
except Exception as e:
    print("execute error:",e)

try:
    for i in range(0,20):
        use_c = dll.get_used(p,i)
        use = ctypes.string_at(use_c).decode("utf-8")
        print("Column",i,", Used:",use)
        dll.recycle(use_c)
except Exception as e:
    print("get used waveform error:",e)

try:
    tst_c = dll.get_pattern(p,"1,2".encode("utf-8"))
    tst = ctypes.string_at(tst_c).decode("utf-8")
    dll.recycle(tst_c)
    print("Print pat:")
    print(tst)
except Exception as e:
    print("get_pattern error:",e)

try:
    dll.end(p)
except Exception as e:
    print("end error:",e)

del dll
print("Time Cost:", time.time()-stt, "s.")