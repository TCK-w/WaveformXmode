# WaveformXmode

## Introduction
A pattern is composed of a series of characters. We defines a waveform as consisting of a sequence of 2 to 16 characters. `WaveformXmode` can find all the waveform combinations needed to represent a pattern and offload the search operation to the GPU.

## Prerequisites
| Requirement      | Description
|---               |---
| OS               | Windows 
| Hardware         | iGPU for Intel Gen9 CPU or newer <br> Intel dGPU
| Software         | Intel® oneAPI Base Toolkit

> Has been verified on :
> * Windows 11
> * iGPU for Intel Gen12 CPU 
> * Intel® oneAPI Base Toolkit 2025.0.1 
> * Intel Graphics Driver 32.0.101.6647 (WHQL Certified)
> * Python 3.11.9

## How it work
This example has two patterns, each consisting of 12 characters. Notice, `#` is the end-of-line key.
[Example of supported pattern file format](Example/Input_example.txt)
```
00#
10#
00#
00#
00#
00#
00#
00#
01#
01#
01#
01#
```
If a waveform is set to consist of 4 characters, the search results are:
* Column0 used: `0100`,`0000`
* Column1 used: `0000`,`1111`
> [!IMPORTANT]
> The numbers of waveform combination cannot be greater than 256.

## How to use
### Code Description
* `void* init();`

  - Initialize an object.
  
* `void pattern_byline(void* p, const char* pattern, const char* end_key, const char* stop_key);`
  
  - Read pattern by line until finding `end_key`. 

  - Read process stop when finding `stop_key`. 

  - `p` is the object from `init()` function.

* `void xmode(void* p, const char* num_list, int xmode);`

  - Set the number of characters for the waveform.
  
  - `num_list` used to choose the column by number. Example: "1,3,4".

  - `xmode` can be set from `2` to `8`. If not, an exception is thrown.

* `void execute(void* p);`

  - Execute the computation.
  
  
* `char* get_used(void* p, int num);`

  - Get the used waveforms. `num` is the number of columns in pattern.

* `char* get_pattern(void* p, const char* num_list);`

  - Get the patterns by column numbers. `num_list` is column numbers. Example: "1,3,4".

* `void end(void* p);`

  - Recycle to prevent the memory leak. Need user to send it back manually.

  - `p` is the object from `init()` function.

* `void recycle(char* c);`

  - Recycle the output characters to prevent the memory leak. Need user to send it back manually.

### Example
[Use this Dynamic Link Library in Python.](Example/Example.py)

Build the cmake project, and then run the example in Windows PowerShell
```
cd Example
$env:ONEAPI_DEVICE_SELECTOR="*:gpu"
python Example.py
```