/*
    Copyright (c) 2005-2020 Intel Corporation

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

/*
Replacing the standard memory allocation routines in Microsoft* C/C++ RTL
(malloc/free, global new/delete, etc.) with the TBB memory allocator.

Include the following header to a source of any binary which is loaded during
application startup

#include "tbb/tbbmalloc_proxy.h"

or add following parameters to the linker options for the binary which is
loaded during application startup. It can be either exe-file or dll.

For win32
tbbmalloc_proxy.lib /INCLUDE:"___TBB_malloc_proxy"
win64
tbbmalloc_proxy.lib /INCLUDE:"__TBB_malloc_proxy"
*/

#ifndef __TBB_tbbmalloc_proxy_H
#define __TBB_tbbmalloc_proxy_H

#if _MSC_VER

#ifdef _DEBUG
    #pragma comment(lib, "tbbmalloc_proxy_debug.lib")
#else
    #pragma comment(lib, "tbbmalloc_proxy.lib")
#endif

#if defined(_WIN64)
    #pragma comment(linker, "/include:__TBB_malloc_proxy")
#else
    #pragma comment(linker, "/include:___TBB_malloc_proxy")
#endif

#else
/* Primarily to support MinGW */

extern "C" void __TBB_malloc_proxy();
struct __TBB_malloc_proxy_caller {
    __TBB_malloc_proxy_caller() { __TBB_malloc_proxy(); }
} volatile __TBB_malloc_proxy_helper_object;

#endif // _MSC_VER

/* Public Windows API */
extern "C" int TBB_malloc_replacement_log(char *** function_replacement_log_ptr);

#endif //__TBB_tbbmalloc_proxy_H
