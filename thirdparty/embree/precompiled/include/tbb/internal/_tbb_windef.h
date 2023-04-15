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

#ifndef __TBB_tbb_windef_H
#error Do not #include this internal file directly; use public TBB headers instead.
#endif /* __TBB_tbb_windef_H */

// Check that the target Windows version has all API calls required for TBB.
// Do not increase the version in condition beyond 0x0500 without prior discussion!
#if defined(_WIN32_WINNT) && _WIN32_WINNT<0x0501
#error TBB is unable to run on old Windows versions; _WIN32_WINNT must be 0x0501 or greater.
#endif

#if !defined(_MT)
#error TBB requires linkage with multithreaded C/C++ runtime library. \
       Choose multithreaded DLL runtime in project settings, or use /MD[d] compiler switch.
#endif

// Workaround for the problem with MVSC headers failing to define namespace std
namespace std {
  using ::size_t; using ::ptrdiff_t;
}

#define __TBB_STRING_AUX(x) #x
#define __TBB_STRING(x) __TBB_STRING_AUX(x)

// Default setting of TBB_USE_DEBUG
#ifdef TBB_USE_DEBUG
#    if TBB_USE_DEBUG
#        if !defined(_DEBUG)
#            pragma message(__FILE__ "(" __TBB_STRING(__LINE__) ") : Warning: Recommend using /MDd if compiling with TBB_USE_DEBUG!=0")
#        endif
#    else
#        if defined(_DEBUG)
#            pragma message(__FILE__ "(" __TBB_STRING(__LINE__) ") : Warning: Recommend using /MD if compiling with TBB_USE_DEBUG==0")
#        endif
#    endif
#endif

#if (__TBB_BUILD || __TBBMALLOC_BUILD || __TBBBIND_BUILD) && !defined(__TBB_NO_IMPLICIT_LINKAGE)
#define __TBB_NO_IMPLICIT_LINKAGE 1
#endif

#if _MSC_VER
    #if !__TBB_NO_IMPLICIT_LINKAGE
        #ifdef __TBB_LIB_NAME
	        #pragma comment(lib, __TBB_STRING(__TBB_LIB_NAME))
        #else
			#ifdef _DEBUG
				#pragma comment(lib, "tbb_debug.lib")
			#else
				#pragma comment(lib, "tbb.lib")
			#endif
        #endif
    #endif
#endif
