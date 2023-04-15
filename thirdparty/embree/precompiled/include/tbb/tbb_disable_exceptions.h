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

//! To disable use of exceptions, include this header before any other header file from the library.

//! The macro that prevents use of exceptions in the library files
#undef  TBB_USE_EXCEPTIONS
#define TBB_USE_EXCEPTIONS 0

//! Prevent compilers from issuing exception related warnings.
/** Note that the warnings are suppressed for all the code after this header is included. */
#if _MSC_VER
#if __INTEL_COMPILER
    #pragma warning (disable: 583)
#else
    #pragma warning (disable: 4530 4577)
#endif
#endif
