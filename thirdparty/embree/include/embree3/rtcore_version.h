// ======================================================================== //
// Copyright 2009-2018 Intel Corporation                                    //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#define RTC_VERSION_MAJOR 3
#define RTC_VERSION_MINOR 5
#define RTC_VERSION_PATCH 2
#define RTC_VERSION 30502
#define RTC_VERSION_STRING "3.5.2"

/* #undef EMBREE_STATIC_LIB */
/* #undef EMBREE_API_NAMESPACE */

#if defined(EMBREE_API_NAMESPACE)
#  define RTC_NAMESPACE 
#  define RTC_NAMESPACE_BEGIN namespace  {
#  define RTC_NAMESPACE_END }
#  define RTC_NAMESPACE_OPEN using namespace ;
#  define RTC_API_EXTERN_C
#  undef EMBREE_API_NAMESPACE
#else
#  define RTC_NAMESPACE_BEGIN
#  define RTC_NAMESPACE_END
#  define RTC_NAMESPACE_OPEN
#  if defined(__cplusplus)
#    define RTC_API_EXTERN_C extern "C"
#  else
#    define RTC_API_EXTERN_C
#  endif
#endif

#if defined(ISPC)
#  define RTC_API_IMPORT extern "C" unmasked
#  define RTC_API_EXPORT extern "C" unmasked
#elif defined(EMBREE_STATIC_LIB)
#  define RTC_API_IMPORT RTC_API_EXTERN_C
#  define RTC_API_EXPORT RTC_API_EXTERN_C
#elif defined(_WIN32)
#  define RTC_API_IMPORT RTC_API_EXTERN_C __declspec(dllimport)
#  define RTC_API_EXPORT RTC_API_EXTERN_C __declspec(dllexport)
#else
#  define RTC_API_IMPORT RTC_API_EXTERN_C
#  define RTC_API_EXPORT RTC_API_EXTERN_C __attribute__ ((visibility ("default")))
#endif

#if defined(RTC_EXPORT_API)
#  define RTC_API RTC_API_EXPORT
#else
#  define RTC_API RTC_API_IMPORT
#endif
