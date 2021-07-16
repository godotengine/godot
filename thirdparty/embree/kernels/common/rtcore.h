// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../../include/embree3/rtcore.h"
RTC_NAMESPACE_USE

namespace embree
{  
  /*! decoding of intersection flags */
  __forceinline bool isCoherent  (RTCIntersectContextFlags flags) { return (flags & RTC_INTERSECT_CONTEXT_FLAG_COHERENT) == RTC_INTERSECT_CONTEXT_FLAG_COHERENT; }
  __forceinline bool isIncoherent(RTCIntersectContextFlags flags) { return (flags & RTC_INTERSECT_CONTEXT_FLAG_COHERENT) == RTC_INTERSECT_CONTEXT_FLAG_INCOHERENT; }

#if defined(TASKING_TBB) && (TBB_INTERFACE_VERSION_MAJOR >= 8)
#  define USE_TASK_ARENA 1
#else
#  define USE_TASK_ARENA 0
#endif

#if defined(TASKING_TBB) && (TBB_INTERFACE_VERSION >= 11009) // TBB 2019 Update 9
#  define TASKING_TBB_USE_TASK_ISOLATION 1
#else
#  define TASKING_TBB_USE_TASK_ISOLATION 0
#endif

/*! Macros used in the rtcore API implementation */
// -- GODOT start --
// #define RTC_CATCH_BEGIN try {
#define RTC_CATCH_BEGIN
  
// #define RTC_CATCH_END(device)                                                \
//   } catch (std::bad_alloc&) {                                                   \
//     Device::process_error(device,RTC_ERROR_OUT_OF_MEMORY,"out of memory");      \
//   } catch (rtcore_error& e) {                                                   \
//     Device::process_error(device,e.error,e.what());                             \
//   } catch (std::exception& e) {                                                 \
//     Device::process_error(device,RTC_ERROR_UNKNOWN,e.what());                   \
//   } catch (...) {                                                               \
//     Device::process_error(device,RTC_ERROR_UNKNOWN,"unknown exception caught"); \
//   }
#define RTC_CATCH_END(device)
  
// #define RTC_CATCH_END2(scene)                                                \
//   } catch (std::bad_alloc&) {                                                   \
//     Device* device = scene ? scene->device : nullptr;                           \
//     Device::process_error(device,RTC_ERROR_OUT_OF_MEMORY,"out of memory");      \
//   } catch (rtcore_error& e) {                                                   \
//     Device* device = scene ? scene->device : nullptr;                           \
//     Device::process_error(device,e.error,e.what());                             \
//   } catch (std::exception& e) {                                                 \
//     Device* device = scene ? scene->device : nullptr;                           \
//     Device::process_error(device,RTC_ERROR_UNKNOWN,e.what());                   \
//   } catch (...) {                                                               \
//     Device* device = scene ? scene->device : nullptr;                           \
//     Device::process_error(device,RTC_ERROR_UNKNOWN,"unknown exception caught"); \
//   }
#define RTC_CATCH_END2(scene)

// #define RTC_CATCH_END2_FALSE(scene)                                             \
//   } catch (std::bad_alloc&) {                                                   \
//     Device* device = scene ? scene->device : nullptr;                           \
//     Device::process_error(device,RTC_ERROR_OUT_OF_MEMORY,"out of memory");      \
//     return false;                                                               \
//   } catch (rtcore_error& e) {                                                   \
//     Device* device = scene ? scene->device : nullptr;                           \
//     Device::process_error(device,e.error,e.what());                             \
//     return false;                                                               \
//   } catch (std::exception& e) {                                                 \
//     Device* device = scene ? scene->device : nullptr;                           \
//     Device::process_error(device,RTC_ERROR_UNKNOWN,e.what());                   \
//     return false;                                                               \
//   } catch (...) {                                                               \
//     Device* device = scene ? scene->device : nullptr;                           \
//     Device::process_error(device,RTC_ERROR_UNKNOWN,"unknown exception caught"); \
//     return false;                                                               \
//   }
#define RTC_CATCH_END2_FALSE(scene) return false;
// -- GODOT end --

#define RTC_VERIFY_HANDLE(handle)                               \
  if (handle == nullptr) {                                         \
    throw_RTCError(RTC_ERROR_INVALID_ARGUMENT,"invalid argument"); \
  }

#define RTC_VERIFY_GEOMID(id)                                   \
  if (id == RTC_INVALID_GEOMETRY_ID) {                             \
    throw_RTCError(RTC_ERROR_INVALID_ARGUMENT,"invalid argument"); \
  }

#define RTC_VERIFY_UPPER(id,upper)                              \
  if (id > upper) {                                                \
    throw_RTCError(RTC_ERROR_INVALID_ARGUMENT,"invalid argument"); \
  }

#define RTC_VERIFY_RANGE(id,lower,upper)	\
  if (id < lower || id > upper)						  \
    throw_RTCError(RTC_ERROR_INVALID_OPERATION,"argument out of bounds");
  
#if 0 // enable to debug print all API calls
#define RTC_TRACE(x) std::cout << #x << std::endl;
#else
#define RTC_TRACE(x) 
#endif

// -- GODOT begin --
//   /*! used to throw embree API errors */
//   struct rtcore_error : public std::exception
//   {
//     __forceinline rtcore_error(RTCError error, const std::string& str)
//       : error(error), str(str) {}
//     
//     ~rtcore_error() throw() {}
//     
//     const char* what () const throw () {
//       return str.c_str();
//     }
//     
//     RTCError error;
//     std::string str;
//   };
// -- GODOT end --

#if defined(DEBUG) // only report file and line in debug mode
  // -- GODOT begin --
  // #define throw_RTCError(error,str) \
  //   throw rtcore_error(error,std::string(__FILE__) + " (" + toString(__LINE__) + "): " + std::string(str));
  #define throw_RTCError(error,str) \
    printf(std::string(__FILE__) + " (" + toString(__LINE__) + "): " + std::string(str)), abort();
  // -- GODOT end --
#else
  // -- GODOT begin --
  // #define throw_RTCError(error,str) \
  //   throw rtcore_error(error,str);
  #define throw_RTCError(error,str) \
    abort();
  // -- GODOT end --
#endif

#define RTC_BUILD_ARGUMENTS_HAS(settings,member) \
  (settings.byteSize > (offsetof(RTCBuildArguments,member)+sizeof(settings.member))) 
}
