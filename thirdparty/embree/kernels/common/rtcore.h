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

#pragma once

#include "../../include/embree3/rtcore.h"
RTC_NAMESPACE_OPEN

namespace embree
{
/*! maximum number of user vertex buffers */
#define RTC_MAX_USER_VERTEX_BUFFERS 65536

/*! maximum number of index buffers for subdivision surfaces */
#define RTC_MAX_INDEX_BUFFERS 65536

  /*! decoding of intersection flags */
  __forceinline bool isCoherent  (RTCIntersectContextFlags flags) { return (flags & RTC_INTERSECT_CONTEXT_FLAG_COHERENT) == RTC_INTERSECT_CONTEXT_FLAG_COHERENT; }
  __forceinline bool isIncoherent(RTCIntersectContextFlags flags) { return (flags & RTC_INTERSECT_CONTEXT_FLAG_COHERENT) == RTC_INTERSECT_CONTEXT_FLAG_INCOHERENT; }

#if defined(TASKING_TBB) && (TBB_INTERFACE_VERSION_MAJOR >= 8)
#  define USE_TASK_ARENA 1
#else
#  define USE_TASK_ARENA 0
#endif

/*! Macros used in the rtcore API implementation */
#define RTC_CATCH_BEGIN try {
  
#define RTC_CATCH_END(device)                                                \
  } catch (std::bad_alloc&) {                                                   \
    Device::process_error(device,RTC_ERROR_OUT_OF_MEMORY,"out of memory");      \
  } catch (rtcore_error& e) {                                                   \
    Device::process_error(device,e.error,e.what());                             \
  } catch (std::exception& e) {                                                 \
    Device::process_error(device,RTC_ERROR_UNKNOWN,e.what());                   \
  } catch (...) {                                                               \
    Device::process_error(device,RTC_ERROR_UNKNOWN,"unknown exception caught"); \
  }
  
#define RTC_CATCH_END2(scene)                                                \
  } catch (std::bad_alloc&) {                                                   \
    Device* device = scene ? scene->device : nullptr;                           \
    Device::process_error(device,RTC_ERROR_OUT_OF_MEMORY,"out of memory");      \
  } catch (rtcore_error& e) {                                                   \
    Device* device = scene ? scene->device : nullptr;                           \
    Device::process_error(device,e.error,e.what());                             \
  } catch (std::exception& e) {                                                 \
    Device* device = scene ? scene->device : nullptr;                           \
    Device::process_error(device,RTC_ERROR_UNKNOWN,e.what());                   \
  } catch (...) {                                                               \
    Device* device = scene ? scene->device : nullptr;                           \
    Device::process_error(device,RTC_ERROR_UNKNOWN,"unknown exception caught"); \
  }

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

  /*! used to throw embree API errors */
  struct rtcore_error : public std::exception
  {
    __forceinline rtcore_error(RTCError error, const std::string& str)
      : error(error), str(str) {}
    
    ~rtcore_error() throw() {}
    
    const char* what () const throw () {
      return str.c_str();
    }
    
    RTCError error;
    std::string str;
  };

#if defined(DEBUG) // only report file and line in debug mode
  #define throw_RTCError(error,str) \
    throw rtcore_error(error,std::string(__FILE__) + " (" + toString(__LINE__) + "): " + std::string(str));
#else
  #define throw_RTCError(error,str) \
    throw rtcore_error(error,str);
#endif

#define RTC_BUILD_ARGUMENTS_HAS(settings,member) \
  (settings.byteSize > (offsetof(RTCBuildArguments,member)+sizeof(settings.member))) 
}
