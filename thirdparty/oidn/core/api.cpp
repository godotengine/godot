// ======================================================================== //
// Copyright 2009-2019 Intel Corporation                                    //
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

#ifdef _WIN32
#  define OIDN_API extern "C" __declspec(dllexport)
#else
#  define OIDN_API extern "C" __attribute__ ((visibility ("default")))
#endif

// Locks the device that owns the specified object
// Use *only* inside OIDN_TRY/CATCH!
#define OIDN_LOCK(obj) \
  std::lock_guard<std::mutex> lock(obj->getDevice()->getMutex());

// Try/catch for converting exceptions to errors
#define OIDN_TRY \
  try {

#define OIDN_CATCH(obj) \
  } catch (Exception& e) {                                                                          \
    Device::setError(obj ? obj->getDevice() : nullptr, e.code(), e.what());                         \
  } catch (std::bad_alloc&) {                                                                       \
    Device::setError(obj ? obj->getDevice() : nullptr, Error::OutOfMemory, "out of memory");        \
  } catch (mkldnn::error& e) {                                                                      \
    if (e.status == mkldnn_out_of_memory)                                                           \
      Device::setError(obj ? obj->getDevice() : nullptr, Error::OutOfMemory, "out of memory");      \
    else                                                                                            \
      Device::setError(obj ? obj->getDevice() : nullptr, Error::Unknown, e.message);                \
  } catch (std::exception& e) {                                                                     \
    Device::setError(obj ? obj->getDevice() : nullptr, Error::Unknown, e.what());                   \
  } catch (...) {                                                                                   \
    Device::setError(obj ? obj->getDevice() : nullptr, Error::Unknown, "unknown exception caught"); \
  }

#include "device.h"
#include "filter.h"
#include <mutex>

namespace oidn {

  namespace
  {
    __forceinline void checkHandle(void* handle)
    {
      if (handle == nullptr)
        throw Exception(Error::InvalidArgument, "invalid handle");
    }

    template<typename T>
    __forceinline void retainObject(T* obj)
    {
      if (obj)
      {
        obj->incRef();
      }
      else
      {
        OIDN_TRY
          checkHandle(obj);
        OIDN_CATCH(obj)
      }
    }

    template<typename T>
    __forceinline void releaseObject(T* obj)
    {
      if (obj == nullptr || obj->decRefKeep() == 0)
      {
        OIDN_TRY
          checkHandle(obj);
          OIDN_LOCK(obj);
          obj->destroy();
        OIDN_CATCH(obj)
      }
    }

    template<>
    __forceinline void releaseObject(Device* obj)
    {
      if (obj == nullptr || obj->decRefKeep() == 0)
      {
        OIDN_TRY
          checkHandle(obj);
          // Do NOT lock the device because it owns the mutex
          obj->destroy();
        OIDN_CATCH(obj)
      }
    }
  }

  OIDN_API OIDNDevice oidnNewDevice(OIDNDeviceType type)
  {
    Ref<Device> device = nullptr;
    OIDN_TRY
      if (type == OIDN_DEVICE_TYPE_CPU || type == OIDN_DEVICE_TYPE_DEFAULT)
        device = makeRef<Device>();
      else
        throw Exception(Error::InvalidArgument, "invalid device type");
    OIDN_CATCH(device)
    return (OIDNDevice)device.detach();
  }

  OIDN_API void oidnRetainDevice(OIDNDevice hDevice)
  {
    Device* device = (Device*)hDevice;
    retainObject(device);
  }

  OIDN_API void oidnReleaseDevice(OIDNDevice hDevice)
  {
    Device* device = (Device*)hDevice;
    releaseObject(device);
  }

  OIDN_API void oidnSetDevice1b(OIDNDevice hDevice, const char* name, bool value)
  {
    Device* device = (Device*)hDevice;
    OIDN_TRY
      checkHandle(hDevice);
      OIDN_LOCK(device);
      device->set1i(name, value);
    OIDN_CATCH(device)
  }

  OIDN_API void oidnSetDevice1i(OIDNDevice hDevice, const char* name, int value)
  {
    Device* device = (Device*)hDevice;
    OIDN_TRY
      checkHandle(hDevice);
      OIDN_LOCK(device);
      device->set1i(name, value);
    OIDN_CATCH(device)
  }

  OIDN_API bool oidnGetDevice1b(OIDNDevice hDevice, const char* name)
  {
    Device* device = (Device*)hDevice;
    OIDN_TRY
      checkHandle(hDevice);
      OIDN_LOCK(device);
      return device->get1i(name);
    OIDN_CATCH(device)
    return false;
  }

  OIDN_API int oidnGetDevice1i(OIDNDevice hDevice, const char* name)
  {
    Device* device = (Device*)hDevice;
    OIDN_TRY
      checkHandle(hDevice);
      OIDN_LOCK(device);
      return device->get1i(name);
    OIDN_CATCH(device)
    return 0;
  }

  OIDN_API void oidnSetDeviceErrorFunction(OIDNDevice hDevice, OIDNErrorFunction func, void* userPtr)
  {
    Device* device = (Device*)hDevice;
    OIDN_TRY
      checkHandle(hDevice);
      OIDN_LOCK(device);
      device->setErrorFunction((ErrorFunction)func, userPtr);
    OIDN_CATCH(device)
  }

  OIDN_API OIDNError oidnGetDeviceError(OIDNDevice hDevice, const char** outMessage)
  {
    Device* device = (Device*)hDevice;
    OIDN_TRY
      return (OIDNError)Device::getError(device, outMessage);
    OIDN_CATCH(device)
    if (outMessage) *outMessage = "";
    return OIDN_ERROR_UNKNOWN;
  }

  OIDN_API void oidnCommitDevice(OIDNDevice hDevice)
  {
    Device* device = (Device*)hDevice;
    OIDN_TRY
      checkHandle(hDevice);
      OIDN_LOCK(device);
      device->commit();
    OIDN_CATCH(device)
  }

  OIDN_API OIDNBuffer oidnNewBuffer(OIDNDevice hDevice, size_t byteSize)
  {
    Device* device = (Device*)hDevice;
    OIDN_TRY
      checkHandle(hDevice);
      OIDN_LOCK(device);
      Ref<Buffer> buffer = device->newBuffer(byteSize);
      return (OIDNBuffer)buffer.detach();
    OIDN_CATCH(device)
    return nullptr;
  }

  OIDN_API OIDNBuffer oidnNewSharedBuffer(OIDNDevice hDevice, void* ptr, size_t byteSize)
  {
    Device* device = (Device*)hDevice;
    OIDN_TRY
      checkHandle(hDevice);
      OIDN_LOCK(device);
      Ref<Buffer> buffer = device->newBuffer(ptr, byteSize);
      return (OIDNBuffer)buffer.detach();
    OIDN_CATCH(device)
    return nullptr;
  }

  OIDN_API void oidnRetainBuffer(OIDNBuffer hBuffer)
  {
    Buffer* buffer = (Buffer*)hBuffer;
    retainObject(buffer);
  }

  OIDN_API void oidnReleaseBuffer(OIDNBuffer hBuffer)
  {
    Buffer* buffer = (Buffer*)hBuffer;
    releaseObject(buffer);
  }

  OIDN_API void* oidnMapBuffer(OIDNBuffer hBuffer, OIDNAccess access, size_t byteOffset, size_t byteSize)
  {
    Buffer* buffer = (Buffer*)hBuffer;
    OIDN_TRY
      checkHandle(hBuffer);
      OIDN_LOCK(buffer);
      return buffer->map(byteOffset, byteSize);
    OIDN_CATCH(buffer)
    return nullptr;
  }

  OIDN_API void oidnUnmapBuffer(OIDNBuffer hBuffer, void* mappedPtr)
  {
    Buffer* buffer = (Buffer*)hBuffer;
    OIDN_TRY
      checkHandle(hBuffer);
      OIDN_LOCK(buffer);
      return buffer->unmap(mappedPtr);
    OIDN_CATCH(buffer)
  }

  OIDN_API OIDNFilter oidnNewFilter(OIDNDevice hDevice, const char* type)
  {
    Device* device = (Device*)hDevice;
    OIDN_TRY
      checkHandle(hDevice);
      OIDN_LOCK(device);
      Ref<Filter> filter = device->newFilter(type);
      return (OIDNFilter)filter.detach();
    OIDN_CATCH(device)
    return nullptr;
  }

  OIDN_API void oidnRetainFilter(OIDNFilter hFilter)
  {
    Filter* filter = (Filter*)hFilter;
    retainObject(filter);
  }

  OIDN_API void oidnReleaseFilter(OIDNFilter hFilter)
  {
    Filter* filter = (Filter*)hFilter;
    releaseObject(filter);
  }

  OIDN_API void oidnSetFilterImage(OIDNFilter hFilter, const char* name,
                                   OIDNBuffer hBuffer, OIDNFormat format,
                                   size_t width, size_t height,
                                   size_t byteOffset,
                                   size_t bytePixelStride, size_t byteRowStride)
  {
    Filter* filter = (Filter*)hFilter;
    OIDN_TRY
      checkHandle(hFilter);
      checkHandle(hBuffer);
      OIDN_LOCK(filter);
      Ref<Buffer> buffer = (Buffer*)hBuffer;
      if (buffer->getDevice() != filter->getDevice())
        throw Exception(Error::InvalidArgument, "the specified objects are bound to different devices");
      Image data(buffer, (Format)format, (int)width, (int)height, byteOffset, bytePixelStride, byteRowStride);
      filter->setImage(name, data);
    OIDN_CATCH(filter)
  }

  OIDN_API void oidnSetSharedFilterImage(OIDNFilter hFilter, const char* name,
                                         void* ptr, OIDNFormat format,
                                         size_t width, size_t height,
                                         size_t byteOffset,
                                         size_t bytePixelStride, size_t byteRowStride)
  {
    Filter* filter = (Filter*)hFilter;
    OIDN_TRY
      checkHandle(hFilter);
      OIDN_LOCK(filter);
      Image data(ptr, (Format)format, (int)width, (int)height, byteOffset, bytePixelStride, byteRowStride);
      filter->setImage(name, data);
    OIDN_CATCH(filter)
  }

  OIDN_API void oidnSetFilter1b(OIDNFilter hFilter, const char* name, bool value)
  {
    Filter* filter = (Filter*)hFilter;
    OIDN_TRY
      checkHandle(hFilter);
      OIDN_LOCK(filter);
      filter->set1i(name, int(value));
    OIDN_CATCH(filter)
  }

  OIDN_API bool oidnGetFilter1b(OIDNFilter hFilter, const char* name)
  {
    Filter* filter = (Filter*)hFilter;
    OIDN_TRY
      checkHandle(hFilter);
      OIDN_LOCK(filter);
      return filter->get1i(name);
    OIDN_CATCH(filter)
    return false;
  }

  OIDN_API void oidnSetFilter1i(OIDNFilter hFilter, const char* name, int value)
  {
    Filter* filter = (Filter*)hFilter;
    OIDN_TRY
      checkHandle(hFilter);
      OIDN_LOCK(filter);
      filter->set1i(name, value);
    OIDN_CATCH(filter)
  }

  OIDN_API int oidnGetFilter1i(OIDNFilter hFilter, const char* name)
  {
    Filter* filter = (Filter*)hFilter;
    OIDN_TRY
      checkHandle(hFilter);
      OIDN_LOCK(filter);
      return filter->get1i(name);
    OIDN_CATCH(filter)
    return 0;
  }

  OIDN_API void oidnSetFilter1f(OIDNFilter hFilter, const char* name, float value)
  {
    Filter* filter = (Filter*)hFilter;
    OIDN_TRY
      checkHandle(hFilter);
      OIDN_LOCK(filter);
      filter->set1f(name, value);
    OIDN_CATCH(filter)
  }

  OIDN_API float oidnGetFilter1f(OIDNFilter hFilter, const char* name)
  {
    Filter* filter = (Filter*)hFilter;
    OIDN_TRY
      checkHandle(hFilter);
      OIDN_LOCK(filter);
      return filter->get1f(name);
    OIDN_CATCH(filter)
    return 0;
  }

  OIDN_API void oidnSetFilterProgressMonitorFunction(OIDNFilter hFilter, OIDNProgressMonitorFunction func, void* userPtr)
  {
    Filter* filter = (Filter*)hFilter;
    OIDN_TRY
      checkHandle(hFilter);
      OIDN_LOCK(filter);
      filter->setProgressMonitorFunction(func, userPtr);
    OIDN_CATCH(filter)
  }

  OIDN_API void oidnCommitFilter(OIDNFilter hFilter)
  {
    Filter* filter = (Filter*)hFilter;
    OIDN_TRY
      checkHandle(hFilter);
      OIDN_LOCK(filter);
      filter->commit();
    OIDN_CATCH(filter)
  }

  OIDN_API void oidnExecuteFilter(OIDNFilter hFilter)
  {
    Filter* filter = (Filter*)hFilter;
    OIDN_TRY
      checkHandle(hFilter);
      OIDN_LOCK(filter);
      filter->execute();
    OIDN_CATCH(filter)
  }

} // namespace oidn
