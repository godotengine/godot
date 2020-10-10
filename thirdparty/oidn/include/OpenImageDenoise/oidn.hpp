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

#pragma once

#include <algorithm>
#include "oidn.h"

namespace oidn {

  // --------------------------------------------------------------------------
  // Buffer
  // --------------------------------------------------------------------------

  // Formats for images and other data stored in buffers
  enum class Format
  {
    Undefined = OIDN_FORMAT_UNDEFINED,

    // 32-bit single-precision floating point scalar and vector formats
    Float  = OIDN_FORMAT_FLOAT,
    Float2 = OIDN_FORMAT_FLOAT2,
    Float3 = OIDN_FORMAT_FLOAT3,
    Float4 = OIDN_FORMAT_FLOAT4,
  };

  // Access modes for mapping buffers
  enum class Access
  {
    Read         = OIDN_ACCESS_READ,          // read-only access
    Write        = OIDN_ACCESS_WRITE,         // write-only access
    ReadWrite    = OIDN_ACCESS_READ_WRITE,    // read and write access
    WriteDiscard = OIDN_ACCESS_WRITE_DISCARD, // write-only access, previous contents discarded
  };

  // Buffer object with automatic reference counting
  class BufferRef
  {
  private:
    OIDNBuffer handle;

  public:
    BufferRef() : handle(nullptr) {}
    BufferRef(OIDNBuffer handle) : handle(handle) {}

    BufferRef(const BufferRef& other) : handle(other.handle)
    {
      if (handle)
        oidnRetainBuffer(handle);
    }

    BufferRef(BufferRef&& other) : handle(other.handle)
    {
      other.handle = nullptr;
    }

    BufferRef& operator =(const BufferRef& other)
    {
      if (&other != this)
      {
        if (other.handle)
          oidnRetainBuffer(other.handle);
        if (handle)
          oidnReleaseBuffer(handle);
        handle = other.handle;
      }
      return *this;
    }

    BufferRef& operator =(BufferRef&& other)
    {
      std::swap(handle, other.handle);
      return *this;
    }

    BufferRef& operator =(OIDNBuffer other)
    {
      if (other)
        oidnRetainBuffer(other);
      if (handle)
        oidnReleaseBuffer(handle);
      handle = other;
      return *this;
    }

    ~BufferRef()
    {
      if (handle)
        oidnReleaseBuffer(handle);
    }

    OIDNBuffer getHandle() const
    {
      return handle;
    }

    operator bool() const
    {
      return handle != nullptr;
    }

    // Maps a region of the buffer to host memory.
    // If byteSize is 0, the maximum available amount of memory will be mapped.
    void* map(Access access = Access::ReadWrite, size_t byteOffset = 0, size_t byteSize = 0)
    {
      return oidnMapBuffer(handle, (OIDNAccess)access, byteOffset, byteSize);
    }

    // Unmaps a region of the buffer.
    // mappedPtr must be a pointer returned by a previous call to map.
    void unmap(void* mappedPtr)
    {
      oidnUnmapBuffer(handle, mappedPtr);
    }
  };

  // --------------------------------------------------------------------------
  // Filter
  // --------------------------------------------------------------------------

  // Progress monitor callback function
  typedef bool (*ProgressMonitorFunction)(void* userPtr, double n);

  // Filter object with automatic reference counting
  class FilterRef
  {
  private:
    OIDNFilter handle;

  public:
    FilterRef() : handle(nullptr) {}
    FilterRef(OIDNFilter handle) : handle(handle) {}

    FilterRef(const FilterRef& other) : handle(other.handle)
    {
      if (handle)
        oidnRetainFilter(handle);
    }

    FilterRef(FilterRef&& other) : handle(other.handle)
    {
      other.handle = nullptr;
    }

    FilterRef& operator =(const FilterRef& other)
    {
      if (&other != this)
      {
        if (other.handle)
          oidnRetainFilter(other.handle);
        if (handle)
          oidnReleaseFilter(handle);
        handle = other.handle;
      }
      return *this;
    }

    FilterRef& operator =(FilterRef&& other)
    {
      std::swap(handle, other.handle);
      return *this;
    }

    FilterRef& operator =(OIDNFilter other)
    {
      if (other)
        oidnRetainFilter(other);
      if (handle)
        oidnReleaseFilter(handle);
      handle = other;
      return *this;
    }

    ~FilterRef()
    {
      if (handle)
        oidnReleaseFilter(handle);
    }

    OIDNFilter getHandle() const
    {
      return handle;
    }

    operator bool() const
    {
      return handle != nullptr;
    }

    // Sets an image parameter of the filter (stored in a buffer).
    void setImage(const char* name,
                  const BufferRef& buffer, Format format,
                  size_t width, size_t height,
                  size_t byteOffset = 0,
                  size_t bytePixelStride = 0, size_t byteRowStride = 0)
    {
      oidnSetFilterImage(handle, name,
                         buffer.getHandle(), (OIDNFormat)format,
                         width, height,
                         byteOffset,
                         bytePixelStride, byteRowStride);
    }

    // Sets an image parameter of the filter (owned by the user).
    void setImage(const char* name,
                  void* ptr, Format format,
                  size_t width, size_t height,
                  size_t byteOffset = 0,
                  size_t bytePixelStride = 0, size_t byteRowStride = 0)
    {
      oidnSetSharedFilterImage(handle, name,
                               ptr, (OIDNFormat)format,
                               width, height,
                               byteOffset,
                               bytePixelStride, byteRowStride);
    }

    // Sets a boolean parameter of the filter.
    void set(const char* name, bool value)
    {
      oidnSetFilter1b(handle, name, value);
    }

    // Sets an integer parameter of the filter.
    void set(const char* name, int value)
    {
      oidnSetFilter1i(handle, name, value);
    }

    // Sets a float parameter of the filter.
    void set(const char* name, float value)
    {
      oidnSetFilter1f(handle, name, value);
    }

    // Gets a parameter of the filter.
    template<typename T>
    T get(const char* name);

    // Sets the progress monitor callback function of the filter.
    void setProgressMonitorFunction(ProgressMonitorFunction func, void* userPtr = nullptr)
    {
      oidnSetFilterProgressMonitorFunction(handle, (OIDNProgressMonitorFunction)func, userPtr);
    }

    // Commits all previous changes to the filter.
    void commit()
    {
      oidnCommitFilter(handle);
    }

    // Executes the filter.
    void execute()
    {
      oidnExecuteFilter(handle);
    }
  };

  // Gets a boolean parameter of the filter.
  template<>
  inline bool FilterRef::get(const char* name)
  {
    return oidnGetFilter1b(handle, name);
  }

  // Gets an integer parameter of the filter.
  template<>
  inline int FilterRef::get(const char* name)
  {
    return oidnGetFilter1i(handle, name);
  }

  // Gets a float parameter of the filter.
  template<>
  inline float FilterRef::get(const char* name)
  {
    return oidnGetFilter1f(handle, name);
  }

  // --------------------------------------------------------------------------
  // Device
  // --------------------------------------------------------------------------

  // Device types
  enum class DeviceType
  {
    Default = OIDN_DEVICE_TYPE_DEFAULT, // select device automatically

    CPU = OIDN_DEVICE_TYPE_CPU, // CPU device
  };

  // Error codes
  enum class Error
  {
    None                = OIDN_ERROR_NONE,                 // no error occurred
    Unknown             = OIDN_ERROR_UNKNOWN,              // an unknown error occurred
    InvalidArgument     = OIDN_ERROR_INVALID_ARGUMENT,     // an invalid argument was specified
    InvalidOperation    = OIDN_ERROR_INVALID_OPERATION,    // the operation is not allowed
    OutOfMemory         = OIDN_ERROR_OUT_OF_MEMORY,        // not enough memory to execute the operation
    UnsupportedHardware = OIDN_ERROR_UNSUPPORTED_HARDWARE, // the hardware (e.g. CPU) is not supported
    Cancelled           = OIDN_ERROR_CANCELLED,            // the operation was cancelled by the user
  };

  // Error callback function
  typedef void (*ErrorFunction)(void* userPtr, Error code, const char* message);

  // Device object with automatic reference counting
  class DeviceRef
  {
  private:
    OIDNDevice handle;

  public:
    DeviceRef() : handle(nullptr) {}
    DeviceRef(OIDNDevice handle) : handle(handle) {}

    DeviceRef(const DeviceRef& other) : handle(other.handle)
    {
      if (handle)
        oidnRetainDevice(handle);
    }

    DeviceRef(DeviceRef&& other) : handle(other.handle)
    {
      other.handle = nullptr;
    }

    DeviceRef& operator =(const DeviceRef& other)
    {
      if (&other != this)
      {
        if (other.handle)
          oidnRetainDevice(other.handle);
        if (handle)
          oidnReleaseDevice(handle);
        handle = other.handle;
      }
      return *this;
    }

    DeviceRef& operator =(DeviceRef&& other)
    {
      std::swap(handle, other.handle);
      return *this;
    }

    DeviceRef& operator =(OIDNDevice other)
    {
      if (other)
        oidnRetainDevice(other);
      if (handle)
        oidnReleaseDevice(handle);
      handle = other;
      return *this;
    }

    ~DeviceRef()
    {
      if (handle)
        oidnReleaseDevice(handle);
    }

    OIDNDevice getHandle() const
    {
      return handle;
    }

    operator bool() const
    {
      return handle != nullptr;
    }

    // Sets a boolean parameter of the device.
    void set(const char* name, bool value)
    {
      oidnSetDevice1b(handle, name, value);
    }

    // Sets an integer parameter of the device.
    void set(const char* name, int value)
    {
      oidnSetDevice1i(handle, name, value);
    }

    // Gets a parameter of the device.
    template<typename T>
    T get(const char* name);

    // Sets the error callback function of the device.
    void setErrorFunction(ErrorFunction func, void* userPtr = nullptr)
    {
      oidnSetDeviceErrorFunction(handle, (OIDNErrorFunction)func, userPtr);
    }

    // Returns the first unqueried error code and clears the stored error.
    // Can be called for a null device as well to check why a device creation failed.
    Error getError()
    {
      return (Error)oidnGetDeviceError(handle, nullptr);
    }

    // Returns the first unqueried error code and string message, and clears the stored error.
    // Can be called for a null device as well to check why a device creation failed.
    Error getError(const char*& outMessage)
    {
      return (Error)oidnGetDeviceError(handle, &outMessage);
    }

    // Commits all previous changes to the device.
    // Must be called before first using the device (e.g. creating filters).
    void commit()
    {
      oidnCommitDevice(handle);
    }

    // Creates a new buffer (data allocated and owned by the device).
    BufferRef newBuffer(size_t byteSize)
    {
      return oidnNewBuffer(handle, byteSize);
    }

    // Creates a new shared buffer (data allocated and owned by the user).
    BufferRef newBuffer(void* ptr, size_t byteSize)
    {
      return oidnNewSharedBuffer(handle, ptr, byteSize);
    }

    // Creates a new filter of the specified type (e.g. "RT").
    FilterRef newFilter(const char* type)
    {
      return oidnNewFilter(handle, type);
    }
  };

  // Gets a boolean parameter of the device.
  template<>
  inline bool DeviceRef::get(const char* name)
  {
    return oidnGetDevice1b(handle, name);
  }

  // Gets an integer parameter of the device (e.g. "version").
  template<>
  inline int DeviceRef::get(const char* name)
  {
    return oidnGetDevice1i(handle, name);
  }

  // Creates a new device.
  inline DeviceRef newDevice(DeviceType type = DeviceType::Default)
  {
    return DeviceRef(oidnNewDevice((OIDNDeviceType)type));
  }

} // namespace oidn
