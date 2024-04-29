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

#include "common.h"
#include "buffer.h"

namespace oidn {

  struct Image
  {
    static constexpr int maxSize = 65536;

    char* ptr;              // pointer to the first pixel
    int width;              // width in number of pixels
    int height;             // height in number of pixels
    size_t bytePixelStride; // pixel stride in number of *bytes*
    size_t rowStride;       // row stride in number of *pixel strides*
    Format format;          // pixel format
    Ref<Buffer> buffer;     // buffer containing the image data

    Image() : ptr(nullptr), width(0), height(0), bytePixelStride(0), rowStride(0), format(Format::Undefined) {}

    Image(void* ptr, Format format, int width, int height, size_t byteOffset, size_t inBytePixelStride, size_t inByteRowStride)
    {
      if (ptr == nullptr)
        throw Exception(Error::InvalidArgument, "buffer pointer null");

      init((char*)ptr + byteOffset, format, width, height, inBytePixelStride, inByteRowStride);
    }

    Image(const Ref<Buffer>& buffer, Format format, int width, int height, size_t byteOffset, size_t inBytePixelStride, size_t inByteRowStride)
    {
      init(buffer->data() + byteOffset, format, width, height, inBytePixelStride, inByteRowStride);

      if (byteOffset + height * rowStride * bytePixelStride > buffer->size())
        throw Exception(Error::InvalidArgument, "buffer region out of range");
    }

    void init(char* ptr, Format format, int width, int height, size_t inBytePixelStride, size_t inByteRowStride)
    {
      assert(width >= 0);
      assert(height >= 0);
      if (width > maxSize || height > maxSize)
        throw Exception(Error::InvalidArgument, "image size too large");

      this->ptr = ptr;
      this->width = width;
      this->height = height;

      const size_t pixelSize = getFormatBytes(format);
      if (inBytePixelStride != 0)
      {
        if (inBytePixelStride < pixelSize)
          throw Exception(Error::InvalidArgument, "pixel stride smaller than pixel size");

        this->bytePixelStride = inBytePixelStride;
      }
      else
      {
        this->bytePixelStride = pixelSize;
      }

      if (inByteRowStride != 0)
      {
        if (inByteRowStride < width * this->bytePixelStride)
          throw Exception(Error::InvalidArgument, "row stride smaller than width * pixel stride");
        if (inByteRowStride % this->bytePixelStride != 0)
          throw Exception(Error::InvalidArgument, "row stride not integer multiple of pixel stride");

        this->rowStride = inByteRowStride / this->bytePixelStride;
      }
      else
      {
        this->rowStride = width;
      }

      this->format = format;
    }

    __forceinline char* get(int y, int x)
    {
      return ptr + ((size_t(y) * rowStride + size_t(x)) * bytePixelStride);
    }

    __forceinline const char* get(int y, int x) const
    {
      return ptr + ((size_t(y) * rowStride + size_t(x)) * bytePixelStride);
    }

    operator bool() const
    {
      return ptr != nullptr;
    }
  };

} // namespace oidn
