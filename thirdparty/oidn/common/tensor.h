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

#include "platform.h"
#include <vector>
#include <map>

namespace oidn {

  template<typename T>
  using shared_vector = std::shared_ptr<std::vector<T>>;

  // Generic tensor
  struct Tensor
  {
    float* data;
    std::vector<int64_t> dims;
    std::string format;
    shared_vector<char> buffer; // optional, only for reference counting

    __forceinline Tensor() : data(nullptr) {}

    __forceinline Tensor(const std::vector<int64_t>& dims, const std::string& format)
      : dims(dims),
        format(format)
    {
      buffer = std::make_shared<std::vector<char>>(size() * sizeof(float));
      data = (float*)buffer->data();
    }

    __forceinline operator bool() const { return data != nullptr; }

    __forceinline int ndims() const { return (int)dims.size(); }

    // Returns the number of values
    __forceinline size_t size() const
    {
      size_t size = 1;
      for (int i = 0; i < ndims(); ++i)
        size *= dims[i];
      return size;
    }

    __forceinline float& operator [](size_t i) { return data[i]; }
    __forceinline const float& operator [](size_t i) const { return data[i]; }
  };

  // Parses tensors from a buffer
  std::map<std::string, Tensor> parseTensors(void* buffer);

} // namespace oidn
