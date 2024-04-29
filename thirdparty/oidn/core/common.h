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

#include "common/platform.h"

#include "mkl-dnn/include/mkldnn.hpp"
#include "mkl-dnn/include/mkldnn_debug.h"
#include "mkl-dnn/src/common/mkldnn_thread.hpp"
#include "mkl-dnn/src/common/type_helpers.hpp"
#include "mkl-dnn/src/cpu/jit_generator.hpp"

#include "common/ref.h"
#include "common/exception.h"
#include "common/thread.h"
// -- GODOT start --
//#include "common/tasking.h"
// -- GODOT end --
#include "math.h"

namespace oidn {

  using namespace mkldnn;
  using namespace mkldnn::impl::cpu;
  using mkldnn::impl::parallel_nd;
  using mkldnn::impl::memory_desc_matches_tag;


  inline size_t getFormatBytes(Format format)
  {
    switch (format)
    {
    case Format::Undefined: return 1;
    case Format::Float:     return sizeof(float);
    case Format::Float2:    return sizeof(float)*2;
    case Format::Float3:    return sizeof(float)*3;
    case Format::Float4:    return sizeof(float)*4;
    }
    assert(0);
    return 0;
  }


  inline memory::dims getTensorDims(const std::shared_ptr<memory>& mem)
  {
    const mkldnn_memory_desc_t& desc = mem->get_desc().data;
    return memory::dims(&desc.dims[0], &desc.dims[desc.ndims]);
  }

  inline memory::data_type getTensorType(const std::shared_ptr<memory>& mem)
  {
    const mkldnn_memory_desc_t& desc = mem->get_desc().data;
    return memory::data_type(desc.data_type);
  }

  // Returns the number of values in a tensor
  inline size_t getTensorSize(const memory::dims& dims)
  {
    size_t res = 1;
    for (int i = 0; i < (int)dims.size(); ++i)
      res *= dims[i];
    return res;
  }

  inline memory::dims getMaxTensorDims(const std::vector<memory::dims>& dims)
  {
    memory::dims result;
    size_t maxSize = 0;

    for (const auto& d : dims)
    {
      const size_t size = getTensorSize(d);
      if (size > maxSize)
      {
        result = d;
        maxSize = size;
      }
    }

    return result;
  }

  inline size_t getTensorSize(const std::shared_ptr<memory>& mem)
  {
    return getTensorSize(getTensorDims(mem));
  }


  template<int K>
  inline int getPadded(int dim)
  {
    return (dim + (K-1)) & ~(K-1);
  }

  template<int K>
  inline memory::dims getPadded_nchw(const memory::dims& dims)
  {
    assert(dims.size() == 4);
    memory::dims padDims = dims;
    padDims[1] = getPadded<K>(dims[1]); // pad C
    return padDims;
  }


  template<int K>
  struct BlockedFormat;

  template<>
  struct BlockedFormat<8>
  {
    static constexpr memory::format_tag nChwKc   = memory::format_tag::nChw8c;
    static constexpr memory::format_tag OIhwKiKo = memory::format_tag::OIhw8i8o;
  };

  template<>
  struct BlockedFormat<16>
  {
    static constexpr memory::format_tag nChwKc   = memory::format_tag::nChw16c;
    static constexpr memory::format_tag OIhwKiKo = memory::format_tag::OIhw16i16o;
  };

} // namespace oidn
