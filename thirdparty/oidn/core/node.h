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
#include <vector>

namespace oidn {

  class Node
  {
  public:
    virtual ~Node() = default;

    virtual void execute(stream& sm) = 0;

    virtual std::shared_ptr<memory> getDst() const { return nullptr; }

    virtual size_t getScratchpadSize() const { return 0; }
    virtual void setScratchpad(const std::shared_ptr<memory>& mem) {}

    virtual void setTile(int h1, int w1, int h2, int w2, int H, int W)
    {
      assert(0); // not supported
    }
  };

  // Node wrapping an MKL-DNN primitive
  class MklNode : public Node
  {
  private:
    primitive prim;
    std::unordered_map<int, memory> args;
    std::shared_ptr<memory> scratchpad;

  public:
    MklNode(const primitive& prim, const std::unordered_map<int, memory>& args)
      : prim(prim),
        args(args)
    {}

    size_t getScratchpadSize() const override
    {
      const auto primDesc = prim.get_primitive_desc();
      const mkldnn_memory_desc_t* scratchpadDesc = mkldnn_primitive_desc_query_md(primDesc, mkldnn_query_scratchpad_md, 0);
      if (scratchpadDesc == nullptr)
        return 0;
      return mkldnn_memory_desc_get_size(scratchpadDesc);
    }

    void setScratchpad(const std::shared_ptr<memory>& mem) override
    {
      scratchpad = mem;
      args.insert(std::make_pair(MKLDNN_ARG_SCRATCHPAD, *scratchpad));
    }

    void execute(stream& sm) override
    {
      prim.execute(sm, args);
    }
  };

  // Convolution node
  class ConvNode : public MklNode
  {
  private:
    std::shared_ptr<memory> src;
    std::shared_ptr<memory> weights;
    std::shared_ptr<memory> bias;
    std::shared_ptr<memory> dst;

  public:
    ConvNode(const convolution_forward::primitive_desc& desc,
             const std::shared_ptr<memory>& src,
             const std::shared_ptr<memory>& weights,
             const std::shared_ptr<memory>& bias,
             const std::shared_ptr<memory>& dst)
      : MklNode(convolution_forward(desc),
                { { MKLDNN_ARG_SRC, *src },
                  { MKLDNN_ARG_WEIGHTS, *weights },
                  { MKLDNN_ARG_BIAS, *bias },
                  { MKLDNN_ARG_DST, *dst } }),
                src(src), weights(weights), bias(bias), dst(dst)
    {}

    std::shared_ptr<memory> getDst() const override { return dst; }
  };

  // Pooling node
  class PoolNode : public MklNode
  {
  private:
    std::shared_ptr<memory> src;
    std::shared_ptr<memory> dst;

  public:
    PoolNode(const pooling_forward::primitive_desc& desc,
             const std::shared_ptr<memory>& src,
             const std::shared_ptr<memory>& dst)
      : MklNode(pooling_forward(desc),
                { { MKLDNN_ARG_SRC, *src },
                  { MKLDNN_ARG_DST, *dst } }),
                src(src), dst(dst)
    {}

    std::shared_ptr<memory> getDst() const override { return dst; }
  };

  // Reorder node
  class ReorderNode : public MklNode
  {
  private:
    std::shared_ptr<memory> src;
    std::shared_ptr<memory> dst;

  public:
    ReorderNode(const std::shared_ptr<memory>& src,
                const std::shared_ptr<memory>& dst)
      : MklNode(reorder(reorder::primitive_desc(*src, *dst)),
                { { MKLDNN_ARG_SRC, *src },
                  { MKLDNN_ARG_DST, *dst } }),
                src(src), dst(dst)
    {}

    std::shared_ptr<memory> getDst() const override { return dst; }
  };

} // namespace oidn
