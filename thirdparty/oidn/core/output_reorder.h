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

#include "node.h"
#include "image.h"

namespace oidn {

  // Output reorder node
  template<int K, class TransferFunction>
  class OutputReorderNode : public Node
  {
  private:
    // Source
    std::shared_ptr<memory> src;
    const float* srcPtr;
    int H1;
    int W1;

    // Destination
    Image output;

    // Tile
    int h1Begin;
    int w1Begin;
    int h2Begin;
    int w2Begin;
    int H;
    int W;

    std::shared_ptr<TransferFunction> transferFunc;

  public:
    OutputReorderNode(const std::shared_ptr<memory>& src,
                      const Image& output,
                      const std::shared_ptr<TransferFunction>& transferFunc)
      : src(src),
        output(output),
        h1Begin(0), w1Begin(0),
        h2Begin(0), w2Begin(0),
        H(output.height), W(output.width),
        transferFunc(transferFunc)
    {
      const mkldnn_memory_desc_t& srcDesc = src->get_desc().data;
      MAYBE_UNUSED(srcDesc);
      assert(memory_desc_matches_tag(srcDesc, mkldnn_format_tag_t(BlockedFormat<K>::nChwKc)));
      assert(srcDesc.ndims == 4);
      assert(srcDesc.data_type == memory::data_type::f32);
      assert(srcDesc.dims[0] == 1);
      // We assume output data is <= K OC
      assert(srcDesc.dims[1] == K);

      srcPtr = (float*)src->get_data_handle();
      H1 = srcDesc.dims[2];
      W1 = srcDesc.dims[3];
    }

    void setTile(int h1, int w1, int h2, int w2, int H, int W) override
    {
      h1Begin = h1;
      w1Begin = w1;
      h2Begin = h2;
      w2Begin = w2;
      this->H = H;
      this->W = W;
    }

    void execute(stream& sm) override
    {
      assert(h1Begin + H <= H1);
      assert(w1Begin + W <= W1);
      assert(h2Begin + H <= output.height);
      assert(w2Begin + W <= output.width);

      const int C1 = K;

      parallel_nd(H, [&](int h)
      {
        const int h1 = h + h1Begin;
        const int h2 = h + h2Begin;

        for (int w = 0; w < W; ++w)
        {
          const int w1 = w + w1Begin;
          const int w2 = w + w2Begin;
          float* dstPtr_C = (float*)output.get(h2, w2);

          // Source is in nChwKc format. In this case C is 1 so this is really nhwc
          const float* srcPtr_C = srcPtr + h1*W1*C1 + w1*C1;

          #pragma unroll
          for (int i = 0; i < 3; ++i)
          {
            // Load the value
            float x = srcPtr_C[i];

            // The CNN output may contain negative values or even NaNs, so it must be sanitized
            x = maxSafe(x, 0.f);

            // Apply the inverse transfer function
            x = transferFunc->inverse(x);

            // Sanitize and store the final value
            dstPtr_C[i] = max(x, 0.f);
          }
        }
      });
    }
  };

} // namespace oidn
