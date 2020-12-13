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

  // Input reorder node
  template<int K, class TransferFunction>
  class InputReorderNode : public Node
  {
  private:
    // Source
    Image color;
    Image albedo;
    Image normal;

    // Destination
    std::shared_ptr<memory> dst;
    float* dstPtr;
    int C2;
    int H2;
    int W2;

    // Tile
    int h1Begin;
    int w1Begin;
    int h2Begin;
    int w2Begin;
    int H;
    int W;

    std::shared_ptr<TransferFunction> transferFunc;

  public:
    InputReorderNode(const Image& color,
                     const Image& albedo,
                     const Image& normal,
                     const std::shared_ptr<memory>& dst,
                     const std::shared_ptr<TransferFunction>& transferFunc)
      : color(color), albedo(albedo), normal(normal),
        dst(dst),
        h1Begin(0), w1Begin(0),
        H(color.height), W(color.width),
        transferFunc(transferFunc)
    {
      const mkldnn_memory_desc_t& dstDesc = dst->get_desc().data;
      assert(memory_desc_matches_tag(dstDesc, mkldnn_format_tag_t(BlockedFormat<K>::nChwKc)));
      assert(dstDesc.ndims == 4);
      assert(dstDesc.data_type == memory::data_type::f32);
      assert(dstDesc.dims[0] == 1);
      //assert(dstDesc.dims[1] >= getPadded<K>(C1));

      dstPtr = (float*)dst->get_data_handle();
      C2 = dstDesc.dims[1];
      H2 = dstDesc.dims[2];
      W2 = dstDesc.dims[3];
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
      assert(H + h1Begin <= color.height);
      assert(W + w1Begin <= color.width);
      assert(H + h2Begin <= H2);
      assert(W + w2Begin <= W2);

      parallel_nd(H2, [&](int h2)
      {
        const int h = h2 - h2Begin;

        if (h >= 0 && h < H)
        {
          const int h1 = h + h1Begin;

          // Zero pad
          for (int w2 = 0; w2 < w2Begin; ++w2)
          {
            int c = 0;
            while (c < C2)
              store(h2, w2, c, 0.f);
          }

          // Reorder
          for (int w = 0; w < W; ++w)
          {
            const int w1 = w + w1Begin;
            const int w2 = w + w2Begin;

            int c = 0;
            storeColor(h2, w2, c, (float*)color.get(h1, w1));
            if (albedo)
              storeAlbedo(h2, w2, c, (float*)albedo.get(h1, w1));
            if (normal)
              storeNormal(h2, w2, c, (float*)normal.get(h1, w1));
            while (c < C2)
              store(h2, w2, c, 0.f);
          }

          // Zero pad
          for (int w2 = W + w2Begin; w2 < W2; ++w2)
          {
            int c = 0;
            while (c < C2)
              store(h2, w2, c, 0.f);
          }
        }
        else
        {
          // Zero pad
          for (int w2 = 0; w2 < W2; ++w2)
          {
            int c = 0;
            while (c < C2)
              store(h2, w2, c, 0.f);
          }
        }
      });
    }

    std::shared_ptr<memory> getDst() const override { return dst; }

  private:
    // Stores a single value
    __forceinline void store(int h, int w, int& c, float value)
    {
      // Destination is in nChwKc format
      float* dst_c = dstPtr + (H2*W2*K*(c/K)) + h*W2*K + w*K + (c%K);
      *dst_c = value;
      c++;
    }

    // Stores a color
    __forceinline void storeColor(int h, int w, int& c, const float* values)
    {
      #pragma unroll
      for (int i = 0; i < 3; ++i)
      {
        // Load the value
        float x = values[i];

        // Sanitize the value
        x = maxSafe(x, 0.f);

        // Apply the transfer function
        x = transferFunc->forward(x);

        // Store the value
        store(h, w, c, x);
      }
    }

    // Stores an albedo
    __forceinline void storeAlbedo(int h, int w, int& c, const float* values)
    {
      #pragma unroll
      for (int i = 0; i < 3; ++i)
      {
        // Load the value
        float x = values[i];

        // Sanitize the value
        x = clampSafe(x, 0.f, 1.f);

        // Store the value
        store(h, w, c, x);
      }
    }

    // Stores a normal
    __forceinline void storeNormal(int h, int w, int& c, const float* values)
    {
      // Load the normal
      float x = values[0];
      float y = values[1];
      float z = values[2];

      // Compute the length of the normal
      const float lengthSqr = sqr(x) + sqr(y) + sqr(z);

      // Normalize the normal and transform it to [0..1]
      if (isfinite(lengthSqr))
      {
        const float invLength = (lengthSqr > minVectorLengthSqr) ? rsqrt(lengthSqr) : 1.f;

        const float scale  = invLength * 0.5f;
        const float offset = 0.5f;

        x = x * scale + offset;
        y = y * scale + offset;
        z = z * scale + offset;
      }
      else
      {
        x = 0.f;
        y = 0.f;
        z = 0.f;
      }

      // Store the normal
      store(h, w, c, x);
      store(h, w, c, y);
      store(h, w, c, z);
    }
  };

} // namespace oidn
