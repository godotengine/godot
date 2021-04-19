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

#include "filter.h"
#include "network.h"
#include "transfer_function.h"

namespace oidn {

  // --------------------------------------------------------------------------
  // AutoencoderFilter - Direct-predicting autoencoder
  // --------------------------------------------------------------------------

  class AutoencoderFilter : public Filter
  {
  protected:
    static constexpr int alignment       = 32;  // required spatial alignment in pixels (padding may be necessary)
    static constexpr int receptiveField  = 222; // receptive field in pixels
    static constexpr int overlap         = roundUp(receptiveField / 2, alignment); // required spatial overlap between tiles in pixels

    static constexpr int estimatedBytesBase       = 16*1024*1024; // estimated base memory usage
    static constexpr int estimatedBytesPerPixel8  = 889;          // estimated memory usage per pixel for K=8
    static constexpr int estimatedBytesPerPixel16 = 2185;         // estimated memory usage per pixel for K=16

    Image color;
    Image albedo;
    Image normal;
    Image output;
    bool hdr = false;
    float hdrScale = std::numeric_limits<float>::quiet_NaN();
    bool srgb = false;
    int maxMemoryMB = 6000; // approximate maximum memory usage in MBs

    int H = 0;          // image height
    int W = 0;          // image width
    int tileH = 0;      // tile height
    int tileW = 0;      // tile width
    int tileCountH = 1; // number of tiles in H dimension
    int tileCountW = 1; // number of tiles in W dimension

    std::shared_ptr<Executable> net;
    std::shared_ptr<Node> inputReorder;
    std::shared_ptr<Node> outputReorder;

    struct
    {
      void* ldr         = nullptr;
      void* ldr_alb     = nullptr;
      void* ldr_alb_nrm = nullptr;
      void* hdr         = nullptr;
      void* hdr_alb     = nullptr;
      void* hdr_alb_nrm = nullptr;
    } weightData;

    explicit AutoencoderFilter(const Ref<Device>& device);
    virtual std::shared_ptr<TransferFunction> makeTransferFunc();

  public:
    void setImage(const std::string& name, const Image& data) override;
    void set1i(const std::string& name, int value) override;
    int get1i(const std::string& name) override;
    void set1f(const std::string& name, float value) override;
    float get1f(const std::string& name) override;

    void commit() override;
    void execute() override;

  private:
    void computeTileSize();

    template<int K>
    std::shared_ptr<Executable> buildNet();

    bool isCommitted() const { return bool(net); }
  };

  // --------------------------------------------------------------------------
  // RTFilter - Generic ray tracing denoiser
  // --------------------------------------------------------------------------

// -- GODOT start --
// Godot doesn't need Raytracing filters. Removing them saves space in the weights files.
#if 0
// -- GODOT end --
  class RTFilter : public AutoencoderFilter
  {
  public:
    explicit RTFilter(const Ref<Device>& device);
  };
// -- GODOT start --
#endif
// -- GODOT end --

  // --------------------------------------------------------------------------
  // RTLightmapFilter - Ray traced lightmap denoiser
  // --------------------------------------------------------------------------

  class RTLightmapFilter : public AutoencoderFilter
  {
  public:
    explicit RTLightmapFilter(const Ref<Device>& device);
    std::shared_ptr<TransferFunction> makeTransferFunc() override;
  };

} // namespace oidn
