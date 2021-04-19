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

#include "autoencoder.h"

namespace oidn {

  // --------------------------------------------------------------------------
  // AutoencoderFilter
  // --------------------------------------------------------------------------

  AutoencoderFilter::AutoencoderFilter(const Ref<Device>& device)
    : Filter(device)
  {
  }

  void AutoencoderFilter::setImage(const std::string& name, const Image& data)
  {
    if (name == "color")
      color = data;
    else if (name == "albedo")
      albedo = data;
    else if (name == "normal")
      normal = data;
    else if (name == "output")
      output = data;

    dirty = true;
  }

  void AutoencoderFilter::set1i(const std::string& name, int value)
  {
    if (name == "hdr")
      hdr = value;
    else if (name == "srgb")
      srgb = value;
    else if (name == "maxMemoryMB")
      maxMemoryMB = value;

    dirty = true;
  }

  int AutoencoderFilter::get1i(const std::string& name)
  {
    if (name == "hdr")
      return hdr;
    else if (name == "srgb")
      return srgb;
    else if (name == "maxMemoryMB")
      return maxMemoryMB;
    else if (name == "alignment")
      return alignment;
    else if (name == "overlap")
      return overlap;
    else
      throw Exception(Error::InvalidArgument, "invalid parameter");
  }

  void AutoencoderFilter::set1f(const std::string& name, float value)
  {
    if (name == "hdrScale")
      hdrScale = value;

    dirty = true;
  }

  float AutoencoderFilter::get1f(const std::string& name)
  {
    if (name == "hdrScale")
      return hdrScale;
    else
      throw Exception(Error::InvalidArgument, "invalid parameter");
  }

  void AutoencoderFilter::commit()
  {
    if (!dirty)
      return;

    // -- GODOT start --
    //device->executeTask([&]()
    //{
    // GODOT end --

      if (mayiuse(avx512_common))
        net = buildNet<16>();
      else
        net = buildNet<8>();

    // GODOT start --    
    //});
    // GODOT end --

    dirty = false;
  }

  void AutoencoderFilter::execute()
  {
    if (dirty)
      throw Exception(Error::InvalidOperation, "changes to the filter are not committed");

    if (!net)
      return;
    // -- GODOT start --
    //device->executeTask([&]()
    //{
    // -- GODOT end --
      Progress progress;
      progress.func = progressFunc;
      progress.userPtr = progressUserPtr;
      progress.taskCount = tileCountH * tileCountW;

      // Iterate over the tiles
      int tileIndex = 0;

      for (int i = 0; i < tileCountH; ++i)
      {
        const int h = i * (tileH - 2*overlap); // input tile position (including overlap)
        const int overlapBeginH = i > 0            ? overlap : 0; // overlap on the top
        const int overlapEndH   = i < tileCountH-1 ? overlap : 0; // overlap on the bottom
        const int tileH1 = min(H - h, tileH); // input tile size (including overlap)
        const int tileH2 = tileH1 - overlapBeginH - overlapEndH; // output tile size
        const int alignOffsetH = tileH - roundUp(tileH1, alignment); // align to the bottom in the tile buffer

        for (int j = 0; j < tileCountW; ++j)
        {
          const int w = j * (tileW - 2*overlap); // input tile position (including overlap)
          const int overlapBeginW = j > 0            ? overlap : 0; // overlap on the left
          const int overlapEndW   = j < tileCountW-1 ? overlap : 0; // overlap on the right
          const int tileW1 = min(W - w, tileW); // input tile size (including overlap)
          const int tileW2 = tileW1 - overlapBeginW - overlapEndW; // output tile size
          const int alignOffsetW = tileW - roundUp(tileW1, alignment); // align to the right in the tile buffer

          // Set the input tile
          inputReorder->setTile(h, w,
                                alignOffsetH, alignOffsetW,
                                tileH1, tileW1);

          // Set the output tile
          outputReorder->setTile(alignOffsetH + overlapBeginH, alignOffsetW + overlapBeginW,
                                 h + overlapBeginH, w + overlapBeginW,
                                 tileH2, tileW2);

          //printf("Tile: %d %d -> %d %d\n", w+overlapBeginW, h+overlapBeginH, w+overlapBeginW+tileW2, h+overlapBeginH+tileH2);

          // Denoise the tile
          net->execute(progress, tileIndex);

          // Next tile
          tileIndex++;
        }
      }
    // -- GODOT start --
    //});
    // -- GODOT end --
  }

  void AutoencoderFilter::computeTileSize()
  {
    const int minTileSize = 3*overlap;
    const int estimatedBytesPerPixel = mayiuse(avx512_common) ? estimatedBytesPerPixel16 : estimatedBytesPerPixel8;
    const int64_t maxTilePixels = (int64_t(maxMemoryMB)*1024*1024 - estimatedBytesBase) / estimatedBytesPerPixel;

    tileCountH = 1;
    tileCountW = 1;
    tileH = roundUp(H, alignment);
    tileW = roundUp(W, alignment);

    // Divide the image into tiles until the tile size gets below the threshold
    while (int64_t(tileH) * tileW > maxTilePixels)
    {
      if (tileH > minTileSize && tileH > tileW)
      {
        tileCountH++;
        tileH = max(roundUp(ceilDiv(H - 2*overlap, tileCountH), alignment) + 2*overlap, minTileSize);
      }
      else if (tileW > minTileSize)
      {
        tileCountW++;
        tileW = max(roundUp(ceilDiv(W - 2*overlap, tileCountW), alignment) + 2*overlap, minTileSize);
      }
      else
        break;
    }

    // Compute the final number of tiles
    tileCountH = (H > tileH) ? ceilDiv(H - 2*overlap, tileH - 2*overlap) : 1;
    tileCountW = (W > tileW) ? ceilDiv(W - 2*overlap, tileW - 2*overlap) : 1;

    if (device->isVerbose(2))
    {
      std::cout << "Tile size : " << tileW << "x" << tileH << std::endl;
      std::cout << "Tile count: " << tileCountW << "x" << tileCountH << std::endl;
    }
  }

  template<int K>
  std::shared_ptr<Executable> AutoencoderFilter::buildNet()
  {
    H = color.height;
    W = color.width;

    // Configure the network
    int inputC;
    void* weightPtr;

    if (srgb && hdr)
      throw Exception(Error::InvalidOperation, "srgb and hdr modes cannot be enabled at the same time");

    if (color && !albedo && !normal && weightData.hdr)
    {
      inputC = 3;
      weightPtr = hdr ? weightData.hdr : weightData.ldr;
    }
    else if (color && albedo && !normal && weightData.hdr_alb)
    {
      inputC = 6;
      weightPtr = hdr ? weightData.hdr_alb : weightData.ldr_alb;
    }
    else if (color && albedo && normal && weightData.hdr_alb_nrm)
    {
      inputC = 9;
      weightPtr = hdr ? weightData.hdr_alb_nrm : weightData.ldr_alb_nrm;
    }
    else
    {
      throw Exception(Error::InvalidOperation, "unsupported combination of input features");
    }

    if (!output)
      throw Exception(Error::InvalidOperation, "output image not specified");

    if ((color.format != Format::Float3)
        || (albedo && albedo.format != Format::Float3)
        || (normal && normal.format != Format::Float3)
        || (output.format != Format::Float3))
      throw Exception(Error::InvalidOperation, "unsupported image format");

    if ((albedo && (albedo.width != W || albedo.height != H))
        || (normal && (normal.width != W || normal.height != H))
        || (output.width != W || output.height != H))
      throw Exception(Error::InvalidOperation, "image size mismatch");

    // Compute the tile size
    computeTileSize();

    // If the image size is zero, there is nothing else to do
    if (H <= 0 || W <= 0)
      return nullptr;

    // Parse the weights
    const auto weightMap = parseTensors(weightPtr);

    // Create the network
    std::shared_ptr<Network<K>> net = std::make_shared<Network<K>>(device, weightMap);

    // Compute the tensor sizes
    const auto inputDims        = memory::dims({1, inputC, tileH, tileW});
    const auto inputReorderDims = net->getInputReorderDims(inputDims, alignment);   //-> concat0

    const auto conv1Dims     = net->getConvDims("conv1", inputReorderDims);         //-> temp0
    const auto conv1bDims    = net->getConvDims("conv1b", conv1Dims);               //-> temp1
    const auto pool1Dims     = net->getPoolDims(conv1bDims);                        //-> concat1
    const auto conv2Dims     = net->getConvDims("conv2", pool1Dims);                //-> temp0
    const auto pool2Dims     = net->getPoolDims(conv2Dims);                         //-> concat2
    const auto conv3Dims     = net->getConvDims("conv3", pool2Dims);                //-> temp0
    const auto pool3Dims     = net->getPoolDims(conv3Dims);                         //-> concat3
    const auto conv4Dims     = net->getConvDims("conv4", pool3Dims);                //-> temp0
    const auto pool4Dims     = net->getPoolDims(conv4Dims);                         //-> concat4
    const auto conv5Dims     = net->getConvDims("conv5", pool4Dims);                //-> temp0
    const auto pool5Dims     = net->getPoolDims(conv5Dims);                         //-> temp1
    const auto upsample4Dims = net->getUpsampleDims(pool5Dims);                     //-> concat4
    const auto concat4Dims   = net->getConcatDims(upsample4Dims, pool4Dims);
    const auto conv6Dims     = net->getConvDims("conv6", concat4Dims);              //-> temp0
    const auto conv6bDims    = net->getConvDims("conv6b", conv6Dims);               //-> temp1
    const auto upsample3Dims = net->getUpsampleDims(conv6bDims);                    //-> concat3
    const auto concat3Dims   = net->getConcatDims(upsample3Dims, pool3Dims);
    const auto conv7Dims     = net->getConvDims("conv7", concat3Dims);              //-> temp0
    const auto conv7bDims    = net->getConvDims("conv7b", conv7Dims);               //-> temp1
    const auto upsample2Dims = net->getUpsampleDims(conv7bDims);                    //-> concat2
    const auto concat2Dims   = net->getConcatDims(upsample2Dims, pool2Dims);
    const auto conv8Dims     = net->getConvDims("conv8", concat2Dims);              //-> temp0
    const auto conv8bDims    = net->getConvDims("conv8b", conv8Dims);               //-> temp1
    const auto upsample1Dims = net->getUpsampleDims(conv8bDims);                    //-> concat1
    const auto concat1Dims   = net->getConcatDims(upsample1Dims, pool1Dims);
    const auto conv9Dims     = net->getConvDims("conv9", concat1Dims);              //-> temp0
    const auto conv9bDims    = net->getConvDims("conv9b", conv9Dims);               //-> temp1
    const auto upsample0Dims = net->getUpsampleDims(conv9bDims);                    //-> concat0
    const auto concat0Dims   = net->getConcatDims(upsample0Dims, inputReorderDims);
    const auto conv10Dims    = net->getConvDims("conv10", concat0Dims);             //-> temp0
    const auto conv10bDims   = net->getConvDims("conv10b", conv10Dims);             //-> temp1
    const auto conv11Dims    = net->getConvDims("conv11", conv10bDims);             //-> temp0

    const auto outputDims = memory::dims({1, 3, tileH, tileW});

    // Allocate two temporary ping-pong buffers to decrease memory usage
    const auto temp0Dims = getMaxTensorDims({
      conv1Dims,
      conv2Dims,
      conv3Dims,
      conv4Dims,
      conv5Dims,
      conv6Dims,
      conv7Dims,
      conv8Dims,
      conv9Dims,
      conv10Dims,
      conv11Dims
    });

    const auto temp1Dims = getMaxTensorDims({
      conv1bDims,
      pool5Dims,
      conv6bDims,
      conv7bDims,
      conv8bDims,
      conv9bDims,
      conv10bDims,
    });

    auto temp0 = net->allocTensor(temp0Dims);
    auto temp1 = net->allocTensor(temp1Dims);

    // Allocate enough memory to hold the concat outputs. Then use the first
    // half to hold the previous conv output and the second half to hold the
    // pool/orig image output. This works because everything is C dimension
    // outermost, padded to K floats, and all the concats are on the C dimension.
    auto concat0Dst = net->allocTensor(concat0Dims);
    auto concat1Dst = net->allocTensor(concat1Dims);
    auto concat2Dst = net->allocTensor(concat2Dims);
    auto concat3Dst = net->allocTensor(concat3Dims);
    auto concat4Dst = net->allocTensor(concat4Dims);

    // Transfer function
    std::shared_ptr<TransferFunction> transferFunc = makeTransferFunc();

    // Autoexposure
    if (auto tf = std::dynamic_pointer_cast<HDRTransferFunction>(transferFunc))
    {
      if (isnan(hdrScale))
        net->addAutoexposure(color, tf);
      else
        tf->setExposure(hdrScale);
    }

    // Input reorder
    auto inputReorderDst = net->castTensor(inputReorderDims, concat0Dst, upsample0Dims);
    inputReorder = net->addInputReorder(color, albedo, normal,
                                        transferFunc,
                                        alignment, inputReorderDst);

    // conv1
    auto conv1 = net->addConv("conv1", inputReorder->getDst(), temp0);

    // conv1b
    auto conv1b = net->addConv("conv1b", conv1->getDst(), temp1);

    // pool1
    // Adjust pointer for pool1 to eliminate concat1
    auto pool1Dst = net->castTensor(pool1Dims, concat1Dst, upsample1Dims);
    auto pool1 = net->addPool(conv1b->getDst(), pool1Dst);

    // conv2
    auto conv2 = net->addConv("conv2", pool1->getDst(), temp0);

    // pool2
    // Adjust pointer for pool2 to eliminate concat2
    auto pool2Dst = net->castTensor(pool2Dims, concat2Dst, upsample2Dims);
    auto pool2 = net->addPool(conv2->getDst(), pool2Dst);

    // conv3
    auto conv3 = net->addConv("conv3", pool2->getDst(), temp0);

    // pool3
    // Adjust pointer for pool3 to eliminate concat3
    auto pool3Dst = net->castTensor(pool3Dims, concat3Dst, upsample3Dims);
    auto pool3 = net->addPool(conv3->getDst(), pool3Dst);

    // conv4
    auto conv4 = net->addConv("conv4", pool3->getDst(), temp0);

    // pool4
    // Adjust pointer for pool4 to eliminate concat4
    auto pool4Dst = net->castTensor(pool4Dims, concat4Dst, upsample4Dims);
    auto pool4 = net->addPool(conv4->getDst(), pool4Dst);

    // conv5
    auto conv5 = net->addConv("conv5", pool4->getDst(), temp0);

    // pool5
    auto pool5 = net->addPool(conv5->getDst(), temp1);

    // upsample4
    auto upsample4Dst = net->castTensor(upsample4Dims, concat4Dst);
    auto upsample4 = net->addUpsample(pool5->getDst(), upsample4Dst);

    // conv6
    auto conv6 = net->addConv("conv6", concat4Dst, temp0);

    // conv6b
    auto conv6b = net->addConv("conv6b", conv6->getDst(), temp1);

    // upsample3
    auto upsample3Dst = net->castTensor(upsample3Dims, concat3Dst);
    auto upsample3 = net->addUpsample(conv6b->getDst(), upsample3Dst);

    // conv7
    auto conv7 = net->addConv("conv7", concat3Dst, temp0);

    // conv7b
    auto conv7b = net->addConv("conv7b", conv7->getDst(), temp1);

    // upsample2
    auto upsample2Dst = net->castTensor(upsample2Dims, concat2Dst);
    auto upsample2 = net->addUpsample(conv7b->getDst(), upsample2Dst);

    // conv8
    auto conv8 = net->addConv("conv8", concat2Dst, temp0);

    // conv8b
    auto conv8b = net->addConv("conv8b", conv8->getDst(), temp1);

    // upsample1
    auto upsample1Dst = net->castTensor(upsample1Dims, concat1Dst);
    auto upsample1 = net->addUpsample(conv8b->getDst(), upsample1Dst);

    // conv9
    auto conv9 = net->addConv("conv9", concat1Dst, temp0);

    // conv9b
    auto conv9b = net->addConv("conv9b", conv9->getDst(), temp1);

    // upsample0
    auto upsample0Dst = net->castTensor(upsample0Dims, concat0Dst);
    auto upsample0 = net->addUpsample(conv9b->getDst(), upsample0Dst);

    // conv10
    auto conv10 = net->addConv("conv10", concat0Dst, temp0);

    // conv10b
    auto conv10b = net->addConv("conv10b", conv10->getDst(), temp1);

    // conv11
    auto conv11 = net->addConv("conv11", conv10b->getDst(), temp0, false /* no relu */);

    // Output reorder
    outputReorder = net->addOutputReorder(conv11->getDst(), transferFunc, output);

    net->finalize();
    return net;
  }

  std::shared_ptr<TransferFunction> AutoencoderFilter::makeTransferFunc()
  {
    if (hdr)
      return std::make_shared<PQXTransferFunction>();
    else if (srgb)
      return std::make_shared<LinearTransferFunction>();
    else
      return std::make_shared<GammaTransferFunction>();
  }

// -- GODOT start --
// Godot doesn't need Raytracing filters. Removing them saves space in the weights files.
#if 0
// -- GODOT end --

  // --------------------------------------------------------------------------
  // RTFilter
  // --------------------------------------------------------------------------

  namespace weights
  {
    // LDR
    extern unsigned char rt_ldr[];         // color
    extern unsigned char rt_ldr_alb[];     // color, albedo
    extern unsigned char rt_ldr_alb_nrm[]; // color, albedo, normal

    // HDR
    extern unsigned char rt_hdr[];         // color
    extern unsigned char rt_hdr_alb[];     // color, albedo
    extern unsigned char rt_hdr_alb_nrm[]; // color, albedo, normal
  }

  RTFilter::RTFilter(const Ref<Device>& device)
    : AutoencoderFilter(device)
  {
    weightData.ldr         = weights::rt_ldr;
    weightData.ldr_alb     = weights::rt_ldr_alb;
    weightData.ldr_alb_nrm = weights::rt_ldr_alb_nrm;
    weightData.hdr         = weights::rt_hdr;
    weightData.hdr_alb     = weights::rt_hdr_alb;
    weightData.hdr_alb_nrm = weights::rt_hdr_alb_nrm;
  }
// -- GODOT start --
#endif
// -- GODOT end --

  // --------------------------------------------------------------------------
  // RTLightmapFilter
  // --------------------------------------------------------------------------

  namespace weights
  {
    // HDR
    extern unsigned char rtlightmap_hdr[]; // color
  }

  RTLightmapFilter::RTLightmapFilter(const Ref<Device>& device)
    : AutoencoderFilter(device)
  {
    weightData.hdr = weights::rtlightmap_hdr;

    hdr = true;
  }

  std::shared_ptr<TransferFunction> RTLightmapFilter::makeTransferFunc()
  {
    return std::make_shared<LogTransferFunction>();
  }

} // namespace oidn
