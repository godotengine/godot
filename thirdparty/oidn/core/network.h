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

#include "common/tensor.h"
#include "image.h"
#include "node.h"
#include "input_reorder.h"
#include "output_reorder.h"
#include "transfer_function.h"

#pragma once

namespace oidn {

  // Progress state
  struct Progress
  {
    ProgressMonitorFunction func;
    void* userPtr;
    int taskCount;
  };

  class Executable
  {
  public:
    virtual ~Executable() {}
    virtual void execute(const Progress& progress, int taskIndex) = 0;
  };

  template<int K>
  class Network : public Executable
  {
  public:
    Network(const Ref<Device>& device, const std::map<std::string, Tensor>& weightMap);

    void execute(const Progress& progress, int taskIndex) override;

    std::shared_ptr<memory> allocTensor(const memory::dims& dims,
                                        memory::format_tag format = memory::format_tag::any,
                                        void* data = nullptr);

    std::shared_ptr<memory> castTensor(const memory::dims& dims,
                                       const std::shared_ptr<memory>& src,
                                       size_t srcOffset = 0,
                                       memory::format_tag format = memory::format_tag::any);

    std::shared_ptr<memory> castTensor(const memory::dims& dims,
                                       const std::shared_ptr<memory>& src,
                                       const memory::dims& srcOffset);

    void zeroTensor(const std::shared_ptr<memory>& dst);

    memory::dims getInputReorderDims(const memory::dims& srcDims, int alignment);

    std::shared_ptr<Node> addInputReorder(const Image& color,
                                          const Image& albedo,
                                          const Image& normal,
                                          const std::shared_ptr<TransferFunction>& transferFunc,
                                          int alignment,
                                          const std::shared_ptr<memory>& userDst = nullptr);

    std::shared_ptr<Node> addOutputReorder(const std::shared_ptr<memory>& src,
                                           const std::shared_ptr<TransferFunction>& transferFunc,
                                           const Image& output);

    memory::dims getConvDims(const std::string& name, const memory::dims& srcDims);
    std::shared_ptr<Node> addConv(const std::string& name,
                                  const std::shared_ptr<memory>& src,
                                  const std::shared_ptr<memory>& userDst = nullptr,
                                  bool relu = true);

    memory::dims getPoolDims(const memory::dims& srcDims);
    std::shared_ptr<Node> addPool(const std::shared_ptr<memory>& src,
                                  const std::shared_ptr<memory>& userDst = nullptr);

    memory::dims getUpsampleDims(const memory::dims& srcDims);
    std::shared_ptr<Node> addUpsample(const std::shared_ptr<memory>& src,
                                      const std::shared_ptr<memory>& userDst = nullptr);

    memory::dims getConcatDims(const memory::dims& src1Dims, const memory::dims& src2Dims);

    std::shared_ptr<Node> addAutoexposure(const Image& color,
                                          const std::shared_ptr<HDRTransferFunction>& transferFunc);

    void finalize();

  private:
    Ref<Device> device;
    engine eng;
    stream sm;
    std::vector<std::shared_ptr<Node>> nodes;
    std::map<std::string, Tensor> weightMap;

    // Memory allocation statistics
    size_t activationAllocBytes = 0; // number of allocated activation bytes
    size_t totalAllocBytes      = 0; // total number of allocated bytes
  };

} // namespace oidn
