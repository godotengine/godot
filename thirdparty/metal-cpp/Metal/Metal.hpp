//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Metal/Metal.hpp
//
// Copyright 2020-2024 Apple Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//-------------------------------------------------------------------------------------------------------------------------------------------------------------

#pragma once

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

#include "MTLAccelerationStructure.hpp"
#include "MTLAccelerationStructureCommandEncoder.hpp"
#include "MTLAccelerationStructureTypes.hpp"
#include "MTLAllocation.hpp"
#include "MTLArgument.hpp"
#include "MTLArgumentEncoder.hpp"
#include "MTLBinaryArchive.hpp"
#include "MTLBlitCommandEncoder.hpp"
#include "MTLBlitPass.hpp"
#include "MTLBuffer.hpp"
#include "MTLCaptureManager.hpp"
#include "MTLCaptureScope.hpp"
#include "MTLCommandBuffer.hpp"
#include "MTLCommandEncoder.hpp"
#include "MTLCommandQueue.hpp"
#include "MTLComputeCommandEncoder.hpp"
#include "MTLComputePass.hpp"
#include "MTLComputePipeline.hpp"
#include "MTLCounters.hpp"
#include "MTLDefines.hpp"
#include "MTLDepthStencil.hpp"
#include "MTLDevice.hpp"
#include "MTLDrawable.hpp"
#include "MTLDynamicLibrary.hpp"
#include "MTLEvent.hpp"
#include "MTLFence.hpp"
#include "MTLFunctionConstantValues.hpp"
#include "MTLFunctionDescriptor.hpp"
#include "MTLFunctionHandle.hpp"
#include "MTLFunctionLog.hpp"
#include "MTLFunctionStitching.hpp"
#include "MTLHeaderBridge.hpp"
#include "MTLHeap.hpp"
#include "MTLIndirectCommandBuffer.hpp"
#include "MTLIndirectCommandEncoder.hpp"
#include "MTLIntersectionFunctionTable.hpp"
#include "MTLIOCommandBuffer.hpp"
#include "MTLIOCommandQueue.hpp"
#include "MTLIOCompressor.hpp"
#include "MTLLibrary.hpp"
#include "MTLLinkedFunctions.hpp"
#include "MTLLogState.hpp"
#include "MTLParallelRenderCommandEncoder.hpp"
#include "MTLPipeline.hpp"
#include "MTLPixelFormat.hpp"
#include "MTLPrivate.hpp"
#include "MTLRasterizationRate.hpp"
#include "MTLRenderCommandEncoder.hpp"
#include "MTLRenderPass.hpp"
#include "MTLRenderPipeline.hpp"
#include "MTLResidencySet.hpp"
#include "MTLResource.hpp"
#include "MTLResourceStateCommandEncoder.hpp"
#include "MTLResourceStatePass.hpp"
#include "MTLSampler.hpp"
#include "MTLStageInputOutputDescriptor.hpp"
#include "MTLTexture.hpp"
#include "MTLTypes.hpp"
#include "MTLVertexDescriptor.hpp"
#include "MTLVisibleFunctionTable.hpp"
#include "MTLVersion.hpp"
#include "MTLTensor.hpp"
#include "MTLResourceViewPool.hpp"
#include "MTLTextureViewPool.hpp"
#include "MTLDataType.hpp"
#include "MTL4ArgumentTable.hpp"
#include "MTL4BinaryFunction.hpp"
#include "MTL4CommandAllocator.hpp"
#include "MTL4CommandBuffer.hpp"
#include "MTL4CommandEncoder.hpp"
#include "MTL4CommandQueue.hpp"
#include "MTL4Counters.hpp"
#include "MTL4RenderPass.hpp"
#include "MTL4RenderCommandEncoder.hpp"
#include "MTL4ComputeCommandEncoder.hpp"
#include "MTL4MachineLearningCommandEncoder.hpp"
#include "MTL4Compiler.hpp"
#include "MTL4CompilerTask.hpp"
#include "MTL4LibraryDescriptor.hpp"
#include "MTL4FunctionDescriptor.hpp"
#include "MTL4LibraryFunctionDescriptor.hpp"
#include "MTL4SpecializedFunctionDescriptor.hpp"
#include "MTL4StitchedFunctionDescriptor.hpp"
#include "MTL4PipelineState.hpp"
#include "MTL4ComputePipeline.hpp"
#include "MTL4RenderPipeline.hpp"
#include "MTL4MachineLearningPipeline.hpp"
#include "MTL4TileRenderPipeline.hpp"
#include "MTL4MeshRenderPipeline.hpp"
#include "MTL4PipelineDataSetSerializer.hpp"
#include "MTL4Archive.hpp"
#include "MTL4CommitFeedback.hpp"
#include "MTL4BinaryFunctionDescriptor.hpp"
#include "MTL4LinkingDescriptor.hpp"

//-------------------------------------------------------------------------------------------------------------------------------------------------------------
