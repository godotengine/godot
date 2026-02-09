//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Metal/MTLRenderPipeline.hpp
//
// Copyright 2020-2025 Apple Inc.
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

#include "../Foundation/Foundation.hpp"
#include "MTLAllocation.hpp"
#include "MTLDefines.hpp"
#include "MTLHeaderBridge.hpp"
#include "MTLPipeline.hpp"
#include "MTLPixelFormat.hpp"
#include "MTLPrivate.hpp"
#include "MTLRenderCommandEncoder.hpp"
#include "MTLTypes.hpp"

namespace MTL
{
class Device;
class Function;
class FunctionHandle;
class IntersectionFunctionTable;
class IntersectionFunctionTableDescriptor;
class LinkedFunctions;
class LogicalToPhysicalColorAttachmentMap;
class MeshRenderPipelineDescriptor;
class PipelineBufferDescriptorArray;
class RenderPipelineColorAttachmentDescriptor;
class RenderPipelineColorAttachmentDescriptorArray;
class RenderPipelineDescriptor;
class RenderPipelineFunctionsDescriptor;
class RenderPipelineReflection;
class RenderPipelineState;
class TileRenderPipelineColorAttachmentDescriptor;
class TileRenderPipelineColorAttachmentDescriptorArray;
class TileRenderPipelineDescriptor;
class VertexDescriptor;
class VisibleFunctionTable;
class VisibleFunctionTableDescriptor;

}
namespace MTL4
{
class BinaryFunction;
class PipelineDescriptor;
class RenderPipelineBinaryFunctionsDescriptor;

}
namespace MTL
{
_MTL_ENUM(NS::UInteger, BlendFactor) {
    BlendFactorZero = 0,
    BlendFactorOne = 1,
    BlendFactorSourceColor = 2,
    BlendFactorOneMinusSourceColor = 3,
    BlendFactorSourceAlpha = 4,
    BlendFactorOneMinusSourceAlpha = 5,
    BlendFactorDestinationColor = 6,
    BlendFactorOneMinusDestinationColor = 7,
    BlendFactorDestinationAlpha = 8,
    BlendFactorOneMinusDestinationAlpha = 9,
    BlendFactorSourceAlphaSaturated = 10,
    BlendFactorBlendColor = 11,
    BlendFactorOneMinusBlendColor = 12,
    BlendFactorBlendAlpha = 13,
    BlendFactorOneMinusBlendAlpha = 14,
    BlendFactorSource1Color = 15,
    BlendFactorOneMinusSource1Color = 16,
    BlendFactorSource1Alpha = 17,
    BlendFactorOneMinusSource1Alpha = 18,
    BlendFactorUnspecialized = 19,
};

_MTL_ENUM(NS::UInteger, BlendOperation) {
    BlendOperationAdd = 0,
    BlendOperationSubtract = 1,
    BlendOperationReverseSubtract = 2,
    BlendOperationMin = 3,
    BlendOperationMax = 4,
    BlendOperationUnspecialized = 5,
};

_MTL_ENUM(NS::UInteger, PrimitiveTopologyClass) {
    PrimitiveTopologyClassUnspecified = 0,
    PrimitiveTopologyClassPoint = 1,
    PrimitiveTopologyClassLine = 2,
    PrimitiveTopologyClassTriangle = 3,
};

_MTL_ENUM(NS::UInteger, TessellationPartitionMode) {
    TessellationPartitionModePow2 = 0,
    TessellationPartitionModeInteger = 1,
    TessellationPartitionModeFractionalOdd = 2,
    TessellationPartitionModeFractionalEven = 3,
};

_MTL_ENUM(NS::UInteger, TessellationFactorStepFunction) {
    TessellationFactorStepFunctionConstant = 0,
    TessellationFactorStepFunctionPerPatch = 1,
    TessellationFactorStepFunctionPerInstance = 2,
    TessellationFactorStepFunctionPerPatchAndPerInstance = 3,
};

_MTL_ENUM(NS::UInteger, TessellationFactorFormat) {
    TessellationFactorFormatHalf = 0,
};

_MTL_ENUM(NS::UInteger, TessellationControlPointIndexType) {
    TessellationControlPointIndexTypeNone = 0,
    TessellationControlPointIndexTypeUInt16 = 1,
    TessellationControlPointIndexTypeUInt32 = 2,
};

_MTL_OPTIONS(NS::UInteger, ColorWriteMask) {
    ColorWriteMaskNone = 0,
    ColorWriteMaskRed = 1 << 3,
    ColorWriteMaskGreen = 1 << 2,
    ColorWriteMaskBlue = 1 << 1,
    ColorWriteMaskAlpha = 1,
    ColorWriteMaskAll = 15,
    ColorWriteMaskUnspecialized = 1 << 4,
};

class RenderPipelineColorAttachmentDescriptor : public NS::Copying<RenderPipelineColorAttachmentDescriptor>
{
public:
    static RenderPipelineColorAttachmentDescriptor* alloc();

    BlendOperation                                  alphaBlendOperation() const;

    [[deprecated("please use isBlendingEnabled instead")]]
    bool                                     blendingEnabled() const;

    BlendFactor                              destinationAlphaBlendFactor() const;

    BlendFactor                              destinationRGBBlendFactor() const;

    RenderPipelineColorAttachmentDescriptor* init();

    bool                                     isBlendingEnabled() const;

    PixelFormat                              pixelFormat() const;

    BlendOperation                           rgbBlendOperation() const;

    void                                     setAlphaBlendOperation(MTL::BlendOperation alphaBlendOperation);

    void                                     setBlendingEnabled(bool blendingEnabled);

    void                                     setDestinationAlphaBlendFactor(MTL::BlendFactor destinationAlphaBlendFactor);

    void                                     setDestinationRGBBlendFactor(MTL::BlendFactor destinationRGBBlendFactor);

    void                                     setPixelFormat(MTL::PixelFormat pixelFormat);

    void                                     setRgbBlendOperation(MTL::BlendOperation rgbBlendOperation);

    void                                     setSourceAlphaBlendFactor(MTL::BlendFactor sourceAlphaBlendFactor);

    void                                     setSourceRGBBlendFactor(MTL::BlendFactor sourceRGBBlendFactor);

    void                                     setWriteMask(MTL::ColorWriteMask writeMask);

    BlendFactor                              sourceAlphaBlendFactor() const;

    BlendFactor                              sourceRGBBlendFactor() const;

    ColorWriteMask                           writeMask() const;
};
class LogicalToPhysicalColorAttachmentMap : public NS::Copying<LogicalToPhysicalColorAttachmentMap>
{
public:
    static LogicalToPhysicalColorAttachmentMap* alloc();

    NS::UInteger                                getPhysicalIndex(NS::UInteger logicalIndex);

    LogicalToPhysicalColorAttachmentMap*        init();

    void                                        reset();

    void                                        setPhysicalIndex(NS::UInteger physicalIndex, NS::UInteger logicalIndex);
};
class RenderPipelineReflection : public NS::Referencing<RenderPipelineReflection>
{
public:
    static RenderPipelineReflection* alloc();

    NS::Array*                       fragmentArguments() const;

    NS::Array*                       fragmentBindings() const;

    RenderPipelineReflection*        init();

    NS::Array*                       meshBindings() const;

    NS::Array*                       objectBindings() const;

    NS::Array*                       tileArguments() const;

    NS::Array*                       tileBindings() const;

    NS::Array*                       vertexArguments() const;

    NS::Array*                       vertexBindings() const;
};
class RenderPipelineDescriptor : public NS::Copying<RenderPipelineDescriptor>
{
public:
    static RenderPipelineDescriptor* alloc();

    [[deprecated("please use isAlphaToCoverageEnabled instead")]]
    bool alphaToCoverageEnabled() const;

    [[deprecated("please use isAlphaToOneEnabled instead")]]
    bool                                          alphaToOneEnabled() const;

    NS::Array*                                    binaryArchives() const;

    RenderPipelineColorAttachmentDescriptorArray* colorAttachments() const;

    PixelFormat                                   depthAttachmentPixelFormat() const;

    PipelineBufferDescriptorArray*                fragmentBuffers() const;

    Function*                                     fragmentFunction() const;

    LinkedFunctions*                              fragmentLinkedFunctions() const;

    NS::Array*                                    fragmentPreloadedLibraries() const;

    RenderPipelineDescriptor*                     init();

    PrimitiveTopologyClass                        inputPrimitiveTopology() const;

    bool                                          isAlphaToCoverageEnabled() const;

    bool                                          isAlphaToOneEnabled() const;

    bool                                          isRasterizationEnabled() const;

    bool                                          isTessellationFactorScaleEnabled() const;

    NS::String*                                   label() const;

    NS::UInteger                                  maxFragmentCallStackDepth() const;

    NS::UInteger                                  maxTessellationFactor() const;

    NS::UInteger                                  maxVertexAmplificationCount() const;

    NS::UInteger                                  maxVertexCallStackDepth() const;

    NS::UInteger                                  rasterSampleCount() const;

    [[deprecated("please use isRasterizationEnabled instead")]]
    bool                              rasterizationEnabled() const;

    void                              reset();

    NS::UInteger                      sampleCount() const;

    void                              setAlphaToCoverageEnabled(bool alphaToCoverageEnabled);

    void                              setAlphaToOneEnabled(bool alphaToOneEnabled);

    void                              setBinaryArchives(const NS::Array* binaryArchives);

    void                              setDepthAttachmentPixelFormat(MTL::PixelFormat depthAttachmentPixelFormat);

    void                              setFragmentFunction(const MTL::Function* fragmentFunction);

    void                              setFragmentLinkedFunctions(const MTL::LinkedFunctions* fragmentLinkedFunctions);

    void                              setFragmentPreloadedLibraries(const NS::Array* fragmentPreloadedLibraries);

    void                              setInputPrimitiveTopology(MTL::PrimitiveTopologyClass inputPrimitiveTopology);

    void                              setLabel(const NS::String* label);

    void                              setMaxFragmentCallStackDepth(NS::UInteger maxFragmentCallStackDepth);

    void                              setMaxTessellationFactor(NS::UInteger maxTessellationFactor);

    void                              setMaxVertexAmplificationCount(NS::UInteger maxVertexAmplificationCount);

    void                              setMaxVertexCallStackDepth(NS::UInteger maxVertexCallStackDepth);

    void                              setRasterSampleCount(NS::UInteger rasterSampleCount);

    void                              setRasterizationEnabled(bool rasterizationEnabled);

    void                              setSampleCount(NS::UInteger sampleCount);

    void                              setShaderValidation(MTL::ShaderValidation shaderValidation);

    void                              setStencilAttachmentPixelFormat(MTL::PixelFormat stencilAttachmentPixelFormat);

    void                              setSupportAddingFragmentBinaryFunctions(bool supportAddingFragmentBinaryFunctions);

    void                              setSupportAddingVertexBinaryFunctions(bool supportAddingVertexBinaryFunctions);

    void                              setSupportIndirectCommandBuffers(bool supportIndirectCommandBuffers);

    void                              setTessellationControlPointIndexType(MTL::TessellationControlPointIndexType tessellationControlPointIndexType);

    void                              setTessellationFactorFormat(MTL::TessellationFactorFormat tessellationFactorFormat);

    void                              setTessellationFactorScaleEnabled(bool tessellationFactorScaleEnabled);

    void                              setTessellationFactorStepFunction(MTL::TessellationFactorStepFunction tessellationFactorStepFunction);

    void                              setTessellationOutputWindingOrder(MTL::Winding tessellationOutputWindingOrder);

    void                              setTessellationPartitionMode(MTL::TessellationPartitionMode tessellationPartitionMode);

    void                              setVertexDescriptor(const MTL::VertexDescriptor* vertexDescriptor);

    void                              setVertexFunction(const MTL::Function* vertexFunction);

    void                              setVertexLinkedFunctions(const MTL::LinkedFunctions* vertexLinkedFunctions);

    void                              setVertexPreloadedLibraries(const NS::Array* vertexPreloadedLibraries);

    ShaderValidation                  shaderValidation() const;

    PixelFormat                       stencilAttachmentPixelFormat() const;

    bool                              supportAddingFragmentBinaryFunctions() const;

    bool                              supportAddingVertexBinaryFunctions() const;

    bool                              supportIndirectCommandBuffers() const;

    TessellationControlPointIndexType tessellationControlPointIndexType() const;

    TessellationFactorFormat          tessellationFactorFormat() const;

    [[deprecated("please use isTessellationFactorScaleEnabled instead")]]
    bool                           tessellationFactorScaleEnabled() const;

    TessellationFactorStepFunction tessellationFactorStepFunction() const;

    Winding                        tessellationOutputWindingOrder() const;

    TessellationPartitionMode      tessellationPartitionMode() const;

    PipelineBufferDescriptorArray* vertexBuffers() const;

    VertexDescriptor*              vertexDescriptor() const;

    Function*                      vertexFunction() const;

    LinkedFunctions*               vertexLinkedFunctions() const;

    NS::Array*                     vertexPreloadedLibraries() const;
};
class RenderPipelineFunctionsDescriptor : public NS::Copying<RenderPipelineFunctionsDescriptor>
{
public:
    static RenderPipelineFunctionsDescriptor* alloc();

    NS::Array*                                fragmentAdditionalBinaryFunctions() const;

    RenderPipelineFunctionsDescriptor*        init();

    void                                      setFragmentAdditionalBinaryFunctions(const NS::Array* fragmentAdditionalBinaryFunctions);

    void                                      setTileAdditionalBinaryFunctions(const NS::Array* tileAdditionalBinaryFunctions);

    void                                      setVertexAdditionalBinaryFunctions(const NS::Array* vertexAdditionalBinaryFunctions);

    NS::Array*                                tileAdditionalBinaryFunctions() const;

    NS::Array*                                vertexAdditionalBinaryFunctions() const;
};
class RenderPipelineState : public NS::Referencing<RenderPipelineState, Allocation>
{
public:
    Device*                    device() const;

    FunctionHandle*            functionHandle(const NS::String* name, MTL::RenderStages stage);
    FunctionHandle*            functionHandle(const MTL4::BinaryFunction* function, MTL::RenderStages stage);
    FunctionHandle*            functionHandle(const MTL::Function* function, MTL::RenderStages stage);

    ResourceID                 gpuResourceID() const;

    NS::UInteger               imageblockMemoryLength(MTL::Size imageblockDimensions);

    NS::UInteger               imageblockSampleLength() const;

    NS::String*                label() const;

    NS::UInteger               maxTotalThreadgroupsPerMeshGrid() const;

    NS::UInteger               maxTotalThreadsPerMeshThreadgroup() const;

    NS::UInteger               maxTotalThreadsPerObjectThreadgroup() const;

    NS::UInteger               maxTotalThreadsPerThreadgroup() const;

    NS::UInteger               meshThreadExecutionWidth() const;

    IntersectionFunctionTable* newIntersectionFunctionTable(const MTL::IntersectionFunctionTableDescriptor* descriptor, MTL::RenderStages stage);

    MTL4::PipelineDescriptor*  newRenderPipelineDescriptor();

    RenderPipelineState*       newRenderPipelineState(const MTL4::RenderPipelineBinaryFunctionsDescriptor* binaryFunctionsDescriptor, NS::Error** error);
    RenderPipelineState*       newRenderPipelineState(const MTL::RenderPipelineFunctionsDescriptor* additionalBinaryFunctions, NS::Error** error);

    VisibleFunctionTable*      newVisibleFunctionTable(const MTL::VisibleFunctionTableDescriptor* descriptor, MTL::RenderStages stage);

    NS::UInteger               objectThreadExecutionWidth() const;

    RenderPipelineReflection*  reflection() const;

    Size                       requiredThreadsPerMeshThreadgroup() const;

    Size                       requiredThreadsPerObjectThreadgroup() const;

    Size                       requiredThreadsPerTileThreadgroup() const;

    ShaderValidation           shaderValidation() const;

    bool                       supportIndirectCommandBuffers() const;

    bool                       threadgroupSizeMatchesTileSize() const;
};
class RenderPipelineColorAttachmentDescriptorArray : public NS::Referencing<RenderPipelineColorAttachmentDescriptorArray>
{
public:
    static RenderPipelineColorAttachmentDescriptorArray* alloc();

    RenderPipelineColorAttachmentDescriptorArray*        init();

    RenderPipelineColorAttachmentDescriptor*             object(NS::UInteger attachmentIndex);
    void                                                 setObject(const MTL::RenderPipelineColorAttachmentDescriptor* attachment, NS::UInteger attachmentIndex);
};
class TileRenderPipelineColorAttachmentDescriptor : public NS::Copying<TileRenderPipelineColorAttachmentDescriptor>
{
public:
    static TileRenderPipelineColorAttachmentDescriptor* alloc();

    TileRenderPipelineColorAttachmentDescriptor*        init();

    PixelFormat                                         pixelFormat() const;
    void                                                setPixelFormat(MTL::PixelFormat pixelFormat);
};
class TileRenderPipelineColorAttachmentDescriptorArray : public NS::Referencing<TileRenderPipelineColorAttachmentDescriptorArray>
{
public:
    static TileRenderPipelineColorAttachmentDescriptorArray* alloc();

    TileRenderPipelineColorAttachmentDescriptorArray*        init();

    TileRenderPipelineColorAttachmentDescriptor*             object(NS::UInteger attachmentIndex);
    void                                                     setObject(const MTL::TileRenderPipelineColorAttachmentDescriptor* attachment, NS::UInteger attachmentIndex);
};
class TileRenderPipelineDescriptor : public NS::Copying<TileRenderPipelineDescriptor>
{
public:
    static TileRenderPipelineDescriptor*              alloc();

    NS::Array*                                        binaryArchives() const;

    TileRenderPipelineColorAttachmentDescriptorArray* colorAttachments() const;

    TileRenderPipelineDescriptor*                     init();

    NS::String*                                       label() const;

    LinkedFunctions*                                  linkedFunctions() const;

    NS::UInteger                                      maxCallStackDepth() const;

    NS::UInteger                                      maxTotalThreadsPerThreadgroup() const;

    NS::Array*                                        preloadedLibraries() const;

    NS::UInteger                                      rasterSampleCount() const;

    Size                                              requiredThreadsPerThreadgroup() const;

    void                                              reset();

    void                                              setBinaryArchives(const NS::Array* binaryArchives);

    void                                              setLabel(const NS::String* label);

    void                                              setLinkedFunctions(const MTL::LinkedFunctions* linkedFunctions);

    void                                              setMaxCallStackDepth(NS::UInteger maxCallStackDepth);

    void                                              setMaxTotalThreadsPerThreadgroup(NS::UInteger maxTotalThreadsPerThreadgroup);

    void                                              setPreloadedLibraries(const NS::Array* preloadedLibraries);

    void                                              setRasterSampleCount(NS::UInteger rasterSampleCount);

    void                                              setRequiredThreadsPerThreadgroup(MTL::Size requiredThreadsPerThreadgroup);

    void                                              setShaderValidation(MTL::ShaderValidation shaderValidation);

    void                                              setSupportAddingBinaryFunctions(bool supportAddingBinaryFunctions);

    void                                              setThreadgroupSizeMatchesTileSize(bool threadgroupSizeMatchesTileSize);

    void                                              setTileFunction(const MTL::Function* tileFunction);

    ShaderValidation                                  shaderValidation() const;

    bool                                              supportAddingBinaryFunctions() const;

    bool                                              threadgroupSizeMatchesTileSize() const;

    PipelineBufferDescriptorArray*                    tileBuffers() const;

    Function*                                         tileFunction() const;
};
class MeshRenderPipelineDescriptor : public NS::Copying<MeshRenderPipelineDescriptor>
{
public:
    static MeshRenderPipelineDescriptor* alloc();

    [[deprecated("please use isAlphaToCoverageEnabled instead")]]
    bool alphaToCoverageEnabled() const;

    [[deprecated("please use isAlphaToOneEnabled instead")]]
    bool                                          alphaToOneEnabled() const;

    NS::Array*                                    binaryArchives() const;

    RenderPipelineColorAttachmentDescriptorArray* colorAttachments() const;

    PixelFormat                                   depthAttachmentPixelFormat() const;

    PipelineBufferDescriptorArray*                fragmentBuffers() const;

    Function*                                     fragmentFunction() const;

    LinkedFunctions*                              fragmentLinkedFunctions() const;

    MeshRenderPipelineDescriptor*                 init();

    bool                                          isAlphaToCoverageEnabled() const;

    bool                                          isAlphaToOneEnabled() const;

    bool                                          isRasterizationEnabled() const;

    NS::String*                                   label() const;

    NS::UInteger                                  maxTotalThreadgroupsPerMeshGrid() const;

    NS::UInteger                                  maxTotalThreadsPerMeshThreadgroup() const;

    NS::UInteger                                  maxTotalThreadsPerObjectThreadgroup() const;

    NS::UInteger                                  maxVertexAmplificationCount() const;

    PipelineBufferDescriptorArray*                meshBuffers() const;

    Function*                                     meshFunction() const;

    LinkedFunctions*                              meshLinkedFunctions() const;

    bool                                          meshThreadgroupSizeIsMultipleOfThreadExecutionWidth() const;

    PipelineBufferDescriptorArray*                objectBuffers() const;

    Function*                                     objectFunction() const;

    LinkedFunctions*                              objectLinkedFunctions() const;

    bool                                          objectThreadgroupSizeIsMultipleOfThreadExecutionWidth() const;

    NS::UInteger                                  payloadMemoryLength() const;

    NS::UInteger                                  rasterSampleCount() const;

    [[deprecated("please use isRasterizationEnabled instead")]]
    bool             rasterizationEnabled() const;

    Size             requiredThreadsPerMeshThreadgroup() const;

    Size             requiredThreadsPerObjectThreadgroup() const;

    void             reset();

    void             setAlphaToCoverageEnabled(bool alphaToCoverageEnabled);

    void             setAlphaToOneEnabled(bool alphaToOneEnabled);

    void             setBinaryArchives(const NS::Array* binaryArchives);

    void             setDepthAttachmentPixelFormat(MTL::PixelFormat depthAttachmentPixelFormat);

    void             setFragmentFunction(const MTL::Function* fragmentFunction);

    void             setFragmentLinkedFunctions(const MTL::LinkedFunctions* fragmentLinkedFunctions);

    void             setLabel(const NS::String* label);

    void             setMaxTotalThreadgroupsPerMeshGrid(NS::UInteger maxTotalThreadgroupsPerMeshGrid);

    void             setMaxTotalThreadsPerMeshThreadgroup(NS::UInteger maxTotalThreadsPerMeshThreadgroup);

    void             setMaxTotalThreadsPerObjectThreadgroup(NS::UInteger maxTotalThreadsPerObjectThreadgroup);

    void             setMaxVertexAmplificationCount(NS::UInteger maxVertexAmplificationCount);

    void             setMeshFunction(const MTL::Function* meshFunction);

    void             setMeshLinkedFunctions(const MTL::LinkedFunctions* meshLinkedFunctions);

    void             setMeshThreadgroupSizeIsMultipleOfThreadExecutionWidth(bool meshThreadgroupSizeIsMultipleOfThreadExecutionWidth);

    void             setObjectFunction(const MTL::Function* objectFunction);

    void             setObjectLinkedFunctions(const MTL::LinkedFunctions* objectLinkedFunctions);

    void             setObjectThreadgroupSizeIsMultipleOfThreadExecutionWidth(bool objectThreadgroupSizeIsMultipleOfThreadExecutionWidth);

    void             setPayloadMemoryLength(NS::UInteger payloadMemoryLength);

    void             setRasterSampleCount(NS::UInteger rasterSampleCount);

    void             setRasterizationEnabled(bool rasterizationEnabled);

    void             setRequiredThreadsPerMeshThreadgroup(MTL::Size requiredThreadsPerMeshThreadgroup);

    void             setRequiredThreadsPerObjectThreadgroup(MTL::Size requiredThreadsPerObjectThreadgroup);

    void             setShaderValidation(MTL::ShaderValidation shaderValidation);

    void             setStencilAttachmentPixelFormat(MTL::PixelFormat stencilAttachmentPixelFormat);

    void             setSupportIndirectCommandBuffers(bool supportIndirectCommandBuffers);

    ShaderValidation shaderValidation() const;

    PixelFormat      stencilAttachmentPixelFormat() const;

    bool             supportIndirectCommandBuffers() const;
};

}
_MTL_INLINE MTL::RenderPipelineColorAttachmentDescriptor* MTL::RenderPipelineColorAttachmentDescriptor::alloc()
{
    return NS::Object::alloc<MTL::RenderPipelineColorAttachmentDescriptor>(_MTL_PRIVATE_CLS(MTLRenderPipelineColorAttachmentDescriptor));
}

_MTL_INLINE MTL::BlendOperation MTL::RenderPipelineColorAttachmentDescriptor::alphaBlendOperation() const
{
    return Object::sendMessage<MTL::BlendOperation>(this, _MTL_PRIVATE_SEL(alphaBlendOperation));
}

_MTL_INLINE bool MTL::RenderPipelineColorAttachmentDescriptor::blendingEnabled() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(isBlendingEnabled));
}

_MTL_INLINE MTL::BlendFactor MTL::RenderPipelineColorAttachmentDescriptor::destinationAlphaBlendFactor() const
{
    return Object::sendMessage<MTL::BlendFactor>(this, _MTL_PRIVATE_SEL(destinationAlphaBlendFactor));
}

_MTL_INLINE MTL::BlendFactor MTL::RenderPipelineColorAttachmentDescriptor::destinationRGBBlendFactor() const
{
    return Object::sendMessage<MTL::BlendFactor>(this, _MTL_PRIVATE_SEL(destinationRGBBlendFactor));
}

_MTL_INLINE MTL::RenderPipelineColorAttachmentDescriptor* MTL::RenderPipelineColorAttachmentDescriptor::init()
{
    return NS::Object::init<MTL::RenderPipelineColorAttachmentDescriptor>();
}

_MTL_INLINE bool MTL::RenderPipelineColorAttachmentDescriptor::isBlendingEnabled() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(isBlendingEnabled));
}

_MTL_INLINE MTL::PixelFormat MTL::RenderPipelineColorAttachmentDescriptor::pixelFormat() const
{
    return Object::sendMessage<MTL::PixelFormat>(this, _MTL_PRIVATE_SEL(pixelFormat));
}

_MTL_INLINE MTL::BlendOperation MTL::RenderPipelineColorAttachmentDescriptor::rgbBlendOperation() const
{
    return Object::sendMessage<MTL::BlendOperation>(this, _MTL_PRIVATE_SEL(rgbBlendOperation));
}

_MTL_INLINE void MTL::RenderPipelineColorAttachmentDescriptor::setAlphaBlendOperation(MTL::BlendOperation alphaBlendOperation)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setAlphaBlendOperation_), alphaBlendOperation);
}

_MTL_INLINE void MTL::RenderPipelineColorAttachmentDescriptor::setBlendingEnabled(bool blendingEnabled)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setBlendingEnabled_), blendingEnabled);
}

_MTL_INLINE void MTL::RenderPipelineColorAttachmentDescriptor::setDestinationAlphaBlendFactor(MTL::BlendFactor destinationAlphaBlendFactor)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setDestinationAlphaBlendFactor_), destinationAlphaBlendFactor);
}

_MTL_INLINE void MTL::RenderPipelineColorAttachmentDescriptor::setDestinationRGBBlendFactor(MTL::BlendFactor destinationRGBBlendFactor)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setDestinationRGBBlendFactor_), destinationRGBBlendFactor);
}

_MTL_INLINE void MTL::RenderPipelineColorAttachmentDescriptor::setPixelFormat(MTL::PixelFormat pixelFormat)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setPixelFormat_), pixelFormat);
}

_MTL_INLINE void MTL::RenderPipelineColorAttachmentDescriptor::setRgbBlendOperation(MTL::BlendOperation rgbBlendOperation)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setRgbBlendOperation_), rgbBlendOperation);
}

_MTL_INLINE void MTL::RenderPipelineColorAttachmentDescriptor::setSourceAlphaBlendFactor(MTL::BlendFactor sourceAlphaBlendFactor)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setSourceAlphaBlendFactor_), sourceAlphaBlendFactor);
}

_MTL_INLINE void MTL::RenderPipelineColorAttachmentDescriptor::setSourceRGBBlendFactor(MTL::BlendFactor sourceRGBBlendFactor)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setSourceRGBBlendFactor_), sourceRGBBlendFactor);
}

_MTL_INLINE void MTL::RenderPipelineColorAttachmentDescriptor::setWriteMask(MTL::ColorWriteMask writeMask)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setWriteMask_), writeMask);
}

_MTL_INLINE MTL::BlendFactor MTL::RenderPipelineColorAttachmentDescriptor::sourceAlphaBlendFactor() const
{
    return Object::sendMessage<MTL::BlendFactor>(this, _MTL_PRIVATE_SEL(sourceAlphaBlendFactor));
}

_MTL_INLINE MTL::BlendFactor MTL::RenderPipelineColorAttachmentDescriptor::sourceRGBBlendFactor() const
{
    return Object::sendMessage<MTL::BlendFactor>(this, _MTL_PRIVATE_SEL(sourceRGBBlendFactor));
}

_MTL_INLINE MTL::ColorWriteMask MTL::RenderPipelineColorAttachmentDescriptor::writeMask() const
{
    return Object::sendMessage<MTL::ColorWriteMask>(this, _MTL_PRIVATE_SEL(writeMask));
}

_MTL_INLINE MTL::LogicalToPhysicalColorAttachmentMap* MTL::LogicalToPhysicalColorAttachmentMap::alloc()
{
    return NS::Object::alloc<MTL::LogicalToPhysicalColorAttachmentMap>(_MTL_PRIVATE_CLS(MTLLogicalToPhysicalColorAttachmentMap));
}

_MTL_INLINE NS::UInteger MTL::LogicalToPhysicalColorAttachmentMap::getPhysicalIndex(NS::UInteger logicalIndex)
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(getPhysicalIndexForLogicalIndex_), logicalIndex);
}

_MTL_INLINE MTL::LogicalToPhysicalColorAttachmentMap* MTL::LogicalToPhysicalColorAttachmentMap::init()
{
    return NS::Object::init<MTL::LogicalToPhysicalColorAttachmentMap>();
}

_MTL_INLINE void MTL::LogicalToPhysicalColorAttachmentMap::reset()
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(reset));
}

_MTL_INLINE void MTL::LogicalToPhysicalColorAttachmentMap::setPhysicalIndex(NS::UInteger physicalIndex, NS::UInteger logicalIndex)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setPhysicalIndex_forLogicalIndex_), physicalIndex, logicalIndex);
}

_MTL_INLINE MTL::RenderPipelineReflection* MTL::RenderPipelineReflection::alloc()
{
    return NS::Object::alloc<MTL::RenderPipelineReflection>(_MTL_PRIVATE_CLS(MTLRenderPipelineReflection));
}

_MTL_INLINE NS::Array* MTL::RenderPipelineReflection::fragmentArguments() const
{
    return Object::sendMessage<NS::Array*>(this, _MTL_PRIVATE_SEL(fragmentArguments));
}

_MTL_INLINE NS::Array* MTL::RenderPipelineReflection::fragmentBindings() const
{
    return Object::sendMessage<NS::Array*>(this, _MTL_PRIVATE_SEL(fragmentBindings));
}

_MTL_INLINE MTL::RenderPipelineReflection* MTL::RenderPipelineReflection::init()
{
    return NS::Object::init<MTL::RenderPipelineReflection>();
}

_MTL_INLINE NS::Array* MTL::RenderPipelineReflection::meshBindings() const
{
    return Object::sendMessage<NS::Array*>(this, _MTL_PRIVATE_SEL(meshBindings));
}

_MTL_INLINE NS::Array* MTL::RenderPipelineReflection::objectBindings() const
{
    return Object::sendMessage<NS::Array*>(this, _MTL_PRIVATE_SEL(objectBindings));
}

_MTL_INLINE NS::Array* MTL::RenderPipelineReflection::tileArguments() const
{
    return Object::sendMessage<NS::Array*>(this, _MTL_PRIVATE_SEL(tileArguments));
}

_MTL_INLINE NS::Array* MTL::RenderPipelineReflection::tileBindings() const
{
    return Object::sendMessage<NS::Array*>(this, _MTL_PRIVATE_SEL(tileBindings));
}

_MTL_INLINE NS::Array* MTL::RenderPipelineReflection::vertexArguments() const
{
    return Object::sendMessage<NS::Array*>(this, _MTL_PRIVATE_SEL(vertexArguments));
}

_MTL_INLINE NS::Array* MTL::RenderPipelineReflection::vertexBindings() const
{
    return Object::sendMessage<NS::Array*>(this, _MTL_PRIVATE_SEL(vertexBindings));
}

_MTL_INLINE MTL::RenderPipelineDescriptor* MTL::RenderPipelineDescriptor::alloc()
{
    return NS::Object::alloc<MTL::RenderPipelineDescriptor>(_MTL_PRIVATE_CLS(MTLRenderPipelineDescriptor));
}

_MTL_INLINE bool MTL::RenderPipelineDescriptor::alphaToCoverageEnabled() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(isAlphaToCoverageEnabled));
}

_MTL_INLINE bool MTL::RenderPipelineDescriptor::alphaToOneEnabled() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(isAlphaToOneEnabled));
}

_MTL_INLINE NS::Array* MTL::RenderPipelineDescriptor::binaryArchives() const
{
    return Object::sendMessage<NS::Array*>(this, _MTL_PRIVATE_SEL(binaryArchives));
}

_MTL_INLINE MTL::RenderPipelineColorAttachmentDescriptorArray* MTL::RenderPipelineDescriptor::colorAttachments() const
{
    return Object::sendMessage<MTL::RenderPipelineColorAttachmentDescriptorArray*>(this, _MTL_PRIVATE_SEL(colorAttachments));
}

_MTL_INLINE MTL::PixelFormat MTL::RenderPipelineDescriptor::depthAttachmentPixelFormat() const
{
    return Object::sendMessage<MTL::PixelFormat>(this, _MTL_PRIVATE_SEL(depthAttachmentPixelFormat));
}

_MTL_INLINE MTL::PipelineBufferDescriptorArray* MTL::RenderPipelineDescriptor::fragmentBuffers() const
{
    return Object::sendMessage<MTL::PipelineBufferDescriptorArray*>(this, _MTL_PRIVATE_SEL(fragmentBuffers));
}

_MTL_INLINE MTL::Function* MTL::RenderPipelineDescriptor::fragmentFunction() const
{
    return Object::sendMessage<MTL::Function*>(this, _MTL_PRIVATE_SEL(fragmentFunction));
}

_MTL_INLINE MTL::LinkedFunctions* MTL::RenderPipelineDescriptor::fragmentLinkedFunctions() const
{
    return Object::sendMessage<MTL::LinkedFunctions*>(this, _MTL_PRIVATE_SEL(fragmentLinkedFunctions));
}

_MTL_INLINE NS::Array* MTL::RenderPipelineDescriptor::fragmentPreloadedLibraries() const
{
    return Object::sendMessage<NS::Array*>(this, _MTL_PRIVATE_SEL(fragmentPreloadedLibraries));
}

_MTL_INLINE MTL::RenderPipelineDescriptor* MTL::RenderPipelineDescriptor::init()
{
    return NS::Object::init<MTL::RenderPipelineDescriptor>();
}

_MTL_INLINE MTL::PrimitiveTopologyClass MTL::RenderPipelineDescriptor::inputPrimitiveTopology() const
{
    return Object::sendMessage<MTL::PrimitiveTopologyClass>(this, _MTL_PRIVATE_SEL(inputPrimitiveTopology));
}

_MTL_INLINE bool MTL::RenderPipelineDescriptor::isAlphaToCoverageEnabled() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(isAlphaToCoverageEnabled));
}

_MTL_INLINE bool MTL::RenderPipelineDescriptor::isAlphaToOneEnabled() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(isAlphaToOneEnabled));
}

_MTL_INLINE bool MTL::RenderPipelineDescriptor::isRasterizationEnabled() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(isRasterizationEnabled));
}

_MTL_INLINE bool MTL::RenderPipelineDescriptor::isTessellationFactorScaleEnabled() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(isTessellationFactorScaleEnabled));
}

_MTL_INLINE NS::String* MTL::RenderPipelineDescriptor::label() const
{
    return Object::sendMessage<NS::String*>(this, _MTL_PRIVATE_SEL(label));
}

_MTL_INLINE NS::UInteger MTL::RenderPipelineDescriptor::maxFragmentCallStackDepth() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(maxFragmentCallStackDepth));
}

_MTL_INLINE NS::UInteger MTL::RenderPipelineDescriptor::maxTessellationFactor() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(maxTessellationFactor));
}

_MTL_INLINE NS::UInteger MTL::RenderPipelineDescriptor::maxVertexAmplificationCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(maxVertexAmplificationCount));
}

_MTL_INLINE NS::UInteger MTL::RenderPipelineDescriptor::maxVertexCallStackDepth() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(maxVertexCallStackDepth));
}

_MTL_INLINE NS::UInteger MTL::RenderPipelineDescriptor::rasterSampleCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(rasterSampleCount));
}

_MTL_INLINE bool MTL::RenderPipelineDescriptor::rasterizationEnabled() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(isRasterizationEnabled));
}

_MTL_INLINE void MTL::RenderPipelineDescriptor::reset()
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(reset));
}

_MTL_INLINE NS::UInteger MTL::RenderPipelineDescriptor::sampleCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(sampleCount));
}

_MTL_INLINE void MTL::RenderPipelineDescriptor::setAlphaToCoverageEnabled(bool alphaToCoverageEnabled)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setAlphaToCoverageEnabled_), alphaToCoverageEnabled);
}

_MTL_INLINE void MTL::RenderPipelineDescriptor::setAlphaToOneEnabled(bool alphaToOneEnabled)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setAlphaToOneEnabled_), alphaToOneEnabled);
}

_MTL_INLINE void MTL::RenderPipelineDescriptor::setBinaryArchives(const NS::Array* binaryArchives)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setBinaryArchives_), binaryArchives);
}

_MTL_INLINE void MTL::RenderPipelineDescriptor::setDepthAttachmentPixelFormat(MTL::PixelFormat depthAttachmentPixelFormat)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setDepthAttachmentPixelFormat_), depthAttachmentPixelFormat);
}

_MTL_INLINE void MTL::RenderPipelineDescriptor::setFragmentFunction(const MTL::Function* fragmentFunction)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setFragmentFunction_), fragmentFunction);
}

_MTL_INLINE void MTL::RenderPipelineDescriptor::setFragmentLinkedFunctions(const MTL::LinkedFunctions* fragmentLinkedFunctions)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setFragmentLinkedFunctions_), fragmentLinkedFunctions);
}

_MTL_INLINE void MTL::RenderPipelineDescriptor::setFragmentPreloadedLibraries(const NS::Array* fragmentPreloadedLibraries)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setFragmentPreloadedLibraries_), fragmentPreloadedLibraries);
}

_MTL_INLINE void MTL::RenderPipelineDescriptor::setInputPrimitiveTopology(MTL::PrimitiveTopologyClass inputPrimitiveTopology)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setInputPrimitiveTopology_), inputPrimitiveTopology);
}

_MTL_INLINE void MTL::RenderPipelineDescriptor::setLabel(const NS::String* label)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setLabel_), label);
}

_MTL_INLINE void MTL::RenderPipelineDescriptor::setMaxFragmentCallStackDepth(NS::UInteger maxFragmentCallStackDepth)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMaxFragmentCallStackDepth_), maxFragmentCallStackDepth);
}

_MTL_INLINE void MTL::RenderPipelineDescriptor::setMaxTessellationFactor(NS::UInteger maxTessellationFactor)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMaxTessellationFactor_), maxTessellationFactor);
}

_MTL_INLINE void MTL::RenderPipelineDescriptor::setMaxVertexAmplificationCount(NS::UInteger maxVertexAmplificationCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMaxVertexAmplificationCount_), maxVertexAmplificationCount);
}

_MTL_INLINE void MTL::RenderPipelineDescriptor::setMaxVertexCallStackDepth(NS::UInteger maxVertexCallStackDepth)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMaxVertexCallStackDepth_), maxVertexCallStackDepth);
}

_MTL_INLINE void MTL::RenderPipelineDescriptor::setRasterSampleCount(NS::UInteger rasterSampleCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setRasterSampleCount_), rasterSampleCount);
}

_MTL_INLINE void MTL::RenderPipelineDescriptor::setRasterizationEnabled(bool rasterizationEnabled)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setRasterizationEnabled_), rasterizationEnabled);
}

_MTL_INLINE void MTL::RenderPipelineDescriptor::setSampleCount(NS::UInteger sampleCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setSampleCount_), sampleCount);
}

_MTL_INLINE void MTL::RenderPipelineDescriptor::setShaderValidation(MTL::ShaderValidation shaderValidation)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setShaderValidation_), shaderValidation);
}

_MTL_INLINE void MTL::RenderPipelineDescriptor::setStencilAttachmentPixelFormat(MTL::PixelFormat stencilAttachmentPixelFormat)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setStencilAttachmentPixelFormat_), stencilAttachmentPixelFormat);
}

_MTL_INLINE void MTL::RenderPipelineDescriptor::setSupportAddingFragmentBinaryFunctions(bool supportAddingFragmentBinaryFunctions)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setSupportAddingFragmentBinaryFunctions_), supportAddingFragmentBinaryFunctions);
}

_MTL_INLINE void MTL::RenderPipelineDescriptor::setSupportAddingVertexBinaryFunctions(bool supportAddingVertexBinaryFunctions)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setSupportAddingVertexBinaryFunctions_), supportAddingVertexBinaryFunctions);
}

_MTL_INLINE void MTL::RenderPipelineDescriptor::setSupportIndirectCommandBuffers(bool supportIndirectCommandBuffers)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setSupportIndirectCommandBuffers_), supportIndirectCommandBuffers);
}

_MTL_INLINE void MTL::RenderPipelineDescriptor::setTessellationControlPointIndexType(MTL::TessellationControlPointIndexType tessellationControlPointIndexType)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setTessellationControlPointIndexType_), tessellationControlPointIndexType);
}

_MTL_INLINE void MTL::RenderPipelineDescriptor::setTessellationFactorFormat(MTL::TessellationFactorFormat tessellationFactorFormat)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setTessellationFactorFormat_), tessellationFactorFormat);
}

_MTL_INLINE void MTL::RenderPipelineDescriptor::setTessellationFactorScaleEnabled(bool tessellationFactorScaleEnabled)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setTessellationFactorScaleEnabled_), tessellationFactorScaleEnabled);
}

_MTL_INLINE void MTL::RenderPipelineDescriptor::setTessellationFactorStepFunction(MTL::TessellationFactorStepFunction tessellationFactorStepFunction)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setTessellationFactorStepFunction_), tessellationFactorStepFunction);
}

_MTL_INLINE void MTL::RenderPipelineDescriptor::setTessellationOutputWindingOrder(MTL::Winding tessellationOutputWindingOrder)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setTessellationOutputWindingOrder_), tessellationOutputWindingOrder);
}

_MTL_INLINE void MTL::RenderPipelineDescriptor::setTessellationPartitionMode(MTL::TessellationPartitionMode tessellationPartitionMode)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setTessellationPartitionMode_), tessellationPartitionMode);
}

_MTL_INLINE void MTL::RenderPipelineDescriptor::setVertexDescriptor(const MTL::VertexDescriptor* vertexDescriptor)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setVertexDescriptor_), vertexDescriptor);
}

_MTL_INLINE void MTL::RenderPipelineDescriptor::setVertexFunction(const MTL::Function* vertexFunction)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setVertexFunction_), vertexFunction);
}

_MTL_INLINE void MTL::RenderPipelineDescriptor::setVertexLinkedFunctions(const MTL::LinkedFunctions* vertexLinkedFunctions)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setVertexLinkedFunctions_), vertexLinkedFunctions);
}

_MTL_INLINE void MTL::RenderPipelineDescriptor::setVertexPreloadedLibraries(const NS::Array* vertexPreloadedLibraries)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setVertexPreloadedLibraries_), vertexPreloadedLibraries);
}

_MTL_INLINE MTL::ShaderValidation MTL::RenderPipelineDescriptor::shaderValidation() const
{
    return Object::sendMessage<MTL::ShaderValidation>(this, _MTL_PRIVATE_SEL(shaderValidation));
}

_MTL_INLINE MTL::PixelFormat MTL::RenderPipelineDescriptor::stencilAttachmentPixelFormat() const
{
    return Object::sendMessage<MTL::PixelFormat>(this, _MTL_PRIVATE_SEL(stencilAttachmentPixelFormat));
}

_MTL_INLINE bool MTL::RenderPipelineDescriptor::supportAddingFragmentBinaryFunctions() const
{
    return Object::sendMessageSafe<bool>(this, _MTL_PRIVATE_SEL(supportAddingFragmentBinaryFunctions));
}

_MTL_INLINE bool MTL::RenderPipelineDescriptor::supportAddingVertexBinaryFunctions() const
{
    return Object::sendMessageSafe<bool>(this, _MTL_PRIVATE_SEL(supportAddingVertexBinaryFunctions));
}

_MTL_INLINE bool MTL::RenderPipelineDescriptor::supportIndirectCommandBuffers() const
{
    return Object::sendMessageSafe<bool>(this, _MTL_PRIVATE_SEL(supportIndirectCommandBuffers));
}

_MTL_INLINE MTL::TessellationControlPointIndexType MTL::RenderPipelineDescriptor::tessellationControlPointIndexType() const
{
    return Object::sendMessage<MTL::TessellationControlPointIndexType>(this, _MTL_PRIVATE_SEL(tessellationControlPointIndexType));
}

_MTL_INLINE MTL::TessellationFactorFormat MTL::RenderPipelineDescriptor::tessellationFactorFormat() const
{
    return Object::sendMessage<MTL::TessellationFactorFormat>(this, _MTL_PRIVATE_SEL(tessellationFactorFormat));
}

_MTL_INLINE bool MTL::RenderPipelineDescriptor::tessellationFactorScaleEnabled() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(isTessellationFactorScaleEnabled));
}

_MTL_INLINE MTL::TessellationFactorStepFunction MTL::RenderPipelineDescriptor::tessellationFactorStepFunction() const
{
    return Object::sendMessage<MTL::TessellationFactorStepFunction>(this, _MTL_PRIVATE_SEL(tessellationFactorStepFunction));
}

_MTL_INLINE MTL::Winding MTL::RenderPipelineDescriptor::tessellationOutputWindingOrder() const
{
    return Object::sendMessage<MTL::Winding>(this, _MTL_PRIVATE_SEL(tessellationOutputWindingOrder));
}

_MTL_INLINE MTL::TessellationPartitionMode MTL::RenderPipelineDescriptor::tessellationPartitionMode() const
{
    return Object::sendMessage<MTL::TessellationPartitionMode>(this, _MTL_PRIVATE_SEL(tessellationPartitionMode));
}

_MTL_INLINE MTL::PipelineBufferDescriptorArray* MTL::RenderPipelineDescriptor::vertexBuffers() const
{
    return Object::sendMessage<MTL::PipelineBufferDescriptorArray*>(this, _MTL_PRIVATE_SEL(vertexBuffers));
}

_MTL_INLINE MTL::VertexDescriptor* MTL::RenderPipelineDescriptor::vertexDescriptor() const
{
    return Object::sendMessage<MTL::VertexDescriptor*>(this, _MTL_PRIVATE_SEL(vertexDescriptor));
}

_MTL_INLINE MTL::Function* MTL::RenderPipelineDescriptor::vertexFunction() const
{
    return Object::sendMessage<MTL::Function*>(this, _MTL_PRIVATE_SEL(vertexFunction));
}

_MTL_INLINE MTL::LinkedFunctions* MTL::RenderPipelineDescriptor::vertexLinkedFunctions() const
{
    return Object::sendMessage<MTL::LinkedFunctions*>(this, _MTL_PRIVATE_SEL(vertexLinkedFunctions));
}

_MTL_INLINE NS::Array* MTL::RenderPipelineDescriptor::vertexPreloadedLibraries() const
{
    return Object::sendMessage<NS::Array*>(this, _MTL_PRIVATE_SEL(vertexPreloadedLibraries));
}

_MTL_INLINE MTL::RenderPipelineFunctionsDescriptor* MTL::RenderPipelineFunctionsDescriptor::alloc()
{
    return NS::Object::alloc<MTL::RenderPipelineFunctionsDescriptor>(_MTL_PRIVATE_CLS(MTLRenderPipelineFunctionsDescriptor));
}

_MTL_INLINE NS::Array* MTL::RenderPipelineFunctionsDescriptor::fragmentAdditionalBinaryFunctions() const
{
    return Object::sendMessage<NS::Array*>(this, _MTL_PRIVATE_SEL(fragmentAdditionalBinaryFunctions));
}

_MTL_INLINE MTL::RenderPipelineFunctionsDescriptor* MTL::RenderPipelineFunctionsDescriptor::init()
{
    return NS::Object::init<MTL::RenderPipelineFunctionsDescriptor>();
}

_MTL_INLINE void MTL::RenderPipelineFunctionsDescriptor::setFragmentAdditionalBinaryFunctions(const NS::Array* fragmentAdditionalBinaryFunctions)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setFragmentAdditionalBinaryFunctions_), fragmentAdditionalBinaryFunctions);
}

_MTL_INLINE void MTL::RenderPipelineFunctionsDescriptor::setTileAdditionalBinaryFunctions(const NS::Array* tileAdditionalBinaryFunctions)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setTileAdditionalBinaryFunctions_), tileAdditionalBinaryFunctions);
}

_MTL_INLINE void MTL::RenderPipelineFunctionsDescriptor::setVertexAdditionalBinaryFunctions(const NS::Array* vertexAdditionalBinaryFunctions)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setVertexAdditionalBinaryFunctions_), vertexAdditionalBinaryFunctions);
}

_MTL_INLINE NS::Array* MTL::RenderPipelineFunctionsDescriptor::tileAdditionalBinaryFunctions() const
{
    return Object::sendMessage<NS::Array*>(this, _MTL_PRIVATE_SEL(tileAdditionalBinaryFunctions));
}

_MTL_INLINE NS::Array* MTL::RenderPipelineFunctionsDescriptor::vertexAdditionalBinaryFunctions() const
{
    return Object::sendMessage<NS::Array*>(this, _MTL_PRIVATE_SEL(vertexAdditionalBinaryFunctions));
}

_MTL_INLINE MTL::Device* MTL::RenderPipelineState::device() const
{
    return Object::sendMessage<MTL::Device*>(this, _MTL_PRIVATE_SEL(device));
}

_MTL_INLINE MTL::FunctionHandle* MTL::RenderPipelineState::functionHandle(const NS::String* name, MTL::RenderStages stage)
{
    return Object::sendMessage<MTL::FunctionHandle*>(this, _MTL_PRIVATE_SEL(functionHandleWithName_stage_), name, stage);
}

_MTL_INLINE MTL::FunctionHandle* MTL::RenderPipelineState::functionHandle(const MTL4::BinaryFunction* function, MTL::RenderStages stage)
{
    return Object::sendMessage<MTL::FunctionHandle*>(this, _MTL_PRIVATE_SEL(functionHandleWithBinaryFunction_stage_), function, stage);
}

_MTL_INLINE MTL::FunctionHandle* MTL::RenderPipelineState::functionHandle(const MTL::Function* function, MTL::RenderStages stage)
{
    return Object::sendMessage<MTL::FunctionHandle*>(this, _MTL_PRIVATE_SEL(functionHandleWithFunction_stage_), function, stage);
}

_MTL_INLINE MTL::ResourceID MTL::RenderPipelineState::gpuResourceID() const
{
    return Object::sendMessage<MTL::ResourceID>(this, _MTL_PRIVATE_SEL(gpuResourceID));
}

_MTL_INLINE NS::UInteger MTL::RenderPipelineState::imageblockMemoryLength(MTL::Size imageblockDimensions)
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(imageblockMemoryLengthForDimensions_), imageblockDimensions);
}

_MTL_INLINE NS::UInteger MTL::RenderPipelineState::imageblockSampleLength() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(imageblockSampleLength));
}

_MTL_INLINE NS::String* MTL::RenderPipelineState::label() const
{
    return Object::sendMessage<NS::String*>(this, _MTL_PRIVATE_SEL(label));
}

_MTL_INLINE NS::UInteger MTL::RenderPipelineState::maxTotalThreadgroupsPerMeshGrid() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(maxTotalThreadgroupsPerMeshGrid));
}

_MTL_INLINE NS::UInteger MTL::RenderPipelineState::maxTotalThreadsPerMeshThreadgroup() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(maxTotalThreadsPerMeshThreadgroup));
}

_MTL_INLINE NS::UInteger MTL::RenderPipelineState::maxTotalThreadsPerObjectThreadgroup() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(maxTotalThreadsPerObjectThreadgroup));
}

_MTL_INLINE NS::UInteger MTL::RenderPipelineState::maxTotalThreadsPerThreadgroup() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(maxTotalThreadsPerThreadgroup));
}

_MTL_INLINE NS::UInteger MTL::RenderPipelineState::meshThreadExecutionWidth() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(meshThreadExecutionWidth));
}

_MTL_INLINE MTL::IntersectionFunctionTable* MTL::RenderPipelineState::newIntersectionFunctionTable(const MTL::IntersectionFunctionTableDescriptor* descriptor, MTL::RenderStages stage)
{
    return Object::sendMessage<MTL::IntersectionFunctionTable*>(this, _MTL_PRIVATE_SEL(newIntersectionFunctionTableWithDescriptor_stage_), descriptor, stage);
}

_MTL_INLINE MTL4::PipelineDescriptor* MTL::RenderPipelineState::newRenderPipelineDescriptor()
{
    return Object::sendMessage<MTL4::PipelineDescriptor*>(this, _MTL_PRIVATE_SEL(newRenderPipelineDescriptorForSpecialization));
}

_MTL_INLINE MTL::RenderPipelineState* MTL::RenderPipelineState::newRenderPipelineState(const MTL4::RenderPipelineBinaryFunctionsDescriptor* binaryFunctionsDescriptor, NS::Error** error)
{
    return Object::sendMessage<MTL::RenderPipelineState*>(this, _MTL_PRIVATE_SEL(newRenderPipelineStateWithBinaryFunctions_error_), binaryFunctionsDescriptor, error);
}

_MTL_INLINE MTL::RenderPipelineState* MTL::RenderPipelineState::newRenderPipelineState(const MTL::RenderPipelineFunctionsDescriptor* additionalBinaryFunctions, NS::Error** error)
{
    return Object::sendMessage<MTL::RenderPipelineState*>(this, _MTL_PRIVATE_SEL(newRenderPipelineStateWithAdditionalBinaryFunctions_error_), additionalBinaryFunctions, error);
}

_MTL_INLINE MTL::VisibleFunctionTable* MTL::RenderPipelineState::newVisibleFunctionTable(const MTL::VisibleFunctionTableDescriptor* descriptor, MTL::RenderStages stage)
{
    return Object::sendMessage<MTL::VisibleFunctionTable*>(this, _MTL_PRIVATE_SEL(newVisibleFunctionTableWithDescriptor_stage_), descriptor, stage);
}

_MTL_INLINE NS::UInteger MTL::RenderPipelineState::objectThreadExecutionWidth() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(objectThreadExecutionWidth));
}

_MTL_INLINE MTL::RenderPipelineReflection* MTL::RenderPipelineState::reflection() const
{
    return Object::sendMessage<MTL::RenderPipelineReflection*>(this, _MTL_PRIVATE_SEL(reflection));
}

_MTL_INLINE MTL::Size MTL::RenderPipelineState::requiredThreadsPerMeshThreadgroup() const
{
    return Object::sendMessage<MTL::Size>(this, _MTL_PRIVATE_SEL(requiredThreadsPerMeshThreadgroup));
}

_MTL_INLINE MTL::Size MTL::RenderPipelineState::requiredThreadsPerObjectThreadgroup() const
{
    return Object::sendMessage<MTL::Size>(this, _MTL_PRIVATE_SEL(requiredThreadsPerObjectThreadgroup));
}

_MTL_INLINE MTL::Size MTL::RenderPipelineState::requiredThreadsPerTileThreadgroup() const
{
    return Object::sendMessage<MTL::Size>(this, _MTL_PRIVATE_SEL(requiredThreadsPerTileThreadgroup));
}

_MTL_INLINE MTL::ShaderValidation MTL::RenderPipelineState::shaderValidation() const
{
    return Object::sendMessage<MTL::ShaderValidation>(this, _MTL_PRIVATE_SEL(shaderValidation));
}

_MTL_INLINE bool MTL::RenderPipelineState::supportIndirectCommandBuffers() const
{
    return Object::sendMessageSafe<bool>(this, _MTL_PRIVATE_SEL(supportIndirectCommandBuffers));
}

_MTL_INLINE bool MTL::RenderPipelineState::threadgroupSizeMatchesTileSize() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(threadgroupSizeMatchesTileSize));
}

_MTL_INLINE MTL::RenderPipelineColorAttachmentDescriptorArray* MTL::RenderPipelineColorAttachmentDescriptorArray::alloc()
{
    return NS::Object::alloc<MTL::RenderPipelineColorAttachmentDescriptorArray>(_MTL_PRIVATE_CLS(MTLRenderPipelineColorAttachmentDescriptorArray));
}

_MTL_INLINE MTL::RenderPipelineColorAttachmentDescriptorArray* MTL::RenderPipelineColorAttachmentDescriptorArray::init()
{
    return NS::Object::init<MTL::RenderPipelineColorAttachmentDescriptorArray>();
}

_MTL_INLINE MTL::RenderPipelineColorAttachmentDescriptor* MTL::RenderPipelineColorAttachmentDescriptorArray::object(NS::UInteger attachmentIndex)
{
    return Object::sendMessage<MTL::RenderPipelineColorAttachmentDescriptor*>(this, _MTL_PRIVATE_SEL(objectAtIndexedSubscript_), attachmentIndex);
}

_MTL_INLINE void MTL::RenderPipelineColorAttachmentDescriptorArray::setObject(const MTL::RenderPipelineColorAttachmentDescriptor* attachment, NS::UInteger attachmentIndex)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setObject_atIndexedSubscript_), attachment, attachmentIndex);
}

_MTL_INLINE MTL::TileRenderPipelineColorAttachmentDescriptor* MTL::TileRenderPipelineColorAttachmentDescriptor::alloc()
{
    return NS::Object::alloc<MTL::TileRenderPipelineColorAttachmentDescriptor>(_MTL_PRIVATE_CLS(MTLTileRenderPipelineColorAttachmentDescriptor));
}

_MTL_INLINE MTL::TileRenderPipelineColorAttachmentDescriptor* MTL::TileRenderPipelineColorAttachmentDescriptor::init()
{
    return NS::Object::init<MTL::TileRenderPipelineColorAttachmentDescriptor>();
}

_MTL_INLINE MTL::PixelFormat MTL::TileRenderPipelineColorAttachmentDescriptor::pixelFormat() const
{
    return Object::sendMessage<MTL::PixelFormat>(this, _MTL_PRIVATE_SEL(pixelFormat));
}

_MTL_INLINE void MTL::TileRenderPipelineColorAttachmentDescriptor::setPixelFormat(MTL::PixelFormat pixelFormat)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setPixelFormat_), pixelFormat);
}

_MTL_INLINE MTL::TileRenderPipelineColorAttachmentDescriptorArray* MTL::TileRenderPipelineColorAttachmentDescriptorArray::alloc()
{
    return NS::Object::alloc<MTL::TileRenderPipelineColorAttachmentDescriptorArray>(_MTL_PRIVATE_CLS(MTLTileRenderPipelineColorAttachmentDescriptorArray));
}

_MTL_INLINE MTL::TileRenderPipelineColorAttachmentDescriptorArray* MTL::TileRenderPipelineColorAttachmentDescriptorArray::init()
{
    return NS::Object::init<MTL::TileRenderPipelineColorAttachmentDescriptorArray>();
}

_MTL_INLINE MTL::TileRenderPipelineColorAttachmentDescriptor* MTL::TileRenderPipelineColorAttachmentDescriptorArray::object(NS::UInteger attachmentIndex)
{
    return Object::sendMessage<MTL::TileRenderPipelineColorAttachmentDescriptor*>(this, _MTL_PRIVATE_SEL(objectAtIndexedSubscript_), attachmentIndex);
}

_MTL_INLINE void MTL::TileRenderPipelineColorAttachmentDescriptorArray::setObject(const MTL::TileRenderPipelineColorAttachmentDescriptor* attachment, NS::UInteger attachmentIndex)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setObject_atIndexedSubscript_), attachment, attachmentIndex);
}

_MTL_INLINE MTL::TileRenderPipelineDescriptor* MTL::TileRenderPipelineDescriptor::alloc()
{
    return NS::Object::alloc<MTL::TileRenderPipelineDescriptor>(_MTL_PRIVATE_CLS(MTLTileRenderPipelineDescriptor));
}

_MTL_INLINE NS::Array* MTL::TileRenderPipelineDescriptor::binaryArchives() const
{
    return Object::sendMessage<NS::Array*>(this, _MTL_PRIVATE_SEL(binaryArchives));
}

_MTL_INLINE MTL::TileRenderPipelineColorAttachmentDescriptorArray* MTL::TileRenderPipelineDescriptor::colorAttachments() const
{
    return Object::sendMessage<MTL::TileRenderPipelineColorAttachmentDescriptorArray*>(this, _MTL_PRIVATE_SEL(colorAttachments));
}

_MTL_INLINE MTL::TileRenderPipelineDescriptor* MTL::TileRenderPipelineDescriptor::init()
{
    return NS::Object::init<MTL::TileRenderPipelineDescriptor>();
}

_MTL_INLINE NS::String* MTL::TileRenderPipelineDescriptor::label() const
{
    return Object::sendMessage<NS::String*>(this, _MTL_PRIVATE_SEL(label));
}

_MTL_INLINE MTL::LinkedFunctions* MTL::TileRenderPipelineDescriptor::linkedFunctions() const
{
    return Object::sendMessage<MTL::LinkedFunctions*>(this, _MTL_PRIVATE_SEL(linkedFunctions));
}

_MTL_INLINE NS::UInteger MTL::TileRenderPipelineDescriptor::maxCallStackDepth() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(maxCallStackDepth));
}

_MTL_INLINE NS::UInteger MTL::TileRenderPipelineDescriptor::maxTotalThreadsPerThreadgroup() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(maxTotalThreadsPerThreadgroup));
}

_MTL_INLINE NS::Array* MTL::TileRenderPipelineDescriptor::preloadedLibraries() const
{
    return Object::sendMessage<NS::Array*>(this, _MTL_PRIVATE_SEL(preloadedLibraries));
}

_MTL_INLINE NS::UInteger MTL::TileRenderPipelineDescriptor::rasterSampleCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(rasterSampleCount));
}

_MTL_INLINE MTL::Size MTL::TileRenderPipelineDescriptor::requiredThreadsPerThreadgroup() const
{
    return Object::sendMessage<MTL::Size>(this, _MTL_PRIVATE_SEL(requiredThreadsPerThreadgroup));
}

_MTL_INLINE void MTL::TileRenderPipelineDescriptor::reset()
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(reset));
}

_MTL_INLINE void MTL::TileRenderPipelineDescriptor::setBinaryArchives(const NS::Array* binaryArchives)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setBinaryArchives_), binaryArchives);
}

_MTL_INLINE void MTL::TileRenderPipelineDescriptor::setLabel(const NS::String* label)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setLabel_), label);
}

_MTL_INLINE void MTL::TileRenderPipelineDescriptor::setLinkedFunctions(const MTL::LinkedFunctions* linkedFunctions)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setLinkedFunctions_), linkedFunctions);
}

_MTL_INLINE void MTL::TileRenderPipelineDescriptor::setMaxCallStackDepth(NS::UInteger maxCallStackDepth)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMaxCallStackDepth_), maxCallStackDepth);
}

_MTL_INLINE void MTL::TileRenderPipelineDescriptor::setMaxTotalThreadsPerThreadgroup(NS::UInteger maxTotalThreadsPerThreadgroup)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMaxTotalThreadsPerThreadgroup_), maxTotalThreadsPerThreadgroup);
}

_MTL_INLINE void MTL::TileRenderPipelineDescriptor::setPreloadedLibraries(const NS::Array* preloadedLibraries)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setPreloadedLibraries_), preloadedLibraries);
}

_MTL_INLINE void MTL::TileRenderPipelineDescriptor::setRasterSampleCount(NS::UInteger rasterSampleCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setRasterSampleCount_), rasterSampleCount);
}

_MTL_INLINE void MTL::TileRenderPipelineDescriptor::setRequiredThreadsPerThreadgroup(MTL::Size requiredThreadsPerThreadgroup)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setRequiredThreadsPerThreadgroup_), requiredThreadsPerThreadgroup);
}

_MTL_INLINE void MTL::TileRenderPipelineDescriptor::setShaderValidation(MTL::ShaderValidation shaderValidation)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setShaderValidation_), shaderValidation);
}

_MTL_INLINE void MTL::TileRenderPipelineDescriptor::setSupportAddingBinaryFunctions(bool supportAddingBinaryFunctions)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setSupportAddingBinaryFunctions_), supportAddingBinaryFunctions);
}

_MTL_INLINE void MTL::TileRenderPipelineDescriptor::setThreadgroupSizeMatchesTileSize(bool threadgroupSizeMatchesTileSize)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setThreadgroupSizeMatchesTileSize_), threadgroupSizeMatchesTileSize);
}

_MTL_INLINE void MTL::TileRenderPipelineDescriptor::setTileFunction(const MTL::Function* tileFunction)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setTileFunction_), tileFunction);
}

_MTL_INLINE MTL::ShaderValidation MTL::TileRenderPipelineDescriptor::shaderValidation() const
{
    return Object::sendMessage<MTL::ShaderValidation>(this, _MTL_PRIVATE_SEL(shaderValidation));
}

_MTL_INLINE bool MTL::TileRenderPipelineDescriptor::supportAddingBinaryFunctions() const
{
    return Object::sendMessageSafe<bool>(this, _MTL_PRIVATE_SEL(supportAddingBinaryFunctions));
}

_MTL_INLINE bool MTL::TileRenderPipelineDescriptor::threadgroupSizeMatchesTileSize() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(threadgroupSizeMatchesTileSize));
}

_MTL_INLINE MTL::PipelineBufferDescriptorArray* MTL::TileRenderPipelineDescriptor::tileBuffers() const
{
    return Object::sendMessage<MTL::PipelineBufferDescriptorArray*>(this, _MTL_PRIVATE_SEL(tileBuffers));
}

_MTL_INLINE MTL::Function* MTL::TileRenderPipelineDescriptor::tileFunction() const
{
    return Object::sendMessage<MTL::Function*>(this, _MTL_PRIVATE_SEL(tileFunction));
}

_MTL_INLINE MTL::MeshRenderPipelineDescriptor* MTL::MeshRenderPipelineDescriptor::alloc()
{
    return NS::Object::alloc<MTL::MeshRenderPipelineDescriptor>(_MTL_PRIVATE_CLS(MTLMeshRenderPipelineDescriptor));
}

_MTL_INLINE bool MTL::MeshRenderPipelineDescriptor::alphaToCoverageEnabled() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(isAlphaToCoverageEnabled));
}

_MTL_INLINE bool MTL::MeshRenderPipelineDescriptor::alphaToOneEnabled() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(isAlphaToOneEnabled));
}

_MTL_INLINE NS::Array* MTL::MeshRenderPipelineDescriptor::binaryArchives() const
{
    return Object::sendMessage<NS::Array*>(this, _MTL_PRIVATE_SEL(binaryArchives));
}

_MTL_INLINE MTL::RenderPipelineColorAttachmentDescriptorArray* MTL::MeshRenderPipelineDescriptor::colorAttachments() const
{
    return Object::sendMessage<MTL::RenderPipelineColorAttachmentDescriptorArray*>(this, _MTL_PRIVATE_SEL(colorAttachments));
}

_MTL_INLINE MTL::PixelFormat MTL::MeshRenderPipelineDescriptor::depthAttachmentPixelFormat() const
{
    return Object::sendMessage<MTL::PixelFormat>(this, _MTL_PRIVATE_SEL(depthAttachmentPixelFormat));
}

_MTL_INLINE MTL::PipelineBufferDescriptorArray* MTL::MeshRenderPipelineDescriptor::fragmentBuffers() const
{
    return Object::sendMessage<MTL::PipelineBufferDescriptorArray*>(this, _MTL_PRIVATE_SEL(fragmentBuffers));
}

_MTL_INLINE MTL::Function* MTL::MeshRenderPipelineDescriptor::fragmentFunction() const
{
    return Object::sendMessage<MTL::Function*>(this, _MTL_PRIVATE_SEL(fragmentFunction));
}

_MTL_INLINE MTL::LinkedFunctions* MTL::MeshRenderPipelineDescriptor::fragmentLinkedFunctions() const
{
    return Object::sendMessage<MTL::LinkedFunctions*>(this, _MTL_PRIVATE_SEL(fragmentLinkedFunctions));
}

_MTL_INLINE MTL::MeshRenderPipelineDescriptor* MTL::MeshRenderPipelineDescriptor::init()
{
    return NS::Object::init<MTL::MeshRenderPipelineDescriptor>();
}

_MTL_INLINE bool MTL::MeshRenderPipelineDescriptor::isAlphaToCoverageEnabled() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(isAlphaToCoverageEnabled));
}

_MTL_INLINE bool MTL::MeshRenderPipelineDescriptor::isAlphaToOneEnabled() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(isAlphaToOneEnabled));
}

_MTL_INLINE bool MTL::MeshRenderPipelineDescriptor::isRasterizationEnabled() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(isRasterizationEnabled));
}

_MTL_INLINE NS::String* MTL::MeshRenderPipelineDescriptor::label() const
{
    return Object::sendMessage<NS::String*>(this, _MTL_PRIVATE_SEL(label));
}

_MTL_INLINE NS::UInteger MTL::MeshRenderPipelineDescriptor::maxTotalThreadgroupsPerMeshGrid() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(maxTotalThreadgroupsPerMeshGrid));
}

_MTL_INLINE NS::UInteger MTL::MeshRenderPipelineDescriptor::maxTotalThreadsPerMeshThreadgroup() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(maxTotalThreadsPerMeshThreadgroup));
}

_MTL_INLINE NS::UInteger MTL::MeshRenderPipelineDescriptor::maxTotalThreadsPerObjectThreadgroup() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(maxTotalThreadsPerObjectThreadgroup));
}

_MTL_INLINE NS::UInteger MTL::MeshRenderPipelineDescriptor::maxVertexAmplificationCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(maxVertexAmplificationCount));
}

_MTL_INLINE MTL::PipelineBufferDescriptorArray* MTL::MeshRenderPipelineDescriptor::meshBuffers() const
{
    return Object::sendMessage<MTL::PipelineBufferDescriptorArray*>(this, _MTL_PRIVATE_SEL(meshBuffers));
}

_MTL_INLINE MTL::Function* MTL::MeshRenderPipelineDescriptor::meshFunction() const
{
    return Object::sendMessage<MTL::Function*>(this, _MTL_PRIVATE_SEL(meshFunction));
}

_MTL_INLINE MTL::LinkedFunctions* MTL::MeshRenderPipelineDescriptor::meshLinkedFunctions() const
{
    return Object::sendMessage<MTL::LinkedFunctions*>(this, _MTL_PRIVATE_SEL(meshLinkedFunctions));
}

_MTL_INLINE bool MTL::MeshRenderPipelineDescriptor::meshThreadgroupSizeIsMultipleOfThreadExecutionWidth() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(meshThreadgroupSizeIsMultipleOfThreadExecutionWidth));
}

_MTL_INLINE MTL::PipelineBufferDescriptorArray* MTL::MeshRenderPipelineDescriptor::objectBuffers() const
{
    return Object::sendMessage<MTL::PipelineBufferDescriptorArray*>(this, _MTL_PRIVATE_SEL(objectBuffers));
}

_MTL_INLINE MTL::Function* MTL::MeshRenderPipelineDescriptor::objectFunction() const
{
    return Object::sendMessage<MTL::Function*>(this, _MTL_PRIVATE_SEL(objectFunction));
}

_MTL_INLINE MTL::LinkedFunctions* MTL::MeshRenderPipelineDescriptor::objectLinkedFunctions() const
{
    return Object::sendMessage<MTL::LinkedFunctions*>(this, _MTL_PRIVATE_SEL(objectLinkedFunctions));
}

_MTL_INLINE bool MTL::MeshRenderPipelineDescriptor::objectThreadgroupSizeIsMultipleOfThreadExecutionWidth() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(objectThreadgroupSizeIsMultipleOfThreadExecutionWidth));
}

_MTL_INLINE NS::UInteger MTL::MeshRenderPipelineDescriptor::payloadMemoryLength() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(payloadMemoryLength));
}

_MTL_INLINE NS::UInteger MTL::MeshRenderPipelineDescriptor::rasterSampleCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(rasterSampleCount));
}

_MTL_INLINE bool MTL::MeshRenderPipelineDescriptor::rasterizationEnabled() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(isRasterizationEnabled));
}

_MTL_INLINE MTL::Size MTL::MeshRenderPipelineDescriptor::requiredThreadsPerMeshThreadgroup() const
{
    return Object::sendMessage<MTL::Size>(this, _MTL_PRIVATE_SEL(requiredThreadsPerMeshThreadgroup));
}

_MTL_INLINE MTL::Size MTL::MeshRenderPipelineDescriptor::requiredThreadsPerObjectThreadgroup() const
{
    return Object::sendMessage<MTL::Size>(this, _MTL_PRIVATE_SEL(requiredThreadsPerObjectThreadgroup));
}

_MTL_INLINE void MTL::MeshRenderPipelineDescriptor::reset()
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(reset));
}

_MTL_INLINE void MTL::MeshRenderPipelineDescriptor::setAlphaToCoverageEnabled(bool alphaToCoverageEnabled)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setAlphaToCoverageEnabled_), alphaToCoverageEnabled);
}

_MTL_INLINE void MTL::MeshRenderPipelineDescriptor::setAlphaToOneEnabled(bool alphaToOneEnabled)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setAlphaToOneEnabled_), alphaToOneEnabled);
}

_MTL_INLINE void MTL::MeshRenderPipelineDescriptor::setBinaryArchives(const NS::Array* binaryArchives)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setBinaryArchives_), binaryArchives);
}

_MTL_INLINE void MTL::MeshRenderPipelineDescriptor::setDepthAttachmentPixelFormat(MTL::PixelFormat depthAttachmentPixelFormat)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setDepthAttachmentPixelFormat_), depthAttachmentPixelFormat);
}

_MTL_INLINE void MTL::MeshRenderPipelineDescriptor::setFragmentFunction(const MTL::Function* fragmentFunction)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setFragmentFunction_), fragmentFunction);
}

_MTL_INLINE void MTL::MeshRenderPipelineDescriptor::setFragmentLinkedFunctions(const MTL::LinkedFunctions* fragmentLinkedFunctions)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setFragmentLinkedFunctions_), fragmentLinkedFunctions);
}

_MTL_INLINE void MTL::MeshRenderPipelineDescriptor::setLabel(const NS::String* label)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setLabel_), label);
}

_MTL_INLINE void MTL::MeshRenderPipelineDescriptor::setMaxTotalThreadgroupsPerMeshGrid(NS::UInteger maxTotalThreadgroupsPerMeshGrid)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMaxTotalThreadgroupsPerMeshGrid_), maxTotalThreadgroupsPerMeshGrid);
}

_MTL_INLINE void MTL::MeshRenderPipelineDescriptor::setMaxTotalThreadsPerMeshThreadgroup(NS::UInteger maxTotalThreadsPerMeshThreadgroup)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMaxTotalThreadsPerMeshThreadgroup_), maxTotalThreadsPerMeshThreadgroup);
}

_MTL_INLINE void MTL::MeshRenderPipelineDescriptor::setMaxTotalThreadsPerObjectThreadgroup(NS::UInteger maxTotalThreadsPerObjectThreadgroup)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMaxTotalThreadsPerObjectThreadgroup_), maxTotalThreadsPerObjectThreadgroup);
}

_MTL_INLINE void MTL::MeshRenderPipelineDescriptor::setMaxVertexAmplificationCount(NS::UInteger maxVertexAmplificationCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMaxVertexAmplificationCount_), maxVertexAmplificationCount);
}

_MTL_INLINE void MTL::MeshRenderPipelineDescriptor::setMeshFunction(const MTL::Function* meshFunction)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMeshFunction_), meshFunction);
}

_MTL_INLINE void MTL::MeshRenderPipelineDescriptor::setMeshLinkedFunctions(const MTL::LinkedFunctions* meshLinkedFunctions)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMeshLinkedFunctions_), meshLinkedFunctions);
}

_MTL_INLINE void MTL::MeshRenderPipelineDescriptor::setMeshThreadgroupSizeIsMultipleOfThreadExecutionWidth(bool meshThreadgroupSizeIsMultipleOfThreadExecutionWidth)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMeshThreadgroupSizeIsMultipleOfThreadExecutionWidth_), meshThreadgroupSizeIsMultipleOfThreadExecutionWidth);
}

_MTL_INLINE void MTL::MeshRenderPipelineDescriptor::setObjectFunction(const MTL::Function* objectFunction)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setObjectFunction_), objectFunction);
}

_MTL_INLINE void MTL::MeshRenderPipelineDescriptor::setObjectLinkedFunctions(const MTL::LinkedFunctions* objectLinkedFunctions)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setObjectLinkedFunctions_), objectLinkedFunctions);
}

_MTL_INLINE void MTL::MeshRenderPipelineDescriptor::setObjectThreadgroupSizeIsMultipleOfThreadExecutionWidth(bool objectThreadgroupSizeIsMultipleOfThreadExecutionWidth)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setObjectThreadgroupSizeIsMultipleOfThreadExecutionWidth_), objectThreadgroupSizeIsMultipleOfThreadExecutionWidth);
}

_MTL_INLINE void MTL::MeshRenderPipelineDescriptor::setPayloadMemoryLength(NS::UInteger payloadMemoryLength)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setPayloadMemoryLength_), payloadMemoryLength);
}

_MTL_INLINE void MTL::MeshRenderPipelineDescriptor::setRasterSampleCount(NS::UInteger rasterSampleCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setRasterSampleCount_), rasterSampleCount);
}

_MTL_INLINE void MTL::MeshRenderPipelineDescriptor::setRasterizationEnabled(bool rasterizationEnabled)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setRasterizationEnabled_), rasterizationEnabled);
}

_MTL_INLINE void MTL::MeshRenderPipelineDescriptor::setRequiredThreadsPerMeshThreadgroup(MTL::Size requiredThreadsPerMeshThreadgroup)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setRequiredThreadsPerMeshThreadgroup_), requiredThreadsPerMeshThreadgroup);
}

_MTL_INLINE void MTL::MeshRenderPipelineDescriptor::setRequiredThreadsPerObjectThreadgroup(MTL::Size requiredThreadsPerObjectThreadgroup)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setRequiredThreadsPerObjectThreadgroup_), requiredThreadsPerObjectThreadgroup);
}

_MTL_INLINE void MTL::MeshRenderPipelineDescriptor::setShaderValidation(MTL::ShaderValidation shaderValidation)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setShaderValidation_), shaderValidation);
}

_MTL_INLINE void MTL::MeshRenderPipelineDescriptor::setStencilAttachmentPixelFormat(MTL::PixelFormat stencilAttachmentPixelFormat)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setStencilAttachmentPixelFormat_), stencilAttachmentPixelFormat);
}

_MTL_INLINE void MTL::MeshRenderPipelineDescriptor::setSupportIndirectCommandBuffers(bool supportIndirectCommandBuffers)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setSupportIndirectCommandBuffers_), supportIndirectCommandBuffers);
}

_MTL_INLINE MTL::ShaderValidation MTL::MeshRenderPipelineDescriptor::shaderValidation() const
{
    return Object::sendMessage<MTL::ShaderValidation>(this, _MTL_PRIVATE_SEL(shaderValidation));
}

_MTL_INLINE MTL::PixelFormat MTL::MeshRenderPipelineDescriptor::stencilAttachmentPixelFormat() const
{
    return Object::sendMessage<MTL::PixelFormat>(this, _MTL_PRIVATE_SEL(stencilAttachmentPixelFormat));
}

_MTL_INLINE bool MTL::MeshRenderPipelineDescriptor::supportIndirectCommandBuffers() const
{
    return Object::sendMessageSafe<bool>(this, _MTL_PRIVATE_SEL(supportIndirectCommandBuffers));
}
