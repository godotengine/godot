//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Metal/MTL4RenderPipeline.hpp
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
#include "MTL4PipelineState.hpp"
#include "MTLDefines.hpp"
#include "MTLHeaderBridge.hpp"
#include "MTLPixelFormat.hpp"
#include "MTLPrivate.hpp"
#include "MTLRenderPipeline.hpp"

namespace MTL4
{
class FunctionDescriptor;
class RenderPipelineBinaryFunctionsDescriptor;
class RenderPipelineColorAttachmentDescriptor;
class RenderPipelineColorAttachmentDescriptorArray;
class RenderPipelineDescriptor;
class StaticLinkingDescriptor;
}

namespace MTL
{
class VertexDescriptor;
}

namespace MTL4
{
_MTL_ENUM(NS::Integer, LogicalToPhysicalColorAttachmentMappingState) {
    LogicalToPhysicalColorAttachmentMappingStateIdentity = 0,
    LogicalToPhysicalColorAttachmentMappingStateInherited = 1,
};

class RenderPipelineColorAttachmentDescriptor : public NS::Copying<RenderPipelineColorAttachmentDescriptor>
{
public:
    static RenderPipelineColorAttachmentDescriptor* alloc();

    MTL::BlendOperation                             alphaBlendOperation() const;

    BlendState                                      blendingState() const;

    MTL::BlendFactor                                destinationAlphaBlendFactor() const;

    MTL::BlendFactor                                destinationRGBBlendFactor() const;

    RenderPipelineColorAttachmentDescriptor*        init();

    MTL::PixelFormat                                pixelFormat() const;

    void                                            reset();

    MTL::BlendOperation                             rgbBlendOperation() const;

    void                                            setAlphaBlendOperation(MTL::BlendOperation alphaBlendOperation);

    void                                            setBlendingState(MTL4::BlendState blendingState);

    void                                            setDestinationAlphaBlendFactor(MTL::BlendFactor destinationAlphaBlendFactor);

    void                                            setDestinationRGBBlendFactor(MTL::BlendFactor destinationRGBBlendFactor);

    void                                            setPixelFormat(MTL::PixelFormat pixelFormat);

    void                                            setRgbBlendOperation(MTL::BlendOperation rgbBlendOperation);

    void                                            setSourceAlphaBlendFactor(MTL::BlendFactor sourceAlphaBlendFactor);

    void                                            setSourceRGBBlendFactor(MTL::BlendFactor sourceRGBBlendFactor);

    void                                            setWriteMask(MTL::ColorWriteMask writeMask);

    MTL::BlendFactor                                sourceAlphaBlendFactor() const;

    MTL::BlendFactor                                sourceRGBBlendFactor() const;

    MTL::ColorWriteMask                             writeMask() const;
};

class RenderPipelineColorAttachmentDescriptorArray : public NS::Copying<RenderPipelineColorAttachmentDescriptorArray>
{
public:
    static RenderPipelineColorAttachmentDescriptorArray* alloc();

    RenderPipelineColorAttachmentDescriptorArray*        init();

    RenderPipelineColorAttachmentDescriptor*             object(NS::UInteger attachmentIndex);

    void                                                 reset();

    void                                                 setObject(const MTL4::RenderPipelineColorAttachmentDescriptor* attachment, NS::UInteger attachmentIndex);
};

class RenderPipelineBinaryFunctionsDescriptor : public NS::Copying<RenderPipelineBinaryFunctionsDescriptor>
{
public:
    static RenderPipelineBinaryFunctionsDescriptor* alloc();

    NS::Array*                                      fragmentAdditionalBinaryFunctions() const;

    RenderPipelineBinaryFunctionsDescriptor*        init();

    NS::Array*                                      meshAdditionalBinaryFunctions() const;

    NS::Array*                                      objectAdditionalBinaryFunctions() const;

    void                                            reset();

    void                                            setFragmentAdditionalBinaryFunctions(const NS::Array* fragmentAdditionalBinaryFunctions);

    void                                            setMeshAdditionalBinaryFunctions(const NS::Array* meshAdditionalBinaryFunctions);

    void                                            setObjectAdditionalBinaryFunctions(const NS::Array* objectAdditionalBinaryFunctions);

    void                                            setTileAdditionalBinaryFunctions(const NS::Array* tileAdditionalBinaryFunctions);

    void                                            setVertexAdditionalBinaryFunctions(const NS::Array* vertexAdditionalBinaryFunctions);

    NS::Array*                                      tileAdditionalBinaryFunctions() const;

    NS::Array*                                      vertexAdditionalBinaryFunctions() const;
};

class RenderPipelineDescriptor : public NS::Copying<RenderPipelineDescriptor, PipelineDescriptor>
{
public:
    static RenderPipelineDescriptor*              alloc();

    AlphaToCoverageState                          alphaToCoverageState() const;

    AlphaToOneState                               alphaToOneState() const;

    LogicalToPhysicalColorAttachmentMappingState  colorAttachmentMappingState() const;

    RenderPipelineColorAttachmentDescriptorArray* colorAttachments() const;

    FunctionDescriptor*                           fragmentFunctionDescriptor() const;

    StaticLinkingDescriptor*                      fragmentStaticLinkingDescriptor() const;

    RenderPipelineDescriptor*                     init();

    MTL::PrimitiveTopologyClass                   inputPrimitiveTopology() const;

    bool                                          isRasterizationEnabled() const;

    NS::UInteger                                  maxVertexAmplificationCount() const;

    NS::UInteger                                  rasterSampleCount() const;

    [[deprecated("please use isRasterizationEnabled instead")]]
    bool                              rasterizationEnabled() const;

    void                              reset();

    void                              setAlphaToCoverageState(MTL4::AlphaToCoverageState alphaToCoverageState);

    void                              setAlphaToOneState(MTL4::AlphaToOneState alphaToOneState);

    void                              setColorAttachmentMappingState(MTL4::LogicalToPhysicalColorAttachmentMappingState colorAttachmentMappingState);

    void                              setFragmentFunctionDescriptor(const MTL4::FunctionDescriptor* fragmentFunctionDescriptor);

    void                              setFragmentStaticLinkingDescriptor(const MTL4::StaticLinkingDescriptor* fragmentStaticLinkingDescriptor);

    void                              setInputPrimitiveTopology(MTL::PrimitiveTopologyClass inputPrimitiveTopology);

    void                              setMaxVertexAmplificationCount(NS::UInteger maxVertexAmplificationCount);

    void                              setRasterSampleCount(NS::UInteger rasterSampleCount);

    void                              setRasterizationEnabled(bool rasterizationEnabled);

    void                              setSupportFragmentBinaryLinking(bool supportFragmentBinaryLinking);

    void                              setSupportIndirectCommandBuffers(MTL4::IndirectCommandBufferSupportState supportIndirectCommandBuffers);

    void                              setSupportVertexBinaryLinking(bool supportVertexBinaryLinking);

    void                              setVertexDescriptor(const MTL::VertexDescriptor* vertexDescriptor);

    void                              setVertexFunctionDescriptor(const MTL4::FunctionDescriptor* vertexFunctionDescriptor);

    void                              setVertexStaticLinkingDescriptor(const MTL4::StaticLinkingDescriptor* vertexStaticLinkingDescriptor);

    bool                              supportFragmentBinaryLinking() const;

    IndirectCommandBufferSupportState supportIndirectCommandBuffers() const;

    bool                              supportVertexBinaryLinking() const;

    MTL::VertexDescriptor*            vertexDescriptor() const;

    FunctionDescriptor*               vertexFunctionDescriptor() const;

    StaticLinkingDescriptor*          vertexStaticLinkingDescriptor() const;
};

}
_MTL_INLINE MTL4::RenderPipelineColorAttachmentDescriptor* MTL4::RenderPipelineColorAttachmentDescriptor::alloc()
{
    return NS::Object::alloc<MTL4::RenderPipelineColorAttachmentDescriptor>(_MTL_PRIVATE_CLS(MTL4RenderPipelineColorAttachmentDescriptor));
}

_MTL_INLINE MTL::BlendOperation MTL4::RenderPipelineColorAttachmentDescriptor::alphaBlendOperation() const
{
    return Object::sendMessage<MTL::BlendOperation>(this, _MTL_PRIVATE_SEL(alphaBlendOperation));
}

_MTL_INLINE MTL4::BlendState MTL4::RenderPipelineColorAttachmentDescriptor::blendingState() const
{
    return Object::sendMessage<MTL4::BlendState>(this, _MTL_PRIVATE_SEL(blendingState));
}

_MTL_INLINE MTL::BlendFactor MTL4::RenderPipelineColorAttachmentDescriptor::destinationAlphaBlendFactor() const
{
    return Object::sendMessage<MTL::BlendFactor>(this, _MTL_PRIVATE_SEL(destinationAlphaBlendFactor));
}

_MTL_INLINE MTL::BlendFactor MTL4::RenderPipelineColorAttachmentDescriptor::destinationRGBBlendFactor() const
{
    return Object::sendMessage<MTL::BlendFactor>(this, _MTL_PRIVATE_SEL(destinationRGBBlendFactor));
}

_MTL_INLINE MTL4::RenderPipelineColorAttachmentDescriptor* MTL4::RenderPipelineColorAttachmentDescriptor::init()
{
    return NS::Object::init<MTL4::RenderPipelineColorAttachmentDescriptor>();
}

_MTL_INLINE MTL::PixelFormat MTL4::RenderPipelineColorAttachmentDescriptor::pixelFormat() const
{
    return Object::sendMessage<MTL::PixelFormat>(this, _MTL_PRIVATE_SEL(pixelFormat));
}

_MTL_INLINE void MTL4::RenderPipelineColorAttachmentDescriptor::reset()
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(reset));
}

_MTL_INLINE MTL::BlendOperation MTL4::RenderPipelineColorAttachmentDescriptor::rgbBlendOperation() const
{
    return Object::sendMessage<MTL::BlendOperation>(this, _MTL_PRIVATE_SEL(rgbBlendOperation));
}

_MTL_INLINE void MTL4::RenderPipelineColorAttachmentDescriptor::setAlphaBlendOperation(MTL::BlendOperation alphaBlendOperation)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setAlphaBlendOperation_), alphaBlendOperation);
}

_MTL_INLINE void MTL4::RenderPipelineColorAttachmentDescriptor::setBlendingState(MTL4::BlendState blendingState)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setBlendingState_), blendingState);
}

_MTL_INLINE void MTL4::RenderPipelineColorAttachmentDescriptor::setDestinationAlphaBlendFactor(MTL::BlendFactor destinationAlphaBlendFactor)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setDestinationAlphaBlendFactor_), destinationAlphaBlendFactor);
}

_MTL_INLINE void MTL4::RenderPipelineColorAttachmentDescriptor::setDestinationRGBBlendFactor(MTL::BlendFactor destinationRGBBlendFactor)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setDestinationRGBBlendFactor_), destinationRGBBlendFactor);
}

_MTL_INLINE void MTL4::RenderPipelineColorAttachmentDescriptor::setPixelFormat(MTL::PixelFormat pixelFormat)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setPixelFormat_), pixelFormat);
}

_MTL_INLINE void MTL4::RenderPipelineColorAttachmentDescriptor::setRgbBlendOperation(MTL::BlendOperation rgbBlendOperation)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setRgbBlendOperation_), rgbBlendOperation);
}

_MTL_INLINE void MTL4::RenderPipelineColorAttachmentDescriptor::setSourceAlphaBlendFactor(MTL::BlendFactor sourceAlphaBlendFactor)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setSourceAlphaBlendFactor_), sourceAlphaBlendFactor);
}

_MTL_INLINE void MTL4::RenderPipelineColorAttachmentDescriptor::setSourceRGBBlendFactor(MTL::BlendFactor sourceRGBBlendFactor)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setSourceRGBBlendFactor_), sourceRGBBlendFactor);
}

_MTL_INLINE void MTL4::RenderPipelineColorAttachmentDescriptor::setWriteMask(MTL::ColorWriteMask writeMask)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setWriteMask_), writeMask);
}

_MTL_INLINE MTL::BlendFactor MTL4::RenderPipelineColorAttachmentDescriptor::sourceAlphaBlendFactor() const
{
    return Object::sendMessage<MTL::BlendFactor>(this, _MTL_PRIVATE_SEL(sourceAlphaBlendFactor));
}

_MTL_INLINE MTL::BlendFactor MTL4::RenderPipelineColorAttachmentDescriptor::sourceRGBBlendFactor() const
{
    return Object::sendMessage<MTL::BlendFactor>(this, _MTL_PRIVATE_SEL(sourceRGBBlendFactor));
}

_MTL_INLINE MTL::ColorWriteMask MTL4::RenderPipelineColorAttachmentDescriptor::writeMask() const
{
    return Object::sendMessage<MTL::ColorWriteMask>(this, _MTL_PRIVATE_SEL(writeMask));
}

_MTL_INLINE MTL4::RenderPipelineColorAttachmentDescriptorArray* MTL4::RenderPipelineColorAttachmentDescriptorArray::alloc()
{
    return NS::Object::alloc<MTL4::RenderPipelineColorAttachmentDescriptorArray>(_MTL_PRIVATE_CLS(MTL4RenderPipelineColorAttachmentDescriptorArray));
}

_MTL_INLINE MTL4::RenderPipelineColorAttachmentDescriptorArray* MTL4::RenderPipelineColorAttachmentDescriptorArray::init()
{
    return NS::Object::init<MTL4::RenderPipelineColorAttachmentDescriptorArray>();
}

_MTL_INLINE MTL4::RenderPipelineColorAttachmentDescriptor* MTL4::RenderPipelineColorAttachmentDescriptorArray::object(NS::UInteger attachmentIndex)
{
    return Object::sendMessage<MTL4::RenderPipelineColorAttachmentDescriptor*>(this, _MTL_PRIVATE_SEL(objectAtIndexedSubscript_), attachmentIndex);
}

_MTL_INLINE void MTL4::RenderPipelineColorAttachmentDescriptorArray::reset()
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(reset));
}

_MTL_INLINE void MTL4::RenderPipelineColorAttachmentDescriptorArray::setObject(const MTL4::RenderPipelineColorAttachmentDescriptor* attachment, NS::UInteger attachmentIndex)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setObject_atIndexedSubscript_), attachment, attachmentIndex);
}

_MTL_INLINE MTL4::RenderPipelineBinaryFunctionsDescriptor* MTL4::RenderPipelineBinaryFunctionsDescriptor::alloc()
{
    return NS::Object::alloc<MTL4::RenderPipelineBinaryFunctionsDescriptor>(_MTL_PRIVATE_CLS(MTL4RenderPipelineBinaryFunctionsDescriptor));
}

_MTL_INLINE NS::Array* MTL4::RenderPipelineBinaryFunctionsDescriptor::fragmentAdditionalBinaryFunctions() const
{
    return Object::sendMessage<NS::Array*>(this, _MTL_PRIVATE_SEL(fragmentAdditionalBinaryFunctions));
}

_MTL_INLINE MTL4::RenderPipelineBinaryFunctionsDescriptor* MTL4::RenderPipelineBinaryFunctionsDescriptor::init()
{
    return NS::Object::init<MTL4::RenderPipelineBinaryFunctionsDescriptor>();
}

_MTL_INLINE NS::Array* MTL4::RenderPipelineBinaryFunctionsDescriptor::meshAdditionalBinaryFunctions() const
{
    return Object::sendMessage<NS::Array*>(this, _MTL_PRIVATE_SEL(meshAdditionalBinaryFunctions));
}

_MTL_INLINE NS::Array* MTL4::RenderPipelineBinaryFunctionsDescriptor::objectAdditionalBinaryFunctions() const
{
    return Object::sendMessage<NS::Array*>(this, _MTL_PRIVATE_SEL(objectAdditionalBinaryFunctions));
}

_MTL_INLINE void MTL4::RenderPipelineBinaryFunctionsDescriptor::reset()
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(reset));
}

_MTL_INLINE void MTL4::RenderPipelineBinaryFunctionsDescriptor::setFragmentAdditionalBinaryFunctions(const NS::Array* fragmentAdditionalBinaryFunctions)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setFragmentAdditionalBinaryFunctions_), fragmentAdditionalBinaryFunctions);
}

_MTL_INLINE void MTL4::RenderPipelineBinaryFunctionsDescriptor::setMeshAdditionalBinaryFunctions(const NS::Array* meshAdditionalBinaryFunctions)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMeshAdditionalBinaryFunctions_), meshAdditionalBinaryFunctions);
}

_MTL_INLINE void MTL4::RenderPipelineBinaryFunctionsDescriptor::setObjectAdditionalBinaryFunctions(const NS::Array* objectAdditionalBinaryFunctions)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setObjectAdditionalBinaryFunctions_), objectAdditionalBinaryFunctions);
}

_MTL_INLINE void MTL4::RenderPipelineBinaryFunctionsDescriptor::setTileAdditionalBinaryFunctions(const NS::Array* tileAdditionalBinaryFunctions)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setTileAdditionalBinaryFunctions_), tileAdditionalBinaryFunctions);
}

_MTL_INLINE void MTL4::RenderPipelineBinaryFunctionsDescriptor::setVertexAdditionalBinaryFunctions(const NS::Array* vertexAdditionalBinaryFunctions)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setVertexAdditionalBinaryFunctions_), vertexAdditionalBinaryFunctions);
}

_MTL_INLINE NS::Array* MTL4::RenderPipelineBinaryFunctionsDescriptor::tileAdditionalBinaryFunctions() const
{
    return Object::sendMessage<NS::Array*>(this, _MTL_PRIVATE_SEL(tileAdditionalBinaryFunctions));
}

_MTL_INLINE NS::Array* MTL4::RenderPipelineBinaryFunctionsDescriptor::vertexAdditionalBinaryFunctions() const
{
    return Object::sendMessage<NS::Array*>(this, _MTL_PRIVATE_SEL(vertexAdditionalBinaryFunctions));
}

_MTL_INLINE MTL4::RenderPipelineDescriptor* MTL4::RenderPipelineDescriptor::alloc()
{
    return NS::Object::alloc<MTL4::RenderPipelineDescriptor>(_MTL_PRIVATE_CLS(MTL4RenderPipelineDescriptor));
}

_MTL_INLINE MTL4::AlphaToCoverageState MTL4::RenderPipelineDescriptor::alphaToCoverageState() const
{
    return Object::sendMessage<MTL4::AlphaToCoverageState>(this, _MTL_PRIVATE_SEL(alphaToCoverageState));
}

_MTL_INLINE MTL4::AlphaToOneState MTL4::RenderPipelineDescriptor::alphaToOneState() const
{
    return Object::sendMessage<MTL4::AlphaToOneState>(this, _MTL_PRIVATE_SEL(alphaToOneState));
}

_MTL_INLINE MTL4::LogicalToPhysicalColorAttachmentMappingState MTL4::RenderPipelineDescriptor::colorAttachmentMappingState() const
{
    return Object::sendMessage<MTL4::LogicalToPhysicalColorAttachmentMappingState>(this, _MTL_PRIVATE_SEL(colorAttachmentMappingState));
}

_MTL_INLINE MTL4::RenderPipelineColorAttachmentDescriptorArray* MTL4::RenderPipelineDescriptor::colorAttachments() const
{
    return Object::sendMessage<MTL4::RenderPipelineColorAttachmentDescriptorArray*>(this, _MTL_PRIVATE_SEL(colorAttachments));
}

_MTL_INLINE MTL4::FunctionDescriptor* MTL4::RenderPipelineDescriptor::fragmentFunctionDescriptor() const
{
    return Object::sendMessage<MTL4::FunctionDescriptor*>(this, _MTL_PRIVATE_SEL(fragmentFunctionDescriptor));
}

_MTL_INLINE MTL4::StaticLinkingDescriptor* MTL4::RenderPipelineDescriptor::fragmentStaticLinkingDescriptor() const
{
    return Object::sendMessage<MTL4::StaticLinkingDescriptor*>(this, _MTL_PRIVATE_SEL(fragmentStaticLinkingDescriptor));
}

_MTL_INLINE MTL4::RenderPipelineDescriptor* MTL4::RenderPipelineDescriptor::init()
{
    return NS::Object::init<MTL4::RenderPipelineDescriptor>();
}

_MTL_INLINE MTL::PrimitiveTopologyClass MTL4::RenderPipelineDescriptor::inputPrimitiveTopology() const
{
    return Object::sendMessage<MTL::PrimitiveTopologyClass>(this, _MTL_PRIVATE_SEL(inputPrimitiveTopology));
}

_MTL_INLINE bool MTL4::RenderPipelineDescriptor::isRasterizationEnabled() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(isRasterizationEnabled));
}

_MTL_INLINE NS::UInteger MTL4::RenderPipelineDescriptor::maxVertexAmplificationCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(maxVertexAmplificationCount));
}

_MTL_INLINE NS::UInteger MTL4::RenderPipelineDescriptor::rasterSampleCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(rasterSampleCount));
}

_MTL_INLINE bool MTL4::RenderPipelineDescriptor::rasterizationEnabled() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(isRasterizationEnabled));
}

_MTL_INLINE void MTL4::RenderPipelineDescriptor::reset()
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(reset));
}

_MTL_INLINE void MTL4::RenderPipelineDescriptor::setAlphaToCoverageState(MTL4::AlphaToCoverageState alphaToCoverageState)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setAlphaToCoverageState_), alphaToCoverageState);
}

_MTL_INLINE void MTL4::RenderPipelineDescriptor::setAlphaToOneState(MTL4::AlphaToOneState alphaToOneState)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setAlphaToOneState_), alphaToOneState);
}

_MTL_INLINE void MTL4::RenderPipelineDescriptor::setColorAttachmentMappingState(MTL4::LogicalToPhysicalColorAttachmentMappingState colorAttachmentMappingState)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setColorAttachmentMappingState_), colorAttachmentMappingState);
}

_MTL_INLINE void MTL4::RenderPipelineDescriptor::setFragmentFunctionDescriptor(const MTL4::FunctionDescriptor* fragmentFunctionDescriptor)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setFragmentFunctionDescriptor_), fragmentFunctionDescriptor);
}

_MTL_INLINE void MTL4::RenderPipelineDescriptor::setFragmentStaticLinkingDescriptor(const MTL4::StaticLinkingDescriptor* fragmentStaticLinkingDescriptor)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setFragmentStaticLinkingDescriptor_), fragmentStaticLinkingDescriptor);
}

_MTL_INLINE void MTL4::RenderPipelineDescriptor::setInputPrimitiveTopology(MTL::PrimitiveTopologyClass inputPrimitiveTopology)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setInputPrimitiveTopology_), inputPrimitiveTopology);
}

_MTL_INLINE void MTL4::RenderPipelineDescriptor::setMaxVertexAmplificationCount(NS::UInteger maxVertexAmplificationCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMaxVertexAmplificationCount_), maxVertexAmplificationCount);
}

_MTL_INLINE void MTL4::RenderPipelineDescriptor::setRasterSampleCount(NS::UInteger rasterSampleCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setRasterSampleCount_), rasterSampleCount);
}

_MTL_INLINE void MTL4::RenderPipelineDescriptor::setRasterizationEnabled(bool rasterizationEnabled)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setRasterizationEnabled_), rasterizationEnabled);
}

_MTL_INLINE void MTL4::RenderPipelineDescriptor::setSupportFragmentBinaryLinking(bool supportFragmentBinaryLinking)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setSupportFragmentBinaryLinking_), supportFragmentBinaryLinking);
}

_MTL_INLINE void MTL4::RenderPipelineDescriptor::setSupportIndirectCommandBuffers(MTL4::IndirectCommandBufferSupportState supportIndirectCommandBuffers)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setSupportIndirectCommandBuffers_), supportIndirectCommandBuffers);
}

_MTL_INLINE void MTL4::RenderPipelineDescriptor::setSupportVertexBinaryLinking(bool supportVertexBinaryLinking)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setSupportVertexBinaryLinking_), supportVertexBinaryLinking);
}

_MTL_INLINE void MTL4::RenderPipelineDescriptor::setVertexDescriptor(const MTL::VertexDescriptor* vertexDescriptor)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setVertexDescriptor_), vertexDescriptor);
}

_MTL_INLINE void MTL4::RenderPipelineDescriptor::setVertexFunctionDescriptor(const MTL4::FunctionDescriptor* vertexFunctionDescriptor)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setVertexFunctionDescriptor_), vertexFunctionDescriptor);
}

_MTL_INLINE void MTL4::RenderPipelineDescriptor::setVertexStaticLinkingDescriptor(const MTL4::StaticLinkingDescriptor* vertexStaticLinkingDescriptor)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setVertexStaticLinkingDescriptor_), vertexStaticLinkingDescriptor);
}

_MTL_INLINE bool MTL4::RenderPipelineDescriptor::supportFragmentBinaryLinking() const
{
    return Object::sendMessageSafe<bool>(this, _MTL_PRIVATE_SEL(supportFragmentBinaryLinking));
}

_MTL_INLINE MTL4::IndirectCommandBufferSupportState MTL4::RenderPipelineDescriptor::supportIndirectCommandBuffers() const
{
    return Object::sendMessage<MTL4::IndirectCommandBufferSupportState>(this, _MTL_PRIVATE_SEL(supportIndirectCommandBuffers));
}

_MTL_INLINE bool MTL4::RenderPipelineDescriptor::supportVertexBinaryLinking() const
{
    return Object::sendMessageSafe<bool>(this, _MTL_PRIVATE_SEL(supportVertexBinaryLinking));
}

_MTL_INLINE MTL::VertexDescriptor* MTL4::RenderPipelineDescriptor::vertexDescriptor() const
{
    return Object::sendMessage<MTL::VertexDescriptor*>(this, _MTL_PRIVATE_SEL(vertexDescriptor));
}

_MTL_INLINE MTL4::FunctionDescriptor* MTL4::RenderPipelineDescriptor::vertexFunctionDescriptor() const
{
    return Object::sendMessage<MTL4::FunctionDescriptor*>(this, _MTL_PRIVATE_SEL(vertexFunctionDescriptor));
}

_MTL_INLINE MTL4::StaticLinkingDescriptor* MTL4::RenderPipelineDescriptor::vertexStaticLinkingDescriptor() const
{
    return Object::sendMessage<MTL4::StaticLinkingDescriptor*>(this, _MTL_PRIVATE_SEL(vertexStaticLinkingDescriptor));
}
