#pragma once

#include "MTL4Defines.hpp"
#include "MTL4Blocks.hpp"
#include "MTL4Structs.hpp"
#include "MTL4Bridge.hpp"
#include "../Foundation/NSObject.hpp"
#include "../Foundation/NSTypes.hpp"
#include "../Foundation/NSRange.hpp"
#include "MTL4PipelineState.hpp"

namespace MTL {
    class VertexDescriptor;
    enum BlendFactor : NS::UInteger;
    enum BlendOperation : NS::UInteger;
    using ColorWriteMask = NS::UInteger;
    enum PixelFormat : NS::UInteger;
    enum PrimitiveTopologyClass : NS::UInteger;
}
namespace MTL4 {
    class FunctionDescriptor;
    class StaticLinkingDescriptor;
    enum AlphaToCoverageState : NS::Integer;
    enum AlphaToOneState : NS::Integer;
    enum BlendState : NS::Integer;
    enum IndirectCommandBufferSupportState : NS::Integer;
}
namespace NS {
    class Array;
}

namespace MTL4
{

_MTL4_ENUM(NS::Integer, LogicalToPhysicalColorAttachmentMappingState) {
    LogicalToPhysicalColorAttachmentMappingStateIdentity = 0,
    LogicalToPhysicalColorAttachmentMappingStateInherited = 1,
};


class RenderPipelineColorAttachmentDescriptor;
class RenderPipelineColorAttachmentDescriptorArray;
class RenderPipelineBinaryFunctionsDescriptor;
class RenderPipelineDescriptor;

class RenderPipelineColorAttachmentDescriptor : public NS::Copying<RenderPipelineColorAttachmentDescriptor>
{
public:
    static RenderPipelineColorAttachmentDescriptor* alloc();
    RenderPipelineColorAttachmentDescriptor*        init() const;

    MTL::BlendOperation alphaBlendOperation() const;
    MTL4::BlendState    blendingState() const;
    MTL::BlendFactor    destinationAlphaBlendFactor() const;
    MTL::BlendFactor    destinationRGBBlendFactor() const;
    MTL::PixelFormat    pixelFormat() const;
    void                reset();
    MTL::BlendOperation rgbBlendOperation() const;
    void                setAlphaBlendOperation(MTL::BlendOperation alphaBlendOperation);
    void                setBlendingState(MTL4::BlendState blendingState);
    void                setDestinationAlphaBlendFactor(MTL::BlendFactor destinationAlphaBlendFactor);
    void                setDestinationRGBBlendFactor(MTL::BlendFactor destinationRGBBlendFactor);
    void                setPixelFormat(MTL::PixelFormat pixelFormat);
    void                setRgbBlendOperation(MTL::BlendOperation rgbBlendOperation);
    void                setSourceAlphaBlendFactor(MTL::BlendFactor sourceAlphaBlendFactor);
    void                setSourceRGBBlendFactor(MTL::BlendFactor sourceRGBBlendFactor);
    void                setWriteMask(MTL::ColorWriteMask writeMask);
    MTL::BlendFactor    sourceAlphaBlendFactor() const;
    MTL::BlendFactor    sourceRGBBlendFactor() const;
    MTL::ColorWriteMask writeMask() const;

};

class RenderPipelineColorAttachmentDescriptorArray : public NS::Copying<RenderPipelineColorAttachmentDescriptorArray>
{
public:
    static RenderPipelineColorAttachmentDescriptorArray* alloc();
    RenderPipelineColorAttachmentDescriptorArray*        init() const;

    MTL4::RenderPipelineColorAttachmentDescriptor* object(NS::UInteger attachmentIndex);
    void                                           reset();
    void                                           setObject(MTL4::RenderPipelineColorAttachmentDescriptor* attachment, NS::UInteger attachmentIndex);

};

class RenderPipelineBinaryFunctionsDescriptor : public NS::Copying<RenderPipelineBinaryFunctionsDescriptor>
{
public:
    static RenderPipelineBinaryFunctionsDescriptor* alloc();
    RenderPipelineBinaryFunctionsDescriptor*        init() const;

    NS::Array* fragmentAdditionalBinaryFunctions() const;
    NS::Array* meshAdditionalBinaryFunctions() const;
    NS::Array* objectAdditionalBinaryFunctions() const;
    void       reset();
    void       setFragmentAdditionalBinaryFunctions(NS::Array* fragmentAdditionalBinaryFunctions);
    void       setMeshAdditionalBinaryFunctions(NS::Array* meshAdditionalBinaryFunctions);
    void       setObjectAdditionalBinaryFunctions(NS::Array* objectAdditionalBinaryFunctions);
    void       setTileAdditionalBinaryFunctions(NS::Array* tileAdditionalBinaryFunctions);
    void       setVertexAdditionalBinaryFunctions(NS::Array* vertexAdditionalBinaryFunctions);
    NS::Array* tileAdditionalBinaryFunctions() const;
    NS::Array* vertexAdditionalBinaryFunctions() const;

};

class RenderPipelineDescriptor : public NS::Referencing<RenderPipelineDescriptor, MTL4::PipelineDescriptor>
{
public:
    static RenderPipelineDescriptor* alloc();
    RenderPipelineDescriptor*        init() const;

    MTL4::AlphaToCoverageState                          alphaToCoverageState() const;
    MTL4::AlphaToOneState                               alphaToOneState() const;
    MTL4::LogicalToPhysicalColorAttachmentMappingState  colorAttachmentMappingState() const;
    MTL4::RenderPipelineColorAttachmentDescriptorArray* colorAttachments() const;
    MTL4::FunctionDescriptor*                           fragmentFunctionDescriptor() const;
    MTL4::StaticLinkingDescriptor*                      fragmentStaticLinkingDescriptor() const;
    MTL::PrimitiveTopologyClass                         inputPrimitiveTopology() const;
    bool                                                isRasterizationEnabled();
    NS::UInteger                                        maxVertexAmplificationCount() const;
    NS::UInteger                                        rasterSampleCount() const;
    bool                                                rasterizationEnabled() const;
    void                                                reset();
    void                                                setAlphaToCoverageState(MTL4::AlphaToCoverageState alphaToCoverageState);
    void                                                setAlphaToOneState(MTL4::AlphaToOneState alphaToOneState);
    void                                                setColorAttachmentMappingState(MTL4::LogicalToPhysicalColorAttachmentMappingState colorAttachmentMappingState);
    void                                                setFragmentFunctionDescriptor(MTL4::FunctionDescriptor* fragmentFunctionDescriptor);
    void                                                setFragmentStaticLinkingDescriptor(MTL4::StaticLinkingDescriptor* fragmentStaticLinkingDescriptor);
    void                                                setInputPrimitiveTopology(MTL::PrimitiveTopologyClass inputPrimitiveTopology);
    void                                                setMaxVertexAmplificationCount(NS::UInteger maxVertexAmplificationCount);
    void                                                setRasterSampleCount(NS::UInteger rasterSampleCount);
    void                                                setRasterizationEnabled(bool rasterizationEnabled);
    void                                                setSupportFragmentBinaryLinking(bool supportFragmentBinaryLinking);
    void                                                setSupportIndirectCommandBuffers(MTL4::IndirectCommandBufferSupportState supportIndirectCommandBuffers);
    void                                                setSupportVertexBinaryLinking(bool supportVertexBinaryLinking);
    void                                                setVertexDescriptor(MTL::VertexDescriptor* vertexDescriptor);
    void                                                setVertexFunctionDescriptor(MTL4::FunctionDescriptor* vertexFunctionDescriptor);
    void                                                setVertexStaticLinkingDescriptor(MTL4::StaticLinkingDescriptor* vertexStaticLinkingDescriptor);
    bool                                                supportFragmentBinaryLinking() const;
    MTL4::IndirectCommandBufferSupportState             supportIndirectCommandBuffers() const;
    bool                                                supportVertexBinaryLinking() const;
    MTL::VertexDescriptor*                              vertexDescriptor() const;
    MTL4::FunctionDescriptor*                           vertexFunctionDescriptor() const;
    MTL4::StaticLinkingDescriptor*                      vertexStaticLinkingDescriptor() const;

};

} // namespace MTL4

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTL4RenderPipelineColorAttachmentDescriptor;
extern "C" void *OBJC_CLASS_$_MTL4RenderPipelineColorAttachmentDescriptorArray;
extern "C" void *OBJC_CLASS_$_MTL4RenderPipelineBinaryFunctionsDescriptor;
extern "C" void *OBJC_CLASS_$_MTL4RenderPipelineDescriptor;

_MTL4_INLINE MTL4::RenderPipelineColorAttachmentDescriptor* MTL4::RenderPipelineColorAttachmentDescriptor::alloc()
{
    return _MTL4_msg_MTL4__RenderPipelineColorAttachmentDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTL4RenderPipelineColorAttachmentDescriptor, nullptr);
}

_MTL4_INLINE MTL4::RenderPipelineColorAttachmentDescriptor* MTL4::RenderPipelineColorAttachmentDescriptor::init() const
{
    return _MTL4_msg_MTL4__RenderPipelineColorAttachmentDescriptorp_init((const void*)this, nullptr);
}

_MTL4_INLINE MTL::PixelFormat MTL4::RenderPipelineColorAttachmentDescriptor::pixelFormat() const
{
    return _MTL4_msg_MTL__PixelFormat_pixelFormat((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::RenderPipelineColorAttachmentDescriptor::setPixelFormat(MTL::PixelFormat pixelFormat)
{
    _MTL4_msg_v_setPixelFormat__MTL__PixelFormat((const void*)this, nullptr, pixelFormat);
}

_MTL4_INLINE MTL4::BlendState MTL4::RenderPipelineColorAttachmentDescriptor::blendingState() const
{
    return _MTL4_msg_MTL4__BlendState_blendingState((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::RenderPipelineColorAttachmentDescriptor::setBlendingState(MTL4::BlendState blendingState)
{
    _MTL4_msg_v_setBlendingState__MTL4__BlendState((const void*)this, nullptr, blendingState);
}

_MTL4_INLINE MTL::BlendFactor MTL4::RenderPipelineColorAttachmentDescriptor::sourceRGBBlendFactor() const
{
    return _MTL4_msg_MTL__BlendFactor_sourceRGBBlendFactor((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::RenderPipelineColorAttachmentDescriptor::setSourceRGBBlendFactor(MTL::BlendFactor sourceRGBBlendFactor)
{
    _MTL4_msg_v_setSourceRGBBlendFactor__MTL__BlendFactor((const void*)this, nullptr, sourceRGBBlendFactor);
}

_MTL4_INLINE MTL::BlendFactor MTL4::RenderPipelineColorAttachmentDescriptor::destinationRGBBlendFactor() const
{
    return _MTL4_msg_MTL__BlendFactor_destinationRGBBlendFactor((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::RenderPipelineColorAttachmentDescriptor::setDestinationRGBBlendFactor(MTL::BlendFactor destinationRGBBlendFactor)
{
    _MTL4_msg_v_setDestinationRGBBlendFactor__MTL__BlendFactor((const void*)this, nullptr, destinationRGBBlendFactor);
}

_MTL4_INLINE MTL::BlendOperation MTL4::RenderPipelineColorAttachmentDescriptor::rgbBlendOperation() const
{
    return _MTL4_msg_MTL__BlendOperation_rgbBlendOperation((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::RenderPipelineColorAttachmentDescriptor::setRgbBlendOperation(MTL::BlendOperation rgbBlendOperation)
{
    _MTL4_msg_v_setRgbBlendOperation__MTL__BlendOperation((const void*)this, nullptr, rgbBlendOperation);
}

_MTL4_INLINE MTL::BlendFactor MTL4::RenderPipelineColorAttachmentDescriptor::sourceAlphaBlendFactor() const
{
    return _MTL4_msg_MTL__BlendFactor_sourceAlphaBlendFactor((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::RenderPipelineColorAttachmentDescriptor::setSourceAlphaBlendFactor(MTL::BlendFactor sourceAlphaBlendFactor)
{
    _MTL4_msg_v_setSourceAlphaBlendFactor__MTL__BlendFactor((const void*)this, nullptr, sourceAlphaBlendFactor);
}

_MTL4_INLINE MTL::BlendFactor MTL4::RenderPipelineColorAttachmentDescriptor::destinationAlphaBlendFactor() const
{
    return _MTL4_msg_MTL__BlendFactor_destinationAlphaBlendFactor((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::RenderPipelineColorAttachmentDescriptor::setDestinationAlphaBlendFactor(MTL::BlendFactor destinationAlphaBlendFactor)
{
    _MTL4_msg_v_setDestinationAlphaBlendFactor__MTL__BlendFactor((const void*)this, nullptr, destinationAlphaBlendFactor);
}

_MTL4_INLINE MTL::BlendOperation MTL4::RenderPipelineColorAttachmentDescriptor::alphaBlendOperation() const
{
    return _MTL4_msg_MTL__BlendOperation_alphaBlendOperation((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::RenderPipelineColorAttachmentDescriptor::setAlphaBlendOperation(MTL::BlendOperation alphaBlendOperation)
{
    _MTL4_msg_v_setAlphaBlendOperation__MTL__BlendOperation((const void*)this, nullptr, alphaBlendOperation);
}

_MTL4_INLINE MTL::ColorWriteMask MTL4::RenderPipelineColorAttachmentDescriptor::writeMask() const
{
    return _MTL4_msg_MTL__ColorWriteMask_writeMask((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::RenderPipelineColorAttachmentDescriptor::setWriteMask(MTL::ColorWriteMask writeMask)
{
    _MTL4_msg_v_setWriteMask__MTL__ColorWriteMask((const void*)this, nullptr, writeMask);
}

_MTL4_INLINE void MTL4::RenderPipelineColorAttachmentDescriptor::reset()
{
    _MTL4_msg_v_reset((const void*)this, nullptr);
}

_MTL4_INLINE MTL4::RenderPipelineColorAttachmentDescriptorArray* MTL4::RenderPipelineColorAttachmentDescriptorArray::alloc()
{
    return _MTL4_msg_MTL4__RenderPipelineColorAttachmentDescriptorArrayp_alloc((const void*)&OBJC_CLASS_$_MTL4RenderPipelineColorAttachmentDescriptorArray, nullptr);
}

_MTL4_INLINE MTL4::RenderPipelineColorAttachmentDescriptorArray* MTL4::RenderPipelineColorAttachmentDescriptorArray::init() const
{
    return _MTL4_msg_MTL4__RenderPipelineColorAttachmentDescriptorArrayp_init((const void*)this, nullptr);
}

_MTL4_INLINE MTL4::RenderPipelineColorAttachmentDescriptor* MTL4::RenderPipelineColorAttachmentDescriptorArray::object(NS::UInteger attachmentIndex)
{
    return _MTL4_msg_MTL4__RenderPipelineColorAttachmentDescriptorp_objectAtIndexedSubscript__NS__UInteger((const void*)this, nullptr, attachmentIndex);
}

_MTL4_INLINE void MTL4::RenderPipelineColorAttachmentDescriptorArray::setObject(MTL4::RenderPipelineColorAttachmentDescriptor* attachment, NS::UInteger attachmentIndex)
{
    _MTL4_msg_v_setObject_atIndexedSubscript__MTL4__RenderPipelineColorAttachmentDescriptorp_NS__UInteger((const void*)this, nullptr, attachment, attachmentIndex);
}

_MTL4_INLINE void MTL4::RenderPipelineColorAttachmentDescriptorArray::reset()
{
    _MTL4_msg_v_reset((const void*)this, nullptr);
}

_MTL4_INLINE MTL4::RenderPipelineBinaryFunctionsDescriptor* MTL4::RenderPipelineBinaryFunctionsDescriptor::alloc()
{
    return _MTL4_msg_MTL4__RenderPipelineBinaryFunctionsDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTL4RenderPipelineBinaryFunctionsDescriptor, nullptr);
}

_MTL4_INLINE MTL4::RenderPipelineBinaryFunctionsDescriptor* MTL4::RenderPipelineBinaryFunctionsDescriptor::init() const
{
    return _MTL4_msg_MTL4__RenderPipelineBinaryFunctionsDescriptorp_init((const void*)this, nullptr);
}

_MTL4_INLINE NS::Array* MTL4::RenderPipelineBinaryFunctionsDescriptor::vertexAdditionalBinaryFunctions() const
{
    return _MTL4_msg_NS__Arrayp_vertexAdditionalBinaryFunctions((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::RenderPipelineBinaryFunctionsDescriptor::setVertexAdditionalBinaryFunctions(NS::Array* vertexAdditionalBinaryFunctions)
{
    _MTL4_msg_v_setVertexAdditionalBinaryFunctions__NS__Arrayp((const void*)this, nullptr, vertexAdditionalBinaryFunctions);
}

_MTL4_INLINE NS::Array* MTL4::RenderPipelineBinaryFunctionsDescriptor::fragmentAdditionalBinaryFunctions() const
{
    return _MTL4_msg_NS__Arrayp_fragmentAdditionalBinaryFunctions((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::RenderPipelineBinaryFunctionsDescriptor::setFragmentAdditionalBinaryFunctions(NS::Array* fragmentAdditionalBinaryFunctions)
{
    _MTL4_msg_v_setFragmentAdditionalBinaryFunctions__NS__Arrayp((const void*)this, nullptr, fragmentAdditionalBinaryFunctions);
}

_MTL4_INLINE NS::Array* MTL4::RenderPipelineBinaryFunctionsDescriptor::tileAdditionalBinaryFunctions() const
{
    return _MTL4_msg_NS__Arrayp_tileAdditionalBinaryFunctions((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::RenderPipelineBinaryFunctionsDescriptor::setTileAdditionalBinaryFunctions(NS::Array* tileAdditionalBinaryFunctions)
{
    _MTL4_msg_v_setTileAdditionalBinaryFunctions__NS__Arrayp((const void*)this, nullptr, tileAdditionalBinaryFunctions);
}

_MTL4_INLINE NS::Array* MTL4::RenderPipelineBinaryFunctionsDescriptor::objectAdditionalBinaryFunctions() const
{
    return _MTL4_msg_NS__Arrayp_objectAdditionalBinaryFunctions((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::RenderPipelineBinaryFunctionsDescriptor::setObjectAdditionalBinaryFunctions(NS::Array* objectAdditionalBinaryFunctions)
{
    _MTL4_msg_v_setObjectAdditionalBinaryFunctions__NS__Arrayp((const void*)this, nullptr, objectAdditionalBinaryFunctions);
}

_MTL4_INLINE NS::Array* MTL4::RenderPipelineBinaryFunctionsDescriptor::meshAdditionalBinaryFunctions() const
{
    return _MTL4_msg_NS__Arrayp_meshAdditionalBinaryFunctions((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::RenderPipelineBinaryFunctionsDescriptor::setMeshAdditionalBinaryFunctions(NS::Array* meshAdditionalBinaryFunctions)
{
    _MTL4_msg_v_setMeshAdditionalBinaryFunctions__NS__Arrayp((const void*)this, nullptr, meshAdditionalBinaryFunctions);
}

_MTL4_INLINE void MTL4::RenderPipelineBinaryFunctionsDescriptor::reset()
{
    _MTL4_msg_v_reset((const void*)this, nullptr);
}

_MTL4_INLINE MTL4::RenderPipelineDescriptor* MTL4::RenderPipelineDescriptor::alloc()
{
    return _MTL4_msg_MTL4__RenderPipelineDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTL4RenderPipelineDescriptor, nullptr);
}

_MTL4_INLINE MTL4::RenderPipelineDescriptor* MTL4::RenderPipelineDescriptor::init() const
{
    return _MTL4_msg_MTL4__RenderPipelineDescriptorp_init((const void*)this, nullptr);
}

_MTL4_INLINE MTL4::FunctionDescriptor* MTL4::RenderPipelineDescriptor::vertexFunctionDescriptor() const
{
    return _MTL4_msg_MTL4__FunctionDescriptorp_vertexFunctionDescriptor((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::RenderPipelineDescriptor::setVertexFunctionDescriptor(MTL4::FunctionDescriptor* vertexFunctionDescriptor)
{
    _MTL4_msg_v_setVertexFunctionDescriptor__MTL4__FunctionDescriptorp((const void*)this, nullptr, vertexFunctionDescriptor);
}

_MTL4_INLINE MTL4::FunctionDescriptor* MTL4::RenderPipelineDescriptor::fragmentFunctionDescriptor() const
{
    return _MTL4_msg_MTL4__FunctionDescriptorp_fragmentFunctionDescriptor((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::RenderPipelineDescriptor::setFragmentFunctionDescriptor(MTL4::FunctionDescriptor* fragmentFunctionDescriptor)
{
    _MTL4_msg_v_setFragmentFunctionDescriptor__MTL4__FunctionDescriptorp((const void*)this, nullptr, fragmentFunctionDescriptor);
}

_MTL4_INLINE MTL::VertexDescriptor* MTL4::RenderPipelineDescriptor::vertexDescriptor() const
{
    return _MTL4_msg_MTL__VertexDescriptorp_vertexDescriptor((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::RenderPipelineDescriptor::setVertexDescriptor(MTL::VertexDescriptor* vertexDescriptor)
{
    _MTL4_msg_v_setVertexDescriptor__MTL__VertexDescriptorp((const void*)this, nullptr, vertexDescriptor);
}

_MTL4_INLINE NS::UInteger MTL4::RenderPipelineDescriptor::rasterSampleCount() const
{
    return _MTL4_msg_NS__UInteger_rasterSampleCount((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::RenderPipelineDescriptor::setRasterSampleCount(NS::UInteger rasterSampleCount)
{
    _MTL4_msg_v_setRasterSampleCount__NS__UInteger((const void*)this, nullptr, rasterSampleCount);
}

_MTL4_INLINE MTL4::AlphaToCoverageState MTL4::RenderPipelineDescriptor::alphaToCoverageState() const
{
    return _MTL4_msg_MTL4__AlphaToCoverageState_alphaToCoverageState((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::RenderPipelineDescriptor::setAlphaToCoverageState(MTL4::AlphaToCoverageState alphaToCoverageState)
{
    _MTL4_msg_v_setAlphaToCoverageState__MTL4__AlphaToCoverageState((const void*)this, nullptr, alphaToCoverageState);
}

_MTL4_INLINE MTL4::AlphaToOneState MTL4::RenderPipelineDescriptor::alphaToOneState() const
{
    return _MTL4_msg_MTL4__AlphaToOneState_alphaToOneState((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::RenderPipelineDescriptor::setAlphaToOneState(MTL4::AlphaToOneState alphaToOneState)
{
    _MTL4_msg_v_setAlphaToOneState__MTL4__AlphaToOneState((const void*)this, nullptr, alphaToOneState);
}

_MTL4_INLINE bool MTL4::RenderPipelineDescriptor::rasterizationEnabled() const
{
    return _MTL4_msg_bool_rasterizationEnabled((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::RenderPipelineDescriptor::setRasterizationEnabled(bool rasterizationEnabled)
{
    _MTL4_msg_v_setRasterizationEnabled__bool((const void*)this, nullptr, rasterizationEnabled);
}

_MTL4_INLINE NS::UInteger MTL4::RenderPipelineDescriptor::maxVertexAmplificationCount() const
{
    return _MTL4_msg_NS__UInteger_maxVertexAmplificationCount((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::RenderPipelineDescriptor::setMaxVertexAmplificationCount(NS::UInteger maxVertexAmplificationCount)
{
    _MTL4_msg_v_setMaxVertexAmplificationCount__NS__UInteger((const void*)this, nullptr, maxVertexAmplificationCount);
}

_MTL4_INLINE MTL4::RenderPipelineColorAttachmentDescriptorArray* MTL4::RenderPipelineDescriptor::colorAttachments() const
{
    return _MTL4_msg_MTL4__RenderPipelineColorAttachmentDescriptorArrayp_colorAttachments((const void*)this, nullptr);
}

_MTL4_INLINE MTL::PrimitiveTopologyClass MTL4::RenderPipelineDescriptor::inputPrimitiveTopology() const
{
    return _MTL4_msg_MTL__PrimitiveTopologyClass_inputPrimitiveTopology((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::RenderPipelineDescriptor::setInputPrimitiveTopology(MTL::PrimitiveTopologyClass inputPrimitiveTopology)
{
    _MTL4_msg_v_setInputPrimitiveTopology__MTL__PrimitiveTopologyClass((const void*)this, nullptr, inputPrimitiveTopology);
}

_MTL4_INLINE MTL4::StaticLinkingDescriptor* MTL4::RenderPipelineDescriptor::vertexStaticLinkingDescriptor() const
{
    return _MTL4_msg_MTL4__StaticLinkingDescriptorp_vertexStaticLinkingDescriptor((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::RenderPipelineDescriptor::setVertexStaticLinkingDescriptor(MTL4::StaticLinkingDescriptor* vertexStaticLinkingDescriptor)
{
    _MTL4_msg_v_setVertexStaticLinkingDescriptor__MTL4__StaticLinkingDescriptorp((const void*)this, nullptr, vertexStaticLinkingDescriptor);
}

_MTL4_INLINE MTL4::StaticLinkingDescriptor* MTL4::RenderPipelineDescriptor::fragmentStaticLinkingDescriptor() const
{
    return _MTL4_msg_MTL4__StaticLinkingDescriptorp_fragmentStaticLinkingDescriptor((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::RenderPipelineDescriptor::setFragmentStaticLinkingDescriptor(MTL4::StaticLinkingDescriptor* fragmentStaticLinkingDescriptor)
{
    _MTL4_msg_v_setFragmentStaticLinkingDescriptor__MTL4__StaticLinkingDescriptorp((const void*)this, nullptr, fragmentStaticLinkingDescriptor);
}

_MTL4_INLINE bool MTL4::RenderPipelineDescriptor::supportVertexBinaryLinking() const
{
    return _MTL4_msg_bool_supportVertexBinaryLinking((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::RenderPipelineDescriptor::setSupportVertexBinaryLinking(bool supportVertexBinaryLinking)
{
    _MTL4_msg_v_setSupportVertexBinaryLinking__bool((const void*)this, nullptr, supportVertexBinaryLinking);
}

_MTL4_INLINE bool MTL4::RenderPipelineDescriptor::supportFragmentBinaryLinking() const
{
    return _MTL4_msg_bool_supportFragmentBinaryLinking((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::RenderPipelineDescriptor::setSupportFragmentBinaryLinking(bool supportFragmentBinaryLinking)
{
    _MTL4_msg_v_setSupportFragmentBinaryLinking__bool((const void*)this, nullptr, supportFragmentBinaryLinking);
}

_MTL4_INLINE MTL4::LogicalToPhysicalColorAttachmentMappingState MTL4::RenderPipelineDescriptor::colorAttachmentMappingState() const
{
    return _MTL4_msg_MTL4__LogicalToPhysicalColorAttachmentMappingState_colorAttachmentMappingState((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::RenderPipelineDescriptor::setColorAttachmentMappingState(MTL4::LogicalToPhysicalColorAttachmentMappingState colorAttachmentMappingState)
{
    _MTL4_msg_v_setColorAttachmentMappingState__MTL4__LogicalToPhysicalColorAttachmentMappingState((const void*)this, nullptr, colorAttachmentMappingState);
}

_MTL4_INLINE MTL4::IndirectCommandBufferSupportState MTL4::RenderPipelineDescriptor::supportIndirectCommandBuffers() const
{
    return _MTL4_msg_MTL4__IndirectCommandBufferSupportState_supportIndirectCommandBuffers((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::RenderPipelineDescriptor::setSupportIndirectCommandBuffers(MTL4::IndirectCommandBufferSupportState supportIndirectCommandBuffers)
{
    _MTL4_msg_v_setSupportIndirectCommandBuffers__MTL4__IndirectCommandBufferSupportState((const void*)this, nullptr, supportIndirectCommandBuffers);
}

_MTL4_INLINE void MTL4::RenderPipelineDescriptor::reset()
{
    _MTL4_msg_v_reset((const void*)this, nullptr);
}

_MTL4_INLINE bool MTL4::RenderPipelineDescriptor::isRasterizationEnabled()
{
    return _MTL4_msg_bool_isRasterizationEnabled((const void*)this, nullptr);
}
