#pragma once

#include "MTL4Defines.hpp"
#include "MTL4Blocks.hpp"
#include "MTL4Structs.hpp"
#include "MTL4Bridge.hpp"
#include "../Foundation/NSObject.hpp"
#include "../Foundation/NSTypes.hpp"
#include "../Foundation/NSRange.hpp"
#include "MTLStructs.hpp"

namespace MTL {
    class Buffer;
    class RasterizationRateMap;
    class RenderPassColorAttachmentDescriptorArray;
    class RenderPassDepthAttachmentDescriptor;
    class RenderPassStencilAttachmentDescriptor;
    enum VisibilityResultType : NS::Integer;
}

namespace MTL4
{

class RenderPassDescriptor : public NS::Copying<RenderPassDescriptor>
{
public:
    static RenderPassDescriptor* alloc();
    RenderPassDescriptor*        init() const;

    MTL::RenderPassColorAttachmentDescriptorArray* colorAttachments() const;
    NS::UInteger                                   defaultRasterSampleCount() const;
    MTL::RenderPassDepthAttachmentDescriptor*      depthAttachment() const;
    NS::UInteger                                   getSamplePositions(MTL::SamplePosition* positions, NS::UInteger count);
    NS::UInteger                                   imageblockSampleLength() const;
    MTL::RasterizationRateMap*                     rasterizationRateMap() const;
    NS::UInteger                                   renderTargetArrayLength() const;
    NS::UInteger                                   renderTargetHeight() const;
    NS::UInteger                                   renderTargetWidth() const;
    void                                           setDefaultRasterSampleCount(NS::UInteger defaultRasterSampleCount);
    void                                           setDepthAttachment(MTL::RenderPassDepthAttachmentDescriptor* depthAttachment);
    void                                           setImageblockSampleLength(NS::UInteger imageblockSampleLength);
    void                                           setRasterizationRateMap(MTL::RasterizationRateMap* rasterizationRateMap);
    void                                           setRenderTargetArrayLength(NS::UInteger renderTargetArrayLength);
    void                                           setRenderTargetHeight(NS::UInteger renderTargetHeight);
    void                                           setRenderTargetWidth(NS::UInteger renderTargetWidth);
    void                                           setSamplePositions(const MTL::SamplePosition * positions, NS::UInteger count);
    void                                           setStencilAttachment(MTL::RenderPassStencilAttachmentDescriptor* stencilAttachment);
    void                                           setSupportColorAttachmentMapping(bool supportColorAttachmentMapping);
    void                                           setThreadgroupMemoryLength(NS::UInteger threadgroupMemoryLength);
    void                                           setTileHeight(NS::UInteger tileHeight);
    void                                           setTileWidth(NS::UInteger tileWidth);
    void                                           setVisibilityResultBuffer(MTL::Buffer* visibilityResultBuffer);
    void                                           setVisibilityResultType(MTL::VisibilityResultType visibilityResultType);
    MTL::RenderPassStencilAttachmentDescriptor*    stencilAttachment() const;
    bool                                           supportColorAttachmentMapping() const;
    NS::UInteger                                   threadgroupMemoryLength() const;
    NS::UInteger                                   tileHeight() const;
    NS::UInteger                                   tileWidth() const;
    MTL::Buffer*                                   visibilityResultBuffer() const;
    MTL::VisibilityResultType                      visibilityResultType() const;

};

} // namespace MTL4

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTL4RenderPassDescriptor;

_MTL4_INLINE MTL4::RenderPassDescriptor* MTL4::RenderPassDescriptor::alloc()
{
    return _MTL4_msg_MTL4__RenderPassDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTL4RenderPassDescriptor, nullptr);
}

_MTL4_INLINE MTL4::RenderPassDescriptor* MTL4::RenderPassDescriptor::init() const
{
    return _MTL4_msg_MTL4__RenderPassDescriptorp_init((const void*)this, nullptr);
}

_MTL4_INLINE MTL::RenderPassColorAttachmentDescriptorArray* MTL4::RenderPassDescriptor::colorAttachments() const
{
    return _MTL4_msg_MTL__RenderPassColorAttachmentDescriptorArrayp_colorAttachments((const void*)this, nullptr);
}

_MTL4_INLINE MTL::RenderPassDepthAttachmentDescriptor* MTL4::RenderPassDescriptor::depthAttachment() const
{
    return _MTL4_msg_MTL__RenderPassDepthAttachmentDescriptorp_depthAttachment((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::RenderPassDescriptor::setDepthAttachment(MTL::RenderPassDepthAttachmentDescriptor* depthAttachment)
{
    _MTL4_msg_v_setDepthAttachment__MTL__RenderPassDepthAttachmentDescriptorp((const void*)this, nullptr, depthAttachment);
}

_MTL4_INLINE MTL::RenderPassStencilAttachmentDescriptor* MTL4::RenderPassDescriptor::stencilAttachment() const
{
    return _MTL4_msg_MTL__RenderPassStencilAttachmentDescriptorp_stencilAttachment((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::RenderPassDescriptor::setStencilAttachment(MTL::RenderPassStencilAttachmentDescriptor* stencilAttachment)
{
    _MTL4_msg_v_setStencilAttachment__MTL__RenderPassStencilAttachmentDescriptorp((const void*)this, nullptr, stencilAttachment);
}

_MTL4_INLINE NS::UInteger MTL4::RenderPassDescriptor::renderTargetArrayLength() const
{
    return _MTL4_msg_NS__UInteger_renderTargetArrayLength((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::RenderPassDescriptor::setRenderTargetArrayLength(NS::UInteger renderTargetArrayLength)
{
    _MTL4_msg_v_setRenderTargetArrayLength__NS__UInteger((const void*)this, nullptr, renderTargetArrayLength);
}

_MTL4_INLINE NS::UInteger MTL4::RenderPassDescriptor::imageblockSampleLength() const
{
    return _MTL4_msg_NS__UInteger_imageblockSampleLength((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::RenderPassDescriptor::setImageblockSampleLength(NS::UInteger imageblockSampleLength)
{
    _MTL4_msg_v_setImageblockSampleLength__NS__UInteger((const void*)this, nullptr, imageblockSampleLength);
}

_MTL4_INLINE NS::UInteger MTL4::RenderPassDescriptor::threadgroupMemoryLength() const
{
    return _MTL4_msg_NS__UInteger_threadgroupMemoryLength((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::RenderPassDescriptor::setThreadgroupMemoryLength(NS::UInteger threadgroupMemoryLength)
{
    _MTL4_msg_v_setThreadgroupMemoryLength__NS__UInteger((const void*)this, nullptr, threadgroupMemoryLength);
}

_MTL4_INLINE NS::UInteger MTL4::RenderPassDescriptor::tileWidth() const
{
    return _MTL4_msg_NS__UInteger_tileWidth((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::RenderPassDescriptor::setTileWidth(NS::UInteger tileWidth)
{
    _MTL4_msg_v_setTileWidth__NS__UInteger((const void*)this, nullptr, tileWidth);
}

_MTL4_INLINE NS::UInteger MTL4::RenderPassDescriptor::tileHeight() const
{
    return _MTL4_msg_NS__UInteger_tileHeight((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::RenderPassDescriptor::setTileHeight(NS::UInteger tileHeight)
{
    _MTL4_msg_v_setTileHeight__NS__UInteger((const void*)this, nullptr, tileHeight);
}

_MTL4_INLINE NS::UInteger MTL4::RenderPassDescriptor::defaultRasterSampleCount() const
{
    return _MTL4_msg_NS__UInteger_defaultRasterSampleCount((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::RenderPassDescriptor::setDefaultRasterSampleCount(NS::UInteger defaultRasterSampleCount)
{
    _MTL4_msg_v_setDefaultRasterSampleCount__NS__UInteger((const void*)this, nullptr, defaultRasterSampleCount);
}

_MTL4_INLINE NS::UInteger MTL4::RenderPassDescriptor::renderTargetWidth() const
{
    return _MTL4_msg_NS__UInteger_renderTargetWidth((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::RenderPassDescriptor::setRenderTargetWidth(NS::UInteger renderTargetWidth)
{
    _MTL4_msg_v_setRenderTargetWidth__NS__UInteger((const void*)this, nullptr, renderTargetWidth);
}

_MTL4_INLINE NS::UInteger MTL4::RenderPassDescriptor::renderTargetHeight() const
{
    return _MTL4_msg_NS__UInteger_renderTargetHeight((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::RenderPassDescriptor::setRenderTargetHeight(NS::UInteger renderTargetHeight)
{
    _MTL4_msg_v_setRenderTargetHeight__NS__UInteger((const void*)this, nullptr, renderTargetHeight);
}

_MTL4_INLINE MTL::RasterizationRateMap* MTL4::RenderPassDescriptor::rasterizationRateMap() const
{
    return _MTL4_msg_MTL__RasterizationRateMapp_rasterizationRateMap((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::RenderPassDescriptor::setRasterizationRateMap(MTL::RasterizationRateMap* rasterizationRateMap)
{
    _MTL4_msg_v_setRasterizationRateMap__MTL__RasterizationRateMapp((const void*)this, nullptr, rasterizationRateMap);
}

_MTL4_INLINE MTL::Buffer* MTL4::RenderPassDescriptor::visibilityResultBuffer() const
{
    return _MTL4_msg_MTL__Bufferp_visibilityResultBuffer((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::RenderPassDescriptor::setVisibilityResultBuffer(MTL::Buffer* visibilityResultBuffer)
{
    _MTL4_msg_v_setVisibilityResultBuffer__MTL__Bufferp((const void*)this, nullptr, visibilityResultBuffer);
}

_MTL4_INLINE MTL::VisibilityResultType MTL4::RenderPassDescriptor::visibilityResultType() const
{
    return _MTL4_msg_MTL__VisibilityResultType_visibilityResultType((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::RenderPassDescriptor::setVisibilityResultType(MTL::VisibilityResultType visibilityResultType)
{
    _MTL4_msg_v_setVisibilityResultType__MTL__VisibilityResultType((const void*)this, nullptr, visibilityResultType);
}

_MTL4_INLINE bool MTL4::RenderPassDescriptor::supportColorAttachmentMapping() const
{
    return _MTL4_msg_bool_supportColorAttachmentMapping((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::RenderPassDescriptor::setSupportColorAttachmentMapping(bool supportColorAttachmentMapping)
{
    _MTL4_msg_v_setSupportColorAttachmentMapping__bool((const void*)this, nullptr, supportColorAttachmentMapping);
}

_MTL4_INLINE void MTL4::RenderPassDescriptor::setSamplePositions(const MTL::SamplePosition * positions, NS::UInteger count)
{
    _MTL4_msg_v_setSamplePositions_count__constMTL__SamplePositionp_NS__UInteger((const void*)this, nullptr, positions, count);
}

_MTL4_INLINE NS::UInteger MTL4::RenderPassDescriptor::getSamplePositions(MTL::SamplePosition* positions, NS::UInteger count)
{
    return _MTL4_msg_NS__UInteger_getSamplePositions_count__MTL__SamplePositionp_NS__UInteger((const void*)this, nullptr, positions, count);
}
