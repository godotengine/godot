#pragma once

#include "MTL4Defines.hpp"
#include "MTL4Blocks.hpp"
#include "MTL4Structs.hpp"
#include "MTL4Bridge.hpp"
#include "../Foundation/NSObject.hpp"
#include "../Foundation/NSTypes.hpp"
#include "../Foundation/NSRange.hpp"
#include "MTL4PipelineState.hpp"
#include "MTLStructs.hpp"

namespace MTL {
    class TileRenderPipelineColorAttachmentDescriptorArray;
}
namespace MTL4 {
    class FunctionDescriptor;
    class StaticLinkingDescriptor;
}

namespace MTL4
{

class TileRenderPipelineDescriptor : public NS::Referencing<TileRenderPipelineDescriptor, MTL4::PipelineDescriptor>
{
public:
    static TileRenderPipelineDescriptor* alloc();
    TileRenderPipelineDescriptor*        init() const;

    MTL::TileRenderPipelineColorAttachmentDescriptorArray* colorAttachments() const;
    NS::UInteger                                           maxTotalThreadsPerThreadgroup() const;
    NS::UInteger                                           rasterSampleCount() const;
    MTL::Size                                              requiredThreadsPerThreadgroup() const;
    void                                                   reset();
    void                                                   setMaxTotalThreadsPerThreadgroup(NS::UInteger maxTotalThreadsPerThreadgroup);
    void                                                   setRasterSampleCount(NS::UInteger rasterSampleCount);
    void                                                   setRequiredThreadsPerThreadgroup(MTL::Size requiredThreadsPerThreadgroup);
    void                                                   setStaticLinkingDescriptor(MTL4::StaticLinkingDescriptor* staticLinkingDescriptor);
    void                                                   setSupportBinaryLinking(bool supportBinaryLinking);
    void                                                   setThreadgroupSizeMatchesTileSize(bool threadgroupSizeMatchesTileSize);
    void                                                   setTileFunctionDescriptor(MTL4::FunctionDescriptor* tileFunctionDescriptor);
    MTL4::StaticLinkingDescriptor*                         staticLinkingDescriptor() const;
    bool                                                   supportBinaryLinking() const;
    bool                                                   threadgroupSizeMatchesTileSize() const;
    MTL4::FunctionDescriptor*                              tileFunctionDescriptor() const;

};

} // namespace MTL4

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTL4TileRenderPipelineDescriptor;

_MTL4_INLINE MTL4::TileRenderPipelineDescriptor* MTL4::TileRenderPipelineDescriptor::alloc()
{
    return _MTL4_msg_MTL4__TileRenderPipelineDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTL4TileRenderPipelineDescriptor, nullptr);
}

_MTL4_INLINE MTL4::TileRenderPipelineDescriptor* MTL4::TileRenderPipelineDescriptor::init() const
{
    return _MTL4_msg_MTL4__TileRenderPipelineDescriptorp_init((const void*)this, nullptr);
}

_MTL4_INLINE MTL4::FunctionDescriptor* MTL4::TileRenderPipelineDescriptor::tileFunctionDescriptor() const
{
    return _MTL4_msg_MTL4__FunctionDescriptorp_tileFunctionDescriptor((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::TileRenderPipelineDescriptor::setTileFunctionDescriptor(MTL4::FunctionDescriptor* tileFunctionDescriptor)
{
    _MTL4_msg_v_setTileFunctionDescriptor__MTL4__FunctionDescriptorp((const void*)this, nullptr, tileFunctionDescriptor);
}

_MTL4_INLINE NS::UInteger MTL4::TileRenderPipelineDescriptor::rasterSampleCount() const
{
    return _MTL4_msg_NS__UInteger_rasterSampleCount((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::TileRenderPipelineDescriptor::setRasterSampleCount(NS::UInteger rasterSampleCount)
{
    _MTL4_msg_v_setRasterSampleCount__NS__UInteger((const void*)this, nullptr, rasterSampleCount);
}

_MTL4_INLINE MTL::TileRenderPipelineColorAttachmentDescriptorArray* MTL4::TileRenderPipelineDescriptor::colorAttachments() const
{
    return _MTL4_msg_MTL__TileRenderPipelineColorAttachmentDescriptorArrayp_colorAttachments((const void*)this, nullptr);
}

_MTL4_INLINE bool MTL4::TileRenderPipelineDescriptor::threadgroupSizeMatchesTileSize() const
{
    return _MTL4_msg_bool_threadgroupSizeMatchesTileSize((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::TileRenderPipelineDescriptor::setThreadgroupSizeMatchesTileSize(bool threadgroupSizeMatchesTileSize)
{
    _MTL4_msg_v_setThreadgroupSizeMatchesTileSize__bool((const void*)this, nullptr, threadgroupSizeMatchesTileSize);
}

_MTL4_INLINE NS::UInteger MTL4::TileRenderPipelineDescriptor::maxTotalThreadsPerThreadgroup() const
{
    return _MTL4_msg_NS__UInteger_maxTotalThreadsPerThreadgroup((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::TileRenderPipelineDescriptor::setMaxTotalThreadsPerThreadgroup(NS::UInteger maxTotalThreadsPerThreadgroup)
{
    _MTL4_msg_v_setMaxTotalThreadsPerThreadgroup__NS__UInteger((const void*)this, nullptr, maxTotalThreadsPerThreadgroup);
}

_MTL4_INLINE MTL::Size MTL4::TileRenderPipelineDescriptor::requiredThreadsPerThreadgroup() const
{
    return _MTL4_msg_MTL__Size_requiredThreadsPerThreadgroup((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::TileRenderPipelineDescriptor::setRequiredThreadsPerThreadgroup(MTL::Size requiredThreadsPerThreadgroup)
{
    _MTL4_msg_v_setRequiredThreadsPerThreadgroup__MTL__Size((const void*)this, nullptr, requiredThreadsPerThreadgroup);
}

_MTL4_INLINE MTL4::StaticLinkingDescriptor* MTL4::TileRenderPipelineDescriptor::staticLinkingDescriptor() const
{
    return _MTL4_msg_MTL4__StaticLinkingDescriptorp_staticLinkingDescriptor((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::TileRenderPipelineDescriptor::setStaticLinkingDescriptor(MTL4::StaticLinkingDescriptor* staticLinkingDescriptor)
{
    _MTL4_msg_v_setStaticLinkingDescriptor__MTL4__StaticLinkingDescriptorp((const void*)this, nullptr, staticLinkingDescriptor);
}

_MTL4_INLINE bool MTL4::TileRenderPipelineDescriptor::supportBinaryLinking() const
{
    return _MTL4_msg_bool_supportBinaryLinking((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::TileRenderPipelineDescriptor::setSupportBinaryLinking(bool supportBinaryLinking)
{
    _MTL4_msg_v_setSupportBinaryLinking__bool((const void*)this, nullptr, supportBinaryLinking);
}

_MTL4_INLINE void MTL4::TileRenderPipelineDescriptor::reset()
{
    _MTL4_msg_v_reset((const void*)this, nullptr);
}
