#pragma once

#include "MTLDefines.hpp"
#include "MTLBlocks.hpp"
#include "MTLStructs.hpp"
#include "MTLBridge.hpp"
#include "../Foundation/NSObject.hpp"
#include "../Foundation/NSTypes.hpp"
#include "../Foundation/NSRange.hpp"

namespace MTL {
    class Buffer;
    class CounterSampleBuffer;
    class RasterizationRateMap;
    class Texture;
}

namespace MTL
{

_MTL_ENUM(NS::UInteger, LoadAction) {
    LoadActionDontCare = 0,
    LoadActionLoad = 1,
    LoadActionClear = 2,
};

_MTL_ENUM(NS::UInteger, StoreAction) {
    StoreActionDontCare = 0,
    StoreActionStore = 1,
    StoreActionMultisampleResolve = 2,
    StoreActionStoreAndMultisampleResolve = 3,
    StoreActionUnknown = 4,
    StoreActionCustomSampleDepthStore = 5,
};

_MTL_OPTIONS(NS::UInteger, StoreActionOptions) {
    StoreActionOptionNone = 0,
    StoreActionOptionCustomSamplePositions = 1 << 0,
};

_MTL_ENUM(NS::Integer, VisibilityResultType) {
    VisibilityResultTypeReset = 0,
    VisibilityResultTypeAccumulate = 1,
};

_MTL_ENUM(NS::UInteger, MultisampleDepthResolveFilter) {
    MultisampleDepthResolveFilterSample0 = 0,
    MultisampleDepthResolveFilterMin = 1,
    MultisampleDepthResolveFilterMax = 2,
};

_MTL_ENUM(NS::UInteger, MultisampleStencilResolveFilter) {
    MultisampleStencilResolveFilterSample0 = 0,
    MultisampleStencilResolveFilterDepthResolvedSample = 1,
};


class RenderPassAttachmentDescriptor;
class RenderPassColorAttachmentDescriptor;
class RenderPassDepthAttachmentDescriptor;
class RenderPassStencilAttachmentDescriptor;
class RenderPassColorAttachmentDescriptorArray;
class RenderPassSampleBufferAttachmentDescriptor;
class RenderPassSampleBufferAttachmentDescriptorArray;
class RenderPassDescriptor;

class RenderPassAttachmentDescriptor : public NS::Copying<RenderPassAttachmentDescriptor>
{
public:
    static RenderPassAttachmentDescriptor* alloc();
    RenderPassAttachmentDescriptor*        init() const;

    NS::UInteger            depthPlane() const;
    NS::UInteger            level() const;
    MTL::LoadAction         loadAction() const;
    NS::UInteger            resolveDepthPlane() const;
    NS::UInteger            resolveLevel() const;
    NS::UInteger            resolveSlice() const;
    MTL::Texture*           resolveTexture() const;
    void                    setDepthPlane(NS::UInteger depthPlane);
    void                    setLevel(NS::UInteger level);
    void                    setLoadAction(MTL::LoadAction loadAction);
    void                    setResolveDepthPlane(NS::UInteger resolveDepthPlane);
    void                    setResolveLevel(NS::UInteger resolveLevel);
    void                    setResolveSlice(NS::UInteger resolveSlice);
    void                    setResolveTexture(MTL::Texture* resolveTexture);
    void                    setSlice(NS::UInteger slice);
    void                    setStoreAction(MTL::StoreAction storeAction);
    void                    setStoreActionOptions(MTL::StoreActionOptions storeActionOptions);
    void                    setTexture(MTL::Texture* texture);
    NS::UInteger            slice() const;
    MTL::StoreAction        storeAction() const;
    MTL::StoreActionOptions storeActionOptions() const;
    MTL::Texture*           texture() const;

};

class RenderPassColorAttachmentDescriptor : public NS::Referencing<RenderPassColorAttachmentDescriptor, MTL::RenderPassAttachmentDescriptor>
{
public:
    static RenderPassColorAttachmentDescriptor* alloc();
    RenderPassColorAttachmentDescriptor*        init() const;

    MTL::ClearColor clearColor() const;
    void            setClearColor(MTL::ClearColor clearColor);

};

class RenderPassDepthAttachmentDescriptor : public NS::Referencing<RenderPassDepthAttachmentDescriptor, MTL::RenderPassAttachmentDescriptor>
{
public:
    static RenderPassDepthAttachmentDescriptor* alloc();
    RenderPassDepthAttachmentDescriptor*        init() const;

    double                             clearDepth() const;
    MTL::MultisampleDepthResolveFilter depthResolveFilter() const;
    void                               setClearDepth(double clearDepth);
    void                               setDepthResolveFilter(MTL::MultisampleDepthResolveFilter depthResolveFilter);

};

class RenderPassStencilAttachmentDescriptor : public NS::Referencing<RenderPassStencilAttachmentDescriptor, MTL::RenderPassAttachmentDescriptor>
{
public:
    static RenderPassStencilAttachmentDescriptor* alloc();
    RenderPassStencilAttachmentDescriptor*        init() const;

    uint32_t                             clearStencil() const;
    void                                 setClearStencil(uint32_t clearStencil);
    void                                 setStencilResolveFilter(MTL::MultisampleStencilResolveFilter stencilResolveFilter);
    MTL::MultisampleStencilResolveFilter stencilResolveFilter() const;

};

class RenderPassColorAttachmentDescriptorArray : public NS::Referencing<RenderPassColorAttachmentDescriptorArray>
{
public:
    static RenderPassColorAttachmentDescriptorArray* alloc();
    RenderPassColorAttachmentDescriptorArray*        init() const;

    MTL::RenderPassColorAttachmentDescriptor* object(NS::UInteger attachmentIndex);
    void                                      setObject(MTL::RenderPassColorAttachmentDescriptor* attachment, NS::UInteger attachmentIndex);

};

class RenderPassSampleBufferAttachmentDescriptor : public NS::Copying<RenderPassSampleBufferAttachmentDescriptor>
{
public:
    static RenderPassSampleBufferAttachmentDescriptor* alloc();
    RenderPassSampleBufferAttachmentDescriptor*        init() const;

    NS::UInteger              endOfFragmentSampleIndex() const;
    NS::UInteger              endOfVertexSampleIndex() const;
    MTL::CounterSampleBuffer* sampleBuffer() const;
    void                      setEndOfFragmentSampleIndex(NS::UInteger endOfFragmentSampleIndex);
    void                      setEndOfVertexSampleIndex(NS::UInteger endOfVertexSampleIndex);
    void                      setSampleBuffer(MTL::CounterSampleBuffer* sampleBuffer);
    void                      setStartOfFragmentSampleIndex(NS::UInteger startOfFragmentSampleIndex);
    void                      setStartOfVertexSampleIndex(NS::UInteger startOfVertexSampleIndex);
    NS::UInteger              startOfFragmentSampleIndex() const;
    NS::UInteger              startOfVertexSampleIndex() const;

};

class RenderPassSampleBufferAttachmentDescriptorArray : public NS::Referencing<RenderPassSampleBufferAttachmentDescriptorArray>
{
public:
    static RenderPassSampleBufferAttachmentDescriptorArray* alloc();
    RenderPassSampleBufferAttachmentDescriptorArray*        init() const;

    MTL::RenderPassSampleBufferAttachmentDescriptor* object(NS::UInteger attachmentIndex);
    void                                             setObject(MTL::RenderPassSampleBufferAttachmentDescriptor* attachment, NS::UInteger attachmentIndex);

};

class RenderPassDescriptor : public NS::Copying<RenderPassDescriptor>
{
public:
    static RenderPassDescriptor* alloc();
    RenderPassDescriptor*        init() const;

    static MTL::RenderPassDescriptor* renderPassDescriptor();

    MTL::RenderPassColorAttachmentDescriptorArray*        colorAttachments() const;
    NS::UInteger                                          defaultRasterSampleCount() const;
    MTL::RenderPassDepthAttachmentDescriptor*             depthAttachment() const;
    NS::UInteger                                          getSamplePositions(MTL::SamplePosition* positions, NS::UInteger count);
    NS::UInteger                                          imageblockSampleLength() const;
    MTL::RasterizationRateMap*                            rasterizationRateMap() const;
    NS::UInteger                                          renderTargetArrayLength() const;
    NS::UInteger                                          renderTargetHeight() const;
    NS::UInteger                                          renderTargetWidth() const;
    MTL::RenderPassSampleBufferAttachmentDescriptorArray* sampleBufferAttachments() const;
    void                                                  setDefaultRasterSampleCount(NS::UInteger defaultRasterSampleCount);
    void                                                  setDepthAttachment(MTL::RenderPassDepthAttachmentDescriptor* depthAttachment);
    void                                                  setImageblockSampleLength(NS::UInteger imageblockSampleLength);
    void                                                  setRasterizationRateMap(MTL::RasterizationRateMap* rasterizationRateMap);
    void                                                  setRenderTargetArrayLength(NS::UInteger renderTargetArrayLength);
    void                                                  setRenderTargetHeight(NS::UInteger renderTargetHeight);
    void                                                  setRenderTargetWidth(NS::UInteger renderTargetWidth);
    void                                                  setSamplePositions(const MTL::SamplePosition * positions, NS::UInteger count);
    void                                                  setStencilAttachment(MTL::RenderPassStencilAttachmentDescriptor* stencilAttachment);
    void                                                  setSupportColorAttachmentMapping(bool supportColorAttachmentMapping);
    void                                                  setThreadgroupMemoryLength(NS::UInteger threadgroupMemoryLength);
    void                                                  setTileHeight(NS::UInteger tileHeight);
    void                                                  setTileWidth(NS::UInteger tileWidth);
    void                                                  setVisibilityResultBuffer(MTL::Buffer* visibilityResultBuffer);
    void                                                  setVisibilityResultType(MTL::VisibilityResultType visibilityResultType);
    MTL::RenderPassStencilAttachmentDescriptor*           stencilAttachment() const;
    bool                                                  supportColorAttachmentMapping() const;
    NS::UInteger                                          threadgroupMemoryLength() const;
    NS::UInteger                                          tileHeight() const;
    NS::UInteger                                          tileWidth() const;
    MTL::Buffer*                                          visibilityResultBuffer() const;
    MTL::VisibilityResultType                             visibilityResultType() const;

};

} // namespace MTL

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTLRenderPassAttachmentDescriptor;
extern "C" void *OBJC_CLASS_$_MTLRenderPassColorAttachmentDescriptor;
extern "C" void *OBJC_CLASS_$_MTLRenderPassDepthAttachmentDescriptor;
extern "C" void *OBJC_CLASS_$_MTLRenderPassStencilAttachmentDescriptor;
extern "C" void *OBJC_CLASS_$_MTLRenderPassColorAttachmentDescriptorArray;
extern "C" void *OBJC_CLASS_$_MTLRenderPassSampleBufferAttachmentDescriptor;
extern "C" void *OBJC_CLASS_$_MTLRenderPassSampleBufferAttachmentDescriptorArray;
extern "C" void *OBJC_CLASS_$_MTLRenderPassDescriptor;

_MTL_INLINE MTL::RenderPassAttachmentDescriptor* MTL::RenderPassAttachmentDescriptor::alloc()
{
    return _MTL_msg_MTL__RenderPassAttachmentDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTLRenderPassAttachmentDescriptor, nullptr);
}

_MTL_INLINE MTL::RenderPassAttachmentDescriptor* MTL::RenderPassAttachmentDescriptor::init() const
{
    return _MTL_msg_MTL__RenderPassAttachmentDescriptorp_init((const void*)this, nullptr);
}

_MTL_INLINE MTL::Texture* MTL::RenderPassAttachmentDescriptor::texture() const
{
    return _MTL_msg_MTL__Texturep_texture((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RenderPassAttachmentDescriptor::setTexture(MTL::Texture* texture)
{
    _MTL_msg_v_setTexture__MTL__Texturep((const void*)this, nullptr, texture);
}

_MTL_INLINE NS::UInteger MTL::RenderPassAttachmentDescriptor::level() const
{
    return _MTL_msg_NS__UInteger_level((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RenderPassAttachmentDescriptor::setLevel(NS::UInteger level)
{
    _MTL_msg_v_setLevel__NS__UInteger((const void*)this, nullptr, level);
}

_MTL_INLINE NS::UInteger MTL::RenderPassAttachmentDescriptor::slice() const
{
    return _MTL_msg_NS__UInteger_slice((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RenderPassAttachmentDescriptor::setSlice(NS::UInteger slice)
{
    _MTL_msg_v_setSlice__NS__UInteger((const void*)this, nullptr, slice);
}

_MTL_INLINE NS::UInteger MTL::RenderPassAttachmentDescriptor::depthPlane() const
{
    return _MTL_msg_NS__UInteger_depthPlane((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RenderPassAttachmentDescriptor::setDepthPlane(NS::UInteger depthPlane)
{
    _MTL_msg_v_setDepthPlane__NS__UInteger((const void*)this, nullptr, depthPlane);
}

_MTL_INLINE MTL::Texture* MTL::RenderPassAttachmentDescriptor::resolveTexture() const
{
    return _MTL_msg_MTL__Texturep_resolveTexture((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RenderPassAttachmentDescriptor::setResolveTexture(MTL::Texture* resolveTexture)
{
    _MTL_msg_v_setResolveTexture__MTL__Texturep((const void*)this, nullptr, resolveTexture);
}

_MTL_INLINE NS::UInteger MTL::RenderPassAttachmentDescriptor::resolveLevel() const
{
    return _MTL_msg_NS__UInteger_resolveLevel((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RenderPassAttachmentDescriptor::setResolveLevel(NS::UInteger resolveLevel)
{
    _MTL_msg_v_setResolveLevel__NS__UInteger((const void*)this, nullptr, resolveLevel);
}

_MTL_INLINE NS::UInteger MTL::RenderPassAttachmentDescriptor::resolveSlice() const
{
    return _MTL_msg_NS__UInteger_resolveSlice((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RenderPassAttachmentDescriptor::setResolveSlice(NS::UInteger resolveSlice)
{
    _MTL_msg_v_setResolveSlice__NS__UInteger((const void*)this, nullptr, resolveSlice);
}

_MTL_INLINE NS::UInteger MTL::RenderPassAttachmentDescriptor::resolveDepthPlane() const
{
    return _MTL_msg_NS__UInteger_resolveDepthPlane((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RenderPassAttachmentDescriptor::setResolveDepthPlane(NS::UInteger resolveDepthPlane)
{
    _MTL_msg_v_setResolveDepthPlane__NS__UInteger((const void*)this, nullptr, resolveDepthPlane);
}

_MTL_INLINE MTL::LoadAction MTL::RenderPassAttachmentDescriptor::loadAction() const
{
    return _MTL_msg_MTL__LoadAction_loadAction((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RenderPassAttachmentDescriptor::setLoadAction(MTL::LoadAction loadAction)
{
    _MTL_msg_v_setLoadAction__MTL__LoadAction((const void*)this, nullptr, loadAction);
}

_MTL_INLINE MTL::StoreAction MTL::RenderPassAttachmentDescriptor::storeAction() const
{
    return _MTL_msg_MTL__StoreAction_storeAction((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RenderPassAttachmentDescriptor::setStoreAction(MTL::StoreAction storeAction)
{
    _MTL_msg_v_setStoreAction__MTL__StoreAction((const void*)this, nullptr, storeAction);
}

_MTL_INLINE MTL::StoreActionOptions MTL::RenderPassAttachmentDescriptor::storeActionOptions() const
{
    return _MTL_msg_MTL__StoreActionOptions_storeActionOptions((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RenderPassAttachmentDescriptor::setStoreActionOptions(MTL::StoreActionOptions storeActionOptions)
{
    _MTL_msg_v_setStoreActionOptions__MTL__StoreActionOptions((const void*)this, nullptr, storeActionOptions);
}

_MTL_INLINE MTL::RenderPassColorAttachmentDescriptor* MTL::RenderPassColorAttachmentDescriptor::alloc()
{
    return _MTL_msg_MTL__RenderPassColorAttachmentDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTLRenderPassColorAttachmentDescriptor, nullptr);
}

_MTL_INLINE MTL::RenderPassColorAttachmentDescriptor* MTL::RenderPassColorAttachmentDescriptor::init() const
{
    return _MTL_msg_MTL__RenderPassColorAttachmentDescriptorp_init((const void*)this, nullptr);
}

_MTL_INLINE MTL::ClearColor MTL::RenderPassColorAttachmentDescriptor::clearColor() const
{
    return _MTL_msg_MTL__ClearColor_clearColor((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RenderPassColorAttachmentDescriptor::setClearColor(MTL::ClearColor clearColor)
{
    _MTL_msg_v_setClearColor__MTL__ClearColor((const void*)this, nullptr, clearColor);
}

_MTL_INLINE MTL::RenderPassDepthAttachmentDescriptor* MTL::RenderPassDepthAttachmentDescriptor::alloc()
{
    return _MTL_msg_MTL__RenderPassDepthAttachmentDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTLRenderPassDepthAttachmentDescriptor, nullptr);
}

_MTL_INLINE MTL::RenderPassDepthAttachmentDescriptor* MTL::RenderPassDepthAttachmentDescriptor::init() const
{
    return _MTL_msg_MTL__RenderPassDepthAttachmentDescriptorp_init((const void*)this, nullptr);
}

_MTL_INLINE double MTL::RenderPassDepthAttachmentDescriptor::clearDepth() const
{
    return _MTL_msg_double_clearDepth((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RenderPassDepthAttachmentDescriptor::setClearDepth(double clearDepth)
{
    _MTL_msg_v_setClearDepth__double((const void*)this, nullptr, clearDepth);
}

_MTL_INLINE MTL::MultisampleDepthResolveFilter MTL::RenderPassDepthAttachmentDescriptor::depthResolveFilter() const
{
    return _MTL_msg_MTL__MultisampleDepthResolveFilter_depthResolveFilter((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RenderPassDepthAttachmentDescriptor::setDepthResolveFilter(MTL::MultisampleDepthResolveFilter depthResolveFilter)
{
    _MTL_msg_v_setDepthResolveFilter__MTL__MultisampleDepthResolveFilter((const void*)this, nullptr, depthResolveFilter);
}

_MTL_INLINE MTL::RenderPassStencilAttachmentDescriptor* MTL::RenderPassStencilAttachmentDescriptor::alloc()
{
    return _MTL_msg_MTL__RenderPassStencilAttachmentDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTLRenderPassStencilAttachmentDescriptor, nullptr);
}

_MTL_INLINE MTL::RenderPassStencilAttachmentDescriptor* MTL::RenderPassStencilAttachmentDescriptor::init() const
{
    return _MTL_msg_MTL__RenderPassStencilAttachmentDescriptorp_init((const void*)this, nullptr);
}

_MTL_INLINE uint32_t MTL::RenderPassStencilAttachmentDescriptor::clearStencil() const
{
    return _MTL_msg_uint32_t_clearStencil((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RenderPassStencilAttachmentDescriptor::setClearStencil(uint32_t clearStencil)
{
    _MTL_msg_v_setClearStencil__uint32_t((const void*)this, nullptr, clearStencil);
}

_MTL_INLINE MTL::MultisampleStencilResolveFilter MTL::RenderPassStencilAttachmentDescriptor::stencilResolveFilter() const
{
    return _MTL_msg_MTL__MultisampleStencilResolveFilter_stencilResolveFilter((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RenderPassStencilAttachmentDescriptor::setStencilResolveFilter(MTL::MultisampleStencilResolveFilter stencilResolveFilter)
{
    _MTL_msg_v_setStencilResolveFilter__MTL__MultisampleStencilResolveFilter((const void*)this, nullptr, stencilResolveFilter);
}

_MTL_INLINE MTL::RenderPassColorAttachmentDescriptorArray* MTL::RenderPassColorAttachmentDescriptorArray::alloc()
{
    return _MTL_msg_MTL__RenderPassColorAttachmentDescriptorArrayp_alloc((const void*)&OBJC_CLASS_$_MTLRenderPassColorAttachmentDescriptorArray, nullptr);
}

_MTL_INLINE MTL::RenderPassColorAttachmentDescriptorArray* MTL::RenderPassColorAttachmentDescriptorArray::init() const
{
    return _MTL_msg_MTL__RenderPassColorAttachmentDescriptorArrayp_init((const void*)this, nullptr);
}

_MTL_INLINE MTL::RenderPassColorAttachmentDescriptor* MTL::RenderPassColorAttachmentDescriptorArray::object(NS::UInteger attachmentIndex)
{
    return _MTL_msg_MTL__RenderPassColorAttachmentDescriptorp_objectAtIndexedSubscript__NS__UInteger((const void*)this, nullptr, attachmentIndex);
}

_MTL_INLINE void MTL::RenderPassColorAttachmentDescriptorArray::setObject(MTL::RenderPassColorAttachmentDescriptor* attachment, NS::UInteger attachmentIndex)
{
    _MTL_msg_v_setObject_atIndexedSubscript__MTL__RenderPassColorAttachmentDescriptorp_NS__UInteger((const void*)this, nullptr, attachment, attachmentIndex);
}

_MTL_INLINE MTL::RenderPassSampleBufferAttachmentDescriptor* MTL::RenderPassSampleBufferAttachmentDescriptor::alloc()
{
    return _MTL_msg_MTL__RenderPassSampleBufferAttachmentDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTLRenderPassSampleBufferAttachmentDescriptor, nullptr);
}

_MTL_INLINE MTL::RenderPassSampleBufferAttachmentDescriptor* MTL::RenderPassSampleBufferAttachmentDescriptor::init() const
{
    return _MTL_msg_MTL__RenderPassSampleBufferAttachmentDescriptorp_init((const void*)this, nullptr);
}

_MTL_INLINE MTL::CounterSampleBuffer* MTL::RenderPassSampleBufferAttachmentDescriptor::sampleBuffer() const
{
    return _MTL_msg_MTL__CounterSampleBufferp_sampleBuffer((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RenderPassSampleBufferAttachmentDescriptor::setSampleBuffer(MTL::CounterSampleBuffer* sampleBuffer)
{
    _MTL_msg_v_setSampleBuffer__MTL__CounterSampleBufferp((const void*)this, nullptr, sampleBuffer);
}

_MTL_INLINE NS::UInteger MTL::RenderPassSampleBufferAttachmentDescriptor::startOfVertexSampleIndex() const
{
    return _MTL_msg_NS__UInteger_startOfVertexSampleIndex((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RenderPassSampleBufferAttachmentDescriptor::setStartOfVertexSampleIndex(NS::UInteger startOfVertexSampleIndex)
{
    _MTL_msg_v_setStartOfVertexSampleIndex__NS__UInteger((const void*)this, nullptr, startOfVertexSampleIndex);
}

_MTL_INLINE NS::UInteger MTL::RenderPassSampleBufferAttachmentDescriptor::endOfVertexSampleIndex() const
{
    return _MTL_msg_NS__UInteger_endOfVertexSampleIndex((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RenderPassSampleBufferAttachmentDescriptor::setEndOfVertexSampleIndex(NS::UInteger endOfVertexSampleIndex)
{
    _MTL_msg_v_setEndOfVertexSampleIndex__NS__UInteger((const void*)this, nullptr, endOfVertexSampleIndex);
}

_MTL_INLINE NS::UInteger MTL::RenderPassSampleBufferAttachmentDescriptor::startOfFragmentSampleIndex() const
{
    return _MTL_msg_NS__UInteger_startOfFragmentSampleIndex((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RenderPassSampleBufferAttachmentDescriptor::setStartOfFragmentSampleIndex(NS::UInteger startOfFragmentSampleIndex)
{
    _MTL_msg_v_setStartOfFragmentSampleIndex__NS__UInteger((const void*)this, nullptr, startOfFragmentSampleIndex);
}

_MTL_INLINE NS::UInteger MTL::RenderPassSampleBufferAttachmentDescriptor::endOfFragmentSampleIndex() const
{
    return _MTL_msg_NS__UInteger_endOfFragmentSampleIndex((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RenderPassSampleBufferAttachmentDescriptor::setEndOfFragmentSampleIndex(NS::UInteger endOfFragmentSampleIndex)
{
    _MTL_msg_v_setEndOfFragmentSampleIndex__NS__UInteger((const void*)this, nullptr, endOfFragmentSampleIndex);
}

_MTL_INLINE MTL::RenderPassSampleBufferAttachmentDescriptorArray* MTL::RenderPassSampleBufferAttachmentDescriptorArray::alloc()
{
    return _MTL_msg_MTL__RenderPassSampleBufferAttachmentDescriptorArrayp_alloc((const void*)&OBJC_CLASS_$_MTLRenderPassSampleBufferAttachmentDescriptorArray, nullptr);
}

_MTL_INLINE MTL::RenderPassSampleBufferAttachmentDescriptorArray* MTL::RenderPassSampleBufferAttachmentDescriptorArray::init() const
{
    return _MTL_msg_MTL__RenderPassSampleBufferAttachmentDescriptorArrayp_init((const void*)this, nullptr);
}

_MTL_INLINE MTL::RenderPassSampleBufferAttachmentDescriptor* MTL::RenderPassSampleBufferAttachmentDescriptorArray::object(NS::UInteger attachmentIndex)
{
    return _MTL_msg_MTL__RenderPassSampleBufferAttachmentDescriptorp_objectAtIndexedSubscript__NS__UInteger((const void*)this, nullptr, attachmentIndex);
}

_MTL_INLINE void MTL::RenderPassSampleBufferAttachmentDescriptorArray::setObject(MTL::RenderPassSampleBufferAttachmentDescriptor* attachment, NS::UInteger attachmentIndex)
{
    _MTL_msg_v_setObject_atIndexedSubscript__MTL__RenderPassSampleBufferAttachmentDescriptorp_NS__UInteger((const void*)this, nullptr, attachment, attachmentIndex);
}

_MTL_INLINE MTL::RenderPassDescriptor* MTL::RenderPassDescriptor::alloc()
{
    return _MTL_msg_MTL__RenderPassDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTLRenderPassDescriptor, nullptr);
}

_MTL_INLINE MTL::RenderPassDescriptor* MTL::RenderPassDescriptor::init() const
{
    return _MTL_msg_MTL__RenderPassDescriptorp_init((const void*)this, nullptr);
}

_MTL_INLINE MTL::RenderPassDescriptor* MTL::RenderPassDescriptor::renderPassDescriptor()
{
    return _MTL_msg_MTL__RenderPassDescriptorp_renderPassDescriptor((const void*)&OBJC_CLASS_$_MTLRenderPassDescriptor, nullptr);
}

_MTL_INLINE MTL::RenderPassColorAttachmentDescriptorArray* MTL::RenderPassDescriptor::colorAttachments() const
{
    return _MTL_msg_MTL__RenderPassColorAttachmentDescriptorArrayp_colorAttachments((const void*)this, nullptr);
}

_MTL_INLINE MTL::RenderPassDepthAttachmentDescriptor* MTL::RenderPassDescriptor::depthAttachment() const
{
    return _MTL_msg_MTL__RenderPassDepthAttachmentDescriptorp_depthAttachment((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RenderPassDescriptor::setDepthAttachment(MTL::RenderPassDepthAttachmentDescriptor* depthAttachment)
{
    _MTL_msg_v_setDepthAttachment__MTL__RenderPassDepthAttachmentDescriptorp((const void*)this, nullptr, depthAttachment);
}

_MTL_INLINE MTL::RenderPassStencilAttachmentDescriptor* MTL::RenderPassDescriptor::stencilAttachment() const
{
    return _MTL_msg_MTL__RenderPassStencilAttachmentDescriptorp_stencilAttachment((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RenderPassDescriptor::setStencilAttachment(MTL::RenderPassStencilAttachmentDescriptor* stencilAttachment)
{
    _MTL_msg_v_setStencilAttachment__MTL__RenderPassStencilAttachmentDescriptorp((const void*)this, nullptr, stencilAttachment);
}

_MTL_INLINE MTL::Buffer* MTL::RenderPassDescriptor::visibilityResultBuffer() const
{
    return _MTL_msg_MTL__Bufferp_visibilityResultBuffer((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RenderPassDescriptor::setVisibilityResultBuffer(MTL::Buffer* visibilityResultBuffer)
{
    _MTL_msg_v_setVisibilityResultBuffer__MTL__Bufferp((const void*)this, nullptr, visibilityResultBuffer);
}

_MTL_INLINE NS::UInteger MTL::RenderPassDescriptor::renderTargetArrayLength() const
{
    return _MTL_msg_NS__UInteger_renderTargetArrayLength((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RenderPassDescriptor::setRenderTargetArrayLength(NS::UInteger renderTargetArrayLength)
{
    _MTL_msg_v_setRenderTargetArrayLength__NS__UInteger((const void*)this, nullptr, renderTargetArrayLength);
}

_MTL_INLINE NS::UInteger MTL::RenderPassDescriptor::imageblockSampleLength() const
{
    return _MTL_msg_NS__UInteger_imageblockSampleLength((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RenderPassDescriptor::setImageblockSampleLength(NS::UInteger imageblockSampleLength)
{
    _MTL_msg_v_setImageblockSampleLength__NS__UInteger((const void*)this, nullptr, imageblockSampleLength);
}

_MTL_INLINE NS::UInteger MTL::RenderPassDescriptor::threadgroupMemoryLength() const
{
    return _MTL_msg_NS__UInteger_threadgroupMemoryLength((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RenderPassDescriptor::setThreadgroupMemoryLength(NS::UInteger threadgroupMemoryLength)
{
    _MTL_msg_v_setThreadgroupMemoryLength__NS__UInteger((const void*)this, nullptr, threadgroupMemoryLength);
}

_MTL_INLINE NS::UInteger MTL::RenderPassDescriptor::tileWidth() const
{
    return _MTL_msg_NS__UInteger_tileWidth((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RenderPassDescriptor::setTileWidth(NS::UInteger tileWidth)
{
    _MTL_msg_v_setTileWidth__NS__UInteger((const void*)this, nullptr, tileWidth);
}

_MTL_INLINE NS::UInteger MTL::RenderPassDescriptor::tileHeight() const
{
    return _MTL_msg_NS__UInteger_tileHeight((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RenderPassDescriptor::setTileHeight(NS::UInteger tileHeight)
{
    _MTL_msg_v_setTileHeight__NS__UInteger((const void*)this, nullptr, tileHeight);
}

_MTL_INLINE NS::UInteger MTL::RenderPassDescriptor::defaultRasterSampleCount() const
{
    return _MTL_msg_NS__UInteger_defaultRasterSampleCount((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RenderPassDescriptor::setDefaultRasterSampleCount(NS::UInteger defaultRasterSampleCount)
{
    _MTL_msg_v_setDefaultRasterSampleCount__NS__UInteger((const void*)this, nullptr, defaultRasterSampleCount);
}

_MTL_INLINE NS::UInteger MTL::RenderPassDescriptor::renderTargetWidth() const
{
    return _MTL_msg_NS__UInteger_renderTargetWidth((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RenderPassDescriptor::setRenderTargetWidth(NS::UInteger renderTargetWidth)
{
    _MTL_msg_v_setRenderTargetWidth__NS__UInteger((const void*)this, nullptr, renderTargetWidth);
}

_MTL_INLINE NS::UInteger MTL::RenderPassDescriptor::renderTargetHeight() const
{
    return _MTL_msg_NS__UInteger_renderTargetHeight((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RenderPassDescriptor::setRenderTargetHeight(NS::UInteger renderTargetHeight)
{
    _MTL_msg_v_setRenderTargetHeight__NS__UInteger((const void*)this, nullptr, renderTargetHeight);
}

_MTL_INLINE MTL::RasterizationRateMap* MTL::RenderPassDescriptor::rasterizationRateMap() const
{
    return _MTL_msg_MTL__RasterizationRateMapp_rasterizationRateMap((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RenderPassDescriptor::setRasterizationRateMap(MTL::RasterizationRateMap* rasterizationRateMap)
{
    _MTL_msg_v_setRasterizationRateMap__MTL__RasterizationRateMapp((const void*)this, nullptr, rasterizationRateMap);
}

_MTL_INLINE MTL::RenderPassSampleBufferAttachmentDescriptorArray* MTL::RenderPassDescriptor::sampleBufferAttachments() const
{
    return _MTL_msg_MTL__RenderPassSampleBufferAttachmentDescriptorArrayp_sampleBufferAttachments((const void*)this, nullptr);
}

_MTL_INLINE MTL::VisibilityResultType MTL::RenderPassDescriptor::visibilityResultType() const
{
    return _MTL_msg_MTL__VisibilityResultType_visibilityResultType((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RenderPassDescriptor::setVisibilityResultType(MTL::VisibilityResultType visibilityResultType)
{
    _MTL_msg_v_setVisibilityResultType__MTL__VisibilityResultType((const void*)this, nullptr, visibilityResultType);
}

_MTL_INLINE bool MTL::RenderPassDescriptor::supportColorAttachmentMapping() const
{
    return _MTL_msg_bool_supportColorAttachmentMapping((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RenderPassDescriptor::setSupportColorAttachmentMapping(bool supportColorAttachmentMapping)
{
    _MTL_msg_v_setSupportColorAttachmentMapping__bool((const void*)this, nullptr, supportColorAttachmentMapping);
}

_MTL_INLINE void MTL::RenderPassDescriptor::setSamplePositions(const MTL::SamplePosition * positions, NS::UInteger count)
{
    _MTL_msg_v_setSamplePositions_count__constMTL__SamplePositionp_NS__UInteger((const void*)this, nullptr, positions, count);
}

_MTL_INLINE NS::UInteger MTL::RenderPassDescriptor::getSamplePositions(MTL::SamplePosition* positions, NS::UInteger count)
{
    return _MTL_msg_NS__UInteger_getSamplePositions_count__MTL__SamplePositionp_NS__UInteger((const void*)this, nullptr, positions, count);
}
