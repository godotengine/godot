//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Metal/MTLRenderPass.hpp
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
#include "MTLDefines.hpp"
#include "MTLHeaderBridge.hpp"
#include "MTLPrivate.hpp"
#include <cstdint>

namespace MTL
{
class Buffer;
class CounterSampleBuffer;
class RasterizationRateMap;
class RenderPassAttachmentDescriptor;
class RenderPassColorAttachmentDescriptor;
class RenderPassColorAttachmentDescriptorArray;
class RenderPassDepthAttachmentDescriptor;
class RenderPassDescriptor;
class RenderPassSampleBufferAttachmentDescriptor;
class RenderPassSampleBufferAttachmentDescriptorArray;
class RenderPassStencilAttachmentDescriptor;
struct SamplePosition;
class Texture;
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

_MTL_OPTIONS(NS::UInteger, StoreActionOptions) {
    StoreActionOptionNone = 0,
    StoreActionOptionCustomSamplePositions = 1,
    StoreActionOptionValidMask = 1,
};

struct ClearColor
{
    ClearColor() = default;

    ClearColor(double red, double green, double blue, double alpha);

    static ClearColor Make(double red, double green, double blue, double alpha);

    double            red;
    double            green;
    double            blue;
    double            alpha;
} _MTL_PACKED;

class RenderPassAttachmentDescriptor : public NS::Copying<RenderPassAttachmentDescriptor>
{
public:
    static RenderPassAttachmentDescriptor* alloc();

    NS::UInteger                           depthPlane() const;

    RenderPassAttachmentDescriptor*        init();

    NS::UInteger                           level() const;

    LoadAction                             loadAction() const;

    NS::UInteger                           resolveDepthPlane() const;

    NS::UInteger                           resolveLevel() const;

    NS::UInteger                           resolveSlice() const;

    Texture*                               resolveTexture() const;

    void                                   setDepthPlane(NS::UInteger depthPlane);

    void                                   setLevel(NS::UInteger level);

    void                                   setLoadAction(MTL::LoadAction loadAction);

    void                                   setResolveDepthPlane(NS::UInteger resolveDepthPlane);

    void                                   setResolveLevel(NS::UInteger resolveLevel);

    void                                   setResolveSlice(NS::UInteger resolveSlice);

    void                                   setResolveTexture(const MTL::Texture* resolveTexture);

    void                                   setSlice(NS::UInteger slice);

    void                                   setStoreAction(MTL::StoreAction storeAction);
    void                                   setStoreActionOptions(MTL::StoreActionOptions storeActionOptions);

    void                                   setTexture(const MTL::Texture* texture);

    NS::UInteger                           slice() const;

    StoreAction                            storeAction() const;
    StoreActionOptions                     storeActionOptions() const;

    Texture*                               texture() const;
};
class RenderPassColorAttachmentDescriptor : public NS::Copying<RenderPassColorAttachmentDescriptor, RenderPassAttachmentDescriptor>
{
public:
    static RenderPassColorAttachmentDescriptor* alloc();

    ClearColor                                  clearColor() const;

    RenderPassColorAttachmentDescriptor*        init();

    void                                        setClearColor(MTL::ClearColor clearColor);
};
class RenderPassDepthAttachmentDescriptor : public NS::Copying<RenderPassDepthAttachmentDescriptor, RenderPassAttachmentDescriptor>
{
public:
    static RenderPassDepthAttachmentDescriptor* alloc();

    double                                      clearDepth() const;

    MultisampleDepthResolveFilter               depthResolveFilter() const;

    RenderPassDepthAttachmentDescriptor*        init();

    void                                        setClearDepth(double clearDepth);

    void                                        setDepthResolveFilter(MTL::MultisampleDepthResolveFilter depthResolveFilter);
};
class RenderPassStencilAttachmentDescriptor : public NS::Copying<RenderPassStencilAttachmentDescriptor, RenderPassAttachmentDescriptor>
{
public:
    static RenderPassStencilAttachmentDescriptor* alloc();

    uint32_t                                      clearStencil() const;

    RenderPassStencilAttachmentDescriptor*        init();

    void                                          setClearStencil(uint32_t clearStencil);

    void                                          setStencilResolveFilter(MTL::MultisampleStencilResolveFilter stencilResolveFilter);
    MultisampleStencilResolveFilter               stencilResolveFilter() const;
};
class RenderPassColorAttachmentDescriptorArray : public NS::Referencing<RenderPassColorAttachmentDescriptorArray>
{
public:
    static RenderPassColorAttachmentDescriptorArray* alloc();

    RenderPassColorAttachmentDescriptorArray*        init();

    RenderPassColorAttachmentDescriptor*             object(NS::UInteger attachmentIndex);
    void                                             setObject(const MTL::RenderPassColorAttachmentDescriptor* attachment, NS::UInteger attachmentIndex);
};
class RenderPassSampleBufferAttachmentDescriptor : public NS::Copying<RenderPassSampleBufferAttachmentDescriptor>
{
public:
    static RenderPassSampleBufferAttachmentDescriptor* alloc();

    NS::UInteger                                       endOfFragmentSampleIndex() const;

    NS::UInteger                                       endOfVertexSampleIndex() const;

    RenderPassSampleBufferAttachmentDescriptor*        init();

    CounterSampleBuffer*                               sampleBuffer() const;

    void                                               setEndOfFragmentSampleIndex(NS::UInteger endOfFragmentSampleIndex);

    void                                               setEndOfVertexSampleIndex(NS::UInteger endOfVertexSampleIndex);

    void                                               setSampleBuffer(const MTL::CounterSampleBuffer* sampleBuffer);

    void                                               setStartOfFragmentSampleIndex(NS::UInteger startOfFragmentSampleIndex);

    void                                               setStartOfVertexSampleIndex(NS::UInteger startOfVertexSampleIndex);

    NS::UInteger                                       startOfFragmentSampleIndex() const;

    NS::UInteger                                       startOfVertexSampleIndex() const;
};
class RenderPassSampleBufferAttachmentDescriptorArray : public NS::Referencing<RenderPassSampleBufferAttachmentDescriptorArray>
{
public:
    static RenderPassSampleBufferAttachmentDescriptorArray* alloc();

    RenderPassSampleBufferAttachmentDescriptorArray*        init();

    RenderPassSampleBufferAttachmentDescriptor*             object(NS::UInteger attachmentIndex);
    void                                                    setObject(const MTL::RenderPassSampleBufferAttachmentDescriptor* attachment, NS::UInteger attachmentIndex);
};
class RenderPassDescriptor : public NS::Copying<RenderPassDescriptor>
{
public:
    static RenderPassDescriptor*                     alloc();

    RenderPassColorAttachmentDescriptorArray*        colorAttachments() const;

    NS::UInteger                                     defaultRasterSampleCount() const;

    RenderPassDepthAttachmentDescriptor*             depthAttachment() const;

    NS::UInteger                                     getSamplePositions(MTL::SamplePosition* positions, NS::UInteger count);

    NS::UInteger                                     imageblockSampleLength() const;

    RenderPassDescriptor*                            init();

    RasterizationRateMap*                            rasterizationRateMap() const;

    static RenderPassDescriptor*                     renderPassDescriptor();

    NS::UInteger                                     renderTargetArrayLength() const;

    NS::UInteger                                     renderTargetHeight() const;

    NS::UInteger                                     renderTargetWidth() const;

    RenderPassSampleBufferAttachmentDescriptorArray* sampleBufferAttachments() const;

    void                                             setDefaultRasterSampleCount(NS::UInteger defaultRasterSampleCount);

    void                                             setDepthAttachment(const MTL::RenderPassDepthAttachmentDescriptor* depthAttachment);

    void                                             setImageblockSampleLength(NS::UInteger imageblockSampleLength);

    void                                             setRasterizationRateMap(const MTL::RasterizationRateMap* rasterizationRateMap);

    void                                             setRenderTargetArrayLength(NS::UInteger renderTargetArrayLength);

    void                                             setRenderTargetHeight(NS::UInteger renderTargetHeight);

    void                                             setRenderTargetWidth(NS::UInteger renderTargetWidth);

    void                                             setSamplePositions(const MTL::SamplePosition* positions, NS::UInteger count);

    void                                             setStencilAttachment(const MTL::RenderPassStencilAttachmentDescriptor* stencilAttachment);

    void                                             setSupportColorAttachmentMapping(bool supportColorAttachmentMapping);

    void                                             setThreadgroupMemoryLength(NS::UInteger threadgroupMemoryLength);

    void                                             setTileHeight(NS::UInteger tileHeight);

    void                                             setTileWidth(NS::UInteger tileWidth);

    void                                             setVisibilityResultBuffer(const MTL::Buffer* visibilityResultBuffer);

    void                                             setVisibilityResultType(MTL::VisibilityResultType visibilityResultType);

    RenderPassStencilAttachmentDescriptor*           stencilAttachment() const;

    bool                                             supportColorAttachmentMapping() const;

    NS::UInteger                                     threadgroupMemoryLength() const;

    NS::UInteger                                     tileHeight() const;

    NS::UInteger                                     tileWidth() const;

    Buffer*                                          visibilityResultBuffer() const;

    VisibilityResultType                             visibilityResultType() const;
};

}
_MTL_INLINE MTL::ClearColor::ClearColor(double red, double green, double blue, double alpha)
    : red(red)
    , green(green)
    , blue(blue)
    , alpha(alpha)
{
}

_MTL_INLINE MTL::ClearColor MTL::ClearColor::Make(double red, double green, double blue, double alpha)
{
    return ClearColor(red, green, blue, alpha);
}

_MTL_INLINE MTL::RenderPassAttachmentDescriptor* MTL::RenderPassAttachmentDescriptor::alloc()
{
    return NS::Object::alloc<MTL::RenderPassAttachmentDescriptor>(_MTL_PRIVATE_CLS(MTLRenderPassAttachmentDescriptor));
}

_MTL_INLINE NS::UInteger MTL::RenderPassAttachmentDescriptor::depthPlane() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(depthPlane));
}

_MTL_INLINE MTL::RenderPassAttachmentDescriptor* MTL::RenderPassAttachmentDescriptor::init()
{
    return NS::Object::init<MTL::RenderPassAttachmentDescriptor>();
}

_MTL_INLINE NS::UInteger MTL::RenderPassAttachmentDescriptor::level() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(level));
}

_MTL_INLINE MTL::LoadAction MTL::RenderPassAttachmentDescriptor::loadAction() const
{
    return Object::sendMessage<MTL::LoadAction>(this, _MTL_PRIVATE_SEL(loadAction));
}

_MTL_INLINE NS::UInteger MTL::RenderPassAttachmentDescriptor::resolveDepthPlane() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(resolveDepthPlane));
}

_MTL_INLINE NS::UInteger MTL::RenderPassAttachmentDescriptor::resolveLevel() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(resolveLevel));
}

_MTL_INLINE NS::UInteger MTL::RenderPassAttachmentDescriptor::resolveSlice() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(resolveSlice));
}

_MTL_INLINE MTL::Texture* MTL::RenderPassAttachmentDescriptor::resolveTexture() const
{
    return Object::sendMessage<MTL::Texture*>(this, _MTL_PRIVATE_SEL(resolveTexture));
}

_MTL_INLINE void MTL::RenderPassAttachmentDescriptor::setDepthPlane(NS::UInteger depthPlane)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setDepthPlane_), depthPlane);
}

_MTL_INLINE void MTL::RenderPassAttachmentDescriptor::setLevel(NS::UInteger level)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setLevel_), level);
}

_MTL_INLINE void MTL::RenderPassAttachmentDescriptor::setLoadAction(MTL::LoadAction loadAction)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setLoadAction_), loadAction);
}

_MTL_INLINE void MTL::RenderPassAttachmentDescriptor::setResolveDepthPlane(NS::UInteger resolveDepthPlane)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setResolveDepthPlane_), resolveDepthPlane);
}

_MTL_INLINE void MTL::RenderPassAttachmentDescriptor::setResolveLevel(NS::UInteger resolveLevel)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setResolveLevel_), resolveLevel);
}

_MTL_INLINE void MTL::RenderPassAttachmentDescriptor::setResolveSlice(NS::UInteger resolveSlice)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setResolveSlice_), resolveSlice);
}

_MTL_INLINE void MTL::RenderPassAttachmentDescriptor::setResolveTexture(const MTL::Texture* resolveTexture)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setResolveTexture_), resolveTexture);
}

_MTL_INLINE void MTL::RenderPassAttachmentDescriptor::setSlice(NS::UInteger slice)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setSlice_), slice);
}

_MTL_INLINE void MTL::RenderPassAttachmentDescriptor::setStoreAction(MTL::StoreAction storeAction)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setStoreAction_), storeAction);
}

_MTL_INLINE void MTL::RenderPassAttachmentDescriptor::setStoreActionOptions(MTL::StoreActionOptions storeActionOptions)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setStoreActionOptions_), storeActionOptions);
}

_MTL_INLINE void MTL::RenderPassAttachmentDescriptor::setTexture(const MTL::Texture* texture)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setTexture_), texture);
}

_MTL_INLINE NS::UInteger MTL::RenderPassAttachmentDescriptor::slice() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(slice));
}

_MTL_INLINE MTL::StoreAction MTL::RenderPassAttachmentDescriptor::storeAction() const
{
    return Object::sendMessage<MTL::StoreAction>(this, _MTL_PRIVATE_SEL(storeAction));
}

_MTL_INLINE MTL::StoreActionOptions MTL::RenderPassAttachmentDescriptor::storeActionOptions() const
{
    return Object::sendMessage<MTL::StoreActionOptions>(this, _MTL_PRIVATE_SEL(storeActionOptions));
}

_MTL_INLINE MTL::Texture* MTL::RenderPassAttachmentDescriptor::texture() const
{
    return Object::sendMessage<MTL::Texture*>(this, _MTL_PRIVATE_SEL(texture));
}

_MTL_INLINE MTL::RenderPassColorAttachmentDescriptor* MTL::RenderPassColorAttachmentDescriptor::alloc()
{
    return NS::Object::alloc<MTL::RenderPassColorAttachmentDescriptor>(_MTL_PRIVATE_CLS(MTLRenderPassColorAttachmentDescriptor));
}

_MTL_INLINE MTL::ClearColor MTL::RenderPassColorAttachmentDescriptor::clearColor() const
{
    return Object::sendMessage<MTL::ClearColor>(this, _MTL_PRIVATE_SEL(clearColor));
}

_MTL_INLINE MTL::RenderPassColorAttachmentDescriptor* MTL::RenderPassColorAttachmentDescriptor::init()
{
    return NS::Object::init<MTL::RenderPassColorAttachmentDescriptor>();
}

_MTL_INLINE void MTL::RenderPassColorAttachmentDescriptor::setClearColor(MTL::ClearColor clearColor)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setClearColor_), clearColor);
}

_MTL_INLINE MTL::RenderPassDepthAttachmentDescriptor* MTL::RenderPassDepthAttachmentDescriptor::alloc()
{
    return NS::Object::alloc<MTL::RenderPassDepthAttachmentDescriptor>(_MTL_PRIVATE_CLS(MTLRenderPassDepthAttachmentDescriptor));
}

_MTL_INLINE double MTL::RenderPassDepthAttachmentDescriptor::clearDepth() const
{
    return Object::sendMessage<double>(this, _MTL_PRIVATE_SEL(clearDepth));
}

_MTL_INLINE MTL::MultisampleDepthResolveFilter MTL::RenderPassDepthAttachmentDescriptor::depthResolveFilter() const
{
    return Object::sendMessage<MTL::MultisampleDepthResolveFilter>(this, _MTL_PRIVATE_SEL(depthResolveFilter));
}

_MTL_INLINE MTL::RenderPassDepthAttachmentDescriptor* MTL::RenderPassDepthAttachmentDescriptor::init()
{
    return NS::Object::init<MTL::RenderPassDepthAttachmentDescriptor>();
}

_MTL_INLINE void MTL::RenderPassDepthAttachmentDescriptor::setClearDepth(double clearDepth)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setClearDepth_), clearDepth);
}

_MTL_INLINE void MTL::RenderPassDepthAttachmentDescriptor::setDepthResolveFilter(MTL::MultisampleDepthResolveFilter depthResolveFilter)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setDepthResolveFilter_), depthResolveFilter);
}

_MTL_INLINE MTL::RenderPassStencilAttachmentDescriptor* MTL::RenderPassStencilAttachmentDescriptor::alloc()
{
    return NS::Object::alloc<MTL::RenderPassStencilAttachmentDescriptor>(_MTL_PRIVATE_CLS(MTLRenderPassStencilAttachmentDescriptor));
}

_MTL_INLINE uint32_t MTL::RenderPassStencilAttachmentDescriptor::clearStencil() const
{
    return Object::sendMessage<uint32_t>(this, _MTL_PRIVATE_SEL(clearStencil));
}

_MTL_INLINE MTL::RenderPassStencilAttachmentDescriptor* MTL::RenderPassStencilAttachmentDescriptor::init()
{
    return NS::Object::init<MTL::RenderPassStencilAttachmentDescriptor>();
}

_MTL_INLINE void MTL::RenderPassStencilAttachmentDescriptor::setClearStencil(uint32_t clearStencil)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setClearStencil_), clearStencil);
}

_MTL_INLINE void MTL::RenderPassStencilAttachmentDescriptor::setStencilResolveFilter(MTL::MultisampleStencilResolveFilter stencilResolveFilter)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setStencilResolveFilter_), stencilResolveFilter);
}

_MTL_INLINE MTL::MultisampleStencilResolveFilter MTL::RenderPassStencilAttachmentDescriptor::stencilResolveFilter() const
{
    return Object::sendMessage<MTL::MultisampleStencilResolveFilter>(this, _MTL_PRIVATE_SEL(stencilResolveFilter));
}

_MTL_INLINE MTL::RenderPassColorAttachmentDescriptorArray* MTL::RenderPassColorAttachmentDescriptorArray::alloc()
{
    return NS::Object::alloc<MTL::RenderPassColorAttachmentDescriptorArray>(_MTL_PRIVATE_CLS(MTLRenderPassColorAttachmentDescriptorArray));
}

_MTL_INLINE MTL::RenderPassColorAttachmentDescriptorArray* MTL::RenderPassColorAttachmentDescriptorArray::init()
{
    return NS::Object::init<MTL::RenderPassColorAttachmentDescriptorArray>();
}

_MTL_INLINE MTL::RenderPassColorAttachmentDescriptor* MTL::RenderPassColorAttachmentDescriptorArray::object(NS::UInteger attachmentIndex)
{
    return Object::sendMessage<MTL::RenderPassColorAttachmentDescriptor*>(this, _MTL_PRIVATE_SEL(objectAtIndexedSubscript_), attachmentIndex);
}

_MTL_INLINE void MTL::RenderPassColorAttachmentDescriptorArray::setObject(const MTL::RenderPassColorAttachmentDescriptor* attachment, NS::UInteger attachmentIndex)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setObject_atIndexedSubscript_), attachment, attachmentIndex);
}

_MTL_INLINE MTL::RenderPassSampleBufferAttachmentDescriptor* MTL::RenderPassSampleBufferAttachmentDescriptor::alloc()
{
    return NS::Object::alloc<MTL::RenderPassSampleBufferAttachmentDescriptor>(_MTL_PRIVATE_CLS(MTLRenderPassSampleBufferAttachmentDescriptor));
}

_MTL_INLINE NS::UInteger MTL::RenderPassSampleBufferAttachmentDescriptor::endOfFragmentSampleIndex() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(endOfFragmentSampleIndex));
}

_MTL_INLINE NS::UInteger MTL::RenderPassSampleBufferAttachmentDescriptor::endOfVertexSampleIndex() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(endOfVertexSampleIndex));
}

_MTL_INLINE MTL::RenderPassSampleBufferAttachmentDescriptor* MTL::RenderPassSampleBufferAttachmentDescriptor::init()
{
    return NS::Object::init<MTL::RenderPassSampleBufferAttachmentDescriptor>();
}

_MTL_INLINE MTL::CounterSampleBuffer* MTL::RenderPassSampleBufferAttachmentDescriptor::sampleBuffer() const
{
    return Object::sendMessage<MTL::CounterSampleBuffer*>(this, _MTL_PRIVATE_SEL(sampleBuffer));
}

_MTL_INLINE void MTL::RenderPassSampleBufferAttachmentDescriptor::setEndOfFragmentSampleIndex(NS::UInteger endOfFragmentSampleIndex)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setEndOfFragmentSampleIndex_), endOfFragmentSampleIndex);
}

_MTL_INLINE void MTL::RenderPassSampleBufferAttachmentDescriptor::setEndOfVertexSampleIndex(NS::UInteger endOfVertexSampleIndex)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setEndOfVertexSampleIndex_), endOfVertexSampleIndex);
}

_MTL_INLINE void MTL::RenderPassSampleBufferAttachmentDescriptor::setSampleBuffer(const MTL::CounterSampleBuffer* sampleBuffer)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setSampleBuffer_), sampleBuffer);
}

_MTL_INLINE void MTL::RenderPassSampleBufferAttachmentDescriptor::setStartOfFragmentSampleIndex(NS::UInteger startOfFragmentSampleIndex)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setStartOfFragmentSampleIndex_), startOfFragmentSampleIndex);
}

_MTL_INLINE void MTL::RenderPassSampleBufferAttachmentDescriptor::setStartOfVertexSampleIndex(NS::UInteger startOfVertexSampleIndex)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setStartOfVertexSampleIndex_), startOfVertexSampleIndex);
}

_MTL_INLINE NS::UInteger MTL::RenderPassSampleBufferAttachmentDescriptor::startOfFragmentSampleIndex() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(startOfFragmentSampleIndex));
}

_MTL_INLINE NS::UInteger MTL::RenderPassSampleBufferAttachmentDescriptor::startOfVertexSampleIndex() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(startOfVertexSampleIndex));
}

_MTL_INLINE MTL::RenderPassSampleBufferAttachmentDescriptorArray* MTL::RenderPassSampleBufferAttachmentDescriptorArray::alloc()
{
    return NS::Object::alloc<MTL::RenderPassSampleBufferAttachmentDescriptorArray>(_MTL_PRIVATE_CLS(MTLRenderPassSampleBufferAttachmentDescriptorArray));
}

_MTL_INLINE MTL::RenderPassSampleBufferAttachmentDescriptorArray* MTL::RenderPassSampleBufferAttachmentDescriptorArray::init()
{
    return NS::Object::init<MTL::RenderPassSampleBufferAttachmentDescriptorArray>();
}

_MTL_INLINE MTL::RenderPassSampleBufferAttachmentDescriptor* MTL::RenderPassSampleBufferAttachmentDescriptorArray::object(NS::UInteger attachmentIndex)
{
    return Object::sendMessage<MTL::RenderPassSampleBufferAttachmentDescriptor*>(this, _MTL_PRIVATE_SEL(objectAtIndexedSubscript_), attachmentIndex);
}

_MTL_INLINE void MTL::RenderPassSampleBufferAttachmentDescriptorArray::setObject(const MTL::RenderPassSampleBufferAttachmentDescriptor* attachment, NS::UInteger attachmentIndex)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setObject_atIndexedSubscript_), attachment, attachmentIndex);
}

_MTL_INLINE MTL::RenderPassDescriptor* MTL::RenderPassDescriptor::alloc()
{
    return NS::Object::alloc<MTL::RenderPassDescriptor>(_MTL_PRIVATE_CLS(MTLRenderPassDescriptor));
}

_MTL_INLINE MTL::RenderPassColorAttachmentDescriptorArray* MTL::RenderPassDescriptor::colorAttachments() const
{
    return Object::sendMessage<MTL::RenderPassColorAttachmentDescriptorArray*>(this, _MTL_PRIVATE_SEL(colorAttachments));
}

_MTL_INLINE NS::UInteger MTL::RenderPassDescriptor::defaultRasterSampleCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(defaultRasterSampleCount));
}

_MTL_INLINE MTL::RenderPassDepthAttachmentDescriptor* MTL::RenderPassDescriptor::depthAttachment() const
{
    return Object::sendMessage<MTL::RenderPassDepthAttachmentDescriptor*>(this, _MTL_PRIVATE_SEL(depthAttachment));
}

_MTL_INLINE NS::UInteger MTL::RenderPassDescriptor::getSamplePositions(MTL::SamplePosition* positions, NS::UInteger count)
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(getSamplePositions_count_), positions, count);
}

_MTL_INLINE NS::UInteger MTL::RenderPassDescriptor::imageblockSampleLength() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(imageblockSampleLength));
}

_MTL_INLINE MTL::RenderPassDescriptor* MTL::RenderPassDescriptor::init()
{
    return NS::Object::init<MTL::RenderPassDescriptor>();
}

_MTL_INLINE MTL::RasterizationRateMap* MTL::RenderPassDescriptor::rasterizationRateMap() const
{
    return Object::sendMessage<MTL::RasterizationRateMap*>(this, _MTL_PRIVATE_SEL(rasterizationRateMap));
}

_MTL_INLINE MTL::RenderPassDescriptor* MTL::RenderPassDescriptor::renderPassDescriptor()
{
    return Object::sendMessage<MTL::RenderPassDescriptor*>(_MTL_PRIVATE_CLS(MTLRenderPassDescriptor), _MTL_PRIVATE_SEL(renderPassDescriptor));
}

_MTL_INLINE NS::UInteger MTL::RenderPassDescriptor::renderTargetArrayLength() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(renderTargetArrayLength));
}

_MTL_INLINE NS::UInteger MTL::RenderPassDescriptor::renderTargetHeight() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(renderTargetHeight));
}

_MTL_INLINE NS::UInteger MTL::RenderPassDescriptor::renderTargetWidth() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(renderTargetWidth));
}

_MTL_INLINE MTL::RenderPassSampleBufferAttachmentDescriptorArray* MTL::RenderPassDescriptor::sampleBufferAttachments() const
{
    return Object::sendMessage<MTL::RenderPassSampleBufferAttachmentDescriptorArray*>(this, _MTL_PRIVATE_SEL(sampleBufferAttachments));
}

_MTL_INLINE void MTL::RenderPassDescriptor::setDefaultRasterSampleCount(NS::UInteger defaultRasterSampleCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setDefaultRasterSampleCount_), defaultRasterSampleCount);
}

_MTL_INLINE void MTL::RenderPassDescriptor::setDepthAttachment(const MTL::RenderPassDepthAttachmentDescriptor* depthAttachment)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setDepthAttachment_), depthAttachment);
}

_MTL_INLINE void MTL::RenderPassDescriptor::setImageblockSampleLength(NS::UInteger imageblockSampleLength)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setImageblockSampleLength_), imageblockSampleLength);
}

_MTL_INLINE void MTL::RenderPassDescriptor::setRasterizationRateMap(const MTL::RasterizationRateMap* rasterizationRateMap)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setRasterizationRateMap_), rasterizationRateMap);
}

_MTL_INLINE void MTL::RenderPassDescriptor::setRenderTargetArrayLength(NS::UInteger renderTargetArrayLength)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setRenderTargetArrayLength_), renderTargetArrayLength);
}

_MTL_INLINE void MTL::RenderPassDescriptor::setRenderTargetHeight(NS::UInteger renderTargetHeight)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setRenderTargetHeight_), renderTargetHeight);
}

_MTL_INLINE void MTL::RenderPassDescriptor::setRenderTargetWidth(NS::UInteger renderTargetWidth)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setRenderTargetWidth_), renderTargetWidth);
}

_MTL_INLINE void MTL::RenderPassDescriptor::setSamplePositions(const MTL::SamplePosition* positions, NS::UInteger count)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setSamplePositions_count_), positions, count);
}

_MTL_INLINE void MTL::RenderPassDescriptor::setStencilAttachment(const MTL::RenderPassStencilAttachmentDescriptor* stencilAttachment)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setStencilAttachment_), stencilAttachment);
}

_MTL_INLINE void MTL::RenderPassDescriptor::setSupportColorAttachmentMapping(bool supportColorAttachmentMapping)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setSupportColorAttachmentMapping_), supportColorAttachmentMapping);
}

_MTL_INLINE void MTL::RenderPassDescriptor::setThreadgroupMemoryLength(NS::UInteger threadgroupMemoryLength)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setThreadgroupMemoryLength_), threadgroupMemoryLength);
}

_MTL_INLINE void MTL::RenderPassDescriptor::setTileHeight(NS::UInteger tileHeight)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setTileHeight_), tileHeight);
}

_MTL_INLINE void MTL::RenderPassDescriptor::setTileWidth(NS::UInteger tileWidth)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setTileWidth_), tileWidth);
}

_MTL_INLINE void MTL::RenderPassDescriptor::setVisibilityResultBuffer(const MTL::Buffer* visibilityResultBuffer)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setVisibilityResultBuffer_), visibilityResultBuffer);
}

_MTL_INLINE void MTL::RenderPassDescriptor::setVisibilityResultType(MTL::VisibilityResultType visibilityResultType)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setVisibilityResultType_), visibilityResultType);
}

_MTL_INLINE MTL::RenderPassStencilAttachmentDescriptor* MTL::RenderPassDescriptor::stencilAttachment() const
{
    return Object::sendMessage<MTL::RenderPassStencilAttachmentDescriptor*>(this, _MTL_PRIVATE_SEL(stencilAttachment));
}

_MTL_INLINE bool MTL::RenderPassDescriptor::supportColorAttachmentMapping() const
{
    return Object::sendMessageSafe<bool>(this, _MTL_PRIVATE_SEL(supportColorAttachmentMapping));
}

_MTL_INLINE NS::UInteger MTL::RenderPassDescriptor::threadgroupMemoryLength() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(threadgroupMemoryLength));
}

_MTL_INLINE NS::UInteger MTL::RenderPassDescriptor::tileHeight() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(tileHeight));
}

_MTL_INLINE NS::UInteger MTL::RenderPassDescriptor::tileWidth() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(tileWidth));
}

_MTL_INLINE MTL::Buffer* MTL::RenderPassDescriptor::visibilityResultBuffer() const
{
    return Object::sendMessage<MTL::Buffer*>(this, _MTL_PRIVATE_SEL(visibilityResultBuffer));
}

_MTL_INLINE MTL::VisibilityResultType MTL::RenderPassDescriptor::visibilityResultType() const
{
    return Object::sendMessage<MTL::VisibilityResultType>(this, _MTL_PRIVATE_SEL(visibilityResultType));
}
