//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Metal/MTLSampler.hpp
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
#include "MTLDepthStencil.hpp"
#include "MTLHeaderBridge.hpp"
#include "MTLPrivate.hpp"
#include "MTLTypes.hpp"

namespace MTL
{
class Device;
class SamplerDescriptor;
_MTL_ENUM(NS::UInteger, SamplerMinMagFilter) {
    SamplerMinMagFilterNearest = 0,
    SamplerMinMagFilterLinear = 1,
};

_MTL_ENUM(NS::UInteger, SamplerMipFilter) {
    SamplerMipFilterNotMipmapped = 0,
    SamplerMipFilterNearest = 1,
    SamplerMipFilterLinear = 2,
};

_MTL_ENUM(NS::UInteger, SamplerAddressMode) {
    SamplerAddressModeClampToEdge = 0,
    SamplerAddressModeMirrorClampToEdge = 1,
    SamplerAddressModeRepeat = 2,
    SamplerAddressModeMirrorRepeat = 3,
    SamplerAddressModeClampToZero = 4,
    SamplerAddressModeClampToBorderColor = 5,
};

_MTL_ENUM(NS::UInteger, SamplerBorderColor) {
    SamplerBorderColorTransparentBlack = 0,
    SamplerBorderColorOpaqueBlack = 1,
    SamplerBorderColorOpaqueWhite = 2,
};

_MTL_ENUM(NS::UInteger, SamplerReductionMode) {
    SamplerReductionModeWeightedAverage = 0,
    SamplerReductionModeMinimum = 1,
    SamplerReductionModeMaximum = 2,
};

class SamplerDescriptor : public NS::Copying<SamplerDescriptor>
{
public:
    static SamplerDescriptor* alloc();

    SamplerBorderColor        borderColor() const;

    CompareFunction           compareFunction() const;

    SamplerDescriptor*        init();

    NS::String*               label() const;

    bool                      lodAverage() const;

    float                     lodBias() const;

    float                     lodMaxClamp() const;

    float                     lodMinClamp() const;

    SamplerMinMagFilter       magFilter() const;

    NS::UInteger              maxAnisotropy() const;

    SamplerMinMagFilter       minFilter() const;

    SamplerMipFilter          mipFilter() const;

    bool                      normalizedCoordinates() const;

    SamplerAddressMode        rAddressMode() const;

    SamplerReductionMode      reductionMode() const;

    SamplerAddressMode        sAddressMode() const;

    void                      setBorderColor(MTL::SamplerBorderColor borderColor);

    void                      setCompareFunction(MTL::CompareFunction compareFunction);

    void                      setLabel(const NS::String* label);

    void                      setLodAverage(bool lodAverage);

    void                      setLodBias(float lodBias);

    void                      setLodMaxClamp(float lodMaxClamp);

    void                      setLodMinClamp(float lodMinClamp);

    void                      setMagFilter(MTL::SamplerMinMagFilter magFilter);

    void                      setMaxAnisotropy(NS::UInteger maxAnisotropy);

    void                      setMinFilter(MTL::SamplerMinMagFilter minFilter);

    void                      setMipFilter(MTL::SamplerMipFilter mipFilter);

    void                      setNormalizedCoordinates(bool normalizedCoordinates);

    void                      setRAddressMode(MTL::SamplerAddressMode rAddressMode);

    void                      setReductionMode(MTL::SamplerReductionMode reductionMode);

    void                      setSAddressMode(MTL::SamplerAddressMode sAddressMode);

    void                      setSupportArgumentBuffers(bool supportArgumentBuffers);

    void                      setTAddressMode(MTL::SamplerAddressMode tAddressMode);

    bool                      supportArgumentBuffers() const;

    SamplerAddressMode        tAddressMode() const;
};
class SamplerState : public NS::Referencing<SamplerState>
{
public:
    Device*     device() const;

    ResourceID  gpuResourceID() const;

    NS::String* label() const;
};

}
_MTL_INLINE MTL::SamplerDescriptor* MTL::SamplerDescriptor::alloc()
{
    return NS::Object::alloc<MTL::SamplerDescriptor>(_MTL_PRIVATE_CLS(MTLSamplerDescriptor));
}

_MTL_INLINE MTL::SamplerBorderColor MTL::SamplerDescriptor::borderColor() const
{
    return Object::sendMessage<MTL::SamplerBorderColor>(this, _MTL_PRIVATE_SEL(borderColor));
}

_MTL_INLINE MTL::CompareFunction MTL::SamplerDescriptor::compareFunction() const
{
    return Object::sendMessage<MTL::CompareFunction>(this, _MTL_PRIVATE_SEL(compareFunction));
}

_MTL_INLINE MTL::SamplerDescriptor* MTL::SamplerDescriptor::init()
{
    return NS::Object::init<MTL::SamplerDescriptor>();
}

_MTL_INLINE NS::String* MTL::SamplerDescriptor::label() const
{
    return Object::sendMessage<NS::String*>(this, _MTL_PRIVATE_SEL(label));
}

_MTL_INLINE bool MTL::SamplerDescriptor::lodAverage() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(lodAverage));
}

_MTL_INLINE float MTL::SamplerDescriptor::lodBias() const
{
    return Object::sendMessage<float>(this, _MTL_PRIVATE_SEL(lodBias));
}

_MTL_INLINE float MTL::SamplerDescriptor::lodMaxClamp() const
{
    return Object::sendMessage<float>(this, _MTL_PRIVATE_SEL(lodMaxClamp));
}

_MTL_INLINE float MTL::SamplerDescriptor::lodMinClamp() const
{
    return Object::sendMessage<float>(this, _MTL_PRIVATE_SEL(lodMinClamp));
}

_MTL_INLINE MTL::SamplerMinMagFilter MTL::SamplerDescriptor::magFilter() const
{
    return Object::sendMessage<MTL::SamplerMinMagFilter>(this, _MTL_PRIVATE_SEL(magFilter));
}

_MTL_INLINE NS::UInteger MTL::SamplerDescriptor::maxAnisotropy() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(maxAnisotropy));
}

_MTL_INLINE MTL::SamplerMinMagFilter MTL::SamplerDescriptor::minFilter() const
{
    return Object::sendMessage<MTL::SamplerMinMagFilter>(this, _MTL_PRIVATE_SEL(minFilter));
}

_MTL_INLINE MTL::SamplerMipFilter MTL::SamplerDescriptor::mipFilter() const
{
    return Object::sendMessage<MTL::SamplerMipFilter>(this, _MTL_PRIVATE_SEL(mipFilter));
}

_MTL_INLINE bool MTL::SamplerDescriptor::normalizedCoordinates() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(normalizedCoordinates));
}

_MTL_INLINE MTL::SamplerAddressMode MTL::SamplerDescriptor::rAddressMode() const
{
    return Object::sendMessage<MTL::SamplerAddressMode>(this, _MTL_PRIVATE_SEL(rAddressMode));
}

_MTL_INLINE MTL::SamplerReductionMode MTL::SamplerDescriptor::reductionMode() const
{
    return Object::sendMessage<MTL::SamplerReductionMode>(this, _MTL_PRIVATE_SEL(reductionMode));
}

_MTL_INLINE MTL::SamplerAddressMode MTL::SamplerDescriptor::sAddressMode() const
{
    return Object::sendMessage<MTL::SamplerAddressMode>(this, _MTL_PRIVATE_SEL(sAddressMode));
}

_MTL_INLINE void MTL::SamplerDescriptor::setBorderColor(MTL::SamplerBorderColor borderColor)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setBorderColor_), borderColor);
}

_MTL_INLINE void MTL::SamplerDescriptor::setCompareFunction(MTL::CompareFunction compareFunction)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setCompareFunction_), compareFunction);
}

_MTL_INLINE void MTL::SamplerDescriptor::setLabel(const NS::String* label)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setLabel_), label);
}

_MTL_INLINE void MTL::SamplerDescriptor::setLodAverage(bool lodAverage)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setLodAverage_), lodAverage);
}

_MTL_INLINE void MTL::SamplerDescriptor::setLodBias(float lodBias)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setLodBias_), lodBias);
}

_MTL_INLINE void MTL::SamplerDescriptor::setLodMaxClamp(float lodMaxClamp)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setLodMaxClamp_), lodMaxClamp);
}

_MTL_INLINE void MTL::SamplerDescriptor::setLodMinClamp(float lodMinClamp)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setLodMinClamp_), lodMinClamp);
}

_MTL_INLINE void MTL::SamplerDescriptor::setMagFilter(MTL::SamplerMinMagFilter magFilter)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMagFilter_), magFilter);
}

_MTL_INLINE void MTL::SamplerDescriptor::setMaxAnisotropy(NS::UInteger maxAnisotropy)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMaxAnisotropy_), maxAnisotropy);
}

_MTL_INLINE void MTL::SamplerDescriptor::setMinFilter(MTL::SamplerMinMagFilter minFilter)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMinFilter_), minFilter);
}

_MTL_INLINE void MTL::SamplerDescriptor::setMipFilter(MTL::SamplerMipFilter mipFilter)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMipFilter_), mipFilter);
}

_MTL_INLINE void MTL::SamplerDescriptor::setNormalizedCoordinates(bool normalizedCoordinates)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setNormalizedCoordinates_), normalizedCoordinates);
}

_MTL_INLINE void MTL::SamplerDescriptor::setRAddressMode(MTL::SamplerAddressMode rAddressMode)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setRAddressMode_), rAddressMode);
}

_MTL_INLINE void MTL::SamplerDescriptor::setReductionMode(MTL::SamplerReductionMode reductionMode)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setReductionMode_), reductionMode);
}

_MTL_INLINE void MTL::SamplerDescriptor::setSAddressMode(MTL::SamplerAddressMode sAddressMode)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setSAddressMode_), sAddressMode);
}

_MTL_INLINE void MTL::SamplerDescriptor::setSupportArgumentBuffers(bool supportArgumentBuffers)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setSupportArgumentBuffers_), supportArgumentBuffers);
}

_MTL_INLINE void MTL::SamplerDescriptor::setTAddressMode(MTL::SamplerAddressMode tAddressMode)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setTAddressMode_), tAddressMode);
}

_MTL_INLINE bool MTL::SamplerDescriptor::supportArgumentBuffers() const
{
    return Object::sendMessageSafe<bool>(this, _MTL_PRIVATE_SEL(supportArgumentBuffers));
}

_MTL_INLINE MTL::SamplerAddressMode MTL::SamplerDescriptor::tAddressMode() const
{
    return Object::sendMessage<MTL::SamplerAddressMode>(this, _MTL_PRIVATE_SEL(tAddressMode));
}

_MTL_INLINE MTL::Device* MTL::SamplerState::device() const
{
    return Object::sendMessage<MTL::Device*>(this, _MTL_PRIVATE_SEL(device));
}

_MTL_INLINE MTL::ResourceID MTL::SamplerState::gpuResourceID() const
{
    return Object::sendMessage<MTL::ResourceID>(this, _MTL_PRIVATE_SEL(gpuResourceID));
}

_MTL_INLINE NS::String* MTL::SamplerState::label() const
{
    return Object::sendMessage<NS::String*>(this, _MTL_PRIVATE_SEL(label));
}
