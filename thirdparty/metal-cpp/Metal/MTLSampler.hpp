#pragma once

#include "MTLDefines.hpp"
#include "MTLBlocks.hpp"
#include "MTLStructs.hpp"
#include "MTLBridge.hpp"
#include "../Foundation/NSObject.hpp"
#include "../Foundation/NSTypes.hpp"
#include "../Foundation/NSRange.hpp"

namespace MTL {
    class Device;
    enum CompareFunction : NS::UInteger;
}
namespace NS {
    class String;
}

namespace MTL
{

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


class SamplerDescriptor;
class SamplerState;

class SamplerDescriptor : public NS::Copying<SamplerDescriptor>
{
public:
    static SamplerDescriptor* alloc();
    SamplerDescriptor*        init() const;

    MTL::SamplerBorderColor   borderColor() const;
    MTL::CompareFunction      compareFunction() const;
    NS::String*               label() const;
    bool                      lodAverage() const;
    float                     lodBias() const;
    float                     lodMaxClamp() const;
    float                     lodMinClamp() const;
    MTL::SamplerMinMagFilter  magFilter() const;
    NS::UInteger              maxAnisotropy() const;
    MTL::SamplerMinMagFilter  minFilter() const;
    MTL::SamplerMipFilter     mipFilter() const;
    bool                      normalizedCoordinates() const;
    MTL::SamplerAddressMode   rAddressMode() const;
    MTL::SamplerReductionMode reductionMode() const;
    MTL::SamplerAddressMode   sAddressMode() const;
    void                      setBorderColor(MTL::SamplerBorderColor borderColor);
    void                      setCompareFunction(MTL::CompareFunction compareFunction);
    void                      setLabel(NS::String* label);
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
    MTL::SamplerAddressMode   tAddressMode() const;

};

class SamplerState : public NS::Referencing<SamplerState>
{
public:
    MTL::Device*    device() const;
    MTL::ResourceID gpuResourceID() const;
    NS::String*     label() const;

};

} // namespace MTL

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTLSamplerDescriptor;
extern "C" void *OBJC_CLASS_$_MTLSamplerState;

_MTL_INLINE MTL::SamplerDescriptor* MTL::SamplerDescriptor::alloc()
{
    return _MTL_msg_MTL__SamplerDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTLSamplerDescriptor, nullptr);
}

_MTL_INLINE MTL::SamplerDescriptor* MTL::SamplerDescriptor::init() const
{
    return _MTL_msg_MTL__SamplerDescriptorp_init((const void*)this, nullptr);
}

_MTL_INLINE MTL::SamplerMinMagFilter MTL::SamplerDescriptor::minFilter() const
{
    return _MTL_msg_MTL__SamplerMinMagFilter_minFilter((const void*)this, nullptr);
}

_MTL_INLINE void MTL::SamplerDescriptor::setMinFilter(MTL::SamplerMinMagFilter minFilter)
{
    _MTL_msg_v_setMinFilter__MTL__SamplerMinMagFilter((const void*)this, nullptr, minFilter);
}

_MTL_INLINE MTL::SamplerMinMagFilter MTL::SamplerDescriptor::magFilter() const
{
    return _MTL_msg_MTL__SamplerMinMagFilter_magFilter((const void*)this, nullptr);
}

_MTL_INLINE void MTL::SamplerDescriptor::setMagFilter(MTL::SamplerMinMagFilter magFilter)
{
    _MTL_msg_v_setMagFilter__MTL__SamplerMinMagFilter((const void*)this, nullptr, magFilter);
}

_MTL_INLINE MTL::SamplerMipFilter MTL::SamplerDescriptor::mipFilter() const
{
    return _MTL_msg_MTL__SamplerMipFilter_mipFilter((const void*)this, nullptr);
}

_MTL_INLINE void MTL::SamplerDescriptor::setMipFilter(MTL::SamplerMipFilter mipFilter)
{
    _MTL_msg_v_setMipFilter__MTL__SamplerMipFilter((const void*)this, nullptr, mipFilter);
}

_MTL_INLINE NS::UInteger MTL::SamplerDescriptor::maxAnisotropy() const
{
    return _MTL_msg_NS__UInteger_maxAnisotropy((const void*)this, nullptr);
}

_MTL_INLINE void MTL::SamplerDescriptor::setMaxAnisotropy(NS::UInteger maxAnisotropy)
{
    _MTL_msg_v_setMaxAnisotropy__NS__UInteger((const void*)this, nullptr, maxAnisotropy);
}

_MTL_INLINE MTL::SamplerAddressMode MTL::SamplerDescriptor::sAddressMode() const
{
    return _MTL_msg_MTL__SamplerAddressMode_sAddressMode((const void*)this, nullptr);
}

_MTL_INLINE void MTL::SamplerDescriptor::setSAddressMode(MTL::SamplerAddressMode sAddressMode)
{
    _MTL_msg_v_setSAddressMode__MTL__SamplerAddressMode((const void*)this, nullptr, sAddressMode);
}

_MTL_INLINE MTL::SamplerAddressMode MTL::SamplerDescriptor::tAddressMode() const
{
    return _MTL_msg_MTL__SamplerAddressMode_tAddressMode((const void*)this, nullptr);
}

_MTL_INLINE void MTL::SamplerDescriptor::setTAddressMode(MTL::SamplerAddressMode tAddressMode)
{
    _MTL_msg_v_setTAddressMode__MTL__SamplerAddressMode((const void*)this, nullptr, tAddressMode);
}

_MTL_INLINE MTL::SamplerAddressMode MTL::SamplerDescriptor::rAddressMode() const
{
    return _MTL_msg_MTL__SamplerAddressMode_rAddressMode((const void*)this, nullptr);
}

_MTL_INLINE void MTL::SamplerDescriptor::setRAddressMode(MTL::SamplerAddressMode rAddressMode)
{
    _MTL_msg_v_setRAddressMode__MTL__SamplerAddressMode((const void*)this, nullptr, rAddressMode);
}

_MTL_INLINE MTL::SamplerBorderColor MTL::SamplerDescriptor::borderColor() const
{
    return _MTL_msg_MTL__SamplerBorderColor_borderColor((const void*)this, nullptr);
}

_MTL_INLINE void MTL::SamplerDescriptor::setBorderColor(MTL::SamplerBorderColor borderColor)
{
    _MTL_msg_v_setBorderColor__MTL__SamplerBorderColor((const void*)this, nullptr, borderColor);
}

_MTL_INLINE MTL::SamplerReductionMode MTL::SamplerDescriptor::reductionMode() const
{
    return _MTL_msg_MTL__SamplerReductionMode_reductionMode((const void*)this, nullptr);
}

_MTL_INLINE void MTL::SamplerDescriptor::setReductionMode(MTL::SamplerReductionMode reductionMode)
{
    _MTL_msg_v_setReductionMode__MTL__SamplerReductionMode((const void*)this, nullptr, reductionMode);
}

_MTL_INLINE bool MTL::SamplerDescriptor::normalizedCoordinates() const
{
    return _MTL_msg_bool_normalizedCoordinates((const void*)this, nullptr);
}

_MTL_INLINE void MTL::SamplerDescriptor::setNormalizedCoordinates(bool normalizedCoordinates)
{
    _MTL_msg_v_setNormalizedCoordinates__bool((const void*)this, nullptr, normalizedCoordinates);
}

_MTL_INLINE float MTL::SamplerDescriptor::lodMinClamp() const
{
    return _MTL_msg_float_lodMinClamp((const void*)this, nullptr);
}

_MTL_INLINE void MTL::SamplerDescriptor::setLodMinClamp(float lodMinClamp)
{
    _MTL_msg_v_setLodMinClamp__float((const void*)this, nullptr, lodMinClamp);
}

_MTL_INLINE float MTL::SamplerDescriptor::lodMaxClamp() const
{
    return _MTL_msg_float_lodMaxClamp((const void*)this, nullptr);
}

_MTL_INLINE void MTL::SamplerDescriptor::setLodMaxClamp(float lodMaxClamp)
{
    _MTL_msg_v_setLodMaxClamp__float((const void*)this, nullptr, lodMaxClamp);
}

_MTL_INLINE bool MTL::SamplerDescriptor::lodAverage() const
{
    return _MTL_msg_bool_lodAverage((const void*)this, nullptr);
}

_MTL_INLINE void MTL::SamplerDescriptor::setLodAverage(bool lodAverage)
{
    _MTL_msg_v_setLodAverage__bool((const void*)this, nullptr, lodAverage);
}

_MTL_INLINE float MTL::SamplerDescriptor::lodBias() const
{
    return _MTL_msg_float_lodBias((const void*)this, nullptr);
}

_MTL_INLINE void MTL::SamplerDescriptor::setLodBias(float lodBias)
{
    _MTL_msg_v_setLodBias__float((const void*)this, nullptr, lodBias);
}

_MTL_INLINE MTL::CompareFunction MTL::SamplerDescriptor::compareFunction() const
{
    return _MTL_msg_MTL__CompareFunction_compareFunction((const void*)this, nullptr);
}

_MTL_INLINE void MTL::SamplerDescriptor::setCompareFunction(MTL::CompareFunction compareFunction)
{
    _MTL_msg_v_setCompareFunction__MTL__CompareFunction((const void*)this, nullptr, compareFunction);
}

_MTL_INLINE bool MTL::SamplerDescriptor::supportArgumentBuffers() const
{
    return _MTL_msg_bool_supportArgumentBuffers((const void*)this, nullptr);
}

_MTL_INLINE void MTL::SamplerDescriptor::setSupportArgumentBuffers(bool supportArgumentBuffers)
{
    _MTL_msg_v_setSupportArgumentBuffers__bool((const void*)this, nullptr, supportArgumentBuffers);
}

_MTL_INLINE NS::String* MTL::SamplerDescriptor::label() const
{
    return _MTL_msg_NS__Stringp_label((const void*)this, nullptr);
}

_MTL_INLINE void MTL::SamplerDescriptor::setLabel(NS::String* label)
{
    _MTL_msg_v_setLabel__NS__Stringp((const void*)this, nullptr, label);
}

_MTL_INLINE NS::String* MTL::SamplerState::label() const
{
    return _MTL_msg_NS__Stringp_label((const void*)this, nullptr);
}

_MTL_INLINE MTL::Device* MTL::SamplerState::device() const
{
    return _MTL_msg_MTL__Devicep_device((const void*)this, nullptr);
}

_MTL_INLINE MTL::ResourceID MTL::SamplerState::gpuResourceID() const
{
    return _MTL_msg_MTL__ResourceID_gpuResourceID((const void*)this, nullptr);
}
