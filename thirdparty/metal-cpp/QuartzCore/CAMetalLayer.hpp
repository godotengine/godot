#pragma once

#include "CADefines.hpp"
#include "CAStructs.hpp"
#include "CABridge.hpp"
#include "../Foundation/NSObject.hpp"
#include "../Foundation/NSTypes.hpp"
#include "../Foundation/NSRange.hpp"
#include "../Metal/MTLDrawable.hpp"
#include "CALayer.hpp"
#include <CoreGraphics/CoreGraphics.h>

namespace MTL {
    class Device;
    class ResidencySet;
    class Texture;
    enum PixelFormat : NS::UInteger;
}

namespace CA
{

class MetalDrawable;
class MetalLayer;

class MetalDrawable : public NS::Referencing<MetalDrawable, MTL::Drawable>
{
public:
    CA::MetalLayer* layer() const;
    MTL::Texture*   texture() const;

};

class MetalLayer : public NS::Referencing<MetalLayer, CA::Layer>
{
public:
    static MetalLayer* alloc();
    MetalLayer*        init() const;

    bool               allowsNextDrawableTimeout() const;
    CGColorSpaceRef    colorspace() const;
    MTL::Device*       device() const;
    bool               displaySyncEnabled() const;
    CGSize             drawableSize() const;
    bool               framebufferOnly() const;
    NS::UInteger       maximumDrawableCount() const;
    CA::MetalDrawable* nextDrawable();
    MTL::PixelFormat   pixelFormat() const;
    MTL::ResidencySet* residencySet() const;
    void               setAllowsNextDrawableTimeout(bool allowsNextDrawableTimeout);
    void               setColorspace(CGColorSpaceRef colorspace);
    void               setDevice(MTL::Device* device);
    void               setDisplaySyncEnabled(bool displaySyncEnabled);
    void               setDrawableSize(CGSize drawableSize);
    void               setFramebufferOnly(bool framebufferOnly);
    void               setMaximumDrawableCount(NS::UInteger maximumDrawableCount);
    void               setPixelFormat(MTL::PixelFormat pixelFormat);
    void               setWantsExtendedDynamicRangeContent(bool wantsExtendedDynamicRangeContent);
    bool               wantsExtendedDynamicRangeContent() const;

};

} // namespace CA

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_CAMetalDrawable;
extern "C" void *OBJC_CLASS_$_CAMetalLayer;

_CA_INLINE MTL::Texture* CA::MetalDrawable::texture() const
{
    return _CA_msg_MTL__Texturep_texture((const void*)this, nullptr);
}

_CA_INLINE CA::MetalLayer* CA::MetalDrawable::layer() const
{
    return _CA_msg_CA__MetalLayerp_layer((const void*)this, nullptr);
}

_CA_INLINE CA::MetalLayer* CA::MetalLayer::alloc()
{
    return _CA_msg_CA__MetalLayerp_alloc((const void*)&OBJC_CLASS_$_CAMetalLayer, nullptr);
}

_CA_INLINE CA::MetalLayer* CA::MetalLayer::init() const
{
    return _CA_msg_CA__MetalLayerp_init((const void*)this, nullptr);
}

_CA_INLINE MTL::Device* CA::MetalLayer::device() const
{
    return _CA_msg_MTL__Devicep_device((const void*)this, nullptr);
}

_CA_INLINE void CA::MetalLayer::setDevice(MTL::Device* device)
{
    _CA_msg_v_setDevice__MTL__Devicep((const void*)this, nullptr, device);
}

_CA_INLINE MTL::PixelFormat CA::MetalLayer::pixelFormat() const
{
    return _CA_msg_MTL__PixelFormat_pixelFormat((const void*)this, nullptr);
}

_CA_INLINE void CA::MetalLayer::setPixelFormat(MTL::PixelFormat pixelFormat)
{
    _CA_msg_v_setPixelFormat__MTL__PixelFormat((const void*)this, nullptr, pixelFormat);
}

_CA_INLINE bool CA::MetalLayer::framebufferOnly() const
{
    return _CA_msg_bool_framebufferOnly((const void*)this, nullptr);
}

_CA_INLINE void CA::MetalLayer::setFramebufferOnly(bool framebufferOnly)
{
    _CA_msg_v_setFramebufferOnly__bool((const void*)this, nullptr, framebufferOnly);
}

_CA_INLINE CGSize CA::MetalLayer::drawableSize() const
{
    return _CA_msg_CGSize_drawableSize((const void*)this, nullptr);
}

_CA_INLINE void CA::MetalLayer::setDrawableSize(CGSize drawableSize)
{
    _CA_msg_v_setDrawableSize__CGSize((const void*)this, nullptr, drawableSize);
}

_CA_INLINE NS::UInteger CA::MetalLayer::maximumDrawableCount() const
{
    return _CA_msg_NS__UInteger_maximumDrawableCount((const void*)this, nullptr);
}

_CA_INLINE void CA::MetalLayer::setMaximumDrawableCount(NS::UInteger maximumDrawableCount)
{
    _CA_msg_v_setMaximumDrawableCount__NS__UInteger((const void*)this, nullptr, maximumDrawableCount);
}

_CA_INLINE CGColorSpaceRef CA::MetalLayer::colorspace() const
{
    return _CA_msg_CGColorSpaceRef_colorspace((const void*)this, nullptr);
}

_CA_INLINE void CA::MetalLayer::setColorspace(CGColorSpaceRef colorspace)
{
    _CA_msg_v_setColorspace__CGColorSpaceRef((const void*)this, nullptr, colorspace);
}

_CA_INLINE bool CA::MetalLayer::wantsExtendedDynamicRangeContent() const
{
    return _CA_msg_bool_wantsExtendedDynamicRangeContent((const void*)this, nullptr);
}

_CA_INLINE void CA::MetalLayer::setWantsExtendedDynamicRangeContent(bool wantsExtendedDynamicRangeContent)
{
    _CA_msg_v_setWantsExtendedDynamicRangeContent__bool((const void*)this, nullptr, wantsExtendedDynamicRangeContent);
}

_CA_INLINE bool CA::MetalLayer::displaySyncEnabled() const
{
    return _CA_msg_bool_displaySyncEnabled((const void*)this, nullptr);
}

_CA_INLINE void CA::MetalLayer::setDisplaySyncEnabled(bool displaySyncEnabled)
{
    _CA_msg_v_setDisplaySyncEnabled__bool((const void*)this, nullptr, displaySyncEnabled);
}

_CA_INLINE bool CA::MetalLayer::allowsNextDrawableTimeout() const
{
    return _CA_msg_bool_allowsNextDrawableTimeout((const void*)this, nullptr);
}

_CA_INLINE void CA::MetalLayer::setAllowsNextDrawableTimeout(bool allowsNextDrawableTimeout)
{
    _CA_msg_v_setAllowsNextDrawableTimeout__bool((const void*)this, nullptr, allowsNextDrawableTimeout);
}

_CA_INLINE MTL::ResidencySet* CA::MetalLayer::residencySet() const
{
    return _CA_msg_MTL__ResidencySetp_residencySet((const void*)this, nullptr);
}

_CA_INLINE CA::MetalDrawable* CA::MetalLayer::nextDrawable()
{
    return _CA_msg_CA__MetalDrawablep_nextDrawable((const void*)this, nullptr);
}
