#pragma once

#include "CADefines.hpp"
#include "CAStructs.hpp"
#include "CABridge.hpp"
#include "../Foundation/NSObject.hpp"
#include "../Foundation/NSTypes.hpp"
#include "../Foundation/NSRange.hpp"
#include <CoreGraphics/CoreGraphics.h>

namespace CA
{

using LayerContentsGravity = NS::String*;
using LayerContentsFormat = NS::String*;
using LayerContentsFilter = NS::String*;
using LayerCornerCurve = NS::String*;
using ToneMapMode = NS::String*;
using DynamicRange = NS::String*;
extern ToneMapMode const ToneMapModeAutomatic __asm__("_CAToneMapModeAutomatic");
extern ToneMapMode const ToneMapModeNever __asm__("_CAToneMapModeNever");
extern ToneMapMode const ToneMapModeIfSupported __asm__("_CAToneMapModeIfSupported");
extern DynamicRange const DynamicRangeAutomatic __asm__("_CADynamicRangeAutomatic");
extern DynamicRange const DynamicRangeStandard __asm__("_CADynamicRangeStandard");
extern DynamicRange const DynamicRangeConstrainedHigh __asm__("_CADynamicRangeConstrainedHigh");
extern DynamicRange const DynamicRangeHigh __asm__("_CADynamicRangeHigh");
_CA_OPTIONS(unsigned int, AutoresizingMask) {
    kCALayerNotSizable = 0,
    kCALayerMinXMargin = 1U << 0,
    kCALayerWidthSizable = 1U << 1,
    kCALayerMaxXMargin = 1U << 2,
    kCALayerMinYMargin = 1U << 3,
    kCALayerHeightSizable = 1U << 4,
    kCALayerMaxYMargin = 1U << 5,
};

_CA_OPTIONS(unsigned int, EdgeAntialiasingMask) {
    kCALayerLeftEdge = 1U << 0,
    kCALayerRightEdge = 1U << 1,
    kCALayerBottomEdge = 1U << 2,
    kCALayerTopEdge = 1U << 3,
};

_CA_OPTIONS(NS::UInteger, CornerMask) {
    kCALayerMinXMinYCorner = 1U << 0,
    kCALayerMaxXMinYCorner = 1U << 1,
    kCALayerMinXMaxYCorner = 1U << 2,
    kCALayerMaxXMaxYCorner = 1U << 3,
};


class Layer : public NS::SecureCoding<Layer>
{
public:
    static Layer* alloc();
    Layer*        init() const;

    CGFloat          contentsHeadroom() const;
    bool             opaque() const;
    CA::DynamicRange preferredDynamicRange() const;
    void             setContentsHeadroom(CGFloat contentsHeadroom);
    void             setOpaque(bool opaque);
    void             setPreferredDynamicRange(CA::DynamicRange preferredDynamicRange);
    void             setWantsExtendedDynamicRangeContent(bool wantsExtendedDynamicRangeContent);
    bool             wantsExtendedDynamicRangeContent() const;

};

} // namespace CA

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_CALayer;

_CA_INLINE CA::Layer* CA::Layer::alloc()
{
    return _CA_msg_CA__Layerp_alloc((const void*)&OBJC_CLASS_$_CALayer, nullptr);
}

_CA_INLINE CA::Layer* CA::Layer::init() const
{
    return _CA_msg_CA__Layerp_init((const void*)this, nullptr);
}

_CA_INLINE bool CA::Layer::wantsExtendedDynamicRangeContent() const
{
    return _CA_msg_bool_wantsExtendedDynamicRangeContent((const void*)this, nullptr);
}

_CA_INLINE void CA::Layer::setWantsExtendedDynamicRangeContent(bool wantsExtendedDynamicRangeContent)
{
    _CA_msg_v_setWantsExtendedDynamicRangeContent__bool((const void*)this, nullptr, wantsExtendedDynamicRangeContent);
}

_CA_INLINE CA::DynamicRange CA::Layer::preferredDynamicRange() const
{
    return _CA_msg_NS__Stringp_preferredDynamicRange((const void*)this, nullptr);
}

_CA_INLINE void CA::Layer::setPreferredDynamicRange(CA::DynamicRange preferredDynamicRange)
{
    _CA_msg_v_setPreferredDynamicRange__NS__Stringp((const void*)this, nullptr, preferredDynamicRange);
}

_CA_INLINE CGFloat CA::Layer::contentsHeadroom() const
{
    return _CA_msg_CGFloat_contentsHeadroom((const void*)this, nullptr);
}

_CA_INLINE void CA::Layer::setContentsHeadroom(CGFloat contentsHeadroom)
{
    _CA_msg_v_setContentsHeadroom__CGFloat((const void*)this, nullptr, contentsHeadroom);
}

_CA_INLINE bool CA::Layer::opaque() const
{
    return _CA_msg_bool_opaque((const void*)this, nullptr);
}

_CA_INLINE void CA::Layer::setOpaque(bool opaque)
{
    _CA_msg_v_setOpaque__bool((const void*)this, nullptr, opaque);
}
