#pragma once

#include "CADefines.hpp"
#include "CAPrivate.hpp"
#include "../Foundation/NSObject.hpp"
#include <CoreGraphics/CoreGraphics.h>

namespace CA
{
using DynamicRange = NS::String*;

_CA_CONST(DynamicRange, DynamicRangeAutomatic);
_CA_CONST(DynamicRange, DynamicRangeStandard);
_CA_CONST(DynamicRange, DynamicRangeConstrainedHigh);
_CA_CONST(DynamicRange, DynamicRangeHigh);


class Layer : public NS::Referencing<Layer>
{
public:
    bool wantsExtendedDynamicRangeContent() const;
    void setWantsExtendedDynamicRangeContent(bool wantsExtendedDynamicRangeContent);
    CA::DynamicRange preferredDynamicRange() const;
    void setPreferredDynamicRange(CA::DynamicRange preferredDynamicRange);
    CGFloat contentsHeadroom() const;
    void setContentsHeadroom(CGFloat contentsHeadroom);
    bool opaque() const;
    void setOpaque(bool opaque);

};

} // namespace CA

// --- Inline implementations ---

_CA_INLINE bool CA::Layer::wantsExtendedDynamicRangeContent() const
{
    return Object::sendMessage<bool>(this, _CA_PRIVATE_SEL(wantsExtendedDynamicRangeContent));
}

_CA_INLINE void CA::Layer::setWantsExtendedDynamicRangeContent(bool wantsExtendedDynamicRangeContent)
{
    Object::sendMessage<void>(this, _CA_PRIVATE_SEL(setWantsExtendedDynamicRangeContent_), wantsExtendedDynamicRangeContent);
}

_CA_INLINE CA::DynamicRange CA::Layer::preferredDynamicRange() const
{
    return Object::sendMessage<CA::DynamicRange>(this, _CA_PRIVATE_SEL(preferredDynamicRange));
}

_CA_INLINE void CA::Layer::setPreferredDynamicRange(CA::DynamicRange preferredDynamicRange)
{
    Object::sendMessage<void>(this, _CA_PRIVATE_SEL(setPreferredDynamicRange_), preferredDynamicRange);
}

_CA_INLINE CGFloat CA::Layer::contentsHeadroom() const
{
    return Object::sendMessage<CGFloat>(this, _CA_PRIVATE_SEL(contentsHeadroom));
}

_CA_INLINE void CA::Layer::setContentsHeadroom(CGFloat contentsHeadroom)
{
    Object::sendMessage<void>(this, _CA_PRIVATE_SEL(setContentsHeadroom_), contentsHeadroom);
}

_CA_INLINE bool CA::Layer::opaque() const
{
    return Object::sendMessage<bool>(this, _CA_PRIVATE_SEL(opaque));
}

_CA_INLINE void CA::Layer::setOpaque(bool opaque)
{
    Object::sendMessage<void>(this, _CA_PRIVATE_SEL(setOpaque_), opaque);
}

_CA_PRIVATE_DEF_CONST(CA::DynamicRange, DynamicRangeAutomatic);
_CA_PRIVATE_DEF_CONST(CA::DynamicRange, DynamicRangeStandard);
_CA_PRIVATE_DEF_CONST(CA::DynamicRange, DynamicRangeConstrainedHigh);
_CA_PRIVATE_DEF_CONST(CA::DynamicRange, DynamicRangeHigh);
