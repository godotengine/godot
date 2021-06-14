#include "maskelement.h"
#include "parser.h"
#include "layoutcontext.h"

using namespace lunasvg;

MaskElement::MaskElement()
    : StyledElement(ElementId::Mask)
{
}

Length MaskElement::x() const
{
    auto& value = get(PropertyId::X);
    if(value.empty())
        return Length{-10, LengthUnits::Percent};

    return Parser::parseLength(value, AllowNegativeLengths);
}

Length MaskElement::y() const
{
    auto& value = get(PropertyId::Y);
    if(value.empty())
        return Length{-10, LengthUnits::Percent};

    return Parser::parseLength(value, AllowNegativeLengths);
}

Length MaskElement::width() const
{
    auto& value = get(PropertyId::Width);
    if(value.empty())
        return Length{20, LengthUnits::Percent};

    return Parser::parseLength(value, ForbidNegativeLengths);
}

Length MaskElement::height() const
{
    auto& value = get(PropertyId::Height);
    if(value.empty())
        return Length{20, LengthUnits::Percent};

    return Parser::parseLength(value, ForbidNegativeLengths);
}

Units MaskElement::maskUnits() const
{
    auto& value = get(PropertyId::MaskUnits);
    if(value.empty())
        return Units::ObjectBoundingBox;

    return Parser::parseUnits(value);
}

Units MaskElement::maskContentUnits() const
{
    auto& value = get(PropertyId::MaskContentUnits);
    if(value.empty())
        return Units::UserSpaceOnUse;

    return Parser::parseUnits(value);
}

std::unique_ptr<LayoutMask> MaskElement::getMasker(LayoutContext* context) const
{
    auto masker = std::make_unique<LayoutMask>();
    masker->units = maskUnits();
    masker->contentUnits = maskContentUnits();
    masker->opacity = opacity();
    masker->masker = context->getMasker(mask());
    masker->clipper = context->getClipper(clip_path());

    LengthContext lengthContext(this, maskUnits());
    masker->x = lengthContext.valueForLength(x(), LengthMode::Width);
    masker->y = lengthContext.valueForLength(y(), LengthMode::Height);
    masker->width = lengthContext.valueForLength(width(), LengthMode::Width);
    masker->height = lengthContext.valueForLength(height(), LengthMode::Height);
    layoutChildren(context, masker.get());
    return masker;
}

std::unique_ptr<Node> MaskElement::clone() const
{
    return cloneElement<MaskElement>();
}
