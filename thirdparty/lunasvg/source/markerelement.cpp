#include "markerelement.h"
#include "parser.h"
#include "layoutcontext.h"

using namespace lunasvg;

MarkerElement::MarkerElement()
    : StyledElement(ElementId::Marker)
{
}

Length MarkerElement::refX() const
{
    auto& value = get(PropertyId::RefX);
    if(value.empty())
        return Length{};

    return Parser::parseLength(value, AllowNegativeLengths);
}

Length MarkerElement::refY() const
{
    auto& value = get(PropertyId::RefY);
    if(value.empty())
        return Length{};

    return Parser::parseLength(value, AllowNegativeLengths);
}

Length MarkerElement::markerWidth() const
{
    auto& value = get(PropertyId::MarkerWidth);
    if(value.empty())
        return Length{3, LengthUnits::Number};

    return Parser::parseLength(value, ForbidNegativeLengths);
}

Length MarkerElement::markerHeight() const
{
    auto& value = get(PropertyId::MarkerHeight);
    if(value.empty())
        return Length{3, LengthUnits::Number};

    return Parser::parseLength(value, ForbidNegativeLengths);
}

Angle MarkerElement::orient() const
{
    auto& value = get(PropertyId::Orient);
    if(value.empty())
        return Angle{};

    return Parser::parseAngle(value);
}

MarkerUnits MarkerElement::markerUnits() const
{
    auto& value = get(PropertyId::MarkerUnits);
    if(value.empty())
        return MarkerUnits::StrokeWidth;

    return Parser::parseMarkerUnits(value);
}

Rect MarkerElement::viewBox() const
{
    auto& value = get(PropertyId::ViewBox);
    if(value.empty())
        return Rect{};

    return Parser::parseViewBox(value);
}

PreserveAspectRatio MarkerElement::preserveAspectRatio() const
{
    auto& value = get(PropertyId::PreserveAspectRatio);
    if(value.empty())
        return PreserveAspectRatio{};

    return Parser::parsePreserveAspectRatio(value);
}

std::unique_ptr<LayoutMarker> MarkerElement::getMarker(LayoutContext* context) const
{
    LengthContext lengthContext(this);
    auto _refX = lengthContext.valueForLength(refX(), LengthMode::Width);
    auto _refY = lengthContext.valueForLength(refY(), LengthMode::Height);

    Rect viewPort;
    viewPort.w = lengthContext.valueForLength(markerWidth(), LengthMode::Width);
    viewPort.h = lengthContext.valueForLength(markerHeight(), LengthMode::Height);

    auto preserveAspectRatio = this->preserveAspectRatio();
    auto viewTransform = preserveAspectRatio.getMatrix(viewPort, viewBox());
    viewTransform.map(_refX, _refY, &_refX, &_refY);

    auto marker = std::make_unique<LayoutMarker>();
    marker->refX = _refX;
    marker->refY = _refY;
    marker->transform = viewTransform;
    marker->orient = orient();
    marker->units = markerUnits();
    marker->opacity = opacity();
    marker->masker = context->getMasker(mask());
    marker->clipper = context->getClipper(clip_path());
    layoutChildren(context, marker.get());
    return marker;
}

std::unique_ptr<Node> MarkerElement::clone() const
{
    return cloneElement<MarkerElement>();
}
