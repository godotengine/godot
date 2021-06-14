#include "paintelement.h"
#include "stopelement.h"
#include "parser.h"
#include "layoutcontext.h"

#include <set>

using namespace lunasvg;

PaintElement::PaintElement(ElementId id)
    : StyledElement(id)
{
}

GradientElement::GradientElement(ElementId id)
    : PaintElement(id)
{
}

Transform GradientElement::gradientTransform() const
{
    auto& value = get(PropertyId::GradientTransform);
    if(value.empty())
        return Transform{};

    return Parser::parseTransform(value);
}

SpreadMethod GradientElement::spreadMethod() const
{
    auto& value = get(PropertyId::SpreadMethod);
    if(value.empty())
        return SpreadMethod::Pad;

    return Parser::parseSpreadMethod(value);
}

Units GradientElement::gradientUnits() const
{
    auto& value = get(PropertyId::GradientUnits);
    if(value.empty())
        return Units::ObjectBoundingBox;

    return Parser::parseUnits(value);
}

std::string GradientElement::href() const
{
    auto& value = get(PropertyId::Href);
    if(value.empty())
        return std::string{};

    return Parser::parseHref(value);
}

GradientStops GradientElement::buildGradientStops() const
{
    GradientStops gradientStops;
    double prevOffset = 0.0;
    for(auto& child : children)
    {
        auto element = static_cast<Element*>(child.get());
        if(child->isText() || element->id != ElementId::Stop)
            continue;
        auto stop = static_cast<StopElement*>(element);
        auto offset = std::min(std::max(prevOffset, stop->offset()), 1.0);
        prevOffset = offset;
        gradientStops.emplace_back(offset, stop->stopColorWithOpacity());
    }

    return gradientStops;
}

LinearGradientElement::LinearGradientElement()
    : GradientElement(ElementId::LinearGradient)
{
}

Length LinearGradientElement::x1() const
{
    auto& value = get(PropertyId::X1);
    if(value.empty())
        return Length{};

    return Parser::parseLength(value, AllowNegativeLengths);
}

Length LinearGradientElement::y1() const
{
    auto& value = get(PropertyId::Y1);
    if(value.empty())
        return Length{};

    return Parser::parseLength(value, AllowNegativeLengths);
}

Length LinearGradientElement::x2() const
{
    auto& value = get(PropertyId::X2);
    if(value.empty())
        return Length{100, LengthUnits::Percent};

    return Parser::parseLength(value, AllowNegativeLengths);
}

Length LinearGradientElement::y2() const
{
    auto& value = get(PropertyId::Y2);
    if(value.empty())
        return Length{};

    return Parser::parseLength(value, AllowNegativeLengths);
}

std::unique_ptr<LayoutPaint> LinearGradientElement::getPainter(LayoutContext* context) const
{
    LinearGradientAttributes attributes;
    std::set<const GradientElement*> processedGradients;
    const GradientElement* current = this;

    while(true)
    {
        if(!attributes.hasGradientTransform() && current->has(PropertyId::GradientTransform))
            attributes.setGradientTransform(current->gradientTransform());
        if(!attributes.hasSpreadMethod() && current->has(PropertyId::SpreadMethod))
            attributes.setSpreadMethod(current->spreadMethod());
        if(!attributes.hasGradientUnits() && current->has(PropertyId::GradientUnits))
            attributes.setGradientUnits(current->gradientUnits());
        if(!attributes.hasGradientStops())
            attributes.setGradientStops(current->buildGradientStops());

        if(current->id == ElementId::LinearGradient)
        {
            auto element = static_cast<const LinearGradientElement*>(current);
            if(!attributes.hasX1() && element->has(PropertyId::X1))
                attributes.setX1(element->x1());
            if(!attributes.hasY1() && element->has(PropertyId::Y1))
                attributes.setY1(element->y1());
            if(!attributes.hasX2() && element->has(PropertyId::X2))
                attributes.setX2(element->x2());
            if(!attributes.hasY2() && element->has(PropertyId::Y2))
                attributes.setY2(element->y2());
        }

        auto ref = context->getElementById(current->href());
        if(!ref || !(ref->id == ElementId::LinearGradient || ref->id == ElementId::RadialGradient))
            break;

        processedGradients.insert(current);
        current = static_cast<const GradientElement*>(ref);
        if(processedGradients.find(current) != processedGradients.end())
            break;
    }

    auto& stops = attributes.gradientStops();
    if(stops.empty())
        return nullptr;

    LengthContext lengthContext(this, attributes.gradientUnits());
    auto x1 = lengthContext.valueForLength(attributes.x1(), LengthMode::Width);
    auto y1 = lengthContext.valueForLength(attributes.y1(), LengthMode::Height);
    auto x2 = lengthContext.valueForLength(attributes.x2(), LengthMode::Width);
    auto y2 = lengthContext.valueForLength(attributes.y2(), LengthMode::Height);
    if((x1 == x2 && y1 == y2) || stops.size() == 1)
    {
        auto solid = std::make_unique<LayoutSolidColor>();
        solid->color = stops.back().second;
        return std::move(solid);
    }

    auto gradient = std::make_unique<LayoutLinearGradient>();
    gradient->transform = attributes.gradientTransform();
    gradient->spreadMethod = attributes.spreadMethod();
    gradient->units = attributes.gradientUnits();
    gradient->stops = attributes.gradientStops();

    gradient->x1 = x1;
    gradient->y1 = y1;
    gradient->x2 = x2;
    gradient->y2 = y2;

    return std::move(gradient);
}

std::unique_ptr<Node> LinearGradientElement::clone() const
{
    return cloneElement<LinearGradientElement>();
}

RadialGradientElement::RadialGradientElement()
    : GradientElement(ElementId::RadialGradient)
{
}

Length RadialGradientElement::cx() const
{
    auto& value = get(PropertyId::Cx);
    if(value.empty())
        return Length{50, LengthUnits::Percent};

    return Parser::parseLength(value, AllowNegativeLengths);
}

Length RadialGradientElement::cy() const
{
    auto& value = get(PropertyId::Cy);
    if(value.empty())
        return Length{50, LengthUnits::Percent};

    return Parser::parseLength(value, AllowNegativeLengths);
}

Length RadialGradientElement::r() const
{
    auto& value = get(PropertyId::R);
    if(value.empty())
        return Length{50, LengthUnits::Percent};

    return Parser::parseLength(value, ForbidNegativeLengths);
}

Length RadialGradientElement::fx() const
{
    auto& value = get(PropertyId::Fx);
    if(value.empty())
        return Length{};

    return Parser::parseLength(value, AllowNegativeLengths);
}

Length RadialGradientElement::fy() const
{
    auto& value = get(PropertyId::Fy);
    if(value.empty())
        return Length{};

    return Parser::parseLength(value, AllowNegativeLengths);
}

std::unique_ptr<LayoutPaint> RadialGradientElement::getPainter(LayoutContext *context) const
{
    RadialGradientAttributes attributes;
    std::set<const GradientElement*> processedGradients;
    const GradientElement* current = this;

    while(true)
    {
        if(!attributes.hasGradientTransform() && current->has(PropertyId::GradientTransform))
            attributes.setGradientTransform(current->gradientTransform());
        if(!attributes.hasSpreadMethod() && current->has(PropertyId::SpreadMethod))
            attributes.setSpreadMethod(current->spreadMethod());
        if(!attributes.hasGradientUnits() && current->has(PropertyId::GradientUnits))
            attributes.setGradientUnits(current->gradientUnits());
        if(!attributes.hasGradientStops())
            attributes.setGradientStops(current->buildGradientStops());

        if(current->id == ElementId::RadialGradient)
        {
            auto element = static_cast<const RadialGradientElement*>(current);
            if(!attributes.hasCx() && element->has(PropertyId::Cx))
                attributes.setCx(element->cx());
            if(!attributes.hasCy() && element->has(PropertyId::Cy))
                attributes.setCy(element->cy());
            if(!attributes.hasR() && element->has(PropertyId::R))
                attributes.setR(element->r());
            if(!attributes.hasFx() && element->has(PropertyId::Fx))
                attributes.setFx(element->fx());
            if(!attributes.hasFy() && element->has(PropertyId::Fy))
                attributes.setFy(element->fy());
        }

        auto ref = context->getElementById(current->href());
        if(!ref || !(ref->id == ElementId::LinearGradient || ref->id == ElementId::RadialGradient))
            break;

        processedGradients.insert(current);
        current = static_cast<const GradientElement*>(ref);
        if(processedGradients.find(current) != processedGradients.end())
            break;
    }

    if(!attributes.hasFx())
        attributes.setFx(attributes.cx());
    if(!attributes.hasFy())
        attributes.setFy(attributes.cy());

    auto& stops = attributes.gradientStops();
    if(stops.empty())
        return nullptr;

    auto& r = attributes.r();
    if(r.isZero() || stops.size() == 1)
    {
        auto solid = std::make_unique<LayoutSolidColor>();
        solid->color = stops.back().second;
        return std::move(solid);
    }

    auto gradient = std::make_unique<LayoutRadialGradient>();
    gradient->transform = attributes.gradientTransform();
    gradient->spreadMethod = attributes.spreadMethod();
    gradient->units = attributes.gradientUnits();
    gradient->stops = attributes.gradientStops();

    LengthContext lengthContext(this, attributes.gradientUnits());
    gradient->cx = lengthContext.valueForLength(attributes.cx(), LengthMode::Width);
    gradient->cy = lengthContext.valueForLength(attributes.cy(), LengthMode::Height);
    gradient->r = lengthContext.valueForLength(attributes.r(), LengthMode::Both);
    gradient->fx = lengthContext.valueForLength(attributes.fx(), LengthMode::Width);
    gradient->fy = lengthContext.valueForLength(attributes.fy(), LengthMode::Height);

    return std::move(gradient);
}

std::unique_ptr<Node> RadialGradientElement::clone() const
{
    return cloneElement<RadialGradientElement>();
}

PatternElement::PatternElement()
    : PaintElement(ElementId::Pattern)
{
}

Length PatternElement::x() const
{
    auto& value = get(PropertyId::X);
    if(value.empty())
        return Length{};

    return Parser::parseLength(value, AllowNegativeLengths);
}

Length PatternElement::y() const
{
    auto& value = get(PropertyId::Y);
    if(value.empty())
        return Length{};

    return Parser::parseLength(value, AllowNegativeLengths);
}

Length PatternElement::width() const
{
    auto& value = get(PropertyId::Width);
    if(value.empty())
        return Length{};

    return Parser::parseLength(value, ForbidNegativeLengths);
}

Length PatternElement::height() const
{
    auto& value = get(PropertyId::Height);
    if(value.empty())
        return Length{};

    return Parser::parseLength(value, ForbidNegativeLengths);
}

Transform PatternElement::patternTransform() const
{
    auto& value = get(PropertyId::PatternTransform);
    if(value.empty())
        return Transform{};

    return Parser::parseTransform(value);
}

Units PatternElement::patternUnits() const
{
    auto& value = get(PropertyId::PatternUnits);
    if(value.empty())
        return Units::ObjectBoundingBox;

    return Parser::parseUnits(value);
}

Units PatternElement::patternContentUnits() const
{
    auto& value = get(PropertyId::PatternContentUnits);
    if(value.empty())
        return Units::UserSpaceOnUse;

    return Parser::parseUnits(value);
}

Rect PatternElement::viewBox() const
{
    auto& value = get(PropertyId::ViewBox);
    if(value.empty())
        return Rect{};

    return Parser::parseViewBox(value);
}

PreserveAspectRatio PatternElement::preserveAspectRatio() const
{
    auto& value = get(PropertyId::PreserveAspectRatio);
    if(value.empty())
        return PreserveAspectRatio{};

    return Parser::parsePreserveAspectRatio(value);
}

std::string PatternElement::href() const
{
    auto& value = get(PropertyId::Href);
    if(value.empty())
        return std::string{};

    return Parser::parseHref(value);
}

std::unique_ptr<LayoutPaint> PatternElement::getPainter(LayoutContext* context) const
{
    PatternAttributes attributes;
    std::set<const PatternElement*> processedPatterns;
    const PatternElement* current = this;

    while(true)
    {
        if(!attributes.hasX() && current->has(PropertyId::X))
            attributes.setX(current->x());
        if(!attributes.hasY() && current->has(PropertyId::Y))
            attributes.setY(current->y());
        if(!attributes.hasWidth() && current->has(PropertyId::Width))
            attributes.setWidth(current->width());
        if(!attributes.hasHeight() && current->has(PropertyId::Height))
            attributes.setHeight(current->height());
        if(!attributes.hasPatternTransform() && current->has(PropertyId::PatternTransform))
            attributes.setPatternTransform(current->patternTransform());
        if(!attributes.hasPatternUnits() && current->has(PropertyId::PatternUnits))
            attributes.setPatternUnits(current->patternUnits());
        if(!attributes.hasPatternContentUnits() && current->has(PropertyId::PatternContentUnits))
            attributes.setPatternContentUnits(current->patternContentUnits());
        if(!attributes.hasViewBox() && current->has(PropertyId::ViewBox))
            attributes.setViewBox(current->viewBox());
        if(!attributes.hasPreserveAspectRatio() && current->has(PropertyId::PreserveAspectRatio))
            attributes.setPreserveAspectRatio(current->preserveAspectRatio());
        if(!attributes.hasPatternContentElement() && current->children.size())
            attributes.setPatternContentElement(current);

        auto ref = context->getElementById(current->href());
        if(!ref || ref->id != ElementId::Pattern)
            break;

        processedPatterns.insert(current);
        current = static_cast<const PatternElement*>(ref);
        if(processedPatterns.find(current) != processedPatterns.end())
            break;
    }

    auto& width = attributes.width();
    auto& height = attributes.height();
    auto element = attributes.patternContentElement();
    if(element == nullptr || width.isZero() || height.isZero())
        return nullptr;

    auto pattern = std::make_unique<LayoutPattern>();
    pattern->transform = attributes.patternTransform();
    pattern->units = attributes.patternUnits();
    pattern->contentUnits = attributes.patternContentUnits();
    pattern->viewBox = attributes.viewBox();
    pattern->preserveAspectRatio = attributes.preserveAspectRatio();

    LengthContext lengthContext(this, attributes.patternUnits());
    pattern->x = lengthContext.valueForLength(attributes.x(), LengthMode::Width);
    pattern->y = lengthContext.valueForLength(attributes.y(), LengthMode::Height);
    pattern->width = lengthContext.valueForLength(attributes.width(), LengthMode::Width);
    pattern->height = lengthContext.valueForLength(attributes.height(), LengthMode::Height);
    element->layoutChildren(context, pattern.get());

    return std::move(pattern);
}

std::unique_ptr<Node> PatternElement::clone() const
{
    return cloneElement<PatternElement>();
}

SolidColorElement::SolidColorElement()
    : PaintElement(ElementId::SolidColor)
{
}

std::unique_ptr<LayoutPaint> SolidColorElement::getPainter(LayoutContext*) const
{
    auto solid = std::make_unique<LayoutSolidColor>();
    solid->color = solid_color();
    solid->color.a = solid_opacity();
    return std::move(solid);
}

std::unique_ptr<Node> SolidColorElement::clone() const
{
    return cloneElement<SolidColorElement>();
}
