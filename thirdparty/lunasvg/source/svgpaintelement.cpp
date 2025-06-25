#include "svgpaintelement.h"
#include "svglayoutstate.h"
#include "svgrenderstate.h"

#include <set>

namespace lunasvg {

SVGPaintElement::SVGPaintElement(Document* document, ElementID id)
    : SVGElement(document, id)
{
}

SVGStopElement::SVGStopElement(Document* document)
    : SVGElement(document, ElementID::Stop)
    , m_offset(PropertyID::Offset, 0.f)
{
    addProperty(m_offset);
}

void SVGStopElement::layoutElement(const SVGLayoutState& state)
{
    m_stop_color = state.stop_color();
    m_stop_opacity = state.stop_opacity();
    SVGElement::layoutElement(state);
}

GradientStop SVGStopElement::gradientStop(float opacity) const
{
    Color stopColor = m_stop_color.colorWithAlpha(m_stop_opacity * opacity);
    GradientStop gradientStop = {
        m_offset.value(), { stopColor.redF(), stopColor.greenF(), stopColor.blueF(), stopColor.alphaF() }
    };

    return gradientStop;
}

SVGGradientElement::SVGGradientElement(Document* document, ElementID id)
    : SVGPaintElement(document, id)
    , SVGURIReference(this)
    , m_gradientTransform(PropertyID::GradientTransform)
    , m_gradientUnits(PropertyID::GradientUnits, Units::ObjectBoundingBox)
    , m_spreadMethod(PropertyID::SpreadMethod, SpreadMethod::Pad)
{
    addProperty(m_gradientTransform);
    addProperty(m_gradientUnits);
    addProperty(m_spreadMethod);
}

void SVGGradientElement::collectGradientAttributes(SVGGradientAttributes& attributes) const
{
    if(!attributes.hasGradientTransform() && hasAttribute(PropertyID::GradientTransform))
        attributes.setGradientTransform(this);
    if(!attributes.hasSpreadMethod() && hasAttribute(PropertyID::SpreadMethod))
        attributes.setSpreadMethod(this);
    if(!attributes.hasGradientUnits() && hasAttribute(PropertyID::GradientUnits))
        attributes.setGradientUnits(this);
    if(!attributes.hasGradientContentElement()) {
        for(const auto& child : children()) {
            if(auto element = toSVGElement(child); element && element->id() == ElementID::Stop) {
                attributes.setGradientContentElement(this);
                break;
            }
        }
    }
}

SVGLinearGradientElement::SVGLinearGradientElement(Document* document)
    : SVGGradientElement(document, ElementID::LinearGradient)
    , m_x1(PropertyID::X1, LengthDirection::Horizontal, LengthNegativeMode::Allow, 0.f, LengthUnits::Percent)
    , m_y1(PropertyID::Y1, LengthDirection::Vertical, LengthNegativeMode::Allow, 0.f, LengthUnits::Percent)
    , m_x2(PropertyID::X2, LengthDirection::Horizontal, LengthNegativeMode::Allow, 100.f, LengthUnits::Percent)
    , m_y2(PropertyID::Y2, LengthDirection::Vertical, LengthNegativeMode::Allow, 0.f, LengthUnits::Percent)
{
    addProperty(m_x1);
    addProperty(m_y1);
    addProperty(m_x2);
    addProperty(m_y2);
}

SVGLinearGradientAttributes SVGLinearGradientElement::collectGradientAttributes() const
{
    SVGLinearGradientAttributes attributes;
    std::set<const SVGGradientElement*> processedGradients;
    const SVGGradientElement* current = this;
    while(true) {
        current->collectGradientAttributes(attributes);
        if(current->id() == ElementID::LinearGradient) {
            auto element = static_cast<const SVGLinearGradientElement*>(current);
            if(!attributes.hasX1() && element->hasAttribute(PropertyID::X1))
                attributes.setX1(element);
            if(!attributes.hasY1() && element->hasAttribute(PropertyID::Y1))
                attributes.setY1(element);
            if(!attributes.hasX2() && element->hasAttribute(PropertyID::X2))
                attributes.setX2(element);
            if(!attributes.hasY2() && element->hasAttribute(PropertyID::Y2)) {
                attributes.setY2(element);
            }
        }

        auto targetElement = current->getTargetElement(document());
        if(!targetElement || !(targetElement->id() == ElementID::LinearGradient || targetElement->id() == ElementID::RadialGradient))
            break;
        processedGradients.insert(current);
        current = static_cast<const SVGGradientElement*>(targetElement);
        if(processedGradients.count(current) > 0) {
            break;
        }
    }

    attributes.setDefaultValues(this);
    return attributes;
}

static GradientStops buildGradientStops(const SVGGradientElement* element, float opacity)
{
    GradientStops gradientStops;
    for(const auto& child : element->children()) {
        if(auto element = toSVGElement(child); element && element->id() == ElementID::Stop) {
            auto stopElement = static_cast<SVGStopElement*>(element);
            gradientStops.push_back(stopElement->gradientStop(opacity));
        }
    }

    return gradientStops;
}

bool SVGLinearGradientElement::applyPaint(SVGRenderState& state, float opacity) const
{
    auto attributes = collectGradientAttributes();
    auto gradientContentElement = attributes.gradientContentElement();
    auto gradientStops = buildGradientStops(gradientContentElement, opacity);
    if(gradientStops.empty())
        return false;
    LengthContext lengthContext(this, attributes.gradientUnits());
    auto x1 = lengthContext.valueForLength(attributes.x1());
    auto y1 = lengthContext.valueForLength(attributes.y1());
    auto x2 = lengthContext.valueForLength(attributes.x2());
    auto y2 = lengthContext.valueForLength(attributes.y2());
    if(gradientStops.size() == 1 || (x1 == x2 && y1 == y2)) {
        const auto& lastStop = gradientStops.back();
        state->setColor(lastStop.color.r, lastStop.color.g, lastStop.color.b, lastStop.color.a);
        return true;
    }

    auto spreadMethod = attributes.spreadMethod();
    auto gradientUnits = attributes.gradientUnits();
    auto gradientTransform = attributes.gradientTransform();
    if(gradientUnits == Units::ObjectBoundingBox) {
        auto bbox = state.fillBoundingBox();
        gradientTransform.postMultiply(Transform(bbox.w, 0, 0, bbox.h, bbox.x, bbox.y));
    }

    state->setLinearGradient(x1, y1, x2, y2, spreadMethod, gradientStops, gradientTransform);
    return true;
}

SVGRadialGradientElement::SVGRadialGradientElement(Document* document)
    : SVGGradientElement(document, ElementID::RadialGradient)
    , m_cx(PropertyID::Cx, LengthDirection::Horizontal, LengthNegativeMode::Allow, 50.f, LengthUnits::Percent)
    , m_cy(PropertyID::Cy, LengthDirection::Vertical, LengthNegativeMode::Allow, 50.f, LengthUnits::Percent)
    , m_r(PropertyID::R, LengthDirection::Diagonal, LengthNegativeMode::Forbid, 50.f, LengthUnits::Percent)
    , m_fx(PropertyID::Fx, LengthDirection::Horizontal, LengthNegativeMode::Allow, 0.f, LengthUnits::None)
    , m_fy(PropertyID::Fy, LengthDirection::Vertical, LengthNegativeMode::Allow, 0.f, LengthUnits::None)
{
    addProperty(m_cx);
    addProperty(m_cy);
    addProperty(m_r);
    addProperty(m_fx);
    addProperty(m_fy);
}

bool SVGRadialGradientElement::applyPaint(SVGRenderState& state, float opacity) const
{
    auto attributes = collectGradientAttributes();
    auto gradientContentElement = attributes.gradientContentElement();
    auto gradientStops = buildGradientStops(gradientContentElement, opacity);
    if(gradientStops.empty())
        return false;
    LengthContext lengthContext(this, attributes.gradientUnits());
    auto fx = lengthContext.valueForLength(attributes.fx());
    auto fy = lengthContext.valueForLength(attributes.fy());
    auto cx = lengthContext.valueForLength(attributes.cx());
    auto cy = lengthContext.valueForLength(attributes.cy());
    auto r = lengthContext.valueForLength(attributes.r());
    if(r == 0.f || gradientStops.size() == 1) {
        const auto& lastStop = gradientStops.back();
        state->setColor(lastStop.color.r, lastStop.color.g, lastStop.color.b, lastStop.color.a);
        return true;
    }

    auto spreadMethod = attributes.spreadMethod();
    auto gradientUnits = attributes.gradientUnits();
    auto gradientTransform = attributes.gradientTransform();
    if(gradientUnits == Units::ObjectBoundingBox) {
        auto bbox = state.fillBoundingBox();
        gradientTransform.postMultiply(Transform(bbox.w, 0, 0, bbox.h, bbox.x, bbox.y));
    }

    state->setRadialGradient(cx, cy, r, fx, fy, spreadMethod, gradientStops, gradientTransform);
    return true;
}

SVGRadialGradientAttributes SVGRadialGradientElement::collectGradientAttributes() const
{
    SVGRadialGradientAttributes attributes;
    std::set<const SVGGradientElement*> processedGradients;
    const SVGGradientElement* current = this;
    while(true) {
        current->collectGradientAttributes(attributes);
        if(current->id() == ElementID::RadialGradient) {
            auto element = static_cast<const SVGRadialGradientElement*>(current);
            if(!attributes.hasCx() && element->hasAttribute(PropertyID::Cx))
                attributes.setCx(element);
            if(!attributes.hasCy() && element->hasAttribute(PropertyID::Cy))
                attributes.setCy(element);
            if(!attributes.hasR() && element->hasAttribute(PropertyID::R))
                attributes.setR(element);
            if(!attributes.hasFx() && element->hasAttribute(PropertyID::Fx))
                attributes.setFx(element);
            if(!attributes.hasFy() && element->hasAttribute(PropertyID::Fy)) {
                attributes.setFy(element);
            }
        }

        auto targetElement = current->getTargetElement(document());
        if(!targetElement || !(targetElement->id() == ElementID::LinearGradient || targetElement->id() == ElementID::RadialGradient))
            break;
        processedGradients.insert(current);
        current = static_cast<const SVGGradientElement*>(targetElement);
        if(processedGradients.count(current) > 0) {
            break;
        }
    }

    attributes.setDefaultValues(this);
    return attributes;
}

SVGPatternElement::SVGPatternElement(Document* document)
    : SVGPaintElement(document, ElementID::Pattern)
    , SVGURIReference(this)
    , SVGFitToViewBox(this)
    , m_x(PropertyID::X, LengthDirection::Horizontal, LengthNegativeMode::Allow)
    , m_y(PropertyID::Y, LengthDirection::Vertical, LengthNegativeMode::Allow)
    , m_width(PropertyID::Width, LengthDirection::Horizontal, LengthNegativeMode::Forbid)
    , m_height(PropertyID::Height, LengthDirection::Vertical, LengthNegativeMode::Forbid)
    , m_patternTransform(PropertyID::PatternTransform)
    , m_patternUnits(PropertyID::PatternUnits, Units::ObjectBoundingBox)
    , m_patternContentUnits(PropertyID::PatternContentUnits, Units::UserSpaceOnUse)
{
    addProperty(m_x);
    addProperty(m_y);
    addProperty(m_width);
    addProperty(m_height);
    addProperty(m_patternTransform);
    addProperty(m_patternUnits);
    addProperty(m_patternContentUnits);
}

bool SVGPatternElement::applyPaint(SVGRenderState& state, float opacity) const
{
    if(state.hasCycleReference(this))
        return false;
    auto attributes = collectPatternAttributes();
    auto patternContentElement = attributes.patternContentElement();
    if(patternContentElement == nullptr)
        return false;
    LengthContext lengthContext(this, attributes.patternUnits());
    Rect patternRect = {
        lengthContext.valueForLength(attributes.x()),
        lengthContext.valueForLength(attributes.y()),
        lengthContext.valueForLength(attributes.width()),
        lengthContext.valueForLength(attributes.height())
    };

    if(attributes.patternUnits() == Units::ObjectBoundingBox) {
        auto bbox = state.fillBoundingBox();
        patternRect.x = patternRect.x * bbox.w + bbox.x;
        patternRect.y = patternRect.y * bbox.h + bbox.y;
        patternRect.w = patternRect.w * bbox.w;
        patternRect.h = patternRect.h * bbox.h;
    }

    auto currentTransform = attributes.patternTransform() * state.currentTransform();
    auto xScale = currentTransform.xScale();
    auto yScale = currentTransform.yScale();

    auto patternImage = Canvas::create(0, 0, patternRect.w * xScale, patternRect.h * yScale);
    auto patternImageTransform = Transform::scaled(xScale, yScale);

    const auto& viewBoxRect = attributes.viewBox();
    if(viewBoxRect.isValid()) {
        const auto& preserveAspectRatio = attributes.preserveAspectRatio();
        patternImageTransform.multiply(preserveAspectRatio.getTransform(viewBoxRect, patternRect.size()));
    } else if(attributes.patternContentUnits() == Units::ObjectBoundingBox) {
        auto bbox = state.fillBoundingBox();
        patternImageTransform.scale(bbox.w, bbox.h);
    }

    SVGRenderState newState(this, &state, patternImageTransform, SVGRenderMode::Painting, patternImage);
    patternContentElement->renderChildren(newState);
    auto patternTransform = attributes.patternTransform();
    patternTransform.translate(patternRect.x, patternRect.y);
    patternTransform.scale(1.f / xScale, 1.f / yScale);
    state->setTexture(*patternImage, TextureType::Tiled, opacity, patternTransform);
    return true;
}

SVGPatternAttributes SVGPatternElement::collectPatternAttributes() const
{
    SVGPatternAttributes attributes;
    std::set<const SVGPatternElement*> processedPatterns;
    const SVGPatternElement* current = this;
    while(true) {
        if(!attributes.hasX() && current->hasAttribute(PropertyID::X))
            attributes.setX(current);
        if(!attributes.hasY() && current->hasAttribute(PropertyID::Y))
            attributes.setY(current);
        if(!attributes.hasWidth() && current->hasAttribute(PropertyID::Width))
            attributes.setWidth(current);
        if(!attributes.hasHeight() && current->hasAttribute(PropertyID::Height))
            attributes.setHeight(current);
        if(!attributes.hasPatternTransform() && current->hasAttribute(PropertyID::PatternTransform))
            attributes.setPatternTransform(current);
        if(!attributes.hasPatternUnits() && current->hasAttribute(PropertyID::PatternUnits))
            attributes.setPatternUnits(current);
        if(!attributes.hasPatternContentUnits() && current->hasAttribute(PropertyID::PatternContentUnits))
            attributes.setPatternContentUnits(current);
        if(!attributes.hasViewBox() && current->hasAttribute(PropertyID::ViewBox))
            attributes.setViewBox(current);
        if(!attributes.hasPreserveAspectRatio() && current->hasAttribute(PropertyID::PreserveAspectRatio))
            attributes.setPreserveAspectRatio(current);
        if(!attributes.hasPatternContentElement()) {
            for(const auto& child : current->children()) {
                if(child->isElement()) {
                    attributes.setPatternContentElement(current);
                    break;
                }
            }
        }

        auto targetElement = current->getTargetElement(document());
        if(!targetElement || targetElement->id() != ElementID::Pattern)
            break;
        processedPatterns.insert(current);
        current = static_cast<const SVGPatternElement*>(targetElement);
        if(processedPatterns.count(current) > 0) {
            break;
        }
    }

    attributes.setDefaultValues(this);
    return attributes;
}

} // namespace lunasvg
