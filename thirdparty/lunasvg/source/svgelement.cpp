#include "svgelement.h"
#include "parser.h"
#include "layoutcontext.h"

using namespace lunasvg;

SVGElement::SVGElement()
    : GraphicsElement(ElementId::Svg)
{
}

Length SVGElement::x() const
{
    auto& value = get(PropertyId::X);
    if(value.empty())
        return Length{};

    return Parser::parseLength(value, AllowNegativeLengths);
}

Length SVGElement::y() const
{
    auto& value = get(PropertyId::Y);
    if(value.empty())
        return Length{};

    return Parser::parseLength(value, AllowNegativeLengths);
}

Length SVGElement::width() const
{
    auto& value = get(PropertyId::Width);
    if(value.empty())
        return Length{100, LengthUnits::Percent};

    return Parser::parseLength(value, ForbidNegativeLengths);
}

Length SVGElement::height() const
{
    auto& value = get(PropertyId::Height);
    if(value.empty())
        return Length{100, LengthUnits::Percent};

    return Parser::parseLength(value, ForbidNegativeLengths);
}

Rect SVGElement::viewBox() const
{
    auto& value = get(PropertyId::ViewBox);
    if(value.empty())
        return Rect{};

    return Parser::parseViewBox(value);
}

PreserveAspectRatio SVGElement::preserveAspectRatio() const
{
    auto& value = get(PropertyId::PreserveAspectRatio);
    if(value.empty())
        return PreserveAspectRatio{};

    return Parser::parsePreserveAspectRatio(value);
}

Rect SVGElement::viewPort() const
{
    LengthContext lengthContext(this);
    auto _x = lengthContext.valueForLength(x(), LengthMode::Width);
    auto _y = lengthContext.valueForLength(y(), LengthMode::Height);
    auto _w = lengthContext.valueForLength(width(), LengthMode::Width);
    auto _h = lengthContext.valueForLength(height(), LengthMode::Height);
    return Rect{_x, _y, _w, _h};
}

std::unique_ptr<LayoutRoot> SVGElement::layoutDocument(const ParseDocument* document) const
{
    if(isDisplayNone())
        return nullptr;

    auto w = this->width();
    auto h = this->height();
    if(w.isZero() || h.isZero())
        return nullptr;

    auto viewPort = this->viewPort();
    auto preserveAspectRatio = this->preserveAspectRatio();

    auto root = std::make_unique<LayoutRoot>();
    root->width = viewPort.w;
    root->height = viewPort.h;
    root->viewTransform = preserveAspectRatio.getMatrix(viewPort, viewBox());
    root->transform = transform();
    root->opacity = opacity();

    LayoutContext context{document, root.get()};
    root->masker = context.getMasker(mask());
    root->clipper = context.getClipper(clip_path());
    layoutChildren(&context, root.get());
    return root;
}

void SVGElement::layout(LayoutContext* context, LayoutContainer* current) const
{
    if(isDisplayNone())
        return;

    auto w = this->width();
    auto h = this->height();
    if(w.isZero() || h.isZero())
        return;

    auto preserveAspectRatio = this->preserveAspectRatio();
    auto viewTransform = preserveAspectRatio.getMatrix(viewPort(), viewBox());

    auto group = std::make_unique<LayoutGroup>();
    group->transform = viewTransform * transform();
    group->opacity = opacity();
    group->masker = context->getMasker(mask());
    group->clipper = context->getClipper(clip_path());
    layoutChildren(context, group.get());
    current->addChild(std::move(group));
}

std::unique_ptr<Node> SVGElement::clone() const
{
    return cloneElement<SVGElement>();
}
