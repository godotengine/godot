#include "useelement.h"
#include "parser.h"
#include "layoutcontext.h"

#include "gelement.h"
#include "svgelement.h"

using namespace lunasvg;

UseElement::UseElement()
    : GraphicsElement(ElementId::Use)
{
}

Length UseElement::x() const
{
    auto& value = get(PropertyId::X);
    if(value.empty())
        return Length{};

    return Parser::parseLength(value, AllowNegativeLengths);
}

Length UseElement::y() const
{
    auto& value = get(PropertyId::Y);
    if(value.empty())
        return Length{};

    return Parser::parseLength(value, AllowNegativeLengths);
}

Length UseElement::width() const
{
    auto& value = get(PropertyId::Width);
    if(value.empty())
        return Length{100, LengthUnits::Percent};

    return Parser::parseLength(value, ForbidNegativeLengths);
}

Length UseElement::height() const
{
    auto& value = get(PropertyId::Height);
    if(value.empty())
        return Length{100, LengthUnits::Percent};

    return Parser::parseLength(value, ForbidNegativeLengths);
}

std::string UseElement::href() const
{
    auto& value = get(PropertyId::Href);
    if(value.empty())
        return std::string{};

    return Parser::parseHref(value);
}

void UseElement::transferWidthAndHeight(Element* element) const
{
    auto& width = get(PropertyId::Width);
    auto& height = get(PropertyId::Height);

    element->set(PropertyId::Width, width);
    element->set(PropertyId::Height, height);
}

void UseElement::layout(LayoutContext* context, LayoutContainer* current) const
{
    if(isDisplayNone())
        return;

    auto ref = context->getElementById(href());
    if(ref == nullptr)
        return;

    auto group = std::make_unique<GElement>();
    group->parent = parent;
    group->properties = properties;

    LengthContext lengthContext(this);
    auto _x = lengthContext.valueForLength(x(), LengthMode::Width);
    auto _y = lengthContext.valueForLength(y(), LengthMode::Height);

    std::string transform;
    transform += get(PropertyId::Transform);
    transform += "translate(";
    transform += std::to_string(_x);
    transform += ' ';
    transform += std::to_string(_y);
    transform += ')';
    group->set(PropertyId::Transform, transform);

    if(ref->id == ElementId::Svg || ref->id == ElementId::Symbol)
    {
        auto element = ref->cloneElement<SVGElement>();
        transferWidthAndHeight(element.get());
        group->addChild(std::move(element));
    }
    else
    {
        group->addChild(ref->clone());
    }

    group->layout(context, current);
}

std::unique_ptr<Node> UseElement::clone() const
{
    return cloneElement<UseElement>();
}
