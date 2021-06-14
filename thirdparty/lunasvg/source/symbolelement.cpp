#include "symbolelement.h"
#include "parser.h"

using namespace lunasvg;

SymbolElement::SymbolElement()
    : StyledElement(ElementId::Symbol)
{
}

Rect SymbolElement::viewBox() const
{
    auto& value = get(PropertyId::ViewBox);
    if(value.empty())
        return Rect{};

    return Parser::parseViewBox(value);
}

PreserveAspectRatio SymbolElement::preserveAspectRatio() const
{
    auto& value = get(PropertyId::PreserveAspectRatio);
    if(value.empty())
        return PreserveAspectRatio{};

    return Parser::parsePreserveAspectRatio(value);
}

std::unique_ptr<Node> SymbolElement::clone() const
{
    return cloneElement<SymbolElement>();
}
