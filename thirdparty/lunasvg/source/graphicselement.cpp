#include "graphicselement.h"
#include "parser.h"

using namespace lunasvg;

GraphicsElement::GraphicsElement(ElementId id)
    : StyledElement(id)
{
}

Transform GraphicsElement::transform() const
{
    auto& value = get(PropertyId::Transform);
    if(value.empty())
        return Transform{};

    return Parser::parseTransform(value);
}
