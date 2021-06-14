#include "stopelement.h"
#include "parser.h"

using namespace lunasvg;

StopElement::StopElement()
    : StyledElement(ElementId::Stop)
{
}

double StopElement::offset() const
{
    auto& value = get(PropertyId::Offset);
    if(value.empty())
        return 1.0;

    return Parser::parseNumberPercentage(value);
}

Color StopElement::stopColorWithOpacity() const
{
    auto color = stop_color();
    color.a = stop_opacity();
    return color;
}

std::unique_ptr<Node> StopElement::clone() const
{
    return cloneElement<StopElement>();
}
