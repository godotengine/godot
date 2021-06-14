#include "styledelement.h"
#include "parser.h"

using namespace lunasvg;

StyledElement::StyledElement(ElementId id)
    : Element(id)
{
}

Paint StyledElement::fill() const
{
    auto& value = find(PropertyId::Fill);
    if(value.empty())
        return Color::Black;

    return Parser::parsePaint(value, this);
}

Paint StyledElement::stroke() const
{
    auto& value = find(PropertyId::Stroke);
    if(value.empty())
        return Color::Transparent;

    return Parser::parsePaint(value, this);
}

Color StyledElement::color() const
{
    auto& value = find(PropertyId::Color);
    if(value.empty())
        return Color::Black;

    return Parser::parseColor(value, this);
}

Color StyledElement::stop_color() const
{
    auto& value = find(PropertyId::Stop_Color);
    if(value.empty())
        return Color::Black;

    return Parser::parseColor(value, this);
}

Color StyledElement::solid_color() const
{
    auto& value = find(PropertyId::Solid_Color);
    if(value.empty())
        return Color::Black;

    return Parser::parseColor(value, this);
}

double StyledElement::opacity() const
{
    auto& value = get(PropertyId::Opacity);
    if(value.empty())
        return 1.0;

    return Parser::parseNumberPercentage(value);
}

double StyledElement::fill_opacity() const
{
    auto& value = find(PropertyId::Fill_Opacity);
    if(value.empty())
        return 1.0;

    return Parser::parseNumberPercentage(value);
}

double StyledElement::stroke_opacity() const
{
    auto& value = find(PropertyId::Stroke_Opacity);
    if(value.empty())
        return 1.0;

    return Parser::parseNumberPercentage(value);
}

double StyledElement::stop_opacity() const
{
    auto& value = find(PropertyId::Stop_Opacity);
    if(value.empty())
        return 1.0;

    return Parser::parseNumberPercentage(value);
}

double StyledElement::solid_opacity() const
{
    auto& value = find(PropertyId::Solid_Opacity);
    if(value.empty())
        return 1.0;

    return Parser::parseNumberPercentage(value);
}

double StyledElement::stroke_miterlimit() const
{
    auto& value = find(PropertyId::Stroke_Miterlimit);
    if(value.empty())
        return 4.0;

    return Parser::parseNumber(value);
}

Length StyledElement::stroke_width() const
{
    auto& value = find(PropertyId::Stroke_Width);
    if(value.empty())
        return Length{1.0, LengthUnits::Number};

    return Parser::parseLength(value, ForbidNegativeLengths);
}

Length StyledElement::stroke_dashoffset() const
{
    auto& value = find(PropertyId::Stroke_Dashoffset);
    if(value.empty())
        return Length{};

    return Parser::parseLength(value, AllowNegativeLengths);
}

LengthList StyledElement::stroke_dasharray() const
{
    auto& value = find(PropertyId::Stroke_Dasharray);
    if(value.empty())
        return LengthList{};

    return Parser::parseLengthList(value, ForbidNegativeLengths);
}

WindRule StyledElement::fill_rule() const
{
    auto& value = find(PropertyId::Fill_Rule);
    if(value.empty())
        return WindRule::NonZero;

    return Parser::parseWindRule(value);
}

WindRule StyledElement::clip_rule() const
{
    auto& value = find(PropertyId::Clip_Rule);
    if(value.empty())
        return WindRule::NonZero;

    return Parser::parseWindRule(value);
}

LineCap StyledElement::stroke_linecap() const
{
    auto& value = find(PropertyId::Stroke_Linecap);
    if(value.empty())
        return LineCap::Butt;

    return Parser::parseLineCap(value);
}

LineJoin StyledElement::stroke_linejoin() const
{
    auto& value = find(PropertyId::Stroke_Linejoin);
    if(value.empty())
        return LineJoin::Miter;

    return Parser::parseLineJoin(value);
}

Display StyledElement::display() const
{
    auto& value = get(PropertyId::Display);
    if(value.empty())
        return Display::Inline;

    return Parser::parseDisplay(value);
}

Visibility StyledElement::visibility() const
{
    auto& value = find(PropertyId::Visibility);
    if(value.empty())
        return Visibility::Visible;

    return Parser::parseVisibility(value);
}

std::string StyledElement::clip_path() const
{
    auto& value = get(PropertyId::Clip_Path);
    if(value.empty())
        return std::string{};

    return Parser::parseUrl(value);
}

std::string StyledElement::mask() const
{
    auto& value = get(PropertyId::Mask);
    if(value.empty())
        return std::string{};

    return Parser::parseUrl(value);
}

std::string StyledElement::marker_start() const
{
    auto& value = find(PropertyId::Marker_Start);
    if(value.empty())
        return std::string{};

    return Parser::parseUrl(value);
}

std::string StyledElement::marker_mid() const
{
    auto& value = find(PropertyId::Marker_Mid);
    if(value.empty())
        return std::string{};

    return Parser::parseUrl(value);
}

std::string StyledElement::marker_end() const
{
    auto& value = find(PropertyId::Marker_End);
    if(value.empty())
        return std::string{};

    return Parser::parseUrl(value);
}

bool StyledElement::isDisplayNone() const
{
    return display() == Display::None;
}
