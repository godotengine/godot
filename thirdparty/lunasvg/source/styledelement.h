#ifndef STYLEDELEMENT_H
#define STYLEDELEMENT_H

#include "element.h"

namespace lunasvg {

class StyledElement : public Element
{
public:
    StyledElement(ElementId id);

    Paint fill() const;
    Paint stroke() const;

    Color color() const;
    Color stop_color() const;
    Color solid_color() const;

    double opacity() const;
    double fill_opacity() const;
    double stroke_opacity() const;
    double stop_opacity() const;
    double solid_opacity() const;
    double stroke_miterlimit() const;

    Length stroke_width() const;
    Length stroke_dashoffset() const;
    LengthList stroke_dasharray() const;

    WindRule fill_rule() const;
    WindRule clip_rule() const;

    LineCap stroke_linecap() const;
    LineJoin stroke_linejoin() const;

    Display display() const;
    Visibility visibility() const;

    std::string clip_path() const;
    std::string mask() const;
    std::string marker_start() const;
    std::string marker_mid() const;
    std::string marker_end() const;

    bool isDisplayNone() const;
};

} // namespace lunasvg

#endif // STYLEDELEMENT_H
