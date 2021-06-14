#ifndef USEELEMENT_H
#define USEELEMENT_H

#include "graphicselement.h"

namespace lunasvg {

class UseElement : public GraphicsElement
{
public:
    UseElement();

    Length x() const;
    Length y() const;
    Length width() const;
    Length height() const;
    std::string href() const;
    void transferWidthAndHeight(Element* element) const;

    void layout(LayoutContext* context, LayoutContainer* current) const;
    std::unique_ptr<Node> clone() const;
};

} // namespace lunasvg

#endif // USEELEMENT_H
