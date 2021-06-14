#ifndef GELEMENT_H
#define GELEMENT_H

#include "graphicselement.h"

namespace lunasvg {

class GElement : public GraphicsElement
{
public:
    GElement();

    void layout(LayoutContext* context, LayoutContainer* current) const;
    std::unique_ptr<Node> clone() const;
};

} // namespace lunasvg

#endif // GELEMENT_H
