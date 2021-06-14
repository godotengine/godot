#ifndef DEFSELEMENT_H
#define DEFSELEMENT_H

#include "graphicselement.h"

namespace lunasvg {

class DefsElement : public GraphicsElement
{
public:
    DefsElement();

    std::unique_ptr<Node> clone() const;
};

} // namespace lunasvg

#endif // DEFSELEMENT_H
