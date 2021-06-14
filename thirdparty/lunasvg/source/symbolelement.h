#ifndef SYMBOLELEMENT_H
#define SYMBOLELEMENT_H

#include "styledelement.h"

namespace lunasvg {

class SymbolElement : public StyledElement
{
public:
    SymbolElement();

    Rect viewBox() const;
    PreserveAspectRatio preserveAspectRatio() const;

    std::unique_ptr<Node> clone() const;
};

} // namespace lunasvg

#endif // SYMBOLELEMENT_H
