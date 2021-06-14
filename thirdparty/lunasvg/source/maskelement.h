#ifndef MASKELEMENT_H
#define MASKELEMENT_H

#include "styledelement.h"

namespace lunasvg {

class LayoutMask;

class MaskElement : public StyledElement
{
public:
    MaskElement();

    Length x() const;
    Length y() const;
    Length width() const;
    Length height() const;
    Units maskUnits() const;
    Units maskContentUnits() const;
    std::unique_ptr<LayoutMask> getMasker(LayoutContext* context) const;

    std::unique_ptr<Node> clone() const;
};

} // namespace lunasvg

#endif // MASKELEMENT_H
