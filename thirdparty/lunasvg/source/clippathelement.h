#ifndef CLIPPATHELEMENT_H
#define CLIPPATHELEMENT_H

#include "graphicselement.h"

namespace lunasvg {

class LayoutClipPath;

class ClipPathElement : public GraphicsElement
{
public:
    ClipPathElement();

    Units clipPathUnits() const;
    std::unique_ptr<LayoutClipPath> getClipper(LayoutContext* context) const;

    std::unique_ptr<Node> clone() const;
};

} // namespace lunasvg

#endif // CLIPPATHELEMENT_H
