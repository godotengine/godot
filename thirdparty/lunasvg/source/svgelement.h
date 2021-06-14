#ifndef SVGELEMENT_H
#define SVGELEMENT_H

#include "graphicselement.h"

namespace lunasvg {

class ParseDocument;
class LayoutRoot;

class SVGElement : public GraphicsElement
{
public:
    SVGElement();

    Length x() const;
    Length y() const;
    Length width() const;
    Length height() const;

    Rect viewBox() const;
    PreserveAspectRatio preserveAspectRatio() const;
    Rect viewPort() const;
    std::unique_ptr<LayoutRoot> layoutDocument(const ParseDocument* document) const;

    void layout(LayoutContext* context, LayoutContainer* current) const;
    std::unique_ptr<Node> clone() const;
};

} // namespace lunasvg

#endif // SVGELEMENT_H
