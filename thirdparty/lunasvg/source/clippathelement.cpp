#include "clippathelement.h"
#include "parser.h"
#include "layoutcontext.h"

using namespace lunasvg;

ClipPathElement::ClipPathElement()
    : GraphicsElement(ElementId::ClipPath)
{
}

Units ClipPathElement::clipPathUnits() const
{
    auto& value = get(PropertyId::ClipPathUnits);
    if(value.empty())
        return Units::UserSpaceOnUse;

    return Parser::parseUnits(value);
}

std::unique_ptr<LayoutClipPath> ClipPathElement::getClipper(LayoutContext* context) const
{
    auto clipper = std::make_unique<LayoutClipPath>();
    clipper->units = clipPathUnits();
    clipper->transform = transform();
    clipper->clipper = context->getClipper(clip_path());
    layoutChildren(context, clipper.get());
    return clipper;
}

std::unique_ptr<Node> ClipPathElement::clone() const
{
    return cloneElement<ClipPathElement>();
}
