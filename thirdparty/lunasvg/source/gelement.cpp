#include "gelement.h"
#include "layoutcontext.h"

using namespace lunasvg;

GElement::GElement()
    : GraphicsElement(ElementId::G)
{
}

void GElement::layout(LayoutContext* context, LayoutContainer* current) const
{
    if(isDisplayNone())
        return;

    auto group = std::make_unique<LayoutGroup>();
    group->transform = transform();
    group->opacity = opacity();
    group->masker = context->getMasker(mask());
    group->clipper = context->getClipper(clip_path());
    layoutChildren(context, group.get());
    current->addChild(std::move(group));
}

std::unique_ptr<Node> GElement::clone() const
{
    return cloneElement<GElement>();
}
