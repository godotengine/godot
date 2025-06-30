#ifndef LUNASVG_SVGRENDERSTATE_H
#define LUNASVG_SVGRENDERSTATE_H

#include "svgelement.h"

namespace lunasvg {

enum class SVGRenderMode {
    Painting,
    Clipping
};

class SVGBlendInfo {
public:
    explicit SVGBlendInfo(const SVGElement* element);
    SVGBlendInfo(const SVGClipPathElement* clipper, const SVGMaskElement* masker, float opacity)
        : m_clipper(clipper), m_masker(masker), m_opacity(opacity)
    {}

    bool requiresCompositing(SVGRenderMode mode) const;
    const SVGClipPathElement* clipper() const { return m_clipper; }
    const SVGMaskElement* masker() const { return m_masker; }
    float opacity() const { return m_opacity; }

private:
    const SVGClipPathElement* m_clipper;
    const SVGMaskElement* m_masker;
    const float m_opacity;
};

class SVGRenderState {
public:
    SVGRenderState(const SVGElement* element, const SVGRenderState& parent, const Transform& localTransform)
        : m_element(element), m_parent(&parent), m_currentTransform(parent.currentTransform() * localTransform)
        , m_mode(parent.mode()), m_canvas(parent.canvas())
    {}

    SVGRenderState(const SVGElement* element, const SVGRenderState* parent, const Transform& currentTransform, SVGRenderMode mode, std::shared_ptr<Canvas> canvas)
        : m_element(element), m_parent(parent), m_currentTransform(currentTransform), m_mode(mode), m_canvas(std::move(canvas))
    {}

    Canvas& operator*() const { return *m_canvas; }
    Canvas* operator->() const { return &*m_canvas; }

    const SVGElement* element() const { return m_element; }
    const SVGRenderState* parent() const { return m_parent; }
    const Transform& currentTransform() const { return m_currentTransform; }
    const SVGRenderMode mode() const { return m_mode; }
    const std::shared_ptr<Canvas>& canvas() const { return m_canvas; }

    Rect fillBoundingBox() const { return m_element->fillBoundingBox(); }
    Rect paintBoundingBox() const { return m_element->paintBoundingBox(); }

    bool hasCycleReference(const SVGElement* element) const;

    void beginGroup(const SVGBlendInfo& blendInfo);
    void endGroup(const SVGBlendInfo& blendInfo);

private:
    const SVGElement* m_element;
    const SVGRenderState* m_parent;
    const Transform m_currentTransform;
    const SVGRenderMode m_mode;
    std::shared_ptr<Canvas> m_canvas;
};

} // namespace lunasvg

#endif // LUNASVG_SVGRENDERSTATE_H
