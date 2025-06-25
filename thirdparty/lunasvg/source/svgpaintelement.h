#ifndef LUNASVG_SVGPAINTELEMENT_H
#define LUNASVG_SVGPAINTELEMENT_H

#include "svgelement.h"

namespace lunasvg {

class SVGPaintElement : public SVGElement {
public:
    SVGPaintElement(Document* document, ElementID id);

    bool isPaintElement() const final { return true; }

    virtual bool applyPaint(SVGRenderState& state, float opacity) const = 0;
};

class SVGStopElement final : public SVGElement {
public:
    SVGStopElement(Document* document);

    void layoutElement(const SVGLayoutState& state) final;
    const SVGNumberPercentage& offset() const { return m_offset; }
    GradientStop gradientStop(float opacity) const;

private:
    SVGNumberPercentage m_offset;
    Color m_stop_color = Color::Black;
    float m_stop_opacity = 1.f;
};

class SVGGradientAttributes;

class SVGGradientElement : public SVGPaintElement, public SVGURIReference {
public:
    SVGGradientElement(Document* document, ElementID id);

    const SVGTransform& gradientTransform() const { return m_gradientTransform; }
    const SVGEnumeration<Units>& gradientUnits() const { return m_gradientUnits; }
    const SVGEnumeration<SpreadMethod>& spreadMethod() const { return m_spreadMethod; }
    void collectGradientAttributes(SVGGradientAttributes& attributes) const;

private:
    SVGTransform m_gradientTransform;
    SVGEnumeration<Units> m_gradientUnits;
    SVGEnumeration<SpreadMethod> m_spreadMethod;
};

class SVGGradientAttributes {
public:
    SVGGradientAttributes() = default;

    const Transform& gradientTransform() const { return m_gradientTransform->gradientTransform().value(); }
    SpreadMethod spreadMethod() const { return m_spreadMethod->spreadMethod().value(); }
    Units gradientUnits() const { return m_gradientUnits->gradientUnits().value(); }
    const SVGGradientElement* gradientContentElement() const { return m_gradientContentElement; }

    bool hasGradientTransform() const { return m_gradientTransform; }
    bool hasSpreadMethod() const { return m_spreadMethod; }
    bool hasGradientUnits() const { return m_gradientUnits; }
    bool hasGradientContentElement() const { return m_gradientContentElement; }

    void setGradientTransform(const SVGGradientElement* value) { m_gradientTransform = value; }
    void setSpreadMethod(const SVGGradientElement* value) { m_spreadMethod = value; }
    void setGradientUnits(const SVGGradientElement* value) { m_gradientUnits = value; }
    void setGradientContentElement(const SVGGradientElement* value) { m_gradientContentElement = value; }

    void setDefaultValues(const SVGGradientElement* element) {
        if(!m_gradientTransform) { m_gradientTransform = element; }
        if(!m_spreadMethod) { m_spreadMethod = element; }
        if(!m_gradientUnits) { m_gradientUnits = element; }
        if(!m_gradientContentElement) { m_gradientContentElement = element; }
    }

private:
    const SVGGradientElement* m_gradientTransform{nullptr};
    const SVGGradientElement* m_spreadMethod{nullptr};
    const SVGGradientElement* m_gradientUnits{nullptr};
    const SVGGradientElement* m_gradientContentElement{nullptr};
};

class SVGLinearGradientAttributes;

class SVGLinearGradientElement final : public SVGGradientElement {
public:
    SVGLinearGradientElement(Document* document);

    const SVGLength& x1() const { return m_x1; }
    const SVGLength& y1() const { return m_y1; }
    const SVGLength& x2() const { return m_x2; }
    const SVGLength& y2() const { return m_y2; }

    bool applyPaint(SVGRenderState& state, float opacity) const final;

private:
    SVGLinearGradientAttributes collectGradientAttributes() const;
    SVGLength m_x1;
    SVGLength m_y1;
    SVGLength m_x2;
    SVGLength m_y2;
};

class SVGLinearGradientAttributes : public SVGGradientAttributes {
public:
    SVGLinearGradientAttributes() = default;

    const SVGLength& x1() const { return m_x1->x1(); }
    const SVGLength& y1() const { return m_y1->y1(); }
    const SVGLength& x2() const { return m_x2->x2(); }
    const SVGLength& y2() const { return m_y2->y2(); }

    bool hasX1() const { return m_x1; }
    bool hasY1() const { return m_y1; }
    bool hasX2() const { return m_x2; }
    bool hasY2() const { return m_y2; }

    void setX1(const SVGLinearGradientElement* value) { m_x1 = value; }
    void setY1(const SVGLinearGradientElement* value) { m_y1 = value; }
    void setX2(const SVGLinearGradientElement* value) { m_x2 = value; }
    void setY2(const SVGLinearGradientElement* value) { m_y2 = value; }

    void setDefaultValues(const SVGLinearGradientElement* element) {
        SVGGradientAttributes::setDefaultValues(element);
        if(!m_x1) { m_x1 = element; }
        if(!m_y1) { m_y1 = element; }
        if(!m_x2) { m_x2 = element; }
        if(!m_y2) { m_y2 = element; }
    }

private:
    const SVGLinearGradientElement* m_x1{nullptr};
    const SVGLinearGradientElement* m_y1{nullptr};
    const SVGLinearGradientElement* m_x2{nullptr};
    const SVGLinearGradientElement* m_y2{nullptr};
};

class SVGRadialGradientAttributes;

class SVGRadialGradientElement final : public SVGGradientElement {
public:
    SVGRadialGradientElement(Document* document);

    const SVGLength& cx() const { return m_cx; }
    const SVGLength& cy() const { return m_cy; }
    const SVGLength& r() const { return m_r; }
    const SVGLength& fx() const { return m_fx; }
    const SVGLength& fy() const { return m_fy; }

    bool applyPaint(SVGRenderState& state, float opacity) const final;

private:
    SVGRadialGradientAttributes collectGradientAttributes() const;
    SVGLength m_cx;
    SVGLength m_cy;
    SVGLength m_r;
    SVGLength m_fx;
    SVGLength m_fy;
};

class SVGRadialGradientAttributes : public SVGGradientAttributes {
public:
    SVGRadialGradientAttributes() = default;

    const SVGLength& cx() const { return m_cx->cx(); }
    const SVGLength& cy() const { return m_cy->cy(); }
    const SVGLength& r() const { return m_r->r(); }
    const SVGLength& fx() const { return m_fx ? m_fx->fx() : m_cx->cx(); }
    const SVGLength& fy() const { return m_fy ? m_fy->fy() : m_cy->cy(); }

    bool hasCx() const { return m_cx; }
    bool hasCy() const { return m_cy; }
    bool hasR() const { return m_r; }
    bool hasFx() const { return m_fx; }
    bool hasFy() const { return m_fy; }

    void setCx(const SVGRadialGradientElement* value) { m_cx = value; }
    void setCy(const SVGRadialGradientElement* value) { m_cy = value; }
    void setR(const SVGRadialGradientElement* value) { m_r = value; }
    void setFx(const SVGRadialGradientElement* value) { m_fx = value; }
    void setFy(const SVGRadialGradientElement* value) { m_fy = value; }

    void setDefaultValues(const SVGRadialGradientElement* element) {
        SVGGradientAttributes::setDefaultValues(element);
        if(!m_cx) { m_cx = element; }
        if(!m_cy) { m_cy = element; }
        if(!m_r) { m_r = element; }
    }

private:
    const SVGRadialGradientElement* m_cx{nullptr};
    const SVGRadialGradientElement* m_cy{nullptr};
    const SVGRadialGradientElement* m_r{nullptr};
    const SVGRadialGradientElement* m_fx{nullptr};
    const SVGRadialGradientElement* m_fy{nullptr};
};

class SVGPatternAttributes;

class SVGPatternElement final : public SVGPaintElement, public SVGURIReference, public SVGFitToViewBox {
public:
    SVGPatternElement(Document* document);

    const SVGLength& x() const { return m_x; }
    const SVGLength& y() const { return m_y; }
    const SVGLength& width() const { return m_width; }
    const SVGLength& height() const { return m_height; }
    const SVGTransform& patternTransform() const { return m_patternTransform; }
    const SVGEnumeration<Units>& patternUnits() const { return m_patternUnits; }
    const SVGEnumeration<Units>& patternContentUnits() const { return m_patternContentUnits; }

    bool applyPaint(SVGRenderState& state, float opacity) const final;

private:
    SVGPatternAttributes collectPatternAttributes() const;
    SVGLength m_x;
    SVGLength m_y;
    SVGLength m_width;
    SVGLength m_height;
    SVGTransform m_patternTransform;
    SVGEnumeration<Units> m_patternUnits;
    SVGEnumeration<Units> m_patternContentUnits;
};

class SVGPatternAttributes {
public:
    SVGPatternAttributes() = default;

    const SVGLength& x() const { return m_x->x(); }
    const SVGLength& y() const { return m_y->y(); }
    const SVGLength& width() const { return m_width->width(); }
    const SVGLength& height() const { return m_height->height(); }
    const Transform& patternTransform() const { return m_patternTransform->patternTransform().value(); }
    Units patternUnits() const { return m_patternUnits->patternUnits().value(); }
    Units patternContentUnits() const { return m_patternContentUnits->patternContentUnits().value(); }
    const Rect& viewBox() const { return m_viewBox->viewBox().value(); }
    const SVGPreserveAspectRatio& preserveAspectRatio() const { return m_preserveAspectRatio->preserveAspectRatio(); }
    const SVGPatternElement* patternContentElement() const { return m_patternContentElement; }

    bool hasX() const { return m_x; }
    bool hasY() const { return m_y; }
    bool hasWidth() const { return m_width; }
    bool hasHeight() const { return m_height; }
    bool hasPatternTransform() const { return m_patternTransform; }
    bool hasPatternUnits() const { return m_patternUnits; }
    bool hasPatternContentUnits() const { return m_patternContentUnits; }
    bool hasViewBox() const { return m_viewBox; }
    bool hasPreserveAspectRatio() const { return m_preserveAspectRatio; }
    bool hasPatternContentElement() const { return m_patternContentElement; }

    void setX(const SVGPatternElement* value) { m_x = value; }
    void setY(const SVGPatternElement* value) { m_y = value; }
    void setWidth(const SVGPatternElement* value) { m_width = value; }
    void setHeight(const SVGPatternElement* value) { m_height = value; }
    void setPatternTransform(const SVGPatternElement* value) { m_patternTransform = value; }
    void setPatternUnits(const SVGPatternElement* value) { m_patternUnits = value; }
    void setPatternContentUnits(const SVGPatternElement* value) { m_patternContentUnits = value; }
    void setViewBox(const SVGPatternElement* value) { m_viewBox = value; }
    void setPreserveAspectRatio(const SVGPatternElement* value) { m_preserveAspectRatio = value; }
    void setPatternContentElement(const SVGPatternElement* value) { m_patternContentElement = value; }

    void setDefaultValues(const SVGPatternElement* element) {
        if(!m_x) { m_x = element; }
        if(!m_y) { m_y = element; }
        if(!m_width) { m_width = element; }
        if(!m_height) { m_height = element; }
        if(!m_patternTransform) { m_patternTransform = element; }
        if(!m_patternUnits) { m_patternUnits = element; }
        if(!m_patternContentUnits) { m_patternContentUnits = element; }
        if(!m_viewBox) { m_viewBox = element; }
        if(!m_preserveAspectRatio) { m_preserveAspectRatio = element; }
        if(!m_patternContentElement) { m_patternContentElement = element; }
    }

private:
    const SVGPatternElement* m_x{nullptr};
    const SVGPatternElement* m_y{nullptr};
    const SVGPatternElement* m_width{nullptr};
    const SVGPatternElement* m_height{nullptr};
    const SVGPatternElement* m_patternTransform{nullptr};
    const SVGPatternElement* m_patternUnits{nullptr};
    const SVGPatternElement* m_patternContentUnits{nullptr};
    const SVGPatternElement* m_viewBox{nullptr};
    const SVGPatternElement* m_preserveAspectRatio{nullptr};
    const SVGPatternElement* m_patternContentElement{nullptr};
};

} // namespace lunasvg

#endif // LUNASVG_SVGPAINTELEMENT_H
