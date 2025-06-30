#ifndef LUNASVG_SVGGEOMETRYELEMENT_H
#define LUNASVG_SVGGEOMETRYELEMENT_H

#include "svgelement.h"

namespace lunasvg {

class SVGMarkerPosition {
public:
    SVGMarkerPosition(const SVGMarkerElement* element, const Point& origin, float angle)
        : m_element(element), m_origin(origin), m_angle(angle)
    {}

    const SVGMarkerElement* element() const { return m_element; }
    const Point& origin() const { return m_origin; }
    float angle() const { return m_angle; }

    Rect markerBoundingBox(float strokeWidth) const;
    void renderMarker(SVGRenderState& state, float strokeWidth) const;

private:
    const SVGMarkerElement* m_element;
    Point m_origin;
    float m_angle;
};

using SVGMarkerPositionList = std::vector<SVGMarkerPosition>;

class SVGGeometryElement : public SVGGraphicsElement {
public:
    SVGGeometryElement(Document* document, ElementID id);

    bool isGeometryElement() const final { return true; }

    Rect fillBoundingBox() const override { return m_fillBoundingBox; }
    Rect strokeBoundingBox() const override;
    void layoutElement(const SVGLayoutState& state) override;

    FillRule fill_rule() const { return m_fill_rule; }
    FillRule clip_rule() const { return m_clip_rule; }

    virtual Rect updateShape(Path& path) = 0;
    void updateMarkerPositions(SVGMarkerPositionList& positions, const SVGLayoutState& state);
    void render(SVGRenderState& state) const override;

    const Path& path() const { return m_path; }

private:
    Path m_path;
    Rect m_fillBoundingBox;
    StrokeData m_strokeData;

    SVGPaintServer m_fill;
    SVGPaintServer m_stroke;
    SVGMarkerPositionList m_markerPositions;

    FillRule m_fill_rule = FillRule::NonZero;
    FillRule m_clip_rule = FillRule::NonZero;
};

class SVGLineElement final : public SVGGeometryElement {
public:
    SVGLineElement(Document* document);

    Rect updateShape(Path& path) final;

private:
    SVGLength m_x1;
    SVGLength m_y1;
    SVGLength m_x2;
    SVGLength m_y2;
};

class SVGRectElement final : public SVGGeometryElement {
public:
    SVGRectElement(Document* document);

    Rect updateShape(Path& path) final;

private:
    SVGLength m_x;
    SVGLength m_y;
    SVGLength m_width;
    SVGLength m_height;
    SVGLength m_rx;
    SVGLength m_ry;
};

class SVGEllipseElement final : public SVGGeometryElement {
public:
    SVGEllipseElement(Document* document);

    Rect updateShape(Path& path) final;

private:
    SVGLength m_cx;
    SVGLength m_cy;
    SVGLength m_rx;
    SVGLength m_ry;
};

class SVGCircleElement final : public SVGGeometryElement {
public:
    SVGCircleElement(Document* document);

    Rect updateShape(Path& path) final;

private:
    SVGLength m_cx;
    SVGLength m_cy;
    SVGLength m_r;
};

class SVGPolyElement final : public SVGGeometryElement {
public:
    SVGPolyElement(Document* document, ElementID id);

    Rect updateShape(Path& path) final;

private:
    SVGPointList m_points;
};

class SVGPathElement final : public SVGGeometryElement {
public:
    SVGPathElement(Document* document);

    Rect updateShape(Path& path) final;

private:
    SVGPath m_d;
};

} // namespace lunasvg

#endif // LUNASVG_SVGGEOMETRYELEMENT_H
