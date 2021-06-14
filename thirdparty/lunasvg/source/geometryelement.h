#ifndef GEOMETRYELEMENT_H
#define GEOMETRYELEMENT_H

#include "graphicselement.h"

namespace lunasvg {

class LayoutShape;

class GeometryElement : public GraphicsElement
{
public:
    GeometryElement(ElementId id);

    virtual void layout(LayoutContext* context, LayoutContainer* current) const;
    virtual void layoutMarkers(LayoutContext* context, LayoutShape* shape) const;
    virtual Path path() const = 0;
};

class PathElement : public GeometryElement
{
public:
    PathElement();

    Path d() const;

    Path path() const;

    std::unique_ptr<Node> clone() const;
};

class PolyElement : public GeometryElement
{
public:
    PolyElement(ElementId id);

    PointList points() const;
};

class PolygonElement : public PolyElement
{
public:
    PolygonElement();

    Path path() const;

    std::unique_ptr<Node> clone() const;
};

class PolylineElement : public PolyElement
{
public:
    PolylineElement();

    Path path() const;

    std::unique_ptr<Node> clone() const;
};

class CircleElement : public GeometryElement
{
public:
    CircleElement();

    Length cx() const;
    Length cy() const;
    Length r() const;

    Path path() const;

    std::unique_ptr<Node> clone() const;
};

class EllipseElement : public GeometryElement
{
public:
    EllipseElement();

    Length cx() const;
    Length cy() const;
    Length rx() const;
    Length ry() const;

    Path path() const;

    std::unique_ptr<Node> clone() const;
};

class LineElement : public GeometryElement
{
public:
    LineElement();

    Length x1() const;
    Length y1() const;
    Length x2() const;
    Length y2() const;

    Path path() const;

    std::unique_ptr<Node> clone() const;
};

class RectElement : public GeometryElement
{
public:
    RectElement();

    Length x() const;
    Length y() const;
    Length rx() const;
    Length ry() const;
    Length width() const;
    Length height() const;

    Path path() const;

    std::unique_ptr<Node> clone() const;
};

} // namespace lunasvg

#endif // GEOMETRYELEMENT_H
