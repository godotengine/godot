#include "geometryelement.h"
#include "parser.h"
#include "layoutcontext.h"

#include <cmath>

using namespace lunasvg;

GeometryElement::GeometryElement(ElementId id)
    : GraphicsElement(id)
{
}

void GeometryElement::layout(LayoutContext* context, LayoutContainer* current) const
{
    if(isDisplayNone())
        return;

    auto path = this->path();
    if(path.empty())
        return;

    auto shape = std::make_unique<LayoutShape>();
    shape->path = std::move(path);
    shape->box = shape->path.box();
    shape->transform = transform();
    shape->fillData = context->fillData(this);
    shape->strokeData = context->strokeData(this);
    shape->visibility = visibility();
    shape->clipRule = clip_rule();
    shape->masker = context->getMasker(mask());
    shape->clipper = context->getClipper(clip_path());
    layoutMarkers(context, shape.get());
    current->addChild(std::move(shape));
}

static const double pi = 3.14159265358979323846;

void GeometryElement::layoutMarkers(LayoutContext* context, LayoutShape* shape) const
{
    auto markerStart = context->getMarker(marker_start());
    auto markerMid = context->getMarker(marker_mid());
    auto markerEnd = context->getMarker(marker_end());

    if(markerStart == nullptr && markerMid == nullptr && markerEnd == nullptr)
        return;

    PathIterator it(shape->path);
    Point origin;
    Point startPoint;
    Point inslopePoints[2];
    Point outslopePoints[2];

    int index = 0;
    std::array<Point, 3> points;
    while(!it.isDone())
    {
        switch(it.currentSegment(points)) {
        case PathCommand::MoveTo:
            startPoint = points[0];
            inslopePoints[0] = origin;
            inslopePoints[1] = points[0];
            origin = points[0];
            break;
        case PathCommand::LineTo:
            inslopePoints[0] = origin;
            inslopePoints[1] = points[0];
            origin = points[0];
            break;
        case PathCommand::CubicTo:
            inslopePoints[0] = points[1];
            inslopePoints[1] = points[2];
            origin = points[2];
            break;
        case PathCommand::Close:
            inslopePoints[0] = origin;
            inslopePoints[1] = points[0];
            origin = startPoint;
            startPoint = Point{};
            break;
        }

        index += 1;
        it.next();

        if(!it.isDone() && (markerStart || markerMid))
        {
            it.currentSegment(points);
            outslopePoints[0] = origin;
            outslopePoints[1] = points[0];

            if(index == 1 && markerStart)
            {
                Point slope{outslopePoints[1].x - outslopePoints[0].x, outslopePoints[1].y - outslopePoints[0].y};
                auto angle = 180.0 * std::atan2(slope.y, slope.x) / pi;

                shape->markers.emplace_back(markerStart, origin, angle);
            }

            if(index > 1 && markerMid)
            {
                Point inslope{inslopePoints[1].x - inslopePoints[0].x, inslopePoints[1].y - inslopePoints[0].y};
                Point outslope{outslopePoints[1].x - outslopePoints[0].x, outslopePoints[1].y - outslopePoints[0].y};
                auto inangle = 180.0 * std::atan2(inslope.y, inslope.x) / pi;
                auto outangle = 180.0 * std::atan2(outslope.y, outslope.x) / pi;
                auto angle = (inangle + outangle) * 0.5;

                shape->markers.emplace_back(markerMid, origin, angle);
            }
        }

        if(it.isDone() && markerEnd)
        {
            Point slope{inslopePoints[1].x - inslopePoints[0].x, inslopePoints[1].y - inslopePoints[0].y};
            auto angle = 180.0 * std::atan2(slope.y, slope.x) / pi;

            shape->markers.emplace_back(markerEnd, origin, angle);
        }
    }
}

PathElement::PathElement()
    : GeometryElement(ElementId::Path)
{
}

Path PathElement::d() const
{
    auto& value = get(PropertyId::D);
    if(value.empty())
        return Path{};

    return Parser::parsePath(value);
}

Path PathElement::path() const
{
    return d();
}

std::unique_ptr<Node> PathElement::clone() const
{
    return cloneElement<PathElement>();
}

PolyElement::PolyElement(ElementId id)
    : GeometryElement(id)
{
}

PointList PolyElement::points() const
{
    auto& value = get(PropertyId::Points);
    if(value.empty())
        return PointList{};

    return Parser::parsePointList(value);
}

PolygonElement::PolygonElement()
    : PolyElement(ElementId::Polygon)
{
}

Path PolygonElement::path() const
{
    auto points = this->points();
    if(points.empty())
        return Path{};

    Path path;
    path.moveTo(points[0].x, points[0].y);
    for(std::size_t i = 1;i < points.size();i++)
        path.lineTo(points[i].x, points[i].y);

    path.close();
    return path;
}

std::unique_ptr<Node> PolygonElement::clone() const
{
    return cloneElement<PolygonElement>();
}

PolylineElement::PolylineElement()
    : PolyElement(ElementId::Polyline)
{
}

Path PolylineElement::path() const
{
    auto points = this->points();
    if(points.empty())
        return Path{};

    Path path;
    path.moveTo(points[0].x, points[0].y);
    for(std::size_t i = 1;i < points.size();i++)
        path.lineTo(points[i].x, points[i].y);

    return path;
}

std::unique_ptr<Node> PolylineElement::clone() const
{
    return cloneElement<PolylineElement>();
}

CircleElement::CircleElement()
    : GeometryElement(ElementId::Circle)
{
}

Length CircleElement::cx() const
{
    auto& value = get(PropertyId::Cx);
    if(value.empty())
        return Length{};

    return Parser::parseLength(value, AllowNegativeLengths);
}

Length CircleElement::cy() const
{
    auto& value = get(PropertyId::Cy);
    if(value.empty())
        return Length{};

    return Parser::parseLength(value, AllowNegativeLengths);
}

Length CircleElement::r() const
{
    auto& value = get(PropertyId::R);
    if(value.empty())
        return Length{};

    return Parser::parseLength(value, ForbidNegativeLengths);
}

Path CircleElement::path() const
{
    auto r = this->r();
    if(r.isZero())
        return Path{};

    LengthContext lengthContext(this);
    auto _cx = lengthContext.valueForLength(cx(), LengthMode::Width);
    auto _cy = lengthContext.valueForLength(cy(), LengthMode::Height);
    auto _r = lengthContext.valueForLength(r, LengthMode::Both);

    Path path;
    path.ellipse(_cx, _cy, _r, _r);
    return path;
}

std::unique_ptr<Node> CircleElement::clone() const
{
    return cloneElement<CircleElement>();
}

EllipseElement::EllipseElement()
    : GeometryElement(ElementId::Ellipse)
{
}

Length EllipseElement::cx() const
{
    auto& value = get(PropertyId::Cx);
    if(value.empty())
        return Length{};

    return Parser::parseLength(value, AllowNegativeLengths);
}

Length EllipseElement::cy() const
{
    auto& value = get(PropertyId::Cy);
    if(value.empty())
        return Length{};

    return Parser::parseLength(value, AllowNegativeLengths);
}

Length EllipseElement::rx() const
{
    auto& value = get(PropertyId::Rx);
    if(value.empty())
        return Length{};

    return Parser::parseLength(value, ForbidNegativeLengths);
}

Length EllipseElement::ry() const
{
    auto& value = get(PropertyId::Ry);
    if(value.empty())
        return Length{};

    return Parser::parseLength(value, ForbidNegativeLengths);
}

Path EllipseElement::path() const
{
    auto rx = this->rx();
    auto ry = this->ry();
    if(rx.isZero() || ry.isZero())
        return Path{};

    LengthContext lengthContext(this);
    auto _cx = lengthContext.valueForLength(cx(), LengthMode::Width);
    auto _cy = lengthContext.valueForLength(cy(), LengthMode::Height);
    auto _rx = lengthContext.valueForLength(rx, LengthMode::Width);
    auto _ry = lengthContext.valueForLength(ry, LengthMode::Height);

    Path path;
    path.ellipse(_cx, _cy, _rx, _ry);
    return path;
}

std::unique_ptr<Node> EllipseElement::clone() const
{
    return cloneElement<EllipseElement>();
}

LineElement::LineElement()
    : GeometryElement(ElementId::Line)
{
}

Length LineElement::x1() const
{
    auto& value = get(PropertyId::X1);
    if(value.empty())
        return Length{};

    return Parser::parseLength(value, AllowNegativeLengths);
}

Length LineElement::y1() const
{
    auto& value = get(PropertyId::Y1);
    if(value.empty())
        return Length{};

    return Parser::parseLength(value, AllowNegativeLengths);
}

Length LineElement::x2() const
{
    auto& value = get(PropertyId::X2);
    if(value.empty())
        return Length{};

    return Parser::parseLength(value, AllowNegativeLengths);
}

Length LineElement::y2() const
{
    auto& value = get(PropertyId::Y2);
    if(value.empty())
        return Length{};

    return Parser::parseLength(value, AllowNegativeLengths);
}

Path LineElement::path() const
{
    LengthContext lengthContext(this);
    auto _x1 = lengthContext.valueForLength(x1(), LengthMode::Width);
    auto _y1 = lengthContext.valueForLength(y1(), LengthMode::Height);
    auto _x2 = lengthContext.valueForLength(x2(), LengthMode::Width);
    auto _y2 = lengthContext.valueForLength(y2(), LengthMode::Height);

    Path path;
    path.moveTo(_x1, _y1);
    path.lineTo(_x2, _y2);
    return path;
}

std::unique_ptr<Node> LineElement::clone() const
{
    return cloneElement<LineElement>();
}

RectElement::RectElement()
    : GeometryElement(ElementId::Rect)
{
}

Length RectElement::x() const
{
    auto& value = get(PropertyId::X);
    if(value.empty())
        return Length{};

    return Parser::parseLength(value, AllowNegativeLengths);
}

Length RectElement::y() const
{
    auto& value = get(PropertyId::Y);
    if(value.empty())
        return Length{};

    return Parser::parseLength(value, AllowNegativeLengths);
}

Length RectElement::rx() const
{
    auto& value = get(PropertyId::Rx);
    if(value.empty())
        return Length{0, LengthUnits::Unknown};

    return Parser::parseLength(value, ForbidNegativeLengths);
}

Length RectElement::ry() const
{
    auto& value = get(PropertyId::Ry);
    if(value.empty())
        return Length{0, LengthUnits::Unknown};

    return Parser::parseLength(value, ForbidNegativeLengths);
}

Length RectElement::width() const
{
    auto& value = get(PropertyId::Width);
    if(value.empty())
        return Length{};

    return Parser::parseLength(value, ForbidNegativeLengths);
}

Length RectElement::height() const
{
    auto& value = get(PropertyId::Height);
    if(value.empty())
        return Length{};

    return Parser::parseLength(value, ForbidNegativeLengths);
}

Path RectElement::path() const
{
    auto w = this->width();
    auto h = this->height();
    if(w.isZero() || h.isZero())
        return Path{};

    LengthContext lengthContext(this);
    auto _x = lengthContext.valueForLength(x(), LengthMode::Width);
    auto _y = lengthContext.valueForLength(y(), LengthMode::Height);
    auto _w = lengthContext.valueForLength(w, LengthMode::Width);
    auto _h = lengthContext.valueForLength(h, LengthMode::Height);

    auto rx = this->rx();
    auto ry = this->ry();

    auto _rx = lengthContext.valueForLength(rx, LengthMode::Width);
    auto _ry = lengthContext.valueForLength(ry, LengthMode::Height);

    if(!rx.isValid()) _rx = _ry;
    if(!ry.isValid()) _ry = _rx;

    Path path;
    path.rect(_x, _y, _w, _h, _rx, _ry);
    return path;
}

std::unique_ptr<Node> RectElement::clone() const
{
    return cloneElement<RectElement>();
}
