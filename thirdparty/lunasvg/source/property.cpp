#include "property.h"
#include "styledelement.h"

#include <cmath>

using namespace lunasvg;

const Color Color::Black = {0, 0, 0, 1};
const Color Color::White = {1, 1, 1, 1};
const Color Color::Red = {1, 0, 0, 1};
const Color Color::Green = {0, 1, 0, 1};
const Color Color::Blue = {0, 0, 1, 1};
const Color Color::Yellow = {1, 1, 0, 1};
const Color Color::Transparent = {0, 0, 0, 0};

Color::Color(double r, double g, double b, double a)
    : r(r), g(g), b(b), a(a)
{
}

Paint::Paint(const Color& color)
    : m_color(color)
{
}

Paint::Paint(const std::string& ref)
    : m_ref(ref)
{
}

Point::Point(double x, double y)
    : x(x), y(y)
{
}

Rect::Rect(double x, double y, double w, double h)
    : x(x), y(y), w(w), h(h)
{
}

Transform::Transform(double m00, double m10, double m01, double m11, double m02, double m12)
    : m00(m00), m10(m10), m01(m01), m11(m11), m02(m02), m12(m12)
{
}

Transform Transform::inverted() const
{
    double det = (this->m00 * this->m11 - this->m10 * this->m01);
    if(det == 0.0)
        return Transform{};

    double inv_det = 1.0 / det;
    double m00 = this->m00 * inv_det;
    double m10 = this->m10 * inv_det;
    double m01 = this->m01 * inv_det;
    double m11 = this->m11 * inv_det;
    double m02 = (this->m01 * this->m12 - this->m11 * this->m02) * inv_det;
    double m12 = (this->m10 * this->m02 - this->m00 * this->m12) * inv_det;

    return Transform{m11, -m10, -m01, m00, m02, m12};
}

Transform Transform::operator*(const Transform& transform) const
{
    double m00 = this->m00 * transform.m00 + this->m10 * transform.m01;
    double m10 = this->m00 * transform.m10 + this->m10 * transform.m11;
    double m01 = this->m01 * transform.m00 + this->m11 * transform.m01;
    double m11 = this->m01 * transform.m10 + this->m11 * transform.m11;
    double m02 = this->m02 * transform.m00 + this->m12 * transform.m01 + transform.m02;
    double m12 = this->m02 * transform.m10 + this->m12 * transform.m11 + transform.m12;

    return Transform{m00, m10, m01, m11, m02, m12};
}

Transform& Transform::operator*=(const Transform& transform)
{
    *this = *this * transform;
    return *this;
}

Transform& Transform::premultiply(const Transform& transform)
{
    *this = transform * *this;
    return *this;
}

Transform& Transform::postmultiply(const Transform& transform)
{
    *this = *this * transform;
    return *this;
}

Transform& Transform::rotate(double angle)
{
    *this = rotated(angle) * *this;
    return *this;
}

Transform& Transform::rotate(double angle, double cx, double cy)
{
    *this = rotated(angle, cx, cy) * *this;
    return *this;
}

Transform& Transform::scale(double sx, double sy)
{
    *this = scaled(sx, sy) * *this;
    return *this;
}

Transform& Transform::shear(double shx, double shy)
{
    *this = sheared(shx, shy) * *this;
    return *this;
}

Transform& Transform::translate(double tx, double ty)
{
    *this = translated(tx, ty) * *this;
    return *this;
}

Transform& Transform::transform(double m00, double m10, double m01, double m11, double m02, double m12)
{
    *this = Transform{m00, m10, m01, m11, m02, m12} * *this;
    return *this;
}

Transform& Transform::identity()
{
    *this = Transform{1, 0, 0, 1, 0, 0};
    return *this;
}

Transform& Transform::invert()
{
    *this = inverted();
    return *this;
}

void Transform::map(double x, double y, double* _x, double* _y) const
{
    *_x = x * m00 + y * m01 + m02;
    *_y = x * m10 + y * m11 + m12;
}

Point Transform::map(const Point& point) const
{
    Point p;
    map(point.x, point.y, &p.x, &p.y);
    return p;
}

Rect Transform::map(const Rect& rect) const
{
    auto left = rect.x;
    auto top = rect.y;
    auto right = rect.x + rect.w;
    auto bottom = rect.y + rect.h;

    Point p[4];
    p[0] = map(Point{left, top});
    p[1] = map(Point{right, top});
    p[2] = map(Point{right, bottom});
    p[3] = map(Point{left, bottom});

    auto l = p[0].x;
    auto t = p[0].y;
    auto r = p[0].x;
    auto b = p[0].y;

    for(int i = 1;i < 4;i++)
    {
        if(p[i].x < l) l = p[i].x;
        if(p[i].x > r) r = p[i].x;
        if(p[i].y < t) t = p[i].y;
        if(p[i].y > b) b = p[i].y;
    }

    return Rect{l, t, r-l, b-t};
}

static const double pi = 3.14159265358979323846;

Transform Transform::rotated(double angle)
{
    auto c = std::cos(angle * pi / 180.0);
    auto s = std::sin(angle * pi / 180.0);

    return Transform{c, s, -s, c, 0, 0};
}

Transform Transform::rotated(double angle, double cx, double cy)
{
    auto c = std::cos(angle * pi / 180.0);
    auto s = std::sin(angle * pi / 180.0);

    auto x = cx * (1 - c) + cy * s;
    auto y = cy * (1 - c) - cx * s;

    return Transform{c, s, -s, c, x, y};
}

Transform Transform::scaled(double sx, double sy)
{
    return Transform{sx, 0, 0, sy, 0, 0};
}

Transform Transform::sheared(double shx, double shy)
{
    auto x = std::tan(shx * pi / 180.0);
    auto y = std::tan(shy * pi / 180.0);

    return Transform{1, y, x, 1, 0, 0};
}

Transform Transform::translated(double tx, double ty)
{
    return Transform{1, 0, 0, 1, tx, ty};
}

void Path::moveTo(double x, double y)
{
    m_commands.push_back(PathCommand::MoveTo);
    m_points.emplace_back(x, y);
}

void Path::lineTo(double x, double y)
{
    m_commands.push_back(PathCommand::LineTo);
    m_points.emplace_back(x, y);
}

void Path::cubicTo(double x1, double y1, double x2, double y2, double x3, double y3)
{
    m_commands.push_back(PathCommand::CubicTo);
    m_points.emplace_back(x1, y1);
    m_points.emplace_back(x2, y2);
    m_points.emplace_back(x3, y3);
}

void Path::close()
{
    if(m_commands.empty())
        return;

    if(m_commands.back() == PathCommand::Close)
        return;

    m_commands.push_back(PathCommand::Close);
}

void Path::reset()
{
    m_commands.clear();
    m_points.clear();
}

bool Path::empty() const
{
    return m_commands.empty();
}

void Path::quadTo(double cx, double cy, double x1, double y1, double x2, double y2)
{
    auto cx1 = 2.0 / 3.0 * x1 + 1.0 / 3.0 * cx;
    auto cy1 = 2.0 / 3.0 * y1 + 1.0 / 3.0 * cy;
    auto cx2 = 2.0 / 3.0 * x1 + 1.0 / 3.0 * x2;
    auto cy2 = 2.0 / 3.0 * y1 + 1.0 / 3.0 * y2;
    cubicTo(cx1, cy1, cx2, cy2, x2, y2);
}

void Path::arcTo(double cx, double cy, double rx, double ry, double xAxisRotation, bool largeArcFlag, bool sweepFlag, double x, double y)
{
    rx = std::fabs(rx);
    ry = std::fabs(ry);

    auto sin_th = std::sin(xAxisRotation * pi / 180.0);
    auto cos_th = std::cos(xAxisRotation * pi / 180.0);

    auto dx = (cx - x) / 2.0;
    auto dy = (cy - y) / 2.0;
    auto dx1 =  cos_th * dx + sin_th * dy;
    auto dy1 = -sin_th * dx + cos_th * dy;
    auto Pr1 = rx * rx;
    auto Pr2 = ry * ry;
    auto Px = dx1 * dx1;
    auto Py = dy1 * dy1;
    auto check = Px / Pr1 + Py / Pr2;
    if(check > 1)
    {
        rx = rx * std::sqrt(check);
        ry = ry * std::sqrt(check);
    }

    auto a00 =  cos_th / rx;
    auto a01 =  sin_th / rx;
    auto a10 = -sin_th / ry;
    auto a11 =  cos_th / ry;
    auto x0 = a00 * cx + a01 * cy;
    auto y0 = a10 * cx + a11 * cy;
    auto x1 = a00 * x + a01 * y;
    auto y1 = a10 * x + a11 * y;
    auto d = (x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0);
    auto sfactor_sq = 1.0 / d - 0.25;
    if(sfactor_sq < 0) sfactor_sq = 0;
    auto sfactor = std::sqrt(sfactor_sq);
    if(sweepFlag == largeArcFlag) sfactor = -sfactor;
    auto xc = 0.5 * (x0 + x1) - sfactor * (y1 - y0);
    auto yc = 0.5 * (y0 + y1) + sfactor * (x1 - x0);

    auto th0 = std::atan2(y0 - yc, x0 - xc);
    auto th1 = std::atan2(y1 - yc, x1 - xc);

    double th_arc = th1 - th0;
    if(th_arc < 0.0 && sweepFlag)
        th_arc += 2.0 * pi;
    else if(th_arc > 0.0 && !sweepFlag)
        th_arc -= 2.0 * pi;

    auto n_segs = static_cast<int>(std::ceil(std::fabs(th_arc / (pi * 0.5 + 0.001))));
    for(int i = 0;i < n_segs;i++)
    {
        auto th2 = th0 + i * th_arc / n_segs;
        auto th3 = th0 + (i + 1) * th_arc / n_segs;

        auto a00 =  cos_th * rx;
        auto a01 = -sin_th * ry;
        auto a10 =  sin_th * rx;
        auto a11 =  cos_th * ry;

        auto thHalf = 0.5 * (th3 - th2);
        auto t = (8.0 / 3.0) * std::sin(thHalf * 0.5) * std::sin(thHalf * 0.5) / std::sin(thHalf);
        auto x1 = xc + std::cos(th2) - t * std::sin(th2);
        auto y1 = yc + std::sin(th2) + t * std::cos(th2);
        auto x3 = xc + std::cos(th3);
        auto y3 = yc + std::sin(th3);
        auto x2 = x3 + t * std::sin(th3);
        auto y2 = y3 - t * std::cos(th3);

        auto cx1 = a00 * x1 + a01 * y1;
        auto cy1 = a10 * x1 + a11 * y1;
        auto cx2 = a00 * x2 + a01 * y2;
        auto cy2 = a10 * x2 + a11 * y2;
        auto cx3 = a00 * x3 + a01 * y3;
        auto cy3 = a10 * x3 + a11 * y3;
        cubicTo(cx1, cy1, cx2, cy2, cx3, cy3);
    }
}

static const double kappa = 0.55228474983079339840;

void Path::ellipse(double cx, double cy, double rx, double ry)
{
    auto left = cx - rx;
    auto top = cy - ry;
    auto right = cx + rx;
    auto bottom = cy + ry;

    auto cpx = rx * kappa;
    auto cpy = ry * kappa;

    moveTo(cx, top);
    cubicTo(cx+cpx, top, right, cy-cpy, right, cy);
    cubicTo(right, cy+cpy, cx+cpx, bottom, cx, bottom);
    cubicTo(cx-cpx, bottom, left, cy+cpy, left, cy);
    cubicTo(left, cy-cpy, cx-cpx, top, cx, top);
    close();
}

void Path::rect(double x, double y, double w, double h, double rx, double ry)
{
    rx = std::min(rx, w * 0.5);
    ry = std::min(ry, h * 0.5);

    auto right = x + w;
    auto bottom = y + h;

    if(rx == 0.0 && ry == 0.0)
    {
        moveTo(x, y);
        lineTo(right, y);
        lineTo(right, bottom);
        lineTo(x, bottom);
        lineTo(x, y);
        close();
    }
    else
    {
        double cpx = rx * kappa;
        double cpy = ry * kappa;
        moveTo(x, y+ry);
        cubicTo(x, y+ry-cpy, x+rx-cpx, y, x+rx, y);
        lineTo(right-rx, y);
        cubicTo(right-rx+cpx, y, right, y+ry-cpy, right, y+ry);
        lineTo(right, bottom-ry);
        cubicTo(right, bottom-ry+cpy, right-rx+cpx, bottom, right-rx, bottom);
        lineTo(x+rx, bottom);
        cubicTo(x+rx-cpx, bottom, x, bottom-ry+cpy, x, bottom-ry);
        lineTo(x, y+ry);
        close();
    }
}

Rect Path::box() const
{
    if(m_points.empty())
        return Rect{};

    auto l = m_points[0].x;
    auto t = m_points[0].y;
    auto r = m_points[0].x;
    auto b = m_points[0].y;

    for(std::size_t i = 1;i < m_points.size();i++)
    {
        if(m_points[i].x < l) l = m_points[i].x;
        if(m_points[i].x > r) r = m_points[i].x;
        if(m_points[i].y < t) t = m_points[i].y;
        if(m_points[i].y > b) b = m_points[i].y;
    }

    return Rect{l, t, r-l, b-t};
}

PathIterator::PathIterator(const Path& path)
    : m_commands(path.commands()),
      m_points(path.points().data())
{
}

PathCommand PathIterator::currentSegment(std::array<Point, 3>& points) const
{
    auto command = m_commands[m_index];
    switch(command) {
    case PathCommand::MoveTo:
        points[0] = m_points[0];
        m_startPoint = points[0];
        break;
    case PathCommand::LineTo:
        points[0] = m_points[0];
        break;
    case PathCommand::CubicTo:
        points[0] = m_points[0];
        points[1] = m_points[1];
        points[2] = m_points[2];
        break;
    case PathCommand::Close:
        points[0] = m_startPoint;
        break;
    }

    return command;
}

bool PathIterator::isDone() const
{
    return (m_index >= m_commands.size());
}

void PathIterator::next()
{
    switch(m_commands[m_index]) {
    case PathCommand::MoveTo:
    case PathCommand::LineTo:
        m_points += 1;
        break;
    case PathCommand::CubicTo:
        m_points += 3;
        break;
    default:
        break;
    }

    m_index += 1;
}

Length::Length(double value)
    : m_value(value)
{
}

Length::Length(double value, LengthUnits units)
    : m_value(value), m_units(units)
{
}

static const double dpi = 96.0;

double Length::value(double max) const
{
    switch(m_units) {
    case LengthUnits::Number:
    case LengthUnits::Px:
        return m_value;
    case LengthUnits::In:
        return m_value * dpi;
    case LengthUnits::Cm:
        return m_value * dpi / 2.54;
    case LengthUnits::Mm:
        return m_value * dpi / 25.4;
    case LengthUnits::Pt:
        return m_value * dpi / 72.0;
    case LengthUnits::Pc:
        return m_value * dpi / 6.0;
    case LengthUnits::Percent:
        return m_value * max / 100.0;
    default:
        break;
    }

    return 0.0;
}

static const double sqrt2 = 1.41421356237309504880;

double Length::value(const Element* element, LengthMode mode) const
{
    if(m_units == LengthUnits::Percent)
    {
        auto viewBox = element->nearestViewBox();
        auto w = viewBox.w;
        auto h = viewBox.h;
        auto max = (mode == LengthMode::Width) ? w : (mode == LengthMode::Height) ? h : std::sqrt(w*w+h*h) / sqrt2;
        return m_value * max / 100.0;
    }

    return value(1.0);
}

LengthContext::LengthContext(const Element* element)
    : m_element(element)
{
}

LengthContext::LengthContext(const Element* element, Units units)
    : m_element(element), m_units(units)
{
}

double LengthContext::valueForLength(const Length& length, LengthMode mode) const
{
    if(m_units == Units::ObjectBoundingBox)
        return length.value(1.0);
    return length.value(m_element, mode);
}

PreserveAspectRatio::PreserveAspectRatio(Align align, MeetOrSlice scale)
    : m_align(align), m_scale(scale)
{
}

Transform PreserveAspectRatio::getMatrix(const Rect& viewPort, const Rect& viewBox) const
{
    auto matrix = Transform::translated(viewPort.x, viewPort.y);
    if(viewBox.w == 0.0 || viewBox.h == 0.0)
        return matrix;

    auto scaleX = viewPort.w  / viewBox.w;
    auto scaleY = viewPort.h  / viewBox.h;
    if(scaleX == 0.0 || scaleY == 0.0)
        return matrix;

    auto transX = -viewBox.x;
    auto transY = -viewBox.y;
    if(m_align == Align::None)
    {
        matrix.scale(scaleX, scaleY);
        matrix.translate(transX, transY);
        return matrix;
    }

    auto scale = (m_scale == MeetOrSlice::Meet) ? std::min(scaleX, scaleY) : std::max(scaleX, scaleY);
    auto viewW = viewPort.w / scale;
    auto viewH = viewPort.h / scale;

    switch(m_align) {
    case Align::xMidYMin:
    case Align::xMidYMid:
    case Align::xMidYMax:
        transX -= (viewBox.w - viewW) * 0.5;
        break;
    case Align::xMaxYMin:
    case Align::xMaxYMid:
    case Align::xMaxYMax:
        transX -= (viewBox.w - viewW);
        break;
    default:
        break;
    }

    switch(m_align) {
    case Align::xMinYMid:
    case Align::xMidYMid:
    case Align::xMaxYMid:
        transY -= (viewBox.h - viewH) * 0.5;
        break;
    case Align::xMinYMax:
    case Align::xMidYMax:
    case Align::xMaxYMax:
        transY -= (viewBox.h - viewH);
        break;
    default:
        break;
    }

    matrix.scale(scale, scale);
    matrix.translate(transX, transY);
    return matrix;
}

Angle::Angle(MarkerOrient type)
    : m_type(type)
{
}

Angle::Angle(double value, MarkerOrient type)
    : m_value(value), m_type(type)
{
}
