#ifndef CANVAS_H
#define CANVAS_H

#include "property.h"

#include <memory>

namespace lunasvg {

using GradientStop = std::pair<double, Color>;
using GradientStops = std::vector<GradientStop>;

class LinearGradientValues
{
public:
    LinearGradientValues(double x1, double y1, double x2, double y2);

public:
    double x1;
    double y1;
    double x2;
    double y2;
};

class RadialGradientValues
{
public:
    RadialGradientValues(double cx, double cy, double r, double fx, double fy);

public:
    double cx;
    double cy;
    double r;
    double fx;
    double fy;
};

using DashArray = std::vector<double>;

class DashData
{
public:
    DashData() = default;

public:
    DashArray array;
    double offset{1.0};
};

enum class TileMode
{
    Plain,
    Tiled
};

enum class BlendMode
{
    Src,
    Src_Over,
    Dst_In,
    Dst_Out
};

class CanvasImpl;

class Canvas
{
public:
    static std::shared_ptr<Canvas> create(unsigned int width, unsigned int height);
    static std::shared_ptr<Canvas> create(unsigned char* data, unsigned int width, unsigned int height, unsigned int stride);

    void setMatrix(const Transform& matrix);
    void setOpacity(double opacity);
    void setColor(const Color& color);
    void setGradient(const LinearGradientValues& values, const Transform& matrix, SpreadMethod spread, const GradientStops& stops);
    void setGradient(const RadialGradientValues& values, const Transform& matrix, SpreadMethod spread, const GradientStops& stops);
    void setPattern(const Canvas& tile, const Transform& matrix, TileMode mode);
    void setWinding(WindRule winding);
    void setLineWidth(double width);
    void setLineCap(LineCap cap);
    void setLineJoin(LineJoin join);
    void setMiterlimit(double miterlimit);
    void setDash(const DashData& dash);

    void fill(const Path& path);
    void stroke(const Path& path);
    void blend(const Canvas& source, BlendMode mode, double opacity);

    void clear(unsigned int value);
    void rgba();
    void luminance();

    unsigned int width() const;
    unsigned int height() const;
    unsigned int stride() const;
    unsigned char* data() const;

    ~Canvas();

private:
    Canvas(unsigned int width, unsigned int height);
    Canvas(unsigned char* data, unsigned int width, unsigned int height, unsigned int stride);

    std::unique_ptr<CanvasImpl> d;
};

} // namespace lunasvg

#endif // CANVAS_H
