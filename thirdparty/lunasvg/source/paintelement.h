#ifndef PAINTELEMENT_H
#define PAINTELEMENT_H

#include "styledelement.h"
#include "canvas.h"

namespace lunasvg {

class LayoutPaint;

class PaintElement : public StyledElement
{
public:
    PaintElement(ElementId id);

    virtual std::unique_ptr<LayoutPaint> getPainter(LayoutContext* context) const = 0;
};

class GradientElement : public PaintElement
{
public:
    GradientElement(ElementId id);

    Transform gradientTransform() const;
    SpreadMethod spreadMethod() const;
    Units gradientUnits() const;
    std::string href() const;
    GradientStops buildGradientStops() const;
};

class LinearGradientElement : public GradientElement
{
public:
    LinearGradientElement();

    Length x1() const;
    Length y1() const;
    Length x2() const;
    Length y2() const;

    std::unique_ptr<LayoutPaint> getPainter(LayoutContext* context) const;
    std::unique_ptr<Node> clone() const;
};

class RadialGradientElement : public GradientElement
{
public:
    RadialGradientElement();

    Length cx() const;
    Length cy() const;
    Length r() const;
    Length fx() const;
    Length fy() const;

    std::unique_ptr<LayoutPaint> getPainter(LayoutContext* context) const;
    std::unique_ptr<Node> clone() const;
};

class PatternElement : public PaintElement
{
public:
    PatternElement();

    Length x() const;
    Length y() const;
    Length width() const;
    Length height() const;
    Transform patternTransform() const;
    Units patternUnits() const;
    Units patternContentUnits() const;

    Rect viewBox() const;
    PreserveAspectRatio preserveAspectRatio() const;
    std::string href() const;

    std::unique_ptr<LayoutPaint> getPainter(LayoutContext* context) const;
    std::unique_ptr<Node> clone() const;
};

class SolidColorElement : public PaintElement
{
public:
    SolidColorElement();

    std::unique_ptr<LayoutPaint> getPainter(LayoutContext*) const;
    std::unique_ptr<Node> clone() const;
};

class GradientAttributes
{
public:
    GradientAttributes() = default;

    const Transform& gradientTransform() const { return m_gradientTransform; }
    SpreadMethod spreadMethod() const { return m_spreadMethod; }
    Units gradientUnits() const { return m_gradientUnits; }
    const GradientStops& gradientStops() const { return m_gradientStops; }

    bool hasGradientTransform() const { return m_hasGradientTransform; }
    bool hasSpreadMethod() const { return m_hasSpreadMethod; }
    bool hasGradientUnits() const { return m_hasGradientUnits; }
    bool hasGradientStops() const { return m_hasGradientStops; }

    void setGradientTransform(const Transform& gradientTransform)
    {
        m_gradientTransform = gradientTransform;
        m_hasGradientTransform = true;
    }

    void setSpreadMethod(SpreadMethod spreadMethod)
    {
        m_spreadMethod = spreadMethod;
        m_hasSpreadMethod = true;
    }

    void setGradientUnits(Units gradientUnits)
    {
        m_gradientUnits = gradientUnits;
        m_hasGradientUnits = true;
    }

    void setGradientStops(const GradientStops& gradientStops)
    {
        m_gradientStops = gradientStops;
        m_hasGradientStops = gradientStops.size();
    }

private:
    Transform m_gradientTransform;
    SpreadMethod m_spreadMethod{SpreadMethod::Pad};
    Units m_gradientUnits{Units::ObjectBoundingBox};
    GradientStops m_gradientStops;

    bool m_hasGradientTransform{false};
    bool m_hasSpreadMethod{false};
    bool m_hasGradientUnits{false};
    bool m_hasGradientStops{false};
};

class LinearGradientAttributes : public GradientAttributes
{
public:
    LinearGradientAttributes() = default;

    const Length& x1() const { return m_x1; }
    const Length& y1() const { return m_y1; }
    const Length& x2() const { return m_x2; }
    const Length& y2() const { return m_y2; }

    bool hasX1() const { return m_hasX1; }
    bool hasY1() const { return m_hasY1; }
    bool hasX2() const { return m_hasX2; }
    bool hasY2() const { return m_hasY2; }

    void setX1(const Length& x1)
    {
        m_x1 = x1;
        m_hasX1 = true;
    }

    void setY1(const Length& y1)
    {
        m_y1 = y1;
        m_hasY1 = true;
    }

    void setX2(const Length& x2)
    {
        m_x2 = x2;
        m_hasX2 = true;
    }

    void setY2(const Length& y2)
    {
        m_y2 = y2;
        m_hasY2 = true;
    }

private:
    Length m_x1;
    Length m_y1;
    Length m_x2{100, LengthUnits::Percent};
    Length m_y2;

    bool m_hasX1{false};
    bool m_hasY1{false};
    bool m_hasX2{false};
    bool m_hasY2{false};
};

class RadialGradientAttributes : public GradientAttributes
{
public:
    RadialGradientAttributes() = default;

    const Length& cx() const { return m_cx; }
    const Length& cy() const { return m_cy; }
    const Length& r() const { return m_r; }
    const Length& fx() const { return m_fx; }
    const Length& fy() const { return m_fy; }

    bool hasCx() const { return m_hasCx; }
    bool hasCy() const { return m_hasCy; }
    bool hasR() const { return m_hasR; }
    bool hasFx() const { return m_hasFx; }
    bool hasFy() const { return m_hasFy; }

    void setCx(const Length& cx)
    {
        m_cx = cx;
        m_hasCx = true;
    }

    void setCy(const Length& cy)
    {
        m_cy = cy;
        m_hasCy = true;
    }

    void setR(const Length& r)
    {
        m_r = r;
        m_hasR = true;
    }

    void setFx(const Length& fx)
    {
        m_fx = fx;
        m_hasFx = true;
    }

    void setFy(const Length& fy)
    {
        m_fy = fy;
        m_hasFy = true;
    }


private:
    Length m_cx{50, LengthUnits::Percent};
    Length m_cy{50, LengthUnits::Percent};
    Length m_r{50, LengthUnits::Percent};
    Length m_fx;
    Length m_fy;

    bool m_hasCx{false};
    bool m_hasCy{false};
    bool m_hasR{false};
    bool m_hasFx{false};
    bool m_hasFy{false};
};

class PatternAttributes
{
public:
    PatternAttributes() = default;

    const Length& x() const { return m_x; }
    const Length& y() const { return m_y; }
    const Length& width() const { return m_width; }
    const Length& height() const { return m_height; }
    const Transform& patternTransform() const { return m_patternTransform; }
    Units patternUnits() const { return m_patternUnits; }
    Units patternContentUnits() const { return m_patternContentUnits; }
    const Rect& viewBox() const { return m_viewBox; }
    const PreserveAspectRatio& preserveAspectRatio() const { return m_preserveAspectRatio; }
    const PatternElement* patternContentElement() const { return m_patternContentElement; }

    bool hasX() const { return m_hasX; }
    bool hasY() const { return m_hasY; }
    bool hasWidth() const { return m_hasWidth; }
    bool hasHeight() const { return m_hasHeight; }
    bool hasPatternTransform() const { return m_hasPatternTransform; }
    bool hasPatternUnits() const { return m_hasPatternUnits; }
    bool hasPatternContentUnits() const { return m_hasPatternContentUnits; }
    bool hasViewBox() const { return m_hasViewBox; }
    bool hasPreserveAspectRatio() const { return m_hasPreserveAspectRatio; }
    bool hasPatternContentElement() const { return m_hasPatternContentElement; }

    void setX(const Length& x)
    {
        m_x = x;
        m_hasX = true;
    }

    void setY(const Length& y)
    {
        m_y = y;
        m_hasY = true;
    }

    void setWidth(const Length& width)
    {
        m_width = width;
        m_hasWidth = true;
    }

    void setHeight(const Length& height)
    {
        m_height = height;
        m_hasHeight = true;
    }

    void setPatternTransform(const Transform& patternTransform)
    {
        m_patternTransform = patternTransform;
        m_hasPatternTransform = true;
    }

    void setPatternUnits(Units patternUnits)
    {
        m_patternUnits = patternUnits;
        m_hasPatternUnits = true;
    }

    void setPatternContentUnits(Units patternContentUnits)
    {
        m_patternContentUnits = patternContentUnits;
        m_hasPatternContentUnits = true;
    }

    void setViewBox(const Rect& viewBox)
    {
        m_viewBox = viewBox;
        m_hasViewBox = true;
    }

    void setPreserveAspectRatio(const PreserveAspectRatio& preserveAspectRatio)
    {
        m_preserveAspectRatio = preserveAspectRatio;
        m_hasPreserveAspectRatio = true;
    }

    void setPatternContentElement(const PatternElement* patternContentElement)
    {
        m_patternContentElement = patternContentElement;
        m_hasPatternContentElement = true;
    }

private:
    Length m_x;
    Length m_y;
    Length m_width;
    Length m_height;
    Transform m_patternTransform;
    Units m_patternUnits{Units::ObjectBoundingBox};
    Units m_patternContentUnits{Units::UserSpaceOnUse};
    Rect m_viewBox;
    PreserveAspectRatio m_preserveAspectRatio;
    const PatternElement* m_patternContentElement{nullptr};

    bool m_hasX{false};
    bool m_hasY{false};
    bool m_hasWidth{false};
    bool m_hasHeight{false};
    bool m_hasPatternTransform{false};
    bool m_hasPatternUnits{false};
    bool m_hasPatternContentUnits{false};
    bool m_hasViewBox{false};
    bool m_hasPreserveAspectRatio{false};
    bool m_hasPatternContentElement{false};
};

} // namespace lunasvg

#endif // PAINTELEMENT_H
