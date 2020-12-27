
#pragma once

#include <vector>

namespace msdfgen {

/// Fill rule dictates how intersection total is interpreted during rasterization.
enum FillRule {
    FILL_NONZERO,
    FILL_ODD, // "even-odd"
    FILL_POSITIVE,
    FILL_NEGATIVE
};

/// Resolves the number of intersection into a binary fill value based on fill rule.
bool interpretFillRule(int intersections, FillRule fillRule);

/// Represents a horizontal scanline intersecting a shape.
class Scanline {

public:
    /// An intersection with the scanline.
    struct Intersection {
        /// X coordinate.
        double x;
        /// Normalized Y direction of the oriented edge at the point of intersection.
        int direction;
    };

    static double overlap(const Scanline &a, const Scanline &b, double xFrom, double xTo, FillRule fillRule);

    Scanline();
    /// Populates the intersection list.
    void setIntersections(const std::vector<Intersection> &intersections);
#ifdef MSDFGEN_USE_CPP11
    void setIntersections(std::vector<Intersection> &&intersections);
#endif
    /// Returns the number of intersections left of x.
    int countIntersections(double x) const;
    /// Returns the total sign of intersections left of x.
    int sumIntersections(double x) const;
    /// Decides whether the scanline is filled at x based on fill rule.
    bool filled(double x, FillRule fillRule) const;

private:
    std::vector<Intersection> intersections;
    mutable int lastIndex;

    void preprocess();
    int moveTo(double x) const;

};

}
