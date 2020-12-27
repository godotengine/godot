
#pragma once

#include <vector>
#include "EdgeHolder.h"

namespace msdfgen {

/// A single closed contour of a shape.
class Contour {

public:
    /// The sequence of edges that make up the contour.
    std::vector<EdgeHolder> edges;

    /// Adds an edge to the contour.
    void addEdge(const EdgeHolder &edge);
#ifdef MSDFGEN_USE_CPP11
    void addEdge(EdgeHolder &&edge);
#endif
    /// Creates a new edge in the contour and returns its reference.
    EdgeHolder & addEdge();
    /// Adjusts the bounding box to fit the contour.
    void bound(double &l, double &b, double &r, double &t) const;
    /// Adjusts the bounding box to fit the contour border's mitered corners.
    void boundMiters(double &l, double &b, double &r, double &t, double border, double miterLimit, int polarity) const;
    /// Computes the winding of the contour. Returns 1 if positive, -1 if negative.
    int winding() const;
    /// Reverses the sequence of edges on the contour.
    void reverse();

};

}
