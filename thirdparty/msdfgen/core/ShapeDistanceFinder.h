
#pragma once

#include <vector>
#include "Vector2.h"
#include "edge-selectors.h"
#include "contour-combiners.h"

namespace msdfgen {

/// Finds the distance between a point and a Shape. ContourCombiner dictates the distance metric and its data type.
template <class ContourCombiner>
class ShapeDistanceFinder {

public:
    typedef typename ContourCombiner::DistanceType DistanceType;

    // Passed shape object must persist until the distance finder is destroyed!
    explicit ShapeDistanceFinder(const Shape &shape);
    /// Finds the distance from origin. Not thread-safe! Is fastest when subsequent queries are close together.
    DistanceType distance(const Point2 &origin);

    /// Finds the distance between shape and origin. Does not allocate result cache used to optimize performance of multiple queries.
    static DistanceType oneShotDistance(const Shape &shape, const Point2 &origin);

private:
    const Shape &shape;
    ContourCombiner contourCombiner;
    std::vector<typename ContourCombiner::EdgeSelectorType::EdgeCache> shapeEdgeCache;

};

typedef ShapeDistanceFinder<SimpleContourCombiner<TrueDistanceSelector> > SimpleTrueShapeDistanceFinder;

}

#include "ShapeDistanceFinder.hpp"
