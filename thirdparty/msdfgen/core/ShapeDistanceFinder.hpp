
#include "ShapeDistanceFinder.h"

namespace msdfgen {

template <class ContourCombiner>
ShapeDistanceFinder<ContourCombiner>::ShapeDistanceFinder(const Shape &shape) : shape(shape), contourCombiner(shape), shapeEdgeCache(shape.edgeCount()) { }

template <class ContourCombiner>
typename ShapeDistanceFinder<ContourCombiner>::DistanceType ShapeDistanceFinder<ContourCombiner>::distance(const Point2 &origin) {
    contourCombiner.reset(origin);
    typename ContourCombiner::EdgeSelectorType::EdgeCache *edgeCache = &shapeEdgeCache[0];

    for (std::vector<Contour>::const_iterator contour = shape.contours.begin(); contour != shape.contours.end(); ++contour) {
        if (!contour->edges.empty()) {
            typename ContourCombiner::EdgeSelectorType &edgeSelector = contourCombiner.edgeSelector(int(contour-shape.contours.begin()));

            const EdgeSegment *prevEdge = contour->edges.size() >= 2 ? *(contour->edges.end()-2) : *contour->edges.begin();
            const EdgeSegment *curEdge = contour->edges.back();
            for (std::vector<EdgeHolder>::const_iterator edge = contour->edges.begin(); edge != contour->edges.end(); ++edge) {
                const EdgeSegment *nextEdge = *edge;
                edgeSelector.addEdge(*edgeCache++, prevEdge, curEdge, nextEdge);
                prevEdge = curEdge;
                curEdge = nextEdge;
            }
        }
    }

    return contourCombiner.distance();
}

template <class ContourCombiner>
typename ShapeDistanceFinder<ContourCombiner>::DistanceType ShapeDistanceFinder<ContourCombiner>::oneShotDistance(const Shape &shape, const Point2 &origin) {
    ContourCombiner contourCombiner(shape);
    contourCombiner.reset(origin);

    for (std::vector<Contour>::const_iterator contour = shape.contours.begin(); contour != shape.contours.end(); ++contour) {
        if (!contour->edges.empty()) {
            typename ContourCombiner::EdgeSelectorType &edgeSelector = contourCombiner.edgeSelector(int(contour-shape.contours.begin()));

            const EdgeSegment *prevEdge = contour->edges.size() >= 2 ? *(contour->edges.end()-2) : *contour->edges.begin();
            const EdgeSegment *curEdge = contour->edges.back();
            for (std::vector<EdgeHolder>::const_iterator edge = contour->edges.begin(); edge != contour->edges.end(); ++edge) {
                const EdgeSegment *nextEdge = *edge;
                typename ContourCombiner::EdgeSelectorType::EdgeCache dummy;
                edgeSelector.addEdge(dummy, prevEdge, curEdge, nextEdge);
                prevEdge = curEdge;
                curEdge = nextEdge;
            }
        }
    }

    return contourCombiner.distance();
}

}
