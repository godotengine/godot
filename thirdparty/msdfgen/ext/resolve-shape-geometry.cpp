
#include "resolve-shape-geometry.h"

#ifdef MSDFGEN_USE_SKIA

#include <core/SkPath.h>
#include <pathops/SkPathOps.h>
#include "../core/Vector2.h"
#include "../core/edge-segments.h"
#include "../core/Contour.h"

namespace msdfgen {

SkPoint pointToSkiaPoint(Point2 p) {
    return SkPoint::Make((SkScalar) p.x, (SkScalar) p.y);
}

Point2 pointFromSkiaPoint(const SkPoint p) {
    return Point2((double) p.x(), (double) p.y());
}

void shapeToSkiaPath(SkPath &skPath, const Shape &shape) {
    for (std::vector<Contour>::const_iterator contour = shape.contours.begin(); contour != shape.contours.end(); ++contour) {
        if (!contour->edges.empty()) {
            skPath.moveTo(pointToSkiaPoint(contour->edges.front()->point(0)));
            for (std::vector<EdgeHolder>::const_iterator edge = contour->edges.begin(); edge != contour->edges.end(); ++edge) {
                if (const LinearSegment *linearSegment = dynamic_cast<const LinearSegment *>(&**edge))
                    skPath.lineTo(pointToSkiaPoint(linearSegment->p[1]));
                else if (const QuadraticSegment *quadraticSegment = dynamic_cast<const QuadraticSegment *>(&**edge))
                    skPath.quadTo(pointToSkiaPoint(quadraticSegment->p[1]), pointToSkiaPoint(quadraticSegment->p[2]));
                else if (const CubicSegment *cubicSegment = dynamic_cast<const CubicSegment *>(&**edge))
                    skPath.cubicTo(pointToSkiaPoint(cubicSegment->p[1]), pointToSkiaPoint(cubicSegment->p[2]), pointToSkiaPoint(cubicSegment->p[3]));
            }
        }
    }
}

void shapeFromSkiaPath(Shape &shape, const SkPath &skPath) {
    shape.contours.clear();
    Contour *contour = &shape.addContour();
    SkPath::Iter pathIterator(skPath, true);
    SkPoint edgePoints[4];
    for (SkPath::Verb op; (op = pathIterator.next(edgePoints)) != SkPath::kDone_Verb;) {
        switch (op) {
            case SkPath::kMove_Verb:
                if (!contour->edges.empty())
                    contour = &shape.addContour();
                break;
            case SkPath::kLine_Verb:
                contour->addEdge(new LinearSegment(pointFromSkiaPoint(edgePoints[0]), pointFromSkiaPoint(edgePoints[1])));
                break;
            case SkPath::kQuad_Verb:
                contour->addEdge(new QuadraticSegment(pointFromSkiaPoint(edgePoints[0]), pointFromSkiaPoint(edgePoints[1]), pointFromSkiaPoint(edgePoints[2])));
                break;
            case SkPath::kCubic_Verb:
                contour->addEdge(new CubicSegment(pointFromSkiaPoint(edgePoints[0]), pointFromSkiaPoint(edgePoints[1]), pointFromSkiaPoint(edgePoints[2]), pointFromSkiaPoint(edgePoints[3])));
                break;
            default:;
        }
    }
    if (contour->edges.empty())
        shape.contours.pop_back();
}

bool resolveShapeGeometry(Shape &shape) {
    SkPath skPath;
    shapeToSkiaPath(skPath, shape);
    if (!Simplify(skPath, &skPath))
        return false;
    // Skia's AsWinding doesn't seem to work for unknown reasons
    shapeFromSkiaPath(shape, skPath);
    shape.orientContours();
    return true;
}

}

#endif
