
#include "../msdfgen.h"

#include <vector>
#include "edge-selectors.h"
#include "contour-combiners.h"
#include "ShapeDistanceFinder.h"
#include "msdf-edge-artifact-patcher.h"

namespace msdfgen {

template <typename DistanceType>
class DistancePixelConversion;

template <>
class DistancePixelConversion<double> {
public:
    typedef BitmapRef<float, 1> BitmapRefType;
    inline static void convert(float *pixels, double distance, double range) {
        *pixels = float(distance/range+.5);
    }
};

template <>
class DistancePixelConversion<MultiDistance> {
public:
    typedef BitmapRef<float, 3> BitmapRefType;
    inline static void convert(float *pixels, const MultiDistance &distance, double range) {
        pixels[0] = float(distance.r/range+.5);
        pixels[1] = float(distance.g/range+.5);
        pixels[2] = float(distance.b/range+.5);
    }
};

template <>
class DistancePixelConversion<MultiAndTrueDistance> {
public:
    typedef BitmapRef<float, 4> BitmapRefType;
    inline static void convert(float *pixels, const MultiAndTrueDistance &distance, double range) {
        pixels[0] = float(distance.r/range+.5);
        pixels[1] = float(distance.g/range+.5);
        pixels[2] = float(distance.b/range+.5);
        pixels[3] = float(distance.a/range+.5);
    }
};

template <class ContourCombiner>
void generateDistanceField(const typename DistancePixelConversion<typename ContourCombiner::DistanceType>::BitmapRefType &output, const Shape &shape, double range, const Vector2 &scale, const Vector2 &translate) {
#ifdef MSDFGEN_USE_OPENMP
    #pragma omp parallel
#endif
    {
        ShapeDistanceFinder<ContourCombiner> distanceFinder(shape);
        bool rightToLeft = false;
        Point2 p;
#ifdef MSDFGEN_USE_OPENMP
        #pragma omp for
#endif
        for (int y = 0; y < output.height; ++y) {
            int row = shape.inverseYAxis ? output.height-y-1 : y;
            p.y = (y+.5)/scale.y-translate.y;
            for (int col = 0; col < output.width; ++col) {
                int x = rightToLeft ? output.width-col-1 : col;
                p.x = (x+.5)/scale.x-translate.x;
                typename ContourCombiner::DistanceType distance = distanceFinder.distance(p);
                DistancePixelConversion<typename ContourCombiner::DistanceType>::convert(output(x, row), distance, range);
            }
            rightToLeft = !rightToLeft;
        }
    }
}

void generateSDF(const BitmapRef<float, 1> &output, const Shape &shape, double range, const Vector2 &scale, const Vector2 &translate, bool overlapSupport) {
    if (overlapSupport)
        generateDistanceField<OverlappingContourCombiner<TrueDistanceSelector> >(output, shape, range, scale, translate);
    else
        generateDistanceField<SimpleContourCombiner<TrueDistanceSelector> >(output, shape, range, scale, translate);
}

void generatePseudoSDF(const BitmapRef<float, 1> &output, const Shape &shape, double range, const Vector2 &scale, const Vector2 &translate, bool overlapSupport) {
    if (overlapSupport)
        generateDistanceField<OverlappingContourCombiner<PseudoDistanceSelector> >(output, shape, range, scale, translate);
    else
        generateDistanceField<SimpleContourCombiner<PseudoDistanceSelector> >(output, shape, range, scale, translate);
}

void generateMSDF(const BitmapRef<float, 3> &output, const Shape &shape, double range, const Vector2 &scale, const Vector2 &translate, double edgeThreshold, bool overlapSupport) {
    if (overlapSupport)
        generateDistanceField<OverlappingContourCombiner<MultiDistanceSelector> >(output, shape, range, scale, translate);
    else
        generateDistanceField<SimpleContourCombiner<MultiDistanceSelector> >(output, shape, range, scale, translate);
    if (edgeThreshold > 0)
        msdfErrorCorrection(output, edgeThreshold/(scale*range));
    msdfPatchEdgeArtifacts(output, shape, range, scale, translate, overlapSupport);
}

void generateMTSDF(const BitmapRef<float, 4> &output, const Shape &shape, double range, const Vector2 &scale, const Vector2 &translate, double edgeThreshold, bool overlapSupport) {
    if (overlapSupport)
        generateDistanceField<OverlappingContourCombiner<MultiAndTrueDistanceSelector> >(output, shape, range, scale, translate);
    else
        generateDistanceField<SimpleContourCombiner<MultiAndTrueDistanceSelector> >(output, shape, range, scale, translate);
    if (edgeThreshold > 0)
        msdfErrorCorrection(output, edgeThreshold/(scale*range));
    msdfPatchEdgeArtifacts(output, shape, range, scale, translate, overlapSupport);
}

// Legacy version

void generateSDF_legacy(const BitmapRef<float, 1> &output, const Shape &shape, double range, const Vector2 &scale, const Vector2 &translate) {
#ifdef MSDFGEN_USE_OPENMP
    #pragma omp parallel for
#endif
    for (int y = 0; y < output.height; ++y) {
        int row = shape.inverseYAxis ? output.height-y-1 : y;
        for (int x = 0; x < output.width; ++x) {
            double dummy;
            Point2 p = Vector2(x+.5, y+.5)/scale-translate;
            SignedDistance minDistance;
            for (std::vector<Contour>::const_iterator contour = shape.contours.begin(); contour != shape.contours.end(); ++contour)
                for (std::vector<EdgeHolder>::const_iterator edge = contour->edges.begin(); edge != contour->edges.end(); ++edge) {
                    SignedDistance distance = (*edge)->signedDistance(p, dummy);
                    if (distance < minDistance)
                        minDistance = distance;
                }
            *output(x, row) = float(minDistance.distance/range+.5);
        }
    }
}

void generatePseudoSDF_legacy(const BitmapRef<float, 1> &output, const Shape &shape, double range, const Vector2 &scale, const Vector2 &translate) {
#ifdef MSDFGEN_USE_OPENMP
    #pragma omp parallel for
#endif
    for (int y = 0; y < output.height; ++y) {
        int row = shape.inverseYAxis ? output.height-y-1 : y;
        for (int x = 0; x < output.width; ++x) {
            Point2 p = Vector2(x+.5, y+.5)/scale-translate;
            SignedDistance minDistance;
            const EdgeHolder *nearEdge = NULL;
            double nearParam = 0;
            for (std::vector<Contour>::const_iterator contour = shape.contours.begin(); contour != shape.contours.end(); ++contour)
                for (std::vector<EdgeHolder>::const_iterator edge = contour->edges.begin(); edge != contour->edges.end(); ++edge) {
                    double param;
                    SignedDistance distance = (*edge)->signedDistance(p, param);
                    if (distance < minDistance) {
                        minDistance = distance;
                        nearEdge = &*edge;
                        nearParam = param;
                    }
                }
            if (nearEdge)
                (*nearEdge)->distanceToPseudoDistance(minDistance, p, nearParam);
            *output(x, row) = float(minDistance.distance/range+.5);
        }
    }
}

void generateMSDF_legacy(const BitmapRef<float, 3> &output, const Shape &shape, double range, const Vector2 &scale, const Vector2 &translate, double edgeThreshold) {
#ifdef MSDFGEN_USE_OPENMP
    #pragma omp parallel for
#endif
    for (int y = 0; y < output.height; ++y) {
        int row = shape.inverseYAxis ? output.height-y-1 : y;
        for (int x = 0; x < output.width; ++x) {
            Point2 p = Vector2(x+.5, y+.5)/scale-translate;

            struct {
                SignedDistance minDistance;
                const EdgeHolder *nearEdge;
                double nearParam;
            } r, g, b;
            r.nearEdge = g.nearEdge = b.nearEdge = NULL;
            r.nearParam = g.nearParam = b.nearParam = 0;

            for (std::vector<Contour>::const_iterator contour = shape.contours.begin(); contour != shape.contours.end(); ++contour)
                for (std::vector<EdgeHolder>::const_iterator edge = contour->edges.begin(); edge != contour->edges.end(); ++edge) {
                    double param;
                    SignedDistance distance = (*edge)->signedDistance(p, param);
                    if ((*edge)->color&RED && distance < r.minDistance) {
                        r.minDistance = distance;
                        r.nearEdge = &*edge;
                        r.nearParam = param;
                    }
                    if ((*edge)->color&GREEN && distance < g.minDistance) {
                        g.minDistance = distance;
                        g.nearEdge = &*edge;
                        g.nearParam = param;
                    }
                    if ((*edge)->color&BLUE && distance < b.minDistance) {
                        b.minDistance = distance;
                        b.nearEdge = &*edge;
                        b.nearParam = param;
                    }
                }

            if (r.nearEdge)
                (*r.nearEdge)->distanceToPseudoDistance(r.minDistance, p, r.nearParam);
            if (g.nearEdge)
                (*g.nearEdge)->distanceToPseudoDistance(g.minDistance, p, g.nearParam);
            if (b.nearEdge)
                (*b.nearEdge)->distanceToPseudoDistance(b.minDistance, p, b.nearParam);
            output(x, row)[0] = float(r.minDistance.distance/range+.5);
            output(x, row)[1] = float(g.minDistance.distance/range+.5);
            output(x, row)[2] = float(b.minDistance.distance/range+.5);
        }
    }

    if (edgeThreshold > 0)
        msdfErrorCorrection(output, edgeThreshold/(scale*range));
}

void generateMTSDF_legacy(const BitmapRef<float, 4> &output, const Shape &shape, double range, const Vector2 &scale, const Vector2 &translate, double edgeThreshold) {
#ifdef MSDFGEN_USE_OPENMP
    #pragma omp parallel for
#endif
    for (int y = 0; y < output.height; ++y) {
        int row = shape.inverseYAxis ? output.height-y-1 : y;
        for (int x = 0; x < output.width; ++x) {
            Point2 p = Vector2(x+.5, y+.5)/scale-translate;

            SignedDistance minDistance;
            struct {
                SignedDistance minDistance;
                const EdgeHolder *nearEdge;
                double nearParam;
            } r, g, b;
            r.nearEdge = g.nearEdge = b.nearEdge = NULL;
            r.nearParam = g.nearParam = b.nearParam = 0;

            for (std::vector<Contour>::const_iterator contour = shape.contours.begin(); contour != shape.contours.end(); ++contour)
                for (std::vector<EdgeHolder>::const_iterator edge = contour->edges.begin(); edge != contour->edges.end(); ++edge) {
                    double param;
                    SignedDistance distance = (*edge)->signedDistance(p, param);
                    if (distance < minDistance)
                        minDistance = distance;
                    if ((*edge)->color&RED && distance < r.minDistance) {
                        r.minDistance = distance;
                        r.nearEdge = &*edge;
                        r.nearParam = param;
                    }
                    if ((*edge)->color&GREEN && distance < g.minDistance) {
                        g.minDistance = distance;
                        g.nearEdge = &*edge;
                        g.nearParam = param;
                    }
                    if ((*edge)->color&BLUE && distance < b.minDistance) {
                        b.minDistance = distance;
                        b.nearEdge = &*edge;
                        b.nearParam = param;
                    }
                }

            if (r.nearEdge)
                (*r.nearEdge)->distanceToPseudoDistance(r.minDistance, p, r.nearParam);
            if (g.nearEdge)
                (*g.nearEdge)->distanceToPseudoDistance(g.minDistance, p, g.nearParam);
            if (b.nearEdge)
                (*b.nearEdge)->distanceToPseudoDistance(b.minDistance, p, b.nearParam);
            output(x, row)[0] = float(r.minDistance.distance/range+.5);
            output(x, row)[1] = float(g.minDistance.distance/range+.5);
            output(x, row)[2] = float(b.minDistance.distance/range+.5);
            output(x, row)[3] = float(minDistance.distance/range+.5);
        }
    }

    if (edgeThreshold > 0)
        msdfErrorCorrection(output, edgeThreshold/(scale*range));
}

}
