
#include "msdf-edge-artifact-patcher.h"

#include <cstring>
#include <vector>
#include <utility>
#include "arithmetics.hpp"
#include "equation-solver.h"
#include "bitmap-interpolation.hpp"
#include "edge-selectors.h"
#include "contour-combiners.h"
#include "ShapeDistanceFinder.h"

namespace msdfgen {

static bool isHotspot(float am, float bm, float xm) {
    return (am > .5f && bm > .5f && xm < .5f) || (am < .5f && bm < .5f && xm > .5f);
    // A much more aggressive version for the entire distance field (not just edges): return median(am, bm, xm) != xm;
}

static int findLinearChannelHotspots(double t[1], const float *a, const float *b, float dA, float dB) {
    int found = 0;
    double x = (double) dA/(dA-dB);
    if (x > 0 && x < 1) {
        float am = median(a[0], a[1], a[2]);
        float bm = median(b[0], b[1], b[2]);
        float xm = median(
            mix(a[0], b[0], x),
            mix(a[1], b[1], x),
            mix(a[2], b[2], x)
        );
        if (isHotspot(am, bm, xm))
            t[found++] = x;
    }
    return found;
}

static int findDiagonalChannelHotspots(double t[2], const float *a, const float *b, const float *c, const float *d, float dA, float dB, float dC, float dD) {
    int found = 0;
    double x[2];
    int solutions = solveQuadratic(x, (dD-dC)-(dB-dA), dC+dB-2*dA, dA);
    for (int i = 0; i < solutions; ++i)
        if (x[i] > 0 && x[i] < 1) {
            float am = median(a[0], a[1], a[2]);
            float bm = median(b[0], b[1], b[2]);
            float xm = median(
                mix(mix(a[0], b[0], x[i]), mix(c[0], d[0], x[i]), x[i]),
                mix(mix(a[1], b[1], x[i]), mix(c[1], d[1], x[i]), x[i]),
                mix(mix(a[2], b[2], x[i]), mix(c[2], d[2], x[i]), x[i])
            );
            if (isHotspot(am, bm, xm))
                t[found++] = x[i];
        }
    return found;
}

static int findLinearHotspots(double t[3], const float *a, const float *b) {
    int found = 0;
    found += findLinearChannelHotspots(t+found, a, b, a[1]-a[0], b[1]-b[0]);
    found += findLinearChannelHotspots(t+found, a, b, a[2]-a[1], b[2]-b[1]);
    found += findLinearChannelHotspots(t+found, a, b, a[0]-a[2], b[0]-b[2]);
    return found;
}

static int findDiagonalHotspots(double t[6], const float *a, const float *b, const float *c, const float *d) {
    int found = 0;
    found += findDiagonalChannelHotspots(t+found, a, b, c, d, a[1]-a[0], b[1]-b[0], c[1]-c[0], d[1]-d[0]);
    found += findDiagonalChannelHotspots(t+found, a, b, c, d, a[2]-a[1], b[2]-b[1], c[2]-c[1], d[2]-d[1]);
    found += findDiagonalChannelHotspots(t+found, a, b, c, d, a[0]-a[2], b[0]-b[2], c[0]-c[2], d[0]-d[2]);
    return found;
}

template <int N>
void findHotspots(std::vector<Point2> &hotspots, const BitmapConstRef<float, N> &sdf) {
    // All hotspots intersect either the horizontal, vertical, or diagonal line that connects neighboring texels
    // Horizontal:
    for (int y = 0; y < sdf.height; ++y) {
        const float *left = sdf(0, y);
        const float *right = sdf(1, y);
        for (int x = 0; x < sdf.width-1; ++x) {
            double t[3];
            int found = findLinearHotspots(t, left, right);
            for (int i = 0; i < found; ++i)
                hotspots.push_back(Point2(x+.5+t[i], y+.5));
            left += N, right += N;
        }
    }
    // Vertical:
    for (int y = 0; y < sdf.height-1; ++y) {
        const float *bottom = sdf(0, y);
        const float *top = sdf(0, y+1);
        for (int x = 0; x < sdf.width; ++x) {
            double t[3];
            int found = findLinearHotspots(t, bottom, top);
            for (int i = 0; i < found; ++i)
                hotspots.push_back(Point2(x+.5, y+.5+t[i]));
            bottom += N, top += N;
        }
    }
    // Diagonal:
    for (int y = 0; y < sdf.height-1; ++y) {
        const float *lb = sdf(0, y);
        const float *rb = sdf(1, y);
        const float *lt = sdf(0, y+1);
        const float *rt = sdf(1, y+1);
        for (int x = 0; x < sdf.width-1; ++x) {
            double t[6];
            int found = 0;
            found = findDiagonalHotspots(t, lb, rb, lt, rt);
            for (int i = 0; i < found; ++i)
                hotspots.push_back(Point2(x+.5+t[i], y+.5+t[i]));
            found = findDiagonalHotspots(t, lt, rt, lb, rb);
            for (int i = 0; i < found; ++i)
                hotspots.push_back(Point2(x+.5+t[i], y+1.5-t[i]));
            lb += N, rb += N, lt += N, rt += N;
        }
    }
}

template <template <typename> class ContourCombiner, int N>
static void msdfPatchEdgeArtifactsInner(const BitmapRef<float, N> &sdf, const Shape &shape, double range, const Vector2 &scale, const Vector2 &translate) {
    ShapeDistanceFinder<ContourCombiner<PseudoDistanceSelector> > distanceFinder(shape);
    std::vector<Point2> hotspots;
    findHotspots(hotspots, BitmapConstRef<float, N>(sdf));
    std::vector<std::pair<int, int> > artifacts;
    artifacts.reserve(hotspots.size());
    for (std::vector<Point2>::const_iterator hotspot = hotspots.begin(); hotspot != hotspots.end(); ++hotspot) {
        Point2 pos = *hotspot/scale-translate;
        double actualDistance = distanceFinder.distance(pos);
        float sd = float(actualDistance/range+.5);

        // Store hotspot's closest texel's current color
        float *subject = sdf((int) hotspot->x, (int) hotspot->y);
        float texel[N];
        memcpy(texel, subject, N*sizeof(float));
        // Sample signed distance at hotspot
        float msd[N];
        interpolate(msd, BitmapConstRef<float, N>(sdf), *hotspot);
        float oldSsd = median(msd[0], msd[1], msd[2]);
        // Flatten hotspot's closest texel
        float med = median(subject[0], subject[1], subject[2]);
        subject[0] = med, subject[1] = med, subject[2] = med;
        // Sample signed distance at hotspot after flattening
        interpolate(msd, BitmapConstRef<float, N>(sdf), *hotspot);
        float newSsd = median(msd[0], msd[1], msd[2]);
        // Revert modified texel
        memcpy(subject, texel, N*sizeof(float));

        // Consider hotspot an artifact if flattening improved the sample
        if (fabsf(newSsd-sd) < fabsf(oldSsd-sd))
            artifacts.push_back(std::make_pair((int) hotspot->x, (int) hotspot->y));
    }
    for (std::vector<std::pair<int, int> >::const_iterator artifact = artifacts.begin(); artifact != artifacts.end(); ++artifact) {
        float *pixel = sdf(artifact->first, artifact->second);
        float med = median(pixel[0], pixel[1], pixel[2]);
        pixel[0] = med, pixel[1] = med, pixel[2] = med;
    }
}

void msdfPatchEdgeArtifacts(const BitmapRef<float, 3> &sdf, const Shape &shape, double range, const Vector2 &scale, const Vector2 &translate, bool overlapSupport) {
    if (overlapSupport)
        msdfPatchEdgeArtifactsInner<OverlappingContourCombiner>(sdf, shape, range, scale, translate);
    else
        msdfPatchEdgeArtifactsInner<SimpleContourCombiner>(sdf, shape, range, scale, translate);
}

void msdfPatchEdgeArtifacts(const BitmapRef<float, 4> &sdf, const Shape &shape, double range, const Vector2 &scale, const Vector2 &translate, bool overlapSupport) {
    if (overlapSupport)
        msdfPatchEdgeArtifactsInner<OverlappingContourCombiner>(sdf, shape, range, scale, translate);
    else
        msdfPatchEdgeArtifactsInner<SimpleContourCombiner>(sdf, shape, range, scale, translate);
}

}
