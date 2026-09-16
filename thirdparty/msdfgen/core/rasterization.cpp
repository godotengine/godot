
#include "rasterization.h"

#include <vector>
#include "arithmetics.hpp"

namespace msdfgen {

void rasterize(BitmapSection<float, 1> output, const Shape &shape, const Projection &projection, FillRule fillRule) {
    output.reorient(shape.getYAxisOrientation());
    Scanline scanline;
    for (int y = 0; y < output.height; ++y) {
        shape.scanline(scanline, projection.unprojectY(y+.5));
        for (int x = 0; x < output.width; ++x)
            *output(x, y) = (float) scanline.filled(projection.unprojectX(x+.5), fillRule);
    }
}

void distanceSignCorrection(BitmapSection<float, 1> sdf, const Shape &shape, const Projection &projection, float sdfZeroValue, FillRule fillRule) {
    sdf.reorient(shape.getYAxisOrientation());
    float doubleSdfZeroValue = sdfZeroValue+sdfZeroValue;
    Scanline scanline;
    for (int y = 0; y < sdf.height; ++y) {
        shape.scanline(scanline, projection.unprojectY(y+.5));
        for (int x = 0; x < sdf.width; ++x) {
            bool fill = scanline.filled(projection.unprojectX(x+.5), fillRule);
            float &sd = *sdf(x, y);
            if ((sd > sdfZeroValue) != fill)
                sd = doubleSdfZeroValue-sd;
        }
    }
}

template <int N>
static void multiDistanceSignCorrection(BitmapSection<float, N> sdf, const Shape &shape, const Projection &projection, float sdfZeroValue, FillRule fillRule) {
    int w = sdf.width, h = sdf.height;
    if (!(w && h))
        return;
    sdf.reorient(shape.getYAxisOrientation());
    float doubleSdfZeroValue = sdfZeroValue+sdfZeroValue;
    Scanline scanline;
    bool ambiguous = false;
    std::vector<char> matchMap;
    matchMap.resize(w*h);
    char *match = &matchMap[0];
    for (int y = 0; y < h; ++y) {
        shape.scanline(scanline, projection.unprojectY(y+.5));
        for (int x = 0; x < w; ++x) {
            bool fill = scanline.filled(projection.unprojectX(x+.5), fillRule);
            float *msd = sdf(x, y);
            float sd = median(msd[0], msd[1], msd[2]);
            if (sd == sdfZeroValue)
                ambiguous = true;
            else if ((sd > sdfZeroValue) != fill) {
                msd[0] = doubleSdfZeroValue-msd[0];
                msd[1] = doubleSdfZeroValue-msd[1];
                msd[2] = doubleSdfZeroValue-msd[2];
                *match = -1;
            } else
                *match = 1;
            if (N >= 4 && (msd[3] > sdfZeroValue) != fill)
                msd[3] = doubleSdfZeroValue-msd[3];
            ++match;
        }
    }
    // This step is necessary to avoid artifacts when whole shape is inverted
    if (ambiguous) {
        match = &matchMap[0];
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                if (!*match) {
                    int neighborMatch = 0;
                    if (x > 0) neighborMatch += *(match-1);
                    if (x < w-1) neighborMatch += *(match+1);
                    if (y > 0) neighborMatch += *(match-w);
                    if (y < h-1) neighborMatch += *(match+w);
                    if (neighborMatch < 0) {
                        float *msd = sdf(x, y);
                        msd[0] = doubleSdfZeroValue-msd[0];
                        msd[1] = doubleSdfZeroValue-msd[1];
                        msd[2] = doubleSdfZeroValue-msd[2];
                    }
                }
                ++match;
            }
        }
    }
}

void distanceSignCorrection(BitmapSection<float, 3> sdf, const Shape &shape, const Projection &projection, float sdfZeroValue, FillRule fillRule) {
    multiDistanceSignCorrection(sdf, shape, projection, sdfZeroValue, fillRule);
}

void distanceSignCorrection(BitmapSection<float, 4> sdf, const Shape &shape, const Projection &projection, float sdfZeroValue, FillRule fillRule) {
    multiDistanceSignCorrection(sdf, shape, projection, sdfZeroValue, fillRule);
}

// Legacy API

void rasterize(const BitmapSection<float, 1> &output, const Shape &shape, const Vector2 &scale, const Vector2 &translate, FillRule fillRule) {
    rasterize(output, shape, Projection(scale, translate), fillRule);
}

void distanceSignCorrection(BitmapSection<float, 1> sdf, const Shape &shape, const Projection &projection, FillRule fillRule) {
    distanceSignCorrection(sdf, shape, projection, .5f, fillRule);
}

void distanceSignCorrection(BitmapSection<float, 3> sdf, const Shape &shape, const Projection &projection, FillRule fillRule) {
    distanceSignCorrection(sdf, shape, projection, .5f, fillRule);
}

void distanceSignCorrection(BitmapSection<float, 4> sdf, const Shape &shape, const Projection &projection, FillRule fillRule) {
    distanceSignCorrection(sdf, shape, projection, .5f, fillRule);
}

void distanceSignCorrection(const BitmapSection<float, 1> &sdf, const Shape &shape, const Vector2 &scale, const Vector2 &translate, FillRule fillRule) {
    distanceSignCorrection(sdf, shape, Projection(scale, translate), fillRule);
}

void distanceSignCorrection(const BitmapSection<float, 3> &sdf, const Shape &shape, const Vector2 &scale, const Vector2 &translate, FillRule fillRule) {
    distanceSignCorrection(sdf, shape, Projection(scale, translate), fillRule);
}

void distanceSignCorrection(const BitmapSection<float, 4> &sdf, const Shape &shape, const Vector2 &scale, const Vector2 &translate, FillRule fillRule) {
    distanceSignCorrection(sdf, shape, Projection(scale, translate), fillRule);
}

}
