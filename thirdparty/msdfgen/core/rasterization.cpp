
#include "rasterization.h"

#include <vector>
#include "arithmetics.hpp"

namespace msdfgen {

void rasterize(const BitmapRef<float, 1> &output, const Shape &shape, const Vector2 &scale, const Vector2 &translate, FillRule fillRule) {
    Point2 p;
    Scanline scanline;
    for (int y = 0; y < output.height; ++y) {
        int row = shape.inverseYAxis ? output.height-y-1 : y;
        p.y = (y+.5)/scale.y-translate.y;
        shape.scanline(scanline, p.y);
        for (int x = 0; x < output.width; ++x) {
            p.x = (x+.5)/scale.x-translate.x;
            bool fill = scanline.filled(p.x, fillRule);
            *output(x, row) = (float) fill;
        }
    }
}

void distanceSignCorrection(const BitmapRef<float, 1> &sdf, const Shape &shape, const Vector2 &scale, const Vector2 &translate, FillRule fillRule) {
    Point2 p;
    Scanline scanline;
    for (int y = 0; y < sdf.height; ++y) {
        int row = shape.inverseYAxis ? sdf.height-y-1 : y;
        p.y = (y+.5)/scale.y-translate.y;
        shape.scanline(scanline, p.y);
        for (int x = 0; x < sdf.width; ++x) {
            p.x = (x+.5)/scale.x-translate.x;
            bool fill = scanline.filled(p.x, fillRule);
            float &sd = *sdf(x, row);
            if ((sd > .5f) != fill)
                sd = 1.f-sd;
        }
    }
}

template <int N>
static void multiDistanceSignCorrection(const BitmapRef<float, N> &sdf, const Shape &shape, const Vector2 &scale, const Vector2 &translate, FillRule fillRule) {
    int w = sdf.width, h = sdf.height;
    if (!(w*h))
        return;
    Point2 p;
    Scanline scanline;
    bool ambiguous = false;
    std::vector<char> matchMap;
    matchMap.resize(w*h);
    char *match = &matchMap[0];
    for (int y = 0; y < h; ++y) {
        int row = shape.inverseYAxis ? h-y-1 : y;
        p.y = (y+.5)/scale.y-translate.y;
        shape.scanline(scanline, p.y);
        for (int x = 0; x < w; ++x) {
            p.x = (x+.5)/scale.x-translate.x;
            bool fill = scanline.filled(p.x, fillRule);
            float *msd = sdf(x, row);
            float sd = median(msd[0], msd[1], msd[2]);
            if (sd == .5f)
                ambiguous = true;
            else if ((sd > .5f) != fill) {
                msd[0] = 1.f-msd[0];
                msd[1] = 1.f-msd[1];
                msd[2] = 1.f-msd[2];
                *match = -1;
            } else
                *match = 1;
            if (N >= 4 && (msd[3] > .5f) != fill)
                msd[3] = 1.f-msd[3];
            ++match;
        }
    }
    // This step is necessary to avoid artifacts when whole shape is inverted
    if (ambiguous) {
        match = &matchMap[0];
        for (int y = 0; y < h; ++y) {
            int row = shape.inverseYAxis ? h-y-1 : y;
            for (int x = 0; x < w; ++x) {
                if (!*match) {
                    int neighborMatch = 0;
                    if (x > 0) neighborMatch += *(match-1);
                    if (x < w-1) neighborMatch += *(match+1);
                    if (y > 0) neighborMatch += *(match-w);
                    if (y < h-1) neighborMatch += *(match+w);
                    if (neighborMatch < 0) {
                        float *msd = sdf(x, row);
                        msd[0] = 1.f-msd[0];
                        msd[1] = 1.f-msd[1];
                        msd[2] = 1.f-msd[2];
                    }
                }
                ++match;
            }
        }
    }
}

void distanceSignCorrection(const BitmapRef<float, 3> &sdf, const Shape &shape, const Vector2 &scale, const Vector2 &translate, FillRule fillRule) {
    multiDistanceSignCorrection(sdf, shape, scale, translate, fillRule);
}

void distanceSignCorrection(const BitmapRef<float, 4> &sdf, const Shape &shape, const Vector2 &scale, const Vector2 &translate, FillRule fillRule) {
    multiDistanceSignCorrection(sdf, shape, scale, translate, fillRule);
}

}
