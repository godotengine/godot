/*
 * Copyright (c) 2020 - 2023 the ThorVG project. All rights reserved.

 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:

 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.

 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef _TVG_SW_COMMON_H_
#define _TVG_SW_COMMON_H_

#include "tvgCommon.h"
#include "tvgRender.h"

#if 0
#include <sys/time.h>
static double timeStamp()
{
   struct timeval tv;
   gettimeofday(&tv, NULL);
   return (tv.tv_sec + tv.tv_usec / 1000000.0);
}
#endif

#define SW_CURVE_TYPE_POINT 0
#define SW_CURVE_TYPE_CUBIC 1
#define SW_ANGLE_PI (180L << 16)
#define SW_ANGLE_2PI (SW_ANGLE_PI << 1)
#define SW_ANGLE_PI2 (SW_ANGLE_PI >> 1)
#define SW_ANGLE_PI4 (SW_ANGLE_PI >> 2)

using SwCoord = signed long;
using SwFixed = signed long long;

struct SwPoint
{
    SwCoord x, y;

    SwPoint& operator+=(const SwPoint& rhs)
    {
        x += rhs.x;
        y += rhs.y;
        return *this;
    }

    SwPoint operator+(const SwPoint& rhs) const
    {
        return {x + rhs.x, y + rhs.y};
    }

    SwPoint operator-(const SwPoint& rhs) const
    {
        return {x - rhs.x, y - rhs.y};
    }

    bool operator==(const SwPoint& rhs) const
    {
        return (x == rhs.x && y == rhs.y);
    }

    bool operator!=(const SwPoint& rhs) const
    {
        return (x != rhs.x || y != rhs.y);
    }

    bool zero() const
    {
        if (x == 0 && y == 0) return true;
        else return false;
    }

    bool small() const
    {
        //2 is epsilon...
        if (abs(x) < 2 && abs(y) < 2) return true;
        else return false;
    }

};

struct SwSize
{
    SwCoord w, h;
};

struct SwOutline
{
    SwPoint*      pts;              //the outline's points
    uint32_t      ptsCnt;           //number of points in the glyph
    uint32_t      reservedPtsCnt;
    uint32_t*     cntrs;            //the contour end points
    uint16_t      cntrsCnt;         //number of contours in glyph
    uint16_t      reservedCntrsCnt;
    uint8_t*      types;            //curve type
    bool*         closed;           //opened or closed path?
    FillRule      fillRule;
};

struct SwSpan
{
    uint16_t x, y;
    uint16_t len;
    uint8_t coverage;
};

struct SwRleData
{
    SwSpan *spans;
    uint32_t alloc;
    uint32_t size;
};

struct SwBBox
{
    SwPoint min, max;

    void reset()
    {
        min.x = min.y = max.x = max.y = 0;
    }
};

struct SwFill
{
    struct SwLinear {
        float dx, dy;
        float len;
        float offset;
    };

    struct SwRadial {
        float a11, a12, shiftX;
        float a21, a22, shiftY;
        float detSecDeriv;
        float a;
    };

    union {
        SwLinear linear;
        SwRadial radial;
    };

    uint32_t* ctable;
    FillSpread spread;

    bool translucent;
};

struct SwStrokeBorder
{
    uint32_t ptsCnt;
    uint32_t maxPts;
    SwPoint* pts;
    uint8_t* tags;
    int32_t start;     //index of current sub-path start point
    bool movable;      //true: for ends of lineto borders
};

struct SwStroke
{
    SwFixed angleIn;
    SwFixed angleOut;
    SwPoint center;
    SwFixed lineLength;
    SwFixed subPathAngle;
    SwPoint ptStartSubPath;
    SwFixed subPathLineLength;
    SwFixed width;

    StrokeCap cap;
    StrokeJoin join;
    StrokeJoin joinSaved;
    SwFill* fill = nullptr;

    SwStrokeBorder borders[2];

    float sx, sy;

    bool firstPt;
    bool closedSubPath;
    bool handleWideStrokes;
};

struct SwDashStroke
{
    SwOutline* outline;
    float curLen;
    int32_t curIdx;
    Point ptStart;
    Point ptCur;
    float* pattern;
    uint32_t cnt;
    bool curOpGap;
};

struct SwShape
{
    SwOutline*   outline = nullptr;
    SwStroke*    stroke = nullptr;
    SwFill*      fill = nullptr;
    SwRleData*   rle = nullptr;
    SwRleData*   strokeRle = nullptr;
    SwBBox       bbox;           //Keep it boundary without stroke region. Using for optimal filling.

    bool         fastTrack = false;   //Fast Track: axis-aligned rectangle without any clips?
};

struct SwImage
{
    SwOutline*   outline = nullptr;
    SwRleData*   rle = nullptr;
    uint32_t*    data = nullptr;
    uint32_t     w, h, stride;
    int32_t      ox = 0;         //offset x
    int32_t      oy = 0;         //offset y
    float        scale;

    bool         direct = false;  //draw image directly (with offset)
    bool         scaled = false;  //draw scaled image
};

struct SwBlender
{
    uint32_t (*join)(uint8_t r, uint8_t g, uint8_t b, uint8_t a);
    uint32_t (*lumaValue)(uint32_t c);
};

struct SwCompositor;

struct SwSurface : Surface
{
    SwBlender blender;                    //mandatory
    SwCompositor* compositor = nullptr;   //compositor (optional)
};

struct SwCompositor : Compositor
{
    SwSurface* recoverSfc;                  //Recover surface when composition is started
    SwCompositor* recoverCmp;               //Recover compositor when composition is done
    SwImage image;
    SwBBox bbox;
    bool valid;
};

struct SwMpool
{
    SwOutline* outline;
    SwOutline* strokeOutline;
    unsigned allocSize;
};

static inline SwCoord TO_SWCOORD(float val)
{
    return SwCoord(val * 64.0f);
}

static inline uint32_t ALPHA_BLEND(uint32_t c, uint32_t a)
{
    return (((((c >> 8) & 0x00ff00ff) * a + 0x00ff00ff) & 0xff00ff00) +
            ((((c & 0x00ff00ff) * a + 0x00ff00ff) >> 8) & 0x00ff00ff));
}

static inline uint32_t INTERPOLATE(uint32_t a, uint32_t c0, uint32_t c1)
{
    return (((((((c0 >> 8) & 0xff00ff) - ((c1 >> 8) & 0xff00ff)) * a) + (c1 & 0xff00ff00)) & 0xff00ff00) + ((((((c0 & 0xff00ff) - (c1 & 0xff00ff)) * a) >> 8) + (c1 & 0xff00ff)) & 0xff00ff));
}

static inline SwCoord HALF_STROKE(float width)
{
    return TO_SWCOORD(width * 0.5f);
}

int64_t mathMultiply(int64_t a, int64_t b);
int64_t mathDivide(int64_t a, int64_t b);
int64_t mathMulDiv(int64_t a, int64_t b, int64_t c);
void mathRotate(SwPoint& pt, SwFixed angle);
SwFixed mathTan(SwFixed angle);
SwFixed mathAtan(const SwPoint& pt);
SwFixed mathCos(SwFixed angle);
SwFixed mathSin(SwFixed angle);
void mathSplitCubic(SwPoint* base);
SwFixed mathDiff(SwFixed angle1, SwFixed angle2);
SwFixed mathLength(const SwPoint& pt);
bool mathSmallCubic(const SwPoint* base, SwFixed& angleIn, SwFixed& angleMid, SwFixed& angleOut);
SwFixed mathMean(SwFixed angle1, SwFixed angle2);
SwPoint mathTransform(const Point* to, const Matrix* transform);
bool mathUpdateOutlineBBox(const SwOutline* outline, const SwBBox& clipRegion, SwBBox& renderRegion, bool fastTrack);
bool mathClipBBox(const SwBBox& clipper, SwBBox& clipee);

void shapeReset(SwShape* shape);
bool shapePrepare(SwShape* shape, const RenderShape* rshape, const Matrix* transform, const SwBBox& clipRegion, SwBBox& renderRegion, SwMpool* mpool, unsigned tid, bool hasComposite);
bool shapePrepared(const SwShape* shape);
bool shapeGenRle(SwShape* shape, const RenderShape* rshape, bool antiAlias);
void shapeDelOutline(SwShape* shape, SwMpool* mpool, uint32_t tid);
void shapeResetStroke(SwShape* shape, const RenderShape* rshape, const Matrix* transform);
bool shapeGenStrokeRle(SwShape* shape, const RenderShape* rshape, const Matrix* transform, const SwBBox& clipRegion, SwBBox& renderRegion, SwMpool* mpool, unsigned tid);
void shapeFree(SwShape* shape);
void shapeDelStroke(SwShape* shape);
bool shapeGenFillColors(SwShape* shape, const Fill* fill, const Matrix* transform, SwSurface* surface, uint32_t opacity, bool ctable);
bool shapeGenStrokeFillColors(SwShape* shape, const Fill* fill, const Matrix* transform, SwSurface* surface, uint32_t opacity, bool ctable);
void shapeResetFill(SwShape* shape);
void shapeResetStrokeFill(SwShape* shape);
void shapeDelFill(SwShape* shape);
void shapeDelStrokeFill(SwShape* shape);

void strokeReset(SwStroke* stroke, const RenderShape* shape, const Matrix* transform);
bool strokeParseOutline(SwStroke* stroke, const SwOutline& outline);
SwOutline* strokeExportOutline(SwStroke* stroke, SwMpool* mpool, unsigned tid);
void strokeFree(SwStroke* stroke);

bool imagePrepare(SwImage* image, Polygon* triangles, uint32_t triangleCount, const Matrix* transform, const SwBBox& clipRegion, SwBBox& renderRegion, SwMpool* mpool, unsigned tid);
bool imageGenRle(SwImage* image, const SwBBox& renderRegion, bool antiAlias);
void imageDelOutline(SwImage* image, SwMpool* mpool, uint32_t tid);
void imageReset(SwImage* image);
void imageFree(SwImage* image);

bool fillGenColorTable(SwFill* fill, const Fill* fdata, const Matrix* transform, SwSurface* surface, uint32_t opacity, bool ctable);
void fillReset(SwFill* fill);
void fillFree(SwFill* fill);
void fillFetchLinear(const SwFill* fill, uint32_t* dst, uint32_t y, uint32_t x, uint32_t len);
void fillFetchRadial(const SwFill* fill, uint32_t* dst, uint32_t y, uint32_t x, uint32_t len);

SwRleData* rleRender(SwRleData* rle, const SwOutline* outline, const SwBBox& renderRegion, bool antiAlias);
void rleFree(SwRleData* rle);
void rleReset(SwRleData* rle);
void rleClipPath(SwRleData *rle, const SwRleData *clip);
void rleClipRect(SwRleData *rle, const SwBBox* clip);

SwMpool* mpoolInit(uint32_t threads);
bool mpoolTerm(SwMpool* mpool);
bool mpoolClear(SwMpool* mpool);
SwOutline* mpoolReqOutline(SwMpool* mpool, unsigned idx);
void mpoolRetOutline(SwMpool* mpool, unsigned idx);
SwOutline* mpoolReqStrokeOutline(SwMpool* mpool, unsigned idx);
void mpoolRetStrokeOutline(SwMpool* mpool, unsigned idx);

bool rasterCompositor(SwSurface* surface);
bool rasterGradientShape(SwSurface* surface, SwShape* shape, unsigned id);
bool rasterShape(SwSurface* surface, SwShape* shape, uint8_t r, uint8_t g, uint8_t b, uint8_t a);
bool rasterImage(SwSurface* surface, SwImage* image, const Matrix* transform, const SwBBox& bbox, uint32_t opacity);
bool rasterImageMesh(SwSurface* surface, SwImage* image, const Polygon* triangles, const uint32_t triangleCount, const Matrix* transform, const SwBBox& bbox, uint32_t opacity);
bool rasterStroke(SwSurface* surface, SwShape* shape, uint8_t r, uint8_t g, uint8_t b, uint8_t a);
bool rasterGradientStroke(SwSurface* surface, SwShape* shape, unsigned id);
bool rasterClear(SwSurface* surface);
void rasterRGBA32(uint32_t *dst, uint32_t val, uint32_t offset, int32_t len);
void rasterUnpremultiply(SwSurface* surface);

#endif /* _TVG_SW_COMMON_H_ */
