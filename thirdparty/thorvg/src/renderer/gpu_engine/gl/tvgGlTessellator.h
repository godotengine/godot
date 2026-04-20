/*
 * Copyright (c) 2023 - 2026 ThorVG project. All rights reserved.

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

#ifndef _TVG_GL_TESSELLATOR_H_
#define _TVG_GL_TESSELLATOR_H_

#include "tvgGlCommon.h"

namespace tvg
{

class Stroker
{
    struct State
    {
        Point firstPt;
        Point firstPtDir;
        Point prevPt;
        Point prevPtDir;
    };
public:
    Stroker(GlGeometryBuffer* buffer, float strokeWidth, StrokeCap cap, StrokeJoin join, float miterLimit = 4.0f);
    void run(const RenderPath& path);
    RenderRegion bounds() const;

private:

    float radius() const
    {
        return mWidth * 0.5f;
    }

    void cap();
    void lineTo(const Point& curr);
    void cubicTo(const Point& cnt1, const Point& cnt2, const Point& end);
    void close();
    void join(const Point& dir);
    void round(const Point& prev, const Point& curr, const Point& center);
    void miter(const Point& prev, const Point& curr, const Point& center);
    void bevel(const Point& prev, const Point& curr, const Point& center);
    void square(const Point& p, const Point& outDir);
    void squarePoint(const Point& p);
    void round(const Point& p, const Point& outDir);
    void roundPoint(const Point& p);

    GlGeometryBuffer* mBuffer;
    float mWidth = 0.0f;
    float mMiterLimit = 4.f;
    StrokeCap mCap = StrokeCap::Square;
    StrokeJoin mJoin = StrokeJoin::Bevel;
    State mState = {};
    Point mLeftTop = {0.0f, 0.0f};
    Point mRightBottom = {0.0f, 0.0f};
};

class BWTessellator
{
public:
    BWTessellator(GlGeometryBuffer* buffer);
    void tessellate(const RenderPath& path);
    RenderRegion bounds() const;
    bool convex = true;

private:
    uint32_t pushVertex(float x, float y);
    void pushTriangle(uint32_t a, uint32_t b, uint32_t c);

    GlGeometryBuffer* mBuffer;
    BBox bbox = {};
};

}  // namespace tvg

#endif /* _TVG_GL_TESSELLATOR_H_ */
