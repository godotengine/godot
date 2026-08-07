/*
 * Copyright (c) 2020 - 2026 ThorVG project. All rights reserved.

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

#ifndef _TVG_GL_COMMON_H_
#define _TVG_GL_COMMON_H_

#include <cassert>
#include "tvgGl.h"
#include "tvgRender.h"
#include "tvgMath.h"
#include "tvgGpuCommon.h"

constexpr float MIN_GL_STROKE_WIDTH = 1.0f;
constexpr float MIN_GL_STROKE_ALPHA = 0.25f;

constexpr uint32_t GL_MAT3_STD140_SIZE = 12; // mat3 is 3 vec4 columns in std140
constexpr uint32_t GL_MAT3_STD140_BYTES = GL_MAT3_STD140_SIZE * sizeof(float);

// All GPU matrices use column major order.
static inline void getMatrix3(const Matrix& mat3, float* matOut)
{
    matOut[0] = mat3.e11; matOut[3] = mat3.e12; matOut[6] = mat3.e13;
    matOut[1] = mat3.e21; matOut[4] = mat3.e22; matOut[7] = mat3.e23;
    matOut[2] = mat3.e31; matOut[5] = mat3.e32; matOut[8] = mat3.e33;
}


// All GPU matrices use column major order. std140 mat3 packs each column into a vec4 stride.
static inline void getMatrix3Std140(const Matrix& mat3, float* matOut)
{
    matOut[0] = mat3.e11; matOut[4] = mat3.e12; matOut[8] = mat3.e13;
    matOut[1] = mat3.e21; matOut[5] = mat3.e22; matOut[9] = mat3.e23;
    matOut[2] = mat3.e31; matOut[6] = mat3.e32; matOut[10] = mat3.e33;
    matOut[3] = 0.0f;     matOut[7] = 0.0f;     matOut[11] = 0.0f;
}


enum class GlStencilMode {
    None,
    FillNonZero,
    FillEvenOdd,
    Stroke,
};


class GlStageBuffer;
class GlRenderTask;

struct GlGeometryBuffer {
    Array<float> vertex;
    Array<uint32_t> index;

    void clear()
    {
        vertex.clear();
        index.clear();
    }

};

struct GlGeometry
{
    const Matrix* inverseMatrix()
    {
        if (!inverseMatrixDirty) return &cachedInverseMatrix;
        inverse(&matrix, &cachedInverseMatrix);
        inverseMatrixDirty = false;
        return &cachedInverseMatrix;
    }

    void setMatrix(const Matrix& tr) { matrix = tr; inverseMatrixDirty = true;}

    void prepare(const RenderShape& rshape);
    bool tesselateShape(const RenderShape& rshape, float* opacityMultiplier = nullptr);
    bool tesselateStroke(const RenderShape& rshape);
    bool tesselateThinPath(const RenderPath& path);
    void tesselateImage(const RenderSurface* image);
    bool draw(GlRenderTask* task, GlStageBuffer* gpuBuffer, RenderUpdateFlag flag);
    GlStencilMode getStencilMode(RenderUpdateFlag flag);
    RenderRegion getBounds() const;

    GlGeometryBuffer fill, stroke;
    Matrix matrix = {};
    RenderRegion viewport = {};
    RenderRegion fillBounds = {};
    RenderRegion strokeBounds = {};
    FillRule fillRule = FillRule::NonZero;
    RenderPath optPath;  //optimal path
    float strokeRenderWidth = 0.0f;
    Matrix cachedInverseMatrix = {};
    bool inverseMatrixDirty = true;
    bool fillWorld = false;
    bool optPathThin = false;
    bool optPathSkipFill = false;
    bool convex;
};


struct GlShape
{
  const RenderShape* rshape = nullptr;
  float viewWd;
  float viewHt;
  uint32_t opacity = 0;
  GLuint texId = 0;
  const RenderSurface* texSource = nullptr;
  FilterMethod texFilter = FilterMethod::Bilinear;
  uint32_t texFlipY = 0;
  ColorSpace texColorSpace = ColorSpace::ABGR8888;
  GlGeometry geometry;
  Array<RenderData> clips;
  uint16_t texStamp = 0;  // Tracks TextureMgr::stamp ownership of texId.
  bool validFill = false;
  bool validStroke = false;
};

struct GlIntersector
{
    bool isPointInTriangle(const Point& p, const Point& a, const Point& b, const Point& c);
    bool isPointInImage(const Point& p, const GlGeometryBuffer& mesh, const Matrix& tr);
    bool isPointInTris(const Point& p, const GlGeometryBuffer& mesh, const Matrix& tr);
    bool isPointInMesh(const Point& p, const GlGeometryBuffer& mesh, const Matrix& tr);
    bool intersectClips(const Point& pt, const tvg::Array<tvg::RenderData>& clips);
    bool intersectShape(const RenderRegion region, const GlShape* shape);
    bool intersectImage(const RenderRegion region, const GlShape* image);
};

#define MAX_GRADIENT_STOPS 16

struct GlLinearGradientBlock
{
    alignas(16) float nStops[4] = {};
    alignas(16) float startPos[2] = {};
    alignas(8) float stopPos[2] = {};
    alignas(8) float stopPoints[MAX_GRADIENT_STOPS] = {};
    alignas(16) float stopColors[4 * MAX_GRADIENT_STOPS] = {};
};

struct GlRadialGradientBlock
{
    alignas(16) float nStops[4] = {};
    alignas(16) float centerPos[4] = {};
    alignas(16) float radius[2] = {};
    alignas(16) float stopPoints[MAX_GRADIENT_STOPS] = {};
    alignas(16) float stopColors[4 * MAX_GRADIENT_STOPS] = {};
};

struct GlCompositor : RenderCompositor
{
    RenderRegion bbox;
    CompositionFlag flags;
    BlendMethod blendMethod = BlendMethod::Normal;

    GlCompositor(const RenderRegion& box, CompositionFlag flags) : bbox(box), flags(flags) {}
};

#endif /* _TVG_GL_COMMON_H_ */
