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

#ifndef _TVG_WG_SHADER_TYPES_H_
#define _TVG_WG_SHADER_TYPES_H_

#include "tvgRender.h"

///////////////////////////////////////////////////////////////////////////////
// shader types
///////////////////////////////////////////////////////////////////////////////

// mat4x4f
struct WgShaderTypeMat4x4f
{
    float mat[16]{};

    WgShaderTypeMat4x4f();
    WgShaderTypeMat4x4f(const Matrix& transform);
    WgShaderTypeMat4x4f(size_t w, size_t h);
    void identity();
    void update(const Matrix& transform);
    void update(size_t w, size_t h);
};

// vec4f
struct WgShaderTypeVec4f
{
    float vec[4]{};

    WgShaderTypeVec4f() {};
    WgShaderTypeVec4f(const ColorSpace colorSpace, uint8_t o);
    WgShaderTypeVec4f(const RenderColor& c);
    void update(const ColorSpace colorSpace, uint8_t o);
    void update(const RenderColor& c);
    void update(const RenderRegion& r);
};

// WGSL: struct GradSettings  { transform: mat4x4f, coords: vec4f, focal: vec4f };
struct WgShaderTypeGradSettings
{
    // gradient transform matrix
    WgShaderTypeMat4x4f transform;
    // linear: [0] - x1, [1] - y1, [2] - x2, [3] - y2
    // radial: [0] - cx, [1] - cy, [2] - cr
    WgShaderTypeVec4f coords;
    // radial: [0] - fx, [1] - fy, [2] - fr
    WgShaderTypeVec4f focal;
    
    void update(const Fill* fill, const Matrix* modelTransform);
};

// WGSL: struct PaintSettings { options: vec4f, color: vec4f, gradient: GradSettings };
struct WgShaderTypePaintSettings
{
    // [0] - color space, [3] - opacity
    WgShaderTypeVec4f options;
    // solid color
    WgShaderTypeVec4f color;
    // gradient settings (linear/radial)
    WgShaderTypeGradSettings gradient;
    // align to 256 bytes (see webgpu spec: minUniformBufferOffsetAlignment)
    uint8_t _padding[256 - sizeof(options) - sizeof(color) - sizeof(gradient)];
};
// see webgpu spec: 3.6.2. Limits - minUniformBufferOffsetAlignment (256)
static_assert(sizeof(WgShaderTypePaintSettings) == 256, "Uniform shader data type size must be aligned to 256 bytes");

// gradient color map
#define WG_TEXTURE_GRADIENT_SIZE 512
struct WgShaderTypeGradientData
{
    uint8_t data[WG_TEXTURE_GRADIENT_SIZE * 4];

    void update(const Fill* fill);
};

// gaussian params: sigma, scale, extend
#define WG_GAUSSIAN_KERNEL_SIZE_MAX (128.0f)
// gaussian blur, drop shadow, fill, tint, tritone
struct WgShaderTypeEffectParams
{
    // gaussian blur: [0]: sigma, [1]: scale, [2]: kernel size
    // drop shadow:   [0]: sigma, [1]: scale, [2]: kernel size, [4..7]: color, [8, 9]: offset
    // fill:          [0..3]: color
    // tint:          [0..2]: black,  [4..6]: white,   [8]: intensity
    // tritone:       [0..2]: shadow, [4..6]: midtone, [8..10]: highlight
    float params[4+4+4]{}; // settings: array<vec4f, 3>;
    uint32_t extend{};     // gaussian blur extend
    Point offset{};        // drop shadow offset

    bool update(RenderEffectGaussianBlur* gaussian, const Matrix& transform);
    bool update(RenderEffectDropShadow* dropShadow, const Matrix& transform);
    bool update(RenderEffectFill* fill);
    bool update(RenderEffectTint* tint);
    bool update(RenderEffectTritone* tritone);
};

#endif // _TVG_WG_SHADER_TYPES_H_
