/*
 * Copyright (c) 2026 ThorVG project. All rights reserved.

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

#include "tvgSwCommon.h"

/************************************************************************/
/* Internal Class Implementation                                        */
/************************************************************************/

static uint32_t _premultiply(uint32_t c1, uint32_t c2, uint8_t a)
{
    if (a == 255) return c1;
    else if (a == 0) return c2;
    return ALPHA_BLEND(c1, a) + ALPHA_BLEND(c2, 255 - a);
}

static RenderColor _unpremultiply(uint32_t c)
{
    RenderColor o = {C1(c), C2(c), C3(c), A(c)};
    if (o.a > 0 && o.a < 255) {
        o.r = std::min(o.r * 255u / o.a, 255u);
        o.g = std::min(o.g * 255u / o.a, 255u);
        o.b = std::min(o.b * 255u / o.a, 255u);
    }
    return o;
}

static void _saturate(uint8_t& c1, uint8_t& c2, uint8_t& c3, const uint8_t& s1, const uint8_t& s2, const uint8_t& s3)
{
    // to saturation
    auto s = std::max(s1, std::max(s2, s3)) - std::min(s1, std::min(s2, s3));

    uint8_t v[3] = {c1, c2, c3};
    int lo = 0, mid = 1, hi = 2;

    if (v[lo] > v[mid]) std::swap(lo, mid);
    if (v[mid] > v[hi]) std::swap(mid, hi);
    if (v[lo] > v[mid]) std::swap(lo, mid);

    auto minv = v[lo];
    auto maxv = v[hi];

    if (maxv > minv) {
        v[mid] = uint8_t((int(v[mid] - minv) * s) / (maxv - minv));
        v[hi] = s;
    } else {
        v[mid] = v[hi] = 0;
    }

    v[lo] = 0;
    c1 = v[0];
    c2 = v[1];
    c3 = v[2];
}

static void _luminance(uint8_t& c1, uint8_t& c2, uint8_t& c3, int cl, int l)
{
    auto d = l - cl;
    auto r = int(c1) + d;
    auto g = int(c2) + d;
    auto b = int(c3) + d;
    auto minv = std::min(r, std::min(g, b));
    auto maxv = std::max(r, std::max(g, b));

    if (minv < 0) {
        r = l + ((r - l) * l) / (l - minv);
        g = l + ((g - l) * l) / (l - minv);
        b = l + ((b - l) * l) / (l - minv);
        maxv = std::max(r, std::max(g, b));
    }

    if (maxv > 255) {
        r = l + ((r - l) * (255 - l)) / (maxv - l);
        g = l + ((g - l) * (255 - l)) / (maxv - l);
        b = l + ((b - l) * (255 - l)) / (maxv - l);
    }

    c1 = static_cast<uint8_t>(r);
    c2 = static_cast<uint8_t>(g);
    c3 = static_cast<uint8_t>(b);
}

/************************************************************************/
/* External Class Implementation                                        */
/************************************************************************/

uint32_t blendDifference(TVG_UNUSED const SwSurface* surface, uint32_t s, uint32_t d)
{
    auto f = [](uint8_t s, uint8_t d) {
        return (s > d) ? (s - d) : (d - s);
    };

    return JOIN(255, f(C1(s), C1(d)), f(C2(s), C2(d)), f(C3(s), C3(d)));
}

uint32_t blendExclusion(TVG_UNUSED const SwSurface* surface, uint32_t s, uint32_t d)
{
    auto f = [](uint8_t s, uint8_t d) {
        return tvg::clamp(s + d - 2 * MULTIPLY(s, d), 0, 255);
    };

    return JOIN(255, f(C1(s), C1(d)), f(C2(s), C2(d)), f(C3(s), C3(d)));
}

uint32_t blendAdd(TVG_UNUSED const SwSurface* surface, uint32_t s, uint32_t d)
{
    auto f = [](uint8_t s, uint8_t d) {
        return std::min(s + d, 255);
    };

    return JOIN(255, f(C1(s), C1(d)), f(C2(s), C2(d)), f(C3(s), C3(d)));
}

uint32_t blendScreen(TVG_UNUSED const SwSurface* surface, uint32_t s, uint32_t d)
{
    auto f = [](uint8_t s, uint8_t d) {
        return s + d - MULTIPLY(s, d);
    };

    return JOIN(255, f(C1(s), C1(d)), f(C2(s), C2(d)), f(C3(s), C3(d)));
}

uint32_t blendMultiply(TVG_UNUSED const SwSurface* surface, uint32_t s, uint32_t d)
{
    auto o = _unpremultiply(d);

    auto f = [](uint8_t s, uint8_t d) {
        return MULTIPLY(s, d);
    };

    return _premultiply(JOIN(255, f(C1(s), o.r), f(C2(s), o.g), f(C3(s), o.b)), s, o.a);
}

uint32_t blendOverlay(TVG_UNUSED const SwSurface* surface, uint32_t s, uint32_t d)
{
    auto o = _unpremultiply(d);

    auto f = [](uint8_t s, uint8_t d) {
        return (d < 128) ? std::min(255, 2 * MULTIPLY(s, d)) : (255 - std::min(255, 2 * MULTIPLY(255 - s, 255 - d)));
    };

    return _premultiply(JOIN(255, f(C1(s), o.r), f(C2(s), o.g), f(C3(s), o.b)), s, o.a);
}

uint32_t blendDarken(TVG_UNUSED const SwSurface* surface, uint32_t s, uint32_t d)
{
    auto o = _unpremultiply(d);

    auto f = [](uint8_t s, uint8_t d) {
        return std::min(s, d);
    };

    return _premultiply(JOIN(255, f(C1(s), o.r), f(C2(s), o.g), f(C3(s), o.b)), s, o.a);
}

uint32_t blendLighten(TVG_UNUSED const SwSurface* surface, uint32_t s, uint32_t d)
{
    auto f = [](uint8_t s, uint8_t d) {
        return std::max(s, d);
    };

    return JOIN(255, f(C1(s), C1(d)), f(C2(s), C2(d)), f(C3(s), C3(d)));
}

uint32_t blendColorDodge(TVG_UNUSED const SwSurface* surface, uint32_t s, uint32_t d)
{
    auto o = _unpremultiply(d);

    auto f = [](uint8_t s, uint8_t d) {
        return d == 0 ? 0 : (s == 255 ? 255 : std::min(d * 255 / (255 - s), 255));
    };

    return _premultiply(JOIN(255, f(C1(s), o.r), f(C2(s), o.g), f(C3(s), o.b)), s, o.a);
}

uint32_t blendColorBurn(TVG_UNUSED const SwSurface* surface, uint32_t s, uint32_t d)
{
    auto o = _unpremultiply(d);

    auto f = [](uint8_t s, uint8_t d) {
        return d == 255 ? 255 : (s == 0 ? 0 : 255 - std::min((255 - d) * 255 / s, 255));
    };

    return _premultiply(JOIN(255, f(C1(s), o.r), f(C2(s), o.g), f(C3(s), o.b)), s, o.a);
}

uint32_t blendHardLight(TVG_UNUSED const SwSurface* surface, uint32_t s, uint32_t d)
{
    auto o = _unpremultiply(d);

    auto f = [](uint8_t s, uint8_t d) {
        return (s < 128) ? std::min(255, 2 * MULTIPLY(s, d)) : (255 - std::min(255, 2 * MULTIPLY(255 - s, 255 - d)));
    };

    return _premultiply(JOIN(255, f(C1(s), o.r), f(C2(s), o.g), f(C3(s), o.b)), s, o.a);
}

uint32_t blendSoftLight(TVG_UNUSED const SwSurface* surface, uint32_t s, uint32_t d)
{
    auto o = _unpremultiply(d);

    auto f = [](uint8_t s, uint8_t d) -> uint8_t {
        if (s <= 127) return (d - ((255 - 2 * s) * d * (255 - d)) / 65025);

        // use Look up table for skip sqrt per every pixels.
        static constexpr uint8_t SQRT_LUT[256] = {
            0, 15, 22, 27, 31, 35, 39, 42, 45, 47, 50, 52, 55, 57, 59, 61,
            63, 65, 67, 69, 71, 73, 74, 76, 78, 79, 81, 82, 84, 85, 87, 88,
            90, 91, 93, 94, 95, 97, 98, 99, 100, 102, 103, 104, 105, 107, 108, 109,
            110, 111, 112, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126,
            127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 141,
            142, 143, 144, 145, 146, 147, 148, 148, 149, 150, 151, 152, 153, 153, 154, 155,
            156, 157, 158, 158, 159, 160, 161, 162, 162, 163, 164, 165, 165, 166, 167, 168,
            168, 169, 170, 171, 171, 172, 173, 174, 174, 175, 176, 177, 177, 178, 179, 179,
            180, 181, 182, 182, 183, 184, 184, 185, 186, 186, 187, 188, 188, 189, 190, 190,
            191, 192, 192, 193, 194, 194, 195, 196, 196, 197, 198, 198, 199, 200, 200, 201,
            201, 202, 203, 203, 204, 205, 205, 206, 206, 207, 208, 208, 209, 210, 210, 211,
            211, 212, 213, 213, 214, 214, 215, 216, 216, 217, 217, 218, 218, 219, 220, 220,
            221, 221, 222, 222, 223, 224, 224, 225, 225, 226, 226, 227, 228, 228, 229, 229,
            230, 230, 231, 231, 232, 233, 233, 234, 234, 235, 235, 236, 236, 237, 237, 238,
            238, 239, 240, 240, 241, 241, 242, 242, 243, 243, 244, 244, 245, 245, 246, 246,
            247, 247, 248, 248, 249, 249, 250, 250, 251, 251, 252, 252, 253, 253, 254, 255};
        auto D = (d <= 64) ? (4 * d - (12 * d * d) / 255 + (16 * d * d * d) / 65025) : SQRT_LUT[d];
        return static_cast<uint8_t>(d + ((2 * s - 255) * (D - d)) / 255);
    };

    return _premultiply(JOIN(255, f(C1(s), o.r), f(C2(s), o.g), f(C3(s), o.b)), s, o.a);
}

uint32_t blendHue(const SwSurface* surface, uint32_t s, uint32_t d)
{
    auto o = _unpremultiply(d);

    auto c1 = C1(s);
    auto c2 = C2(s);
    auto c3 = C3(s);
    _saturate(c1, c2, c3, o.r, o.g, o.b);
    _luminance(c1, c2, c3, surface->luma((c1 << 16 | c2 << 8 | c3)), surface->luma((o.r << 16 | o.g << 8 | o.b)));

    return _premultiply(JOIN(255, c1, c2, c3), s, o.a);
}

uint32_t blendSaturation(const SwSurface* surface, uint32_t s, uint32_t d)
{
    auto o = _unpremultiply(d);

    auto c1 = o.r;
    auto c2 = o.g;
    auto c3 = o.b;
    _saturate(c1, c2, c3, C1(s), C2(s), C3(s));
    _luminance(c1, c2, c3, surface->luma((c1 << 16 | c2 << 8 | c3)), surface->luma((o.r << 16 | o.g << 8 | o.b)));

    return _premultiply(JOIN(255, c1, c2, c3), s, o.a);
}

uint32_t blendColor(const SwSurface* surface, uint32_t s, uint32_t d)
{
    auto o = _unpremultiply(d);

    auto c1 = C1(s);
    auto c2 = C2(s);
    auto c3 = C3(s);
    _luminance(c1, c2, c3, surface->luma(s), surface->luma((o.r << 16 | o.g << 8 | o.b)));

    return _premultiply(JOIN(255, c1, c2, c3), s, o.a);
}

uint32_t blendLuminosity(const SwSurface* surface, uint32_t s, uint32_t d)
{
    auto o = _unpremultiply(d);

    _luminance(o.r, o.g, o.b, surface->luma((o.r << 16 | o.g << 8 | o.b)), surface->luma(s));

    return _premultiply(JOIN(255, o.r, o.g, o.b), s, o.a);
}
