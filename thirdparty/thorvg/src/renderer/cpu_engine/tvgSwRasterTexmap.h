/*
 * Copyright (c) 2021 - 2026 ThorVG project. All rights reserved.

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

namespace tvg
{

struct Vertex
{
   Point pt;
   Point uv;
};

struct Polygon
{
   Vertex vertex[3];
};

struct TexmapCtx
{
    SwSurface* surface;
    const SwImage& image;
    const RenderRegion& bbox;
    uint8_t opacity;

    // Each textured polygon owns its scan-converter ctx.
    float dudx, dvdx;
    float dxdya, dxdyb, dudya, dvdya;
    float xa, xb, ua, va;
};

static inline int32_t _modf(float v)
{
    return 255 - ((int(v * 256.0f)) & 255);
}

static uint8_t _feathering(int iru, int irv, int ar, int ab, int sw, int sh)
{
    if (irv == 1) {
        if (iru == 1) return 255 - MULTIPLY(ar, ab);
        else if (iru == sw) return MULTIPLY(ar, 255 - ab);
        return 255 - ab;
    } else if (irv == sh) {
        if (iru == 1) return MULTIPLY(255 - ar, ab);
        else if (iru == sw) return MULTIPLY(ar, ab);
        return ab;
    } else {
        if (iru == 1) return 255 - ar;
        else if (iru == sw) return ar;
    }
    return 255;
}

static bool _rasterMaskedPolygonImageSegment(TexmapCtx& ctx, int yStart, int yEnd, bool needAA)
{
    TVGERR("SW_ENGINE", "TODO: _rasterMaskedPolygonImageSegment()");
    return false;
}

static void _rasterBlendingPolygonImageSegment(TexmapCtx& ctx, int yStart, int yEnd, bool needAA)
{
    if (ctx.surface->channelSize == sizeof(uint8_t)) {
        TVGERR("SW_ENGINE", "Not supported grayscale Textmap polygon!");
        return;
    }

    auto _dudx = ctx.dudx, _dvdx = ctx.dvdx;
    auto _dxdya = ctx.dxdya, _dxdyb = ctx.dxdyb, _dudya = ctx.dudya, _dvdya = ctx.dvdya;
    auto _xa = ctx.xa, _xb = ctx.xb, _ua = ctx.ua, _va = ctx.va;
    auto sbuf = ctx.image.buf32;
    auto dbuf = ctx.surface->buf32;
    auto sw = static_cast<int32_t>(ctx.image.w);
    auto sh = static_cast<int32_t>(ctx.image.h);
    int32_t x1, x2, x, y, ar, ab, iru, irv, px;
    int32_t vv = 0, uu = 0;
    float dx, u, v;
    uint32_t* buf;

    if (yStart < ctx.bbox.yMin) yStart = ctx.bbox.yMin;
    if (yEnd > ctx.bbox.yMax) yEnd = ctx.bbox.yMax;

    y = yStart;

    while (y < yEnd) {
        x1 = std::max((int32_t)_xa, ctx.bbox.xMin);
        x2 = std::min((int32_t)_xb, ctx.bbox.xMax);

        //Range allowed
        if ((x2 - x1) >= 1 && (x1 < ctx.bbox.xMax) && (x2 > ctx.bbox.xMin)) {
            //Perform subtexel pre-stepping on UV
            dx = 1 - (_xa - x1);
            u = _ua + dx * _dudx;
            v = _va + dx * _dvdx;
            buf = dbuf + ((y * ctx.surface->stride) + x1);
            x = x1;

            //Draw horizontal line
            while (x++ < x2) {
                uu = (int) u;
                vv = (int) v;

                if ((uint32_t)uu >= ctx.image.w || (uint32_t)vv >= ctx.image.h) continue;

                ar = _modf(u);
                ab = _modf(v);
                iru = uu + 1;
                irv = vv + 1;

                px = *(sbuf + (vv * ctx.image.stride) + uu);

                if (ctx.image.filter == FilterMethod::Bilinear) {
                    // horizontal interpolate
                    if (iru < sw) {
                        int px2 = *(sbuf + (vv * ctx.image.stride) + iru);
                        px = INTERPOLATE(px, px2, ar);
                    }
                    // vertical interpolate
                    if (irv < sh) {
                        int px2 = *(sbuf + (irv * ctx.image.stride) + uu);
                        // horizontal interpolate
                        if (iru < sw) {
                            int px3 = *(sbuf + (irv * ctx.image.stride) + iru);
                            px2 = INTERPOLATE(px2, px3, ar);
                        }
                        px = INTERPOLATE(px, px2, ab);
                    }
                }

                // anti-aliasing
                if (needAA) {
                    auto feather = _feathering(iru, irv, ar, ab, sw, sh);
                    if (feather < 255) px = ALPHA_BLEND(px, feather);
                }

                *buf = INTERPOLATE(ctx.surface->blender(ctx.surface, UNPREMULTIPLY(px), *buf), *buf, MULTIPLY(ctx.opacity, A(px)));
                ++buf;

                //Step UV horizontally
                u += _dudx;
                v += _dvdx;
            }
        }

        //Step along both edges
        _xa += _dxdya;
        _xb += _dxdyb;
        _ua += _dudya;
        _va += _dvdya;

        ++y;
    }
    ctx.xa = _xa;
    ctx.xb = _xb;
    ctx.ua = _ua;
    ctx.va = _va;
}

static void _rasterPolygonImageSegment32(TexmapCtx& ctx, int yStart, int yEnd, bool matting, bool needAA)
{
    auto _dudx = ctx.dudx, _dvdx = ctx.dvdx;
    auto _dxdya = ctx.dxdya, _dxdyb = ctx.dxdyb, _dudya = ctx.dudya, _dvdya = ctx.dvdya;
    auto _xa = ctx.xa, _xb = ctx.xb, _ua = ctx.ua, _va = ctx.va;
    auto sbuf = ctx.image.buf32;
    auto dbuf = ctx.surface->buf32;
    auto sw = static_cast<int32_t>(ctx.image.w);
    auto sh = static_cast<int32_t>(ctx.image.h);
    int32_t x1, x2, x, y, ar, ab, iru, irv, px;
    int32_t vv = 0, uu = 0;
    float dx, u, v;
    uint32_t* buf;
    auto fullOpacity = (ctx.opacity == 255);

    //for matting(composition)
    auto csize = matting ? ctx.surface->compositor->image.channelSize : 0;
    auto alpha = matting ? ctx.surface->alpha(ctx.surface->compositor->method) : nullptr;
    uint8_t* cmp = nullptr;

    if (yStart < ctx.bbox.yMin) yStart = ctx.bbox.yMin;
    if (yEnd > ctx.bbox.yMax) yEnd = ctx.bbox.yMax;

    y = yStart;

    while (y < yEnd) {
        x1 = std::max((int32_t)_xa, ctx.bbox.xMin);
        x2 = std::min((int32_t)_xb, ctx.bbox.xMax);

        //Range allowed
        if ((x2 - x1) >= 1 && (x1 < ctx.bbox.xMax) && (x2 > ctx.bbox.xMin)) {
            //Perform subtexel pre-stepping on UV
            dx = 1 - (_xa - x1);
            u = _ua + dx * _dudx;
            v = _va + dx * _dvdx;
            buf = dbuf + ((y * ctx.surface->stride) + x1);
            x = x1;

            if (matting) cmp = &ctx.surface->compositor->image.buf8[(y * ctx.surface->compositor->image.stride + x1) * csize];

            //Draw horizontal line
            while (x++ < x2) {
                uu = (int) u;
                vv = (int) v;

                if ((uint32_t)uu >= ctx.image.w || (uint32_t)vv >= ctx.image.h) continue;

                ar = _modf(u);
                ab = _modf(v);
                iru = uu + 1;
                irv = vv + 1;

                px = *(sbuf + (vv * ctx.image.stride) + uu);

                if (ctx.image.filter == FilterMethod::Bilinear) {
                    // horizontal interpolate
                    if (iru < sw) {
                        int px2 = *(sbuf + (vv * ctx.image.stride) + iru);
                        px = INTERPOLATE(px, px2, ar);
                    }
                    // vertical interpolate
                    if (irv < sh) {
                        int px2 = *(sbuf + (irv * ctx.image.stride) + uu);
                        // horizontal interpolate
                        if (iru < sw) {
                            int px3 = *(sbuf + (irv * ctx.image.stride) + iru);
                            px2 = INTERPOLATE(px2, px3, ar);
                        }
                        px = INTERPOLATE(px, px2, ab);
                    }
                }

                uint32_t src;
                if (matting) {
                    auto a = alpha(cmp);
                    src = fullOpacity ? ALPHA_BLEND(px, a) : ALPHA_BLEND(px, MULTIPLY(ctx.opacity, a));
                    cmp += csize;
                } else {
                    src = fullOpacity ? px : ALPHA_BLEND(px, ctx.opacity);
                }

                // anti-aliasing
                if (needAA) {
                    auto feather = _feathering(iru, irv, ar, ab, sw, sh);
                    if (feather < 255) src = ALPHA_BLEND(src, feather);
                }

                *buf = src + ALPHA_BLEND(*buf, IA(src));
                ++buf;

                //Step UV horizontally
                u += _dudx;
                v += _dvdx;
            }
        }

        //Step along both edges
        _xa += _dxdya;
        _xb += _dxdyb;
        _ua += _dudya;
        _va += _dvdya;

        ++y;
    }
    ctx.xa = _xa;
    ctx.xb = _xb;
    ctx.ua = _ua;
    ctx.va = _va;
}

// no anti-aliasing, no interpolation for the fastest cheap masking
static void _rasterPolygonImageSegment8(TexmapCtx& ctx, int yStart, int yEnd, TVG_UNUSED bool needAA)
{
    auto _dudx = ctx.dudx, _dvdx = ctx.dvdx;
    auto _dxdya = ctx.dxdya, _dxdyb = ctx.dxdyb, _dudya = ctx.dudya, _dvdya = ctx.dvdya;
    auto _xa = ctx.xa, _xb = ctx.xb, _ua = ctx.ua, _va = ctx.va;
    auto sbuf = ctx.image.buf32;
    auto dbuf = ctx.surface->buf8;
    int32_t x1, x2, x, y;
    float dx, u, v;
    uint8_t* buf;
    uint8_t px;

    if (yStart < ctx.bbox.yMin) yStart = ctx.bbox.yMin;
    if (yEnd > ctx.bbox.yMax) yEnd = ctx.bbox.yMax;

    y = yStart;

    while (y < yEnd) {
        x1 = std::max((int32_t)_xa, ctx.bbox.xMin);
        x2 = std::min((int32_t)_xb, ctx.bbox.xMax);

        //Range allowed
        if ((x2 - x1) >= 1 && (x1 < ctx.bbox.xMax) && (x2 > ctx.bbox.xMin)) {
            //Perform subtexel pre-stepping on UV
            dx = 1 - (_xa - x1);
            u = _ua + dx * _dudx;
            v = _va + dx * _dvdx;
            buf = dbuf + ((y * ctx.surface->stride) + x1);
            x = x1;
            //Draw horizontal line
            while (x++ < x2) {
                auto uu = (int) u;
                auto vv = (int) v;
                if ((uint32_t)uu >= ctx.image.w || (uint32_t)vv >= ctx.image.h) continue;

                px = A(*(sbuf + (vv * ctx.image.stride) + uu));
                *buf = MULTIPLY(px, ctx.opacity);
                ++buf;
                //Step UV horizontally
                u += _dudx;
                v += _dvdx;
            }
        }
        //Step along both edges
        _xa += _dxdya;
        _xb += _dxdyb;
        _ua += _dudya;
        _va += _dvdya;
        ++y;
    }
    ctx.xa = _xa;
    ctx.xb = _xb;
    ctx.ua = _ua;
    ctx.va = _va;
}

static void _rasterPolygonImageSegment(TexmapCtx& ctx, int yStart, int yEnd, bool matting, bool needAA)
{
    if (ctx.surface->channelSize == sizeof(uint32_t)) _rasterPolygonImageSegment32(ctx, yStart, yEnd, matting, needAA);
    else if (ctx.surface->channelSize == sizeof(uint8_t)) _rasterPolygonImageSegment8(ctx, yStart, yEnd, needAA);
}


/* This mapping algorithm is based on Mikael Kalms's. */
static void _rasterPolygonImage(TexmapCtx& ctx, Polygon& polygon, bool needAA)
{
    float x[3] = {polygon.vertex[0].pt.x, polygon.vertex[1].pt.x, polygon.vertex[2].pt.x};
    float y[3] = {polygon.vertex[0].pt.y, polygon.vertex[1].pt.y, polygon.vertex[2].pt.y};
    float u[3] = {polygon.vertex[0].uv.x, polygon.vertex[1].uv.x, polygon.vertex[2].uv.x};
    float v[3] = {polygon.vertex[0].uv.y, polygon.vertex[1].uv.y, polygon.vertex[2].uv.y};

    float off_y;
    float dxdy[3] = {0.0f, 0.0f, 0.0f};

    auto upper = false;

    //Sort the vertices in ascending Y order
    if (y[0] > y[1]) {
        std::swap(x[0], x[1]);
        std::swap(y[0], y[1]);
        std::swap(u[0], u[1]);
        std::swap(v[0], v[1]);
    }
    if (y[0] > y[2])  {
        std::swap(x[0], x[2]);
        std::swap(y[0], y[2]);
        std::swap(u[0], u[2]);
        std::swap(v[0], v[2]);
    }
    if (y[1] > y[2]) {
        std::swap(x[1], x[2]);
        std::swap(y[1], y[2]);
        std::swap(u[1], u[2]);
        std::swap(v[1], v[2]);
    }

    //Y indexes
    int yi[3] = {(int)y[0], (int)y[1], (int)y[2]};

    //Skip drawing if it's too thin to cover any pixels at all.
    if ((yi[0] == yi[1] && yi[0] == yi[2]) || ((int) x[0] == (int) x[1] && (int) x[0] == (int) x[2])) return;

    //Calculate horizontal and vertical increments for UV axes (these calcs are certainly not optimal, although they're stable (handles any dy being 0)
    auto denom = ((x[2] - x[0]) * (y[1] - y[0]) - (x[1] - x[0]) * (y[2] - y[0]));

    //Skip poly if it's an infinitely thin line
    if (tvg::zero(denom)) return;

    denom = 1 / denom;   //Reciprocal for speeding up
    ctx.dudx = ((u[2] - u[0]) * (y[1] - y[0]) - (u[1] - u[0]) * (y[2] - y[0])) * denom;
    ctx.dvdx = ((v[2] - v[0]) * (y[1] - y[0]) - (v[1] - v[0]) * (y[2] - y[0])) * denom;
    auto dudy = ((u[1] - u[0]) * (x[2] - x[0]) - (u[2] - u[0]) * (x[1] - x[0])) * denom;
    auto dvdy = ((v[1] - v[0]) * (x[2] - x[0]) - (v[2] - v[0]) * (x[1] - x[0])) * denom;

    //Calculate X-slopes along the edges
    if (y[1] > y[0]) dxdy[0] = (x[1] - x[0]) / (y[1] - y[0]);
    if (y[2] > y[0]) dxdy[1] = (x[2] - x[0]) / (y[2] - y[0]);
    if (y[2] > y[1]) dxdy[2] = (x[2] - x[1]) / (y[2] - y[1]);

    //Determine which side of the polygon the longer edge is on
    auto side = (dxdy[1] > dxdy[0]) ? true : false;

    if (tvg::equal(y[0], y[1])) side = x[0] > x[1];
    if (tvg::equal(y[1], y[2])) side = x[2] > x[1];

    auto compositing = _compositing(ctx.surface);  // Composition required
    auto blending = _blending(ctx.surface);        // Blending required

    //Longer edge is on the left side
    if (!side) {
        //Calculate slopes along left edge
        ctx.dxdya = dxdy[1];
        ctx.dudya = ctx.dxdya * ctx.dudx + dudy;
        ctx.dvdya = ctx.dxdya * ctx.dvdx + dvdy;

        //Perform subpixel pre-stepping along left edge
        auto dy = 1.0f - (y[0] - yi[0]);
        ctx.xa = x[0] + dy * ctx.dxdya;
        ctx.ua = u[0] + dy * ctx.dudya;
        ctx.va = v[0] + dy * ctx.dvdya;

        //Draw upper segment if possibly visible
        if (yi[0] < yi[1]) {
            off_y = y[0] < ctx.bbox.yMin ? (ctx.bbox.yMin - y[0]) : 0;
            ctx.xa += (off_y * ctx.dxdya);
            ctx.ua += (off_y * ctx.dudya);
            ctx.va += (off_y * ctx.dvdya);

            // Set right edge X-slope and perform subpixel pre-stepping
            ctx.dxdyb = dxdy[0];
            ctx.xb = x[0] + dy * ctx.dxdyb + (off_y * ctx.dxdyb);

            if (compositing) {
                if (_matting(ctx.surface)) _rasterPolygonImageSegment(ctx, yi[0], yi[1], true, needAA);
                else _rasterMaskedPolygonImageSegment(ctx, yi[0], yi[1], needAA);
            } else if (blending) {
                _rasterBlendingPolygonImageSegment(ctx, yi[0], yi[1], needAA);
            } else {
                _rasterPolygonImageSegment(ctx, yi[0], yi[1], false, needAA);
            }
            upper = true;
        }
        //Draw lower segment if possibly visible
        if (yi[1] < yi[2]) {
            off_y = y[1] < ctx.bbox.yMin ? (ctx.bbox.yMin - y[1]) : 0;
            if (!upper) {
                ctx.xa += (off_y * ctx.dxdya);
                ctx.ua += (off_y * ctx.dudya);
                ctx.va += (off_y * ctx.dvdya);
            }
            // Set right edge X-slope and perform subpixel pre-stepping
            ctx.dxdyb = dxdy[2];
            ctx.xb = x[1] + (1 - (y[1] - yi[1])) * ctx.dxdyb + (off_y * ctx.dxdyb);
            if (compositing) {
                if (_matting(ctx.surface)) _rasterPolygonImageSegment(ctx, yi[1], yi[2], true, needAA);
                else _rasterMaskedPolygonImageSegment(ctx, yi[1], yi[2], needAA);
            } else if (blending) {
                _rasterBlendingPolygonImageSegment(ctx, yi[1], yi[2], needAA);
            } else {
                _rasterPolygonImageSegment(ctx, yi[1], yi[2], false, needAA);
            }
        }
    //Longer edge is on the right side
    } else {
        //Set right edge X-slope and perform subpixel pre-stepping
        ctx.dxdyb = dxdy[1];
        auto dy = 1.0f - (y[0] - yi[0]);
        ctx.xb = x[0] + dy * ctx.dxdyb;

        //Draw upper segment if possibly visible
        if (yi[0] < yi[1]) {
            off_y = y[0] < ctx.bbox.yMin ? (ctx.bbox.yMin - y[0]) : 0;
            ctx.xb += (off_y * ctx.dxdyb);

            // Set slopes along left edge and perform subpixel pre-stepping
            ctx.dxdya = dxdy[0];
            ctx.dudya = ctx.dxdya * ctx.dudx + dudy;
            ctx.dvdya = ctx.dxdya * ctx.dvdx + dvdy;

            ctx.xa = x[0] + dy * ctx.dxdya + (off_y * ctx.dxdya);
            ctx.ua = u[0] + dy * ctx.dudya + (off_y * ctx.dudya);
            ctx.va = v[0] + dy * ctx.dvdya + (off_y * ctx.dvdya);

            if (compositing) {
                if (_matting(ctx.surface)) _rasterPolygonImageSegment(ctx, yi[0], yi[1], true, needAA);
                else _rasterMaskedPolygonImageSegment(ctx, yi[0], yi[1], needAA);
            } else if (blending) {
                _rasterBlendingPolygonImageSegment(ctx, yi[0], yi[1], needAA);
            } else {
                _rasterPolygonImageSegment(ctx, yi[0], yi[1], false, needAA);
            }
            upper = true;
        }
        //Draw lower segment if possibly visible
        if (yi[1] < yi[2]) {
            off_y = y[1] < ctx.bbox.yMin ? (ctx.bbox.yMin - y[1]) : 0;
            if (!upper) ctx.xb += (off_y * ctx.dxdyb);

            // Set slopes along left edge and perform subpixel pre-stepping
            ctx.dxdya = dxdy[2];
            ctx.dudya = ctx.dxdya * ctx.dudx + dudy;
            ctx.dvdya = ctx.dxdya * ctx.dvdx + dvdy;
            dy = 1 - (y[1] - yi[1]);
            ctx.xa = x[1] + dy * ctx.dxdya + (off_y * ctx.dxdya);
            ctx.ua = u[1] + dy * ctx.dudya + (off_y * ctx.dudya);
            ctx.va = v[1] + dy * ctx.dvdya + (off_y * ctx.dvdya);

            if (compositing) {
                if (_matting(ctx.surface)) _rasterPolygonImageSegment(ctx, yi[1], yi[2], true, needAA);
                else _rasterMaskedPolygonImageSegment(ctx, yi[1], yi[2], needAA);
            } else if (blending) {
                _rasterBlendingPolygonImageSegment(ctx, yi[1], yi[2], needAA);
            } else {
                _rasterPolygonImageSegment(ctx, yi[1], yi[2], false, needAA);
            }
        }
    }
}

} //namespace


/*
    2 triangles constructs 1 mesh.
    below figure illustrates vert[4] index info.
    If you need better quality, please divide a mesh by more number of triangles.

    0 -- 1
    |  / |
    | /  |
    3 -- 2
*/
bool rasterTexmapPolygon(SwSurface* surface, const SwImage& image, const Matrix& transform, const RenderRegion& bbox, uint8_t opacity)
{
    //Prepare vertices. Shift XY coordinates to match the sub-pixeling technique.
    TexmapCtx ctx = {surface, image, bbox, opacity};
    Vertex vertices[4];
    vertices[0] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
    vertices[1] = {{float(image.w), 0.0f}, {float(image.w), 0.0f}};
    vertices[2] = {{float(image.w), float(image.h)}, {float(image.w), float(image.h)}};
    vertices[3] = {{0.0f, float(image.h)}, {0.0f, float(image.h)}};

    float ys = FLT_MAX, ye = -1.0f;
    for (int i = 0; i < 4; i++) {
        vertices[i].pt *= transform;
        if (vertices[i].pt.y < ys) ys = vertices[i].pt.y;
        if (vertices[i].pt.y > ye) ye = vertices[i].pt.y;
    }

    auto needAA = rightAngle(transform) ? false : true;

    tvg::Polygon polygon;

    //Draw the first polygon
    polygon.vertex[0] = vertices[0];
    polygon.vertex[1] = vertices[1];
    polygon.vertex[2] = vertices[3];

    _rasterPolygonImage(ctx, polygon, needAA);

    //Draw the second polygon
    polygon.vertex[0] = vertices[1];
    polygon.vertex[1] = vertices[2];
    polygon.vertex[2] = vertices[3];

    _rasterPolygonImage(ctx, polygon, needAA);

#if 0
    if (_compositing(surface) && _masking(surface) && !_direct(surface->compositor->method)) {
        _compositeMaskImage(surface, &surface->compositor->image, surface->compositor->bbox);
    }
#endif
    return true;
}
