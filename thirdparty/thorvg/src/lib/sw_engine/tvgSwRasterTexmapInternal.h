/*
 * Copyright (c) 2021 - 2022 Samsung Electronics Co., Ltd. All rights reserved.

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
{
    float _dudx = dudx, _dvdx = dvdx;
    float _dxdya = dxdya, _dxdyb = dxdyb, _dudya = dudya, _dvdya = dvdya;
    float _xa = xa, _xb = xb, _ua = ua, _va = va;
    auto sbuf = image->data;
    auto dbuf = surface->buffer;
    int32_t sw = static_cast<int32_t>(image->stride);
    int32_t sh = image->h;
    int32_t dw = surface->stride;
    int32_t x1, x2, x, y, ar, ab, iru, irv, px, ay;
    int32_t vv = 0, uu = 0;
    int32_t minx = INT32_MAX, maxx = INT32_MIN;
    float dx, u, v, iptr;
    uint32_t* buf;
    SwSpan* span = nullptr;         //used only when rle based.

#ifdef TEXMAP_MASKING
    uint32_t* cmp;
#endif

    if (!_arrange(image, region, yStart, yEnd)) return;

    //Loop through all lines in the segment
    uint32_t spanIdx = 0;

    if (region) {
        minx = region->min.x;
        maxx = region->max.x;
    } else {
        span = image->rle->spans;
        while (span->y < yStart) {
            ++span;
            ++spanIdx;
        }
    }

    y = yStart;

    while (y < yEnd) {
        x1 = (int32_t)_xa;
        x2 = (int32_t)_xb;

        if (!region) {
            minx = INT32_MAX;
            maxx = INT32_MIN;
            //one single row, could be consisted of multiple spans.
            while (span->y == y && spanIdx < image->rle->size) {
                if (minx > span->x) minx = span->x;
                if (maxx < span->x + span->len) maxx = span->x + span->len;
                ++span;
                ++spanIdx;
            }
        }
        if (x1 < minx) x1 = minx;
        if (x2 > maxx) x2 = maxx;

        //Anti-Aliasing frames
        ay = y - aaSpans->yStart;
        if (aaSpans->lines[ay].x[0] > x1) aaSpans->lines[ay].x[0] = x1;
        if (aaSpans->lines[ay].x[1] < x2) aaSpans->lines[ay].x[1] = x2;

        //Range exception
        if ((x2 - x1) < 1 || (x1 >= maxx) || (x2 <= minx)) goto next;

        //Perform subtexel pre-stepping on UV
        dx = 1 - (_xa - x1);
        u = _ua + dx * _dudx;
        v = _va + dx * _dvdx;

        buf = dbuf + ((y * dw) + x1);

        x = x1;

#ifdef TEXMAP_MASKING
        cmp = &surface->compositor->image.data[y * surface->compositor->image.stride + x1];
#endif
        //Draw horizontal line
        while (x++ < x2) {
            uu = (int) u;
            vv = (int) v;

            ar = (int)(255 * (1 - modff(u, &iptr)));
            ab = (int)(255 * (1 - modff(v, &iptr)));
            iru = uu + 1;
            irv = vv + 1;
            px = *(sbuf + (vv * sw) + uu);

            /* horizontal interpolate */
            if (iru < sw) {
                /* right pixel */
                int px2 = *(sbuf + (vv * sw) + iru);
                px = INTERPOLATE(ar, px, px2);
            }
            /* vertical interpolate */
            if (irv < sh) {
                /* bottom pixel */
                int px2 = *(sbuf + (irv * sw) + uu);

                /* horizontal interpolate */
                if (iru < sw) {
                    /* bottom right pixel */
                    int px3 = *(sbuf + (irv * sw) + iru);
                    px2 = INTERPOLATE(ar, px2, px3);
                }
                px = INTERPOLATE(ab, px, px2);
            }
#if defined(TEXMAP_MASKING) && defined(TEXMAP_TRANSLUCENT)
            auto src = ALPHA_BLEND(px, _multiplyAlpha(opacity, blendMethod(*cmp)));
#elif defined(TEXMAP_MASKING)
            auto src = ALPHA_BLEND(px, blendMethod(*cmp));
#elif defined(TEXMAP_TRANSLUCENT)
            auto src = ALPHA_BLEND(px, opacity);
#else
            auto src = px;
#endif
            *buf = src + ALPHA_BLEND(*buf, _ialpha(src));
            ++buf;
#ifdef TEXMAP_MASKING
            ++cmp;
#endif
            //Step UV horizontally
            u += _dudx;
            v += _dvdx;
            //range over?
            if ((uint32_t)v >= image->h) break;
        }
next:
        //Step along both edges
        _xa += _dxdya;
        _xb += _dxdyb;
        _ua += _dudya;
        _va += _dvdya;

        if (!region && spanIdx >= image->rle->size) break;

        ++y;
    }
    xa = _xa;
    xb = _xb;
    ua = _ua;
    va = _va;
}
