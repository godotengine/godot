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

#include "tvgMath.h"
#include "tvgColor.h"
namespace tvg
{

void hsl2rgb(float h, float s, float l, uint8_t& r, uint8_t& g, uint8_t& b)
{
    if (tvg::zero(s))  {
        r = g = b = (uint8_t)nearbyint(l * 255.0f);
        return;
    } 

    if (tvg::equal(h, 360.0f)) {
        h = 0.0f;
    } else {
        h = fmod(h, 360.0f);
        if (h < 0.0f) h += 360.0f;
        h /= 60.0f;
    }

    auto v = (l <= 0.5f) ? (l * (1.0f + s)) : (l + s - (l * s));
    auto p = l + l - v;
    auto sv = tvg::zero(v) ? 0.0f : (v - p) / v;
    auto i = static_cast<uint8_t>(h);
    auto f = h - i;
    auto vsf = v * sv * f;
    auto t = p + vsf;
    auto q = v - vsf;
    float tr, tg, tb;

    switch (i) {
        case 0: tr = v; tg = t; tb = p; break;
        case 1: tr = q; tg = v; tb = p; break;
        case 2: tr = p; tg = v; tb = t; break;
        case 3: tr = p; tg = q; tb = v; break;
        case 4: tr = t; tg = p; tb = v; break;
        case 5: tr = v; tg = p; tb = q; break;
        default: tr = tg = tb = 0.0f; break;
    }
    r = (uint8_t)nearbyint(tr * 255.0f);
    g = (uint8_t)nearbyint(tg * 255.0f);
    b = (uint8_t)nearbyint(tb * 255.0f);
}

}
