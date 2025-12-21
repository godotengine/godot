
#pragma once

#include "arithmetics.hpp"
#include "Vector2.hpp"
#include "BitmapRef.hpp"

namespace msdfgen {

template <typename T, int N>
inline void interpolate(T *output, const BitmapConstSection<T, N> &bitmap, Point2 pos) {
    pos.x = clamp(pos.x, double(bitmap.width));
    pos.y = clamp(pos.y, double(bitmap.height));
    pos -= .5;
    int l = (int) floor(pos.x);
    int b = (int) floor(pos.y);
    int r = l+1;
    int t = b+1;
    double lr = pos.x-l;
    double bt = pos.y-b;
    l = clamp(l, bitmap.width-1), r = clamp(r, bitmap.width-1);
    b = clamp(b, bitmap.height-1), t = clamp(t, bitmap.height-1);
    for (int i = 0; i < N; ++i)
        output[i] = mix(mix(bitmap(l, b)[i], bitmap(r, b)[i], lr), mix(bitmap(l, t)[i], bitmap(r, t)[i], lr), bt);
}

}
