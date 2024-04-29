
#pragma once

#include "arithmetics.hpp"
#include "Vector2.h"
#include "BitmapRef.hpp"

namespace msdfgen {

template <typename T, int N>
static void interpolate(T *output, const BitmapConstRef<T, N> &bitmap, Point2 pos) {
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
