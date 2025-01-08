
#pragma once

#include "arithmetics.hpp"

namespace msdfgen {

inline byte pixelFloatToByte(float x) {
    return byte(clamp(256.f*x, 255.f));
}

inline float pixelByteToFloat(byte x) {
    return 1.f/255.f*float(x);
}

}
