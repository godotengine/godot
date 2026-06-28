#pragma once

#include <stdint.h>

namespace cvtt { namespace Tables { namespace BC7Prio {
    extern const uint16_t *g_bc7PrioCodesRGB;
    extern const int g_bc7NumPrioCodesRGB;

    extern const uint16_t *g_bc7PrioCodesRGBA;
    extern const int g_bc7NumPrioCodesRGBA;

    int UnpackMode(uint16_t packed);
    int UnpackSeedPointCount(uint16_t packed);
    int UnpackPartition(uint16_t packed);
    int UnpackRotation(uint16_t packed);
    int UnpackIndexSelector(uint16_t packed);
}}}
