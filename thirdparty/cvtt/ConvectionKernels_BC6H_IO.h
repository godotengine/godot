#pragma once

#include <stdint.h>
#include "ConvectionKernels_BC6H_IO.h"

namespace cvtt
{
    namespace BC6H_IO
    {
        typedef void (*ReadFunc_t)(const uint32_t *encoded, uint16_t &d, uint16_t &rw, uint16_t &rx, uint16_t &ry, uint16_t &rz, uint16_t &gw, uint16_t &gx, uint16_t &gy, uint16_t &gz, uint16_t &bw, uint16_t &bx, uint16_t &by, uint16_t &bz);
        typedef void (*WriteFunc_t)(uint32_t *encoded, uint16_t m, uint16_t d, uint16_t rw, uint16_t rx, uint16_t ry, uint16_t rz, uint16_t gw, uint16_t gx, uint16_t gy, uint16_t gz, uint16_t bw, uint16_t bx, uint16_t by, uint16_t bz);

        extern const ReadFunc_t g_readFuncs[14];
        extern const WriteFunc_t g_writeFuncs[14];
    }
}
