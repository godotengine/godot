/*
Convection Texture Tools
Copyright (c) 2018-2019 Eric Lasota

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject
to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

-------------------------------------------------------------------------------------

Portions based on DirectX Texture Library (DirectXTex)

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.

http://go.microsoft.com/fwlink/?LinkId=248926
*/
#include "ConvectionKernels_Config.h"

#if !defined(CVTT_SINGLE_FILE) || defined(CVTT_SINGLE_FILE_IMPL)

#include "ConvectionKernels_BC6H_IO.h"

namespace cvtt
{
    namespace BC6H_IO
    {
        void WriteMode0(uint32_t *encoded, uint16_t m, uint16_t d, uint16_t rw, uint16_t rx, uint16_t ry, uint16_t rz, uint16_t gw, uint16_t gx, uint16_t gy, uint16_t gz, uint16_t bw, uint16_t bx, uint16_t by, uint16_t bz)
        {
            encoded[0] = (m & 0x3u) | ((gy >> 2) & 0x4u) | ((by >> 1) & 0x8u) | (bz & 0x10u) | ((rw << 5) & 0x7fe0u) | ((gw << 15) & 0x1ff8000u) | ((bw << 25) & 0xfe000000u);
            encoded[1] = ((bw >> 7) & 0x7u) | ((rx << 3) & 0xf8u) | ((gz << 4) & 0x100u) | ((gy << 9) & 0x1e00u) | ((gx << 13) & 0x3e000u) | ((bz << 18) & 0x40000u) | ((gz << 19) & 0x780000u) | ((bx << 23) & 0xf800000u) | ((bz << 27) & 0x10000000u) | ((by << 29) & 0xe0000000u);
            encoded[2] = ((by >> 3) & 0x1u) | ((ry << 1) & 0x3eu) | ((bz << 4) & 0x40u) | ((rz << 7) & 0xf80u) | ((bz << 9) & 0x1000u) | ((d << 13) & 0x3e000u);
        }

        void WriteMode1(uint32_t *encoded, uint16_t m, uint16_t d, uint16_t rw, uint16_t rx, uint16_t ry, uint16_t rz, uint16_t gw, uint16_t gx, uint16_t gy, uint16_t gz, uint16_t bw, uint16_t bx, uint16_t by, uint16_t bz)
        {
            encoded[0] = (m & 0x3u) | ((gy >> 3) & 0x4u) | ((gz >> 1) & 0x18u) | ((rw << 5) & 0xfe0u) | ((bz << 12) & 0x3000u) | ((by << 10) & 0x4000u) | ((gw << 15) & 0x3f8000u) | ((by << 17) & 0x400000u) | ((bz << 21) & 0x800000u) | ((gy << 20) & 0x1000000u) | ((bw << 25) & 0xfe000000u);
            encoded[1] = ((bz >> 3) & 0x1u) | ((bz >> 4) & 0x2u) | ((bz >> 2) & 0x4u) | ((rx << 3) & 0x1f8u) | ((gy << 9) & 0x1e00u) | ((gx << 13) & 0x7e000u) | ((gz << 19) & 0x780000u) | ((bx << 23) & 0x1f800000u) | ((by << 29) & 0xe0000000u);
            encoded[2] = ((by >> 3) & 0x1u) | ((ry << 1) & 0x7eu) | ((rz << 7) & 0x1f80u) | ((d << 13) & 0x3e000u);
        }

        void WriteMode2(uint32_t *encoded, uint16_t m, uint16_t d, uint16_t rw, uint16_t rx, uint16_t ry, uint16_t rz, uint16_t gw, uint16_t gx, uint16_t gy, uint16_t gz, uint16_t bw, uint16_t bx, uint16_t by, uint16_t bz)
        {
            encoded[0] = (m & 0x1fu) | ((rw << 5) & 0x7fe0u) | ((gw << 15) & 0x1ff8000u) | ((bw << 25) & 0xfe000000u);
            encoded[1] = ((bw >> 7) & 0x7u) | ((rx << 3) & 0xf8u) | ((rw >> 2) & 0x100u) | ((gy << 9) & 0x1e00u) | ((gx << 13) & 0x1e000u) | ((gw << 7) & 0x20000u) | ((bz << 18) & 0x40000u) | ((gz << 19) & 0x780000u) | ((bx << 23) & 0x7800000u) | ((bw << 17) & 0x8000000u) | ((bz << 27) & 0x10000000u) | ((by << 29) & 0xe0000000u);
            encoded[2] = ((by >> 3) & 0x1u) | ((ry << 1) & 0x3eu) | ((bz << 4) & 0x40u) | ((rz << 7) & 0xf80u) | ((bz << 9) & 0x1000u) | ((d << 13) & 0x3e000u);
        }

        void WriteMode3(uint32_t *encoded, uint16_t m, uint16_t d, uint16_t rw, uint16_t rx, uint16_t ry, uint16_t rz, uint16_t gw, uint16_t gx, uint16_t gy, uint16_t gz, uint16_t bw, uint16_t bx, uint16_t by, uint16_t bz)
        {
            encoded[0] = (m & 0x1fu) | ((rw << 5) & 0x7fe0u) | ((gw << 15) & 0x1ff8000u) | ((bw << 25) & 0xfe000000u);
            encoded[1] = ((bw >> 7) & 0x7u) | ((rx << 3) & 0x78u) | ((rw >> 3) & 0x80u) | ((gz << 4) & 0x100u) | ((gy << 9) & 0x1e00u) | ((gx << 13) & 0x3e000u) | ((gw << 8) & 0x40000u) | ((gz << 19) & 0x780000u) | ((bx << 23) & 0x7800000u) | ((bw << 17) & 0x8000000u) | ((bz << 27) & 0x10000000u) | ((by << 29) & 0xe0000000u);
            encoded[2] = ((by >> 3) & 0x1u) | ((ry << 1) & 0x1eu) | ((bz << 5) & 0x20u) | ((bz << 4) & 0x40u) | ((rz << 7) & 0x780u) | ((gy << 7) & 0x800u) | ((bz << 9) & 0x1000u) | ((d << 13) & 0x3e000u);
        }

        void WriteMode4(uint32_t *encoded, uint16_t m, uint16_t d, uint16_t rw, uint16_t rx, uint16_t ry, uint16_t rz, uint16_t gw, uint16_t gx, uint16_t gy, uint16_t gz, uint16_t bw, uint16_t bx, uint16_t by, uint16_t bz)
        {
            encoded[0] = (m & 0x1fu) | ((rw << 5) & 0x7fe0u) | ((gw << 15) & 0x1ff8000u) | ((bw << 25) & 0xfe000000u);
            encoded[1] = ((bw >> 7) & 0x7u) | ((rx << 3) & 0x78u) | ((rw >> 3) & 0x80u) | ((by << 4) & 0x100u) | ((gy << 9) & 0x1e00u) | ((gx << 13) & 0x1e000u) | ((gw << 7) & 0x20000u) | ((bz << 18) & 0x40000u) | ((gz << 19) & 0x780000u) | ((bx << 23) & 0xf800000u) | ((bw << 18) & 0x10000000u) | ((by << 29) & 0xe0000000u);
            encoded[2] = ((by >> 3) & 0x1u) | ((ry << 1) & 0x1eu) | ((bz << 4) & 0x60u) | ((rz << 7) & 0x780u) | ((bz << 7) & 0x800u) | ((bz << 9) & 0x1000u) | ((d << 13) & 0x3e000u);
        }

        void WriteMode5(uint32_t *encoded, uint16_t m, uint16_t d, uint16_t rw, uint16_t rx, uint16_t ry, uint16_t rz, uint16_t gw, uint16_t gx, uint16_t gy, uint16_t gz, uint16_t bw, uint16_t bx, uint16_t by, uint16_t bz)
        {
            encoded[0] = (m & 0x1fu) | ((rw << 5) & 0x3fe0u) | ((by << 10) & 0x4000u) | ((gw << 15) & 0xff8000u) | ((gy << 20) & 0x1000000u) | ((bw << 25) & 0xfe000000u);
            encoded[1] = ((bw >> 7) & 0x3u) | ((bz >> 2) & 0x4u) | ((rx << 3) & 0xf8u) | ((gz << 4) & 0x100u) | ((gy << 9) & 0x1e00u) | ((gx << 13) & 0x3e000u) | ((bz << 18) & 0x40000u) | ((gz << 19) & 0x780000u) | ((bx << 23) & 0xf800000u) | ((bz << 27) & 0x10000000u) | ((by << 29) & 0xe0000000u);
            encoded[2] = ((by >> 3) & 0x1u) | ((ry << 1) & 0x3eu) | ((bz << 4) & 0x40u) | ((rz << 7) & 0xf80u) | ((bz << 9) & 0x1000u) | ((d << 13) & 0x3e000u);
        }

        void WriteMode6(uint32_t *encoded, uint16_t m, uint16_t d, uint16_t rw, uint16_t rx, uint16_t ry, uint16_t rz, uint16_t gw, uint16_t gx, uint16_t gy, uint16_t gz, uint16_t bw, uint16_t bx, uint16_t by, uint16_t bz)
        {
            encoded[0] = (m & 0x1fu) | ((rw << 5) & 0x1fe0u) | ((gz << 9) & 0x2000u) | ((by << 10) & 0x4000u) | ((gw << 15) & 0x7f8000u) | ((bz << 21) & 0x800000u) | ((gy << 20) & 0x1000000u) | ((bw << 25) & 0xfe000000u);
            encoded[1] = ((bw >> 7) & 0x1u) | ((bz >> 2) & 0x6u) | ((rx << 3) & 0x1f8u) | ((gy << 9) & 0x1e00u) | ((gx << 13) & 0x3e000u) | ((bz << 18) & 0x40000u) | ((gz << 19) & 0x780000u) | ((bx << 23) & 0xf800000u) | ((bz << 27) & 0x10000000u) | ((by << 29) & 0xe0000000u);
            encoded[2] = ((by >> 3) & 0x1u) | ((ry << 1) & 0x7eu) | ((rz << 7) & 0x1f80u) | ((d << 13) & 0x3e000u);
        }

        void WriteMode7(uint32_t *encoded, uint16_t m, uint16_t d, uint16_t rw, uint16_t rx, uint16_t ry, uint16_t rz, uint16_t gw, uint16_t gx, uint16_t gy, uint16_t gz, uint16_t bw, uint16_t bx, uint16_t by, uint16_t bz)
        {
            encoded[0] = (m & 0x1fu) | ((rw << 5) & 0x1fe0u) | ((bz << 13) & 0x2000u) | ((by << 10) & 0x4000u) | ((gw << 15) & 0x7f8000u) | ((gy << 18) & 0x800000u) | ((gy << 20) & 0x1000000u) | ((bw << 25) & 0xfe000000u);
            encoded[1] = ((bw >> 7) & 0x1u) | ((gz >> 4) & 0x2u) | ((bz >> 2) & 0x4u) | ((rx << 3) & 0xf8u) | ((gz << 4) & 0x100u) | ((gy << 9) & 0x1e00u) | ((gx << 13) & 0x7e000u) | ((gz << 19) & 0x780000u) | ((bx << 23) & 0xf800000u) | ((bz << 27) & 0x10000000u) | ((by << 29) & 0xe0000000u);
            encoded[2] = ((by >> 3) & 0x1u) | ((ry << 1) & 0x3eu) | ((bz << 4) & 0x40u) | ((rz << 7) & 0xf80u) | ((bz << 9) & 0x1000u) | ((d << 13) & 0x3e000u);
        }

        void WriteMode8(uint32_t *encoded, uint16_t m, uint16_t d, uint16_t rw, uint16_t rx, uint16_t ry, uint16_t rz, uint16_t gw, uint16_t gx, uint16_t gy, uint16_t gz, uint16_t bw, uint16_t bx, uint16_t by, uint16_t bz)
        {
            encoded[0] = (m & 0x1fu) | ((rw << 5) & 0x1fe0u) | ((bz << 12) & 0x2000u) | ((by << 10) & 0x4000u) | ((gw << 15) & 0x7f8000u) | ((by << 18) & 0x800000u) | ((gy << 20) & 0x1000000u) | ((bw << 25) & 0xfe000000u);
            encoded[1] = ((bw >> 7) & 0x1u) | ((bz >> 4) & 0x2u) | ((bz >> 2) & 0x4u) | ((rx << 3) & 0xf8u) | ((gz << 4) & 0x100u) | ((gy << 9) & 0x1e00u) | ((gx << 13) & 0x3e000u) | ((bz << 18) & 0x40000u) | ((gz << 19) & 0x780000u) | ((bx << 23) & 0x1f800000u) | ((by << 29) & 0xe0000000u);
            encoded[2] = ((by >> 3) & 0x1u) | ((ry << 1) & 0x3eu) | ((bz << 4) & 0x40u) | ((rz << 7) & 0xf80u) | ((bz << 9) & 0x1000u) | ((d << 13) & 0x3e000u);
        }

        void WriteMode9(uint32_t *encoded, uint16_t m, uint16_t d, uint16_t rw, uint16_t rx, uint16_t ry, uint16_t rz, uint16_t gw, uint16_t gx, uint16_t gy, uint16_t gz, uint16_t bw, uint16_t bx, uint16_t by, uint16_t bz)
        {
            encoded[0] = (m & 0x1fu) | ((rw << 5) & 0x7e0u) | ((gz << 7) & 0x800u) | ((bz << 12) & 0x3000u) | ((by << 10) & 0x4000u) | ((gw << 15) & 0x1f8000u) | ((gy << 16) & 0x200000u) | ((by << 17) & 0x400000u) | ((bz << 21) & 0x800000u) | ((gy << 20) & 0x1000000u) | ((bw << 25) & 0x7e000000u) | ((gz << 26) & 0x80000000u);
            encoded[1] = ((bz >> 3) & 0x1u) | ((bz >> 4) & 0x2u) | ((bz >> 2) & 0x4u) | ((rx << 3) & 0x1f8u) | ((gy << 9) & 0x1e00u) | ((gx << 13) & 0x7e000u) | ((gz << 19) & 0x780000u) | ((bx << 23) & 0x1f800000u) | ((by << 29) & 0xe0000000u);
            encoded[2] = ((by >> 3) & 0x1u) | ((ry << 1) & 0x7eu) | ((rz << 7) & 0x1f80u) | ((d << 13) & 0x3e000u);
        }

        void WriteMode10(uint32_t *encoded, uint16_t m, uint16_t d, uint16_t rw, uint16_t rx, uint16_t ry, uint16_t rz, uint16_t gw, uint16_t gx, uint16_t gy, uint16_t gz, uint16_t bw, uint16_t bx, uint16_t by, uint16_t bz)
        {
            encoded[0] = (m & 0x1fu) | ((rw << 5) & 0x7fe0u) | ((gw << 15) & 0x1ff8000u) | ((bw << 25) & 0xfe000000u);
            encoded[1] = ((bw >> 7) & 0x7u) | ((rx << 3) & 0x1ff8u) | ((gx << 13) & 0x7fe000u) | ((bx << 23) & 0xff800000u);
            encoded[2] = ((bx >> 9) & 0x1u);
        }

        void WriteMode11(uint32_t *encoded, uint16_t m, uint16_t d, uint16_t rw, uint16_t rx, uint16_t ry, uint16_t rz, uint16_t gw, uint16_t gx, uint16_t gy, uint16_t gz, uint16_t bw, uint16_t bx, uint16_t by, uint16_t bz)
        {
            encoded[0] = (m & 0x1fu) | ((rw << 5) & 0x7fe0u) | ((gw << 15) & 0x1ff8000u) | ((bw << 25) & 0xfe000000u);
            encoded[1] = ((bw >> 7) & 0x7u) | ((rx << 3) & 0xff8u) | ((rw << 2) & 0x1000u) | ((gx << 13) & 0x3fe000u) | ((gw << 12) & 0x400000u) | ((bx << 23) & 0xff800000u);
            encoded[2] = ((bw >> 10) & 0x1u);
        }

        void WriteMode12(uint32_t *encoded, uint16_t m, uint16_t d, uint16_t rw, uint16_t rx, uint16_t ry, uint16_t rz, uint16_t gw, uint16_t gx, uint16_t gy, uint16_t gz, uint16_t bw, uint16_t bx, uint16_t by, uint16_t bz)
        {
            encoded[0] = (m & 0x1fu) | ((rw << 5) & 0x7fe0u) | ((gw << 15) & 0x1ff8000u) | ((bw << 25) & 0xfe000000u);
            encoded[1] = ((bw >> 7) & 0x7u) | ((rx << 3) & 0x7f8u) | (rw & 0x800u) | ((rw << 2) & 0x1000u) | ((gx << 13) & 0x1fe000u) | ((gw << 10) & 0x200000u) | ((gw << 12) & 0x400000u) | ((bx << 23) & 0x7f800000u) | ((bw << 20) & 0x80000000u);
            encoded[2] = ((bw >> 10) & 0x1u);
        }

        void WriteMode13(uint32_t *encoded, uint16_t m, uint16_t d, uint16_t rw, uint16_t rx, uint16_t ry, uint16_t rz, uint16_t gw, uint16_t gx, uint16_t gy, uint16_t gz, uint16_t bw, uint16_t bx, uint16_t by, uint16_t bz)
        {
            encoded[0] = (m & 0x1fu) | ((rw << 5) & 0x7fe0u) | ((gw << 15) & 0x1ff8000u) | ((bw << 25) & 0xfe000000u);
            encoded[1] = ((bw >> 7) & 0x7u) | ((rx << 3) & 0x78u) | ((rw >> 8) & 0x80u) | ((rw >> 6) & 0x100u) | ((rw >> 4) & 0x200u) | ((rw >> 2) & 0x400u) | (rw & 0x800u) | ((rw << 2) & 0x1000u) | ((gx << 13) & 0x1e000u) | ((gw << 2) & 0x20000u) | ((gw << 4) & 0x40000u) | ((gw << 6) & 0x80000u) | ((gw << 8) & 0x100000u) | ((gw << 10) & 0x200000u) | ((gw << 12) & 0x400000u) | ((bx << 23) & 0x7800000u) | ((bw << 12) & 0x8000000u) | ((bw << 14) & 0x10000000u) | ((bw << 16) & 0x20000000u) | ((bw << 18) & 0x40000000u) | ((bw << 20) & 0x80000000u);
            encoded[2] = ((bw >> 10) & 0x1u);
        }

        void ReadMode0(const uint32_t *encoded, uint16_t &outD, uint16_t &outRW, uint16_t &outRX, uint16_t &outRY, uint16_t &outRZ, uint16_t &outGW, uint16_t &outGX, uint16_t &outGY, uint16_t &outGZ, uint16_t &outBW, uint16_t &outBX, uint16_t &outBY, uint16_t &outBZ)
        {
            uint16_t d = 0;
            uint16_t rw = 0;
            uint16_t rx = 0;
            uint16_t ry = 0;
            uint16_t rz = 0;
            uint16_t gw = 0;
            uint16_t gx = 0;
            uint16_t gy = 0;
            uint16_t gz = 0;
            uint16_t bw = 0;
            uint16_t bx = 0;
            uint16_t by = 0;
            uint16_t bz = 0;
            gy |= ((encoded[0] << 2) & 0x10u);
            by |= ((encoded[0] << 1) & 0x10u);
            bz |= (encoded[0] & 0x10u);
            rw |= ((encoded[0] >> 5) & 0x3ffu);
            gw |= ((encoded[0] >> 15) & 0x3ffu);
            bw |= ((encoded[0] >> 25) & 0x7fu);
            bw |= ((encoded[1] << 7) & 0x380u);
            rx |= ((encoded[1] >> 3) & 0x1fu);
            gz |= ((encoded[1] >> 4) & 0x10u);
            gy |= ((encoded[1] >> 9) & 0xfu);
            gx |= ((encoded[1] >> 13) & 0x1fu);
            bz |= ((encoded[1] >> 18) & 0x1u);
            gz |= ((encoded[1] >> 19) & 0xfu);
            bx |= ((encoded[1] >> 23) & 0x1fu);
            bz |= ((encoded[1] >> 27) & 0x2u);
            by |= ((encoded[1] >> 29) & 0x7u);
            by |= ((encoded[2] << 3) & 0x8u);
            ry |= ((encoded[2] >> 1) & 0x1fu);
            bz |= ((encoded[2] >> 4) & 0x4u);
            rz |= ((encoded[2] >> 7) & 0x1fu);
            bz |= ((encoded[2] >> 9) & 0x8u);
            d |= ((encoded[2] >> 13) & 0x1fu);
            outD = d;
            outRW = rw;
            outRX = rx;
            outRY = ry;
            outRZ = rz;
            outGW = gw;
            outGX = gx;
            outGY = gy;
            outGZ = gz;
            outBW = bw;
            outBX = bx;
            outBY = by;
            outBZ = bz;
        }

        void ReadMode1(const uint32_t *encoded, uint16_t &outD, uint16_t &outRW, uint16_t &outRX, uint16_t &outRY, uint16_t &outRZ, uint16_t &outGW, uint16_t &outGX, uint16_t &outGY, uint16_t &outGZ, uint16_t &outBW, uint16_t &outBX, uint16_t &outBY, uint16_t &outBZ)
        {
            uint16_t d = 0;
            uint16_t rw = 0;
            uint16_t rx = 0;
            uint16_t ry = 0;
            uint16_t rz = 0;
            uint16_t gw = 0;
            uint16_t gx = 0;
            uint16_t gy = 0;
            uint16_t gz = 0;
            uint16_t bw = 0;
            uint16_t bx = 0;
            uint16_t by = 0;
            uint16_t bz = 0;
            gy |= ((encoded[0] << 3) & 0x20u);
            gz |= ((encoded[0] << 1) & 0x30u);
            rw |= ((encoded[0] >> 5) & 0x7fu);
            bz |= ((encoded[0] >> 12) & 0x3u);
            by |= ((encoded[0] >> 10) & 0x10u);
            gw |= ((encoded[0] >> 15) & 0x7fu);
            by |= ((encoded[0] >> 17) & 0x20u);
            bz |= ((encoded[0] >> 21) & 0x4u);
            gy |= ((encoded[0] >> 20) & 0x10u);
            bw |= ((encoded[0] >> 25) & 0x7fu);
            bz |= ((encoded[1] << 3) & 0x8u);
            bz |= ((encoded[1] << 4) & 0x20u);
            bz |= ((encoded[1] << 2) & 0x10u);
            rx |= ((encoded[1] >> 3) & 0x3fu);
            gy |= ((encoded[1] >> 9) & 0xfu);
            gx |= ((encoded[1] >> 13) & 0x3fu);
            gz |= ((encoded[1] >> 19) & 0xfu);
            bx |= ((encoded[1] >> 23) & 0x3fu);
            by |= ((encoded[1] >> 29) & 0x7u);
            by |= ((encoded[2] << 3) & 0x8u);
            ry |= ((encoded[2] >> 1) & 0x3fu);
            rz |= ((encoded[2] >> 7) & 0x3fu);
            d |= ((encoded[2] >> 13) & 0x1fu);
            outD = d;
            outRW = rw;
            outRX = rx;
            outRY = ry;
            outRZ = rz;
            outGW = gw;
            outGX = gx;
            outGY = gy;
            outGZ = gz;
            outBW = bw;
            outBX = bx;
            outBY = by;
            outBZ = bz;
        }

        void ReadMode2(const uint32_t *encoded, uint16_t &outD, uint16_t &outRW, uint16_t &outRX, uint16_t &outRY, uint16_t &outRZ, uint16_t &outGW, uint16_t &outGX, uint16_t &outGY, uint16_t &outGZ, uint16_t &outBW, uint16_t &outBX, uint16_t &outBY, uint16_t &outBZ)
        {
            uint16_t d = 0;
            uint16_t rw = 0;
            uint16_t rx = 0;
            uint16_t ry = 0;
            uint16_t rz = 0;
            uint16_t gw = 0;
            uint16_t gx = 0;
            uint16_t gy = 0;
            uint16_t gz = 0;
            uint16_t bw = 0;
            uint16_t bx = 0;
            uint16_t by = 0;
            uint16_t bz = 0;
            rw |= ((encoded[0] >> 5) & 0x3ffu);
            gw |= ((encoded[0] >> 15) & 0x3ffu);
            bw |= ((encoded[0] >> 25) & 0x7fu);
            bw |= ((encoded[1] << 7) & 0x380u);
            rx |= ((encoded[1] >> 3) & 0x1fu);
            rw |= ((encoded[1] << 2) & 0x400u);
            gy |= ((encoded[1] >> 9) & 0xfu);
            gx |= ((encoded[1] >> 13) & 0xfu);
            gw |= ((encoded[1] >> 7) & 0x400u);
            bz |= ((encoded[1] >> 18) & 0x1u);
            gz |= ((encoded[1] >> 19) & 0xfu);
            bx |= ((encoded[1] >> 23) & 0xfu);
            bw |= ((encoded[1] >> 17) & 0x400u);
            bz |= ((encoded[1] >> 27) & 0x2u);
            by |= ((encoded[1] >> 29) & 0x7u);
            by |= ((encoded[2] << 3) & 0x8u);
            ry |= ((encoded[2] >> 1) & 0x1fu);
            bz |= ((encoded[2] >> 4) & 0x4u);
            rz |= ((encoded[2] >> 7) & 0x1fu);
            bz |= ((encoded[2] >> 9) & 0x8u);
            d |= ((encoded[2] >> 13) & 0x1fu);
            outD = d;
            outRW = rw;
            outRX = rx;
            outRY = ry;
            outRZ = rz;
            outGW = gw;
            outGX = gx;
            outGY = gy;
            outGZ = gz;
            outBW = bw;
            outBX = bx;
            outBY = by;
            outBZ = bz;
        }

        void ReadMode3(const uint32_t *encoded, uint16_t &outD, uint16_t &outRW, uint16_t &outRX, uint16_t &outRY, uint16_t &outRZ, uint16_t &outGW, uint16_t &outGX, uint16_t &outGY, uint16_t &outGZ, uint16_t &outBW, uint16_t &outBX, uint16_t &outBY, uint16_t &outBZ)
        {
            uint16_t d = 0;
            uint16_t rw = 0;
            uint16_t rx = 0;
            uint16_t ry = 0;
            uint16_t rz = 0;
            uint16_t gw = 0;
            uint16_t gx = 0;
            uint16_t gy = 0;
            uint16_t gz = 0;
            uint16_t bw = 0;
            uint16_t bx = 0;
            uint16_t by = 0;
            uint16_t bz = 0;
            rw |= ((encoded[0] >> 5) & 0x3ffu);
            gw |= ((encoded[0] >> 15) & 0x3ffu);
            bw |= ((encoded[0] >> 25) & 0x7fu);
            bw |= ((encoded[1] << 7) & 0x380u);
            rx |= ((encoded[1] >> 3) & 0xfu);
            rw |= ((encoded[1] << 3) & 0x400u);
            gz |= ((encoded[1] >> 4) & 0x10u);
            gy |= ((encoded[1] >> 9) & 0xfu);
            gx |= ((encoded[1] >> 13) & 0x1fu);
            gw |= ((encoded[1] >> 8) & 0x400u);
            gz |= ((encoded[1] >> 19) & 0xfu);
            bx |= ((encoded[1] >> 23) & 0xfu);
            bw |= ((encoded[1] >> 17) & 0x400u);
            bz |= ((encoded[1] >> 27) & 0x2u);
            by |= ((encoded[1] >> 29) & 0x7u);
            by |= ((encoded[2] << 3) & 0x8u);
            ry |= ((encoded[2] >> 1) & 0xfu);
            bz |= ((encoded[2] >> 5) & 0x1u);
            bz |= ((encoded[2] >> 4) & 0x4u);
            rz |= ((encoded[2] >> 7) & 0xfu);
            gy |= ((encoded[2] >> 7) & 0x10u);
            bz |= ((encoded[2] >> 9) & 0x8u);
            d |= ((encoded[2] >> 13) & 0x1fu);
            outD = d;
            outRW = rw;
            outRX = rx;
            outRY = ry;
            outRZ = rz;
            outGW = gw;
            outGX = gx;
            outGY = gy;
            outGZ = gz;
            outBW = bw;
            outBX = bx;
            outBY = by;
            outBZ = bz;
        }

        void ReadMode4(const uint32_t *encoded, uint16_t &outD, uint16_t &outRW, uint16_t &outRX, uint16_t &outRY, uint16_t &outRZ, uint16_t &outGW, uint16_t &outGX, uint16_t &outGY, uint16_t &outGZ, uint16_t &outBW, uint16_t &outBX, uint16_t &outBY, uint16_t &outBZ)
        {
            uint16_t d = 0;
            uint16_t rw = 0;
            uint16_t rx = 0;
            uint16_t ry = 0;
            uint16_t rz = 0;
            uint16_t gw = 0;
            uint16_t gx = 0;
            uint16_t gy = 0;
            uint16_t gz = 0;
            uint16_t bw = 0;
            uint16_t bx = 0;
            uint16_t by = 0;
            uint16_t bz = 0;
            rw |= ((encoded[0] >> 5) & 0x3ffu);
            gw |= ((encoded[0] >> 15) & 0x3ffu);
            bw |= ((encoded[0] >> 25) & 0x7fu);
            bw |= ((encoded[1] << 7) & 0x380u);
            rx |= ((encoded[1] >> 3) & 0xfu);
            rw |= ((encoded[1] << 3) & 0x400u);
            by |= ((encoded[1] >> 4) & 0x10u);
            gy |= ((encoded[1] >> 9) & 0xfu);
            gx |= ((encoded[1] >> 13) & 0xfu);
            gw |= ((encoded[1] >> 7) & 0x400u);
            bz |= ((encoded[1] >> 18) & 0x1u);
            gz |= ((encoded[1] >> 19) & 0xfu);
            bx |= ((encoded[1] >> 23) & 0x1fu);
            bw |= ((encoded[1] >> 18) & 0x400u);
            by |= ((encoded[1] >> 29) & 0x7u);
            by |= ((encoded[2] << 3) & 0x8u);
            ry |= ((encoded[2] >> 1) & 0xfu);
            bz |= ((encoded[2] >> 4) & 0x6u);
            rz |= ((encoded[2] >> 7) & 0xfu);
            bz |= ((encoded[2] >> 7) & 0x10u);
            bz |= ((encoded[2] >> 9) & 0x8u);
            d |= ((encoded[2] >> 13) & 0x1fu);
            outD = d;
            outRW = rw;
            outRX = rx;
            outRY = ry;
            outRZ = rz;
            outGW = gw;
            outGX = gx;
            outGY = gy;
            outGZ = gz;
            outBW = bw;
            outBX = bx;
            outBY = by;
            outBZ = bz;
        }

        void ReadMode5(const uint32_t *encoded, uint16_t &outD, uint16_t &outRW, uint16_t &outRX, uint16_t &outRY, uint16_t &outRZ, uint16_t &outGW, uint16_t &outGX, uint16_t &outGY, uint16_t &outGZ, uint16_t &outBW, uint16_t &outBX, uint16_t &outBY, uint16_t &outBZ)
        {
            uint16_t d = 0;
            uint16_t rw = 0;
            uint16_t rx = 0;
            uint16_t ry = 0;
            uint16_t rz = 0;
            uint16_t gw = 0;
            uint16_t gx = 0;
            uint16_t gy = 0;
            uint16_t gz = 0;
            uint16_t bw = 0;
            uint16_t bx = 0;
            uint16_t by = 0;
            uint16_t bz = 0;
            rw |= ((encoded[0] >> 5) & 0x1ffu);
            by |= ((encoded[0] >> 10) & 0x10u);
            gw |= ((encoded[0] >> 15) & 0x1ffu);
            gy |= ((encoded[0] >> 20) & 0x10u);
            bw |= ((encoded[0] >> 25) & 0x7fu);
            bw |= ((encoded[1] << 7) & 0x180u);
            bz |= ((encoded[1] << 2) & 0x10u);
            rx |= ((encoded[1] >> 3) & 0x1fu);
            gz |= ((encoded[1] >> 4) & 0x10u);
            gy |= ((encoded[1] >> 9) & 0xfu);
            gx |= ((encoded[1] >> 13) & 0x1fu);
            bz |= ((encoded[1] >> 18) & 0x1u);
            gz |= ((encoded[1] >> 19) & 0xfu);
            bx |= ((encoded[1] >> 23) & 0x1fu);
            bz |= ((encoded[1] >> 27) & 0x2u);
            by |= ((encoded[1] >> 29) & 0x7u);
            by |= ((encoded[2] << 3) & 0x8u);
            ry |= ((encoded[2] >> 1) & 0x1fu);
            bz |= ((encoded[2] >> 4) & 0x4u);
            rz |= ((encoded[2] >> 7) & 0x1fu);
            bz |= ((encoded[2] >> 9) & 0x8u);
            d |= ((encoded[2] >> 13) & 0x1fu);
            outD = d;
            outRW = rw;
            outRX = rx;
            outRY = ry;
            outRZ = rz;
            outGW = gw;
            outGX = gx;
            outGY = gy;
            outGZ = gz;
            outBW = bw;
            outBX = bx;
            outBY = by;
            outBZ = bz;
        }

        void ReadMode6(const uint32_t *encoded, uint16_t &outD, uint16_t &outRW, uint16_t &outRX, uint16_t &outRY, uint16_t &outRZ, uint16_t &outGW, uint16_t &outGX, uint16_t &outGY, uint16_t &outGZ, uint16_t &outBW, uint16_t &outBX, uint16_t &outBY, uint16_t &outBZ)
        {
            uint16_t d = 0;
            uint16_t rw = 0;
            uint16_t rx = 0;
            uint16_t ry = 0;
            uint16_t rz = 0;
            uint16_t gw = 0;
            uint16_t gx = 0;
            uint16_t gy = 0;
            uint16_t gz = 0;
            uint16_t bw = 0;
            uint16_t bx = 0;
            uint16_t by = 0;
            uint16_t bz = 0;
            rw |= ((encoded[0] >> 5) & 0xffu);
            gz |= ((encoded[0] >> 9) & 0x10u);
            by |= ((encoded[0] >> 10) & 0x10u);
            gw |= ((encoded[0] >> 15) & 0xffu);
            bz |= ((encoded[0] >> 21) & 0x4u);
            gy |= ((encoded[0] >> 20) & 0x10u);
            bw |= ((encoded[0] >> 25) & 0x7fu);
            bw |= ((encoded[1] << 7) & 0x80u);
            bz |= ((encoded[1] << 2) & 0x18u);
            rx |= ((encoded[1] >> 3) & 0x3fu);
            gy |= ((encoded[1] >> 9) & 0xfu);
            gx |= ((encoded[1] >> 13) & 0x1fu);
            bz |= ((encoded[1] >> 18) & 0x1u);
            gz |= ((encoded[1] >> 19) & 0xfu);
            bx |= ((encoded[1] >> 23) & 0x1fu);
            bz |= ((encoded[1] >> 27) & 0x2u);
            by |= ((encoded[1] >> 29) & 0x7u);
            by |= ((encoded[2] << 3) & 0x8u);
            ry |= ((encoded[2] >> 1) & 0x3fu);
            rz |= ((encoded[2] >> 7) & 0x3fu);
            d |= ((encoded[2] >> 13) & 0x1fu);
            outD = d;
            outRW = rw;
            outRX = rx;
            outRY = ry;
            outRZ = rz;
            outGW = gw;
            outGX = gx;
            outGY = gy;
            outGZ = gz;
            outBW = bw;
            outBX = bx;
            outBY = by;
            outBZ = bz;
        }

        void ReadMode7(const uint32_t *encoded, uint16_t &outD, uint16_t &outRW, uint16_t &outRX, uint16_t &outRY, uint16_t &outRZ, uint16_t &outGW, uint16_t &outGX, uint16_t &outGY, uint16_t &outGZ, uint16_t &outBW, uint16_t &outBX, uint16_t &outBY, uint16_t &outBZ)
        {
            uint16_t d = 0;
            uint16_t rw = 0;
            uint16_t rx = 0;
            uint16_t ry = 0;
            uint16_t rz = 0;
            uint16_t gw = 0;
            uint16_t gx = 0;
            uint16_t gy = 0;
            uint16_t gz = 0;
            uint16_t bw = 0;
            uint16_t bx = 0;
            uint16_t by = 0;
            uint16_t bz = 0;
            rw |= ((encoded[0] >> 5) & 0xffu);
            bz |= ((encoded[0] >> 13) & 0x1u);
            by |= ((encoded[0] >> 10) & 0x10u);
            gw |= ((encoded[0] >> 15) & 0xffu);
            gy |= ((encoded[0] >> 18) & 0x20u);
            gy |= ((encoded[0] >> 20) & 0x10u);
            bw |= ((encoded[0] >> 25) & 0x7fu);
            bw |= ((encoded[1] << 7) & 0x80u);
            gz |= ((encoded[1] << 4) & 0x20u);
            bz |= ((encoded[1] << 2) & 0x10u);
            rx |= ((encoded[1] >> 3) & 0x1fu);
            gz |= ((encoded[1] >> 4) & 0x10u);
            gy |= ((encoded[1] >> 9) & 0xfu);
            gx |= ((encoded[1] >> 13) & 0x3fu);
            gz |= ((encoded[1] >> 19) & 0xfu);
            bx |= ((encoded[1] >> 23) & 0x1fu);
            bz |= ((encoded[1] >> 27) & 0x2u);
            by |= ((encoded[1] >> 29) & 0x7u);
            by |= ((encoded[2] << 3) & 0x8u);
            ry |= ((encoded[2] >> 1) & 0x1fu);
            bz |= ((encoded[2] >> 4) & 0x4u);
            rz |= ((encoded[2] >> 7) & 0x1fu);
            bz |= ((encoded[2] >> 9) & 0x8u);
            d |= ((encoded[2] >> 13) & 0x1fu);
            outD = d;
            outRW = rw;
            outRX = rx;
            outRY = ry;
            outRZ = rz;
            outGW = gw;
            outGX = gx;
            outGY = gy;
            outGZ = gz;
            outBW = bw;
            outBX = bx;
            outBY = by;
            outBZ = bz;
        }

        void ReadMode8(const uint32_t *encoded, uint16_t &outD, uint16_t &outRW, uint16_t &outRX, uint16_t &outRY, uint16_t &outRZ, uint16_t &outGW, uint16_t &outGX, uint16_t &outGY, uint16_t &outGZ, uint16_t &outBW, uint16_t &outBX, uint16_t &outBY, uint16_t &outBZ)
        {
            uint16_t d = 0;
            uint16_t rw = 0;
            uint16_t rx = 0;
            uint16_t ry = 0;
            uint16_t rz = 0;
            uint16_t gw = 0;
            uint16_t gx = 0;
            uint16_t gy = 0;
            uint16_t gz = 0;
            uint16_t bw = 0;
            uint16_t bx = 0;
            uint16_t by = 0;
            uint16_t bz = 0;
            rw |= ((encoded[0] >> 5) & 0xffu);
            bz |= ((encoded[0] >> 12) & 0x2u);
            by |= ((encoded[0] >> 10) & 0x10u);
            gw |= ((encoded[0] >> 15) & 0xffu);
            by |= ((encoded[0] >> 18) & 0x20u);
            gy |= ((encoded[0] >> 20) & 0x10u);
            bw |= ((encoded[0] >> 25) & 0x7fu);
            bw |= ((encoded[1] << 7) & 0x80u);
            bz |= ((encoded[1] << 4) & 0x20u);
            bz |= ((encoded[1] << 2) & 0x10u);
            rx |= ((encoded[1] >> 3) & 0x1fu);
            gz |= ((encoded[1] >> 4) & 0x10u);
            gy |= ((encoded[1] >> 9) & 0xfu);
            gx |= ((encoded[1] >> 13) & 0x1fu);
            bz |= ((encoded[1] >> 18) & 0x1u);
            gz |= ((encoded[1] >> 19) & 0xfu);
            bx |= ((encoded[1] >> 23) & 0x3fu);
            by |= ((encoded[1] >> 29) & 0x7u);
            by |= ((encoded[2] << 3) & 0x8u);
            ry |= ((encoded[2] >> 1) & 0x1fu);
            bz |= ((encoded[2] >> 4) & 0x4u);
            rz |= ((encoded[2] >> 7) & 0x1fu);
            bz |= ((encoded[2] >> 9) & 0x8u);
            d |= ((encoded[2] >> 13) & 0x1fu);
            outD = d;
            outRW = rw;
            outRX = rx;
            outRY = ry;
            outRZ = rz;
            outGW = gw;
            outGX = gx;
            outGY = gy;
            outGZ = gz;
            outBW = bw;
            outBX = bx;
            outBY = by;
            outBZ = bz;
        }

        void ReadMode9(const uint32_t *encoded, uint16_t &outD, uint16_t &outRW, uint16_t &outRX, uint16_t &outRY, uint16_t &outRZ, uint16_t &outGW, uint16_t &outGX, uint16_t &outGY, uint16_t &outGZ, uint16_t &outBW, uint16_t &outBX, uint16_t &outBY, uint16_t &outBZ)
        {
            uint16_t d = 0;
            uint16_t rw = 0;
            uint16_t rx = 0;
            uint16_t ry = 0;
            uint16_t rz = 0;
            uint16_t gw = 0;
            uint16_t gx = 0;
            uint16_t gy = 0;
            uint16_t gz = 0;
            uint16_t bw = 0;
            uint16_t bx = 0;
            uint16_t by = 0;
            uint16_t bz = 0;
            rw |= ((encoded[0] >> 5) & 0x3fu);
            gz |= ((encoded[0] >> 7) & 0x10u);
            bz |= ((encoded[0] >> 12) & 0x3u);
            by |= ((encoded[0] >> 10) & 0x10u);
            gw |= ((encoded[0] >> 15) & 0x3fu);
            gy |= ((encoded[0] >> 16) & 0x20u);
            by |= ((encoded[0] >> 17) & 0x20u);
            bz |= ((encoded[0] >> 21) & 0x4u);
            gy |= ((encoded[0] >> 20) & 0x10u);
            bw |= ((encoded[0] >> 25) & 0x3fu);
            gz |= ((encoded[0] >> 26) & 0x20u);
            bz |= ((encoded[1] << 3) & 0x8u);
            bz |= ((encoded[1] << 4) & 0x20u);
            bz |= ((encoded[1] << 2) & 0x10u);
            rx |= ((encoded[1] >> 3) & 0x3fu);
            gy |= ((encoded[1] >> 9) & 0xfu);
            gx |= ((encoded[1] >> 13) & 0x3fu);
            gz |= ((encoded[1] >> 19) & 0xfu);
            bx |= ((encoded[1] >> 23) & 0x3fu);
            by |= ((encoded[1] >> 29) & 0x7u);
            by |= ((encoded[2] << 3) & 0x8u);
            ry |= ((encoded[2] >> 1) & 0x3fu);
            rz |= ((encoded[2] >> 7) & 0x3fu);
            d |= ((encoded[2] >> 13) & 0x1fu);
            outD = d;
            outRW = rw;
            outRX = rx;
            outRY = ry;
            outRZ = rz;
            outGW = gw;
            outGX = gx;
            outGY = gy;
            outGZ = gz;
            outBW = bw;
            outBX = bx;
            outBY = by;
            outBZ = bz;
        }

        void ReadMode10(const uint32_t *encoded, uint16_t &outD, uint16_t &outRW, uint16_t &outRX, uint16_t &outRY, uint16_t &outRZ, uint16_t &outGW, uint16_t &outGX, uint16_t &outGY, uint16_t &outGZ, uint16_t &outBW, uint16_t &outBX, uint16_t &outBY, uint16_t &outBZ)
        {
            uint16_t d = 0;
            uint16_t rw = 0;
            uint16_t rx = 0;
            uint16_t ry = 0;
            uint16_t rz = 0;
            uint16_t gw = 0;
            uint16_t gx = 0;
            uint16_t gy = 0;
            uint16_t gz = 0;
            uint16_t bw = 0;
            uint16_t bx = 0;
            uint16_t by = 0;
            uint16_t bz = 0;
            rw |= ((encoded[0] >> 5) & 0x3ffu);
            gw |= ((encoded[0] >> 15) & 0x3ffu);
            bw |= ((encoded[0] >> 25) & 0x7fu);
            bw |= ((encoded[1] << 7) & 0x380u);
            rx |= ((encoded[1] >> 3) & 0x3ffu);
            gx |= ((encoded[1] >> 13) & 0x3ffu);
            bx |= ((encoded[1] >> 23) & 0x1ffu);
            bx |= ((encoded[2] << 9) & 0x200u);
            outD = d;
            outRW = rw;
            outRX = rx;
            outRY = ry;
            outRZ = rz;
            outGW = gw;
            outGX = gx;
            outGY = gy;
            outGZ = gz;
            outBW = bw;
            outBX = bx;
            outBY = by;
            outBZ = bz;
        }

        void ReadMode11(const uint32_t *encoded, uint16_t &outD, uint16_t &outRW, uint16_t &outRX, uint16_t &outRY, uint16_t &outRZ, uint16_t &outGW, uint16_t &outGX, uint16_t &outGY, uint16_t &outGZ, uint16_t &outBW, uint16_t &outBX, uint16_t &outBY, uint16_t &outBZ)
        {
            uint16_t d = 0;
            uint16_t rw = 0;
            uint16_t rx = 0;
            uint16_t ry = 0;
            uint16_t rz = 0;
            uint16_t gw = 0;
            uint16_t gx = 0;
            uint16_t gy = 0;
            uint16_t gz = 0;
            uint16_t bw = 0;
            uint16_t bx = 0;
            uint16_t by = 0;
            uint16_t bz = 0;
            rw |= ((encoded[0] >> 5) & 0x3ffu);
            gw |= ((encoded[0] >> 15) & 0x3ffu);
            bw |= ((encoded[0] >> 25) & 0x7fu);
            bw |= ((encoded[1] << 7) & 0x380u);
            rx |= ((encoded[1] >> 3) & 0x1ffu);
            rw |= ((encoded[1] >> 2) & 0x400u);
            gx |= ((encoded[1] >> 13) & 0x1ffu);
            gw |= ((encoded[1] >> 12) & 0x400u);
            bx |= ((encoded[1] >> 23) & 0x1ffu);
            bw |= ((encoded[2] << 10) & 0x400u);
            outD = d;
            outRW = rw;
            outRX = rx;
            outRY = ry;
            outRZ = rz;
            outGW = gw;
            outGX = gx;
            outGY = gy;
            outGZ = gz;
            outBW = bw;
            outBX = bx;
            outBY = by;
            outBZ = bz;
        }

        void ReadMode12(const uint32_t *encoded, uint16_t &outD, uint16_t &outRW, uint16_t &outRX, uint16_t &outRY, uint16_t &outRZ, uint16_t &outGW, uint16_t &outGX, uint16_t &outGY, uint16_t &outGZ, uint16_t &outBW, uint16_t &outBX, uint16_t &outBY, uint16_t &outBZ)
        {
            uint16_t d = 0;
            uint16_t rw = 0;
            uint16_t rx = 0;
            uint16_t ry = 0;
            uint16_t rz = 0;
            uint16_t gw = 0;
            uint16_t gx = 0;
            uint16_t gy = 0;
            uint16_t gz = 0;
            uint16_t bw = 0;
            uint16_t bx = 0;
            uint16_t by = 0;
            uint16_t bz = 0;
            rw |= ((encoded[0] >> 5) & 0x3ffu);
            gw |= ((encoded[0] >> 15) & 0x3ffu);
            bw |= ((encoded[0] >> 25) & 0x7fu);
            bw |= ((encoded[1] << 7) & 0x380u);
            rx |= ((encoded[1] >> 3) & 0xffu);
            rw |= (encoded[1] & 0x800u);
            rw |= ((encoded[1] >> 2) & 0x400u);
            gx |= ((encoded[1] >> 13) & 0xffu);
            gw |= ((encoded[1] >> 10) & 0x800u);
            gw |= ((encoded[1] >> 12) & 0x400u);
            bx |= ((encoded[1] >> 23) & 0xffu);
            bw |= ((encoded[1] >> 20) & 0x800u);
            bw |= ((encoded[2] << 10) & 0x400u);
            outD = d;
            outRW = rw;
            outRX = rx;
            outRY = ry;
            outRZ = rz;
            outGW = gw;
            outGX = gx;
            outGY = gy;
            outGZ = gz;
            outBW = bw;
            outBX = bx;
            outBY = by;
            outBZ = bz;
        }

        void ReadMode13(const uint32_t *encoded, uint16_t &outD, uint16_t &outRW, uint16_t &outRX, uint16_t &outRY, uint16_t &outRZ, uint16_t &outGW, uint16_t &outGX, uint16_t &outGY, uint16_t &outGZ, uint16_t &outBW, uint16_t &outBX, uint16_t &outBY, uint16_t &outBZ)
        {
            uint16_t d = 0;
            uint16_t rw = 0;
            uint16_t rx = 0;
            uint16_t ry = 0;
            uint16_t rz = 0;
            uint16_t gw = 0;
            uint16_t gx = 0;
            uint16_t gy = 0;
            uint16_t gz = 0;
            uint16_t bw = 0;
            uint16_t bx = 0;
            uint16_t by = 0;
            uint16_t bz = 0;
            rw |= ((encoded[0] >> 5) & 0x3ffu);
            gw |= ((encoded[0] >> 15) & 0x3ffu);
            bw |= ((encoded[0] >> 25) & 0x7fu);
            bw |= ((encoded[1] << 7) & 0x380u);
            rx |= ((encoded[1] >> 3) & 0xfu);
            rw |= ((encoded[1] << 8) & 0x8000u);
            rw |= ((encoded[1] << 6) & 0x4000u);
            rw |= ((encoded[1] << 4) & 0x2000u);
            rw |= ((encoded[1] << 2) & 0x1000u);
            rw |= (encoded[1] & 0x800u);
            rw |= ((encoded[1] >> 2) & 0x400u);
            gx |= ((encoded[1] >> 13) & 0xfu);
            gw |= ((encoded[1] >> 2) & 0x8000u);
            gw |= ((encoded[1] >> 4) & 0x4000u);
            gw |= ((encoded[1] >> 6) & 0x2000u);
            gw |= ((encoded[1] >> 8) & 0x1000u);
            gw |= ((encoded[1] >> 10) & 0x800u);
            gw |= ((encoded[1] >> 12) & 0x400u);
            bx |= ((encoded[1] >> 23) & 0xfu);
            bw |= ((encoded[1] >> 12) & 0x8000u);
            bw |= ((encoded[1] >> 14) & 0x4000u);
            bw |= ((encoded[1] >> 16) & 0x2000u);
            bw |= ((encoded[1] >> 18) & 0x1000u);
            bw |= ((encoded[1] >> 20) & 0x800u);
            bw |= ((encoded[2] << 10) & 0x400u);
            outD = d;
            outRW = rw;
            outRX = rx;
            outRY = ry;
            outRZ = rz;
            outGW = gw;
            outGX = gx;
            outGY = gy;
            outGZ = gz;
            outBW = bw;
            outBX = bx;
            outBY = by;
            outBZ = bz;
        }

        const ReadFunc_t g_readFuncs[14] =
        {
            ReadMode0,
            ReadMode1,
            ReadMode2,
            ReadMode3,
            ReadMode4,
            ReadMode5,
            ReadMode6,
            ReadMode7,
            ReadMode8,
            ReadMode9,
            ReadMode10,
            ReadMode11,
            ReadMode12,
            ReadMode13
        };

        const WriteFunc_t g_writeFuncs[14] =
        {
            WriteMode0,
            WriteMode1,
            WriteMode2,
            WriteMode3,
            WriteMode4,
            WriteMode5,
            WriteMode6,
            WriteMode7,
            WriteMode8,
            WriteMode9,
            WriteMode10,
            WriteMode11,
            WriteMode12,
            WriteMode13
        };
    }
}

#endif
