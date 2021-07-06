//
// Copyright 2013 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// loadimage_etc.cpp: Decodes ETC and EAC encoded textures.

#include "image_util/loadimage.h"

#include <type_traits>
#include "common/mathutil.h"

#include "image_util/imageformats.h"

namespace angle
{
namespace
{

using IntensityModifier = const int[4];

// Table 3.17.2 sorted according to table 3.17.3
// clang-format off
static IntensityModifier intensityModifierDefault[] =
{
    {  2,   8,  -2,   -8 },
    {  5,  17,  -5,  -17 },
    {  9,  29,  -9,  -29 },
    { 13,  42, -13,  -42 },
    { 18,  60, -18,  -60 },
    { 24,  80, -24,  -80 },
    { 33, 106, -33, -106 },
    { 47, 183, -47, -183 },
};
// clang-format on

// Table C.12, intensity modifier for non opaque punchthrough alpha
// clang-format off
static IntensityModifier intensityModifierNonOpaque[] =
{
    { 0,   8, 0,   -8 },
    { 0,  17, 0,  -17 },
    { 0,  29, 0,  -29 },
    { 0,  42, 0,  -42 },
    { 0,  60, 0,  -60 },
    { 0,  80, 0,  -80 },
    { 0, 106, 0, -106 },
    { 0, 183, 0, -183 },
};
// clang-format on

static const int kNumPixelsInBlock = 16;

struct ETC2Block
{
    // Decodes unsigned single or dual channel ETC2 block to 8-bit color
    void decodeAsSingleETC2Channel(uint8_t *dest,
                                   size_t x,
                                   size_t y,
                                   size_t w,
                                   size_t h,
                                   size_t destPixelStride,
                                   size_t destRowPitch,
                                   bool isSigned) const
    {
        for (size_t j = 0; j < 4 && (y + j) < h; j++)
        {
            uint8_t *row = dest + (j * destRowPitch);
            for (size_t i = 0; i < 4 && (x + i) < w; i++)
            {
                uint8_t *pixel = row + (i * destPixelStride);
                if (isSigned)
                {
                    *pixel = clampSByte(getSingleETC2Channel(i, j, isSigned));
                }
                else
                {
                    *pixel = clampByte(getSingleETC2Channel(i, j, isSigned));
                }
            }
        }
    }

    // Decodes unsigned single or dual channel EAC block to 16-bit color
    void decodeAsSingleEACChannel(uint16_t *dest,
                                  size_t x,
                                  size_t y,
                                  size_t w,
                                  size_t h,
                                  size_t destPixelStride,
                                  size_t destRowPitch,
                                  bool isSigned,
                                  bool isFloat) const
    {
        for (size_t j = 0; j < 4 && (y + j) < h; j++)
        {
            uint16_t *row = reinterpret_cast<uint16_t *>(reinterpret_cast<uint8_t *>(dest) +
                                                         (j * destRowPitch));
            for (size_t i = 0; i < 4 && (x + i) < w; i++)
            {
                uint16_t *pixel = row + (i * destPixelStride);
                if (isSigned)
                {
                    int16_t tempPixel =
                        renormalizeEAC<int16_t>(getSingleEACChannel(i, j, isSigned));
                    *pixel =
                        isFloat ? gl::float32ToFloat16(float(gl::normalize(tempPixel))) : tempPixel;
                }
                else
                {
                    uint16_t tempPixel =
                        renormalizeEAC<uint16_t>(getSingleEACChannel(i, j, isSigned));
                    *pixel =
                        isFloat ? gl::float32ToFloat16(float(gl::normalize(tempPixel))) : tempPixel;
                }
            }
        }
    }

    // Decodes RGB block to rgba8
    void decodeAsRGB(uint8_t *dest,
                     size_t x,
                     size_t y,
                     size_t w,
                     size_t h,
                     size_t destRowPitch,
                     const uint8_t alphaValues[4][4],
                     bool punchThroughAlpha) const
    {
        bool opaqueBit                  = u.idht.mode.idm.diffbit;
        bool nonOpaquePunchThroughAlpha = punchThroughAlpha && !opaqueBit;
        // Select mode
        if (u.idht.mode.idm.diffbit || punchThroughAlpha)
        {
            const auto &block = u.idht.mode.idm.colors.diff;
            int r             = (block.R + block.dR);
            int g             = (block.G + block.dG);
            int b             = (block.B + block.dB);
            if (r < 0 || r > 31)
            {
                decodeTBlock(dest, x, y, w, h, destRowPitch, alphaValues,
                             nonOpaquePunchThroughAlpha);
            }
            else if (g < 0 || g > 31)
            {
                decodeHBlock(dest, x, y, w, h, destRowPitch, alphaValues,
                             nonOpaquePunchThroughAlpha);
            }
            else if (b < 0 || b > 31)
            {
                decodePlanarBlock(dest, x, y, w, h, destRowPitch, alphaValues);
            }
            else
            {
                decodeDifferentialBlock(dest, x, y, w, h, destRowPitch, alphaValues,
                                        nonOpaquePunchThroughAlpha);
            }
        }
        else
        {
            decodeIndividualBlock(dest, x, y, w, h, destRowPitch, alphaValues,
                                  nonOpaquePunchThroughAlpha);
        }
    }

    // Transcodes RGB block to BC1
    void transcodeAsBC1(uint8_t *dest,
                        size_t x,
                        size_t y,
                        size_t w,
                        size_t h,
                        const uint8_t alphaValues[4][4],
                        bool punchThroughAlpha) const
    {
        bool opaqueBit                  = u.idht.mode.idm.diffbit;
        bool nonOpaquePunchThroughAlpha = punchThroughAlpha && !opaqueBit;
        // Select mode
        if (u.idht.mode.idm.diffbit || punchThroughAlpha)
        {
            const auto &block = u.idht.mode.idm.colors.diff;
            int r             = (block.R + block.dR);
            int g             = (block.G + block.dG);
            int b             = (block.B + block.dB);
            if (r < 0 || r > 31)
            {
                transcodeTBlockToBC1(dest, x, y, w, h, alphaValues, nonOpaquePunchThroughAlpha);
            }
            else if (g < 0 || g > 31)
            {
                transcodeHBlockToBC1(dest, x, y, w, h, alphaValues, nonOpaquePunchThroughAlpha);
            }
            else if (b < 0 || b > 31)
            {
                transcodePlanarBlockToBC1(dest, x, y, w, h, alphaValues);
            }
            else
            {
                transcodeDifferentialBlockToBC1(dest, x, y, w, h, alphaValues,
                                                nonOpaquePunchThroughAlpha);
            }
        }
        else
        {
            transcodeIndividualBlockToBC1(dest, x, y, w, h, alphaValues,
                                          nonOpaquePunchThroughAlpha);
        }
    }

  private:
    union
    {
        // Individual, differential, H and T modes
        struct
        {
            union
            {
                // Individual and differential modes
                struct
                {
                    union
                    {
                        struct  // Individual colors
                        {
                            unsigned char R2 : 4;
                            unsigned char R1 : 4;
                            unsigned char G2 : 4;
                            unsigned char G1 : 4;
                            unsigned char B2 : 4;
                            unsigned char B1 : 4;
                        } indiv;
                        struct  // Differential colors
                        {
                            signed char dR : 3;
                            unsigned char R : 5;
                            signed char dG : 3;
                            unsigned char G : 5;
                            signed char dB : 3;
                            unsigned char B : 5;
                        } diff;
                    } colors;
                    bool flipbit : 1;
                    bool diffbit : 1;
                    unsigned char cw2 : 3;
                    unsigned char cw1 : 3;
                } idm;
                // T mode
                struct
                {
                    // Byte 1
                    unsigned char TR1b : 2;
                    unsigned char TdummyB : 1;
                    unsigned char TR1a : 2;
                    unsigned char TdummyA : 3;
                    // Byte 2
                    unsigned char TB1 : 4;
                    unsigned char TG1 : 4;
                    // Byte 3
                    unsigned char TG2 : 4;
                    unsigned char TR2 : 4;
                    // Byte 4
                    unsigned char Tdb : 1;
                    bool Tflipbit : 1;
                    unsigned char Tda : 2;
                    unsigned char TB2 : 4;
                } tm;
                // H mode
                struct
                {
                    // Byte 1
                    unsigned char HG1a : 3;
                    unsigned char HR1 : 4;
                    unsigned char HdummyA : 1;
                    // Byte 2
                    unsigned char HB1b : 2;
                    unsigned char HdummyC : 1;
                    unsigned char HB1a : 1;
                    unsigned char HG1b : 1;
                    unsigned char HdummyB : 3;
                    // Byte 3
                    unsigned char HG2a : 3;
                    unsigned char HR2 : 4;
                    unsigned char HB1c : 1;
                    // Byte 4
                    unsigned char Hdb : 1;
                    bool Hflipbit : 1;
                    unsigned char Hda : 1;
                    unsigned char HB2 : 4;
                    unsigned char HG2b : 1;
                } hm;
            } mode;
            unsigned char pixelIndexMSB[2];
            unsigned char pixelIndexLSB[2];
        } idht;
        // planar mode
        struct
        {
            // Byte 1
            unsigned char GO1 : 1;
            unsigned char RO : 6;
            unsigned char PdummyA : 1;
            // Byte 2
            unsigned char BO1 : 1;
            unsigned char GO2 : 6;
            unsigned char PdummyB : 1;
            // Byte 3
            unsigned char BO3a : 2;
            unsigned char PdummyD : 1;
            unsigned char BO2 : 2;
            unsigned char PdummyC : 3;
            // Byte 4
            unsigned char RH2 : 1;
            bool Pflipbit : 1;
            unsigned char RH1 : 5;
            unsigned char BO3b : 1;
            // Byte 5
            unsigned char BHa : 1;
            unsigned char GH : 7;
            // Byte 6
            unsigned char RVa : 3;
            unsigned char BHb : 5;
            // Byte 7
            unsigned char GVa : 5;
            unsigned char RVb : 3;
            // Byte 8
            unsigned char BV : 6;
            unsigned char GVb : 2;
        } pblk;
        // Single channel block
        struct
        {
            union
            {
                unsigned char us;
                signed char s;
            } base_codeword;
            unsigned char table_index : 4;
            unsigned char multiplier : 4;
            unsigned char mc1 : 2;
            unsigned char mb : 3;
            unsigned char ma : 3;
            unsigned char mf1 : 1;
            unsigned char me : 3;
            unsigned char md : 3;
            unsigned char mc2 : 1;
            unsigned char mh : 3;
            unsigned char mg : 3;
            unsigned char mf2 : 2;
            unsigned char mk1 : 2;
            unsigned char mj : 3;
            unsigned char mi : 3;
            unsigned char mn1 : 1;
            unsigned char mm : 3;
            unsigned char ml : 3;
            unsigned char mk2 : 1;
            unsigned char mp : 3;
            unsigned char mo : 3;
            unsigned char mn2 : 2;
        } scblk;
    } u;

    static unsigned char clampByte(int value)
    {
        return static_cast<unsigned char>(gl::clamp(value, 0, 255));
    }

    static signed char clampSByte(int value)
    {
        return static_cast<signed char>(gl::clamp(value, -128, 127));
    }

    template <typename T>
    static T renormalizeEAC(int value)
    {
        int upper = 0;
        int lower = 0;
        int shift = 0;

        if (std::is_same<T, int16_t>::value)
        {
            // The spec states that -1024 invalid and should be clamped to -1023
            upper = 1023;
            lower = -1023;
            shift = 5;
        }
        else if (std::is_same<T, uint16_t>::value)
        {
            upper = 2047;
            lower = 0;
            shift = 5;
        }
        else
        {
            // We currently only support renormalizing int16_t or uint16_t
            UNREACHABLE();
        }

        return static_cast<T>(gl::clamp(value, lower, upper)) << shift;
    }

    static R8G8B8A8 createRGBA(int red, int green, int blue, int alpha)
    {
        R8G8B8A8 rgba;
        rgba.R = clampByte(red);
        rgba.G = clampByte(green);
        rgba.B = clampByte(blue);
        rgba.A = clampByte(alpha);
        return rgba;
    }

    static R8G8B8A8 createRGBA(int red, int green, int blue)
    {
        return createRGBA(red, green, blue, 255);
    }

    static int extend_4to8bits(int x) { return (x << 4) | x; }
    static int extend_5to8bits(int x) { return (x << 3) | (x >> 2); }
    static int extend_6to8bits(int x) { return (x << 2) | (x >> 4); }
    static int extend_7to8bits(int x) { return (x << 1) | (x >> 6); }

    void decodeIndividualBlock(uint8_t *dest,
                               size_t x,
                               size_t y,
                               size_t w,
                               size_t h,
                               size_t destRowPitch,
                               const uint8_t alphaValues[4][4],
                               bool nonOpaquePunchThroughAlpha) const
    {
        const auto &block = u.idht.mode.idm.colors.indiv;
        int r1            = extend_4to8bits(block.R1);
        int g1            = extend_4to8bits(block.G1);
        int b1            = extend_4to8bits(block.B1);
        int r2            = extend_4to8bits(block.R2);
        int g2            = extend_4to8bits(block.G2);
        int b2            = extend_4to8bits(block.B2);
        decodeIndividualOrDifferentialBlock(dest, x, y, w, h, destRowPitch, r1, g1, b1, r2, g2, b2,
                                            alphaValues, nonOpaquePunchThroughAlpha);
    }

    void decodeDifferentialBlock(uint8_t *dest,
                                 size_t x,
                                 size_t y,
                                 size_t w,
                                 size_t h,
                                 size_t destRowPitch,
                                 const uint8_t alphaValues[4][4],
                                 bool nonOpaquePunchThroughAlpha) const
    {
        const auto &block = u.idht.mode.idm.colors.diff;
        int b1            = extend_5to8bits(block.B);
        int g1            = extend_5to8bits(block.G);
        int r1            = extend_5to8bits(block.R);
        int r2            = extend_5to8bits(block.R + block.dR);
        int g2            = extend_5to8bits(block.G + block.dG);
        int b2            = extend_5to8bits(block.B + block.dB);
        decodeIndividualOrDifferentialBlock(dest, x, y, w, h, destRowPitch, r1, g1, b1, r2, g2, b2,
                                            alphaValues, nonOpaquePunchThroughAlpha);
    }

    void decodeIndividualOrDifferentialBlock(uint8_t *dest,
                                             size_t x,
                                             size_t y,
                                             size_t w,
                                             size_t h,
                                             size_t destRowPitch,
                                             int r1,
                                             int g1,
                                             int b1,
                                             int r2,
                                             int g2,
                                             int b2,
                                             const uint8_t alphaValues[4][4],
                                             bool nonOpaquePunchThroughAlpha) const
    {
        const IntensityModifier *intensityModifier =
            nonOpaquePunchThroughAlpha ? intensityModifierNonOpaque : intensityModifierDefault;

        R8G8B8A8 subblockColors0[4];
        R8G8B8A8 subblockColors1[4];
        for (size_t modifierIdx = 0; modifierIdx < 4; modifierIdx++)
        {
            const int i1                 = intensityModifier[u.idht.mode.idm.cw1][modifierIdx];
            subblockColors0[modifierIdx] = createRGBA(r1 + i1, g1 + i1, b1 + i1);

            const int i2                 = intensityModifier[u.idht.mode.idm.cw2][modifierIdx];
            subblockColors1[modifierIdx] = createRGBA(r2 + i2, g2 + i2, b2 + i2);
        }

        if (u.idht.mode.idm.flipbit)
        {
            uint8_t *curPixel = dest;
            for (size_t j = 0; j < 2 && (y + j) < h; j++)
            {
                R8G8B8A8 *row = reinterpret_cast<R8G8B8A8 *>(curPixel);
                for (size_t i = 0; i < 4 && (x + i) < w; i++)
                {
                    row[i]   = subblockColors0[getIndex(i, j)];
                    row[i].A = alphaValues[j][i];
                }
                curPixel += destRowPitch;
            }
            for (size_t j = 2; j < 4 && (y + j) < h; j++)
            {
                R8G8B8A8 *row = reinterpret_cast<R8G8B8A8 *>(curPixel);
                for (size_t i = 0; i < 4 && (x + i) < w; i++)
                {
                    row[i]   = subblockColors1[getIndex(i, j)];
                    row[i].A = alphaValues[j][i];
                }
                curPixel += destRowPitch;
            }
        }
        else
        {
            uint8_t *curPixel = dest;
            for (size_t j = 0; j < 4 && (y + j) < h; j++)
            {
                R8G8B8A8 *row = reinterpret_cast<R8G8B8A8 *>(curPixel);
                for (size_t i = 0; i < 2 && (x + i) < w; i++)
                {
                    row[i]   = subblockColors0[getIndex(i, j)];
                    row[i].A = alphaValues[j][i];
                }
                for (size_t i = 2; i < 4 && (x + i) < w; i++)
                {
                    row[i]   = subblockColors1[getIndex(i, j)];
                    row[i].A = alphaValues[j][i];
                }
                curPixel += destRowPitch;
            }
        }
        if (nonOpaquePunchThroughAlpha)
        {
            decodePunchThroughAlphaBlock(dest, x, y, w, h, destRowPitch);
        }
    }

    void decodeTBlock(uint8_t *dest,
                      size_t x,
                      size_t y,
                      size_t w,
                      size_t h,
                      size_t destRowPitch,
                      const uint8_t alphaValues[4][4],
                      bool nonOpaquePunchThroughAlpha) const
    {
        // Table C.8, distance index for T and H modes
        const auto &block = u.idht.mode.tm;

        int r1 = extend_4to8bits(block.TR1a << 2 | block.TR1b);
        int g1 = extend_4to8bits(block.TG1);
        int b1 = extend_4to8bits(block.TB1);
        int r2 = extend_4to8bits(block.TR2);
        int g2 = extend_4to8bits(block.TG2);
        int b2 = extend_4to8bits(block.TB2);

        static int distance[8] = {3, 6, 11, 16, 23, 32, 41, 64};
        const int d            = distance[block.Tda << 1 | block.Tdb];

        const R8G8B8A8 paintColors[4] = {
            createRGBA(r1, g1, b1),
            createRGBA(r2 + d, g2 + d, b2 + d),
            createRGBA(r2, g2, b2),
            createRGBA(r2 - d, g2 - d, b2 - d),
        };

        uint8_t *curPixel = dest;
        for (size_t j = 0; j < 4 && (y + j) < h; j++)
        {
            R8G8B8A8 *row = reinterpret_cast<R8G8B8A8 *>(curPixel);
            for (size_t i = 0; i < 4 && (x + i) < w; i++)
            {
                row[i]   = paintColors[getIndex(i, j)];
                row[i].A = alphaValues[j][i];
            }
            curPixel += destRowPitch;
        }

        if (nonOpaquePunchThroughAlpha)
        {
            decodePunchThroughAlphaBlock(dest, x, y, w, h, destRowPitch);
        }
    }

    void decodeHBlock(uint8_t *dest,
                      size_t x,
                      size_t y,
                      size_t w,
                      size_t h,
                      size_t destRowPitch,
                      const uint8_t alphaValues[4][4],
                      bool nonOpaquePunchThroughAlpha) const
    {
        // Table C.8, distance index for T and H modes
        const auto &block = u.idht.mode.hm;

        int r1 = extend_4to8bits(block.HR1);
        int g1 = extend_4to8bits(block.HG1a << 1 | block.HG1b);
        int b1 = extend_4to8bits(block.HB1a << 3 | block.HB1b << 1 | block.HB1c);
        int r2 = extend_4to8bits(block.HR2);
        int g2 = extend_4to8bits(block.HG2a << 1 | block.HG2b);
        int b2 = extend_4to8bits(block.HB2);

        static const int distance[8] = {3, 6, 11, 16, 23, 32, 41, 64};
        const int orderingTrickBit =
            ((r1 << 16 | g1 << 8 | b1) >= (r2 << 16 | g2 << 8 | b2) ? 1 : 0);
        const int d = distance[(block.Hda << 2) | (block.Hdb << 1) | orderingTrickBit];

        const R8G8B8A8 paintColors[4] = {
            createRGBA(r1 + d, g1 + d, b1 + d),
            createRGBA(r1 - d, g1 - d, b1 - d),
            createRGBA(r2 + d, g2 + d, b2 + d),
            createRGBA(r2 - d, g2 - d, b2 - d),
        };

        uint8_t *curPixel = dest;
        for (size_t j = 0; j < 4 && (y + j) < h; j++)
        {
            R8G8B8A8 *row = reinterpret_cast<R8G8B8A8 *>(curPixel);
            for (size_t i = 0; i < 4 && (x + i) < w; i++)
            {
                row[i]   = paintColors[getIndex(i, j)];
                row[i].A = alphaValues[j][i];
            }
            curPixel += destRowPitch;
        }

        if (nonOpaquePunchThroughAlpha)
        {
            decodePunchThroughAlphaBlock(dest, x, y, w, h, destRowPitch);
        }
    }

    void decodePlanarBlock(uint8_t *dest,
                           size_t x,
                           size_t y,
                           size_t w,
                           size_t h,
                           size_t pitch,
                           const uint8_t alphaValues[4][4]) const
    {
        int ro = extend_6to8bits(u.pblk.RO);
        int go = extend_7to8bits(u.pblk.GO1 << 6 | u.pblk.GO2);
        int bo =
            extend_6to8bits(u.pblk.BO1 << 5 | u.pblk.BO2 << 3 | u.pblk.BO3a << 1 | u.pblk.BO3b);
        int rh = extend_6to8bits(u.pblk.RH1 << 1 | u.pblk.RH2);
        int gh = extend_7to8bits(u.pblk.GH);
        int bh = extend_6to8bits(u.pblk.BHa << 5 | u.pblk.BHb);
        int rv = extend_6to8bits(u.pblk.RVa << 3 | u.pblk.RVb);
        int gv = extend_7to8bits(u.pblk.GVa << 2 | u.pblk.GVb);
        int bv = extend_6to8bits(u.pblk.BV);

        uint8_t *curPixel = dest;
        for (size_t j = 0; j < 4 && (y + j) < h; j++)
        {
            R8G8B8A8 *row = reinterpret_cast<R8G8B8A8 *>(curPixel);

            int ry = static_cast<int>(j) * (rv - ro) + 2;
            int gy = static_cast<int>(j) * (gv - go) + 2;
            int by = static_cast<int>(j) * (bv - bo) + 2;
            for (size_t i = 0; i < 4 && (x + i) < w; i++)
            {
                row[i] = createRGBA(((static_cast<int>(i) * (rh - ro) + ry) >> 2) + ro,
                                    ((static_cast<int>(i) * (gh - go) + gy) >> 2) + go,
                                    ((static_cast<int>(i) * (bh - bo) + by) >> 2) + bo,
                                    alphaValues[j][i]);
            }
            curPixel += pitch;
        }
    }

    // Index for individual, differential, H and T modes
    size_t getIndex(size_t x, size_t y) const
    {
        size_t bitIndex  = x * 4 + y;
        size_t bitOffset = bitIndex & 7;
        size_t lsb       = (u.idht.pixelIndexLSB[1 - (bitIndex >> 3)] >> bitOffset) & 1;
        size_t msb       = (u.idht.pixelIndexMSB[1 - (bitIndex >> 3)] >> bitOffset) & 1;
        return (msb << 1) | lsb;
    }

    void decodePunchThroughAlphaBlock(uint8_t *dest,
                                      size_t x,
                                      size_t y,
                                      size_t w,
                                      size_t h,
                                      size_t destRowPitch) const
    {
        uint8_t *curPixel = dest;
        for (size_t j = 0; j < 4 && (y + j) < h; j++)
        {
            R8G8B8A8 *row = reinterpret_cast<R8G8B8A8 *>(curPixel);
            for (size_t i = 0; i < 4 && (x + i) < w; i++)
            {
                if (getIndex(i, j) == 2)  //  msb == 1 && lsb == 0
                {
                    row[i] = createRGBA(0, 0, 0, 0);
                }
            }
            curPixel += destRowPitch;
        }
    }

    uint16_t RGB8ToRGB565(const R8G8B8A8 &rgba) const
    {
        return (static_cast<uint16_t>(rgba.R >> 3) << 11) |
               (static_cast<uint16_t>(rgba.G >> 2) << 5) |
               (static_cast<uint16_t>(rgba.B >> 3) << 0);
    }

    uint32_t matchBC1Bits(const int *pixelIndices,
                          const int *pixelIndexCounts,
                          const R8G8B8A8 *subblockColors,
                          size_t numColors,
                          const R8G8B8A8 &minColor,
                          const R8G8B8A8 &maxColor,
                          bool nonOpaquePunchThroughAlpha) const
    {
        // Project each pixel on the (maxColor, minColor) line to decide which
        // BC1 code to assign to it.

        uint8_t decodedColors[2][3] = {{maxColor.R, maxColor.G, maxColor.B},
                                       {minColor.R, minColor.G, minColor.B}};

        int direction[3];
        for (int ch = 0; ch < 3; ch++)
        {
            direction[ch] = decodedColors[0][ch] - decodedColors[1][ch];
        }

        int stops[2];
        for (int i = 0; i < 2; i++)
        {
            stops[i] = decodedColors[i][0] * direction[0] + decodedColors[i][1] * direction[1] +
                       decodedColors[i][2] * direction[2];
        }

        ASSERT(numColors <= kNumPixelsInBlock);

        int encodedColors[kNumPixelsInBlock];
        if (nonOpaquePunchThroughAlpha)
        {
            for (size_t i = 0; i < numColors; i++)
            {
                const int count = pixelIndexCounts[i];
                if (count > 0)
                {
                    // In non-opaque mode, 3 is for tranparent pixels.

                    if (0 == subblockColors[i].A)
                    {
                        encodedColors[i] = 3;
                    }
                    else
                    {
                        const R8G8B8A8 &pixel = subblockColors[i];
                        const int dot         = pixel.R * direction[0] + pixel.G * direction[1] +
                                        pixel.B * direction[2];
                        const int factor = gl::clamp(
                            static_cast<int>(
                                (static_cast<float>(dot - stops[1]) / (stops[0] - stops[1])) * 2 +
                                0.5f),
                            0, 2);
                        switch (factor)
                        {
                            case 0:
                                encodedColors[i] = 0;
                                break;
                            case 1:
                                encodedColors[i] = 2;
                                break;
                            case 2:
                            default:
                                encodedColors[i] = 1;
                                break;
                        }
                    }
                }
            }
        }
        else
        {
            for (size_t i = 0; i < numColors; i++)
            {
                const int count = pixelIndexCounts[i];
                if (count > 0)
                {
                    // In opaque mode, the code is from 0 to 3.

                    const R8G8B8A8 &pixel = subblockColors[i];
                    const int dot =
                        pixel.R * direction[0] + pixel.G * direction[1] + pixel.B * direction[2];
                    const int factor = gl::clamp(
                        static_cast<int>(
                            (static_cast<float>(dot - stops[1]) / (stops[0] - stops[1])) * 3 +
                            0.5f),
                        0, 3);
                    switch (factor)
                    {
                        case 0:
                            encodedColors[i] = 1;
                            break;
                        case 1:
                            encodedColors[i] = 3;
                            break;
                        case 2:
                            encodedColors[i] = 2;
                            break;
                        case 3:
                        default:
                            encodedColors[i] = 0;
                            break;
                    }
                }
            }
        }

        uint32_t bits = 0;
        for (int i = kNumPixelsInBlock - 1; i >= 0; i--)
        {
            bits <<= 2;
            bits |= encodedColors[pixelIndices[i]];
        }

        return bits;
    }

    void packBC1(void *bc1,
                 const int *pixelIndices,
                 const int *pixelIndexCounts,
                 const R8G8B8A8 *subblockColors,
                 size_t numColors,
                 int minColorIndex,
                 int maxColorIndex,
                 bool nonOpaquePunchThroughAlpha) const
    {
        const R8G8B8A8 &minColor = subblockColors[minColorIndex];
        const R8G8B8A8 &maxColor = subblockColors[maxColorIndex];

        uint32_t bits;
        uint16_t max16 = RGB8ToRGB565(maxColor);
        uint16_t min16 = RGB8ToRGB565(minColor);
        if (max16 != min16)
        {
            // Find the best BC1 code for each pixel
            bits = matchBC1Bits(pixelIndices, pixelIndexCounts, subblockColors, numColors, minColor,
                                maxColor, nonOpaquePunchThroughAlpha);
        }
        else
        {
            // Same colors, BC1 index 0 is the color in both opaque and transparent mode
            bits = 0;
            // BC1 index 3 is transparent
            if (nonOpaquePunchThroughAlpha)
            {
                for (int i = 0; i < kNumPixelsInBlock; i++)
                {
                    if (0 == subblockColors[pixelIndices[i]].A)
                    {
                        bits |= (3 << (i * 2));
                    }
                }
            }
        }

        if (max16 < min16)
        {
            std::swap(max16, min16);

            uint32_t xorMask = 0;
            if (nonOpaquePunchThroughAlpha)
            {
                // In transparent mode switching the colors is doing the
                // following code swap: 0 <-> 1. 0xA selects the second bit of
                // each code, bits >> 1 selects the first bit of the code when
                // the seconds bit is set (case 2 and 3). We invert all the
                // non-selected bits, that is the first bit when the code is
                // 0 or 1.
                xorMask = ~((bits >> 1) | 0xAAAAAAAA);
            }
            else
            {
                // In opaque mode switching the two colors is doing the
                // following code swaps: 0 <-> 1 and 2 <-> 3. This is
                // equivalent to flipping the first bit of each code
                // (5 = 0b0101)
                xorMask = 0x55555555;
            }
            bits ^= xorMask;
        }

        struct BC1Block
        {
            uint16_t color0;
            uint16_t color1;
            uint32_t bits;
        };

        // Encode the opaqueness in the order of the two BC1 colors
        BC1Block *dest = reinterpret_cast<BC1Block *>(bc1);
        if (nonOpaquePunchThroughAlpha)
        {
            dest->color0 = min16;
            dest->color1 = max16;
        }
        else
        {
            dest->color0 = max16;
            dest->color1 = min16;
        }
        dest->bits = bits;
    }

    void transcodeIndividualBlockToBC1(uint8_t *dest,
                                       size_t x,
                                       size_t y,
                                       size_t w,
                                       size_t h,
                                       const uint8_t alphaValues[4][4],
                                       bool nonOpaquePunchThroughAlpha) const
    {
        const auto &block = u.idht.mode.idm.colors.indiv;
        int r1            = extend_4to8bits(block.R1);
        int g1            = extend_4to8bits(block.G1);
        int b1            = extend_4to8bits(block.B1);
        int r2            = extend_4to8bits(block.R2);
        int g2            = extend_4to8bits(block.G2);
        int b2            = extend_4to8bits(block.B2);
        transcodeIndividualOrDifferentialBlockToBC1(dest, x, y, w, h, r1, g1, b1, r2, g2, b2,
                                                    alphaValues, nonOpaquePunchThroughAlpha);
    }

    void transcodeDifferentialBlockToBC1(uint8_t *dest,
                                         size_t x,
                                         size_t y,
                                         size_t w,
                                         size_t h,
                                         const uint8_t alphaValues[4][4],
                                         bool nonOpaquePunchThroughAlpha) const
    {
        const auto &block = u.idht.mode.idm.colors.diff;
        int b1            = extend_5to8bits(block.B);
        int g1            = extend_5to8bits(block.G);
        int r1            = extend_5to8bits(block.R);
        int r2            = extend_5to8bits(block.R + block.dR);
        int g2            = extend_5to8bits(block.G + block.dG);
        int b2            = extend_5to8bits(block.B + block.dB);
        transcodeIndividualOrDifferentialBlockToBC1(dest, x, y, w, h, r1, g1, b1, r2, g2, b2,
                                                    alphaValues, nonOpaquePunchThroughAlpha);
    }

    void extractPixelIndices(int *pixelIndices,
                             int *pixelIndicesCounts,
                             size_t x,
                             size_t y,
                             size_t w,
                             size_t h,
                             bool flipbit,
                             size_t subblockIdx) const
    {
        size_t dxBegin = 0;
        size_t dxEnd   = 4;
        size_t dyBegin = subblockIdx * 2;
        size_t dyEnd   = dyBegin + 2;
        if (!flipbit)
        {
            std::swap(dxBegin, dyBegin);
            std::swap(dxEnd, dyEnd);
        }

        for (size_t j = dyBegin; j < dyEnd; j++)
        {
            int *row = &pixelIndices[j * 4];
            for (size_t i = dxBegin; i < dxEnd; i++)
            {
                const size_t pixelIndex = subblockIdx * 4 + getIndex(i, j);
                row[i]                  = static_cast<int>(pixelIndex);
                pixelIndicesCounts[pixelIndex]++;
            }
        }
    }

    void selectEndPointPCA(const int *pixelIndexCounts,
                           const R8G8B8A8 *subblockColors,
                           size_t numColors,
                           int *minColorIndex,
                           int *maxColorIndex) const
    {
        // determine color distribution
        int mu[3], min[3], max[3];
        for (int ch = 0; ch < 3; ch++)
        {
            int muv  = 0;
            int minv = 255;
            int maxv = 0;
            for (size_t i = 0; i < numColors; i++)
            {
                const int count = pixelIndexCounts[i];
                if (count > 0)
                {
                    const auto &pixel = subblockColors[i];
                    if (pixel.A > 0)
                    {
                        // Non-transparent pixels
                        muv += (&pixel.R)[ch] * count;
                        minv = std::min<int>(minv, (&pixel.R)[ch]);
                        maxv = std::max<int>(maxv, (&pixel.R)[ch]);
                    }
                }
            }

            mu[ch]  = (muv + kNumPixelsInBlock / 2) / kNumPixelsInBlock;
            min[ch] = minv;
            max[ch] = maxv;
        }

        // determine covariance matrix
        int cov[6] = {0, 0, 0, 0, 0, 0};
        for (size_t i = 0; i < numColors; i++)
        {
            const int count = pixelIndexCounts[i];
            if (count > 0)
            {
                const auto &pixel = subblockColors[i];
                if (pixel.A > 0)
                {
                    int r = pixel.R - mu[0];
                    int g = pixel.G - mu[1];
                    int b = pixel.B - mu[2];

                    cov[0] += r * r * count;
                    cov[1] += r * g * count;
                    cov[2] += r * b * count;
                    cov[3] += g * g * count;
                    cov[4] += g * b * count;
                    cov[5] += b * b * count;
                }
            }
        }

        // Power iteration algorithm to get the eigenvalues and eigenvector

        // Starts with diagonal vector
        float vfr = static_cast<float>(max[0] - min[0]);
        float vfg = static_cast<float>(max[1] - min[1]);
        float vfb = static_cast<float>(max[2] - min[2]);
        float eigenvalue;

        static const size_t kPowerIterations = 4;
        for (size_t i = 0; i < kPowerIterations; i++)
        {
            float r = vfr * cov[0] + vfg * cov[1] + vfb * cov[2];
            float g = vfr * cov[1] + vfg * cov[3] + vfb * cov[4];
            float b = vfr * cov[2] + vfg * cov[4] + vfb * cov[5];

            vfr = r;
            vfg = g;
            vfb = b;

            eigenvalue = sqrt(r * r + g * g + b * b);
            if (eigenvalue > 0)
            {
                float invNorm = 1.0f / eigenvalue;
                vfr *= invNorm;
                vfg *= invNorm;
                vfb *= invNorm;
            }
        }

        int vr, vg, vb;

        static const float kDefaultLuminanceThreshold = 4.0f * 255;
        static const float kQuantizeRange             = 512.0f;
        if (eigenvalue < kDefaultLuminanceThreshold)  // too small, default to luminance
        {
            // Luminance weights defined by ITU-R Recommendation BT.601, scaled by 1000
            vr = 299;
            vg = 587;
            vb = 114;
        }
        else
        {
            // From the eigenvalue and eigenvector, choose the axis to project
            // colors on. When projecting colors we want to do integer computations
            // for speed, so we normalize the eigenvector to the [0, 512] range.
            float magn = std::max(std::max(std::abs(vfr), std::abs(vfg)), std::abs(vfb));
            magn       = kQuantizeRange / magn;
            vr         = static_cast<int>(vfr * magn);
            vg         = static_cast<int>(vfg * magn);
            vb         = static_cast<int>(vfb * magn);
        }

        // Pick colors at extreme points
        int minD        = INT_MAX;
        int maxD        = 0;
        size_t minIndex = 0;
        size_t maxIndex = 0;
        for (size_t i = 0; i < numColors; i++)
        {
            const int count = pixelIndexCounts[i];
            if (count > 0)
            {
                const auto &pixel = subblockColors[i];
                if (pixel.A > 0)
                {
                    int dot = pixel.R * vr + pixel.G * vg + pixel.B * vb;
                    if (dot < minD)
                    {
                        minD     = dot;
                        minIndex = i;
                    }
                    if (dot > maxD)
                    {
                        maxD     = dot;
                        maxIndex = i;
                    }
                }
            }
        }

        *minColorIndex = static_cast<int>(minIndex);
        *maxColorIndex = static_cast<int>(maxIndex);
    }

    void transcodeIndividualOrDifferentialBlockToBC1(uint8_t *dest,
                                                     size_t x,
                                                     size_t y,
                                                     size_t w,
                                                     size_t h,
                                                     int r1,
                                                     int g1,
                                                     int b1,
                                                     int r2,
                                                     int g2,
                                                     int b2,
                                                     const uint8_t alphaValues[4][4],
                                                     bool nonOpaquePunchThroughAlpha) const
    {
        // A BC1 block has 2 endpoints, pixels is encoded as linear
        // interpolations of them. A ETC1/ETC2 individual or differential block
        // has 2 subblocks. Each subblock has one color and a modifier. We
        // select axis by principal component analysis (PCA) to use as
        // our two BC1 endpoints and then map pixels to BC1 by projecting on the
        // line between the two endpoints and choosing the right fraction.

        // The goal of this algorithm is make it faster than decode ETC to RGBs
        //   and then encode to BC. To achieve this, we only extract subblock
        //   colors, pixel indices, and counts of each pixel indices from ETC.
        //   With those information, we can only encode used subblock colors
        //   to BC1, and copy the bits to the right pixels.
        // Fully decode and encode need to process 16 RGBA pixels. With this
        //   algorithm, it's 8 pixels at maximum for a individual or
        //   differential block. Saves us bandwidth and computations.

        static const size_t kNumColors = 8;

        const IntensityModifier *intensityModifier =
            nonOpaquePunchThroughAlpha ? intensityModifierNonOpaque : intensityModifierDefault;

        // Compute the colors that pixels can have in each subblock both for
        // the decoding of the RGBA data and BC1 encoding
        R8G8B8A8 subblockColors[kNumColors];
        for (size_t modifierIdx = 0; modifierIdx < 4; modifierIdx++)
        {
            if (nonOpaquePunchThroughAlpha && (modifierIdx == 2))
            {
                // In ETC opaque punch through formats, individual and
                // differential blocks take index 2 as transparent pixel.
                // Thus we don't need to compute its color, just assign it
                // as black.
                subblockColors[modifierIdx]     = createRGBA(0, 0, 0, 0);
                subblockColors[4 + modifierIdx] = createRGBA(0, 0, 0, 0);
            }
            else
            {
                const int i1                = intensityModifier[u.idht.mode.idm.cw1][modifierIdx];
                subblockColors[modifierIdx] = createRGBA(r1 + i1, g1 + i1, b1 + i1);

                const int i2 = intensityModifier[u.idht.mode.idm.cw2][modifierIdx];
                subblockColors[4 + modifierIdx] = createRGBA(r2 + i2, g2 + i2, b2 + i2);
            }
        }

        int pixelIndices[kNumPixelsInBlock];
        int pixelIndexCounts[kNumColors] = {0};
        // Extract pixel indices from a ETC block.
        for (size_t blockIdx = 0; blockIdx < 2; blockIdx++)
        {
            extractPixelIndices(pixelIndices, pixelIndexCounts, x, y, w, h, u.idht.mode.idm.flipbit,
                                blockIdx);
        }

        int minColorIndex, maxColorIndex;
        selectEndPointPCA(pixelIndexCounts, subblockColors, kNumColors, &minColorIndex,
                          &maxColorIndex);

        packBC1(dest, pixelIndices, pixelIndexCounts, subblockColors, kNumColors, minColorIndex,
                maxColorIndex, nonOpaquePunchThroughAlpha);
    }

    void transcodeTBlockToBC1(uint8_t *dest,
                              size_t x,
                              size_t y,
                              size_t w,
                              size_t h,
                              const uint8_t alphaValues[4][4],
                              bool nonOpaquePunchThroughAlpha) const
    {
        static const size_t kNumColors = 4;

        // Table C.8, distance index for T and H modes
        const auto &block = u.idht.mode.tm;

        int r1 = extend_4to8bits(block.TR1a << 2 | block.TR1b);
        int g1 = extend_4to8bits(block.TG1);
        int b1 = extend_4to8bits(block.TB1);
        int r2 = extend_4to8bits(block.TR2);
        int g2 = extend_4to8bits(block.TG2);
        int b2 = extend_4to8bits(block.TB2);

        static int distance[8] = {3, 6, 11, 16, 23, 32, 41, 64};
        const int d            = distance[block.Tda << 1 | block.Tdb];

        // In ETC opaque punch through formats, index == 2 means transparent pixel.
        // Thus we don't need to compute its color, just assign it as black.
        const R8G8B8A8 paintColors[kNumColors] = {
            createRGBA(r1, g1, b1),
            createRGBA(r2 + d, g2 + d, b2 + d),
            nonOpaquePunchThroughAlpha ? createRGBA(0, 0, 0, 0) : createRGBA(r2, g2, b2),
            createRGBA(r2 - d, g2 - d, b2 - d),
        };

        int pixelIndices[kNumPixelsInBlock];
        int pixelIndexCounts[kNumColors] = {0};
        for (size_t j = 0; j < 4; j++)
        {
            int *row = &pixelIndices[j * 4];
            for (size_t i = 0; i < 4; i++)
            {
                const size_t pixelIndex = getIndex(i, j);
                row[i]                  = static_cast<int>(pixelIndex);
                pixelIndexCounts[pixelIndex]++;
            }
        }

        int minColorIndex, maxColorIndex;
        selectEndPointPCA(pixelIndexCounts, paintColors, kNumColors, &minColorIndex,
                          &maxColorIndex);

        packBC1(dest, pixelIndices, pixelIndexCounts, paintColors, kNumColors, minColorIndex,
                maxColorIndex, nonOpaquePunchThroughAlpha);
    }

    void transcodeHBlockToBC1(uint8_t *dest,
                              size_t x,
                              size_t y,
                              size_t w,
                              size_t h,
                              const uint8_t alphaValues[4][4],
                              bool nonOpaquePunchThroughAlpha) const
    {
        static const size_t kNumColors = 4;

        // Table C.8, distance index for T and H modes
        const auto &block = u.idht.mode.hm;

        int r1 = extend_4to8bits(block.HR1);
        int g1 = extend_4to8bits(block.HG1a << 1 | block.HG1b);
        int b1 = extend_4to8bits(block.HB1a << 3 | block.HB1b << 1 | block.HB1c);
        int r2 = extend_4to8bits(block.HR2);
        int g2 = extend_4to8bits(block.HG2a << 1 | block.HG2b);
        int b2 = extend_4to8bits(block.HB2);

        static const int distance[8] = {3, 6, 11, 16, 23, 32, 41, 64};
        const int orderingTrickBit =
            ((r1 << 16 | g1 << 8 | b1) >= (r2 << 16 | g2 << 8 | b2) ? 1 : 0);
        const int d = distance[(block.Hda << 2) | (block.Hdb << 1) | orderingTrickBit];

        // In ETC opaque punch through formats, index == 2 means transparent pixel.
        // Thus we don't need to compute its color, just assign it as black.
        const R8G8B8A8 paintColors[kNumColors] = {
            createRGBA(r1 + d, g1 + d, b1 + d),
            createRGBA(r1 - d, g1 - d, b1 - d),
            nonOpaquePunchThroughAlpha ? createRGBA(0, 0, 0, 0)
                                       : createRGBA(r2 + d, g2 + d, b2 + d),
            createRGBA(r2 - d, g2 - d, b2 - d),
        };

        int pixelIndices[kNumPixelsInBlock];
        int pixelIndexCounts[kNumColors] = {0};
        for (size_t j = 0; j < 4; j++)
        {
            int *row = &pixelIndices[j * 4];
            for (size_t i = 0; i < 4; i++)
            {
                const size_t pixelIndex = getIndex(i, j);
                row[i]                  = static_cast<int>(pixelIndex);
                pixelIndexCounts[pixelIndex]++;
            }
        }

        int minColorIndex, maxColorIndex;
        selectEndPointPCA(pixelIndexCounts, paintColors, kNumColors, &minColorIndex,
                          &maxColorIndex);

        packBC1(dest, pixelIndices, pixelIndexCounts, paintColors, kNumColors, minColorIndex,
                maxColorIndex, nonOpaquePunchThroughAlpha);
    }

    void transcodePlanarBlockToBC1(uint8_t *dest,
                                   size_t x,
                                   size_t y,
                                   size_t w,
                                   size_t h,
                                   const uint8_t alphaValues[4][4]) const
    {
        static const size_t kNumColors = kNumPixelsInBlock;

        R8G8B8A8 rgbaBlock[kNumColors];
        decodePlanarBlock(reinterpret_cast<uint8_t *>(rgbaBlock), x, y, w, h, sizeof(R8G8B8A8) * 4,
                          alphaValues);

        // Planar block doesn't have a color table, fill indices as full
        int pixelIndices[kNumPixelsInBlock] = {0, 1, 2,  3,  4,  5,  6,  7,
                                               8, 9, 10, 11, 12, 13, 14, 15};
        int pixelIndexCounts[kNumColors]    = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

        int minColorIndex, maxColorIndex;
        selectEndPointPCA(pixelIndexCounts, rgbaBlock, kNumColors, &minColorIndex, &maxColorIndex);

        packBC1(dest, pixelIndices, pixelIndexCounts, rgbaBlock, kNumColors, minColorIndex,
                maxColorIndex, false);
    }

    // Single channel utility functions
    int getSingleEACChannel(size_t x, size_t y, bool isSigned) const
    {
        int codeword   = isSigned ? u.scblk.base_codeword.s : u.scblk.base_codeword.us;
        int multiplier = (u.scblk.multiplier == 0) ? 1 : u.scblk.multiplier * 8;
        return codeword * 8 + 4 + getSingleChannelModifier(x, y) * multiplier;
    }

    int getSingleETC2Channel(size_t x, size_t y, bool isSigned) const
    {
        int codeword = isSigned ? u.scblk.base_codeword.s : u.scblk.base_codeword.us;
        return codeword + getSingleChannelModifier(x, y) * u.scblk.multiplier;
    }

    int getSingleChannelIndex(size_t x, size_t y) const
    {
        ASSERT(x < 4 && y < 4);

        // clang-format off
        switch (x * 4 + y)
        {
            case 0: return u.scblk.ma;
            case 1: return u.scblk.mb;
            case 2: return u.scblk.mc1 << 1 | u.scblk.mc2;
            case 3: return u.scblk.md;
            case 4: return u.scblk.me;
            case 5: return u.scblk.mf1 << 2 | u.scblk.mf2;
            case 6: return u.scblk.mg;
            case 7: return u.scblk.mh;
            case 8: return u.scblk.mi;
            case 9: return u.scblk.mj;
            case 10: return u.scblk.mk1 << 1 | u.scblk.mk2;
            case 11: return u.scblk.ml;
            case 12: return u.scblk.mm;
            case 13: return u.scblk.mn1 << 2 | u.scblk.mn2;
            case 14: return u.scblk.mo;
            case 15: return u.scblk.mp;
            default: UNREACHABLE(); return 0;
        }
        // clang-format on
    }

    int getSingleChannelModifier(size_t x, size_t y) const
    {
        // clang-format off
        static const int modifierTable[16][8] =
        {
            { -3, -6,  -9, -15, 2, 5, 8, 14 },
            { -3, -7, -10, -13, 2, 6, 9, 12 },
            { -2, -5,  -8, -13, 1, 4, 7, 12 },
            { -2, -4,  -6, -13, 1, 3, 5, 12 },
            { -3, -6,  -8, -12, 2, 5, 7, 11 },
            { -3, -7,  -9, -11, 2, 6, 8, 10 },
            { -4, -7,  -8, -11, 3, 6, 7, 10 },
            { -3, -5,  -8, -11, 2, 4, 7, 10 },
            { -2, -6,  -8, -10, 1, 5, 7,  9 },
            { -2, -5,  -8, -10, 1, 4, 7,  9 },
            { -2, -4,  -8, -10, 1, 3, 7,  9 },
            { -2, -5,  -7, -10, 1, 4, 6,  9 },
            { -3, -4,  -7, -10, 2, 3, 6,  9 },
            { -1, -2,  -3, -10, 0, 1, 2,  9 },
            { -4, -6,  -8,  -9, 3, 5, 7,  8 },
            { -3, -5,  -7,  -9, 2, 4, 6,  8 }
        };
        // clang-format on

        return modifierTable[u.scblk.table_index][getSingleChannelIndex(x, y)];
    }
};

// clang-format off
static const uint8_t DefaultETCAlphaValues[4][4] =
{
    { 255, 255, 255, 255 },
    { 255, 255, 255, 255 },
    { 255, 255, 255, 255 },
    { 255, 255, 255, 255 },
};

// clang-format on
void LoadR11EACToR8(size_t width,
                    size_t height,
                    size_t depth,
                    const uint8_t *input,
                    size_t inputRowPitch,
                    size_t inputDepthPitch,
                    uint8_t *output,
                    size_t outputRowPitch,
                    size_t outputDepthPitch,
                    bool isSigned)
{
    for (size_t z = 0; z < depth; z++)
    {
        for (size_t y = 0; y < height; y += 4)
        {
            const ETC2Block *sourceRow =
                priv::OffsetDataPointer<ETC2Block>(input, y / 4, z, inputRowPitch, inputDepthPitch);
            uint8_t *destRow =
                priv::OffsetDataPointer<uint8_t>(output, y, z, outputRowPitch, outputDepthPitch);

            for (size_t x = 0; x < width; x += 4)
            {
                const ETC2Block *sourceBlock = sourceRow + (x / 4);
                uint8_t *destPixels          = destRow + x;

                sourceBlock->decodeAsSingleETC2Channel(destPixels, x, y, width, height, 1,
                                                       outputRowPitch, isSigned);
            }
        }
    }
}

void LoadRG11EACToRG8(size_t width,
                      size_t height,
                      size_t depth,
                      const uint8_t *input,
                      size_t inputRowPitch,
                      size_t inputDepthPitch,
                      uint8_t *output,
                      size_t outputRowPitch,
                      size_t outputDepthPitch,
                      bool isSigned)
{
    for (size_t z = 0; z < depth; z++)
    {
        for (size_t y = 0; y < height; y += 4)
        {
            const ETC2Block *sourceRow =
                priv::OffsetDataPointer<ETC2Block>(input, y / 4, z, inputRowPitch, inputDepthPitch);
            uint8_t *destRow =
                priv::OffsetDataPointer<uint8_t>(output, y, z, outputRowPitch, outputDepthPitch);

            for (size_t x = 0; x < width; x += 4)
            {
                uint8_t *destPixelsRed          = destRow + (x * 2);
                const ETC2Block *sourceBlockRed = sourceRow + (x / 2);
                sourceBlockRed->decodeAsSingleETC2Channel(destPixelsRed, x, y, width, height, 2,
                                                          outputRowPitch, isSigned);

                uint8_t *destPixelsGreen          = destPixelsRed + 1;
                const ETC2Block *sourceBlockGreen = sourceBlockRed + 1;
                sourceBlockGreen->decodeAsSingleETC2Channel(destPixelsGreen, x, y, width, height, 2,
                                                            outputRowPitch, isSigned);
            }
        }
    }
}

void LoadR11EACToR16(size_t width,
                     size_t height,
                     size_t depth,
                     const uint8_t *input,
                     size_t inputRowPitch,
                     size_t inputDepthPitch,
                     uint8_t *output,
                     size_t outputRowPitch,
                     size_t outputDepthPitch,
                     bool isSigned,
                     bool isFloat)
{
    for (size_t z = 0; z < depth; z++)
    {
        for (size_t y = 0; y < height; y += 4)
        {
            const ETC2Block *sourceRow =
                priv::OffsetDataPointer<ETC2Block>(input, y / 4, z, inputRowPitch, inputDepthPitch);
            uint16_t *destRow =
                priv::OffsetDataPointer<uint16_t>(output, y, z, outputRowPitch, outputDepthPitch);

            for (size_t x = 0; x < width; x += 4)
            {
                const ETC2Block *sourceBlock = sourceRow + (x / 4);
                uint16_t *destPixels         = destRow + x;

                sourceBlock->decodeAsSingleEACChannel(destPixels, x, y, width, height, 1,
                                                      outputRowPitch, isSigned, isFloat);
            }
        }
    }
}

void LoadRG11EACToRG16(size_t width,
                       size_t height,
                       size_t depth,
                       const uint8_t *input,
                       size_t inputRowPitch,
                       size_t inputDepthPitch,
                       uint8_t *output,
                       size_t outputRowPitch,
                       size_t outputDepthPitch,
                       bool isSigned,
                       bool isFloat)
{
    for (size_t z = 0; z < depth; z++)
    {
        for (size_t y = 0; y < height; y += 4)
        {
            const ETC2Block *sourceRow =
                priv::OffsetDataPointer<ETC2Block>(input, y / 4, z, inputRowPitch, inputDepthPitch);
            uint16_t *destRow =
                priv::OffsetDataPointer<uint16_t>(output, y, z, outputRowPitch, outputDepthPitch);

            for (size_t x = 0; x < width; x += 4)
            {
                uint16_t *destPixelsRed         = destRow + (x * 2);
                const ETC2Block *sourceBlockRed = sourceRow + (x / 2);
                sourceBlockRed->decodeAsSingleEACChannel(destPixelsRed, x, y, width, height, 2,
                                                         outputRowPitch, isSigned, isFloat);

                uint16_t *destPixelsGreen         = destPixelsRed + 1;
                const ETC2Block *sourceBlockGreen = sourceBlockRed + 1;
                sourceBlockGreen->decodeAsSingleEACChannel(destPixelsGreen, x, y, width, height, 2,
                                                           outputRowPitch, isSigned, isFloat);
            }
        }
    }
}

void LoadETC2RGB8ToRGBA8(size_t width,
                         size_t height,
                         size_t depth,
                         const uint8_t *input,
                         size_t inputRowPitch,
                         size_t inputDepthPitch,
                         uint8_t *output,
                         size_t outputRowPitch,
                         size_t outputDepthPitch,
                         bool punchthroughAlpha)
{
    for (size_t z = 0; z < depth; z++)
    {
        for (size_t y = 0; y < height; y += 4)
        {
            const ETC2Block *sourceRow =
                priv::OffsetDataPointer<ETC2Block>(input, y / 4, z, inputRowPitch, inputDepthPitch);
            uint8_t *destRow =
                priv::OffsetDataPointer<uint8_t>(output, y, z, outputRowPitch, outputDepthPitch);

            for (size_t x = 0; x < width; x += 4)
            {
                const ETC2Block *sourceBlock = sourceRow + (x / 4);
                uint8_t *destPixels          = destRow + (x * 4);

                sourceBlock->decodeAsRGB(destPixels, x, y, width, height, outputRowPitch,
                                         DefaultETCAlphaValues, punchthroughAlpha);
            }
        }
    }
}

void LoadETC2RGB8ToBC1(size_t width,
                       size_t height,
                       size_t depth,
                       const uint8_t *input,
                       size_t inputRowPitch,
                       size_t inputDepthPitch,
                       uint8_t *output,
                       size_t outputRowPitch,
                       size_t outputDepthPitch,
                       bool punchthroughAlpha)
{
    for (size_t z = 0; z < depth; z++)
    {
        for (size_t y = 0; y < height; y += 4)
        {
            const ETC2Block *sourceRow =
                priv::OffsetDataPointer<ETC2Block>(input, y / 4, z, inputRowPitch, inputDepthPitch);
            uint8_t *destRow = priv::OffsetDataPointer<uint8_t>(output, y / 4, z, outputRowPitch,
                                                                outputDepthPitch);

            for (size_t x = 0; x < width; x += 4)
            {
                const ETC2Block *sourceBlock = sourceRow + (x / 4);
                uint8_t *destPixels          = destRow + (x * 2);

                sourceBlock->transcodeAsBC1(destPixels, x, y, width, height, DefaultETCAlphaValues,
                                            punchthroughAlpha);
            }
        }
    }
}

void LoadETC2RGBA8ToRGBA8(size_t width,
                          size_t height,
                          size_t depth,
                          const uint8_t *input,
                          size_t inputRowPitch,
                          size_t inputDepthPitch,
                          uint8_t *output,
                          size_t outputRowPitch,
                          size_t outputDepthPitch,
                          bool srgb)
{
    uint8_t decodedAlphaValues[4][4];

    for (size_t z = 0; z < depth; z++)
    {
        for (size_t y = 0; y < height; y += 4)
        {
            const ETC2Block *sourceRow =
                priv::OffsetDataPointer<ETC2Block>(input, y / 4, z, inputRowPitch, inputDepthPitch);
            uint8_t *destRow =
                priv::OffsetDataPointer<uint8_t>(output, y, z, outputRowPitch, outputDepthPitch);

            for (size_t x = 0; x < width; x += 4)
            {
                const ETC2Block *sourceBlockAlpha = sourceRow + (x / 2);
                sourceBlockAlpha->decodeAsSingleETC2Channel(
                    reinterpret_cast<uint8_t *>(decodedAlphaValues), x, y, width, height, 1, 4,
                    false);

                uint8_t *destPixels             = destRow + (x * 4);
                const ETC2Block *sourceBlockRGB = sourceBlockAlpha + 1;
                sourceBlockRGB->decodeAsRGB(destPixels, x, y, width, height, outputRowPitch,
                                            decodedAlphaValues, false);
            }
        }
    }
}

}  // anonymous namespace

void LoadETC1RGB8ToRGBA8(size_t width,
                         size_t height,
                         size_t depth,
                         const uint8_t *input,
                         size_t inputRowPitch,
                         size_t inputDepthPitch,
                         uint8_t *output,
                         size_t outputRowPitch,
                         size_t outputDepthPitch)
{
    LoadETC2RGB8ToRGBA8(width, height, depth, input, inputRowPitch, inputDepthPitch, output,
                        outputRowPitch, outputDepthPitch, false);
}

void LoadETC1RGB8ToBC1(size_t width,
                       size_t height,
                       size_t depth,
                       const uint8_t *input,
                       size_t inputRowPitch,
                       size_t inputDepthPitch,
                       uint8_t *output,
                       size_t outputRowPitch,
                       size_t outputDepthPitch)
{
    LoadETC2RGB8ToBC1(width, height, depth, input, inputRowPitch, inputDepthPitch, output,
                      outputRowPitch, outputDepthPitch, false);
}

void LoadEACR11ToR8(size_t width,
                    size_t height,
                    size_t depth,
                    const uint8_t *input,
                    size_t inputRowPitch,
                    size_t inputDepthPitch,
                    uint8_t *output,
                    size_t outputRowPitch,
                    size_t outputDepthPitch)
{
    LoadR11EACToR8(width, height, depth, input, inputRowPitch, inputDepthPitch, output,
                   outputRowPitch, outputDepthPitch, false);
}

void LoadEACR11SToR8(size_t width,
                     size_t height,
                     size_t depth,
                     const uint8_t *input,
                     size_t inputRowPitch,
                     size_t inputDepthPitch,
                     uint8_t *output,
                     size_t outputRowPitch,
                     size_t outputDepthPitch)
{
    LoadR11EACToR8(width, height, depth, input, inputRowPitch, inputDepthPitch, output,
                   outputRowPitch, outputDepthPitch, true);
}

void LoadEACRG11ToRG8(size_t width,
                      size_t height,
                      size_t depth,
                      const uint8_t *input,
                      size_t inputRowPitch,
                      size_t inputDepthPitch,
                      uint8_t *output,
                      size_t outputRowPitch,
                      size_t outputDepthPitch)
{
    LoadRG11EACToRG8(width, height, depth, input, inputRowPitch, inputDepthPitch, output,
                     outputRowPitch, outputDepthPitch, false);
}

void LoadEACRG11SToRG8(size_t width,
                       size_t height,
                       size_t depth,
                       const uint8_t *input,
                       size_t inputRowPitch,
                       size_t inputDepthPitch,
                       uint8_t *output,
                       size_t outputRowPitch,
                       size_t outputDepthPitch)
{
    LoadRG11EACToRG8(width, height, depth, input, inputRowPitch, inputDepthPitch, output,
                     outputRowPitch, outputDepthPitch, true);
}

void LoadEACR11ToR16(size_t width,
                     size_t height,
                     size_t depth,
                     const uint8_t *input,
                     size_t inputRowPitch,
                     size_t inputDepthPitch,
                     uint8_t *output,
                     size_t outputRowPitch,
                     size_t outputDepthPitch)
{
    LoadR11EACToR16(width, height, depth, input, inputRowPitch, inputDepthPitch, output,
                    outputRowPitch, outputDepthPitch, false, false);
}

void LoadEACR11SToR16(size_t width,
                      size_t height,
                      size_t depth,
                      const uint8_t *input,
                      size_t inputRowPitch,
                      size_t inputDepthPitch,
                      uint8_t *output,
                      size_t outputRowPitch,
                      size_t outputDepthPitch)
{
    LoadR11EACToR16(width, height, depth, input, inputRowPitch, inputDepthPitch, output,
                    outputRowPitch, outputDepthPitch, true, false);
}

void LoadEACRG11ToRG16(size_t width,
                       size_t height,
                       size_t depth,
                       const uint8_t *input,
                       size_t inputRowPitch,
                       size_t inputDepthPitch,
                       uint8_t *output,
                       size_t outputRowPitch,
                       size_t outputDepthPitch)
{
    LoadRG11EACToRG16(width, height, depth, input, inputRowPitch, inputDepthPitch, output,
                      outputRowPitch, outputDepthPitch, false, false);
}

void LoadEACRG11SToRG16(size_t width,
                        size_t height,
                        size_t depth,
                        const uint8_t *input,
                        size_t inputRowPitch,
                        size_t inputDepthPitch,
                        uint8_t *output,
                        size_t outputRowPitch,
                        size_t outputDepthPitch)
{
    LoadRG11EACToRG16(width, height, depth, input, inputRowPitch, inputDepthPitch, output,
                      outputRowPitch, outputDepthPitch, true, false);
}

void LoadEACR11ToR16F(size_t width,
                      size_t height,
                      size_t depth,
                      const uint8_t *input,
                      size_t inputRowPitch,
                      size_t inputDepthPitch,
                      uint8_t *output,
                      size_t outputRowPitch,
                      size_t outputDepthPitch)
{
    LoadR11EACToR16(width, height, depth, input, inputRowPitch, inputDepthPitch, output,
                    outputRowPitch, outputDepthPitch, false, true);
}

void LoadEACR11SToR16F(size_t width,
                       size_t height,
                       size_t depth,
                       const uint8_t *input,
                       size_t inputRowPitch,
                       size_t inputDepthPitch,
                       uint8_t *output,
                       size_t outputRowPitch,
                       size_t outputDepthPitch)
{
    LoadR11EACToR16(width, height, depth, input, inputRowPitch, inputDepthPitch, output,
                    outputRowPitch, outputDepthPitch, true, true);
}

void LoadEACRG11ToRG16F(size_t width,
                        size_t height,
                        size_t depth,
                        const uint8_t *input,
                        size_t inputRowPitch,
                        size_t inputDepthPitch,
                        uint8_t *output,
                        size_t outputRowPitch,
                        size_t outputDepthPitch)
{
    LoadRG11EACToRG16(width, height, depth, input, inputRowPitch, inputDepthPitch, output,
                      outputRowPitch, outputDepthPitch, false, true);
}

void LoadEACRG11SToRG16F(size_t width,
                         size_t height,
                         size_t depth,
                         const uint8_t *input,
                         size_t inputRowPitch,
                         size_t inputDepthPitch,
                         uint8_t *output,
                         size_t outputRowPitch,
                         size_t outputDepthPitch)
{
    LoadRG11EACToRG16(width, height, depth, input, inputRowPitch, inputDepthPitch, output,
                      outputRowPitch, outputDepthPitch, true, true);
}

void LoadETC2RGB8ToRGBA8(size_t width,
                         size_t height,
                         size_t depth,
                         const uint8_t *input,
                         size_t inputRowPitch,
                         size_t inputDepthPitch,
                         uint8_t *output,
                         size_t outputRowPitch,
                         size_t outputDepthPitch)
{
    LoadETC2RGB8ToRGBA8(width, height, depth, input, inputRowPitch, inputDepthPitch, output,
                        outputRowPitch, outputDepthPitch, false);
}

void LoadETC2RGB8ToBC1(size_t width,
                       size_t height,
                       size_t depth,
                       const uint8_t *input,
                       size_t inputRowPitch,
                       size_t inputDepthPitch,
                       uint8_t *output,
                       size_t outputRowPitch,
                       size_t outputDepthPitch)
{
    LoadETC2RGB8ToBC1(width, height, depth, input, inputRowPitch, inputDepthPitch, output,
                      outputRowPitch, outputDepthPitch, false);
}

void LoadETC2SRGB8ToRGBA8(size_t width,
                          size_t height,
                          size_t depth,
                          const uint8_t *input,
                          size_t inputRowPitch,
                          size_t inputDepthPitch,
                          uint8_t *output,
                          size_t outputRowPitch,
                          size_t outputDepthPitch)
{
    LoadETC2RGB8ToRGBA8(width, height, depth, input, inputRowPitch, inputDepthPitch, output,
                        outputRowPitch, outputDepthPitch, false);
}

void LoadETC2SRGB8ToBC1(size_t width,
                        size_t height,
                        size_t depth,
                        const uint8_t *input,
                        size_t inputRowPitch,
                        size_t inputDepthPitch,
                        uint8_t *output,
                        size_t outputRowPitch,
                        size_t outputDepthPitch)
{
    LoadETC2RGB8ToBC1(width, height, depth, input, inputRowPitch, inputDepthPitch, output,
                      outputRowPitch, outputDepthPitch, false);
}

void LoadETC2RGB8A1ToRGBA8(size_t width,
                           size_t height,
                           size_t depth,
                           const uint8_t *input,
                           size_t inputRowPitch,
                           size_t inputDepthPitch,
                           uint8_t *output,
                           size_t outputRowPitch,
                           size_t outputDepthPitch)
{
    LoadETC2RGB8ToRGBA8(width, height, depth, input, inputRowPitch, inputDepthPitch, output,
                        outputRowPitch, outputDepthPitch, true);
}

void LoadETC2RGB8A1ToBC1(size_t width,
                         size_t height,
                         size_t depth,
                         const uint8_t *input,
                         size_t inputRowPitch,
                         size_t inputDepthPitch,
                         uint8_t *output,
                         size_t outputRowPitch,
                         size_t outputDepthPitch)
{
    LoadETC2RGB8ToBC1(width, height, depth, input, inputRowPitch, inputDepthPitch, output,
                      outputRowPitch, outputDepthPitch, true);
}

void LoadETC2SRGB8A1ToRGBA8(size_t width,
                            size_t height,
                            size_t depth,
                            const uint8_t *input,
                            size_t inputRowPitch,
                            size_t inputDepthPitch,
                            uint8_t *output,
                            size_t outputRowPitch,
                            size_t outputDepthPitch)
{
    LoadETC2RGB8ToRGBA8(width, height, depth, input, inputRowPitch, inputDepthPitch, output,
                        outputRowPitch, outputDepthPitch, true);
}

void LoadETC2SRGB8A1ToBC1(size_t width,
                          size_t height,
                          size_t depth,
                          const uint8_t *input,
                          size_t inputRowPitch,
                          size_t inputDepthPitch,
                          uint8_t *output,
                          size_t outputRowPitch,
                          size_t outputDepthPitch)
{
    LoadETC2RGB8ToBC1(width, height, depth, input, inputRowPitch, inputDepthPitch, output,
                      outputRowPitch, outputDepthPitch, true);
}

void LoadETC2RGBA8ToRGBA8(size_t width,
                          size_t height,
                          size_t depth,
                          const uint8_t *input,
                          size_t inputRowPitch,
                          size_t inputDepthPitch,
                          uint8_t *output,
                          size_t outputRowPitch,
                          size_t outputDepthPitch)
{
    LoadETC2RGBA8ToRGBA8(width, height, depth, input, inputRowPitch, inputDepthPitch, output,
                         outputRowPitch, outputDepthPitch, false);
}

void LoadETC2SRGBA8ToSRGBA8(size_t width,
                            size_t height,
                            size_t depth,
                            const uint8_t *input,
                            size_t inputRowPitch,
                            size_t inputDepthPitch,
                            uint8_t *output,
                            size_t outputRowPitch,
                            size_t outputDepthPitch)
{
    LoadETC2RGBA8ToRGBA8(width, height, depth, input, inputRowPitch, inputDepthPitch, output,
                         outputRowPitch, outputDepthPitch, true);
}

}  // namespace angle
