// This file is part of the FidelityFX SDK.
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#define GROUP_SIZE  8
#define FSR_RCAS_DENOISE 1

#include "../ffx_core.h"

#if FFX_HALF

    #define FFX_FSR_EASU_HALF 1
    FfxFloat16x4 FsrEasuRH(FfxFloat32x2 p) { return GatherEasuRed(p); }
    FfxFloat16x4 FsrEasuGH(FfxFloat32x2 p) { return GatherEasuGreen(p); }
    FfxFloat16x4 FsrEasuBH(FfxFloat32x2 p) { return GatherEasuBlue(p); }

#else

    #define FFX_FSR_EASU_FLOAT 1
    FfxFloat32x4 FsrEasuRF(FfxFloat32x2 p) { return GatherEasuRed(p); }
    FfxFloat32x4 FsrEasuGF(FfxFloat32x2 p) { return GatherEasuGreen(p); }
    FfxFloat32x4 FsrEasuBF(FfxFloat32x2 p) { return GatherEasuBlue(p); }

#endif // FFX_HALF

#if FFX_FSR1_OPTION_RCAS_PASSTHROUGH_ALPHA
    #define FSR_RCAS_PASSTHROUGH_ALPHA
#endif // FFX_FSR1_OPTION_RCAS_PASSTHROUGH_ALPHA

#include "ffx_fsr1.h"

void CurrFilter(FfxUInt32x2 pos)
{
#if FFX_HALF

    FfxFloat16x3 c;
    FsrEasuH(c, pos, Const0(), Const1(), Const2(), Const3());
    if (EASUSample().x == 1)
    {
        c *= c;
    }

#if FFX_FSR1_OPTION_SRGB_CONVERSIONS
    // Apply gamma if this is an sRGB format (auto-degamma'd on sampler read)
    c = pow(c, FfxFloat16x3(1.0 / 2.2, 1.0 / 2.2, 1.0 / 2.2));
#endif // FFX_FSR1_OPTION_SRGB_CONVERSIONS

    StoreEASUOutput(pos, c);

#else

    FfxFloat32x3 c;
    ffxFsrEasuFloat(c, pos, Const0(), Const1(), Const2(), Const3());
    if (EASUSample().x == 1)
    {
        c *= c;
    }

#if FFX_FSR1_OPTION_SRGB_CONVERSIONS
    // Apply gamma if this is an sRGB format (auto-degamma'd on sampler read)
    c = pow(c, FfxFloat32x3(1.f / 2.2f, 1.f / 2.2f, 1.f / 2.2f));
#endif // FFX_FSR1_OPTION_SRGB_CONVERSIONS

    StoreEASUOutput(pos, c);

#endif // FFX_HALF
}

void EASU(FfxUInt32x3 LocalThreadId, FfxUInt32x3 WorkGroupId, FfxUInt32x3 Dtid)
{
    // Do remapping of local xy in workgroup for a more PS-like swizzle pattern.
    FfxUInt32x2 gxy = ffxRemapForQuad(LocalThreadId.x) + FfxUInt32x2(WorkGroupId.x << 4u, WorkGroupId.y << 4u);
    CurrFilter(gxy);
    gxy.x += 8u;
    CurrFilter(gxy);
    gxy.y += 8u;
    CurrFilter(gxy);
    gxy.x -= 8u;
    CurrFilter(gxy);
}
