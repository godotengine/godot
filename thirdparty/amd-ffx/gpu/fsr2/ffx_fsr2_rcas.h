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

#if FFX_HALF && defined(__XBOX_SCARLETT) && defined(__XBATG_EXTRA_16_BIT_OPTIMISATION) && (__XBATG_EXTRA_16_BIT_OPTIMISATION == 1)
    #define FSR_RCAS_PREFER_PAIRED_VERSION 1
#else
    #define FSR_RCAS_PREFER_PAIRED_VERSION 0
#endif

void WriteUpscaledOutput(FFX_MIN16_U2 iPxHrPos, FfxFloat32x3 fUpscaledColor)
{
    StoreUpscaledOutput(FFX_MIN16_I2(iPxHrPos), fUpscaledColor);
}

#if FSR_RCAS_PREFER_PAIRED_VERSION

    #define FSR_RCAS_HX2 1

    FfxFloat16x4 FsrRcasLoadHx2(FfxInt16x2 p)
    {
        return FfxFloat16x4(LoadRCAS_Input(p));
    }
    void FsrRcasInputHx2(inout FfxFloat16x2 r, inout FfxFloat16x2 g, inout FfxFloat16x2 b)
    {
        PrepareRgbPaired(r, g, b, Exposure(), PreExposure());
    }

#else

    #define FSR_RCAS_F 1
    FfxFloat32x4 FsrRcasLoadF(FfxInt32x2 p)
    {
        FfxFloat32x4 fColor = LoadRCAS_Input(p);

        fColor.rgb = PrepareRgb(fColor.rgb, Exposure(), PreExposure());

        return fColor;
    }
    void FsrRcasInputF(inout FfxFloat32 r, inout FfxFloat32 g, inout FfxFloat32 b) {}

#endif // #if FSR_RCAS_PREFER_PAIRED_VERSION

// GODOT BEGINS
// Workaround for Godot GLSL processor not supporting conditional include.
// Thus we have to take the include statement out of the conditional block.

#include "../fsr1/ffx_fsr1.h"

#if FSR_RCAS_PREFER_PAIRED_VERSION

void CurrFilterPaired(FFX_MIN16_U2 pos)
{
	FfxFloat16x2 cr;
	FfxFloat16x2 cg;
	FfxFloat16x2 cb;
	FsrRcasHx2(cr, cg, cb, pos, RCASConfig());

	UnprepareRgbPaired(cr, cg, cb, Exposure());

	WriteUpscaledOutput(pos, FfxFloat16x3(cr.x, cg.x, cb.x)); //TODO: fix type
	pos.x += 8;
	WriteUpscaledOutput(pos, FfxFloat16x3(cr.y, cg.y, cb.y)); //TODO: fix type
}

#else

void CurrFilter(FFX_MIN16_U2 pos)
{
	FfxFloat32x3 c;
	FsrRcasF(c.r, c.g, c.b, pos, RCASConfig());

	c = UnprepareRgb(c, Exposure());

	WriteUpscaledOutput(pos, c);
}

#endif // #if FSR_RCAS_PREFER_PAIRED_VERSION
// GODOT ENDS

void RCAS(FfxUInt32x3 LocalThreadId, FfxUInt32x3 WorkGroupId, FfxUInt32x3 Dtid)
{
    // Do remapping of local xy in workgroup for a more PS-like swizzle pattern.
    FfxUInt32x2 gxy = ffxRemapForQuad(LocalThreadId.x) + FfxUInt32x2(WorkGroupId.x << 4u, WorkGroupId.y << 4u);
#if FSR_RCAS_PREFER_PAIRED_VERSION
    CurrFilterPaired(FFX_MIN16_U2(gxy));
    gxy.y += 8u;
    CurrFilterPaired(FFX_MIN16_U2(gxy));
#else
    CurrFilter(FFX_MIN16_U2(gxy));
    gxy.x += 8u;
    CurrFilter(FFX_MIN16_U2(gxy));
    gxy.y += 8u;
    CurrFilter(FFX_MIN16_U2(gxy));
    gxy.x -= 8u;
    CurrFilter(FFX_MIN16_U2(gxy));
#endif
}
