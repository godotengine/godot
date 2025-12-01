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

#ifndef FFX_OPTICALFLOW_PREPARE_LUMA_H
#define FFX_OPTICALFLOW_PREPARE_LUMA_H

FfxFloat32 LuminanceToPerceivedLuminance(FfxFloat32 fLuminance)
{
    FfxFloat32 fPercievedLuminance = 0;
    if (fLuminance <= 216.0f / 24389.0f) {
        fPercievedLuminance = fLuminance * (24389.0f / 27.0f);
    }
    else {
        fPercievedLuminance = ffxPow(fLuminance, 1.0f / 3.0f) * 116.0f - 16.0f;
    }

    return fPercievedLuminance * 0.01f;
}

FfxFloat32 LinearLdrToLuminance(FfxFloat32x3 linearRec709RGB)
{
    FfxFloat32 fY = 0.2126 * linearRec709RGB.x + 0.7152 * linearRec709RGB.y + 0.0722 * linearRec709RGB.z;
    return fY;
}

FfxFloat32 LinearRec2020ToLuminance(FfxFloat32x3 linearRec2020RGB)
{
    FfxFloat32 fY = 0.2627 * linearRec2020RGB.x + 0.678 * linearRec2020RGB.y + 0.0593 * linearRec2020RGB.z;
    return fY;
}

FfxFloat32 PQCorrectedHdrToLuminance(FfxFloat32x3 pq, FfxFloat32 maxLuminance)
{
    FfxFloat32 fY = LinearRec2020ToLuminance(ffxLinearFromPQ(pq) * (10000.0f / maxLuminance));
    return fY;
}

FfxFloat32x3 ffxscRGBToLinear(FfxFloat32x3 value, FfxFloat32 minLuminance, FfxFloat32 maxLuminance)
{
    FfxFloat32x3 p = value - ffxBroadcast3(minLuminance / 80.0f);
    return p / ffxBroadcast3((maxLuminance - minLuminance) / 80.0f);
}

FfxFloat32 SCRGBCorrectedHdrToLuminance(FfxFloat32x3 scRGB, FfxFloat32 minLuminance, FfxFloat32 maxLuminance)
{
    FfxFloat32 fY = LinearLdrToLuminance(ffxscRGBToLinear(scRGB, minLuminance, maxLuminance));
    return fY;
}

void PrepareLuma(FfxInt32x2 iGlobalId, FfxInt32 iLocalIndex)
{
#define PixelsPerThreadX 2
#define PixelsPerThreadY 2
#pragma unroll
    for (FfxInt32 y = 0; y < PixelsPerThreadY; y++)
    {
#pragma unroll
        for (FfxInt32 x = 0; x < PixelsPerThreadX; x++)
        {
            FfxInt32x2 pos = iGlobalId * FfxInt32x2(PixelsPerThreadX, PixelsPerThreadY) + FfxInt32x2(x, y);
            FfxInt32x2 iPxHrPos = pos;
            FfxFloat32 fY = 0.0;

            FfxFloat32x3 inputColor = LoadInputColor(iPxHrPos).rgb;

            FfxUInt32 backbufferTransferFunction = BackbufferTransferFunction();
            if (backbufferTransferFunction == 0)
            {
                fY = LinearLdrToLuminance(inputColor);
            }
            else if (backbufferTransferFunction == 1)
            {
                fY = PQCorrectedHdrToLuminance(inputColor, MinMaxLuminance()[1]);
                fY = LuminanceToPerceivedLuminance(fY);
            }
            else if (backbufferTransferFunction == 2)
            {
                fY = SCRGBCorrectedHdrToLuminance(inputColor, MinMaxLuminance()[0], MinMaxLuminance()[1]);
                fY = LuminanceToPerceivedLuminance(fY);
            }

            StoreOpticalFlowInput(pos, FfxUInt32(fY * 255));
        }
    }
}

#endif // FFX_OPTICALFLOW_PREPARE_LUMA_H
