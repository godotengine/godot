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

FFX_STATIC const FfxInt32 s_MipLevelsToUse = 3;

struct ShadingChangeLumaInfo
{
    FfxFloat32 fSamples[s_MipLevelsToUse];
};

ShadingChangeLumaInfo ComputeShadingChangeLuma(FfxInt32x2 iPxPos, FfxFloat32x2 fUv, const FfxInt32x2 iCurrentSize)
{
    ShadingChangeLumaInfo info;

    const FfxFloat32x2 fMipUv = ClampUv(fUv, ShadingChangeRenderSize(), GetSPDMipDimensions(0));

    FFX_UNROLL
    for (FfxInt32 iMipLevel = iShadingChangeMipStart; iMipLevel < s_MipLevelsToUse; iMipLevel++) {

        const FfxFloat32x2 fSample = SampleSPDMipLevel(fMipUv, iMipLevel);

        info.fSamples[iMipLevel] = abs(fSample.x * fSample.y);
    }

    return info;
}

void ShadingChange(FfxInt32x2 iPxPos)
{
    if (IsOnScreen(FfxInt32x2(iPxPos), ShadingChangeRenderSize())) {

        const FfxFloat32x2 fUv = (iPxPos + 0.5f) / ShadingChangeRenderSize();
        const FfxFloat32x2 fUvJittered = fUv + Jitter() / RenderSize();

        const ShadingChangeLumaInfo info = ComputeShadingChangeLuma(iPxPos, fUvJittered, ShadingChangeRenderSize());

        const FfxFloat32 fScale = 1.0f + iShadingChangeMipStart / s_MipLevelsToUse;
        FfxFloat32 fShadingChange = 0.0f;
        FFX_UNROLL
        for (int iMipLevel = iShadingChangeMipStart; iMipLevel < s_MipLevelsToUse; iMipLevel++)
        {
            if (info.fSamples[iMipLevel] > 0) {
                fShadingChange = ffxMax(fShadingChange, info.fSamples[iMipLevel]) * fScale;
            }
        }
        
        StoreShadingChange(iPxPos, ffxSaturate(fShadingChange));
    }
}
