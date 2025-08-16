// This file is part of the FidelityFX SDK.
//
// Copyright (c) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
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

#define USE_YCOCG 1

#define fAutogenEpsilon 0.01f

// EXPERIMENTAL

FFX_MIN16_F ComputeAutoTC_01(FFX_MIN16_I2 uDispatchThreadId, FFX_MIN16_I2 iPrevIdx)
{
    FfxFloat32x3 colorPreAlpha = LoadOpaqueOnly(uDispatchThreadId);
    FfxFloat32x3 colorPostAlpha = LoadInputColor(uDispatchThreadId);
    FfxFloat32x3 colorPrevPreAlpha = LoadPrevPreAlpha(iPrevIdx);
    FfxFloat32x3 colorPrevPostAlpha = LoadPrevPostAlpha(iPrevIdx);

#if USE_YCOCG    
    colorPreAlpha = RGBToYCoCg(colorPreAlpha);
    colorPostAlpha = RGBToYCoCg(colorPostAlpha);
    colorPrevPreAlpha = RGBToYCoCg(colorPrevPreAlpha);
    colorPrevPostAlpha = RGBToYCoCg(colorPrevPostAlpha);
#endif

    FfxFloat32x3 colorDeltaCurr = colorPostAlpha - colorPreAlpha;
    FfxFloat32x3 colorDeltaPrev = colorPrevPostAlpha - colorPrevPreAlpha;
    bool hasAlpha = any(FFX_GREATER_THAN(abs(colorDeltaCurr), FfxFloat32x3(fAutogenEpsilon, fAutogenEpsilon, fAutogenEpsilon)));
    bool hadAlpha = any(FFX_GREATER_THAN(abs(colorDeltaPrev), FfxFloat32x3(fAutogenEpsilon, fAutogenEpsilon, fAutogenEpsilon)));

    FfxFloat32x3 X = colorPreAlpha;
    FfxFloat32x3 Y = colorPostAlpha;
    FfxFloat32x3 Z = colorPrevPreAlpha;
    FfxFloat32x3 W = colorPrevPostAlpha;

    FFX_MIN16_F retVal = FFX_MIN16_F(ffxSaturate(dot(abs(abs(Y - X) - abs(W - Z)), FfxFloat32x3(1, 1, 1))));

    // cleanup very small values
    retVal = (retVal < getTcThreshold()) ? FFX_MIN16_F(0.0f) : FFX_MIN16_F(1.f);

    return retVal;
}

// works ok: thin edges
FFX_MIN16_F ComputeAutoTC_02(FFX_MIN16_I2 uDispatchThreadId, FFX_MIN16_I2 iPrevIdx)
{
    FfxFloat32x3 colorPreAlpha = LoadOpaqueOnly(uDispatchThreadId);
    FfxFloat32x3 colorPostAlpha = LoadInputColor(uDispatchThreadId);
    FfxFloat32x3 colorPrevPreAlpha = LoadPrevPreAlpha(iPrevIdx);
    FfxFloat32x3 colorPrevPostAlpha = LoadPrevPostAlpha(iPrevIdx);

#if USE_YCOCG    
    colorPreAlpha = RGBToYCoCg(colorPreAlpha);
    colorPostAlpha = RGBToYCoCg(colorPostAlpha);
    colorPrevPreAlpha = RGBToYCoCg(colorPrevPreAlpha);
    colorPrevPostAlpha = RGBToYCoCg(colorPrevPostAlpha);
#endif

    FfxFloat32x3 colorDelta = colorPostAlpha - colorPreAlpha;
    FfxFloat32x3 colorPrevDelta = colorPrevPostAlpha - colorPrevPreAlpha;
    bool hasAlpha = any(FFX_GREATER_THAN(abs(colorDelta), FfxFloat32x3(fAutogenEpsilon, fAutogenEpsilon, fAutogenEpsilon)));
    bool hadAlpha = any(FFX_GREATER_THAN(abs(colorPrevDelta), FfxFloat32x3(fAutogenEpsilon, fAutogenEpsilon, fAutogenEpsilon)));

    FfxFloat32x3 delta = colorPostAlpha - colorPreAlpha;              //prev+1*d = post   => d = color, alpha =
    FfxFloat32x3 deltaPrev = colorPrevPostAlpha - colorPrevPreAlpha;

    FfxFloat32x3 X = colorPrevPreAlpha;
    FfxFloat32x3 N = colorPreAlpha - colorPrevPreAlpha;
    FfxFloat32x3 YAminusXA = colorPrevPostAlpha - colorPrevPreAlpha;
    FfxFloat32x3 NminusNA = colorPostAlpha - colorPrevPostAlpha;

    FfxFloat32x3 A = (hasAlpha || hadAlpha) ? NminusNA / max(FfxFloat32x3(fAutogenEpsilon, fAutogenEpsilon, fAutogenEpsilon), N) : FfxFloat32x3(0, 0, 0);

    FFX_MIN16_F retVal = FFX_MIN16_F( max(max(A.x, A.y), A.z) );

    // only pixels that have significantly changed in color shuold be considered
    retVal = ffxSaturate(retVal * FFX_MIN16_F(length(colorPostAlpha - colorPrevPostAlpha)) );

    return retVal;
}

// This function computes the TransparencyAndComposition mask:
// This mask indicates pixels that should discard locks and apply color clamping.
// 
// Typically this is the case for translucent pixels (that don't write depth values) or pixels where the correctness of 
// the MVs can not be guaranteed (e.g. procedutal movement or vegetation that does not have MVs to reduce the cost during rasterization)
// Also, large changes in color due to changed lighting should be marked to remove locks on pixels with "old" lighting.
//
// This function takes a opaque only and a final texture and uses internal copies of those textures from the last frame.
// The function tries to determine where the color changes between opaque only and final image to determine the pixels that use transparency.
// Also it uses the previous frames and detects where the use of transparency changed to mark those pixels.
// Additionally it marks pixels where the color changed significantly in the opaque only image, e.g. due to lighting or texture animation.
// 
// In the final step it stores the current textures in internal textures for the next frame

FFX_MIN16_F ComputeTransparencyAndComposition(FFX_MIN16_I2 uDispatchThreadId, FFX_MIN16_I2 iPrevIdx)
{
    FFX_MIN16_F retVal = ComputeAutoTC_02(uDispatchThreadId, iPrevIdx);

    // [branch]
    if (retVal > FFX_MIN16_F(0.01f))
    {
        retVal = ComputeAutoTC_01(uDispatchThreadId, iPrevIdx);
    }
    return retVal;
}

float computeSolidEdge(FFX_MIN16_I2 curPos, FFX_MIN16_I2 prevPos)
{
    float lum[9];
    int i = 0;
    for (int y = -1; y < 2; ++y)
    {
        for (int x = -1; x < 2; ++x)
        {
            FfxFloat32x3 curCol  = LoadOpaqueOnly(curPos + FFX_MIN16_I2(x, y)).rgb;
            FfxFloat32x3 prevCol = LoadPrevPreAlpha(prevPos + FFX_MIN16_I2(x, y)).rgb;
            lum[i++] = length(curCol - prevCol);
        }
    }

    //float gradX = abs(lum[3] - lum[4]) + abs(lum[5] - lum[4]);
    //float gradY = abs(lum[1] - lum[4]) + abs(lum[7] - lum[4]);

    //return sqrt(gradX * gradX + gradY * gradY);

    float gradX = abs(lum[3] - lum[4]) * abs(lum[5] - lum[4]);
    float gradY = abs(lum[1] - lum[4]) * abs(lum[7] - lum[4]);

    return sqrt(sqrt(gradX * gradY));
}

float computeAlphaEdge(FFX_MIN16_I2 curPos, FFX_MIN16_I2 prevPos)
{
    float lum[9];
    int i = 0;
    for (int y = -1; y < 2; ++y)
    {
        for (int x = -1; x < 2; ++x)
        {
            FfxFloat32x3 curCol  = abs(LoadInputColor(curPos + FFX_MIN16_I2(x, y)).rgb - LoadOpaqueOnly(curPos + FFX_MIN16_I2(x, y)).rgb);
            FfxFloat32x3 prevCol = abs(LoadPrevPostAlpha(prevPos + FFX_MIN16_I2(x, y)).rgb - LoadPrevPreAlpha(prevPos + FFX_MIN16_I2(x, y)).rgb);
            lum[i++] = length(curCol - prevCol);
        }
    }

    //float gradX = abs(lum[3] - lum[4]) + abs(lum[5] - lum[4]);
    //float gradY = abs(lum[1] - lum[4]) + abs(lum[7] - lum[4]);

    //return sqrt(gradX * gradX + gradY * gradY);

    float gradX = abs(lum[3] - lum[4]) * abs(lum[5] - lum[4]);
    float gradY = abs(lum[1] - lum[4]) * abs(lum[7] - lum[4]);

    return sqrt(sqrt(gradX * gradY));
}

FFX_MIN16_F ComputeAabbOverlap(FFX_MIN16_I2 uDispatchThreadId, FFX_MIN16_I2 iPrevIdx)
{
    FFX_MIN16_F retVal = FFX_MIN16_F(0.f);

    FfxFloat32x2 fMotionVector = LoadInputMotionVector(uDispatchThreadId);
    FfxFloat32x3 colorPreAlpha = LoadOpaqueOnly(uDispatchThreadId);
    FfxFloat32x3 colorPostAlpha = LoadInputColor(uDispatchThreadId);
    FfxFloat32x3 colorPrevPreAlpha = LoadPrevPreAlpha(iPrevIdx);
    FfxFloat32x3 colorPrevPostAlpha = LoadPrevPostAlpha(iPrevIdx);

#if USE_YCOCG    
    colorPreAlpha = RGBToYCoCg(colorPreAlpha);
    colorPostAlpha = RGBToYCoCg(colorPostAlpha);
    colorPrevPreAlpha = RGBToYCoCg(colorPrevPreAlpha);
    colorPrevPostAlpha = RGBToYCoCg(colorPrevPostAlpha);
#endif
    FfxFloat32x3 minPrev = FFX_MIN16_F3(+1000.f, +1000.f, +1000.f);
    FfxFloat32x3 maxPrev = FFX_MIN16_F3(-1000.f, -1000.f, -1000.f);
    for (int y = -1; y < 2; ++y)
    {
        for (int x = -1; x < 2; ++x)
        {
            FfxFloat32x3 W = LoadPrevPostAlpha(iPrevIdx + FFX_MIN16_I2(x, y));

#if USE_YCOCG
            W = RGBToYCoCg(W);
#endif
            minPrev = min(minPrev, W);
            maxPrev = max(maxPrev, W);
        }
    }
    // instead of computing the overlap: simply count how many samples are outside
    // set reactive based on that
    FFX_MIN16_F count = FFX_MIN16_F(0.f);
    for (int y = -1; y < 2; ++y)
    {
        for (int x = -1; x < 2; ++x)
        {
            FfxFloat32x3 Y = LoadInputColor(uDispatchThreadId + FFX_MIN16_I2(x, y));

#if USE_YCOCG
            Y = RGBToYCoCg(Y);
#endif
            count += ((Y.x < minPrev.x) || (Y.x > maxPrev.x)) ? FFX_MIN16_F(1.f) : FFX_MIN16_F(0.f);
            count += ((Y.y < minPrev.y) || (Y.y > maxPrev.y)) ? FFX_MIN16_F(1.f) : FFX_MIN16_F(0.f);
            count += ((Y.z < minPrev.z) || (Y.z > maxPrev.z)) ? FFX_MIN16_F(1.f) : FFX_MIN16_F(0.f);
        }
    }
    retVal = count / FFX_MIN16_F(27.f);

    return retVal;
}


// This function computes the Reactive mask:
// We want pixels marked where the alpha portion of the frame changes a lot between neighbours
// Those pixels are expected to change quickly between frames, too. (e.g. small particles, reflections on curved surfaces...)
// As a result history would not be trustworthy.
// On the other hand we don't want pixels marked where pre-alpha has a large differnce, since those would profit from accumulation
// For mirrors we may assume the pre-alpha is pretty uniform color.
// 
// This works well generally, but also marks edge pixels
FFX_MIN16_F ComputeReactive(FFX_MIN16_I2 uDispatchThreadId, FFX_MIN16_I2 iPrevIdx)
{
    // we only get here if alpha has a significant contribution and has changed since last frame.
    FFX_MIN16_F retVal = FFX_MIN16_F(0.f);

    // mark pixels with huge variance in alpha as reactive
    FFX_MIN16_F alphaEdge = FFX_MIN16_F(computeAlphaEdge(uDispatchThreadId, iPrevIdx));
    FFX_MIN16_F opaqueEdge = FFX_MIN16_F(computeSolidEdge(uDispatchThreadId, iPrevIdx));
    retVal = ffxSaturate(alphaEdge - opaqueEdge);

    // the above also marks edge pixels due to jitter, so we need to cancel those out


    return retVal;
}
