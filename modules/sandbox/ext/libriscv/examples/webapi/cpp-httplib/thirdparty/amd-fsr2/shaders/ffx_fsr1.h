// This file is part of the FidelityFX SDK.
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#ifdef __clang__
#pragma clang diagnostic ignored "-Wunused-variable"
#endif

/// Setup required constant values for EASU (works on CPU or GPU).
///
/// @param [out] con0
/// @param [out] con1
/// @param [out] con2
/// @param [out] con3
/// @param [in] inputViewportInPixelsX                  The rendered image resolution being upscaled in X dimension.
/// @param [in] inputViewportInPixelsY                  The rendered image resolution being upscaled in Y dimension.
/// @param [in] inputSizeInPixelsX                      The resolution of the resource containing the input image (useful for dynamic resolution) in X dimension.
/// @param [in] inputSizeInPixelsY                      The resolution of the resource containing the input image (useful for dynamic resolution) in Y dimension.
/// @param [in] outputSizeInPixelsX                     The display resolution which the input image gets upscaled to in X dimension.
/// @param [in] outputSizeInPixelsY                     The display resolution which the input image gets upscaled to in Y dimension.
/// 
/// @ingroup FSR1
FFX_STATIC void ffxFsrPopulateEasuConstants(
    FFX_PARAMETER_INOUT FfxUInt32x4 con0,
    FFX_PARAMETER_INOUT FfxUInt32x4 con1,
    FFX_PARAMETER_INOUT FfxUInt32x4 con2,
    FFX_PARAMETER_INOUT FfxUInt32x4 con3,
    FFX_PARAMETER_IN FfxFloat32 inputViewportInPixelsX,
    FFX_PARAMETER_IN FfxFloat32 inputViewportInPixelsY,
    FFX_PARAMETER_IN FfxFloat32 inputSizeInPixelsX,
    FFX_PARAMETER_IN FfxFloat32 inputSizeInPixelsY,
    FFX_PARAMETER_IN FfxFloat32 outputSizeInPixelsX,
    FFX_PARAMETER_IN FfxFloat32 outputSizeInPixelsY)
{
    // Output integer position to a pixel position in viewport.
    con0[0] = ffxAsUInt32(inputViewportInPixelsX * ffxReciprocal(outputSizeInPixelsX));
    con0[1] = ffxAsUInt32(inputViewportInPixelsY * ffxReciprocal(outputSizeInPixelsY));
    con0[2] = ffxAsUInt32(FfxFloat32(0.5) * inputViewportInPixelsX * ffxReciprocal(outputSizeInPixelsX) - FfxFloat32(0.5));
    con0[3] = ffxAsUInt32(FfxFloat32(0.5) * inputViewportInPixelsY * ffxReciprocal(outputSizeInPixelsY) - FfxFloat32(0.5));

    // Viewport pixel position to normalized image space.
    // This is used to get upper-left of 'F' tap.
    con1[0] = ffxAsUInt32(ffxReciprocal(inputSizeInPixelsX));
    con1[1] = ffxAsUInt32(ffxReciprocal(inputSizeInPixelsY));

    // Centers of gather4, first offset from upper-left of 'F'.
    //      +---+---+
    //      |   |   |
    //      +--(0)--+
    //      | b | c |
    //  +---F---+---+---+
    //  | e | f | g | h |
    //  +--(1)--+--(2)--+
    //  | i | j | k | l |
    //  +---+---+---+---+
    //      | n | o |
    //      +--(3)--+
    //      |   |   |
    //      +---+---+
    con1[2] = ffxAsUInt32(FfxFloat32(1.0) * ffxReciprocal(inputSizeInPixelsX));
    con1[3] = ffxAsUInt32(FfxFloat32(-1.0) * ffxReciprocal(inputSizeInPixelsY));

    // These are from (0) instead of 'F'.
    con2[0] = ffxAsUInt32(FfxFloat32(-1.0) * ffxReciprocal(inputSizeInPixelsX));
    con2[1] = ffxAsUInt32(FfxFloat32(2.0) * ffxReciprocal(inputSizeInPixelsY));
    con2[2] = ffxAsUInt32(FfxFloat32(1.0) * ffxReciprocal(inputSizeInPixelsX));
    con2[3] = ffxAsUInt32(FfxFloat32(2.0) * ffxReciprocal(inputSizeInPixelsY));
    con3[0] = ffxAsUInt32(FfxFloat32(0.0) * ffxReciprocal(inputSizeInPixelsX));
    con3[1] = ffxAsUInt32(FfxFloat32(4.0) * ffxReciprocal(inputSizeInPixelsY));
    con3[2] = con3[3] = 0;
}

/// Setup required constant values for EASU (works on CPU or GPU).
///
/// @param [out] con0
/// @param [out] con1
/// @param [out] con2
/// @param [out] con3
/// @param [in] inputViewportInPixelsX              The resolution of the input in the X dimension.
/// @param [in] inputViewportInPixelsY              The resolution of the input in the Y dimension.
/// @param [in] inputSizeInPixelsX                  The input size in pixels in the X dimension.
/// @param [in] inputSizeInPixelsY                  The input size in pixels in the Y dimension.
/// @param [in] outputSizeInPixelsX                 The output size in pixels in the X dimension.
/// @param [in] outputSizeInPixelsY                 The output size in pixels in the Y dimension.
/// @param [in] inputOffsetInPixelsX                The input image offset in the X dimension into the resource containing it (useful for dynamic resolution).
/// @param [in] inputOffsetInPixelsY                The input image offset in the Y dimension into the resource containing it (useful for dynamic resolution).
///
/// @ingroup FSR1
FFX_STATIC void ffxFsrPopulateEasuConstantsOffset(
    FFX_PARAMETER_INOUT FfxUInt32x4 con0,
    FFX_PARAMETER_INOUT FfxUInt32x4 con1,
    FFX_PARAMETER_INOUT FfxUInt32x4 con2,
    FFX_PARAMETER_INOUT FfxUInt32x4 con3,
    FFX_PARAMETER_IN FfxFloat32 inputViewportInPixelsX,
    FFX_PARAMETER_IN FfxFloat32 inputViewportInPixelsY,
    FFX_PARAMETER_IN FfxFloat32 inputSizeInPixelsX,
    FFX_PARAMETER_IN FfxFloat32 inputSizeInPixelsY,
    FFX_PARAMETER_IN FfxFloat32 outputSizeInPixelsX,
    FFX_PARAMETER_IN FfxFloat32 outputSizeInPixelsY,
    FFX_PARAMETER_IN FfxFloat32 inputOffsetInPixelsX,
    FFX_PARAMETER_IN FfxFloat32 inputOffsetInPixelsY)
{
    ffxFsrPopulateEasuConstants(
        con0,
        con1,
        con2,
        con3,
        inputViewportInPixelsX,
        inputViewportInPixelsY,
        inputSizeInPixelsX,
        inputSizeInPixelsY,
        outputSizeInPixelsX,
        outputSizeInPixelsY);

    // override 
    con0[2] = ffxAsUInt32(FfxFloat32(0.5) * inputViewportInPixelsX * ffxReciprocal(outputSizeInPixelsX) - FfxFloat32(0.5) + inputOffsetInPixelsX);
    con0[3] = ffxAsUInt32(FfxFloat32(0.5) * inputViewportInPixelsY * ffxReciprocal(outputSizeInPixelsY) - FfxFloat32(0.5) + inputOffsetInPixelsY);
}

#if defined(FFX_GPU) && defined(FFX_FSR_EASU_FLOAT)
// Input callback prototypes, need to be implemented by calling shader
FfxFloat32x4 FsrEasuRF(FfxFloat32x2 p);
FfxFloat32x4 FsrEasuGF(FfxFloat32x2 p);
FfxFloat32x4 FsrEasuBF(FfxFloat32x2 p);

// Filtering for a given tap for the scalar.
void fsrEasuTapFloat(
    FFX_PARAMETER_INOUT FfxFloat32x3 accumulatedColor,   // Accumulated color, with negative lobe.
    FFX_PARAMETER_INOUT FfxFloat32 accumulatedWeight,    // Accumulated weight.
    FFX_PARAMETER_IN FfxFloat32x2 pixelOffset,           // Pixel offset from resolve position to tap.
    FFX_PARAMETER_IN FfxFloat32x2 gradientDirection,     // Gradient direction.
    FFX_PARAMETER_IN FfxFloat32x2 length,                // Length.
    FFX_PARAMETER_IN FfxFloat32 negativeLobeStrength,    // Negative lobe strength.
    FFX_PARAMETER_IN FfxFloat32 clippingPoint,           // Clipping point.
    FFX_PARAMETER_IN FfxFloat32x3 color)                 // Tap color.
{
    // Rotate offset by direction.
    FfxFloat32x2 rotatedOffset;
    rotatedOffset.x = (pixelOffset.x * (gradientDirection.x)) + (pixelOffset.y * gradientDirection.y);
    rotatedOffset.y = (pixelOffset.x * (-gradientDirection.y)) + (pixelOffset.y * gradientDirection.x);

    // Anisotropy.
    rotatedOffset *= length;

    // Compute distance^2.
    FfxFloat32 distanceSquared = rotatedOffset.x * rotatedOffset.x + rotatedOffset.y * rotatedOffset.y;

    // Limit to the window as at corner, 2 taps can easily be outside.
    distanceSquared = ffxMin(distanceSquared, clippingPoint);

    // Approximation of lancos2 without sin() or rcp(), or sqrt() to get x.
    //  (25/16 * (2/5 * x^2 - 1)^2 - (25/16 - 1)) * (1/4 * x^2 - 1)^2
    //  |_______________________________________|   |_______________|
    //                   base                             window
    // The general form of the 'base' is,
    //  (a*(b*x^2-1)^2-(a-1))
    // Where 'a=1/(2*b-b^2)' and 'b' moves around the negative lobe.
    FfxFloat32 weightB = FfxFloat32(2.0 / 5.0) * distanceSquared + FfxFloat32(-1.0);
    FfxFloat32 weightA = negativeLobeStrength * distanceSquared + FfxFloat32(-1.0);
    weightB *= weightB;
    weightA *= weightA;
    weightB = FfxFloat32(25.0 / 16.0) * weightB + FfxFloat32(-(25.0 / 16.0 - 1.0));
    FfxFloat32 weight = weightB * weightA;

    // Do weighted average.
    accumulatedColor += color * weight;
    accumulatedWeight += weight;
}

// Accumulate direction and length.
void fsrEasuSetFloat(
    FFX_PARAMETER_INOUT FfxFloat32x2 direction,
    FFX_PARAMETER_INOUT FfxFloat32 length,
    FFX_PARAMETER_IN FfxFloat32x2 pp,
    FFX_PARAMETER_IN FfxBoolean biS,
    FFX_PARAMETER_IN FfxBoolean biT,
    FFX_PARAMETER_IN FfxBoolean biU,
    FFX_PARAMETER_IN FfxBoolean biV,
    FFX_PARAMETER_IN FfxFloat32 lA,
    FFX_PARAMETER_IN FfxFloat32 lB,
    FFX_PARAMETER_IN FfxFloat32 lC,
    FFX_PARAMETER_IN FfxFloat32 lD,
    FFX_PARAMETER_IN FfxFloat32 lE)
{
    // Compute bilinear weight, branches factor out as predicates are compiler time immediates.
    //  s t
    //  u v
    FfxFloat32 weight = FfxFloat32(0.0);
    if (biS)
        weight = (FfxFloat32(1.0) - pp.x) * (FfxFloat32(1.0) - pp.y);
    if (biT)
        weight = pp.x * (FfxFloat32(1.0) - pp.y);
    if (biU)
        weight = (FfxFloat32(1.0) - pp.x) * pp.y;
    if (biV)
        weight = pp.x * pp.y;

    // Direction is the '+' diff.
    //    a
    //  b c d
    //    e
    // Then takes magnitude from abs average of both sides of 'c'.
    // Length converts gradient reversal to 0, smoothly to non-reversal at 1, shaped, then adding horz and vert terms.
    FfxFloat32 dc = lD - lC;
    FfxFloat32 cb = lC - lB;
    FfxFloat32 lengthX = max(abs(dc), abs(cb));
    lengthX = ffxApproximateReciprocal(lengthX);
    FfxFloat32 directionX = lD - lB;
    direction.x += directionX * weight;
    lengthX = ffxSaturate(abs(directionX) * lengthX);
    lengthX *= lengthX;
    length += lengthX * weight;

    // Repeat for the y axis.
    FfxFloat32 ec = lE - lC;
    FfxFloat32 ca = lC - lA;
    FfxFloat32 lengthY = max(abs(ec), abs(ca));
    lengthY = ffxApproximateReciprocal(lengthY);
    FfxFloat32 directionY = lE - lA;
    direction.y += directionY * weight;
    lengthY = ffxSaturate(abs(directionY) * lengthY);
    lengthY *= lengthY;
    length += lengthY * weight;
}

/// Apply edge-aware spatial upsampling using 32bit floating point precision calculations.
///
/// @param [out] outPixel               The computed color of a pixel.
/// @param [in]  integerPosition        Integer pixel position within the output.
/// @param [in]  con0                   The first constant value generated by <c><i>ffxFsrPopulateEasuConstants</i></c>.
/// @param [in]  con1                   The second constant value generated by <c><i>ffxFsrPopulateEasuConstants</i></c>.
/// @param [in]  con2                   The third constant value generated by <c><i>ffxFsrPopulateEasuConstants</i></c>.
/// @param [in]  con3                   The fourth constant value generated by <c><i>ffxFsrPopulateEasuConstants</i></c>.
/// 
/// @ingroup FSR
void ffxFsrEasuFloat(
    FFX_PARAMETER_OUT FfxFloat32x3 pix,
    FFX_PARAMETER_IN FfxUInt32x2 ip,
    FFX_PARAMETER_IN FfxUInt32x4 con0,
    FFX_PARAMETER_IN FfxUInt32x4 con1,
    FFX_PARAMETER_IN FfxUInt32x4 con2,
    FFX_PARAMETER_IN FfxUInt32x4 con3)
{
    // Get position of 'f'.
    FfxFloat32x2 pp = FfxFloat32x2(ip) * ffxAsFloat(con0.xy) + ffxAsFloat(con0.zw);
    FfxFloat32x2 fp = floor(pp);
    pp -= fp;

    // 12-tap kernel.
    //    b c
    //  e f g h
    //  i j k l
    //    n o
    // Gather 4 ordering.
    //  a b
    //  r g
    // For packed FP16, need either {rg} or {ab} so using the following setup for gather in all versions,
    //    a b    <- unused (z)
    //    r g
    //  a b a b
    //  r g r g
    //    a b
    //    r g    <- unused (z)
    // Allowing dead-code removal to remove the 'z's.
    FfxFloat32x2 p0 = fp * ffxAsFloat(con1.xy) + ffxAsFloat(con1.zw);

    // These are from p0 to avoid pulling two constants on pre-Navi hardware.
    FfxFloat32x2 p1    = p0 + ffxAsFloat(con2.xy);
    FfxFloat32x2 p2    = p0 + ffxAsFloat(con2.zw);
    FfxFloat32x2 p3    = p0 + ffxAsFloat(con3.xy);
    FfxFloat32x4 bczzR = FsrEasuRF(p0);
    FfxFloat32x4 bczzG = FsrEasuGF(p0);
    FfxFloat32x4 bczzB = FsrEasuBF(p0);
    FfxFloat32x4 ijfeR = FsrEasuRF(p1);
    FfxFloat32x4 ijfeG = FsrEasuGF(p1);
    FfxFloat32x4 ijfeB = FsrEasuBF(p1);
    FfxFloat32x4 klhgR = FsrEasuRF(p2);
    FfxFloat32x4 klhgG = FsrEasuGF(p2);
    FfxFloat32x4 klhgB = FsrEasuBF(p2);
    FfxFloat32x4 zzonR = FsrEasuRF(p3);
    FfxFloat32x4 zzonG = FsrEasuGF(p3);
    FfxFloat32x4 zzonB = FsrEasuBF(p3);

    // Simplest multi-channel approximate luma possible (luma times 2, in 2 FMA/MAD).
    FfxFloat32x4 bczzL = bczzB * ffxBroadcast4(0.5) + (bczzR * ffxBroadcast4(0.5) + bczzG);
    FfxFloat32x4 ijfeL = ijfeB * ffxBroadcast4(0.5) + (ijfeR * ffxBroadcast4(0.5) + ijfeG);
    FfxFloat32x4 klhgL = klhgB * ffxBroadcast4(0.5) + (klhgR * ffxBroadcast4(0.5) + klhgG);
    FfxFloat32x4 zzonL = zzonB * ffxBroadcast4(0.5) + (zzonR * ffxBroadcast4(0.5) + zzonG);

    // Rename.
    FfxFloat32 bL = bczzL.x;
    FfxFloat32 cL = bczzL.y;
    FfxFloat32 iL = ijfeL.x;
    FfxFloat32 jL = ijfeL.y;
    FfxFloat32 fL = ijfeL.z;
    FfxFloat32 eL = ijfeL.w;
    FfxFloat32 kL = klhgL.x;
    FfxFloat32 lL = klhgL.y;
    FfxFloat32 hL = klhgL.z;
    FfxFloat32 gL = klhgL.w;
    FfxFloat32 oL = zzonL.z;
    FfxFloat32 nL = zzonL.w;

    // Accumulate for bilinear interpolation.
    FfxFloat32x2 dir = ffxBroadcast2(0.0);
    FfxFloat32  len = FfxFloat32(0.0);
    fsrEasuSetFloat(dir, len, pp, FFX_TRUE,  FFX_FALSE, FFX_FALSE, FFX_FALSE, bL, eL, fL, gL, jL);
    fsrEasuSetFloat(dir, len, pp, FFX_FALSE, FFX_TRUE,  FFX_FALSE, FFX_FALSE, cL, fL, gL, hL, kL);
    fsrEasuSetFloat(dir, len, pp, FFX_FALSE, FFX_FALSE, FFX_TRUE,  FFX_FALSE, fL, iL, jL, kL, nL);
    fsrEasuSetFloat(dir, len, pp, FFX_FALSE, FFX_FALSE, FFX_FALSE, FFX_TRUE,  gL, jL, kL, lL, oL);

    // Normalize with approximation, and cleanup close to zero.
    FfxFloat32x2 dir2 = dir * dir;
    FfxFloat32 dirR = dir2.x + dir2.y;
    FfxUInt32 zro  = dirR < FfxFloat32(1.0 / 32768.0);
    dirR = ffxApproximateReciprocalSquareRoot(dirR);
    dirR = zro ? FfxFloat32(1.0) : dirR;
    dir.x = zro ? FfxFloat32(1.0) : dir.x;
    dir *= ffxBroadcast2(dirR);

    // Transform from {0 to 2} to {0 to 1} range, and shape with square.
    len = len * FfxFloat32(0.5);
    len *= len;

    // Stretch kernel {1.0 vert|horz, to sqrt(2.0) on diagonal}.
    FfxFloat32 stretch = (dir.x * dir.x + dir.y * dir.y) * ffxApproximateReciprocal(max(abs(dir.x), abs(dir.y)));

    // Anisotropic length after rotation,
    //  x := 1.0 lerp to 'stretch' on edges
    //  y := 1.0 lerp to 2x on edges
    FfxFloat32x2 len2 = FfxFloat32x2(FfxFloat32(1.0) + (stretch - FfxFloat32(1.0)) * len, FfxFloat32(1.0) + FfxFloat32(-0.5) * len);

    // Based on the amount of 'edge',
    // the window shifts from +/-{sqrt(2.0) to slightly beyond 2.0}.
    FfxFloat32 lob = FfxFloat32(0.5) + FfxFloat32((1.0 / 4.0 - 0.04) - 0.5) * len;

    // Set distance^2 clipping point to the end of the adjustable window.
    FfxFloat32 clp = ffxApproximateReciprocal(lob);

    // Accumulation mixed with min/max of 4 nearest.
    //    b c
    //  e f g h
    //  i j k l
    //    n o
    FfxFloat32x3 min4 =
        ffxMin(ffxMin3(FfxFloat32x3(ijfeR.z, ijfeG.z, ijfeB.z), FfxFloat32x3(klhgR.w, klhgG.w, klhgB.w), FfxFloat32x3(ijfeR.y, ijfeG.y, ijfeB.y)),
               FfxFloat32x3(klhgR.x, klhgG.x, klhgB.x));
    FfxFloat32x3 max4 =
        max(ffxMax3(FfxFloat32x3(ijfeR.z, ijfeG.z, ijfeB.z), FfxFloat32x3(klhgR.w, klhgG.w, klhgB.w), FfxFloat32x3(ijfeR.y, ijfeG.y, ijfeB.y)), FfxFloat32x3(klhgR.x, klhgG.x, klhgB.x));

    // Accumulation.
    FfxFloat32x3 aC = ffxBroadcast3(0.0);
    FfxFloat32  aW = FfxFloat32(0.0);
    fsrEasuTapFloat(aC, aW, FfxFloat32x2(0.0, -1.0) - pp, dir, len2, lob, clp, FfxFloat32x3(bczzR.x, bczzG.x, bczzB.x));  // b
    fsrEasuTapFloat(aC, aW, FfxFloat32x2(1.0, -1.0) - pp, dir, len2, lob, clp, FfxFloat32x3(bczzR.y, bczzG.y, bczzB.y));  // c
    fsrEasuTapFloat(aC, aW, FfxFloat32x2(-1.0, 1.0) - pp, dir, len2, lob, clp, FfxFloat32x3(ijfeR.x, ijfeG.x, ijfeB.x));  // i
    fsrEasuTapFloat(aC, aW, FfxFloat32x2(0.0, 1.0) - pp, dir, len2, lob, clp, FfxFloat32x3(ijfeR.y, ijfeG.y, ijfeB.y));   // j
    fsrEasuTapFloat(aC, aW, FfxFloat32x2(0.0, 0.0) - pp, dir, len2, lob, clp, FfxFloat32x3(ijfeR.z, ijfeG.z, ijfeB.z));   // f
    fsrEasuTapFloat(aC, aW, FfxFloat32x2(-1.0, 0.0) - pp, dir, len2, lob, clp, FfxFloat32x3(ijfeR.w, ijfeG.w, ijfeB.w));  // e
    fsrEasuTapFloat(aC, aW, FfxFloat32x2(1.0, 1.0) - pp, dir, len2, lob, clp, FfxFloat32x3(klhgR.x, klhgG.x, klhgB.x));   // k
    fsrEasuTapFloat(aC, aW, FfxFloat32x2(2.0, 1.0) - pp, dir, len2, lob, clp, FfxFloat32x3(klhgR.y, klhgG.y, klhgB.y));   // l
    fsrEasuTapFloat(aC, aW, FfxFloat32x2(2.0, 0.0) - pp, dir, len2, lob, clp, FfxFloat32x3(klhgR.z, klhgG.z, klhgB.z));   // h
    fsrEasuTapFloat(aC, aW, FfxFloat32x2(1.0, 0.0) - pp, dir, len2, lob, clp, FfxFloat32x3(klhgR.w, klhgG.w, klhgB.w));   // g
    fsrEasuTapFloat(aC, aW, FfxFloat32x2(1.0, 2.0) - pp, dir, len2, lob, clp, FfxFloat32x3(zzonR.z, zzonG.z, zzonB.z));   // o
    fsrEasuTapFloat(aC, aW, FfxFloat32x2(0.0, 2.0) - pp, dir, len2, lob, clp, FfxFloat32x3(zzonR.w, zzonG.w, zzonB.w));   // n

    // Normalize and dering.
    pix = ffxMin(max4, max(min4, aC * ffxBroadcast3(rcp(aW))));
}
#endif // #if defined(FFX_GPU) && defined(FFX_FSR_EASU_FLOAT)

#if defined(FFX_GPU) && FFX_HALF == 1 && defined(FFX_FSR_EASU_HALF)
// Input callback prototypes, need to be implemented by calling shader
FfxFloat16x4 FsrEasuRH(FfxFloat32x2 p);
FfxFloat16x4 FsrEasuGH(FfxFloat32x2 p);
FfxFloat16x4 FsrEasuBH(FfxFloat32x2 p);

// This runs 2 taps in parallel.
void FsrEasuTapH(
    FFX_PARAMETER_INOUT FfxFloat16x2 aCR,
    FFX_PARAMETER_INOUT FfxFloat16x2 aCG,
    FFX_PARAMETER_INOUT FfxFloat16x2 aCB,
    FFX_PARAMETER_INOUT FfxFloat16x2 aW,
    FFX_PARAMETER_IN FfxFloat16x2 offX,
    FFX_PARAMETER_IN FfxFloat16x2 offY,
    FFX_PARAMETER_IN FfxFloat16x2 dir,
    FFX_PARAMETER_IN FfxFloat16x2 len,
    FFX_PARAMETER_IN FfxFloat16 lob,
    FFX_PARAMETER_IN FfxFloat16 clp,
    FFX_PARAMETER_IN FfxFloat16x2 cR,
    FFX_PARAMETER_IN FfxFloat16x2 cG,
    FFX_PARAMETER_IN FfxFloat16x2 cB)
{
    FfxFloat16x2 vX, vY;
    vX = offX * dir.xx + offY * dir.yy;
    vY = offX * (-dir.yy) + offY * dir.xx;
    vX *= len.x;
    vY *= len.y;
    FfxFloat16x2 d2 = vX * vX + vY * vY;
    d2              = min(d2, FFX_BROADCAST_FLOAT16X2(clp));
    FfxFloat16x2 wB = FFX_BROADCAST_FLOAT16X2(2.0 / 5.0) * d2 + FFX_BROADCAST_FLOAT16X2(-1.0);
    FfxFloat16x2 wA = FFX_BROADCAST_FLOAT16X2(lob) * d2 + FFX_BROADCAST_FLOAT16X2(-1.0);
    wB *= wB;
    wA *= wA;
    wB             = FFX_BROADCAST_FLOAT16X2(25.0 / 16.0) * wB + FFX_BROADCAST_FLOAT16X2(-(25.0 / 16.0 - 1.0));
    FfxFloat16x2 w = wB * wA;
    aCR += cR * w;
    aCG += cG * w;
    aCB += cB * w;
    aW += w;
}

// This runs 2 taps in parallel.
void FsrEasuSetH(
    FFX_PARAMETER_INOUT FfxFloat16x2 dirPX,
    FFX_PARAMETER_INOUT FfxFloat16x2  dirPY,
    FFX_PARAMETER_INOUT FfxFloat16x2 lenP,
    FFX_PARAMETER_IN FfxFloat16x2 pp,
    FFX_PARAMETER_IN FfxBoolean biST,
    FFX_PARAMETER_IN FfxBoolean biUV,
    FFX_PARAMETER_IN FfxFloat16x2 lA,
    FFX_PARAMETER_IN FfxFloat16x2 lB,
    FFX_PARAMETER_IN FfxFloat16x2 lC,
    FFX_PARAMETER_IN FfxFloat16x2 lD,
    FFX_PARAMETER_IN FfxFloat16x2 lE)
{
    FfxFloat16x2 w = FFX_BROADCAST_FLOAT16X2(0.0);
    
    if (biST)
        w = (FfxFloat16x2(1.0, 0.0) + FfxFloat16x2(-pp.x, pp.x)) * FFX_BROADCAST_FLOAT16X2(FFX_BROADCAST_FLOAT16(1.0) - pp.y);

    if (biUV)
        w = (FfxFloat16x2(1.0, 0.0) + FfxFloat16x2(-pp.x, pp.x)) * FFX_BROADCAST_FLOAT16X2(pp.y);

    // ABS is not free in the packed FP16 path.
    FfxFloat16x2 dc   = lD - lC;
    FfxFloat16x2 cb   = lC - lB;
    FfxFloat16x2 lenX = max(abs(dc), abs(cb));
    lenX              = ffxReciprocalHalf(lenX);

    FfxFloat16x2 dirX = lD - lB;
    dirPX += dirX * w;
    lenX = ffxSaturate(abs(dirX) * lenX);
    lenX *= lenX;
    lenP += lenX * w;
    FfxFloat16x2 ec   = lE - lC;
    FfxFloat16x2 ca   = lC - lA;
    FfxFloat16x2 lenY = max(abs(ec), abs(ca));
    lenY              = ffxReciprocalHalf(lenY);
    FfxFloat16x2 dirY = lE - lA;
    dirPY += dirY * w;
    lenY = ffxSaturate(abs(dirY) * lenY);
    lenY *= lenY;
    lenP += lenY * w;
}

void FsrEasuH(
    FFX_PARAMETER_OUT FfxFloat16x3 pix, 
    FFX_PARAMETER_IN FfxUInt32x2 ip,
    FFX_PARAMETER_IN FfxUInt32x4 con0,
    FFX_PARAMETER_IN FfxUInt32x4 con1,
    FFX_PARAMETER_IN FfxUInt32x4 con2,
    FFX_PARAMETER_IN FfxUInt32x4 con3)
{
    FfxFloat32x2 pp = FfxFloat32x2(ip) * ffxAsFloat(con0.xy) + ffxAsFloat(con0.zw);
    FfxFloat32x2 fp = floor(pp);
    pp -= fp;
    FfxFloat16x2 ppp = FfxFloat16x2(pp);

    FfxFloat32x2 p0    = fp * ffxAsFloat(con1.xy) + ffxAsFloat(con1.zw);
    FfxFloat32x2 p1    = p0 + ffxAsFloat(con2.xy);
    FfxFloat32x2 p2    = p0 + ffxAsFloat(con2.zw);
    FfxFloat32x2 p3    = p0 + ffxAsFloat(con3.xy);
    FfxFloat16x4 bczzR = FsrEasuRH(p0);
    FfxFloat16x4 bczzG = FsrEasuGH(p0);
    FfxFloat16x4 bczzB = FsrEasuBH(p0);
    FfxFloat16x4 ijfeR = FsrEasuRH(p1);
    FfxFloat16x4 ijfeG = FsrEasuGH(p1);
    FfxFloat16x4 ijfeB = FsrEasuBH(p1);
    FfxFloat16x4 klhgR = FsrEasuRH(p2);
    FfxFloat16x4 klhgG = FsrEasuGH(p2);
    FfxFloat16x4 klhgB = FsrEasuBH(p2);
    FfxFloat16x4 zzonR = FsrEasuRH(p3);
    FfxFloat16x4 zzonG = FsrEasuGH(p3);
    FfxFloat16x4 zzonB = FsrEasuBH(p3);

    FfxFloat16x4 bczzL = bczzB * FFX_BROADCAST_FLOAT16X4(0.5) + (bczzR * FFX_BROADCAST_FLOAT16X4(0.5) + bczzG);
    FfxFloat16x4 ijfeL = ijfeB * FFX_BROADCAST_FLOAT16X4(0.5) + (ijfeR * FFX_BROADCAST_FLOAT16X4(0.5) + ijfeG);
    FfxFloat16x4 klhgL = klhgB * FFX_BROADCAST_FLOAT16X4(0.5) + (klhgR * FFX_BROADCAST_FLOAT16X4(0.5) + klhgG);
    FfxFloat16x4 zzonL = zzonB * FFX_BROADCAST_FLOAT16X4(0.5) + (zzonR * FFX_BROADCAST_FLOAT16X4(0.5) + zzonG);
    FfxFloat16   bL    = bczzL.x;
    FfxFloat16   cL    = bczzL.y;
    FfxFloat16   iL    = ijfeL.x;
    FfxFloat16   jL    = ijfeL.y;
    FfxFloat16   fL    = ijfeL.z;
    FfxFloat16   eL    = ijfeL.w;
    FfxFloat16   kL    = klhgL.x;
    FfxFloat16   lL    = klhgL.y;
    FfxFloat16   hL    = klhgL.z;
    FfxFloat16   gL    = klhgL.w;
    FfxFloat16   oL    = zzonL.z;
    FfxFloat16   nL    = zzonL.w;

    // This part is different, accumulating 2 taps in parallel.
    FfxFloat16x2 dirPX = FFX_BROADCAST_FLOAT16X2(0.0);
    FfxFloat16x2 dirPY = FFX_BROADCAST_FLOAT16X2(0.0);
    FfxFloat16x2 lenP  = FFX_BROADCAST_FLOAT16X2(0.0);
    FsrEasuSetH(dirPX,
                dirPY,
                lenP,
                ppp,
                FfxUInt32(true),
                FfxUInt32(false),
                FfxFloat16x2(bL, cL),
                FfxFloat16x2(eL, fL),
                FfxFloat16x2(fL, gL),
                FfxFloat16x2(gL, hL),
                FfxFloat16x2(jL, kL));
    FsrEasuSetH(dirPX,
                dirPY,
                lenP,
                ppp,
                FfxUInt32(false),
                FfxUInt32(true),
                FfxFloat16x2(fL, gL),
                FfxFloat16x2(iL, jL),
                FfxFloat16x2(jL, kL),
                FfxFloat16x2(kL, lL),
                FfxFloat16x2(nL, oL));
    FfxFloat16x2 dir = FfxFloat16x2(dirPX.r + dirPX.g, dirPY.r + dirPY.g);
    FfxFloat16   len = lenP.r + lenP.g;

    FfxFloat16x2 dir2 = dir * dir;
    FfxFloat16   dirR = dir2.x + dir2.y;
    FfxBoolean   zro  = FfxBoolean(dirR < FFX_BROADCAST_FLOAT16(1.0 / 32768.0));
    dirR              = ffxApproximateReciprocalSquareRootHalf(dirR);
    dirR              = (zro > 0) ? FFX_BROADCAST_FLOAT16(1.0) : dirR;
    dir.x             = (zro > 0) ? FFX_BROADCAST_FLOAT16(1.0) : dir.x;
    dir *= FFX_BROADCAST_FLOAT16X2(dirR);
    len = len * FFX_BROADCAST_FLOAT16(0.5);
    len *= len;
    FfxFloat16   stretch = (dir.x * dir.x + dir.y * dir.y) * ffxApproximateReciprocalHalf(max(abs(dir.x), abs(dir.y)));
    FfxFloat16x2 len2 =
        FfxFloat16x2(FFX_BROADCAST_FLOAT16(1.0) + (stretch - FFX_BROADCAST_FLOAT16(1.0)) * len, FFX_BROADCAST_FLOAT16(1.0) + FFX_BROADCAST_FLOAT16(-0.5) * len);
    FfxFloat16 lob = FFX_BROADCAST_FLOAT16(0.5) + FFX_BROADCAST_FLOAT16((1.0 / 4.0 - 0.04) - 0.5) * len;
    FfxFloat16 clp = ffxApproximateReciprocalHalf(lob);

    // FP16 is different, using packed trick to do min and max in same operation.
    FfxFloat16x2 bothR =
        max(max(FfxFloat16x2(-ijfeR.z, ijfeR.z), FfxFloat16x2(-klhgR.w, klhgR.w)), max(FfxFloat16x2(-ijfeR.y, ijfeR.y), FfxFloat16x2(-klhgR.x, klhgR.x)));
    FfxFloat16x2 bothG =
        max(max(FfxFloat16x2(-ijfeG.z, ijfeG.z), FfxFloat16x2(-klhgG.w, klhgG.w)), max(FfxFloat16x2(-ijfeG.y, ijfeG.y), FfxFloat16x2(-klhgG.x, klhgG.x)));
    FfxFloat16x2 bothB =
        max(max(FfxFloat16x2(-ijfeB.z, ijfeB.z), FfxFloat16x2(-klhgB.w, klhgB.w)), max(FfxFloat16x2(-ijfeB.y, ijfeB.y), FfxFloat16x2(-klhgB.x, klhgB.x)));

    // This part is different for FP16, working pairs of taps at a time.
    FfxFloat16x2 pR = FFX_BROADCAST_FLOAT16X2(0.0);
    FfxFloat16x2 pG = FFX_BROADCAST_FLOAT16X2(0.0);
    FfxFloat16x2 pB = FFX_BROADCAST_FLOAT16X2(0.0);
    FfxFloat16x2 pW = FFX_BROADCAST_FLOAT16X2(0.0);
    FsrEasuTapH(pR, pG, pB, pW, FfxFloat16x2(0.0, 1.0) - ppp.xx, FfxFloat16x2(-1.0, -1.0) - ppp.yy, dir, len2, lob, clp, bczzR.xy, bczzG.xy, bczzB.xy);
    FsrEasuTapH(pR, pG, pB, pW, FfxFloat16x2(-1.0, 0.0) - ppp.xx, FfxFloat16x2(1.0, 1.0) - ppp.yy, dir, len2, lob, clp, ijfeR.xy, ijfeG.xy, ijfeB.xy);
    FsrEasuTapH(pR, pG, pB, pW, FfxFloat16x2(0.0, -1.0) - ppp.xx, FfxFloat16x2(0.0, 0.0) - ppp.yy, dir, len2, lob, clp, ijfeR.zw, ijfeG.zw, ijfeB.zw);
    FsrEasuTapH(pR, pG, pB, pW, FfxFloat16x2(1.0, 2.0) - ppp.xx, FfxFloat16x2(1.0, 1.0) - ppp.yy, dir, len2, lob, clp, klhgR.xy, klhgG.xy, klhgB.xy);
    FsrEasuTapH(pR, pG, pB, pW, FfxFloat16x2(2.0, 1.0) - ppp.xx, FfxFloat16x2(0.0, 0.0) - ppp.yy, dir, len2, lob, clp, klhgR.zw, klhgG.zw, klhgB.zw);
    FsrEasuTapH(pR, pG, pB, pW, FfxFloat16x2(1.0, 0.0) - ppp.xx, FfxFloat16x2(2.0, 2.0) - ppp.yy, dir, len2, lob, clp, zzonR.zw, zzonG.zw, zzonB.zw);
    FfxFloat16x3 aC = FfxFloat16x3(pR.x + pR.y, pG.x + pG.y, pB.x + pB.y);
    FfxFloat16   aW = pW.x + pW.y;

    // Slightly different for FP16 version due to combined min and max.
    pix = min(FfxFloat16x3(bothR.y, bothG.y, bothB.y), max(-FfxFloat16x3(bothR.x, bothG.x, bothB.x), aC * FFX_BROADCAST_FLOAT16X3(ffxReciprocalHalf(aW))));
}
#endif // #if defined(FFX_GPU) && defined(FFX_HALF) && defined(FFX_FSR_EASU_HALF)

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//_____________________________________________________________/\_______________________________________________________________
//==============================================================================================================================
//
//                                      FSR - [RCAS] ROBUST CONTRAST ADAPTIVE SHARPENING
//
//------------------------------------------------------------------------------------------------------------------------------
// CAS uses a simplified mechanism to convert local contrast into a variable amount of sharpness.
// RCAS uses a more exact mechanism, solving for the maximum local sharpness possible before clipping.
// RCAS also has a built in process to limit sharpening of what it detects as possible noise.
// RCAS sharper does not support scaling, as it should be applied after EASU scaling.
// Pass EASU output straight into RCAS, no color conversions necessary.
//------------------------------------------------------------------------------------------------------------------------------
// RCAS is based on the following logic.
// RCAS uses a 5 tap filter in a cross pattern (same as CAS),
//    w                n
//  w 1 w  for taps  w m e 
//    w                s
// Where 'w' is the negative lobe weight.
//  output = (w*(n+e+w+s)+m)/(4*w+1)
// RCAS solves for 'w' by seeing where the signal might clip out of the {0 to 1} input range,
//  0 == (w*(n+e+w+s)+m)/(4*w+1) -> w = -m/(n+e+w+s)
//  1 == (w*(n+e+w+s)+m)/(4*w+1) -> w = (1-m)/(n+e+w+s-4*1)
// Then chooses the 'w' which results in no clipping, limits 'w', and multiplies by the 'sharp' amount.
// This solution above has issues with MSAA input as the steps along the gradient cause edge detection issues.
// So RCAS uses 4x the maximum and 4x the minimum (depending on equation)in place of the individual taps.
// As well as switching from 'm' to either the minimum or maximum (depending on side), to help in energy conservation.
// This stabilizes RCAS.
// RCAS does a simple highpass which is normalized against the local contrast then shaped,
//       0.25
//  0.25  -1  0.25
//       0.25
// This is used as a noise detection filter, to reduce the effect of RCAS on grain, and focus on real edges.
//
//  GLSL example for the required callbacks :
// 
//  FfxFloat16x4 FsrRcasLoadH(FfxInt16x2 p){return FfxFloat16x4(imageLoad(imgSrc,FfxInt32x2(p)));}
//  void FsrRcasInputH(inout FfxFloat16 r,inout FfxFloat16 g,inout FfxFloat16 b)
//  {
//    //do any simple input color conversions here or leave empty if none needed
//  }
//  
//  FsrRcasCon need to be called from the CPU or GPU to set up constants.
//  Including a GPU example here, the 'con' value would be stored out to a constant buffer.
// 
//  FfxUInt32x4 con;
//  FsrRcasCon(con,
//   0.0); // The scale is {0.0 := maximum sharpness, to N>0, where N is the number of stops (halving) of the reduction of sharpness}.
// ---------------
// RCAS sharpening supports a CAS-like pass-through alpha via,
//  #define FSR_RCAS_PASSTHROUGH_ALPHA 1
// RCAS also supports a define to enable a more expensive path to avoid some sharpening of noise.
// Would suggest it is better to apply film grain after RCAS sharpening (and after scaling) instead of using this define,
//  #define FSR_RCAS_DENOISE 1
//==============================================================================================================================
// This is set at the limit of providing unnatural results for sharpening.
#define FSR_RCAS_LIMIT (0.25-(1.0/16.0))
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//_____________________________________________________________/\_______________________________________________________________
//==============================================================================================================================
//                                                      CONSTANT SETUP
//==============================================================================================================================
// Call to setup required constant values (works on CPU or GPU).
 FFX_STATIC void FsrRcasCon(FfxUInt32x4 con,
                            // The scale is {0.0 := maximum, to N>0, where N is the number of stops (halving) of the reduction of sharpness}.
                            FfxFloat32 sharpness)
 {
     // Transform from stops to linear value.
     sharpness = exp2(-sharpness);
     FfxFloat32x2 hSharp  = {sharpness, sharpness};
     con[0] = ffxAsUInt32(sharpness);
     con[1] = packHalf2x16(hSharp);
     con[2] = 0;
     con[3] = 0;
 }
 ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//_____________________________________________________________/\_______________________________________________________________
//==============================================================================================================================
//                                                   NON-PACKED 32-BIT VERSION
//==============================================================================================================================
#if defined(FFX_GPU)&&defined(FSR_RCAS_F)
 // Input callback prototypes that need to be implemented by calling shader
 FfxFloat32x4 FsrRcasLoadF(FfxInt32x2 p);
 void FsrRcasInputF(inout FfxFloat32 r,inout FfxFloat32 g,inout FfxFloat32 b);
//------------------------------------------------------------------------------------------------------------------------------
 void FsrRcasF(out FfxFloat32 pixR,  // Output values, non-vector so port between RcasFilter() and RcasFilterH() is easy.
               out FfxFloat32 pixG,
               out FfxFloat32 pixB,
#ifdef FSR_RCAS_PASSTHROUGH_ALPHA
               out FfxFloat32 pixA,
#endif
               FfxUInt32x2 ip,  // Integer pixel position in output.
               FfxUInt32x4 con)
 {  // Constant generated by RcasSetup().
     // Algorithm uses minimal 3x3 pixel neighborhood.
     //    b
     //  d e f
     //    h
     FfxInt32x2   sp = FfxInt32x2(ip);
     FfxFloat32x3 b  = FsrRcasLoadF(sp + FfxInt32x2(0, -1)).rgb;
     FfxFloat32x3 d  = FsrRcasLoadF(sp + FfxInt32x2(-1, 0)).rgb;
#ifdef FSR_RCAS_PASSTHROUGH_ALPHA
     FfxFloat32x4 ee = FsrRcasLoadF(sp);
     FfxFloat32x3 e  = ee.rgb;
     pixA            = ee.a;
#else
     FfxFloat32x3 e = FsrRcasLoadF(sp).rgb;
#endif
     FfxFloat32x3 f = FsrRcasLoadF(sp + FfxInt32x2(1, 0)).rgb;
     FfxFloat32x3 h = FsrRcasLoadF(sp + FfxInt32x2(0, 1)).rgb;
     // Rename (32-bit) or regroup (16-bit).
     FfxFloat32 bR = b.r;
     FfxFloat32 bG = b.g;
     FfxFloat32 bB = b.b;
     FfxFloat32 dR = d.r;
     FfxFloat32 dG = d.g;
     FfxFloat32 dB = d.b;
     FfxFloat32 eR = e.r;
     FfxFloat32 eG = e.g;
     FfxFloat32 eB = e.b;
     FfxFloat32 fR = f.r;
     FfxFloat32 fG = f.g;
     FfxFloat32 fB = f.b;
     FfxFloat32 hR = h.r;
     FfxFloat32 hG = h.g;
     FfxFloat32 hB = h.b;
     // Run optional input transform.
     FsrRcasInputF(bR, bG, bB);
     FsrRcasInputF(dR, dG, dB);
     FsrRcasInputF(eR, eG, eB);
     FsrRcasInputF(fR, fG, fB);
     FsrRcasInputF(hR, hG, hB);
     // Luma times 2.
     FfxFloat32 bL = bB * FfxFloat32(0.5) + (bR * FfxFloat32(0.5) + bG);
     FfxFloat32 dL = dB * FfxFloat32(0.5) + (dR * FfxFloat32(0.5) + dG);
     FfxFloat32 eL = eB * FfxFloat32(0.5) + (eR * FfxFloat32(0.5) + eG);
     FfxFloat32 fL = fB * FfxFloat32(0.5) + (fR * FfxFloat32(0.5) + fG);
     FfxFloat32 hL = hB * FfxFloat32(0.5) + (hR * FfxFloat32(0.5) + hG);
     // Noise detection.
     FfxFloat32 nz = FfxFloat32(0.25) * bL + FfxFloat32(0.25) * dL + FfxFloat32(0.25) * fL + FfxFloat32(0.25) * hL - eL;
     nz            = ffxSaturate(abs(nz) * ffxApproximateReciprocalMedium(ffxMax3(ffxMax3(bL, dL, eL), fL, hL) - ffxMin3(ffxMin3(bL, dL, eL), fL, hL)));
     nz            = FfxFloat32(-0.5) * nz + FfxFloat32(1.0);
     // Min and max of ring.
     FfxFloat32 mn4R = ffxMin(ffxMin3(bR, dR, fR), hR);
     FfxFloat32 mn4G = ffxMin(ffxMin3(bG, dG, fG), hG);
     FfxFloat32 mn4B = ffxMin(ffxMin3(bB, dB, fB), hB);
     FfxFloat32 mx4R = max(ffxMax3(bR, dR, fR), hR);
     FfxFloat32 mx4G = max(ffxMax3(bG, dG, fG), hG);
     FfxFloat32 mx4B = max(ffxMax3(bB, dB, fB), hB);
     // Immediate constants for peak range.
     FfxFloat32x2 peakC = FfxFloat32x2(1.0, -1.0 * 4.0);
     // Limiters, these need to be high precision RCPs.
     FfxFloat32 hitMinR = mn4R * rcp(FfxFloat32(4.0) * mx4R);
     FfxFloat32 hitMinG = mn4G * rcp(FfxFloat32(4.0) * mx4G);
     FfxFloat32 hitMinB = mn4B * rcp(FfxFloat32(4.0) * mx4B);
     FfxFloat32 hitMaxR = (peakC.x - mx4R) * rcp(FfxFloat32(4.0) * mn4R + peakC.y);
     FfxFloat32 hitMaxG = (peakC.x - mx4G) * rcp(FfxFloat32(4.0) * mn4G + peakC.y);
     FfxFloat32 hitMaxB = (peakC.x - mx4B) * rcp(FfxFloat32(4.0) * mn4B + peakC.y);
     FfxFloat32 lobeR   = max(-hitMinR, hitMaxR);
     FfxFloat32 lobeG   = max(-hitMinG, hitMaxG);
     FfxFloat32 lobeB   = max(-hitMinB, hitMaxB);
     FfxFloat32 lobe    = max(FfxFloat32(-FSR_RCAS_LIMIT), ffxMin(ffxMax3(lobeR, lobeG, lobeB), FfxFloat32(0.0))) * ffxAsFloat
     (con.x);
 // Apply noise removal.
#ifdef FSR_RCAS_DENOISE
     lobe *= nz;
#endif
     // Resolve, which needs the medium precision rcp approximation to avoid visible tonality changes.
     FfxFloat32 rcpL = ffxApproximateReciprocalMedium(FfxFloat32(4.0) * lobe + FfxFloat32(1.0));
     pixR            = (lobe * bR + lobe * dR + lobe * hR + lobe * fR + eR) * rcpL;
     pixG            = (lobe * bG + lobe * dG + lobe * hG + lobe * fG + eG) * rcpL;
     pixB            = (lobe * bB + lobe * dB + lobe * hB + lobe * fB + eB) * rcpL;
     return;
 }
#endif
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//_____________________________________________________________/\_______________________________________________________________
//==============================================================================================================================
//                                                  NON-PACKED 16-BIT VERSION
//==============================================================================================================================
#if defined(FFX_GPU) && FFX_HALF == 1 && defined(FSR_RCAS_H)
 // Input callback prototypes that need to be implemented by calling shader
 FfxFloat16x4 FsrRcasLoadH(FfxInt16x2 p);
 void FsrRcasInputH(inout FfxFloat16 r,inout FfxFloat16 g,inout FfxFloat16 b);
//------------------------------------------------------------------------------------------------------------------------------
 void FsrRcasH(
 out FfxFloat16 pixR, // Output values, non-vector so port between RcasFilter() and RcasFilterH() is easy.
 out FfxFloat16 pixG,
 out FfxFloat16 pixB,
 #ifdef FSR_RCAS_PASSTHROUGH_ALPHA
  out FfxFloat16 pixA,
 #endif
 FfxUInt32x2 ip, // Integer pixel position in output.
 FfxUInt32x4 con){ // Constant generated by RcasSetup().
  // Sharpening algorithm uses minimal 3x3 pixel neighborhood.
  //    b 
  //  d e f
  //    h
  FfxInt16x2 sp=FfxInt16x2(ip);
  FfxFloat16x3 b=FsrRcasLoadH(sp+FfxInt16x2( 0,-1)).rgb;
  FfxFloat16x3 d=FsrRcasLoadH(sp+FfxInt16x2(-1, 0)).rgb;
  #ifdef FSR_RCAS_PASSTHROUGH_ALPHA
   FfxFloat16x4 ee=FsrRcasLoadH(sp);
   FfxFloat16x3 e=ee.rgb;pixA=ee.a;
  #else
   FfxFloat16x3 e=FsrRcasLoadH(sp).rgb;
  #endif
  FfxFloat16x3 f=FsrRcasLoadH(sp+FfxInt16x2( 1, 0)).rgb;
  FfxFloat16x3 h=FsrRcasLoadH(sp+FfxInt16x2( 0, 1)).rgb;
  // Rename (32-bit) or regroup (16-bit).
  FfxFloat16 bR=b.r;
  FfxFloat16 bG=b.g;
  FfxFloat16 bB=b.b;
  FfxFloat16 dR=d.r;
  FfxFloat16 dG=d.g;
  FfxFloat16 dB=d.b;
  FfxFloat16 eR=e.r;
  FfxFloat16 eG=e.g;
  FfxFloat16 eB=e.b;
  FfxFloat16 fR=f.r;
  FfxFloat16 fG=f.g;
  FfxFloat16 fB=f.b;
  FfxFloat16 hR=h.r;
  FfxFloat16 hG=h.g;
  FfxFloat16 hB=h.b;
  // Run optional input transform.
  FsrRcasInputH(bR,bG,bB);
  FsrRcasInputH(dR,dG,dB);
  FsrRcasInputH(eR,eG,eB);
  FsrRcasInputH(fR,fG,fB);
  FsrRcasInputH(hR,hG,hB);
  // Luma times 2.
  FfxFloat16 bL=bB*FFX_BROADCAST_FLOAT16(0.5)+(bR*FFX_BROADCAST_FLOAT16(0.5)+bG);
  FfxFloat16 dL=dB*FFX_BROADCAST_FLOAT16(0.5)+(dR*FFX_BROADCAST_FLOAT16(0.5)+dG);
  FfxFloat16 eL=eB*FFX_BROADCAST_FLOAT16(0.5)+(eR*FFX_BROADCAST_FLOAT16(0.5)+eG);
  FfxFloat16 fL=fB*FFX_BROADCAST_FLOAT16(0.5)+(fR*FFX_BROADCAST_FLOAT16(0.5)+fG);
  FfxFloat16 hL=hB*FFX_BROADCAST_FLOAT16(0.5)+(hR*FFX_BROADCAST_FLOAT16(0.5)+hG);
  // Noise detection.
  FfxFloat16 nz=FFX_BROADCAST_FLOAT16(0.25)*bL+FFX_BROADCAST_FLOAT16(0.25)*dL+FFX_BROADCAST_FLOAT16(0.25)*fL+FFX_BROADCAST_FLOAT16(0.25)*hL-eL;
  nz=ffxSaturate(abs(nz)*ffxApproximateReciprocalMediumHalf(ffxMax3Half(ffxMax3Half(bL,dL,eL),fL,hL)-ffxMin3Half(ffxMin3Half(bL,dL,eL),fL,hL)));
  nz=FFX_BROADCAST_FLOAT16(-0.5)*nz+FFX_BROADCAST_FLOAT16(1.0);
  // Min and max of ring.
  FfxFloat16 mn4R=min(ffxMin3Half(bR,dR,fR),hR);
  FfxFloat16 mn4G=min(ffxMin3Half(bG,dG,fG),hG);
  FfxFloat16 mn4B=min(ffxMin3Half(bB,dB,fB),hB);
  FfxFloat16 mx4R=max(ffxMax3Half(bR,dR,fR),hR);
  FfxFloat16 mx4G=max(ffxMax3Half(bG,dG,fG),hG);
  FfxFloat16 mx4B=max(ffxMax3Half(bB,dB,fB),hB);
  // Immediate constants for peak range.
  FfxFloat16x2 peakC=FfxFloat16x2(1.0,-1.0*4.0);
  // Limiters, these need to be high precision RCPs.
  FfxFloat16 hitMinR=mn4R*ffxReciprocalHalf(FFX_BROADCAST_FLOAT16(4.0)*mx4R);
  FfxFloat16 hitMinG=mn4G*ffxReciprocalHalf(FFX_BROADCAST_FLOAT16(4.0)*mx4G);
  FfxFloat16 hitMinB=mn4B*ffxReciprocalHalf(FFX_BROADCAST_FLOAT16(4.0)*mx4B);
  FfxFloat16 hitMaxR=(peakC.x-mx4R)*ffxReciprocalHalf(FFX_BROADCAST_FLOAT16(4.0)*mn4R+peakC.y);
  FfxFloat16 hitMaxG=(peakC.x-mx4G)*ffxReciprocalHalf(FFX_BROADCAST_FLOAT16(4.0)*mn4G+peakC.y);
  FfxFloat16 hitMaxB=(peakC.x-mx4B)*ffxReciprocalHalf(FFX_BROADCAST_FLOAT16(4.0)*mn4B+peakC.y);
  FfxFloat16 lobeR=max(-hitMinR,hitMaxR);
  FfxFloat16 lobeG=max(-hitMinG,hitMaxG);
  FfxFloat16 lobeB=max(-hitMinB,hitMaxB);
  FfxFloat16 lobe=max(FFX_BROADCAST_FLOAT16(-FSR_RCAS_LIMIT),min(ffxMax3Half(lobeR,lobeG,lobeB),FFX_BROADCAST_FLOAT16(0.0)))*FFX_UINT32_TO_FLOAT16X2(con.y).x;
  // Apply noise removal.
  #ifdef FSR_RCAS_DENOISE
   lobe*=nz;
  #endif
  // Resolve, which needs the medium precision rcp approximation to avoid visible tonality changes.
  FfxFloat16 rcpL=ffxApproximateReciprocalMediumHalf(FFX_BROADCAST_FLOAT16(4.0)*lobe+FFX_BROADCAST_FLOAT16(1.0));
  pixR=(lobe*bR+lobe*dR+lobe*hR+lobe*fR+eR)*rcpL;
  pixG=(lobe*bG+lobe*dG+lobe*hG+lobe*fG+eG)*rcpL;
  pixB=(lobe*bB+lobe*dB+lobe*hB+lobe*fB+eB)*rcpL;
}
#endif
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//_____________________________________________________________/\_______________________________________________________________
//==============================================================================================================================
//                                                     PACKED 16-BIT VERSION
//==============================================================================================================================
#if defined(FFX_GPU)&& FFX_HALF == 1 && defined(FSR_RCAS_HX2)
 // Input callback prototypes that need to be implemented by the calling shader
 FfxFloat16x4 FsrRcasLoadHx2(FfxInt16x2 p);
 void FsrRcasInputHx2(inout FfxFloat16x2 r,inout FfxFloat16x2 g,inout FfxFloat16x2 b);
//------------------------------------------------------------------------------------------------------------------------------
 // Can be used to convert from packed Structures of Arrays to Arrays of Structures for store.
 void FsrRcasDepackHx2(out FfxFloat16x4 pix0,out FfxFloat16x4 pix1,FfxFloat16x2 pixR,FfxFloat16x2 pixG,FfxFloat16x2 pixB){
  #ifdef FFX_HLSL
   // Invoke a slower path for DX only, since it won't allow uninitialized values.
   pix0.a=pix1.a=0.0;
  #endif
  pix0.rgb=FfxFloat16x3(pixR.x,pixG.x,pixB.x);
  pix1.rgb=FfxFloat16x3(pixR.y,pixG.y,pixB.y);}
//------------------------------------------------------------------------------------------------------------------------------
 void FsrRcasHx2(
 // Output values are for 2 8x8 tiles in a 16x8 region.
 //  pix<R,G,B>.x =  left 8x8 tile
 //  pix<R,G,B>.y = right 8x8 tile
 // This enables later processing to easily be packed as well.
 out FfxFloat16x2 pixR,
 out FfxFloat16x2 pixG,
 out FfxFloat16x2 pixB,
 #ifdef FSR_RCAS_PASSTHROUGH_ALPHA
  out FfxFloat16x2 pixA,
 #endif
 FfxUInt32x2 ip, // Integer pixel position in output.
 FfxUInt32x4 con){ // Constant generated by RcasSetup().
  // No scaling algorithm uses minimal 3x3 pixel neighborhood.
  FfxInt16x2 sp0=FfxInt16x2(ip);
  FfxFloat16x3 b0=FsrRcasLoadHx2(sp0+FfxInt16x2( 0,-1)).rgb;
  FfxFloat16x3 d0=FsrRcasLoadHx2(sp0+FfxInt16x2(-1, 0)).rgb;
  #ifdef FSR_RCAS_PASSTHROUGH_ALPHA
   FfxFloat16x4 ee0=FsrRcasLoadHx2(sp0);
   FfxFloat16x3 e0=ee0.rgb;pixA.r=ee0.a;
  #else
   FfxFloat16x3 e0=FsrRcasLoadHx2(sp0).rgb;
  #endif
  FfxFloat16x3 f0=FsrRcasLoadHx2(sp0+FfxInt16x2( 1, 0)).rgb;
  FfxFloat16x3 h0=FsrRcasLoadHx2(sp0+FfxInt16x2( 0, 1)).rgb;
  FfxInt16x2 sp1=sp0+FfxInt16x2(8,0);
  FfxFloat16x3 b1=FsrRcasLoadHx2(sp1+FfxInt16x2( 0,-1)).rgb;
  FfxFloat16x3 d1=FsrRcasLoadHx2(sp1+FfxInt16x2(-1, 0)).rgb;
  #ifdef FSR_RCAS_PASSTHROUGH_ALPHA
   FfxFloat16x4 ee1=FsrRcasLoadHx2(sp1);
   FfxFloat16x3 e1=ee1.rgb;pixA.g=ee1.a;
  #else
   FfxFloat16x3 e1=FsrRcasLoadHx2(sp1).rgb;
  #endif
  FfxFloat16x3 f1=FsrRcasLoadHx2(sp1+FfxInt16x2( 1, 0)).rgb;
  FfxFloat16x3 h1=FsrRcasLoadHx2(sp1+FfxInt16x2( 0, 1)).rgb;
  // Arrays of Structures to Structures of Arrays conversion.
  FfxFloat16x2 bR=FfxFloat16x2(b0.r,b1.r);
  FfxFloat16x2 bG=FfxFloat16x2(b0.g,b1.g);
  FfxFloat16x2 bB=FfxFloat16x2(b0.b,b1.b);
  FfxFloat16x2 dR=FfxFloat16x2(d0.r,d1.r);
  FfxFloat16x2 dG=FfxFloat16x2(d0.g,d1.g);
  FfxFloat16x2 dB=FfxFloat16x2(d0.b,d1.b);
  FfxFloat16x2 eR=FfxFloat16x2(e0.r,e1.r);
  FfxFloat16x2 eG=FfxFloat16x2(e0.g,e1.g);
  FfxFloat16x2 eB=FfxFloat16x2(e0.b,e1.b);
  FfxFloat16x2 fR=FfxFloat16x2(f0.r,f1.r);
  FfxFloat16x2 fG=FfxFloat16x2(f0.g,f1.g);
  FfxFloat16x2 fB=FfxFloat16x2(f0.b,f1.b);
  FfxFloat16x2 hR=FfxFloat16x2(h0.r,h1.r);
  FfxFloat16x2 hG=FfxFloat16x2(h0.g,h1.g);
  FfxFloat16x2 hB=FfxFloat16x2(h0.b,h1.b);
  // Run optional input transform.
  FsrRcasInputHx2(bR,bG,bB);
  FsrRcasInputHx2(dR,dG,dB);
  FsrRcasInputHx2(eR,eG,eB);
  FsrRcasInputHx2(fR,fG,fB);
  FsrRcasInputHx2(hR,hG,hB);
  // Luma times 2.
  FfxFloat16x2 bL=bB*FFX_BROADCAST_FLOAT16X2(0.5)+(bR*FFX_BROADCAST_FLOAT16X2(0.5)+bG);
  FfxFloat16x2 dL=dB*FFX_BROADCAST_FLOAT16X2(0.5)+(dR*FFX_BROADCAST_FLOAT16X2(0.5)+dG);
  FfxFloat16x2 eL=eB*FFX_BROADCAST_FLOAT16X2(0.5)+(eR*FFX_BROADCAST_FLOAT16X2(0.5)+eG);
  FfxFloat16x2 fL=fB*FFX_BROADCAST_FLOAT16X2(0.5)+(fR*FFX_BROADCAST_FLOAT16X2(0.5)+fG);
  FfxFloat16x2 hL=hB*FFX_BROADCAST_FLOAT16X2(0.5)+(hR*FFX_BROADCAST_FLOAT16X2(0.5)+hG);
  // Noise detection.
  FfxFloat16x2 nz=FFX_BROADCAST_FLOAT16X2(0.25)*bL+FFX_BROADCAST_FLOAT16X2(0.25)*dL+FFX_BROADCAST_FLOAT16X2(0.25)*fL+FFX_BROADCAST_FLOAT16X2(0.25)*hL-eL;
  nz=ffxSaturate(abs(nz)*ffxApproximateReciprocalMediumHalf(ffxMax3Half(ffxMax3Half(bL,dL,eL),fL,hL)-ffxMin3Half(ffxMin3Half(bL,dL,eL),fL,hL)));
  nz=FFX_BROADCAST_FLOAT16X2(-0.5)*nz+FFX_BROADCAST_FLOAT16X2(1.0);
  // Min and max of ring.
  FfxFloat16x2 mn4R=min(ffxMin3Half(bR,dR,fR),hR);
  FfxFloat16x2 mn4G=min(ffxMin3Half(bG,dG,fG),hG);
  FfxFloat16x2 mn4B=min(ffxMin3Half(bB,dB,fB),hB);
  FfxFloat16x2 mx4R=max(ffxMax3Half(bR,dR,fR),hR);
  FfxFloat16x2 mx4G=max(ffxMax3Half(bG,dG,fG),hG);
  FfxFloat16x2 mx4B=max(ffxMax3Half(bB,dB,fB),hB);
  // Immediate constants for peak range.
  FfxFloat16x2 peakC=FfxFloat16x2(1.0,-1.0*4.0);
  // Limiters, these need to be high precision RCPs.
  FfxFloat16x2 hitMinR=mn4R*ffxReciprocalHalf(FFX_BROADCAST_FLOAT16X2(4.0)*mx4R);
  FfxFloat16x2 hitMinG=mn4G*ffxReciprocalHalf(FFX_BROADCAST_FLOAT16X2(4.0)*mx4G);
  FfxFloat16x2 hitMinB=mn4B*ffxReciprocalHalf(FFX_BROADCAST_FLOAT16X2(4.0)*mx4B);
  FfxFloat16x2 hitMaxR=(peakC.x-mx4R)*ffxReciprocalHalf(FFX_BROADCAST_FLOAT16X2(4.0)*mn4R+peakC.y);
  FfxFloat16x2 hitMaxG=(peakC.x-mx4G)*ffxReciprocalHalf(FFX_BROADCAST_FLOAT16X2(4.0)*mn4G+peakC.y);
  FfxFloat16x2 hitMaxB=(peakC.x-mx4B)*ffxReciprocalHalf(FFX_BROADCAST_FLOAT16X2(4.0)*mn4B+peakC.y);
  FfxFloat16x2 lobeR=max(-hitMinR,hitMaxR);
  FfxFloat16x2 lobeG=max(-hitMinG,hitMaxG);
  FfxFloat16x2 lobeB=max(-hitMinB,hitMaxB);
  FfxFloat16x2 lobe=max(FFX_BROADCAST_FLOAT16X2(-FSR_RCAS_LIMIT),min(ffxMax3Half(lobeR,lobeG,lobeB),FFX_BROADCAST_FLOAT16X2(0.0)))*FFX_BROADCAST_FLOAT16X2(FFX_UINT32_TO_FLOAT16X2(con.y).x);
  // Apply noise removal.
  #ifdef FSR_RCAS_DENOISE
   lobe*=nz;
  #endif
  // Resolve, which needs the medium precision rcp approximation to avoid visible tonality changes.
  FfxFloat16x2 rcpL=ffxApproximateReciprocalMediumHalf(FFX_BROADCAST_FLOAT16X2(4.0)*lobe+FFX_BROADCAST_FLOAT16X2(1.0));
  pixR=(lobe*bR+lobe*dR+lobe*hR+lobe*fR+eR)*rcpL;
  pixG=(lobe*bG+lobe*dG+lobe*hG+lobe*fG+eG)*rcpL;
  pixB=(lobe*bB+lobe*dB+lobe*hB+lobe*fB+eB)*rcpL;}
#endif
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//_____________________________________________________________/\_______________________________________________________________
//==============================================================================================================================
//
//                                          FSR - [LFGA] LINEAR FILM GRAIN APPLICATOR
//
//------------------------------------------------------------------------------------------------------------------------------
// Adding output-resolution film grain after scaling is a good way to mask both rendering and scaling artifacts.
// Suggest using tiled blue noise as film grain input, with peak noise frequency set for a specific look and feel.
// The 'Lfga*()' functions provide a convenient way to introduce grain.
// These functions limit grain based on distance to signal limits.
// This is done so that the grain is temporally energy preserving, and thus won't modify image tonality.
// Grain application should be done in a linear colorspace.
// The grain should be temporally changing, but have a temporal sum per pixel that adds to zero (non-biased).
//------------------------------------------------------------------------------------------------------------------------------
// Usage,
//   FsrLfga*(
//    color, // In/out linear colorspace color {0 to 1} ranged.
//    grain, // Per pixel grain texture value {-0.5 to 0.5} ranged, input is 3-channel to support colored grain.
//    amount); // Amount of grain (0 to 1} ranged.
//------------------------------------------------------------------------------------------------------------------------------
// Example if grain texture is monochrome: 'FsrLfgaF(color,ffxBroadcast3(grain),amount)'
//==============================================================================================================================
#if defined(FFX_GPU)
 // Maximum grain is the minimum distance to the signal limit.
 void FsrLfgaF(inout FfxFloat32x3 c, FfxFloat32x3 t, FfxFloat32 a)
 {
     c += (t * ffxBroadcast3(a)) * ffxMin(ffxBroadcast3(1.0) - c, c);
 }
#endif
//==============================================================================================================================
#if defined(FFX_GPU)&& FFX_HALF == 1
 // Half precision version (slower).
 void FsrLfgaH(inout FfxFloat16x3 c, FfxFloat16x3 t, FfxFloat16 a)
 {
     c += (t * FFX_BROADCAST_FLOAT16X3(a)) * min(FFX_BROADCAST_FLOAT16X3(1.0) - c, c);
 }
 //------------------------------------------------------------------------------------------------------------------------------
 // Packed half precision version (faster).
 void FsrLfgaHx2(inout FfxFloat16x2 cR,inout FfxFloat16x2 cG,inout FfxFloat16x2 cB,FfxFloat16x2 tR,FfxFloat16x2 tG,FfxFloat16x2 tB,FfxFloat16 a){
  cR+=(tR*FFX_BROADCAST_FLOAT16X2(a))*min(FFX_BROADCAST_FLOAT16X2(1.0)-cR,cR);cG+=(tG*FFX_BROADCAST_FLOAT16X2(a))*min(FFX_BROADCAST_FLOAT16X2(1.0)-cG,cG);cB+=(tB*FFX_BROADCAST_FLOAT16X2(a))*min(FFX_BROADCAST_FLOAT16X2(1.0)-cB,cB);}
#endif
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//_____________________________________________________________/\_______________________________________________________________
//==============================================================================================================================
//
//                                          FSR - [SRTM] SIMPLE REVERSIBLE TONE-MAPPER
//
//------------------------------------------------------------------------------------------------------------------------------
// This provides a way to take linear HDR color {0 to FP16_MAX} and convert it into a temporary {0 to 1} ranged post-tonemapped linear.
// The tonemapper preserves RGB ratio, which helps maintain HDR color bleed during filtering.
//------------------------------------------------------------------------------------------------------------------------------
// Reversible tonemapper usage,
//  FsrSrtm*(color); // {0 to FP16_MAX} converted to {0 to 1}.
//  FsrSrtmInv*(color); // {0 to 1} converted into {0 to 32768, output peak safe for FP16}.
//==============================================================================================================================
#if defined(FFX_GPU)
 void FsrSrtmF(inout FfxFloat32x3 c)
 {
     c *= ffxBroadcast3(rcp(ffxMax3(c.r, c.g, c.b) + FfxFloat32(1.0)));
 }
 // The extra max solves the c=1.0 case (which is a /0).
 void FsrSrtmInvF(inout FfxFloat32x3 c){c*=ffxBroadcast3(rcp(max(FfxFloat32(1.0/32768.0),FfxFloat32(1.0)-ffxMax3(c.r,c.g,c.b))));}
#endif
//==============================================================================================================================
#if defined(FFX_GPU )&& FFX_HALF == 1
 void FsrSrtmH(inout FfxFloat16x3 c)
 {
     c *= FFX_BROADCAST_FLOAT16X3(ffxReciprocalHalf(ffxMax3Half(c.r, c.g, c.b) + FFX_BROADCAST_FLOAT16(1.0)));
 }
 void FsrSrtmInvH(inout FfxFloat16x3 c)
 {
     c *= FFX_BROADCAST_FLOAT16X3(ffxReciprocalHalf(max(FFX_BROADCAST_FLOAT16(1.0 / 32768.0), FFX_BROADCAST_FLOAT16(1.0) - ffxMax3Half(c.r, c.g, c.b))));
 }
 //------------------------------------------------------------------------------------------------------------------------------
 void FsrSrtmHx2(inout FfxFloat16x2 cR, inout FfxFloat16x2 cG, inout FfxFloat16x2 cB)
 {
     FfxFloat16x2 rcp = ffxReciprocalHalf(ffxMax3Half(cR, cG, cB) + FFX_BROADCAST_FLOAT16X2(1.0));
     cR *= rcp;
     cG *= rcp;
     cB *= rcp;
 }
 void FsrSrtmInvHx2(inout FfxFloat16x2 cR,inout FfxFloat16x2 cG,inout FfxFloat16x2 cB)
 {
     FfxFloat16x2 rcp=ffxReciprocalHalf(max(FFX_BROADCAST_FLOAT16X2(1.0/32768.0),FFX_BROADCAST_FLOAT16X2(1.0)-ffxMax3Half(cR,cG,cB)));
     cR*=rcp;
     cG*=rcp;
     cB*=rcp;
 }
#endif
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//_____________________________________________________________/\_______________________________________________________________
//==============================================================================================================================
//
//                                       FSR - [TEPD] TEMPORAL ENERGY PRESERVING DITHER
//
//------------------------------------------------------------------------------------------------------------------------------
// Temporally energy preserving dithered {0 to 1} linear to gamma 2.0 conversion.
// Gamma 2.0 is used so that the conversion back to linear is just to square the color.
// The conversion comes in 8-bit and 10-bit modes, designed for output to 8-bit UNORM or 10:10:10:2 respectively.
// Given good non-biased temporal blue noise as dither input,
// the output dither will temporally conserve energy.
// This is done by choosing the linear nearest step point instead of perceptual nearest.
// See code below for details.
//------------------------------------------------------------------------------------------------------------------------------
// DX SPEC RULES FOR FLOAT->UNORM 8-BIT CONVERSION
// ===============================================
// - Output is 'FfxUInt32(floor(saturate(n)*255.0+0.5))'.
// - Thus rounding is to nearest.
// - NaN gets converted to zero.
// - INF is clamped to {0.0 to 1.0}.
//==============================================================================================================================
#if defined(FFX_GPU)
 // Hand tuned integer position to dither value, with more values than simple checkerboard.
 // Only 32-bit has enough precision for this compddation.
 // Output is {0 to <1}.
 FfxFloat32 FsrTepdDitF(FfxUInt32x2 p, FfxUInt32 f)
 {
     FfxFloat32 x = FfxFloat32(p.x + f);
     FfxFloat32 y = FfxFloat32(p.y);
     // The 1.61803 golden ratio.
     FfxFloat32 a = FfxFloat32((1.0 + ffxSqrt(5.0f)) / 2.0);
     // Number designed to provide a good visual pattern.
     FfxFloat32 b = FfxFloat32(1.0 / 3.69);
     x            = x * a + (y * b);
     return ffxFract(x);
 }
  //------------------------------------------------------------------------------------------------------------------------------
 // This version is 8-bit gamma 2.0.
 // The 'c' input is {0 to 1}.
 // Output is {0 to 1} ready for image store.
 void FsrTepdC8F(inout FfxFloat32x3 c, FfxFloat32 dit)
 {
     FfxFloat32x3 n = ffxSqrt(c);
     n              = floor(n * ffxBroadcast3(255.0)) * ffxBroadcast3(1.0 / 255.0);
     FfxFloat32x3 a = n * n;
     FfxFloat32x3 b = n + ffxBroadcast3(1.0 / 255.0);
     b              = b * b;
     // Ratio of 'a' to 'b' required to produce 'c'.
     // ffxApproximateReciprocal() won't work here (at least for very high dynamic ranges).
     // ffxApproximateReciprocalMedium() is an IADD,FMA,MUL.
     FfxFloat32x3 r = (c - b) * ffxApproximateReciprocalMedium(a - b);
     // Use the ratio as a cutoff to choose 'a' or 'b'.
     // ffxIsGreaterThanZero() is a MUL.
     c = ffxSaturate(n + ffxIsGreaterThanZero(ffxBroadcast3(dit) - r) * ffxBroadcast3(1.0 / 255.0));
 }
 //------------------------------------------------------------------------------------------------------------------------------
 // This version is 10-bit gamma 2.0.
 // The 'c' input is {0 to 1}.
 // Output is {0 to 1} ready for image store.
 void FsrTepdC10F(inout FfxFloat32x3 c, FfxFloat32 dit)
 {
     FfxFloat32x3 n = ffxSqrt(c);
     n              = floor(n * ffxBroadcast3(1023.0)) * ffxBroadcast3(1.0 / 1023.0);
     FfxFloat32x3 a = n * n;
     FfxFloat32x3 b = n + ffxBroadcast3(1.0 / 1023.0);
     b              = b * b;
     FfxFloat32x3 r = (c - b) * ffxApproximateReciprocalMedium(a - b);
     c              = ffxSaturate(n + ffxIsGreaterThanZero(ffxBroadcast3(dit) - r) * ffxBroadcast3(1.0 / 1023.0));
 }
#endif
//==============================================================================================================================
#if defined(FFX_GPU)&& FFX_HALF == 1
 FfxFloat16 FsrTepdDitH(FfxUInt32x2 p, FfxUInt32 f)
 {
     FfxFloat32 x = FfxFloat32(p.x + f);
     FfxFloat32 y = FfxFloat32(p.y);
     FfxFloat32 a = FfxFloat32((1.0 + ffxSqrt(5.0f)) / 2.0);
     FfxFloat32 b = FfxFloat32(1.0 / 3.69);
     x       = x * a + (y * b);
     return FfxFloat16(ffxFract(x));
 }
 //------------------------------------------------------------------------------------------------------------------------------
 void FsrTepdC8H(inout FfxFloat16x3 c, FfxFloat16 dit)
 {
     FfxFloat16x3 n = sqrt(c);
     n     = floor(n * FFX_BROADCAST_FLOAT16X3(255.0)) * FFX_BROADCAST_FLOAT16X3(1.0 / 255.0);
     FfxFloat16x3 a = n * n;
     FfxFloat16x3 b = n + FFX_BROADCAST_FLOAT16X3(1.0 / 255.0);
     b     = b * b;
     FfxFloat16x3 r = (c - b) * ffxApproximateReciprocalMediumHalf(a - b);
     c     = ffxSaturate(n + ffxIsGreaterThanZeroHalf(FFX_BROADCAST_FLOAT16X3(dit) - r) * FFX_BROADCAST_FLOAT16X3(1.0 / 255.0));
 }
 //------------------------------------------------------------------------------------------------------------------------------
 void FsrTepdC10H(inout FfxFloat16x3 c, FfxFloat16 dit)
 {
     FfxFloat16x3 n = sqrt(c);
     n     = floor(n * FFX_BROADCAST_FLOAT16X3(1023.0)) * FFX_BROADCAST_FLOAT16X3(1.0 / 1023.0);
     FfxFloat16x3 a = n * n;
     FfxFloat16x3 b = n + FFX_BROADCAST_FLOAT16X3(1.0 / 1023.0);
     b     = b * b;
     FfxFloat16x3 r = (c - b) * ffxApproximateReciprocalMediumHalf(a - b);
     c     = ffxSaturate(n + ffxIsGreaterThanZeroHalf(FFX_BROADCAST_FLOAT16X3(dit) - r) * FFX_BROADCAST_FLOAT16X3(1.0 / 1023.0));
 }
 //==============================================================================================================================
 // This computes dither for positions 'p' and 'p+{8,0}'.
 FfxFloat16x2 FsrTepdDitHx2(FfxUInt32x2 p, FfxUInt32 f)
 {
     FfxFloat32x2 x;
     x.x     = FfxFloat32(p.x + f);
     x.y     = x.x + FfxFloat32(8.0);
     FfxFloat32 y = FfxFloat32(p.y);
     FfxFloat32 a = FfxFloat32((1.0 + ffxSqrt(5.0f)) / 2.0);
     FfxFloat32 b = FfxFloat32(1.0 / 3.69);
     x       = x * ffxBroadcast2(a) + ffxBroadcast2(y * b);
     return FfxFloat16x2(ffxFract(x));
 }
 //------------------------------------------------------------------------------------------------------------------------------
 void FsrTepdC8Hx2(inout FfxFloat16x2 cR, inout FfxFloat16x2 cG, inout FfxFloat16x2 cB, FfxFloat16x2 dit)
 {
     FfxFloat16x2 nR = sqrt(cR);
     FfxFloat16x2 nG = sqrt(cG);
     FfxFloat16x2 nB = sqrt(cB);
     nR     = floor(nR * FFX_BROADCAST_FLOAT16X2(255.0)) * FFX_BROADCAST_FLOAT16X2(1.0 / 255.0);
     nG     = floor(nG * FFX_BROADCAST_FLOAT16X2(255.0)) * FFX_BROADCAST_FLOAT16X2(1.0 / 255.0);
     nB     = floor(nB * FFX_BROADCAST_FLOAT16X2(255.0)) * FFX_BROADCAST_FLOAT16X2(1.0 / 255.0);
     FfxFloat16x2 aR = nR * nR;
     FfxFloat16x2 aG = nG * nG;
     FfxFloat16x2 aB = nB * nB;
     FfxFloat16x2 bR = nR + FFX_BROADCAST_FLOAT16X2(1.0 / 255.0);
     bR     = bR * bR;
     FfxFloat16x2 bG = nG + FFX_BROADCAST_FLOAT16X2(1.0 / 255.0);
     bG     = bG * bG;
     FfxFloat16x2 bB = nB + FFX_BROADCAST_FLOAT16X2(1.0 / 255.0);
     bB     = bB * bB;
     FfxFloat16x2 rR = (cR - bR) * ffxApproximateReciprocalMediumHalf(aR - bR);
     FfxFloat16x2 rG = (cG - bG) * ffxApproximateReciprocalMediumHalf(aG - bG);
     FfxFloat16x2 rB = (cB - bB) * ffxApproximateReciprocalMediumHalf(aB - bB);
     cR     = ffxSaturate(nR + ffxIsGreaterThanZeroHalf(dit - rR) * FFX_BROADCAST_FLOAT16X2(1.0 / 255.0));
     cG     = ffxSaturate(nG + ffxIsGreaterThanZeroHalf(dit - rG) * FFX_BROADCAST_FLOAT16X2(1.0 / 255.0));
     cB     = ffxSaturate(nB + ffxIsGreaterThanZeroHalf(dit - rB) * FFX_BROADCAST_FLOAT16X2(1.0 / 255.0));
 }
 //------------------------------------------------------------------------------------------------------------------------------
 void FsrTepdC10Hx2(inout FfxFloat16x2 cR,inout FfxFloat16x2 cG,inout FfxFloat16x2 cB,FfxFloat16x2 dit){
  FfxFloat16x2 nR=sqrt(cR);
  FfxFloat16x2 nG=sqrt(cG);
  FfxFloat16x2 nB=sqrt(cB);
  nR=floor(nR*FFX_BROADCAST_FLOAT16X2(1023.0))*FFX_BROADCAST_FLOAT16X2(1.0/1023.0);
  nG=floor(nG*FFX_BROADCAST_FLOAT16X2(1023.0))*FFX_BROADCAST_FLOAT16X2(1.0/1023.0);
  nB=floor(nB*FFX_BROADCAST_FLOAT16X2(1023.0))*FFX_BROADCAST_FLOAT16X2(1.0/1023.0);
  FfxFloat16x2 aR=nR*nR;
  FfxFloat16x2 aG=nG*nG;
  FfxFloat16x2 aB=nB*nB;
  FfxFloat16x2 bR=nR+FFX_BROADCAST_FLOAT16X2(1.0/1023.0);bR=bR*bR;
  FfxFloat16x2 bG=nG+FFX_BROADCAST_FLOAT16X2(1.0/1023.0);bG=bG*bG;
  FfxFloat16x2 bB=nB+FFX_BROADCAST_FLOAT16X2(1.0/1023.0);bB=bB*bB;
  FfxFloat16x2 rR=(cR-bR)*ffxApproximateReciprocalMediumHalf(aR-bR);
  FfxFloat16x2 rG=(cG-bG)*ffxApproximateReciprocalMediumHalf(aG-bG);
  FfxFloat16x2 rB=(cB-bB)*ffxApproximateReciprocalMediumHalf(aB-bB);
  cR=ffxSaturate(nR+ffxIsGreaterThanZeroHalf(dit-rR)*FFX_BROADCAST_FLOAT16X2(1.0/1023.0));
  cG=ffxSaturate(nG+ffxIsGreaterThanZeroHalf(dit-rG)*FFX_BROADCAST_FLOAT16X2(1.0/1023.0));
  cB                                                       = ffxSaturate(nB + ffxIsGreaterThanZeroHalf(dit - rB) * FFX_BROADCAST_FLOAT16X2(1.0 / 1023.0));
}
#endif
