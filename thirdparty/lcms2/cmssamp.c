//---------------------------------------------------------------------------------
//
//  Little Color Management System
//  Copyright (c) 1998-2017 Marti Maria Saguer
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software
// is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//
//---------------------------------------------------------------------------------
//

#include "lcms2_internal.h"


#define cmsmin(a, b) (((a) < (b)) ? (a) : (b))
#define cmsmax(a, b) (((a) > (b)) ? (a) : (b))

// This file contains routines for resampling and LUT optimization, black point detection
// and black preservation.

// Black point detection -------------------------------------------------------------------------


// PCS -> PCS round trip transform, always uses relative intent on the device -> pcs
static
cmsHTRANSFORM CreateRoundtripXForm(cmsHPROFILE hProfile, cmsUInt32Number nIntent)
{
    cmsContext ContextID = cmsGetProfileContextID(hProfile);
    cmsHPROFILE hLab = cmsCreateLab4ProfileTHR(ContextID, NULL);
    cmsHTRANSFORM xform;
    cmsBool BPC[4] = { FALSE, FALSE, FALSE, FALSE };
    cmsFloat64Number States[4] = { 1.0, 1.0, 1.0, 1.0 };
    cmsHPROFILE hProfiles[4];
    cmsUInt32Number Intents[4];

    hProfiles[0] = hLab; hProfiles[1] = hProfile; hProfiles[2] = hProfile; hProfiles[3] = hLab;
    Intents[0]   = INTENT_RELATIVE_COLORIMETRIC; Intents[1] = nIntent; Intents[2] = INTENT_RELATIVE_COLORIMETRIC; Intents[3] = INTENT_RELATIVE_COLORIMETRIC;

    xform =  cmsCreateExtendedTransform(ContextID, 4, hProfiles, BPC, Intents,
        States, NULL, 0, TYPE_Lab_DBL, TYPE_Lab_DBL, cmsFLAGS_NOCACHE|cmsFLAGS_NOOPTIMIZE);

    cmsCloseProfile(hLab);
    return xform;
}

// Use darker colorants to obtain black point. This works in the relative colorimetric intent and
// assumes more ink results in darker colors. No ink limit is assumed.
static
cmsBool  BlackPointAsDarkerColorant(cmsHPROFILE    hInput,
                                    cmsUInt32Number Intent,
                                    cmsCIEXYZ* BlackPoint,
                                    cmsUInt32Number dwFlags)
{
    cmsUInt16Number *Black;
    cmsHTRANSFORM xform;
    cmsColorSpaceSignature Space;
    cmsUInt32Number nChannels;
    cmsUInt32Number dwFormat;
    cmsHPROFILE hLab;
    cmsCIELab  Lab;
    cmsCIEXYZ  BlackXYZ;
    cmsContext ContextID = cmsGetProfileContextID(hInput);

    // If the profile does not support input direction, assume Black point 0
    if (!cmsIsIntentSupported(hInput, Intent, LCMS_USED_AS_INPUT)) {

        BlackPoint -> X = BlackPoint ->Y = BlackPoint -> Z = 0.0;
        return FALSE;
    }

    // Create a formatter which has n channels and floating point
    dwFormat = cmsFormatterForColorspaceOfProfile(hInput, 2, FALSE);

   // Try to get black by using black colorant
    Space = cmsGetColorSpace(hInput);

    // This function returns darker colorant in 16 bits for several spaces
    if (!_cmsEndPointsBySpace(Space, NULL, &Black, &nChannels)) {

        BlackPoint -> X = BlackPoint ->Y = BlackPoint -> Z = 0.0;
        return FALSE;
    }

    if (nChannels != T_CHANNELS(dwFormat)) {
       BlackPoint -> X = BlackPoint ->Y = BlackPoint -> Z = 0.0;
       return FALSE;
    }

    // Lab will be used as the output space, but lab2 will avoid recursion
    hLab = cmsCreateLab2ProfileTHR(ContextID, NULL);
    if (hLab == NULL) {
       BlackPoint -> X = BlackPoint ->Y = BlackPoint -> Z = 0.0;
       return FALSE;
    }

    // Create the transform
    xform = cmsCreateTransformTHR(ContextID, hInput, dwFormat,
                                hLab, TYPE_Lab_DBL, Intent, cmsFLAGS_NOOPTIMIZE|cmsFLAGS_NOCACHE);
    cmsCloseProfile(hLab);

    if (xform == NULL) {

        // Something went wrong. Get rid of open resources and return zero as black
        BlackPoint -> X = BlackPoint ->Y = BlackPoint -> Z = 0.0;
        return FALSE;
    }

    // Convert black to Lab
    cmsDoTransform(xform, Black, &Lab, 1);

    // Force it to be neutral, clip to max. L* of 50
    Lab.a = Lab.b = 0;
    if (Lab.L > 50) Lab.L = 50;

    // Free the resources
    cmsDeleteTransform(xform);

    // Convert from Lab (which is now clipped) to XYZ.
    cmsLab2XYZ(NULL, &BlackXYZ, &Lab);

    if (BlackPoint != NULL)
        *BlackPoint = BlackXYZ;

    return TRUE;

    cmsUNUSED_PARAMETER(dwFlags);
}

// Get a black point of output CMYK profile, discounting any ink-limiting embedded
// in the profile. For doing that, we use perceptual intent in input direction:
// Lab (0, 0, 0) -> [Perceptual] Profile -> CMYK -> [Rel. colorimetric] Profile -> Lab
static
cmsBool BlackPointUsingPerceptualBlack(cmsCIEXYZ* BlackPoint, cmsHPROFILE hProfile)
{
    cmsHTRANSFORM hRoundTrip;
    cmsCIELab LabIn, LabOut;
    cmsCIEXYZ  BlackXYZ;

     // Is the intent supported by the profile?
    if (!cmsIsIntentSupported(hProfile, INTENT_PERCEPTUAL, LCMS_USED_AS_INPUT)) {

        BlackPoint -> X = BlackPoint ->Y = BlackPoint -> Z = 0.0;
        return TRUE;
    }

    hRoundTrip = CreateRoundtripXForm(hProfile, INTENT_PERCEPTUAL);
    if (hRoundTrip == NULL) {
        BlackPoint -> X = BlackPoint ->Y = BlackPoint -> Z = 0.0;
        return FALSE;
    }

    LabIn.L = LabIn.a = LabIn.b = 0;
    cmsDoTransform(hRoundTrip, &LabIn, &LabOut, 1);

    // Clip Lab to reasonable limits
    if (LabOut.L > 50) LabOut.L = 50;
    LabOut.a = LabOut.b = 0;

    cmsDeleteTransform(hRoundTrip);

    // Convert it to XYZ
    cmsLab2XYZ(NULL, &BlackXYZ, &LabOut);

    if (BlackPoint != NULL)
        *BlackPoint = BlackXYZ;

    return TRUE;
}

// This function shouldn't exist at all -- there is such quantity of broken
// profiles on black point tag, that we must somehow fix chromaticity to
// avoid huge tint when doing Black point compensation. This function does
// just that. There is a special flag for using black point tag, but turned
// off by default because it is bogus on most profiles. The detection algorithm
// involves to turn BP to neutral and to use only L component.
cmsBool CMSEXPORT cmsDetectBlackPoint(cmsCIEXYZ* BlackPoint, cmsHPROFILE hProfile, cmsUInt32Number Intent, cmsUInt32Number dwFlags)
{
    cmsProfileClassSignature devClass;

    // Make sure the device class is adequate
    devClass = cmsGetDeviceClass(hProfile);
    if (devClass == cmsSigLinkClass ||
        devClass == cmsSigAbstractClass ||
        devClass == cmsSigNamedColorClass) {
            BlackPoint -> X = BlackPoint ->Y = BlackPoint -> Z = 0.0;
            return FALSE;
    }

    // Make sure intent is adequate
    if (Intent != INTENT_PERCEPTUAL &&
        Intent != INTENT_RELATIVE_COLORIMETRIC &&
        Intent != INTENT_SATURATION) {
            BlackPoint -> X = BlackPoint ->Y = BlackPoint -> Z = 0.0;
            return FALSE;
    }

    // v4 + perceptual & saturation intents does have its own black point, and it is
    // well specified enough to use it. Black point tag is deprecated in V4.
    if ((cmsGetEncodedICCversion(hProfile) >= 0x4000000) &&
        (Intent == INTENT_PERCEPTUAL || Intent == INTENT_SATURATION)) {

            // Matrix shaper share MRC & perceptual intents
            if (cmsIsMatrixShaper(hProfile))
                return BlackPointAsDarkerColorant(hProfile, INTENT_RELATIVE_COLORIMETRIC, BlackPoint, 0);

            // Get Perceptual black out of v4 profiles. That is fixed for perceptual & saturation intents
            BlackPoint -> X = cmsPERCEPTUAL_BLACK_X;
            BlackPoint -> Y = cmsPERCEPTUAL_BLACK_Y;
            BlackPoint -> Z = cmsPERCEPTUAL_BLACK_Z;

            return TRUE;
    }


#ifdef CMS_USE_PROFILE_BLACK_POINT_TAG

    // v2, v4 rel/abs colorimetric
    if (cmsIsTag(hProfile, cmsSigMediaBlackPointTag) &&
        Intent == INTENT_RELATIVE_COLORIMETRIC) {

            cmsCIEXYZ *BlackPtr, BlackXYZ, UntrustedBlackPoint, TrustedBlackPoint, MediaWhite;
            cmsCIELab Lab;

            // If black point is specified, then use it,

            BlackPtr = cmsReadTag(hProfile, cmsSigMediaBlackPointTag);
            if (BlackPtr != NULL) {

                BlackXYZ = *BlackPtr;
                _cmsReadMediaWhitePoint(&MediaWhite, hProfile);

                // Black point is absolute XYZ, so adapt to D50 to get PCS value
                cmsAdaptToIlluminant(&UntrustedBlackPoint, &MediaWhite, cmsD50_XYZ(), &BlackXYZ);

                // Force a=b=0 to get rid of any chroma
                cmsXYZ2Lab(NULL, &Lab, &UntrustedBlackPoint);
                Lab.a = Lab.b = 0;
                if (Lab.L > 50) Lab.L = 50; // Clip to L* <= 50
                cmsLab2XYZ(NULL, &TrustedBlackPoint, &Lab);

                if (BlackPoint != NULL)
                    *BlackPoint = TrustedBlackPoint;

                return TRUE;
            }
    }
#endif

    // That is about v2 profiles.

    // If output profile, discount ink-limiting and that's all
    if (Intent == INTENT_RELATIVE_COLORIMETRIC &&
        (cmsGetDeviceClass(hProfile) == cmsSigOutputClass) &&
        (cmsGetColorSpace(hProfile)  == cmsSigCmykData))
        return BlackPointUsingPerceptualBlack(BlackPoint, hProfile);

    // Nope, compute BP using current intent.
    return BlackPointAsDarkerColorant(hProfile, Intent, BlackPoint, dwFlags);
}



// ---------------------------------------------------------------------------------------------------------

// Least Squares Fit of a Quadratic Curve to Data
// http://www.personal.psu.edu/jhm/f90/lectures/lsq2.html

static
cmsFloat64Number RootOfLeastSquaresFitQuadraticCurve(int n, cmsFloat64Number x[], cmsFloat64Number y[])
{
    double sum_x = 0, sum_x2 = 0, sum_x3 = 0, sum_x4 = 0;
    double sum_y = 0, sum_yx = 0, sum_yx2 = 0;
    double d, a, b, c;
    int i;
    cmsMAT3 m;
    cmsVEC3 v, res;

    if (n < 4) return 0;

    for (i=0; i < n; i++) {

        double xn = x[i];
        double yn = y[i];

        sum_x  += xn;
        sum_x2 += xn*xn;
        sum_x3 += xn*xn*xn;
        sum_x4 += xn*xn*xn*xn;

        sum_y += yn;
        sum_yx += yn*xn;
        sum_yx2 += yn*xn*xn;
    }

    _cmsVEC3init(&m.v[0], n,      sum_x,  sum_x2);
    _cmsVEC3init(&m.v[1], sum_x,  sum_x2, sum_x3);
    _cmsVEC3init(&m.v[2], sum_x2, sum_x3, sum_x4);

    _cmsVEC3init(&v, sum_y, sum_yx, sum_yx2);

    if (!_cmsMAT3solve(&res, &m, &v)) return 0;

      
    a = res.n[2];
    b = res.n[1];
    c = res.n[0];

    if (fabs(a) < 1.0E-10) {
    
        return cmsmin(0, cmsmax(50, -c/b ));
    }
    else {

         d = b*b - 4.0 * a * c;
         if (d <= 0) {
             return 0;
         }
         else {

             double rt = (-b + sqrt(d)) / (2.0 * a);

             return cmsmax(0, cmsmin(50, rt));
         }
   }

}



// Calculates the black point of a destination profile.
// This algorithm comes from the Adobe paper disclosing its black point compensation method.
cmsBool CMSEXPORT cmsDetectDestinationBlackPoint(cmsCIEXYZ* BlackPoint, cmsHPROFILE hProfile, cmsUInt32Number Intent, cmsUInt32Number dwFlags)
{
    cmsColorSpaceSignature ColorSpace;
    cmsHTRANSFORM hRoundTrip = NULL;
    cmsCIELab InitialLab, destLab, Lab;
    cmsFloat64Number inRamp[256], outRamp[256];
    cmsFloat64Number MinL, MaxL;
    cmsBool NearlyStraightMidrange = TRUE;  
    cmsFloat64Number yRamp[256];
    cmsFloat64Number x[256], y[256];
    cmsFloat64Number lo, hi;
    int n, l;
    cmsProfileClassSignature devClass;

    // Make sure the device class is adequate
    devClass = cmsGetDeviceClass(hProfile);
    if (devClass == cmsSigLinkClass ||
        devClass == cmsSigAbstractClass ||
        devClass == cmsSigNamedColorClass) {
            BlackPoint -> X = BlackPoint ->Y = BlackPoint -> Z = 0.0;
            return FALSE;
    }

    // Make sure intent is adequate
    if (Intent != INTENT_PERCEPTUAL &&
        Intent != INTENT_RELATIVE_COLORIMETRIC &&
        Intent != INTENT_SATURATION) {
            BlackPoint -> X = BlackPoint ->Y = BlackPoint -> Z = 0.0;
            return FALSE;
    }


    // v4 + perceptual & saturation intents does have its own black point, and it is
    // well specified enough to use it. Black point tag is deprecated in V4.
    if ((cmsGetEncodedICCversion(hProfile) >= 0x4000000) &&
        (Intent == INTENT_PERCEPTUAL || Intent == INTENT_SATURATION)) {

            // Matrix shaper share MRC & perceptual intents
            if (cmsIsMatrixShaper(hProfile))
                return BlackPointAsDarkerColorant(hProfile, INTENT_RELATIVE_COLORIMETRIC, BlackPoint, 0);

            // Get Perceptual black out of v4 profiles. That is fixed for perceptual & saturation intents
            BlackPoint -> X = cmsPERCEPTUAL_BLACK_X;
            BlackPoint -> Y = cmsPERCEPTUAL_BLACK_Y;
            BlackPoint -> Z = cmsPERCEPTUAL_BLACK_Z;
            return TRUE;
    }


    // Check if the profile is lut based and gray, rgb or cmyk (7.2 in Adobe's document)
    ColorSpace = cmsGetColorSpace(hProfile);
    if (!cmsIsCLUT(hProfile, Intent, LCMS_USED_AS_OUTPUT ) ||
        (ColorSpace != cmsSigGrayData &&
         ColorSpace != cmsSigRgbData  &&
         ColorSpace != cmsSigCmykData)) {

        // In this case, handle as input case
        return cmsDetectBlackPoint(BlackPoint, hProfile, Intent, dwFlags);
    }

    // It is one of the valid cases!, use Adobe algorithm

    
    // Set a first guess, that should work on good profiles.
    if (Intent == INTENT_RELATIVE_COLORIMETRIC) {

        cmsCIEXYZ IniXYZ;

        // calculate initial Lab as source black point
        if (!cmsDetectBlackPoint(&IniXYZ, hProfile, Intent, dwFlags)) {
            return FALSE;
        }

        // convert the XYZ to lab
        cmsXYZ2Lab(NULL, &InitialLab, &IniXYZ);

    } else {

        // set the initial Lab to zero, that should be the black point for perceptual and saturation
        InitialLab.L = 0;
        InitialLab.a = 0;
        InitialLab.b = 0;
    }


    // Step 2
    // ======

    // Create a roundtrip. Define a Transform BT for all x in L*a*b*
    hRoundTrip = CreateRoundtripXForm(hProfile, Intent);
    if (hRoundTrip == NULL)  return FALSE;

    // Compute ramps

    for (l=0; l < 256; l++) {

        Lab.L = (cmsFloat64Number) (l * 100.0) / 255.0;
        Lab.a = cmsmin(50, cmsmax(-50, InitialLab.a));
        Lab.b = cmsmin(50, cmsmax(-50, InitialLab.b));

        cmsDoTransform(hRoundTrip, &Lab, &destLab, 1);

        inRamp[l]  = Lab.L;
        outRamp[l] = destLab.L;
    }

    // Make monotonic
    for (l = 254; l > 0; --l) {
        outRamp[l] = cmsmin(outRamp[l], outRamp[l+1]);
    }

    // Check
    if (! (outRamp[0] < outRamp[255])) {

        cmsDeleteTransform(hRoundTrip);
        BlackPoint -> X = BlackPoint ->Y = BlackPoint -> Z = 0.0;
        return FALSE;
    }


    // Test for mid range straight (only on relative colorimetric)
    NearlyStraightMidrange = TRUE;
    MinL = outRamp[0]; MaxL = outRamp[255];
    if (Intent == INTENT_RELATIVE_COLORIMETRIC) {
      
        for (l=0; l < 256; l++) {

            if (! ((inRamp[l] <= MinL + 0.2 * (MaxL - MinL) ) ||   
                (fabs(inRamp[l] - outRamp[l]) < 4.0 )))
                NearlyStraightMidrange = FALSE;
        }

        // If the mid range is straight (as determined above) then the 
        // DestinationBlackPoint shall be the same as initialLab. 
        // Otherwise, the DestinationBlackPoint shall be determined 
        // using curve fitting.
        if (NearlyStraightMidrange) {

            cmsLab2XYZ(NULL, BlackPoint, &InitialLab);
            cmsDeleteTransform(hRoundTrip);
            return TRUE;
        }
    }

 
    // curve fitting: The round-trip curve normally looks like a nearly constant section at the black point,
    // with a corner and a nearly straight line to the white point.  
    for (l=0; l < 256; l++) {
    
        yRamp[l] = (outRamp[l] - MinL) / (MaxL - MinL);
    }

    // find the black point using the least squares error quadratic curve fitting
    if (Intent == INTENT_RELATIVE_COLORIMETRIC) {
        lo = 0.1;
        hi = 0.5;
    }
    else {

        // Perceptual and saturation
        lo = 0.03;
        hi = 0.25;
    }

    // Capture shadow points for the fitting.
    n = 0;
    for (l=0; l < 256; l++) {
    
        cmsFloat64Number ff = yRamp[l];

        if (ff >= lo && ff < hi) {
            x[n] = inRamp[l];
            y[n] = yRamp[l];
            n++;
        }    
    }

    
    // No suitable points
    if (n < 3 ) {
        cmsDeleteTransform(hRoundTrip);
        BlackPoint -> X = BlackPoint ->Y = BlackPoint -> Z = 0.0;
        return FALSE;
    }

  
    // fit and get the vertex of quadratic curve
    Lab.L = RootOfLeastSquaresFitQuadraticCurve(n, x, y);

    if (Lab.L < 0.0) { // clip to zero L* if the vertex is negative
        Lab.L = 0;
    }

    Lab.a = InitialLab.a;
    Lab.b = InitialLab.b;

    cmsLab2XYZ(NULL, BlackPoint, &Lab);

    cmsDeleteTransform(hRoundTrip);
    return TRUE;
}
