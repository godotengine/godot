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

// Read tags using low-level functions, provides necessary glue code to adapt versions, etc.

// LUT tags
static const cmsTagSignature Device2PCS16[]   =  {cmsSigAToB0Tag,     // Perceptual
                                                  cmsSigAToB1Tag,     // Relative colorimetric
                                                  cmsSigAToB2Tag,     // Saturation
                                                  cmsSigAToB1Tag };   // Absolute colorimetric

static const cmsTagSignature Device2PCSFloat[] = {cmsSigDToB0Tag,     // Perceptual
                                                  cmsSigDToB1Tag,     // Relative colorimetric
                                                  cmsSigDToB2Tag,     // Saturation
                                                  cmsSigDToB3Tag };   // Absolute colorimetric

static const cmsTagSignature PCS2Device16[]    = {cmsSigBToA0Tag,     // Perceptual
                                                  cmsSigBToA1Tag,     // Relative colorimetric
                                                  cmsSigBToA2Tag,     // Saturation
                                                  cmsSigBToA1Tag };   // Absolute colorimetric

static const cmsTagSignature PCS2DeviceFloat[] = {cmsSigBToD0Tag,     // Perceptual
                                                  cmsSigBToD1Tag,     // Relative colorimetric
                                                  cmsSigBToD2Tag,     // Saturation
                                                  cmsSigBToD3Tag };   // Absolute colorimetric


// Factors to convert from 1.15 fixed point to 0..1.0 range and vice-versa
#define InpAdj   (1.0/MAX_ENCODEABLE_XYZ)     // (65536.0/(65535.0*2.0))
#define OutpAdj  (MAX_ENCODEABLE_XYZ)         // ((2.0*65535.0)/65536.0)

// Several resources for gray conversions.
static const cmsFloat64Number GrayInputMatrix[] = { (InpAdj*cmsD50X),  (InpAdj*cmsD50Y),  (InpAdj*cmsD50Z) };
static const cmsFloat64Number OneToThreeInputMatrix[] = { 1, 1, 1 };
static const cmsFloat64Number PickYMatrix[] = { 0, (OutpAdj*cmsD50Y), 0 };
static const cmsFloat64Number PickLstarMatrix[] = { 1, 0, 0 };

// Get a media white point fixing some issues found in certain old profiles
cmsBool  _cmsReadMediaWhitePoint(cmsCIEXYZ* Dest, cmsHPROFILE hProfile)
{
    cmsCIEXYZ* Tag;

    _cmsAssert(Dest != NULL);

    Tag = (cmsCIEXYZ*) cmsReadTag(hProfile, cmsSigMediaWhitePointTag);

    // If no wp, take D50
    if (Tag == NULL) {
        *Dest = *cmsD50_XYZ();
        return TRUE;
    }

    // V2 display profiles should give D50
    if (cmsGetEncodedICCversion(hProfile) < 0x4000000) {

        if (cmsGetDeviceClass(hProfile) == cmsSigDisplayClass) {
            *Dest = *cmsD50_XYZ();
            return TRUE;
        }
    }

    // All seems ok
    *Dest = *Tag;
    return TRUE;
}


// Chromatic adaptation matrix. Fix some issues as well
cmsBool  _cmsReadCHAD(cmsMAT3* Dest, cmsHPROFILE hProfile)
{
    cmsMAT3* Tag;

    _cmsAssert(Dest != NULL);

    Tag = (cmsMAT3*) cmsReadTag(hProfile, cmsSigChromaticAdaptationTag);

    if (Tag != NULL) {
        *Dest = *Tag;
        return TRUE;
    }

    // No CHAD available, default it to identity
    _cmsMAT3identity(Dest);

    // V2 display profiles should give D50
    if (cmsGetEncodedICCversion(hProfile) < 0x4000000) {

        if (cmsGetDeviceClass(hProfile) == cmsSigDisplayClass) {

            cmsCIEXYZ* White = (cmsCIEXYZ*) cmsReadTag(hProfile, cmsSigMediaWhitePointTag);

            if (White == NULL) {

                _cmsMAT3identity(Dest);
                return TRUE;
            }

            return _cmsAdaptationMatrix(Dest, NULL, White, cmsD50_XYZ());
        }
    }

    return TRUE;
}


// Auxiliary, read colorants as a MAT3 structure. Used by any function that needs a matrix-shaper
static
cmsBool ReadICCMatrixRGB2XYZ(cmsMAT3* r, cmsHPROFILE hProfile)
{
    cmsCIEXYZ *PtrRed, *PtrGreen, *PtrBlue;

    _cmsAssert(r != NULL);

    PtrRed   = (cmsCIEXYZ *) cmsReadTag(hProfile, cmsSigRedColorantTag);
    PtrGreen = (cmsCIEXYZ *) cmsReadTag(hProfile, cmsSigGreenColorantTag);
    PtrBlue  = (cmsCIEXYZ *) cmsReadTag(hProfile, cmsSigBlueColorantTag);

    if (PtrRed == NULL || PtrGreen == NULL || PtrBlue == NULL)
        return FALSE;

    _cmsVEC3init(&r -> v[0], PtrRed -> X, PtrGreen -> X,  PtrBlue -> X);
    _cmsVEC3init(&r -> v[1], PtrRed -> Y, PtrGreen -> Y,  PtrBlue -> Y);
    _cmsVEC3init(&r -> v[2], PtrRed -> Z, PtrGreen -> Z,  PtrBlue -> Z);

    return TRUE;
}


// Gray input pipeline
static
cmsPipeline* BuildGrayInputMatrixPipeline(cmsHPROFILE hProfile)
{
    cmsToneCurve *GrayTRC;
    cmsPipeline* Lut;
    cmsContext ContextID = cmsGetProfileContextID(hProfile);

    GrayTRC = (cmsToneCurve *) cmsReadTag(hProfile, cmsSigGrayTRCTag);
    if (GrayTRC == NULL) return NULL;

    Lut = cmsPipelineAlloc(ContextID, 1, 3);
    if (Lut == NULL)
        goto Error;

    if (cmsGetPCS(hProfile) == cmsSigLabData) {

        // In this case we implement the profile as an  identity matrix plus 3 tone curves
        cmsUInt16Number Zero[2] = { 0x8080, 0x8080 };
        cmsToneCurve* EmptyTab;
        cmsToneCurve* LabCurves[3];

        EmptyTab = cmsBuildTabulatedToneCurve16(ContextID, 2, Zero);

        if (EmptyTab == NULL)
            goto Error;

        LabCurves[0] = GrayTRC;
        LabCurves[1] = EmptyTab;
        LabCurves[2] = EmptyTab;

        if (!cmsPipelineInsertStage(Lut, cmsAT_END, cmsStageAllocMatrix(ContextID, 3,  1, OneToThreeInputMatrix, NULL)) ||
            !cmsPipelineInsertStage(Lut, cmsAT_END, cmsStageAllocToneCurves(ContextID, 3, LabCurves))) {
                cmsFreeToneCurve(EmptyTab);
                goto Error;
        }

        cmsFreeToneCurve(EmptyTab);

    }
    else  {

        if (!cmsPipelineInsertStage(Lut, cmsAT_END, cmsStageAllocToneCurves(ContextID, 1, &GrayTRC)) ||
            !cmsPipelineInsertStage(Lut, cmsAT_END, cmsStageAllocMatrix(ContextID, 3,  1, GrayInputMatrix, NULL)))
            goto Error;
    }

    return Lut;

Error:
    cmsFreeToneCurve(GrayTRC);
    cmsPipelineFree(Lut);
    return NULL;
}

// RGB Matrix shaper
static
cmsPipeline* BuildRGBInputMatrixShaper(cmsHPROFILE hProfile)
{
    cmsPipeline* Lut;
    cmsMAT3 Mat;
    cmsToneCurve *Shapes[3];
    cmsContext ContextID = cmsGetProfileContextID(hProfile);
    int i, j;

    if (!ReadICCMatrixRGB2XYZ(&Mat, hProfile)) return NULL;

    // XYZ PCS in encoded in 1.15 format, and the matrix output comes in 0..0xffff range, so
    // we need to adjust the output by a factor of (0x10000/0xffff) to put data in
    // a 1.16 range, and then a >> 1 to obtain 1.15. The total factor is (65536.0)/(65535.0*2)

    for (i=0; i < 3; i++)
        for (j=0; j < 3; j++)
            Mat.v[i].n[j] *= InpAdj;


    Shapes[0] = (cmsToneCurve *) cmsReadTag(hProfile, cmsSigRedTRCTag);
    Shapes[1] = (cmsToneCurve *) cmsReadTag(hProfile, cmsSigGreenTRCTag);
    Shapes[2] = (cmsToneCurve *) cmsReadTag(hProfile, cmsSigBlueTRCTag);

    if (!Shapes[0] || !Shapes[1] || !Shapes[2])
        return NULL;

    Lut = cmsPipelineAlloc(ContextID, 3, 3);
    if (Lut != NULL) {

        if (!cmsPipelineInsertStage(Lut, cmsAT_END, cmsStageAllocToneCurves(ContextID, 3, Shapes)) ||
            !cmsPipelineInsertStage(Lut, cmsAT_END, cmsStageAllocMatrix(ContextID, 3, 3, (cmsFloat64Number*) &Mat, NULL)))
            goto Error;

        // Note that it is certainly possible a single profile would have a LUT based
        // tag for output working in lab and a matrix-shaper for the fallback cases. 
        // This is not allowed by the spec, but this code is tolerant to those cases    
        if (cmsGetPCS(hProfile) == cmsSigLabData) {

            if (!cmsPipelineInsertStage(Lut, cmsAT_END, _cmsStageAllocXYZ2Lab(ContextID)))
                goto Error;
        }

    }

    return Lut;

Error:
    cmsPipelineFree(Lut);
    return NULL;
}



// Read the DToAX tag, adjusting the encoding of Lab or XYZ if neded
static
cmsPipeline* _cmsReadFloatInputTag(cmsHPROFILE hProfile, cmsTagSignature tagFloat)
{
    cmsContext ContextID       = cmsGetProfileContextID(hProfile);
    cmsPipeline* Lut           = cmsPipelineDup((cmsPipeline*) cmsReadTag(hProfile, tagFloat));
    cmsColorSpaceSignature spc = cmsGetColorSpace(hProfile);
    cmsColorSpaceSignature PCS = cmsGetPCS(hProfile);
    
    if (Lut == NULL) return NULL;
    
    // input and output of transform are in lcms 0..1 encoding.  If XYZ or Lab spaces are used, 
    //  these need to be normalized into the appropriate ranges (Lab = 100,0,0, XYZ=1.0,1.0,1.0)
    if ( spc == cmsSigLabData)
    {
        if (!cmsPipelineInsertStage(Lut, cmsAT_BEGIN, _cmsStageNormalizeToLabFloat(ContextID)))
            goto Error;
    }
    else if (spc == cmsSigXYZData)
    {
        if (!cmsPipelineInsertStage(Lut, cmsAT_BEGIN, _cmsStageNormalizeToXyzFloat(ContextID)))
            goto Error;
    }
    
    if ( PCS == cmsSigLabData)
    {
        if (!cmsPipelineInsertStage(Lut, cmsAT_END, _cmsStageNormalizeFromLabFloat(ContextID)))
            goto Error;
    }
    else if( PCS == cmsSigXYZData)
    {
        if (!cmsPipelineInsertStage(Lut, cmsAT_END, _cmsStageNormalizeFromXyzFloat(ContextID)))
            goto Error;
    }
    
    return Lut;

Error:
    cmsPipelineFree(Lut);
    return NULL;
}


// Read and create a BRAND NEW MPE LUT from a given profile. All stuff dependent of version, etc
// is adjusted here in order to create a LUT that takes care of all those details.
// We add intent = 0xffffffff as a way to read matrix shaper always, no matter of other LUT
cmsPipeline* CMSEXPORT _cmsReadInputLUT(cmsHPROFILE hProfile, cmsUInt32Number Intent)
{
    cmsTagTypeSignature OriginalType;
    cmsTagSignature tag16;
    cmsTagSignature tagFloat;
    cmsContext ContextID = cmsGetProfileContextID(hProfile);

    // On named color, take the appropriate tag
    if (cmsGetDeviceClass(hProfile) == cmsSigNamedColorClass) {

        cmsPipeline* Lut;
        cmsNAMEDCOLORLIST* nc = (cmsNAMEDCOLORLIST*) cmsReadTag(hProfile, cmsSigNamedColor2Tag);

        if (nc == NULL) return NULL;

        Lut = cmsPipelineAlloc(ContextID, 0, 0);
        if (Lut == NULL) {
            cmsFreeNamedColorList(nc);
            return NULL;
        }

        if (!cmsPipelineInsertStage(Lut, cmsAT_BEGIN, _cmsStageAllocNamedColor(nc, TRUE)) ||
            !cmsPipelineInsertStage(Lut, cmsAT_END, _cmsStageAllocLabV2ToV4(ContextID))) {
            cmsPipelineFree(Lut);
            return NULL;
        }
        return Lut;
    }

    // This is an attempt to reuse this function to retrieve the matrix-shaper as pipeline no
    // matter other LUT are present and have precedence. Intent = 0xffffffff can be used for that.
    if (Intent <= INTENT_ABSOLUTE_COLORIMETRIC) {

        tag16 = Device2PCS16[Intent];
        tagFloat = Device2PCSFloat[Intent];

        if (cmsIsTag(hProfile, tagFloat)) {  // Float tag takes precedence

            // Floating point LUT are always V4, but the encoding range is no
            // longer 0..1.0, so we need to add an stage depending on the color space
            return _cmsReadFloatInputTag(hProfile, tagFloat);
        }

        // Revert to perceptual if no tag is found
        if (!cmsIsTag(hProfile, tag16)) {
            tag16 = Device2PCS16[0];
        }

        if (cmsIsTag(hProfile, tag16)) { // Is there any LUT-Based table?

            // Check profile version and LUT type. Do the necessary adjustments if needed

            // First read the tag
            cmsPipeline* Lut = (cmsPipeline*) cmsReadTag(hProfile, tag16);
            if (Lut == NULL) return NULL;

            // After reading it, we have now info about the original type
            OriginalType =  _cmsGetTagTrueType(hProfile, tag16);

            // The profile owns the Lut, so we need to copy it
            Lut = cmsPipelineDup(Lut);

            // We need to adjust data only for Lab16 on output
            if (OriginalType != cmsSigLut16Type || cmsGetPCS(hProfile) != cmsSigLabData)
                return Lut;

            // If the input is Lab, add also a conversion at the begin
            if (cmsGetColorSpace(hProfile) == cmsSigLabData &&
                !cmsPipelineInsertStage(Lut, cmsAT_BEGIN, _cmsStageAllocLabV4ToV2(ContextID)))
                goto Error;

            // Add a matrix for conversion V2 to V4 Lab PCS
            if (!cmsPipelineInsertStage(Lut, cmsAT_END, _cmsStageAllocLabV2ToV4(ContextID)))
                goto Error;

            return Lut;
Error:
            cmsPipelineFree(Lut);
            return NULL;
        }
    }

    // Lut was not found, try to create a matrix-shaper

    // Check if this is a grayscale profile.
    if (cmsGetColorSpace(hProfile) == cmsSigGrayData) {

        // if so, build appropriate conversion tables.
        // The tables are the PCS iluminant, scaled across GrayTRC
        return BuildGrayInputMatrixPipeline(hProfile);
    }

    // Not gray, create a normal matrix-shaper
    return BuildRGBInputMatrixShaper(hProfile);
}

// ---------------------------------------------------------------------------------------------------------------

// Gray output pipeline.
// XYZ -> Gray or Lab -> Gray. Since we only know the GrayTRC, we need to do some assumptions. Gray component will be
// given by Y on XYZ PCS and by L* on Lab PCS, Both across inverse TRC curve.
// The complete pipeline on XYZ is Matrix[3:1] -> Tone curve and in Lab Matrix[3:1] -> Tone Curve as well.

static
cmsPipeline* BuildGrayOutputPipeline(cmsHPROFILE hProfile)
{
    cmsToneCurve *GrayTRC, *RevGrayTRC;
    cmsPipeline* Lut;
    cmsContext ContextID = cmsGetProfileContextID(hProfile);

    GrayTRC = (cmsToneCurve *) cmsReadTag(hProfile, cmsSigGrayTRCTag);
    if (GrayTRC == NULL) return NULL;

    RevGrayTRC = cmsReverseToneCurve(GrayTRC);
    if (RevGrayTRC == NULL) return NULL;

    Lut = cmsPipelineAlloc(ContextID, 3, 1);
    if (Lut == NULL) {
        cmsFreeToneCurve(RevGrayTRC);
        return NULL;
    }

    if (cmsGetPCS(hProfile) == cmsSigLabData) {

        if (!cmsPipelineInsertStage(Lut, cmsAT_END, cmsStageAllocMatrix(ContextID, 1,  3, PickLstarMatrix, NULL)))
            goto Error;
    }
    else  {
        if (!cmsPipelineInsertStage(Lut, cmsAT_END, cmsStageAllocMatrix(ContextID, 1,  3, PickYMatrix, NULL)))
            goto Error;
    }

    if (!cmsPipelineInsertStage(Lut, cmsAT_END, cmsStageAllocToneCurves(ContextID, 1, &RevGrayTRC)))
        goto Error;

    cmsFreeToneCurve(RevGrayTRC);
    return Lut;

Error:
    cmsFreeToneCurve(RevGrayTRC);
    cmsPipelineFree(Lut);
    return NULL;
}


static
cmsPipeline* BuildRGBOutputMatrixShaper(cmsHPROFILE hProfile)
{
    cmsPipeline* Lut;
    cmsToneCurve *Shapes[3], *InvShapes[3];
    cmsMAT3 Mat, Inv;
    int i, j;
    cmsContext ContextID = cmsGetProfileContextID(hProfile);

    if (!ReadICCMatrixRGB2XYZ(&Mat, hProfile))
        return NULL;

    if (!_cmsMAT3inverse(&Mat, &Inv))
        return NULL;

    // XYZ PCS in encoded in 1.15 format, and the matrix input should come in 0..0xffff range, so
    // we need to adjust the input by a << 1 to obtain a 1.16 fixed and then by a factor of
    // (0xffff/0x10000) to put data in 0..0xffff range. Total factor is (2.0*65535.0)/65536.0;

    for (i=0; i < 3; i++)
        for (j=0; j < 3; j++)
            Inv.v[i].n[j] *= OutpAdj;

    Shapes[0] = (cmsToneCurve *) cmsReadTag(hProfile, cmsSigRedTRCTag);
    Shapes[1] = (cmsToneCurve *) cmsReadTag(hProfile, cmsSigGreenTRCTag);
    Shapes[2] = (cmsToneCurve *) cmsReadTag(hProfile, cmsSigBlueTRCTag);

    if (!Shapes[0] || !Shapes[1] || !Shapes[2])
        return NULL;

    InvShapes[0] = cmsReverseToneCurve(Shapes[0]);
    InvShapes[1] = cmsReverseToneCurve(Shapes[1]);
    InvShapes[2] = cmsReverseToneCurve(Shapes[2]);

    if (!InvShapes[0] || !InvShapes[1] || !InvShapes[2]) {
        return NULL;
    }

    Lut = cmsPipelineAlloc(ContextID, 3, 3);
    if (Lut != NULL) {

        // Note that it is certainly possible a single profile would have a LUT based
        // tag for output working in lab and a matrix-shaper for the fallback cases. 
        // This is not allowed by the spec, but this code is tolerant to those cases    
        if (cmsGetPCS(hProfile) == cmsSigLabData) {

            if (!cmsPipelineInsertStage(Lut, cmsAT_END, _cmsStageAllocLab2XYZ(ContextID)))
                goto Error;
        }

        if (!cmsPipelineInsertStage(Lut, cmsAT_END, cmsStageAllocMatrix(ContextID, 3, 3, (cmsFloat64Number*) &Inv, NULL)) ||
            !cmsPipelineInsertStage(Lut, cmsAT_END, cmsStageAllocToneCurves(ContextID, 3, InvShapes)))
            goto Error;
    }

    cmsFreeToneCurveTriple(InvShapes);
    return Lut;
Error:
    cmsFreeToneCurveTriple(InvShapes);
    cmsPipelineFree(Lut);
    return NULL;
}


// Change CLUT interpolation to trilinear
static
void ChangeInterpolationToTrilinear(cmsPipeline* Lut)
{
    cmsStage* Stage;

    for (Stage = cmsPipelineGetPtrToFirstStage(Lut);
        Stage != NULL;
        Stage = cmsStageNext(Stage)) {

            if (cmsStageType(Stage) == cmsSigCLutElemType) {

                _cmsStageCLutData* CLUT = (_cmsStageCLutData*) Stage ->Data;

                CLUT ->Params->dwFlags |= CMS_LERP_FLAGS_TRILINEAR;
                _cmsSetInterpolationRoutine(Lut->ContextID, CLUT ->Params);
            }
    }
}


// Read the DToAX tag, adjusting the encoding of Lab or XYZ if neded
static
cmsPipeline* _cmsReadFloatOutputTag(cmsHPROFILE hProfile, cmsTagSignature tagFloat)
{
    cmsContext ContextID       = cmsGetProfileContextID(hProfile);
    cmsPipeline* Lut           = cmsPipelineDup((cmsPipeline*) cmsReadTag(hProfile, tagFloat));
    cmsColorSpaceSignature PCS = cmsGetPCS(hProfile);
    cmsColorSpaceSignature dataSpace = cmsGetColorSpace(hProfile);
    
    if (Lut == NULL) return NULL;
    
    // If PCS is Lab or XYZ, the floating point tag is accepting data in the space encoding,
    // and since the formatter has already accommodated to 0..1.0, we should undo this change
    if ( PCS == cmsSigLabData)
    {
        if (!cmsPipelineInsertStage(Lut, cmsAT_BEGIN, _cmsStageNormalizeToLabFloat(ContextID)))
            goto Error;
    }
    else
        if (PCS == cmsSigXYZData)
        {
            if (!cmsPipelineInsertStage(Lut, cmsAT_BEGIN, _cmsStageNormalizeToXyzFloat(ContextID)))
                goto Error;
        }
    
    // the output can be Lab or XYZ, in which case normalisation is needed on the end of the pipeline
    if ( dataSpace == cmsSigLabData)
    {
        if (!cmsPipelineInsertStage(Lut, cmsAT_END, _cmsStageNormalizeFromLabFloat(ContextID)))
            goto Error;
    }
    else if (dataSpace == cmsSigXYZData)
    {
        if (!cmsPipelineInsertStage(Lut, cmsAT_END, _cmsStageNormalizeFromXyzFloat(ContextID)))
            goto Error;
    }
    
    return Lut;

Error:
    cmsPipelineFree(Lut);
    return NULL;
}

// Create an output MPE LUT from agiven profile. Version mismatches are handled here
cmsPipeline* CMSEXPORT _cmsReadOutputLUT(cmsHPROFILE hProfile, cmsUInt32Number Intent)
{
    cmsTagTypeSignature OriginalType;
    cmsTagSignature tag16;
    cmsTagSignature tagFloat;
    cmsContext ContextID  = cmsGetProfileContextID(hProfile);


    if (Intent <= INTENT_ABSOLUTE_COLORIMETRIC) {

        tag16 = PCS2Device16[Intent];
        tagFloat = PCS2DeviceFloat[Intent];

        if (cmsIsTag(hProfile, tagFloat)) {  // Float tag takes precedence

            // Floating point LUT are always V4
            return _cmsReadFloatOutputTag(hProfile, tagFloat);
        }

        // Revert to perceptual if no tag is found
        if (!cmsIsTag(hProfile, tag16)) {
            tag16 = PCS2Device16[0];
        }

        if (cmsIsTag(hProfile, tag16)) { // Is there any LUT-Based table?

            // Check profile version and LUT type. Do the necessary adjustments if needed

            // First read the tag
            cmsPipeline* Lut = (cmsPipeline*) cmsReadTag(hProfile, tag16);
            if (Lut == NULL) return NULL;

            // After reading it, we have info about the original type
            OriginalType =  _cmsGetTagTrueType(hProfile, tag16);

            // The profile owns the Lut, so we need to copy it
            Lut = cmsPipelineDup(Lut);
            if (Lut == NULL) return NULL;

            // Now it is time for a controversial stuff. I found that for 3D LUTS using
            // Lab used as indexer space,  trilinear interpolation should be used
            if (cmsGetPCS(hProfile) == cmsSigLabData)
                ChangeInterpolationToTrilinear(Lut);

            // We need to adjust data only for Lab and Lut16 type
            if (OriginalType != cmsSigLut16Type || cmsGetPCS(hProfile) != cmsSigLabData)
                return Lut;

            // Add a matrix for conversion V4 to V2 Lab PCS
            if (!cmsPipelineInsertStage(Lut, cmsAT_BEGIN, _cmsStageAllocLabV4ToV2(ContextID)))
                goto Error;

            // If the output is Lab, add also a conversion at the end
            if (cmsGetColorSpace(hProfile) == cmsSigLabData)
                if (!cmsPipelineInsertStage(Lut, cmsAT_END, _cmsStageAllocLabV2ToV4(ContextID)))
                    goto Error;

            return Lut;
Error:
            cmsPipelineFree(Lut);
            return NULL;
        }
    }

    // Lut not found, try to create a matrix-shaper

    // Check if this is a grayscale profile.
    if (cmsGetColorSpace(hProfile) == cmsSigGrayData) {

        // if so, build appropriate conversion tables.
        // The tables are the PCS iluminant, scaled across GrayTRC
        return BuildGrayOutputPipeline(hProfile);
    }

    // Not gray, create a normal matrix-shaper, which only operates in XYZ space  
    return BuildRGBOutputMatrixShaper(hProfile);
}

// ---------------------------------------------------------------------------------------------------------------

// Read the AToD0 tag, adjusting the encoding of Lab or XYZ if neded
static
cmsPipeline* _cmsReadFloatDevicelinkTag(cmsHPROFILE hProfile, cmsTagSignature tagFloat)
{
    cmsContext ContextID = cmsGetProfileContextID(hProfile);
    cmsPipeline* Lut = cmsPipelineDup((cmsPipeline*)cmsReadTag(hProfile, tagFloat));
    cmsColorSpaceSignature PCS = cmsGetPCS(hProfile);
    cmsColorSpaceSignature spc = cmsGetColorSpace(hProfile);

    if (Lut == NULL) return NULL;

    if (spc == cmsSigLabData)
    {
        if (!cmsPipelineInsertStage(Lut, cmsAT_BEGIN, _cmsStageNormalizeToLabFloat(ContextID)))
            goto Error;
    }
    else
        if (spc == cmsSigXYZData)
        {
            if (!cmsPipelineInsertStage(Lut, cmsAT_BEGIN, _cmsStageNormalizeToXyzFloat(ContextID)))
                goto Error;
        }

    if (PCS == cmsSigLabData)
    {
        if (!cmsPipelineInsertStage(Lut, cmsAT_END, _cmsStageNormalizeFromLabFloat(ContextID)))
            goto Error;
    }
    else
        if (PCS == cmsSigXYZData)
        {
            if (!cmsPipelineInsertStage(Lut, cmsAT_END, _cmsStageNormalizeFromXyzFloat(ContextID)))
                goto Error;
        }

    return Lut;
Error:
    cmsPipelineFree(Lut);
    return NULL;
}

// This one includes abstract profiles as well. Matrix-shaper cannot be obtained on that device class. The
// tag name here may default to AToB0
cmsPipeline* CMSEXPORT _cmsReadDevicelinkLUT(cmsHPROFILE hProfile, cmsUInt32Number Intent)
{
    cmsPipeline* Lut;
    cmsTagTypeSignature OriginalType;
    cmsTagSignature tag16;
    cmsTagSignature tagFloat;
    cmsContext ContextID = cmsGetProfileContextID(hProfile);


    if (Intent > INTENT_ABSOLUTE_COLORIMETRIC)
        return NULL;

    tag16 = Device2PCS16[Intent];
    tagFloat = Device2PCSFloat[Intent];

    // On named color, take the appropriate tag
    if (cmsGetDeviceClass(hProfile) == cmsSigNamedColorClass) {

        cmsNAMEDCOLORLIST* nc = (cmsNAMEDCOLORLIST*)cmsReadTag(hProfile, cmsSigNamedColor2Tag);

        if (nc == NULL) return NULL;

        Lut = cmsPipelineAlloc(ContextID, 0, 0);
        if (Lut == NULL)
            goto Error;

        if (!cmsPipelineInsertStage(Lut, cmsAT_BEGIN, _cmsStageAllocNamedColor(nc, FALSE)))
            goto Error;

        if (cmsGetColorSpace(hProfile) == cmsSigLabData)
            if (!cmsPipelineInsertStage(Lut, cmsAT_END, _cmsStageAllocLabV2ToV4(ContextID)))
                goto Error;

        return Lut;
    Error:
        cmsPipelineFree(Lut);
        cmsFreeNamedColorList(nc);
        return NULL;
    }


    if (cmsIsTag(hProfile, tagFloat)) {  // Float tag takes precedence

        // Floating point LUT are always V
        return _cmsReadFloatDevicelinkTag(hProfile, tagFloat);
    }

    tagFloat = Device2PCSFloat[0];
    if (cmsIsTag(hProfile, tagFloat)) {

        return cmsPipelineDup((cmsPipeline*)cmsReadTag(hProfile, tagFloat));
    }

    if (!cmsIsTag(hProfile, tag16)) {  // Is there any LUT-Based table?

        tag16 = Device2PCS16[0];
        if (!cmsIsTag(hProfile, tag16)) return NULL;
    }

    // Check profile version and LUT type. Do the necessary adjustments if needed

    // Read the tag
    Lut = (cmsPipeline*)cmsReadTag(hProfile, tag16);
    if (Lut == NULL) return NULL;

    // The profile owns the Lut, so we need to copy it
    Lut = cmsPipelineDup(Lut);
    if (Lut == NULL) return NULL;

    // Now it is time for a controversial stuff. I found that for 3D LUTS using
    // Lab used as indexer space,  trilinear interpolation should be used
    if (cmsGetPCS(hProfile) == cmsSigLabData)
        ChangeInterpolationToTrilinear(Lut);

    // After reading it, we have info about the original type
    OriginalType = _cmsGetTagTrueType(hProfile, tag16);

    // We need to adjust data for Lab16 on output
    if (OriginalType != cmsSigLut16Type) return Lut;

    // Here it is possible to get Lab on both sides

    if (cmsGetColorSpace(hProfile) == cmsSigLabData) {
        if (!cmsPipelineInsertStage(Lut, cmsAT_BEGIN, _cmsStageAllocLabV4ToV2(ContextID)))
            goto Error2;
    }

    if (cmsGetPCS(hProfile) == cmsSigLabData) {
        if (!cmsPipelineInsertStage(Lut, cmsAT_END, _cmsStageAllocLabV2ToV4(ContextID)))
            goto Error2;
    }

    return Lut;

Error2:
    cmsPipelineFree(Lut);
    return NULL;
}

// ---------------------------------------------------------------------------------------------------------------

// Returns TRUE if the profile is implemented as matrix-shaper
cmsBool  CMSEXPORT cmsIsMatrixShaper(cmsHPROFILE hProfile)
{
    switch (cmsGetColorSpace(hProfile)) {

    case cmsSigGrayData:

        return cmsIsTag(hProfile, cmsSigGrayTRCTag);

    case cmsSigRgbData:

        return (cmsIsTag(hProfile, cmsSigRedColorantTag) &&
                cmsIsTag(hProfile, cmsSigGreenColorantTag) &&
                cmsIsTag(hProfile, cmsSigBlueColorantTag) &&
                cmsIsTag(hProfile, cmsSigRedTRCTag) &&
                cmsIsTag(hProfile, cmsSigGreenTRCTag) &&
                cmsIsTag(hProfile, cmsSigBlueTRCTag));

    default:

        return FALSE;
    }
}

// Returns TRUE if the intent is implemented as CLUT
cmsBool  CMSEXPORT cmsIsCLUT(cmsHPROFILE hProfile, cmsUInt32Number Intent, cmsUInt32Number UsedDirection)
{
    const cmsTagSignature* TagTable;

    // For devicelinks, the supported intent is that one stated in the header
    if (cmsGetDeviceClass(hProfile) == cmsSigLinkClass) {
            return (cmsGetHeaderRenderingIntent(hProfile) == Intent);
    }

    switch (UsedDirection) {

       case LCMS_USED_AS_INPUT: TagTable = Device2PCS16; break;
       case LCMS_USED_AS_OUTPUT:TagTable = PCS2Device16; break;

       // For proofing, we need rel. colorimetric in output. Let's do some recursion
       case LCMS_USED_AS_PROOF:
           return cmsIsIntentSupported(hProfile, Intent, LCMS_USED_AS_INPUT) &&
                  cmsIsIntentSupported(hProfile, INTENT_RELATIVE_COLORIMETRIC, LCMS_USED_AS_OUTPUT);

       default:
           cmsSignalError(cmsGetProfileContextID(hProfile), cmsERROR_RANGE, "Unexpected direction (%d)", UsedDirection);
           return FALSE;
    }

    return cmsIsTag(hProfile, TagTable[Intent]);

}


// Return info about supported intents
cmsBool  CMSEXPORT cmsIsIntentSupported(cmsHPROFILE hProfile,
                                        cmsUInt32Number Intent, cmsUInt32Number UsedDirection)
{

    if (cmsIsCLUT(hProfile, Intent, UsedDirection)) return TRUE;

    // Is there any matrix-shaper? If so, the intent is supported. This is a bit odd, since V2 matrix shaper
    // does not fully support relative colorimetric because they cannot deal with non-zero black points, but
    // many profiles claims that, and this is certainly not true for V4 profiles. Lets answer "yes" no matter
    // the accuracy would be less than optimal in rel.col and v2 case.

    return cmsIsMatrixShaper(hProfile);
}


// ---------------------------------------------------------------------------------------------------------------

// Read both, profile sequence description and profile sequence id if present. Then combine both to
// create qa unique structure holding both. Shame on ICC to store things in such complicated way.
cmsSEQ* _cmsReadProfileSequence(cmsHPROFILE hProfile)
{
    cmsSEQ* ProfileSeq;
    cmsSEQ* ProfileId;
    cmsSEQ* NewSeq;
    cmsUInt32Number i;

    // Take profile sequence description first
    ProfileSeq = (cmsSEQ*) cmsReadTag(hProfile, cmsSigProfileSequenceDescTag);

    // Take profile sequence ID
    ProfileId  = (cmsSEQ*) cmsReadTag(hProfile, cmsSigProfileSequenceIdTag);

    if (ProfileSeq == NULL && ProfileId == NULL) return NULL;

    if (ProfileSeq == NULL) return cmsDupProfileSequenceDescription(ProfileId);
    if (ProfileId  == NULL) return cmsDupProfileSequenceDescription(ProfileSeq);

    // We have to mix both together. For that they must agree
    if (ProfileSeq ->n != ProfileId ->n) return cmsDupProfileSequenceDescription(ProfileSeq);

    NewSeq = cmsDupProfileSequenceDescription(ProfileSeq);

    // Ok, proceed to the mixing
    if (NewSeq != NULL) {
        for (i=0; i < ProfileSeq ->n; i++) {

            memmove(&NewSeq ->seq[i].ProfileID, &ProfileId ->seq[i].ProfileID, sizeof(cmsProfileID));
            NewSeq ->seq[i].Description = cmsMLUdup(ProfileId ->seq[i].Description);
        }
    }
    return NewSeq;
}

// Dump the contents of profile sequence in both tags (if v4 available)
cmsBool _cmsWriteProfileSequence(cmsHPROFILE hProfile, const cmsSEQ* seq)
{
    if (!cmsWriteTag(hProfile, cmsSigProfileSequenceDescTag, seq)) return FALSE;

    if (cmsGetEncodedICCversion(hProfile) >= 0x4000000) {

            if (!cmsWriteTag(hProfile, cmsSigProfileSequenceIdTag, seq)) return FALSE;
    }

    return TRUE;
}


// Auxiliary, read and duplicate a MLU if found.
static
cmsMLU* GetMLUFromProfile(cmsHPROFILE h, cmsTagSignature sig)
{
    cmsMLU* mlu = (cmsMLU*) cmsReadTag(h, sig);
    if (mlu == NULL) return NULL;

    return cmsMLUdup(mlu);
}

// Create a sequence description out of an array of profiles
cmsSEQ* _cmsCompileProfileSequence(cmsContext ContextID, cmsUInt32Number nProfiles, cmsHPROFILE hProfiles[])
{
    cmsUInt32Number i;
    cmsSEQ* seq = cmsAllocProfileSequenceDescription(ContextID, nProfiles);

    if (seq == NULL) return NULL;

    for (i=0; i < nProfiles; i++) {

        cmsPSEQDESC* ps = &seq ->seq[i];
        cmsHPROFILE h = hProfiles[i];
        cmsTechnologySignature* techpt;

        cmsGetHeaderAttributes(h, &ps ->attributes);
        cmsGetHeaderProfileID(h, ps ->ProfileID.ID8);
        ps ->deviceMfg   = cmsGetHeaderManufacturer(h);
        ps ->deviceModel = cmsGetHeaderModel(h);

        techpt = (cmsTechnologySignature*) cmsReadTag(h, cmsSigTechnologyTag);
        if (techpt == NULL)
            ps ->technology   =  (cmsTechnologySignature) 0;
        else
            ps ->technology   = *techpt;

        ps ->Manufacturer = GetMLUFromProfile(h,  cmsSigDeviceMfgDescTag);
        ps ->Model        = GetMLUFromProfile(h,  cmsSigDeviceModelDescTag);
        ps ->Description  = GetMLUFromProfile(h, cmsSigProfileDescriptionTag);

    }

    return seq;
}

// -------------------------------------------------------------------------------------------------------------------


static
const cmsMLU* GetInfo(cmsHPROFILE hProfile, cmsInfoType Info)
{
    cmsTagSignature sig;

    switch (Info) {

    case cmsInfoDescription:
        sig = cmsSigProfileDescriptionTag;
        break;

    case cmsInfoManufacturer:
        sig = cmsSigDeviceMfgDescTag;
        break;

    case cmsInfoModel:
        sig = cmsSigDeviceModelDescTag;
         break;

    case cmsInfoCopyright:
        sig = cmsSigCopyrightTag;
        break;

    default: return NULL;
    }


    return (cmsMLU*) cmsReadTag(hProfile, sig);
}



cmsUInt32Number CMSEXPORT cmsGetProfileInfo(cmsHPROFILE hProfile, cmsInfoType Info,
                                            const char LanguageCode[3], const char CountryCode[3],
                                            wchar_t* Buffer, cmsUInt32Number BufferSize)
{
    const cmsMLU* mlu = GetInfo(hProfile, Info);
    if (mlu == NULL) return 0;

    return cmsMLUgetWide(mlu, LanguageCode, CountryCode, Buffer, BufferSize);
}


cmsUInt32Number  CMSEXPORT cmsGetProfileInfoASCII(cmsHPROFILE hProfile, cmsInfoType Info,
                                                          const char LanguageCode[3], const char CountryCode[3],
                                                          char* Buffer, cmsUInt32Number BufferSize)
{
    const cmsMLU* mlu = GetInfo(hProfile, Info);
    if (mlu == NULL) return 0;

    return cmsMLUgetASCII(mlu, LanguageCode, CountryCode, Buffer, BufferSize);
}
