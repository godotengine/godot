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

// Virtual (built-in) profiles
// -----------------------------------------------------------------------------------

static
cmsBool SetTextTags(cmsHPROFILE hProfile, const wchar_t* Description)
{
    cmsMLU *DescriptionMLU, *CopyrightMLU;
    cmsBool  rc = FALSE;
    cmsContext ContextID = cmsGetProfileContextID(hProfile);

    DescriptionMLU  = cmsMLUalloc(ContextID, 1);
    CopyrightMLU    = cmsMLUalloc(ContextID, 1);

    if (DescriptionMLU == NULL || CopyrightMLU == NULL) goto Error;

    if (!cmsMLUsetWide(DescriptionMLU,  "en", "US", Description)) goto Error;
    if (!cmsMLUsetWide(CopyrightMLU,    "en", "US", L"No copyright, use freely")) goto Error;

    if (!cmsWriteTag(hProfile, cmsSigProfileDescriptionTag,  DescriptionMLU)) goto Error;
    if (!cmsWriteTag(hProfile, cmsSigCopyrightTag,           CopyrightMLU)) goto Error;

    rc = TRUE;

Error:

    if (DescriptionMLU)
        cmsMLUfree(DescriptionMLU);
    if (CopyrightMLU)
        cmsMLUfree(CopyrightMLU);
    return rc;
}


static
cmsBool  SetSeqDescTag(cmsHPROFILE hProfile, const char* Model)
{
    cmsBool  rc = FALSE;
    cmsContext ContextID = cmsGetProfileContextID(hProfile);
    cmsSEQ* Seq = cmsAllocProfileSequenceDescription(ContextID, 1);

    if (Seq == NULL) return FALSE;

    Seq->seq[0].deviceMfg = (cmsSignature) 0;
    Seq->seq[0].deviceModel = (cmsSignature) 0;

#ifdef CMS_DONT_USE_INT64
    Seq->seq[0].attributes[0] = 0;
    Seq->seq[0].attributes[1] = 0;
#else
    Seq->seq[0].attributes = 0;
#endif

    Seq->seq[0].technology = (cmsTechnologySignature) 0;

    cmsMLUsetASCII( Seq->seq[0].Manufacturer, cmsNoLanguage, cmsNoCountry, "Little CMS");
    cmsMLUsetASCII( Seq->seq[0].Model,        cmsNoLanguage, cmsNoCountry, Model);

    if (!_cmsWriteProfileSequence(hProfile, Seq)) goto Error;

    rc = TRUE;

Error:
    if (Seq)
        cmsFreeProfileSequenceDescription(Seq);

    return rc;
}



// This function creates a profile based on White point, primaries and
// transfer functions.
cmsHPROFILE CMSEXPORT cmsCreateRGBProfileTHR(cmsContext ContextID,
                                          const cmsCIExyY* WhitePoint,
                                          const cmsCIExyYTRIPLE* Primaries,
                                          cmsToneCurve* const TransferFunction[3])
{
    cmsHPROFILE hICC;
    cmsMAT3 MColorants;
    cmsCIEXYZTRIPLE Colorants;
    cmsCIExyY MaxWhite;
    cmsMAT3 CHAD;
    cmsCIEXYZ WhitePointXYZ;

    hICC = cmsCreateProfilePlaceholder(ContextID);
    if (!hICC)                          // can't allocate
        return NULL;

    cmsSetProfileVersion(hICC, 4.3);

    cmsSetDeviceClass(hICC,      cmsSigDisplayClass);
    cmsSetColorSpace(hICC,       cmsSigRgbData);
    cmsSetPCS(hICC,              cmsSigXYZData);

    cmsSetHeaderRenderingIntent(hICC,  INTENT_PERCEPTUAL);


    // Implement profile using following tags:
    //
    //  1 cmsSigProfileDescriptionTag
    //  2 cmsSigMediaWhitePointTag
    //  3 cmsSigRedColorantTag
    //  4 cmsSigGreenColorantTag
    //  5 cmsSigBlueColorantTag
    //  6 cmsSigRedTRCTag
    //  7 cmsSigGreenTRCTag
    //  8 cmsSigBlueTRCTag
    //  9 Chromatic adaptation Tag
    // This conforms a standard RGB DisplayProfile as says ICC, and then I add (As per addendum II)
    // 10 cmsSigChromaticityTag


    if (!SetTextTags(hICC, L"RGB built-in")) goto Error;

    if (WhitePoint) {

        if (!cmsWriteTag(hICC, cmsSigMediaWhitePointTag, cmsD50_XYZ())) goto Error;

        cmsxyY2XYZ(&WhitePointXYZ, WhitePoint);
        _cmsAdaptationMatrix(&CHAD, NULL, &WhitePointXYZ, cmsD50_XYZ());

        // This is a V4 tag, but many CMM does read and understand it no matter which version
        if (!cmsWriteTag(hICC, cmsSigChromaticAdaptationTag, (void*) &CHAD)) goto Error;
    }

    if (WhitePoint && Primaries) {

        MaxWhite.x =  WhitePoint -> x;
        MaxWhite.y =  WhitePoint -> y;
        MaxWhite.Y =  1.0;

        if (!_cmsBuildRGB2XYZtransferMatrix(&MColorants, &MaxWhite, Primaries)) goto Error;

        Colorants.Red.X   = MColorants.v[0].n[0];
        Colorants.Red.Y   = MColorants.v[1].n[0];
        Colorants.Red.Z   = MColorants.v[2].n[0];

        Colorants.Green.X = MColorants.v[0].n[1];
        Colorants.Green.Y = MColorants.v[1].n[1];
        Colorants.Green.Z = MColorants.v[2].n[1];

        Colorants.Blue.X  = MColorants.v[0].n[2];
        Colorants.Blue.Y  = MColorants.v[1].n[2];
        Colorants.Blue.Z  = MColorants.v[2].n[2];

        if (!cmsWriteTag(hICC, cmsSigRedColorantTag,   (void*) &Colorants.Red)) goto Error;
        if (!cmsWriteTag(hICC, cmsSigBlueColorantTag,  (void*) &Colorants.Blue)) goto Error;
        if (!cmsWriteTag(hICC, cmsSigGreenColorantTag, (void*) &Colorants.Green)) goto Error;
    }


    if (TransferFunction) {

        // Tries to minimize space. Thanks to Richard Hughes for this nice idea         
        if (!cmsWriteTag(hICC, cmsSigRedTRCTag,   (void*) TransferFunction[0])) goto Error;

        if (TransferFunction[1] == TransferFunction[0]) {

            if (!cmsLinkTag (hICC, cmsSigGreenTRCTag, cmsSigRedTRCTag)) goto Error;

        } else {

            if (!cmsWriteTag(hICC, cmsSigGreenTRCTag, (void*) TransferFunction[1])) goto Error;
        }

        if (TransferFunction[2] == TransferFunction[0]) {

            if (!cmsLinkTag (hICC, cmsSigBlueTRCTag, cmsSigRedTRCTag)) goto Error;

        } else {

            if (!cmsWriteTag(hICC, cmsSigBlueTRCTag, (void*) TransferFunction[2])) goto Error;
        }
    }

    if (Primaries) {
        if (!cmsWriteTag(hICC, cmsSigChromaticityTag, (void*) Primaries)) goto Error;
    }


    return hICC;

Error:
    if (hICC)
        cmsCloseProfile(hICC);
    return NULL;
}

cmsHPROFILE CMSEXPORT cmsCreateRGBProfile(const cmsCIExyY* WhitePoint,
                                          const cmsCIExyYTRIPLE* Primaries,
                                          cmsToneCurve* const TransferFunction[3])
{
    return cmsCreateRGBProfileTHR(NULL, WhitePoint, Primaries, TransferFunction);
}



// This function creates a profile based on White point and transfer function.
cmsHPROFILE CMSEXPORT cmsCreateGrayProfileTHR(cmsContext ContextID,
                                           const cmsCIExyY* WhitePoint,
                                           const cmsToneCurve* TransferFunction)
{
    cmsHPROFILE hICC;
    cmsCIEXYZ tmp;

    hICC = cmsCreateProfilePlaceholder(ContextID);
    if (!hICC)                          // can't allocate
        return NULL;

    cmsSetProfileVersion(hICC, 4.3);

    cmsSetDeviceClass(hICC,      cmsSigDisplayClass);
    cmsSetColorSpace(hICC,       cmsSigGrayData);
    cmsSetPCS(hICC,              cmsSigXYZData);
    cmsSetHeaderRenderingIntent(hICC,  INTENT_PERCEPTUAL);


    // Implement profile using following tags:
    //
    //  1 cmsSigProfileDescriptionTag
    //  2 cmsSigMediaWhitePointTag
    //  3 cmsSigGrayTRCTag

    // This conforms a standard Gray DisplayProfile

    // Fill-in the tags

    if (!SetTextTags(hICC, L"gray built-in")) goto Error;


    if (WhitePoint) {

        cmsxyY2XYZ(&tmp, WhitePoint);
        if (!cmsWriteTag(hICC, cmsSigMediaWhitePointTag, (void*) &tmp)) goto Error;
    }

    if (TransferFunction) {

        if (!cmsWriteTag(hICC, cmsSigGrayTRCTag, (void*) TransferFunction)) goto Error;
    }

    return hICC;

Error:
    if (hICC)
        cmsCloseProfile(hICC);
    return NULL;
}



cmsHPROFILE CMSEXPORT cmsCreateGrayProfile(const cmsCIExyY* WhitePoint,
                                                    const cmsToneCurve* TransferFunction)
{
    return cmsCreateGrayProfileTHR(NULL, WhitePoint, TransferFunction);
}

// This is a devicelink operating in the target colorspace with as many transfer functions as components

cmsHPROFILE CMSEXPORT cmsCreateLinearizationDeviceLinkTHR(cmsContext ContextID,
                                                          cmsColorSpaceSignature ColorSpace,
                                                          cmsToneCurve* const TransferFunctions[])
{
    cmsHPROFILE hICC;
    cmsPipeline* Pipeline;
    cmsUInt32Number nChannels;

    hICC = cmsCreateProfilePlaceholder(ContextID);
    if (!hICC)
        return NULL;

    cmsSetProfileVersion(hICC, 4.3);

    cmsSetDeviceClass(hICC,      cmsSigLinkClass);
    cmsSetColorSpace(hICC,       ColorSpace);
    cmsSetPCS(hICC,              ColorSpace);

    cmsSetHeaderRenderingIntent(hICC,  INTENT_PERCEPTUAL);

    // Set up channels
    nChannels = cmsChannelsOf(ColorSpace);

    // Creates a Pipeline with prelinearization step only
    Pipeline = cmsPipelineAlloc(ContextID, nChannels, nChannels);
    if (Pipeline == NULL) goto Error;


    // Copy tables to Pipeline
    if (!cmsPipelineInsertStage(Pipeline, cmsAT_BEGIN, cmsStageAllocToneCurves(ContextID, nChannels, TransferFunctions)))
        goto Error;

    // Create tags
    if (!SetTextTags(hICC, L"Linearization built-in")) goto Error;
    if (!cmsWriteTag(hICC, cmsSigAToB0Tag, (void*) Pipeline)) goto Error;
    if (!SetSeqDescTag(hICC, "Linearization built-in")) goto Error;

    // Pipeline is already on virtual profile
    cmsPipelineFree(Pipeline);

    // Ok, done
    return hICC;

Error:
    cmsPipelineFree(Pipeline);
    if (hICC)
        cmsCloseProfile(hICC);


    return NULL;
}

cmsHPROFILE CMSEXPORT cmsCreateLinearizationDeviceLink(cmsColorSpaceSignature ColorSpace,
                                                                 cmsToneCurve* const TransferFunctions[])
{
    return cmsCreateLinearizationDeviceLinkTHR(NULL, ColorSpace, TransferFunctions);
}

// Ink-limiting algorithm
//
//  Sum = C + M + Y + K
//  If Sum > InkLimit
//        Ratio= 1 - (Sum - InkLimit) / (C + M + Y)
//        if Ratio <0
//              Ratio=0
//        endif
//     Else
//         Ratio=1
//     endif
//
//     C = Ratio * C
//     M = Ratio * M
//     Y = Ratio * Y
//     K: Does not change

static
int InkLimitingSampler(register const cmsUInt16Number In[], register cmsUInt16Number Out[], register void* Cargo)
{
    cmsFloat64Number InkLimit = *(cmsFloat64Number *) Cargo;
    cmsFloat64Number SumCMY, SumCMYK, Ratio;

    InkLimit = (InkLimit * 655.35);

    SumCMY   = In[0]  + In[1] + In[2];
    SumCMYK  = SumCMY + In[3];

    if (SumCMYK > InkLimit) {

        Ratio = 1 - ((SumCMYK - InkLimit) / SumCMY);
        if (Ratio < 0)
            Ratio = 0;
    }
    else Ratio = 1;

    Out[0] = _cmsQuickSaturateWord(In[0] * Ratio);     // C
    Out[1] = _cmsQuickSaturateWord(In[1] * Ratio);     // M
    Out[2] = _cmsQuickSaturateWord(In[2] * Ratio);     // Y

    Out[3] = In[3];                                 // K (untouched)

    return TRUE;
}

// This is a devicelink operating in CMYK for ink-limiting

cmsHPROFILE CMSEXPORT cmsCreateInkLimitingDeviceLinkTHR(cmsContext ContextID,
                                                     cmsColorSpaceSignature ColorSpace,
                                                     cmsFloat64Number Limit)
{
    cmsHPROFILE hICC;
    cmsPipeline* LUT;
    cmsStage* CLUT;
    cmsUInt32Number nChannels;

    if (ColorSpace != cmsSigCmykData) {
        cmsSignalError(ContextID, cmsERROR_COLORSPACE_CHECK, "InkLimiting: Only CMYK currently supported");
        return NULL;
    }

    if (Limit < 0.0 || Limit > 400) {

        cmsSignalError(ContextID, cmsERROR_RANGE, "InkLimiting: Limit should be between 0..400");
        if (Limit < 0) Limit = 0;
        if (Limit > 400) Limit = 400;

    }

    hICC = cmsCreateProfilePlaceholder(ContextID);
    if (!hICC)                          // can't allocate
        return NULL;

    cmsSetProfileVersion(hICC, 4.3);

    cmsSetDeviceClass(hICC,      cmsSigLinkClass);
    cmsSetColorSpace(hICC,       ColorSpace);
    cmsSetPCS(hICC,              ColorSpace);

    cmsSetHeaderRenderingIntent(hICC,  INTENT_PERCEPTUAL);


    // Creates a Pipeline with 3D grid only
    LUT = cmsPipelineAlloc(ContextID, 4, 4);
    if (LUT == NULL) goto Error;


    nChannels = cmsChannelsOf(ColorSpace);

    CLUT = cmsStageAllocCLut16bit(ContextID, 17, nChannels, nChannels, NULL);
    if (CLUT == NULL) goto Error;

    if (!cmsStageSampleCLut16bit(CLUT, InkLimitingSampler, (void*) &Limit, 0)) goto Error;

    if (!cmsPipelineInsertStage(LUT, cmsAT_BEGIN, _cmsStageAllocIdentityCurves(ContextID, nChannels)) ||
        !cmsPipelineInsertStage(LUT, cmsAT_END, CLUT) ||
        !cmsPipelineInsertStage(LUT, cmsAT_END, _cmsStageAllocIdentityCurves(ContextID, nChannels)))
        goto Error;

    // Create tags
    if (!SetTextTags(hICC, L"ink-limiting built-in")) goto Error;

    if (!cmsWriteTag(hICC, cmsSigAToB0Tag, (void*) LUT))  goto Error;
    if (!SetSeqDescTag(hICC, "ink-limiting built-in")) goto Error;

    // cmsPipeline is already on virtual profile
    cmsPipelineFree(LUT);

    // Ok, done
    return hICC;

Error:
    if (LUT != NULL)
        cmsPipelineFree(LUT);

    if (hICC != NULL)
        cmsCloseProfile(hICC);

    return NULL;
}

cmsHPROFILE CMSEXPORT cmsCreateInkLimitingDeviceLink(cmsColorSpaceSignature ColorSpace, cmsFloat64Number Limit)
{
    return cmsCreateInkLimitingDeviceLinkTHR(NULL, ColorSpace, Limit);
}


// Creates a fake Lab identity.
cmsHPROFILE CMSEXPORT cmsCreateLab2ProfileTHR(cmsContext ContextID, const cmsCIExyY* WhitePoint)
{
    cmsHPROFILE hProfile;
    cmsPipeline* LUT = NULL;

    hProfile = cmsCreateRGBProfileTHR(ContextID, WhitePoint == NULL ? cmsD50_xyY() : WhitePoint, NULL, NULL);
    if (hProfile == NULL) return NULL;

    cmsSetProfileVersion(hProfile, 2.1);

    cmsSetDeviceClass(hProfile, cmsSigAbstractClass);
    cmsSetColorSpace(hProfile,  cmsSigLabData);
    cmsSetPCS(hProfile,         cmsSigLabData);

    if (!SetTextTags(hProfile, L"Lab identity built-in")) return NULL;

    // An identity LUT is all we need
    LUT = cmsPipelineAlloc(ContextID, 3, 3);
    if (LUT == NULL) goto Error;

    if (!cmsPipelineInsertStage(LUT, cmsAT_BEGIN, _cmsStageAllocIdentityCLut(ContextID, 3)))
        goto Error;

    if (!cmsWriteTag(hProfile, cmsSigAToB0Tag, LUT)) goto Error;
    cmsPipelineFree(LUT);

    return hProfile;

Error:

    if (LUT != NULL)
        cmsPipelineFree(LUT);

    if (hProfile != NULL)
        cmsCloseProfile(hProfile);

    return NULL;
}


cmsHPROFILE CMSEXPORT cmsCreateLab2Profile(const cmsCIExyY* WhitePoint)
{
    return cmsCreateLab2ProfileTHR(NULL, WhitePoint);
}


// Creates a fake Lab V4 identity.
cmsHPROFILE CMSEXPORT cmsCreateLab4ProfileTHR(cmsContext ContextID, const cmsCIExyY* WhitePoint)
{
    cmsHPROFILE hProfile;
    cmsPipeline* LUT = NULL;

    hProfile = cmsCreateRGBProfileTHR(ContextID, WhitePoint == NULL ? cmsD50_xyY() : WhitePoint, NULL, NULL);
    if (hProfile == NULL) return NULL;

    cmsSetProfileVersion(hProfile, 4.3);

    cmsSetDeviceClass(hProfile, cmsSigAbstractClass);
    cmsSetColorSpace(hProfile,  cmsSigLabData);
    cmsSetPCS(hProfile,         cmsSigLabData);

    if (!SetTextTags(hProfile, L"Lab identity built-in")) goto Error;

    // An empty LUTs is all we need
    LUT = cmsPipelineAlloc(ContextID, 3, 3);
    if (LUT == NULL) goto Error;

    if (!cmsPipelineInsertStage(LUT, cmsAT_BEGIN, _cmsStageAllocIdentityCurves(ContextID, 3)))
        goto Error;

    if (!cmsWriteTag(hProfile, cmsSigAToB0Tag, LUT)) goto Error;
    cmsPipelineFree(LUT);

    return hProfile;

Error:

    if (LUT != NULL)
        cmsPipelineFree(LUT);

    if (hProfile != NULL)
        cmsCloseProfile(hProfile);

    return NULL;
}

cmsHPROFILE CMSEXPORT cmsCreateLab4Profile(const cmsCIExyY* WhitePoint)
{
    return cmsCreateLab4ProfileTHR(NULL, WhitePoint);
}


// Creates a fake XYZ identity
cmsHPROFILE CMSEXPORT cmsCreateXYZProfileTHR(cmsContext ContextID)
{
    cmsHPROFILE hProfile;
    cmsPipeline* LUT = NULL;

    hProfile = cmsCreateRGBProfileTHR(ContextID, cmsD50_xyY(), NULL, NULL);
    if (hProfile == NULL) return NULL;

    cmsSetProfileVersion(hProfile, 4.3);

    cmsSetDeviceClass(hProfile, cmsSigAbstractClass);
    cmsSetColorSpace(hProfile,  cmsSigXYZData);
    cmsSetPCS(hProfile,         cmsSigXYZData);

    if (!SetTextTags(hProfile, L"XYZ identity built-in")) goto Error;

    // An identity LUT is all we need
    LUT = cmsPipelineAlloc(ContextID, 3, 3);
    if (LUT == NULL) goto Error;

    if (!cmsPipelineInsertStage(LUT, cmsAT_BEGIN, _cmsStageAllocIdentityCurves(ContextID, 3)))
        goto Error;

    if (!cmsWriteTag(hProfile, cmsSigAToB0Tag, LUT)) goto Error;
    cmsPipelineFree(LUT);

    return hProfile;

Error:

    if (LUT != NULL)
        cmsPipelineFree(LUT);

    if (hProfile != NULL)
        cmsCloseProfile(hProfile);

    return NULL;
}


cmsHPROFILE CMSEXPORT cmsCreateXYZProfile(void)
{
    return cmsCreateXYZProfileTHR(NULL);
}


//sRGB Curves are defined by:
//
//If  R’sRGB,G’sRGB, B’sRGB < 0.04045
//
//    R =  R’sRGB / 12.92
//    G =  G’sRGB / 12.92
//    B =  B’sRGB / 12.92
//
//
//else if  R’sRGB,G’sRGB, B’sRGB >= 0.04045
//
//    R = ((R’sRGB + 0.055) / 1.055)^2.4
//    G = ((G’sRGB + 0.055) / 1.055)^2.4
//    B = ((B’sRGB + 0.055) / 1.055)^2.4

static
cmsToneCurve* Build_sRGBGamma(cmsContext ContextID)
{
    cmsFloat64Number Parameters[5];

    Parameters[0] = 2.4;
    Parameters[1] = 1. / 1.055;
    Parameters[2] = 0.055 / 1.055;
    Parameters[3] = 1. / 12.92;
    Parameters[4] = 0.04045;

    return cmsBuildParametricToneCurve(ContextID, 4, Parameters);
}

// Create the ICC virtual profile for sRGB space
cmsHPROFILE CMSEXPORT cmsCreate_sRGBProfileTHR(cmsContext ContextID)
{
       cmsCIExyY       D65 = { 0.3127, 0.3290, 1.0 };
       cmsCIExyYTRIPLE Rec709Primaries = {
                                   {0.6400, 0.3300, 1.0},
                                   {0.3000, 0.6000, 1.0},
                                   {0.1500, 0.0600, 1.0}
                                   };
       cmsToneCurve* Gamma22[3];
       cmsHPROFILE  hsRGB;

      // cmsWhitePointFromTemp(&D65, 6504);
       Gamma22[0] = Gamma22[1] = Gamma22[2] = Build_sRGBGamma(ContextID);
       if (Gamma22[0] == NULL) return NULL;

       hsRGB = cmsCreateRGBProfileTHR(ContextID, &D65, &Rec709Primaries, Gamma22);
       cmsFreeToneCurve(Gamma22[0]);
       if (hsRGB == NULL) return NULL;

       if (!SetTextTags(hsRGB, L"sRGB built-in")) {
           cmsCloseProfile(hsRGB);
           return NULL;
       }

       return hsRGB;
}

cmsHPROFILE CMSEXPORT cmsCreate_sRGBProfile(void)
{
    return cmsCreate_sRGBProfileTHR(NULL);
}



typedef struct {
                cmsFloat64Number Brightness;
                cmsFloat64Number Contrast;
                cmsFloat64Number Hue;
                cmsFloat64Number Saturation;
                cmsBool          lAdjustWP;
                cmsCIEXYZ WPsrc, WPdest;

} BCHSWADJUSTS, *LPBCHSWADJUSTS;


static
int bchswSampler(register const cmsUInt16Number In[], register cmsUInt16Number Out[], register void* Cargo)
{
    cmsCIELab LabIn, LabOut;
    cmsCIELCh LChIn, LChOut;
    cmsCIEXYZ XYZ;
    LPBCHSWADJUSTS bchsw = (LPBCHSWADJUSTS) Cargo;


    cmsLabEncoded2Float(&LabIn, In);


    cmsLab2LCh(&LChIn, &LabIn);

    // Do some adjusts on LCh

    LChOut.L = LChIn.L * bchsw ->Contrast + bchsw ->Brightness;
    LChOut.C = LChIn.C + bchsw -> Saturation;
    LChOut.h = LChIn.h + bchsw -> Hue;


    cmsLCh2Lab(&LabOut, &LChOut);

    // Move white point in Lab
    if (bchsw->lAdjustWP) {
           cmsLab2XYZ(&bchsw->WPsrc, &XYZ, &LabOut);
           cmsXYZ2Lab(&bchsw->WPdest, &LabOut, &XYZ);
    }

    // Back to encoded

    cmsFloat2LabEncoded(Out, &LabOut);

    return TRUE;
}


// Creates an abstract profile operating in Lab space for Brightness,
// contrast, Saturation and white point displacement

cmsHPROFILE CMSEXPORT cmsCreateBCHSWabstractProfileTHR(cmsContext ContextID,
                                                       cmsUInt32Number nLUTPoints,
                                                       cmsFloat64Number Bright,
                                                       cmsFloat64Number Contrast,
                                                       cmsFloat64Number Hue,
                                                       cmsFloat64Number Saturation,
                                                       cmsUInt32Number TempSrc,
                                                       cmsUInt32Number TempDest)
{
    cmsHPROFILE hICC;
    cmsPipeline* Pipeline;
    BCHSWADJUSTS bchsw;
    cmsCIExyY WhitePnt;
    cmsStage* CLUT;
    cmsUInt32Number Dimensions[MAX_INPUT_DIMENSIONS];
    cmsUInt32Number i;

    bchsw.Brightness = Bright;
    bchsw.Contrast   = Contrast;
    bchsw.Hue        = Hue;
    bchsw.Saturation = Saturation;
    if (TempSrc == TempDest) {

           bchsw.lAdjustWP = FALSE;
    }
    else {
           bchsw.lAdjustWP = TRUE;
           cmsWhitePointFromTemp(&WhitePnt, TempSrc);
           cmsxyY2XYZ(&bchsw.WPsrc, &WhitePnt);
           cmsWhitePointFromTemp(&WhitePnt, TempDest);
           cmsxyY2XYZ(&bchsw.WPdest, &WhitePnt);
     
    }

    hICC = cmsCreateProfilePlaceholder(ContextID);
    if (!hICC)                          // can't allocate
        return NULL;

    cmsSetDeviceClass(hICC,      cmsSigAbstractClass);
    cmsSetColorSpace(hICC,       cmsSigLabData);
    cmsSetPCS(hICC,              cmsSigLabData);

    cmsSetHeaderRenderingIntent(hICC,  INTENT_PERCEPTUAL);

    // Creates a Pipeline with 3D grid only
    Pipeline = cmsPipelineAlloc(ContextID, 3, 3);
    if (Pipeline == NULL) {
        cmsCloseProfile(hICC);
        return NULL;
    }

    for (i=0; i < MAX_INPUT_DIMENSIONS; i++) Dimensions[i] = nLUTPoints;
    CLUT = cmsStageAllocCLut16bitGranular(ContextID, Dimensions, 3, 3, NULL);
    if (CLUT == NULL) goto Error;


    if (!cmsStageSampleCLut16bit(CLUT, bchswSampler, (void*) &bchsw, 0)) {

        // Shouldn't reach here
        goto Error;
    }

    if (!cmsPipelineInsertStage(Pipeline, cmsAT_END, CLUT)) {
        goto Error;
    }

    // Create tags
    if (!SetTextTags(hICC, L"BCHS built-in")) return NULL;

    cmsWriteTag(hICC, cmsSigMediaWhitePointTag, (void*) cmsD50_XYZ());

    cmsWriteTag(hICC, cmsSigAToB0Tag, (void*) Pipeline);

    // Pipeline is already on virtual profile
    cmsPipelineFree(Pipeline);

    // Ok, done
    return hICC;

Error:
    cmsPipelineFree(Pipeline);
    cmsCloseProfile(hICC);
    return NULL;
}


CMSAPI cmsHPROFILE   CMSEXPORT cmsCreateBCHSWabstractProfile(cmsUInt32Number nLUTPoints,
                                                             cmsFloat64Number Bright,
                                                             cmsFloat64Number Contrast,
                                                             cmsFloat64Number Hue,
                                                             cmsFloat64Number Saturation,
                                                             cmsUInt32Number TempSrc,
                                                             cmsUInt32Number TempDest)
{
    return cmsCreateBCHSWabstractProfileTHR(NULL, nLUTPoints, Bright, Contrast, Hue, Saturation, TempSrc, TempDest);
}


// Creates a fake NULL profile. This profile return 1 channel as always 0.
// Is useful only for gamut checking tricks
cmsHPROFILE CMSEXPORT cmsCreateNULLProfileTHR(cmsContext ContextID)
{
    cmsHPROFILE hProfile;
    cmsPipeline* LUT = NULL;
    cmsStage* PostLin;
    cmsStage* OutLin;
    cmsToneCurve* EmptyTab[3];
    cmsUInt16Number Zero[2] = { 0, 0 };
    const cmsFloat64Number PickLstarMatrix[] = { 1, 0, 0 };

    hProfile = cmsCreateProfilePlaceholder(ContextID);
    if (!hProfile)                          // can't allocate
        return NULL;

    cmsSetProfileVersion(hProfile, 4.3);

    if (!SetTextTags(hProfile, L"NULL profile built-in")) goto Error;


    cmsSetDeviceClass(hProfile, cmsSigOutputClass);
    cmsSetColorSpace(hProfile,  cmsSigGrayData);
    cmsSetPCS(hProfile,         cmsSigLabData);

    // Create a valid ICC 4 structure
    LUT = cmsPipelineAlloc(ContextID, 3, 1);
    if (LUT == NULL) goto Error;
    
    EmptyTab[0] = EmptyTab[1] = EmptyTab[2] = cmsBuildTabulatedToneCurve16(ContextID, 2, Zero);
    PostLin = cmsStageAllocToneCurves(ContextID, 3, EmptyTab);
    OutLin  = cmsStageAllocToneCurves(ContextID, 1, EmptyTab);
    cmsFreeToneCurve(EmptyTab[0]);

    if (!cmsPipelineInsertStage(LUT, cmsAT_END, PostLin))
        goto Error;

    if (!cmsPipelineInsertStage(LUT, cmsAT_END, cmsStageAllocMatrix(ContextID, 1, 3, PickLstarMatrix, NULL)))
        goto Error;

    if (!cmsPipelineInsertStage(LUT, cmsAT_END, OutLin))
        goto Error;

    if (!cmsWriteTag(hProfile, cmsSigBToA0Tag, (void*) LUT)) goto Error;
    if (!cmsWriteTag(hProfile, cmsSigMediaWhitePointTag, cmsD50_XYZ())) goto Error;

    cmsPipelineFree(LUT);
    return hProfile;

Error:

    if (LUT != NULL)
        cmsPipelineFree(LUT);

    if (hProfile != NULL)
        cmsCloseProfile(hProfile);

    return NULL;
}

cmsHPROFILE CMSEXPORT cmsCreateNULLProfile(void)
{
    return cmsCreateNULLProfileTHR(NULL);
}


static
int IsPCS(cmsColorSpaceSignature ColorSpace)
{
    return (ColorSpace == cmsSigXYZData ||
            ColorSpace == cmsSigLabData);
}


static
void FixColorSpaces(cmsHPROFILE hProfile,
                              cmsColorSpaceSignature ColorSpace,
                              cmsColorSpaceSignature PCS,
                              cmsUInt32Number dwFlags)
{
    if (dwFlags & cmsFLAGS_GUESSDEVICECLASS) {

            if (IsPCS(ColorSpace) && IsPCS(PCS)) {

                    cmsSetDeviceClass(hProfile,      cmsSigAbstractClass);
                    cmsSetColorSpace(hProfile,       ColorSpace);
                    cmsSetPCS(hProfile,              PCS);
                    return;
            }

            if (IsPCS(ColorSpace) && !IsPCS(PCS)) {

                    cmsSetDeviceClass(hProfile, cmsSigOutputClass);
                    cmsSetPCS(hProfile,         ColorSpace);
                    cmsSetColorSpace(hProfile,  PCS);
                    return;
            }

            if (IsPCS(PCS) && !IsPCS(ColorSpace)) {

                   cmsSetDeviceClass(hProfile,  cmsSigInputClass);
                   cmsSetColorSpace(hProfile,   ColorSpace);
                   cmsSetPCS(hProfile,          PCS);
                   return;
            }
    }

    cmsSetDeviceClass(hProfile,      cmsSigLinkClass);
    cmsSetColorSpace(hProfile,       ColorSpace);
    cmsSetPCS(hProfile,              PCS);
}



// This function creates a named color profile dumping all the contents of transform to a single profile
// In this way, LittleCMS may be used to "group" several named color databases into a single profile.
// It has, however, several minor limitations. PCS is always Lab, which is not very critic since this
// is the normal PCS for named color profiles.
static
cmsHPROFILE CreateNamedColorDevicelink(cmsHTRANSFORM xform)
{
    _cmsTRANSFORM* v = (_cmsTRANSFORM*) xform;
    cmsHPROFILE hICC = NULL;
    cmsUInt32Number i, nColors;
    cmsNAMEDCOLORLIST *nc2 = NULL, *Original = NULL;

    // Create an empty placeholder
    hICC = cmsCreateProfilePlaceholder(v->ContextID);
    if (hICC == NULL) return NULL;

    // Critical information
    cmsSetDeviceClass(hICC, cmsSigNamedColorClass);
    cmsSetColorSpace(hICC, v ->ExitColorSpace);
    cmsSetPCS(hICC, cmsSigLabData);

    // Tag profile with information
    if (!SetTextTags(hICC, L"Named color devicelink")) goto Error;

    Original = cmsGetNamedColorList(xform);
    if (Original == NULL) goto Error;

    nColors = cmsNamedColorCount(Original);
    nc2     = cmsDupNamedColorList(Original);
    if (nc2 == NULL) goto Error;

    // Colorant count now depends on the output space
    nc2 ->ColorantCount = cmsPipelineOutputChannels(v ->Lut);

    // Make sure we have proper formatters
    cmsChangeBuffersFormat(xform, TYPE_NAMED_COLOR_INDEX,
        FLOAT_SH(0) | COLORSPACE_SH(_cmsLCMScolorSpace(v ->ExitColorSpace))
        | BYTES_SH(2) | CHANNELS_SH(cmsChannelsOf(v ->ExitColorSpace)));

    // Apply the transfor to colorants.
    for (i=0; i < nColors; i++) {
        cmsDoTransform(xform, &i, nc2 ->List[i].DeviceColorant, 1);
    }

    if (!cmsWriteTag(hICC, cmsSigNamedColor2Tag, (void*) nc2)) goto Error;
    cmsFreeNamedColorList(nc2);

    return hICC;

Error:
    if (hICC != NULL) cmsCloseProfile(hICC);
    return NULL;
}


// This structure holds information about which MPU can be stored on a profile based on the version

typedef struct {
    cmsBool              IsV4;             // Is a V4 tag?
    cmsTagSignature      RequiredTag;      // Set to 0 for both types
    cmsTagTypeSignature  LutType;          // The LUT type
    int                  nTypes;           // Number of types (up to 5)
    cmsStageSignature    MpeTypes[5];      // 5 is the maximum number

} cmsAllowedLUT;

#define cmsSig0 ((cmsTagSignature) 0) 

static const cmsAllowedLUT AllowedLUTTypes[] = {

    { FALSE, cmsSig0,        cmsSigLut16Type, 4, { cmsSigMatrixElemType, cmsSigCurveSetElemType, cmsSigCLutElemType, cmsSigCurveSetElemType } },
    { FALSE, cmsSig0,        cmsSigLut16Type, 3, { cmsSigCurveSetElemType, cmsSigCLutElemType, cmsSigCurveSetElemType } },
    { FALSE, cmsSig0,        cmsSigLut16Type, 2, { cmsSigCurveSetElemType, cmsSigCLutElemType } },
    { TRUE,  cmsSig0,        cmsSigLutAtoBType, 1, { cmsSigCurveSetElemType } },
    { TRUE , cmsSigAToB0Tag, cmsSigLutAtoBType,  3,  { cmsSigCurveSetElemType, cmsSigMatrixElemType, cmsSigCurveSetElemType } },
    { TRUE , cmsSigAToB0Tag, cmsSigLutAtoBType,  3,  { cmsSigCurveSetElemType, cmsSigCLutElemType, cmsSigCurveSetElemType   } },
    { TRUE , cmsSigAToB0Tag, cmsSigLutAtoBType,  5,  { cmsSigCurveSetElemType, cmsSigCLutElemType, cmsSigCurveSetElemType, cmsSigMatrixElemType, cmsSigCurveSetElemType }},
    { TRUE , cmsSigBToA0Tag, cmsSigLutBtoAType,  1,  { cmsSigCurveSetElemType }},
    { TRUE , cmsSigBToA0Tag, cmsSigLutBtoAType,  3,  { cmsSigCurveSetElemType, cmsSigMatrixElemType, cmsSigCurveSetElemType }},
    { TRUE , cmsSigBToA0Tag, cmsSigLutBtoAType,  3,  { cmsSigCurveSetElemType, cmsSigCLutElemType, cmsSigCurveSetElemType }},
    { TRUE , cmsSigBToA0Tag, cmsSigLutBtoAType,  5,  { cmsSigCurveSetElemType, cmsSigMatrixElemType, cmsSigCurveSetElemType, cmsSigCLutElemType, cmsSigCurveSetElemType }}
};

#define SIZE_OF_ALLOWED_LUT (sizeof(AllowedLUTTypes)/sizeof(cmsAllowedLUT))

// Check a single entry
static
cmsBool CheckOne(const cmsAllowedLUT* Tab, const cmsPipeline* Lut)
{
    cmsStage* mpe;
    int n;

    for (n=0, mpe = Lut ->Elements; mpe != NULL; mpe = mpe ->Next, n++) {

        if (n > Tab ->nTypes) return FALSE;
        if (cmsStageType(mpe) != Tab ->MpeTypes[n]) return FALSE;
    }

    return (n == Tab ->nTypes);
}


static
const cmsAllowedLUT* FindCombination(const cmsPipeline* Lut, cmsBool IsV4, cmsTagSignature DestinationTag)
{
    cmsUInt32Number n;

    for (n=0; n < SIZE_OF_ALLOWED_LUT; n++) {

        const cmsAllowedLUT* Tab = AllowedLUTTypes + n;

        if (IsV4 ^ Tab -> IsV4) continue;
        if ((Tab ->RequiredTag != 0) && (Tab ->RequiredTag != DestinationTag)) continue;

        if (CheckOne(Tab, Lut)) return Tab;
    }

    return NULL;
}


// Does convert a transform into a device link profile
cmsHPROFILE CMSEXPORT cmsTransform2DeviceLink(cmsHTRANSFORM hTransform, cmsFloat64Number Version, cmsUInt32Number dwFlags)
{
    cmsHPROFILE hProfile = NULL;
    cmsUInt32Number FrmIn, FrmOut, ChansIn, ChansOut;
    int ColorSpaceBitsIn, ColorSpaceBitsOut;
    _cmsTRANSFORM* xform = (_cmsTRANSFORM*) hTransform;
    cmsPipeline* LUT = NULL;
    cmsStage* mpe;
    cmsContext ContextID = cmsGetTransformContextID(hTransform);
    const cmsAllowedLUT* AllowedLUT;
    cmsTagSignature DestinationTag;
    cmsProfileClassSignature deviceClass; 

    _cmsAssert(hTransform != NULL);

    // Get the first mpe to check for named color
    mpe = cmsPipelineGetPtrToFirstStage(xform ->Lut);

    // Check if is a named color transform
    if (mpe != NULL) {

        if (cmsStageType(mpe) == cmsSigNamedColorElemType) {
            return CreateNamedColorDevicelink(hTransform);
        }
    }

    // First thing to do is to get a copy of the transformation
    LUT = cmsPipelineDup(xform ->Lut);
    if (LUT == NULL) return NULL;

    // Time to fix the Lab2/Lab4 issue.
    if ((xform ->EntryColorSpace == cmsSigLabData) && (Version < 4.0)) {

        if (!cmsPipelineInsertStage(LUT, cmsAT_BEGIN, _cmsStageAllocLabV2ToV4curves(ContextID)))
            goto Error;
    }

    // On the output side too
    if ((xform ->ExitColorSpace) == cmsSigLabData && (Version < 4.0)) {

        if (!cmsPipelineInsertStage(LUT, cmsAT_END, _cmsStageAllocLabV4ToV2(ContextID)))
            goto Error;
    }


    hProfile = cmsCreateProfilePlaceholder(ContextID);
    if (!hProfile) goto Error;                    // can't allocate

    cmsSetProfileVersion(hProfile, Version);

    FixColorSpaces(hProfile, xform -> EntryColorSpace, xform -> ExitColorSpace, dwFlags);

    // Optimize the LUT and precalculate a devicelink

    ChansIn  = cmsChannelsOf(xform -> EntryColorSpace);
    ChansOut = cmsChannelsOf(xform -> ExitColorSpace);

    ColorSpaceBitsIn  = _cmsLCMScolorSpace(xform -> EntryColorSpace);
    ColorSpaceBitsOut = _cmsLCMScolorSpace(xform -> ExitColorSpace);

    FrmIn  = COLORSPACE_SH(ColorSpaceBitsIn) | CHANNELS_SH(ChansIn)|BYTES_SH(2);
    FrmOut = COLORSPACE_SH(ColorSpaceBitsOut) | CHANNELS_SH(ChansOut)|BYTES_SH(2);

    deviceClass = cmsGetDeviceClass(hProfile);

     if (deviceClass == cmsSigOutputClass)
         DestinationTag = cmsSigBToA0Tag;
     else
         DestinationTag = cmsSigAToB0Tag;

    // Check if the profile/version can store the result
    if (dwFlags & cmsFLAGS_FORCE_CLUT)
        AllowedLUT = NULL;
    else
        AllowedLUT = FindCombination(LUT, Version >= 4.0, DestinationTag);

    if (AllowedLUT == NULL) {

        // Try to optimize
        _cmsOptimizePipeline(ContextID, &LUT, xform ->RenderingIntent, &FrmIn, &FrmOut, &dwFlags);
        AllowedLUT = FindCombination(LUT, Version >= 4.0, DestinationTag);

    }

    // If no way, then force CLUT that for sure can be written
    if (AllowedLUT == NULL) {

        cmsStage* FirstStage;
        cmsStage* LastStage;

        dwFlags |= cmsFLAGS_FORCE_CLUT;
        _cmsOptimizePipeline(ContextID, &LUT, xform ->RenderingIntent, &FrmIn, &FrmOut, &dwFlags);

        // Put identity curves if needed
        FirstStage = cmsPipelineGetPtrToFirstStage(LUT);
        if (FirstStage != NULL && FirstStage ->Type != cmsSigCurveSetElemType)
             if (!cmsPipelineInsertStage(LUT, cmsAT_BEGIN, _cmsStageAllocIdentityCurves(ContextID, ChansIn)))
                 goto Error;

        LastStage = cmsPipelineGetPtrToLastStage(LUT);
        if (LastStage != NULL && LastStage ->Type != cmsSigCurveSetElemType)
             if (!cmsPipelineInsertStage(LUT, cmsAT_END,   _cmsStageAllocIdentityCurves(ContextID, ChansOut)))
                 goto Error;

        AllowedLUT = FindCombination(LUT, Version >= 4.0, DestinationTag);
    }

    // Somethings is wrong...
    if (AllowedLUT == NULL) {
        goto Error;
    }


    if (dwFlags & cmsFLAGS_8BITS_DEVICELINK)
                     cmsPipelineSetSaveAs8bitsFlag(LUT, TRUE);

    // Tag profile with information
    if (!SetTextTags(hProfile, L"devicelink")) goto Error;

    // Store result
    if (!cmsWriteTag(hProfile, DestinationTag, LUT)) goto Error;


    if (xform -> InputColorant != NULL) {
           if (!cmsWriteTag(hProfile, cmsSigColorantTableTag, xform->InputColorant)) goto Error;
    }

    if (xform -> OutputColorant != NULL) {
           if (!cmsWriteTag(hProfile, cmsSigColorantTableOutTag, xform->OutputColorant)) goto Error;
    }

    if ((deviceClass == cmsSigLinkClass) && (xform ->Sequence != NULL)) {
        if (!_cmsWriteProfileSequence(hProfile, xform ->Sequence)) goto Error;
    }

    // Set the white point
    if (deviceClass == cmsSigInputClass) {
        if (!cmsWriteTag(hProfile, cmsSigMediaWhitePointTag, &xform ->EntryWhitePoint)) goto Error;
    }
    else {
         if (!cmsWriteTag(hProfile, cmsSigMediaWhitePointTag, &xform ->ExitWhitePoint)) goto Error;
    }

  
    // Per 7.2.15 in spec 4.3
    cmsSetHeaderRenderingIntent(hProfile, xform ->RenderingIntent);

    cmsPipelineFree(LUT);
    return hProfile;

Error:
    if (LUT != NULL) cmsPipelineFree(LUT);
    cmsCloseProfile(hProfile);
    return NULL;
}
