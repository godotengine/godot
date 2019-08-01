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


// Link several profiles to obtain a single LUT modelling the whole color transform. Intents, Black point
// compensation and Adaptation parameters may vary across profiles. BPC and Adaptation refers to the PCS
// after the profile. I.e, BPC[0] refers to connexion between profile(0) and profile(1)
cmsPipeline* _cmsLinkProfiles(cmsContext     ContextID,
                              cmsUInt32Number nProfiles,
                              cmsUInt32Number Intents[],
                              cmsHPROFILE     hProfiles[],
                              cmsBool         BPC[],
                              cmsFloat64Number AdaptationStates[],
                              cmsUInt32Number dwFlags);

//---------------------------------------------------------------------------------

// This is the default routine for ICC-style intents. A user may decide to override it by using a plugin.
// Supported intents are perceptual, relative colorimetric, saturation and ICC-absolute colorimetric
static
cmsPipeline* DefaultICCintents(cmsContext     ContextID,
                               cmsUInt32Number nProfiles,
                               cmsUInt32Number Intents[],
                               cmsHPROFILE     hProfiles[],
                               cmsBool         BPC[],
                               cmsFloat64Number AdaptationStates[],
                               cmsUInt32Number dwFlags);

//---------------------------------------------------------------------------------

// This is the entry for black-preserving K-only intents, which are non-ICC. Last profile have to be a output profile
// to do the trick (no devicelinks allowed at that position)
static
cmsPipeline*  BlackPreservingKOnlyIntents(cmsContext     ContextID,
                                          cmsUInt32Number nProfiles,
                                          cmsUInt32Number Intents[],
                                          cmsHPROFILE     hProfiles[],
                                          cmsBool         BPC[],
                                          cmsFloat64Number AdaptationStates[],
                                          cmsUInt32Number dwFlags);

//---------------------------------------------------------------------------------

// This is the entry for black-plane preserving, which are non-ICC. Again, Last profile have to be a output profile
// to do the trick (no devicelinks allowed at that position)
static
cmsPipeline*  BlackPreservingKPlaneIntents(cmsContext     ContextID,
                                           cmsUInt32Number nProfiles,
                                           cmsUInt32Number Intents[],
                                           cmsHPROFILE     hProfiles[],
                                           cmsBool         BPC[],
                                           cmsFloat64Number AdaptationStates[],
                                           cmsUInt32Number dwFlags);

//---------------------------------------------------------------------------------


// This is a structure holding implementations for all supported intents.
typedef struct _cms_intents_list {

    cmsUInt32Number Intent;
    char            Description[256];
    cmsIntentFn     Link;
    struct _cms_intents_list*  Next;

} cmsIntentsList;


// Built-in intents
static cmsIntentsList DefaultIntents[] = {

    { INTENT_PERCEPTUAL,                            "Perceptual",                                   DefaultICCintents,            &DefaultIntents[1] },
    { INTENT_RELATIVE_COLORIMETRIC,                 "Relative colorimetric",                        DefaultICCintents,            &DefaultIntents[2] },
    { INTENT_SATURATION,                            "Saturation",                                   DefaultICCintents,            &DefaultIntents[3] },
    { INTENT_ABSOLUTE_COLORIMETRIC,                 "Absolute colorimetric",                        DefaultICCintents,            &DefaultIntents[4] },
    { INTENT_PRESERVE_K_ONLY_PERCEPTUAL,            "Perceptual preserving black ink",              BlackPreservingKOnlyIntents,  &DefaultIntents[5] },
    { INTENT_PRESERVE_K_ONLY_RELATIVE_COLORIMETRIC, "Relative colorimetric preserving black ink",   BlackPreservingKOnlyIntents,  &DefaultIntents[6] },
    { INTENT_PRESERVE_K_ONLY_SATURATION,            "Saturation preserving black ink",              BlackPreservingKOnlyIntents,  &DefaultIntents[7] },
    { INTENT_PRESERVE_K_PLANE_PERCEPTUAL,           "Perceptual preserving black plane",            BlackPreservingKPlaneIntents, &DefaultIntents[8] },
    { INTENT_PRESERVE_K_PLANE_RELATIVE_COLORIMETRIC,"Relative colorimetric preserving black plane", BlackPreservingKPlaneIntents, &DefaultIntents[9] },
    { INTENT_PRESERVE_K_PLANE_SATURATION,           "Saturation preserving black plane",            BlackPreservingKPlaneIntents, NULL }
};


// A pointer to the beginning of the list
_cmsIntentsPluginChunkType _cmsIntentsPluginChunk = { NULL };

// Duplicates the zone of memory used by the plug-in in the new context
static
void DupPluginIntentsList(struct _cmsContext_struct* ctx, 
                                               const struct _cmsContext_struct* src)
{
   _cmsIntentsPluginChunkType newHead = { NULL };
   cmsIntentsList*  entry;
   cmsIntentsList*  Anterior = NULL;
   _cmsIntentsPluginChunkType* head = (_cmsIntentsPluginChunkType*) src->chunks[IntentPlugin];

    // Walk the list copying all nodes
   for (entry = head->Intents;
        entry != NULL;
        entry = entry ->Next) {

            cmsIntentsList *newEntry = ( cmsIntentsList *) _cmsSubAllocDup(ctx ->MemPool, entry, sizeof(cmsIntentsList));
   
            if (newEntry == NULL) 
                return;

            // We want to keep the linked list order, so this is a little bit tricky
            newEntry -> Next = NULL;
            if (Anterior)
                Anterior -> Next = newEntry;
     
            Anterior = newEntry;

            if (newHead.Intents == NULL)
                newHead.Intents = newEntry;
    }

  ctx ->chunks[IntentPlugin] = _cmsSubAllocDup(ctx->MemPool, &newHead, sizeof(_cmsIntentsPluginChunkType));
}

void  _cmsAllocIntentsPluginChunk(struct _cmsContext_struct* ctx, 
                                         const struct _cmsContext_struct* src)
{
    if (src != NULL) {

        // Copy all linked list
        DupPluginIntentsList(ctx, src);
    }
    else {
        static _cmsIntentsPluginChunkType IntentsPluginChunkType = { NULL };
        ctx ->chunks[IntentPlugin] = _cmsSubAllocDup(ctx ->MemPool, &IntentsPluginChunkType, sizeof(_cmsIntentsPluginChunkType));
    }
}


// Search the list for a suitable intent. Returns NULL if not found
static
cmsIntentsList* SearchIntent(cmsContext ContextID, cmsUInt32Number Intent)
{
    _cmsIntentsPluginChunkType* ctx = ( _cmsIntentsPluginChunkType*) _cmsContextGetClientChunk(ContextID, IntentPlugin);
    cmsIntentsList* pt;

    for (pt = ctx -> Intents; pt != NULL; pt = pt -> Next)
        if (pt ->Intent == Intent) return pt;

    for (pt = DefaultIntents; pt != NULL; pt = pt -> Next)
        if (pt ->Intent == Intent) return pt;

    return NULL;
}

// Black point compensation. Implemented as a linear scaling in XYZ. Black points
// should come relative to the white point. Fills an matrix/offset element m
// which is organized as a 4x4 matrix.
static
void ComputeBlackPointCompensation(const cmsCIEXYZ* BlackPointIn,
                                   const cmsCIEXYZ* BlackPointOut,
                                   cmsMAT3* m, cmsVEC3* off)
{
  cmsFloat64Number ax, ay, az, bx, by, bz, tx, ty, tz;

   // Now we need to compute a matrix plus an offset m and of such of
   // [m]*bpin + off = bpout
   // [m]*D50  + off = D50
   //
   // This is a linear scaling in the form ax+b, where
   // a = (bpout - D50) / (bpin - D50)
   // b = - D50* (bpout - bpin) / (bpin - D50)

   tx = BlackPointIn->X - cmsD50_XYZ()->X;
   ty = BlackPointIn->Y - cmsD50_XYZ()->Y;
   tz = BlackPointIn->Z - cmsD50_XYZ()->Z;

   ax = (BlackPointOut->X - cmsD50_XYZ()->X) / tx;
   ay = (BlackPointOut->Y - cmsD50_XYZ()->Y) / ty;
   az = (BlackPointOut->Z - cmsD50_XYZ()->Z) / tz;

   bx = - cmsD50_XYZ()-> X * (BlackPointOut->X - BlackPointIn->X) / tx;
   by = - cmsD50_XYZ()-> Y * (BlackPointOut->Y - BlackPointIn->Y) / ty;
   bz = - cmsD50_XYZ()-> Z * (BlackPointOut->Z - BlackPointIn->Z) / tz;

   _cmsVEC3init(&m ->v[0], ax, 0,  0);
   _cmsVEC3init(&m ->v[1], 0, ay,  0);
   _cmsVEC3init(&m ->v[2], 0,  0,  az);
   _cmsVEC3init(off, bx, by, bz);

}


// Approximate a blackbody illuminant based on CHAD information
static
cmsFloat64Number CHAD2Temp(const cmsMAT3* Chad)
{
    // Convert D50 across inverse CHAD to get the absolute white point
    cmsVEC3 d, s;
    cmsCIEXYZ Dest;
    cmsCIExyY DestChromaticity;
    cmsFloat64Number TempK;
    cmsMAT3 m1, m2;

    m1 = *Chad;
    if (!_cmsMAT3inverse(&m1, &m2)) return FALSE;

    s.n[VX] = cmsD50_XYZ() -> X;
    s.n[VY] = cmsD50_XYZ() -> Y;
    s.n[VZ] = cmsD50_XYZ() -> Z;

    _cmsMAT3eval(&d, &m2, &s);

    Dest.X = d.n[VX];
    Dest.Y = d.n[VY];
    Dest.Z = d.n[VZ];

    cmsXYZ2xyY(&DestChromaticity, &Dest);

    if (!cmsTempFromWhitePoint(&TempK, &DestChromaticity))
        return -1.0;

    return TempK;
}

// Compute a CHAD based on a given temperature
static
    void Temp2CHAD(cmsMAT3* Chad, cmsFloat64Number Temp)
{
    cmsCIEXYZ White;
    cmsCIExyY ChromaticityOfWhite;

    cmsWhitePointFromTemp(&ChromaticityOfWhite, Temp);
    cmsxyY2XYZ(&White, &ChromaticityOfWhite);
    _cmsAdaptationMatrix(Chad, NULL, &White, cmsD50_XYZ());
}

// Join scalings to obtain relative input to absolute and then to relative output.
// Result is stored in a 3x3 matrix
static
cmsBool  ComputeAbsoluteIntent(cmsFloat64Number AdaptationState,
                               const cmsCIEXYZ* WhitePointIn,
                               const cmsMAT3* ChromaticAdaptationMatrixIn,
                               const cmsCIEXYZ* WhitePointOut,
                               const cmsMAT3* ChromaticAdaptationMatrixOut,
                               cmsMAT3* m)
{
    cmsMAT3 Scale, m1, m2, m3, m4;

    // TODO: Follow Marc Mahy's recommendation to check if CHAD is same by using M1*M2 == M2*M1. If so, do nothing.
    // TODO: Add support for ArgyllArts tag

    // Adaptation state
    if (AdaptationState == 1.0) {

        // Observer is fully adapted. Keep chromatic adaptation.
        // That is the standard V4 behaviour
        _cmsVEC3init(&m->v[0], WhitePointIn->X / WhitePointOut->X, 0, 0);
        _cmsVEC3init(&m->v[1], 0, WhitePointIn->Y / WhitePointOut->Y, 0);
        _cmsVEC3init(&m->v[2], 0, 0, WhitePointIn->Z / WhitePointOut->Z);

    }
    else  {

        // Incomplete adaptation. This is an advanced feature.
        _cmsVEC3init(&Scale.v[0], WhitePointIn->X / WhitePointOut->X, 0, 0);
        _cmsVEC3init(&Scale.v[1], 0,  WhitePointIn->Y / WhitePointOut->Y, 0);
        _cmsVEC3init(&Scale.v[2], 0, 0,  WhitePointIn->Z / WhitePointOut->Z);


        if (AdaptationState == 0.0) {
        
            m1 = *ChromaticAdaptationMatrixOut;
            _cmsMAT3per(&m2, &m1, &Scale);
            // m2 holds CHAD from output white to D50 times abs. col. scaling

            // Observer is not adapted, undo the chromatic adaptation
            _cmsMAT3per(m, &m2, ChromaticAdaptationMatrixOut);

            m3 = *ChromaticAdaptationMatrixIn;
            if (!_cmsMAT3inverse(&m3, &m4)) return FALSE;
            _cmsMAT3per(m, &m2, &m4);

        } else {

            cmsMAT3 MixedCHAD;
            cmsFloat64Number TempSrc, TempDest, Temp;

            m1 = *ChromaticAdaptationMatrixIn;
            if (!_cmsMAT3inverse(&m1, &m2)) return FALSE;
            _cmsMAT3per(&m3, &m2, &Scale);
            // m3 holds CHAD from input white to D50 times abs. col. scaling

            TempSrc  = CHAD2Temp(ChromaticAdaptationMatrixIn);
            TempDest = CHAD2Temp(ChromaticAdaptationMatrixOut);

            if (TempSrc < 0.0 || TempDest < 0.0) return FALSE; // Something went wrong

            if (_cmsMAT3isIdentity(&Scale) && fabs(TempSrc - TempDest) < 0.01) {

                _cmsMAT3identity(m);
                return TRUE;
            }

            Temp = (1.0 - AdaptationState) * TempDest + AdaptationState * TempSrc;

            // Get a CHAD from whatever output temperature to D50. This replaces output CHAD
            Temp2CHAD(&MixedCHAD, Temp);

            _cmsMAT3per(m, &m3, &MixedCHAD);
        }

    }
    return TRUE;

}

// Just to see if m matrix should be applied
static
cmsBool IsEmptyLayer(cmsMAT3* m, cmsVEC3* off)
{
    cmsFloat64Number diff = 0;
    cmsMAT3 Ident;
    int i;

    if (m == NULL && off == NULL) return TRUE;  // NULL is allowed as an empty layer
    if (m == NULL && off != NULL) return FALSE; // This is an internal error

    _cmsMAT3identity(&Ident);

    for (i=0; i < 3*3; i++)
        diff += fabs(((cmsFloat64Number*)m)[i] - ((cmsFloat64Number*)&Ident)[i]);

    for (i=0; i < 3; i++)
        diff += fabs(((cmsFloat64Number*)off)[i]);


    return (diff < 0.002);
}


// Compute the conversion layer
static
cmsBool ComputeConversion(cmsUInt32Number i, 
                          cmsHPROFILE hProfiles[],
                          cmsUInt32Number Intent,
                          cmsBool BPC,
                          cmsFloat64Number AdaptationState,
                          cmsMAT3* m, cmsVEC3* off)
{

    int k;

    // m  and off are set to identity and this is detected latter on
    _cmsMAT3identity(m);
    _cmsVEC3init(off, 0, 0, 0);

    // If intent is abs. colorimetric,
    if (Intent == INTENT_ABSOLUTE_COLORIMETRIC) {

        cmsCIEXYZ WhitePointIn, WhitePointOut;
        cmsMAT3 ChromaticAdaptationMatrixIn, ChromaticAdaptationMatrixOut;

        _cmsReadMediaWhitePoint(&WhitePointIn,  hProfiles[i-1]);
        _cmsReadCHAD(&ChromaticAdaptationMatrixIn, hProfiles[i-1]);

        _cmsReadMediaWhitePoint(&WhitePointOut,  hProfiles[i]);
        _cmsReadCHAD(&ChromaticAdaptationMatrixOut, hProfiles[i]);

        if (!ComputeAbsoluteIntent(AdaptationState,
                                  &WhitePointIn,  &ChromaticAdaptationMatrixIn,
                                  &WhitePointOut, &ChromaticAdaptationMatrixOut, m)) return FALSE;

    }
    else {
        // Rest of intents may apply BPC.

        if (BPC) {

            cmsCIEXYZ BlackPointIn, BlackPointOut;

            cmsDetectBlackPoint(&BlackPointIn,  hProfiles[i-1], Intent, 0);
            cmsDetectDestinationBlackPoint(&BlackPointOut, hProfiles[i], Intent, 0);

            // If black points are equal, then do nothing
            if (BlackPointIn.X != BlackPointOut.X ||
                BlackPointIn.Y != BlackPointOut.Y ||
                BlackPointIn.Z != BlackPointOut.Z)
                    ComputeBlackPointCompensation(&BlackPointIn, &BlackPointOut, m, off);
        }
    }

    // Offset should be adjusted because the encoding. We encode XYZ normalized to 0..1.0,
    // to do that, we divide by MAX_ENCODEABLE_XZY. The conversion stage goes XYZ -> XYZ so
    // we have first to convert from encoded to XYZ and then convert back to encoded.
    // y = Mx + Off
    // x = x'c
    // y = M x'c + Off
    // y = y'c; y' = y / c
    // y' = (Mx'c + Off) /c = Mx' + (Off / c)

    for (k=0; k < 3; k++) {
        off ->n[k] /= MAX_ENCODEABLE_XYZ;
    }

    return TRUE;
}


// Add a conversion stage if needed. If a matrix/offset m is given, it applies to XYZ space
static
cmsBool AddConversion(cmsPipeline* Result, cmsColorSpaceSignature InPCS, cmsColorSpaceSignature OutPCS, cmsMAT3* m, cmsVEC3* off)
{
    cmsFloat64Number* m_as_dbl = (cmsFloat64Number*) m;
    cmsFloat64Number* off_as_dbl = (cmsFloat64Number*) off;

    // Handle PCS mismatches. A specialized stage is added to the LUT in such case
    switch (InPCS) {

    case cmsSigXYZData: // Input profile operates in XYZ

        switch (OutPCS) {

        case cmsSigXYZData:  // XYZ -> XYZ
            if (!IsEmptyLayer(m, off) &&
                !cmsPipelineInsertStage(Result, cmsAT_END, cmsStageAllocMatrix(Result ->ContextID, 3, 3, m_as_dbl, off_as_dbl)))
                return FALSE;
            break;

        case cmsSigLabData:  // XYZ -> Lab
            if (!IsEmptyLayer(m, off) &&
                !cmsPipelineInsertStage(Result, cmsAT_END, cmsStageAllocMatrix(Result ->ContextID, 3, 3, m_as_dbl, off_as_dbl)))
                return FALSE;
            if (!cmsPipelineInsertStage(Result, cmsAT_END, _cmsStageAllocXYZ2Lab(Result ->ContextID)))
                return FALSE;
            break;

        default:
            return FALSE;   // Colorspace mismatch
        }
        break;

    case cmsSigLabData: // Input profile operates in Lab

        switch (OutPCS) {

        case cmsSigXYZData:  // Lab -> XYZ

            if (!cmsPipelineInsertStage(Result, cmsAT_END, _cmsStageAllocLab2XYZ(Result ->ContextID)))
                return FALSE;
            if (!IsEmptyLayer(m, off) &&
                !cmsPipelineInsertStage(Result, cmsAT_END, cmsStageAllocMatrix(Result ->ContextID, 3, 3, m_as_dbl, off_as_dbl)))
                return FALSE;
            break;

        case cmsSigLabData:  // Lab -> Lab

            if (!IsEmptyLayer(m, off)) {
                if (!cmsPipelineInsertStage(Result, cmsAT_END, _cmsStageAllocLab2XYZ(Result ->ContextID)) ||
                    !cmsPipelineInsertStage(Result, cmsAT_END, cmsStageAllocMatrix(Result ->ContextID, 3, 3, m_as_dbl, off_as_dbl)) ||
                    !cmsPipelineInsertStage(Result, cmsAT_END, _cmsStageAllocXYZ2Lab(Result ->ContextID)))
                    return FALSE;
            }
            break;

        default:
            return FALSE;  // Mismatch
        }
        break;

        // On colorspaces other than PCS, check for same space
    default:
        if (InPCS != OutPCS) return FALSE;
        break;
    }

    return TRUE;
}


// Is a given space compatible with another?
static
cmsBool ColorSpaceIsCompatible(cmsColorSpaceSignature a, cmsColorSpaceSignature b)
{
    // If they are same, they are compatible.
    if (a == b) return TRUE;

    // Check for MCH4 substitution of CMYK
    if ((a == cmsSig4colorData) && (b == cmsSigCmykData)) return TRUE;
    if ((a == cmsSigCmykData) && (b == cmsSig4colorData)) return TRUE;

    // Check for XYZ/Lab. Those spaces are interchangeable as they can be computed one from other.
    if ((a == cmsSigXYZData) && (b == cmsSigLabData)) return TRUE;
    if ((a == cmsSigLabData) && (b == cmsSigXYZData)) return TRUE;

    return FALSE;
}


// Default handler for ICC-style intents
static
cmsPipeline* DefaultICCintents(cmsContext       ContextID,
                               cmsUInt32Number  nProfiles,
                               cmsUInt32Number  TheIntents[],
                               cmsHPROFILE      hProfiles[],
                               cmsBool          BPC[],
                               cmsFloat64Number AdaptationStates[],
                               cmsUInt32Number  dwFlags)
{
    cmsPipeline* Lut = NULL;
    cmsPipeline* Result;
    cmsHPROFILE hProfile;
    cmsMAT3 m;
    cmsVEC3 off;
    cmsColorSpaceSignature ColorSpaceIn, ColorSpaceOut = cmsSigLabData, CurrentColorSpace;
    cmsProfileClassSignature ClassSig;
    cmsUInt32Number  i, Intent;

    // For safety
    if (nProfiles == 0) return NULL;

    // Allocate an empty LUT for holding the result. 0 as channel count means 'undefined'
    Result = cmsPipelineAlloc(ContextID, 0, 0);
    if (Result == NULL) return NULL;

    CurrentColorSpace = cmsGetColorSpace(hProfiles[0]);

    for (i=0; i < nProfiles; i++) {

        cmsBool  lIsDeviceLink, lIsInput;

        hProfile      = hProfiles[i];
        ClassSig      = cmsGetDeviceClass(hProfile);
        lIsDeviceLink = (ClassSig == cmsSigLinkClass || ClassSig == cmsSigAbstractClass );

        // First profile is used as input unless devicelink or abstract
        if ((i == 0) && !lIsDeviceLink) {
            lIsInput = TRUE;
        }
        else {
          // Else use profile in the input direction if current space is not PCS
        lIsInput      = (CurrentColorSpace != cmsSigXYZData) &&
                        (CurrentColorSpace != cmsSigLabData);
        }

        Intent        = TheIntents[i];

        if (lIsInput || lIsDeviceLink) {

            ColorSpaceIn    = cmsGetColorSpace(hProfile);
            ColorSpaceOut   = cmsGetPCS(hProfile);
        }
        else {

            ColorSpaceIn    = cmsGetPCS(hProfile);
            ColorSpaceOut   = cmsGetColorSpace(hProfile);
        }

        if (!ColorSpaceIsCompatible(ColorSpaceIn, CurrentColorSpace)) {

            cmsSignalError(ContextID, cmsERROR_COLORSPACE_CHECK, "ColorSpace mismatch");
            goto Error;
        }

        // If devicelink is found, then no custom intent is allowed and we can
        // read the LUT to be applied. Settings don't apply here.
        if (lIsDeviceLink || ((ClassSig == cmsSigNamedColorClass) && (nProfiles == 1))) {

            // Get the involved LUT from the profile
            Lut = _cmsReadDevicelinkLUT(hProfile, Intent);
            if (Lut == NULL) goto Error;

            // What about abstract profiles?
             if (ClassSig == cmsSigAbstractClass && i > 0) {
                if (!ComputeConversion(i, hProfiles, Intent, BPC[i], AdaptationStates[i], &m, &off)) goto Error;
             }
             else {
                _cmsMAT3identity(&m);
                _cmsVEC3init(&off, 0, 0, 0);
             }


            if (!AddConversion(Result, CurrentColorSpace, ColorSpaceIn, &m, &off)) goto Error;

        }
        else {

            if (lIsInput) {
                // Input direction means non-pcs connection, so proceed like devicelinks
                Lut = _cmsReadInputLUT(hProfile, Intent);
                if (Lut == NULL) goto Error;
            }
            else {

                // Output direction means PCS connection. Intent may apply here
                Lut = _cmsReadOutputLUT(hProfile, Intent);
                if (Lut == NULL) goto Error;


                if (!ComputeConversion(i, hProfiles, Intent, BPC[i], AdaptationStates[i], &m, &off)) goto Error;
                if (!AddConversion(Result, CurrentColorSpace, ColorSpaceIn, &m, &off)) goto Error;

            }
        }

        // Concatenate to the output LUT
        if (!cmsPipelineCat(Result, Lut))
            goto Error;

        cmsPipelineFree(Lut);
        Lut = NULL;

        // Update current space
        CurrentColorSpace = ColorSpaceOut;
    }

    // Check for non-negatives clip
    if (dwFlags & cmsFLAGS_NONEGATIVES) {

           if (ColorSpaceOut == cmsSigGrayData ||
                  ColorSpaceOut == cmsSigRgbData ||
                  ColorSpaceOut == cmsSigCmykData) {

                  cmsStage* clip = _cmsStageClipNegatives(Result->ContextID, cmsChannelsOf(ColorSpaceOut));
                  if (clip == NULL) goto Error;

                  if (!cmsPipelineInsertStage(Result, cmsAT_END, clip))
                         goto Error;
           }

    }

    return Result;

Error:

    if (Lut != NULL) cmsPipelineFree(Lut);
    if (Result != NULL) cmsPipelineFree(Result);
    return NULL;

    cmsUNUSED_PARAMETER(dwFlags);
}


// Wrapper for DLL calling convention
cmsPipeline*  CMSEXPORT _cmsDefaultICCintents(cmsContext     ContextID,
                                              cmsUInt32Number nProfiles,
                                              cmsUInt32Number TheIntents[],
                                              cmsHPROFILE     hProfiles[],
                                              cmsBool         BPC[],
                                              cmsFloat64Number AdaptationStates[],
                                              cmsUInt32Number dwFlags)
{
    return DefaultICCintents(ContextID, nProfiles, TheIntents, hProfiles, BPC, AdaptationStates, dwFlags);
}

// Black preserving intents ---------------------------------------------------------------------------------------------

// Translate black-preserving intents to ICC ones
static
cmsUInt32Number TranslateNonICCIntents(cmsUInt32Number Intent)
{
    switch (Intent) {
        case INTENT_PRESERVE_K_ONLY_PERCEPTUAL:
        case INTENT_PRESERVE_K_PLANE_PERCEPTUAL:
            return INTENT_PERCEPTUAL;

        case INTENT_PRESERVE_K_ONLY_RELATIVE_COLORIMETRIC:
        case INTENT_PRESERVE_K_PLANE_RELATIVE_COLORIMETRIC:
            return INTENT_RELATIVE_COLORIMETRIC;

        case INTENT_PRESERVE_K_ONLY_SATURATION:
        case INTENT_PRESERVE_K_PLANE_SATURATION:
            return INTENT_SATURATION;

        default: return Intent;
    }
}

// Sampler for Black-only preserving CMYK->CMYK transforms

typedef struct {
    cmsPipeline*    cmyk2cmyk;      // The original transform
    cmsToneCurve*   KTone;          // Black-to-black tone curve

} GrayOnlyParams;


// Preserve black only if that is the only ink used
static
int BlackPreservingGrayOnlySampler(register const cmsUInt16Number In[], register cmsUInt16Number Out[], register void* Cargo)
{
    GrayOnlyParams* bp = (GrayOnlyParams*) Cargo;

    // If going across black only, keep black only
    if (In[0] == 0 && In[1] == 0 && In[2] == 0) {

        // TAC does not apply because it is black ink!
        Out[0] = Out[1] = Out[2] = 0;
        Out[3] = cmsEvalToneCurve16(bp->KTone, In[3]);
        return TRUE;
    }

    // Keep normal transform for other colors
    bp ->cmyk2cmyk ->Eval16Fn(In, Out, bp ->cmyk2cmyk->Data);
    return TRUE;
}

// This is the entry for black-preserving K-only intents, which are non-ICC
static
cmsPipeline*  BlackPreservingKOnlyIntents(cmsContext     ContextID,
                                          cmsUInt32Number nProfiles,
                                          cmsUInt32Number TheIntents[],
                                          cmsHPROFILE     hProfiles[],
                                          cmsBool         BPC[],
                                          cmsFloat64Number AdaptationStates[],
                                          cmsUInt32Number dwFlags)
{
    GrayOnlyParams  bp;
    cmsPipeline*    Result;
    cmsUInt32Number ICCIntents[256];
    cmsStage*         CLUT;
    cmsUInt32Number i, nGridPoints;


    // Sanity check
    if (nProfiles < 1 || nProfiles > 255) return NULL;

    // Translate black-preserving intents to ICC ones
    for (i=0; i < nProfiles; i++)
        ICCIntents[i] = TranslateNonICCIntents(TheIntents[i]);

    // Check for non-cmyk profiles
    if (cmsGetColorSpace(hProfiles[0]) != cmsSigCmykData ||
        cmsGetColorSpace(hProfiles[nProfiles-1]) != cmsSigCmykData)
           return DefaultICCintents(ContextID, nProfiles, ICCIntents, hProfiles, BPC, AdaptationStates, dwFlags);

    memset(&bp, 0, sizeof(bp));

    // Allocate an empty LUT for holding the result
    Result = cmsPipelineAlloc(ContextID, 4, 4);
    if (Result == NULL) return NULL;

    // Create a LUT holding normal ICC transform
    bp.cmyk2cmyk = DefaultICCintents(ContextID,
        nProfiles,
        ICCIntents,
        hProfiles,
        BPC,
        AdaptationStates,
        dwFlags);

    if (bp.cmyk2cmyk == NULL) goto Error;

    // Now, compute the tone curve
    bp.KTone = _cmsBuildKToneCurve(ContextID,
        4096,
        nProfiles,
        ICCIntents,
        hProfiles,
        BPC,
        AdaptationStates,
        dwFlags);

    if (bp.KTone == NULL) goto Error;


    // How many gridpoints are we going to use?
    nGridPoints = _cmsReasonableGridpointsByColorspace(cmsSigCmykData, dwFlags);

    // Create the CLUT. 16 bits
    CLUT = cmsStageAllocCLut16bit(ContextID, nGridPoints, 4, 4, NULL);
    if (CLUT == NULL) goto Error;

    // This is the one and only MPE in this LUT
    if (!cmsPipelineInsertStage(Result, cmsAT_BEGIN, CLUT))
        goto Error;

    // Sample it. We cannot afford pre/post linearization this time.
    if (!cmsStageSampleCLut16bit(CLUT, BlackPreservingGrayOnlySampler, (void*) &bp, 0))
        goto Error;

    // Get rid of xform and tone curve
    cmsPipelineFree(bp.cmyk2cmyk);
    cmsFreeToneCurve(bp.KTone);

    return Result;

Error:

    if (bp.cmyk2cmyk != NULL) cmsPipelineFree(bp.cmyk2cmyk);
    if (bp.KTone != NULL)  cmsFreeToneCurve(bp.KTone);
    if (Result != NULL) cmsPipelineFree(Result);
    return NULL;

}

// K Plane-preserving CMYK to CMYK ------------------------------------------------------------------------------------

typedef struct {

    cmsPipeline*     cmyk2cmyk;     // The original transform
    cmsHTRANSFORM    hProofOutput;  // Output CMYK to Lab (last profile)
    cmsHTRANSFORM    cmyk2Lab;      // The input chain
    cmsToneCurve*    KTone;         // Black-to-black tone curve
    cmsPipeline*     LabK2cmyk;     // The output profile
    cmsFloat64Number MaxError;

    cmsHTRANSFORM    hRoundTrip;
    cmsFloat64Number MaxTAC;


} PreserveKPlaneParams;


// The CLUT will be stored at 16 bits, but calculations are performed at cmsFloat32Number precision
static
int BlackPreservingSampler(register const cmsUInt16Number In[], register cmsUInt16Number Out[], register void* Cargo)
{
    int i;
    cmsFloat32Number Inf[4], Outf[4];
    cmsFloat32Number LabK[4];
    cmsFloat64Number SumCMY, SumCMYK, Error, Ratio;
    cmsCIELab ColorimetricLab, BlackPreservingLab;
    PreserveKPlaneParams* bp = (PreserveKPlaneParams*) Cargo;

    // Convert from 16 bits to floating point
    for (i=0; i < 4; i++)
        Inf[i] = (cmsFloat32Number) (In[i] / 65535.0);

    // Get the K across Tone curve
    LabK[3] = cmsEvalToneCurveFloat(bp ->KTone, Inf[3]);

    // If going across black only, keep black only
    if (In[0] == 0 && In[1] == 0 && In[2] == 0) {

        Out[0] = Out[1] = Out[2] = 0;
        Out[3] = _cmsQuickSaturateWord(LabK[3] * 65535.0);
        return TRUE;
    }

    // Try the original transform,
    cmsPipelineEvalFloat( Inf, Outf, bp ->cmyk2cmyk);

    // Store a copy of the floating point result into 16-bit
    for (i=0; i < 4; i++)
            Out[i] = _cmsQuickSaturateWord(Outf[i] * 65535.0);

    // Maybe K is already ok (mostly on K=0)
    if ( fabs(Outf[3] - LabK[3]) < (3.0 / 65535.0) ) {
        return TRUE;
    }

    // K differ, mesure and keep Lab measurement for further usage
    // this is done in relative colorimetric intent
    cmsDoTransform(bp->hProofOutput, Out, &ColorimetricLab, 1);

    // Is not black only and the transform doesn't keep black.
    // Obtain the Lab of output CMYK. After that we have Lab + K
    cmsDoTransform(bp ->cmyk2Lab, Outf, LabK, 1);

    // Obtain the corresponding CMY using reverse interpolation
    // (K is fixed in LabK[3])
    if (!cmsPipelineEvalReverseFloat(LabK, Outf, Outf, bp ->LabK2cmyk)) {

        // Cannot find a suitable value, so use colorimetric xform
        // which is already stored in Out[]
        return TRUE;
    }

    // Make sure to pass through K (which now is fixed)
    Outf[3] = LabK[3];

    // Apply TAC if needed
    SumCMY   = Outf[0]  + Outf[1] + Outf[2];
    SumCMYK  = SumCMY + Outf[3];

    if (SumCMYK > bp ->MaxTAC) {

        Ratio = 1 - ((SumCMYK - bp->MaxTAC) / SumCMY);
        if (Ratio < 0)
            Ratio = 0;
    }
    else
       Ratio = 1.0;

    Out[0] = _cmsQuickSaturateWord(Outf[0] * Ratio * 65535.0);     // C
    Out[1] = _cmsQuickSaturateWord(Outf[1] * Ratio * 65535.0);     // M
    Out[2] = _cmsQuickSaturateWord(Outf[2] * Ratio * 65535.0);     // Y
    Out[3] = _cmsQuickSaturateWord(Outf[3] * 65535.0);

    // Estimate the error (this goes 16 bits to Lab DBL)
    cmsDoTransform(bp->hProofOutput, Out, &BlackPreservingLab, 1);
    Error = cmsDeltaE(&ColorimetricLab, &BlackPreservingLab);
    if (Error > bp -> MaxError)
        bp->MaxError = Error;

    return TRUE;
}

// This is the entry for black-plane preserving, which are non-ICC
static
cmsPipeline* BlackPreservingKPlaneIntents(cmsContext     ContextID,
                                          cmsUInt32Number nProfiles,
                                          cmsUInt32Number TheIntents[],
                                          cmsHPROFILE     hProfiles[],
                                          cmsBool         BPC[],
                                          cmsFloat64Number AdaptationStates[],
                                          cmsUInt32Number dwFlags)
{
    PreserveKPlaneParams bp;
    cmsPipeline*    Result = NULL;
    cmsUInt32Number ICCIntents[256];
    cmsStage*         CLUT;
    cmsUInt32Number i, nGridPoints;
    cmsHPROFILE hLab;

    // Sanity check
    if (nProfiles < 1 || nProfiles > 255) return NULL;

    // Translate black-preserving intents to ICC ones
    for (i=0; i < nProfiles; i++)
        ICCIntents[i] = TranslateNonICCIntents(TheIntents[i]);

    // Check for non-cmyk profiles
    if (cmsGetColorSpace(hProfiles[0]) != cmsSigCmykData ||
        !(cmsGetColorSpace(hProfiles[nProfiles-1]) == cmsSigCmykData ||
        cmsGetDeviceClass(hProfiles[nProfiles-1]) == cmsSigOutputClass))
           return  DefaultICCintents(ContextID, nProfiles, ICCIntents, hProfiles, BPC, AdaptationStates, dwFlags);

    // Allocate an empty LUT for holding the result
    Result = cmsPipelineAlloc(ContextID, 4, 4);
    if (Result == NULL) return NULL;


    memset(&bp, 0, sizeof(bp));

    // We need the input LUT of the last profile, assuming this one is responsible of
    // black generation. This LUT will be searched in inverse order.
    bp.LabK2cmyk = _cmsReadInputLUT(hProfiles[nProfiles-1], INTENT_RELATIVE_COLORIMETRIC);
    if (bp.LabK2cmyk == NULL) goto Cleanup;

    // Get total area coverage (in 0..1 domain)
    bp.MaxTAC = cmsDetectTAC(hProfiles[nProfiles-1]) / 100.0;
    if (bp.MaxTAC <= 0) goto Cleanup;


    // Create a LUT holding normal ICC transform
    bp.cmyk2cmyk = DefaultICCintents(ContextID,
                                         nProfiles,
                                         ICCIntents,
                                         hProfiles,
                                         BPC,
                                         AdaptationStates,
                                         dwFlags);
    if (bp.cmyk2cmyk == NULL) goto Cleanup;

    // Now the tone curve
    bp.KTone = _cmsBuildKToneCurve(ContextID, 4096, nProfiles,
                                   ICCIntents,
                                   hProfiles,
                                   BPC,
                                   AdaptationStates,
                                   dwFlags);
    if (bp.KTone == NULL) goto Cleanup;

    // To measure the output, Last profile to Lab
    hLab = cmsCreateLab4ProfileTHR(ContextID, NULL);
    bp.hProofOutput = cmsCreateTransformTHR(ContextID, hProfiles[nProfiles-1],
                                         CHANNELS_SH(4)|BYTES_SH(2), hLab, TYPE_Lab_DBL,
                                         INTENT_RELATIVE_COLORIMETRIC,
                                         cmsFLAGS_NOCACHE|cmsFLAGS_NOOPTIMIZE);
    if ( bp.hProofOutput == NULL) goto Cleanup;

    // Same as anterior, but lab in the 0..1 range
    bp.cmyk2Lab = cmsCreateTransformTHR(ContextID, hProfiles[nProfiles-1],
                                         FLOAT_SH(1)|CHANNELS_SH(4)|BYTES_SH(4), hLab,
                                         FLOAT_SH(1)|CHANNELS_SH(3)|BYTES_SH(4),
                                         INTENT_RELATIVE_COLORIMETRIC,
                                         cmsFLAGS_NOCACHE|cmsFLAGS_NOOPTIMIZE);
    if (bp.cmyk2Lab == NULL) goto Cleanup;
    cmsCloseProfile(hLab);

    // Error estimation (for debug only)
    bp.MaxError = 0;

    // How many gridpoints are we going to use?
    nGridPoints = _cmsReasonableGridpointsByColorspace(cmsSigCmykData, dwFlags);


    CLUT = cmsStageAllocCLut16bit(ContextID, nGridPoints, 4, 4, NULL);
    if (CLUT == NULL) goto Cleanup;

    if (!cmsPipelineInsertStage(Result, cmsAT_BEGIN, CLUT))
        goto Cleanup;

    cmsStageSampleCLut16bit(CLUT, BlackPreservingSampler, (void*) &bp, 0);

Cleanup:

    if (bp.cmyk2cmyk) cmsPipelineFree(bp.cmyk2cmyk);
    if (bp.cmyk2Lab) cmsDeleteTransform(bp.cmyk2Lab);
    if (bp.hProofOutput) cmsDeleteTransform(bp.hProofOutput);

    if (bp.KTone) cmsFreeToneCurve(bp.KTone);
    if (bp.LabK2cmyk) cmsPipelineFree(bp.LabK2cmyk);

    return Result;
}

// Link routines ------------------------------------------------------------------------------------------------------

// Chain several profiles into a single LUT. It just checks the parameters and then calls the handler
// for the first intent in chain. The handler may be user-defined. Is up to the handler to deal with the
// rest of intents in chain. A maximum of 255 profiles at time are supported, which is pretty reasonable.
cmsPipeline* _cmsLinkProfiles(cmsContext     ContextID,
                              cmsUInt32Number nProfiles,
                              cmsUInt32Number TheIntents[],
                              cmsHPROFILE     hProfiles[],
                              cmsBool         BPC[],
                              cmsFloat64Number AdaptationStates[],
                              cmsUInt32Number dwFlags)
{
    cmsUInt32Number i;
    cmsIntentsList* Intent;

    // Make sure a reasonable number of profiles is provided
    if (nProfiles <= 0 || nProfiles > 255) {
         cmsSignalError(ContextID, cmsERROR_RANGE, "Couldn't link '%d' profiles", nProfiles);
        return NULL;
    }

    for (i=0; i < nProfiles; i++) {

        // Check if black point is really needed or allowed. Note that
        // following Adobe's document:
        // BPC does not apply to devicelink profiles, nor to abs colorimetric,
        // and applies always on V4 perceptual and saturation.

        if (TheIntents[i] == INTENT_ABSOLUTE_COLORIMETRIC)
            BPC[i] = FALSE;

        if (TheIntents[i] == INTENT_PERCEPTUAL || TheIntents[i] == INTENT_SATURATION) {

            // Force BPC for V4 profiles in perceptual and saturation
            if (cmsGetEncodedICCversion(hProfiles[i]) >= 0x4000000)
                BPC[i] = TRUE;
        }
    }

    // Search for a handler. The first intent in the chain defines the handler. That would
    // prevent using multiple custom intents in a multiintent chain, but the behaviour of
    // this case would present some issues if the custom intent tries to do things like
    // preserve primaries. This solution is not perfect, but works well on most cases.

    Intent = SearchIntent(ContextID, TheIntents[0]);
    if (Intent == NULL) {
        cmsSignalError(ContextID, cmsERROR_UNKNOWN_EXTENSION, "Unsupported intent '%d'", TheIntents[0]);
        return NULL;
    }

    // Call the handler
    return Intent ->Link(ContextID, nProfiles, TheIntents, hProfiles, BPC, AdaptationStates, dwFlags);
}

// -------------------------------------------------------------------------------------------------

// Get information about available intents. nMax is the maximum space for the supplied "Codes"
// and "Descriptions" the function returns the total number of intents, which may be greater
// than nMax, although the matrices are not populated beyond this level.
cmsUInt32Number CMSEXPORT cmsGetSupportedIntentsTHR(cmsContext ContextID, cmsUInt32Number nMax, cmsUInt32Number* Codes, char** Descriptions)
{
    _cmsIntentsPluginChunkType* ctx = ( _cmsIntentsPluginChunkType*) _cmsContextGetClientChunk(ContextID, IntentPlugin);
    cmsIntentsList* pt;
    cmsUInt32Number nIntents;


    for (nIntents=0, pt = ctx->Intents; pt != NULL; pt = pt -> Next)
    {
        if (nIntents < nMax) {
            if (Codes != NULL)
                Codes[nIntents] = pt ->Intent;

            if (Descriptions != NULL)
                Descriptions[nIntents] = pt ->Description;
        }

        nIntents++;
    }

    for (nIntents=0, pt = DefaultIntents; pt != NULL; pt = pt -> Next)
    {
        if (nIntents < nMax) {
            if (Codes != NULL)
                Codes[nIntents] = pt ->Intent;

            if (Descriptions != NULL)
                Descriptions[nIntents] = pt ->Description;
        }

        nIntents++;
    }
    return nIntents;
}

cmsUInt32Number CMSEXPORT cmsGetSupportedIntents(cmsUInt32Number nMax, cmsUInt32Number* Codes, char** Descriptions)
{
    return cmsGetSupportedIntentsTHR(NULL, nMax, Codes, Descriptions);
}

// The plug-in registration. User can add new intents or override default routines
cmsBool  _cmsRegisterRenderingIntentPlugin(cmsContext id, cmsPluginBase* Data)
{
    _cmsIntentsPluginChunkType* ctx = ( _cmsIntentsPluginChunkType*) _cmsContextGetClientChunk(id, IntentPlugin);
    cmsPluginRenderingIntent* Plugin = (cmsPluginRenderingIntent*) Data;
    cmsIntentsList* fl;

    // Do we have to reset the custom intents?
    if (Data == NULL) {

        ctx->Intents = NULL;
        return TRUE;
    }

    fl = (cmsIntentsList*) _cmsPluginMalloc(id, sizeof(cmsIntentsList));
    if (fl == NULL) return FALSE;


    fl ->Intent  = Plugin ->Intent;
    strncpy(fl ->Description, Plugin ->Description, sizeof(fl ->Description)-1);
    fl ->Description[sizeof(fl ->Description)-1] = 0;

    fl ->Link    = Plugin ->Link;

    fl ->Next = ctx ->Intents;
    ctx ->Intents = fl;

    return TRUE;
}

