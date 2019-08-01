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


//----------------------------------------------------------------------------------

// Optimization for 8 bits, Shaper-CLUT (3 inputs only)
typedef struct {

    cmsContext ContextID;

    const cmsInterpParams* p;   // Tetrahedrical interpolation parameters. This is a not-owned pointer.

    cmsUInt16Number rx[256], ry[256], rz[256];
    cmsUInt32Number X0[256], Y0[256], Z0[256];  // Precomputed nodes and offsets for 8-bit input data


} Prelin8Data;


// Generic optimization for 16 bits Shaper-CLUT-Shaper (any inputs)
typedef struct {

    cmsContext ContextID;

    // Number of channels
    cmsUInt32Number nInputs;
    cmsUInt32Number nOutputs;

    _cmsInterpFn16 EvalCurveIn16[MAX_INPUT_DIMENSIONS];       // The maximum number of input channels is known in advance
    cmsInterpParams*  ParamsCurveIn16[MAX_INPUT_DIMENSIONS];

    _cmsInterpFn16 EvalCLUT;            // The evaluator for 3D grid
    const cmsInterpParams* CLUTparams;  // (not-owned pointer)


    _cmsInterpFn16* EvalCurveOut16;       // Points to an array of curve evaluators in 16 bits (not-owned pointer)
    cmsInterpParams**  ParamsCurveOut16;  // Points to an array of references to interpolation params (not-owned pointer)


} Prelin16Data;


// Optimization for matrix-shaper in 8 bits. Numbers are operated in n.14 signed, tables are stored in 1.14 fixed

typedef cmsInt32Number cmsS1Fixed14Number;   // Note that this may hold more than 16 bits!

#define DOUBLE_TO_1FIXED14(x) ((cmsS1Fixed14Number) floor((x) * 16384.0 + 0.5))

typedef struct {

    cmsContext ContextID;

    cmsS1Fixed14Number Shaper1R[256];  // from 0..255 to 1.14  (0.0...1.0)
    cmsS1Fixed14Number Shaper1G[256];
    cmsS1Fixed14Number Shaper1B[256];

    cmsS1Fixed14Number Mat[3][3];     // n.14 to n.14 (needs a saturation after that)
    cmsS1Fixed14Number Off[3];

    cmsUInt16Number Shaper2R[16385];    // 1.14 to 0..255
    cmsUInt16Number Shaper2G[16385];
    cmsUInt16Number Shaper2B[16385];

} MatShaper8Data;

// Curves, optimization is shared between 8 and 16 bits
typedef struct {

    cmsContext ContextID;

    cmsUInt32Number nCurves;      // Number of curves
    cmsUInt32Number nElements;    // Elements in curves
    cmsUInt16Number** Curves;     // Points to a dynamically  allocated array

} Curves16Data;


// Simple optimizations ----------------------------------------------------------------------------------------------------------


// Remove an element in linked chain
static
void _RemoveElement(cmsStage** head)
{
    cmsStage* mpe = *head;
    cmsStage* next = mpe ->Next;
    *head = next;
    cmsStageFree(mpe);
}

// Remove all identities in chain. Note that pt actually is a double pointer to the element that holds the pointer.
static
cmsBool _Remove1Op(cmsPipeline* Lut, cmsStageSignature UnaryOp)
{
    cmsStage** pt = &Lut ->Elements;
    cmsBool AnyOpt = FALSE;

    while (*pt != NULL) {

        if ((*pt) ->Implements == UnaryOp) {
            _RemoveElement(pt);
            AnyOpt = TRUE;
        }
        else
            pt = &((*pt) -> Next);
    }

    return AnyOpt;
}

// Same, but only if two adjacent elements are found
static
cmsBool _Remove2Op(cmsPipeline* Lut, cmsStageSignature Op1, cmsStageSignature Op2)
{
    cmsStage** pt1;
    cmsStage** pt2;
    cmsBool AnyOpt = FALSE;

    pt1 = &Lut ->Elements;
    if (*pt1 == NULL) return AnyOpt;

    while (*pt1 != NULL) {

        pt2 = &((*pt1) -> Next);
        if (*pt2 == NULL) return AnyOpt;

        if ((*pt1) ->Implements == Op1 && (*pt2) ->Implements == Op2) {
            _RemoveElement(pt2);
            _RemoveElement(pt1);
            AnyOpt = TRUE;
        }
        else
            pt1 = &((*pt1) -> Next);
    }

    return AnyOpt;
}


static
cmsBool CloseEnoughFloat(cmsFloat64Number a, cmsFloat64Number b)
{
       return fabs(b - a) < 0.00001f;
}

static
cmsBool  isFloatMatrixIdentity(const cmsMAT3* a)
{
       cmsMAT3 Identity;
       int i, j;

       _cmsMAT3identity(&Identity);

       for (i = 0; i < 3; i++)
              for (j = 0; j < 3; j++)
                     if (!CloseEnoughFloat(a->v[i].n[j], Identity.v[i].n[j])) return FALSE;

       return TRUE;
}
// if two adjacent matrices are found, multiply them. 
static
cmsBool _MultiplyMatrix(cmsPipeline* Lut)
{
       cmsStage** pt1;
       cmsStage** pt2;
       cmsStage*  chain;
       cmsBool AnyOpt = FALSE;

       pt1 = &Lut->Elements;
       if (*pt1 == NULL) return AnyOpt;

       while (*pt1 != NULL) {

              pt2 = &((*pt1)->Next);
              if (*pt2 == NULL) return AnyOpt;

              if ((*pt1)->Implements == cmsSigMatrixElemType && (*pt2)->Implements == cmsSigMatrixElemType) {

                     // Get both matrices
                     _cmsStageMatrixData* m1 = (_cmsStageMatrixData*) cmsStageData(*pt1);
                     _cmsStageMatrixData* m2 = (_cmsStageMatrixData*) cmsStageData(*pt2);
                     cmsMAT3 res;
                     
                     // Input offset and output offset should be zero to use this optimization
                     if (m1->Offset != NULL || m2 ->Offset != NULL || 
                            cmsStageInputChannels(*pt1) != 3 || cmsStageOutputChannels(*pt1) != 3 ||                            
                            cmsStageInputChannels(*pt2) != 3 || cmsStageOutputChannels(*pt2) != 3)
                            return FALSE;

                     // Multiply both matrices to get the result
                     _cmsMAT3per(&res, (cmsMAT3*)m2->Double, (cmsMAT3*)m1->Double);

                     // Get the next in chain after the matrices
                     chain = (*pt2)->Next;

                     // Remove both matrices
                     _RemoveElement(pt2);
                     _RemoveElement(pt1);

                     // Now what if the result is a plain identity?                     
                     if (!isFloatMatrixIdentity(&res)) {

                            // We can not get rid of full matrix                            
                            cmsStage* Multmat = cmsStageAllocMatrix(Lut->ContextID, 3, 3, (const cmsFloat64Number*) &res, NULL);
                            if (Multmat == NULL) return FALSE;  // Should never happen

                            // Recover the chain
                            Multmat->Next = chain;
                            *pt1 = Multmat;
                     }

                     AnyOpt = TRUE;
              }
              else
                     pt1 = &((*pt1)->Next);
       }

       return AnyOpt;
}


// Preoptimize just gets rif of no-ops coming paired. Conversion from v2 to v4 followed
// by a v4 to v2 and vice-versa. The elements are then discarded.
static
cmsBool PreOptimize(cmsPipeline* Lut)
{
    cmsBool AnyOpt = FALSE, Opt;

    do {

        Opt = FALSE;

        // Remove all identities
        Opt |= _Remove1Op(Lut, cmsSigIdentityElemType);

        // Remove XYZ2Lab followed by Lab2XYZ
        Opt |= _Remove2Op(Lut, cmsSigXYZ2LabElemType, cmsSigLab2XYZElemType);

        // Remove Lab2XYZ followed by XYZ2Lab
        Opt |= _Remove2Op(Lut, cmsSigLab2XYZElemType, cmsSigXYZ2LabElemType);

        // Remove V4 to V2 followed by V2 to V4
        Opt |= _Remove2Op(Lut, cmsSigLabV4toV2, cmsSigLabV2toV4);

        // Remove V2 to V4 followed by V4 to V2
        Opt |= _Remove2Op(Lut, cmsSigLabV2toV4, cmsSigLabV4toV2);

        // Remove float pcs Lab conversions
        Opt |= _Remove2Op(Lut, cmsSigLab2FloatPCS, cmsSigFloatPCS2Lab);

        // Remove float pcs Lab conversions
        Opt |= _Remove2Op(Lut, cmsSigXYZ2FloatPCS, cmsSigFloatPCS2XYZ);

        // Simplify matrix. 
        Opt |= _MultiplyMatrix(Lut);

        if (Opt) AnyOpt = TRUE;

    } while (Opt);

    return AnyOpt;
}

static
void Eval16nop1D(register const cmsUInt16Number Input[],
                 register cmsUInt16Number Output[],
                 register const struct _cms_interp_struc* p)
{
    Output[0] = Input[0];

    cmsUNUSED_PARAMETER(p);
}

static
void PrelinEval16(register const cmsUInt16Number Input[],
                  register cmsUInt16Number Output[],
                  register const void* D)
{
    Prelin16Data* p16 = (Prelin16Data*) D;
    cmsUInt16Number  StageABC[MAX_INPUT_DIMENSIONS];
    cmsUInt16Number  StageDEF[cmsMAXCHANNELS];
    cmsUInt32Number i;

    for (i=0; i < p16 ->nInputs; i++) {

        p16 ->EvalCurveIn16[i](&Input[i], &StageABC[i], p16 ->ParamsCurveIn16[i]);
    }

    p16 ->EvalCLUT(StageABC, StageDEF, p16 ->CLUTparams);

    for (i=0; i < p16 ->nOutputs; i++) {

        p16 ->EvalCurveOut16[i](&StageDEF[i], &Output[i], p16 ->ParamsCurveOut16[i]);
    }
}


static
void PrelinOpt16free(cmsContext ContextID, void* ptr)
{
    Prelin16Data* p16 = (Prelin16Data*) ptr;

    _cmsFree(ContextID, p16 ->EvalCurveOut16);
    _cmsFree(ContextID, p16 ->ParamsCurveOut16);

    _cmsFree(ContextID, p16);
}

static
void* Prelin16dup(cmsContext ContextID, const void* ptr)
{
    Prelin16Data* p16 = (Prelin16Data*) ptr;
    Prelin16Data* Duped = (Prelin16Data*) _cmsDupMem(ContextID, p16, sizeof(Prelin16Data));

    if (Duped == NULL) return NULL;

    Duped->EvalCurveOut16 = (_cmsInterpFn16*) _cmsDupMem(ContextID, p16->EvalCurveOut16, p16->nOutputs * sizeof(_cmsInterpFn16));
    Duped->ParamsCurveOut16 = (cmsInterpParams**)_cmsDupMem(ContextID, p16->ParamsCurveOut16, p16->nOutputs * sizeof(cmsInterpParams*));

    return Duped;
}


static
Prelin16Data* PrelinOpt16alloc(cmsContext ContextID,
                               const cmsInterpParams* ColorMap,
                               cmsUInt32Number nInputs, cmsToneCurve** In,
                               cmsUInt32Number nOutputs, cmsToneCurve** Out )
{
    cmsUInt32Number i;
    Prelin16Data* p16 = (Prelin16Data*)_cmsMallocZero(ContextID, sizeof(Prelin16Data));
    if (p16 == NULL) return NULL;

    p16 ->nInputs = nInputs;
    p16 ->nOutputs = nOutputs;


    for (i=0; i < nInputs; i++) {

        if (In == NULL) {
            p16 -> ParamsCurveIn16[i] = NULL;
            p16 -> EvalCurveIn16[i] = Eval16nop1D;

        }
        else {
            p16 -> ParamsCurveIn16[i] = In[i] ->InterpParams;
            p16 -> EvalCurveIn16[i] = p16 ->ParamsCurveIn16[i]->Interpolation.Lerp16;
        }
    }

    p16 ->CLUTparams = ColorMap;
    p16 ->EvalCLUT   = ColorMap ->Interpolation.Lerp16;


    p16 -> EvalCurveOut16 = (_cmsInterpFn16*) _cmsCalloc(ContextID, nOutputs, sizeof(_cmsInterpFn16));
    p16 -> ParamsCurveOut16 = (cmsInterpParams**) _cmsCalloc(ContextID, nOutputs, sizeof(cmsInterpParams* ));

    for (i=0; i < nOutputs; i++) {

        if (Out == NULL) {
            p16 ->ParamsCurveOut16[i] = NULL;
            p16 -> EvalCurveOut16[i] = Eval16nop1D;
        }
        else {

            p16 ->ParamsCurveOut16[i] = Out[i] ->InterpParams;
            p16 -> EvalCurveOut16[i] = p16 ->ParamsCurveOut16[i]->Interpolation.Lerp16;
        }
    }

    return p16;
}



// Resampling ---------------------------------------------------------------------------------

#define PRELINEARIZATION_POINTS 4096

// Sampler implemented by another LUT. This is a clean way to precalculate the devicelink 3D CLUT for
// almost any transform. We use floating point precision and then convert from floating point to 16 bits.
static
cmsInt32Number XFormSampler16(register const cmsUInt16Number In[], register cmsUInt16Number Out[], register void* Cargo)
{
    cmsPipeline* Lut = (cmsPipeline*) Cargo;
    cmsFloat32Number InFloat[cmsMAXCHANNELS], OutFloat[cmsMAXCHANNELS];
    cmsUInt32Number i;

    _cmsAssert(Lut -> InputChannels < cmsMAXCHANNELS);
    _cmsAssert(Lut -> OutputChannels < cmsMAXCHANNELS);

    // From 16 bit to floating point
    for (i=0; i < Lut ->InputChannels; i++)
        InFloat[i] = (cmsFloat32Number) (In[i] / 65535.0);

    // Evaluate in floating point
    cmsPipelineEvalFloat(InFloat, OutFloat, Lut);

    // Back to 16 bits representation
    for (i=0; i < Lut ->OutputChannels; i++)
        Out[i] = _cmsQuickSaturateWord(OutFloat[i] * 65535.0);

    // Always succeed
    return TRUE;
}

// Try to see if the curves of a given MPE are linear
static
cmsBool AllCurvesAreLinear(cmsStage* mpe)
{
    cmsToneCurve** Curves;
    cmsUInt32Number i, n;

    Curves = _cmsStageGetPtrToCurveSet(mpe);
    if (Curves == NULL) return FALSE;

    n = cmsStageOutputChannels(mpe);

    for (i=0; i < n; i++) {
        if (!cmsIsToneCurveLinear(Curves[i])) return FALSE;
    }

    return TRUE;
}

// This function replaces a specific node placed in "At" by the "Value" numbers. Its purpose
// is to fix scum dot on broken profiles/transforms. Works on 1, 3 and 4 channels
static
cmsBool  PatchLUT(cmsStage* CLUT, cmsUInt16Number At[], cmsUInt16Number Value[],
                  cmsUInt32Number nChannelsOut, cmsUInt32Number nChannelsIn)
{
    _cmsStageCLutData* Grid = (_cmsStageCLutData*) CLUT ->Data;
    cmsInterpParams* p16  = Grid ->Params;
    cmsFloat64Number px, py, pz, pw;
    int        x0, y0, z0, w0;
    int        i, index;

    if (CLUT -> Type != cmsSigCLutElemType) {
        cmsSignalError(CLUT->ContextID, cmsERROR_INTERNAL, "(internal) Attempt to PatchLUT on non-lut stage");
        return FALSE;
    }

    if (nChannelsIn == 4) {

        px = ((cmsFloat64Number) At[0] * (p16->Domain[0])) / 65535.0;
        py = ((cmsFloat64Number) At[1] * (p16->Domain[1])) / 65535.0;
        pz = ((cmsFloat64Number) At[2] * (p16->Domain[2])) / 65535.0;
        pw = ((cmsFloat64Number) At[3] * (p16->Domain[3])) / 65535.0;

        x0 = (int) floor(px);
        y0 = (int) floor(py);
        z0 = (int) floor(pz);
        w0 = (int) floor(pw);

        if (((px - x0) != 0) ||
            ((py - y0) != 0) ||
            ((pz - z0) != 0) ||
            ((pw - w0) != 0)) return FALSE; // Not on exact node

        index = (int) p16 -> opta[3] * x0 +
                (int) p16 -> opta[2] * y0 +
                (int) p16 -> opta[1] * z0 +
                (int) p16 -> opta[0] * w0;
    }
    else
        if (nChannelsIn == 3) {

            px = ((cmsFloat64Number) At[0] * (p16->Domain[0])) / 65535.0;
            py = ((cmsFloat64Number) At[1] * (p16->Domain[1])) / 65535.0;
            pz = ((cmsFloat64Number) At[2] * (p16->Domain[2])) / 65535.0;
           
            x0 = (int) floor(px);
            y0 = (int) floor(py);
            z0 = (int) floor(pz);
           
            if (((px - x0) != 0) ||
                ((py - y0) != 0) ||
                ((pz - z0) != 0)) return FALSE;  // Not on exact node

            index = (int) p16 -> opta[2] * x0 +
                    (int) p16 -> opta[1] * y0 +
                    (int) p16 -> opta[0] * z0;
        }
        else
            if (nChannelsIn == 1) {

                px = ((cmsFloat64Number) At[0] * (p16->Domain[0])) / 65535.0;
                
                x0 = (int) floor(px);
                
                if (((px - x0) != 0)) return FALSE; // Not on exact node

                index = (int) p16 -> opta[0] * x0;
            }
            else {
                cmsSignalError(CLUT->ContextID, cmsERROR_INTERNAL, "(internal) %d Channels are not supported on PatchLUT", nChannelsIn);
                return FALSE;
            }

            for (i = 0; i < (int) nChannelsOut; i++)
                Grid->Tab.T[index + i] = Value[i];

            return TRUE;
}

// Auxiliary, to see if two values are equal or very different
static
cmsBool WhitesAreEqual(cmsUInt32Number n, cmsUInt16Number White1[], cmsUInt16Number White2[] )
{
    cmsUInt32Number i;

    for (i=0; i < n; i++) {

        if (abs(White1[i] - White2[i]) > 0xf000) return TRUE;  // Values are so extremely different that the fixup should be avoided
        if (White1[i] != White2[i]) return FALSE;
    }
    return TRUE;
}


// Locate the node for the white point and fix it to pure white in order to avoid scum dot.
static
cmsBool FixWhiteMisalignment(cmsPipeline* Lut, cmsColorSpaceSignature EntryColorSpace, cmsColorSpaceSignature ExitColorSpace)
{
    cmsUInt16Number *WhitePointIn, *WhitePointOut;
    cmsUInt16Number  WhiteIn[cmsMAXCHANNELS], WhiteOut[cmsMAXCHANNELS], ObtainedOut[cmsMAXCHANNELS];
    cmsUInt32Number i, nOuts, nIns;
    cmsStage *PreLin = NULL, *CLUT = NULL, *PostLin = NULL;

    if (!_cmsEndPointsBySpace(EntryColorSpace,
        &WhitePointIn, NULL, &nIns)) return FALSE;

    if (!_cmsEndPointsBySpace(ExitColorSpace,
        &WhitePointOut, NULL, &nOuts)) return FALSE;

    // It needs to be fixed?
    if (Lut ->InputChannels != nIns) return FALSE;
    if (Lut ->OutputChannels != nOuts) return FALSE;

    cmsPipelineEval16(WhitePointIn, ObtainedOut, Lut);

    if (WhitesAreEqual(nOuts, WhitePointOut, ObtainedOut)) return TRUE; // whites already match

    // Check if the LUT comes as Prelin, CLUT or Postlin. We allow all combinations
    if (!cmsPipelineCheckAndRetreiveStages(Lut, 3, cmsSigCurveSetElemType, cmsSigCLutElemType, cmsSigCurveSetElemType, &PreLin, &CLUT, &PostLin))
        if (!cmsPipelineCheckAndRetreiveStages(Lut, 2, cmsSigCurveSetElemType, cmsSigCLutElemType, &PreLin, &CLUT))
            if (!cmsPipelineCheckAndRetreiveStages(Lut, 2, cmsSigCLutElemType, cmsSigCurveSetElemType, &CLUT, &PostLin))
                if (!cmsPipelineCheckAndRetreiveStages(Lut, 1, cmsSigCLutElemType, &CLUT))
                    return FALSE;

    // We need to interpolate white points of both, pre and post curves
    if (PreLin) {

        cmsToneCurve** Curves = _cmsStageGetPtrToCurveSet(PreLin);

        for (i=0; i < nIns; i++) {
            WhiteIn[i] = cmsEvalToneCurve16(Curves[i], WhitePointIn[i]);
        }
    }
    else {
        for (i=0; i < nIns; i++)
            WhiteIn[i] = WhitePointIn[i];
    }

    // If any post-linearization, we need to find how is represented white before the curve, do
    // a reverse interpolation in this case.
    if (PostLin) {

        cmsToneCurve** Curves = _cmsStageGetPtrToCurveSet(PostLin);

        for (i=0; i < nOuts; i++) {

            cmsToneCurve* InversePostLin = cmsReverseToneCurve(Curves[i]);
            if (InversePostLin == NULL) {
                WhiteOut[i] = WhitePointOut[i];    

            } else {

                WhiteOut[i] = cmsEvalToneCurve16(InversePostLin, WhitePointOut[i]);
                cmsFreeToneCurve(InversePostLin);
            }
        }
    }
    else {
        for (i=0; i < nOuts; i++)
            WhiteOut[i] = WhitePointOut[i];
    }

    // Ok, proceed with patching. May fail and we don't care if it fails
    PatchLUT(CLUT, WhiteIn, WhiteOut, nOuts, nIns);

    return TRUE;
}

// -----------------------------------------------------------------------------------------------------------------------------------------------
// This function creates simple LUT from complex ones. The generated LUT has an optional set of
// prelinearization curves, a CLUT of nGridPoints and optional postlinearization tables.
// These curves have to exist in the original LUT in order to be used in the simplified output.
// Caller may also use the flags to allow this feature.
// LUTS with all curves will be simplified to a single curve. Parametric curves are lost.
// This function should be used on 16-bits LUTS only, as floating point losses precision when simplified
// -----------------------------------------------------------------------------------------------------------------------------------------------

static
cmsBool OptimizeByResampling(cmsPipeline** Lut, cmsUInt32Number Intent, cmsUInt32Number* InputFormat, cmsUInt32Number* OutputFormat, cmsUInt32Number* dwFlags)
{
    cmsPipeline* Src = NULL;
    cmsPipeline* Dest = NULL;
    cmsStage* mpe;
    cmsStage* CLUT;
    cmsStage *KeepPreLin = NULL, *KeepPostLin = NULL;
    cmsUInt32Number nGridPoints;
    cmsColorSpaceSignature ColorSpace, OutputColorSpace;
    cmsStage *NewPreLin = NULL;
    cmsStage *NewPostLin = NULL;
    _cmsStageCLutData* DataCLUT;
    cmsToneCurve** DataSetIn;
    cmsToneCurve** DataSetOut;
    Prelin16Data* p16;

    // This is a loosy optimization! does not apply in floating-point cases
    if (_cmsFormatterIsFloat(*InputFormat) || _cmsFormatterIsFloat(*OutputFormat)) return FALSE;

    ColorSpace       = _cmsICCcolorSpace((int) T_COLORSPACE(*InputFormat));
    OutputColorSpace = _cmsICCcolorSpace((int) T_COLORSPACE(*OutputFormat));

    // Color space must be specified
    if (ColorSpace == (cmsColorSpaceSignature)0 ||
        OutputColorSpace == (cmsColorSpaceSignature)0) return FALSE;

    nGridPoints      = _cmsReasonableGridpointsByColorspace(ColorSpace, *dwFlags);

    // For empty LUTs, 2 points are enough
    if (cmsPipelineStageCount(*Lut) == 0)
        nGridPoints = 2;

    Src = *Lut;

    // Named color pipelines cannot be optimized either
    for (mpe = cmsPipelineGetPtrToFirstStage(Src);
        mpe != NULL;
        mpe = cmsStageNext(mpe)) {
            if (cmsStageType(mpe) == cmsSigNamedColorElemType) return FALSE;
    }

    // Allocate an empty LUT
    Dest =  cmsPipelineAlloc(Src ->ContextID, Src ->InputChannels, Src ->OutputChannels);
    if (!Dest) return FALSE;

    // Prelinearization tables are kept unless indicated by flags
    if (*dwFlags & cmsFLAGS_CLUT_PRE_LINEARIZATION) {

        // Get a pointer to the prelinearization element
        cmsStage* PreLin = cmsPipelineGetPtrToFirstStage(Src);

        // Check if suitable
        if (PreLin && PreLin ->Type == cmsSigCurveSetElemType) {

            // Maybe this is a linear tram, so we can avoid the whole stuff
            if (!AllCurvesAreLinear(PreLin)) {

                // All seems ok, proceed.
                NewPreLin = cmsStageDup(PreLin);
                if(!cmsPipelineInsertStage(Dest, cmsAT_BEGIN, NewPreLin))
                    goto Error;

                // Remove prelinearization. Since we have duplicated the curve
                // in destination LUT, the sampling should be applied after this stage.
                cmsPipelineUnlinkStage(Src, cmsAT_BEGIN, &KeepPreLin);
            }
        }
    }

    // Allocate the CLUT
    CLUT = cmsStageAllocCLut16bit(Src ->ContextID, nGridPoints, Src ->InputChannels, Src->OutputChannels, NULL);
    if (CLUT == NULL) goto Error;

    // Add the CLUT to the destination LUT
    if (!cmsPipelineInsertStage(Dest, cmsAT_END, CLUT)) {
        goto Error;
    }

    // Postlinearization tables are kept unless indicated by flags
    if (*dwFlags & cmsFLAGS_CLUT_POST_LINEARIZATION) {

        // Get a pointer to the postlinearization if present
        cmsStage* PostLin = cmsPipelineGetPtrToLastStage(Src);

        // Check if suitable
        if (PostLin && cmsStageType(PostLin) == cmsSigCurveSetElemType) {

            // Maybe this is a linear tram, so we can avoid the whole stuff
            if (!AllCurvesAreLinear(PostLin)) {

                // All seems ok, proceed.
                NewPostLin = cmsStageDup(PostLin);
                if (!cmsPipelineInsertStage(Dest, cmsAT_END, NewPostLin))
                    goto Error;

                // In destination LUT, the sampling should be applied after this stage.
                cmsPipelineUnlinkStage(Src, cmsAT_END, &KeepPostLin);
            }
        }
    }

    // Now its time to do the sampling. We have to ignore pre/post linearization
    // The source LUT without pre/post curves is passed as parameter.
    if (!cmsStageSampleCLut16bit(CLUT, XFormSampler16, (void*) Src, 0)) {
Error:
        // Ops, something went wrong, Restore stages
        if (KeepPreLin != NULL) {
            if (!cmsPipelineInsertStage(Src, cmsAT_BEGIN, KeepPreLin)) {
                _cmsAssert(0); // This never happens
            }
        }
        if (KeepPostLin != NULL) {
            if (!cmsPipelineInsertStage(Src, cmsAT_END,   KeepPostLin)) {
                _cmsAssert(0); // This never happens
            }
        }
        cmsPipelineFree(Dest);
        return FALSE;
    }

    // Done.

    if (KeepPreLin != NULL) cmsStageFree(KeepPreLin);
    if (KeepPostLin != NULL) cmsStageFree(KeepPostLin);
    cmsPipelineFree(Src);

    DataCLUT = (_cmsStageCLutData*) CLUT ->Data;

    if (NewPreLin == NULL) DataSetIn = NULL;
    else DataSetIn = ((_cmsStageToneCurvesData*) NewPreLin ->Data) ->TheCurves;

    if (NewPostLin == NULL) DataSetOut = NULL;
    else  DataSetOut = ((_cmsStageToneCurvesData*) NewPostLin ->Data) ->TheCurves;


    if (DataSetIn == NULL && DataSetOut == NULL) {

        _cmsPipelineSetOptimizationParameters(Dest, (_cmsOPTeval16Fn) DataCLUT->Params->Interpolation.Lerp16, DataCLUT->Params, NULL, NULL);
    }
    else {

        p16 = PrelinOpt16alloc(Dest ->ContextID,
            DataCLUT ->Params,
            Dest ->InputChannels,
            DataSetIn,
            Dest ->OutputChannels,
            DataSetOut);

        _cmsPipelineSetOptimizationParameters(Dest, PrelinEval16, (void*) p16, PrelinOpt16free, Prelin16dup);
    }


    // Don't fix white on absolute colorimetric
    if (Intent == INTENT_ABSOLUTE_COLORIMETRIC)
        *dwFlags |= cmsFLAGS_NOWHITEONWHITEFIXUP;

    if (!(*dwFlags & cmsFLAGS_NOWHITEONWHITEFIXUP)) {

        FixWhiteMisalignment(Dest, ColorSpace, OutputColorSpace);
    }

    *Lut = Dest;
    return TRUE;

    cmsUNUSED_PARAMETER(Intent);
}


// -----------------------------------------------------------------------------------------------------------------------------------------------
// Fixes the gamma balancing of transform. This is described in my paper "Prelinearization Stages on
// Color-Management Application-Specific Integrated Circuits (ASICs)" presented at NIP24. It only works
// for RGB transforms. See the paper for more details
// -----------------------------------------------------------------------------------------------------------------------------------------------


// Normalize endpoints by slope limiting max and min. This assures endpoints as well.
// Descending curves are handled as well.
static
void SlopeLimiting(cmsToneCurve* g)
{
    int BeginVal, EndVal;
    int AtBegin = (int) floor((cmsFloat64Number) g ->nEntries * 0.02 + 0.5);   // Cutoff at 2%
    int AtEnd   = (int) g ->nEntries - AtBegin - 1;                                  // And 98%
    cmsFloat64Number Val, Slope, beta;
    int i;

    if (cmsIsToneCurveDescending(g)) {
        BeginVal = 0xffff; EndVal = 0;
    }
    else {
        BeginVal = 0; EndVal = 0xffff;
    }

    // Compute slope and offset for begin of curve
    Val   = g ->Table16[AtBegin];
    Slope = (Val - BeginVal) / AtBegin;
    beta  = Val - Slope * AtBegin;

    for (i=0; i < AtBegin; i++)
        g ->Table16[i] = _cmsQuickSaturateWord(i * Slope + beta);

    // Compute slope and offset for the end
    Val   = g ->Table16[AtEnd];
    Slope = (EndVal - Val) / AtBegin;   // AtBegin holds the X interval, which is same in both cases
    beta  = Val - Slope * AtEnd;

    for (i = AtEnd; i < (int) g ->nEntries; i++)
        g ->Table16[i] = _cmsQuickSaturateWord(i * Slope + beta);
}


// Precomputes tables for 8-bit on input devicelink.
static
Prelin8Data* PrelinOpt8alloc(cmsContext ContextID, const cmsInterpParams* p, cmsToneCurve* G[3])
{
    int i;
    cmsUInt16Number Input[3];
    cmsS15Fixed16Number v1, v2, v3;
    Prelin8Data* p8;

    p8 = (Prelin8Data*)_cmsMallocZero(ContextID, sizeof(Prelin8Data));
    if (p8 == NULL) return NULL;

    // Since this only works for 8 bit input, values comes always as x * 257,
    // we can safely take msb byte (x << 8 + x)

    for (i=0; i < 256; i++) {

        if (G != NULL) {

            // Get 16-bit representation
            Input[0] = cmsEvalToneCurve16(G[0], FROM_8_TO_16(i));
            Input[1] = cmsEvalToneCurve16(G[1], FROM_8_TO_16(i));
            Input[2] = cmsEvalToneCurve16(G[2], FROM_8_TO_16(i));
        }
        else {
            Input[0] = FROM_8_TO_16(i);
            Input[1] = FROM_8_TO_16(i);
            Input[2] = FROM_8_TO_16(i);
        }


        // Move to 0..1.0 in fixed domain
        v1 = _cmsToFixedDomain((int) (Input[0] * p -> Domain[0]));
        v2 = _cmsToFixedDomain((int) (Input[1] * p -> Domain[1]));
        v3 = _cmsToFixedDomain((int) (Input[2] * p -> Domain[2]));

        // Store the precalculated table of nodes
        p8 ->X0[i] = (p->opta[2] * FIXED_TO_INT(v1));
        p8 ->Y0[i] = (p->opta[1] * FIXED_TO_INT(v2));
        p8 ->Z0[i] = (p->opta[0] * FIXED_TO_INT(v3));

        // Store the precalculated table of offsets
        p8 ->rx[i] = (cmsUInt16Number) FIXED_REST_TO_INT(v1);
        p8 ->ry[i] = (cmsUInt16Number) FIXED_REST_TO_INT(v2);
        p8 ->rz[i] = (cmsUInt16Number) FIXED_REST_TO_INT(v3);
    }

    p8 ->ContextID = ContextID;
    p8 ->p = p;

    return p8;
}

static
void Prelin8free(cmsContext ContextID, void* ptr)
{
    _cmsFree(ContextID, ptr);
}

static
void* Prelin8dup(cmsContext ContextID, const void* ptr)
{
    return _cmsDupMem(ContextID, ptr, sizeof(Prelin8Data));
}



// A optimized interpolation for 8-bit input.
#define DENS(i,j,k) (LutTable[(i)+(j)+(k)+OutChan])
static
void PrelinEval8(register const cmsUInt16Number Input[],
                  register cmsUInt16Number Output[],
                  register const void* D)
{

    cmsUInt8Number         r, g, b;
    cmsS15Fixed16Number    rx, ry, rz;
    cmsS15Fixed16Number    c0, c1, c2, c3, Rest;
    int                    OutChan;
    register cmsS15Fixed16Number X0, X1, Y0, Y1, Z0, Z1;
    Prelin8Data* p8 = (Prelin8Data*) D;
    register const cmsInterpParams* p = p8 ->p;
    int                    TotalOut = (int) p -> nOutputs;
    const cmsUInt16Number* LutTable = (const cmsUInt16Number*) p->Table;

    r = (cmsUInt8Number) (Input[0] >> 8);
    g = (cmsUInt8Number) (Input[1] >> 8);
    b = (cmsUInt8Number) (Input[2] >> 8);

    X0 = X1 = (cmsS15Fixed16Number) p8->X0[r];
    Y0 = Y1 = (cmsS15Fixed16Number) p8->Y0[g];
    Z0 = Z1 = (cmsS15Fixed16Number) p8->Z0[b];

    rx = p8 ->rx[r];
    ry = p8 ->ry[g];
    rz = p8 ->rz[b];

    X1 = X0 + (cmsS15Fixed16Number)((rx == 0) ? 0 :  p ->opta[2]);
    Y1 = Y0 + (cmsS15Fixed16Number)((ry == 0) ? 0 :  p ->opta[1]);
    Z1 = Z0 + (cmsS15Fixed16Number)((rz == 0) ? 0 :  p ->opta[0]);


    // These are the 6 Tetrahedral
    for (OutChan=0; OutChan < TotalOut; OutChan++) {

        c0 = DENS(X0, Y0, Z0);

        if (rx >= ry && ry >= rz)
        {
            c1 = DENS(X1, Y0, Z0) - c0;
            c2 = DENS(X1, Y1, Z0) - DENS(X1, Y0, Z0);
            c3 = DENS(X1, Y1, Z1) - DENS(X1, Y1, Z0);
        }
        else
            if (rx >= rz && rz >= ry)
            {
                c1 = DENS(X1, Y0, Z0) - c0;
                c2 = DENS(X1, Y1, Z1) - DENS(X1, Y0, Z1);
                c3 = DENS(X1, Y0, Z1) - DENS(X1, Y0, Z0);
            }
            else
                if (rz >= rx && rx >= ry)
                {
                    c1 = DENS(X1, Y0, Z1) - DENS(X0, Y0, Z1);
                    c2 = DENS(X1, Y1, Z1) - DENS(X1, Y0, Z1);
                    c3 = DENS(X0, Y0, Z1) - c0;
                }
                else
                    if (ry >= rx && rx >= rz)
                    {
                        c1 = DENS(X1, Y1, Z0) - DENS(X0, Y1, Z0);
                        c2 = DENS(X0, Y1, Z0) - c0;
                        c3 = DENS(X1, Y1, Z1) - DENS(X1, Y1, Z0);
                    }
                    else
                        if (ry >= rz && rz >= rx)
                        {
                            c1 = DENS(X1, Y1, Z1) - DENS(X0, Y1, Z1);
                            c2 = DENS(X0, Y1, Z0) - c0;
                            c3 = DENS(X0, Y1, Z1) - DENS(X0, Y1, Z0);
                        }
                        else
                            if (rz >= ry && ry >= rx)
                            {
                                c1 = DENS(X1, Y1, Z1) - DENS(X0, Y1, Z1);
                                c2 = DENS(X0, Y1, Z1) - DENS(X0, Y0, Z1);
                                c3 = DENS(X0, Y0, Z1) - c0;
                            }
                            else  {
                                c1 = c2 = c3 = 0;
                            }

                            Rest = c1 * rx + c2 * ry + c3 * rz + 0x8001;
                            Output[OutChan] = (cmsUInt16Number) (c0 + ((Rest + (Rest >> 16)) >> 16));

    }
}

#undef DENS


// Curves that contain wide empty areas are not optimizeable
static
cmsBool IsDegenerated(const cmsToneCurve* g)
{
    cmsUInt32Number i, Zeros = 0, Poles = 0;
    cmsUInt32Number nEntries = g ->nEntries;

    for (i=0; i < nEntries; i++) {

        if (g ->Table16[i] == 0x0000) Zeros++;
        if (g ->Table16[i] == 0xffff) Poles++;
    }

    if (Zeros == 1 && Poles == 1) return FALSE;  // For linear tables
    if (Zeros > (nEntries / 20)) return TRUE;  // Degenerated, many zeros
    if (Poles > (nEntries / 20)) return TRUE;  // Degenerated, many poles

    return FALSE;
}

// --------------------------------------------------------------------------------------------------------------
// We need xput over here

static
cmsBool OptimizeByComputingLinearization(cmsPipeline** Lut, cmsUInt32Number Intent, cmsUInt32Number* InputFormat, cmsUInt32Number* OutputFormat, cmsUInt32Number* dwFlags)
{
    cmsPipeline* OriginalLut;
    cmsUInt32Number nGridPoints;
    cmsToneCurve *Trans[cmsMAXCHANNELS], *TransReverse[cmsMAXCHANNELS];
    cmsUInt32Number t, i;
    cmsFloat32Number v, In[cmsMAXCHANNELS], Out[cmsMAXCHANNELS];
    cmsBool lIsSuitable, lIsLinear;
    cmsPipeline* OptimizedLUT = NULL, *LutPlusCurves = NULL;
    cmsStage* OptimizedCLUTmpe;
    cmsColorSpaceSignature ColorSpace, OutputColorSpace;
    cmsStage* OptimizedPrelinMpe;
    cmsStage* mpe;
    cmsToneCurve** OptimizedPrelinCurves;
    _cmsStageCLutData* OptimizedPrelinCLUT;


    // This is a loosy optimization! does not apply in floating-point cases
    if (_cmsFormatterIsFloat(*InputFormat) || _cmsFormatterIsFloat(*OutputFormat)) return FALSE;

    // Only on chunky RGB
    if (T_COLORSPACE(*InputFormat)  != PT_RGB) return FALSE;
    if (T_PLANAR(*InputFormat)) return FALSE;

    if (T_COLORSPACE(*OutputFormat) != PT_RGB) return FALSE;
    if (T_PLANAR(*OutputFormat)) return FALSE;

    // On 16 bits, user has to specify the feature
    if (!_cmsFormatterIs8bit(*InputFormat)) {
        if (!(*dwFlags & cmsFLAGS_CLUT_PRE_LINEARIZATION)) return FALSE;
    }

    OriginalLut = *Lut;

   // Named color pipelines cannot be optimized either
   for (mpe = cmsPipelineGetPtrToFirstStage(OriginalLut);
         mpe != NULL;
         mpe = cmsStageNext(mpe)) {
            if (cmsStageType(mpe) == cmsSigNamedColorElemType) return FALSE;
    }

    ColorSpace       = _cmsICCcolorSpace((int) T_COLORSPACE(*InputFormat));
    OutputColorSpace = _cmsICCcolorSpace((int) T_COLORSPACE(*OutputFormat));

    // Color space must be specified
    if (ColorSpace == (cmsColorSpaceSignature)0 ||
        OutputColorSpace == (cmsColorSpaceSignature)0) return FALSE;

    nGridPoints      = _cmsReasonableGridpointsByColorspace(ColorSpace, *dwFlags);

    // Empty gamma containers
    memset(Trans, 0, sizeof(Trans));
    memset(TransReverse, 0, sizeof(TransReverse));

    // If the last stage of the original lut are curves, and those curves are
    // degenerated, it is likely the transform is squeezing and clipping
    // the output from previous CLUT. We cannot optimize this case     
    {
        cmsStage* last = cmsPipelineGetPtrToLastStage(OriginalLut);

        if (cmsStageType(last) == cmsSigCurveSetElemType) {

            _cmsStageToneCurvesData* Data = (_cmsStageToneCurvesData*)cmsStageData(last);
            for (i = 0; i < Data->nCurves; i++) {
                if (IsDegenerated(Data->TheCurves[i]))
                    goto Error;
            }
        }
    }

    for (t = 0; t < OriginalLut ->InputChannels; t++) {
        Trans[t] = cmsBuildTabulatedToneCurve16(OriginalLut ->ContextID, PRELINEARIZATION_POINTS, NULL);
        if (Trans[t] == NULL) goto Error;
    }

    // Populate the curves
    for (i=0; i < PRELINEARIZATION_POINTS; i++) {

        v = (cmsFloat32Number) ((cmsFloat64Number) i / (PRELINEARIZATION_POINTS - 1));

        // Feed input with a gray ramp
        for (t=0; t < OriginalLut ->InputChannels; t++)
            In[t] = v;

        // Evaluate the gray value
        cmsPipelineEvalFloat(In, Out, OriginalLut);

        // Store result in curve
        for (t=0; t < OriginalLut ->InputChannels; t++)
            Trans[t] ->Table16[i] = _cmsQuickSaturateWord(Out[t] * 65535.0);
    }

    // Slope-limit the obtained curves
    for (t = 0; t < OriginalLut ->InputChannels; t++)
        SlopeLimiting(Trans[t]);

    // Check for validity
    lIsSuitable = TRUE;
    lIsLinear   = TRUE;
    for (t=0; (lIsSuitable && (t < OriginalLut ->InputChannels)); t++) {

        // Exclude if already linear
        if (!cmsIsToneCurveLinear(Trans[t]))
            lIsLinear = FALSE;

        // Exclude if non-monotonic
        if (!cmsIsToneCurveMonotonic(Trans[t]))
            lIsSuitable = FALSE;

        if (IsDegenerated(Trans[t]))
            lIsSuitable = FALSE;
    }

    // If it is not suitable, just quit
    if (!lIsSuitable) goto Error;

    // Invert curves if possible
    for (t = 0; t < OriginalLut ->InputChannels; t++) {
        TransReverse[t] = cmsReverseToneCurveEx(PRELINEARIZATION_POINTS, Trans[t]);
        if (TransReverse[t] == NULL) goto Error;
    }

    // Now inset the reversed curves at the begin of transform
    LutPlusCurves = cmsPipelineDup(OriginalLut);
    if (LutPlusCurves == NULL) goto Error;

    if (!cmsPipelineInsertStage(LutPlusCurves, cmsAT_BEGIN, cmsStageAllocToneCurves(OriginalLut ->ContextID, OriginalLut ->InputChannels, TransReverse)))
        goto Error;

    // Create the result LUT
    OptimizedLUT = cmsPipelineAlloc(OriginalLut ->ContextID, OriginalLut ->InputChannels, OriginalLut ->OutputChannels);
    if (OptimizedLUT == NULL) goto Error;

    OptimizedPrelinMpe = cmsStageAllocToneCurves(OriginalLut ->ContextID, OriginalLut ->InputChannels, Trans);

    // Create and insert the curves at the beginning
    if (!cmsPipelineInsertStage(OptimizedLUT, cmsAT_BEGIN, OptimizedPrelinMpe))
        goto Error;

    // Allocate the CLUT for result
    OptimizedCLUTmpe = cmsStageAllocCLut16bit(OriginalLut ->ContextID, nGridPoints, OriginalLut ->InputChannels, OriginalLut ->OutputChannels, NULL);

    // Add the CLUT to the destination LUT
    if (!cmsPipelineInsertStage(OptimizedLUT, cmsAT_END, OptimizedCLUTmpe))
        goto Error;

    // Resample the LUT
    if (!cmsStageSampleCLut16bit(OptimizedCLUTmpe, XFormSampler16, (void*) LutPlusCurves, 0)) goto Error;

    // Free resources
    for (t = 0; t < OriginalLut ->InputChannels; t++) {

        if (Trans[t]) cmsFreeToneCurve(Trans[t]);
        if (TransReverse[t]) cmsFreeToneCurve(TransReverse[t]);
    }

    cmsPipelineFree(LutPlusCurves);


    OptimizedPrelinCurves = _cmsStageGetPtrToCurveSet(OptimizedPrelinMpe);
    OptimizedPrelinCLUT   = (_cmsStageCLutData*) OptimizedCLUTmpe ->Data;

    // Set the evaluator if 8-bit
    if (_cmsFormatterIs8bit(*InputFormat)) {

        Prelin8Data* p8 = PrelinOpt8alloc(OptimizedLUT ->ContextID,
                                                OptimizedPrelinCLUT ->Params,
                                                OptimizedPrelinCurves);
        if (p8 == NULL) return FALSE;

        _cmsPipelineSetOptimizationParameters(OptimizedLUT, PrelinEval8, (void*) p8, Prelin8free, Prelin8dup);

    }
    else
    {
        Prelin16Data* p16 = PrelinOpt16alloc(OptimizedLUT ->ContextID,
            OptimizedPrelinCLUT ->Params,
            3, OptimizedPrelinCurves, 3, NULL);
        if (p16 == NULL) return FALSE;

        _cmsPipelineSetOptimizationParameters(OptimizedLUT, PrelinEval16, (void*) p16, PrelinOpt16free, Prelin16dup);

    }

    // Don't fix white on absolute colorimetric
    if (Intent == INTENT_ABSOLUTE_COLORIMETRIC)
        *dwFlags |= cmsFLAGS_NOWHITEONWHITEFIXUP;

    if (!(*dwFlags & cmsFLAGS_NOWHITEONWHITEFIXUP)) {

        if (!FixWhiteMisalignment(OptimizedLUT, ColorSpace, OutputColorSpace)) {

            return FALSE;
        }
    }

    // And return the obtained LUT

    cmsPipelineFree(OriginalLut);
    *Lut = OptimizedLUT;
    return TRUE;

Error:

    for (t = 0; t < OriginalLut ->InputChannels; t++) {

        if (Trans[t]) cmsFreeToneCurve(Trans[t]);
        if (TransReverse[t]) cmsFreeToneCurve(TransReverse[t]);
    }

    if (LutPlusCurves != NULL) cmsPipelineFree(LutPlusCurves);
    if (OptimizedLUT != NULL) cmsPipelineFree(OptimizedLUT);

    return FALSE;

    cmsUNUSED_PARAMETER(Intent);
    cmsUNUSED_PARAMETER(lIsLinear);
}


// Curves optimizer ------------------------------------------------------------------------------------------------------------------

static
void CurvesFree(cmsContext ContextID, void* ptr)
{
     Curves16Data* Data = (Curves16Data*) ptr;
     cmsUInt32Number i;

     for (i=0; i < Data -> nCurves; i++) {

         _cmsFree(ContextID, Data ->Curves[i]);
     }

     _cmsFree(ContextID, Data ->Curves);
     _cmsFree(ContextID, ptr);
}

static
void* CurvesDup(cmsContext ContextID, const void* ptr)
{
    Curves16Data* Data = (Curves16Data*)_cmsDupMem(ContextID, ptr, sizeof(Curves16Data));
    cmsUInt32Number i;

    if (Data == NULL) return NULL;

    Data->Curves = (cmsUInt16Number**) _cmsDupMem(ContextID, Data->Curves, Data->nCurves * sizeof(cmsUInt16Number*));

    for (i=0; i < Data -> nCurves; i++) {
        Data->Curves[i] = (cmsUInt16Number*) _cmsDupMem(ContextID, Data->Curves[i], Data->nElements * sizeof(cmsUInt16Number));
    }

    return (void*) Data;
}

// Precomputes tables for 8-bit on input devicelink.
static
Curves16Data* CurvesAlloc(cmsContext ContextID, cmsUInt32Number nCurves, cmsUInt32Number nElements, cmsToneCurve** G)
{
    cmsUInt32Number i, j;
    Curves16Data* c16;

    c16 = (Curves16Data*)_cmsMallocZero(ContextID, sizeof(Curves16Data));
    if (c16 == NULL) return NULL;

    c16 ->nCurves = nCurves;
    c16 ->nElements = nElements;

    c16->Curves = (cmsUInt16Number**) _cmsCalloc(ContextID, nCurves, sizeof(cmsUInt16Number*));
    if (c16->Curves == NULL) {
        _cmsFree(ContextID, c16);
        return NULL;
    }

    for (i=0; i < nCurves; i++) {

        c16->Curves[i] = (cmsUInt16Number*) _cmsCalloc(ContextID, nElements, sizeof(cmsUInt16Number));

        if (c16->Curves[i] == NULL) {

            for (j=0; j < i; j++) {
                _cmsFree(ContextID, c16->Curves[j]);
            }
            _cmsFree(ContextID, c16->Curves);
            _cmsFree(ContextID, c16);
            return NULL;
        }

        if (nElements == 256U) {

            for (j=0; j < nElements; j++) {

                c16 ->Curves[i][j] = cmsEvalToneCurve16(G[i], FROM_8_TO_16(j));
            }
        }
        else {

            for (j=0; j < nElements; j++) {
                c16 ->Curves[i][j] = cmsEvalToneCurve16(G[i], (cmsUInt16Number) j);
            }
        }
    }

    return c16;
}

static
void FastEvaluateCurves8(register const cmsUInt16Number In[],
                          register cmsUInt16Number Out[],
                          register const void* D)
{
    Curves16Data* Data = (Curves16Data*) D;
    int x;
    cmsUInt32Number i;

    for (i=0; i < Data ->nCurves; i++) {

         x = (In[i] >> 8);
         Out[i] = Data -> Curves[i][x];
    }
}


static
void FastEvaluateCurves16(register const cmsUInt16Number In[],
                          register cmsUInt16Number Out[],
                          register const void* D)
{
    Curves16Data* Data = (Curves16Data*) D;
    cmsUInt32Number i;

    for (i=0; i < Data ->nCurves; i++) {
         Out[i] = Data -> Curves[i][In[i]];
    }
}


static
void FastIdentity16(register const cmsUInt16Number In[],
                    register cmsUInt16Number Out[],
                    register const void* D)
{
    cmsPipeline* Lut = (cmsPipeline*) D;
    cmsUInt32Number i;

    for (i=0; i < Lut ->InputChannels; i++) {
         Out[i] = In[i];
    }
}


// If the target LUT holds only curves, the optimization procedure is to join all those
// curves together. That only works on curves and does not work on matrices.
static
cmsBool OptimizeByJoiningCurves(cmsPipeline** Lut, cmsUInt32Number Intent, cmsUInt32Number* InputFormat, cmsUInt32Number* OutputFormat, cmsUInt32Number* dwFlags)
{
    cmsToneCurve** GammaTables = NULL;
    cmsFloat32Number InFloat[cmsMAXCHANNELS], OutFloat[cmsMAXCHANNELS];
    cmsUInt32Number i, j;
    cmsPipeline* Src = *Lut;
    cmsPipeline* Dest = NULL;
    cmsStage* mpe;
    cmsStage* ObtainedCurves = NULL;


    // This is a loosy optimization! does not apply in floating-point cases
    if (_cmsFormatterIsFloat(*InputFormat) || _cmsFormatterIsFloat(*OutputFormat)) return FALSE;

    //  Only curves in this LUT?
    for (mpe = cmsPipelineGetPtrToFirstStage(Src);
         mpe != NULL;
         mpe = cmsStageNext(mpe)) {
            if (cmsStageType(mpe) != cmsSigCurveSetElemType) return FALSE;
    }

    // Allocate an empty LUT
    Dest =  cmsPipelineAlloc(Src ->ContextID, Src ->InputChannels, Src ->OutputChannels);
    if (Dest == NULL) return FALSE;

    // Create target curves
    GammaTables = (cmsToneCurve**) _cmsCalloc(Src ->ContextID, Src ->InputChannels, sizeof(cmsToneCurve*));
    if (GammaTables == NULL) goto Error;

    for (i=0; i < Src ->InputChannels; i++) {
        GammaTables[i] = cmsBuildTabulatedToneCurve16(Src ->ContextID, PRELINEARIZATION_POINTS, NULL);
        if (GammaTables[i] == NULL) goto Error;
    }

    // Compute 16 bit result by using floating point
    for (i=0; i < PRELINEARIZATION_POINTS; i++) {

        for (j=0; j < Src ->InputChannels; j++)
            InFloat[j] = (cmsFloat32Number) ((cmsFloat64Number) i / (PRELINEARIZATION_POINTS - 1));

        cmsPipelineEvalFloat(InFloat, OutFloat, Src);

        for (j=0; j < Src ->InputChannels; j++)
            GammaTables[j] -> Table16[i] = _cmsQuickSaturateWord(OutFloat[j] * 65535.0);
    }

    ObtainedCurves = cmsStageAllocToneCurves(Src ->ContextID, Src ->InputChannels, GammaTables);
    if (ObtainedCurves == NULL) goto Error;

    for (i=0; i < Src ->InputChannels; i++) {
        cmsFreeToneCurve(GammaTables[i]);
        GammaTables[i] = NULL;
    }

    if (GammaTables != NULL) {
        _cmsFree(Src->ContextID, GammaTables);
        GammaTables = NULL;
    }

    // Maybe the curves are linear at the end
    if (!AllCurvesAreLinear(ObtainedCurves)) {

        if (!cmsPipelineInsertStage(Dest, cmsAT_BEGIN, ObtainedCurves))
            goto Error;

        // If the curves are to be applied in 8 bits, we can save memory
        if (_cmsFormatterIs8bit(*InputFormat)) {

            _cmsStageToneCurvesData* Data = (_cmsStageToneCurvesData*) ObtainedCurves ->Data;
             Curves16Data* c16 = CurvesAlloc(Dest ->ContextID, Data ->nCurves, 256, Data ->TheCurves);

             if (c16 == NULL) goto Error; 
             *dwFlags |= cmsFLAGS_NOCACHE;
            _cmsPipelineSetOptimizationParameters(Dest, FastEvaluateCurves8, c16, CurvesFree, CurvesDup);

        }
        else {

            _cmsStageToneCurvesData* Data = (_cmsStageToneCurvesData*) cmsStageData(ObtainedCurves);
             Curves16Data* c16 = CurvesAlloc(Dest ->ContextID, Data ->nCurves, 65536, Data ->TheCurves);

             if (c16 == NULL) goto Error; 
             *dwFlags |= cmsFLAGS_NOCACHE;
            _cmsPipelineSetOptimizationParameters(Dest, FastEvaluateCurves16, c16, CurvesFree, CurvesDup);
        }
    }
    else {

        // LUT optimizes to nothing. Set the identity LUT
        cmsStageFree(ObtainedCurves);
        ObtainedCurves = NULL;

        if (!cmsPipelineInsertStage(Dest, cmsAT_BEGIN, cmsStageAllocIdentity(Dest ->ContextID, Src ->InputChannels)))
            goto Error;

        *dwFlags |= cmsFLAGS_NOCACHE;
        _cmsPipelineSetOptimizationParameters(Dest, FastIdentity16, (void*) Dest, NULL, NULL);
    }

    // We are done.
    cmsPipelineFree(Src);
    *Lut = Dest;
    return TRUE;

Error:

    if (ObtainedCurves != NULL) cmsStageFree(ObtainedCurves);
    if (GammaTables != NULL) {
        for (i=0; i < Src ->InputChannels; i++) {
            if (GammaTables[i] != NULL) cmsFreeToneCurve(GammaTables[i]);
        }

        _cmsFree(Src ->ContextID, GammaTables);
    }

    if (Dest != NULL) cmsPipelineFree(Dest);
    return FALSE;

    cmsUNUSED_PARAMETER(Intent);
    cmsUNUSED_PARAMETER(InputFormat);
    cmsUNUSED_PARAMETER(OutputFormat);
    cmsUNUSED_PARAMETER(dwFlags);
}

// -------------------------------------------------------------------------------------------------------------------------------------
// LUT is Shaper - Matrix - Matrix - Shaper, which is very frequent when combining two matrix-shaper profiles


static
void  FreeMatShaper(cmsContext ContextID, void* Data)
{
    if (Data != NULL) _cmsFree(ContextID, Data);
}

static
void* DupMatShaper(cmsContext ContextID, const void* Data)
{
    return _cmsDupMem(ContextID, Data, sizeof(MatShaper8Data));
}


// A fast matrix-shaper evaluator for 8 bits. This is a bit ticky since I'm using 1.14 signed fixed point
// to accomplish some performance. Actually it takes 256x3 16 bits tables and 16385 x 3 tables of 8 bits,
// in total about 50K, and the performance boost is huge!
static
void MatShaperEval16(register const cmsUInt16Number In[],
                     register cmsUInt16Number Out[],
                     register const void* D)
{
    MatShaper8Data* p = (MatShaper8Data*) D;
    cmsS1Fixed14Number l1, l2, l3, r, g, b;
    cmsUInt32Number ri, gi, bi;

    // In this case (and only in this case!) we can use this simplification since
    // In[] is assured to come from a 8 bit number. (a << 8 | a)
    ri = In[0] & 0xFFU;
    gi = In[1] & 0xFFU;
    bi = In[2] & 0xFFU;

    // Across first shaper, which also converts to 1.14 fixed point
    r = p->Shaper1R[ri];
    g = p->Shaper1G[gi];
    b = p->Shaper1B[bi];

    // Evaluate the matrix in 1.14 fixed point
    l1 =  (p->Mat[0][0] * r + p->Mat[0][1] * g + p->Mat[0][2] * b + p->Off[0] + 0x2000) >> 14;
    l2 =  (p->Mat[1][0] * r + p->Mat[1][1] * g + p->Mat[1][2] * b + p->Off[1] + 0x2000) >> 14;
    l3 =  (p->Mat[2][0] * r + p->Mat[2][1] * g + p->Mat[2][2] * b + p->Off[2] + 0x2000) >> 14;

    // Now we have to clip to 0..1.0 range
    ri = (l1 < 0) ? 0 : ((l1 > 16384) ? 16384U : (cmsUInt32Number) l1);
    gi = (l2 < 0) ? 0 : ((l2 > 16384) ? 16384U : (cmsUInt32Number) l2);
    bi = (l3 < 0) ? 0 : ((l3 > 16384) ? 16384U : (cmsUInt32Number) l3);

    // And across second shaper,
    Out[0] = p->Shaper2R[ri];
    Out[1] = p->Shaper2G[gi];
    Out[2] = p->Shaper2B[bi];

}

// This table converts from 8 bits to 1.14 after applying the curve
static
void FillFirstShaper(cmsS1Fixed14Number* Table, cmsToneCurve* Curve)
{
    int i;
    cmsFloat32Number R, y;

    for (i=0; i < 256; i++) {

        R   = (cmsFloat32Number) (i / 255.0);
        y   = cmsEvalToneCurveFloat(Curve, R);

        if (y < 131072.0)
            Table[i] = DOUBLE_TO_1FIXED14(y);
        else
            Table[i] = 0x7fffffff;
    }
}

// This table converts form 1.14 (being 0x4000 the last entry) to 8 bits after applying the curve
static
void FillSecondShaper(cmsUInt16Number* Table, cmsToneCurve* Curve, cmsBool Is8BitsOutput)
{
    int i;
    cmsFloat32Number R, Val;

    for (i=0; i < 16385; i++) {

        R   = (cmsFloat32Number) (i / 16384.0);
        Val = cmsEvalToneCurveFloat(Curve, R);    // Val comes 0..1.0

        if (Val < 0)
            Val = 0;

        if (Val > 1.0)
            Val = 1.0;

        if (Is8BitsOutput) {

            // If 8 bits output, we can optimize further by computing the / 257 part.
            // first we compute the resulting byte and then we store the byte times
            // 257. This quantization allows to round very quick by doing a >> 8, but
            // since the low byte is always equal to msb, we can do a & 0xff and this works!
            cmsUInt16Number w = _cmsQuickSaturateWord(Val * 65535.0);
            cmsUInt8Number  b = FROM_16_TO_8(w);

            Table[i] = FROM_8_TO_16(b);
        }
        else Table[i]  = _cmsQuickSaturateWord(Val * 65535.0);
    }
}

// Compute the matrix-shaper structure
static
cmsBool SetMatShaper(cmsPipeline* Dest, cmsToneCurve* Curve1[3], cmsMAT3* Mat, cmsVEC3* Off, cmsToneCurve* Curve2[3], cmsUInt32Number* OutputFormat)
{
    MatShaper8Data* p;
    int i, j;
    cmsBool Is8Bits = _cmsFormatterIs8bit(*OutputFormat);

    // Allocate a big chuck of memory to store precomputed tables
    p = (MatShaper8Data*) _cmsMalloc(Dest ->ContextID, sizeof(MatShaper8Data));
    if (p == NULL) return FALSE;

    p -> ContextID = Dest -> ContextID;

    // Precompute tables
    FillFirstShaper(p ->Shaper1R, Curve1[0]);
    FillFirstShaper(p ->Shaper1G, Curve1[1]);
    FillFirstShaper(p ->Shaper1B, Curve1[2]);

    FillSecondShaper(p ->Shaper2R, Curve2[0], Is8Bits);
    FillSecondShaper(p ->Shaper2G, Curve2[1], Is8Bits);
    FillSecondShaper(p ->Shaper2B, Curve2[2], Is8Bits);

    // Convert matrix to nFixed14. Note that those values may take more than 16 bits 
    for (i=0; i < 3; i++) {
        for (j=0; j < 3; j++) {
            p ->Mat[i][j] = DOUBLE_TO_1FIXED14(Mat->v[i].n[j]);
        }
    }

    for (i=0; i < 3; i++) {

        if (Off == NULL) {
            p ->Off[i] = 0;
        }
        else {
            p ->Off[i] = DOUBLE_TO_1FIXED14(Off->n[i]);
        }
    }

    // Mark as optimized for faster formatter
    if (Is8Bits)
        *OutputFormat |= OPTIMIZED_SH(1);

    // Fill function pointers
    _cmsPipelineSetOptimizationParameters(Dest, MatShaperEval16, (void*) p, FreeMatShaper, DupMatShaper);
    return TRUE;
}

//  8 bits on input allows matrix-shaper boot up to 25 Mpixels per second on RGB. That's fast!
static
cmsBool OptimizeMatrixShaper(cmsPipeline** Lut, cmsUInt32Number Intent, cmsUInt32Number* InputFormat, cmsUInt32Number* OutputFormat, cmsUInt32Number* dwFlags)
{
       cmsStage* Curve1, *Curve2;
       cmsStage* Matrix1, *Matrix2;
       cmsMAT3 res;
       cmsBool IdentityMat;
       cmsPipeline* Dest, *Src;
       cmsFloat64Number* Offset;

       // Only works on RGB to RGB
       if (T_CHANNELS(*InputFormat) != 3 || T_CHANNELS(*OutputFormat) != 3) return FALSE;

       // Only works on 8 bit input
       if (!_cmsFormatterIs8bit(*InputFormat)) return FALSE;

       // Seems suitable, proceed
       Src = *Lut;

       // Check for:
       // 
       //    shaper-matrix-matrix-shaper 
       //    shaper-matrix-shaper
       // 
       // Both of those constructs are possible (first because abs. colorimetric). 
       // additionally, In the first case, the input matrix offset should be zero.

       IdentityMat = FALSE;
       if (cmsPipelineCheckAndRetreiveStages(Src, 4,
              cmsSigCurveSetElemType, cmsSigMatrixElemType, cmsSigMatrixElemType, cmsSigCurveSetElemType,
              &Curve1, &Matrix1, &Matrix2, &Curve2)) {

              // Get both matrices
              _cmsStageMatrixData* Data1 = (_cmsStageMatrixData*)cmsStageData(Matrix1);
              _cmsStageMatrixData* Data2 = (_cmsStageMatrixData*)cmsStageData(Matrix2);

              // Input offset should be zero
              if (Data1->Offset != NULL) return FALSE;

              // Multiply both matrices to get the result
              _cmsMAT3per(&res, (cmsMAT3*)Data2->Double, (cmsMAT3*)Data1->Double);

              // Only 2nd matrix has offset, or it is zero 
              Offset = Data2->Offset;

              // Now the result is in res + Data2 -> Offset. Maybe is a plain identity?
              if (_cmsMAT3isIdentity(&res) && Offset == NULL) {

                     // We can get rid of full matrix
                     IdentityMat = TRUE;
              }

       }
       else {

              if (cmsPipelineCheckAndRetreiveStages(Src, 3,
                     cmsSigCurveSetElemType, cmsSigMatrixElemType, cmsSigCurveSetElemType,
                     &Curve1, &Matrix1, &Curve2)) {

                     _cmsStageMatrixData* Data = (_cmsStageMatrixData*)cmsStageData(Matrix1);

                     // Copy the matrix to our result
                     memcpy(&res, Data->Double, sizeof(res));

                     // Preserve the Odffset (may be NULL as a zero offset)
                     Offset = Data->Offset;

                     if (_cmsMAT3isIdentity(&res) && Offset == NULL) {

                            // We can get rid of full matrix
                            IdentityMat = TRUE;
                     }
              }
              else
                     return FALSE; // Not optimizeable this time

       }

      // Allocate an empty LUT
    Dest =  cmsPipelineAlloc(Src ->ContextID, Src ->InputChannels, Src ->OutputChannels);
    if (!Dest) return FALSE;

    // Assamble the new LUT
    if (!cmsPipelineInsertStage(Dest, cmsAT_BEGIN, cmsStageDup(Curve1)))
        goto Error;

    if (!IdentityMat) {

           if (!cmsPipelineInsertStage(Dest, cmsAT_END, cmsStageAllocMatrix(Dest->ContextID, 3, 3, (const cmsFloat64Number*)&res, Offset)))
                  goto Error;
    }

    if (!cmsPipelineInsertStage(Dest, cmsAT_END, cmsStageDup(Curve2)))
        goto Error;

    // If identity on matrix, we can further optimize the curves, so call the join curves routine
    if (IdentityMat) {

        OptimizeByJoiningCurves(&Dest, Intent, InputFormat, OutputFormat, dwFlags);
    }
    else {
        _cmsStageToneCurvesData* mpeC1 = (_cmsStageToneCurvesData*) cmsStageData(Curve1);
        _cmsStageToneCurvesData* mpeC2 = (_cmsStageToneCurvesData*) cmsStageData(Curve2);

        // In this particular optimization, caché does not help as it takes more time to deal with
        // the caché that with the pixel handling
        *dwFlags |= cmsFLAGS_NOCACHE;

        // Setup the optimizarion routines
        SetMatShaper(Dest, mpeC1 ->TheCurves, &res, (cmsVEC3*) Offset, mpeC2->TheCurves, OutputFormat);
    }

    cmsPipelineFree(Src);
    *Lut = Dest;
    return TRUE;
Error:
    // Leave Src unchanged
    cmsPipelineFree(Dest);
    return FALSE;
}


// -------------------------------------------------------------------------------------------------------------------------------------
// Optimization plug-ins

// List of optimizations
typedef struct _cmsOptimizationCollection_st {

    _cmsOPToptimizeFn  OptimizePtr;

    struct _cmsOptimizationCollection_st *Next;

} _cmsOptimizationCollection;


// The built-in list. We currently implement 4 types of optimizations. Joining of curves, matrix-shaper, linearization and resampling
static _cmsOptimizationCollection DefaultOptimization[] = {

    { OptimizeByJoiningCurves,            &DefaultOptimization[1] },
    { OptimizeMatrixShaper,               &DefaultOptimization[2] },
    { OptimizeByComputingLinearization,   &DefaultOptimization[3] },
    { OptimizeByResampling,               NULL }
};

// The linked list head
_cmsOptimizationPluginChunkType _cmsOptimizationPluginChunk = { NULL };


// Duplicates the zone of memory used by the plug-in in the new context
static
void DupPluginOptimizationList(struct _cmsContext_struct* ctx, 
                               const struct _cmsContext_struct* src)
{
   _cmsOptimizationPluginChunkType newHead = { NULL };
   _cmsOptimizationCollection*  entry;
   _cmsOptimizationCollection*  Anterior = NULL;
   _cmsOptimizationPluginChunkType* head = (_cmsOptimizationPluginChunkType*) src->chunks[OptimizationPlugin];

    _cmsAssert(ctx != NULL);
    _cmsAssert(head != NULL);

    // Walk the list copying all nodes
   for (entry = head->OptimizationCollection;
        entry != NULL;
        entry = entry ->Next) {

            _cmsOptimizationCollection *newEntry = ( _cmsOptimizationCollection *) _cmsSubAllocDup(ctx ->MemPool, entry, sizeof(_cmsOptimizationCollection));
   
            if (newEntry == NULL) 
                return;

            // We want to keep the linked list order, so this is a little bit tricky
            newEntry -> Next = NULL;
            if (Anterior)
                Anterior -> Next = newEntry;
     
            Anterior = newEntry;

            if (newHead.OptimizationCollection == NULL)
                newHead.OptimizationCollection = newEntry;
    }

  ctx ->chunks[OptimizationPlugin] = _cmsSubAllocDup(ctx->MemPool, &newHead, sizeof(_cmsOptimizationPluginChunkType));
}

void  _cmsAllocOptimizationPluginChunk(struct _cmsContext_struct* ctx, 
                                         const struct _cmsContext_struct* src)
{
  if (src != NULL) {

        // Copy all linked list
       DupPluginOptimizationList(ctx, src);
    }
    else {
        static _cmsOptimizationPluginChunkType OptimizationPluginChunkType = { NULL };
        ctx ->chunks[OptimizationPlugin] = _cmsSubAllocDup(ctx ->MemPool, &OptimizationPluginChunkType, sizeof(_cmsOptimizationPluginChunkType));
    }
}


// Register new ways to optimize
cmsBool  _cmsRegisterOptimizationPlugin(cmsContext ContextID, cmsPluginBase* Data)
{
    cmsPluginOptimization* Plugin = (cmsPluginOptimization*) Data;
    _cmsOptimizationPluginChunkType* ctx = ( _cmsOptimizationPluginChunkType*) _cmsContextGetClientChunk(ContextID, OptimizationPlugin);
    _cmsOptimizationCollection* fl;

    if (Data == NULL) {

        ctx->OptimizationCollection = NULL;
        return TRUE;
    }

    // Optimizer callback is required
    if (Plugin ->OptimizePtr == NULL) return FALSE;

    fl = (_cmsOptimizationCollection*) _cmsPluginMalloc(ContextID, sizeof(_cmsOptimizationCollection));
    if (fl == NULL) return FALSE;

    // Copy the parameters
    fl ->OptimizePtr = Plugin ->OptimizePtr;

    // Keep linked list
    fl ->Next = ctx->OptimizationCollection;

    // Set the head
    ctx ->OptimizationCollection = fl;

    // All is ok
    return TRUE;
}

// The entry point for LUT optimization
cmsBool _cmsOptimizePipeline(cmsContext ContextID,
                             cmsPipeline**    PtrLut,
                             cmsUInt32Number  Intent,
                             cmsUInt32Number* InputFormat,
                             cmsUInt32Number* OutputFormat,
                             cmsUInt32Number* dwFlags)
{
    _cmsOptimizationPluginChunkType* ctx = ( _cmsOptimizationPluginChunkType*) _cmsContextGetClientChunk(ContextID, OptimizationPlugin);
    _cmsOptimizationCollection* Opts;
    cmsBool AnySuccess = FALSE;

    // A CLUT is being asked, so force this specific optimization
    if (*dwFlags & cmsFLAGS_FORCE_CLUT) {

        PreOptimize(*PtrLut);
        return OptimizeByResampling(PtrLut, Intent, InputFormat, OutputFormat, dwFlags);
    }

    // Anything to optimize?
    if ((*PtrLut) ->Elements == NULL) {
        _cmsPipelineSetOptimizationParameters(*PtrLut, FastIdentity16, (void*) *PtrLut, NULL, NULL);
        return TRUE;
    }

    // Try to get rid of identities and trivial conversions.
    AnySuccess = PreOptimize(*PtrLut);

    // After removal do we end with an identity?
    if ((*PtrLut) ->Elements == NULL) {
        _cmsPipelineSetOptimizationParameters(*PtrLut, FastIdentity16, (void*) *PtrLut, NULL, NULL);
        return TRUE;
    }

    // Do not optimize, keep all precision
    if (*dwFlags & cmsFLAGS_NOOPTIMIZE)
        return FALSE;

    // Try plug-in optimizations 
    for (Opts = ctx->OptimizationCollection;
         Opts != NULL;
         Opts = Opts ->Next) {

            // If one schema succeeded, we are done
            if (Opts ->OptimizePtr(PtrLut, Intent, InputFormat, OutputFormat, dwFlags)) {

                return TRUE;    // Optimized!
            }
    }

   // Try built-in optimizations 
    for (Opts = DefaultOptimization;
         Opts != NULL;
         Opts = Opts ->Next) {

            if (Opts ->OptimizePtr(PtrLut, Intent, InputFormat, OutputFormat, dwFlags)) {

                return TRUE;  
            }
    }

    // Only simple optimizations succeeded
    return AnySuccess;
}



