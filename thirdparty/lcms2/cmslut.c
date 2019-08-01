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


// Allocates an empty multi profile element
cmsStage* CMSEXPORT _cmsStageAllocPlaceholder(cmsContext ContextID,
                                cmsStageSignature Type,
                                cmsUInt32Number InputChannels,
                                cmsUInt32Number OutputChannels,
                                _cmsStageEvalFn     EvalPtr,
                                _cmsStageDupElemFn  DupElemPtr,
                                _cmsStageFreeElemFn FreePtr,
                                void*             Data)
{
    cmsStage* ph = (cmsStage*) _cmsMallocZero(ContextID, sizeof(cmsStage));

    if (ph == NULL) return NULL;


    ph ->ContextID = ContextID;

    ph ->Type       = Type;
    ph ->Implements = Type;   // By default, no clue on what is implementing

    ph ->InputChannels  = InputChannels;
    ph ->OutputChannels = OutputChannels;
    ph ->EvalPtr        = EvalPtr;
    ph ->DupElemPtr     = DupElemPtr;
    ph ->FreePtr        = FreePtr;
    ph ->Data           = Data;

    return ph;
}


static
void EvaluateIdentity(const cmsFloat32Number In[],
                            cmsFloat32Number Out[],
                      const cmsStage *mpe)
{
    memmove(Out, In, mpe ->InputChannels * sizeof(cmsFloat32Number));
}


cmsStage* CMSEXPORT cmsStageAllocIdentity(cmsContext ContextID, cmsUInt32Number nChannels)
{
    return _cmsStageAllocPlaceholder(ContextID,
                                   cmsSigIdentityElemType,
                                   nChannels, nChannels,
                                   EvaluateIdentity,
                                   NULL,
                                   NULL,
                                   NULL);
 }

// Conversion functions. From floating point to 16 bits
static
void FromFloatTo16(const cmsFloat32Number In[], cmsUInt16Number Out[], cmsUInt32Number n)
{
    cmsUInt32Number i;

    for (i=0; i < n; i++) {
        Out[i] = _cmsQuickSaturateWord(In[i] * 65535.0);
    }
}

// From 16 bits to floating point
static
void From16ToFloat(const cmsUInt16Number In[], cmsFloat32Number Out[], cmsUInt32Number n)
{
    cmsUInt32Number i;

    for (i=0; i < n; i++) {
        Out[i] = (cmsFloat32Number) In[i] / 65535.0F;
    }
}


// This function is quite useful to analyze the structure of a LUT and retrieve the MPE elements
// that conform the LUT. It should be called with the LUT, the number of expected elements and
// then a list of expected types followed with a list of cmsFloat64Number pointers to MPE elements. If
// the function founds a match with current pipeline, it fills the pointers and returns TRUE
// if not, returns FALSE without touching anything. Setting pointers to NULL does bypass
// the storage process.
cmsBool  CMSEXPORT cmsPipelineCheckAndRetreiveStages(const cmsPipeline* Lut, cmsUInt32Number n, ...)
{
    va_list args;
    cmsUInt32Number i;
    cmsStage* mpe;
    cmsStageSignature Type;
    void** ElemPtr;

    // Make sure same number of elements
    if (cmsPipelineStageCount(Lut) != n) return FALSE;

    va_start(args, n);

    // Iterate across asked types
    mpe = Lut ->Elements;
    for (i=0; i < n; i++) {

        // Get asked type. cmsStageSignature is promoted to int by compiler
        Type  = (cmsStageSignature)va_arg(args, int);
        if (mpe ->Type != Type) {

            va_end(args);       // Mismatch. We are done.
            return FALSE;
        }
        mpe = mpe ->Next;
    }

    // Found a combination, fill pointers if not NULL
    mpe = Lut ->Elements;
    for (i=0; i < n; i++) {

        ElemPtr = va_arg(args, void**);
        if (ElemPtr != NULL)
            *ElemPtr = mpe;

        mpe = mpe ->Next;
    }

    va_end(args);
    return TRUE;
}

// Below there are implementations for several types of elements. Each type may be implemented by a
// evaluation function, a duplication function, a function to free resources and a constructor.

// *************************************************************************************************
// Type cmsSigCurveSetElemType (curves)
// *************************************************************************************************

cmsToneCurve** _cmsStageGetPtrToCurveSet(const cmsStage* mpe)
{
    _cmsStageToneCurvesData* Data = (_cmsStageToneCurvesData*) mpe ->Data;

    return Data ->TheCurves;
}

static
void EvaluateCurves(const cmsFloat32Number In[],
                    cmsFloat32Number Out[],
                    const cmsStage *mpe)
{
    _cmsStageToneCurvesData* Data;
    cmsUInt32Number i;

    _cmsAssert(mpe != NULL);

    Data = (_cmsStageToneCurvesData*) mpe ->Data;
    if (Data == NULL) return;

    if (Data ->TheCurves == NULL) return;

    for (i=0; i < Data ->nCurves; i++) {
        Out[i] = cmsEvalToneCurveFloat(Data ->TheCurves[i], In[i]);
    }
}

static
void CurveSetElemTypeFree(cmsStage* mpe)
{
    _cmsStageToneCurvesData* Data;
    cmsUInt32Number i;

    _cmsAssert(mpe != NULL);

    Data = (_cmsStageToneCurvesData*) mpe ->Data;
    if (Data == NULL) return;

    if (Data ->TheCurves != NULL) {
        for (i=0; i < Data ->nCurves; i++) {
            if (Data ->TheCurves[i] != NULL)
                cmsFreeToneCurve(Data ->TheCurves[i]);
        }
    }
    _cmsFree(mpe ->ContextID, Data ->TheCurves);
    _cmsFree(mpe ->ContextID, Data);
}


static
void* CurveSetDup(cmsStage* mpe)
{
    _cmsStageToneCurvesData* Data = (_cmsStageToneCurvesData*) mpe ->Data;
    _cmsStageToneCurvesData* NewElem;
    cmsUInt32Number i;

    NewElem = (_cmsStageToneCurvesData*) _cmsMallocZero(mpe ->ContextID, sizeof(_cmsStageToneCurvesData));
    if (NewElem == NULL) return NULL;

    NewElem ->nCurves   = Data ->nCurves;
    NewElem ->TheCurves = (cmsToneCurve**) _cmsCalloc(mpe ->ContextID, NewElem ->nCurves, sizeof(cmsToneCurve*));

    if (NewElem ->TheCurves == NULL) goto Error;

    for (i=0; i < NewElem ->nCurves; i++) {

        // Duplicate each curve. It may fail.
        NewElem ->TheCurves[i] = cmsDupToneCurve(Data ->TheCurves[i]);
        if (NewElem ->TheCurves[i] == NULL) goto Error;


    }
    return (void*) NewElem;

Error:

    if (NewElem ->TheCurves != NULL) {
        for (i=0; i < NewElem ->nCurves; i++) {
            if (NewElem ->TheCurves[i])
                cmsFreeToneCurve(NewElem ->TheCurves[i]);
        }
    }
    _cmsFree(mpe ->ContextID, NewElem ->TheCurves);
    _cmsFree(mpe ->ContextID, NewElem);
    return NULL;
}


// Curves == NULL forces identity curves
cmsStage* CMSEXPORT cmsStageAllocToneCurves(cmsContext ContextID, cmsUInt32Number nChannels, cmsToneCurve* const Curves[])
{
    cmsUInt32Number i;
    _cmsStageToneCurvesData* NewElem;
    cmsStage* NewMPE;


    NewMPE = _cmsStageAllocPlaceholder(ContextID, cmsSigCurveSetElemType, nChannels, nChannels,
                                     EvaluateCurves, CurveSetDup, CurveSetElemTypeFree, NULL );
    if (NewMPE == NULL) return NULL;

    NewElem = (_cmsStageToneCurvesData*) _cmsMallocZero(ContextID, sizeof(_cmsStageToneCurvesData));
    if (NewElem == NULL) {
        cmsStageFree(NewMPE);
        return NULL;
    }

    NewMPE ->Data  = (void*) NewElem;

    NewElem ->nCurves   = nChannels;
    NewElem ->TheCurves = (cmsToneCurve**) _cmsCalloc(ContextID, nChannels, sizeof(cmsToneCurve*));
    if (NewElem ->TheCurves == NULL) {
        cmsStageFree(NewMPE);
        return NULL;
    }

    for (i=0; i < nChannels; i++) {

        if (Curves == NULL) {
            NewElem ->TheCurves[i] = cmsBuildGamma(ContextID, 1.0);
        }
        else {
            NewElem ->TheCurves[i] = cmsDupToneCurve(Curves[i]);
        }

        if (NewElem ->TheCurves[i] == NULL) {
            cmsStageFree(NewMPE);
            return NULL;
        }

    }

   return NewMPE;
}


// Create a bunch of identity curves
cmsStage* CMSEXPORT _cmsStageAllocIdentityCurves(cmsContext ContextID, cmsUInt32Number nChannels)
{
    cmsStage* mpe = cmsStageAllocToneCurves(ContextID, nChannels, NULL);

    if (mpe == NULL) return NULL;
    mpe ->Implements = cmsSigIdentityElemType;
    return mpe;
}


// *************************************************************************************************
// Type cmsSigMatrixElemType (Matrices)
// *************************************************************************************************


// Special care should be taken here because precision loss. A temporary cmsFloat64Number buffer is being used
static
void EvaluateMatrix(const cmsFloat32Number In[],
                    cmsFloat32Number Out[],
                    const cmsStage *mpe)
{
    cmsUInt32Number i, j;
    _cmsStageMatrixData* Data = (_cmsStageMatrixData*) mpe ->Data;
    cmsFloat64Number Tmp;

    // Input is already in 0..1.0 notation
    for (i=0; i < mpe ->OutputChannels; i++) {

        Tmp = 0;
        for (j=0; j < mpe->InputChannels; j++) {
            Tmp += In[j] * Data->Double[i*mpe->InputChannels + j];
        }

        if (Data ->Offset != NULL)
            Tmp += Data->Offset[i];

        Out[i] = (cmsFloat32Number) Tmp;
    }


    // Output in 0..1.0 domain
}


// Duplicate a yet-existing matrix element
static
void* MatrixElemDup(cmsStage* mpe)
{
    _cmsStageMatrixData* Data = (_cmsStageMatrixData*) mpe ->Data;
    _cmsStageMatrixData* NewElem;
    cmsUInt32Number sz;

    NewElem = (_cmsStageMatrixData*) _cmsMallocZero(mpe ->ContextID, sizeof(_cmsStageMatrixData));
    if (NewElem == NULL) return NULL;

    sz = mpe ->InputChannels * mpe ->OutputChannels;

    NewElem ->Double = (cmsFloat64Number*) _cmsDupMem(mpe ->ContextID, Data ->Double, sz * sizeof(cmsFloat64Number)) ;

    if (Data ->Offset)
        NewElem ->Offset = (cmsFloat64Number*) _cmsDupMem(mpe ->ContextID,
                                                Data ->Offset, mpe -> OutputChannels * sizeof(cmsFloat64Number)) ;

    return (void*) NewElem;
}


static
void MatrixElemTypeFree(cmsStage* mpe)
{
    _cmsStageMatrixData* Data = (_cmsStageMatrixData*) mpe ->Data;
    if (Data == NULL)
        return;
    if (Data ->Double)
        _cmsFree(mpe ->ContextID, Data ->Double);

    if (Data ->Offset)
        _cmsFree(mpe ->ContextID, Data ->Offset);

    _cmsFree(mpe ->ContextID, mpe ->Data);
}



cmsStage*  CMSEXPORT cmsStageAllocMatrix(cmsContext ContextID, cmsUInt32Number Rows, cmsUInt32Number Cols,
                                     const cmsFloat64Number* Matrix, const cmsFloat64Number* Offset)
{
    cmsUInt32Number i, n;
    _cmsStageMatrixData* NewElem;
    cmsStage* NewMPE;

    n = Rows * Cols;

    // Check for overflow
    if (n == 0) return NULL;
    if (n >= UINT_MAX / Cols) return NULL;
    if (n >= UINT_MAX / Rows) return NULL;
    if (n < Rows || n < Cols) return NULL;

    NewMPE = _cmsStageAllocPlaceholder(ContextID, cmsSigMatrixElemType, Cols, Rows,
                                     EvaluateMatrix, MatrixElemDup, MatrixElemTypeFree, NULL );
    if (NewMPE == NULL) return NULL;


    NewElem = (_cmsStageMatrixData*) _cmsMallocZero(ContextID, sizeof(_cmsStageMatrixData));
    if (NewElem == NULL) return NULL;


    NewElem ->Double = (cmsFloat64Number*) _cmsCalloc(ContextID, n, sizeof(cmsFloat64Number));

    if (NewElem->Double == NULL) {
        MatrixElemTypeFree(NewMPE);
        return NULL;
    }

    for (i=0; i < n; i++) {
        NewElem ->Double[i] = Matrix[i];
    }


    if (Offset != NULL) {

        NewElem ->Offset = (cmsFloat64Number*) _cmsCalloc(ContextID, Rows, sizeof(cmsFloat64Number));
        if (NewElem->Offset == NULL) {
           MatrixElemTypeFree(NewMPE);
           return NULL;
        }

        for (i=0; i < Rows; i++) {
                NewElem ->Offset[i] = Offset[i];
        }

    }

    NewMPE ->Data  = (void*) NewElem;
    return NewMPE;
}


// *************************************************************************************************
// Type cmsSigCLutElemType
// *************************************************************************************************


// Evaluate in true floating point
static
void EvaluateCLUTfloat(const cmsFloat32Number In[], cmsFloat32Number Out[], const cmsStage *mpe)
{
    _cmsStageCLutData* Data = (_cmsStageCLutData*) mpe ->Data;

    Data -> Params ->Interpolation.LerpFloat(In, Out, Data->Params);
}


// Convert to 16 bits, evaluate, and back to floating point
static
void EvaluateCLUTfloatIn16(const cmsFloat32Number In[], cmsFloat32Number Out[], const cmsStage *mpe)
{
    _cmsStageCLutData* Data = (_cmsStageCLutData*) mpe ->Data;
    cmsUInt16Number In16[MAX_STAGE_CHANNELS], Out16[MAX_STAGE_CHANNELS];

    _cmsAssert(mpe ->InputChannels  <= MAX_STAGE_CHANNELS);
    _cmsAssert(mpe ->OutputChannels <= MAX_STAGE_CHANNELS);

    FromFloatTo16(In, In16, mpe ->InputChannels);
    Data -> Params ->Interpolation.Lerp16(In16, Out16, Data->Params);
    From16ToFloat(Out16, Out,  mpe ->OutputChannels);
}


// Given an hypercube of b dimensions, with Dims[] number of nodes by dimension, calculate the total amount of nodes
static
cmsUInt32Number CubeSize(const cmsUInt32Number Dims[], cmsUInt32Number b)
{
    cmsUInt32Number rv, dim;

    _cmsAssert(Dims != NULL);

    for (rv = 1; b > 0; b--) {

        dim = Dims[b-1];
        if (dim == 0) return 0;  // Error

        rv *= dim;

        // Check for overflow
        if (rv > UINT_MAX / dim) return 0;
    }

    return rv;
}

static
void* CLUTElemDup(cmsStage* mpe)
{
    _cmsStageCLutData* Data = (_cmsStageCLutData*) mpe ->Data;
    _cmsStageCLutData* NewElem;


    NewElem = (_cmsStageCLutData*) _cmsMallocZero(mpe ->ContextID, sizeof(_cmsStageCLutData));
    if (NewElem == NULL) return NULL;

    NewElem ->nEntries       = Data ->nEntries;
    NewElem ->HasFloatValues = Data ->HasFloatValues;

    if (Data ->Tab.T) {

        if (Data ->HasFloatValues) {
            NewElem ->Tab.TFloat = (cmsFloat32Number*) _cmsDupMem(mpe ->ContextID, Data ->Tab.TFloat, Data ->nEntries * sizeof (cmsFloat32Number));
            if (NewElem ->Tab.TFloat == NULL)
                goto Error;
        } else {
            NewElem ->Tab.T = (cmsUInt16Number*) _cmsDupMem(mpe ->ContextID, Data ->Tab.T, Data ->nEntries * sizeof (cmsUInt16Number));
            if (NewElem ->Tab.T == NULL)
                goto Error;
        }
    }

    NewElem ->Params   = _cmsComputeInterpParamsEx(mpe ->ContextID,
                                                   Data ->Params ->nSamples,
                                                   Data ->Params ->nInputs,
                                                   Data ->Params ->nOutputs,
                                                   NewElem ->Tab.T,
                                                   Data ->Params ->dwFlags);
    if (NewElem->Params != NULL)
        return (void*) NewElem;
 Error:
    if (NewElem->Tab.T)
        // This works for both types
        _cmsFree(mpe ->ContextID, NewElem -> Tab.T);
    _cmsFree(mpe ->ContextID, NewElem);
    return NULL;
}


static
void CLutElemTypeFree(cmsStage* mpe)
{

    _cmsStageCLutData* Data = (_cmsStageCLutData*) mpe ->Data;

    // Already empty
    if (Data == NULL) return;

    // This works for both types
    if (Data -> Tab.T)
        _cmsFree(mpe ->ContextID, Data -> Tab.T);

    _cmsFreeInterpParams(Data ->Params);
    _cmsFree(mpe ->ContextID, mpe ->Data);
}


// Allocates a 16-bit multidimensional CLUT. This is evaluated at 16-bit precision. Table may have different
// granularity on each dimension.
cmsStage* CMSEXPORT cmsStageAllocCLut16bitGranular(cmsContext ContextID,
                                         const cmsUInt32Number clutPoints[],
                                         cmsUInt32Number inputChan,
                                         cmsUInt32Number outputChan,
                                         const cmsUInt16Number* Table)
{
    cmsUInt32Number i, n;
    _cmsStageCLutData* NewElem;
    cmsStage* NewMPE;

    _cmsAssert(clutPoints != NULL);

    if (inputChan > MAX_INPUT_DIMENSIONS) {
        cmsSignalError(ContextID, cmsERROR_RANGE, "Too many input channels (%d channels, max=%d)", inputChan, MAX_INPUT_DIMENSIONS);
        return NULL;
    }

    NewMPE = _cmsStageAllocPlaceholder(ContextID, cmsSigCLutElemType, inputChan, outputChan,
                                     EvaluateCLUTfloatIn16, CLUTElemDup, CLutElemTypeFree, NULL );

    if (NewMPE == NULL) return NULL;

    NewElem = (_cmsStageCLutData*) _cmsMallocZero(ContextID, sizeof(_cmsStageCLutData));
    if (NewElem == NULL) {
        cmsStageFree(NewMPE);
        return NULL;
    }

    NewMPE ->Data  = (void*) NewElem;

    NewElem -> nEntries = n = outputChan * CubeSize(clutPoints, inputChan);
    NewElem -> HasFloatValues = FALSE;

    if (n == 0) {
        cmsStageFree(NewMPE);
        return NULL;
    }


    NewElem ->Tab.T  = (cmsUInt16Number*) _cmsCalloc(ContextID, n, sizeof(cmsUInt16Number));
    if (NewElem ->Tab.T == NULL) {
        cmsStageFree(NewMPE);
        return NULL;
    }

    if (Table != NULL) {
        for (i=0; i < n; i++) {
            NewElem ->Tab.T[i] = Table[i];
        }
    }

    NewElem ->Params = _cmsComputeInterpParamsEx(ContextID, clutPoints, inputChan, outputChan, NewElem ->Tab.T, CMS_LERP_FLAGS_16BITS);
    if (NewElem ->Params == NULL) {
        cmsStageFree(NewMPE);
        return NULL;
    }

    return NewMPE;
}

cmsStage* CMSEXPORT cmsStageAllocCLut16bit(cmsContext ContextID,
                                    cmsUInt32Number nGridPoints,
                                    cmsUInt32Number inputChan,
                                    cmsUInt32Number outputChan,
                                    const cmsUInt16Number* Table)
{
    cmsUInt32Number Dimensions[MAX_INPUT_DIMENSIONS];
    int i;

   // Our resulting LUT would be same gridpoints on all dimensions
    for (i=0; i < MAX_INPUT_DIMENSIONS; i++)
        Dimensions[i] = nGridPoints;

    return cmsStageAllocCLut16bitGranular(ContextID, Dimensions, inputChan, outputChan, Table);
}


cmsStage* CMSEXPORT cmsStageAllocCLutFloat(cmsContext ContextID,
                                       cmsUInt32Number nGridPoints,
                                       cmsUInt32Number inputChan,
                                       cmsUInt32Number outputChan,
                                       const cmsFloat32Number* Table)
{
   cmsUInt32Number Dimensions[MAX_INPUT_DIMENSIONS];
   int i;

    // Our resulting LUT would be same gridpoints on all dimensions
    for (i=0; i < MAX_INPUT_DIMENSIONS; i++)
        Dimensions[i] = nGridPoints;

    return cmsStageAllocCLutFloatGranular(ContextID, Dimensions, inputChan, outputChan, Table);
}



cmsStage* CMSEXPORT cmsStageAllocCLutFloatGranular(cmsContext ContextID, const cmsUInt32Number clutPoints[], cmsUInt32Number inputChan, cmsUInt32Number outputChan, const cmsFloat32Number* Table)
{
    cmsUInt32Number i, n;
    _cmsStageCLutData* NewElem;
    cmsStage* NewMPE;

    _cmsAssert(clutPoints != NULL);

    if (inputChan > MAX_INPUT_DIMENSIONS) {
        cmsSignalError(ContextID, cmsERROR_RANGE, "Too many input channels (%d channels, max=%d)", inputChan, MAX_INPUT_DIMENSIONS);
        return NULL;
    }

    NewMPE = _cmsStageAllocPlaceholder(ContextID, cmsSigCLutElemType, inputChan, outputChan,
                                             EvaluateCLUTfloat, CLUTElemDup, CLutElemTypeFree, NULL);
    if (NewMPE == NULL) return NULL;


    NewElem = (_cmsStageCLutData*) _cmsMallocZero(ContextID, sizeof(_cmsStageCLutData));
    if (NewElem == NULL) {
        cmsStageFree(NewMPE);
        return NULL;
    }

    NewMPE ->Data  = (void*) NewElem;

    // There is a potential integer overflow on conputing n and nEntries.
    NewElem -> nEntries = n = outputChan * CubeSize(clutPoints, inputChan);
    NewElem -> HasFloatValues = TRUE;

    if (n == 0) {
        cmsStageFree(NewMPE);
        return NULL;
    }

    NewElem ->Tab.TFloat  = (cmsFloat32Number*) _cmsCalloc(ContextID, n, sizeof(cmsFloat32Number));
    if (NewElem ->Tab.TFloat == NULL) {
        cmsStageFree(NewMPE);
        return NULL;
    }

    if (Table != NULL) {
        for (i=0; i < n; i++) {
            NewElem ->Tab.TFloat[i] = Table[i];
        }
    }

    NewElem ->Params = _cmsComputeInterpParamsEx(ContextID, clutPoints,  inputChan, outputChan, NewElem ->Tab.TFloat, CMS_LERP_FLAGS_FLOAT);
    if (NewElem ->Params == NULL) {
        cmsStageFree(NewMPE);
        return NULL;
    }

    return NewMPE;
}


static
int IdentitySampler(register const cmsUInt16Number In[], register cmsUInt16Number Out[], register void * Cargo)
{
    int nChan = *(int*) Cargo;
    int i;

    for (i=0; i < nChan; i++)
        Out[i] = In[i];

    return 1;
}

// Creates an MPE that just copies input to output
cmsStage* CMSEXPORT _cmsStageAllocIdentityCLut(cmsContext ContextID, cmsUInt32Number nChan)
{
    cmsUInt32Number Dimensions[MAX_INPUT_DIMENSIONS];
    cmsStage* mpe ;
    int i;

    for (i=0; i < MAX_INPUT_DIMENSIONS; i++)
        Dimensions[i] = 2;

    mpe = cmsStageAllocCLut16bitGranular(ContextID, Dimensions, nChan, nChan, NULL);
    if (mpe == NULL) return NULL;

    if (!cmsStageSampleCLut16bit(mpe, IdentitySampler, &nChan, 0)) {
        cmsStageFree(mpe);
        return NULL;
    }

    mpe ->Implements = cmsSigIdentityElemType;
    return mpe;
}



// Quantize a value 0 <= i < MaxSamples to 0..0xffff
cmsUInt16Number CMSEXPORT _cmsQuantizeVal(cmsFloat64Number i, cmsUInt32Number MaxSamples)
{
    cmsFloat64Number x;

    x = ((cmsFloat64Number) i * 65535.) / (cmsFloat64Number) (MaxSamples - 1);
    return _cmsQuickSaturateWord(x);
}


// This routine does a sweep on whole input space, and calls its callback
// function on knots. returns TRUE if all ok, FALSE otherwise.
cmsBool CMSEXPORT cmsStageSampleCLut16bit(cmsStage* mpe, cmsSAMPLER16 Sampler, void * Cargo, cmsUInt32Number dwFlags)
{
    int i, t, index, rest;
    cmsUInt32Number nTotalPoints;
    cmsUInt32Number nInputs, nOutputs;
    cmsUInt32Number* nSamples;
    cmsUInt16Number In[MAX_INPUT_DIMENSIONS+1], Out[MAX_STAGE_CHANNELS];
    _cmsStageCLutData* clut;

    if (mpe == NULL) return FALSE;

    clut = (_cmsStageCLutData*) mpe->Data;

    if (clut == NULL) return FALSE;

    nSamples = clut->Params ->nSamples;
    nInputs  = clut->Params ->nInputs;
    nOutputs = clut->Params ->nOutputs;

    if (nInputs <= 0) return FALSE;
    if (nOutputs <= 0) return FALSE;
    if (nInputs > MAX_INPUT_DIMENSIONS) return FALSE;
    if (nOutputs >= MAX_STAGE_CHANNELS) return FALSE;

    memset(In, 0, sizeof(In));
    memset(Out, 0, sizeof(Out));

    nTotalPoints = CubeSize(nSamples, nInputs);
    if (nTotalPoints == 0) return FALSE;

    index = 0;
    for (i = 0; i < (int) nTotalPoints; i++) {

        rest = i;
        for (t = (int)nInputs - 1; t >= 0; --t) {

            cmsUInt32Number  Colorant = rest % nSamples[t];

            rest /= nSamples[t];

            In[t] = _cmsQuantizeVal(Colorant, nSamples[t]);
        }

        if (clut ->Tab.T != NULL) {
            for (t = 0; t < (int)nOutputs; t++)
                Out[t] = clut->Tab.T[index + t];
        }

        if (!Sampler(In, Out, Cargo))
            return FALSE;

        if (!(dwFlags & SAMPLER_INSPECT)) {

            if (clut ->Tab.T != NULL) {
                for (t=0; t < (int) nOutputs; t++)
                    clut->Tab.T[index + t] = Out[t];
            }
        }

        index += nOutputs;
    }

    return TRUE;
}

// Same as anterior, but for floating point
cmsBool CMSEXPORT cmsStageSampleCLutFloat(cmsStage* mpe, cmsSAMPLERFLOAT Sampler, void * Cargo, cmsUInt32Number dwFlags)
{
    int i, t, index, rest;
    cmsUInt32Number nTotalPoints;
    cmsUInt32Number nInputs, nOutputs;
    cmsUInt32Number* nSamples;
    cmsFloat32Number In[MAX_INPUT_DIMENSIONS+1], Out[MAX_STAGE_CHANNELS];
    _cmsStageCLutData* clut = (_cmsStageCLutData*) mpe->Data;

    nSamples = clut->Params ->nSamples;
    nInputs  = clut->Params ->nInputs;
    nOutputs = clut->Params ->nOutputs;

    if (nInputs <= 0) return FALSE;
    if (nOutputs <= 0) return FALSE;
    if (nInputs  > MAX_INPUT_DIMENSIONS) return FALSE;
    if (nOutputs >= MAX_STAGE_CHANNELS) return FALSE;

    nTotalPoints = CubeSize(nSamples, nInputs);
    if (nTotalPoints == 0) return FALSE;

    index = 0;
    for (i = 0; i < (int)nTotalPoints; i++) {

        rest = i;
        for (t = (int) nInputs-1; t >=0; --t) {

            cmsUInt32Number  Colorant = rest % nSamples[t];

            rest /= nSamples[t];

            In[t] =  (cmsFloat32Number) (_cmsQuantizeVal(Colorant, nSamples[t]) / 65535.0);
        }

        if (clut ->Tab.TFloat != NULL) {
            for (t=0; t < (int) nOutputs; t++)
                Out[t] = clut->Tab.TFloat[index + t];
        }

        if (!Sampler(In, Out, Cargo))
            return FALSE;

        if (!(dwFlags & SAMPLER_INSPECT)) {

            if (clut ->Tab.TFloat != NULL) {
                for (t=0; t < (int) nOutputs; t++)
                    clut->Tab.TFloat[index + t] = Out[t];
            }
        }

        index += nOutputs;
    }

    return TRUE;
}



// This routine does a sweep on whole input space, and calls its callback
// function on knots. returns TRUE if all ok, FALSE otherwise.
cmsBool CMSEXPORT cmsSliceSpace16(cmsUInt32Number nInputs, const cmsUInt32Number clutPoints[],
                                         cmsSAMPLER16 Sampler, void * Cargo)
{
    int i, t, rest;
    cmsUInt32Number nTotalPoints;
    cmsUInt16Number In[cmsMAXCHANNELS];

    if (nInputs >= cmsMAXCHANNELS) return FALSE;

    nTotalPoints = CubeSize(clutPoints, nInputs);
    if (nTotalPoints == 0) return FALSE;

    for (i = 0; i < (int) nTotalPoints; i++) {

        rest = i;
        for (t = (int) nInputs-1; t >=0; --t) {

            cmsUInt32Number  Colorant = rest % clutPoints[t];

            rest /= clutPoints[t];
            In[t] = _cmsQuantizeVal(Colorant, clutPoints[t]);

        }

        if (!Sampler(In, NULL, Cargo))
            return FALSE;
    }

    return TRUE;
}

cmsInt32Number CMSEXPORT cmsSliceSpaceFloat(cmsUInt32Number nInputs, const cmsUInt32Number clutPoints[],
                                            cmsSAMPLERFLOAT Sampler, void * Cargo)
{
    int i, t, rest;
    cmsUInt32Number nTotalPoints;
    cmsFloat32Number In[cmsMAXCHANNELS];

    if (nInputs >= cmsMAXCHANNELS) return FALSE;

    nTotalPoints = CubeSize(clutPoints, nInputs);
    if (nTotalPoints == 0) return FALSE;

    for (i = 0; i < (int) nTotalPoints; i++) {

        rest = i;
        for (t = (int) nInputs-1; t >=0; --t) {

            cmsUInt32Number  Colorant = rest % clutPoints[t];

            rest /= clutPoints[t];
            In[t] =  (cmsFloat32Number) (_cmsQuantizeVal(Colorant, clutPoints[t]) / 65535.0);

        }

        if (!Sampler(In, NULL, Cargo))
            return FALSE;
    }

    return TRUE;
}

// ********************************************************************************
// Type cmsSigLab2XYZElemType
// ********************************************************************************


static
void EvaluateLab2XYZ(const cmsFloat32Number In[],
                     cmsFloat32Number Out[],
                     const cmsStage *mpe)
{
    cmsCIELab Lab;
    cmsCIEXYZ XYZ;
    const cmsFloat64Number XYZadj = MAX_ENCODEABLE_XYZ;

    // V4 rules
    Lab.L = In[0] * 100.0;
    Lab.a = In[1] * 255.0 - 128.0;
    Lab.b = In[2] * 255.0 - 128.0;

    cmsLab2XYZ(NULL, &XYZ, &Lab);

    // From XYZ, range 0..19997 to 0..1.0, note that 1.99997 comes from 0xffff
    // encoded as 1.15 fixed point, so 1 + (32767.0 / 32768.0)

    Out[0] = (cmsFloat32Number) ((cmsFloat64Number) XYZ.X / XYZadj);
    Out[1] = (cmsFloat32Number) ((cmsFloat64Number) XYZ.Y / XYZadj);
    Out[2] = (cmsFloat32Number) ((cmsFloat64Number) XYZ.Z / XYZadj);
    return;

    cmsUNUSED_PARAMETER(mpe);
}


// No dup or free routines needed, as the structure has no pointers in it.
cmsStage* CMSEXPORT _cmsStageAllocLab2XYZ(cmsContext ContextID)
{
    return _cmsStageAllocPlaceholder(ContextID, cmsSigLab2XYZElemType, 3, 3, EvaluateLab2XYZ, NULL, NULL, NULL);
}

// ********************************************************************************

// v2 L=100 is supposed to be placed on 0xFF00. There is no reasonable
// number of gridpoints that would make exact match. However, a prelinearization
// of 258 entries, would map 0xFF00 exactly on entry 257, and this is good to avoid scum dot.
// Almost all what we need but unfortunately, the rest of entries should be scaled by
// (255*257/256) and this is not exact.

cmsStage* _cmsStageAllocLabV2ToV4curves(cmsContext ContextID)
{
    cmsStage* mpe;
    cmsToneCurve* LabTable[3];
    int i, j;

    LabTable[0] = cmsBuildTabulatedToneCurve16(ContextID, 258, NULL);
    LabTable[1] = cmsBuildTabulatedToneCurve16(ContextID, 258, NULL);
    LabTable[2] = cmsBuildTabulatedToneCurve16(ContextID, 258, NULL);

    for (j=0; j < 3; j++) {

        if (LabTable[j] == NULL) {
            cmsFreeToneCurveTriple(LabTable);
            return NULL;
        }

        // We need to map * (0xffff / 0xff00), that's same as (257 / 256)
        // So we can use 258-entry tables to do the trick (i / 257) * (255 * 257) * (257 / 256);
        for (i=0; i < 257; i++)  {

            LabTable[j]->Table16[i] = (cmsUInt16Number) ((i * 0xffff + 0x80) >> 8);
        }

        LabTable[j] ->Table16[257] = 0xffff;
    }

    mpe = cmsStageAllocToneCurves(ContextID, 3, LabTable);
    cmsFreeToneCurveTriple(LabTable);

    if (mpe == NULL) return NULL;
    mpe ->Implements = cmsSigLabV2toV4;
    return mpe;
}

// ********************************************************************************

// Matrix-based conversion, which is more accurate, but slower and cannot properly be saved in devicelink profiles
cmsStage* CMSEXPORT _cmsStageAllocLabV2ToV4(cmsContext ContextID)
{
    static const cmsFloat64Number V2ToV4[] = { 65535.0/65280.0, 0, 0,
                                     0, 65535.0/65280.0, 0,
                                     0, 0, 65535.0/65280.0
                                     };

    cmsStage *mpe = cmsStageAllocMatrix(ContextID, 3, 3, V2ToV4, NULL);

    if (mpe == NULL) return mpe;
    mpe ->Implements = cmsSigLabV2toV4;
    return mpe;
}


// Reverse direction
cmsStage* CMSEXPORT _cmsStageAllocLabV4ToV2(cmsContext ContextID)
{
    static const cmsFloat64Number V4ToV2[] = { 65280.0/65535.0, 0, 0,
                                     0, 65280.0/65535.0, 0,
                                     0, 0, 65280.0/65535.0
                                     };

     cmsStage *mpe = cmsStageAllocMatrix(ContextID, 3, 3, V4ToV2, NULL);

    if (mpe == NULL) return mpe;
    mpe ->Implements = cmsSigLabV4toV2;
    return mpe;
}


// To Lab to float. Note that the MPE gives numbers in normal Lab range
// and we need 0..1.0 range for the formatters
// L* : 0...100 => 0...1.0  (L* / 100)
// ab* : -128..+127 to 0..1  ((ab* + 128) / 255)

cmsStage* _cmsStageNormalizeFromLabFloat(cmsContext ContextID)
{
    static const cmsFloat64Number a1[] = {
        1.0/100.0, 0, 0,
        0, 1.0/255.0, 0,
        0, 0, 1.0/255.0
    };

    static const cmsFloat64Number o1[] = {
        0,
        128.0/255.0,
        128.0/255.0
    };

    cmsStage *mpe = cmsStageAllocMatrix(ContextID, 3, 3, a1, o1);

    if (mpe == NULL) return mpe;
    mpe ->Implements = cmsSigLab2FloatPCS;
    return mpe;
}

// Fom XYZ to floating point PCS
cmsStage* _cmsStageNormalizeFromXyzFloat(cmsContext ContextID)
{
#define n (32768.0/65535.0)
    static const cmsFloat64Number a1[] = {
        n, 0, 0,
        0, n, 0,
        0, 0, n
    };
#undef n

    cmsStage *mpe =  cmsStageAllocMatrix(ContextID, 3, 3, a1, NULL);

    if (mpe == NULL) return mpe;
    mpe ->Implements = cmsSigXYZ2FloatPCS;
    return mpe;
}

cmsStage* _cmsStageNormalizeToLabFloat(cmsContext ContextID)
{
    static const cmsFloat64Number a1[] = {
        100.0, 0, 0,
        0, 255.0, 0,
        0, 0, 255.0
    };

    static const cmsFloat64Number o1[] = {
        0,
        -128.0,
        -128.0
    };

    cmsStage *mpe =  cmsStageAllocMatrix(ContextID, 3, 3, a1, o1);
    if (mpe == NULL) return mpe;
    mpe ->Implements = cmsSigFloatPCS2Lab;
    return mpe;
}

cmsStage* _cmsStageNormalizeToXyzFloat(cmsContext ContextID)
{
#define n (65535.0/32768.0)

    static const cmsFloat64Number a1[] = {
        n, 0, 0,
        0, n, 0,
        0, 0, n
    };
#undef n

    cmsStage *mpe = cmsStageAllocMatrix(ContextID, 3, 3, a1, NULL);
    if (mpe == NULL) return mpe;
    mpe ->Implements = cmsSigFloatPCS2XYZ;
    return mpe;
}

// Clips values smaller than zero
static
void Clipper(const cmsFloat32Number In[], cmsFloat32Number Out[], const cmsStage *mpe)
{
       cmsUInt32Number i;
       for (i = 0; i < mpe->InputChannels; i++) {

              cmsFloat32Number n = In[i];
              Out[i] = n < 0 ? 0 : n;
       }
}

cmsStage*  _cmsStageClipNegatives(cmsContext ContextID, cmsUInt32Number nChannels)
{
       return _cmsStageAllocPlaceholder(ContextID, cmsSigClipNegativesElemType,
              nChannels, nChannels, Clipper, NULL, NULL, NULL);
}

// ********************************************************************************
// Type cmsSigXYZ2LabElemType
// ********************************************************************************

static
void EvaluateXYZ2Lab(const cmsFloat32Number In[], cmsFloat32Number Out[], const cmsStage *mpe)
{
    cmsCIELab Lab;
    cmsCIEXYZ XYZ;
    const cmsFloat64Number XYZadj = MAX_ENCODEABLE_XYZ;

    // From 0..1.0 to XYZ

    XYZ.X = In[0] * XYZadj;
    XYZ.Y = In[1] * XYZadj;
    XYZ.Z = In[2] * XYZadj;

    cmsXYZ2Lab(NULL, &Lab, &XYZ);

    // From V4 Lab to 0..1.0

    Out[0] = (cmsFloat32Number) (Lab.L / 100.0);
    Out[1] = (cmsFloat32Number) ((Lab.a + 128.0) / 255.0);
    Out[2] = (cmsFloat32Number) ((Lab.b + 128.0) / 255.0);
    return;

    cmsUNUSED_PARAMETER(mpe);
}

cmsStage* CMSEXPORT _cmsStageAllocXYZ2Lab(cmsContext ContextID)
{
    return _cmsStageAllocPlaceholder(ContextID, cmsSigXYZ2LabElemType, 3, 3, EvaluateXYZ2Lab, NULL, NULL, NULL);

}

// ********************************************************************************

// For v4, S-Shaped curves are placed in a/b axis to increase resolution near gray

cmsStage* _cmsStageAllocLabPrelin(cmsContext ContextID)
{
    cmsToneCurve* LabTable[3];
    cmsFloat64Number Params[1] =  {2.4} ;

    LabTable[0] = cmsBuildGamma(ContextID, 1.0);
    LabTable[1] = cmsBuildParametricToneCurve(ContextID, 108, Params);
    LabTable[2] = cmsBuildParametricToneCurve(ContextID, 108, Params);

    return cmsStageAllocToneCurves(ContextID, 3, LabTable);
}


// Free a single MPE
void CMSEXPORT cmsStageFree(cmsStage* mpe)
{
    if (mpe ->FreePtr)
        mpe ->FreePtr(mpe);

    _cmsFree(mpe ->ContextID, mpe);
}


cmsUInt32Number  CMSEXPORT cmsStageInputChannels(const cmsStage* mpe)
{
    return mpe ->InputChannels;
}

cmsUInt32Number  CMSEXPORT cmsStageOutputChannels(const cmsStage* mpe)
{
    return mpe ->OutputChannels;
}

cmsStageSignature CMSEXPORT cmsStageType(const cmsStage* mpe)
{
    return mpe -> Type;
}

void* CMSEXPORT cmsStageData(const cmsStage* mpe)
{
    return mpe -> Data;
}

cmsStage*  CMSEXPORT cmsStageNext(const cmsStage* mpe)
{
    return mpe -> Next;
}


// Duplicates an MPE
cmsStage* CMSEXPORT cmsStageDup(cmsStage* mpe)
{
    cmsStage* NewMPE;

    if (mpe == NULL) return NULL;
    NewMPE = _cmsStageAllocPlaceholder(mpe ->ContextID,
                                     mpe ->Type,
                                     mpe ->InputChannels,
                                     mpe ->OutputChannels,
                                     mpe ->EvalPtr,
                                     mpe ->DupElemPtr,
                                     mpe ->FreePtr,
                                     NULL);
    if (NewMPE == NULL) return NULL;

    NewMPE ->Implements = mpe ->Implements;

    if (mpe ->DupElemPtr) {

        NewMPE ->Data = mpe ->DupElemPtr(mpe);

        if (NewMPE->Data == NULL) {

            cmsStageFree(NewMPE);
            return NULL;
        }

    } else {

        NewMPE ->Data       = NULL;
    }

    return NewMPE;
}


// ***********************************************************************************************************

// This function sets up the channel count
static
cmsBool BlessLUT(cmsPipeline* lut)
{
    // We can set the input/output channels only if we have elements.
    if (lut ->Elements != NULL) {

        cmsStage* prev;
        cmsStage* next;
        cmsStage* First;
        cmsStage* Last;

        First  = cmsPipelineGetPtrToFirstStage(lut);
        Last   = cmsPipelineGetPtrToLastStage(lut);

        if (First == NULL || Last == NULL) return FALSE;

        lut->InputChannels = First->InputChannels;
        lut->OutputChannels = Last->OutputChannels;

        // Check chain consistency
        prev = First;
        next = prev->Next;

        while (next != NULL)
        {
            if (next->InputChannels != prev->OutputChannels)
                return FALSE;

            next = next->Next;
            prev = prev->Next;
    }
}

    return TRUE;    
}


// Default to evaluate the LUT on 16 bit-basis. Precision is retained.
static
void _LUTeval16(register const cmsUInt16Number In[], register cmsUInt16Number Out[],  register const void* D)
{
    cmsPipeline* lut = (cmsPipeline*) D;
    cmsStage *mpe;
    cmsFloat32Number Storage[2][MAX_STAGE_CHANNELS];
    int Phase = 0, NextPhase;

    From16ToFloat(In, &Storage[Phase][0], lut ->InputChannels);

    for (mpe = lut ->Elements;
         mpe != NULL;
         mpe = mpe ->Next) {

             NextPhase = Phase ^ 1;
             mpe ->EvalPtr(&Storage[Phase][0], &Storage[NextPhase][0], mpe);
             Phase = NextPhase;
    }


    FromFloatTo16(&Storage[Phase][0], Out, lut ->OutputChannels);
}



// Does evaluate the LUT on cmsFloat32Number-basis.
static
void _LUTevalFloat(register const cmsFloat32Number In[], register cmsFloat32Number Out[], const void* D)
{
    cmsPipeline* lut = (cmsPipeline*) D;
    cmsStage *mpe;
    cmsFloat32Number Storage[2][MAX_STAGE_CHANNELS];
    int Phase = 0, NextPhase;

    memmove(&Storage[Phase][0], In, lut ->InputChannels  * sizeof(cmsFloat32Number));

    for (mpe = lut ->Elements;
         mpe != NULL;
         mpe = mpe ->Next) {

              NextPhase = Phase ^ 1;
              mpe ->EvalPtr(&Storage[Phase][0], &Storage[NextPhase][0], mpe);
              Phase = NextPhase;
    }

    memmove(Out, &Storage[Phase][0], lut ->OutputChannels * sizeof(cmsFloat32Number));
}


// LUT Creation & Destruction
cmsPipeline* CMSEXPORT cmsPipelineAlloc(cmsContext ContextID, cmsUInt32Number InputChannels, cmsUInt32Number OutputChannels)
{
       cmsPipeline* NewLUT;

       // A value of zero in channels is allowed as placeholder
       if (InputChannels >= cmsMAXCHANNELS ||
           OutputChannels >= cmsMAXCHANNELS) return NULL;

       NewLUT = (cmsPipeline*) _cmsMallocZero(ContextID, sizeof(cmsPipeline));
       if (NewLUT == NULL) return NULL;

       NewLUT -> InputChannels  = InputChannels;
       NewLUT -> OutputChannels = OutputChannels;

       NewLUT ->Eval16Fn    = _LUTeval16;
       NewLUT ->EvalFloatFn = _LUTevalFloat;
       NewLUT ->DupDataFn   = NULL;
       NewLUT ->FreeDataFn  = NULL;
       NewLUT ->Data        = NewLUT;
       NewLUT ->ContextID   = ContextID;

       if (!BlessLUT(NewLUT))
       {
           _cmsFree(ContextID, NewLUT);
           return NULL;
       }

       return NewLUT;
}

cmsContext CMSEXPORT cmsGetPipelineContextID(const cmsPipeline* lut)
{
    _cmsAssert(lut != NULL);
    return lut ->ContextID;
}

cmsUInt32Number CMSEXPORT cmsPipelineInputChannels(const cmsPipeline* lut)
{
    _cmsAssert(lut != NULL);
    return lut ->InputChannels;
}

cmsUInt32Number CMSEXPORT cmsPipelineOutputChannels(const cmsPipeline* lut)
{
    _cmsAssert(lut != NULL);
    return lut ->OutputChannels;
}

// Free a profile elements LUT
void CMSEXPORT cmsPipelineFree(cmsPipeline* lut)
{
    cmsStage *mpe, *Next;

    if (lut == NULL) return;

    for (mpe = lut ->Elements;
        mpe != NULL;
        mpe = Next) {

            Next = mpe ->Next;
            cmsStageFree(mpe);
    }

    if (lut ->FreeDataFn) lut ->FreeDataFn(lut ->ContextID, lut ->Data);

    _cmsFree(lut ->ContextID, lut);
}


// Default to evaluate the LUT on 16 bit-basis.
void CMSEXPORT cmsPipelineEval16(const cmsUInt16Number In[], cmsUInt16Number Out[],  const cmsPipeline* lut)
{
    _cmsAssert(lut != NULL);
    lut ->Eval16Fn(In, Out, lut->Data);
}


// Does evaluate the LUT on cmsFloat32Number-basis.
void CMSEXPORT cmsPipelineEvalFloat(const cmsFloat32Number In[], cmsFloat32Number Out[], const cmsPipeline* lut)
{
    _cmsAssert(lut != NULL);
    lut ->EvalFloatFn(In, Out, lut);
}



// Duplicates a LUT
cmsPipeline* CMSEXPORT cmsPipelineDup(const cmsPipeline* lut)
{
    cmsPipeline* NewLUT;
    cmsStage *NewMPE, *Anterior = NULL, *mpe;
    cmsBool  First = TRUE;

    if (lut == NULL) return NULL;

    NewLUT = cmsPipelineAlloc(lut ->ContextID, lut ->InputChannels, lut ->OutputChannels);
    if (NewLUT == NULL) return NULL;

    for (mpe = lut ->Elements;
         mpe != NULL;
         mpe = mpe ->Next) {

             NewMPE = cmsStageDup(mpe);

             if (NewMPE == NULL) {
                 cmsPipelineFree(NewLUT);
                 return NULL;
             }

             if (First) {
                 NewLUT ->Elements = NewMPE;
                 First = FALSE;
             }
             else {
                if (Anterior != NULL) 
                    Anterior ->Next = NewMPE;
             }

            Anterior = NewMPE;
    }

    NewLUT ->Eval16Fn    = lut ->Eval16Fn;
    NewLUT ->EvalFloatFn = lut ->EvalFloatFn;
    NewLUT ->DupDataFn   = lut ->DupDataFn;
    NewLUT ->FreeDataFn  = lut ->FreeDataFn;

    if (NewLUT ->DupDataFn != NULL)
        NewLUT ->Data = NewLUT ->DupDataFn(lut ->ContextID, lut->Data);


    NewLUT ->SaveAs8Bits    = lut ->SaveAs8Bits;

    if (!BlessLUT(NewLUT))
    {
        _cmsFree(lut->ContextID, NewLUT);
        return NULL;
    }

    return NewLUT;
}


int CMSEXPORT cmsPipelineInsertStage(cmsPipeline* lut, cmsStageLoc loc, cmsStage* mpe)
{
    cmsStage* Anterior = NULL, *pt;

    if (lut == NULL || mpe == NULL)
        return FALSE;

    switch (loc) {

        case cmsAT_BEGIN:
            mpe ->Next = lut ->Elements;
            lut ->Elements = mpe;
            break;

        case cmsAT_END:

            if (lut ->Elements == NULL)
                lut ->Elements = mpe;
            else {

                for (pt = lut ->Elements;
                     pt != NULL;
                     pt = pt -> Next) Anterior = pt;
                
                Anterior ->Next = mpe;
                mpe ->Next = NULL;
            }
            break;
        default:;
            return FALSE;
    }

    return BlessLUT(lut);    
}

// Unlink an element and return the pointer to it
void CMSEXPORT cmsPipelineUnlinkStage(cmsPipeline* lut, cmsStageLoc loc, cmsStage** mpe)
{
    cmsStage *Anterior, *pt, *Last;
    cmsStage *Unlinked = NULL;


    // If empty LUT, there is nothing to remove
    if (lut ->Elements == NULL) {
        if (mpe) *mpe = NULL;
        return;
    }

    // On depending on the strategy...
    switch (loc) {

        case cmsAT_BEGIN:
            {
                cmsStage* elem = lut ->Elements;

                lut ->Elements = elem -> Next;
                elem ->Next = NULL;
                Unlinked = elem;

            }
            break;

        case cmsAT_END:
            Anterior = Last = NULL;
            for (pt = lut ->Elements;
                pt != NULL;
                pt = pt -> Next) {
                    Anterior = Last;
                    Last = pt;
            }

            Unlinked = Last;  // Next already points to NULL

            // Truncate the chain
            if (Anterior)
                Anterior ->Next = NULL;
            else
                lut ->Elements = NULL;
            break;
        default:;
    }

    if (mpe)
        *mpe = Unlinked;
    else
        cmsStageFree(Unlinked);

    // May fail, but we ignore it
    BlessLUT(lut);
}


// Concatenate two LUT into a new single one
cmsBool  CMSEXPORT cmsPipelineCat(cmsPipeline* l1, const cmsPipeline* l2)
{
    cmsStage* mpe;

    // If both LUTS does not have elements, we need to inherit
    // the number of channels
    if (l1 ->Elements == NULL && l2 ->Elements == NULL) {
        l1 ->InputChannels  = l2 ->InputChannels;
        l1 ->OutputChannels = l2 ->OutputChannels;
    }

    // Cat second
    for (mpe = l2 ->Elements;
         mpe != NULL;
         mpe = mpe ->Next) {

            // We have to dup each element
            if (!cmsPipelineInsertStage(l1, cmsAT_END, cmsStageDup(mpe)))
                return FALSE;
    }

    return BlessLUT(l1);    
}


cmsBool CMSEXPORT cmsPipelineSetSaveAs8bitsFlag(cmsPipeline* lut, cmsBool On)
{
    cmsBool Anterior = lut ->SaveAs8Bits;

    lut ->SaveAs8Bits = On;
    return Anterior;
}


cmsStage* CMSEXPORT cmsPipelineGetPtrToFirstStage(const cmsPipeline* lut)
{
    return lut ->Elements;
}

cmsStage* CMSEXPORT cmsPipelineGetPtrToLastStage(const cmsPipeline* lut)
{
    cmsStage *mpe, *Anterior = NULL;

    for (mpe = lut ->Elements; mpe != NULL; mpe = mpe ->Next)
        Anterior = mpe;

    return Anterior;
}

cmsUInt32Number CMSEXPORT cmsPipelineStageCount(const cmsPipeline* lut)
{
    cmsStage *mpe;
    cmsUInt32Number n;

    for (n=0, mpe = lut ->Elements; mpe != NULL; mpe = mpe ->Next)
            n++;

    return n;
}

// This function may be used to set the optional evaluator and a block of private data. If private data is being used, an optional
// duplicator and free functions should also be specified in order to duplicate the LUT construct. Use NULL to inhibit such functionality.
void CMSEXPORT _cmsPipelineSetOptimizationParameters(cmsPipeline* Lut,
                                        _cmsOPTeval16Fn Eval16,
                                        void* PrivateData,
                                        _cmsFreeUserDataFn FreePrivateDataFn,
                                        _cmsDupUserDataFn  DupPrivateDataFn)
{

    Lut ->Eval16Fn = Eval16;
    Lut ->DupDataFn = DupPrivateDataFn;
    Lut ->FreeDataFn = FreePrivateDataFn;
    Lut ->Data = PrivateData;
}


// ----------------------------------------------------------- Reverse interpolation
// Here's how it goes. The derivative Df(x) of the function f is the linear
// transformation that best approximates f near the point x. It can be represented
// by a matrix A whose entries are the partial derivatives of the components of f
// with respect to all the coordinates. This is know as the Jacobian
//
// The best linear approximation to f is given by the matrix equation:
//
// y-y0 = A (x-x0)
//
// So, if x0 is a good "guess" for the zero of f, then solving for the zero of this
// linear approximation will give a "better guess" for the zero of f. Thus let y=0,
// and since y0=f(x0) one can solve the above equation for x. This leads to the
// Newton's method formula:
//
// xn+1 = xn - A-1 f(xn)
//
// where xn+1 denotes the (n+1)-st guess, obtained from the n-th guess xn in the
// fashion described above. Iterating this will give better and better approximations
// if you have a "good enough" initial guess.


#define JACOBIAN_EPSILON            0.001f
#define INVERSION_MAX_ITERATIONS    30

// Increment with reflexion on boundary
static
void IncDelta(cmsFloat32Number *Val)
{
    if (*Val < (1.0 - JACOBIAN_EPSILON))

        *Val += JACOBIAN_EPSILON;

    else
        *Val -= JACOBIAN_EPSILON;

}



// Euclidean distance between two vectors of n elements each one
static
cmsFloat32Number EuclideanDistance(cmsFloat32Number a[], cmsFloat32Number b[], int n)
{
    cmsFloat32Number sum = 0;
    int i;

    for (i=0; i < n; i++) {
        cmsFloat32Number dif = b[i] - a[i];
        sum +=  dif * dif;
    }

    return sqrtf(sum);
}


// Evaluate a LUT in reverse direction. It only searches on 3->3 LUT. Uses Newton method
//
// x1 <- x - [J(x)]^-1 * f(x)
//
// lut: The LUT on where to do the search
// Target: LabK, 3 values of Lab plus destination K which is fixed
// Result: The obtained CMYK
// Hint:   Location where begin the search

cmsBool CMSEXPORT cmsPipelineEvalReverseFloat(cmsFloat32Number Target[],
                                              cmsFloat32Number Result[],
                                              cmsFloat32Number Hint[],
                                              const cmsPipeline* lut)
{
    cmsUInt32Number  i, j;
    cmsFloat64Number  error, LastError = 1E20;
    cmsFloat32Number  fx[4], x[4], xd[4], fxd[4];
    cmsVEC3 tmp, tmp2;
    cmsMAT3 Jacobian;
    
    // Only 3->3 and 4->3 are supported
    if (lut ->InputChannels != 3 && lut ->InputChannels != 4) return FALSE;
    if (lut ->OutputChannels != 3) return FALSE;
   
    // Take the hint as starting point if specified
    if (Hint == NULL) {

        // Begin at any point, we choose 1/3 of CMY axis
        x[0] = x[1] = x[2] = 0.3f;
    }
    else {

        // Only copy 3 channels from hint...
        for (j=0; j < 3; j++)
            x[j] = Hint[j];
    }

    // If Lut is 4-dimensions, then grab target[3], which is fixed
    if (lut ->InputChannels == 4) {
        x[3] = Target[3];
    }
    else x[3] = 0; // To keep lint happy


    // Iterate
    for (i = 0; i < INVERSION_MAX_ITERATIONS; i++) {

        // Get beginning fx
        cmsPipelineEvalFloat(x, fx, lut);

        // Compute error
        error = EuclideanDistance(fx, Target, 3);

        // If not convergent, return last safe value
        if (error >= LastError)
            break;

        // Keep latest values
        LastError     = error;
        for (j=0; j < lut ->InputChannels; j++)
                Result[j] = x[j];

        // Found an exact match?
        if (error <= 0)
            break;

        // Obtain slope (the Jacobian)
        for (j = 0; j < 3; j++) {

            xd[0] = x[0];
            xd[1] = x[1];
            xd[2] = x[2];
            xd[3] = x[3];  // Keep fixed channel

            IncDelta(&xd[j]);

            cmsPipelineEvalFloat(xd, fxd, lut);

            Jacobian.v[0].n[j] = ((fxd[0] - fx[0]) / JACOBIAN_EPSILON);
            Jacobian.v[1].n[j] = ((fxd[1] - fx[1]) / JACOBIAN_EPSILON);
            Jacobian.v[2].n[j] = ((fxd[2] - fx[2]) / JACOBIAN_EPSILON);
        }

        // Solve system
        tmp2.n[0] = fx[0] - Target[0];
        tmp2.n[1] = fx[1] - Target[1];
        tmp2.n[2] = fx[2] - Target[2];

        if (!_cmsMAT3solve(&tmp, &Jacobian, &tmp2))
            return FALSE;

        // Move our guess
        x[0] -= (cmsFloat32Number) tmp.n[0];
        x[1] -= (cmsFloat32Number) tmp.n[1];
        x[2] -= (cmsFloat32Number) tmp.n[2];

        // Some clipping....
        for (j=0; j < 3; j++) {
            if (x[j] < 0) x[j] = 0;
            else
                if (x[j] > 1.0) x[j] = 1.0;
        }
    }

    return TRUE;
}


