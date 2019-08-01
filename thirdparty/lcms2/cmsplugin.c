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


// ----------------------------------------------------------------------------------
// Encoding & Decoding support functions
// ----------------------------------------------------------------------------------

//      Little-Endian to Big-Endian

// Adjust a word value after being readed/ before being written from/to an ICC profile
cmsUInt16Number CMSEXPORT  _cmsAdjustEndianess16(cmsUInt16Number Word)
{
#ifndef CMS_USE_BIG_ENDIAN

    cmsUInt8Number* pByte = (cmsUInt8Number*) &Word;
    cmsUInt8Number tmp;

    tmp = pByte[0];
    pByte[0] = pByte[1];
    pByte[1] = tmp;
#endif

    return Word;
}


// Transports to properly encoded values - note that icc profiles does use big endian notation.

// 1 2 3 4
// 4 3 2 1

cmsUInt32Number CMSEXPORT  _cmsAdjustEndianess32(cmsUInt32Number DWord)
{
#ifndef CMS_USE_BIG_ENDIAN

    cmsUInt8Number* pByte = (cmsUInt8Number*) &DWord;
    cmsUInt8Number temp1;
    cmsUInt8Number temp2;

    temp1 = *pByte++;
    temp2 = *pByte++;
    *(pByte-1) = *pByte;
    *pByte++ = temp2;
    *(pByte-3) = *pByte;
    *pByte = temp1;
#endif
    return DWord;
}

// 1 2 3 4 5 6 7 8
// 8 7 6 5 4 3 2 1

void CMSEXPORT  _cmsAdjustEndianess64(cmsUInt64Number* Result, cmsUInt64Number* QWord)
{

#ifndef CMS_USE_BIG_ENDIAN

    cmsUInt8Number* pIn  = (cmsUInt8Number*) QWord;
    cmsUInt8Number* pOut = (cmsUInt8Number*) Result;

    _cmsAssert(Result != NULL);

    pOut[7] = pIn[0];
    pOut[6] = pIn[1];
    pOut[5] = pIn[2];
    pOut[4] = pIn[3];
    pOut[3] = pIn[4];
    pOut[2] = pIn[5];
    pOut[1] = pIn[6];
    pOut[0] = pIn[7];

#else
    _cmsAssert(Result != NULL);

#  ifdef CMS_DONT_USE_INT64
    (*Result)[0] = QWord[0];
    (*Result)[1] = QWord[1];
#  else
    *Result = *QWord;
#  endif
#endif
}

// Auxiliary -- read 8, 16 and 32-bit numbers
cmsBool CMSEXPORT  _cmsReadUInt8Number(cmsIOHANDLER* io, cmsUInt8Number* n)
{
    cmsUInt8Number tmp;

    _cmsAssert(io != NULL);

    if (io -> Read(io, &tmp, sizeof(cmsUInt8Number), 1) != 1)
            return FALSE;

    if (n != NULL) *n = tmp;
    return TRUE;
}

cmsBool CMSEXPORT  _cmsReadUInt16Number(cmsIOHANDLER* io, cmsUInt16Number* n)
{
    cmsUInt16Number tmp;

    _cmsAssert(io != NULL);

    if (io -> Read(io, &tmp, sizeof(cmsUInt16Number), 1) != 1)
            return FALSE;

    if (n != NULL) *n = _cmsAdjustEndianess16(tmp);
    return TRUE;
}

cmsBool CMSEXPORT  _cmsReadUInt16Array(cmsIOHANDLER* io, cmsUInt32Number n, cmsUInt16Number* Array)
{
    cmsUInt32Number i;

    _cmsAssert(io != NULL);

    for (i=0; i < n; i++) {

        if (Array != NULL) {
            if (!_cmsReadUInt16Number(io, Array + i)) return FALSE;
        }
        else {
            if (!_cmsReadUInt16Number(io, NULL)) return FALSE;
        }

    }
    return TRUE;
}

cmsBool CMSEXPORT  _cmsReadUInt32Number(cmsIOHANDLER* io, cmsUInt32Number* n)
{
    cmsUInt32Number tmp;

    _cmsAssert(io != NULL);

    if (io -> Read(io, &tmp, sizeof(cmsUInt32Number), 1) != 1)
            return FALSE;

    if (n != NULL) *n = _cmsAdjustEndianess32(tmp);
    return TRUE;
}

cmsBool CMSEXPORT  _cmsReadFloat32Number(cmsIOHANDLER* io, cmsFloat32Number* n)
{
    cmsUInt32Number tmp;

    _cmsAssert(io != NULL);

    if (io->Read(io, &tmp, sizeof(cmsUInt32Number), 1) != 1)
        return FALSE;

    if (n != NULL) {

        tmp = _cmsAdjustEndianess32(tmp);
        *n = *(cmsFloat32Number*)(void*)&tmp;
        
        // Safeguard which covers against absurd values
        if (*n > 1E+20 || *n < -1E+20) return FALSE;

        #if defined(_MSC_VER) && _MSC_VER < 1800
           return TRUE;
        #elif defined (__BORLANDC__)
           return TRUE;
        #else

           // fpclassify() required by C99 (only provided by MSVC >= 1800, VS2013 onwards)
           return ((fpclassify(*n) == FP_ZERO) || (fpclassify(*n) == FP_NORMAL));
        #endif        
    }

    return TRUE;
}


cmsBool CMSEXPORT   _cmsReadUInt64Number(cmsIOHANDLER* io, cmsUInt64Number* n)
{
    cmsUInt64Number tmp;

    _cmsAssert(io != NULL);

    if (io -> Read(io, &tmp, sizeof(cmsUInt64Number), 1) != 1)
            return FALSE;

    if (n != NULL) {

        _cmsAdjustEndianess64(n, &tmp);
    }

    return TRUE;
}


cmsBool CMSEXPORT  _cmsRead15Fixed16Number(cmsIOHANDLER* io, cmsFloat64Number* n)
{
    cmsUInt32Number tmp;

    _cmsAssert(io != NULL);

    if (io -> Read(io, &tmp, sizeof(cmsUInt32Number), 1) != 1)
            return FALSE;

    if (n != NULL) {
        *n = _cms15Fixed16toDouble((cmsS15Fixed16Number) _cmsAdjustEndianess32(tmp));
    }

    return TRUE;
}


cmsBool CMSEXPORT  _cmsReadXYZNumber(cmsIOHANDLER* io, cmsCIEXYZ* XYZ)
{
    cmsEncodedXYZNumber xyz;

    _cmsAssert(io != NULL);

    if (io ->Read(io, &xyz, sizeof(cmsEncodedXYZNumber), 1) != 1) return FALSE;

    if (XYZ != NULL) {

        XYZ->X = _cms15Fixed16toDouble((cmsS15Fixed16Number) _cmsAdjustEndianess32((cmsUInt32Number) xyz.X));
        XYZ->Y = _cms15Fixed16toDouble((cmsS15Fixed16Number) _cmsAdjustEndianess32((cmsUInt32Number) xyz.Y));
        XYZ->Z = _cms15Fixed16toDouble((cmsS15Fixed16Number) _cmsAdjustEndianess32((cmsUInt32Number) xyz.Z));
    }
    return TRUE;
}

cmsBool CMSEXPORT  _cmsWriteUInt8Number(cmsIOHANDLER* io, cmsUInt8Number n)
{
    _cmsAssert(io != NULL);

    if (io -> Write(io, sizeof(cmsUInt8Number), &n) != 1)
            return FALSE;

    return TRUE;
}

cmsBool CMSEXPORT  _cmsWriteUInt16Number(cmsIOHANDLER* io, cmsUInt16Number n)
{
    cmsUInt16Number tmp;

    _cmsAssert(io != NULL);

    tmp = _cmsAdjustEndianess16(n);
    if (io -> Write(io, sizeof(cmsUInt16Number), &tmp) != 1)
            return FALSE;

    return TRUE;
}

cmsBool CMSEXPORT  _cmsWriteUInt16Array(cmsIOHANDLER* io, cmsUInt32Number n, const cmsUInt16Number* Array)
{
    cmsUInt32Number i;

    _cmsAssert(io != NULL);
    _cmsAssert(Array != NULL);

    for (i=0; i < n; i++) {
        if (!_cmsWriteUInt16Number(io, Array[i])) return FALSE;
    }

    return TRUE;
}

cmsBool CMSEXPORT  _cmsWriteUInt32Number(cmsIOHANDLER* io, cmsUInt32Number n)
{
    cmsUInt32Number tmp;

    _cmsAssert(io != NULL);

    tmp = _cmsAdjustEndianess32(n);
    if (io -> Write(io, sizeof(cmsUInt32Number), &tmp) != 1)
            return FALSE;

    return TRUE;
}


cmsBool CMSEXPORT  _cmsWriteFloat32Number(cmsIOHANDLER* io, cmsFloat32Number n)
{
    cmsUInt32Number tmp;

    _cmsAssert(io != NULL);

    tmp = *(cmsUInt32Number*) (void*) &n;
    tmp = _cmsAdjustEndianess32(tmp);
    if (io -> Write(io, sizeof(cmsUInt32Number), &tmp) != 1)
            return FALSE;

    return TRUE;
}

cmsBool CMSEXPORT  _cmsWriteUInt64Number(cmsIOHANDLER* io, cmsUInt64Number* n)
{
    cmsUInt64Number tmp;

    _cmsAssert(io != NULL);

    _cmsAdjustEndianess64(&tmp, n);
    if (io -> Write(io, sizeof(cmsUInt64Number), &tmp) != 1)
            return FALSE;

    return TRUE;
}

cmsBool CMSEXPORT  _cmsWrite15Fixed16Number(cmsIOHANDLER* io, cmsFloat64Number n)
{
    cmsUInt32Number tmp;

    _cmsAssert(io != NULL);

    tmp = _cmsAdjustEndianess32((cmsUInt32Number) _cmsDoubleTo15Fixed16(n));
    if (io -> Write(io, sizeof(cmsUInt32Number), &tmp) != 1)
            return FALSE;

    return TRUE;
}

cmsBool CMSEXPORT  _cmsWriteXYZNumber(cmsIOHANDLER* io, const cmsCIEXYZ* XYZ)
{
    cmsEncodedXYZNumber xyz;

    _cmsAssert(io != NULL);
    _cmsAssert(XYZ != NULL);

    xyz.X = (cmsS15Fixed16Number) _cmsAdjustEndianess32((cmsUInt32Number) _cmsDoubleTo15Fixed16(XYZ->X));
    xyz.Y = (cmsS15Fixed16Number) _cmsAdjustEndianess32((cmsUInt32Number) _cmsDoubleTo15Fixed16(XYZ->Y));
    xyz.Z = (cmsS15Fixed16Number) _cmsAdjustEndianess32((cmsUInt32Number) _cmsDoubleTo15Fixed16(XYZ->Z));

    return io -> Write(io,  sizeof(cmsEncodedXYZNumber), &xyz);
}

// from Fixed point 8.8 to double
cmsFloat64Number CMSEXPORT _cms8Fixed8toDouble(cmsUInt16Number fixed8)
{
       cmsUInt8Number  msb, lsb;

       lsb = (cmsUInt8Number) (fixed8 & 0xff);
       msb = (cmsUInt8Number) (((cmsUInt16Number) fixed8 >> 8) & 0xff);

       return (cmsFloat64Number) ((cmsFloat64Number) msb + ((cmsFloat64Number) lsb / 256.0));
}

cmsUInt16Number CMSEXPORT _cmsDoubleTo8Fixed8(cmsFloat64Number val)
{
    cmsS15Fixed16Number GammaFixed32 = _cmsDoubleTo15Fixed16(val);
    return  (cmsUInt16Number) ((GammaFixed32 >> 8) & 0xFFFF);
}

// from Fixed point 15.16 to double
cmsFloat64Number CMSEXPORT _cms15Fixed16toDouble(cmsS15Fixed16Number fix32)
{
    cmsFloat64Number floater, sign, mid;
    int Whole, FracPart;

    sign  = (fix32 < 0 ? -1 : 1);
    fix32 = abs(fix32);

    Whole     = (cmsUInt16Number)(fix32 >> 16) & 0xffff;
    FracPart  = (cmsUInt16Number)(fix32 & 0xffff);

    mid     = (cmsFloat64Number) FracPart / 65536.0;
    floater = (cmsFloat64Number) Whole + mid;

    return sign * floater;
}

// from double to Fixed point 15.16
cmsS15Fixed16Number CMSEXPORT _cmsDoubleTo15Fixed16(cmsFloat64Number v)
{
    return ((cmsS15Fixed16Number) floor((v)*65536.0 + 0.5));
}

// Date/Time functions

void CMSEXPORT _cmsDecodeDateTimeNumber(const cmsDateTimeNumber *Source, struct tm *Dest)
{

    _cmsAssert(Dest != NULL);
    _cmsAssert(Source != NULL);

    Dest->tm_sec   = _cmsAdjustEndianess16(Source->seconds);
    Dest->tm_min   = _cmsAdjustEndianess16(Source->minutes);
    Dest->tm_hour  = _cmsAdjustEndianess16(Source->hours);
    Dest->tm_mday  = _cmsAdjustEndianess16(Source->day);
    Dest->tm_mon   = _cmsAdjustEndianess16(Source->month) - 1;
    Dest->tm_year  = _cmsAdjustEndianess16(Source->year) - 1900;
    Dest->tm_wday  = -1;
    Dest->tm_yday  = -1;
    Dest->tm_isdst = 0;
}

void CMSEXPORT _cmsEncodeDateTimeNumber(cmsDateTimeNumber *Dest, const struct tm *Source)
{
    _cmsAssert(Dest != NULL);
    _cmsAssert(Source != NULL);

    Dest->seconds = _cmsAdjustEndianess16((cmsUInt16Number) Source->tm_sec);
    Dest->minutes = _cmsAdjustEndianess16((cmsUInt16Number) Source->tm_min);
    Dest->hours   = _cmsAdjustEndianess16((cmsUInt16Number) Source->tm_hour);
    Dest->day     = _cmsAdjustEndianess16((cmsUInt16Number) Source->tm_mday);
    Dest->month   = _cmsAdjustEndianess16((cmsUInt16Number) (Source->tm_mon + 1));
    Dest->year    = _cmsAdjustEndianess16((cmsUInt16Number) (Source->tm_year + 1900));
}

// Read base and return type base
cmsTagTypeSignature CMSEXPORT _cmsReadTypeBase(cmsIOHANDLER* io)
{
    _cmsTagBase Base;

    _cmsAssert(io != NULL);

    if (io -> Read(io, &Base, sizeof(_cmsTagBase), 1) != 1)
        return (cmsTagTypeSignature) 0;

    return (cmsTagTypeSignature) _cmsAdjustEndianess32(Base.sig);
}

// Setup base marker
cmsBool  CMSEXPORT _cmsWriteTypeBase(cmsIOHANDLER* io, cmsTagTypeSignature sig)
{
    _cmsTagBase  Base;

    _cmsAssert(io != NULL);

    Base.sig = (cmsTagTypeSignature) _cmsAdjustEndianess32(sig);
    memset(&Base.reserved, 0, sizeof(Base.reserved));
    return io -> Write(io, sizeof(_cmsTagBase), &Base);
}

cmsBool CMSEXPORT _cmsReadAlignment(cmsIOHANDLER* io)
{
    cmsUInt8Number  Buffer[4];
    cmsUInt32Number NextAligned, At;
    cmsUInt32Number BytesToNextAlignedPos;

    _cmsAssert(io != NULL);

    At = io -> Tell(io);
    NextAligned = _cmsALIGNLONG(At);
    BytesToNextAlignedPos = NextAligned - At;
    if (BytesToNextAlignedPos == 0) return TRUE;
    if (BytesToNextAlignedPos > 4)  return FALSE;

    return (io ->Read(io, Buffer, BytesToNextAlignedPos, 1) == 1);
}

cmsBool CMSEXPORT _cmsWriteAlignment(cmsIOHANDLER* io)
{
    cmsUInt8Number  Buffer[4];
    cmsUInt32Number NextAligned, At;
    cmsUInt32Number BytesToNextAlignedPos;

    _cmsAssert(io != NULL);

    At = io -> Tell(io);
    NextAligned = _cmsALIGNLONG(At);
    BytesToNextAlignedPos = NextAligned - At;
    if (BytesToNextAlignedPos == 0) return TRUE;
    if (BytesToNextAlignedPos > 4)  return FALSE;

    memset(Buffer, 0, BytesToNextAlignedPos);
    return io -> Write(io, BytesToNextAlignedPos, Buffer);
}


// To deal with text streams. 2K at most
cmsBool CMSEXPORT _cmsIOPrintf(cmsIOHANDLER* io, const char* frm, ...)
{
    va_list args;
    int len;
    cmsUInt8Number Buffer[2048];
    cmsBool rc;

    _cmsAssert(io != NULL);
    _cmsAssert(frm != NULL);

    va_start(args, frm);

    len = vsnprintf((char*) Buffer, 2047, frm, args);
    if (len < 0) {
        va_end(args);
        return FALSE;   // Truncated, which is a fatal error for us
    }

    rc = io ->Write(io, (cmsUInt32Number) len, Buffer);

    va_end(args);

    return rc;
}


// Plugin memory management -------------------------------------------------------------------------------------------------

// Specialized malloc for plug-ins, that is freed upon exit.
void* _cmsPluginMalloc(cmsContext ContextID, cmsUInt32Number size)
{
    struct _cmsContext_struct* ctx = _cmsGetContext(ContextID);

    if (ctx ->MemPool == NULL) {

        if (ContextID == NULL) {

            ctx->MemPool = _cmsCreateSubAlloc(0, 2*1024);
            if (ctx->MemPool == NULL) return NULL;
        }
        else {
            cmsSignalError(ContextID, cmsERROR_CORRUPTION_DETECTED, "NULL memory pool on context");
            return NULL;
        }
    }

    return _cmsSubAlloc(ctx->MemPool, size);
}


// Main plug-in dispatcher
cmsBool CMSEXPORT cmsPlugin(void* Plug_in)
{
    return cmsPluginTHR(NULL, Plug_in);
}

cmsBool CMSEXPORT cmsPluginTHR(cmsContext id, void* Plug_in)
{
    cmsPluginBase* Plugin;

    for (Plugin = (cmsPluginBase*) Plug_in;
         Plugin != NULL;
         Plugin = Plugin -> Next) {

            if (Plugin -> Magic != cmsPluginMagicNumber) {
                cmsSignalError(id, cmsERROR_UNKNOWN_EXTENSION, "Unrecognized plugin");
                return FALSE;
            }

            if (Plugin ->ExpectedVersion > LCMS_VERSION) {
                cmsSignalError(id, cmsERROR_UNKNOWN_EXTENSION, "plugin needs Little CMS %d, current version is %d",
                    Plugin ->ExpectedVersion, LCMS_VERSION);
                return FALSE;
            }

            switch (Plugin -> Type) {

                case cmsPluginMemHandlerSig:
                    if (!_cmsRegisterMemHandlerPlugin(id, Plugin)) return FALSE;
                    break;

                case cmsPluginInterpolationSig:
                    if (!_cmsRegisterInterpPlugin(id, Plugin)) return FALSE;
                    break;

                case cmsPluginTagTypeSig:
                    if (!_cmsRegisterTagTypePlugin(id, Plugin)) return FALSE;
                    break;

                case cmsPluginTagSig:
                    if (!_cmsRegisterTagPlugin(id, Plugin)) return FALSE;
                    break;

                case cmsPluginFormattersSig:
                    if (!_cmsRegisterFormattersPlugin(id, Plugin)) return FALSE;
                    break;

                case cmsPluginRenderingIntentSig:
                    if (!_cmsRegisterRenderingIntentPlugin(id, Plugin)) return FALSE;
                    break;

                case cmsPluginParametricCurveSig:
                    if (!_cmsRegisterParametricCurvesPlugin(id, Plugin)) return FALSE;
                    break;

                case cmsPluginMultiProcessElementSig:
                    if (!_cmsRegisterMultiProcessElementPlugin(id, Plugin)) return FALSE;
                    break;

                case cmsPluginOptimizationSig:
                    if (!_cmsRegisterOptimizationPlugin(id, Plugin)) return FALSE;
                    break;

                case cmsPluginTransformSig:
                    if (!_cmsRegisterTransformPlugin(id, Plugin)) return FALSE;
                    break;

                case cmsPluginMutexSig:
                    if (!_cmsRegisterMutexPlugin(id, Plugin)) return FALSE;
                    break;

                default:
                    cmsSignalError(id, cmsERROR_UNKNOWN_EXTENSION, "Unrecognized plugin type '%X'", Plugin -> Type);
                    return FALSE;
            }
    }

    // Keep a reference to the plug-in
    return TRUE;
}


// Revert all plug-ins to default
void CMSEXPORT cmsUnregisterPlugins(void)
{
    cmsUnregisterPluginsTHR(NULL);
}


// The Global storage for system context. This is the one and only global variable
// pointers structure. All global vars are referenced here.
static struct _cmsContext_struct globalContext = {

    NULL,                              // Not in the linked list
    NULL,                              // No suballocator
    {
        NULL,                          //  UserPtr,            
        &_cmsLogErrorChunk,            //  Logger,
        &_cmsAlarmCodesChunk,          //  AlarmCodes,
        &_cmsAdaptationStateChunk,     //  AdaptationState, 
        &_cmsMemPluginChunk,           //  MemPlugin,
        &_cmsInterpPluginChunk,        //  InterpPlugin,
        &_cmsCurvesPluginChunk,        //  CurvesPlugin,
        &_cmsFormattersPluginChunk,    //  FormattersPlugin,
        &_cmsTagTypePluginChunk,       //  TagTypePlugin,
        &_cmsTagPluginChunk,           //  TagPlugin,
        &_cmsIntentsPluginChunk,       //  IntentPlugin,
        &_cmsMPETypePluginChunk,       //  MPEPlugin,
        &_cmsOptimizationPluginChunk,  //  OptimizationPlugin,
        &_cmsTransformPluginChunk,     //  TransformPlugin,
        &_cmsMutexPluginChunk          //  MutexPlugin
    },
    
    { NULL, NULL, NULL, NULL, NULL, NULL } // The default memory allocator is not used for context 0
};


// The context pool (linked list head)
static _cmsMutex _cmsContextPoolHeadMutex = CMS_MUTEX_INITIALIZER;
static struct _cmsContext_struct* _cmsContextPoolHead = NULL;

// Internal, get associated pointer, with guessing. Never returns NULL.
struct _cmsContext_struct* _cmsGetContext(cmsContext ContextID)
{
    struct _cmsContext_struct* id = (struct _cmsContext_struct*) ContextID;
    struct _cmsContext_struct* ctx;


    // On 0, use global settings
    if (id == NULL) 
        return &globalContext;

    // Search
    for (ctx = _cmsContextPoolHead;
         ctx != NULL;
         ctx = ctx ->Next) {

            // Found it?
            if (id == ctx)
                return ctx; // New-style context, 
    }

    return &globalContext;
}


// Internal: get the memory area associanted with each context client
// Returns the block assigned to the specific zone. Never return NULL.
void* _cmsContextGetClientChunk(cmsContext ContextID, _cmsMemoryClient mc)
{
    struct _cmsContext_struct* ctx;
    void *ptr;

    if ((int) mc < 0 || mc >= MemoryClientMax) {
        
           cmsSignalError(ContextID, cmsERROR_INTERNAL, "Bad context client -- possible corruption");

           // This is catastrophic. Should never reach here
           _cmsAssert(0);

           // Reverts to global context
           return globalContext.chunks[UserPtr];
    }
    
    ctx = _cmsGetContext(ContextID);
    ptr = ctx ->chunks[mc];

    if (ptr != NULL)
        return ptr;

    // A null ptr means no special settings for that context, and this 
    // reverts to Context0 globals
    return globalContext.chunks[mc];    
}


// This function returns the given context its default pristine state,
// as no plug-ins were declared. There is no way to unregister a single 
// plug-in, as a single call to cmsPluginTHR() function may register 
// many different plug-ins simultaneously, then there is no way to 
// identify which plug-in to unregister.
void CMSEXPORT cmsUnregisterPluginsTHR(cmsContext ContextID)
{
    _cmsRegisterMemHandlerPlugin(ContextID, NULL);
    _cmsRegisterInterpPlugin(ContextID, NULL);
    _cmsRegisterTagTypePlugin(ContextID, NULL);
    _cmsRegisterTagPlugin(ContextID, NULL);
    _cmsRegisterFormattersPlugin(ContextID, NULL);
    _cmsRegisterRenderingIntentPlugin(ContextID, NULL);
    _cmsRegisterParametricCurvesPlugin(ContextID, NULL);
    _cmsRegisterMultiProcessElementPlugin(ContextID, NULL);
    _cmsRegisterOptimizationPlugin(ContextID, NULL);
    _cmsRegisterTransformPlugin(ContextID, NULL);    
    _cmsRegisterMutexPlugin(ContextID, NULL);
}


// Returns the memory manager plug-in, if any, from the Plug-in bundle
static
cmsPluginMemHandler* _cmsFindMemoryPlugin(void* PluginBundle)
{
    cmsPluginBase* Plugin;

    for (Plugin = (cmsPluginBase*) PluginBundle;
        Plugin != NULL;
        Plugin = Plugin -> Next) {

            if (Plugin -> Magic == cmsPluginMagicNumber && 
                Plugin -> ExpectedVersion <= LCMS_VERSION && 
                Plugin -> Type == cmsPluginMemHandlerSig) {

                    // Found!
                    return (cmsPluginMemHandler*) Plugin;  
            }
    }

    // Nope, revert to defaults 
    return NULL;
}


// Creates a new context with optional associated plug-ins. Caller may also specify an optional pointer to user-defined 
// data that will be forwarded to plug-ins and logger.
cmsContext CMSEXPORT cmsCreateContext(void* Plugin, void* UserData)
{
    struct _cmsContext_struct* ctx;
    struct _cmsContext_struct  fakeContext;
        
    // See the comments regarding locking in lcms2_internal.h
    // for an explanation of why we need the following code.
#ifdef CMS_IS_WINDOWS_
#ifndef CMS_RELY_ON_WINDOWS_STATIC_MUTEX_INIT
    {
        static HANDLE _cmsWindowsInitMutex = NULL;
        static volatile HANDLE* mutex = &_cmsWindowsInitMutex;

        if (*mutex == NULL)
        {
            HANDLE p = CreateMutex(NULL, FALSE, NULL);
            if (p && InterlockedCompareExchangePointer((void **)mutex, (void*)p, NULL) != NULL)
                CloseHandle(p);
        }
        if (*mutex == NULL || WaitForSingleObject(*mutex, INFINITE) == WAIT_FAILED)
            return NULL;
        if (((void **)&_cmsContextPoolHeadMutex)[0] == NULL)
            InitializeCriticalSection(&_cmsContextPoolHeadMutex);
        if (*mutex == NULL || !ReleaseMutex(*mutex))
            return NULL;
    }
#endif
#endif

    _cmsInstallAllocFunctions(_cmsFindMemoryPlugin(Plugin), &fakeContext.DefaultMemoryManager);
    
    fakeContext.chunks[UserPtr]     = UserData;
    fakeContext.chunks[MemPlugin]   = &fakeContext.DefaultMemoryManager;

    // Create the context structure.
    ctx = (struct _cmsContext_struct*) _cmsMalloc(&fakeContext, sizeof(struct _cmsContext_struct));
    if (ctx == NULL)   
        return NULL;     // Something very wrong happened!

    // Init the structure and the memory manager
    memset(ctx, 0, sizeof(struct _cmsContext_struct));

    // Keep memory manager
    memcpy(&ctx->DefaultMemoryManager, &fakeContext.DefaultMemoryManager, sizeof(_cmsMemPluginChunk)); 
   
    // Maintain the linked list (with proper locking)
    _cmsEnterCriticalSectionPrimitive(&_cmsContextPoolHeadMutex);
       ctx ->Next = _cmsContextPoolHead;
       _cmsContextPoolHead = ctx;
    _cmsLeaveCriticalSectionPrimitive(&_cmsContextPoolHeadMutex);

    ctx ->chunks[UserPtr]     = UserData;
    ctx ->chunks[MemPlugin]   = &ctx->DefaultMemoryManager;
   
    // Now we can allocate the pool by using default memory manager
    ctx ->MemPool = _cmsCreateSubAlloc(ctx, 22 * sizeof(void*));  // default size about 22 pointers
    if (ctx ->MemPool == NULL) {

         cmsDeleteContext(ctx);
        return NULL;
    }

    _cmsAllocLogErrorChunk(ctx, NULL);
    _cmsAllocAlarmCodesChunk(ctx, NULL);
    _cmsAllocAdaptationStateChunk(ctx, NULL);
    _cmsAllocMemPluginChunk(ctx, NULL);
    _cmsAllocInterpPluginChunk(ctx, NULL);
    _cmsAllocCurvesPluginChunk(ctx, NULL);
    _cmsAllocFormattersPluginChunk(ctx, NULL);
    _cmsAllocTagTypePluginChunk(ctx, NULL);
    _cmsAllocMPETypePluginChunk(ctx, NULL);
    _cmsAllocTagPluginChunk(ctx, NULL);
    _cmsAllocIntentsPluginChunk(ctx, NULL);
    _cmsAllocOptimizationPluginChunk(ctx, NULL);
    _cmsAllocTransformPluginChunk(ctx, NULL);
    _cmsAllocMutexPluginChunk(ctx, NULL);

    // Setup the plug-ins
    if (!cmsPluginTHR(ctx, Plugin)) {
    
        cmsDeleteContext(ctx);
        return NULL;
    }

    return (cmsContext) ctx;  
}

// Duplicates a context with all associated plug-ins. 
// Caller may specify an optional pointer to user-defined 
// data that will be forwarded to plug-ins and logger. 
cmsContext CMSEXPORT cmsDupContext(cmsContext ContextID, void* NewUserData)
{
    int i;
    struct _cmsContext_struct* ctx;
    const struct _cmsContext_struct* src = _cmsGetContext(ContextID);

    void* userData = (NewUserData != NULL) ? NewUserData : src -> chunks[UserPtr];
    
    
    ctx = (struct _cmsContext_struct*) _cmsMalloc(ContextID, sizeof(struct _cmsContext_struct));
    if (ctx == NULL)   
        return NULL;     // Something very wrong happened

    // Setup default memory allocators
    memcpy(&ctx->DefaultMemoryManager, &src->DefaultMemoryManager, sizeof(ctx->DefaultMemoryManager));

    // Maintain the linked list
    _cmsEnterCriticalSectionPrimitive(&_cmsContextPoolHeadMutex);
       ctx ->Next = _cmsContextPoolHead;
       _cmsContextPoolHead = ctx;
    _cmsLeaveCriticalSectionPrimitive(&_cmsContextPoolHeadMutex);

    ctx ->chunks[UserPtr]    = userData;
    ctx ->chunks[MemPlugin]  = &ctx->DefaultMemoryManager;

    ctx ->MemPool = _cmsCreateSubAlloc(ctx, 22 * sizeof(void*));
    if (ctx ->MemPool == NULL) {

         cmsDeleteContext(ctx);
        return NULL;
    }

    // Allocate all required chunks.
    _cmsAllocLogErrorChunk(ctx, src);
    _cmsAllocAlarmCodesChunk(ctx, src);
    _cmsAllocAdaptationStateChunk(ctx, src);
    _cmsAllocMemPluginChunk(ctx, src);
    _cmsAllocInterpPluginChunk(ctx, src);
    _cmsAllocCurvesPluginChunk(ctx, src);
    _cmsAllocFormattersPluginChunk(ctx, src);
    _cmsAllocTagTypePluginChunk(ctx, src);
    _cmsAllocMPETypePluginChunk(ctx, src);
    _cmsAllocTagPluginChunk(ctx, src);
    _cmsAllocIntentsPluginChunk(ctx, src);
    _cmsAllocOptimizationPluginChunk(ctx, src);
    _cmsAllocTransformPluginChunk(ctx, src);
    _cmsAllocMutexPluginChunk(ctx, src);

    // Make sure no one failed
    for (i=Logger; i < MemoryClientMax; i++) {

        if (src ->chunks[i] == NULL) {
            cmsDeleteContext((cmsContext) ctx);
            return NULL;
        }
    }

    return (cmsContext) ctx;
}


/*
static
struct _cmsContext_struct* FindPrev(struct _cmsContext_struct* id)
{
    struct _cmsContext_struct* prev;

    // Search for previous
    for (prev = _cmsContextPoolHead; 
             prev != NULL;
             prev = prev ->Next)
    {
        if (prev ->Next == id)
            return prev;
    }

    return NULL;  // List is empty or only one element!
}
*/

// Frees any resources associated with the given context, 
// and destroys the context placeholder. 
// The ContextID can no longer be used in any THR operation.  
void CMSEXPORT cmsDeleteContext(cmsContext ContextID)
{
    if (ContextID != NULL) {

        struct _cmsContext_struct* ctx = (struct _cmsContext_struct*) ContextID;              
        struct _cmsContext_struct  fakeContext;  
        struct _cmsContext_struct* prev;

        memcpy(&fakeContext.DefaultMemoryManager, &ctx->DefaultMemoryManager, sizeof(ctx->DefaultMemoryManager));

        fakeContext.chunks[UserPtr]     = ctx ->chunks[UserPtr];
        fakeContext.chunks[MemPlugin]   = &fakeContext.DefaultMemoryManager;

        // Get rid of plugins
        cmsUnregisterPluginsTHR(ContextID); 

        // Since all memory is allocated in the private pool, all what we need to do is destroy the pool
        if (ctx -> MemPool != NULL)
              _cmsSubAllocDestroy(ctx ->MemPool);
        ctx -> MemPool = NULL;

        // Maintain list
        _cmsEnterCriticalSectionPrimitive(&_cmsContextPoolHeadMutex);
        if (_cmsContextPoolHead == ctx) { 

            _cmsContextPoolHead = ctx->Next;
        }
        else {

            // Search for previous
            for (prev = _cmsContextPoolHead; 
                 prev != NULL;
                 prev = prev ->Next)
            {
                if (prev -> Next == ctx) {
                    prev -> Next = ctx ->Next;
                    break;
                }
            }
        }
        _cmsLeaveCriticalSectionPrimitive(&_cmsContextPoolHeadMutex);

        // free the memory block itself
        _cmsFree(&fakeContext, ctx);
    }
}

// Returns the user data associated to the given ContextID, or NULL if no user data was attached on context creation
void* CMSEXPORT cmsGetContextUserData(cmsContext ContextID)
{
    return _cmsContextGetClientChunk(ContextID, UserPtr);
}


