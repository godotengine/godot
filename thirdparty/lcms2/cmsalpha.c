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

// Alpha copy ------------------------------------------------------------------------------------------------------------------

// Floor to byte, taking care of saturation
cmsINLINE cmsUInt8Number _cmsQuickSaturateByte(cmsFloat64Number d)
{
       d += 0.5;
       if (d <= 0) return 0;
       if (d >= 255.0) return 255;

       return (cmsUInt8Number) _cmsQuickFloorWord(d);
}


// Return the size in bytes of a given formatter
static
cmsUInt32Number trueBytesSize(cmsUInt32Number Format)
{
    cmsUInt32Number fmt_bytes = T_BYTES(Format);

    // For double, the T_BYTES field returns zero
    if (fmt_bytes == 0)
        return sizeof(double);

    // Otherwise, it is already correct for all formats
    return fmt_bytes;
}


// Several format converters

typedef void(*cmsFormatterAlphaFn)(void* dst, const void* src);


// From 8

static
void copy8(void* dst, const void* src)
{
       memmove(dst, src, 1);
}

static
void from8to16(void* dst, const void* src)
{
       cmsUInt8Number n = *(cmsUInt8Number*)src;
       *(cmsUInt16Number*) dst = FROM_8_TO_16(n);
}

static
void from8toFLT(void* dst, const void* src)
{
       *(cmsFloat32Number*)dst = (*(cmsUInt8Number*)src) / 255.0f;
}

static
void from8toDBL(void* dst, const void* src)
{
       *(cmsFloat64Number*)dst = (*(cmsUInt8Number*)src) / 255.0;
}

static
void from8toHLF(void* dst, const void* src)
{
#ifndef CMS_NO_HALF_SUPPORT
       cmsFloat32Number n = (*(cmsUInt8Number*)src) / 255.0f;
       *(cmsUInt16Number*)dst = _cmsFloat2Half(n);
#else
    cmsUNUSED_PARAMETER(dst);
    cmsUNUSED_PARAMETER(src);
#endif
}

// From 16

static
void from16to8(void* dst, const void* src)
{
       cmsUInt16Number n = *(cmsUInt16Number*)src;
       *(cmsUInt8Number*) dst = FROM_16_TO_8(n);
}

static
void copy16(void* dst, const void* src)
{
       memmove(dst, src, 2);
}

void from16toFLT(void* dst, const void* src)
{
       *(cmsFloat32Number*)dst = (*(cmsUInt16Number*)src) / 65535.0f;
}

void from16toDBL(void* dst, const void* src)
{
       *(cmsFloat64Number*)dst = (*(cmsUInt16Number*)src) / 65535.0f;
}

static
void from16toHLF(void* dst, const void* src)
{
#ifndef CMS_NO_HALF_SUPPORT
       cmsFloat32Number n = (*(cmsUInt16Number*)src) / 65535.0f;
       *(cmsUInt16Number*)dst = _cmsFloat2Half(n);
#else
    cmsUNUSED_PARAMETER(dst);
    cmsUNUSED_PARAMETER(src);
#endif
}

// From Float

static 
void fromFLTto8(void* dst, const void* src)
{
       cmsFloat32Number n = *(cmsFloat32Number*)src;   
       *(cmsUInt8Number*)dst = _cmsQuickSaturateByte(n * 255.0f);
}

static
void fromFLTto16(void* dst, const void* src)
{
       cmsFloat32Number n = *(cmsFloat32Number*)src;      
       *(cmsUInt16Number*)dst = _cmsQuickSaturateWord(n * 65535.0f);
}

static
void copy32(void* dst, const void* src)
{
       memmove(dst, src, sizeof(cmsFloat32Number));
}

static
void fromFLTtoDBL(void* dst, const void* src)
{
       cmsFloat32Number n = *(cmsFloat32Number*)src;
       *(cmsFloat64Number*)dst = (cmsFloat64Number)n;
}

static
void fromFLTtoHLF(void* dst, const void* src)
{
#ifndef CMS_NO_HALF_SUPPORT
       cmsFloat32Number n = *(cmsFloat32Number*)src;
       *(cmsUInt16Number*)dst = _cmsFloat2Half(n);
#else
    cmsUNUSED_PARAMETER(dst);
    cmsUNUSED_PARAMETER(src);
#endif
}


// From HALF

static
void fromHLFto8(void* dst, const void* src)
{
#ifndef CMS_NO_HALF_SUPPORT
       cmsFloat32Number n = _cmsHalf2Float(*(cmsUInt16Number*)src);
       *(cmsUInt8Number*)dst = _cmsQuickSaturateByte(n * 255.0f);
#else
    cmsUNUSED_PARAMETER(dst);
    cmsUNUSED_PARAMETER(src);
#endif

}

static
void fromHLFto16(void* dst, const void* src)
{
#ifndef CMS_NO_HALF_SUPPORT
       cmsFloat32Number n = _cmsHalf2Float(*(cmsUInt16Number*)src);
       *(cmsUInt16Number*)dst = _cmsQuickSaturateWord(n * 65535.0f);
#else
    cmsUNUSED_PARAMETER(dst);
    cmsUNUSED_PARAMETER(src);
#endif
}

static
void fromHLFtoFLT(void* dst, const void* src)
{
#ifndef CMS_NO_HALF_SUPPORT
       *(cmsFloat32Number*)dst = _cmsHalf2Float(*(cmsUInt16Number*)src);
#else
    cmsUNUSED_PARAMETER(dst);
    cmsUNUSED_PARAMETER(src);
#endif
}

static
void fromHLFtoDBL(void* dst, const void* src)
{
#ifndef CMS_NO_HALF_SUPPORT
       *(cmsFloat64Number*)dst = (cmsFloat64Number)_cmsHalf2Float(*(cmsUInt16Number*)src);
#else
    cmsUNUSED_PARAMETER(dst);
    cmsUNUSED_PARAMETER(src);
#endif
}

// From double
static
void fromDBLto8(void* dst, const void* src)
{
       cmsFloat64Number n = *(cmsFloat64Number*)src;
       *(cmsUInt8Number*)dst = _cmsQuickSaturateByte(n * 255.0);
}

static
void fromDBLto16(void* dst, const void* src)
{
       cmsFloat64Number n = *(cmsFloat64Number*)src;
       *(cmsUInt16Number*)dst = _cmsQuickSaturateWord(n * 65535.0f);
}

static
void fromDBLtoFLT(void* dst, const void* src)
{
       cmsFloat64Number n = *(cmsFloat64Number*)src;
       *(cmsFloat32Number*)dst = (cmsFloat32Number) n;
}

static
void fromDBLtoHLF(void* dst, const void* src)
{
#ifndef CMS_NO_HALF_SUPPORT
       cmsFloat32Number n = (cmsFloat32Number) *(cmsFloat64Number*)src;
       *(cmsUInt16Number*)dst = _cmsFloat2Half(n);
#else
    cmsUNUSED_PARAMETER(dst);
    cmsUNUSED_PARAMETER(src);
#endif
}

static
void copy64(void* dst, const void* src)
{
       memmove(dst, src, sizeof(cmsFloat64Number));
}


// Returns the position (x or y) of the formatter in the table of functions
static
int FormatterPos(cmsUInt32Number frm)
{
    cmsUInt32Number  b = T_BYTES(frm);

    if (b == 0 && T_FLOAT(frm))
        return 4; // DBL
#ifndef CMS_NO_HALF_SUPPORT
    if (b == 2 && T_FLOAT(frm))
        return 2; // HLF
#endif
    if (b == 4 && T_FLOAT(frm))
        return 3; // FLT
    if (b == 2 && !T_FLOAT(frm))
        return 1; // 16
    if (b == 1 && !T_FLOAT(frm))
        return 0; // 8

    return -1; // not recognized
}

// Obtains a alpha-to-alpha funmction formatter
static
cmsFormatterAlphaFn _cmsGetFormatterAlpha(cmsContext id, cmsUInt32Number in, cmsUInt32Number out)
{
static cmsFormatterAlphaFn FormattersAlpha[5][5] = {

       /* from 8 */  { copy8,      from8to16,   from8toHLF,   from8toFLT,   from8toDBL   },
       /* from 16*/  { from16to8,  copy16,      from16toHLF,  from16toFLT,  from16toDBL  },
       /* from HLF*/ { fromHLFto8, fromHLFto16, copy16,       fromHLFtoFLT, fromHLFtoDBL },
       /* from FLT*/ { fromFLTto8, fromFLTto16, fromFLTtoHLF, copy32,       fromFLTtoDBL },
       /* from DBL*/ { fromDBLto8, fromDBLto16, fromDBLtoHLF, fromDBLtoFLT, copy64 }};

        int in_n  = FormatterPos(in);
        int out_n = FormatterPos(out);

        if (in_n < 0 || out_n < 0 || in_n > 4 || out_n > 4) {

               cmsSignalError(id, cmsERROR_UNKNOWN_EXTENSION, "Unrecognized alpha channel width");
               return NULL;
        }

        return FormattersAlpha[in_n][out_n];
}



// This function computes the distance from each component to the next one in bytes. 
static
void ComputeIncrementsForChunky(cmsUInt32Number Format,                                 
                                cmsUInt32Number ComponentStartingOrder[], 
                                cmsUInt32Number ComponentPointerIncrements[])
{
       cmsUInt32Number channels[cmsMAXCHANNELS];
       cmsUInt32Number extra = T_EXTRA(Format);
       cmsUInt32Number nchannels = T_CHANNELS(Format);
       cmsUInt32Number total_chans = nchannels + extra;
       cmsUInt32Number i;
       cmsUInt32Number channelSize = trueBytesSize(Format);
       cmsUInt32Number pixelSize = channelSize * total_chans;
       
	   // Sanity check
	   if (total_chans <= 0 || total_chans >= cmsMAXCHANNELS)
		   return;

        memset(channels, 0, sizeof(channels));

       // Separation is independent of starting point and only depends on channel size
       for (i = 0; i < extra; i++)
              ComponentPointerIncrements[i] = pixelSize;

       // Handle do swap
       for (i = 0; i < total_chans; i++)
       {
              if (T_DOSWAP(Format)) {
                     channels[i] = total_chans - i - 1;
              }
              else {
                     channels[i] = i;
              }
       }

       // Handle swap first (ROL of positions), example CMYK -> KCMY | 0123 -> 3012
       if (T_SWAPFIRST(Format) && total_chans > 1) {
              
              cmsUInt32Number tmp = channels[0];
              for (i = 0; i < total_chans-1; i++)
                     channels[i] = channels[i + 1];

              channels[total_chans - 1] = tmp;
       }

       // Handle size
       if (channelSize > 1)
              for (i = 0; i < total_chans; i++) {
                     channels[i] *= channelSize;
              }

       for (i = 0; i < extra; i++)
              ComponentStartingOrder[i] = channels[i + nchannels];
}



//  On planar configurations, the distance is the stride added to any non-negative
static
void ComputeIncrementsForPlanar(cmsUInt32Number Format, 
                                cmsUInt32Number BytesPerPlane,
                                cmsUInt32Number ComponentStartingOrder[], 
                                cmsUInt32Number ComponentPointerIncrements[])
{
       cmsUInt32Number channels[cmsMAXCHANNELS];       
       cmsUInt32Number extra = T_EXTRA(Format);
       cmsUInt32Number nchannels = T_CHANNELS(Format);
       cmsUInt32Number total_chans = nchannels + extra;
       cmsUInt32Number i;
       cmsUInt32Number channelSize = trueBytesSize(Format);
      
       // Sanity check
       if (total_chans <= 0 || total_chans >= cmsMAXCHANNELS)
           return;

       memset(channels, 0, sizeof(channels));

       // Separation is independent of starting point and only depends on channel size
       for (i = 0; i < extra; i++)
              ComponentPointerIncrements[i] = channelSize;

       // Handle do swap
       for (i = 0; i < total_chans; i++)
       {
              if (T_DOSWAP(Format)) {
                     channels[i] = total_chans - i - 1;
              }
              else {
                     channels[i] = i;
              }
       }

       // Handle swap first (ROL of positions), example CMYK -> KCMY | 0123 -> 3012
       if (T_SWAPFIRST(Format) && total_chans > 0) {

              cmsUInt32Number tmp = channels[0];
              for (i = 0; i < total_chans - 1; i++)
                     channels[i] = channels[i + 1];

              channels[total_chans - 1] = tmp;
       }

       // Handle size
       for (i = 0; i < total_chans; i++) {
              channels[i] *= BytesPerPlane;
       }

       for (i = 0; i < extra; i++)
              ComponentStartingOrder[i] = channels[i + nchannels];
}



// Dispatcher por chunky and planar RGB
static
void  ComputeComponentIncrements(cmsUInt32Number Format,
                                 cmsUInt32Number BytesPerPlane,
                                 cmsUInt32Number ComponentStartingOrder[], 
                                 cmsUInt32Number ComponentPointerIncrements[])
{
       if (T_PLANAR(Format)) {

              ComputeIncrementsForPlanar(Format,  BytesPerPlane, ComponentStartingOrder, ComponentPointerIncrements);
       }
       else {
              ComputeIncrementsForChunky(Format,  ComponentStartingOrder, ComponentPointerIncrements);
       }

}



// Handles extra channels copying alpha if requested by the flags
void _cmsHandleExtraChannels(_cmsTRANSFORM* p, const void* in,
                                               void* out,
                                               cmsUInt32Number PixelsPerLine,
                                               cmsUInt32Number LineCount,
                                               const cmsStride* Stride)
{
    cmsUInt32Number i, j, k;
    cmsUInt32Number nExtra;
    cmsUInt32Number SourceStartingOrder[cmsMAXCHANNELS];
    cmsUInt32Number SourceIncrements[cmsMAXCHANNELS];
    cmsUInt32Number DestStartingOrder[cmsMAXCHANNELS];
    cmsUInt32Number DestIncrements[cmsMAXCHANNELS];

    cmsFormatterAlphaFn copyValueFn;

    // Make sure we need some copy
    if (!(p->dwOriginalFlags & cmsFLAGS_COPY_ALPHA))
        return;

    // Exit early if in-place color-management is occurring - no need to copy extra channels to themselves.
    if (p->InputFormat == p->OutputFormat && in == out)
        return;

    // Make sure we have same number of alpha channels. If not, just return as this should be checked at transform creation time.
    nExtra = T_EXTRA(p->InputFormat);
    if (nExtra != T_EXTRA(p->OutputFormat))
        return;

    // Anything to do?
    if (nExtra == 0)
        return;

    // Compute the increments 
    ComputeComponentIncrements(p->InputFormat, Stride->BytesPerPlaneIn, SourceStartingOrder, SourceIncrements);
    ComputeComponentIncrements(p->OutputFormat, Stride->BytesPerPlaneOut, DestStartingOrder, DestIncrements);

    // Check for conversions 8, 16, half, float, dbl
    copyValueFn = _cmsGetFormatterAlpha(p->ContextID, p->InputFormat, p->OutputFormat);

    if (nExtra == 1) { // Optimized routine for copying a single extra channel quickly

        cmsUInt8Number* SourcePtr;
        cmsUInt8Number* DestPtr;

        cmsUInt32Number SourceStrideIncrement = 0;
        cmsUInt32Number DestStrideIncrement = 0;

        // The loop itself
        for (i = 0; i < LineCount; i++) {

            // Prepare pointers for the loop
            SourcePtr = (cmsUInt8Number*)in + SourceStartingOrder[0] + SourceStrideIncrement;
            DestPtr = (cmsUInt8Number*)out + DestStartingOrder[0] + DestStrideIncrement;

            for (j = 0; j < PixelsPerLine; j++) {

                copyValueFn(DestPtr, SourcePtr);

                SourcePtr += SourceIncrements[0];
                DestPtr += DestIncrements[0];
            }

            SourceStrideIncrement += Stride->BytesPerLineIn;
            DestStrideIncrement += Stride->BytesPerLineOut;
        }

    }
    else { // General case with more than one extra channel

        cmsUInt8Number* SourcePtr[cmsMAXCHANNELS];
        cmsUInt8Number* DestPtr[cmsMAXCHANNELS];

        cmsUInt32Number SourceStrideIncrements[cmsMAXCHANNELS];
        cmsUInt32Number DestStrideIncrements[cmsMAXCHANNELS];

        memset(SourceStrideIncrements, 0, sizeof(SourceStrideIncrements));
        memset(DestStrideIncrements, 0, sizeof(DestStrideIncrements));

        // The loop itself       
        for (i = 0; i < LineCount; i++) {

            // Prepare pointers for the loop
            for (j = 0; j < nExtra; j++) {

                SourcePtr[j] = (cmsUInt8Number*)in + SourceStartingOrder[j] + SourceStrideIncrements[j];
                DestPtr[j] = (cmsUInt8Number*)out + DestStartingOrder[j] + DestStrideIncrements[j];
            }

            for (j = 0; j < PixelsPerLine; j++) {

                for (k = 0; k < nExtra; k++) {

                    copyValueFn(DestPtr[k], SourcePtr[k]);

                    SourcePtr[k] += SourceIncrements[k];
                    DestPtr[k] += DestIncrements[k];
                }
            }

            for (j = 0; j < nExtra; j++) {

                SourceStrideIncrements[j] += Stride->BytesPerLineIn;
                DestStrideIncrements[j] += Stride->BytesPerLineOut;
            }
        }
    }
}


