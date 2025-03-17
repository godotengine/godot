/*
 * Copyright (C)2009-2024 D. R. Commander.  All Rights Reserved.
 * Copyright (C)2021 Alex Richardson.  All Rights Reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * - Neither the name of the libjpeg-turbo Project nor the names of its
 *   contributors may be used to endorse or promote products derived from this
 *   software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS",
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

/* TurboJPEG/LJT:  this implements the TurboJPEG API using libjpeg or
   libjpeg-turbo */

#include <ctype.h>
#include <limits.h>
#if !defined(_MSC_VER) || _MSC_VER > 1600
#include <stdint.h>
#endif
#include <jinclude.h>
#define JPEG_INTERNALS
#include <jpeglib.h>
#include <jerror.h>
#include <setjmp.h>
#include <errno.h>
#include "./turbojpeg.h"
#include "./tjutil.h"
#include "transupp.h"
#include "./jpegapicomp.h"
#include "./cdjpeg.h"

extern void jpeg_mem_dest_tj(j_compress_ptr, unsigned char **, size_t *,
                             boolean);
extern void jpeg_mem_src_tj(j_decompress_ptr, const unsigned char *, size_t);

#define PAD(v, p)  ((v + (p) - 1) & (~((p) - 1)))
#define IS_POW2(x)  (((x) & (x - 1)) == 0)


/* Error handling (based on example in example.c) */

static THREAD_LOCAL char errStr[JMSG_LENGTH_MAX] = "No error";

struct my_error_mgr {
  struct jpeg_error_mgr pub;
  jmp_buf setjmp_buffer;
  void (*emit_message) (j_common_ptr, int);
  boolean warning, stopOnWarning;
};
typedef struct my_error_mgr *my_error_ptr;

#define JMESSAGE(code, string)  string,
static const char *turbojpeg_message_table[] = {
#include "cderror.h"
  NULL
};

static void my_error_exit(j_common_ptr cinfo)
{
  my_error_ptr myerr = (my_error_ptr)cinfo->err;

  (*cinfo->err->output_message) (cinfo);
  longjmp(myerr->setjmp_buffer, 1);
}

/* Based on output_message() in jerror.c */

static void my_output_message(j_common_ptr cinfo)
{
  (*cinfo->err->format_message) (cinfo, errStr);
}

static void my_emit_message(j_common_ptr cinfo, int msg_level)
{
  my_error_ptr myerr = (my_error_ptr)cinfo->err;

  myerr->emit_message(cinfo, msg_level);
  if (msg_level < 0) {
    myerr->warning = TRUE;
    if (myerr->stopOnWarning) longjmp(myerr->setjmp_buffer, 1);
  }
}


/********************** Global structures, macros, etc. **********************/

enum { COMPRESS = 1, DECOMPRESS = 2 };

typedef struct _tjinstance {
  struct jpeg_compress_struct cinfo;
  struct jpeg_decompress_struct dinfo;
  struct my_error_mgr jerr;
  int init;
  char errStr[JMSG_LENGTH_MAX];
  boolean isInstanceError;
  /* Parameters */
  boolean bottomUp;
  boolean noRealloc;
  int quality;
  int subsamp;
  int jpegWidth;
  int jpegHeight;
  int precision;
  int colorspace;
  boolean fastUpsample;
  boolean fastDCT;
  boolean optimize;
  boolean progressive;
  int scanLimit;
  boolean arithmetic;
  boolean lossless;
  int losslessPSV;
  int losslessPt;
  int restartIntervalBlocks;
  int restartIntervalRows;
  int xDensity;
  int yDensity;
  int densityUnits;
  tjscalingfactor scalingFactor;
  tjregion croppingRegion;
  int maxMemory;
  int maxPixels;
  int saveMarkers;
  unsigned char *iccBuf, *tempICCBuf;
  size_t iccSize, tempICCSize;
} tjinstance;

static tjhandle _tjInitCompress(tjinstance *this);
static tjhandle _tjInitDecompress(tjinstance *this);

struct my_progress_mgr {
  struct jpeg_progress_mgr pub;
  tjinstance *this;
};
typedef struct my_progress_mgr *my_progress_ptr;

static void my_progress_monitor(j_common_ptr dinfo)
{
  my_error_ptr myerr = (my_error_ptr)dinfo->err;
  my_progress_ptr myprog = (my_progress_ptr)dinfo->progress;

  if (dinfo->is_decompressor) {
    int scan_no = ((j_decompress_ptr)dinfo)->input_scan_number;

    if (scan_no > myprog->this->scanLimit) {
      SNPRINTF(myprog->this->errStr, JMSG_LENGTH_MAX,
               "Progressive JPEG image has more than %d scans",
               myprog->this->scanLimit);
      SNPRINTF(errStr, JMSG_LENGTH_MAX,
               "Progressive JPEG image has more than %d scans",
               myprog->this->scanLimit);
      myprog->this->isInstanceError = TRUE;
      myerr->warning = FALSE;
      longjmp(myerr->setjmp_buffer, 1);
    }
  }
}

static const JXFORM_CODE xformtypes[TJ_NUMXOP] = {
  JXFORM_NONE, JXFORM_FLIP_H, JXFORM_FLIP_V, JXFORM_TRANSPOSE,
  JXFORM_TRANSVERSE, JXFORM_ROT_90, JXFORM_ROT_180, JXFORM_ROT_270
};

#define NUMSF  16
static const tjscalingfactor sf[NUMSF] = {
  { 2, 1 },
  { 15, 8 },
  { 7, 4 },
  { 13, 8 },
  { 3, 2 },
  { 11, 8 },
  { 5, 4 },
  { 9, 8 },
  { 1, 1 },
  { 7, 8 },
  { 3, 4 },
  { 5, 8 },
  { 1, 2 },
  { 3, 8 },
  { 1, 4 },
  { 1, 8 }
};

static J_COLOR_SPACE pf2cs[TJ_NUMPF] = {
  JCS_EXT_RGB, JCS_EXT_BGR, JCS_EXT_RGBX, JCS_EXT_BGRX, JCS_EXT_XBGR,
  JCS_EXT_XRGB, JCS_GRAYSCALE, JCS_EXT_RGBA, JCS_EXT_BGRA, JCS_EXT_ABGR,
  JCS_EXT_ARGB, JCS_CMYK
};

static int cs2pf[JPEG_NUMCS] = {
  TJPF_UNKNOWN, TJPF_GRAY,
#if RGB_RED == 0 && RGB_GREEN == 1 && RGB_BLUE == 2 && RGB_PIXELSIZE == 3
  TJPF_RGB,
#elif RGB_RED == 2 && RGB_GREEN == 1 && RGB_BLUE == 0 && RGB_PIXELSIZE == 3
  TJPF_BGR,
#elif RGB_RED == 0 && RGB_GREEN == 1 && RGB_BLUE == 2 && RGB_PIXELSIZE == 4
  TJPF_RGBX,
#elif RGB_RED == 2 && RGB_GREEN == 1 && RGB_BLUE == 0 && RGB_PIXELSIZE == 4
  TJPF_BGRX,
#elif RGB_RED == 3 && RGB_GREEN == 2 && RGB_BLUE == 1 && RGB_PIXELSIZE == 4
  TJPF_XBGR,
#elif RGB_RED == 1 && RGB_GREEN == 2 && RGB_BLUE == 3 && RGB_PIXELSIZE == 4
  TJPF_XRGB,
#endif
  TJPF_UNKNOWN, TJPF_CMYK, TJPF_UNKNOWN, TJPF_RGB, TJPF_RGBX, TJPF_BGR,
  TJPF_BGRX, TJPF_XBGR, TJPF_XRGB, TJPF_RGBA, TJPF_BGRA, TJPF_ABGR, TJPF_ARGB,
  TJPF_UNKNOWN
};

#define THROWG(m, rv) { \
  SNPRINTF(errStr, JMSG_LENGTH_MAX, "%s(): %s", FUNCTION_NAME, m); \
  retval = rv;  goto bailout; \
}
#ifdef _MSC_VER
#define THROW_UNIX(m) { \
  char strerrorBuf[80] = { 0 }; \
  strerror_s(strerrorBuf, 80, errno); \
  SNPRINTF(this->errStr, JMSG_LENGTH_MAX, "%s(): %s\n%s", FUNCTION_NAME, m, \
           strerrorBuf); \
  this->isInstanceError = TRUE; \
  SNPRINTF(errStr, JMSG_LENGTH_MAX, "%s(): %s\n%s", FUNCTION_NAME, m, \
           strerrorBuf); \
  retval = -1;  goto bailout; \
}
#else
#define THROW_UNIX(m) { \
  SNPRINTF(this->errStr, JMSG_LENGTH_MAX, "%s(): %s\n%s", FUNCTION_NAME, m, \
           strerror(errno)); \
  this->isInstanceError = TRUE; \
  SNPRINTF(errStr, JMSG_LENGTH_MAX, "%s(): %s\n%s", FUNCTION_NAME, m, \
           strerror(errno)); \
  retval = -1;  goto bailout; \
}
#endif
#define THROWRV(m, rv) { \
  SNPRINTF(this->errStr, JMSG_LENGTH_MAX, "%s(): %s", FUNCTION_NAME, m); \
  this->isInstanceError = TRUE;  THROWG(m, rv) \
}
#define THROW(m)  THROWRV(m, -1)
#define THROWI(format, val1, val2) { \
  SNPRINTF(this->errStr, JMSG_LENGTH_MAX, "%s(): " format, FUNCTION_NAME, \
           val1, val2); \
  this->isInstanceError = TRUE; \
  SNPRINTF(errStr, JMSG_LENGTH_MAX, "%s(): " format, FUNCTION_NAME, val1, \
           val2); \
  retval = -1;  goto bailout; \
}

#define GET_INSTANCE(handle) \
  tjinstance *this = (tjinstance *)handle; \
  j_compress_ptr cinfo = NULL; \
  j_decompress_ptr dinfo = NULL; \
  \
  if (!this) { \
    SNPRINTF(errStr, JMSG_LENGTH_MAX, "%s(): Invalid handle", FUNCTION_NAME); \
    return -1; \
  } \
  cinfo = &this->cinfo;  dinfo = &this->dinfo; \
  this->jerr.warning = FALSE; \
  this->isInstanceError = FALSE;

#define GET_CINSTANCE(handle) \
  tjinstance *this = (tjinstance *)handle; \
  j_compress_ptr cinfo = NULL; \
  \
  if (!this) { \
    SNPRINTF(errStr, JMSG_LENGTH_MAX, "%s(): Invalid handle", FUNCTION_NAME); \
    return -1; \
  } \
  cinfo = &this->cinfo; \
  this->jerr.warning = FALSE; \
  this->isInstanceError = FALSE;

#define GET_DINSTANCE(handle) \
  tjinstance *this = (tjinstance *)handle; \
  j_decompress_ptr dinfo = NULL; \
  \
  if (!this) { \
    SNPRINTF(errStr, JMSG_LENGTH_MAX, "%s(): Invalid handle", FUNCTION_NAME); \
    return -1; \
  } \
  dinfo = &this->dinfo; \
  this->jerr.warning = FALSE; \
  this->isInstanceError = FALSE;

#define GET_TJINSTANCE(handle, errorReturn) \
  tjinstance *this = (tjinstance *)handle; \
  \
  if (!this) { \
    SNPRINTF(errStr, JMSG_LENGTH_MAX, "%s(): Invalid handle", FUNCTION_NAME); \
    return errorReturn; \
  } \
  this->jerr.warning = FALSE; \
  this->isInstanceError = FALSE;

static int getPixelFormat(int pixelSize, int flags)
{
  if (pixelSize == 1) return TJPF_GRAY;
  if (pixelSize == 3) {
    if (flags & TJ_BGR) return TJPF_BGR;
    else return TJPF_RGB;
  }
  if (pixelSize == 4) {
    if (flags & TJ_ALPHAFIRST) {
      if (flags & TJ_BGR) return TJPF_XBGR;
      else return TJPF_XRGB;
    } else {
      if (flags & TJ_BGR) return TJPF_BGRX;
      else return TJPF_RGBX;
    }
  }
  return -1;
}

static void setCompDefaults(tjinstance *this, int pixelFormat)
{
  int subsamp = this->subsamp;

  this->cinfo.in_color_space = pf2cs[pixelFormat];
  this->cinfo.input_components = tjPixelSize[pixelFormat];
  jpeg_set_defaults(&this->cinfo);

  this->cinfo.restart_interval = this->restartIntervalBlocks;
  this->cinfo.restart_in_rows = this->restartIntervalRows;
  this->cinfo.X_density = (UINT16)this->xDensity;
  this->cinfo.Y_density = (UINT16)this->yDensity;
  this->cinfo.density_unit = (UINT8)this->densityUnits;
  this->cinfo.mem->max_memory_to_use = (long)this->maxMemory * 1048576L;

  if (this->lossless) {
#ifdef C_LOSSLESS_SUPPORTED
    jpeg_enable_lossless(&this->cinfo, this->losslessPSV, this->losslessPt);
#endif
    if (pixelFormat == TJPF_GRAY)
      subsamp = TJSAMP_GRAY;
    else if (subsamp != TJSAMP_GRAY)
      subsamp = TJSAMP_444;
    return;
  }

  jpeg_set_quality(&this->cinfo, this->quality, TRUE);
  this->cinfo.dct_method = this->fastDCT ? JDCT_FASTEST : JDCT_ISLOW;

  switch (this->colorspace) {
  case TJCS_RGB:
    jpeg_set_colorspace(&this->cinfo, JCS_RGB);  break;
  case TJCS_YCbCr:
    jpeg_set_colorspace(&this->cinfo, JCS_YCbCr);  break;
  case TJCS_GRAY:
    jpeg_set_colorspace(&this->cinfo, JCS_GRAYSCALE);  break;
  case TJCS_CMYK:
    jpeg_set_colorspace(&this->cinfo, JCS_CMYK);  break;
  case TJCS_YCCK:
    jpeg_set_colorspace(&this->cinfo, JCS_YCCK);  break;
  default:
    if (subsamp == TJSAMP_GRAY)
      jpeg_set_colorspace(&this->cinfo, JCS_GRAYSCALE);
    else if (pixelFormat == TJPF_CMYK)
      jpeg_set_colorspace(&this->cinfo, JCS_YCCK);
    else
      jpeg_set_colorspace(&this->cinfo, JCS_YCbCr);
  }

  if (this->cinfo.data_precision == 8)
    this->cinfo.optimize_coding = this->optimize;
#ifdef C_PROGRESSIVE_SUPPORTED
  if (this->progressive) jpeg_simple_progression(&this->cinfo);
#endif
  this->cinfo.arith_code = this->arithmetic;

  this->cinfo.comp_info[0].h_samp_factor = tjMCUWidth[subsamp] / 8;
  this->cinfo.comp_info[1].h_samp_factor = 1;
  this->cinfo.comp_info[2].h_samp_factor = 1;
  if (this->cinfo.num_components > 3)
    this->cinfo.comp_info[3].h_samp_factor = tjMCUWidth[subsamp] / 8;
  this->cinfo.comp_info[0].v_samp_factor = tjMCUHeight[subsamp] / 8;
  this->cinfo.comp_info[1].v_samp_factor = 1;
  this->cinfo.comp_info[2].v_samp_factor = 1;
  if (this->cinfo.num_components > 3)
    this->cinfo.comp_info[3].v_samp_factor = tjMCUHeight[subsamp] / 8;
}


static int getSubsamp(j_decompress_ptr dinfo)
{
  int retval = TJSAMP_UNKNOWN, i, k;

  /* The sampling factors actually have no meaning with grayscale JPEG files,
     and in fact it's possible to generate grayscale JPEGs with sampling
     factors > 1 (even though those sampling factors are ignored by the
     decompressor.)  Thus, we need to treat grayscale as a special case. */
  if (dinfo->num_components == 1 && dinfo->jpeg_color_space == JCS_GRAYSCALE)
    return TJSAMP_GRAY;

  for (i = 0; i < TJ_NUMSAMP; i++) {
    if (i == TJSAMP_GRAY) continue;

    if (dinfo->num_components == 3 ||
        ((dinfo->jpeg_color_space == JCS_YCCK ||
          dinfo->jpeg_color_space == JCS_CMYK) &&
         dinfo->num_components == 4)) {
      if (dinfo->comp_info[0].h_samp_factor == tjMCUWidth[i] / 8 &&
          dinfo->comp_info[0].v_samp_factor == tjMCUHeight[i] / 8) {
        int match = 0;

        for (k = 1; k < dinfo->num_components; k++) {
          int href = 1, vref = 1;

          if ((dinfo->jpeg_color_space == JCS_YCCK ||
               dinfo->jpeg_color_space == JCS_CMYK) && k == 3) {
            href = tjMCUWidth[i] / 8;  vref = tjMCUHeight[i] / 8;
          }
          if (dinfo->comp_info[k].h_samp_factor == href &&
              dinfo->comp_info[k].v_samp_factor == vref)
            match++;
        }
        if (match == dinfo->num_components - 1) {
          retval = i;  break;
        }
      }
      /* Handle 4:2:2 and 4:4:0 images whose sampling factors are specified
         in non-standard ways. */
      if (dinfo->comp_info[0].h_samp_factor == 2 &&
          dinfo->comp_info[0].v_samp_factor == 2 &&
          (i == TJSAMP_422 || i == TJSAMP_440)) {
        int match = 0;

        for (k = 1; k < dinfo->num_components; k++) {
          int href = tjMCUHeight[i] / 8, vref = tjMCUWidth[i] / 8;

          if ((dinfo->jpeg_color_space == JCS_YCCK ||
               dinfo->jpeg_color_space == JCS_CMYK) && k == 3) {
            href = vref = 2;
          }
          if (dinfo->comp_info[k].h_samp_factor == href &&
              dinfo->comp_info[k].v_samp_factor == vref)
            match++;
        }
        if (match == dinfo->num_components - 1) {
          retval = i;  break;
        }
      }
      /* Handle 4:4:4 images whose sampling factors are specified in
         non-standard ways. */
      if (dinfo->comp_info[0].h_samp_factor *
          dinfo->comp_info[0].v_samp_factor <=
          D_MAX_BLOCKS_IN_MCU / 3 && i == TJSAMP_444) {
        int match = 0;
        for (k = 1; k < dinfo->num_components; k++) {
          if (dinfo->comp_info[k].h_samp_factor ==
              dinfo->comp_info[0].h_samp_factor &&
              dinfo->comp_info[k].v_samp_factor ==
              dinfo->comp_info[0].v_samp_factor)
            match++;
          if (match == dinfo->num_components - 1) {
            retval = i;  break;
          }
        }
      }
    }
  }
  return retval;
}


static void setDecompParameters(tjinstance *this)
{
  this->subsamp = getSubsamp(&this->dinfo);
  this->jpegWidth = this->dinfo.image_width;
  this->jpegHeight = this->dinfo.image_height;
  this->precision = this->dinfo.data_precision;
  switch (this->dinfo.jpeg_color_space) {
  case JCS_GRAYSCALE:  this->colorspace = TJCS_GRAY;  break;
  case JCS_RGB:        this->colorspace = TJCS_RGB;  break;
  case JCS_YCbCr:      this->colorspace = TJCS_YCbCr;  break;
  case JCS_CMYK:       this->colorspace = TJCS_CMYK;  break;
  case JCS_YCCK:       this->colorspace = TJCS_YCCK;  break;
  default:             this->colorspace = -1;  break;
  }
  this->progressive = this->dinfo.progressive_mode;
  this->arithmetic = this->dinfo.arith_code;
  this->lossless = this->dinfo.master->lossless;
  this->losslessPSV = this->dinfo.Ss;
  this->losslessPt = this->dinfo.Al;
  this->xDensity = this->dinfo.X_density;
  this->yDensity = this->dinfo.Y_density;
  this->densityUnits = this->dinfo.density_unit;
}


static void processFlags(tjhandle handle, int flags, int operation)
{
  tjinstance *this = (tjinstance *)handle;

  this->bottomUp = !!(flags & TJFLAG_BOTTOMUP);

#ifndef NO_PUTENV
  if (flags & TJFLAG_FORCEMMX) PUTENV_S("JSIMD_FORCEMMX", "1");
  else if (flags & TJFLAG_FORCESSE) PUTENV_S("JSIMD_FORCESSE", "1");
  else if (flags & TJFLAG_FORCESSE2) PUTENV_S("JSIMD_FORCESSE2", "1");
#endif

  this->fastUpsample = !!(flags & TJFLAG_FASTUPSAMPLE);
  this->noRealloc = !!(flags & TJFLAG_NOREALLOC);

  if (operation == COMPRESS) {
    if (this->quality >= 96 || flags & TJFLAG_ACCURATEDCT)
      this->fastDCT = FALSE;
    else
      this->fastDCT = TRUE;
  } else
    this->fastDCT = !!(flags & TJFLAG_FASTDCT);

  this->jerr.stopOnWarning = !!(flags & TJFLAG_STOPONWARNING);
  this->progressive = !!(flags & TJFLAG_PROGRESSIVE);

  if (flags & TJFLAG_LIMITSCANS) this->scanLimit = 500;
}


/*************************** General API functions ***************************/

/* TurboJPEG 3.0+ */
DLLEXPORT tjhandle tj3Init(int initType)
{
  static const char FUNCTION_NAME[] = "tj3Init";
  tjinstance *this = NULL;
  tjhandle retval = NULL;

  if (initType < 0 || initType >= TJ_NUMINIT)
    THROWG("Invalid argument", NULL);

  if ((this = (tjinstance *)malloc(sizeof(tjinstance))) == NULL)
    THROWG("Memory allocation failure", NULL);
  memset(this, 0, sizeof(tjinstance));
  SNPRINTF(this->errStr, JMSG_LENGTH_MAX, "No error");

  this->quality = -1;
  this->subsamp = TJSAMP_UNKNOWN;
  this->jpegWidth = -1;
  this->jpegHeight = -1;
  this->precision = 8;
  this->colorspace = -1;
  this->losslessPSV = 1;
  this->xDensity = 1;
  this->yDensity = 1;
  this->scalingFactor = TJUNSCALED;
  this->saveMarkers = 2;

  switch (initType) {
  case TJINIT_COMPRESS:  return _tjInitCompress(this);
  case TJINIT_DECOMPRESS:  return _tjInitDecompress(this);
  case TJINIT_TRANSFORM:
    retval = _tjInitCompress(this);
    if (!retval) return NULL;
    retval = _tjInitDecompress(this);
    return retval;
  }

bailout:
  return retval;
}


/* TurboJPEG 3.0+ */
DLLEXPORT void tj3Destroy(tjhandle handle)
{
  tjinstance *this = (tjinstance *)handle;
  j_compress_ptr cinfo = NULL;
  j_decompress_ptr dinfo = NULL;

  if (!this) return;

  cinfo = &this->cinfo;  dinfo = &this->dinfo;
  this->jerr.warning = FALSE;
  this->isInstanceError = FALSE;

  if (setjmp(this->jerr.setjmp_buffer)) return;
  if (this->init & COMPRESS) jpeg_destroy_compress(cinfo);
  if (this->init & DECOMPRESS) jpeg_destroy_decompress(dinfo);
  free(this->iccBuf);
  free(this->tempICCBuf);
  free(this);
}

/* TurboJPEG 1.0+ */
DLLEXPORT int tjDestroy(tjhandle handle)
{
  static const char FUNCTION_NAME[] = "tjDestroy";
  int retval = 0;

  if (!handle) THROWG("Invalid handle", -1);

  SNPRINTF(errStr, JMSG_LENGTH_MAX, "No error");
  tj3Destroy(handle);
  if (strcmp(errStr, "No error")) retval = -1;

bailout:
  return retval;
}


/* TurboJPEG 3.0+ */
DLLEXPORT char *tj3GetErrorStr(tjhandle handle)
{
  tjinstance *this = (tjinstance *)handle;

  if (this && this->isInstanceError) {
    this->isInstanceError = FALSE;
    return this->errStr;
  } else
    return errStr;
}

/* TurboJPEG 2.0+ */
DLLEXPORT char *tjGetErrorStr2(tjhandle handle)
{
  return tj3GetErrorStr(handle);
}

/* TurboJPEG 1.0+ */
DLLEXPORT char *tjGetErrorStr(void)
{
  return errStr;
}


/* TurboJPEG 3.0+ */
DLLEXPORT int tj3GetErrorCode(tjhandle handle)
{
  tjinstance *this = (tjinstance *)handle;

  if (this && this->jerr.warning) return TJERR_WARNING;
  else return TJERR_FATAL;
}

/* TurboJPEG 2.0+ */
DLLEXPORT int tjGetErrorCode(tjhandle handle)
{
  return tj3GetErrorCode(handle);
}


#define SET_PARAM(field, minValue, maxValue) { \
  if (value < minValue || (maxValue > 0 && value > maxValue)) \
    THROW("Parameter value out of range"); \
  this->field = value; \
}

#define SET_BOOL_PARAM(field) { \
  if (value < 0 || value > 1) \
    THROW("Parameter value out of range"); \
  this->field = (boolean)value; \
}

/* TurboJPEG 3.0+ */
DLLEXPORT int tj3Set(tjhandle handle, int param, int value)
{
  static const char FUNCTION_NAME[] = "tj3Set";
  int retval = 0;

  GET_TJINSTANCE(handle, -1);

  switch (param) {
  case TJPARAM_STOPONWARNING:
    SET_BOOL_PARAM(jerr.stopOnWarning);
    break;
  case TJPARAM_BOTTOMUP:
    SET_BOOL_PARAM(bottomUp);
    break;
  case TJPARAM_NOREALLOC:
    if (!(this->init & COMPRESS))
      THROW("TJPARAM_NOREALLOC is not applicable to decompression instances.");
    SET_BOOL_PARAM(noRealloc);
    break;
  case TJPARAM_QUALITY:
    if (!(this->init & COMPRESS))
      THROW("TJPARAM_QUALITY is not applicable to decompression instances.");
    SET_PARAM(quality, 1, 100);
    break;
  case TJPARAM_SUBSAMP:
    SET_PARAM(subsamp, 0, TJ_NUMSAMP - 1);
    break;
  case TJPARAM_JPEGWIDTH:
    if (!(this->init & DECOMPRESS))
      THROW("TJPARAM_JPEGWIDTH is not applicable to compression instances.");
    THROW("TJPARAM_JPEGWIDTH is read-only in decompression instances.");
    break;
  case TJPARAM_JPEGHEIGHT:
    if (!(this->init & DECOMPRESS))
      THROW("TJPARAM_JPEGHEIGHT is not applicable to compression instances.");
    THROW("TJPARAM_JPEGHEIGHT is read-only in decompression instances.");
    break;
  case TJPARAM_PRECISION:
    SET_PARAM(precision, 2, 16);
    break;
  case TJPARAM_COLORSPACE:
    if (!(this->init & COMPRESS))
      THROW("TJPARAM_COLORSPACE is read-only in decompression instances.");
    SET_PARAM(colorspace, 0, TJ_NUMCS - 1);
    break;
  case TJPARAM_FASTUPSAMPLE:
    if (!(this->init & DECOMPRESS))
      THROW("TJPARAM_FASTUPSAMPLE is not applicable to compression instances.");
    SET_BOOL_PARAM(fastUpsample);
    break;
  case TJPARAM_FASTDCT:
    SET_BOOL_PARAM(fastDCT);
    break;
  case TJPARAM_OPTIMIZE:
    if (!(this->init & COMPRESS))
      THROW("TJPARAM_OPTIMIZE is not applicable to decompression instances.");
    SET_BOOL_PARAM(optimize);
    break;
  case TJPARAM_PROGRESSIVE:
    if (!(this->init & COMPRESS))
      THROW("TJPARAM_PROGRESSIVE is read-only in decompression instances.");
    SET_BOOL_PARAM(progressive);
    break;
  case TJPARAM_SCANLIMIT:
    if (!(this->init & DECOMPRESS))
      THROW("TJPARAM_SCANLIMIT is not applicable to compression instances.");
    SET_PARAM(scanLimit, 0, -1);
    break;
  case TJPARAM_ARITHMETIC:
    if (!(this->init & COMPRESS))
      THROW("TJPARAM_ARITHMETIC is read-only in decompression instances.");
    SET_BOOL_PARAM(arithmetic);
    break;
  case TJPARAM_LOSSLESS:
    if (!(this->init & COMPRESS))
      THROW("TJPARAM_LOSSLESS is read-only in decompression instances.");
    SET_BOOL_PARAM(lossless);
    break;
  case TJPARAM_LOSSLESSPSV:
    if (!(this->init & COMPRESS))
      THROW("TJPARAM_LOSSLESSPSV is read-only in decompression instances.");
    SET_PARAM(losslessPSV, 1, 7);
    break;
  case TJPARAM_LOSSLESSPT:
    if (!(this->init & COMPRESS))
      THROW("TJPARAM_LOSSLESSPT is read-only in decompression instances.");
    SET_PARAM(losslessPt, 0, 15);
    break;
  case TJPARAM_RESTARTBLOCKS:
    if (!(this->init & COMPRESS))
      THROW("TJPARAM_RESTARTBLOCKS is not applicable to decompression instances.");
    SET_PARAM(restartIntervalBlocks, 0, 65535);
    if (value != 0) this->restartIntervalRows = 0;
    break;
  case TJPARAM_RESTARTROWS:
    if (!(this->init & COMPRESS))
      THROW("TJPARAM_RESTARTROWS is not applicable to decompression instances.");
    SET_PARAM(restartIntervalRows, 0, 65535);
    if (value != 0) this->restartIntervalBlocks = 0;
    break;
  case TJPARAM_XDENSITY:
    if (!(this->init & COMPRESS))
      THROW("TJPARAM_XDENSITY is read-only in decompression instances.");
    SET_PARAM(xDensity, 1, 65535);
    break;
  case TJPARAM_YDENSITY:
    if (!(this->init & COMPRESS))
      THROW("TJPARAM_YDENSITY is read-only in decompression instances.");
    SET_PARAM(yDensity, 1, 65535);
    break;
  case TJPARAM_DENSITYUNITS:
    if (!(this->init & COMPRESS))
      THROW("TJPARAM_DENSITYUNITS is read-only in decompression instances.");
    SET_PARAM(densityUnits, 0, 2);
    break;
  case TJPARAM_MAXMEMORY:
    SET_PARAM(maxMemory, 0, (int)(min(LONG_MAX / 1048576L, (long)INT_MAX)));
    break;
  case TJPARAM_MAXPIXELS:
    SET_PARAM(maxPixels, 0, -1);
    break;
  case TJPARAM_SAVEMARKERS:
    if (!(this->init & DECOMPRESS))
      THROW("TJPARAM_SAVEMARKERS is not applicable to compression instances.");
    SET_PARAM(saveMarkers, 0, 4);
    break;
  default:
    THROW("Invalid parameter");
  }

bailout:
  return retval;
}


/* TurboJPEG 3.0+ */
DLLEXPORT int tj3Get(tjhandle handle, int param)
{
  tjinstance *this = (tjinstance *)handle;
  if (!this) return -1;

  switch (param) {
  case TJPARAM_STOPONWARNING:
    return this->jerr.stopOnWarning;
  case TJPARAM_BOTTOMUP:
    return this->bottomUp;
  case TJPARAM_NOREALLOC:
    return this->noRealloc;
  case TJPARAM_QUALITY:
    return this->quality;
  case TJPARAM_SUBSAMP:
    return this->subsamp;
  case TJPARAM_JPEGWIDTH:
    return this->jpegWidth;
  case TJPARAM_JPEGHEIGHT:
    return this->jpegHeight;
  case TJPARAM_PRECISION:
    return this->precision;
  case TJPARAM_COLORSPACE:
    return this->colorspace;
  case TJPARAM_FASTUPSAMPLE:
    return this->fastUpsample;
  case TJPARAM_FASTDCT:
    return this->fastDCT;
  case TJPARAM_OPTIMIZE:
    return this->optimize;
  case TJPARAM_PROGRESSIVE:
    return this->progressive;
  case TJPARAM_SCANLIMIT:
    return this->scanLimit;
  case TJPARAM_ARITHMETIC:
    return this->arithmetic;
  case TJPARAM_LOSSLESS:
    return this->lossless;
  case TJPARAM_LOSSLESSPSV:
    return this->losslessPSV;
  case TJPARAM_LOSSLESSPT:
    return this->losslessPt;
  case TJPARAM_RESTARTBLOCKS:
    return this->restartIntervalBlocks;
  case TJPARAM_RESTARTROWS:
    return this->restartIntervalRows;
  case TJPARAM_XDENSITY:
    return this->xDensity;
  case TJPARAM_YDENSITY:
    return this->yDensity;
  case TJPARAM_DENSITYUNITS:
    return this->densityUnits;
  case TJPARAM_MAXMEMORY:
    return this->maxMemory;
  case TJPARAM_MAXPIXELS:
    return this->maxPixels;
  case TJPARAM_SAVEMARKERS:
    return this->saveMarkers;
  }

  return -1;
}


/* These are exposed mainly because Windows can't malloc() and free() across
   DLL boundaries except when the CRT DLL is used, and we don't use the CRT DLL
   with turbojpeg.dll for compatibility reasons.  However, these functions
   can potentially be used for other purposes by different implementations. */

/* TurboJPEG 3.0+ */
DLLEXPORT void *tj3Alloc(size_t bytes)
{
  return MALLOC(bytes);
}

/* TurboJPEG 1.2+ */
DLLEXPORT unsigned char *tjAlloc(int bytes)
{
  return (unsigned char *)tj3Alloc((size_t)bytes);
}


/* TurboJPEG 3.0+ */
DLLEXPORT void tj3Free(void *buf)
{
  free(buf);
}

/* TurboJPEG 1.2+ */
DLLEXPORT void tjFree(unsigned char *buf)
{
  tj3Free(buf);
}


/* TurboJPEG 3.0+ */
DLLEXPORT size_t tj3JPEGBufSize(int width, int height, int jpegSubsamp)
{
  static const char FUNCTION_NAME[] = "tj3JPEGBufSize";
  unsigned long long retval = 0;
  int mcuw, mcuh, chromasf;

  if (width < 1 || height < 1 || jpegSubsamp < TJSAMP_UNKNOWN ||
      jpegSubsamp >= TJ_NUMSAMP)
    THROWG("Invalid argument", 0);

  if (jpegSubsamp == TJSAMP_UNKNOWN)
    jpegSubsamp = TJSAMP_444;

  /* This allows for rare corner cases in which a JPEG image can actually be
     larger than the uncompressed input (we wouldn't mention it if it hadn't
     happened before.) */
  mcuw = tjMCUWidth[jpegSubsamp];
  mcuh = tjMCUHeight[jpegSubsamp];
  chromasf = jpegSubsamp == TJSAMP_GRAY ? 0 : 4 * 64 / (mcuw * mcuh);
  retval = PAD(width, mcuw) * PAD(height, mcuh) * (2ULL + chromasf) + 2048ULL;
#if ULLONG_MAX > ULONG_MAX
  if (retval > (unsigned long long)((unsigned long)-1))
    THROWG("Image is too large", 0);
#endif

bailout:
  return (size_t)retval;
}

/* TurboJPEG 1.2+ */
DLLEXPORT unsigned long tjBufSize(int width, int height, int jpegSubsamp)
{
  static const char FUNCTION_NAME[] = "tjBufSize";
  size_t retval;

  if (jpegSubsamp < 0)
    THROWG("Invalid argument", 0);

  retval = tj3JPEGBufSize(width, height, jpegSubsamp);

bailout:
  return (retval == 0) ? (unsigned long)-1 : (unsigned long)retval;
}

/* TurboJPEG 1.0+ */
DLLEXPORT unsigned long TJBUFSIZE(int width, int height)
{
  static const char FUNCTION_NAME[] = "TJBUFSIZE";
  unsigned long long retval = 0;

  if (width < 1 || height < 1)
    THROWG("Invalid argument", (unsigned long)-1);

  /* This allows for rare corner cases in which a JPEG image can actually be
     larger than the uncompressed input (we wouldn't mention it if it hadn't
     happened before.) */
  retval = PAD(width, 16) * PAD(height, 16) * 6ULL + 2048ULL;
#if ULLONG_MAX > ULONG_MAX
  if (retval > (unsigned long long)((unsigned long)-1))
    THROWG("Image is too large", (unsigned long)-1);
#endif

bailout:
  return (unsigned long)retval;
}


/* TurboJPEG 3.0+ */
DLLEXPORT size_t tj3YUVBufSize(int width, int align, int height, int subsamp)
{
  static const char FUNCTION_NAME[] = "tj3YUVBufSize";
  unsigned long long retval = 0;
  int nc, i;

  if (align < 1 || !IS_POW2(align) || subsamp < 0 || subsamp >= TJ_NUMSAMP)
    THROWG("Invalid argument", 0);

  nc = (subsamp == TJSAMP_GRAY ? 1 : 3);
  for (i = 0; i < nc; i++) {
    int pw = tj3YUVPlaneWidth(i, width, subsamp);
    int stride = PAD(pw, align);
    int ph = tj3YUVPlaneHeight(i, height, subsamp);

    if (pw == 0 || ph == 0) return 0;
    else retval += (unsigned long long)stride * ph;
  }
#if ULLONG_MAX > ULONG_MAX
  if (retval > (unsigned long long)((unsigned long)-1))
    THROWG("Image is too large", 0);
#endif

bailout:
  return (size_t)retval;
}

/* TurboJPEG 1.4+ */
DLLEXPORT unsigned long tjBufSizeYUV2(int width, int align, int height,
                                      int subsamp)
{
  size_t retval = tj3YUVBufSize(width, align, height, subsamp);
  return (retval == 0) ? (unsigned long)-1 : (unsigned long)retval;
}

/* TurboJPEG 1.2+ */
DLLEXPORT unsigned long tjBufSizeYUV(int width, int height, int subsamp)
{
  return tjBufSizeYUV2(width, 4, height, subsamp);
}

/* TurboJPEG 1.1+ */
DLLEXPORT unsigned long TJBUFSIZEYUV(int width, int height, int subsamp)
{
  return tjBufSizeYUV(width, height, subsamp);
}


/* TurboJPEG 3.0+ */
DLLEXPORT size_t tj3YUVPlaneSize(int componentID, int width, int stride,
                                 int height, int subsamp)
{
  static const char FUNCTION_NAME[] = "tj3YUVPlaneSize";
  unsigned long long retval = 0;
  int pw, ph;

  if (width < 1 || height < 1 || subsamp < 0 || subsamp >= TJ_NUMSAMP)
    THROWG("Invalid argument", 0);

  pw = tj3YUVPlaneWidth(componentID, width, subsamp);
  ph = tj3YUVPlaneHeight(componentID, height, subsamp);
  if (pw == 0 || ph == 0) return 0;

  if (stride == 0) stride = pw;
  else stride = abs(stride);

  retval = (unsigned long long)stride * (ph - 1) + pw;
#if ULLONG_MAX > ULONG_MAX
  if (retval > (unsigned long long)((unsigned long)-1))
    THROWG("Image is too large", 0);
#endif

bailout:
  return (size_t)retval;
}

/* TurboJPEG 1.4+ */
DLLEXPORT unsigned long tjPlaneSizeYUV(int componentID, int width, int stride,
                                       int height, int subsamp)
{
  size_t retval = tj3YUVPlaneSize(componentID, width, stride, height, subsamp);
  return (retval == 0) ? -1 : (unsigned long)retval;
}


/* TurboJPEG 3.0+ */
DLLEXPORT int tj3YUVPlaneWidth(int componentID, int width, int subsamp)
{
  static const char FUNCTION_NAME[] = "tj3YUVPlaneWidth";
  unsigned long long pw, retval = 0;
  int nc;

  if (width < 1 || subsamp < 0 || subsamp >= TJ_NUMSAMP)
    THROWG("Invalid argument", 0);
  nc = (subsamp == TJSAMP_GRAY ? 1 : 3);
  if (componentID < 0 || componentID >= nc)
    THROWG("Invalid argument", 0);

  pw = PAD((unsigned long long)width, tjMCUWidth[subsamp] / 8);
  if (componentID == 0)
    retval = pw;
  else
    retval = pw * 8 / tjMCUWidth[subsamp];

  if (retval > (unsigned long long)INT_MAX)
    THROWG("Width is too large", 0);

bailout:
  return (int)retval;
}

/* TurboJPEG 1.4+ */
DLLEXPORT int tjPlaneWidth(int componentID, int width, int subsamp)
{
  int retval = tj3YUVPlaneWidth(componentID, width, subsamp);
  return (retval == 0) ? -1 : retval;
}


/* TurboJPEG 3.0+ */
DLLEXPORT int tj3YUVPlaneHeight(int componentID, int height, int subsamp)
{
  static const char FUNCTION_NAME[] = "tj3YUVPlaneHeight";
  unsigned long long ph, retval = 0;
  int nc;

  if (height < 1 || subsamp < 0 || subsamp >= TJ_NUMSAMP)
    THROWG("Invalid argument", 0);
  nc = (subsamp == TJSAMP_GRAY ? 1 : 3);
  if (componentID < 0 || componentID >= nc)
    THROWG("Invalid argument", 0);

  ph = PAD((unsigned long long)height, tjMCUHeight[subsamp] / 8);
  if (componentID == 0)
    retval = ph;
  else
    retval = ph * 8 / tjMCUHeight[subsamp];

  if (retval > (unsigned long long)INT_MAX)
    THROWG("Height is too large", 0);

bailout:
  return (int)retval;
}

/* TurboJPEG 1.4+ */
DLLEXPORT int tjPlaneHeight(int componentID, int height, int subsamp)
{
  int retval = tj3YUVPlaneHeight(componentID, height, subsamp);
  return (retval == 0) ? -1 : retval;
}


/******************************** Compressor *********************************/

static tjhandle _tjInitCompress(tjinstance *this)
{
  static unsigned char buffer[1];
  unsigned char *buf = buffer;
  size_t size = 1;

  /* This is also straight out of example.c */
  this->cinfo.err = jpeg_std_error(&this->jerr.pub);
  this->jerr.pub.error_exit = my_error_exit;
  this->jerr.pub.output_message = my_output_message;
  this->jerr.emit_message = this->jerr.pub.emit_message;
  this->jerr.pub.emit_message = my_emit_message;
  this->jerr.pub.addon_message_table = turbojpeg_message_table;
  this->jerr.pub.first_addon_message = JMSG_FIRSTADDONCODE;
  this->jerr.pub.last_addon_message = JMSG_LASTADDONCODE;

  if (setjmp(this->jerr.setjmp_buffer)) {
    /* If we get here, the JPEG code has signaled an error. */
    free(this);
    return NULL;
  }

  jpeg_create_compress(&this->cinfo);
  /* Make an initial call so it will create the destination manager */
  jpeg_mem_dest_tj(&this->cinfo, &buf, &size, 0);

  this->init |= COMPRESS;
  return (tjhandle)this;
}

/* TurboJPEG 1.0+ */
DLLEXPORT tjhandle tjInitCompress(void)
{
  return tj3Init(TJINIT_COMPRESS);
}


/* TurboJPEG 3.1+ */
DLLEXPORT int tj3SetICCProfile(tjhandle handle, unsigned char *iccBuf,
                               size_t iccSize)
{
  static const char FUNCTION_NAME[] = "tj3SetICCProfile";
  int retval = 0;

  GET_TJINSTANCE(handle, -1)
  if ((this->init & COMPRESS) == 0)
    THROW("Instance has not been initialized for compression");

  if (iccBuf == this->iccBuf && iccSize == this->iccSize)
    return 0;

  free(this->iccBuf);
  this->iccBuf = NULL;
  this->iccSize = 0;
  if (iccBuf && iccSize) {
    if ((this->iccBuf = (unsigned char *)malloc(iccSize)) == NULL)
      THROW("Memory allocation failure");
    memcpy(this->iccBuf, iccBuf, iccSize);
    this->iccSize = iccSize;
  }

bailout:
  return retval;
}


/* tj3Compress*() is implemented in turbojpeg-mp.c */
#define BITS_IN_JSAMPLE  8
#include "turbojpeg-mp.c"
#undef BITS_IN_JSAMPLE
#define BITS_IN_JSAMPLE  12
#include "turbojpeg-mp.c"
#undef BITS_IN_JSAMPLE
#define BITS_IN_JSAMPLE  16
#include "turbojpeg-mp.c"
#undef BITS_IN_JSAMPLE

/* TurboJPEG 1.2+ */
DLLEXPORT int tjCompress2(tjhandle handle, const unsigned char *srcBuf,
                          int width, int pitch, int height, int pixelFormat,
                          unsigned char **jpegBuf, unsigned long *jpegSize,
                          int jpegSubsamp, int jpegQual, int flags)
{
  static const char FUNCTION_NAME[] = "tjCompress2";
  int retval = 0;
  size_t size;

  GET_TJINSTANCE(handle, -1);

  if (jpegSize == NULL || jpegSubsamp < 0 || jpegSubsamp >= TJ_NUMSAMP ||
      jpegQual < 0 || jpegQual > 100)
    THROW("Invalid argument");

  this->quality = jpegQual;
  this->subsamp = jpegSubsamp;
  processFlags(handle, flags, COMPRESS);

  size = (size_t)(*jpegSize);
  if (this->noRealloc)
    size = tj3JPEGBufSize(width, height, this->subsamp);
  retval = tj3Compress8(handle, srcBuf, width, pitch, height, pixelFormat,
                        jpegBuf, &size);
  *jpegSize = (unsigned long)size;

bailout:
  return retval;
}

/* TurboJPEG 1.0+ */
DLLEXPORT int tjCompress(tjhandle handle, unsigned char *srcBuf, int width,
                         int pitch, int height, int pixelSize,
                         unsigned char *jpegBuf, unsigned long *jpegSize,
                         int jpegSubsamp, int jpegQual, int flags)
{
  int retval = 0;
  unsigned long size = jpegSize ? *jpegSize : 0;

  if (flags & TJ_YUV) {
    size = tjBufSizeYUV(width, height, jpegSubsamp);
    retval = tjEncodeYUV2(handle, srcBuf, width, pitch, height,
                          getPixelFormat(pixelSize, flags), jpegBuf,
                          jpegSubsamp, flags);
  } else {
    retval = tjCompress2(handle, srcBuf, width, pitch, height,
                         getPixelFormat(pixelSize, flags), &jpegBuf, &size,
                         jpegSubsamp, jpegQual, flags | TJFLAG_NOREALLOC);
  }
  *jpegSize = size;
  return retval;
}


/* TurboJPEG 3.0+ */
DLLEXPORT int tj3CompressFromYUVPlanes8(tjhandle handle,
                                        const unsigned char * const *srcPlanes,
                                        int width, const int *strides,
                                        int height, unsigned char **jpegBuf,
                                        size_t *jpegSize)
{
  static const char FUNCTION_NAME[] = "tj3CompressFromYUVPlanes8";
  int i, row, retval = 0;
  boolean alloc = TRUE;
  int pw[MAX_COMPONENTS], ph[MAX_COMPONENTS], iw[MAX_COMPONENTS],
    tmpbufsize = 0, usetmpbuf = 0, th[MAX_COMPONENTS];
  JSAMPLE *_tmpbuf = NULL, *ptr;
  JSAMPROW *inbuf[MAX_COMPONENTS], *tmpbuf[MAX_COMPONENTS];

  GET_CINSTANCE(handle)

  for (i = 0; i < MAX_COMPONENTS; i++) {
    tmpbuf[i] = NULL;  inbuf[i] = NULL;
  }

  if ((this->init & COMPRESS) == 0)
    THROW("Instance has not been initialized for compression");

  if (!srcPlanes || !srcPlanes[0] || width <= 0 || height <= 0 ||
      jpegBuf == NULL || jpegSize == NULL)
    THROW("Invalid argument");
  if (this->subsamp != TJSAMP_GRAY && (!srcPlanes[1] || !srcPlanes[2]))
    THROW("Invalid argument");

  if (this->quality == -1)
    THROW("TJPARAM_QUALITY must be specified");
  if (this->subsamp == TJSAMP_UNKNOWN)
    THROW("TJPARAM_SUBSAMP must be specified");

  if (setjmp(this->jerr.setjmp_buffer)) {
    /* If we get here, the JPEG code has signaled an error. */
    retval = -1;  goto bailout;
  }

  cinfo->image_width = width;
  cinfo->image_height = height;
  cinfo->data_precision = 8;

  if (this->noRealloc) alloc = FALSE;
  jpeg_mem_dest_tj(cinfo, jpegBuf, jpegSize, alloc);
  setCompDefaults(this, TJPF_RGB);
  cinfo->raw_data_in = TRUE;

  jpeg_start_compress(cinfo, TRUE);
  if (this->iccBuf != NULL && this->iccSize != 0)
    jpeg_write_icc_profile(cinfo, this->iccBuf, (unsigned int)this->iccSize);
  for (i = 0; i < cinfo->num_components; i++) {
    jpeg_component_info *compptr = &cinfo->comp_info[i];
    int ih;

    iw[i] = compptr->width_in_blocks * DCTSIZE;
    ih = compptr->height_in_blocks * DCTSIZE;
    pw[i] = PAD(cinfo->image_width, cinfo->max_h_samp_factor) *
            compptr->h_samp_factor / cinfo->max_h_samp_factor;
    ph[i] = PAD(cinfo->image_height, cinfo->max_v_samp_factor) *
            compptr->v_samp_factor / cinfo->max_v_samp_factor;
    if (iw[i] != pw[i] || ih != ph[i]) usetmpbuf = 1;
    th[i] = compptr->v_samp_factor * DCTSIZE;
    tmpbufsize += iw[i] * th[i];
    if ((inbuf[i] = (JSAMPROW *)malloc(sizeof(JSAMPROW) * ph[i])) == NULL)
      THROW("Memory allocation failure");
    ptr = (JSAMPLE *)srcPlanes[i];
    for (row = 0; row < ph[i]; row++) {
      inbuf[i][row] = ptr;
      ptr += (strides && strides[i] != 0) ? strides[i] : pw[i];
    }
  }
  if (usetmpbuf) {
    if ((_tmpbuf = (JSAMPLE *)malloc(sizeof(JSAMPLE) * tmpbufsize)) == NULL)
      THROW("Memory allocation failure");
    ptr = _tmpbuf;
    for (i = 0; i < cinfo->num_components; i++) {
      if ((tmpbuf[i] = (JSAMPROW *)malloc(sizeof(JSAMPROW) * th[i])) == NULL)
        THROW("Memory allocation failure");
      for (row = 0; row < th[i]; row++) {
        tmpbuf[i][row] = ptr;
        ptr += iw[i];
      }
    }
  }

  if (setjmp(this->jerr.setjmp_buffer)) {
    /* If we get here, the JPEG code has signaled an error. */
    retval = -1;  goto bailout;
  }

  for (row = 0; row < (int)cinfo->image_height;
       row += cinfo->max_v_samp_factor * DCTSIZE) {
    JSAMPARRAY yuvptr[MAX_COMPONENTS];
    int crow[MAX_COMPONENTS];

    for (i = 0; i < cinfo->num_components; i++) {
      jpeg_component_info *compptr = &cinfo->comp_info[i];

      crow[i] = row * compptr->v_samp_factor / cinfo->max_v_samp_factor;
      if (usetmpbuf) {
        int j, k;

        for (j = 0; j < MIN(th[i], ph[i] - crow[i]); j++) {
          memcpy(tmpbuf[i][j], inbuf[i][crow[i] + j], pw[i]);
          /* Duplicate last sample in row to fill out MCU */
          for (k = pw[i]; k < iw[i]; k++)
            tmpbuf[i][j][k] = tmpbuf[i][j][pw[i] - 1];
        }
        /* Duplicate last row to fill out MCU */
        for (j = ph[i] - crow[i]; j < th[i]; j++)
          memcpy(tmpbuf[i][j], tmpbuf[i][ph[i] - crow[i] - 1], iw[i]);
        yuvptr[i] = tmpbuf[i];
      } else
        yuvptr[i] = &inbuf[i][crow[i]];
    }
    jpeg_write_raw_data(cinfo, yuvptr, cinfo->max_v_samp_factor * DCTSIZE);
  }
  jpeg_finish_compress(cinfo);

bailout:
  if (cinfo->global_state > CSTATE_START && alloc)
    (*cinfo->dest->term_destination) (cinfo);
  if (cinfo->global_state > CSTATE_START || retval == -1)
    jpeg_abort_compress(cinfo);
  for (i = 0; i < MAX_COMPONENTS; i++) {
    free(tmpbuf[i]);
    free(inbuf[i]);
  }
  free(_tmpbuf);
  if (this->jerr.warning) retval = -1;
  return retval;
}

/* TurboJPEG 1.4+ */
DLLEXPORT int tjCompressFromYUVPlanes(tjhandle handle,
                                      const unsigned char **srcPlanes,
                                      int width, const int *strides,
                                      int height, int subsamp,
                                      unsigned char **jpegBuf,
                                      unsigned long *jpegSize, int jpegQual,
                                      int flags)
{
  static const char FUNCTION_NAME[] = "tjCompressFromYUVPlanes";
  int retval = 0;
  size_t size;

  GET_TJINSTANCE(handle, -1);

  if (subsamp < 0 || subsamp >= TJ_NUMSAMP || jpegSize == NULL ||
      jpegQual < 0 || jpegQual > 100)
    THROW("Invalid argument");

  this->quality = jpegQual;
  this->subsamp = subsamp;
  processFlags(handle, flags, COMPRESS);

  size = (size_t)(*jpegSize);
  if (this->noRealloc)
    size = tj3JPEGBufSize(width, height, this->subsamp);
  retval = tj3CompressFromYUVPlanes8(handle, srcPlanes, width, strides, height,
                                     jpegBuf, &size);
  *jpegSize = (unsigned long)size;

bailout:
  return retval;
}


/* TurboJPEG 3.0+ */
DLLEXPORT int tj3CompressFromYUV8(tjhandle handle,
                                  const unsigned char *srcBuf, int width,
                                  int align, int height,
                                  unsigned char **jpegBuf, size_t *jpegSize)
{
  static const char FUNCTION_NAME[] = "tj3CompressFromYUV8";
  const unsigned char *srcPlanes[3];
  int pw0, ph0, strides[3], retval = -1;

  GET_TJINSTANCE(handle, -1);

  if (srcBuf == NULL || width <= 0 || align < 1 || !IS_POW2(align) ||
      height <= 0)
    THROW("Invalid argument");

  if (this->subsamp == TJSAMP_UNKNOWN)
    THROW("TJPARAM_SUBSAMP must be specified");

  pw0 = tj3YUVPlaneWidth(0, width, this->subsamp);
  ph0 = tj3YUVPlaneHeight(0, height, this->subsamp);
  srcPlanes[0] = srcBuf;
  strides[0] = PAD(pw0, align);
  if (this->subsamp == TJSAMP_GRAY) {
    strides[1] = strides[2] = 0;
    srcPlanes[1] = srcPlanes[2] = NULL;
  } else {
    int pw1 = tjPlaneWidth(1, width, this->subsamp);
    int ph1 = tjPlaneHeight(1, height, this->subsamp);

    strides[1] = strides[2] = PAD(pw1, align);
    if ((unsigned long long)strides[0] * (unsigned long long)ph0 >
        (unsigned long long)INT_MAX ||
        (unsigned long long)strides[1] * (unsigned long long)ph1 >
        (unsigned long long)INT_MAX)
      THROW("Image or row alignment is too large");
    srcPlanes[1] = srcPlanes[0] + strides[0] * ph0;
    srcPlanes[2] = srcPlanes[1] + strides[1] * ph1;
  }

  return tj3CompressFromYUVPlanes8(handle, srcPlanes, width, strides, height,
                                   jpegBuf, jpegSize);

bailout:
  return retval;
}

/* TurboJPEG 1.4+ */
DLLEXPORT int tjCompressFromYUV(tjhandle handle, const unsigned char *srcBuf,
                                int width, int align, int height, int subsamp,
                                unsigned char **jpegBuf,
                                unsigned long *jpegSize, int jpegQual,
                                int flags)
{
  static const char FUNCTION_NAME[] = "tjCompressFromYUV";
  int retval = -1;
  size_t size;

  GET_TJINSTANCE(handle, -1);

  if (subsamp < 0 || subsamp >= TJ_NUMSAMP)
    THROW("Invalid argument");

  this->quality = jpegQual;
  this->subsamp = subsamp;
  processFlags(handle, flags, COMPRESS);

  size = (size_t)(*jpegSize);
  if (this->noRealloc)
    size = tj3JPEGBufSize(width, height, this->subsamp);
  retval = tj3CompressFromYUV8(handle, srcBuf, width, align, height, jpegBuf,
                               &size);
  *jpegSize = (unsigned long)size;

bailout:
  return retval;
}


/* TurboJPEG 3.0+ */
DLLEXPORT int tj3EncodeYUVPlanes8(tjhandle handle, const unsigned char *srcBuf,
                                  int width, int pitch, int height,
                                  int pixelFormat, unsigned char **dstPlanes,
                                  int *strides)
{
  static const char FUNCTION_NAME[] = "tj3EncodeYUVPlanes8";
  JSAMPROW *row_pointer = NULL;
  JSAMPLE *_tmpbuf[MAX_COMPONENTS], *_tmpbuf2[MAX_COMPONENTS];
  JSAMPROW *tmpbuf[MAX_COMPONENTS], *tmpbuf2[MAX_COMPONENTS];
  JSAMPROW *outbuf[MAX_COMPONENTS];
  int i, retval = 0, row, pw0, ph0, pw[MAX_COMPONENTS], ph[MAX_COMPONENTS];
  JSAMPLE *ptr;
  jpeg_component_info *compptr;

  GET_CINSTANCE(handle)

  for (i = 0; i < MAX_COMPONENTS; i++) {
    tmpbuf[i] = NULL;  _tmpbuf[i] = NULL;
    tmpbuf2[i] = NULL;  _tmpbuf2[i] = NULL;  outbuf[i] = NULL;
  }

  if ((this->init & COMPRESS) == 0)
    THROW("Instance has not been initialized for compression");

  if (srcBuf == NULL || width <= 0 || pitch < 0 || height <= 0 ||
      pixelFormat < 0 || pixelFormat >= TJ_NUMPF || !dstPlanes ||
      !dstPlanes[0])
    THROW("Invalid argument");
  if (this->subsamp != TJSAMP_GRAY && (!dstPlanes[1] || !dstPlanes[2]))
    THROW("Invalid argument");

  if (this->subsamp == TJSAMP_UNKNOWN)
    THROW("TJPARAM_SUBSAMP must be specified");
  if (pixelFormat == TJPF_CMYK)
    THROW("Cannot generate YUV images from packed-pixel CMYK images");

  if (pitch == 0) pitch = width * tjPixelSize[pixelFormat];

  if (setjmp(this->jerr.setjmp_buffer)) {
    /* If we get here, the JPEG code has signaled an error. */
    retval = -1;  goto bailout;
  }

  cinfo->image_width = width;
  cinfo->image_height = height;
  cinfo->data_precision = 8;

  setCompDefaults(this, pixelFormat);

  /* Execute only the parts of jpeg_start_compress() that we need.  If we
     were to call the whole jpeg_start_compress() function, then it would try
     to write the file headers, which could overflow the output buffer if the
     YUV image were very small. */
  if (cinfo->global_state != CSTATE_START)
    THROW("libjpeg API is in the wrong state");
  (*cinfo->err->reset_error_mgr) ((j_common_ptr)cinfo);
  jinit_c_master_control(cinfo, FALSE);
  jinit_color_converter(cinfo);
  jinit_downsampler(cinfo);
  (*cinfo->cconvert->start_pass) (cinfo);

  pw0 = PAD(width, cinfo->max_h_samp_factor);
  ph0 = PAD(height, cinfo->max_v_samp_factor);

  if ((row_pointer = (JSAMPROW *)malloc(sizeof(JSAMPROW) * ph0)) == NULL)
    THROW("Memory allocation failure");
  for (i = 0; i < height; i++) {
    if (this->bottomUp)
      row_pointer[i] = (JSAMPROW)&srcBuf[(height - i - 1) * (size_t)pitch];
    else
      row_pointer[i] = (JSAMPROW)&srcBuf[i * (size_t)pitch];
  }
  if (height < ph0)
    for (i = height; i < ph0; i++) row_pointer[i] = row_pointer[height - 1];

  for (i = 0; i < cinfo->num_components; i++) {
    compptr = &cinfo->comp_info[i];
    _tmpbuf[i] = (JSAMPLE *)MALLOC(
      PAD((compptr->width_in_blocks * cinfo->max_h_samp_factor * DCTSIZE) /
          compptr->h_samp_factor, 32) *
      cinfo->max_v_samp_factor + 32);
    if (!_tmpbuf[i])
      THROW("Memory allocation failure");
    tmpbuf[i] =
      (JSAMPROW *)malloc(sizeof(JSAMPROW) * cinfo->max_v_samp_factor);
    if (!tmpbuf[i])
      THROW("Memory allocation failure");
    for (row = 0; row < cinfo->max_v_samp_factor; row++) {
      unsigned char *_tmpbuf_aligned =
        (unsigned char *)PAD((JUINTPTR)_tmpbuf[i], 32);

      tmpbuf[i][row] = &_tmpbuf_aligned[
        PAD((compptr->width_in_blocks * cinfo->max_h_samp_factor * DCTSIZE) /
            compptr->h_samp_factor, 32) * row];
    }
    _tmpbuf2[i] =
      (JSAMPLE *)MALLOC(PAD(compptr->width_in_blocks * DCTSIZE, 32) *
                        compptr->v_samp_factor + 32);
    if (!_tmpbuf2[i])
      THROW("Memory allocation failure");
    tmpbuf2[i] = (JSAMPROW *)malloc(sizeof(JSAMPROW) * compptr->v_samp_factor);
    if (!tmpbuf2[i])
      THROW("Memory allocation failure");
    for (row = 0; row < compptr->v_samp_factor; row++) {
      unsigned char *_tmpbuf2_aligned =
        (unsigned char *)PAD((JUINTPTR)_tmpbuf2[i], 32);

      tmpbuf2[i][row] =
        &_tmpbuf2_aligned[PAD(compptr->width_in_blocks * DCTSIZE, 32) * row];
    }
    pw[i] = pw0 * compptr->h_samp_factor / cinfo->max_h_samp_factor;
    ph[i] = ph0 * compptr->v_samp_factor / cinfo->max_v_samp_factor;
    outbuf[i] = (JSAMPROW *)malloc(sizeof(JSAMPROW) * ph[i]);
    if (!outbuf[i])
      THROW("Memory allocation failure");
    ptr = dstPlanes[i];
    for (row = 0; row < ph[i]; row++) {
      outbuf[i][row] = ptr;
      ptr += (strides && strides[i] != 0) ? strides[i] : pw[i];
    }
  }

  if (setjmp(this->jerr.setjmp_buffer)) {
    /* If we get here, the JPEG code has signaled an error. */
    retval = -1;  goto bailout;
  }

  for (row = 0; row < ph0; row += cinfo->max_v_samp_factor) {
    (*cinfo->cconvert->color_convert) (cinfo, &row_pointer[row], tmpbuf, 0,
                                       cinfo->max_v_samp_factor);
    (cinfo->downsample->downsample) (cinfo, tmpbuf, 0, tmpbuf2, 0);
    for (i = 0, compptr = cinfo->comp_info; i < cinfo->num_components;
         i++, compptr++)
      jcopy_sample_rows(tmpbuf2[i], 0, outbuf[i],
        row * compptr->v_samp_factor / cinfo->max_v_samp_factor,
        compptr->v_samp_factor, pw[i]);
  }
  cinfo->next_scanline += height;
  jpeg_abort_compress(cinfo);

bailout:
  if (cinfo->global_state > CSTATE_START) jpeg_abort_compress(cinfo);
  free(row_pointer);
  for (i = 0; i < MAX_COMPONENTS; i++) {
    free(tmpbuf[i]);
    free(_tmpbuf[i]);
    free(tmpbuf2[i]);
    free(_tmpbuf2[i]);
    free(outbuf[i]);
  }
  if (this->jerr.warning) retval = -1;
  return retval;
}

/* TurboJPEG 1.4+ */
DLLEXPORT int tjEncodeYUVPlanes(tjhandle handle, const unsigned char *srcBuf,
                                int width, int pitch, int height,
                                int pixelFormat, unsigned char **dstPlanes,
                                int *strides, int subsamp, int flags)
{
  static const char FUNCTION_NAME[] = "tjEncodeYUVPlanes";
  int retval = 0;

  GET_TJINSTANCE(handle, -1);

  if (subsamp < 0 || subsamp >= TJ_NUMSAMP)
    THROW("Invalid argument");

  this->subsamp = subsamp;
  processFlags(handle, flags, COMPRESS);

  return tj3EncodeYUVPlanes8(handle, srcBuf, width, pitch, height, pixelFormat,
                             dstPlanes, strides);

bailout:
  return retval;
}


/* TurboJPEG 3.0+ */
DLLEXPORT int tj3EncodeYUV8(tjhandle handle, const unsigned char *srcBuf,
                            int width, int pitch, int height, int pixelFormat,
                            unsigned char *dstBuf, int align)
{
  static const char FUNCTION_NAME[] = "tj3EncodeYUV8";
  unsigned char *dstPlanes[3];
  int pw0, ph0, strides[3], retval = -1;

  GET_TJINSTANCE(handle, -1);

  if (width <= 0 || height <= 0 || dstBuf == NULL || align < 1 ||
      !IS_POW2(align))
    THROW("Invalid argument");

  if (this->subsamp == TJSAMP_UNKNOWN)
    THROW("TJPARAM_SUBSAMP must be specified");

  pw0 = tj3YUVPlaneWidth(0, width, this->subsamp);
  ph0 = tj3YUVPlaneHeight(0, height, this->subsamp);
  dstPlanes[0] = dstBuf;
  strides[0] = PAD(pw0, align);
  if (this->subsamp == TJSAMP_GRAY) {
    strides[1] = strides[2] = 0;
    dstPlanes[1] = dstPlanes[2] = NULL;
  } else {
    int pw1 = tj3YUVPlaneWidth(1, width, this->subsamp);
    int ph1 = tj3YUVPlaneHeight(1, height, this->subsamp);

    strides[1] = strides[2] = PAD(pw1, align);
    if ((unsigned long long)strides[0] * (unsigned long long)ph0 >
        (unsigned long long)INT_MAX ||
        (unsigned long long)strides[1] * (unsigned long long)ph1 >
        (unsigned long long)INT_MAX)
      THROW("Image or row alignment is too large");
    dstPlanes[1] = dstPlanes[0] + strides[0] * ph0;
    dstPlanes[2] = dstPlanes[1] + strides[1] * ph1;
  }

  return tj3EncodeYUVPlanes8(handle, srcBuf, width, pitch, height, pixelFormat,
                             dstPlanes, strides);

bailout:
  return retval;
}

/* TurboJPEG 1.4+ */
DLLEXPORT int tjEncodeYUV3(tjhandle handle, const unsigned char *srcBuf,
                           int width, int pitch, int height, int pixelFormat,
                           unsigned char *dstBuf, int align, int subsamp,
                           int flags)
{
  static const char FUNCTION_NAME[] = "tjEncodeYUV3";
  int retval = 0;

  GET_TJINSTANCE(handle, -1);

  if (subsamp < 0 || subsamp >= TJ_NUMSAMP)
    THROW("Invalid argument");

  this->subsamp = subsamp;
  processFlags(handle, flags, COMPRESS);

  return tj3EncodeYUV8(handle, srcBuf, width, pitch, height, pixelFormat,
                       dstBuf, align);

bailout:
  return retval;
}

/* TurboJPEG 1.2+ */
DLLEXPORT int tjEncodeYUV2(tjhandle handle, unsigned char *srcBuf, int width,
                           int pitch, int height, int pixelFormat,
                           unsigned char *dstBuf, int subsamp, int flags)
{
  return tjEncodeYUV3(handle, srcBuf, width, pitch, height, pixelFormat,
                      dstBuf, 4, subsamp, flags);
}

/* TurboJPEG 1.1+ */
DLLEXPORT int tjEncodeYUV(tjhandle handle, unsigned char *srcBuf, int width,
                          int pitch, int height, int pixelSize,
                          unsigned char *dstBuf, int subsamp, int flags)
{
  return tjEncodeYUV2(handle, srcBuf, width, pitch, height,
                      getPixelFormat(pixelSize, flags), dstBuf, subsamp,
                      flags);
}


/******************************* Decompressor ********************************/

static tjhandle _tjInitDecompress(tjinstance *this)
{
  static unsigned char buffer[1];

  /* This is also straight out of example.c */
  this->dinfo.err = jpeg_std_error(&this->jerr.pub);
  this->jerr.pub.error_exit = my_error_exit;
  this->jerr.pub.output_message = my_output_message;
  this->jerr.emit_message = this->jerr.pub.emit_message;
  this->jerr.pub.emit_message = my_emit_message;
  this->jerr.pub.addon_message_table = turbojpeg_message_table;
  this->jerr.pub.first_addon_message = JMSG_FIRSTADDONCODE;
  this->jerr.pub.last_addon_message = JMSG_LASTADDONCODE;

  if (setjmp(this->jerr.setjmp_buffer)) {
    /* If we get here, the JPEG code has signaled an error. */
    free(this);
    return NULL;
  }

  jpeg_create_decompress(&this->dinfo);
  /* Make an initial call so it will create the source manager */
  jpeg_mem_src_tj(&this->dinfo, buffer, 1);

  this->init |= DECOMPRESS;
  return (tjhandle)this;
}

/* TurboJPEG 1.0+ */
DLLEXPORT tjhandle tjInitDecompress(void)
{
  return tj3Init(TJINIT_DECOMPRESS);
}


/* TurboJPEG 3.0+ */
DLLEXPORT int tj3DecompressHeader(tjhandle handle,
                                  const unsigned char *jpegBuf,
                                  size_t jpegSize)
{
  static const char FUNCTION_NAME[] = "tj3DecompressHeader";
  int retval = 0;
  unsigned char *iccPtr = NULL;
  unsigned int iccLen = 0;

  GET_DINSTANCE(handle);
  if ((this->init & DECOMPRESS) == 0)
    THROW("Instance has not been initialized for decompression");

  if (jpegBuf == NULL || jpegSize <= 0)
    THROW("Invalid argument");

  if (setjmp(this->jerr.setjmp_buffer)) {
    /* If we get here, the JPEG code has signaled an error. */
    return -1;
  }

  jpeg_mem_src_tj(dinfo, jpegBuf, jpegSize);

  /* Extract ICC profile if TJPARAM_SAVEMARKERS is 2 or 4.  (We could
     eventually reuse this mechanism to save other markers, if needed.)
     Because ICC profiles can be large, we extract them by default but allow
     the user to override that behavior. */
  if (this->saveMarkers == 2 || this->saveMarkers == 4)
    jpeg_save_markers(dinfo, JPEG_APP0 + 2, 0xFFFF);
  /* jpeg_read_header() calls jpeg_abort() and returns JPEG_HEADER_TABLES_ONLY
     if the datastream is a tables-only datastream.  Since we aren't using a
     suspending data source, the only other value it can return is
     JPEG_HEADER_OK. */
  if (jpeg_read_header(dinfo, FALSE) == JPEG_HEADER_TABLES_ONLY)
    return 0;

  setDecompParameters(this);

  if (this->saveMarkers == 2 || this->saveMarkers == 4) {
    if (jpeg_read_icc_profile(dinfo, &iccPtr, &iccLen)) {
      free(this->tempICCBuf);
      this->tempICCBuf = iccPtr;
      this->tempICCSize = (size_t)iccLen;
    }
  }

  jpeg_abort_decompress(dinfo);

  if (this->colorspace < 0)
    THROW("Could not determine colorspace of JPEG image");
  if (this->jpegWidth < 1 || this->jpegHeight < 1)
    THROW("Invalid data returned in header");

bailout:
  if (this->jerr.warning) retval = -1;
  return retval;
}

/* TurboJPEG 1.4+ */
DLLEXPORT int tjDecompressHeader3(tjhandle handle,
                                  const unsigned char *jpegBuf,
                                  unsigned long jpegSize, int *width,
                                  int *height, int *jpegSubsamp,
                                  int *jpegColorspace)
{
  static const char FUNCTION_NAME[] = "tjDecompressHeader3";
  int retval = 0;

  GET_TJINSTANCE(handle, -1);

  if (width == NULL || height == NULL || jpegSubsamp == NULL ||
      jpegColorspace == NULL)
    THROW("Invalid argument");

  retval = tj3DecompressHeader(handle, jpegBuf, jpegSize);

  *width = tj3Get(handle, TJPARAM_JPEGWIDTH);
  *height = tj3Get(handle, TJPARAM_JPEGHEIGHT);
  *jpegSubsamp = tj3Get(handle, TJPARAM_SUBSAMP);
  if (*jpegSubsamp == TJSAMP_UNKNOWN)
    THROW("Could not determine subsampling level of JPEG image");
  *jpegColorspace = tj3Get(handle, TJPARAM_COLORSPACE);

bailout:
  return retval;
}

/* TurboJPEG 1.1+ */
DLLEXPORT int tjDecompressHeader2(tjhandle handle, unsigned char *jpegBuf,
                                  unsigned long jpegSize, int *width,
                                  int *height, int *jpegSubsamp)
{
  int jpegColorspace;

  return tjDecompressHeader3(handle, jpegBuf, jpegSize, width, height,
                             jpegSubsamp, &jpegColorspace);
}

/* TurboJPEG 1.0+ */
DLLEXPORT int tjDecompressHeader(tjhandle handle, unsigned char *jpegBuf,
                                 unsigned long jpegSize, int *width,
                                 int *height)
{
  int jpegSubsamp;

  return tjDecompressHeader2(handle, jpegBuf, jpegSize, width, height,
                             &jpegSubsamp);
}


/* TurboJPEG 3.1+ */
DLLEXPORT int tj3GetICCProfile(tjhandle handle, unsigned char **iccBuf,
                               size_t *iccSize)
{
  static const char FUNCTION_NAME[] = "tj3GetICCProfile";
  int retval = 0;

  GET_TJINSTANCE(handle, -1);
  if ((this->init & DECOMPRESS) == 0)
    THROW("Instance has not been initialized for decompression");

  if (iccSize == NULL)
    THROW("Invalid argument");

  if (!this->tempICCBuf || !this->tempICCSize) {
    if (iccBuf) *iccBuf = NULL;
    *iccSize = 0;
    this->jerr.warning = TRUE;
    THROW("No ICC profile data has been extracted");
  }

  *iccSize = this->tempICCSize;
  if (iccBuf == NULL)
    return 0;
  *iccBuf = this->tempICCBuf;
  this->tempICCBuf = NULL;
  this->tempICCSize = 0;

bailout:
  return retval;
}


/* TurboJPEG 3.0+ */
DLLEXPORT tjscalingfactor *tj3GetScalingFactors(int *numScalingFactors)
{
  static const char FUNCTION_NAME[] = "tj3GetScalingFactors";
  tjscalingfactor *retval = (tjscalingfactor *)sf;

  if (numScalingFactors == NULL)
    THROWG("Invalid argument", NULL);

  *numScalingFactors = NUMSF;

bailout:
  return retval;
}

/* TurboJPEG 1.2+ */
DLLEXPORT tjscalingfactor *tjGetScalingFactors(int *numScalingFactors)
{
  return tj3GetScalingFactors(numScalingFactors);
}


/* TurboJPEG 3.0+ */
DLLEXPORT int tj3SetScalingFactor(tjhandle handle,
                                  tjscalingfactor scalingFactor)
{
  static const char FUNCTION_NAME[] = "tj3SetScalingFactor";
  int i, retval = 0;

  GET_TJINSTANCE(handle, -1);
  if ((this->init & DECOMPRESS) == 0)
    THROW("Instance has not been initialized for decompression");

  for (i = 0; i < NUMSF; i++) {
    if (scalingFactor.num == sf[i].num && scalingFactor.denom == sf[i].denom)
      break;
  }
  if (i >= NUMSF)
    THROW("Unsupported scaling factor");

  this->scalingFactor = scalingFactor;

bailout:
  return retval;
}


/* TurboJPEG 3.0+ */
DLLEXPORT int tj3SetCroppingRegion(tjhandle handle, tjregion croppingRegion)
{
  static const char FUNCTION_NAME[] = "tj3SetCroppingRegion";
  int retval = 0, scaledWidth, scaledHeight;

  GET_TJINSTANCE(handle, -1);
  if ((this->init & DECOMPRESS) == 0)
    THROW("Instance has not been initialized for decompression");

  if (croppingRegion.x == 0 && croppingRegion.y == 0 &&
      croppingRegion.w == 0 && croppingRegion.h == 0) {
    this->croppingRegion = croppingRegion;
    return 0;
  }

  if (croppingRegion.x < 0 || croppingRegion.y < 0 || croppingRegion.w < 0 ||
      croppingRegion.h < 0)
    THROW("Invalid cropping region");
  if (this->jpegWidth < 0 || this->jpegHeight < 0)
    THROW("JPEG header has not yet been read");
  if ((this->precision != 8 && this->precision != 12) || this->lossless)
    THROW("Cannot partially decompress lossless JPEG images");
  if (this->subsamp == TJSAMP_UNKNOWN)
    THROW("Could not determine subsampling level of JPEG image");

  scaledWidth = TJSCALED(this->jpegWidth, this->scalingFactor);
  scaledHeight = TJSCALED(this->jpegHeight, this->scalingFactor);

  if (croppingRegion.x %
      TJSCALED(tjMCUWidth[this->subsamp], this->scalingFactor) != 0)
    THROWI("The left boundary of the cropping region (%d) is not\n"
           "divisible by the scaled iMCU width (%d)",
           croppingRegion.x,
           TJSCALED(tjMCUWidth[this->subsamp], this->scalingFactor));
  if (croppingRegion.w == 0)
    croppingRegion.w = scaledWidth - croppingRegion.x;
  if (croppingRegion.h == 0)
    croppingRegion.h = scaledHeight - croppingRegion.y;
  if (croppingRegion.w <= 0 || croppingRegion.h <= 0 ||
      croppingRegion.x + croppingRegion.w > scaledWidth ||
      croppingRegion.y + croppingRegion.h > scaledHeight)
    THROW("The cropping region exceeds the scaled image dimensions");

  this->croppingRegion = croppingRegion;

bailout:
  return retval;
}


/* tj3Decompress*() is implemented in turbojpeg-mp.c */

/* TurboJPEG 1.2+ */
DLLEXPORT int tjDecompress2(tjhandle handle, const unsigned char *jpegBuf,
                            unsigned long jpegSize, unsigned char *dstBuf,
                            int width, int pitch, int height, int pixelFormat,
                            int flags)
{
  static const char FUNCTION_NAME[] = "tjDecompress2";
  int i, retval = 0, jpegwidth, jpegheight, scaledw, scaledh;

  GET_DINSTANCE(handle);
  if ((this->init & DECOMPRESS) == 0)
    THROW("Instance has not been initialized for decompression");

  if (jpegBuf == NULL || jpegSize <= 0 || width < 0 || height < 0)
    THROW("Invalid argument");

  if (setjmp(this->jerr.setjmp_buffer)) {
    /* If we get here, the JPEG code has signaled an error. */
    retval = -1;  goto bailout;
  }

  jpeg_mem_src_tj(dinfo, jpegBuf, jpegSize);
  jpeg_read_header(dinfo, TRUE);
  jpegwidth = dinfo->image_width;  jpegheight = dinfo->image_height;
  if (width == 0) width = jpegwidth;
  if (height == 0) height = jpegheight;
  for (i = 0; i < NUMSF; i++) {
    scaledw = TJSCALED(jpegwidth, sf[i]);
    scaledh = TJSCALED(jpegheight, sf[i]);
    if (scaledw <= width && scaledh <= height)
      break;
  }
  if (i >= NUMSF)
    THROW("Could not scale down to desired image dimensions");

  processFlags(handle, flags, DECOMPRESS);

  if (tj3SetScalingFactor(handle, sf[i]) == -1)
    return -1;
  if (tj3SetCroppingRegion(handle, TJUNCROPPED) == -1)
    return -1;
  return tj3Decompress8(handle, jpegBuf, jpegSize, dstBuf, pitch, pixelFormat);

bailout:
  if (dinfo->global_state > DSTATE_START) jpeg_abort_decompress(dinfo);
  if (this->jerr.warning) retval = -1;
  return retval;
}

/* TurboJPEG 1.0+ */
DLLEXPORT int tjDecompress(tjhandle handle, unsigned char *jpegBuf,
                           unsigned long jpegSize, unsigned char *dstBuf,
                           int width, int pitch, int height, int pixelSize,
                           int flags)
{
  if (flags & TJ_YUV)
    return tjDecompressToYUV(handle, jpegBuf, jpegSize, dstBuf, flags);
  else
    return tjDecompress2(handle, jpegBuf, jpegSize, dstBuf, width, pitch,
                         height, getPixelFormat(pixelSize, flags), flags);
}


/* TurboJPEG 3.0+ */
DLLEXPORT int tj3DecompressToYUVPlanes8(tjhandle handle,
                                        const unsigned char *jpegBuf,
                                        size_t jpegSize,
                                        unsigned char **dstPlanes,
                                        int *strides)
{
  static const char FUNCTION_NAME[] = "tj3DecompressToYUVPlanes8";
  int i, row, retval = 0;
  int pw[MAX_COMPONENTS], ph[MAX_COMPONENTS], iw[MAX_COMPONENTS],
    tmpbufsize = 0, usetmpbuf = 0, th[MAX_COMPONENTS];
  JSAMPLE *_tmpbuf = NULL, *ptr;
  JSAMPROW *outbuf[MAX_COMPONENTS], *tmpbuf[MAX_COMPONENTS];
  int dctsize;
  struct my_progress_mgr progress;

  GET_DINSTANCE(handle);

  for (i = 0; i < MAX_COMPONENTS; i++) {
    tmpbuf[i] = NULL;  outbuf[i] = NULL;
  }

  if ((this->init & DECOMPRESS) == 0)
    THROW("Instance has not been initialized for decompression");

  if (jpegBuf == NULL || jpegSize <= 0 || !dstPlanes || !dstPlanes[0])
    THROW("Invalid argument");

  if (this->scanLimit) {
    memset(&progress, 0, sizeof(struct my_progress_mgr));
    progress.pub.progress_monitor = my_progress_monitor;
    progress.this = this;
    dinfo->progress = &progress.pub;
  } else
    dinfo->progress = NULL;

  dinfo->mem->max_memory_to_use = (long)this->maxMemory * 1048576L;

  if (setjmp(this->jerr.setjmp_buffer)) {
    /* If we get here, the JPEG code has signaled an error. */
    retval = -1;  goto bailout;
  }

  if (dinfo->global_state <= DSTATE_INHEADER) {
    jpeg_mem_src_tj(dinfo, jpegBuf, jpegSize);
    jpeg_read_header(dinfo, TRUE);
  }
  setDecompParameters(this);
  if (this->maxPixels &&
      (unsigned long long)this->jpegWidth * this->jpegHeight >
      (unsigned long long)this->maxPixels)
    THROW("Image is too large");
  if (this->subsamp == TJSAMP_UNKNOWN)
    THROW("Could not determine subsampling level of JPEG image");

  if (this->subsamp != TJSAMP_GRAY && (!dstPlanes[1] || !dstPlanes[2]))
    THROW("Invalid argument");

  if (dinfo->num_components > 3)
    THROW("JPEG image must have 3 or fewer components");

  dinfo->scale_num = this->scalingFactor.num;
  dinfo->scale_denom = this->scalingFactor.denom;
  jpeg_calc_output_dimensions(dinfo);

  dctsize = DCTSIZE * this->scalingFactor.num / this->scalingFactor.denom;

  for (i = 0; i < dinfo->num_components; i++) {
    jpeg_component_info *compptr = &dinfo->comp_info[i];
    int ih;

    iw[i] = compptr->width_in_blocks * dctsize;
    ih = compptr->height_in_blocks * dctsize;
    pw[i] = tj3YUVPlaneWidth(i, dinfo->output_width, this->subsamp);
    ph[i] = tj3YUVPlaneHeight(i, dinfo->output_height, this->subsamp);
    if (iw[i] != pw[i] || ih != ph[i]) usetmpbuf = 1;
    th[i] = compptr->v_samp_factor * dctsize;
    tmpbufsize += iw[i] * th[i];
    if ((outbuf[i] = (JSAMPROW *)malloc(sizeof(JSAMPROW) * ph[i])) == NULL)
      THROW("Memory allocation failure");
    ptr = dstPlanes[i];
    for (row = 0; row < ph[i]; row++) {
      outbuf[i][row] = ptr;
      ptr += (strides && strides[i] != 0) ? strides[i] : pw[i];
    }
  }
  if (usetmpbuf) {
    if ((_tmpbuf = (JSAMPLE *)MALLOC(sizeof(JSAMPLE) * tmpbufsize)) == NULL)
      THROW("Memory allocation failure");
    ptr = _tmpbuf;
    for (i = 0; i < dinfo->num_components; i++) {
      if ((tmpbuf[i] = (JSAMPROW *)malloc(sizeof(JSAMPROW) * th[i])) == NULL)
        THROW("Memory allocation failure");
      for (row = 0; row < th[i]; row++) {
        tmpbuf[i][row] = ptr;
        ptr += iw[i];
      }
    }
  }

  if (setjmp(this->jerr.setjmp_buffer)) {
    /* If we get here, the JPEG code has signaled an error. */
    retval = -1;  goto bailout;
  }

  dinfo->do_fancy_upsampling = !this->fastUpsample;
  dinfo->dct_method = this->fastDCT ? JDCT_FASTEST : JDCT_ISLOW;
  dinfo->raw_data_out = TRUE;

  dinfo->mem->max_memory_to_use = (long)this->maxMemory * 1048576L;

  jpeg_start_decompress(dinfo);
  for (row = 0; row < (int)dinfo->output_height;
       row += dinfo->max_v_samp_factor * dinfo->_min_DCT_scaled_size) {
    JSAMPARRAY yuvptr[MAX_COMPONENTS];
    int crow[MAX_COMPONENTS];

    for (i = 0; i < dinfo->num_components; i++) {
      jpeg_component_info *compptr = &dinfo->comp_info[i];

      if (this->subsamp == TJSAMP_420) {
        /* When 4:2:0 subsampling is used with IDCT scaling, libjpeg will try
           to be clever and use the IDCT to perform upsampling on the U and V
           planes.  For instance, if the output image is to be scaled by 1/2
           relative to the JPEG image, then the scaling factor and upsampling
           effectively cancel each other, so a normal 8x8 IDCT can be used.
           However, this is not desirable when using the decompress-to-YUV
           functionality in TurboJPEG, since we want to output the U and V
           planes in their subsampled form.  Thus, we have to override some
           internal libjpeg parameters to force it to use the "scaled" IDCT
           functions on the U and V planes. */
        compptr->_DCT_scaled_size = dctsize;
        compptr->MCU_sample_width = tjMCUWidth[this->subsamp] *
          this->scalingFactor.num / this->scalingFactor.denom *
          compptr->v_samp_factor / dinfo->max_v_samp_factor;
        dinfo->idct->inverse_DCT[i] = dinfo->idct->inverse_DCT[0];
      }
      crow[i] = row * compptr->v_samp_factor / dinfo->max_v_samp_factor;
      if (usetmpbuf) yuvptr[i] = tmpbuf[i];
      else yuvptr[i] = &outbuf[i][crow[i]];
    }
    jpeg_read_raw_data(dinfo, yuvptr,
                       dinfo->max_v_samp_factor * dinfo->_min_DCT_scaled_size);
    if (usetmpbuf) {
      int j;

      for (i = 0; i < dinfo->num_components; i++) {
        for (j = 0; j < MIN(th[i], ph[i] - crow[i]); j++) {
          memcpy(outbuf[i][crow[i] + j], tmpbuf[i][j], pw[i]);
        }
      }
    }
  }
  jpeg_finish_decompress(dinfo);

bailout:
  if (dinfo->global_state > DSTATE_START) jpeg_abort_decompress(dinfo);
  for (i = 0; i < MAX_COMPONENTS; i++) {
    free(tmpbuf[i]);
    free(outbuf[i]);
  }
  free(_tmpbuf);
  if (this->jerr.warning) retval = -1;
  return retval;
}

/* TurboJPEG 1.4+ */
DLLEXPORT int tjDecompressToYUVPlanes(tjhandle handle,
                                      const unsigned char *jpegBuf,
                                      unsigned long jpegSize,
                                      unsigned char **dstPlanes, int width,
                                      int *strides, int height, int flags)
{
  static const char FUNCTION_NAME[] = "tjDecompressToYUVPlanes";
  int i, retval = 0, jpegwidth, jpegheight, scaledw, scaledh;

  GET_DINSTANCE(handle);
  if ((this->init & DECOMPRESS) == 0)
    THROW("Instance has not been initialized for decompression");

  if (jpegBuf == NULL || jpegSize <= 0 || width < 0 || height < 0)
    THROW("Invalid argument");

  if (setjmp(this->jerr.setjmp_buffer)) {
    /* If we get here, the JPEG code has signaled an error. */
    retval = -1;  goto bailout;
  }

  jpeg_mem_src_tj(dinfo, jpegBuf, jpegSize);
  jpeg_read_header(dinfo, TRUE);
  jpegwidth = dinfo->image_width;  jpegheight = dinfo->image_height;
  if (width == 0) width = jpegwidth;
  if (height == 0) height = jpegheight;
  for (i = 0; i < NUMSF; i++) {
    scaledw = TJSCALED(jpegwidth, sf[i]);
    scaledh = TJSCALED(jpegheight, sf[i]);
    if (scaledw <= width && scaledh <= height)
      break;
  }
  if (i >= NUMSF)
    THROW("Could not scale down to desired image dimensions");

  processFlags(handle, flags, DECOMPRESS);

  if (tj3SetScalingFactor(handle, sf[i]) == -1)
    return -1;
  return tj3DecompressToYUVPlanes8(handle, jpegBuf, jpegSize, dstPlanes,
                                   strides);

bailout:
  if (dinfo->global_state > DSTATE_START) jpeg_abort_decompress(dinfo);
  if (this->jerr.warning) retval = -1;
  return retval;
}


/* TurboJPEG 3.0+ */
DLLEXPORT int tj3DecompressToYUV8(tjhandle handle,
                                  const unsigned char *jpegBuf,
                                  size_t jpegSize,
                                  unsigned char *dstBuf, int align)
{
  static const char FUNCTION_NAME[] = "tj3DecompressToYUV8";
  unsigned char *dstPlanes[3];
  int pw0, ph0, strides[3], retval = -1;
  int width, height;

  GET_DINSTANCE(handle);

  if (jpegBuf == NULL || jpegSize <= 0 || dstBuf == NULL || align < 1 ||
      !IS_POW2(align))
    THROW("Invalid argument");

  if (setjmp(this->jerr.setjmp_buffer)) {
    /* If we get here, the JPEG code has signaled an error. */
    retval = -1;  goto bailout;
  }

  if (dinfo->global_state <= DSTATE_INHEADER) {
    jpeg_mem_src_tj(dinfo, jpegBuf, jpegSize);
    jpeg_read_header(dinfo, TRUE);
  }
  setDecompParameters(this);
  if (this->subsamp == TJSAMP_UNKNOWN)
    THROW("Could not determine subsampling level of JPEG image");

  width = TJSCALED(dinfo->image_width, this->scalingFactor);
  height = TJSCALED(dinfo->image_height, this->scalingFactor);

  pw0 = tj3YUVPlaneWidth(0, width, this->subsamp);
  ph0 = tj3YUVPlaneHeight(0, height, this->subsamp);
  dstPlanes[0] = dstBuf;
  strides[0] = PAD(pw0, align);
  if (this->subsamp == TJSAMP_GRAY) {
    strides[1] = strides[2] = 0;
    dstPlanes[1] = dstPlanes[2] = NULL;
  } else {
    int pw1 = tj3YUVPlaneWidth(1, width, this->subsamp);
    int ph1 = tj3YUVPlaneHeight(1, height, this->subsamp);

    strides[1] = strides[2] = PAD(pw1, align);
    if ((unsigned long long)strides[0] * (unsigned long long)ph0 >
        (unsigned long long)INT_MAX ||
        (unsigned long long)strides[1] * (unsigned long long)ph1 >
        (unsigned long long)INT_MAX)
      THROW("Image or row alignment is too large");
    dstPlanes[1] = dstPlanes[0] + strides[0] * ph0;
    dstPlanes[2] = dstPlanes[1] + strides[1] * ph1;
  }

  return tj3DecompressToYUVPlanes8(handle, jpegBuf, jpegSize, dstPlanes,
                                   strides);

bailout:
  if (dinfo->global_state > DSTATE_START) jpeg_abort_decompress(dinfo);
  if (this->jerr.warning) retval = -1;
  return retval;
}

/* TurboJPEG 1.4+ */
DLLEXPORT int tjDecompressToYUV2(tjhandle handle, const unsigned char *jpegBuf,
                                 unsigned long jpegSize, unsigned char *dstBuf,
                                 int width, int align, int height, int flags)
{
  static const char FUNCTION_NAME[] = "tjDecompressToYUV2";
  int i, retval = 0, jpegwidth, jpegheight, scaledw, scaledh;

  GET_DINSTANCE(handle);
  if ((this->init & DECOMPRESS) == 0)
    THROW("Instance has not been initialized for decompression");

  if (jpegBuf == NULL || jpegSize <= 0 || width < 0 || height < 0)
    THROW("Invalid argument");

  if (setjmp(this->jerr.setjmp_buffer)) {
    /* If we get here, the JPEG code has signaled an error. */
    retval = -1;  goto bailout;
  }

  jpeg_mem_src_tj(dinfo, jpegBuf, jpegSize);
  jpeg_read_header(dinfo, TRUE);
  jpegwidth = dinfo->image_width;  jpegheight = dinfo->image_height;
  if (width == 0) width = jpegwidth;
  if (height == 0) height = jpegheight;
  for (i = 0; i < NUMSF; i++) {
    scaledw = TJSCALED(jpegwidth, sf[i]);
    scaledh = TJSCALED(jpegheight, sf[i]);
    if (scaledw <= width && scaledh <= height)
      break;
  }
  if (i >= NUMSF)
    THROW("Could not scale down to desired image dimensions");

  width = scaledw;  height = scaledh;

  processFlags(handle, flags, DECOMPRESS);

  if (tj3SetScalingFactor(handle, sf[i]) == -1)
    return -1;
  return tj3DecompressToYUV8(handle, jpegBuf, (size_t)jpegSize, dstBuf, align);

bailout:
  if (dinfo->global_state > DSTATE_START) jpeg_abort_decompress(dinfo);
  if (this->jerr.warning) retval = -1;
  return retval;
}

/* TurboJPEG 1.1+ */
DLLEXPORT int tjDecompressToYUV(tjhandle handle, unsigned char *jpegBuf,
                                unsigned long jpegSize, unsigned char *dstBuf,
                                int flags)
{
  return tjDecompressToYUV2(handle, jpegBuf, jpegSize, dstBuf, 0, 4, 0, flags);
}


static void setDecodeDefaults(tjinstance *this, int pixelFormat)
{
  int i;

  this->dinfo.scale_num = this->dinfo.scale_denom = 1;

  if (this->subsamp == TJSAMP_GRAY) {
    this->dinfo.num_components = this->dinfo.comps_in_scan = 1;
    this->dinfo.jpeg_color_space = JCS_GRAYSCALE;
  } else {
    this->dinfo.num_components = this->dinfo.comps_in_scan = 3;
    this->dinfo.jpeg_color_space = JCS_YCbCr;
  }

  this->dinfo.comp_info = (jpeg_component_info *)
    (*this->dinfo.mem->alloc_small) ((j_common_ptr)&this->dinfo, JPOOL_IMAGE,
                                     this->dinfo.num_components *
                                     sizeof(jpeg_component_info));

  for (i = 0; i < this->dinfo.num_components; i++) {
    jpeg_component_info *compptr = &this->dinfo.comp_info[i];

    compptr->h_samp_factor = (i == 0) ? tjMCUWidth[this->subsamp] / 8 : 1;
    compptr->v_samp_factor = (i == 0) ? tjMCUHeight[this->subsamp] / 8 : 1;
    compptr->component_index = i;
    compptr->component_id = i + 1;
    compptr->quant_tbl_no = compptr->dc_tbl_no =
      compptr->ac_tbl_no = (i == 0) ? 0 : 1;
    this->dinfo.cur_comp_info[i] = compptr;
  }
  this->dinfo.data_precision = 8;
  for (i = 0; i < 2; i++) {
    if (this->dinfo.quant_tbl_ptrs[i] == NULL)
      this->dinfo.quant_tbl_ptrs[i] =
        jpeg_alloc_quant_table((j_common_ptr)&this->dinfo);
  }

  this->dinfo.mem->max_memory_to_use = (long)this->maxMemory * 1048576L;
}


static int my_read_markers(j_decompress_ptr dinfo)
{
  return JPEG_REACHED_SOS;
}

static void my_reset_marker_reader(j_decompress_ptr dinfo)
{
}

/* TurboJPEG 3.0+ */
DLLEXPORT int tj3DecodeYUVPlanes8(tjhandle handle,
                                  const unsigned char * const *srcPlanes,
                                  const int *strides, unsigned char *dstBuf,
                                  int width, int pitch, int height,
                                  int pixelFormat)
{
  static const char FUNCTION_NAME[] = "tj3DecodeYUVPlanes8";
  JSAMPROW *row_pointer = NULL;
  JSAMPLE *_tmpbuf[MAX_COMPONENTS];
  JSAMPROW *tmpbuf[MAX_COMPONENTS], *inbuf[MAX_COMPONENTS];
  int i, retval = 0, row, pw0, ph0, pw[MAX_COMPONENTS], ph[MAX_COMPONENTS];
  JSAMPLE *ptr;
  jpeg_component_info *compptr;
  int (*old_read_markers) (j_decompress_ptr);
  void (*old_reset_marker_reader) (j_decompress_ptr);

  GET_DINSTANCE(handle);

  for (i = 0; i < MAX_COMPONENTS; i++) {
    tmpbuf[i] = NULL;  _tmpbuf[i] = NULL;  inbuf[i] = NULL;
  }

  if ((this->init & DECOMPRESS) == 0)
    THROW("Instance has not been initialized for decompression");

  if (!srcPlanes || !srcPlanes[0] || dstBuf == NULL || width <= 0 ||
      pitch < 0 || height <= 0 || pixelFormat < 0 || pixelFormat >= TJ_NUMPF)
    THROW("Invalid argument");
  if (this->subsamp != TJSAMP_GRAY && (!srcPlanes[1] || !srcPlanes[2]))
    THROW("Invalid argument");

  if (setjmp(this->jerr.setjmp_buffer)) {
    /* If we get here, the JPEG code has signaled an error. */
    retval = -1;  goto bailout;
  }

  if (this->subsamp == TJSAMP_UNKNOWN)
    THROW("TJPARAM_SUBSAMP must be specified");
  if (pixelFormat == TJPF_CMYK)
    THROW("Cannot decode YUV images into packed-pixel CMYK images.");

  if (pitch == 0) pitch = width * tjPixelSize[pixelFormat];
  dinfo->image_width = width;
  dinfo->image_height = height;

  dinfo->progressive_mode = dinfo->inputctl->has_multiple_scans = FALSE;
  dinfo->Ss = dinfo->Ah = dinfo->Al = 0;
  dinfo->Se = DCTSIZE2 - 1;
  setDecodeDefaults(this, pixelFormat);
  old_read_markers = dinfo->marker->read_markers;
  dinfo->marker->read_markers = my_read_markers;
  old_reset_marker_reader = dinfo->marker->reset_marker_reader;
  dinfo->marker->reset_marker_reader = my_reset_marker_reader;
  jpeg_read_header(dinfo, TRUE);
  dinfo->marker->read_markers = old_read_markers;
  dinfo->marker->reset_marker_reader = old_reset_marker_reader;

  this->dinfo.out_color_space = pf2cs[pixelFormat];
  this->dinfo.dct_method = this->fastDCT ? JDCT_FASTEST : JDCT_ISLOW;
  dinfo->do_fancy_upsampling = FALSE;
  dinfo->Se = DCTSIZE2 - 1;
  jinit_master_decompress(dinfo);
  (*dinfo->upsample->start_pass) (dinfo);

  pw0 = PAD(width, dinfo->max_h_samp_factor);
  ph0 = PAD(height, dinfo->max_v_samp_factor);

  if (pitch == 0) pitch = dinfo->output_width * tjPixelSize[pixelFormat];

  if ((row_pointer = (JSAMPROW *)malloc(sizeof(JSAMPROW) * ph0)) == NULL)
    THROW("Memory allocation failure");
  for (i = 0; i < height; i++) {
    if (this->bottomUp)
      row_pointer[i] = &dstBuf[(height - i - 1) * (size_t)pitch];
    else
      row_pointer[i] = &dstBuf[i * (size_t)pitch];
  }
  if (height < ph0)
    for (i = height; i < ph0; i++) row_pointer[i] = row_pointer[height - 1];

  for (i = 0; i < dinfo->num_components; i++) {
    compptr = &dinfo->comp_info[i];
    _tmpbuf[i] =
      (JSAMPLE *)malloc(PAD(compptr->width_in_blocks * DCTSIZE, 32) *
                        compptr->v_samp_factor + 32);
    if (!_tmpbuf[i])
      THROW("Memory allocation failure");
    tmpbuf[i] = (JSAMPROW *)malloc(sizeof(JSAMPROW) * compptr->v_samp_factor);
    if (!tmpbuf[i])
      THROW("Memory allocation failure");
    for (row = 0; row < compptr->v_samp_factor; row++) {
      unsigned char *_tmpbuf_aligned =
        (unsigned char *)PAD((JUINTPTR)_tmpbuf[i], 32);

      tmpbuf[i][row] =
        &_tmpbuf_aligned[PAD(compptr->width_in_blocks * DCTSIZE, 32) * row];
    }
    pw[i] = pw0 * compptr->h_samp_factor / dinfo->max_h_samp_factor;
    ph[i] = ph0 * compptr->v_samp_factor / dinfo->max_v_samp_factor;
    inbuf[i] = (JSAMPROW *)malloc(sizeof(JSAMPROW) * ph[i]);
    if (!inbuf[i])
      THROW("Memory allocation failure");
    ptr = (JSAMPLE *)srcPlanes[i];
    for (row = 0; row < ph[i]; row++) {
      inbuf[i][row] = ptr;
      ptr += (strides && strides[i] != 0) ? strides[i] : pw[i];
    }
  }

  if (setjmp(this->jerr.setjmp_buffer)) {
    /* If we get here, the JPEG code has signaled an error. */
    retval = -1;  goto bailout;
  }

  for (row = 0; row < ph0; row += dinfo->max_v_samp_factor) {
    JDIMENSION inrow = 0, outrow = 0;

    for (i = 0, compptr = dinfo->comp_info; i < dinfo->num_components;
         i++, compptr++)
      jcopy_sample_rows(inbuf[i],
        row * compptr->v_samp_factor / dinfo->max_v_samp_factor, tmpbuf[i], 0,
        compptr->v_samp_factor, pw[i]);
    (dinfo->upsample->upsample) (dinfo, tmpbuf, &inrow,
                                 dinfo->max_v_samp_factor, &row_pointer[row],
                                 &outrow, dinfo->max_v_samp_factor);
  }
  jpeg_abort_decompress(dinfo);

bailout:
  if (dinfo->global_state > DSTATE_START) jpeg_abort_decompress(dinfo);
  free(row_pointer);
  for (i = 0; i < MAX_COMPONENTS; i++) {
    free(tmpbuf[i]);
    free(_tmpbuf[i]);
    free(inbuf[i]);
  }
  if (this->jerr.warning) retval = -1;
  return retval;
}

/* TurboJPEG 1.4+ */
DLLEXPORT int tjDecodeYUVPlanes(tjhandle handle,
                                const unsigned char **srcPlanes,
                                const int *strides, int subsamp,
                                unsigned char *dstBuf, int width, int pitch,
                                int height, int pixelFormat, int flags)
{
  static const char FUNCTION_NAME[] = "tjDecodeYUVPlanes";
  int retval = 0;

  GET_TJINSTANCE(handle, -1);

  if (subsamp < 0 || subsamp >= TJ_NUMSAMP)
    THROW("Invalid argument");

  this->subsamp = subsamp;
  processFlags(handle, flags, DECOMPRESS);

  return tj3DecodeYUVPlanes8(handle, srcPlanes, strides, dstBuf, width, pitch,
                             height, pixelFormat);

bailout:
  return retval;
}


/* TurboJPEG 3.0+ */
DLLEXPORT int tj3DecodeYUV8(tjhandle handle, const unsigned char *srcBuf,
                            int align, unsigned char *dstBuf, int width,
                            int pitch, int height, int pixelFormat)
{
  static const char FUNCTION_NAME[] = "tj3DecodeYUV8";
  const unsigned char *srcPlanes[3];
  int pw0, ph0, strides[3], retval = -1;

  GET_TJINSTANCE(handle, -1);

  if (srcBuf == NULL || align < 1 || !IS_POW2(align) || width <= 0 ||
      height <= 0)
    THROW("Invalid argument");

  if (this->subsamp == TJSAMP_UNKNOWN)
    THROW("TJPARAM_SUBSAMP must be specified");

  pw0 = tj3YUVPlaneWidth(0, width, this->subsamp);
  ph0 = tj3YUVPlaneHeight(0, height, this->subsamp);
  srcPlanes[0] = srcBuf;
  strides[0] = PAD(pw0, align);
  if (this->subsamp == TJSAMP_GRAY) {
    strides[1] = strides[2] = 0;
    srcPlanes[1] = srcPlanes[2] = NULL;
  } else {
    int pw1 = tj3YUVPlaneWidth(1, width, this->subsamp);
    int ph1 = tj3YUVPlaneHeight(1, height, this->subsamp);

    strides[1] = strides[2] = PAD(pw1, align);
    if ((unsigned long long)strides[0] * (unsigned long long)ph0 >
        (unsigned long long)INT_MAX ||
        (unsigned long long)strides[1] * (unsigned long long)ph1 >
        (unsigned long long)INT_MAX)
      THROW("Image or row alignment is too large");
    srcPlanes[1] = srcPlanes[0] + strides[0] * ph0;
    srcPlanes[2] = srcPlanes[1] + strides[1] * ph1;
  }

  return tj3DecodeYUVPlanes8(handle, srcPlanes, strides, dstBuf, width, pitch,
                             height, pixelFormat);

bailout:
  return retval;
}

/* TurboJPEG 1.4+ */
DLLEXPORT int tjDecodeYUV(tjhandle handle, const unsigned char *srcBuf,
                          int align, int subsamp, unsigned char *dstBuf,
                          int width, int pitch, int height, int pixelFormat,
                          int flags)
{
  static const char FUNCTION_NAME[] = "tjDecodeYUV";
  int retval = -1;

  GET_TJINSTANCE(handle, -1);

  if (subsamp < 0 || subsamp >= TJ_NUMSAMP)
    THROW("Invalid argument");

  this->subsamp = subsamp;
  processFlags(handle, flags, DECOMPRESS);

  return tj3DecodeYUV8(handle, srcBuf, align, dstBuf, width, pitch, height,
                       pixelFormat);

bailout:
  return retval;
}


/******************************** Transformer ********************************/

/* TurboJPEG 1.2+ */
DLLEXPORT tjhandle tjInitTransform(void)
{
  return tj3Init(TJINIT_TRANSFORM);
}


static int getDstSubsamp(int srcSubsamp, const tjtransform *transform)
{
  int dstSubsamp;

  if (!transform)
    return srcSubsamp;

  dstSubsamp = (transform->options & TJXOPT_GRAY) ? TJSAMP_GRAY : srcSubsamp;

  if (transform->op == TJXOP_TRANSPOSE || transform->op == TJXOP_TRANSVERSE ||
      transform->op == TJXOP_ROT90 || transform->op == TJXOP_ROT270) {
    if (dstSubsamp == TJSAMP_422) dstSubsamp = TJSAMP_440;
    else if (dstSubsamp == TJSAMP_440) dstSubsamp = TJSAMP_422;
    else if (dstSubsamp == TJSAMP_411) dstSubsamp = TJSAMP_441;
    else if (dstSubsamp == TJSAMP_441) dstSubsamp = TJSAMP_411;
  }

  return dstSubsamp;
}

static int getTransformedSpecs(tjhandle handle, int *width, int *height,
                               int *subsamp, const tjtransform *transform,
                               const char *FUNCTION_NAME)
{
  int retval = 0, dstWidth, dstHeight, dstSubsamp;

  GET_TJINSTANCE(handle, -1);
  if ((this->init & COMPRESS) == 0 || (this->init & DECOMPRESS) == 0)
    THROW("Instance has not been initialized for transformation");

  if (!width || !height || !subsamp || !transform || *width < 1 ||
      *height < 1 || *subsamp < TJSAMP_UNKNOWN || *subsamp >= TJ_NUMSAMP)
    THROW("Invalid argument");

  dstWidth = *width;  dstHeight = *height;
  if (transform->op == TJXOP_TRANSPOSE || transform->op == TJXOP_TRANSVERSE ||
      transform->op == TJXOP_ROT90 || transform->op == TJXOP_ROT270) {
    dstWidth = *height;  dstHeight = *width;
  }
  dstSubsamp = getDstSubsamp(*subsamp, transform);

  if (transform->options & TJXOPT_CROP) {
    int croppedWidth, croppedHeight;

    if (transform->r.x < 0 || transform->r.y < 0 || transform->r.w < 0 ||
        transform->r.h < 0)
      THROW("Invalid cropping region");
    if (dstSubsamp == TJSAMP_UNKNOWN)
      THROW("Could not determine subsampling level of JPEG image");
    if ((transform->r.x % tjMCUWidth[dstSubsamp]) != 0 ||
        (transform->r.y % tjMCUHeight[dstSubsamp]) != 0)
      THROWI("To crop this JPEG image, x must be a multiple of %d\n"
             "and y must be a multiple of %d.", tjMCUWidth[dstSubsamp],
             tjMCUHeight[dstSubsamp]);
    if (transform->r.x >= dstWidth || transform->r.y >= dstHeight)
      THROW("The cropping region exceeds the destination image dimensions");
    croppedWidth = transform->r.w == 0 ? dstWidth - transform->r.x :
                                         transform->r.w;
    croppedHeight = transform->r.h == 0 ? dstHeight - transform->r.y :
                                          transform->r.h;
    if (transform->r.x + croppedWidth > dstWidth ||
        transform->r.y + croppedHeight > dstHeight)
      THROW("The cropping region exceeds the destination image dimensions");
    dstWidth = croppedWidth;  dstHeight = croppedHeight;
  }

  *width = dstWidth;  *height = dstHeight;  *subsamp = dstSubsamp;

bailout:
  return retval;
}


/* TurboJPEG 3.1+ */
DLLEXPORT size_t tj3TransformBufSize(tjhandle handle,
                                     const tjtransform *transform)
{
  static const char FUNCTION_NAME[] = "tj3TransformBufSize";
  size_t retval = 0;
  int dstWidth, dstHeight, dstSubsamp;

  GET_TJINSTANCE(handle, 0);
  if ((this->init & COMPRESS) == 0 || (this->init & DECOMPRESS) == 0)
    THROWRV("Instance has not been initialized for transformation", 0);

  if (transform == NULL)
    THROWRV("Invalid argument", 0)

  if (this->jpegWidth < 0 || this->jpegHeight < 0)
    THROWRV("JPEG header has not yet been read", 0);

  dstWidth = this->jpegWidth;
  dstHeight = this->jpegHeight;
  dstSubsamp = this->subsamp;
  if (getTransformedSpecs(handle, &dstWidth, &dstHeight, &dstSubsamp,
                          transform, FUNCTION_NAME) == -1) {
    retval = 0;
    goto bailout;
  }

  retval = tj3JPEGBufSize(dstWidth, dstHeight, dstSubsamp);
  if ((this->saveMarkers == 2 || this->saveMarkers == 4) &&
      !(transform->options & TJXOPT_COPYNONE))
    retval += this->tempICCSize;
  else
    retval += this->iccSize;

bailout:
  return retval;
}


/* TurboJPEG 3.0+ */
DLLEXPORT int tj3Transform(tjhandle handle, const unsigned char *jpegBuf,
                           size_t jpegSize, int n, unsigned char **dstBufs,
                           size_t *dstSizes, const tjtransform *t)
{
  static const char FUNCTION_NAME[] = "tj3Transform";
  jpeg_transform_info *xinfo = NULL;
  jvirt_barray_ptr *srccoefs, *dstcoefs;
  int retval = 0, i, saveMarkers = 0, srcSubsamp;
  boolean alloc = TRUE;
  struct my_progress_mgr progress;

  GET_INSTANCE(handle);
  if ((this->init & COMPRESS) == 0 || (this->init & DECOMPRESS) == 0)
    THROW("Instance has not been initialized for transformation");

  if (jpegBuf == NULL || jpegSize <= 0 || n < 1 || dstBufs == NULL ||
      dstSizes == NULL || t == NULL)
    THROW("Invalid argument");

  if (this->scanLimit) {
    memset(&progress, 0, sizeof(struct my_progress_mgr));
    progress.pub.progress_monitor = my_progress_monitor;
    progress.this = this;
    dinfo->progress = &progress.pub;
  } else
    dinfo->progress = NULL;

  dinfo->mem->max_memory_to_use = (long)this->maxMemory * 1048576L;

  if ((xinfo =
       (jpeg_transform_info *)malloc(sizeof(jpeg_transform_info) * n)) == NULL)
    THROW("Memory allocation failure");
  memset(xinfo, 0, sizeof(jpeg_transform_info) * n);

  if (setjmp(this->jerr.setjmp_buffer)) {
    /* If we get here, the JPEG code has signaled an error. */
    retval = -1;  goto bailout;
  }

  if (dinfo->global_state <= DSTATE_INHEADER)
    jpeg_mem_src_tj(dinfo, jpegBuf, jpegSize);

  for (i = 0; i < n; i++) {
    if (t[i].op < 0 || t[i].op >= TJ_NUMXOP)
      THROW("Invalid transform operation");
    xinfo[i].transform = xformtypes[t[i].op];
    xinfo[i].perfect = (t[i].options & TJXOPT_PERFECT) ? 1 : 0;
    xinfo[i].trim = (t[i].options & TJXOPT_TRIM) ? 1 : 0;
    xinfo[i].force_grayscale = (t[i].options & TJXOPT_GRAY) ? 1 : 0;
    xinfo[i].crop = (t[i].options & TJXOPT_CROP) ? 1 : 0;
    if (n != 1 && t[i].op == TJXOP_HFLIP) xinfo[i].slow_hflip = 1;
    else xinfo[i].slow_hflip = 0;

    if (xinfo[i].crop) {
      if (t[i].r.x < 0 || t[i].r.y < 0 || t[i].r.w < 0 || t[i].r.h < 0)
        THROW("Invalid cropping region");
      xinfo[i].crop_xoffset = t[i].r.x;  xinfo[i].crop_xoffset_set = JCROP_POS;
      xinfo[i].crop_yoffset = t[i].r.y;  xinfo[i].crop_yoffset_set = JCROP_POS;
      if (t[i].r.w != 0) {
        xinfo[i].crop_width = t[i].r.w;  xinfo[i].crop_width_set = JCROP_POS;
      } else
        xinfo[i].crop_width = JCROP_UNSET;
      if (t[i].r.h != 0) {
        xinfo[i].crop_height = t[i].r.h;  xinfo[i].crop_height_set = JCROP_POS;
      } else
        xinfo[i].crop_height = JCROP_UNSET;
    }
    if (!(t[i].options & TJXOPT_COPYNONE)) saveMarkers = 1;
  }

  jcopy_markers_setup(dinfo, saveMarkers ?
                             (JCOPY_OPTION)this->saveMarkers : JCOPYOPT_NONE);
  if (dinfo->global_state <= DSTATE_INHEADER)
    jpeg_read_header(dinfo, TRUE);
  if (this->maxPixels &&
      (unsigned long long)dinfo->image_width * dinfo->image_height >
      (unsigned long long)this->maxPixels)
    THROW("Image is too large");
  srcSubsamp = getSubsamp(&this->dinfo);

  for (i = 0; i < n; i++) {
    if (!jtransform_request_workspace(dinfo, &xinfo[i]))
      THROW("Transform is not perfect");

    if (xinfo[i].crop) {
      int dstSubsamp = getDstSubsamp(srcSubsamp, &t[i]);

      if (dstSubsamp == TJSAMP_UNKNOWN)
        THROW("Could not determine subsampling level of destination image");
      if ((t[i].r.x % tjMCUWidth[dstSubsamp]) != 0 ||
          (t[i].r.y % tjMCUHeight[dstSubsamp]) != 0)
        THROWI("To crop this JPEG image, x must be a multiple of %d\n"
               "and y must be a multiple of %d.", tjMCUWidth[dstSubsamp],
               tjMCUHeight[dstSubsamp]);
    }
  }

  srccoefs = jpeg_read_coefficients(dinfo);

  for (i = 0; i < n; i++) {
    if (this->noRealloc) alloc = FALSE;
    if (!(t[i].options & TJXOPT_NOOUTPUT))
      jpeg_mem_dest_tj(cinfo, &dstBufs[i], &dstSizes[i], alloc);
    jpeg_copy_critical_parameters(dinfo, cinfo);
    dstcoefs = jtransform_adjust_parameters(dinfo, cinfo, srccoefs, &xinfo[i]);
    if (this->optimize || t[i].options & TJXOPT_OPTIMIZE)
      cinfo->optimize_coding = TRUE;
#ifdef C_PROGRESSIVE_SUPPORTED
    if (this->progressive || t[i].options & TJXOPT_PROGRESSIVE)
      jpeg_simple_progression(cinfo);
#endif
    if (this->arithmetic || t[i].options & TJXOPT_ARITHMETIC) {
      cinfo->arith_code = TRUE;
      cinfo->optimize_coding = FALSE;
    }
    cinfo->restart_interval = this->restartIntervalBlocks;
    cinfo->restart_in_rows = this->restartIntervalRows;
    if (!(t[i].options & TJXOPT_NOOUTPUT)) {
      jpeg_write_coefficients(cinfo, dstcoefs);
      jcopy_markers_execute(dinfo, cinfo, t[i].options & TJXOPT_COPYNONE ?
                                          JCOPYOPT_NONE :
                                          (JCOPY_OPTION)this->saveMarkers);
      if (this->iccBuf != NULL && this->iccSize != 0)
        jpeg_write_icc_profile(cinfo, this->iccBuf,
                               (unsigned int)this->iccSize);
    } else
      jinit_c_master_control(cinfo, TRUE);
    jtransform_execute_transformation(dinfo, cinfo, srccoefs, &xinfo[i]);
    if (t[i].customFilter) {
      int ci, y;
      JDIMENSION by;

      for (ci = 0; ci < cinfo->num_components; ci++) {
        jpeg_component_info *compptr = &cinfo->comp_info[ci];
        tjregion arrayRegion = { 0, 0, 0, 0 };
        tjregion planeRegion = { 0, 0, 0, 0 };

        arrayRegion.w = compptr->width_in_blocks * DCTSIZE;
        arrayRegion.h = DCTSIZE;
        planeRegion.w = compptr->width_in_blocks * DCTSIZE;
        planeRegion.h = compptr->height_in_blocks * DCTSIZE;

        for (by = 0; by < compptr->height_in_blocks;
             by += compptr->v_samp_factor) {
          JBLOCKARRAY barray = (dinfo->mem->access_virt_barray)
            ((j_common_ptr)dinfo, dstcoefs[ci], by, compptr->v_samp_factor,
             TRUE);

          for (y = 0; y < compptr->v_samp_factor; y++) {
            if (t[i].customFilter(barray[y][0], arrayRegion, planeRegion, ci,
                                  i, (tjtransform *)&t[i]) == -1)
              THROW("Error in custom filter");
            arrayRegion.y += DCTSIZE;
          }
        }
      }
    }
    if (!(t[i].options & TJXOPT_NOOUTPUT)) jpeg_finish_compress(cinfo);
  }

  jpeg_finish_decompress(dinfo);

bailout:
  if (cinfo->global_state > CSTATE_START) {
    if (alloc) (*cinfo->dest->term_destination) (cinfo);
    jpeg_abort_compress(cinfo);
  }
  if (dinfo->global_state > DSTATE_START) jpeg_abort_decompress(dinfo);
  free(xinfo);
  if (this->jerr.warning) retval = -1;
  return retval;
}

/* TurboJPEG 1.2+ */
DLLEXPORT int tjTransform(tjhandle handle, const unsigned char *jpegBuf,
                          unsigned long jpegSize, int n,
                          unsigned char **dstBufs, unsigned long *dstSizes,
                          tjtransform *t, int flags)
{
  static const char FUNCTION_NAME[] = "tjTransform";
  int i, retval = 0, srcSubsamp = -1;
  size_t *sizes = NULL;

  GET_DINSTANCE(handle);
  if ((this->init & DECOMPRESS) == 0)
    THROW("Instance has not been initialized for decompression");

  if (n < 1 || dstSizes == NULL)
    THROW("Invalid argument");

  if (setjmp(this->jerr.setjmp_buffer)) {
    /* If we get here, the JPEG code has signaled an error. */
    retval = -1;  goto bailout;
  }

  processFlags(handle, flags, COMPRESS);

  if (this->noRealloc) {
    jpeg_mem_src_tj(dinfo, jpegBuf, jpegSize);
    jpeg_read_header(dinfo, TRUE);
    srcSubsamp = getSubsamp(dinfo);
  }

  if ((sizes = (size_t *)malloc(n * sizeof(size_t))) == NULL)
    THROW("Memory allocation failure");
  for (i = 0; i < n; i++) {
    sizes[i] = (size_t)dstSizes[i];
    if (this->noRealloc) {
      int dstWidth = dinfo->image_width, dstHeight = dinfo->image_height;
      int dstSubsamp = srcSubsamp;

      if (getTransformedSpecs(handle, &dstWidth, &dstHeight, &dstSubsamp,
                              &t[i], FUNCTION_NAME) == -1) {
        retval = -1;
        goto bailout;
      }
      sizes[i] = tj3JPEGBufSize(dstWidth, dstHeight, dstSubsamp);
    }
  }
  retval = tj3Transform(handle, jpegBuf, (size_t)jpegSize, n, dstBufs, sizes,
                        t);
  for (i = 0; i < n; i++)
    dstSizes[i] = (unsigned long)sizes[i];

bailout:
  if (dinfo->global_state > DSTATE_START) jpeg_abort_decompress(dinfo);
  if (this->jerr.warning) retval = -1;
  free(sizes);
  return retval;
}


/*************************** Packed-Pixel Image I/O **************************/

/* tj3LoadImage*() is implemented in turbojpeg-mp.c */

/* TurboJPEG 2.0+ */
DLLEXPORT unsigned char *tjLoadImage(const char *filename, int *width,
                                     int align, int *height,
                                     int *pixelFormat, int flags)
{
  tjhandle handle = NULL;
  unsigned char *dstBuf = NULL;

  if ((handle = tj3Init(TJINIT_COMPRESS)) == NULL) return NULL;

  processFlags(handle, flags, COMPRESS);

  dstBuf = tj3LoadImage8(handle, filename, width, align, height, pixelFormat);

  tj3Destroy(handle);
  return dstBuf;
}


/* tj3SaveImage*() is implemented in turbojpeg-mp.c */

/* TurboJPEG 2.0+ */
DLLEXPORT int tjSaveImage(const char *filename, unsigned char *buffer,
                          int width, int pitch, int height, int pixelFormat,
                          int flags)
{
  tjhandle handle = NULL;
  int retval = -1;

  if ((handle = tj3Init(TJINIT_DECOMPRESS)) == NULL) return -1;

  processFlags(handle, flags, DECOMPRESS);

  retval = tj3SaveImage8(handle, filename, buffer, width, pitch, height,
                         pixelFormat);

  tj3Destroy(handle);
  return retval;
}
