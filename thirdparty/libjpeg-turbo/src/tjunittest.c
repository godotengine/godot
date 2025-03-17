/*
 * Copyright (C)2009-2014, 2017-2019, 2022-2024 D. R. Commander.
 *                                              All Rights Reserved.
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

/*
 * This program tests the various code paths in the TurboJPEG C Wrapper
 */

#ifdef _MSC_VER
#define _CRT_SECURE_NO_DEPRECATE
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <errno.h>
#include "tjutil.h"
#include "turbojpeg.h"
#include "md5/md5.h"
#include "jconfigint.h"
#ifdef _WIN32
#include <time.h>
#include <process.h>
#define random()  rand()
#define getpid()  _getpid()
#else
#include <unistd.h>
#endif


static void usage(char *progName)
{
  printf("\nUSAGE: %s [options]\n\n", progName);
  printf("Options:\n");
  printf("-yuv = test YUV encoding/compression/decompression/decoding\n");
  printf("       (8-bit data precision only)\n");
  printf("-noyuvpad = do not pad each row in each Y, U, and V plane to the nearest\n");
  printf("            multiple of 4 bytes\n");
  printf("-precision N = test N-bit data precision (N=2..16; default is 8; if N is not 8\n");
  printf("               or 12, then -lossless is implied)\n");
  printf("-lossless = test lossless JPEG compression/decompression\n");
  printf("-alloc = test automatic JPEG buffer allocation\n");
  printf("-bmp = test packed-pixel image I/O\n");
  exit(1);
}


#define THROW_TJ(handle) { \
  printf("TurboJPEG ERROR:\n%s\n", tj3GetErrorStr(handle)); \
  BAILOUT() \
}
#define TRY_TJ(handle, f) { if ((f) == -1) THROW_TJ(handle); }
#define THROW(m) { printf("ERROR: %s\n", m);  BAILOUT() }
#define THROW_MD5(filename, md5sum, ref) { \
  printf("\n%s has an MD5 sum of %s.\n   Should be %s.\n", filename, md5sum, \
         ref); \
  BAILOUT() \
}

static const char *subNameLong[TJ_NUMSAMP] = {
  "4:4:4", "4:2:2", "4:2:0", "GRAY", "4:4:0", "4:1:1", "4:4:1"
};
static const char *subName[TJ_NUMSAMP] = {
  "444", "422", "420", "GRAY", "440", "411", "441"
};

static const char *pixFormatStr[TJ_NUMPF] = {
  "RGB", "BGR", "RGBX", "BGRX", "XBGR", "XRGB", "Grayscale",
  "RGBA", "BGRA", "ABGR", "ARGB", "CMYK"
};

static const int _3sampleFormats[] = { TJPF_RGB, TJPF_BGR };
static const int _4sampleFormats[] = {
  TJPF_RGBX, TJPF_BGRX, TJPF_XBGR, TJPF_XRGB, TJPF_CMYK
};
static const int _onlyGray[] = { TJPF_GRAY };
static const int _onlyRGB[] = { TJPF_RGB };

static int doYUV = 0, lossless = 0, psv = 1, alloc = 0, yuvAlign = 4;
static int precision = 8, sampleSize, maxSample, tolerance, redToY, yellowToY;

static int exitStatus = 0;
#define BAILOUT() { exitStatus = -1;  goto bailout; }


static void setVal(void *buf, int index, int value)
{
  if (precision <= 8)
    ((unsigned char *)buf)[index] = (unsigned char)value;
  else if (precision <= 12)
    ((short *)buf)[index] = (short)value;
  else
    ((unsigned short *)buf)[index] = (unsigned short)value;
}

static void initBuf(void *buf, int w, int h, int pf, int bottomUp)
{
  int roffset = tjRedOffset[pf];
  int goffset = tjGreenOffset[pf];
  int boffset = tjBlueOffset[pf];
  int ps = tjPixelSize[pf];
  int i, index, row, col, halfway = 16;

  if (pf == TJPF_GRAY) {
    memset(buf, 0, w * h * ps * sampleSize);
    for (row = 0; row < h; row++) {
      for (col = 0; col < w; col++) {
        if (bottomUp) index = (h - row - 1) * w + col;
        else index = row * w + col;
        if (((row / 8) + (col / 8)) % 2 == 0)
          setVal(buf, index, (row < halfway) ? maxSample : 0);
        else setVal(buf, index, (row < halfway) ? redToY : yellowToY);
      }
    }
  } else if (pf == TJPF_CMYK) {
    for (i = 0; i < w * h * ps; i++)
      setVal(buf, i, maxSample);
    for (row = 0; row < h; row++) {
      for (col = 0; col < w; col++) {
        if (bottomUp) index = (h - row - 1) * w + col;
        else index = row * w + col;
        if (((row / 8) + (col / 8)) % 2 == 0) {
          if (row >= halfway) setVal(buf, index * ps + 3, 0);
        } else {
          setVal(buf, index * ps + 2, 0);
          if (row < halfway) setVal(buf, index * ps + 1, 0);
        }
      }
    }
  } else {
    memset(buf, 0, w * h * ps * sampleSize);
    for (row = 0; row < h; row++) {
      for (col = 0; col < w; col++) {
        if (bottomUp) index = (h - row - 1) * w + col;
        else index = row * w + col;
        if (((row / 8) + (col / 8)) % 2 == 0) {
          if (row < halfway) {
            setVal(buf, index * ps + roffset, maxSample);
            setVal(buf, index * ps + goffset, maxSample);
            setVal(buf, index * ps + boffset, maxSample);
          }
        } else {
          setVal(buf, index * ps + roffset, maxSample);
          if (row >= halfway) setVal(buf, index * ps + goffset, maxSample);
        }
      }
    }
  }
}


#define CHECKVAL(v, cv) { \
  if (v < cv - tolerance || v > cv + tolerance) { \
    printf("\nComp. %s at %d,%d should be %d, not %d\n", #v, row, col, cv, \
           v); \
    retval = 0;  exitStatus = -1;  goto bailout; \
  } \
}

#define CHECKVAL0(v) { \
  if (v > tolerance) { \
    printf("\nComp. %s at %d,%d should be 0, not %d\n", #v, row, col, v); \
    retval = 0;  exitStatus = -1;  goto bailout; \
  } \
}

#define CHECKVALMAX(v) { \
  if (v < maxSample - tolerance) { \
    printf("\nComp. %s at %d,%d should be %d, not %d\n", #v, row, col, \
           maxSample, v); \
    retval = 0;  exitStatus = -1;  goto bailout; \
  } \
}


static int getVal(void *buf, int index)
{
  if (precision <= 8)
    return ((unsigned char *)buf)[index];
  else if (precision <= 12)
    return ((short *)buf)[index];
  else
    return ((unsigned short *)buf)[index];
}

static int checkBuf(void *buf, int w, int h, int pf,  int subsamp,
                    tjscalingfactor sf, int bottomUp)
{
  int roffset = tjRedOffset[pf];
  int goffset = tjGreenOffset[pf];
  int boffset = tjBlueOffset[pf];
  int aoffset = tjAlphaOffset[pf];
  int ps = tjPixelSize[pf];
  int index, row, col, retval = 1;
  int halfway = 16 * sf.num / sf.denom;
  int blocksize = 8 * sf.num / sf.denom;

  if (pf == TJPF_GRAY) roffset = goffset = boffset = 0;

  if (pf == TJPF_CMYK) {
    for (row = 0; row < h; row++) {
      for (col = 0; col < w; col++) {
        int c, m, y, k;

        if (bottomUp) index = (h - row - 1) * w + col;
        else index = row * w + col;
        c = getVal(buf, index * ps);
        m = getVal(buf, index * ps + 1);
        y = getVal(buf, index * ps + 2);
        k = getVal(buf, index * ps + 3);
        if (((row / blocksize) + (col / blocksize)) % 2 == 0) {
          CHECKVALMAX(c);  CHECKVALMAX(m);  CHECKVALMAX(y);
          if (row < halfway) CHECKVALMAX(k)
          else CHECKVAL0(k)
        } else {
          CHECKVALMAX(c);  CHECKVAL0(y);  CHECKVALMAX(k);
          if (row < halfway) CHECKVAL0(m)
          else CHECKVALMAX(m)
        }
      }
    }
    return 1;
  }

  for (row = 0; row < h; row++) {
    for (col = 0; col < w; col++) {
      int r, g, b, a;

      if (bottomUp) index = (h - row - 1) * w + col;
      else index = row * w + col;
      r = getVal(buf, index * ps + roffset);
      g = getVal(buf, index * ps + goffset);
      b = getVal(buf, index * ps + boffset);
      a = aoffset >= 0 ? getVal(buf, index * ps + aoffset) : maxSample;
      if (((row / blocksize) + (col / blocksize)) % 2 == 0) {
        if (row < halfway) {
          CHECKVALMAX(r);  CHECKVALMAX(g);  CHECKVALMAX(b);
        } else {
          CHECKVAL0(r);  CHECKVAL0(g);  CHECKVAL0(b);
        }
      } else {
        if (subsamp == TJSAMP_GRAY) {
          if (row < halfway) {
            CHECKVAL(r, redToY);  CHECKVAL(g, redToY);  CHECKVAL(b, redToY);
          } else {
            CHECKVAL(r, yellowToY);  CHECKVAL(g, yellowToY);
            CHECKVAL(b, yellowToY);
          }
        } else {
          if (row < halfway) {
            CHECKVALMAX(r);  CHECKVAL0(g);  CHECKVAL0(b);
          } else {
            CHECKVALMAX(r);  CHECKVALMAX(g);  CHECKVAL0(b);
          }
        }
      }
      CHECKVALMAX(a);
    }
  }

bailout:
  if (retval == 0) {
    for (row = 0; row < h; row++) {
      for (col = 0; col < w; col++) {
        if (pf == TJPF_CMYK)
          printf("%.3d/%.3d/%.3d/%.3d ", getVal(buf, (row * w + col) * ps),
                 getVal(buf, (row * w + col) * ps + 1),
                 getVal(buf, (row * w + col) * ps + 2),
                 getVal(buf, (row * w + col) * ps + 3));
        else
          printf("%.3d/%.3d/%.3d ",
                 getVal(buf, (row * w + col) * ps + roffset),
                 getVal(buf, (row * w + col) * ps + goffset),
                 getVal(buf, (row * w + col) * ps + boffset));
      }
      printf("\n");
    }
  }
  return retval;
}


#define PAD(v, p)  ((v + (p) - 1) & (~((p) - 1)))

static int checkBufYUV(unsigned char *buf, int w, int h, int subsamp,
                       tjscalingfactor sf)
{
  int row, col;
  int hsf = tjMCUWidth[subsamp] / 8, vsf = tjMCUHeight[subsamp] / 8;
  int pw = PAD(w, hsf), ph = PAD(h, vsf);
  int cw = pw / hsf, ch = ph / vsf;
  int ypitch = PAD(pw, yuvAlign), uvpitch = PAD(cw, yuvAlign);
  int retval = 1;
  int halfway = 16 * sf.num / sf.denom;
  int blocksize = 8 * sf.num / sf.denom;

  for (row = 0; row < ph; row++) {
    for (col = 0; col < pw; col++) {
      unsigned char y = buf[ypitch * row + col];

      if (((row / blocksize) + (col / blocksize)) % 2 == 0) {
        if (row < halfway) CHECKVALMAX(y)
        else CHECKVAL0(y);
      } else {
        if (row < halfway) CHECKVAL(y, 76)
        else CHECKVAL(y, 225);
      }
    }
  }
  if (subsamp != TJSAMP_GRAY) {
    halfway = 16 / vsf * sf.num / sf.denom;

    for (row = 0; row < ch; row++) {
      for (col = 0; col < cw; col++) {
        unsigned char u = buf[ypitch * ph + (uvpitch * row + col)],
          v = buf[ypitch * ph + uvpitch * ch + (uvpitch * row + col)];

        if (((row * vsf / blocksize) + (col * hsf / blocksize)) % 2 == 0) {
          CHECKVAL(u, 128);  CHECKVAL(v, 128);
        } else {
          if (row < halfway) {
            CHECKVAL(u, 85);  CHECKVALMAX(v);
          } else {
            CHECKVAL0(u);  CHECKVAL(v, 149);
          }
        }
      }
    }
  }

bailout:
  if (retval == 0) {
    for (row = 0; row < ph; row++) {
      for (col = 0; col < pw; col++)
        printf("%.3d ", buf[ypitch * row + col]);
      printf("\n");
    }
    printf("\n");
    for (row = 0; row < ch; row++) {
      for (col = 0; col < cw; col++)
        printf("%.3d ", buf[ypitch * ph + (uvpitch * row + col)]);
      printf("\n");
    }
    printf("\n");
    for (row = 0; row < ch; row++) {
      for (col = 0; col < cw; col++)
        printf("%.3d ",
               buf[ypitch * ph + uvpitch * ch + (uvpitch * row + col)]);
      printf("\n");
    }
  }

  return retval;
}


static void writeJPEG(unsigned char *jpegBuf, size_t jpegSize, char *filename)
{
  FILE *file = fopen(filename, "wb");

  if (!file || fwrite(jpegBuf, jpegSize, 1, file) != 1) {
    printf("ERROR: Could not write to %s.\n%s\n", filename, strerror(errno));
    BAILOUT()
  }

bailout:
  if (file) fclose(file);
}


static void compTest(tjhandle handle, unsigned char **dstBuf, size_t *dstSize,
                     int w, int h, int pf, char *basename)
{
  char tempStr[1024];
  void *srcBuf = NULL;
  unsigned char *yuvBuf = NULL;
  const char *pfStr = pixFormatStr[pf];
  int bottomUp = tj3Get(handle, TJPARAM_BOTTOMUP);
  int subsamp = tj3Get(handle, TJPARAM_SUBSAMP);
  int jpegPSV = tj3Get(handle, TJPARAM_LOSSLESSPSV);
  int jpegQual = tj3Get(handle, TJPARAM_QUALITY);
  const char *buStrLong = bottomUp ? "Bottom-Up" : "Top-Down ";
  const char *buStr = bottomUp ? "BU" : "TD";

  if ((srcBuf = malloc(w * h * tjPixelSize[pf] * sampleSize)) == NULL)
      THROW("Memory allocation failure");
  initBuf(srcBuf, w, h, pf, bottomUp);

  if (*dstBuf && *dstSize > 0) memset(*dstBuf, 0, *dstSize);

  if (doYUV) {
    size_t yuvSize = tj3YUVBufSize(w, yuvAlign, h, subsamp);
    tjscalingfactor sf = { 1, 1 };
    tjhandle handle2 = NULL;

    if ((handle2 = tj3Init(TJINIT_COMPRESS)) == NULL)
      THROW_TJ(NULL);
    TRY_TJ(handle2, tj3Set(handle2, TJPARAM_BOTTOMUP, bottomUp));
    TRY_TJ(handle2, tj3Set(handle2, TJPARAM_SUBSAMP, subsamp));

    if ((yuvBuf = (unsigned char *)malloc(yuvSize)) == NULL)
      THROW("Memory allocation failure");
    memset(yuvBuf, 0, yuvSize);

    printf("%s %s -> YUV %s ... ", pfStr, buStrLong, subNameLong[subsamp]);
    TRY_TJ(handle2, tj3EncodeYUV8(handle2, (unsigned char *)srcBuf, w, 0, h,
                                  pf, yuvBuf, yuvAlign));
    tj3Destroy(handle2);
    if (checkBufYUV(yuvBuf, w, h, subsamp, sf)) printf("Passed.\n");
    else printf("FAILED!\n");

    printf("YUV %s %s -> JPEG Q%d ... ", subNameLong[subsamp], buStrLong,
           jpegQual);
    TRY_TJ(handle, tj3CompressFromYUV8(handle, yuvBuf, w, yuvAlign, h, dstBuf,
                                       dstSize));
  } else {
    if (lossless) {
      TRY_TJ(handle, tj3Set(handle, TJPARAM_PRECISION, precision));
      printf("%s %s -> LOSSLESS PSV%d ... ", pfStr, buStrLong, jpegPSV);
    } else
      printf("%s %s -> %s Q%d ... ", pfStr, buStrLong, subNameLong[subsamp],
             jpegQual);
    if (precision <= 8) {
      TRY_TJ(handle, tj3Compress8(handle, (unsigned char *)srcBuf, w, 0, h, pf,
                                  dstBuf, dstSize));
    } else if (precision <= 12) {
      TRY_TJ(handle, tj3Compress12(handle, (short *)srcBuf, w, 0, h, pf,
                                   dstBuf, dstSize));
    } else {
      TRY_TJ(handle, tj3Compress16(handle, (unsigned short *)srcBuf, w, 0, h,
                                   pf, dstBuf, dstSize));
    }
  }

  if (lossless)
    SNPRINTF(tempStr, 1024, "%s_enc%d_%s_%s_LOSSLESS_PSV%d.jpg", basename,
             precision, pfStr, buStr, jpegPSV);
  else
    SNPRINTF(tempStr, 1024, "%s_enc%d_%s_%s_%s_Q%d.jpg", basename, precision,
             pfStr, buStr, subName[subsamp], jpegQual);
  writeJPEG(*dstBuf, *dstSize, tempStr);
  printf("Done.\n  Result in %s\n", tempStr);

bailout:
  free(yuvBuf);
  free(srcBuf);
}


static void _decompTest(tjhandle handle, unsigned char *jpegBuf,
                        size_t jpegSize, int w, int h, int pf, char *basename,
                        int subsamp, tjscalingfactor sf)
{
  void *dstBuf = NULL;
  unsigned char *yuvBuf = NULL;
  int _hdrw = 0, _hdrh = 0, _hdrsubsamp;
  int scaledWidth = TJSCALED(w, sf);
  int scaledHeight = TJSCALED(h, sf);
  size_t dstSize = 0;
  int bottomUp = tj3Get(handle, TJPARAM_BOTTOMUP);

  TRY_TJ(handle, tj3SetScalingFactor(handle, sf));

  TRY_TJ(handle, tj3DecompressHeader(handle, jpegBuf, jpegSize));
  _hdrw = tj3Get(handle, TJPARAM_JPEGWIDTH);
  _hdrh = tj3Get(handle, TJPARAM_JPEGHEIGHT);
  _hdrsubsamp = tj3Get(handle, TJPARAM_SUBSAMP);
  if (lossless && subsamp != TJSAMP_444 && subsamp != TJSAMP_GRAY)
    subsamp = TJSAMP_444;
  if (_hdrw != w || _hdrh != h || _hdrsubsamp != subsamp)
    THROW("Incorrect JPEG header");

  dstSize = scaledWidth * scaledHeight * tjPixelSize[pf];
  if ((dstBuf = malloc(dstSize * sampleSize)) == NULL)
    THROW("Memory allocation failure");
  memset(dstBuf, 0, dstSize * sampleSize);

  if (doYUV) {
    size_t yuvSize = tj3YUVBufSize(scaledWidth, yuvAlign, scaledHeight,
                                   subsamp);
    tjhandle handle2 = NULL;

    if ((handle2 = tj3Init(TJINIT_DECOMPRESS)) == NULL)
      THROW_TJ(NULL);
    TRY_TJ(handle2, tj3Set(handle2, TJPARAM_BOTTOMUP, bottomUp));
    TRY_TJ(handle2, tj3Set(handle2, TJPARAM_SUBSAMP, subsamp));

    if ((yuvBuf = (unsigned char *)malloc(yuvSize)) == NULL)
      THROW("Memory allocation failure");
    memset(yuvBuf, 0, yuvSize);

    printf("JPEG -> YUV %s ", subNameLong[subsamp]);
    if (sf.num != 1 || sf.denom != 1)
      printf("%d/%d ... ", sf.num, sf.denom);
    else printf("... ");
    TRY_TJ(handle, tj3DecompressToYUV8(handle, jpegBuf, jpegSize, yuvBuf,
                                       yuvAlign));
    if (checkBufYUV(yuvBuf, scaledWidth, scaledHeight, subsamp, sf))
      printf("Passed.\n");
    else printf("FAILED!\n");

    printf("YUV %s -> %s %s ... ", subNameLong[subsamp], pixFormatStr[pf],
           bottomUp ? "Bottom-Up" : "Top-Down ");
    TRY_TJ(handle2, tj3DecodeYUV8(handle2, yuvBuf, yuvAlign,
                                  (unsigned char *)dstBuf, scaledWidth, 0,
                                  scaledHeight, pf));
    tj3Destroy(handle2);
  } else {
    printf("JPEG -> %s %s ", pixFormatStr[pf],
           bottomUp ? "Bottom-Up" : "Top-Down ");
    if (sf.num != 1 || sf.denom != 1)
      printf("%d/%d ... ", sf.num, sf.denom);
    else printf("... ");
    if (precision <= 8) {
      TRY_TJ(handle, tj3Decompress8(handle, jpegBuf, jpegSize,
                                    (unsigned char *)dstBuf, 0, pf));
    } else if (precision <= 12) {
      TRY_TJ(handle, tj3Decompress12(handle, jpegBuf, jpegSize,
                                     (short *)dstBuf, 0, pf));
    } else {
      TRY_TJ(handle, tj3Decompress16(handle, jpegBuf, jpegSize,
                                     (unsigned short *)dstBuf, 0, pf));
    }
  }

  if (checkBuf(dstBuf, scaledWidth, scaledHeight, pf, subsamp, sf, bottomUp))
    printf("Passed.");
  else printf("FAILED!");
  printf("\n");

bailout:
  free(yuvBuf);
  free(dstBuf);
}


static void decompTest(tjhandle handle, unsigned char *jpegBuf,
                       size_t jpegSize, int w, int h, int pf, char *basename,
                       int subsamp)
{
  int i, n = 0;
  tjscalingfactor *sf = NULL;

  if (lossless) {
    _decompTest(handle, jpegBuf, jpegSize, w, h, pf, basename, subsamp,
                TJUNSCALED);
    return;
  }

  sf = tj3GetScalingFactors(&n);
  if (!sf || !n) THROW_TJ(NULL);

  for (i = 0; i < n; i++) {
    if (subsamp == TJSAMP_444 || subsamp == TJSAMP_GRAY ||
        ((subsamp == TJSAMP_411 || subsamp == TJSAMP_441) && sf[i].num == 1 &&
         (sf[i].denom == 2 || sf[i].denom == 1)) ||
        (subsamp != TJSAMP_411 && subsamp != TJSAMP_441 && sf[i].num == 1 &&
         (sf[i].denom == 4 || sf[i].denom == 2 || sf[i].denom == 1)))
      _decompTest(handle, jpegBuf, jpegSize, w, h, pf, basename, subsamp,
                  sf[i]);
  }

bailout:
  return;
}


static void doTest(int w, int h, const int *formats, int nformats, int subsamp,
                   char *basename)
{
  tjhandle chandle = NULL, dhandle = NULL;
  unsigned char *dstBuf = NULL;
  size_t size = 0, bufSize = 0;
  int pfi, pf, i;

  if (lossless && subsamp != TJSAMP_GRAY)
    subsamp = TJSAMP_444;

  if (!alloc) {
    size = bufSize = tj3JPEGBufSize(w, h, subsamp);
    if (size == 0)
      THROW_TJ(NULL);
    if ((dstBuf = (unsigned char *)tj3Alloc(size)) == NULL)
      THROW("Memory allocation failure.");
  }

  if ((chandle = tj3Init(TJINIT_COMPRESS)) == NULL ||
      (dhandle = tj3Init(TJINIT_DECOMPRESS)) == NULL)
    THROW_TJ(NULL);

  TRY_TJ(chandle, tj3Set(chandle, TJPARAM_NOREALLOC, !alloc));
  if (lossless) {
    TRY_TJ(chandle, tj3Set(chandle, TJPARAM_LOSSLESS, lossless));
    TRY_TJ(chandle, tj3Set(chandle, TJPARAM_LOSSLESSPSV,
                           ((psv++ - 1) % 7) + 1));
  } else {
    TRY_TJ(chandle, tj3Set(chandle, TJPARAM_QUALITY, 100));
    if (subsamp == TJSAMP_422 || subsamp == TJSAMP_420 ||
        subsamp == TJSAMP_440 || subsamp == TJSAMP_411 ||
        subsamp == TJSAMP_441)
      TRY_TJ(dhandle, tj3Set(dhandle, TJPARAM_FASTUPSAMPLE, 1));
  }
  TRY_TJ(chandle, tj3Set(chandle, TJPARAM_SUBSAMP, subsamp));

  for (pfi = 0; pfi < nformats; pfi++) {
    for (i = 0; i < 2; i++) {
      TRY_TJ(chandle, tj3Set(chandle, TJPARAM_BOTTOMUP, i == 1));
      TRY_TJ(dhandle, tj3Set(dhandle, TJPARAM_BOTTOMUP, i == 1));
      pf = formats[pfi];
      if (!alloc) size = bufSize;
      compTest(chandle, &dstBuf, &size, w, h, pf, basename);
      decompTest(dhandle, dstBuf, size, w, h, pf, basename, subsamp);
      if (pf >= TJPF_RGBX && pf <= TJPF_XRGB) {
        printf("\n");
        decompTest(dhandle, dstBuf, size, w, h, pf + (TJPF_RGBA - TJPF_RGBX),
                   basename, subsamp);
      }
      printf("\n");
    }
  }
  printf("--------------------\n\n");

bailout:
  tj3Destroy(chandle);
  tj3Destroy(dhandle);
  tj3Free(dstBuf);
}


#if SIZEOF_SIZE_T == 8
#define CHECKSIZE(function) { \
  if (size && size < (size_t)0xFFFFFFFF) \
    THROW(#function " overflow"); \
}
#define CHECKSIZEUL(function) { \
  if ((unsigned long long)ulsize < (unsigned long long)0xFFFFFFFF) \
    THROW(#function " overflow"); \
}
#else
#define CHECKSIZE(function) { \
  if (size != 0 || !strcmp(tj3GetErrorStr(NULL), "No error")) \
    THROW(#function " overflow"); \
}
#define CHECKSIZEUL(function) { \
  if (ulsize != (unsigned long)(-1) || \
      !strcmp(tj3GetErrorStr(NULL), "No error")) \
    THROW(#function " overflow"); \
}
#endif
#define CHECKSIZEINT(function) { \
  if (intsize != 0 || !strcmp(tj3GetErrorStr(NULL), "No error")) \
    THROW(#function " overflow"); \
}

static void overflowTest(void)
{
  /* Ensure that the various buffer size functions don't overflow */
  size_t size;
  unsigned long ulsize;
  int intsize;

  size = tj3JPEGBufSize(26755, 26755, TJSAMP_444);
  CHECKSIZE(tj3JPEGBufSize());
  ulsize = tjBufSize(26755, 26755, TJSAMP_444);
  CHECKSIZEUL(tjBufSize());
  ulsize = TJBUFSIZE(26755, 26755);
  CHECKSIZEUL(TJBUFSIZE());
  size = tj3YUVBufSize(37838, 1, 37838, TJSAMP_444);
  CHECKSIZE(tj3YUVBufSize());
  size = tj3YUVBufSize(37837, 3, 37837, TJSAMP_444);
  CHECKSIZE(tj3YUVBufSize());
  size = tj3YUVBufSize(37837, -1, 37837, TJSAMP_444);
  CHECKSIZE(tj3YUVBufSize());
  ulsize = tjBufSizeYUV2(37838, 1, 37838, TJSAMP_444);
  CHECKSIZEUL(tjBufSizeYUV2());
  ulsize = tjBufSizeYUV2(37837, 3, 37837, TJSAMP_444);
  CHECKSIZEUL(tjBufSizeYUV2());
  ulsize = tjBufSizeYUV2(37837, -1, 37837, TJSAMP_444);
  CHECKSIZEUL(tjBufSizeYUV2());
  ulsize = TJBUFSIZEYUV(37838, 37838, TJSAMP_444);
  CHECKSIZEUL(TJBUFSIZEYUV());
  ulsize = tjBufSizeYUV(37838, 37838, TJSAMP_444);
  CHECKSIZEUL(tjBufSizeYUV());
  size = tj3YUVPlaneSize(0, 65536, 0, 65536, TJSAMP_444);
  CHECKSIZE(tj3YUVPlaneSize());
  ulsize = tjPlaneSizeYUV(0, 65536, 0, 65536, TJSAMP_444);
  CHECKSIZEUL(tjPlaneSizeYUV());
  intsize = tj3YUVPlaneWidth(0, INT_MAX, TJSAMP_420);
  CHECKSIZEINT(tj3YUVPlaneWidth());
  intsize = tj3YUVPlaneHeight(0, INT_MAX, TJSAMP_420);
  CHECKSIZEINT(tj3YUVPlaneHeight());

bailout:
  return;
}


static void bufSizeTest(void)
{
  int w, h, i, subsamp;
  void *srcBuf = NULL;
  unsigned char *dstBuf = NULL;
  tjhandle handle = NULL;
  size_t dstSize = 0;
  int numSamp = TJ_NUMSAMP;

  if ((handle = tj3Init(TJINIT_COMPRESS)) == NULL)
    THROW_TJ(NULL);

  TRY_TJ(handle, tj3Set(handle, TJPARAM_NOREALLOC, !alloc));
  if (lossless) {
    TRY_TJ(handle, tj3Set(handle, TJPARAM_PRECISION, precision));
    TRY_TJ(handle, tj3Set(handle, TJPARAM_LOSSLESS, lossless));
    TRY_TJ(handle, tj3Set(handle, TJPARAM_LOSSLESSPSV,
                          ((psv++ - 1) % 7) + 1));
    numSamp = 1;
  } else
    TRY_TJ(handle, tj3Set(handle, TJPARAM_QUALITY, 100));

  printf("Buffer size regression test\n");
  for (subsamp = 0; subsamp < numSamp; subsamp++) {
    TRY_TJ(handle, tj3Set(handle, TJPARAM_SUBSAMP, subsamp));
    for (w = 1; w < 48; w++) {
      int maxh = (w == 1) ? 2048 : 48;

      for (h = 1; h < maxh; h++) {
        if (h % 100 == 0) printf("%.4d x %.4d\b\b\b\b\b\b\b\b\b\b\b", w, h);
        if ((srcBuf = malloc(w * h * 4 * sampleSize)) == NULL)
          THROW("Memory allocation failure");
        if (!alloc || doYUV) {
          if (doYUV) dstSize = tj3YUVBufSize(w, yuvAlign, h, subsamp);
          else dstSize = tj3JPEGBufSize(w, h, subsamp);
          if ((dstBuf = (unsigned char *)tj3Alloc(dstSize)) == NULL)
            THROW("Memory allocation failure");
        }

        for (i = 0; i < w * h * 4; i++) {
          if (random() < RAND_MAX / 2) setVal(srcBuf, i, 0);
          else setVal(srcBuf, i, maxSample);
        }

        if (doYUV) {
          TRY_TJ(handle, tj3EncodeYUV8(handle, (unsigned char *)srcBuf, w, 0,
                                       h, TJPF_BGRX, dstBuf, yuvAlign));
        } else {
          if (precision <= 8) {
            TRY_TJ(handle, tj3Compress8(handle, (unsigned char *)srcBuf, w, 0,
                                        h, TJPF_BGRX, &dstBuf, &dstSize));
          } else if (precision <= 12) {
            TRY_TJ(handle, tj3Compress12(handle, (short *)srcBuf, w, 0, h,
                                         TJPF_BGRX, &dstBuf, &dstSize));
          } else {
            TRY_TJ(handle, tj3Compress16(handle, (unsigned short *)srcBuf, w,
                                         0, h, TJPF_BGRX, &dstBuf, &dstSize));
          }
        }
        free(srcBuf);  srcBuf = NULL;
        if (!alloc || doYUV) {
          tj3Free(dstBuf);  dstBuf = NULL;
        }

        if ((srcBuf = malloc(h * w * 4 * sampleSize)) == NULL)
          THROW("Memory allocation failure");
        if (!alloc || doYUV) {
          if (doYUV) dstSize = tj3YUVBufSize(h, yuvAlign, w, subsamp);
          else dstSize = tj3JPEGBufSize(h, w, subsamp);
          if ((dstBuf = (unsigned char *)tj3Alloc(dstSize)) == NULL)
            THROW("Memory allocation failure");
        }

        for (i = 0; i < h * w * 4; i++) {
          if (random() < RAND_MAX / 2) setVal(srcBuf, i, 0);
          else setVal(srcBuf, i, maxSample);
        }

        if (doYUV) {
          TRY_TJ(handle, tj3EncodeYUV8(handle, (unsigned char *)srcBuf, h, 0,
                                       w, TJPF_BGRX, dstBuf, yuvAlign));
        } else {
          if (precision <= 8) {
            TRY_TJ(handle, tj3Compress8(handle, (unsigned char *)srcBuf, h, 0,
                                        w, TJPF_BGRX, &dstBuf, &dstSize));
          } else if (precision <= 12) {
            TRY_TJ(handle, tj3Compress12(handle, (short *)srcBuf, h, 0, w,
                                         TJPF_BGRX, &dstBuf, &dstSize));
          } else {
            TRY_TJ(handle, tj3Compress16(handle, (unsigned short *)srcBuf, h,
                                         0, w, TJPF_BGRX, &dstBuf, &dstSize));
          }
        }
        free(srcBuf);  srcBuf = NULL;
        if (!alloc || doYUV) {
          tj3Free(dstBuf);  dstBuf = NULL;
        }
      }
    }
  }
  printf("Done.      \n");

bailout:
  free(srcBuf);
  tj3Free(dstBuf);
  tj3Destroy(handle);
}


static void rgb_to_cmyk(int r, int g, int b, int *c, int *m, int *y, int *k)
{
  double ctmp = 1.0 - ((double)r / (double)maxSample);
  double mtmp = 1.0 - ((double)g / (double)maxSample);
  double ytmp = 1.0 - ((double)b / (double)maxSample);
  double ktmp = min(min(ctmp, mtmp), ytmp);

  if (ktmp == 1.0) ctmp = mtmp = ytmp = 0.0;
  else {
    ctmp = (ctmp - ktmp) / (1.0 - ktmp);
    mtmp = (mtmp - ktmp) / (1.0 - ktmp);
    ytmp = (ytmp - ktmp) / (1.0 - ktmp);
  }
  *c = (int)((double)maxSample - ctmp * (double)maxSample + 0.5);
  *m = (int)((double)maxSample - mtmp * (double)maxSample + 0.5);
  *y = (int)((double)maxSample - ytmp * (double)maxSample + 0.5);
  *k = (int)((double)maxSample - ktmp * (double)maxSample + 0.5);
}

static void initBitmap(void *buf, int width, int pitch, int height, int pf,
                       int bottomUp)
{
  int roffset = tjRedOffset[pf];
  int goffset = tjGreenOffset[pf];
  int boffset = tjBlueOffset[pf];
  int ps = tjPixelSize[pf];
  int i, j, ci;

  for (j = 0; j < height; j++) {
    int row = bottomUp ? height - j - 1 : j;

    for (i = 0; i < width; i++) {
      int r = (i * (maxSample + 1) / width) % (maxSample + 1);
      int g = (j * (maxSample + 1) / height) % (maxSample + 1);
      int b = (j * (maxSample + 1) / height +
               i * (maxSample + 1) / width) % (maxSample + 1);

      for (ci = 0; ci < ps; ci++)
        setVal(buf, row * pitch + i * ps + ci, 0);
      if (pf == TJPF_GRAY) setVal(buf, row * pitch + i * ps, b);
      else if (pf == TJPF_CMYK) {
        int c, m, y, k;

        rgb_to_cmyk(r, g, b, &c, &m, &y, &k);
        setVal(buf, row * pitch + i * ps + 0, c);
        setVal(buf, row * pitch + i * ps + 1, m);
        setVal(buf, row * pitch + i * ps + 2, y);
        setVal(buf, row * pitch + i * ps + 3, k);
      } else {
        setVal(buf, row * pitch + i * ps + roffset, r);
        setVal(buf, row * pitch + i * ps + goffset, g);
        setVal(buf, row * pitch + i * ps + boffset, b);
      }
    }
  }
}


static void cmyk_to_rgb(int c, int m, int y, int k, int *r, int *g, int *b)
{
  *r = (int)((double)c * (double)k / (double)maxSample + 0.5);
  *g = (int)((double)m * (double)k / (double)maxSample + 0.5);
  *b = (int)((double)y * (double)k / (double)maxSample + 0.5);
}

static int cmpBitmap(void *buf, int width, int pitch, int height, int pf,
                     int bottomUp, int gray2rgb)
{
  int roffset = tjRedOffset[pf];
  int goffset = tjGreenOffset[pf];
  int boffset = tjBlueOffset[pf];
  int aoffset = tjAlphaOffset[pf];
  int ps = tjPixelSize[pf];
  int i, j;

  for (j = 0; j < height; j++) {
    int row = bottomUp ? height - j - 1 : j;

    for (i = 0; i < width; i++) {
      int r = (i * (maxSample + 1) / width) % (maxSample + 1);
      int g = (j * (maxSample + 1) / height) % (maxSample + 1);
      int b = (j * (maxSample + 1) / height +
               i * (maxSample + 1) / width) % (maxSample + 1);

      if (pf == TJPF_GRAY) {
        if (getVal(buf, row * pitch + i * ps) != b)
          return 0;
      } else if (pf == TJPF_CMYK) {
        int rf, gf, bf;

        cmyk_to_rgb(getVal(buf, row * pitch + i * ps + 0),
                    getVal(buf, row * pitch + i * ps + 1),
                    getVal(buf, row * pitch + i * ps + 2),
                    getVal(buf, row * pitch + i * ps + 3), &rf, &gf, &bf);
        if (gray2rgb) {
          if (rf != b || gf != b || bf != b)
            return 0;
        } else if (rf != r || gf != g || bf != b) return 0;
      } else {
        if (gray2rgb) {
          if (getVal(buf, row * pitch + i * ps + roffset) != b ||
              getVal(buf, row * pitch + i * ps + goffset) != b ||
              getVal(buf, row * pitch + i * ps + boffset) != b)
            return 0;
        } else if (getVal(buf, row * pitch + i * ps + roffset) != r ||
                   getVal(buf, row * pitch + i * ps + goffset) != g ||
                   getVal(buf, row * pitch + i * ps + boffset) != b)
          return 0;
        if (aoffset >= 0 &&
            getVal(buf, row * pitch + i * ps + aoffset) != maxSample)
          return 0;
      }
    }
  }
  return 1;
}


static int doBmpTest(const char *ext, int width, int align, int height, int pf,
                     int bottomUp)
{
  tjhandle handle = NULL;
  char filename[80], *md5sum, md5buf[65];
  int ps = tjPixelSize[pf], pitch = PAD(width * ps, align), loadWidth = 0,
    loadHeight = 0, retval = 0, pixelFormat = pf;
  void *buf = NULL;
  char *md5ref;
  char *colorPPMRefs[17] = {
    "", "", "0bad09d9ef38eda566848fb7c0b7fd0a",
    "7ef2c87261a8bd6838303b541563cf27", "28a37cf9636ff6bb9ed6b206bdac60db",
    "723307791d42e0b5f9e91625c7636086", "d729c4bcd3addc14abc16b656c6bbc98",
    "5d7636eedae3cf579b6de13078227548", "c0c9f772b464d1896326883a5c79c545",
    "fcf6490e0445569427f1d95baf5f8fcb", "5cbc3b0ccba23f5781d950a72e0ccc83",
    "0d4e26d6d16d7bfee380f6feb10f7e53", "2ff5299287017502832c99718450c90a",
    "44ae6cd70c798ea583ab0c8c03621092", "697b2fe03892bc9a75396ad3e73d9203",
    "599732f973eb7c0849a888e783bbe27e", "623f54661b928d170bd2324bc3620565"
  };
  char *grayPPMRefs[17] = {
    "", "", "7565be35a2ce909cae016fa282af8efa",
    "e86b9ea57f7d53f6b5497653740992b5", "8924d4d81fe0220c684719294f93407a",
    "e2e69ba70efcfae317528c91651c7ae2", "e6154aafc1eb9e4333d68ce7ad9df051",
    "3d7fe831d6fbe55d3fa12f52059c15d3", "112c682e82ce5de1cca089e20d60000b",
    "05a7ce86c649dda86d6fed185ab78a67", "0b723c0bc087592816523fbc906b7c3a",
    "5da422b1ddfd44c7659094d42ba5580c", "0d1895c7e6f2b2c9af6e821a655c239c",
    "00fc2803bca103ff75785ea0dca992aa", "d8c91fac522c16b029e514d331a22bc4",
    "e50cff0b3562ed7e64dbfc093440e333", "64f3320b226ea37fb58080713b4df1b2"
  };

  if ((handle = tj3Init(TJINIT_TRANSFORM)) == NULL)
    THROW_TJ(NULL);
  TRY_TJ(handle, tj3Set(handle, TJPARAM_BOTTOMUP, bottomUp));
  TRY_TJ(handle, tj3Set(handle, TJPARAM_PRECISION, precision));

  if (precision == 8 && !strcasecmp(ext, "bmp"))
    md5ref = (pf == TJPF_GRAY ? "51976530acf75f02beddf5d21149101d" :
                                "6d659071b9bfcdee2def22cb58ddadca");
  else
    md5ref = (pf == TJPF_GRAY ? grayPPMRefs[precision] :
                                colorPPMRefs[precision]);

  if ((buf = tj3Alloc(pitch * height * sampleSize)) == NULL)
    THROW("Could not allocate memory");
  initBitmap(buf, width, pitch, height, pf, bottomUp);

  SNPRINTF(filename, 80, "test_bmp%d_%s_%d_%s_%d.%s", precision,
           pixFormatStr[pf], align, bottomUp ? "bu" : "td", getpid(), ext);
  if (precision <= 8) {
    TRY_TJ(handle, tj3SaveImage8(handle, filename, (unsigned char *)buf, width,
                                 pitch, height, pf));
  } else if (precision <= 12) {
    TRY_TJ(handle, tj3SaveImage12(handle, filename, (short *)buf, width, pitch,
                                  height, pf));
  } else {
    TRY_TJ(handle, tj3SaveImage16(handle, filename, (unsigned short *)buf,
                                  width, pitch, height, pf));
  }
  md5sum = MD5File(filename, md5buf);
  if (!md5sum) {
    printf("\n   Could not determine MD5 sum of %s\n", filename);
    retval = -1;  goto bailout;
  }
  if (strcasecmp(md5sum, md5ref))
    THROW_MD5(filename, md5sum, md5ref);

  tj3Free(buf);  buf = NULL;
  if (precision <= 8) {
    if ((buf = tj3LoadImage8(handle, filename, &loadWidth, align, &loadHeight,
                             &pf)) == NULL)
      THROW_TJ(handle);
  } else if (precision <= 12) {
    if ((buf = tj3LoadImage12(handle, filename, &loadWidth, align, &loadHeight,
                              &pf)) == NULL)
      THROW_TJ(handle);
  } else {
    if ((buf = tj3LoadImage16(handle, filename, &loadWidth, align, &loadHeight,
                              &pf)) == NULL)
      THROW_TJ(handle);
  }
  if (width != loadWidth || height != loadHeight) {
    printf("\n   Image dimensions of %s are bogus\n", filename);
    retval = -1;  goto bailout;
  }
  if (!cmpBitmap(buf, width, pitch, height, pf, bottomUp, 0)) {
    printf("\n   Pixel data in %s is bogus\n", filename);
    retval = -1;  goto bailout;
  }
  if (pf == TJPF_GRAY) {
    tj3Free(buf);  buf = NULL;
    pf = TJPF_XBGR;
    if (precision <= 8) {
      if ((buf = tj3LoadImage8(handle, filename, &loadWidth, align,
                               &loadHeight, &pf)) == NULL)
        THROW_TJ(handle);
    } else if (precision <= 12) {
      if ((buf = tj3LoadImage12(handle, filename, &loadWidth, align,
                                &loadHeight, &pf)) == NULL)
        THROW_TJ(handle);
    } else {
      if ((buf = tj3LoadImage16(handle, filename, &loadWidth, align,
                                &loadHeight, &pf)) == NULL)
        THROW_TJ(handle);
    }
    pitch = PAD(width * tjPixelSize[pf], align);
    if (!cmpBitmap(buf, width, pitch, height, pf, bottomUp, 1)) {
      printf("\n   Converting %s to RGB failed\n", filename);
      retval = -1;  goto bailout;
    }

    tj3Free(buf);  buf = NULL;
    pf = TJPF_CMYK;
    if (precision <= 8) {
      if ((buf = tj3LoadImage8(handle, filename, &loadWidth, align,
                               &loadHeight, &pf)) == NULL)
        THROW_TJ(handle);
    } else if (precision <= 12) {
      if ((buf = tj3LoadImage12(handle, filename, &loadWidth, align,
                                &loadHeight, &pf)) == NULL)
        THROW_TJ(handle);
    } else {
      if ((buf = tj3LoadImage16(handle, filename, &loadWidth, align,
                                &loadHeight, &pf)) == NULL)
        THROW_TJ(handle);
    }
    pitch = PAD(width * tjPixelSize[pf], align);
    if (!cmpBitmap(buf, width, pitch, height, pf, bottomUp, 1)) {
      printf("\n   Converting %s to CMYK failed\n", filename);
      retval = -1;  goto bailout;
    }
  }
  /* Verify that tj3LoadImage*() returns the proper "preferred" pixel format
     for the file type. */
  tj3Free(buf);  buf = NULL;
  pf = pixelFormat;
  pixelFormat = TJPF_UNKNOWN;
  if (precision <= 8) {
    if ((buf = tj3LoadImage8(handle, filename, &loadWidth, align, &loadHeight,
                             &pixelFormat)) == NULL)
      THROW_TJ(handle);
  } else if (precision <= 12) {
    if ((buf = tj3LoadImage12(handle, filename, &loadWidth, align, &loadHeight,
                              &pixelFormat)) == NULL)
      THROW_TJ(handle);
  } else {
    if ((buf = tj3LoadImage16(handle, filename, &loadWidth, align, &loadHeight,
                              &pixelFormat)) == NULL)
      THROW_TJ(handle);
  }
  if ((pf == TJPF_GRAY && pixelFormat != TJPF_GRAY) ||
      (pf != TJPF_GRAY && !strcasecmp(ext, "bmp") &&
       pixelFormat != TJPF_BGR) ||
      (pf != TJPF_GRAY && !strcasecmp(ext, "ppm") &&
       pixelFormat != TJPF_RGB)) {
    printf("\n   tj3LoadImage8() returned unexpected pixel format: %s\n",
           pixFormatStr[pixelFormat]);
    retval = -1;
  }
  unlink(filename);

bailout:
  tj3Destroy(handle);
  tj3Free(buf);
  if (exitStatus < 0) return exitStatus;
  return retval;
}


static int bmpTest(void)
{
  int align, width = 35, height = 39, format;

  for (align = 1; align <= 8; align *= 2) {
    for (format = 0; format < TJ_NUMPF; format++) {
      if (precision == 8) {
        printf("%s Top-Down BMP (row alignment = %d samples)  ...  ",
               pixFormatStr[format], align);
        if (doBmpTest("bmp", width, align, height, format, 0) == -1)
          return -1;
        printf("OK.\n");
      }

      printf("%s Top-Down PPM (row alignment = %d samples)  ...  ",
             pixFormatStr[format], align);
      if (doBmpTest("ppm", width, align, height, format, 1) == -1)
        return -1;
      printf("OK.\n");

      if (precision == 8) {
        printf("%s Bottom-Up BMP (row alignment = %d samples)  ...  ",
               pixFormatStr[format], align);
        if (doBmpTest("bmp", width, align, height, format, 0) == -1)
          return -1;
        printf("OK.\n");
      }

      printf("%s Bottom-Up PPM (row alignment = %d samples)  ...  ",
             pixFormatStr[format], align);
      if (doBmpTest("ppm", width, align, height, format, 1) == -1)
        return -1;
      printf("OK.\n");
    }
  }

  return 0;
}


int main(int argc, char *argv[])
{
  int i, bmp = 0, num4bf = 5;

#ifdef _WIN32
  srand((unsigned int)time(NULL));
#endif
  if (argc > 1) {
    for (i = 1; i < argc; i++) {
      if (!strcasecmp(argv[i], "-yuv")) doYUV = 1;
      else if (!strcasecmp(argv[i], "-noyuvpad")) yuvAlign = 1;
      else if (!strcasecmp(argv[i], "-lossless")) lossless = 1;
      else if (!strcasecmp(argv[i], "-alloc")) alloc = 1;
      else if (!strcasecmp(argv[i], "-bmp")) bmp = 1;
      else if (!strcasecmp(argv[i], "-precision") && i < argc - 1) {
        int tempi = atoi(argv[++i]);

        if (tempi < 2 || tempi > 16)
          usage(argv[0]);
        precision = tempi;
        if (precision != 8 && precision != 12) lossless = 1;
      } else
        usage(argv[0]);
    }
  }
  if (lossless && doYUV)
    THROW("Lossless JPEG and YUV encoding/decoding are incompatible.");
  if (precision != 8 && doYUV)
    THROW("YUV encoding/decoding requires 8-bit data precision.");

  printf("Testing %d-bit precision\n", precision);
  sampleSize = (precision <= 8 ? sizeof(unsigned char) : sizeof(short));
  maxSample = (1 << precision) - 1;
  tolerance = (lossless ? 0 : (precision > 8 ? 2 : 1));
  redToY = (19595U * maxSample) >> 16;
  yellowToY = (58065U * maxSample) >> 16;

  if (bmp) return bmpTest();
  if (alloc) printf("Testing automatic buffer allocation\n");
  if (doYUV) num4bf = 4;
  overflowTest();
  doTest(35, 39, _3sampleFormats, 2, TJSAMP_444, "test");
  doTest(39, 41, _4sampleFormats, num4bf, TJSAMP_444, "test");
  doTest(41, 35, _3sampleFormats, 2, TJSAMP_422, "test");
  if (!lossless) {
    doTest(35, 39, _4sampleFormats, num4bf, TJSAMP_422, "test");
    doTest(39, 41, _3sampleFormats, 2, TJSAMP_420, "test");
    doTest(41, 35, _4sampleFormats, num4bf, TJSAMP_420, "test");
    doTest(35, 39, _3sampleFormats, 2, TJSAMP_440, "test");
    doTest(39, 41, _4sampleFormats, num4bf, TJSAMP_440, "test");
    doTest(41, 35, _3sampleFormats, 2, TJSAMP_411, "test");
    doTest(35, 39, _4sampleFormats, num4bf, TJSAMP_411, "test");
    doTest(39, 41, _3sampleFormats, 2, TJSAMP_441, "test");
    doTest(41, 35, _4sampleFormats, num4bf, TJSAMP_441, "test");
  }
  doTest(39, 41, _onlyGray, 1, TJSAMP_GRAY, "test");
  if (!lossless) {
    doTest(41, 35, _3sampleFormats, 2, TJSAMP_GRAY, "test");
    doTest(35, 39, _4sampleFormats, 4, TJSAMP_GRAY, "test");
  }
  bufSizeTest();
  if (doYUV) {
    printf("\n--------------------\n\n");
    doTest(48, 48, _onlyRGB, 1, TJSAMP_444, "test_yuv0");
    doTest(48, 48, _onlyRGB, 1, TJSAMP_422, "test_yuv0");
    doTest(48, 48, _onlyRGB, 1, TJSAMP_420, "test_yuv0");
    doTest(48, 48, _onlyRGB, 1, TJSAMP_440, "test_yuv0");
    doTest(48, 48, _onlyRGB, 1, TJSAMP_411, "test_yuv0");
    doTest(48, 48, _onlyRGB, 1, TJSAMP_441, "test_yuv0");
    doTest(48, 48, _onlyRGB, 1, TJSAMP_GRAY, "test_yuv0");
    doTest(48, 48, _onlyGray, 1, TJSAMP_GRAY, "test_yuv0");
  }

  bailout:
  return exitStatus;
}
