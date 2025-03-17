/*
 * Copyright (C)2011-2012, 2014-2015, 2017, 2019, 2021-2024
 *           D. R. Commander.  All Rights Reserved.
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
 * This program demonstrates how to use the TurboJPEG C API to approximate the
 * functionality of the IJG's djpeg program.  djpeg features that are not
 * covered:
 *
 * - OS/2 BMP, GIF, and Targa output file formats [legacy feature]
 * - Color quantization and dithering [legacy feature]
 * - The floating-point IDCT method [legacy feature]
 * - Extracting an ICC color management profile
 * - Progress reporting
 * - Skipping rows (i.e. exclusive rather than inclusive partial decompression)
 * - Debug output
 */

#ifdef _MSC_VER
#define _CRT_SECURE_NO_DEPRECATE
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <limits.h>
#if !defined(_MSC_VER) || _MSC_VER > 1600
#include <stdint.h>
#endif
#include <turbojpeg.h>


#ifdef _WIN32
#define strncasecmp  strnicmp
#endif

#ifndef max
#define max(a, b)  ((a) > (b) ? (a) : (b))
#endif

#define MATCH_ARG(arg, string, minChars) \
  !strncasecmp(arg, string, max(strlen(arg), minChars))

#define IS_CROPPED(cr)  (cr.x != 0 || cr.y != 0 || cr.w != 0 || cr.h != 0)

#define THROW(action, message) { \
  printf("ERROR in line %d while %s:\n%s\n", __LINE__, action, message); \
  retval = -1;  goto bailout; \
}

#define THROW_TJ(action) { \
  int errorCode = tj3GetErrorCode(tjInstance); \
  printf("%s in line %d while %s:\n%s\n", \
         errorCode == TJERR_WARNING ? "WARNING" : "ERROR", __LINE__, action, \
         tj3GetErrorStr(tjInstance)); \
  if (errorCode == TJERR_FATAL || stopOnWarning == 1) { \
    retval = -1;  goto bailout; \
  } \
}

#define THROW_UNIX(action)  THROW(action, strerror(errno))


static tjscalingfactor *scalingFactors = NULL;
static int numScalingFactors = 0;


static void usage(char *programName)
{
  int i;

  printf("\nUSAGE: %s [options] <JPEG image> <Output image>\n\n", programName);

  printf("The output image will be in Windows BMP or PBMPLUS (PPM/PGM) format, depending\n");
  printf("on the file extension.\n\n");

  printf("GENERAL OPTIONS (CAN BE ABBREVBIATED)\n");
  printf("-------------------------------------\n");
  printf("-icc FILE\n");
  printf("    Extract the ICC (International Color Consortium) color profile from the\n");
  printf("    JPEG image to the specified file\n");
  printf("-strict\n");
  printf("    Treat all warnings as fatal; abort immediately if incomplete or corrupt\n");
  printf("    data is encountered in the JPEG image, rather than trying to salvage the\n");
  printf("    rest of the image\n\n");

  printf("LOSSY JPEG OPTIONS (CAN BE ABBREVIATED)\n");
  printf("---------------------------------------\n");
  printf("-crop WxH+X+Y\n");
  printf("    Decompress only the specified region of the JPEG image.  (W, H, X, and Y\n");
  printf("    are the width, height, left boundary, and upper boundary of the region, all\n");
  printf("    specified relative to the scaled image dimensions.)  If necessary, X will\n");
  printf("    be shifted left to the nearest iMCU boundary, and W will be increased\n");
  printf("    accordingly.\n");
  printf("-dct fast\n");
  printf("    Use less accurate IDCT algorithm [legacy feature]\n");
  printf("-dct int\n");
  printf("    Use more accurate IDCT algorithm [default]\n");
  printf("-grayscale\n");
  printf("    Decompress a full-color JPEG image into a grayscale output image\n");
  printf("-maxmemory N\n");
  printf("    Memory limit (in megabytes) for intermediate buffers used with progressive\n");
  printf("    JPEG decompression [default = no limit]\n");
  printf("-maxscans N\n");
  printf("    Refuse to decompress progressive JPEG images that have more than N scans\n");
  printf("-nosmooth\n");
  printf("    Use the fastest chrominance upsampling algorithm available\n");
  printf("-rgb\n");
  printf("    Decompress a grayscale JPEG image into a full-color output image\n");
  printf("-scale M/N\n");
  printf("    Scale the width/height of the JPEG image by a factor of M/N when\n");
  printf("    decompressing it (M/N = ");
  for (i = 0; i < numScalingFactors; i++) {
    printf("%d/%d", scalingFactors[i].num, scalingFactors[i].denom);
    if (numScalingFactors == 2 && i != numScalingFactors - 1)
      printf(" or ");
    else if (numScalingFactors > 2) {
      if (i != numScalingFactors - 1)
        printf(", ");
      if (i == numScalingFactors - 2)
        printf("or ");
    }
    if (i % 8 == 0 && i != 0) printf("\n    ");
  }
  printf(")\n\n");

  exit(1);
}


int main(int argc, char **argv)
{
  int i, retval = 0;
  int colorspace, fastDCT = -1, fastUpsample = -1, maxMemory = -1,
    maxScans = -1, pixelFormat = TJPF_UNKNOWN, precision, stopOnWarning = -1,
    subsamp;
  tjregion croppingRegion = TJUNCROPPED;
  tjscalingfactor scalingFactor = TJUNSCALED;
  char *iccFilename = NULL;
  tjhandle tjInstance = NULL;
  FILE *jpegFile = NULL, *iccFile = NULL;
  long size = 0;
  size_t jpegSize, sampleSize, iccSize;
  int width, height;
  unsigned char *jpegBuf = NULL, *iccBuf = NULL;
  void *dstBuf = NULL;

  if ((scalingFactors = tj3GetScalingFactors(&numScalingFactors)) == NULL)
    THROW_TJ("getting scaling factors");

  for (i = 1; i < argc; i++) {

    if (MATCH_ARG(argv[i], "-crop", 2) && i < argc - 1) {
      char tempc = -1;

      if (sscanf(argv[++i], "%d%c%d+%d+%d", &croppingRegion.w, &tempc,
                 &croppingRegion.h, &croppingRegion.x,
                 &croppingRegion.y) != 5 || croppingRegion.w < 1 ||
          (tempc != 'x' && tempc != 'X') || croppingRegion.h < 1 ||
          croppingRegion.x < 0 || croppingRegion.y < 0)
        usage(argv[0]);
    } else if (MATCH_ARG(argv[i], "-dct", 2) && i < argc - 1) {
      i++;
      if (MATCH_ARG(argv[i], "fast", 1))
        fastDCT = 1;
      else if (!MATCH_ARG(argv[i], "int", 1))
        usage(argv[0]);
    } else if (MATCH_ARG(argv[i], "-grayscale", 2) ||
               MATCH_ARG(argv[i], "-greyscale", 2))
      pixelFormat = TJPF_GRAY;
    else if (MATCH_ARG(argv[i], "-icc", 2) && i < argc - 1)
      iccFilename = argv[++i];
    else if (MATCH_ARG(argv[i], "-maxscans", 5) && i < argc - 1) {
      int tempi = atoi(argv[++i]);

      if (tempi < 0) usage(argv[0]);
      maxScans = tempi;
    } else if (MATCH_ARG(argv[i], "-maxmemory", 2) && i < argc - 1) {
      int tempi = atoi(argv[++i]);

      if (tempi < 0) usage(argv[0]);
      maxMemory = tempi;
    } else if (MATCH_ARG(argv[i], "-nosmooth", 2))
      fastUpsample = 1;
    else if (MATCH_ARG(argv[i], "-rgb", 2))
      pixelFormat = TJPF_RGB;
    else if (MATCH_ARG(argv[i], "-strict", 3))
      stopOnWarning = 1;
    else if (MATCH_ARG(argv[i], "-scale", 2) && i < argc - 1) {
      int match = 0, temp_num = 0, temp_denom = 0, j;

      if (sscanf(argv[++i], "%d/%d", &temp_num, &temp_denom) < 2)
        usage(argv[0]);
      if (temp_num < 1 || temp_denom < 1)
        usage(argv[0]);
      for (j = 0; j < numScalingFactors; j++) {
        if ((double)temp_num / (double)temp_denom ==
            (double)scalingFactors[j].num / (double)scalingFactors[j].denom) {
          scalingFactor = scalingFactors[j];
          match = 1;
          break;
        }
      }
      if (match != 1)
        usage(argv[0]);
    } else break;
  }

  if (i != argc - 2)
    usage(argv[0]);

  if ((tjInstance = tj3Init(TJINIT_DECOMPRESS)) == NULL)
    THROW_TJ("creating TurboJPEG instance");

  if (stopOnWarning >= 0 &&
      tj3Set(tjInstance, TJPARAM_STOPONWARNING, stopOnWarning) < 0)
    THROW_TJ("setting TJPARAM_STOPONWARNING");
  if (fastUpsample >= 0 &&
      tj3Set(tjInstance, TJPARAM_FASTUPSAMPLE, fastUpsample) < 0)
    THROW_TJ("setting TJPARAM_FASTUPSAMPLE");
  if (fastDCT >= 0 && tj3Set(tjInstance, TJPARAM_FASTDCT, fastDCT) < 0)
    THROW_TJ("setting TJPARAM_FASTDCT");
  if (maxScans >= 0 && tj3Set(tjInstance, TJPARAM_SCANLIMIT, maxScans) < 0)
    THROW_TJ("setting TJPARAM_SCANLIMIT");
  if (maxMemory >= 0 && tj3Set(tjInstance, TJPARAM_MAXMEMORY, maxMemory) < 0)
    THROW_TJ("setting TJPARAM_MAXMEMORY");

  if ((jpegFile = fopen(argv[i++], "rb")) == NULL)
    THROW_UNIX("opening input file");
  if (fseek(jpegFile, 0, SEEK_END) < 0 || ((size = ftell(jpegFile)) < 0) ||
      fseek(jpegFile, 0, SEEK_SET) < 0)
    THROW_UNIX("determining input file size");
  if (size == 0)
    THROW("determining input file size", "Input file contains no data");
  jpegSize = size;
  if ((jpegBuf = (unsigned char *)malloc(jpegSize)) == NULL)
    THROW_UNIX("allocating JPEG buffer");
  if (fread(jpegBuf, jpegSize, 1, jpegFile) < 1)
    THROW_UNIX("reading input file");
  fclose(jpegFile);  jpegFile = NULL;

  if (tj3DecompressHeader(tjInstance, jpegBuf, jpegSize) < 0)
    THROW_TJ("reading JPEG header");
  subsamp = tj3Get(tjInstance, TJPARAM_SUBSAMP);
  width = tj3Get(tjInstance, TJPARAM_JPEGWIDTH);
  height = tj3Get(tjInstance, TJPARAM_JPEGHEIGHT);
  precision = tj3Get(tjInstance, TJPARAM_PRECISION);
  sampleSize = (precision <= 8 ? sizeof(unsigned char) : sizeof(short));
  colorspace = tj3Get(tjInstance, TJPARAM_COLORSPACE);

  if (iccFilename) {
    if (tj3GetICCProfile(tjInstance, &iccBuf, &iccSize) < 0) {
      THROW_TJ("getting ICC profile");
    } else {
      if ((iccFile = fopen(iccFilename, "wb")) == NULL)
        THROW_UNIX("opening ICC file");
      if (fwrite(iccBuf, iccSize, 1, iccFile) < 1)
        THROW_UNIX("writing ICC profile");
      tj3Free(iccBuf);  iccBuf = NULL;
      fclose(iccFile);  iccFile = NULL;
    }
  }

  if (pixelFormat == TJPF_UNKNOWN) {
    if (colorspace == TJCS_GRAY)
      pixelFormat = TJPF_GRAY;
    else if (colorspace == TJCS_CMYK || colorspace == TJCS_YCCK)
      pixelFormat = TJPF_CMYK;
    else
      pixelFormat = TJPF_RGB;
  }

  if (!tj3Get(tjInstance, TJPARAM_LOSSLESS)) {
    if (tj3SetScalingFactor(tjInstance, scalingFactor) < 0)
      THROW_TJ("setting scaling factor");
    width = TJSCALED(width, scalingFactor);
    height = TJSCALED(height, scalingFactor);

    if (IS_CROPPED(croppingRegion)) {
      int adjustment;

      if (subsamp == TJSAMP_UNKNOWN)
        THROW("adjusting cropping region",
              "Could not determine subsampling level of JPEG image");
      adjustment =
        croppingRegion.x % TJSCALED(tjMCUWidth[subsamp], scalingFactor);
      croppingRegion.x -= adjustment;
      croppingRegion.w += adjustment;
      if (tj3SetCroppingRegion(tjInstance, croppingRegion) < 0)
        THROW_TJ("setting cropping region");
      width = croppingRegion.w;
      height = croppingRegion.h;
    }
  }

#if ULLONG_MAX > SIZE_MAX
  if ((unsigned long long)width * height * tjPixelSize[pixelFormat] *
      sampleSize > (unsigned long long)((size_t)-1))
    THROW("allocating uncompressed image buffer", "Image is too large");
#endif
  if ((dstBuf =
       (unsigned char *)malloc(sizeof(unsigned char) * width * height *
                               tjPixelSize[pixelFormat] * sampleSize)) == NULL)
    THROW_UNIX("allocating uncompressed image buffer");

  if (precision <= 8) {
    if (tj3Decompress8(tjInstance, jpegBuf, jpegSize, dstBuf, 0,
                       pixelFormat) < 0)
      THROW_TJ("decompressing JPEG image");
  } else if (precision <= 12) {
    if (tj3Decompress12(tjInstance, jpegBuf, jpegSize, dstBuf, 0,
                        pixelFormat) < 0)
      THROW_TJ("decompressing JPEG image");
  } else {
    if (tj3Decompress16(tjInstance, jpegBuf, jpegSize, dstBuf, 0,
                        pixelFormat) < 0)
      THROW_TJ("decompressing JPEG image");
  }
  tj3Free(jpegBuf);  jpegBuf = NULL;

  if (precision <= 8) {
    if (tj3SaveImage8(tjInstance, argv[i], dstBuf, width, 0, height,
                      pixelFormat) < 0)
      THROW_TJ("saving output image");
  } else if (precision <= 12) {
    if (tj3SaveImage12(tjInstance, argv[i], dstBuf, width, 0, height,
                       pixelFormat) < 0)
      THROW_TJ("saving output image");
  } else {
    if (tj3SaveImage16(tjInstance, argv[i], dstBuf, width, 0, height,
                       pixelFormat) < 0)
      THROW_TJ("saving output image");
  }

bailout:
  tj3Destroy(tjInstance);
  if (jpegFile) fclose(jpegFile);
  tj3Free(jpegBuf);
  tj3Free(iccBuf);
  if (iccFile) fclose(iccFile);
  free(dstBuf);
  return retval;
}
