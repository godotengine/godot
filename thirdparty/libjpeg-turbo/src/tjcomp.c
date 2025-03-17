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
 * functionality of the IJG's cjpeg program.  cjpeg features that are not
 * covered:
 *
 * - GIF and Targa input file formats [legacy feature]
 * - Separate quality settings for luminance and chrominance
 * - The floating-point DCT method [legacy feature]
 * - Input image smoothing
 * - Progress reporting
 * - Debug output
 * - Forcing baseline-compatible quantization tables
 * - Specifying arbitrary quantization tables
 * - Specifying arbitrary sampling factors
 * - Scan scripts
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

#define THROW(action, message) { \
  printf("ERROR in line %d while %s:\n%s\n", __LINE__, action, message); \
  retval = -1;  goto bailout; \
}

#define THROW_TJ(action)  THROW(action, tj3GetErrorStr(tjInstance))

#define THROW_UNIX(action)  THROW(action, strerror(errno))

#define DEFAULT_SUBSAMP  TJSAMP_420
#define DEFAULT_QUALITY  75


static const char *subsampName[TJ_NUMSAMP] = {
  "444", "422", "420", "GRAY", "440", "411", "441"
};


static void usage(char *programName)
{
  printf("\nUSAGE: %s [options] <Input image> <JPEG image>\n\n", programName);

  printf("The input image can be in Windows BMP or PBMPLUS (PPM/PGM) format.\n\n");

  printf("GENERAL OPTIONS (CAN BE ABBREVIATED)\n");
  printf("------------------------------------\n");
  printf("-icc FILE\n");
  printf("    Embed the ICC (International Color Consortium) color management profile\n");
  printf("    from the specified file into the JPEG image\n");
  printf("-lossless PSV[,Pt]\n");
  printf("    Create a lossless JPEG image (implies -subsamp 444) using predictor\n");
  printf("    selection value PSV (1-7) and optional point transform Pt (0 through\n");
  printf("    {data precision} - 1)\n");
  printf("-maxmemory N\n");
  printf("    Memory limit (in megabytes) for intermediate buffers used with progressive\n");
  printf("    JPEG compression, lossless JPEG compression, and Huffman table optimization\n");
  printf("    [default = no limit]\n");
  printf("-precision N\n");
  printf("    Create a JPEG image with N-bit data precision [N = 2..16; default = 8; if N\n");
  printf("    is not 8 or 12, then -lossless must also be specified] (-precision 12\n");
  printf("    implies -optimize unless -arithmetic is also specified)\n");
  printf("-restart N\n");
  printf("    Add a restart marker every N MCU rows [default = 0 (no restart markers)].\n");
  printf("    Append 'B' to specify the restart marker interval in MCUs (lossy only.)\n\n");

  printf("LOSSY JPEG OPTIONS (CAN BE ABBREVIATED)\n");
  printf("---------------------------------------\n");
  printf("-arithmetic\n");
  printf("    Use arithmetic entropy coding instead of Huffman entropy coding (can be\n");
  printf("    combined with -progressive)\n");
  printf("-dct fast\n");
  printf("    Use less accurate DCT algorithm [legacy feature]\n");
  printf("-dct int\n");
  printf("    Use more accurate DCT algorithm [default]\n");
  printf("-grayscale\n");
  printf("    Create a grayscale JPEG image from a full-color input image\n");
  printf("-optimize\n");
  printf("    Use Huffman table optimization\n");
  printf("-progressive\n");
  printf("    Create a progressive JPEG image instead of a single-scan JPEG image (can be\n");
  printf("    combined with -arithmetic; implies -optimize unless -arithmetic is also\n");
  printf("    specified)\n");
  printf("-quality {1..100}\n");
  printf("    Create a JPEG image with the specified quality level [default = %d]\n",
         DEFAULT_QUALITY);
  printf("-rgb\n");
  printf("    Create a JPEG image that uses the RGB colorspace instead of the YCbCr\n");
  printf("    colorspace\n");
  printf("-subsamp {444|422|440|420|411|441}\n");
  printf("    Create a JPEG image that uses the specified chrominance subsampling level\n");
  printf("    [default = %s]\n\n", subsampName[DEFAULT_SUBSAMP]);

  exit(1);
}


int main(int argc, char **argv)
{
  int i, retval = 0;
  int arithmetic = -1, colorspace = -1, fastDCT = -1, losslessPSV = -1,
    losslessPt = -1, maxMemory = -1, optimize = -1, pixelFormat = TJPF_UNKNOWN,
    precision = 8, progressive = -1, quality = DEFAULT_QUALITY,
    restartIntervalBlocks = -1, restartIntervalRows = -1,
    subsamp = DEFAULT_SUBSAMP;
  char *iccFilename = NULL;
  tjhandle tjInstance = NULL;
  void *srcBuf = NULL;
  int width, height;
  long size;
  unsigned char *iccBuf = NULL, *jpegBuf = NULL;
  size_t iccSize = 0, jpegSize = 0;
  FILE *iccFile = NULL, *jpegFile = NULL;

  for (i = 1; i < argc; i++) {
    if (MATCH_ARG(argv[i], "-arithmetic", 2))
      arithmetic = 1;
    else if (MATCH_ARG(argv[i], "-dct", 2) && i < argc - 1) {
      i++;
      if (MATCH_ARG(argv[i], "fast", 1))
        fastDCT = 1;
      else if (!MATCH_ARG(argv[i], "int", 1))
        usage(argv[0]);
    } else if (MATCH_ARG(argv[i], "-grayscale", 2) ||
               MATCH_ARG(argv[i], "-greyscale", 2))
      colorspace = TJCS_GRAY;
    else if (MATCH_ARG(argv[i], "-icc", 2) && i < argc - 1)
      iccFilename = argv[++i];
    else if (MATCH_ARG(argv[i], "-lossless", 2) && i < argc - 1) {
      if (sscanf(argv[++i], "%d,%d", &losslessPSV, &losslessPt) < 1 ||
          losslessPSV < 1 || losslessPSV > 7)
        usage(argv[0]);
    } else if (MATCH_ARG(argv[i], "-maxmemory", 2) && i < argc - 1) {
      int tempi = atoi(argv[++i]);

      if (tempi < 0) usage(argv[0]);
      maxMemory = tempi;
    } else if (MATCH_ARG(argv[i], "-optimize", 2) ||
               MATCH_ARG(argv[i], "-optimise", 2))
      optimize = 1;
    else if (MATCH_ARG(argv[i], "-precision", 4) && i < argc - 1) {
      int tempi = atoi(argv[++i]);

      if (tempi < 2 || tempi > 16)
        usage(argv[0]);
      precision = tempi;
    } else if (MATCH_ARG(argv[i], "-progressive", 2))
      progressive = 1;
    else if (MATCH_ARG(argv[i], "-quality", 2) && i < argc - 1) {
      int tempi = atoi(argv[++i]);

      if (tempi < 1 || tempi > 100)
        usage(argv[0]);
      quality = tempi;
    } else if (MATCH_ARG(argv[i], "-rgb", 3))
      colorspace = TJCS_RGB;
    else if (MATCH_ARG(argv[i], "-restart", 2) && i < argc - 1) {
      int tempi = -1, nscan;  char tempc = 0;

      if ((nscan = sscanf(argv[++i], "%d%c", &tempi, &tempc)) < 1 ||
          tempi < 0 || tempi > 65535 ||
          (nscan == 2 && tempc != 'B' && tempc != 'b'))
        usage(argv[0]);

      if (tempc == 'B' || tempc == 'b')
        restartIntervalBlocks = tempi;
      else
        restartIntervalRows = tempi;
    } else if (MATCH_ARG(argv[i], "-subsamp", 2) && i < argc - 1) {
      i++;
      if (MATCH_ARG(argv[i], "444", 3))
        subsamp = TJSAMP_444;
      else if (MATCH_ARG(argv[i], "422", 3))
        subsamp = TJSAMP_422;
      else if (MATCH_ARG(argv[i], "440", 3))
        subsamp = TJSAMP_440;
      else if (MATCH_ARG(argv[i], "420", 3))
        subsamp = TJSAMP_420;
      else if (MATCH_ARG(argv[i], "411", 3))
        subsamp = TJSAMP_411;
      else if (MATCH_ARG(argv[i], "441", 3))
        subsamp = TJSAMP_441;
      else
        usage(argv[0]);
    } else break;
  }

  if (i != argc - 2)
    usage(argv[0]);
  if (losslessPSV == -1 && precision != 8 && precision != 12)
    usage(argv[0]);

  if ((tjInstance = tj3Init(TJINIT_COMPRESS)) == NULL)
    THROW_TJ("creating TurboJPEG instance");

  if (tj3Set(tjInstance, TJPARAM_QUALITY, quality) < 0)
    THROW_TJ("setting TJPARAM_QUALITY");
  if (tj3Set(tjInstance, TJPARAM_SUBSAMP, subsamp) < 0)
    THROW_TJ("setting TJPARAM_SUBSAMP");
  if (tj3Set(tjInstance, TJPARAM_PRECISION, precision) < 0)
    THROW_TJ("setting TJPARAM_PRECISION");
  if (fastDCT >= 0 && tj3Set(tjInstance, TJPARAM_FASTDCT, fastDCT) < 0)
    THROW_TJ("setting TJPARAM_FASTDCT");
  if (optimize >= 0 && tj3Set(tjInstance, TJPARAM_OPTIMIZE, optimize) < 0)
    THROW_TJ("setting TJPARAM_OPTIMIZE");
  if (progressive >= 0 &&
      tj3Set(tjInstance, TJPARAM_PROGRESSIVE, progressive) < 0)
    THROW_TJ("setting TJPARAM_PROGRESSIVE");
  if (arithmetic >= 0 &&
      tj3Set(tjInstance, TJPARAM_ARITHMETIC, arithmetic) < 0)
    THROW_TJ("setting TJPARAM_ARITHMETIC");
  if (losslessPSV >= 1 && losslessPSV <= 7) {
    if (tj3Set(tjInstance, TJPARAM_LOSSLESS, 1) < 0)
      THROW_TJ("setting TJPARAM_LOSSLESS");
    if (tj3Set(tjInstance, TJPARAM_LOSSLESSPSV, losslessPSV) < 0)
      THROW_TJ("setting TJPARAM_LOSSLESSPSV");
    if (losslessPt >= 0 &&
        tj3Set(tjInstance, TJPARAM_LOSSLESSPT, losslessPt) < 0)
      THROW_TJ("setting TJPARAM_LOSSLESSPT");
  }
  if (restartIntervalBlocks >= 0 &&
      tj3Set(tjInstance, TJPARAM_RESTARTBLOCKS, restartIntervalBlocks) < 0)
    THROW_TJ("setting TJPARAM_RESTARTBLOCKS");
  if (restartIntervalRows >= 0 &&
      tj3Set(tjInstance, TJPARAM_RESTARTROWS, restartIntervalRows) < 0)
    THROW_TJ("setting TJPARAM_RESTARTROWS");
  if (maxMemory >= 0 && tj3Set(tjInstance, TJPARAM_MAXMEMORY, maxMemory) < 0)
    THROW_TJ("setting TJPARAM_MAXMEMORY");

  if (precision <= 8) {
    if ((srcBuf = tj3LoadImage8(tjInstance, argv[i], &width, 1, &height,
                                &pixelFormat)) == NULL)
      THROW_TJ("loading input image");
  } else if (precision <= 12) {
    if ((srcBuf = tj3LoadImage12(tjInstance, argv[i], &width, 1, &height,
                                 &pixelFormat)) == NULL)
      THROW_TJ("loading input image");
  } else {
    if ((srcBuf = tj3LoadImage16(tjInstance, argv[i], &width, 1, &height,
                                 &pixelFormat)) == NULL)
      THROW_TJ("loading input image");
  }

  if (pixelFormat == TJPF_GRAY && colorspace < 0)
    colorspace = TJCS_GRAY;
  if (colorspace >= 0 &&
      tj3Set(tjInstance, TJPARAM_COLORSPACE, colorspace) < 0)
    THROW_TJ("setting TJPARAM_COLORSPACE");

  if (iccFilename) {
    if ((iccFile = fopen(iccFilename, "rb")) == NULL)
      THROW_UNIX("opening ICC profile");
    if (fseek(iccFile, 0, SEEK_END) < 0 || ((size = ftell(iccFile)) < 0) ||
        fseek(iccFile, 0, SEEK_SET) < 0)
      THROW_UNIX("determining ICC profile size");
    if (size == 0)
      THROW("determining ICC profile size", "ICC profile contains no data");
    iccSize = size;
    if ((iccBuf = (unsigned char *)malloc(iccSize)) == NULL)
      THROW_UNIX("allocating ICC profile buffer");
    if (fread(iccBuf, iccSize, 1, iccFile) < 1)
      THROW_UNIX("reading ICC profile");
    fclose(iccFile);  iccFile = NULL;
    if (tj3SetICCProfile(tjInstance, iccBuf, iccSize) < 0)
      THROW_TJ("setting ICC profile");
    free(iccBuf);  iccBuf = NULL;
  }

  if (precision <= 8) {
    if (tj3Compress8(tjInstance, srcBuf, width, 0, height, pixelFormat,
                     &jpegBuf, &jpegSize) < 0)
      THROW_TJ("compressing image");
  } else if (precision <= 12) {
    if (tj3Compress12(tjInstance, srcBuf, width, 0, height, pixelFormat,
                      &jpegBuf, &jpegSize) < 0)
      THROW_TJ("compressing image");
  } else {
    if (tj3Compress16(tjInstance, srcBuf, width, 0, height, pixelFormat,
                      &jpegBuf, &jpegSize) < 0)
      THROW_TJ("compressing image");
  }

  if ((jpegFile = fopen(argv[++i], "wb")) == NULL)
    THROW_UNIX("opening output file");
  if (fwrite(jpegBuf, jpegSize, 1, jpegFile) < 1)
    THROW_UNIX("writing output file");

bailout:
  tj3Destroy(tjInstance);
  tj3Free(srcBuf);
  if (iccFile) fclose(iccFile);
  free(iccBuf);
  tj3Free(jpegBuf);
  if (jpegFile) fclose(jpegFile);
  return retval;
}
