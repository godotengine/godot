/*
 * Small jpeg decoder library - testing application
 *
 * Copyright (c) 2006, Luc Saillard <luc@saillard.org>
 * All rights reserved.
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * - Redistributions of source code must retain the above copyright notice,
 *  this list of conditions and the following disclaimer.
 *
 * - Redistributions in binary form must reproduce the above copyright notice,
 *  this list of conditions and the following disclaimer in the documentation
 *  and/or other materials provided with the distribution.
 *
 * - Neither the name of the author nor the names of its contributors may be
 *  used to endorse or promote products derived from this software without
 *  specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include "tinyjpeg.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define snprintf(buf, size, fmt, ...) sprintf(buf, fmt, __VA_ARGS__)

static void exitmessage(const char *message) __attribute__((noreturn));
static void exitmessage(const char *message)
{
  printf("%s\n", message);
  exit(0);
}

static int filesize(FILE *fp)
{
  long pos;
  fseek(fp, 0, SEEK_END);
  pos = ftell(fp);
  fseek(fp, 0, SEEK_SET);
  return pos;
}

/**
 * Save a buffer in 24bits Targa format 
 * (BGR byte order)
 */
static void write_tga(const char *filename, int output_format, int width, int height, unsigned char **components)
{
  unsigned char targaheader[18];
  FILE *F;
  char temp[1024];
  unsigned int bufferlen = width * height * 3;
  unsigned char *rgb_data = components[0];

  sprintf(temp, sizeof(temp), filename);

  memset(targaheader,0,sizeof(targaheader));

  targaheader[12] = (unsigned char) (width & 0xFF);
  targaheader[13] = (unsigned char) (width >> 8);
  targaheader[14] = (unsigned char) (height & 0xFF);
  targaheader[15] = (unsigned char) (height >> 8);
  targaheader[17] = 0x20;    /* Top-down, non-interlaced */
  targaheader[2]  = 2;       /* image type = uncompressed RGB */
  targaheader[16] = 24;

  if (output_format == TINYJPEG_FMT_RGB24)
   {
     unsigned char *data = rgb_data + bufferlen - 3;
     do
      { 
	unsigned char c = data[0];
	data[0] = data[2];
	data[2] = c;
	data-=3;
      }
     while (data > rgb_data);
   }

  F = fopen(temp, "wb");
  fwrite(targaheader, sizeof(targaheader), 1, F);
  fwrite(rgb_data, 1, bufferlen, F);
  fclose(F);
}

/**
 * Save a buffer in three files (.Y, .U, .V) useable by yuvsplittoppm
 */
static void write_yuv(const char *filename, int width, int height, unsigned char **components)
{
  FILE *F;
  char temp[1024];

  snprintf(temp, 1024, "%s.Y", filename);
  F = fopen(temp, "wb");
  fwrite(components[0], width, height, F);
  fclose(F);
  snprintf(temp, 1024, "%s.U", filename);
  F = fopen(temp, "wb");
  fwrite(components[1], width*height/4, 1, F);
  fclose(F);
  snprintf(temp, 1024, "%s.V", filename);
  F = fopen(temp, "wb");
  fwrite(components[2], width*height/4, 1, F);
  fclose(F);
}

/**
 * Save a buffer in grey image (pgm format)
 */
static void write_pgm(const char *filename, int width, int height, unsigned char **components)
{
  FILE *F;
  char temp[1024];

  snprintf(temp, 1024, "%s", filename);
  F = fopen(temp, "wb");
  fprintf(F, "P5\n%d %d\n255\n", width, height);
  fwrite(components[0], width, height, F);
  fclose(F);
}

/**
 * Load one jpeg image, and try to decompress 1000 times, and save the result.
 * This is mainly used for benchmarking the decoder, or to test if between each
 * called of the library the DCT is corrected reset (a bug was found).
 */
int load_multiple_times(const char *filename, const char *outfilename, int output_format)
{
  FILE *fp;
  int count, length_of_file;
  unsigned int width, height;
  unsigned char *buf;
  struct jdec_private *jdec;
  unsigned char *components[4];

  jdec = tinyjpeg_init();
  count = 0;

  /* Load the Jpeg into memory */
  fp = fopen(filename, "rb");
  if (fp == NULL)
    exitmessage("Cannot open filename\n");
  length_of_file = filesize(fp);
  buf = (unsigned char *)malloc(length_of_file + 4);
  fread(buf, length_of_file, 1, fp);
  fclose(fp);

  while (count<1000)
   {
     if (tinyjpeg_parse_header(jdec, buf, length_of_file)<0)
       exitmessage(tinyjpeg_get_errorstring(jdec));

     tinyjpeg_decode(jdec, output_format);

     count++;
   }

  /* 
   * Get address for each plane (not only max 3 planes is supported), and
   * depending of the output mode, only some components will be filled 
   * RGB: 1 plane, YUV420P: 3 planes, GREY: 1 plane
   */
  tinyjpeg_get_components(jdec, components);
  tinyjpeg_get_size(jdec, &width, &height);

  /* Save it */
  switch (output_format)
   {
    case TINYJPEG_FMT_RGB24:
    case TINYJPEG_FMT_BGR24:
      write_tga(outfilename, output_format, width, height, components);
      break;
    case TINYJPEG_FMT_YUV420P:
      write_yuv(outfilename, width, height, components);
      break;
    case TINYJPEG_FMT_GREY:
      write_pgm(outfilename, width, height, components);
      break;
   }

  free(buf);
  tinyjpeg_free(jdec);
  return 0;
}

/**
 * Load one jpeg image, and decompress it, and save the result.
 */
int convert_one_image(const char *infilename, const char *outfilename, int output_format)
{
  FILE *fp;
  unsigned int length_of_file;
  unsigned int width, height;
  unsigned char *buf;
  struct jdec_private *jdec;
  unsigned char *components[3];

  /* Load the Jpeg into memory */
  fp = fopen(infilename, "rb");
  if (fp == NULL)
    exitmessage("Cannot open filename\n");
  length_of_file = filesize(fp);
  buf = (unsigned char *)malloc(length_of_file + 4);
  if (buf == NULL)
    exitmessage("Not enough memory for loading file\n");
  fread(buf, length_of_file, 1, fp);
  fclose(fp);

  /* Decompress it */
  jdec = tinyjpeg_init();
  if (jdec == NULL)
    exitmessage("Not enough memory to alloc the structure need for decompressing\n");

  if (tinyjpeg_parse_header(jdec, buf, length_of_file)<0)
    exitmessage(tinyjpeg_get_errorstring(jdec));

  /* Get the size of the image */
  tinyjpeg_get_size(jdec, &width, &height);

  printf("Decoding JPEG image...\n");
  if (tinyjpeg_decode(jdec, output_format) < 0)
    exitmessage(tinyjpeg_get_errorstring(jdec));

  /* 
   * Get address for each plane (not only max 3 planes is supported), and
   * depending of the output mode, only some components will be filled 
   * RGB: 1 plane, YUV420P: 3 planes, GREY: 1 plane
   */
  tinyjpeg_get_components(jdec, components);

  /* Save it */
  switch (output_format)
   {
    case TINYJPEG_FMT_RGB24:
    case TINYJPEG_FMT_BGR24:
      write_tga(outfilename, output_format, width, height, components);
      break;
    case TINYJPEG_FMT_YUV420P:
      write_yuv(outfilename, width, height, components);
      break;
    case TINYJPEG_FMT_GREY:
      write_pgm(outfilename, width, height, components);
      break;
   }

  /* Only called this if the buffers were allocated by tinyjpeg_decode() */
  tinyjpeg_free(jdec);
  /* else called just free(jdec); */

  free(buf);
  return 0;
}

static void usage(void)
{
    fprintf(stderr, "Usage: loadjpeg [options] <input_filename.jpeg> <format> <output_filename>\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  --benchmark - Convert 1000 times the same image\n");
    fprintf(stderr, "format:\n");
    fprintf(stderr, "  yuv420p - output 3 files .Y,.U,.V\n");
    fprintf(stderr, "  rgb24   - output a .tga image\n");
    fprintf(stderr, "  bgr24   - output a .tga image\n");
    fprintf(stderr, "  gray    - output a .pgm image\n");
    exit(1);
}

/**
 * main
 *
 */
int main(int argc, char *argv[])
{
  int output_format = TINYJPEG_FMT_YUV420P;
  char *output_filename, *input_filename;
  clock_t start_time, finish_time;
  unsigned int duration;
  int current_argument;
  int benchmark_mode = 0;

  if (argc < 3)
    usage();

  current_argument = 1;
  while (1)
   {
     if (strcmp(argv[current_argument], "--benchmark")==0)
       benchmark_mode = 1;
     else
       break;
     current_argument++;
   }

  if (argc < current_argument+2)
    usage();

  input_filename = argv[current_argument];
  if (strcmp(argv[current_argument+1],"yuv420p")==0)
    output_format = TINYJPEG_FMT_YUV420P;
  else if (strcmp(argv[current_argument+1],"rgb24")==0)
    output_format = TINYJPEG_FMT_RGB24;
  else if (strcmp(argv[current_argument+1],"bgr24")==0)
    output_format = TINYJPEG_FMT_BGR24;
  else if (strcmp(argv[current_argument+1],"grey")==0)
    output_format = TINYJPEG_FMT_GREY;
  else
    exitmessage("Bad format: need to be one of yuv420p, rgb24, bgr24, grey\n");
  output_filename = argv[current_argument+2];

  start_time = clock();

  if (benchmark_mode)
    load_multiple_times(input_filename, output_filename, output_format);
  else
    convert_one_image(input_filename, output_filename, output_format);

  finish_time = clock();
  duration = finish_time - start_time;
  printf("Decoding finished in %u ticks\n", duration);

  return 0;
}




