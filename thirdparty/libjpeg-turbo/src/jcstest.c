/*
 * Copyright (C)2011 D. R. Commander.  All Rights Reserved.
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

/* This program demonstrates how to check for the colorspace extension
   capabilities of libjpeg-turbo at both compile time and run time. */

#include <stdio.h>
#include <jpeglib.h>
#include <jerror.h>
#include <setjmp.h>

#ifndef JCS_EXTENSIONS
#define JCS_EXT_RGB  6
#endif
#if !defined(JCS_EXTENSIONS) || !defined(JCS_ALPHA_EXTENSIONS)
#define JCS_EXT_RGBA  12
#endif

static char lasterror[JMSG_LENGTH_MAX] = "No error";

typedef struct _error_mgr {
  struct jpeg_error_mgr pub;
  jmp_buf jb;
} error_mgr;

static void my_error_exit(j_common_ptr cinfo)
{
  error_mgr *myerr = (error_mgr *)cinfo->err;
  (*cinfo->err->output_message) (cinfo);
  longjmp(myerr->jb, 1);
}

static void my_output_message(j_common_ptr cinfo)
{
  (*cinfo->err->format_message) (cinfo, lasterror);
}

int main(void)
{
  int jcs_valid = -1, jcs_alpha_valid = -1;
  struct jpeg_compress_struct cinfo;
  error_mgr jerr;

  printf("libjpeg-turbo colorspace extensions:\n");
#if JCS_EXTENSIONS
  printf("  Present at compile time\n");
#else
  printf("  Not present at compile time\n");
#endif

  cinfo.err = jpeg_std_error(&jerr.pub);
  jerr.pub.error_exit = my_error_exit;
  jerr.pub.output_message = my_output_message;

  if (setjmp(jerr.jb)) {
    /* this will execute if libjpeg has an error */
    jcs_valid = 0;
    goto done;
  }

  jpeg_create_compress(&cinfo);
  cinfo.input_components = 3;
  jpeg_set_defaults(&cinfo);
  cinfo.in_color_space = JCS_EXT_RGB;
  jpeg_default_colorspace(&cinfo);
  jcs_valid = 1;

done:
  if (jcs_valid)
    printf("  Working properly\n");
  else
    printf("  Not working properly.  Error returned was:\n    %s\n",
           lasterror);

  printf("libjpeg-turbo alpha colorspace extensions:\n");
#if JCS_ALPHA_EXTENSIONS
  printf("  Present at compile time\n");
#else
  printf("  Not present at compile time\n");
#endif

  if (setjmp(jerr.jb)) {
    /* this will execute if libjpeg has an error */
    jcs_alpha_valid = 0;
    goto done2;
  }

  cinfo.in_color_space = JCS_EXT_RGBA;
  jpeg_default_colorspace(&cinfo);
  jcs_alpha_valid = 1;

done2:
  if (jcs_alpha_valid)
    printf("  Working properly\n");
  else
    printf("  Not working properly.  Error returned was:\n    %s\n",
           lasterror);

  jpeg_destroy_compress(&cinfo);
  return 0;
}
