/*
 * Copyright (C)2009-2024 D. R. Commander.  All Rights Reserved.
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

/* TurboJPEG API functions that must be compiled for multiple data
   precisions */

#if BITS_IN_JSAMPLE == 8
#define _JSAMPLE  JSAMPLE
#define _JSAMPROW  JSAMPROW
#define _buffer  buffer
#define _jinit_read_ppm  jinit_read_ppm
#define _jinit_write_ppm  jinit_write_ppm
#define _jpeg_crop_scanline  jpeg_crop_scanline
#define _jpeg_read_scanlines  jpeg_read_scanlines
#define _jpeg_skip_scanlines  jpeg_skip_scanlines
#define _jpeg_write_scanlines  jpeg_write_scanlines
#elif BITS_IN_JSAMPLE == 12
#define _JSAMPLE  J12SAMPLE
#define _JSAMPROW  J12SAMPROW
#define _buffer  buffer12
#define _jinit_read_ppm  j12init_read_ppm
#define _jinit_write_ppm  j12init_write_ppm
#define _jpeg_crop_scanline  jpeg12_crop_scanline
#define _jpeg_read_scanlines  jpeg12_read_scanlines
#define _jpeg_skip_scanlines  jpeg12_skip_scanlines
#define _jpeg_write_scanlines  jpeg12_write_scanlines
#elif BITS_IN_JSAMPLE == 16
#define _JSAMPLE  J16SAMPLE
#define _JSAMPROW  J16SAMPROW
#define _buffer  buffer16
#define _jinit_read_ppm  j16init_read_ppm
#define _jinit_write_ppm  j16init_write_ppm
#define _jpeg_read_scanlines  jpeg16_read_scanlines
#define _jpeg_write_scanlines  jpeg16_write_scanlines
#endif

#define _GET_NAME(name, suffix)  name##suffix
#define GET_NAME(name, suffix)  _GET_NAME(name, suffix)
#define _GET_STRING(name, suffix)  #name #suffix
#define GET_STRING(name, suffix)  _GET_STRING(name, suffix)


/******************************** Compressor *********************************/

/* TurboJPEG 3.0+ */
DLLEXPORT int GET_NAME(tj3Compress, BITS_IN_JSAMPLE)
  (tjhandle handle, const _JSAMPLE *srcBuf, int width, int pitch, int height,
   int pixelFormat, unsigned char **jpegBuf, size_t *jpegSize)
{
  static const char FUNCTION_NAME[] = GET_STRING(tj3Compress, BITS_IN_JSAMPLE);
  int i, retval = 0;
  boolean alloc = TRUE;
  _JSAMPROW *row_pointer = NULL;

  GET_CINSTANCE(handle)
  if ((this->init & COMPRESS) == 0)
    THROW("Instance has not been initialized for compression");

  if (srcBuf == NULL || width <= 0 || pitch < 0 || height <= 0 ||
      pixelFormat < 0 || pixelFormat >= TJ_NUMPF || jpegBuf == NULL ||
      jpegSize == NULL)
    THROW("Invalid argument");

  if (!this->lossless && this->quality == -1)
    THROW("TJPARAM_QUALITY must be specified");
  if (!this->lossless && this->subsamp == TJSAMP_UNKNOWN)
    THROW("TJPARAM_SUBSAMP must be specified");

  if (pitch == 0) pitch = width * tjPixelSize[pixelFormat];

  if ((row_pointer = (_JSAMPROW *)malloc(sizeof(_JSAMPROW) * height)) == NULL)
    THROW("Memory allocation failure");

  if (setjmp(this->jerr.setjmp_buffer)) {
    /* If we get here, the JPEG code has signaled an error. */
    retval = -1;  goto bailout;
  }

  cinfo->image_width = width;
  cinfo->image_height = height;
  cinfo->data_precision = BITS_IN_JSAMPLE;
#if BITS_IN_JSAMPLE == 8
  if (this->lossless && this->precision >= 2 &&
      this->precision <= BITS_IN_JSAMPLE)
#else
  if (this->lossless && this->precision >= BITS_IN_JSAMPLE - 3 &&
      this->precision <= BITS_IN_JSAMPLE)
#endif
    cinfo->data_precision = this->precision;

  setCompDefaults(this, pixelFormat);
  if (this->noRealloc) alloc = FALSE;
  jpeg_mem_dest_tj(cinfo, jpegBuf, jpegSize, alloc);

  jpeg_start_compress(cinfo, TRUE);
  if (this->iccBuf != NULL && this->iccSize != 0)
    jpeg_write_icc_profile(cinfo, this->iccBuf, (unsigned int)this->iccSize);
  for (i = 0; i < height; i++) {
    if (this->bottomUp)
      row_pointer[i] = (_JSAMPROW)&srcBuf[(height - i - 1) * (size_t)pitch];
    else
      row_pointer[i] = (_JSAMPROW)&srcBuf[i * (size_t)pitch];
  }
  while (cinfo->next_scanline < cinfo->image_height)
    _jpeg_write_scanlines(cinfo, &row_pointer[cinfo->next_scanline],
                          cinfo->image_height - cinfo->next_scanline);
  jpeg_finish_compress(cinfo);

bailout:
  if (cinfo->global_state > CSTATE_START && alloc)
    (*cinfo->dest->term_destination) (cinfo);
  if (cinfo->global_state > CSTATE_START || retval == -1)
    jpeg_abort_compress(cinfo);
  free(row_pointer);
  if (this->jerr.warning) retval = -1;
  return retval;
}


/******************************* Decompressor ********************************/

/* TurboJPEG 3.0+ */
DLLEXPORT int GET_NAME(tj3Decompress, BITS_IN_JSAMPLE)
  (tjhandle handle, const unsigned char *jpegBuf, size_t jpegSize,
   _JSAMPLE *dstBuf, int pitch, int pixelFormat)
{
  static const char FUNCTION_NAME[] =
    GET_STRING(tj3Decompress, BITS_IN_JSAMPLE);
  _JSAMPROW *row_pointer = NULL;
  int croppedHeight, i, retval = 0;
#if BITS_IN_JSAMPLE != 16
  int scaledWidth;
#endif
  struct my_progress_mgr progress;

  GET_DINSTANCE(handle);
  if ((this->init & DECOMPRESS) == 0)
    THROW("Instance has not been initialized for decompression");

  if (jpegBuf == NULL || jpegSize <= 0 || dstBuf == NULL || pitch < 0 ||
      pixelFormat < 0 || pixelFormat >= TJ_NUMPF)
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
  this->dinfo.out_color_space = pf2cs[pixelFormat];
#if BITS_IN_JSAMPLE != 16
  scaledWidth = TJSCALED(dinfo->image_width, this->scalingFactor);
#endif
  dinfo->do_fancy_upsampling = !this->fastUpsample;
  this->dinfo.dct_method = this->fastDCT ? JDCT_FASTEST : JDCT_ISLOW;

  dinfo->scale_num = this->scalingFactor.num;
  dinfo->scale_denom = this->scalingFactor.denom;

  jpeg_start_decompress(dinfo);

#if BITS_IN_JSAMPLE != 16
  if (this->croppingRegion.x != 0 ||
      (this->croppingRegion.w != 0 && this->croppingRegion.w != scaledWidth)) {
    JDIMENSION crop_x = this->croppingRegion.x;
    JDIMENSION crop_w = this->croppingRegion.w;

    _jpeg_crop_scanline(dinfo, &crop_x, &crop_w);
    if ((int)crop_x != this->croppingRegion.x)
      THROWI("Unexplained mismatch between specified (%d) and\n"
             "actual (%d) cropping region left boundary",
             this->croppingRegion.x, (int)crop_x);
    if ((int)crop_w != this->croppingRegion.w)
      THROWI("Unexplained mismatch between specified (%d) and\n"
             "actual (%d) cropping region width",
             this->croppingRegion.w, (int)crop_w);
  }
#endif

  if (pitch == 0) pitch = dinfo->output_width * tjPixelSize[pixelFormat];

  croppedHeight = dinfo->output_height;
#if BITS_IN_JSAMPLE != 16
  if (this->croppingRegion.y != 0 || this->croppingRegion.h != 0)
    croppedHeight = this->croppingRegion.h;
#endif
  if ((row_pointer =
       (_JSAMPROW *)malloc(sizeof(_JSAMPROW) * croppedHeight)) == NULL)
    THROW("Memory allocation failure");
  if (setjmp(this->jerr.setjmp_buffer)) {
    /* If we get here, the JPEG code has signaled an error. */
    retval = -1;  goto bailout;
  }
  for (i = 0; i < (int)croppedHeight; i++) {
    if (this->bottomUp)
      row_pointer[i] = &dstBuf[(croppedHeight - i - 1) * (size_t)pitch];
    else
      row_pointer[i] = &dstBuf[i * (size_t)pitch];
  }

#if BITS_IN_JSAMPLE != 16
  if (this->croppingRegion.y != 0 || this->croppingRegion.h != 0) {
    if (this->croppingRegion.y != 0) {
      JDIMENSION lines = _jpeg_skip_scanlines(dinfo, this->croppingRegion.y);

      if ((int)lines != this->croppingRegion.y)
        THROWI("Unexplained mismatch between specified (%d) and\n"
               "actual (%d) cropping region upper boundary",
               this->croppingRegion.y, (int)lines);
    }
    while ((int)dinfo->output_scanline <
           this->croppingRegion.y + this->croppingRegion.h)
      _jpeg_read_scanlines(dinfo, &row_pointer[dinfo->output_scanline -
                                               this->croppingRegion.y],
                           this->croppingRegion.y + this->croppingRegion.h -
                           dinfo->output_scanline);
    if (this->croppingRegion.y + this->croppingRegion.h !=
        (int)dinfo->output_height) {
      JDIMENSION lines = _jpeg_skip_scanlines(dinfo, dinfo->output_height -
                                                     this->croppingRegion.y -
                                                     this->croppingRegion.h);

      if (lines != dinfo->output_height - this->croppingRegion.y -
                   this->croppingRegion.h)
        THROWI("Unexplained mismatch between specified (%d) and\n"
               "actual (%d) cropping region lower boundary",
               this->croppingRegion.y + this->croppingRegion.h,
               (int)(dinfo->output_height - lines));
    }
  } else
#endif
  {
    while (dinfo->output_scanline < dinfo->output_height)
      _jpeg_read_scanlines(dinfo, &row_pointer[dinfo->output_scanline],
                           dinfo->output_height - dinfo->output_scanline);
  }
  jpeg_finish_decompress(dinfo);

bailout:
  if (dinfo->global_state > DSTATE_START) jpeg_abort_decompress(dinfo);
  free(row_pointer);
  if (this->jerr.warning) retval = -1;
  return retval;
}


/*************************** Packed-Pixel Image I/O **************************/

/* TurboJPEG 3.0+ */
DLLEXPORT _JSAMPLE *GET_NAME(tj3LoadImage, BITS_IN_JSAMPLE)
  (tjhandle handle, const char *filename, int *width, int align, int *height,
   int *pixelFormat)
{
  static const char FUNCTION_NAME[] =
    GET_STRING(tj3LoadImage, BITS_IN_JSAMPLE);

#if BITS_IN_JSAMPLE != 16 || defined(C_LOSSLESS_SUPPORTED)

  int retval = 0, tempc;
  size_t pitch;
  tjhandle handle2 = NULL;
  tjinstance *this2;
  j_compress_ptr cinfo = NULL;
  cjpeg_source_ptr src;
  _JSAMPLE *dstBuf = NULL;
  FILE *file = NULL;
  boolean invert;

  GET_TJINSTANCE(handle, NULL)

  if (!filename || !width || align < 1 || !height || !pixelFormat ||
      *pixelFormat < TJPF_UNKNOWN || *pixelFormat >= TJ_NUMPF)
    THROW("Invalid argument");
  if ((align & (align - 1)) != 0)
    THROW("Alignment must be a power of 2");

  /* The instance handle passed to this function is used only for parameter
     retrieval.  Create a new temporary instance to avoid interfering with the
     libjpeg state of the primary instance. */
  if ((handle2 = tj3Init(TJINIT_COMPRESS)) == NULL) return NULL;
  this2 = (tjinstance *)handle2;
  cinfo = &this2->cinfo;

#ifdef _MSC_VER
  if (fopen_s(&file, filename, "rb") || file == NULL)
#else
  if ((file = fopen(filename, "rb")) == NULL)
#endif
    THROW_UNIX("Cannot open input file");

  if ((tempc = getc(file)) < 0 || ungetc(tempc, file) == EOF)
    THROW_UNIX("Could not read input file")
  else if (tempc == EOF)
    THROW("Input file contains no data");

  if (setjmp(this2->jerr.setjmp_buffer)) {
    /* If we get here, the JPEG code has signaled an error. */
    retval = -1;  goto bailout;
  }

  cinfo->data_precision = BITS_IN_JSAMPLE;
  if (*pixelFormat == TJPF_UNKNOWN) cinfo->in_color_space = JCS_UNKNOWN;
  else cinfo->in_color_space = pf2cs[*pixelFormat];
  if (tempc == 'B') {
    if ((src = jinit_read_bmp(cinfo, FALSE)) == NULL)
      THROW("Could not initialize bitmap loader");
    invert = !this->bottomUp;
  } else if (tempc == 'P') {
#if BITS_IN_JSAMPLE == 8
    if (this->precision >= 2 && this->precision <= BITS_IN_JSAMPLE)
#else
    if (this->precision >= BITS_IN_JSAMPLE - 3 &&
        this->precision <= BITS_IN_JSAMPLE)
#endif
      cinfo->data_precision = this->precision;
    if ((src = _jinit_read_ppm(cinfo)) == NULL)
      THROW("Could not initialize PPM loader");
    invert = this->bottomUp;
  } else
    THROW("Unsupported file type");

  cinfo->mem->max_memory_to_use = (long)this->maxMemory * 1048576L;

  src->input_file = file;
  /* Refuse to load images larger than the specified size. */
  src->max_pixels = this->maxPixels;
  (*src->start_input) (cinfo, src);
  if (tempc == 'B') {
    if (cinfo->X_density && cinfo->Y_density) {
      this->xDensity = cinfo->X_density;
      this->yDensity = cinfo->Y_density;
      this->densityUnits = cinfo->density_unit;
    }
  }
  (*cinfo->mem->realize_virt_arrays) ((j_common_ptr)cinfo);

  *width = cinfo->image_width;  *height = cinfo->image_height;
  *pixelFormat = cs2pf[cinfo->in_color_space];

  pitch = PAD((*width) * tjPixelSize[*pixelFormat], align);
  if (
#if ULLONG_MAX > SIZE_MAX
      (unsigned long long)pitch * (unsigned long long)(*height) >
      (unsigned long long)((size_t)-1) ||
#endif
      (dstBuf = (_JSAMPLE *)malloc(pitch * (*height) *
                                   sizeof(_JSAMPLE))) == NULL)
    THROW("Memory allocation failure");

  if (setjmp(this2->jerr.setjmp_buffer)) {
    /* If we get here, the JPEG code has signaled an error. */
    retval = -1;  goto bailout;
  }

  while (cinfo->next_scanline < cinfo->image_height) {
    int i, nlines = (*src->get_pixel_rows) (cinfo, src);

    for (i = 0; i < nlines; i++) {
      _JSAMPLE *dstptr;
      int row;

      row = cinfo->next_scanline + i;
      if (invert) dstptr = &dstBuf[((*height) - row - 1) * pitch];
      else dstptr = &dstBuf[row * pitch];
      memcpy(dstptr, src->_buffer[i],
             (*width) * tjPixelSize[*pixelFormat] * sizeof(_JSAMPLE));
    }
    cinfo->next_scanline += nlines;
  }

  (*src->finish_input) (cinfo, src);

bailout:
  tj3Destroy(handle2);
  if (file) fclose(file);
  if (retval < 0) { free(dstBuf);  dstBuf = NULL; }
  return dstBuf;

#else /* BITS_IN_JSAMPLE != 16 || defined(C_LOSSLESS_SUPPORTED) */

  static const char ERROR_MSG[] =
    "16-bit data precision requires lossless JPEG,\n"
    "which was disabled at build time.";
  _JSAMPLE *retval = NULL;

  GET_TJINSTANCE(handle, NULL)
  SNPRINTF(this->errStr, JMSG_LENGTH_MAX, "%s(): %s", FUNCTION_NAME,
           ERROR_MSG);
  this->isInstanceError = TRUE;  THROWG(ERROR_MSG, NULL)

bailout:
  return retval;

#endif
}


/* TurboJPEG 3.0+ */
DLLEXPORT int GET_NAME(tj3SaveImage, BITS_IN_JSAMPLE)
  (tjhandle handle, const char *filename, const _JSAMPLE *buffer, int width,
   int pitch, int height, int pixelFormat)
{
  static const char FUNCTION_NAME[] =
    GET_STRING(tj3SaveImage, BITS_IN_JSAMPLE);
  int retval = 0;

#if BITS_IN_JSAMPLE != 16 || defined(D_LOSSLESS_SUPPORTED)

  tjhandle handle2 = NULL;
  tjinstance *this2;
  j_decompress_ptr dinfo = NULL;
  djpeg_dest_ptr dst;
  FILE *file = NULL;
  char *ptr = NULL;
  boolean invert;

  GET_TJINSTANCE(handle, -1)

  if (!filename || !buffer || width < 1 || pitch < 0 || height < 1 ||
      pixelFormat < 0 || pixelFormat >= TJ_NUMPF)
    THROW("Invalid argument");

  /* The instance handle passed to this function is used only for parameter
     retrieval.  Create a new temporary instance to avoid interfering with the
     libjpeg state of the primary instance. */
  if ((handle2 = tj3Init(TJINIT_DECOMPRESS)) == NULL)
    return -1;
  this2 = (tjinstance *)handle2;
  dinfo = &this2->dinfo;

#ifdef _MSC_VER
  if (fopen_s(&file, filename, "wb") || file == NULL)
#else
  if ((file = fopen(filename, "wb")) == NULL)
#endif
    THROW_UNIX("Cannot open output file");

  if (setjmp(this2->jerr.setjmp_buffer)) {
    /* If we get here, the JPEG code has signaled an error. */
    retval = -1;  goto bailout;
  }

  this2->dinfo.out_color_space = pf2cs[pixelFormat];
  dinfo->image_width = width;  dinfo->image_height = height;
  dinfo->global_state = DSTATE_READY;
  dinfo->scale_num = dinfo->scale_denom = 1;
  dinfo->data_precision = BITS_IN_JSAMPLE;

  ptr = strrchr(filename, '.');
  if (ptr && !strcasecmp(ptr, ".bmp")) {
    if ((dst = jinit_write_bmp(dinfo, FALSE, FALSE)) == NULL)
      THROW("Could not initialize bitmap writer");
    invert = !this->bottomUp;
    dinfo->X_density = (UINT16)this->xDensity;
    dinfo->Y_density = (UINT16)this->yDensity;
    dinfo->density_unit = (UINT8)this->densityUnits;
  } else {
#if BITS_IN_JSAMPLE == 8
    if (this->precision >= 2 && this->precision <= BITS_IN_JSAMPLE)
#else
    if (this->precision >= BITS_IN_JSAMPLE - 3 &&
        this->precision <= BITS_IN_JSAMPLE)
#endif
      dinfo->data_precision = this->precision;
    if ((dst = _jinit_write_ppm(dinfo)) == NULL)
      THROW("Could not initialize PPM writer");
    invert = this->bottomUp;
  }

  dinfo->mem->max_memory_to_use = (long)this->maxMemory * 1048576L;

  dst->output_file = file;
  (*dst->start_output) (dinfo, dst);
  (*dinfo->mem->realize_virt_arrays) ((j_common_ptr)dinfo);

  if (pitch == 0) pitch = width * tjPixelSize[pixelFormat];

  while (dinfo->output_scanline < dinfo->output_height) {
    _JSAMPLE *rowptr;

    if (invert)
      rowptr =
        (_JSAMPLE *)&buffer[(height - dinfo->output_scanline - 1) * pitch];
    else
      rowptr = (_JSAMPLE *)&buffer[dinfo->output_scanline * pitch];
    memcpy(dst->_buffer[0], rowptr,
           width * tjPixelSize[pixelFormat] * sizeof(_JSAMPLE));
    (*dst->put_pixel_rows) (dinfo, dst, 1);
    dinfo->output_scanline++;
  }

  (*dst->finish_output) (dinfo, dst);

bailout:
  tj3Destroy(handle2);
  if (file) fclose(file);
  return retval;

#else /* BITS_IN_JSAMPLE != 16 || defined(D_LOSSLESS_SUPPORTED) */

  GET_TJINSTANCE(handle, -1)
  THROW("16-bit data precision requires lossless JPEG,\n"
        "which was disabled at build time.")
bailout:
  return retval;

#endif
}


#undef _JSAMPLE
#undef _JSAMPROW
#undef _buffer
#undef _jinit_read_ppm
#undef _jinit_write_ppm
#undef _jpeg_crop_scanline
#undef _jpeg_read_scanlines
#undef _jpeg_skip_scanlines
#undef _jpeg_write_scanlines
