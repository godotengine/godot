/**************************************************************************
 *
 * Copyright 2010 VMware, Inc.
 * All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sub license, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice (including the
 * next paragraph) shall be included in all copies or substantial portions
 * of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
 * IN NO EVENT SHALL VMWARE AND/OR ITS SUPPLIERS BE LIABLE FOR
 * ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 **************************************************************************/

/**
 * @file
 * Pixel format accessor functions.
 *
 * @author Jose Fonseca <jfonseca@vmware.com>
 */

#include "c11/threads.h"
#include "util/detect_arch.h"
#include "util/format/u_format.h"
#include "util/format/u_format_s3tc.h"
#include "util/u_math.h"

#include "pipe/p_defines.h"
#include "pipe/p_screen.h"


/**
 * Copy 2D rect from one place to another.
 * Position and sizes are in pixels.
 * src_stride may be negative to do vertical flip of pixels from source.
 */
void
util_copy_rect(void * dst_in,
               enum pipe_format format,
               unsigned dst_stride,
               unsigned dst_x,
               unsigned dst_y,
               unsigned width,
               unsigned height,
               const void * src_in,
               int src_stride,
               unsigned src_x,
               unsigned src_y)
{
   uint8_t *dst = dst_in;
   const uint8_t *src = src_in;
   unsigned i;
   int src_stride_pos = src_stride < 0 ? -src_stride : src_stride;
   int blocksize = util_format_get_blocksize(format);
   int blockwidth = util_format_get_blockwidth(format);
   int blockheight = util_format_get_blockheight(format);

   assert(blocksize > 0);
   assert(blockwidth > 0);
   assert(blockheight > 0);

   dst_x /= blockwidth;
   dst_y /= blockheight;
   width = (width + blockwidth - 1)/blockwidth;
   height = (height + blockheight - 1)/blockheight;
   src_x /= blockwidth;
   src_y /= blockheight;

   dst += dst_x * blocksize;
   src += src_x * blocksize;
   dst += dst_y * dst_stride;
   src += src_y * src_stride_pos;
   width *= blocksize;

   if (width == dst_stride && width == (unsigned)src_stride)
      memcpy(dst, src, height * width);
   else {
      for (i = 0; i < height; i++) {
         memcpy(dst, src, width);
         dst += dst_stride;
         src += src_stride;
      }
   }
}


bool
util_format_is_float(enum pipe_format format)
{
   const struct util_format_description *desc = util_format_description(format);
   int i;

   i = util_format_get_first_non_void_channel(format);
   if (i < 0) {
      return false;
   }

   return desc->channel[i].type == UTIL_FORMAT_TYPE_FLOAT ? true : false;
}


/** Test if the format contains RGB, but not alpha */
bool
util_format_has_alpha(enum pipe_format format)
{
   const struct util_format_description *desc =
      util_format_description(format);

   return (desc->colorspace == UTIL_FORMAT_COLORSPACE_RGB ||
           desc->colorspace == UTIL_FORMAT_COLORSPACE_SRGB) &&
          desc->swizzle[3] != PIPE_SWIZZLE_1;
}

/** Test if format has alpha as 1 (like RGBX) */
bool
util_format_has_alpha1(enum pipe_format format)
{
   const struct util_format_description *desc =
      util_format_description(format);

   return (desc->colorspace == UTIL_FORMAT_COLORSPACE_RGB ||
           desc->colorspace == UTIL_FORMAT_COLORSPACE_SRGB) &&
           desc->nr_channels == 4 &&
           desc->swizzle[3] == PIPE_SWIZZLE_1;
}

bool
util_format_is_luminance(enum pipe_format format)
{
   const struct util_format_description *desc =
      util_format_description(format);

   if ((desc->colorspace == UTIL_FORMAT_COLORSPACE_RGB ||
        desc->colorspace == UTIL_FORMAT_COLORSPACE_SRGB) &&
       desc->swizzle[0] == PIPE_SWIZZLE_X &&
       desc->swizzle[1] == PIPE_SWIZZLE_X &&
       desc->swizzle[2] == PIPE_SWIZZLE_X &&
       desc->swizzle[3] == PIPE_SWIZZLE_1) {
      return true;
   }
   return false;
}

bool
util_format_is_alpha(enum pipe_format format)
{
   const struct util_format_description *desc =
      util_format_description(format);

   if ((desc->colorspace == UTIL_FORMAT_COLORSPACE_RGB ||
        desc->colorspace == UTIL_FORMAT_COLORSPACE_SRGB) &&
       desc->swizzle[0] == PIPE_SWIZZLE_0 &&
       desc->swizzle[1] == PIPE_SWIZZLE_0 &&
       desc->swizzle[2] == PIPE_SWIZZLE_0 &&
       desc->swizzle[3] == PIPE_SWIZZLE_X) {
      return true;
   }
   return false;
}

bool
util_format_is_pure_integer(enum pipe_format format)
{
   const struct util_format_description *desc = util_format_description(format);
   int i;

   if (desc->colorspace == UTIL_FORMAT_COLORSPACE_ZS) {
      if (util_format_has_depth(desc))
         return false;

      assert(util_format_has_stencil(desc));
      return true;
   }

   /* Find the first non-void channel. */
   i = util_format_get_first_non_void_channel(format);
   if (i == -1)
      return false;

   return desc->channel[i].pure_integer ? true : false;
}

bool
util_format_is_pure_sint(enum pipe_format format)
{
   const struct util_format_description *desc = util_format_description(format);
   int i;

   i = util_format_get_first_non_void_channel(format);
   if (i == -1)
      return false;

   return (desc->channel[i].type == UTIL_FORMAT_TYPE_SIGNED && desc->channel[i].pure_integer) ? true : false;
}

bool
util_format_is_pure_uint(enum pipe_format format)
{
   const struct util_format_description *desc = util_format_description(format);
   int i;

   i = util_format_get_first_non_void_channel(format);
   if (i == -1)
      return false;

   return (desc->channel[i].type == UTIL_FORMAT_TYPE_UNSIGNED && desc->channel[i].pure_integer) ? true : false;
}

/**
 * Returns true if the format contains normalized signed channels.
 */
bool
util_format_is_snorm(enum pipe_format format)
{
   const struct util_format_description *desc = util_format_description(format);

   return desc->is_snorm;
}

/**
 * Returns true if the format contains normalized unsigned channels.
 */
bool
util_format_is_unorm(enum pipe_format format)
{
   const struct util_format_description *desc = util_format_description(format);

   return desc->is_unorm;
}

/**
 * Returns true if the format contains scaled integer format channels.
 */
bool
util_format_is_scaled(enum pipe_format format)
{
   const struct util_format_description *desc = util_format_description(format);
   int i;

   /* format none is described as scaled but not for this check */
   if (format == PIPE_FORMAT_NONE)
      return false;

   /* Find the first non-void channel. */
   i = util_format_get_first_non_void_channel(format);
   if (i == -1)
      return false;

   return !desc->channel[i].pure_integer && !desc->channel[i].normalized &&
      (desc->channel[i].type == UTIL_FORMAT_TYPE_SIGNED ||
       desc->channel[i].type == UTIL_FORMAT_TYPE_UNSIGNED);
}

bool
util_format_is_snorm8(enum pipe_format format)
{
   const struct util_format_description *desc = util_format_description(format);
   int i;

   if (desc->is_mixed)
      return false;

   i = util_format_get_first_non_void_channel(format);
   if (i == -1)
      return false;

   return desc->channel[i].type == UTIL_FORMAT_TYPE_SIGNED &&
          !desc->channel[i].pure_integer &&
          desc->channel[i].normalized &&
          desc->channel[i].size == 8;
}

bool
util_format_is_luminance_alpha(enum pipe_format format)
{
   const struct util_format_description *desc =
      util_format_description(format);

   if ((desc->colorspace == UTIL_FORMAT_COLORSPACE_RGB ||
        desc->colorspace == UTIL_FORMAT_COLORSPACE_SRGB) &&
       desc->swizzle[0] == PIPE_SWIZZLE_X &&
       desc->swizzle[1] == PIPE_SWIZZLE_X &&
       desc->swizzle[2] == PIPE_SWIZZLE_X &&
       desc->swizzle[3] == PIPE_SWIZZLE_Y) {
      return true;
   }
   return false;
}


bool
util_format_is_intensity(enum pipe_format format)
{
   const struct util_format_description *desc =
      util_format_description(format);

   if ((desc->colorspace == UTIL_FORMAT_COLORSPACE_RGB ||
        desc->colorspace == UTIL_FORMAT_COLORSPACE_SRGB) &&
       desc->swizzle[0] == PIPE_SWIZZLE_X &&
       desc->swizzle[1] == PIPE_SWIZZLE_X &&
       desc->swizzle[2] == PIPE_SWIZZLE_X &&
       desc->swizzle[3] == PIPE_SWIZZLE_X) {
      return true;
   }
   return false;
}

bool
util_format_is_subsampled_422(enum pipe_format format)
{
   const struct util_format_description *desc =
      util_format_description(format);

   return desc->layout == UTIL_FORMAT_LAYOUT_SUBSAMPLED &&
      desc->block.width == 2 &&
      desc->block.height == 1 &&
      desc->block.bits == 32;
}

/**
 * Calculates the MRD for the depth format. MRD is used in depth bias
 * for UNORM and unbound depth buffers. When the depth buffer is floating
 * point, the depth bias calculation does not use the MRD. However, the
 * default MRD will be 1.0 / ((1 << 24) - 1).
 */
double
util_get_depth_format_mrd(const struct util_format_description *desc)
{
   /*
    * Depth buffer formats without a depth component OR scenarios
    * without a bound depth buffer default to D24.
    */
   double mrd = 1.0 / ((1 << 24) - 1);
   unsigned depth_channel;

   /*
    * Some depth formats do not store the depth component in the first
    * channel, detect the format and adjust the depth channel. Get the
    * swizzled depth component channel.
    */
   depth_channel = desc->swizzle[0];

   if (desc->channel[depth_channel].type == UTIL_FORMAT_TYPE_UNSIGNED &&
       desc->channel[depth_channel].normalized) {
      int depth_bits;

      depth_bits = desc->channel[depth_channel].size;
      mrd = 1.0 / ((1ULL << depth_bits) - 1);
   }

   return mrd;
}

void
util_format_unpack_rgba_rect(enum pipe_format format,
                   void *dst, unsigned dst_stride,
                   const void *src, unsigned src_stride,
                   unsigned w, unsigned h)
{
   const struct util_format_unpack_description *unpack =
      util_format_unpack_description(format);

   /* Optimized function for block-compressed formats */
   if (unpack->unpack_rgba_rect) {
      unpack->unpack_rgba_rect(dst, dst_stride, src, src_stride, w, h);
   } else {
     for (unsigned y = 0; y < h; y++) {
        unpack->unpack_rgba(dst, src, w);
        src = (const char *)src + src_stride;
        dst = (char *)dst + dst_stride;
     }
  }
}

void
util_format_unpack_rgba_8unorm_rect(enum pipe_format format,
                   void *dst, unsigned dst_stride,
                   const void *src, unsigned src_stride,
                   unsigned w, unsigned h)
{
   const struct util_format_unpack_description *unpack =
      util_format_unpack_description(format);

   /* Optimized function for block-compressed formats */
   if (unpack->unpack_rgba_8unorm_rect) {
      unpack->unpack_rgba_8unorm_rect(dst, dst_stride, src, src_stride, w, h);
   } else {
     for (unsigned y = 0; y < h; y++) {
        unpack->unpack_rgba_8unorm(dst, src, w);
        src = (const char *)src + src_stride;
        dst = (char *)dst + dst_stride;
     }
  }
}

void
util_format_read_4(enum pipe_format format,
                   void *dst, unsigned dst_stride,
                   const void *src, unsigned src_stride,
                   unsigned x, unsigned y, unsigned w, unsigned h)
{
   const struct util_format_description *format_desc;
   const uint8_t *src_row;

   format_desc = util_format_description(format);

   assert(x % format_desc->block.width == 0);
   assert(y % format_desc->block.height == 0);

   src_row = (const uint8_t *)src + y*src_stride + x*(format_desc->block.bits/8);

   util_format_unpack_rgba_rect(format, dst, dst_stride, src_row, src_stride, w, h);
}


void
util_format_write_4(enum pipe_format format,
                     const void *src, unsigned src_stride,
                     void *dst, unsigned dst_stride,
                     unsigned x, unsigned y, unsigned w, unsigned h)
{
   const struct util_format_description *format_desc;
   const struct util_format_pack_description *pack =
      util_format_pack_description(format);
   uint8_t *dst_row;

   format_desc = util_format_description(format);

   assert(x % format_desc->block.width == 0);
   assert(y % format_desc->block.height == 0);

   dst_row = (uint8_t *)dst + y*dst_stride + x*(format_desc->block.bits/8);

   if (util_format_is_pure_uint(format))
      pack->pack_rgba_uint(dst_row, dst_stride, src, src_stride, w, h);
   else if (util_format_is_pure_sint(format))
      pack->pack_rgba_sint(dst_row, dst_stride, src, src_stride, w, h);
   else
      pack->pack_rgba_float(dst_row, dst_stride, src, src_stride, w, h);
}


void
util_format_read_4ub(enum pipe_format format, uint8_t *dst, unsigned dst_stride, const void *src, unsigned src_stride, unsigned x, unsigned y, unsigned w, unsigned h)
{
   const struct util_format_description *format_desc;
   const uint8_t *src_row;

   format_desc = util_format_description(format);

   assert(x % format_desc->block.width == 0);
   assert(y % format_desc->block.height == 0);

   src_row = (const uint8_t *)src + y*src_stride + x*(format_desc->block.bits/8);

   util_format_unpack_rgba_8unorm_rect(format, dst, dst_stride, src_row, src_stride, w, h);
}


void
util_format_write_4ub(enum pipe_format format, const uint8_t *src, unsigned src_stride, void *dst, unsigned dst_stride, unsigned x, unsigned y, unsigned w, unsigned h)
{
   const struct util_format_description *format_desc;
   const struct util_format_pack_description *pack =
      util_format_pack_description(format);
   uint8_t *dst_row;
   const uint8_t *src_row;

   format_desc = util_format_description(format);

   assert(x % format_desc->block.width == 0);
   assert(y % format_desc->block.height == 0);

   dst_row = (uint8_t *)dst + y*dst_stride + x*(format_desc->block.bits/8);
   src_row = src;

   pack->pack_rgba_8unorm(dst_row, dst_stride, src_row, src_stride, w, h);
}

/**
 * Check if we can safely memcopy from the source format to the dest format.
 * This basically covers the cases of a "used" channel copied to a typeless
 * channel, plus some 1-channel cases.
 * Examples of compatible copy formats include:
 *    b8g8r8a8_unorm -> b8g8r8x8_unorm
 *    a8r8g8b8_unorm -> x8r8g8b8_unorm
 *    b5g5r5a1_unorm -> b5g5r5x1_unorm
 *    b4g4r4a4_unorm -> b4g4r4x4_unorm
 *    l8_unorm -> r8_unorm
 *    i8_unorm -> l8_unorm
 *    i8_unorm -> a8_unorm
 *    i8_unorm -> r8_unorm
 *    l16_unorm -> r16_unorm
 *    z24_unorm_s8_uint -> z24x8_unorm
 *    s8_uint_z24_unorm -> x8z24_unorm
 *    r8g8b8a8_unorm -> r8g8b8x8_unorm
 *    a8b8g8r8_srgb -> x8b8g8r8_srgb
 *    b8g8r8a8_srgb -> b8g8r8x8_srgb
 *    a8r8g8b8_srgb -> x8r8g8b8_srgb
 *    a8b8g8r8_unorm -> x8b8g8r8_unorm
 *    r10g10b10a2_uscaled -> r10g10b10x2_uscaled
 *    r10sg10sb10sa2u_norm -> r10g10b10x2_snorm
 */
bool
util_is_format_compatible(const struct util_format_description *src_desc,
                          const struct util_format_description *dst_desc)
{
   unsigned chan;

   if (src_desc->format == dst_desc->format) {
      return true;
   }

   if (src_desc->layout != UTIL_FORMAT_LAYOUT_PLAIN ||
       dst_desc->layout != UTIL_FORMAT_LAYOUT_PLAIN) {
      return false;
   }

   if (src_desc->block.bits != dst_desc->block.bits ||
       src_desc->nr_channels != dst_desc->nr_channels ||
       src_desc->colorspace != dst_desc->colorspace) {
      return false;
   }

   for (chan = 0; chan < 4; ++chan) {
      if (src_desc->channel[chan].size !=
          dst_desc->channel[chan].size) {
         return false;
      }
   }

   for (chan = 0; chan < 4; ++chan) {
      enum pipe_swizzle swizzle = dst_desc->swizzle[chan];

      if (swizzle < 4) {
         if (src_desc->swizzle[chan] != swizzle) {
            return false;
         }
         if ((src_desc->channel[swizzle].type !=
              dst_desc->channel[swizzle].type) ||
             (src_desc->channel[swizzle].normalized !=
              dst_desc->channel[swizzle].normalized)) {
            return false;
         }
      }
   }

   return true;
}


bool
util_format_fits_8unorm(const struct util_format_description *format_desc)
{
   unsigned chan;

   /*
    * After linearized sRGB values require more than 8bits.
    */

   if (format_desc->colorspace == UTIL_FORMAT_COLORSPACE_SRGB) {
      return false;
   }

   switch (format_desc->layout) {

   case UTIL_FORMAT_LAYOUT_S3TC:
      /*
       * These are straight forward.
       */
      return true;
   case UTIL_FORMAT_LAYOUT_RGTC:
      if (format_desc->format == PIPE_FORMAT_RGTC1_SNORM ||
          format_desc->format == PIPE_FORMAT_RGTC2_SNORM ||
          format_desc->format == PIPE_FORMAT_LATC1_SNORM ||
          format_desc->format == PIPE_FORMAT_LATC2_SNORM)
         return false;
      return true;
   case UTIL_FORMAT_LAYOUT_BPTC:
      if (format_desc->format == PIPE_FORMAT_BPTC_RGBA_UNORM)
         return true;
      return false;

   case UTIL_FORMAT_LAYOUT_ETC:
      if (format_desc->format == PIPE_FORMAT_ETC1_RGB8)
         return true;
      return false;

   case UTIL_FORMAT_LAYOUT_PLAIN:
      /*
       * For these we can find a generic rule.
       */

      for (chan = 0; chan < format_desc->nr_channels; ++chan) {
         switch (format_desc->channel[chan].type) {
         case UTIL_FORMAT_TYPE_VOID:
            break;
         case UTIL_FORMAT_TYPE_UNSIGNED:
            if (!format_desc->channel[chan].normalized ||
                format_desc->channel[chan].size > 8) {
               return false;
            }
            break;
         default:
            return false;
         }
      }
      return true;

   default:
      /*
       * Handle all others on a case by case basis.
       */

      switch (format_desc->format) {
      case PIPE_FORMAT_R1_UNORM:
      case PIPE_FORMAT_UYVY:
      case PIPE_FORMAT_YUYV:
      case PIPE_FORMAT_R8G8_B8G8_UNORM:
      case PIPE_FORMAT_G8R8_G8B8_UNORM:
         return true;

      default:
         return false;
      }
   }
}


bool
util_format_translate(enum pipe_format dst_format,
                      void *dst, unsigned dst_stride,
                      unsigned dst_x, unsigned dst_y,
                      enum pipe_format src_format,
                      const void *src, unsigned src_stride,
                      unsigned src_x, unsigned src_y,
                      unsigned width, unsigned height)
{
   const struct util_format_description *dst_format_desc;
   const struct util_format_description *src_format_desc;
   const struct util_format_pack_description *pack =
      util_format_pack_description(dst_format);
   const struct util_format_unpack_description *unpack =
      util_format_unpack_description(src_format);
   uint8_t *dst_row;
   const uint8_t *src_row;
   unsigned x_step, y_step;
   unsigned dst_step;
   unsigned src_step;

   dst_format_desc = util_format_description(dst_format);
   src_format_desc = util_format_description(src_format);

   if (util_is_format_compatible(src_format_desc, dst_format_desc)) {
      /*
       * Trivial case.
       */

      util_copy_rect(dst, dst_format, dst_stride,  dst_x, dst_y,
                     width, height, src, (int)src_stride,
                     src_x, src_y);
      return true;
   }

   assert(dst_x % dst_format_desc->block.width == 0);
   assert(dst_y % dst_format_desc->block.height == 0);
   assert(src_x % src_format_desc->block.width == 0);
   assert(src_y % src_format_desc->block.height == 0);

   dst_row = (uint8_t *)dst + dst_y*dst_stride + dst_x*(dst_format_desc->block.bits/8);
   src_row = (const uint8_t *)src + src_y*src_stride + src_x*(src_format_desc->block.bits/8);

   /*
    * This works because all pixel formats have pixel blocks with power of two
    * sizes.
    */

   y_step = MAX2(dst_format_desc->block.height, src_format_desc->block.height);
   x_step = MAX2(dst_format_desc->block.width, src_format_desc->block.width);
   assert(y_step % dst_format_desc->block.height == 0);
   assert(y_step % src_format_desc->block.height == 0);

   dst_step = y_step / dst_format_desc->block.height * dst_stride;
   src_step = y_step / src_format_desc->block.height * src_stride;

   /*
    * TODO: double formats will loose precision
    * TODO: Add a special case for formats that are mere swizzles of each other
    */

   if (src_format_desc->colorspace == UTIL_FORMAT_COLORSPACE_ZS ||
       dst_format_desc->colorspace == UTIL_FORMAT_COLORSPACE_ZS) {
      float *tmp_z = NULL;
      uint8_t *tmp_s = NULL;

      assert(x_step == 1);
      assert(y_step == 1);

      if (unpack->unpack_z_float && pack->pack_z_float) {
         tmp_z = malloc(width * sizeof *tmp_z);
      }

      if (unpack->unpack_s_8uint && pack->pack_s_8uint) {
         tmp_s = malloc(width * sizeof *tmp_s);
      }

      while (height--) {
         if (tmp_z) {
            util_format_unpack_z_float(src_format, tmp_z, src_row, width);
            util_format_pack_z_float(dst_format, dst_row, tmp_z, width);
         }

         if (tmp_s) {
            util_format_unpack_s_8uint(src_format, tmp_s, src_row, width);
            util_format_pack_s_8uint(dst_format, dst_row, tmp_s, width);
         }

         dst_row += dst_step;
         src_row += src_step;
      }

      free(tmp_s);

      free(tmp_z);

      return true;
   }

   if (util_format_fits_8unorm(src_format_desc) ||
       util_format_fits_8unorm(dst_format_desc)) {
      unsigned tmp_stride;
      uint8_t *tmp_row;

      if ((!unpack->unpack_rgba_8unorm && !unpack->unpack_rgba_8unorm_rect) ||
          !pack->pack_rgba_8unorm) {
         return false;
      }

      tmp_stride = MAX2(width, x_step) * 4 * sizeof *tmp_row;
      tmp_row = malloc(y_step * tmp_stride);
      if (!tmp_row)
         return false;

      while (height >= y_step) {
         util_format_unpack_rgba_8unorm_rect(src_format, tmp_row, tmp_stride, src_row, src_stride, width, y_step);
         pack->pack_rgba_8unorm(dst_row, dst_stride, tmp_row, tmp_stride, width, y_step);

         dst_row += dst_step;
         src_row += src_step;
         height -= y_step;
      }

      if (height) {
         util_format_unpack_rgba_8unorm_rect(src_format, tmp_row, tmp_stride, src_row, src_stride, width, height);
         pack->pack_rgba_8unorm(dst_row, dst_stride, tmp_row, tmp_stride, width, height);
      }

      free(tmp_row);
   }
   else if (util_format_is_pure_sint(src_format) ||
            util_format_is_pure_sint(dst_format)) {
      unsigned tmp_stride;
      int *tmp_row;

      if (util_format_is_pure_sint(src_format) !=
          util_format_is_pure_sint(dst_format)) {
         return false;
      }

      tmp_stride = MAX2(width, x_step) * 4 * sizeof *tmp_row;
      tmp_row = malloc(y_step * tmp_stride);
      if (!tmp_row)
         return false;

      while (height >= y_step) {
         util_format_unpack_rgba_rect(src_format, tmp_row, tmp_stride, src_row, src_stride, width, y_step);
         pack->pack_rgba_sint(dst_row, dst_stride, tmp_row, tmp_stride, width, y_step);

         dst_row += dst_step;
         src_row += src_step;
         height -= y_step;
      }

      if (height) {
         util_format_unpack_rgba_rect(src_format, tmp_row, tmp_stride, src_row, src_stride, width, height);
         pack->pack_rgba_sint(dst_row, dst_stride, tmp_row, tmp_stride, width, height);
      }

      free(tmp_row);
   }
   else if (util_format_is_pure_uint(src_format) ||
            util_format_is_pure_uint(dst_format)) {
      unsigned tmp_stride;
      unsigned int *tmp_row;

      if ((!unpack->unpack_rgba && !unpack->unpack_rgba_rect) ||
          !pack->pack_rgba_uint) {
         return false;
      }

      tmp_stride = MAX2(width, x_step) * 4 * sizeof *tmp_row;
      tmp_row = malloc(y_step * tmp_stride);
      if (!tmp_row)
         return false;

      while (height >= y_step) {
         util_format_unpack_rgba_rect(src_format, tmp_row, tmp_stride, src_row, src_stride, width, y_step);
         pack->pack_rgba_uint(dst_row, dst_stride, tmp_row, tmp_stride, width, y_step);

         dst_row += dst_step;
         src_row += src_step;
         height -= y_step;
      }

      if (height) {
         util_format_unpack_rgba_rect(src_format, tmp_row, tmp_stride, src_row, src_stride, width, height);
         pack->pack_rgba_uint(dst_row, dst_stride, tmp_row, tmp_stride, width, height);
      }

      free(tmp_row);
   }
   else {
      unsigned tmp_stride;
      float *tmp_row;

      if ((!unpack->unpack_rgba && !unpack->unpack_rgba_rect) ||
          !pack->pack_rgba_float) {
         return false;
      }

      tmp_stride = MAX2(width, x_step) * 4 * sizeof *tmp_row;
      tmp_row = malloc(y_step * tmp_stride);
      if (!tmp_row)
         return false;

      while (height >= y_step) {
         util_format_unpack_rgba_rect(src_format, tmp_row, tmp_stride, src_row, src_stride, width, y_step);
         pack->pack_rgba_float(dst_row, dst_stride, tmp_row, tmp_stride, width, y_step);

         dst_row += dst_step;
         src_row += src_step;
         height -= y_step;
      }

      if (height) {
         util_format_unpack_rgba_rect(src_format, tmp_row, tmp_stride, src_row, src_stride, width, height);
         pack->pack_rgba_float(dst_row, dst_stride, tmp_row, tmp_stride, width, height);
      }

      free(tmp_row);
   }
   return true;
}

bool
util_format_translate_3d(enum pipe_format dst_format,
                         void *dst, unsigned dst_stride,
                         unsigned dst_slice_stride,
                         unsigned dst_x, unsigned dst_y,
                         unsigned dst_z,
                         enum pipe_format src_format,
                         const void *src, unsigned src_stride,
                         unsigned src_slice_stride,
                         unsigned src_x, unsigned src_y,
                         unsigned src_z, unsigned width,
                         unsigned height, unsigned depth)
{
   uint8_t *dst_layer;
   const uint8_t *src_layer;
   unsigned z;
   dst_layer = dst;
   src_layer = src;
   dst_layer += dst_z * dst_slice_stride;
   src_layer += src_z * src_slice_stride;
   for (z = 0; z < depth; ++z) {
      if (!util_format_translate(dst_format, dst_layer, dst_stride,
                                 dst_x, dst_y,
                                 src_format, src_layer, src_stride,
                                 src_x, src_y,
                                 width, height))
          return false;

      dst_layer += dst_slice_stride;
      src_layer += src_slice_stride;
   }
   return true;
}

void util_format_compose_swizzles(const unsigned char swz1[4],
                                  const unsigned char swz2[4],
                                  unsigned char dst[4])
{
   unsigned i;

   for (i = 0; i < 4; i++) {
      dst[i] = swz2[i] <= PIPE_SWIZZLE_W ?
               swz1[swz2[i]] : swz2[i];
   }
}

void util_format_apply_color_swizzle(union pipe_color_union *dst,
                                     const union pipe_color_union *src,
                                     const unsigned char swz[4],
                                     const bool is_integer)
{
   unsigned c;

   if (is_integer) {
      for (c = 0; c < 4; ++c) {
         switch (swz[c]) {
         case PIPE_SWIZZLE_X:   dst->ui[c] = src->ui[0]; break;
         case PIPE_SWIZZLE_Y: dst->ui[c] = src->ui[1]; break;
         case PIPE_SWIZZLE_Z:  dst->ui[c] = src->ui[2]; break;
         case PIPE_SWIZZLE_W: dst->ui[c] = src->ui[3]; break;
         default:
            dst->ui[c] = (swz[c] == PIPE_SWIZZLE_1) ? 1 : 0;
            break;
         }
      }
   } else {
      for (c = 0; c < 4; ++c) {
         switch (swz[c]) {
         case PIPE_SWIZZLE_X:   dst->f[c] = src->f[0]; break;
         case PIPE_SWIZZLE_Y: dst->f[c] = src->f[1]; break;
         case PIPE_SWIZZLE_Z:  dst->f[c] = src->f[2]; break;
         case PIPE_SWIZZLE_W: dst->f[c] = src->f[3]; break;
         default:
            dst->f[c] = (swz[c] == PIPE_SWIZZLE_1) ? 1.0f : 0.0f;
            break;
         }
      }
   }
}

void pipe_swizzle_4f(float *dst, const float *src,
                            const unsigned char swz[4])
{
   unsigned i;

   for (i = 0; i < 4; i++) {
      if (swz[i] <= PIPE_SWIZZLE_W)
         dst[i] = src[swz[i]];
      else if (swz[i] == PIPE_SWIZZLE_0)
         dst[i] = 0;
      else if (swz[i] == PIPE_SWIZZLE_1)
         dst[i] = 1;
   }
}

void util_format_unswizzle_4f(float *dst, const float *src,
                              const unsigned char swz[4])
{
   unsigned i;

   for (i = 0; i < 4; i++) {
      switch (swz[i]) {
      case PIPE_SWIZZLE_X:
         dst[0] = src[i];
         break;
      case PIPE_SWIZZLE_Y:
         dst[1] = src[i];
         break;
      case PIPE_SWIZZLE_Z:
         dst[2] = src[i];
         break;
      case PIPE_SWIZZLE_W:
         dst[3] = src[i];
         break;
      }
   }
}

enum pipe_format
util_format_snorm_to_sint(enum pipe_format format)
{
   switch (format) {
   case PIPE_FORMAT_R32_SNORM:
      return PIPE_FORMAT_R32_SINT;
   case PIPE_FORMAT_R32G32_SNORM:
      return PIPE_FORMAT_R32G32_SINT;
   case PIPE_FORMAT_R32G32B32_SNORM:
      return PIPE_FORMAT_R32G32B32_SINT;
   case PIPE_FORMAT_R32G32B32A32_SNORM:
      return PIPE_FORMAT_R32G32B32A32_SINT;

   case PIPE_FORMAT_R16_SNORM:
      return PIPE_FORMAT_R16_SINT;
   case PIPE_FORMAT_R16G16_SNORM:
      return PIPE_FORMAT_R16G16_SINT;
   case PIPE_FORMAT_R16G16B16_SNORM:
      return PIPE_FORMAT_R16G16B16_SINT;
   case PIPE_FORMAT_R16G16B16A16_SNORM:
      return PIPE_FORMAT_R16G16B16A16_SINT;

   case PIPE_FORMAT_R8_SNORM:
      return PIPE_FORMAT_R8_SINT;
   case PIPE_FORMAT_R8G8_SNORM:
      return PIPE_FORMAT_R8G8_SINT;
   case PIPE_FORMAT_R8G8B8_SNORM:
      return PIPE_FORMAT_R8G8B8_SINT;
   case PIPE_FORMAT_B8G8R8_SNORM:
      return PIPE_FORMAT_B8G8R8_SINT;
   case PIPE_FORMAT_R8G8B8A8_SNORM:
      return PIPE_FORMAT_R8G8B8A8_SINT;
   case PIPE_FORMAT_B8G8R8A8_SNORM:
      return PIPE_FORMAT_B8G8R8A8_SINT;

   case PIPE_FORMAT_R10G10B10A2_SNORM:
      return PIPE_FORMAT_R10G10B10A2_SINT;
   case PIPE_FORMAT_B10G10R10A2_SNORM:
      return PIPE_FORMAT_B10G10R10A2_SINT;

   case PIPE_FORMAT_R10G10B10X2_SNORM:
      return PIPE_FORMAT_R10G10B10X2_SINT;

   case PIPE_FORMAT_A8_SNORM:
      return PIPE_FORMAT_A8_SINT;
   case PIPE_FORMAT_L8_SNORM:
      return PIPE_FORMAT_L8_SINT;
   case PIPE_FORMAT_L8A8_SNORM:
      return PIPE_FORMAT_L8A8_SINT;
   case PIPE_FORMAT_I8_SNORM:
      return PIPE_FORMAT_I8_SINT;

   case PIPE_FORMAT_A16_SNORM:
      return PIPE_FORMAT_A16_SINT;
   case PIPE_FORMAT_L16_SNORM:
      return PIPE_FORMAT_L16_SINT;
   case PIPE_FORMAT_L16A16_SNORM:
      return PIPE_FORMAT_L16A16_SINT;
   case PIPE_FORMAT_I16_SNORM:
      return PIPE_FORMAT_I16_SINT;

   case PIPE_FORMAT_R8G8B8X8_SNORM:
      return PIPE_FORMAT_R8G8B8X8_SINT;
   case PIPE_FORMAT_R16G16B16X16_SNORM:
      return PIPE_FORMAT_R16G16B16X16_SINT;

   case PIPE_FORMAT_R8A8_SNORM:
      return PIPE_FORMAT_R8A8_SINT;
   case PIPE_FORMAT_R16A16_SNORM:
      return PIPE_FORMAT_R16A16_SINT;

   case PIPE_FORMAT_G8R8_SNORM:
      return PIPE_FORMAT_G8R8_SINT;
   case PIPE_FORMAT_G16R16_SNORM:
      return PIPE_FORMAT_G16R16_SINT;

   case PIPE_FORMAT_A8B8G8R8_SNORM:
      return PIPE_FORMAT_A8B8G8R8_SINT;
   case PIPE_FORMAT_X8B8G8R8_SNORM:
      return PIPE_FORMAT_X8B8G8R8_SINT;

   case PIPE_FORMAT_B8G8R8X8_SNORM:
      return PIPE_FORMAT_B8G8R8X8_SINT;
   case PIPE_FORMAT_A8R8G8B8_SNORM:
      return PIPE_FORMAT_A8R8G8B8_SINT;
   case PIPE_FORMAT_X8R8G8B8_SNORM:
      return PIPE_FORMAT_X8R8G8B8_SINT;
   case PIPE_FORMAT_B10G10R10X2_SNORM:
      return PIPE_FORMAT_B10G10R10X2_SINT;

   default:
      return format;
   }
}

/**
 * If the format is RGB, return BGR. If the format is BGR, return RGB.
 * This may fail by returning PIPE_FORMAT_NONE.
 */
enum pipe_format
util_format_rgb_to_bgr(enum pipe_format format)
{
#define REMAP_RGB_ONE(r, rs, g, gs, b, bs, type) \
   case PIPE_FORMAT_##r##rs##g##gs##b##bs##_##type: \
      return PIPE_FORMAT_##b##bs##g##gs##r##rs##_##type;

#define REMAP_RGB(rs, gs, bs, type) \
   REMAP_RGB_ONE(R, rs, G, gs, B, bs, type) \
   REMAP_RGB_ONE(B, bs, G, gs, R, rs, type) \

#define REMAP_RGBA_ONE(r, rs, g, gs, b, bs, a, as, type) \
   case PIPE_FORMAT_##r##rs##g##gs##b##bs##a##as##_##type: \
      return PIPE_FORMAT_##b##bs##g##gs##r##rs##a##as##_##type;

#define REMAP_ARGB_ONE(a, as, r, rs, g, gs, b, bs, type) \
   case PIPE_FORMAT_##a##as##r##rs##g##gs##b##bs##_##type: \
      return PIPE_FORMAT_##a##as##b##bs##g##gs##r##rs##_##type;

#define REMAP_RGB_AX(A, rs, gs, bs, as, type) \
   REMAP_RGBA_ONE(R, rs, G, gs, B, bs, A, as, type) \
   REMAP_RGBA_ONE(B, bs, G, gs, R, rs, A, as, type) \

#define REMAP_AX_RGB(A, rs, gs, bs, as, type) \
   REMAP_ARGB_ONE(A, as, R, rs, G, gs, B, bs, type) \
   REMAP_ARGB_ONE(A, as, B, bs, G, gs, R, rs, type) \

#define REMAP_RGBA(rs, gs, bs, as, type) REMAP_RGB_AX(A, rs, gs, bs, as, type)
#define REMAP_RGBX(rs, gs, bs, as, type) REMAP_RGB_AX(X, rs, gs, bs, as, type)
#define REMAP_ARGB(rs, gs, bs, as, type) REMAP_AX_RGB(A, rs, gs, bs, as, type)
#define REMAP_XRGB(rs, gs, bs, as, type) REMAP_AX_RGB(X, rs, gs, bs, as, type)

#define REMAP_RGBA_ALL(rs, gs, bs, as, type) \
   REMAP_RGBA(rs, gs, bs, as, type) \
   REMAP_RGBX(rs, gs, bs, as, type) \
   REMAP_ARGB(rs, gs, bs, as, type) \
   REMAP_XRGB(rs, gs, bs, as, type)

   switch (format) {
   REMAP_RGB(3, 3, 2, UNORM);
   REMAP_RGB(3, 3, 2, UINT);
   REMAP_RGB(5, 6, 5, SRGB);
   REMAP_RGB(5, 6, 5, UNORM);
   REMAP_RGB(5, 6, 5, UINT);
   REMAP_RGB(8, 8, 8, SRGB);
   REMAP_RGB(8, 8, 8, UNORM);
   REMAP_RGB(8, 8, 8, SNORM);
   REMAP_RGB(8, 8, 8, UINT);
   REMAP_RGB(8, 8, 8, SINT);
   REMAP_RGB(8, 8, 8, USCALED);
   REMAP_RGB(8, 8, 8, SSCALED);

   /* Complete format sets. */
   REMAP_RGBA_ALL(5, 5, 5, 1, UNORM);
   REMAP_RGBA_ALL(8, 8, 8, 8, SRGB);
   REMAP_RGBA_ALL(8, 8, 8, 8, UNORM);
   REMAP_RGBA_ALL(8, 8, 8, 8, SNORM);
   REMAP_RGBA_ALL(8, 8, 8, 8, SINT);

   /* Format sets missing XRGB/XBGR. */
   REMAP_RGBA(4, 4, 4, 4, UNORM);
   REMAP_RGBX(4, 4, 4, 4, UNORM);
   REMAP_ARGB(4, 4, 4, 4, UNORM);

   REMAP_RGBA(8, 8, 8, 8, UINT);
   REMAP_RGBX(8, 8, 8, 8, UINT);
   REMAP_ARGB(8, 8, 8, 8, UINT);

   REMAP_RGBA(10, 10, 10, 2, UNORM);
   REMAP_RGBX(10, 10, 10, 2, UNORM);
   REMAP_ARGB(10, 10, 10, 2, UNORM);

   /* Format sets missing a half of combinations. */
   REMAP_RGBA(4, 4, 4, 4, UINT);
   REMAP_ARGB(4, 4, 4, 4, UINT);

   REMAP_RGBA(5, 5, 5, 1, UINT);
   REMAP_ARGB(5, 5, 5, 1, UINT);

   REMAP_RGBA(10, 10, 10, 2, SNORM);
   REMAP_RGBX(10, 10, 10, 2, SNORM);

   REMAP_RGBA(10, 10, 10, 2, UINT);
   REMAP_ARGB(10, 10, 10, 2, UINT);

   REMAP_RGBA(10, 10, 10, 2, SINT);
   REMAP_RGBX(10, 10, 10, 2, SINT);

   /* Format sets having only RGBA/BGRA. */
   REMAP_RGBA(8, 8, 8, 8, USCALED);
   REMAP_RGBA(8, 8, 8, 8, SSCALED);
   REMAP_RGBA(10, 10, 10, 2, USCALED);
   REMAP_RGBA(10, 10, 10, 2, SSCALED);

   default:
      return PIPE_FORMAT_NONE;
   }
}

static const struct util_format_unpack_description *util_format_unpack_table[PIPE_FORMAT_COUNT];

static void
util_format_unpack_table_init(void)
{
   for (enum pipe_format format = PIPE_FORMAT_NONE; format < PIPE_FORMAT_COUNT; format++) {
#if (DETECT_ARCH_AARCH64 || DETECT_ARCH_ARM) && !defined(NO_FORMAT_ASM) && !defined(__SOFTFP__)
      const struct util_format_unpack_description *unpack = util_format_unpack_description_neon(format);
      if (unpack) {
         util_format_unpack_table[format] = unpack;
         continue;
      }
#endif

      util_format_unpack_table[format] = util_format_unpack_description_generic(format);
   }
}

const struct util_format_unpack_description *
util_format_unpack_description(enum pipe_format format)
{
   static once_flag flag = ONCE_FLAG_INIT;
   call_once(&flag, util_format_unpack_table_init);

   return util_format_unpack_table[format];
}

enum pipe_format
util_format_snorm_to_unorm(enum pipe_format format)
{
#define CASE(x) case PIPE_FORMAT_##x##_SNORM: return PIPE_FORMAT_##x##_UNORM

   switch (format) {
   CASE(R8G8B8A8);
   CASE(R8G8B8X8);
   CASE(B8G8R8A8);
   CASE(B8G8R8X8);
   CASE(A8R8G8B8);
   CASE(X8R8G8B8);
   CASE(A8B8G8R8);
   CASE(X8B8G8R8);

   CASE(R10G10B10A2);
   CASE(R10G10B10X2);
   CASE(B10G10R10A2);
   CASE(B10G10R10X2);

   CASE(R8);
   CASE(R8G8);
   CASE(G8R8);
   CASE(R8G8B8);
   CASE(B8G8R8);

   CASE(R16);
   CASE(R16G16);
   CASE(G16R16);
   CASE(R16G16B16);

   CASE(R16G16B16A16);
   CASE(R16G16B16X16);

   CASE(R32);
   CASE(R32G32);
   CASE(R32G32B32);
   CASE(R32G32B32A32);

   CASE(RGTC1);
   CASE(RGTC2);
   CASE(ETC2_R11);
   CASE(ETC2_RG11);

   CASE(A8);
   CASE(A16);
   CASE(L8);
   CASE(L16);
   CASE(I8);
   CASE(I16);

   CASE(L8A8);
   CASE(L16A16);
   CASE(R8A8);
   CASE(R16A16);

   CASE(LATC1);
   CASE(LATC2);

   default:
      return format;
   }

#undef CASE
}

enum pipe_format
util_format_rgbx_to_rgba(enum pipe_format format)
{
   switch (format) {
   case PIPE_FORMAT_B8G8R8X8_UNORM:
      return PIPE_FORMAT_B8G8R8A8_UNORM;
   case PIPE_FORMAT_X8B8G8R8_UNORM:
      return PIPE_FORMAT_A8B8G8R8_UNORM;
   case PIPE_FORMAT_X8R8G8B8_UNORM:
      return PIPE_FORMAT_A8R8G8B8_UNORM;
   case PIPE_FORMAT_X8B8G8R8_SRGB:
      return PIPE_FORMAT_A8B8G8R8_SRGB;
   case PIPE_FORMAT_B8G8R8X8_SRGB:
      return PIPE_FORMAT_B8G8R8A8_SRGB;
   case PIPE_FORMAT_X8R8G8B8_SRGB:
      return PIPE_FORMAT_A8R8G8B8_SRGB;
   case PIPE_FORMAT_B5G5R5X1_UNORM:
      return PIPE_FORMAT_B5G5R5A1_UNORM;
   case PIPE_FORMAT_R10G10B10X2_USCALED:
      return PIPE_FORMAT_R10G10B10A2_USCALED;
   case PIPE_FORMAT_R10G10B10X2_SNORM:
      return PIPE_FORMAT_R10G10B10A2_SNORM;
   case PIPE_FORMAT_R8G8B8X8_UNORM:
      return PIPE_FORMAT_R8G8B8A8_UNORM;
   case PIPE_FORMAT_B4G4R4X4_UNORM:
      return PIPE_FORMAT_B4G4R4A4_UNORM;
   case PIPE_FORMAT_R8G8B8X8_SNORM:
      return PIPE_FORMAT_R8G8B8A8_SNORM;
   case PIPE_FORMAT_R8G8B8X8_SRGB:
      return PIPE_FORMAT_R8G8B8A8_SRGB;
   case PIPE_FORMAT_R8G8B8X8_UINT:
      return PIPE_FORMAT_R8G8B8A8_UINT;
   case PIPE_FORMAT_R8G8B8X8_SINT:
      return PIPE_FORMAT_R8G8B8A8_SINT;
   case PIPE_FORMAT_B10G10R10X2_UNORM:
      return PIPE_FORMAT_B10G10R10A2_UNORM;
   case PIPE_FORMAT_R16G16B16X16_UNORM:
      return PIPE_FORMAT_R16G16B16A16_UNORM;
   case PIPE_FORMAT_R16G16B16X16_SNORM:
      return PIPE_FORMAT_R16G16B16A16_SNORM;
   case PIPE_FORMAT_R16G16B16X16_FLOAT:
      return PIPE_FORMAT_R16G16B16A16_FLOAT;
   case PIPE_FORMAT_R16G16B16X16_UINT:
      return PIPE_FORMAT_R16G16B16A16_UINT;
   case PIPE_FORMAT_R16G16B16X16_SINT:
      return PIPE_FORMAT_R16G16B16A16_SINT;
   case PIPE_FORMAT_R32G32B32X32_FLOAT:
      return PIPE_FORMAT_R32G32B32A32_FLOAT;
   case PIPE_FORMAT_R32G32B32X32_UINT:
      return PIPE_FORMAT_R32G32B32A32_UINT;
   case PIPE_FORMAT_R32G32B32X32_SINT:
      return PIPE_FORMAT_R32G32B32A32_SINT;
   case PIPE_FORMAT_X8B8G8R8_SNORM:
      return PIPE_FORMAT_A8B8G8R8_SNORM;
   case PIPE_FORMAT_R10G10B10X2_UNORM:
      return PIPE_FORMAT_R10G10B10A2_UNORM;
   case PIPE_FORMAT_X1B5G5R5_UNORM:
      return PIPE_FORMAT_A1B5G5R5_UNORM;
   case PIPE_FORMAT_X8B8G8R8_SINT:
      return PIPE_FORMAT_A8B8G8R8_SINT;
   case PIPE_FORMAT_B8G8R8X8_SNORM:
      return PIPE_FORMAT_B8G8R8A8_SNORM;
   case PIPE_FORMAT_B8G8R8X8_UINT:
      return PIPE_FORMAT_B8G8R8A8_UINT;
   case PIPE_FORMAT_B8G8R8X8_SINT:
      return PIPE_FORMAT_B8G8R8A8_SINT;
   case PIPE_FORMAT_X8R8G8B8_SNORM:
      return PIPE_FORMAT_A8R8G8B8_SNORM;
   case PIPE_FORMAT_X8R8G8B8_SINT:
      return PIPE_FORMAT_A8R8G8B8_SINT;
   case PIPE_FORMAT_R5G5B5X1_UNORM:
      return PIPE_FORMAT_R5G5B5A1_UNORM;
   case PIPE_FORMAT_X1R5G5B5_UNORM:
      return PIPE_FORMAT_A1R5G5B5_UNORM;
   case PIPE_FORMAT_R4G4B4X4_UNORM:
      return PIPE_FORMAT_R4G4B4A4_UNORM;
   case PIPE_FORMAT_B10G10R10X2_SNORM:
      return PIPE_FORMAT_B10G10R10A2_SNORM;
   case PIPE_FORMAT_R10G10B10X2_SINT:
      return PIPE_FORMAT_R10G10B10A2_SINT;
   case PIPE_FORMAT_B10G10R10X2_SINT:
      return PIPE_FORMAT_B10G10R10A2_SINT;
   default: {
      ASSERTED const struct util_format_description *desc = util_format_description(format);

      /* Assert that the format doesn't contain X instead of A. */
      assert(desc->colorspace != UTIL_FORMAT_COLORSPACE_ZS ||
             (desc->channel[0].type != UTIL_FORMAT_TYPE_VOID &&
              desc->channel[desc->nr_channels - 1].type != UTIL_FORMAT_TYPE_VOID));
      return format;
   }
   }
}
