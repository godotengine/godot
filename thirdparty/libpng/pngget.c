/* pngget.c - retrieval of values from info struct
 *
 * Copyright (c) 2018-2025 Cosmin Truta
 * Copyright (c) 1998-2002,2004,2006-2018 Glenn Randers-Pehrson
 * Copyright (c) 1996-1997 Andreas Dilger
 * Copyright (c) 1995-1996 Guy Eric Schalnat, Group 42, Inc.
 *
 * This code is released under the libpng license.
 * For conditions of distribution and use, see the disclaimer
 * and license in png.h
 *
 */

#include "pngpriv.h"

#if defined(PNG_READ_SUPPORTED) || defined(PNG_WRITE_SUPPORTED)

png_uint_32 PNGAPI
png_get_valid(png_const_structrp png_ptr, png_const_inforp info_ptr,
    png_uint_32 flag)
{
   if (png_ptr != NULL && info_ptr != NULL)
   {
#ifdef PNG_READ_tRNS_SUPPORTED
      /* png_handle_PLTE() may have canceled a valid tRNS chunk but left the
       * 'valid' flag for the detection of duplicate chunks. Do not report a
       * valid tRNS chunk in this case.
       */
      if (flag == PNG_INFO_tRNS && png_ptr->num_trans == 0)
         return 0;
#endif

      return info_ptr->valid & flag;
   }

   return 0;
}

size_t PNGAPI
png_get_rowbytes(png_const_structrp png_ptr, png_const_inforp info_ptr)
{
   if (png_ptr != NULL && info_ptr != NULL)
      return info_ptr->rowbytes;

   return 0;
}

#ifdef PNG_INFO_IMAGE_SUPPORTED
png_bytepp PNGAPI
png_get_rows(png_const_structrp png_ptr, png_const_inforp info_ptr)
{
   if (png_ptr != NULL && info_ptr != NULL)
      return info_ptr->row_pointers;

   return 0;
}
#endif

#ifdef PNG_EASY_ACCESS_SUPPORTED
/* Easy access to info, added in libpng-0.99 */
png_uint_32 PNGAPI
png_get_image_width(png_const_structrp png_ptr, png_const_inforp info_ptr)
{
   if (png_ptr != NULL && info_ptr != NULL)
      return info_ptr->width;

   return 0;
}

png_uint_32 PNGAPI
png_get_image_height(png_const_structrp png_ptr, png_const_inforp info_ptr)
{
   if (png_ptr != NULL && info_ptr != NULL)
      return info_ptr->height;

   return 0;
}

png_byte PNGAPI
png_get_bit_depth(png_const_structrp png_ptr, png_const_inforp info_ptr)
{
   if (png_ptr != NULL && info_ptr != NULL)
      return info_ptr->bit_depth;

   return 0;
}

png_byte PNGAPI
png_get_color_type(png_const_structrp png_ptr, png_const_inforp info_ptr)
{
   if (png_ptr != NULL && info_ptr != NULL)
      return info_ptr->color_type;

   return 0;
}

png_byte PNGAPI
png_get_filter_type(png_const_structrp png_ptr, png_const_inforp info_ptr)
{
   if (png_ptr != NULL && info_ptr != NULL)
      return info_ptr->filter_type;

   return 0;
}

png_byte PNGAPI
png_get_interlace_type(png_const_structrp png_ptr, png_const_inforp info_ptr)
{
   if (png_ptr != NULL && info_ptr != NULL)
      return info_ptr->interlace_type;

   return 0;
}

png_byte PNGAPI
png_get_compression_type(png_const_structrp png_ptr, png_const_inforp info_ptr)
{
   if (png_ptr != NULL && info_ptr != NULL)
      return info_ptr->compression_type;

   return 0;
}

png_uint_32 PNGAPI
png_get_x_pixels_per_meter(png_const_structrp png_ptr, png_const_inforp
   info_ptr)
{
#ifdef PNG_pHYs_SUPPORTED
   png_debug(1, "in png_get_x_pixels_per_meter");

   if (png_ptr != NULL && info_ptr != NULL &&
       (info_ptr->valid & PNG_INFO_pHYs) != 0)
   {
      if (info_ptr->phys_unit_type == PNG_RESOLUTION_METER)
         return info_ptr->x_pixels_per_unit;
   }
#else
   PNG_UNUSED(png_ptr)
   PNG_UNUSED(info_ptr)
#endif

   return 0;
}

png_uint_32 PNGAPI
png_get_y_pixels_per_meter(png_const_structrp png_ptr, png_const_inforp
    info_ptr)
{
#ifdef PNG_pHYs_SUPPORTED
   png_debug(1, "in png_get_y_pixels_per_meter");

   if (png_ptr != NULL && info_ptr != NULL &&
       (info_ptr->valid & PNG_INFO_pHYs) != 0)
   {
      if (info_ptr->phys_unit_type == PNG_RESOLUTION_METER)
         return info_ptr->y_pixels_per_unit;
   }
#else
   PNG_UNUSED(png_ptr)
   PNG_UNUSED(info_ptr)
#endif

   return 0;
}

png_uint_32 PNGAPI
png_get_pixels_per_meter(png_const_structrp png_ptr, png_const_inforp info_ptr)
{
#ifdef PNG_pHYs_SUPPORTED
   png_debug(1, "in png_get_pixels_per_meter");

   if (png_ptr != NULL && info_ptr != NULL &&
       (info_ptr->valid & PNG_INFO_pHYs) != 0)
   {
      if (info_ptr->phys_unit_type == PNG_RESOLUTION_METER &&
          info_ptr->x_pixels_per_unit == info_ptr->y_pixels_per_unit)
         return info_ptr->x_pixels_per_unit;
   }
#else
   PNG_UNUSED(png_ptr)
   PNG_UNUSED(info_ptr)
#endif

   return 0;
}

#ifdef PNG_FLOATING_POINT_SUPPORTED
float PNGAPI
png_get_pixel_aspect_ratio(png_const_structrp png_ptr, png_const_inforp
   info_ptr)
{
#ifdef PNG_READ_pHYs_SUPPORTED
   png_debug(1, "in png_get_pixel_aspect_ratio");

   if (png_ptr != NULL && info_ptr != NULL &&
       (info_ptr->valid & PNG_INFO_pHYs) != 0)
   {
      if (info_ptr->x_pixels_per_unit != 0)
         return (float)info_ptr->y_pixels_per_unit
              / (float)info_ptr->x_pixels_per_unit;
   }
#else
   PNG_UNUSED(png_ptr)
   PNG_UNUSED(info_ptr)
#endif

   return (float)0.0;
}
#endif

#ifdef PNG_FIXED_POINT_SUPPORTED
png_fixed_point PNGAPI
png_get_pixel_aspect_ratio_fixed(png_const_structrp png_ptr,
    png_const_inforp info_ptr)
{
#ifdef PNG_READ_pHYs_SUPPORTED
   png_debug(1, "in png_get_pixel_aspect_ratio_fixed");

   if (png_ptr != NULL && info_ptr != NULL &&
       (info_ptr->valid & PNG_INFO_pHYs) != 0 &&
       info_ptr->x_pixels_per_unit > 0 && info_ptr->y_pixels_per_unit > 0 &&
       info_ptr->x_pixels_per_unit <= PNG_UINT_31_MAX &&
       info_ptr->y_pixels_per_unit <= PNG_UINT_31_MAX)
   {
      png_fixed_point res;

      /* The following casts work because a PNG 4 byte integer only has a valid
       * range of 0..2^31-1; otherwise the cast might overflow.
       */
      if (png_muldiv(&res, (png_int_32)info_ptr->y_pixels_per_unit, PNG_FP_1,
          (png_int_32)info_ptr->x_pixels_per_unit) != 0)
         return res;
   }
#else
   PNG_UNUSED(png_ptr)
   PNG_UNUSED(info_ptr)
#endif

   return 0;
}
#endif

png_int_32 PNGAPI
png_get_x_offset_microns(png_const_structrp png_ptr, png_const_inforp info_ptr)
{
#ifdef PNG_oFFs_SUPPORTED
   png_debug(1, "in png_get_x_offset_microns");

   if (png_ptr != NULL && info_ptr != NULL &&
       (info_ptr->valid & PNG_INFO_oFFs) != 0)
   {
      if (info_ptr->offset_unit_type == PNG_OFFSET_MICROMETER)
         return info_ptr->x_offset;
   }
#else
   PNG_UNUSED(png_ptr)
   PNG_UNUSED(info_ptr)
#endif

   return 0;
}

png_int_32 PNGAPI
png_get_y_offset_microns(png_const_structrp png_ptr, png_const_inforp info_ptr)
{
#ifdef PNG_oFFs_SUPPORTED
   png_debug(1, "in png_get_y_offset_microns");

   if (png_ptr != NULL && info_ptr != NULL &&
       (info_ptr->valid & PNG_INFO_oFFs) != 0)
   {
      if (info_ptr->offset_unit_type == PNG_OFFSET_MICROMETER)
         return info_ptr->y_offset;
   }
#else
   PNG_UNUSED(png_ptr)
   PNG_UNUSED(info_ptr)
#endif

   return 0;
}

png_int_32 PNGAPI
png_get_x_offset_pixels(png_const_structrp png_ptr, png_const_inforp info_ptr)
{
#ifdef PNG_oFFs_SUPPORTED
   png_debug(1, "in png_get_x_offset_pixels");

   if (png_ptr != NULL && info_ptr != NULL &&
       (info_ptr->valid & PNG_INFO_oFFs) != 0)
   {
      if (info_ptr->offset_unit_type == PNG_OFFSET_PIXEL)
         return info_ptr->x_offset;
   }
#else
   PNG_UNUSED(png_ptr)
   PNG_UNUSED(info_ptr)
#endif

   return 0;
}

png_int_32 PNGAPI
png_get_y_offset_pixels(png_const_structrp png_ptr, png_const_inforp info_ptr)
{
#ifdef PNG_oFFs_SUPPORTED
   png_debug(1, "in png_get_y_offset_pixels");

   if (png_ptr != NULL && info_ptr != NULL &&
       (info_ptr->valid & PNG_INFO_oFFs) != 0)
   {
      if (info_ptr->offset_unit_type == PNG_OFFSET_PIXEL)
         return info_ptr->y_offset;
   }
#else
   PNG_UNUSED(png_ptr)
   PNG_UNUSED(info_ptr)
#endif

   return 0;
}

#ifdef PNG_INCH_CONVERSIONS_SUPPORTED
static png_uint_32
ppi_from_ppm(png_uint_32 ppm)
{
#if 0
   /* The conversion is *(2.54/100), in binary (32 digits):
    * .00000110100000001001110101001001
    */
   png_uint_32 t1001, t1101;
   ppm >>= 1;                  /* .1 */
   t1001 = ppm + (ppm >> 3);   /* .1001 */
   t1101 = t1001 + (ppm >> 1); /* .1101 */
   ppm >>= 20;                 /* .000000000000000000001 */
   t1101 += t1101 >> 15;       /* .1101000000000001101 */
   t1001 >>= 11;               /* .000000000001001 */
   t1001 += t1001 >> 12;       /* .000000000001001000000001001 */
   ppm += t1001;               /* .000000000001001000001001001 */
   ppm += t1101;               /* .110100000001001110101001001 */
   return (ppm + 16) >> 5;/* .00000110100000001001110101001001 */
#else
   /* The argument is a PNG unsigned integer, so it is not permitted
    * to be bigger than 2^31.
    */
   png_fixed_point result;
   if (ppm <= PNG_UINT_31_MAX && png_muldiv(&result, (png_int_32)ppm, 127,
       5000) != 0)
      return (png_uint_32)result;

   /* Overflow. */
   return 0;
#endif
}

png_uint_32 PNGAPI
png_get_pixels_per_inch(png_const_structrp png_ptr, png_const_inforp info_ptr)
{
   return ppi_from_ppm(png_get_pixels_per_meter(png_ptr, info_ptr));
}

png_uint_32 PNGAPI
png_get_x_pixels_per_inch(png_const_structrp png_ptr, png_const_inforp info_ptr)
{
   return ppi_from_ppm(png_get_x_pixels_per_meter(png_ptr, info_ptr));
}

png_uint_32 PNGAPI
png_get_y_pixels_per_inch(png_const_structrp png_ptr, png_const_inforp info_ptr)
{
   return ppi_from_ppm(png_get_y_pixels_per_meter(png_ptr, info_ptr));
}

#ifdef PNG_FIXED_POINT_SUPPORTED
static png_fixed_point
png_fixed_inches_from_microns(png_const_structrp png_ptr, png_int_32 microns)
{
   /* Convert from meters * 1,000,000 to inches * 100,000, meters to
    * inches is simply *(100/2.54), so we want *(10/2.54) == 500/127.
    * Notice that this can overflow - a warning is output and 0 is
    * returned.
    */
   png_fixed_point result;

   if (png_muldiv(&result, microns, 500, 127) != 0)
      return result;

   png_warning(png_ptr, "fixed point overflow ignored");
   return 0;
}

png_fixed_point PNGAPI
png_get_x_offset_inches_fixed(png_const_structrp png_ptr,
    png_const_inforp info_ptr)
{
   return png_fixed_inches_from_microns(png_ptr,
       png_get_x_offset_microns(png_ptr, info_ptr));
}
#endif /* FIXED_POINT */

#ifdef PNG_FIXED_POINT_SUPPORTED
png_fixed_point PNGAPI
png_get_y_offset_inches_fixed(png_const_structrp png_ptr,
    png_const_inforp info_ptr)
{
   return png_fixed_inches_from_microns(png_ptr,
       png_get_y_offset_microns(png_ptr, info_ptr));
}
#endif

#ifdef PNG_FLOATING_POINT_SUPPORTED
float PNGAPI
png_get_x_offset_inches(png_const_structrp png_ptr, png_const_inforp info_ptr)
{
   /* To avoid the overflow do the conversion directly in floating
    * point.
    */
   return (float)(png_get_x_offset_microns(png_ptr, info_ptr) * .00003937);
}
#endif

#ifdef PNG_FLOATING_POINT_SUPPORTED
float PNGAPI
png_get_y_offset_inches(png_const_structrp png_ptr, png_const_inforp info_ptr)
{
   /* To avoid the overflow do the conversion directly in floating
    * point.
    */
   return (float)(png_get_y_offset_microns(png_ptr, info_ptr) * .00003937);
}
#endif

#ifdef PNG_pHYs_SUPPORTED
png_uint_32 PNGAPI
png_get_pHYs_dpi(png_const_structrp png_ptr, png_const_inforp info_ptr,
    png_uint_32 *res_x, png_uint_32 *res_y, int *unit_type)
{
   png_uint_32 retval = 0;

   png_debug1(1, "in %s retrieval function", "pHYs");

   if (png_ptr != NULL && info_ptr != NULL &&
       (info_ptr->valid & PNG_INFO_pHYs) != 0)
   {
      if (res_x != NULL)
      {
         *res_x = info_ptr->x_pixels_per_unit;
         retval |= PNG_INFO_pHYs;
      }

      if (res_y != NULL)
      {
         *res_y = info_ptr->y_pixels_per_unit;
         retval |= PNG_INFO_pHYs;
      }

      if (unit_type != NULL)
      {
         *unit_type = (int)info_ptr->phys_unit_type;
         retval |= PNG_INFO_pHYs;

         if (*unit_type == 1)
         {
            if (res_x != NULL) *res_x = (png_uint_32)(*res_x * .0254 + .50);
            if (res_y != NULL) *res_y = (png_uint_32)(*res_y * .0254 + .50);
         }
      }
   }

   return retval;
}
#endif /* pHYs */
#endif /* INCH_CONVERSIONS */

/* png_get_channels really belongs in here, too, but it's been around longer */

#endif /* EASY_ACCESS */


png_byte PNGAPI
png_get_channels(png_const_structrp png_ptr, png_const_inforp info_ptr)
{
   if (png_ptr != NULL && info_ptr != NULL)
      return info_ptr->channels;

   return 0;
}

#ifdef PNG_READ_SUPPORTED
png_const_bytep PNGAPI
png_get_signature(png_const_structrp png_ptr, png_const_inforp info_ptr)
{
   if (png_ptr != NULL && info_ptr != NULL)
      return info_ptr->signature;

   return NULL;
}
#endif

#ifdef PNG_bKGD_SUPPORTED
png_uint_32 PNGAPI
png_get_bKGD(png_const_structrp png_ptr, png_inforp info_ptr,
    png_color_16p *background)
{
   png_debug1(1, "in %s retrieval function", "bKGD");

   if (png_ptr != NULL && info_ptr != NULL &&
       (info_ptr->valid & PNG_INFO_bKGD) != 0 &&
       background != NULL)
   {
      *background = &(info_ptr->background);
      return PNG_INFO_bKGD;
   }

   return 0;
}
#endif

#ifdef PNG_cHRM_SUPPORTED
/* The XYZ APIs were added in 1.5.5 to take advantage of the code added at the
 * same time to correct the rgb grayscale coefficient defaults obtained from the
 * cHRM chunk in 1.5.4
 */
#  ifdef PNG_FLOATING_POINT_SUPPORTED
png_uint_32 PNGAPI
png_get_cHRM(png_const_structrp png_ptr, png_const_inforp info_ptr,
    double *whitex, double *whitey, double *redx, double *redy,
    double *greenx, double *greeny, double *bluex, double *bluey)
{
   png_debug1(1, "in %s retrieval function", "cHRM");

   /* PNGv3: this just returns the values store from the cHRM, if any. */
   if (png_ptr != NULL && info_ptr != NULL &&
       (info_ptr->valid & PNG_INFO_cHRM) != 0)
   {
      if (whitex != NULL)
         *whitex = png_float(png_ptr, info_ptr->cHRM.whitex, "cHRM wx");
      if (whitey != NULL)
         *whitey = png_float(png_ptr, info_ptr->cHRM.whitey, "cHRM wy");
      if (redx   != NULL)
         *redx   = png_float(png_ptr, info_ptr->cHRM.redx,   "cHRM rx");
      if (redy   != NULL)
         *redy   = png_float(png_ptr, info_ptr->cHRM.redy,   "cHRM ry");
      if (greenx != NULL)
         *greenx = png_float(png_ptr, info_ptr->cHRM.greenx, "cHRM gx");
      if (greeny != NULL)
         *greeny = png_float(png_ptr, info_ptr->cHRM.greeny, "cHRM gy");
      if (bluex  != NULL)
         *bluex  = png_float(png_ptr, info_ptr->cHRM.bluex,  "cHRM bx");
      if (bluey  != NULL)
         *bluey  = png_float(png_ptr, info_ptr->cHRM.bluey,  "cHRM by");
      return PNG_INFO_cHRM;
   }

   return 0;
}

png_uint_32 PNGAPI
png_get_cHRM_XYZ(png_const_structrp png_ptr, png_const_inforp info_ptr,
    double *red_X, double *red_Y, double *red_Z, double *green_X,
    double *green_Y, double *green_Z, double *blue_X, double *blue_Y,
    double *blue_Z)
{
   png_XYZ XYZ;
   png_debug1(1, "in %s retrieval function", "cHRM_XYZ(float)");

   if (png_ptr != NULL && info_ptr != NULL &&
       (info_ptr->valid & PNG_INFO_cHRM) != 0 &&
       png_XYZ_from_xy(&XYZ, &info_ptr->cHRM) == 0)
   {
      if (red_X != NULL)
         *red_X = png_float(png_ptr, XYZ.red_X, "cHRM red X");
      if (red_Y != NULL)
         *red_Y = png_float(png_ptr, XYZ.red_Y, "cHRM red Y");
      if (red_Z != NULL)
         *red_Z = png_float(png_ptr, XYZ.red_Z, "cHRM red Z");
      if (green_X != NULL)
         *green_X = png_float(png_ptr, XYZ.green_X, "cHRM green X");
      if (green_Y != NULL)
         *green_Y = png_float(png_ptr, XYZ.green_Y, "cHRM green Y");
      if (green_Z != NULL)
         *green_Z = png_float(png_ptr, XYZ.green_Z, "cHRM green Z");
      if (blue_X != NULL)
         *blue_X = png_float(png_ptr, XYZ.blue_X, "cHRM blue X");
      if (blue_Y != NULL)
         *blue_Y = png_float(png_ptr, XYZ.blue_Y, "cHRM blue Y");
      if (blue_Z != NULL)
         *blue_Z = png_float(png_ptr, XYZ.blue_Z, "cHRM blue Z");
      return PNG_INFO_cHRM;
   }

   return 0;
}
#  endif

#  ifdef PNG_FIXED_POINT_SUPPORTED
png_uint_32 PNGAPI
png_get_cHRM_XYZ_fixed(png_const_structrp png_ptr, png_const_inforp info_ptr,
    png_fixed_point *int_red_X, png_fixed_point *int_red_Y,
    png_fixed_point *int_red_Z, png_fixed_point *int_green_X,
    png_fixed_point *int_green_Y, png_fixed_point *int_green_Z,
    png_fixed_point *int_blue_X, png_fixed_point *int_blue_Y,
    png_fixed_point *int_blue_Z)
{
   png_XYZ XYZ;
   png_debug1(1, "in %s retrieval function", "cHRM_XYZ");

   if (png_ptr != NULL && info_ptr != NULL &&
       (info_ptr->valid & PNG_INFO_cHRM) != 0U &&
       png_XYZ_from_xy(&XYZ, &info_ptr->cHRM) == 0)
   {
      if (int_red_X != NULL) *int_red_X = XYZ.red_X;
      if (int_red_Y != NULL) *int_red_Y = XYZ.red_Y;
      if (int_red_Z != NULL) *int_red_Z = XYZ.red_Z;
      if (int_green_X != NULL) *int_green_X = XYZ.green_X;
      if (int_green_Y != NULL) *int_green_Y = XYZ.green_Y;
      if (int_green_Z != NULL) *int_green_Z = XYZ.green_Z;
      if (int_blue_X != NULL) *int_blue_X = XYZ.blue_X;
      if (int_blue_Y != NULL) *int_blue_Y = XYZ.blue_Y;
      if (int_blue_Z != NULL) *int_blue_Z = XYZ.blue_Z;
      return PNG_INFO_cHRM;
   }

   return 0;
}

png_uint_32 PNGAPI
png_get_cHRM_fixed(png_const_structrp png_ptr, png_const_inforp info_ptr,
    png_fixed_point *whitex, png_fixed_point *whitey, png_fixed_point *redx,
    png_fixed_point *redy, png_fixed_point *greenx, png_fixed_point *greeny,
    png_fixed_point *bluex, png_fixed_point *bluey)
{
   png_debug1(1, "in %s retrieval function", "cHRM");

   /* PNGv3: this just returns the values store from the cHRM, if any. */
   if (png_ptr != NULL && info_ptr != NULL &&
       (info_ptr->valid & PNG_INFO_cHRM) != 0)
   {
      if (whitex != NULL) *whitex = info_ptr->cHRM.whitex;
      if (whitey != NULL) *whitey = info_ptr->cHRM.whitey;
      if (redx   != NULL) *redx   = info_ptr->cHRM.redx;
      if (redy   != NULL) *redy   = info_ptr->cHRM.redy;
      if (greenx != NULL) *greenx = info_ptr->cHRM.greenx;
      if (greeny != NULL) *greeny = info_ptr->cHRM.greeny;
      if (bluex  != NULL) *bluex  = info_ptr->cHRM.bluex;
      if (bluey  != NULL) *bluey  = info_ptr->cHRM.bluey;
      return PNG_INFO_cHRM;
   }

   return 0;
}
#  endif
#endif

#ifdef PNG_gAMA_SUPPORTED
#  ifdef PNG_FIXED_POINT_SUPPORTED
png_uint_32 PNGAPI
png_get_gAMA_fixed(png_const_structrp png_ptr, png_const_inforp info_ptr,
    png_fixed_point *file_gamma)
{
   png_debug1(1, "in %s retrieval function", "gAMA");

   /* PNGv3 compatibility: only report gAMA if it is really present. */
   if (png_ptr != NULL && info_ptr != NULL &&
       (info_ptr->valid & PNG_INFO_gAMA) != 0)
   {
      if (file_gamma != NULL) *file_gamma = info_ptr->gamma;
      return PNG_INFO_gAMA;
   }

   return 0;
}
#  endif

#  ifdef PNG_FLOATING_POINT_SUPPORTED
png_uint_32 PNGAPI
png_get_gAMA(png_const_structrp png_ptr, png_const_inforp info_ptr,
    double *file_gamma)
{
   png_debug1(1, "in %s retrieval function", "gAMA(float)");

   /* PNGv3 compatibility: only report gAMA if it is really present. */
   if (png_ptr != NULL && info_ptr != NULL &&
       (info_ptr->valid & PNG_INFO_gAMA) != 0)
   {
      if (file_gamma != NULL)
         *file_gamma = png_float(png_ptr, info_ptr->gamma, "gAMA");

      return PNG_INFO_gAMA;
   }

   return 0;
}
#  endif
#endif

#ifdef PNG_sRGB_SUPPORTED
png_uint_32 PNGAPI
png_get_sRGB(png_const_structrp png_ptr, png_const_inforp info_ptr,
    int *file_srgb_intent)
{
   png_debug1(1, "in %s retrieval function", "sRGB");

   if (png_ptr != NULL && info_ptr != NULL &&
      (info_ptr->valid & PNG_INFO_sRGB) != 0)
   {
      if (file_srgb_intent != NULL)
         *file_srgb_intent = info_ptr->rendering_intent;
      return PNG_INFO_sRGB;
   }

   return 0;
}
#endif

#ifdef PNG_iCCP_SUPPORTED
png_uint_32 PNGAPI
png_get_iCCP(png_const_structrp png_ptr, png_inforp info_ptr,
    png_charpp name, int *compression_type,
    png_bytepp profile, png_uint_32 *proflen)
{
   png_debug1(1, "in %s retrieval function", "iCCP");

   if (png_ptr != NULL && info_ptr != NULL &&
       (info_ptr->valid & PNG_INFO_iCCP) != 0 &&
       name != NULL && profile != NULL && proflen != NULL)
   {
      *name = info_ptr->iccp_name;
      *profile = info_ptr->iccp_profile;
      *proflen = png_get_uint_32(info_ptr->iccp_profile);
      /* This is somewhat irrelevant since the profile data returned has
       * actually been uncompressed.
       */
      if (compression_type != NULL)
         *compression_type = PNG_COMPRESSION_TYPE_BASE;
      return PNG_INFO_iCCP;
   }

   return 0;

}
#endif

#ifdef PNG_sPLT_SUPPORTED
int PNGAPI
png_get_sPLT(png_const_structrp png_ptr, png_inforp info_ptr,
    png_sPLT_tpp spalettes)
{
   png_debug1(1, "in %s retrieval function", "sPLT");

   if (png_ptr != NULL && info_ptr != NULL && spalettes != NULL)
   {
      *spalettes = info_ptr->splt_palettes;
      return info_ptr->splt_palettes_num;
   }

   return 0;
}
#endif

#ifdef PNG_cICP_SUPPORTED
png_uint_32 PNGAPI
png_get_cICP(png_const_structrp png_ptr,
             png_const_inforp info_ptr, png_bytep colour_primaries,
             png_bytep transfer_function, png_bytep matrix_coefficients,
             png_bytep video_full_range_flag)
{
    png_debug1(1, "in %s retrieval function", "cICP");

    if (png_ptr != NULL && info_ptr != NULL &&
        (info_ptr->valid & PNG_INFO_cICP) != 0 &&
        colour_primaries != NULL && transfer_function != NULL &&
        matrix_coefficients != NULL && video_full_range_flag != NULL)
    {
        *colour_primaries = info_ptr->cicp_colour_primaries;
        *transfer_function = info_ptr->cicp_transfer_function;
        *matrix_coefficients = info_ptr->cicp_matrix_coefficients;
        *video_full_range_flag = info_ptr->cicp_video_full_range_flag;
        return (PNG_INFO_cICP);
    }

    return 0;
}
#endif

#ifdef PNG_cLLI_SUPPORTED
#  ifdef PNG_FIXED_POINT_SUPPORTED
png_uint_32 PNGAPI
png_get_cLLI_fixed(png_const_structrp png_ptr, png_const_inforp info_ptr,
    png_uint_32p maxCLL,
    png_uint_32p maxFALL)
{
   png_debug1(1, "in %s retrieval function", "cLLI");

   if (png_ptr != NULL && info_ptr != NULL &&
       (info_ptr->valid & PNG_INFO_cLLI) != 0)
   {
      if (maxCLL != NULL) *maxCLL = info_ptr->maxCLL;
      if (maxFALL != NULL) *maxFALL = info_ptr->maxFALL;
      return PNG_INFO_cLLI;
   }

   return 0;
}
#  endif

#  ifdef PNG_FLOATING_POINT_SUPPORTED
png_uint_32 PNGAPI
png_get_cLLI(png_const_structrp png_ptr, png_const_inforp info_ptr,
      double *maxCLL, double *maxFALL)
{
   png_debug1(1, "in %s retrieval function", "cLLI(float)");

   if (png_ptr != NULL && info_ptr != NULL &&
       (info_ptr->valid & PNG_INFO_cLLI) != 0)
   {
      if (maxCLL != NULL) *maxCLL = info_ptr->maxCLL * .0001;
      if (maxFALL != NULL) *maxFALL = info_ptr->maxFALL * .0001;
      return PNG_INFO_cLLI;
   }

   return 0;
}
#  endif
#endif /* cLLI */

#ifdef PNG_mDCV_SUPPORTED
#  ifdef PNG_FIXED_POINT_SUPPORTED
png_uint_32 PNGAPI
png_get_mDCV_fixed(png_const_structrp png_ptr, png_const_inforp info_ptr,
    png_fixed_point *white_x, png_fixed_point *white_y,
    png_fixed_point *red_x, png_fixed_point *red_y,
    png_fixed_point *green_x, png_fixed_point *green_y,
    png_fixed_point *blue_x, png_fixed_point *blue_y,
    png_uint_32p mastering_maxDL, png_uint_32p mastering_minDL)
{
   png_debug1(1, "in %s retrieval function", "mDCV");

   if (png_ptr != NULL && info_ptr != NULL &&
       (info_ptr->valid & PNG_INFO_mDCV) != 0)
   {
      if (white_x != NULL) *white_x = info_ptr->mastering_white_x * 2;
      if (white_y != NULL) *white_y = info_ptr->mastering_white_y * 2;
      if (red_x != NULL) *red_x = info_ptr->mastering_red_x * 2;
      if (red_y != NULL) *red_y = info_ptr->mastering_red_y * 2;
      if (green_x != NULL) *green_x = info_ptr->mastering_green_x * 2;
      if (green_y != NULL) *green_y = info_ptr->mastering_green_y * 2;
      if (blue_x != NULL) *blue_x = info_ptr->mastering_blue_x * 2;
      if (blue_y != NULL) *blue_y = info_ptr->mastering_blue_y * 2;
      if (mastering_maxDL != NULL) *mastering_maxDL = info_ptr->mastering_maxDL;
      if (mastering_minDL != NULL) *mastering_minDL = info_ptr->mastering_minDL;
      return PNG_INFO_mDCV;
   }

   return 0;
}
#  endif

#  ifdef PNG_FLOATING_POINT_SUPPORTED
png_uint_32 PNGAPI
png_get_mDCV(png_const_structrp png_ptr, png_const_inforp info_ptr,
    double *white_x, double *white_y, double *red_x, double *red_y,
    double *green_x, double *green_y, double *blue_x, double *blue_y,
    double *mastering_maxDL, double *mastering_minDL)
{
   png_debug1(1, "in %s retrieval function", "mDCV(float)");

   if (png_ptr != NULL && info_ptr != NULL &&
       (info_ptr->valid & PNG_INFO_mDCV) != 0)
   {
      if (white_x != NULL) *white_x = info_ptr->mastering_white_x * .00002;
      if (white_y != NULL) *white_y = info_ptr->mastering_white_y * .00002;
      if (red_x != NULL) *red_x = info_ptr->mastering_red_x * .00002;
      if (red_y != NULL) *red_y = info_ptr->mastering_red_y * .00002;
      if (green_x != NULL) *green_x = info_ptr->mastering_green_x * .00002;
      if (green_y != NULL) *green_y = info_ptr->mastering_green_y * .00002;
      if (blue_x != NULL) *blue_x = info_ptr->mastering_blue_x * .00002;
      if (blue_y != NULL) *blue_y = info_ptr->mastering_blue_y * .00002;
      if (mastering_maxDL != NULL)
         *mastering_maxDL = info_ptr->mastering_maxDL * .0001;
      if (mastering_minDL != NULL)
         *mastering_minDL = info_ptr->mastering_minDL * .0001;
      return PNG_INFO_mDCV;
   }

   return 0;
}
#  endif /* FLOATING_POINT */
#endif /* mDCV */

#ifdef PNG_eXIf_SUPPORTED
png_uint_32 PNGAPI
png_get_eXIf(png_const_structrp png_ptr, png_inforp info_ptr,
    png_bytep *exif)
{
  png_warning(png_ptr, "png_get_eXIf does not work; use png_get_eXIf_1");
  PNG_UNUSED(info_ptr)
  PNG_UNUSED(exif)
  return 0;
}

png_uint_32 PNGAPI
png_get_eXIf_1(png_const_structrp png_ptr, png_const_inforp info_ptr,
    png_uint_32 *num_exif, png_bytep *exif)
{
   png_debug1(1, "in %s retrieval function", "eXIf");

   if (png_ptr != NULL && info_ptr != NULL &&
       (info_ptr->valid & PNG_INFO_eXIf) != 0 && exif != NULL)
   {
      *num_exif = info_ptr->num_exif;
      *exif = info_ptr->exif;
      return PNG_INFO_eXIf;
   }

   return 0;
}
#endif

#ifdef PNG_hIST_SUPPORTED
png_uint_32 PNGAPI
png_get_hIST(png_const_structrp png_ptr, png_inforp info_ptr,
    png_uint_16p *hist)
{
   png_debug1(1, "in %s retrieval function", "hIST");

   if (png_ptr != NULL && info_ptr != NULL &&
       (info_ptr->valid & PNG_INFO_hIST) != 0 && hist != NULL)
   {
      *hist = info_ptr->hist;
      return PNG_INFO_hIST;
   }

   return 0;
}
#endif

png_uint_32 PNGAPI
png_get_IHDR(png_const_structrp png_ptr, png_const_inforp info_ptr,
    png_uint_32 *width, png_uint_32 *height, int *bit_depth,
    int *color_type, int *interlace_type, int *compression_type,
    int *filter_type)
{
   png_debug1(1, "in %s retrieval function", "IHDR");

   if (png_ptr == NULL || info_ptr == NULL)
      return 0;

   if (width != NULL)
       *width = info_ptr->width;

   if (height != NULL)
       *height = info_ptr->height;

   if (bit_depth != NULL)
       *bit_depth = info_ptr->bit_depth;

   if (color_type != NULL)
       *color_type = info_ptr->color_type;

   if (compression_type != NULL)
      *compression_type = info_ptr->compression_type;

   if (filter_type != NULL)
      *filter_type = info_ptr->filter_type;

   if (interlace_type != NULL)
      *interlace_type = info_ptr->interlace_type;

   /* This is redundant if we can be sure that the info_ptr values were all
    * assigned in png_set_IHDR().  We do the check anyhow in case an
    * application has ignored our advice not to mess with the members
    * of info_ptr directly.
    */
   png_check_IHDR(png_ptr, info_ptr->width, info_ptr->height,
       info_ptr->bit_depth, info_ptr->color_type, info_ptr->interlace_type,
       info_ptr->compression_type, info_ptr->filter_type);

   return 1;
}

#ifdef PNG_oFFs_SUPPORTED
png_uint_32 PNGAPI
png_get_oFFs(png_const_structrp png_ptr, png_const_inforp info_ptr,
    png_int_32 *offset_x, png_int_32 *offset_y, int *unit_type)
{
   png_debug1(1, "in %s retrieval function", "oFFs");

   if (png_ptr != NULL && info_ptr != NULL &&
       (info_ptr->valid & PNG_INFO_oFFs) != 0 &&
       offset_x != NULL && offset_y != NULL && unit_type != NULL)
   {
      *offset_x = info_ptr->x_offset;
      *offset_y = info_ptr->y_offset;
      *unit_type = (int)info_ptr->offset_unit_type;
      return PNG_INFO_oFFs;
   }

   return 0;
}
#endif

#ifdef PNG_pCAL_SUPPORTED
png_uint_32 PNGAPI
png_get_pCAL(png_const_structrp png_ptr, png_inforp info_ptr,
    png_charp *purpose, png_int_32 *X0, png_int_32 *X1, int *type, int *nparams,
    png_charp *units, png_charpp *params)
{
   png_debug1(1, "in %s retrieval function", "pCAL");

   if (png_ptr != NULL && info_ptr != NULL &&
       (info_ptr->valid & PNG_INFO_pCAL) != 0 &&
       purpose != NULL && X0 != NULL && X1 != NULL && type != NULL &&
       nparams != NULL && units != NULL && params != NULL)
   {
      *purpose = info_ptr->pcal_purpose;
      *X0 = info_ptr->pcal_X0;
      *X1 = info_ptr->pcal_X1;
      *type = (int)info_ptr->pcal_type;
      *nparams = (int)info_ptr->pcal_nparams;
      *units = info_ptr->pcal_units;
      *params = info_ptr->pcal_params;
      return PNG_INFO_pCAL;
   }

   return 0;
}
#endif

#ifdef PNG_sCAL_SUPPORTED
#  ifdef PNG_FIXED_POINT_SUPPORTED
#    if defined(PNG_FLOATING_ARITHMETIC_SUPPORTED) || \
         defined(PNG_FLOATING_POINT_SUPPORTED)
png_uint_32 PNGAPI
png_get_sCAL_fixed(png_const_structrp png_ptr, png_const_inforp info_ptr,
    int *unit, png_fixed_point *width, png_fixed_point *height)
{
   png_debug1(1, "in %s retrieval function", "sCAL");

   if (png_ptr != NULL && info_ptr != NULL &&
       (info_ptr->valid & PNG_INFO_sCAL) != 0)
   {
      *unit = info_ptr->scal_unit;
      /*TODO: make this work without FP support; the API is currently eliminated
       * if neither floating point APIs nor internal floating point arithmetic
       * are enabled.
       */
      *width = png_fixed(png_ptr, atof(info_ptr->scal_s_width), "sCAL width");
      *height = png_fixed(png_ptr, atof(info_ptr->scal_s_height),
          "sCAL height");
      return PNG_INFO_sCAL;
   }

   return 0;
}
#    endif /* FLOATING_ARITHMETIC */
#  endif /* FIXED_POINT */
#  ifdef PNG_FLOATING_POINT_SUPPORTED
png_uint_32 PNGAPI
png_get_sCAL(png_const_structrp png_ptr, png_const_inforp info_ptr,
    int *unit, double *width, double *height)
{
   png_debug1(1, "in %s retrieval function", "sCAL(float)");

   if (png_ptr != NULL && info_ptr != NULL &&
       (info_ptr->valid & PNG_INFO_sCAL) != 0)
   {
      *unit = info_ptr->scal_unit;
      *width = atof(info_ptr->scal_s_width);
      *height = atof(info_ptr->scal_s_height);
      return PNG_INFO_sCAL;
   }

   return 0;
}
#  endif /* FLOATING POINT */
png_uint_32 PNGAPI
png_get_sCAL_s(png_const_structrp png_ptr, png_const_inforp info_ptr,
    int *unit, png_charpp width, png_charpp height)
{
   png_debug1(1, "in %s retrieval function", "sCAL(str)");

   if (png_ptr != NULL && info_ptr != NULL &&
       (info_ptr->valid & PNG_INFO_sCAL) != 0)
   {
      *unit = info_ptr->scal_unit;
      *width = info_ptr->scal_s_width;
      *height = info_ptr->scal_s_height;
      return PNG_INFO_sCAL;
   }

   return 0;
}
#endif /* sCAL */

#ifdef PNG_pHYs_SUPPORTED
png_uint_32 PNGAPI
png_get_pHYs(png_const_structrp png_ptr, png_const_inforp info_ptr,
    png_uint_32 *res_x, png_uint_32 *res_y, int *unit_type)
{
   png_uint_32 retval = 0;

   png_debug1(1, "in %s retrieval function", "pHYs");

   if (png_ptr != NULL && info_ptr != NULL &&
       (info_ptr->valid & PNG_INFO_pHYs) != 0)
   {
      if (res_x != NULL)
      {
         *res_x = info_ptr->x_pixels_per_unit;
         retval |= PNG_INFO_pHYs;
      }

      if (res_y != NULL)
      {
         *res_y = info_ptr->y_pixels_per_unit;
         retval |= PNG_INFO_pHYs;
      }

      if (unit_type != NULL)
      {
         *unit_type = (int)info_ptr->phys_unit_type;
         retval |= PNG_INFO_pHYs;
      }
   }

   return retval;
}
#endif /* pHYs */

png_uint_32 PNGAPI
png_get_PLTE(png_const_structrp png_ptr, png_inforp info_ptr,
    png_colorp *palette, int *num_palette)
{
   png_debug1(1, "in %s retrieval function", "PLTE");

   if (png_ptr != NULL && info_ptr != NULL &&
       (info_ptr->valid & PNG_INFO_PLTE) != 0 && palette != NULL)
   {
      *palette = info_ptr->palette;
      *num_palette = info_ptr->num_palette;
      png_debug1(3, "num_palette = %d", *num_palette);
      return PNG_INFO_PLTE;
   }

   return 0;
}

#ifdef PNG_sBIT_SUPPORTED
png_uint_32 PNGAPI
png_get_sBIT(png_const_structrp png_ptr, png_inforp info_ptr,
    png_color_8p *sig_bit)
{
   png_debug1(1, "in %s retrieval function", "sBIT");

   if (png_ptr != NULL && info_ptr != NULL &&
       (info_ptr->valid & PNG_INFO_sBIT) != 0 && sig_bit != NULL)
   {
      *sig_bit = &(info_ptr->sig_bit);
      return PNG_INFO_sBIT;
   }

   return 0;
}
#endif

#ifdef PNG_TEXT_SUPPORTED
int PNGAPI
png_get_text(png_const_structrp png_ptr, png_inforp info_ptr,
    png_textp *text_ptr, int *num_text)
{
   if (png_ptr != NULL && info_ptr != NULL && info_ptr->num_text > 0)
   {
      png_debug1(1, "in text retrieval function, chunk typeid = 0x%lx",
         (unsigned long)png_ptr->chunk_name);

      if (text_ptr != NULL)
         *text_ptr = info_ptr->text;

      if (num_text != NULL)
         *num_text = info_ptr->num_text;

      return info_ptr->num_text;
   }

   if (num_text != NULL)
      *num_text = 0;

   return 0;
}
#endif

#ifdef PNG_tIME_SUPPORTED
png_uint_32 PNGAPI
png_get_tIME(png_const_structrp png_ptr, png_inforp info_ptr,
    png_timep *mod_time)
{
   png_debug1(1, "in %s retrieval function", "tIME");

   if (png_ptr != NULL && info_ptr != NULL &&
       (info_ptr->valid & PNG_INFO_tIME) != 0 && mod_time != NULL)
   {
      *mod_time = &(info_ptr->mod_time);
      return PNG_INFO_tIME;
   }

   return 0;
}
#endif

#ifdef PNG_tRNS_SUPPORTED
png_uint_32 PNGAPI
png_get_tRNS(png_const_structrp png_ptr, png_inforp info_ptr,
    png_bytep *trans_alpha, int *num_trans, png_color_16p *trans_color)
{
   png_uint_32 retval = 0;

   png_debug1(1, "in %s retrieval function", "tRNS");

   if (png_ptr != NULL && info_ptr != NULL &&
       (info_ptr->valid & PNG_INFO_tRNS) != 0)
   {
      if (info_ptr->color_type == PNG_COLOR_TYPE_PALETTE)
      {
         if (trans_alpha != NULL)
         {
            *trans_alpha = info_ptr->trans_alpha;
            retval |= PNG_INFO_tRNS;
         }

         if (trans_color != NULL)
            *trans_color = &(info_ptr->trans_color);
      }

      else /* if (info_ptr->color_type != PNG_COLOR_TYPE_PALETTE) */
      {
         if (trans_color != NULL)
         {
            *trans_color = &(info_ptr->trans_color);
            retval |= PNG_INFO_tRNS;
         }

         if (trans_alpha != NULL)
            *trans_alpha = NULL;
      }

      if (num_trans != NULL)
      {
         *num_trans = info_ptr->num_trans;
         retval |= PNG_INFO_tRNS;
      }
   }

   return retval;
}
#endif

#ifdef PNG_STORE_UNKNOWN_CHUNKS_SUPPORTED
int PNGAPI
png_get_unknown_chunks(png_const_structrp png_ptr, png_inforp info_ptr,
    png_unknown_chunkpp unknowns)
{
   if (png_ptr != NULL && info_ptr != NULL && unknowns != NULL)
   {
      *unknowns = info_ptr->unknown_chunks;
      return info_ptr->unknown_chunks_num;
   }

   return 0;
}
#endif

#ifdef PNG_READ_RGB_TO_GRAY_SUPPORTED
png_byte PNGAPI
png_get_rgb_to_gray_status(png_const_structrp png_ptr)
{
   return (png_byte)(png_ptr ? png_ptr->rgb_to_gray_status : 0);
}
#endif

#ifdef PNG_USER_CHUNKS_SUPPORTED
png_voidp PNGAPI
png_get_user_chunk_ptr(png_const_structrp png_ptr)
{
   return (png_ptr ? png_ptr->user_chunk_ptr : NULL);
}
#endif

size_t PNGAPI
png_get_compression_buffer_size(png_const_structrp png_ptr)
{
   if (png_ptr == NULL)
      return 0;

#ifdef PNG_WRITE_SUPPORTED
   if ((png_ptr->mode & PNG_IS_READ_STRUCT) != 0)
#endif
   {
#ifdef PNG_SEQUENTIAL_READ_SUPPORTED
      return png_ptr->IDAT_read_size;
#else
      return PNG_IDAT_READ_SIZE;
#endif
   }

#ifdef PNG_WRITE_SUPPORTED
   else
      return png_ptr->zbuffer_size;
#endif
}

#ifdef PNG_SET_USER_LIMITS_SUPPORTED
/* These functions were added to libpng 1.2.6 and were enabled
 * by default in libpng-1.4.0 */
png_uint_32 PNGAPI
png_get_user_width_max(png_const_structrp png_ptr)
{
   return (png_ptr ? png_ptr->user_width_max : 0);
}

png_uint_32 PNGAPI
png_get_user_height_max(png_const_structrp png_ptr)
{
   return (png_ptr ? png_ptr->user_height_max : 0);
}

/* This function was added to libpng 1.4.0 */
png_uint_32 PNGAPI
png_get_chunk_cache_max(png_const_structrp png_ptr)
{
   return (png_ptr ? png_ptr->user_chunk_cache_max : 0);
}

/* This function was added to libpng 1.4.1 */
png_alloc_size_t PNGAPI
png_get_chunk_malloc_max(png_const_structrp png_ptr)
{
   return (png_ptr ? png_ptr->user_chunk_malloc_max : 0);
}
#endif /* SET_USER_LIMITS */

/* These functions were added to libpng 1.4.0 */
#ifdef PNG_IO_STATE_SUPPORTED
png_uint_32 PNGAPI
png_get_io_state(png_const_structrp png_ptr)
{
   return png_ptr->io_state;
}

png_uint_32 PNGAPI
png_get_io_chunk_type(png_const_structrp png_ptr)
{
   return png_ptr->chunk_name;
}
#endif /* IO_STATE */

#ifdef PNG_CHECK_FOR_INVALID_INDEX_SUPPORTED
#  ifdef PNG_GET_PALETTE_MAX_SUPPORTED
int PNGAPI
png_get_palette_max(png_const_structp png_ptr, png_const_infop info_ptr)
{
   if (png_ptr != NULL && info_ptr != NULL)
      return png_ptr->num_palette_max;

   return -1;
}
#  endif
#endif

#endif /* READ || WRITE */
