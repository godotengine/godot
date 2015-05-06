
/* pngset.c - storage of image information into info struct
 *
 * Last changed in libpng 1.5.7 [December 15, 2011]
 * Copyright (c) 1998-2011 Glenn Randers-Pehrson
 * (Version 0.96 Copyright (c) 1996, 1997 Andreas Dilger)
 * (Version 0.88 Copyright (c) 1995, 1996 Guy Eric Schalnat, Group 42, Inc.)
 *
 * This code is released under the libpng license.
 * For conditions of distribution and use, see the disclaimer
 * and license in png.h
 *
 * The functions here are used during reads to store data from the file
 * into the info struct, and during writes to store application data
 * into the info struct for writing into the file.  This abstracts the
 * info struct and allows us to change the structure in the future.
 */

#include "pngpriv.h"

#if defined(PNG_READ_SUPPORTED) || defined(PNG_WRITE_SUPPORTED)

#ifdef PNG_bKGD_SUPPORTED
void PNGAPI
png_set_bKGD(png_structp png_ptr, png_infop info_ptr,
    png_const_color_16p background)
{
   png_debug1(1, "in %s storage function", "bKGD");

   if (png_ptr == NULL || info_ptr == NULL)
      return;

   png_memcpy(&(info_ptr->background), background, png_sizeof(png_color_16));
   info_ptr->valid |= PNG_INFO_bKGD;
}
#endif

#ifdef PNG_cHRM_SUPPORTED
void PNGFAPI
png_set_cHRM_fixed(png_structp png_ptr, png_infop info_ptr,
    png_fixed_point white_x, png_fixed_point white_y, png_fixed_point red_x,
    png_fixed_point red_y, png_fixed_point green_x, png_fixed_point green_y,
    png_fixed_point blue_x, png_fixed_point blue_y)
{
   png_debug1(1, "in %s storage function", "cHRM fixed");

   if (png_ptr == NULL || info_ptr == NULL)
      return;

#  ifdef PNG_CHECK_cHRM_SUPPORTED
   if (png_check_cHRM_fixed(png_ptr,
       white_x, white_y, red_x, red_y, green_x, green_y, blue_x, blue_y))
#  endif
   {
      info_ptr->x_white = white_x;
      info_ptr->y_white = white_y;
      info_ptr->x_red   = red_x;
      info_ptr->y_red   = red_y;
      info_ptr->x_green = green_x;
      info_ptr->y_green = green_y;
      info_ptr->x_blue  = blue_x;
      info_ptr->y_blue  = blue_y;
      info_ptr->valid |= PNG_INFO_cHRM;
   }
}

void PNGFAPI
png_set_cHRM_XYZ_fixed(png_structp png_ptr, png_infop info_ptr,
    png_fixed_point int_red_X, png_fixed_point int_red_Y,
    png_fixed_point int_red_Z, png_fixed_point int_green_X,
    png_fixed_point int_green_Y, png_fixed_point int_green_Z,
    png_fixed_point int_blue_X, png_fixed_point int_blue_Y,
    png_fixed_point int_blue_Z)
{
   png_XYZ XYZ;
   png_xy xy;

   png_debug1(1, "in %s storage function", "cHRM XYZ fixed");

   if (png_ptr == NULL || info_ptr == NULL)
      return;

   XYZ.redX = int_red_X;
   XYZ.redY = int_red_Y;
   XYZ.redZ = int_red_Z;
   XYZ.greenX = int_green_X;
   XYZ.greenY = int_green_Y;
   XYZ.greenZ = int_green_Z;
   XYZ.blueX = int_blue_X;
   XYZ.blueY = int_blue_Y;
   XYZ.blueZ = int_blue_Z;

   if (png_xy_from_XYZ(&xy, XYZ))
      png_error(png_ptr, "XYZ values out of representable range");

   png_set_cHRM_fixed(png_ptr, info_ptr, xy.whitex, xy.whitey, xy.redx, xy.redy,
      xy.greenx, xy.greeny, xy.bluex, xy.bluey);
}

#  ifdef PNG_FLOATING_POINT_SUPPORTED
void PNGAPI
png_set_cHRM(png_structp png_ptr, png_infop info_ptr,
    double white_x, double white_y, double red_x, double red_y,
    double green_x, double green_y, double blue_x, double blue_y)
{
   png_set_cHRM_fixed(png_ptr, info_ptr,
      png_fixed(png_ptr, white_x, "cHRM White X"),
      png_fixed(png_ptr, white_y, "cHRM White Y"),
      png_fixed(png_ptr, red_x, "cHRM Red X"),
      png_fixed(png_ptr, red_y, "cHRM Red Y"),
      png_fixed(png_ptr, green_x, "cHRM Green X"),
      png_fixed(png_ptr, green_y, "cHRM Green Y"),
      png_fixed(png_ptr, blue_x, "cHRM Blue X"),
      png_fixed(png_ptr, blue_y, "cHRM Blue Y"));
}

void PNGAPI
png_set_cHRM_XYZ(png_structp png_ptr, png_infop info_ptr, double red_X,
    double red_Y, double red_Z, double green_X, double green_Y, double green_Z,
    double blue_X, double blue_Y, double blue_Z)
{
   png_set_cHRM_XYZ_fixed(png_ptr, info_ptr,
      png_fixed(png_ptr, red_X, "cHRM Red X"),
      png_fixed(png_ptr, red_Y, "cHRM Red Y"),
      png_fixed(png_ptr, red_Z, "cHRM Red Z"),
      png_fixed(png_ptr, green_X, "cHRM Red X"),
      png_fixed(png_ptr, green_Y, "cHRM Red Y"),
      png_fixed(png_ptr, green_Z, "cHRM Red Z"),
      png_fixed(png_ptr, blue_X, "cHRM Red X"),
      png_fixed(png_ptr, blue_Y, "cHRM Red Y"),
      png_fixed(png_ptr, blue_Z, "cHRM Red Z"));
}
#  endif /* PNG_FLOATING_POINT_SUPPORTED */

#endif /* PNG_cHRM_SUPPORTED */

#ifdef PNG_gAMA_SUPPORTED
void PNGFAPI
png_set_gAMA_fixed(png_structp png_ptr, png_infop info_ptr, png_fixed_point
    file_gamma)
{
   png_debug1(1, "in %s storage function", "gAMA");

   if (png_ptr == NULL || info_ptr == NULL)
      return;

   /* Changed in libpng-1.5.4 to limit the values to ensure overflow can't
    * occur.  Since the fixed point representation is assymetrical it is
    * possible for 1/gamma to overflow the limit of 21474 and this means the
    * gamma value must be at least 5/100000 and hence at most 20000.0.  For
    * safety the limits here are a little narrower.  The values are 0.00016 to
    * 6250.0, which are truly ridiculous gammma values (and will produce
    * displays that are all black or all white.)
    */
   if (file_gamma < 16 || file_gamma > 625000000)
      png_warning(png_ptr, "Out of range gamma value ignored");

   else
   {
      info_ptr->gamma = file_gamma;
      info_ptr->valid |= PNG_INFO_gAMA;
   }
}

#  ifdef PNG_FLOATING_POINT_SUPPORTED
void PNGAPI
png_set_gAMA(png_structp png_ptr, png_infop info_ptr, double file_gamma)
{
   png_set_gAMA_fixed(png_ptr, info_ptr, png_fixed(png_ptr, file_gamma,
       "png_set_gAMA"));
}
#  endif
#endif

#ifdef PNG_hIST_SUPPORTED
void PNGAPI
png_set_hIST(png_structp png_ptr, png_infop info_ptr, png_const_uint_16p hist)
{
   int i;

   png_debug1(1, "in %s storage function", "hIST");

   if (png_ptr == NULL || info_ptr == NULL)
      return;

   if (info_ptr->num_palette == 0 || info_ptr->num_palette
       > PNG_MAX_PALETTE_LENGTH)
   {
      png_warning(png_ptr,
          "Invalid palette size, hIST allocation skipped");

      return;
   }

   png_free_data(png_ptr, info_ptr, PNG_FREE_HIST, 0);

   /* Changed from info->num_palette to PNG_MAX_PALETTE_LENGTH in
    * version 1.2.1
    */
   png_ptr->hist = (png_uint_16p)png_malloc_warn(png_ptr,
       PNG_MAX_PALETTE_LENGTH * png_sizeof(png_uint_16));

   if (png_ptr->hist == NULL)
   {
      png_warning(png_ptr, "Insufficient memory for hIST chunk data");
      return;
   }

   for (i = 0; i < info_ptr->num_palette; i++)
      png_ptr->hist[i] = hist[i];

   info_ptr->hist = png_ptr->hist;
   info_ptr->valid |= PNG_INFO_hIST;
   info_ptr->free_me |= PNG_FREE_HIST;
}
#endif

void PNGAPI
png_set_IHDR(png_structp png_ptr, png_infop info_ptr,
    png_uint_32 width, png_uint_32 height, int bit_depth,
    int color_type, int interlace_type, int compression_type,
    int filter_type)
{
   png_debug1(1, "in %s storage function", "IHDR");

   if (png_ptr == NULL || info_ptr == NULL)
      return;

   info_ptr->width = width;
   info_ptr->height = height;
   info_ptr->bit_depth = (png_byte)bit_depth;
   info_ptr->color_type = (png_byte)color_type;
   info_ptr->compression_type = (png_byte)compression_type;
   info_ptr->filter_type = (png_byte)filter_type;
   info_ptr->interlace_type = (png_byte)interlace_type;

   png_check_IHDR (png_ptr, info_ptr->width, info_ptr->height,
       info_ptr->bit_depth, info_ptr->color_type, info_ptr->interlace_type,
       info_ptr->compression_type, info_ptr->filter_type);

   if (info_ptr->color_type == PNG_COLOR_TYPE_PALETTE)
      info_ptr->channels = 1;

   else if (info_ptr->color_type & PNG_COLOR_MASK_COLOR)
      info_ptr->channels = 3;

   else
      info_ptr->channels = 1;

   if (info_ptr->color_type & PNG_COLOR_MASK_ALPHA)
      info_ptr->channels++;

   info_ptr->pixel_depth = (png_byte)(info_ptr->channels * info_ptr->bit_depth);

   /* Check for potential overflow */
   if (width >
       (PNG_UINT_32_MAX >> 3)      /* 8-byte RRGGBBAA pixels */
       - 48       /* bigrowbuf hack */
       - 1        /* filter byte */
       - 7*8      /* rounding of width to multiple of 8 pixels */
       - 8)       /* extra max_pixel_depth pad */
      info_ptr->rowbytes = 0;
   else
      info_ptr->rowbytes = PNG_ROWBYTES(info_ptr->pixel_depth, width);
}

#ifdef PNG_oFFs_SUPPORTED
void PNGAPI
png_set_oFFs(png_structp png_ptr, png_infop info_ptr,
    png_int_32 offset_x, png_int_32 offset_y, int unit_type)
{
   png_debug1(1, "in %s storage function", "oFFs");

   if (png_ptr == NULL || info_ptr == NULL)
      return;

   info_ptr->x_offset = offset_x;
   info_ptr->y_offset = offset_y;
   info_ptr->offset_unit_type = (png_byte)unit_type;
   info_ptr->valid |= PNG_INFO_oFFs;
}
#endif

#ifdef PNG_pCAL_SUPPORTED
void PNGAPI
png_set_pCAL(png_structp png_ptr, png_infop info_ptr,
    png_const_charp purpose, png_int_32 X0, png_int_32 X1, int type,
    int nparams, png_const_charp units, png_charpp params)
{
   png_size_t length;
   int i;

   png_debug1(1, "in %s storage function", "pCAL");

   if (png_ptr == NULL || info_ptr == NULL)
      return;

   length = png_strlen(purpose) + 1;
   png_debug1(3, "allocating purpose for info (%lu bytes)",
       (unsigned long)length);

   /* TODO: validate format of calibration name and unit name */

   /* Check that the type matches the specification. */
   if (type < 0 || type > 3)
      png_error(png_ptr, "Invalid pCAL equation type");

   /* Validate params[nparams] */
   for (i=0; i<nparams; ++i)
      if (!png_check_fp_string(params[i], png_strlen(params[i])))
         png_error(png_ptr, "Invalid format for pCAL parameter");

   info_ptr->pcal_purpose = (png_charp)png_malloc_warn(png_ptr, length);

   if (info_ptr->pcal_purpose == NULL)
   {
      png_warning(png_ptr, "Insufficient memory for pCAL purpose");
      return;
   }

   png_memcpy(info_ptr->pcal_purpose, purpose, length);

   png_debug(3, "storing X0, X1, type, and nparams in info");
   info_ptr->pcal_X0 = X0;
   info_ptr->pcal_X1 = X1;
   info_ptr->pcal_type = (png_byte)type;
   info_ptr->pcal_nparams = (png_byte)nparams;

   length = png_strlen(units) + 1;
   png_debug1(3, "allocating units for info (%lu bytes)",
     (unsigned long)length);

   info_ptr->pcal_units = (png_charp)png_malloc_warn(png_ptr, length);

   if (info_ptr->pcal_units == NULL)
   {
      png_warning(png_ptr, "Insufficient memory for pCAL units");
      return;
   }

   png_memcpy(info_ptr->pcal_units, units, length);

   info_ptr->pcal_params = (png_charpp)png_malloc_warn(png_ptr,
       (png_size_t)((nparams + 1) * png_sizeof(png_charp)));

   if (info_ptr->pcal_params == NULL)
   {
      png_warning(png_ptr, "Insufficient memory for pCAL params");
      return;
   }

   png_memset(info_ptr->pcal_params, 0, (nparams + 1) * png_sizeof(png_charp));

   for (i = 0; i < nparams; i++)
   {
      length = png_strlen(params[i]) + 1;
      png_debug2(3, "allocating parameter %d for info (%lu bytes)", i,
          (unsigned long)length);

      info_ptr->pcal_params[i] = (png_charp)png_malloc_warn(png_ptr, length);

      if (info_ptr->pcal_params[i] == NULL)
      {
         png_warning(png_ptr, "Insufficient memory for pCAL parameter");
         return;
      }

      png_memcpy(info_ptr->pcal_params[i], params[i], length);
   }

   info_ptr->valid |= PNG_INFO_pCAL;
   info_ptr->free_me |= PNG_FREE_PCAL;
}
#endif

#ifdef PNG_sCAL_SUPPORTED
void PNGAPI
png_set_sCAL_s(png_structp png_ptr, png_infop info_ptr,
    int unit, png_const_charp swidth, png_const_charp sheight)
{
   png_size_t lengthw = 0, lengthh = 0;

   png_debug1(1, "in %s storage function", "sCAL");

   if (png_ptr == NULL || info_ptr == NULL)
      return;

   /* Double check the unit (should never get here with an invalid
    * unit unless this is an API call.)
    */
   if (unit != 1 && unit != 2)
      png_error(png_ptr, "Invalid sCAL unit");

   if (swidth == NULL || (lengthw = png_strlen(swidth)) == 0 ||
       swidth[0] == 45 /* '-' */ || !png_check_fp_string(swidth, lengthw))
      png_error(png_ptr, "Invalid sCAL width");

   if (sheight == NULL || (lengthh = png_strlen(sheight)) == 0 ||
       sheight[0] == 45 /* '-' */ || !png_check_fp_string(sheight, lengthh))
      png_error(png_ptr, "Invalid sCAL height");

   info_ptr->scal_unit = (png_byte)unit;

   ++lengthw;

   png_debug1(3, "allocating unit for info (%u bytes)", (unsigned int)lengthw);

   info_ptr->scal_s_width = (png_charp)png_malloc_warn(png_ptr, lengthw);

   if (info_ptr->scal_s_width == NULL)
   {
      png_warning(png_ptr, "Memory allocation failed while processing sCAL");
      return;
   }

   png_memcpy(info_ptr->scal_s_width, swidth, lengthw);

   ++lengthh;

   png_debug1(3, "allocating unit for info (%u bytes)", (unsigned int)lengthh);

   info_ptr->scal_s_height = (png_charp)png_malloc_warn(png_ptr, lengthh);

   if (info_ptr->scal_s_height == NULL)
   {
      png_free (png_ptr, info_ptr->scal_s_width);
      info_ptr->scal_s_width = NULL;

      png_warning(png_ptr, "Memory allocation failed while processing sCAL");
      return;
   }

   png_memcpy(info_ptr->scal_s_height, sheight, lengthh);

   info_ptr->valid |= PNG_INFO_sCAL;
   info_ptr->free_me |= PNG_FREE_SCAL;
}

#  ifdef PNG_FLOATING_POINT_SUPPORTED
void PNGAPI
png_set_sCAL(png_structp png_ptr, png_infop info_ptr, int unit, double width,
    double height)
{
   png_debug1(1, "in %s storage function", "sCAL");

   /* Check the arguments. */
   if (width <= 0)
      png_warning(png_ptr, "Invalid sCAL width ignored");

   else if (height <= 0)
      png_warning(png_ptr, "Invalid sCAL height ignored");

   else
   {
      /* Convert 'width' and 'height' to ASCII. */
      char swidth[PNG_sCAL_MAX_DIGITS+1];
      char sheight[PNG_sCAL_MAX_DIGITS+1];

      png_ascii_from_fp(png_ptr, swidth, sizeof swidth, width,
         PNG_sCAL_PRECISION);
      png_ascii_from_fp(png_ptr, sheight, sizeof sheight, height,
         PNG_sCAL_PRECISION);

      png_set_sCAL_s(png_ptr, info_ptr, unit, swidth, sheight);
   }
}
#  endif

#  ifdef PNG_FIXED_POINT_SUPPORTED
void PNGAPI
png_set_sCAL_fixed(png_structp png_ptr, png_infop info_ptr, int unit,
    png_fixed_point width, png_fixed_point height)
{
   png_debug1(1, "in %s storage function", "sCAL");

   /* Check the arguments. */
   if (width <= 0)
      png_warning(png_ptr, "Invalid sCAL width ignored");

   else if (height <= 0)
      png_warning(png_ptr, "Invalid sCAL height ignored");

   else
   {
      /* Convert 'width' and 'height' to ASCII. */
      char swidth[PNG_sCAL_MAX_DIGITS+1];
      char sheight[PNG_sCAL_MAX_DIGITS+1];

      png_ascii_from_fixed(png_ptr, swidth, sizeof swidth, width);
      png_ascii_from_fixed(png_ptr, sheight, sizeof sheight, height);

      png_set_sCAL_s(png_ptr, info_ptr, unit, swidth, sheight);
   }
}
#  endif
#endif

#ifdef PNG_pHYs_SUPPORTED
void PNGAPI
png_set_pHYs(png_structp png_ptr, png_infop info_ptr,
    png_uint_32 res_x, png_uint_32 res_y, int unit_type)
{
   png_debug1(1, "in %s storage function", "pHYs");

   if (png_ptr == NULL || info_ptr == NULL)
      return;

   info_ptr->x_pixels_per_unit = res_x;
   info_ptr->y_pixels_per_unit = res_y;
   info_ptr->phys_unit_type = (png_byte)unit_type;
   info_ptr->valid |= PNG_INFO_pHYs;
}
#endif

void PNGAPI
png_set_PLTE(png_structp png_ptr, png_infop info_ptr,
    png_const_colorp palette, int num_palette)
{

   png_debug1(1, "in %s storage function", "PLTE");

   if (png_ptr == NULL || info_ptr == NULL)
      return;

   if (num_palette < 0 || num_palette > PNG_MAX_PALETTE_LENGTH)
   {
      if (info_ptr->color_type == PNG_COLOR_TYPE_PALETTE)
         png_error(png_ptr, "Invalid palette length");

      else
      {
         png_warning(png_ptr, "Invalid palette length");
         return;
      }
   }

   /* It may not actually be necessary to set png_ptr->palette here;
    * we do it for backward compatibility with the way the png_handle_tRNS
    * function used to do the allocation.
    */
   png_free_data(png_ptr, info_ptr, PNG_FREE_PLTE, 0);

   /* Changed in libpng-1.2.1 to allocate PNG_MAX_PALETTE_LENGTH instead
    * of num_palette entries, in case of an invalid PNG file that has
    * too-large sample values.
    */
   png_ptr->palette = (png_colorp)png_calloc(png_ptr,
       PNG_MAX_PALETTE_LENGTH * png_sizeof(png_color));

   png_memcpy(png_ptr->palette, palette, num_palette * png_sizeof(png_color));
   info_ptr->palette = png_ptr->palette;
   info_ptr->num_palette = png_ptr->num_palette = (png_uint_16)num_palette;

   info_ptr->free_me |= PNG_FREE_PLTE;

   info_ptr->valid |= PNG_INFO_PLTE;
}

#ifdef PNG_sBIT_SUPPORTED
void PNGAPI
png_set_sBIT(png_structp png_ptr, png_infop info_ptr,
    png_const_color_8p sig_bit)
{
   png_debug1(1, "in %s storage function", "sBIT");

   if (png_ptr == NULL || info_ptr == NULL)
      return;

   png_memcpy(&(info_ptr->sig_bit), sig_bit, png_sizeof(png_color_8));
   info_ptr->valid |= PNG_INFO_sBIT;
}
#endif

#ifdef PNG_sRGB_SUPPORTED
void PNGAPI
png_set_sRGB(png_structp png_ptr, png_infop info_ptr, int srgb_intent)
{
   png_debug1(1, "in %s storage function", "sRGB");

   if (png_ptr == NULL || info_ptr == NULL)
      return;

   info_ptr->srgb_intent = (png_byte)srgb_intent;
   info_ptr->valid |= PNG_INFO_sRGB;
}

void PNGAPI
png_set_sRGB_gAMA_and_cHRM(png_structp png_ptr, png_infop info_ptr,
    int srgb_intent)
{
   png_debug1(1, "in %s storage function", "sRGB_gAMA_and_cHRM");

   if (png_ptr == NULL || info_ptr == NULL)
      return;

   png_set_sRGB(png_ptr, info_ptr, srgb_intent);

#  ifdef PNG_gAMA_SUPPORTED
   png_set_gAMA_fixed(png_ptr, info_ptr, PNG_GAMMA_sRGB_INVERSE);
#  endif

#  ifdef PNG_cHRM_SUPPORTED
   png_set_cHRM_fixed(png_ptr, info_ptr,
      /* color      x       y */
      /* white */ 31270, 32900,
      /* red   */ 64000, 33000,
      /* green */ 30000, 60000,
      /* blue  */ 15000,  6000
   );
#  endif /* cHRM */
}
#endif /* sRGB */


#ifdef PNG_iCCP_SUPPORTED
void PNGAPI
png_set_iCCP(png_structp png_ptr, png_infop info_ptr,
    png_const_charp name, int compression_type,
    png_const_bytep profile, png_uint_32 proflen)
{
   png_charp new_iccp_name;
   png_bytep new_iccp_profile;
   png_size_t length;

   png_debug1(1, "in %s storage function", "iCCP");

   if (png_ptr == NULL || info_ptr == NULL || name == NULL || profile == NULL)
      return;

   length = png_strlen(name)+1;
   new_iccp_name = (png_charp)png_malloc_warn(png_ptr, length);

   if (new_iccp_name == NULL)
   {
        png_warning(png_ptr, "Insufficient memory to process iCCP chunk");
      return;
   }

   png_memcpy(new_iccp_name, name, length);
   new_iccp_profile = (png_bytep)png_malloc_warn(png_ptr, proflen);

   if (new_iccp_profile == NULL)
   {
      png_free (png_ptr, new_iccp_name);
      png_warning(png_ptr,
          "Insufficient memory to process iCCP profile");
      return;
   }

   png_memcpy(new_iccp_profile, profile, (png_size_t)proflen);

   png_free_data(png_ptr, info_ptr, PNG_FREE_ICCP, 0);

   info_ptr->iccp_proflen = proflen;
   info_ptr->iccp_name = new_iccp_name;
   info_ptr->iccp_profile = new_iccp_profile;
   /* Compression is always zero but is here so the API and info structure
    * does not have to change if we introduce multiple compression types
    */
   info_ptr->iccp_compression = (png_byte)compression_type;
   info_ptr->free_me |= PNG_FREE_ICCP;
   info_ptr->valid |= PNG_INFO_iCCP;
}
#endif

#ifdef PNG_TEXT_SUPPORTED
void PNGAPI
png_set_text(png_structp png_ptr, png_infop info_ptr, png_const_textp text_ptr,
    int num_text)
{
   int ret;
   ret = png_set_text_2(png_ptr, info_ptr, text_ptr, num_text);

   if (ret)
      png_error(png_ptr, "Insufficient memory to store text");
}

int /* PRIVATE */
png_set_text_2(png_structp png_ptr, png_infop info_ptr,
    png_const_textp text_ptr, int num_text)
{
   int i;

   png_debug1(1, "in %lx storage function", png_ptr == NULL ? "unexpected" :
      (unsigned long)png_ptr->chunk_name);

   if (png_ptr == NULL || info_ptr == NULL || num_text == 0)
      return(0);

   /* Make sure we have enough space in the "text" array in info_struct
    * to hold all of the incoming text_ptr objects.
    */
   if (info_ptr->num_text + num_text > info_ptr->max_text)
   {
      if (info_ptr->text != NULL)
      {
         png_textp old_text;
         int old_max;

         old_max = info_ptr->max_text;
         info_ptr->max_text = info_ptr->num_text + num_text + 8;
         old_text = info_ptr->text;
         info_ptr->text = (png_textp)png_malloc_warn(png_ptr,
            (png_size_t)(info_ptr->max_text * png_sizeof(png_text)));

         if (info_ptr->text == NULL)
         {
            png_free(png_ptr, old_text);
            return(1);
         }

         png_memcpy(info_ptr->text, old_text, (png_size_t)(old_max *
             png_sizeof(png_text)));
         png_free(png_ptr, old_text);
      }

      else
      {
         info_ptr->max_text = num_text + 8;
         info_ptr->num_text = 0;
         info_ptr->text = (png_textp)png_malloc_warn(png_ptr,
             (png_size_t)(info_ptr->max_text * png_sizeof(png_text)));
         if (info_ptr->text == NULL)
            return(1);
         info_ptr->free_me |= PNG_FREE_TEXT;
      }

      png_debug1(3, "allocated %d entries for info_ptr->text",
          info_ptr->max_text);
   }
   for (i = 0; i < num_text; i++)
   {
      png_size_t text_length, key_len;
      png_size_t lang_len, lang_key_len;
      png_textp textp = &(info_ptr->text[info_ptr->num_text]);

      if (text_ptr[i].key == NULL)
          continue;

      if (text_ptr[i].compression < PNG_TEXT_COMPRESSION_NONE ||
          text_ptr[i].compression >= PNG_TEXT_COMPRESSION_LAST)
      {
         png_warning(png_ptr, "text compression mode is out of range");
         continue;
      }

      key_len = png_strlen(text_ptr[i].key);

      if (text_ptr[i].compression <= 0)
      {
         lang_len = 0;
         lang_key_len = 0;
      }

      else
#  ifdef PNG_iTXt_SUPPORTED
      {
         /* Set iTXt data */

         if (text_ptr[i].lang != NULL)
            lang_len = png_strlen(text_ptr[i].lang);

         else
            lang_len = 0;

         if (text_ptr[i].lang_key != NULL)
            lang_key_len = png_strlen(text_ptr[i].lang_key);

         else
            lang_key_len = 0;
      }
#  else /* PNG_iTXt_SUPPORTED */
      {
         png_warning(png_ptr, "iTXt chunk not supported");
         continue;
      }
#  endif

      if (text_ptr[i].text == NULL || text_ptr[i].text[0] == '\0')
      {
         text_length = 0;
#  ifdef PNG_iTXt_SUPPORTED
         if (text_ptr[i].compression > 0)
            textp->compression = PNG_ITXT_COMPRESSION_NONE;

         else
#  endif
            textp->compression = PNG_TEXT_COMPRESSION_NONE;
      }

      else
      {
         text_length = png_strlen(text_ptr[i].text);
         textp->compression = text_ptr[i].compression;
      }

      textp->key = (png_charp)png_malloc_warn(png_ptr,
          (png_size_t)
          (key_len + text_length + lang_len + lang_key_len + 4));

      if (textp->key == NULL)
         return(1);

      png_debug2(2, "Allocated %lu bytes at %p in png_set_text",
          (unsigned long)(png_uint_32)
          (key_len + lang_len + lang_key_len + text_length + 4),
          textp->key);

      png_memcpy(textp->key, text_ptr[i].key,(png_size_t)(key_len));
      *(textp->key + key_len) = '\0';

      if (text_ptr[i].compression > 0)
      {
         textp->lang = textp->key + key_len + 1;
         png_memcpy(textp->lang, text_ptr[i].lang, lang_len);
         *(textp->lang + lang_len) = '\0';
         textp->lang_key = textp->lang + lang_len + 1;
         png_memcpy(textp->lang_key, text_ptr[i].lang_key, lang_key_len);
         *(textp->lang_key + lang_key_len) = '\0';
         textp->text = textp->lang_key + lang_key_len + 1;
      }

      else
      {
         textp->lang=NULL;
         textp->lang_key=NULL;
         textp->text = textp->key + key_len + 1;
      }

      if (text_length)
         png_memcpy(textp->text, text_ptr[i].text,
             (png_size_t)(text_length));

      *(textp->text + text_length) = '\0';

#  ifdef PNG_iTXt_SUPPORTED
      if (textp->compression > 0)
      {
         textp->text_length = 0;
         textp->itxt_length = text_length;
      }

      else
#  endif
      {
         textp->text_length = text_length;
         textp->itxt_length = 0;
      }

      info_ptr->num_text++;
      png_debug1(3, "transferred text chunk %d", info_ptr->num_text);
   }
   return(0);
}
#endif

#ifdef PNG_tIME_SUPPORTED
void PNGAPI
png_set_tIME(png_structp png_ptr, png_infop info_ptr, png_const_timep mod_time)
{
   png_debug1(1, "in %s storage function", "tIME");

   if (png_ptr == NULL || info_ptr == NULL ||
       (png_ptr->mode & PNG_WROTE_tIME))
      return;

   if (mod_time->month == 0   || mod_time->month > 12  ||
       mod_time->day   == 0   || mod_time->day   > 31  ||
       mod_time->hour  > 23   || mod_time->minute > 59 ||
       mod_time->second > 60)
   {
      png_warning(png_ptr, "Ignoring invalid time value");
      return;
   }

   png_memcpy(&(info_ptr->mod_time), mod_time, png_sizeof(png_time));
   info_ptr->valid |= PNG_INFO_tIME;
}
#endif

#ifdef PNG_tRNS_SUPPORTED
void PNGAPI
png_set_tRNS(png_structp png_ptr, png_infop info_ptr,
    png_const_bytep trans_alpha, int num_trans, png_const_color_16p trans_color)
{
   png_debug1(1, "in %s storage function", "tRNS");

   if (png_ptr == NULL || info_ptr == NULL)
      return;

   if (trans_alpha != NULL)
   {
       /* It may not actually be necessary to set png_ptr->trans_alpha here;
        * we do it for backward compatibility with the way the png_handle_tRNS
        * function used to do the allocation.
        */

       png_free_data(png_ptr, info_ptr, PNG_FREE_TRNS, 0);

       /* Changed from num_trans to PNG_MAX_PALETTE_LENGTH in version 1.2.1 */
       png_ptr->trans_alpha = info_ptr->trans_alpha =
           (png_bytep)png_malloc(png_ptr, (png_size_t)PNG_MAX_PALETTE_LENGTH);

       if (num_trans > 0 && num_trans <= PNG_MAX_PALETTE_LENGTH)
          png_memcpy(info_ptr->trans_alpha, trans_alpha, (png_size_t)num_trans);
   }

   if (trans_color != NULL)
   {
      int sample_max = (1 << info_ptr->bit_depth);

      if ((info_ptr->color_type == PNG_COLOR_TYPE_GRAY &&
          (int)trans_color->gray > sample_max) ||
          (info_ptr->color_type == PNG_COLOR_TYPE_RGB &&
          ((int)trans_color->red > sample_max ||
          (int)trans_color->green > sample_max ||
          (int)trans_color->blue > sample_max)))
         png_warning(png_ptr,
            "tRNS chunk has out-of-range samples for bit_depth");

      png_memcpy(&(info_ptr->trans_color), trans_color,
         png_sizeof(png_color_16));

      if (num_trans == 0)
         num_trans = 1;
   }

   info_ptr->num_trans = (png_uint_16)num_trans;

   if (num_trans != 0)
   {
      info_ptr->valid |= PNG_INFO_tRNS;
      info_ptr->free_me |= PNG_FREE_TRNS;
   }
}
#endif

#ifdef PNG_sPLT_SUPPORTED
void PNGAPI
png_set_sPLT(png_structp png_ptr,
    png_infop info_ptr, png_const_sPLT_tp entries, int nentries)
/*
 *  entries        - array of png_sPLT_t structures
 *                   to be added to the list of palettes
 *                   in the info structure.
 *
 *  nentries       - number of palette structures to be
 *                   added.
 */
{
   png_sPLT_tp np;
   int i;

   if (png_ptr == NULL || info_ptr == NULL)
      return;

   np = (png_sPLT_tp)png_malloc_warn(png_ptr,
       (info_ptr->splt_palettes_num + nentries) *
       (png_size_t)png_sizeof(png_sPLT_t));

   if (np == NULL)
   {
      png_warning(png_ptr, "No memory for sPLT palettes");
      return;
   }

   png_memcpy(np, info_ptr->splt_palettes,
       info_ptr->splt_palettes_num * png_sizeof(png_sPLT_t));

   png_free(png_ptr, info_ptr->splt_palettes);
   info_ptr->splt_palettes=NULL;

   for (i = 0; i < nentries; i++)
   {
      png_sPLT_tp to = np + info_ptr->splt_palettes_num + i;
      png_const_sPLT_tp from = entries + i;
      png_size_t length;

      length = png_strlen(from->name) + 1;
      to->name = (png_charp)png_malloc_warn(png_ptr, length);

      if (to->name == NULL)
      {
         png_warning(png_ptr,
             "Out of memory while processing sPLT chunk");
         continue;
      }

      png_memcpy(to->name, from->name, length);
      to->entries = (png_sPLT_entryp)png_malloc_warn(png_ptr,
          from->nentries * png_sizeof(png_sPLT_entry));

      if (to->entries == NULL)
      {
         png_warning(png_ptr,
             "Out of memory while processing sPLT chunk");
         png_free(png_ptr, to->name);
         to->name = NULL;
         continue;
      }

      png_memcpy(to->entries, from->entries,
          from->nentries * png_sizeof(png_sPLT_entry));

      to->nentries = from->nentries;
      to->depth = from->depth;
   }

   info_ptr->splt_palettes = np;
   info_ptr->splt_palettes_num += nentries;
   info_ptr->valid |= PNG_INFO_sPLT;
   info_ptr->free_me |= PNG_FREE_SPLT;
}
#endif /* PNG_sPLT_SUPPORTED */

#ifdef PNG_UNKNOWN_CHUNKS_SUPPORTED
void PNGAPI
png_set_unknown_chunks(png_structp png_ptr,
   png_infop info_ptr, png_const_unknown_chunkp unknowns, int num_unknowns)
{
   png_unknown_chunkp np;
   int i;

   if (png_ptr == NULL || info_ptr == NULL || num_unknowns == 0)
      return;

   np = (png_unknown_chunkp)png_malloc_warn(png_ptr,
       (png_size_t)(info_ptr->unknown_chunks_num + num_unknowns) *
       png_sizeof(png_unknown_chunk));

   if (np == NULL)
   {
      png_warning(png_ptr,
          "Out of memory while processing unknown chunk");
      return;
   }

   png_memcpy(np, info_ptr->unknown_chunks,
       (png_size_t)info_ptr->unknown_chunks_num *
       png_sizeof(png_unknown_chunk));

   png_free(png_ptr, info_ptr->unknown_chunks);
   info_ptr->unknown_chunks = NULL;

   for (i = 0; i < num_unknowns; i++)
   {
      png_unknown_chunkp to = np + info_ptr->unknown_chunks_num + i;
      png_const_unknown_chunkp from = unknowns + i;

      png_memcpy(to->name, from->name, png_sizeof(from->name));
      to->name[png_sizeof(to->name)-1] = '\0';
      to->size = from->size;

      /* Note our location in the read or write sequence */
      to->location = (png_byte)(png_ptr->mode & 0xff);

      if (from->size == 0)
         to->data=NULL;

      else
      {
         to->data = (png_bytep)png_malloc_warn(png_ptr,
             (png_size_t)from->size);

         if (to->data == NULL)
         {
            png_warning(png_ptr,
                "Out of memory while processing unknown chunk");
            to->size = 0;
         }

         else
            png_memcpy(to->data, from->data, from->size);
      }
   }

   info_ptr->unknown_chunks = np;
   info_ptr->unknown_chunks_num += num_unknowns;
   info_ptr->free_me |= PNG_FREE_UNKN;
}

void PNGAPI
png_set_unknown_chunk_location(png_structp png_ptr, png_infop info_ptr,
    int chunk, int location)
{
   if (png_ptr != NULL && info_ptr != NULL && chunk >= 0 && chunk <
       info_ptr->unknown_chunks_num)
      info_ptr->unknown_chunks[chunk].location = (png_byte)location;
}
#endif


#ifdef PNG_MNG_FEATURES_SUPPORTED
png_uint_32 PNGAPI
png_permit_mng_features (png_structp png_ptr, png_uint_32 mng_features)
{
   png_debug(1, "in png_permit_mng_features");

   if (png_ptr == NULL)
      return (png_uint_32)0;

   png_ptr->mng_features_permitted =
       (png_byte)(mng_features & PNG_ALL_MNG_FEATURES);

   return (png_uint_32)png_ptr->mng_features_permitted;
}
#endif

#ifdef PNG_HANDLE_AS_UNKNOWN_SUPPORTED
void PNGAPI
png_set_keep_unknown_chunks(png_structp png_ptr, int keep, png_const_bytep
    chunk_list, int num_chunks)
{
   png_bytep new_list, p;
   int i, old_num_chunks;
   if (png_ptr == NULL)
      return;

   if (num_chunks == 0)
   {
      if (keep == PNG_HANDLE_CHUNK_ALWAYS || keep == PNG_HANDLE_CHUNK_IF_SAFE)
         png_ptr->flags |= PNG_FLAG_KEEP_UNKNOWN_CHUNKS;

      else
         png_ptr->flags &= ~PNG_FLAG_KEEP_UNKNOWN_CHUNKS;

      if (keep == PNG_HANDLE_CHUNK_ALWAYS)
         png_ptr->flags |= PNG_FLAG_KEEP_UNSAFE_CHUNKS;

      else
         png_ptr->flags &= ~PNG_FLAG_KEEP_UNSAFE_CHUNKS;

      return;
   }

   if (chunk_list == NULL)
      return;

   old_num_chunks = png_ptr->num_chunk_list;
   new_list=(png_bytep)png_malloc(png_ptr,
       (png_size_t)(5*(num_chunks + old_num_chunks)));

   if (png_ptr->chunk_list != NULL)
   {
      png_memcpy(new_list, png_ptr->chunk_list,
          (png_size_t)(5*old_num_chunks));
      png_free(png_ptr, png_ptr->chunk_list);
      png_ptr->chunk_list=NULL;
   }

   png_memcpy(new_list + 5*old_num_chunks, chunk_list,
       (png_size_t)(5*num_chunks));

   for (p = new_list + 5*old_num_chunks + 4, i = 0; i<num_chunks; i++, p += 5)
      *p=(png_byte)keep;

   png_ptr->num_chunk_list = old_num_chunks + num_chunks;
   png_ptr->chunk_list = new_list;
   png_ptr->free_me |= PNG_FREE_LIST;
}
#endif

#ifdef PNG_READ_USER_CHUNKS_SUPPORTED
void PNGAPI
png_set_read_user_chunk_fn(png_structp png_ptr, png_voidp user_chunk_ptr,
    png_user_chunk_ptr read_user_chunk_fn)
{
   png_debug(1, "in png_set_read_user_chunk_fn");

   if (png_ptr == NULL)
      return;

   png_ptr->read_user_chunk_fn = read_user_chunk_fn;
   png_ptr->user_chunk_ptr = user_chunk_ptr;
}
#endif

#ifdef PNG_INFO_IMAGE_SUPPORTED
void PNGAPI
png_set_rows(png_structp png_ptr, png_infop info_ptr, png_bytepp row_pointers)
{
   png_debug1(1, "in %s storage function", "rows");

   if (png_ptr == NULL || info_ptr == NULL)
      return;

   if (info_ptr->row_pointers && (info_ptr->row_pointers != row_pointers))
      png_free_data(png_ptr, info_ptr, PNG_FREE_ROWS, 0);

   info_ptr->row_pointers = row_pointers;

   if (row_pointers)
      info_ptr->valid |= PNG_INFO_IDAT;
}
#endif

void PNGAPI
png_set_compression_buffer_size(png_structp png_ptr, png_size_t size)
{
    if (png_ptr == NULL)
       return;

    png_free(png_ptr, png_ptr->zbuf);

    if (size > ZLIB_IO_MAX)
    {
       png_warning(png_ptr, "Attempt to set buffer size beyond max ignored");
       png_ptr->zbuf_size = ZLIB_IO_MAX;
       size = ZLIB_IO_MAX; /* must fit */
    }

    else
       png_ptr->zbuf_size = (uInt)size;

    png_ptr->zbuf = (png_bytep)png_malloc(png_ptr, size);

    /* The following ensures a relatively safe failure if this gets called while
     * the buffer is actually in use.
     */
    png_ptr->zstream.next_out = png_ptr->zbuf;
    png_ptr->zstream.avail_out = 0;
    png_ptr->zstream.avail_in = 0;
}

void PNGAPI
png_set_invalid(png_structp png_ptr, png_infop info_ptr, int mask)
{
   if (png_ptr && info_ptr)
      info_ptr->valid &= ~mask;
}



#ifdef PNG_SET_USER_LIMITS_SUPPORTED
/* This function was added to libpng 1.2.6 */
void PNGAPI
png_set_user_limits (png_structp png_ptr, png_uint_32 user_width_max,
    png_uint_32 user_height_max)
{
   /* Images with dimensions larger than these limits will be
    * rejected by png_set_IHDR().  To accept any PNG datastream
    * regardless of dimensions, set both limits to 0x7ffffffL.
    */
   if (png_ptr == NULL)
      return;

   png_ptr->user_width_max = user_width_max;
   png_ptr->user_height_max = user_height_max;
}

/* This function was added to libpng 1.4.0 */
void PNGAPI
png_set_chunk_cache_max (png_structp png_ptr,
   png_uint_32 user_chunk_cache_max)
{
    if (png_ptr)
       png_ptr->user_chunk_cache_max = user_chunk_cache_max;
}

/* This function was added to libpng 1.4.1 */
void PNGAPI
png_set_chunk_malloc_max (png_structp png_ptr,
    png_alloc_size_t user_chunk_malloc_max)
{
   if (png_ptr)
      png_ptr->user_chunk_malloc_max = user_chunk_malloc_max;
}
#endif /* ?PNG_SET_USER_LIMITS_SUPPORTED */


#ifdef PNG_BENIGN_ERRORS_SUPPORTED
void PNGAPI
png_set_benign_errors(png_structp png_ptr, int allowed)
{
   png_debug(1, "in png_set_benign_errors");

   if (allowed)
      png_ptr->flags |= PNG_FLAG_BENIGN_ERRORS_WARN;

   else
      png_ptr->flags &= ~PNG_FLAG_BENIGN_ERRORS_WARN;
}
#endif /* PNG_BENIGN_ERRORS_SUPPORTED */
#endif /* PNG_READ_SUPPORTED || PNG_WRITE_SUPPORTED */
