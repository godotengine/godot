
/* pngpread.c - read a png file in push mode
 *
 * Last changed in libpng 1.5.7 [December 15, 2011]
 * Copyright (c) 1998-2011 Glenn Randers-Pehrson
 * (Version 0.96 Copyright (c) 1996, 1997 Andreas Dilger)
 * (Version 0.88 Copyright (c) 1995, 1996 Guy Eric Schalnat, Group 42, Inc.)
 *
 * This code is released under the libpng license.
 * For conditions of distribution and use, see the disclaimer
 * and license in png.h
 */

#include "pngpriv.h"

#ifdef PNG_PROGRESSIVE_READ_SUPPORTED

/* Push model modes */
#define PNG_READ_SIG_MODE   0
#define PNG_READ_CHUNK_MODE 1
#define PNG_READ_IDAT_MODE  2
#define PNG_SKIP_MODE       3
#define PNG_READ_tEXt_MODE  4
#define PNG_READ_zTXt_MODE  5
#define PNG_READ_DONE_MODE  6
#define PNG_READ_iTXt_MODE  7
#define PNG_ERROR_MODE      8

void PNGAPI
png_process_data(png_structp png_ptr, png_infop info_ptr,
    png_bytep buffer, png_size_t buffer_size)
{
   if (png_ptr == NULL || info_ptr == NULL)
      return;

   png_push_restore_buffer(png_ptr, buffer, buffer_size);

   while (png_ptr->buffer_size)
   {
      png_process_some_data(png_ptr, info_ptr);
   }
}

png_size_t PNGAPI
png_process_data_pause(png_structp png_ptr, int save)
{
   if (png_ptr != NULL)
   {
      /* It's easiest for the caller if we do the save, then the caller doesn't
       * have to supply the same data again:
       */
      if (save)
         png_push_save_buffer(png_ptr);
      else
      {
         /* This includes any pending saved bytes: */
         png_size_t remaining = png_ptr->buffer_size;
         png_ptr->buffer_size = 0;

         /* So subtract the saved buffer size, unless all the data
          * is actually 'saved', in which case we just return 0
          */
         if (png_ptr->save_buffer_size < remaining)
            return remaining - png_ptr->save_buffer_size;
      }
   }

   return 0;
}

png_uint_32 PNGAPI
png_process_data_skip(png_structp png_ptr)
{
   png_uint_32 remaining = 0;

   if (png_ptr != NULL && png_ptr->process_mode == PNG_SKIP_MODE &&
      png_ptr->skip_length > 0)
   {
      /* At the end of png_process_data the buffer size must be 0 (see the loop
       * above) so we can detect a broken call here:
       */
      if (png_ptr->buffer_size != 0)
         png_error(png_ptr,
            "png_process_data_skip called inside png_process_data");

      /* If is impossible for there to be a saved buffer at this point -
       * otherwise we could not be in SKIP mode.  This will also happen if
       * png_process_skip is called inside png_process_data (but only very
       * rarely.)
       */
      if (png_ptr->save_buffer_size != 0)
         png_error(png_ptr, "png_process_data_skip called with saved data");

      remaining = png_ptr->skip_length;
      png_ptr->skip_length = 0;
      png_ptr->process_mode = PNG_READ_CHUNK_MODE;
   }

   return remaining;
}

/* What we do with the incoming data depends on what we were previously
 * doing before we ran out of data...
 */
void /* PRIVATE */
png_process_some_data(png_structp png_ptr, png_infop info_ptr)
{
   if (png_ptr == NULL)
      return;

   switch (png_ptr->process_mode)
   {
      case PNG_READ_SIG_MODE:
      {
         png_push_read_sig(png_ptr, info_ptr);
         break;
      }

      case PNG_READ_CHUNK_MODE:
      {
         png_push_read_chunk(png_ptr, info_ptr);
         break;
      }

      case PNG_READ_IDAT_MODE:
      {
         png_push_read_IDAT(png_ptr);
         break;
      }

#ifdef PNG_READ_tEXt_SUPPORTED
      case PNG_READ_tEXt_MODE:
      {
         png_push_read_tEXt(png_ptr, info_ptr);
         break;
      }

#endif
#ifdef PNG_READ_zTXt_SUPPORTED
      case PNG_READ_zTXt_MODE:
      {
         png_push_read_zTXt(png_ptr, info_ptr);
         break;
      }

#endif
#ifdef PNG_READ_iTXt_SUPPORTED
      case PNG_READ_iTXt_MODE:
      {
         png_push_read_iTXt(png_ptr, info_ptr);
         break;
      }

#endif
      case PNG_SKIP_MODE:
      {
         png_push_crc_finish(png_ptr);
         break;
      }

      default:
      {
         png_ptr->buffer_size = 0;
         break;
      }
   }
}

/* Read any remaining signature bytes from the stream and compare them with
 * the correct PNG signature.  It is possible that this routine is called
 * with bytes already read from the signature, either because they have been
 * checked by the calling application, or because of multiple calls to this
 * routine.
 */
void /* PRIVATE */
png_push_read_sig(png_structp png_ptr, png_infop info_ptr)
{
   png_size_t num_checked = png_ptr->sig_bytes,
             num_to_check = 8 - num_checked;

   if (png_ptr->buffer_size < num_to_check)
   {
      num_to_check = png_ptr->buffer_size;
   }

   png_push_fill_buffer(png_ptr, &(info_ptr->signature[num_checked]),
       num_to_check);
   png_ptr->sig_bytes = (png_byte)(png_ptr->sig_bytes + num_to_check);

   if (png_sig_cmp(info_ptr->signature, num_checked, num_to_check))
   {
      if (num_checked < 4 &&
          png_sig_cmp(info_ptr->signature, num_checked, num_to_check - 4))
         png_error(png_ptr, "Not a PNG file");

      else
         png_error(png_ptr, "PNG file corrupted by ASCII conversion");
   }
   else
   {
      if (png_ptr->sig_bytes >= 8)
      {
         png_ptr->process_mode = PNG_READ_CHUNK_MODE;
      }
   }
}

void /* PRIVATE */
png_push_read_chunk(png_structp png_ptr, png_infop info_ptr)
{
   png_uint_32 chunk_name;

   /* First we make sure we have enough data for the 4 byte chunk name
    * and the 4 byte chunk length before proceeding with decoding the
    * chunk data.  To fully decode each of these chunks, we also make
    * sure we have enough data in the buffer for the 4 byte CRC at the
    * end of every chunk (except IDAT, which is handled separately).
    */
   if (!(png_ptr->mode & PNG_HAVE_CHUNK_HEADER))
   {
      png_byte chunk_length[4];
      png_byte chunk_tag[4];

      if (png_ptr->buffer_size < 8)
      {
         png_push_save_buffer(png_ptr);
         return;
      }

      png_push_fill_buffer(png_ptr, chunk_length, 4);
      png_ptr->push_length = png_get_uint_31(png_ptr, chunk_length);
      png_reset_crc(png_ptr);
      png_crc_read(png_ptr, chunk_tag, 4);
      png_ptr->chunk_name = PNG_CHUNK_FROM_STRING(chunk_tag);
      png_check_chunk_name(png_ptr, png_ptr->chunk_name);
      png_ptr->mode |= PNG_HAVE_CHUNK_HEADER;
   }

   chunk_name = png_ptr->chunk_name;

   if (chunk_name == png_IDAT)
   {
      /* This is here above the if/else case statement below because if the
       * unknown handling marks 'IDAT' as unknown then the IDAT handling case is
       * completely skipped.
       *
       * TODO: there must be a better way of doing this.
       */
      if (png_ptr->mode & PNG_AFTER_IDAT)
         png_ptr->mode |= PNG_HAVE_CHUNK_AFTER_IDAT;
   }

   if (chunk_name == png_IHDR)
   {
      if (png_ptr->push_length != 13)
         png_error(png_ptr, "Invalid IHDR length");

      if (png_ptr->push_length + 4 > png_ptr->buffer_size)
      {
         png_push_save_buffer(png_ptr);
         return;
      }

      png_handle_IHDR(png_ptr, info_ptr, png_ptr->push_length);
   }

   else if (chunk_name == png_IEND)
   {
      if (png_ptr->push_length + 4 > png_ptr->buffer_size)
      {
         png_push_save_buffer(png_ptr);
         return;
      }

      png_handle_IEND(png_ptr, info_ptr, png_ptr->push_length);

      png_ptr->process_mode = PNG_READ_DONE_MODE;
      png_push_have_end(png_ptr, info_ptr);
   }

#ifdef PNG_HANDLE_AS_UNKNOWN_SUPPORTED
   else if (png_chunk_unknown_handling(png_ptr, chunk_name))
   {
      if (png_ptr->push_length + 4 > png_ptr->buffer_size)
      {
         png_push_save_buffer(png_ptr);
         return;
      }

      if (chunk_name == png_IDAT)
         png_ptr->mode |= PNG_HAVE_IDAT;

      png_handle_unknown(png_ptr, info_ptr, png_ptr->push_length);

      if (chunk_name == png_PLTE)
         png_ptr->mode |= PNG_HAVE_PLTE;

      else if (chunk_name == png_IDAT)
      {
         if (!(png_ptr->mode & PNG_HAVE_IHDR))
            png_error(png_ptr, "Missing IHDR before IDAT");

         else if (png_ptr->color_type == PNG_COLOR_TYPE_PALETTE &&
             !(png_ptr->mode & PNG_HAVE_PLTE))
            png_error(png_ptr, "Missing PLTE before IDAT");
      }
   }

#endif
   else if (chunk_name == png_PLTE)
   {
      if (png_ptr->push_length + 4 > png_ptr->buffer_size)
      {
         png_push_save_buffer(png_ptr);
         return;
      }
      png_handle_PLTE(png_ptr, info_ptr, png_ptr->push_length);
   }

   else if (chunk_name == png_IDAT)
   {
      /* If we reach an IDAT chunk, this means we have read all of the
       * header chunks, and we can start reading the image (or if this
       * is called after the image has been read - we have an error).
       */

      if (!(png_ptr->mode & PNG_HAVE_IHDR))
         png_error(png_ptr, "Missing IHDR before IDAT");

      else if (png_ptr->color_type == PNG_COLOR_TYPE_PALETTE &&
          !(png_ptr->mode & PNG_HAVE_PLTE))
         png_error(png_ptr, "Missing PLTE before IDAT");

      if (png_ptr->mode & PNG_HAVE_IDAT)
      {
         if (!(png_ptr->mode & PNG_HAVE_CHUNK_AFTER_IDAT))
            if (png_ptr->push_length == 0)
               return;

         if (png_ptr->mode & PNG_AFTER_IDAT)
            png_benign_error(png_ptr, "Too many IDATs found");
      }

      png_ptr->idat_size = png_ptr->push_length;
      png_ptr->mode |= PNG_HAVE_IDAT;
      png_ptr->process_mode = PNG_READ_IDAT_MODE;
      png_push_have_info(png_ptr, info_ptr);
      png_ptr->zstream.avail_out =
          (uInt) PNG_ROWBYTES(png_ptr->pixel_depth,
          png_ptr->iwidth) + 1;
      png_ptr->zstream.next_out = png_ptr->row_buf;
      return;
   }

#ifdef PNG_READ_gAMA_SUPPORTED
   else if (png_ptr->chunk_name == png_gAMA)
   {
      if (png_ptr->push_length + 4 > png_ptr->buffer_size)
      {
         png_push_save_buffer(png_ptr);
         return;
      }

      png_handle_gAMA(png_ptr, info_ptr, png_ptr->push_length);
   }

#endif
#ifdef PNG_READ_sBIT_SUPPORTED
   else if (png_ptr->chunk_name == png_sBIT)
   {
      if (png_ptr->push_length + 4 > png_ptr->buffer_size)
      {
         png_push_save_buffer(png_ptr);
         return;
      }

      png_handle_sBIT(png_ptr, info_ptr, png_ptr->push_length);
   }

#endif
#ifdef PNG_READ_cHRM_SUPPORTED
   else if (png_ptr->chunk_name == png_cHRM)
   {
      if (png_ptr->push_length + 4 > png_ptr->buffer_size)
      {
         png_push_save_buffer(png_ptr);
         return;
      }

      png_handle_cHRM(png_ptr, info_ptr, png_ptr->push_length);
   }

#endif
#ifdef PNG_READ_sRGB_SUPPORTED
   else if (chunk_name == png_sRGB)
   {
      if (png_ptr->push_length + 4 > png_ptr->buffer_size)
      {
         png_push_save_buffer(png_ptr);
         return;
      }

      png_handle_sRGB(png_ptr, info_ptr, png_ptr->push_length);
   }

#endif
#ifdef PNG_READ_iCCP_SUPPORTED
   else if (png_ptr->chunk_name == png_iCCP)
   {
      if (png_ptr->push_length + 4 > png_ptr->buffer_size)
      {
         png_push_save_buffer(png_ptr);
         return;
      }

      png_handle_iCCP(png_ptr, info_ptr, png_ptr->push_length);
   }

#endif
#ifdef PNG_READ_sPLT_SUPPORTED
   else if (chunk_name == png_sPLT)
   {
      if (png_ptr->push_length + 4 > png_ptr->buffer_size)
      {
         png_push_save_buffer(png_ptr);
         return;
      }

      png_handle_sPLT(png_ptr, info_ptr, png_ptr->push_length);
   }

#endif
#ifdef PNG_READ_tRNS_SUPPORTED
   else if (chunk_name == png_tRNS)
   {
      if (png_ptr->push_length + 4 > png_ptr->buffer_size)
      {
         png_push_save_buffer(png_ptr);
         return;
      }

      png_handle_tRNS(png_ptr, info_ptr, png_ptr->push_length);
   }

#endif
#ifdef PNG_READ_bKGD_SUPPORTED
   else if (chunk_name == png_bKGD)
   {
      if (png_ptr->push_length + 4 > png_ptr->buffer_size)
      {
         png_push_save_buffer(png_ptr);
         return;
      }

      png_handle_bKGD(png_ptr, info_ptr, png_ptr->push_length);
   }

#endif
#ifdef PNG_READ_hIST_SUPPORTED
   else if (chunk_name == png_hIST)
   {
      if (png_ptr->push_length + 4 > png_ptr->buffer_size)
      {
         png_push_save_buffer(png_ptr);
         return;
      }

      png_handle_hIST(png_ptr, info_ptr, png_ptr->push_length);
   }

#endif
#ifdef PNG_READ_pHYs_SUPPORTED
   else if (chunk_name == png_pHYs)
   {
      if (png_ptr->push_length + 4 > png_ptr->buffer_size)
      {
         png_push_save_buffer(png_ptr);
         return;
      }

      png_handle_pHYs(png_ptr, info_ptr, png_ptr->push_length);
   }

#endif
#ifdef PNG_READ_oFFs_SUPPORTED
   else if (chunk_name == png_oFFs)
   {
      if (png_ptr->push_length + 4 > png_ptr->buffer_size)
      {
         png_push_save_buffer(png_ptr);
         return;
      }

      png_handle_oFFs(png_ptr, info_ptr, png_ptr->push_length);
   }
#endif

#ifdef PNG_READ_pCAL_SUPPORTED
   else if (chunk_name == png_pCAL)
   {
      if (png_ptr->push_length + 4 > png_ptr->buffer_size)
      {
         png_push_save_buffer(png_ptr);
         return;
      }

      png_handle_pCAL(png_ptr, info_ptr, png_ptr->push_length);
   }

#endif
#ifdef PNG_READ_sCAL_SUPPORTED
   else if (chunk_name == png_sCAL)
   {
      if (png_ptr->push_length + 4 > png_ptr->buffer_size)
      {
         png_push_save_buffer(png_ptr);
         return;
      }

      png_handle_sCAL(png_ptr, info_ptr, png_ptr->push_length);
   }

#endif
#ifdef PNG_READ_tIME_SUPPORTED
   else if (chunk_name == png_tIME)
   {
      if (png_ptr->push_length + 4 > png_ptr->buffer_size)
      {
         png_push_save_buffer(png_ptr);
         return;
      }

      png_handle_tIME(png_ptr, info_ptr, png_ptr->push_length);
   }

#endif
#ifdef PNG_READ_tEXt_SUPPORTED
   else if (chunk_name == png_tEXt)
   {
      if (png_ptr->push_length + 4 > png_ptr->buffer_size)
      {
         png_push_save_buffer(png_ptr);
         return;
      }

      png_push_handle_tEXt(png_ptr, info_ptr, png_ptr->push_length);
   }

#endif
#ifdef PNG_READ_zTXt_SUPPORTED
   else if (chunk_name == png_zTXt)
   {
      if (png_ptr->push_length + 4 > png_ptr->buffer_size)
      {
         png_push_save_buffer(png_ptr);
         return;
      }

      png_push_handle_zTXt(png_ptr, info_ptr, png_ptr->push_length);
   }

#endif
#ifdef PNG_READ_iTXt_SUPPORTED
   else if (chunk_name == png_iTXt)
   {
      if (png_ptr->push_length + 4 > png_ptr->buffer_size)
      {
         png_push_save_buffer(png_ptr);
         return;
      }

      png_push_handle_iTXt(png_ptr, info_ptr, png_ptr->push_length);
   }

#endif
   else
   {
      if (png_ptr->push_length + 4 > png_ptr->buffer_size)
      {
         png_push_save_buffer(png_ptr);
         return;
      }
      png_push_handle_unknown(png_ptr, info_ptr, png_ptr->push_length);
   }

   png_ptr->mode &= ~PNG_HAVE_CHUNK_HEADER;
}

void /* PRIVATE */
png_push_crc_skip(png_structp png_ptr, png_uint_32 skip)
{
   png_ptr->process_mode = PNG_SKIP_MODE;
   png_ptr->skip_length = skip;
}

void /* PRIVATE */
png_push_crc_finish(png_structp png_ptr)
{
   if (png_ptr->skip_length && png_ptr->save_buffer_size)
   {
      png_size_t save_size = png_ptr->save_buffer_size;
      png_uint_32 skip_length = png_ptr->skip_length;

      /* We want the smaller of 'skip_length' and 'save_buffer_size', but
       * they are of different types and we don't know which variable has the
       * fewest bits.  Carefully select the smaller and cast it to the type of
       * the larger - this cannot overflow.  Do not cast in the following test
       * - it will break on either 16 or 64 bit platforms.
       */
      if (skip_length < save_size)
         save_size = (png_size_t)skip_length;

      else
         skip_length = (png_uint_32)save_size;

      png_calculate_crc(png_ptr, png_ptr->save_buffer_ptr, save_size);

      png_ptr->skip_length -= skip_length;
      png_ptr->buffer_size -= save_size;
      png_ptr->save_buffer_size -= save_size;
      png_ptr->save_buffer_ptr += save_size;
   }
   if (png_ptr->skip_length && png_ptr->current_buffer_size)
   {
      png_size_t save_size = png_ptr->current_buffer_size;
      png_uint_32 skip_length = png_ptr->skip_length;

      /* We want the smaller of 'skip_length' and 'current_buffer_size', here,
       * the same problem exists as above and the same solution.
       */
      if (skip_length < save_size)
         save_size = (png_size_t)skip_length;

      else
         skip_length = (png_uint_32)save_size;

      png_calculate_crc(png_ptr, png_ptr->current_buffer_ptr, save_size);

      png_ptr->skip_length -= skip_length;
      png_ptr->buffer_size -= save_size;
      png_ptr->current_buffer_size -= save_size;
      png_ptr->current_buffer_ptr += save_size;
   }
   if (!png_ptr->skip_length)
   {
      if (png_ptr->buffer_size < 4)
      {
         png_push_save_buffer(png_ptr);
         return;
      }

      png_crc_finish(png_ptr, 0);
      png_ptr->process_mode = PNG_READ_CHUNK_MODE;
   }
}

void PNGCBAPI
png_push_fill_buffer(png_structp png_ptr, png_bytep buffer, png_size_t length)
{
   png_bytep ptr;

   if (png_ptr == NULL)
      return;

   ptr = buffer;
   if (png_ptr->save_buffer_size)
   {
      png_size_t save_size;

      if (length < png_ptr->save_buffer_size)
         save_size = length;

      else
         save_size = png_ptr->save_buffer_size;

      png_memcpy(ptr, png_ptr->save_buffer_ptr, save_size);
      length -= save_size;
      ptr += save_size;
      png_ptr->buffer_size -= save_size;
      png_ptr->save_buffer_size -= save_size;
      png_ptr->save_buffer_ptr += save_size;
   }
   if (length && png_ptr->current_buffer_size)
   {
      png_size_t save_size;

      if (length < png_ptr->current_buffer_size)
         save_size = length;

      else
         save_size = png_ptr->current_buffer_size;

      png_memcpy(ptr, png_ptr->current_buffer_ptr, save_size);
      png_ptr->buffer_size -= save_size;
      png_ptr->current_buffer_size -= save_size;
      png_ptr->current_buffer_ptr += save_size;
   }
}

void /* PRIVATE */
png_push_save_buffer(png_structp png_ptr)
{
   if (png_ptr->save_buffer_size)
   {
      if (png_ptr->save_buffer_ptr != png_ptr->save_buffer)
      {
         png_size_t i, istop;
         png_bytep sp;
         png_bytep dp;

         istop = png_ptr->save_buffer_size;
         for (i = 0, sp = png_ptr->save_buffer_ptr, dp = png_ptr->save_buffer;
             i < istop; i++, sp++, dp++)
         {
            *dp = *sp;
         }
      }
   }
   if (png_ptr->save_buffer_size + png_ptr->current_buffer_size >
       png_ptr->save_buffer_max)
   {
      png_size_t new_max;
      png_bytep old_buffer;

      if (png_ptr->save_buffer_size > PNG_SIZE_MAX -
          (png_ptr->current_buffer_size + 256))
      {
         png_error(png_ptr, "Potential overflow of save_buffer");
      }

      new_max = png_ptr->save_buffer_size + png_ptr->current_buffer_size + 256;
      old_buffer = png_ptr->save_buffer;
      png_ptr->save_buffer = (png_bytep)png_malloc_warn(png_ptr,
          (png_size_t)new_max);

      if (png_ptr->save_buffer == NULL)
      {
         png_free(png_ptr, old_buffer);
         png_error(png_ptr, "Insufficient memory for save_buffer");
      }

      png_memcpy(png_ptr->save_buffer, old_buffer, png_ptr->save_buffer_size);
      png_free(png_ptr, old_buffer);
      png_ptr->save_buffer_max = new_max;
   }
   if (png_ptr->current_buffer_size)
   {
      png_memcpy(png_ptr->save_buffer + png_ptr->save_buffer_size,
         png_ptr->current_buffer_ptr, png_ptr->current_buffer_size);
      png_ptr->save_buffer_size += png_ptr->current_buffer_size;
      png_ptr->current_buffer_size = 0;
   }
   png_ptr->save_buffer_ptr = png_ptr->save_buffer;
   png_ptr->buffer_size = 0;
}

void /* PRIVATE */
png_push_restore_buffer(png_structp png_ptr, png_bytep buffer,
   png_size_t buffer_length)
{
   png_ptr->current_buffer = buffer;
   png_ptr->current_buffer_size = buffer_length;
   png_ptr->buffer_size = buffer_length + png_ptr->save_buffer_size;
   png_ptr->current_buffer_ptr = png_ptr->current_buffer;
}

void /* PRIVATE */
png_push_read_IDAT(png_structp png_ptr)
{
   if (!(png_ptr->mode & PNG_HAVE_CHUNK_HEADER))
   {
      png_byte chunk_length[4];
      png_byte chunk_tag[4];

      /* TODO: this code can be commoned up with the same code in push_read */
      if (png_ptr->buffer_size < 8)
      {
         png_push_save_buffer(png_ptr);
         return;
      }

      png_push_fill_buffer(png_ptr, chunk_length, 4);
      png_ptr->push_length = png_get_uint_31(png_ptr, chunk_length);
      png_reset_crc(png_ptr);
      png_crc_read(png_ptr, chunk_tag, 4);
      png_ptr->chunk_name = PNG_CHUNK_FROM_STRING(chunk_tag);
      png_ptr->mode |= PNG_HAVE_CHUNK_HEADER;

      if (png_ptr->chunk_name != png_IDAT)
      {
         png_ptr->process_mode = PNG_READ_CHUNK_MODE;

         if (!(png_ptr->flags & PNG_FLAG_ZLIB_FINISHED))
            png_error(png_ptr, "Not enough compressed data");

         return;
      }

      png_ptr->idat_size = png_ptr->push_length;
   }

   if (png_ptr->idat_size && png_ptr->save_buffer_size)
   {
      png_size_t save_size = png_ptr->save_buffer_size;
      png_uint_32 idat_size = png_ptr->idat_size;

      /* We want the smaller of 'idat_size' and 'current_buffer_size', but they
       * are of different types and we don't know which variable has the fewest
       * bits.  Carefully select the smaller and cast it to the type of the
       * larger - this cannot overflow.  Do not cast in the following test - it
       * will break on either 16 or 64 bit platforms.
       */
      if (idat_size < save_size)
         save_size = (png_size_t)idat_size;

      else
         idat_size = (png_uint_32)save_size;

      png_calculate_crc(png_ptr, png_ptr->save_buffer_ptr, save_size);

      png_process_IDAT_data(png_ptr, png_ptr->save_buffer_ptr, save_size);

      png_ptr->idat_size -= idat_size;
      png_ptr->buffer_size -= save_size;
      png_ptr->save_buffer_size -= save_size;
      png_ptr->save_buffer_ptr += save_size;
   }

   if (png_ptr->idat_size && png_ptr->current_buffer_size)
   {
      png_size_t save_size = png_ptr->current_buffer_size;
      png_uint_32 idat_size = png_ptr->idat_size;

      /* We want the smaller of 'idat_size' and 'current_buffer_size', but they
       * are of different types and we don't know which variable has the fewest
       * bits.  Carefully select the smaller and cast it to the type of the
       * larger - this cannot overflow.
       */
      if (idat_size < save_size)
         save_size = (png_size_t)idat_size;

      else
         idat_size = (png_uint_32)save_size;

      png_calculate_crc(png_ptr, png_ptr->current_buffer_ptr, save_size);

      png_process_IDAT_data(png_ptr, png_ptr->current_buffer_ptr, save_size);

      png_ptr->idat_size -= idat_size;
      png_ptr->buffer_size -= save_size;
      png_ptr->current_buffer_size -= save_size;
      png_ptr->current_buffer_ptr += save_size;
   }
   if (!png_ptr->idat_size)
   {
      if (png_ptr->buffer_size < 4)
      {
         png_push_save_buffer(png_ptr);
         return;
      }

      png_crc_finish(png_ptr, 0);
      png_ptr->mode &= ~PNG_HAVE_CHUNK_HEADER;
      png_ptr->mode |= PNG_AFTER_IDAT;
   }
}

void /* PRIVATE */
png_process_IDAT_data(png_structp png_ptr, png_bytep buffer,
   png_size_t buffer_length)
{
   /* The caller checks for a non-zero buffer length. */
   if (!(buffer_length > 0) || buffer == NULL)
      png_error(png_ptr, "No IDAT data (internal error)");

   /* This routine must process all the data it has been given
    * before returning, calling the row callback as required to
    * handle the uncompressed results.
    */
   png_ptr->zstream.next_in = buffer;
   png_ptr->zstream.avail_in = (uInt)buffer_length;

   /* Keep going until the decompressed data is all processed
    * or the stream marked as finished.
    */
   while (png_ptr->zstream.avail_in > 0 &&
          !(png_ptr->flags & PNG_FLAG_ZLIB_FINISHED))
   {
      int ret;

      /* We have data for zlib, but we must check that zlib
       * has someplace to put the results.  It doesn't matter
       * if we don't expect any results -- it may be the input
       * data is just the LZ end code.
       */
      if (!(png_ptr->zstream.avail_out > 0))
      {
         png_ptr->zstream.avail_out =
             (uInt) PNG_ROWBYTES(png_ptr->pixel_depth,
             png_ptr->iwidth) + 1;

         png_ptr->zstream.next_out = png_ptr->row_buf;
      }

      /* Using Z_SYNC_FLUSH here means that an unterminated
       * LZ stream (a stream with a missing end code) can still
       * be handled, otherwise (Z_NO_FLUSH) a future zlib
       * implementation might defer output and therefore
       * change the current behavior (see comments in inflate.c
       * for why this doesn't happen at present with zlib 1.2.5).
       */
      ret = inflate(&png_ptr->zstream, Z_SYNC_FLUSH);

      /* Check for any failure before proceeding. */
      if (ret != Z_OK && ret != Z_STREAM_END)
      {
         /* Terminate the decompression. */
         png_ptr->flags |= PNG_FLAG_ZLIB_FINISHED;

         /* This may be a truncated stream (missing or
          * damaged end code).  Treat that as a warning.
          */
         if (png_ptr->row_number >= png_ptr->num_rows ||
             png_ptr->pass > 6)
            png_warning(png_ptr, "Truncated compressed data in IDAT");

         else
            png_error(png_ptr, "Decompression error in IDAT");

         /* Skip the check on unprocessed input */
         return;
      }

      /* Did inflate output any data? */
      if (png_ptr->zstream.next_out != png_ptr->row_buf)
      {
         /* Is this unexpected data after the last row?
          * If it is, artificially terminate the LZ output
          * here.
          */
         if (png_ptr->row_number >= png_ptr->num_rows ||
             png_ptr->pass > 6)
         {
            /* Extra data. */
            png_warning(png_ptr, "Extra compressed data in IDAT");
            png_ptr->flags |= PNG_FLAG_ZLIB_FINISHED;

            /* Do no more processing; skip the unprocessed
             * input check below.
             */
            return;
         }

         /* Do we have a complete row? */
         if (png_ptr->zstream.avail_out == 0)
            png_push_process_row(png_ptr);
      }

      /* And check for the end of the stream. */
      if (ret == Z_STREAM_END)
         png_ptr->flags |= PNG_FLAG_ZLIB_FINISHED;
   }

   /* All the data should have been processed, if anything
    * is left at this point we have bytes of IDAT data
    * after the zlib end code.
    */
   if (png_ptr->zstream.avail_in > 0)
      png_warning(png_ptr, "Extra compression data in IDAT");
}

void /* PRIVATE */
png_push_process_row(png_structp png_ptr)
{
   /* 1.5.6: row_info moved out of png_struct to a local here. */
   png_row_info row_info;

   row_info.width = png_ptr->iwidth; /* NOTE: width of current interlaced row */
   row_info.color_type = png_ptr->color_type;
   row_info.bit_depth = png_ptr->bit_depth;
   row_info.channels = png_ptr->channels;
   row_info.pixel_depth = png_ptr->pixel_depth;
   row_info.rowbytes = PNG_ROWBYTES(row_info.pixel_depth, row_info.width);

   if (png_ptr->row_buf[0] > PNG_FILTER_VALUE_NONE)
   {
      if (png_ptr->row_buf[0] < PNG_FILTER_VALUE_LAST)
         png_read_filter_row(png_ptr, &row_info, png_ptr->row_buf + 1,
            png_ptr->prev_row + 1, png_ptr->row_buf[0]);
      else
         png_error(png_ptr, "bad adaptive filter value");
   }

   /* libpng 1.5.6: the following line was copying png_ptr->rowbytes before
    * 1.5.6, while the buffer really is this big in current versions of libpng
    * it may not be in the future, so this was changed just to copy the
    * interlaced row count:
    */
   png_memcpy(png_ptr->prev_row, png_ptr->row_buf, row_info.rowbytes + 1);

#ifdef PNG_READ_TRANSFORMS_SUPPORTED
   if (png_ptr->transformations)
      png_do_read_transformations(png_ptr, &row_info);
#endif

   /* The transformed pixel depth should match the depth now in row_info. */
   if (png_ptr->transformed_pixel_depth == 0)
   {
      png_ptr->transformed_pixel_depth = row_info.pixel_depth;
      if (row_info.pixel_depth > png_ptr->maximum_pixel_depth)
         png_error(png_ptr, "progressive row overflow");
   }

   else if (png_ptr->transformed_pixel_depth != row_info.pixel_depth)
      png_error(png_ptr, "internal progressive row size calculation error");


#ifdef PNG_READ_INTERLACING_SUPPORTED
   /* Blow up interlaced rows to full size */
   if (png_ptr->interlaced && (png_ptr->transformations & PNG_INTERLACE))
   {
      if (png_ptr->pass < 6)
         png_do_read_interlace(&row_info, png_ptr->row_buf + 1, png_ptr->pass,
            png_ptr->transformations);

    switch (png_ptr->pass)
    {
         case 0:
         {
            int i;
            for (i = 0; i < 8 && png_ptr->pass == 0; i++)
            {
               png_push_have_row(png_ptr, png_ptr->row_buf + 1);
               png_read_push_finish_row(png_ptr); /* Updates png_ptr->pass */
            }

            if (png_ptr->pass == 2) /* Pass 1 might be empty */
            {
               for (i = 0; i < 4 && png_ptr->pass == 2; i++)
               {
                  png_push_have_row(png_ptr, NULL);
                  png_read_push_finish_row(png_ptr);
               }
            }

            if (png_ptr->pass == 4 && png_ptr->height <= 4)
            {
               for (i = 0; i < 2 && png_ptr->pass == 4; i++)
               {
                  png_push_have_row(png_ptr, NULL);
                  png_read_push_finish_row(png_ptr);
               }
            }

            if (png_ptr->pass == 6 && png_ptr->height <= 4)
            {
                png_push_have_row(png_ptr, NULL);
                png_read_push_finish_row(png_ptr);
            }

            break;
         }

         case 1:
         {
            int i;
            for (i = 0; i < 8 && png_ptr->pass == 1; i++)
            {
               png_push_have_row(png_ptr, png_ptr->row_buf + 1);
               png_read_push_finish_row(png_ptr);
            }

            if (png_ptr->pass == 2) /* Skip top 4 generated rows */
            {
               for (i = 0; i < 4 && png_ptr->pass == 2; i++)
               {
                  png_push_have_row(png_ptr, NULL);
                  png_read_push_finish_row(png_ptr);
               }
            }

            break;
         }

         case 2:
         {
            int i;

            for (i = 0; i < 4 && png_ptr->pass == 2; i++)
            {
               png_push_have_row(png_ptr, png_ptr->row_buf + 1);
               png_read_push_finish_row(png_ptr);
            }

            for (i = 0; i < 4 && png_ptr->pass == 2; i++)
            {
               png_push_have_row(png_ptr, NULL);
               png_read_push_finish_row(png_ptr);
            }

            if (png_ptr->pass == 4) /* Pass 3 might be empty */
            {
               for (i = 0; i < 2 && png_ptr->pass == 4; i++)
               {
                  png_push_have_row(png_ptr, NULL);
                  png_read_push_finish_row(png_ptr);
               }
            }

            break;
         }

         case 3:
         {
            int i;

            for (i = 0; i < 4 && png_ptr->pass == 3; i++)
            {
               png_push_have_row(png_ptr, png_ptr->row_buf + 1);
               png_read_push_finish_row(png_ptr);
            }

            if (png_ptr->pass == 4) /* Skip top two generated rows */
            {
               for (i = 0; i < 2 && png_ptr->pass == 4; i++)
               {
                  png_push_have_row(png_ptr, NULL);
                  png_read_push_finish_row(png_ptr);
               }
            }

            break;
         }

         case 4:
         {
            int i;

            for (i = 0; i < 2 && png_ptr->pass == 4; i++)
            {
               png_push_have_row(png_ptr, png_ptr->row_buf + 1);
               png_read_push_finish_row(png_ptr);
            }

            for (i = 0; i < 2 && png_ptr->pass == 4; i++)
            {
               png_push_have_row(png_ptr, NULL);
               png_read_push_finish_row(png_ptr);
            }

            if (png_ptr->pass == 6) /* Pass 5 might be empty */
            {
               png_push_have_row(png_ptr, NULL);
               png_read_push_finish_row(png_ptr);
            }

            break;
         }

         case 5:
         {
            int i;

            for (i = 0; i < 2 && png_ptr->pass == 5; i++)
            {
               png_push_have_row(png_ptr, png_ptr->row_buf + 1);
               png_read_push_finish_row(png_ptr);
            }

            if (png_ptr->pass == 6) /* Skip top generated row */
            {
               png_push_have_row(png_ptr, NULL);
               png_read_push_finish_row(png_ptr);
            }

            break;
         }

         default:
         case 6:
         {
            png_push_have_row(png_ptr, png_ptr->row_buf + 1);
            png_read_push_finish_row(png_ptr);

            if (png_ptr->pass != 6)
               break;

            png_push_have_row(png_ptr, NULL);
            png_read_push_finish_row(png_ptr);
         }
      }
   }
   else
#endif
   {
      png_push_have_row(png_ptr, png_ptr->row_buf + 1);
      png_read_push_finish_row(png_ptr);
   }
}

void /* PRIVATE */
png_read_push_finish_row(png_structp png_ptr)
{
   /* Arrays to facilitate easy interlacing - use pass (0 - 6) as index */

   /* Start of interlace block */
   static PNG_CONST png_byte FARDATA png_pass_start[] = {0, 4, 0, 2, 0, 1, 0};

   /* Offset to next interlace block */
   static PNG_CONST png_byte FARDATA png_pass_inc[] = {8, 8, 4, 4, 2, 2, 1};

   /* Start of interlace block in the y direction */
   static PNG_CONST png_byte FARDATA png_pass_ystart[] = {0, 0, 4, 0, 2, 0, 1};

   /* Offset to next interlace block in the y direction */
   static PNG_CONST png_byte FARDATA png_pass_yinc[] = {8, 8, 8, 4, 4, 2, 2};

   /* Height of interlace block.  This is not currently used - if you need
    * it, uncomment it here and in png.h
   static PNG_CONST png_byte FARDATA png_pass_height[] = {8, 8, 4, 4, 2, 2, 1};
   */

   png_ptr->row_number++;
   if (png_ptr->row_number < png_ptr->num_rows)
      return;

#ifdef PNG_READ_INTERLACING_SUPPORTED
   if (png_ptr->interlaced)
   {
      png_ptr->row_number = 0;
      png_memset(png_ptr->prev_row, 0, png_ptr->rowbytes + 1);

      do
      {
         png_ptr->pass++;
         if ((png_ptr->pass == 1 && png_ptr->width < 5) ||
             (png_ptr->pass == 3 && png_ptr->width < 3) ||
             (png_ptr->pass == 5 && png_ptr->width < 2))
            png_ptr->pass++;

         if (png_ptr->pass > 7)
            png_ptr->pass--;

         if (png_ptr->pass >= 7)
            break;

         png_ptr->iwidth = (png_ptr->width +
             png_pass_inc[png_ptr->pass] - 1 -
             png_pass_start[png_ptr->pass]) /
             png_pass_inc[png_ptr->pass];

         if (png_ptr->transformations & PNG_INTERLACE)
            break;

         png_ptr->num_rows = (png_ptr->height +
             png_pass_yinc[png_ptr->pass] - 1 -
             png_pass_ystart[png_ptr->pass]) /
             png_pass_yinc[png_ptr->pass];

      } while (png_ptr->iwidth == 0 || png_ptr->num_rows == 0);
   }
#endif /* PNG_READ_INTERLACING_SUPPORTED */
}

#ifdef PNG_READ_tEXt_SUPPORTED
void /* PRIVATE */
png_push_handle_tEXt(png_structp png_ptr, png_infop info_ptr, png_uint_32
    length)
{
   if (!(png_ptr->mode & PNG_HAVE_IHDR) || (png_ptr->mode & PNG_HAVE_IEND))
      {
         PNG_UNUSED(info_ptr) /* To quiet some compiler warnings */
         png_error(png_ptr, "Out of place tEXt");
         /* NOT REACHED */
      }

#ifdef PNG_MAX_MALLOC_64K
   png_ptr->skip_length = 0;  /* This may not be necessary */

   if (length > (png_uint_32)65535L) /* Can't hold entire string in memory */
   {
      png_warning(png_ptr, "tEXt chunk too large to fit in memory");
      png_ptr->skip_length = length - (png_uint_32)65535L;
      length = (png_uint_32)65535L;
   }
#endif

   png_ptr->current_text = (png_charp)png_malloc(png_ptr,
       (png_size_t)(length + 1));
   png_ptr->current_text[length] = '\0';
   png_ptr->current_text_ptr = png_ptr->current_text;
   png_ptr->current_text_size = (png_size_t)length;
   png_ptr->current_text_left = (png_size_t)length;
   png_ptr->process_mode = PNG_READ_tEXt_MODE;
}

void /* PRIVATE */
png_push_read_tEXt(png_structp png_ptr, png_infop info_ptr)
{
   if (png_ptr->buffer_size && png_ptr->current_text_left)
   {
      png_size_t text_size;

      if (png_ptr->buffer_size < png_ptr->current_text_left)
         text_size = png_ptr->buffer_size;

      else
         text_size = png_ptr->current_text_left;

      png_crc_read(png_ptr, (png_bytep)png_ptr->current_text_ptr, text_size);
      png_ptr->current_text_left -= text_size;
      png_ptr->current_text_ptr += text_size;
   }
   if (!(png_ptr->current_text_left))
   {
      png_textp text_ptr;
      png_charp text;
      png_charp key;
      int ret;

      if (png_ptr->buffer_size < 4)
      {
         png_push_save_buffer(png_ptr);
         return;
      }

      png_push_crc_finish(png_ptr);

#ifdef PNG_MAX_MALLOC_64K
      if (png_ptr->skip_length)
         return;
#endif

      key = png_ptr->current_text;

      for (text = key; *text; text++)
         /* Empty loop */ ;

      if (text < key + png_ptr->current_text_size)
         text++;

      text_ptr = (png_textp)png_malloc(png_ptr, png_sizeof(png_text));
      text_ptr->compression = PNG_TEXT_COMPRESSION_NONE;
      text_ptr->key = key;
      text_ptr->itxt_length = 0;
      text_ptr->lang = NULL;
      text_ptr->lang_key = NULL;
      text_ptr->text = text;

      ret = png_set_text_2(png_ptr, info_ptr, text_ptr, 1);

      png_free(png_ptr, key);
      png_free(png_ptr, text_ptr);
      png_ptr->current_text = NULL;

      if (ret)
         png_warning(png_ptr, "Insufficient memory to store text chunk");
   }
}
#endif

#ifdef PNG_READ_zTXt_SUPPORTED
void /* PRIVATE */
png_push_handle_zTXt(png_structp png_ptr, png_infop info_ptr, png_uint_32
   length)
{
   if (!(png_ptr->mode & PNG_HAVE_IHDR) || (png_ptr->mode & PNG_HAVE_IEND))
   {
      PNG_UNUSED(info_ptr) /* To quiet some compiler warnings */
      png_error(png_ptr, "Out of place zTXt");
      /* NOT REACHED */
   }

#ifdef PNG_MAX_MALLOC_64K
   /* We can't handle zTXt chunks > 64K, since we don't have enough space
    * to be able to store the uncompressed data.  Actually, the threshold
    * is probably around 32K, but it isn't as definite as 64K is.
    */
   if (length > (png_uint_32)65535L)
   {
      png_warning(png_ptr, "zTXt chunk too large to fit in memory");
      png_push_crc_skip(png_ptr, length);
      return;
   }
#endif

   png_ptr->current_text = (png_charp)png_malloc(png_ptr,
       (png_size_t)(length + 1));
   png_ptr->current_text[length] = '\0';
   png_ptr->current_text_ptr = png_ptr->current_text;
   png_ptr->current_text_size = (png_size_t)length;
   png_ptr->current_text_left = (png_size_t)length;
   png_ptr->process_mode = PNG_READ_zTXt_MODE;
}

void /* PRIVATE */
png_push_read_zTXt(png_structp png_ptr, png_infop info_ptr)
{
   if (png_ptr->buffer_size && png_ptr->current_text_left)
   {
      png_size_t text_size;

      if (png_ptr->buffer_size < (png_uint_32)png_ptr->current_text_left)
         text_size = png_ptr->buffer_size;

      else
         text_size = png_ptr->current_text_left;

      png_crc_read(png_ptr, (png_bytep)png_ptr->current_text_ptr, text_size);
      png_ptr->current_text_left -= text_size;
      png_ptr->current_text_ptr += text_size;
   }
   if (!(png_ptr->current_text_left))
   {
      png_textp text_ptr;
      png_charp text;
      png_charp key;
      int ret;
      png_size_t text_size, key_size;

      if (png_ptr->buffer_size < 4)
      {
         png_push_save_buffer(png_ptr);
         return;
      }

      png_push_crc_finish(png_ptr);

      key = png_ptr->current_text;

      for (text = key; *text; text++)
         /* Empty loop */ ;

      /* zTXt can't have zero text */
      if (text >= key + png_ptr->current_text_size)
      {
         png_ptr->current_text = NULL;
         png_free(png_ptr, key);
         return;
      }

      text++;

      if (*text != PNG_TEXT_COMPRESSION_zTXt) /* Check compression byte */
      {
         png_ptr->current_text = NULL;
         png_free(png_ptr, key);
         return;
      }

      text++;

      png_ptr->zstream.next_in = (png_bytep)text;
      png_ptr->zstream.avail_in = (uInt)(png_ptr->current_text_size -
          (text - key));
      png_ptr->zstream.next_out = png_ptr->zbuf;
      png_ptr->zstream.avail_out = (uInt)png_ptr->zbuf_size;

      key_size = text - key;
      text_size = 0;
      text = NULL;
      ret = Z_STREAM_END;

      while (png_ptr->zstream.avail_in)
      {
         ret = inflate(&png_ptr->zstream, Z_PARTIAL_FLUSH);
         if (ret != Z_OK && ret != Z_STREAM_END)
         {
            inflateReset(&png_ptr->zstream);
            png_ptr->zstream.avail_in = 0;
            png_ptr->current_text = NULL;
            png_free(png_ptr, key);
            png_free(png_ptr, text);
            return;
         }

         if (!(png_ptr->zstream.avail_out) || ret == Z_STREAM_END)
         {
            if (text == NULL)
            {
               text = (png_charp)png_malloc(png_ptr,
                   (png_ptr->zbuf_size
                   - png_ptr->zstream.avail_out + key_size + 1));

               png_memcpy(text + key_size, png_ptr->zbuf,
                   png_ptr->zbuf_size - png_ptr->zstream.avail_out);

               png_memcpy(text, key, key_size);

               text_size = key_size + png_ptr->zbuf_size -
                   png_ptr->zstream.avail_out;

               *(text + text_size) = '\0';
            }

            else
            {
               png_charp tmp;

               tmp = text;
               text = (png_charp)png_malloc(png_ptr, text_size +
                   (png_ptr->zbuf_size
                   - png_ptr->zstream.avail_out + 1));

               png_memcpy(text, tmp, text_size);
               png_free(png_ptr, tmp);

               png_memcpy(text + text_size, png_ptr->zbuf,
                   png_ptr->zbuf_size - png_ptr->zstream.avail_out);

               text_size += png_ptr->zbuf_size - png_ptr->zstream.avail_out;
               *(text + text_size) = '\0';
            }

            if (ret != Z_STREAM_END)
            {
               png_ptr->zstream.next_out = png_ptr->zbuf;
               png_ptr->zstream.avail_out = (uInt)png_ptr->zbuf_size;
            }
         }
         else
         {
            break;
         }

         if (ret == Z_STREAM_END)
            break;
      }

      inflateReset(&png_ptr->zstream);
      png_ptr->zstream.avail_in = 0;

      if (ret != Z_STREAM_END)
      {
         png_ptr->current_text = NULL;
         png_free(png_ptr, key);
         png_free(png_ptr, text);
         return;
      }

      png_ptr->current_text = NULL;
      png_free(png_ptr, key);
      key = text;
      text += key_size;

      text_ptr = (png_textp)png_malloc(png_ptr,
          png_sizeof(png_text));
      text_ptr->compression = PNG_TEXT_COMPRESSION_zTXt;
      text_ptr->key = key;
      text_ptr->itxt_length = 0;
      text_ptr->lang = NULL;
      text_ptr->lang_key = NULL;
      text_ptr->text = text;

      ret = png_set_text_2(png_ptr, info_ptr, text_ptr, 1);

      png_free(png_ptr, key);
      png_free(png_ptr, text_ptr);

      if (ret)
         png_warning(png_ptr, "Insufficient memory to store text chunk");
   }
}
#endif

#ifdef PNG_READ_iTXt_SUPPORTED
void /* PRIVATE */
png_push_handle_iTXt(png_structp png_ptr, png_infop info_ptr, png_uint_32
    length)
{
   if (!(png_ptr->mode & PNG_HAVE_IHDR) || (png_ptr->mode & PNG_HAVE_IEND))
   {
      PNG_UNUSED(info_ptr) /* To quiet some compiler warnings */
      png_error(png_ptr, "Out of place iTXt");
      /* NOT REACHED */
   }

#ifdef PNG_MAX_MALLOC_64K
   png_ptr->skip_length = 0;  /* This may not be necessary */

   if (length > (png_uint_32)65535L) /* Can't hold entire string in memory */
   {
      png_warning(png_ptr, "iTXt chunk too large to fit in memory");
      png_ptr->skip_length = length - (png_uint_32)65535L;
      length = (png_uint_32)65535L;
   }
#endif

   png_ptr->current_text = (png_charp)png_malloc(png_ptr,
       (png_size_t)(length + 1));
   png_ptr->current_text[length] = '\0';
   png_ptr->current_text_ptr = png_ptr->current_text;
   png_ptr->current_text_size = (png_size_t)length;
   png_ptr->current_text_left = (png_size_t)length;
   png_ptr->process_mode = PNG_READ_iTXt_MODE;
}

void /* PRIVATE */
png_push_read_iTXt(png_structp png_ptr, png_infop info_ptr)
{

   if (png_ptr->buffer_size && png_ptr->current_text_left)
   {
      png_size_t text_size;

      if (png_ptr->buffer_size < png_ptr->current_text_left)
         text_size = png_ptr->buffer_size;

      else
         text_size = png_ptr->current_text_left;

      png_crc_read(png_ptr, (png_bytep)png_ptr->current_text_ptr, text_size);
      png_ptr->current_text_left -= text_size;
      png_ptr->current_text_ptr += text_size;
   }

   if (!(png_ptr->current_text_left))
   {
      png_textp text_ptr;
      png_charp key;
      int comp_flag;
      png_charp lang;
      png_charp lang_key;
      png_charp text;
      int ret;

      if (png_ptr->buffer_size < 4)
      {
         png_push_save_buffer(png_ptr);
         return;
      }

      png_push_crc_finish(png_ptr);

#ifdef PNG_MAX_MALLOC_64K
      if (png_ptr->skip_length)
         return;
#endif

      key = png_ptr->current_text;

      for (lang = key; *lang; lang++)
         /* Empty loop */ ;

      if (lang < key + png_ptr->current_text_size - 3)
         lang++;

      comp_flag = *lang++;
      lang++;     /* Skip comp_type, always zero */

      for (lang_key = lang; *lang_key; lang_key++)
         /* Empty loop */ ;

      lang_key++;        /* Skip NUL separator */

      text=lang_key;

      if (lang_key < key + png_ptr->current_text_size - 1)
      {
         for (; *text; text++)
            /* Empty loop */ ;
      }

      if (text < key + png_ptr->current_text_size)
         text++;

      text_ptr = (png_textp)png_malloc(png_ptr,
          png_sizeof(png_text));

      text_ptr->compression = comp_flag + 2;
      text_ptr->key = key;
      text_ptr->lang = lang;
      text_ptr->lang_key = lang_key;
      text_ptr->text = text;
      text_ptr->text_length = 0;
      text_ptr->itxt_length = png_strlen(text);

      ret = png_set_text_2(png_ptr, info_ptr, text_ptr, 1);

      png_ptr->current_text = NULL;

      png_free(png_ptr, text_ptr);
      if (ret)
         png_warning(png_ptr, "Insufficient memory to store iTXt chunk");
   }
}
#endif

/* This function is called when we haven't found a handler for this
 * chunk.  If there isn't a problem with the chunk itself (ie a bad chunk
 * name or a critical chunk), the chunk is (currently) silently ignored.
 */
void /* PRIVATE */
png_push_handle_unknown(png_structp png_ptr, png_infop info_ptr, png_uint_32
    length)
{
   png_uint_32 skip = 0;
   png_uint_32 chunk_name = png_ptr->chunk_name;

   if (PNG_CHUNK_CRITICAL(chunk_name))
   {
#ifdef PNG_READ_UNKNOWN_CHUNKS_SUPPORTED
      if (png_chunk_unknown_handling(png_ptr, chunk_name) !=
          PNG_HANDLE_CHUNK_ALWAYS
#ifdef PNG_READ_USER_CHUNKS_SUPPORTED
          && png_ptr->read_user_chunk_fn == NULL
#endif
          )
#endif
         png_chunk_error(png_ptr, "unknown critical chunk");

      PNG_UNUSED(info_ptr) /* To quiet some compiler warnings */
   }

#ifdef PNG_READ_UNKNOWN_CHUNKS_SUPPORTED
   /* TODO: the code below is apparently just using the
    * png_struct::unknown_chunk member as a temporarily variable, it should be
    * possible to eliminate both it and the temporary buffer.
    */
   if (png_ptr->flags & PNG_FLAG_KEEP_UNKNOWN_CHUNKS)
   {
#ifdef PNG_MAX_MALLOC_64K
      if (length > 65535)
      {
         png_warning(png_ptr, "unknown chunk too large to fit in memory");
         skip = length - 65535;
         length = 65535;
      }
#endif
      /* This is just a record for the user; libpng doesn't use the character
       * form of the name.
       */
      PNG_CSTRING_FROM_CHUNK(png_ptr->unknown_chunk.name, png_ptr->chunk_name);

      /* The following cast should be safe because of the check above. */
      png_ptr->unknown_chunk.size = (png_size_t)length;

      if (length == 0)
         png_ptr->unknown_chunk.data = NULL;

      else
      {
         png_ptr->unknown_chunk.data = (png_bytep)png_malloc(png_ptr,
            png_ptr->unknown_chunk.size);
         png_crc_read(png_ptr, (png_bytep)png_ptr->unknown_chunk.data,
            png_ptr->unknown_chunk.size);
      }

#ifdef PNG_READ_USER_CHUNKS_SUPPORTED
      if (png_ptr->read_user_chunk_fn != NULL)
      {
         /* Callback to user unknown chunk handler */
         int ret;
         ret = (*(png_ptr->read_user_chunk_fn))
             (png_ptr, &png_ptr->unknown_chunk);

         if (ret < 0)
            png_chunk_error(png_ptr, "error in user chunk");

         if (ret == 0)
         {
            if (PNG_CHUNK_CRITICAL(png_ptr->chunk_name))
               if (png_chunk_unknown_handling(png_ptr, chunk_name) !=
                   PNG_HANDLE_CHUNK_ALWAYS)
                  png_chunk_error(png_ptr, "unknown critical chunk");
            png_set_unknown_chunks(png_ptr, info_ptr,
                &png_ptr->unknown_chunk, 1);
         }
      }

      else
#endif
         png_set_unknown_chunks(png_ptr, info_ptr, &png_ptr->unknown_chunk, 1);
      png_free(png_ptr, png_ptr->unknown_chunk.data);
      png_ptr->unknown_chunk.data = NULL;
   }

   else
#endif
      skip=length;
   png_push_crc_skip(png_ptr, skip);
}

void /* PRIVATE */
png_push_have_info(png_structp png_ptr, png_infop info_ptr)
{
   if (png_ptr->info_fn != NULL)
      (*(png_ptr->info_fn))(png_ptr, info_ptr);
}

void /* PRIVATE */
png_push_have_end(png_structp png_ptr, png_infop info_ptr)
{
   if (png_ptr->end_fn != NULL)
      (*(png_ptr->end_fn))(png_ptr, info_ptr);
}

void /* PRIVATE */
png_push_have_row(png_structp png_ptr, png_bytep row)
{
   if (png_ptr->row_fn != NULL)
      (*(png_ptr->row_fn))(png_ptr, row, png_ptr->row_number,
         (int)png_ptr->pass);
}

#ifdef PNG_READ_INTERLACING_SUPPORTED
void PNGAPI
png_progressive_combine_row (png_structp png_ptr, png_bytep old_row,
    png_const_bytep new_row)
{
   if (png_ptr == NULL)
      return;

   /* new_row is a flag here - if it is NULL then the app callback was called
    * from an empty row (see the calls to png_struct::row_fn below), otherwise
    * it must be png_ptr->row_buf+1
    */
   if (new_row != NULL)
      png_combine_row(png_ptr, old_row, 1/*display*/);
}
#endif /* PNG_READ_INTERLACING_SUPPORTED */

void PNGAPI
png_set_progressive_read_fn(png_structp png_ptr, png_voidp progressive_ptr,
    png_progressive_info_ptr info_fn, png_progressive_row_ptr row_fn,
    png_progressive_end_ptr end_fn)
{
   if (png_ptr == NULL)
      return;

   png_ptr->info_fn = info_fn;
   png_ptr->row_fn = row_fn;
   png_ptr->end_fn = end_fn;

   png_set_read_fn(png_ptr, progressive_ptr, png_push_fill_buffer);
}

png_voidp PNGAPI
png_get_progressive_ptr(png_const_structp png_ptr)
{
   if (png_ptr == NULL)
      return (NULL);

   return png_ptr->io_ptr;
}
#endif /* PNG_PROGRESSIVE_READ_SUPPORTED */
