
/* pngstruct.h - header file for PNG reference library
 *
 * Last changed in libpng 1.5.23 [July 23, 2015]
 * Copyright (c) 1998-2002,2004,2006-2015 Glenn Randers-Pehrson
 * (Version 0.96 Copyright (c) 1996, 1997 Andreas Dilger)
 * (Version 0.88 Copyright (c) 1995, 1996 Guy Eric Schalnat, Group 42, Inc.)
 *
 * This code is released under the libpng license.
 * For conditions of distribution and use, see the disclaimer
 * and license in png.h
 */

/* The structure that holds the information to read and write PNG files.
 * The only people who need to care about what is inside of this are the
 * people who will be modifying the library for their own special needs.
 * It should NOT be accessed directly by an application.
 */

#ifndef PNGSTRUCT_H
#define PNGSTRUCT_H
/* zlib.h defines the structure z_stream, an instance of which is included
 * in this structure and is required for decompressing the LZ compressed
 * data in PNG files.
 */
#include "zlib.h"

struct png_struct_def
{
#ifdef PNG_SETJMP_SUPPORTED
   jmp_buf longjmp_buffer;    /* used in png_error */
   png_longjmp_ptr longjmp_fn;/* setjmp non-local goto function. */
#endif
   png_error_ptr error_fn;    /* function for printing errors and aborting */
#ifdef PNG_WARNINGS_SUPPORTED
   png_error_ptr warning_fn;  /* function for printing warnings */
#endif
   png_voidp error_ptr;       /* user supplied struct for error functions */
   png_rw_ptr write_data_fn;  /* function for writing output data */
   png_rw_ptr read_data_fn;   /* function for reading input data */
   png_voidp io_ptr;          /* ptr to application struct for I/O functions */

#ifdef PNG_READ_USER_TRANSFORM_SUPPORTED
   png_user_transform_ptr read_user_transform_fn; /* user read transform */
#endif

#ifdef PNG_WRITE_USER_TRANSFORM_SUPPORTED
   png_user_transform_ptr write_user_transform_fn; /* user write transform */
#endif

/* These were added in libpng-1.0.2 */
#ifdef PNG_USER_TRANSFORM_PTR_SUPPORTED
#if defined(PNG_READ_USER_TRANSFORM_SUPPORTED) || \
    defined(PNG_WRITE_USER_TRANSFORM_SUPPORTED)
   png_voidp user_transform_ptr; /* user supplied struct for user transform */
   png_byte user_transform_depth;    /* bit depth of user transformed pixels */
   png_byte user_transform_channels; /* channels in user transformed pixels */
#endif
#endif

   png_uint_32 mode;          /* tells us where we are in the PNG file */
   png_uint_32 flags;         /* flags indicating various things to libpng */
   png_uint_32 transformations; /* which transformations to perform */

   z_stream zstream;          /* pointer to decompression structure (below) */
   png_bytep zbuf;            /* buffer for zlib */
   uInt zbuf_size;            /* size of zbuf (typically 65536) */
#ifdef PNG_WRITE_SUPPORTED

/* Added in 1.5.4: state to keep track of whether the zstream has been
 * initialized and if so whether it is for IDAT or some other chunk.
 */
#define PNG_ZLIB_UNINITIALIZED 0
#define PNG_ZLIB_FOR_IDAT      1
#define PNG_ZLIB_FOR_TEXT      2 /* anything other than IDAT */
#define PNG_ZLIB_USE_MASK      3 /* bottom two bits */
#define PNG_ZLIB_IN_USE        4 /* a flag value */

   png_uint_32 zlib_state;       /* State of zlib initialization */
/* End of material added at libpng 1.5.4 */

   int zlib_level;            /* holds zlib compression level */
   int zlib_method;           /* holds zlib compression method */
   int zlib_window_bits;      /* holds zlib compression window bits */
   int zlib_mem_level;        /* holds zlib compression memory level */
   int zlib_strategy;         /* holds zlib compression strategy */
#endif
/* Added at libpng 1.5.4 */
#if defined(PNG_WRITE_COMPRESSED_TEXT_SUPPORTED) || \
    defined(PNG_WRITE_CUSTOMIZE_ZTXT_COMPRESSION_SUPPORTED)
   int zlib_text_level;            /* holds zlib compression level */
   int zlib_text_method;           /* holds zlib compression method */
   int zlib_text_window_bits;      /* holds zlib compression window bits */
   int zlib_text_mem_level;        /* holds zlib compression memory level */
   int zlib_text_strategy;         /* holds zlib compression strategy */
#endif
/* End of material added at libpng 1.5.4 */

   png_uint_32 width;         /* width of image in pixels */
   png_uint_32 height;        /* height of image in pixels */
   png_uint_32 num_rows;      /* number of rows in current pass */
   png_uint_32 usr_width;     /* width of row at start of write */
   png_size_t rowbytes;       /* size of row in bytes */
   png_uint_32 iwidth;        /* width of current interlaced row in pixels */
   png_uint_32 row_number;    /* current row in interlace pass */
   png_uint_32 chunk_name;    /* PNG_CHUNK() id of current chunk */
   png_bytep prev_row;        /* buffer to save previous (unfiltered) row.
                               * This is a pointer into big_prev_row
                               */
   png_bytep row_buf;         /* buffer to save current (unfiltered) row.
                               * This is a pointer into big_row_buf
                               */
   png_bytep sub_row;         /* buffer to save "sub" row when filtering */
   png_bytep up_row;          /* buffer to save "up" row when filtering */
   png_bytep avg_row;         /* buffer to save "avg" row when filtering */
   png_bytep paeth_row;       /* buffer to save "Paeth" row when filtering */
   png_size_t info_rowbytes;  /* Added in 1.5.4: cache of updated row bytes */

   png_uint_32 idat_size;     /* current IDAT size for read */
   png_uint_32 crc;           /* current chunk CRC value */
   png_colorp palette;        /* palette from the input file */
   png_uint_16 num_palette;   /* number of color entries in palette */

/* Added at libpng-1.5.10 */
#ifdef PNG_CHECK_FOR_INVALID_INDEX_SUPPORTED
   int num_palette_max;       /* maximum palette index found in IDAT */
#endif

   png_uint_16 num_trans;     /* number of transparency values */
   png_byte compression;      /* file compression type (always 0) */
   png_byte filter;           /* file filter type (always 0) */
   png_byte interlaced;       /* PNG_INTERLACE_NONE, PNG_INTERLACE_ADAM7 */
   png_byte pass;             /* current interlace pass (0 - 6) */
   png_byte do_filter;        /* row filter flags (see PNG_FILTER_ below ) */
   png_byte color_type;       /* color type of file */
   png_byte bit_depth;        /* bit depth of file */
   png_byte usr_bit_depth;    /* bit depth of users row: write only */
   png_byte pixel_depth;      /* number of bits per pixel */
   png_byte channels;         /* number of channels in file */
   png_byte usr_channels;     /* channels at start of write: write only */
   png_byte sig_bytes;        /* magic bytes read/written from start of file */
   png_byte maximum_pixel_depth;
                              /* pixel depth used for the row buffers */
   png_byte transformed_pixel_depth;
                              /* pixel depth after read/write transforms */
   png_byte io_chunk_string[5];
                              /* string name of chunk */

#if defined(PNG_READ_FILLER_SUPPORTED) || defined(PNG_WRITE_FILLER_SUPPORTED)
   png_uint_16 filler;           /* filler bytes for pixel expansion */
#endif

#if defined(PNG_bKGD_SUPPORTED) || defined(PNG_READ_BACKGROUND_SUPPORTED) ||\
   defined(PNG_READ_ALPHA_MODE_SUPPORTED)
   png_byte background_gamma_type;
   png_fixed_point background_gamma;
   png_color_16 background;   /* background color in screen gamma space */
#ifdef PNG_READ_GAMMA_SUPPORTED
   png_color_16 background_1; /* background normalized to gamma 1.0 */
#endif
#endif /* PNG_bKGD_SUPPORTED */

#ifdef PNG_WRITE_FLUSH_SUPPORTED
   png_flush_ptr output_flush_fn; /* Function for flushing output */
   png_uint_32 flush_dist;    /* how many rows apart to flush, 0 - no flush */
   png_uint_32 flush_rows;    /* number of rows written since last flush */
#endif

#ifdef PNG_READ_GAMMA_SUPPORTED
   int gamma_shift;      /* number of "insignificant" bits in 16-bit gamma */
   png_fixed_point gamma;        /* file gamma value */
   png_fixed_point screen_gamma; /* screen gamma value (display_exponent) */

   png_bytep gamma_table;     /* gamma table for 8-bit depth files */
   png_uint_16pp gamma_16_table; /* gamma table for 16-bit depth files */
#if defined(PNG_READ_BACKGROUND_SUPPORTED) || \
   defined(PNG_READ_ALPHA_MODE_SUPPORTED) || \
   defined(PNG_READ_RGB_TO_GRAY_SUPPORTED)
   png_bytep gamma_from_1;    /* converts from 1.0 to screen */
   png_bytep gamma_to_1;      /* converts from file to 1.0 */
   png_uint_16pp gamma_16_from_1; /* converts from 1.0 to screen */
   png_uint_16pp gamma_16_to_1; /* converts from file to 1.0 */
#endif /* READ_BACKGROUND || READ_ALPHA_MODE || RGB_TO_GRAY */
#endif

#if defined(PNG_READ_GAMMA_SUPPORTED) || defined(PNG_sBIT_SUPPORTED)
   png_color_8 sig_bit;       /* significant bits in each available channel */
#endif

#if defined(PNG_READ_SHIFT_SUPPORTED) || defined(PNG_WRITE_SHIFT_SUPPORTED)
   png_color_8 shift;         /* shift for significant bit tranformation */
#endif

#if defined(PNG_tRNS_SUPPORTED) || defined(PNG_READ_BACKGROUND_SUPPORTED) \
 || defined(PNG_READ_EXPAND_SUPPORTED) || defined(PNG_READ_BACKGROUND_SUPPORTED)
   png_bytep trans_alpha;           /* alpha values for paletted files */
   png_color_16 trans_color;  /* transparent color for non-paletted files */
#endif

   png_read_status_ptr read_row_fn;   /* called after each row is decoded */
   png_write_status_ptr write_row_fn; /* called after each row is encoded */
#ifdef PNG_PROGRESSIVE_READ_SUPPORTED
   png_progressive_info_ptr info_fn; /* called after header data fully read */
   png_progressive_row_ptr row_fn;   /* called after a prog. row is decoded */
   png_progressive_end_ptr end_fn;   /* called after image is complete */
   png_bytep save_buffer_ptr;        /* current location in save_buffer */
   png_bytep save_buffer;            /* buffer for previously read data */
   png_bytep current_buffer_ptr;     /* current location in current_buffer */
   png_bytep current_buffer;         /* buffer for recently used data */
   png_uint_32 push_length;          /* size of current input chunk */
   png_uint_32 skip_length;          /* bytes to skip in input data */
   png_size_t save_buffer_size;      /* amount of data now in save_buffer */
   png_size_t save_buffer_max;       /* total size of save_buffer */
   png_size_t buffer_size;           /* total amount of available input data */
   png_size_t current_buffer_size;   /* amount of data now in current_buffer */
   int process_mode;                 /* what push library is currently doing */
   int cur_palette;                  /* current push library palette index */

#endif /* PNG_PROGRESSIVE_READ_SUPPORTED */

#if defined(__TURBOC__) && !defined(_Windows) && !defined(__FLAT__)
/* For the Borland special 64K segment handler */
   png_bytepp offset_table_ptr;
   png_bytep offset_table;
   png_uint_16 offset_table_number;
   png_uint_16 offset_table_count;
   png_uint_16 offset_table_count_free;
#endif

#ifdef PNG_READ_QUANTIZE_SUPPORTED
   png_bytep palette_lookup; /* lookup table for quantizing */
   png_bytep quantize_index; /* index translation for palette files */
#endif

#if defined(PNG_READ_QUANTIZE_SUPPORTED) || defined(PNG_hIST_SUPPORTED)
   png_uint_16p hist;                /* histogram */
#endif

#ifdef PNG_TIME_RFC1123_SUPPORTED
   /* This is going to be unused in libpng16 and removed from libpng17 */
   char time_buffer[29]; /* String to hold RFC 1123 time text */
#endif

/* New members added in libpng-1.0.6 */

   png_uint_32 free_me;    /* flags items libpng is responsible for freeing */

#ifdef PNG_USER_CHUNKS_SUPPORTED
   png_voidp user_chunk_ptr;
   png_user_chunk_ptr read_user_chunk_fn; /* user read chunk handler */
#endif

#ifdef PNG_HANDLE_AS_UNKNOWN_SUPPORTED
   int num_chunk_list;
   png_bytep chunk_list;
#endif

#ifdef PNG_READ_sRGB_SUPPORTED
   /* Added in 1.5.5 to record an sRGB chunk in the png. */
   png_byte is_sRGB;
#endif

/* New members added in libpng-1.0.3 */
#ifdef PNG_READ_RGB_TO_GRAY_SUPPORTED
   png_byte rgb_to_gray_status;
   /* Added in libpng 1.5.5 to record setting of coefficients: */
   png_byte rgb_to_gray_coefficients_set;
   /* These were changed from png_byte in libpng-1.0.6 */
   png_uint_16 rgb_to_gray_red_coeff;
   png_uint_16 rgb_to_gray_green_coeff;
   /* deleted in 1.5.5: rgb_to_gray_blue_coeff; */
#endif

/* New member added in libpng-1.0.4 (renamed in 1.0.9) */
#if defined(PNG_MNG_FEATURES_SUPPORTED)
/* Changed from png_byte to png_uint_32 at version 1.2.0 */
   png_uint_32 mng_features_permitted;
#endif

/* New member added in libpng-1.0.9, ifdef'ed out in 1.0.12, enabled in 1.2.0 */
#ifdef PNG_MNG_FEATURES_SUPPORTED
   png_byte filter_type;
#endif

/* New members added in libpng-1.2.0 */

/* New members added in libpng-1.0.2 but first enabled by default in 1.2.0 */
#ifdef PNG_USER_MEM_SUPPORTED
   png_voidp mem_ptr;             /* user supplied struct for mem functions */
   png_malloc_ptr malloc_fn;      /* function for allocating memory */
   png_free_ptr free_fn;          /* function for freeing memory */
#endif

/* New member added in libpng-1.0.13 and 1.2.0 */
   png_bytep big_row_buf;         /* buffer to save current (unfiltered) row */

#ifdef PNG_READ_QUANTIZE_SUPPORTED
/* The following three members were added at version 1.0.14 and 1.2.4 */
   png_bytep quantize_sort;          /* working sort array */
   png_bytep index_to_palette;       /* where the original index currently is
                                        in the palette */
   png_bytep palette_to_index;       /* which original index points to this
                                         palette color */
#endif

/* New members added in libpng-1.0.16 and 1.2.6 */
   png_byte compression_type;

#ifdef PNG_USER_LIMITS_SUPPORTED
   png_uint_32 user_width_max;
   png_uint_32 user_height_max;

   /* Added in libpng-1.4.0: Total number of sPLT, text, and unknown
    * chunks that can be stored (0 means unlimited).
    */
   png_uint_32 user_chunk_cache_max;

   /* Total memory that a zTXt, sPLT, iTXt, iCCP, or unknown chunk
    * can occupy when decompressed.  0 means unlimited.
    */
   png_alloc_size_t user_chunk_malloc_max;
#endif

/* New member added in libpng-1.0.25 and 1.2.17 */
#ifdef PNG_UNKNOWN_CHUNKS_SUPPORTED
   /* Storage for unknown chunk that the library doesn't recognize. */
   png_unknown_chunk unknown_chunk;
#endif

/* New member added in libpng-1.2.26 */
  png_size_t old_big_row_buf_size;

/* New member added in libpng-1.2.30 */
  png_charp chunkdata;  /* buffer for reading chunk data */

#ifdef PNG_IO_STATE_SUPPORTED
/* New member added in libpng-1.4.0 */
   png_uint_32 io_state;
#endif

/* New member added in libpng-1.5.6 */
   png_bytep big_prev_row;

/* New member added in libpng-1.5.7 */
   void (*read_filter[PNG_FILTER_VALUE_LAST-1])(png_row_infop row_info,
      png_bytep row, png_const_bytep prev_row);

   /* Options */
#ifdef PNG_SET_OPTION_SUPPORTED
   png_byte options;           /* On/off state (up to 4 options) */
#endif
};
#endif /* PNGSTRUCT_H */
