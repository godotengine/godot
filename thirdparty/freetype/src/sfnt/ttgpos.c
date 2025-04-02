/****************************************************************************
 *
 * ttgpos.c
 *
 *   Load the TrueType GPOS table.  The only GPOS layout feature this
 *   currently supports is kerning, from x advances in the pair adjustment
 *   layout feature.
 *
 *   Parts of the implementation were adapted from:
 *   https://github.com/nothings/stb/blob/master/stb_truetype.h
 *
 *   GPOS spec reference available at:
 *   https://learn.microsoft.com/en-us/typography/opentype/spec/gpos
 *
 * Copyright (C) 2024 by
 * David Saltzman
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 */

#include <freetype/internal/ftdebug.h>
#include <freetype/internal/ftstream.h>
#include <freetype/tttags.h>
#include "freetype/fttypes.h"
#include "freetype/internal/ftobjs.h"
#include "ttgpos.h"


#ifdef TT_CONFIG_OPTION_GPOS_KERNING

  /**************************************************************************
   *
   * The macro FT_COMPONENT is used in trace mode.  It is an implicit
   * parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log
   * messages during execution.
   */
#undef  FT_COMPONENT
#define FT_COMPONENT  ttgpos


  typedef enum  coverage_table_format_type_
  {
    COVERAGE_TABLE_FORMAT_LIST  = 1,
    COVERAGE_TABLE_FORMAT_RANGE = 2

  } coverage_table_format_type;

  typedef enum  class_def_table_format_type_
  {
    CLASS_DEF_TABLE_FORMAT_ARRAY        = 1,
    CLASS_DEF_TABLE_FORMAT_RANGE_GROUPS = 2

  } class_def_table_format_type;

  typedef enum  gpos_lookup_type_
  {
    GPOS_LOOKUP_TYPE_NONE                        = 0,
    GPOS_LOOKUP_TYPE_SINGLE_ADJUSTMENT           = 1,
    GPOS_LOOKUP_TYPE_PAIR_ADJUSTMENT             = 2,
    GPOS_LOOKUP_TYPE_CURSIVE_ATTACHMENT          = 3,
    GPOS_LOOKUP_TYPE_MARK_TO_BASE_ATTACHMENT     = 4,
    GPOS_LOOKUP_TYPE_MARK_TO_LIGATURE_ATTACHMENT = 5,
    GPOS_LOOKUP_TYPE_MARK_TO_MARK_ATTACHMENT     = 6,
    GPOS_LOOKUP_TYPE_CONTEXT_POSITIONING         = 7,
    GPOS_LOOKUP_TYPE_CHAINED_CONTEXT_POSITIONING = 8,
    GPOS_LOOKUP_TYPE_EXTENSION_POSITIONING       = 9

  } gpos_lookup_type;

  typedef enum  gpos_pair_adjustment_format_
  {
    GPOS_PAIR_ADJUSTMENT_FORMAT_GLYPH_PAIR = 1,
    GPOS_PAIR_ADJUSTMENT_FORMAT_CLASS_PAIR = 2

  } gpos_pair_adjustment_format;

  typedef enum  gpos_value_format_bitmask_
  {
    GPOS_VALUE_FORMAT_NONE               = 0x0000,
    GPOS_VALUE_FORMAT_X_PLACEMENT        = 0x0001,
    GPOS_VALUE_FORMAT_Y_PLACEMENT        = 0x0002,
    GPOS_VALUE_FORMAT_X_ADVANCE          = 0x0004,
    GPOS_VALUE_FORMAT_Y_ADVANCE          = 0x0008,
    GPOS_VALUE_FORMAT_X_PLACEMENT_DEVICE = 0x0010,
    GPOS_VALUE_FORMAT_Y_PLACEMENT_DEVICE = 0x0020,
    GPOS_VALUE_FORMAT_X_ADVANCE_DEVICE   = 0x0040,
    GPOS_VALUE_FORMAT_Y_ADVANCE_DEVICE   = 0x0080

  } gpos_value_format_bitmask;


  typedef struct TT_GPOS_Subtable_Iterator_Context_
  {
    /* Iteration state. */
    FT_Byte*          current_lookup_table;
    gpos_lookup_type  current_lookup_type;
    FT_UShort         subtable_count;
    FT_Byte*          subtable_offsets;
    FT_UInt           subtable_idx;

    /* Element for the current iteration. */
    FT_Byte*          subtable;
    gpos_lookup_type  subtable_type;

  } TT_GPOS_Subtable_Iterator_Context;


  /* Initialize a subtable iterator for a given lookup list index. */
  static void
  tt_gpos_subtable_iterator_init(
    TT_GPOS_Subtable_Iterator_Context*  context,
    FT_Byte*                            gpos_table,
    FT_ULong                            lookup_list_idx )
  {
    FT_Byte*   lookup_list  = gpos_table + FT_PEEK_USHORT( gpos_table + 8 );
    FT_UInt16  lookup_count = FT_PEEK_USHORT( lookup_list );


    if ( lookup_list_idx < lookup_count )
    {
      context->current_lookup_table =
        lookup_list + FT_PEEK_USHORT( lookup_list + 2 + 2 * lookup_list_idx );
      context->current_lookup_type =
        (gpos_lookup_type)FT_PEEK_USHORT( context->current_lookup_table );
      context->subtable_count =
        FT_PEEK_USHORT( context->current_lookup_table + 4 );
      context->subtable_offsets = context->current_lookup_table + 6;
    }
    else
    {
      context->current_lookup_table = NULL;
      context->current_lookup_type  = GPOS_LOOKUP_TYPE_NONE;
      context->subtable_count       = 0;
      context->subtable_offsets     = NULL;
    }

    context->subtable_idx  = 0;
    context->subtable      = NULL;
    context->subtable_type = GPOS_LOOKUP_TYPE_NONE;
  }


  /* Get the next subtable.  Return whether there was a next one. */
  static FT_Bool
  tt_gpos_subtable_iterator_next(
    TT_GPOS_Subtable_Iterator_Context*  context )
  {
    if ( context->subtable_idx < context->subtable_count )
    {
      FT_UShort  subtable_offset =
        FT_PEEK_USHORT( context->subtable_offsets +
                        2 * context->subtable_idx );


      context->subtable = context->current_lookup_table + subtable_offset;

      if ( context->current_lookup_type ==
           GPOS_LOOKUP_TYPE_EXTENSION_POSITIONING )
      {
        /* Update type and subtable based on extension positioning header. */
        context->subtable_type =
          (gpos_lookup_type)FT_PEEK_USHORT( context->subtable + 2 );
        context->subtable += FT_PEEK_ULONG( context->subtable + 4 );
      }
      else
        context->subtable_type = context->current_lookup_type;

      context->subtable_idx++;
      return TRUE;
    }

    return FALSE;
  }


  static FT_Int
  tt_gpos_get_coverage_index( FT_Byte  *coverage_table,
                              FT_UInt   glyph )
  {
    coverage_table_format_type  coverage_format =
      (coverage_table_format_type)FT_PEEK_USHORT( coverage_table );


    switch ( coverage_format )
    {
    case COVERAGE_TABLE_FORMAT_LIST:
      {
        FT_UShort  glyph_count = FT_PEEK_USHORT( coverage_table + 2 );

        FT_Int  l = 0;
        FT_Int  r = glyph_count - 1;
        FT_Int  m;

        FT_Int  straw;
        FT_Int  needle = (FT_Int)glyph;


        /* Binary search. */
        while ( l <= r )
        {
          FT_Byte   *glyph_array = coverage_table + 4;
          FT_UShort  glyph_id;


          m        = ( l + r ) >> 1;
          glyph_id = FT_PEEK_USHORT( glyph_array + 2 * m );
          straw    = glyph_id;

          if ( needle < straw )
            r = m - 1;
          else if ( needle > straw )
            l = m + 1;
          else
            return m;
        }
        break;
      }

    case COVERAGE_TABLE_FORMAT_RANGE:
      {
        FT_UShort  range_count = FT_PEEK_USHORT( coverage_table + 2 );
        FT_Byte   *range_array = coverage_table + 4;

        FT_Int  l = 0;
        FT_Int  r = range_count - 1;
        FT_Int  m;

        FT_Int  straw_start;
        FT_Int  straw_end;
        FT_Int  needle = (FT_Int)glyph;


        /* Binary search. */
        while ( l <= r )
        {
          FT_Byte  *range_record;


          m            = ( l + r ) >> 1;
          range_record = range_array + 6 * m;
          straw_start  = FT_PEEK_USHORT( range_record );
          straw_end    = FT_PEEK_USHORT( range_record + 2 );

          if ( needle < straw_start )
            r = m - 1;
          else if ( needle > straw_end )
            l = m + 1;
          else
          {
            FT_UShort start_coverage_index =
                        FT_PEEK_USHORT( range_record + 4 );


            return (FT_Int)start_coverage_index + (FT_Int)glyph - straw_start;
          }
        }
        break;
      }
    }

    return -1;
  }


  static FT_Int
  tt_gpos_get_glyph_class( FT_Byte  *class_def_table,
                           FT_UInt   glyph )
  {
    class_def_table_format_type  class_def_format =
      (class_def_table_format_type)FT_PEEK_USHORT( class_def_table );


    switch ( class_def_format )
    {
    case CLASS_DEF_TABLE_FORMAT_ARRAY:
      {
        FT_UInt  start_glyph_id    = FT_PEEK_USHORT( class_def_table + 2 );
        FT_UInt  glyph_count       = FT_PEEK_USHORT( class_def_table + 4 );
        FT_Byte  *class_value_array = class_def_table + 6;


        if ( glyph >= start_glyph_id              &&
             glyph < start_glyph_id + glyph_count )
          return (FT_Int)FT_PEEK_USHORT( class_value_array +
                                         2 * ( glyph - start_glyph_id ) );
        break;
      }

    case CLASS_DEF_TABLE_FORMAT_RANGE_GROUPS:
      {
        FT_UShort  class_range_count   = FT_PEEK_USHORT( class_def_table + 2 );
        FT_Byte   *class_range_records = class_def_table + 4;

        FT_Int  l = 0;
        FT_Int  r = class_range_count - 1;
        FT_Int  m;

        FT_Int  straw_start;
        FT_Int  straw_end;
        FT_Int  needle = (FT_Int)glyph;


        while ( l <= r )
        {
          FT_Byte *class_range_record;


          m                  = ( l + r ) >> 1;
          class_range_record = class_range_records + 6 * m;
          straw_start        = FT_PEEK_USHORT( class_range_record );
          straw_end          = FT_PEEK_USHORT( class_range_record + 2 );

          if ( needle < straw_start )
            r = m - 1;
          else if ( needle > straw_end )
            l = m + 1;
          else
            return (FT_Int)FT_PEEK_USHORT( class_range_record + 4 );
        }
        break;
      }
    }

    /* "All glyphs not assigned to a class fall into class 0." */
    /* (OpenType spec)                                         */
    return 0;
  }


  FT_LOCAL_DEF( FT_Error )
  tt_face_load_gpos( TT_Face    face,
                     FT_Stream  stream )
  {
    FT_Error  error;
    FT_ULong  table_size;


    /* The GPOS table is optional; exit silently if it is missing. */
    error = face->goto_table( face, TTAG_GPOS, stream, &table_size );
    if ( error )
      goto Exit;

    if ( table_size < 4 )  /* the case of a malformed table */
    {
      FT_ERROR(( "tt_face_load_gpos:"
                 " GPOS table is too small - ignored\n" ));
      error = FT_THROW( Table_Missing );
      goto Exit;
    }

    if ( FT_FRAME_EXTRACT( table_size, face->gpos_table ) )
    {
      FT_ERROR(( "tt_face_load_gpos:"
                 " could not extract GPOS table\n" ));
      goto Exit;
    }

    face->gpos_kerning_available = FALSE;

    if ( face->gpos_table )
    {
      FT_Byte*   feature_list    = face->gpos_table +
                                   FT_PEEK_USHORT( face->gpos_table + 6 );
      FT_UInt16  feature_count   = FT_PEEK_USHORT( feature_list );
      FT_Byte*   feature_records = feature_list + 2;

      FT_UInt  idx;


      for ( idx = 0; idx < feature_count; idx++, feature_records += 6 )
      {
        FT_ULong  feature_tag = FT_PEEK_ULONG( feature_records );


        if ( feature_tag == TTAG_kern )
        {
          face->gpos_kerning_available = TRUE;
          break;
        }
      }
    }

  Exit:
    return error;
  }


  FT_LOCAL_DEF( void )
  tt_face_done_gpos( TT_Face  face )
  {
    FT_Stream  stream = face->root.stream;


    FT_FRAME_RELEASE( face->gpos_table );
  }


  FT_LOCAL_DEF( FT_Int )
  tt_face_get_gpos_kerning( TT_Face  face,
                            FT_UInt  left_glyph,
                            FT_UInt  right_glyph )
  {
    FT_Byte*   feature_list;
    FT_UInt16  feature_count;
    FT_Byte*   feature_records;
    FT_UInt    feature_idx;


    if ( !face->gpos_kerning_available )
      return 0;

    feature_list    = face->gpos_table +
                      FT_PEEK_USHORT( face->gpos_table + 6 );
    feature_count   = FT_PEEK_USHORT( feature_list );
    feature_records = feature_list + 2;

    for ( feature_idx = 0;
          feature_idx < feature_count;
          feature_idx++, feature_records += 6 )
    {
      FT_ULong   feature_tag = FT_PEEK_ULONG( feature_records );
      FT_Byte*   feature_table;
      FT_UInt16  lookup_idx_count;
      FT_UInt16  lookup_idx;


      if ( feature_tag != TTAG_kern )
        continue;

      feature_table    = feature_list + FT_PEEK_USHORT( feature_records + 4 );
      lookup_idx_count = FT_PEEK_USHORT( feature_table + 2 );

      for ( lookup_idx = 0; lookup_idx < lookup_idx_count; lookup_idx++ )
      {
        FT_UInt16 lookup_list_idx =
          FT_PEEK_USHORT( feature_table + 4 + 2 * lookup_idx );
        TT_GPOS_Subtable_Iterator_Context  subtable_iter;


        tt_gpos_subtable_iterator_init( &subtable_iter,
                                        face->gpos_table,
                                        lookup_list_idx );

        while ( tt_gpos_subtable_iterator_next( &subtable_iter ) )
        {
          FT_Byte*  subtable;

          gpos_value_format_bitmask    value_format_1;
          gpos_value_format_bitmask    value_format_2;
          gpos_pair_adjustment_format  format;

          FT_UShort  coverage_offset;
          FT_Int     coverage_index;


          if ( subtable_iter.subtable_type !=
               GPOS_LOOKUP_TYPE_PAIR_ADJUSTMENT )
            continue;

          subtable = subtable_iter.subtable;

          value_format_1 =
            (gpos_value_format_bitmask)FT_PEEK_USHORT( subtable + 4 );
          value_format_2 =
            (gpos_value_format_bitmask)FT_PEEK_USHORT( subtable + 6 );

          if ( !( value_format_1 == GPOS_VALUE_FORMAT_X_ADVANCE &&
                  value_format_2 == GPOS_VALUE_FORMAT_NONE      ) )
            continue;

          format = (gpos_pair_adjustment_format)FT_PEEK_USHORT( subtable );

          coverage_offset = FT_PEEK_USHORT( subtable + 2 );
          coverage_index  =
            tt_gpos_get_coverage_index( subtable + coverage_offset,
                                        left_glyph );

          if ( coverage_index == -1 )
            continue;

          switch ( format )
          {
          case GPOS_PAIR_ADJUSTMENT_FORMAT_GLYPH_PAIR:
            {
              FT_Int  l, r, m;
              FT_Int  straw, needle;

              FT_Int  value_record_pair_size_in_bytes = 2;

              FT_UShort  pair_set_count = FT_PEEK_USHORT( subtable + 8 );
              FT_UShort  pair_pos_offset;

              FT_Byte*   pair_value_table;
              FT_UShort  pair_value_count;
              FT_Byte*   pair_value_array;


              if ( coverage_index >= pair_set_count )
                return 0;

              pair_pos_offset =
                FT_PEEK_USHORT( subtable + 10 + 2 * coverage_index );

              pair_value_table = subtable + pair_pos_offset;
              pair_value_count = FT_PEEK_USHORT( pair_value_table );
              pair_value_array = pair_value_table + 2;

              needle = (FT_Int)right_glyph;
              r      = pair_value_count - 1;
              l      = 0;

              /* Binary search. */
              while ( l <= r )
              {
                FT_UShort  second_glyph;
                FT_Byte*   pair_value;


                m            = ( l + r ) >> 1;
                pair_value   = pair_value_array +
                               ( 2 + value_record_pair_size_in_bytes ) * m;
                second_glyph = FT_PEEK_USHORT( pair_value );
                straw        = second_glyph;

                if ( needle < straw )
                  r = m - 1;
                else if ( needle > straw )
                  l = m + 1;
                else
                {
                  FT_Short  x_advance = FT_PEEK_SHORT( pair_value + 2 );


                  return x_advance;
                }
              }
              break;
            }

          case GPOS_PAIR_ADJUSTMENT_FORMAT_CLASS_PAIR:
            {
              FT_UShort  class_def1_offset = FT_PEEK_USHORT( subtable + 8 );
              FT_UShort  class_def2_offset = FT_PEEK_USHORT( subtable + 10 );

              FT_Int  left_glyph_class =
                tt_gpos_get_glyph_class( subtable + class_def1_offset,
                                         left_glyph );
              FT_Int  right_glyph_class =
                tt_gpos_get_glyph_class( subtable + class_def2_offset,
                                         right_glyph );

              FT_UShort class1_count = FT_PEEK_USHORT( subtable + 12 );
              FT_UShort class2_count = FT_PEEK_USHORT( subtable + 14 );

              FT_Byte *class1_records, *class2_records;
              FT_Short x_advance;


              if ( left_glyph_class < 0             ||
                   left_glyph_class >= class1_count )
                return 0;  /* malformed */
              if ( right_glyph_class < 0             ||
                   right_glyph_class >= class2_count )
                return 0;  /* malformed */

              if ( right_glyph_class == 0 )
                continue; /* right glyph not found in this table */

              class1_records = subtable + 16;
              class2_records =
                class1_records + 2 * ( left_glyph_class * class2_count );

              x_advance =
                FT_PEEK_SHORT( class2_records + 2 * right_glyph_class );

              return x_advance;
            }
          }
        }
      }
    }

    return 0;
  }

#else /* !TT_CONFIG_OPTION_GPOS_KERNING */

  /* ANSI C doesn't like empty source files */
  typedef int  tt_gpos_dummy_;

#endif /* !TT_CONFIG_OPTION_GPOS_KERNING */


/* END */
