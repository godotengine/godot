/****************************************************************************
 *
 * ttcolr.c
 *
 *   TrueType and OpenType colored glyph layer support (body).
 *
 * Copyright (C) 2018-2023 by
 * David Turner, Robert Wilhelm, Dominik Röttsches, and Werner Lemberg.
 *
 * Originally written by Shao Yu Zhang <shaozhang@fb.com>.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


  /**************************************************************************
   *
   * `COLR' table specification:
   *
   *   https://www.microsoft.com/typography/otspec/colr.htm
   *
   */


#include <freetype/internal/ftcalc.h>
#include <freetype/internal/ftdebug.h>
#include <freetype/internal/ftstream.h>
#include <freetype/tttags.h>
#include <freetype/ftcolor.h>
#include <freetype/config/integer-types.h>

#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT
#include <freetype/internal/services/svmm.h>
#endif

#ifdef TT_CONFIG_OPTION_COLOR_LAYERS

#include "ttcolr.h"


  /* NOTE: These are the table sizes calculated through the specs. */
#define BASE_GLYPH_SIZE                   6U
#define BASE_GLYPH_PAINT_RECORD_SIZE      6U
#define LAYER_V1_LIST_PAINT_OFFSET_SIZE   4U
#define LAYER_V1_LIST_NUM_LAYERS_SIZE     4U
#define COLOR_STOP_SIZE                   6U
#define VAR_IDX_BASE_SIZE                 4U
#define LAYER_SIZE                        4U
/* https://docs.microsoft.com/en-us/typography/opentype/spec/colr#colr-header */
/* 3 * uint16 + 2 * Offset32 */
#define COLRV0_HEADER_SIZE               14U
/* COLRV0_HEADER_SIZE + 5 * Offset32 */
#define COLRV1_HEADER_SIZE               34U


#define ENSURE_READ_BYTES( byte_size )                             \
  if ( p < colr->paints_start_v1                                || \
       p > (FT_Byte*)colr->table + colr->table_size - byte_size )  \
    return 0


  typedef enum  FT_PaintFormat_Internal_
  {
    FT_COLR_PAINTFORMAT_INTERNAL_VAR_SOLID                = 3,
    FT_COLR_PAINTFORMAT_INTERNAL_VAR_LINEAR_GRADIENT      = 5,
    FT_COLR_PAINTFORMAT_INTERNAL_VAR_RADIAL_GRADIENT      = 7,
    FT_COLR_PAINTFORMAT_INTERNAL_VAR_SWEEP_GRADIENT       = 9,
    FT_COLR_PAINTFORMAT_INTERNAL_VAR_TRANSFORM            = 13,
    FT_COLR_PAINTFORMAT_INTERNAL_VAR_TRANSLATE            = 15,
    FT_COLR_PAINTFORMAT_INTERNAL_VAR_SCALE                = 17,
    FT_COLR_PAINTFORMAT_INTERNAL_SCALE_CENTER             = 18,
    FT_COLR_PAINTFORMAT_INTERNAL_VAR_SCALE_CENTER         = 19,
    FT_COLR_PAINTFORMAT_INTERNAL_SCALE_UNIFORM            = 20,
    FT_COLR_PAINTFORMAT_INTERNAL_VAR_SCALE_UNIFORM        = 21,
    FT_COLR_PAINTFORMAT_INTERNAL_SCALE_UNIFORM_CENTER     = 22,
    FT_COLR_PAINTFORMAT_INTERNAL_VAR_SCALE_UNIFORM_CENTER = 23,
    FT_COLR_PAINTFORMAT_INTERNAL_VAR_ROTATE               = 25,
    FT_COLR_PAINTFORMAT_INTERNAL_ROTATE_CENTER            = 26,
    FT_COLR_PAINTFORMAT_INTERNAL_VAR_ROTATE_CENTER        = 27,
    FT_COLR_PAINTFORMAT_INTERNAL_VAR_SKEW                 = 29,
    FT_COLR_PAINTFORMAT_INTERNAL_SKEW_CENTER              = 30,
    FT_COLR_PAINTFORMAT_INTERNAL_VAR_SKEW_CENTER          = 31,

  } FT_PaintFormat_Internal;


  typedef struct  BaseGlyphRecord_
  {
    FT_UShort  gid;
    FT_UShort  first_layer_index;
    FT_UShort  num_layers;

  } BaseGlyphRecord;


  typedef struct  BaseGlyphV1Record_
  {
    FT_UShort  gid;
    /* Offset from start of BaseGlyphV1List, i.e., from base_glyphs_v1. */
    FT_ULong   paint_offset;

  } BaseGlyphV1Record;


  typedef struct  Colr_
  {
    FT_UShort  version;
    FT_UShort  num_base_glyphs;
    FT_UShort  num_layers;

    FT_Byte*  base_glyphs;
    FT_Byte*  layers;

    FT_ULong  num_base_glyphs_v1;
    /* Points at beginning of BaseGlyphV1List. */
    FT_Byte*  base_glyphs_v1;

    FT_ULong  num_layers_v1;
    FT_Byte*  layers_v1;

    FT_Byte*  clip_list;

    /*
     * Paint tables start at the minimum of the end of the LayerList and the
     * end of the BaseGlyphList.  Record this location in a field here for
     * safety checks when accessing paint tables.
     */
    FT_Byte*  paints_start_v1;

#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT
    /* Item Variation Store for variable 'COLR' v1. */
    GX_ItemVarStoreRec    var_store;
    GX_DeltaSetIdxMapRec  delta_set_idx_map;
#endif

    /* The memory that backs up the `COLR' table. */
    void*     table;
    FT_ULong  table_size;

  } Colr;


  /**************************************************************************
   *
   * The macro FT_COMPONENT is used in trace mode.  It is an implicit
   * parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log
   * messages during execution.
   */
#undef  FT_COMPONENT
#define FT_COMPONENT  ttcolr


  FT_LOCAL_DEF( FT_Error )
  tt_face_load_colr( TT_Face    face,
                     FT_Stream  stream )
  {
    FT_Error   error;
    FT_Memory  memory = face->root.memory;

    FT_Byte*  table = NULL;
    FT_Byte*  p     = NULL;
    /* Needed for reading array lengths in referenced tables. */
    FT_Byte*  p1    = NULL;

    Colr*  colr = NULL;

    FT_ULong  base_glyph_offset, layer_offset;
    FT_ULong  base_glyphs_offset_v1, num_base_glyphs_v1;
    FT_ULong  layer_offset_v1, num_layers_v1, clip_list_offset;
    FT_ULong  table_size;
#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT
    FT_ULong  colr_offset_in_stream;
#endif


    /* `COLR' always needs `CPAL' */
    if ( !face->cpal )
      return FT_THROW( Invalid_File_Format );

    error = face->goto_table( face, TTAG_COLR, stream, &table_size );
    if ( error )
      goto NoColr;

#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT
    colr_offset_in_stream = FT_STREAM_POS();
#endif

    if ( table_size < COLRV0_HEADER_SIZE )
      goto NoColr;

    if ( FT_FRAME_EXTRACT( table_size, table ) )
      goto NoColr;

    p = table;

    if ( FT_NEW( colr ) )
      goto NoColr;

    colr->version = FT_NEXT_USHORT( p );
    if ( colr->version != 0 && colr->version != 1 )
      goto InvalidTable;

    colr->num_base_glyphs = FT_NEXT_USHORT( p );
    base_glyph_offset     = FT_NEXT_ULONG( p );

    if ( base_glyph_offset >= table_size )
      goto InvalidTable;
    if ( colr->num_base_glyphs * BASE_GLYPH_SIZE >
           table_size - base_glyph_offset )
      goto InvalidTable;

    layer_offset     = FT_NEXT_ULONG( p );
    colr->num_layers = FT_NEXT_USHORT( p );

    if ( layer_offset >= table_size )
      goto InvalidTable;
    if ( colr->num_layers * LAYER_SIZE > table_size - layer_offset )
      goto InvalidTable;

    if ( colr->version == 1 )
    {
      if ( table_size < COLRV1_HEADER_SIZE )
        goto InvalidTable;

      base_glyphs_offset_v1 = FT_NEXT_ULONG( p );

      if ( base_glyphs_offset_v1 + 4 >= table_size )
        goto InvalidTable;

      p1                 = (FT_Byte*)( table + base_glyphs_offset_v1 );
      num_base_glyphs_v1 = FT_PEEK_ULONG( p1 );

      if ( num_base_glyphs_v1 * BASE_GLYPH_PAINT_RECORD_SIZE >
             table_size - base_glyphs_offset_v1 )
        goto InvalidTable;

      colr->num_base_glyphs_v1 = num_base_glyphs_v1;
      colr->base_glyphs_v1     = p1;

      layer_offset_v1 = FT_NEXT_ULONG( p );

      if ( layer_offset_v1 >= table_size )
        goto InvalidTable;

      if ( layer_offset_v1 )
      {
        if ( layer_offset_v1 + 4 >= table_size )
          goto InvalidTable;

        p1            = (FT_Byte*)( table + layer_offset_v1 );
        num_layers_v1 = FT_PEEK_ULONG( p1 );

        if ( num_layers_v1 * LAYER_V1_LIST_PAINT_OFFSET_SIZE >
               table_size - layer_offset_v1 )
          goto InvalidTable;

        colr->num_layers_v1 = num_layers_v1;
        colr->layers_v1     = p1;

        colr->paints_start_v1 =
            FT_MIN( colr->base_glyphs_v1 +
                    colr->num_base_glyphs_v1 * BASE_GLYPH_PAINT_RECORD_SIZE,
                    colr->layers_v1 +
                    colr->num_layers_v1 * LAYER_V1_LIST_PAINT_OFFSET_SIZE );
      }
      else
      {
        colr->num_layers_v1   = 0;
        colr->layers_v1       = 0;
        colr->paints_start_v1 =
          colr->base_glyphs_v1 +
          colr->num_base_glyphs_v1 * BASE_GLYPH_PAINT_RECORD_SIZE;
      }

      clip_list_offset = FT_NEXT_ULONG( p );

      if ( clip_list_offset >= table_size )
        goto InvalidTable;

      if ( clip_list_offset )
        colr->clip_list = (FT_Byte*)( table + clip_list_offset );
      else
        colr->clip_list = 0;

#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT
      colr->var_store.dataCount     = 0;
      colr->var_store.varData       = NULL;
      colr->var_store.axisCount     = 0;
      colr->var_store.regionCount   = 0;
      colr->var_store.varRegionList = 0;

      colr->delta_set_idx_map.mapCount   = 0;
      colr->delta_set_idx_map.outerIndex = NULL;
      colr->delta_set_idx_map.innerIndex = NULL;

      if ( face->variation_support & TT_FACE_FLAG_VAR_FVAR )
      {
        FT_ULong  var_idx_map_offset, var_store_offset;

        FT_Service_MultiMasters  mm = (FT_Service_MultiMasters)face->mm;


        var_idx_map_offset = FT_NEXT_ULONG( p );

        if ( var_idx_map_offset >= table_size )
          goto InvalidTable;

        var_store_offset = FT_NEXT_ULONG( p );
        if ( var_store_offset >= table_size )
          goto InvalidTable;

        if ( var_store_offset )
        {
          /* If variation info has not been initialized yet, try doing so, */
          /* otherwise loading the variation store will fail as it         */
          /* requires access to `blend` for checking the number of axes.   */
          if ( !face->blend )
            if ( mm->get_mm_var( FT_FACE( face ), NULL ) )
              goto InvalidTable;

          /* Try loading `VarIdxMap` and `VarStore`. */
          error = mm->load_item_var_store(
                    FT_FACE( face ),
                    colr_offset_in_stream + var_store_offset,
                    &colr->var_store );
          if ( error != FT_Err_Ok )
            goto InvalidTable;
        }

        if ( colr->var_store.axisCount && var_idx_map_offset )
        {
          error = mm->load_delta_set_idx_map(
                    FT_FACE( face ),
                    colr_offset_in_stream + var_idx_map_offset,
                    &colr->delta_set_idx_map,
                    &colr->var_store,
                    table_size );
          if ( error != FT_Err_Ok )
            goto InvalidTable;
        }
      }
#endif /* TT_CONFIG_OPTION_GX_VAR_SUPPORT */
    }

    colr->base_glyphs = (FT_Byte*)( table + base_glyph_offset );
    colr->layers      = (FT_Byte*)( table + layer_offset      );
    colr->table       = table;
    colr->table_size  = table_size;

    face->colr = colr;

    return FT_Err_Ok;

  InvalidTable:
#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT
    {
      FT_Service_MultiMasters  mm = (FT_Service_MultiMasters)face->mm;


      mm->done_delta_set_idx_map( FT_FACE( face ),
                                  &colr->delta_set_idx_map );
      mm->done_item_var_store( FT_FACE( face ),
                               &colr->var_store );
    }
#endif

    error = FT_THROW( Invalid_Table );

  NoColr:
    FT_FRAME_RELEASE( table );
    FT_FREE( colr );

    return error;
  }


  FT_LOCAL_DEF( void )
  tt_face_free_colr( TT_Face  face )
  {
    FT_Stream  stream = face->root.stream;
    FT_Memory  memory = face->root.memory;

    Colr*  colr = (Colr*)face->colr;


    if ( colr )
    {
#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT
      {
        FT_Service_MultiMasters  mm = (FT_Service_MultiMasters)face->mm;


        mm->done_delta_set_idx_map( FT_FACE( face ),
                                    &colr->delta_set_idx_map );
        mm->done_item_var_store( FT_FACE( face ),
                                 &colr->var_store );
      }
#endif
      FT_FRAME_RELEASE( colr->table );
      FT_FREE( colr );
    }
  }


  static FT_Bool
  find_base_glyph_record( FT_Byte*          base_glyph_begin,
                          FT_UInt           num_base_glyph,
                          FT_UInt           glyph_id,
                          BaseGlyphRecord*  record )
  {
    FT_UInt  min = 0;
    FT_UInt  max = num_base_glyph;


    while ( min < max )
    {
      FT_UInt   mid = min + ( max - min ) / 2;
      FT_Byte*  p   = base_glyph_begin + mid * BASE_GLYPH_SIZE;

      FT_UShort  gid = FT_NEXT_USHORT( p );


      if ( gid < glyph_id )
        min = mid + 1;
      else if (gid > glyph_id )
        max = mid;
      else
      {
        record->gid               = gid;
        record->first_layer_index = FT_NEXT_USHORT( p );
        record->num_layers        = FT_NEXT_USHORT( p );

        return 1;
      }
    }

    return 0;
  }


  FT_LOCAL_DEF( FT_Bool )
  tt_face_get_colr_layer( TT_Face            face,
                          FT_UInt            base_glyph,
                          FT_UInt           *aglyph_index,
                          FT_UInt           *acolor_index,
                          FT_LayerIterator*  iterator )
  {
    Colr*            colr = (Colr*)face->colr;
    BaseGlyphRecord  glyph_record;


    if ( !colr )
      return 0;

    if ( !iterator->p )
    {
      FT_ULong  offset;


      /* first call to function */
      iterator->layer = 0;

      if ( !find_base_glyph_record( colr->base_glyphs,
                                    colr->num_base_glyphs,
                                    base_glyph,
                                    &glyph_record ) )
        return 0;

      if ( glyph_record.num_layers )
        iterator->num_layers = glyph_record.num_layers;
      else
        return 0;

      offset = LAYER_SIZE * glyph_record.first_layer_index;
      if ( offset + LAYER_SIZE * glyph_record.num_layers > colr->table_size )
        return 0;

      iterator->p = colr->layers + offset;
    }

    if ( iterator->layer >= iterator->num_layers                     ||
         iterator->p < colr->layers                                  ||
         iterator->p >= ( (FT_Byte*)colr->table + colr->table_size ) )
      return 0;

    *aglyph_index = FT_NEXT_USHORT( iterator->p );
    *acolor_index = FT_NEXT_USHORT( iterator->p );

    if ( *aglyph_index >= (FT_UInt)( FT_FACE( face )->num_glyphs )   ||
         ( *acolor_index != 0xFFFF                                 &&
           *acolor_index >= face->palette_data.num_palette_entries ) )
      return 0;

    iterator->layer++;

    return 1;
  }


  static FT_Bool
  read_color_line( Colr*          colr,
                   FT_Byte*       color_line_p,
                   FT_ColorLine*  colorline,
                   FT_Bool        read_variable )
  {
    FT_Byte*        p = color_line_p;
    FT_PaintExtend  paint_extend;


    ENSURE_READ_BYTES( 3 );

    paint_extend = (FT_PaintExtend)FT_NEXT_BYTE( p );
    if ( paint_extend > FT_COLR_PAINT_EXTEND_REFLECT )
      return 0;

    colorline->extend = paint_extend;

    colorline->color_stop_iterator.num_color_stops    = FT_NEXT_USHORT( p );
    colorline->color_stop_iterator.p                  = p;
    colorline->color_stop_iterator.current_color_stop = 0;
    colorline->color_stop_iterator.read_variable      = read_variable;

    return 1;
  }


  /*
   * Read a paint offset for `FT_Paint*` objects that have them and check
   * whether it is within reasonable limits within the font and the COLR
   * table.
   *
   * Return 1 on success, 0 on failure.
   */
  static FT_Bool
  get_child_table_pointer ( Colr*      colr,
                            FT_Byte*   paint_base,
                            FT_Byte**  p,
                            FT_Byte**  child_table_pointer )
  {
    FT_UInt32  paint_offset;
    FT_Byte*   child_table_p;


    if ( !child_table_pointer )
      return 0;

    if ( *p < colr->paints_start_v1                            ||
         *p > (FT_Byte*)colr->table + colr->table_size - 1 - 3 )
      return 0;

    paint_offset = FT_NEXT_UOFF3( *p );
    if ( !paint_offset )
      return 0;

    child_table_p = (FT_Byte*)( paint_base + paint_offset );

    if ( child_table_p < colr->paints_start_v1                         ||
         child_table_p >= ( (FT_Byte*)colr->table + colr->table_size ) )
      return 0;

    *child_table_pointer = child_table_p;
    return 1;
  }


#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT

  static FT_Bool
  get_deltas_for_var_index_base ( TT_Face           face,
                                  Colr*             colr,
                                  FT_ULong          var_index_base,
                                  FT_UInt           num_deltas,
                                  FT_ItemVarDelta*  deltas )
  {
    FT_UInt   outer_index    = 0;
    FT_UInt   inner_index    = 0;
    FT_ULong  loop_var_index = var_index_base;

    FT_Service_MultiMasters  mm = (FT_Service_MultiMasters)face->mm;

    FT_UInt  i = 0;


    if ( var_index_base == 0xFFFFFFFF )
    {
      for ( i = 0; i < num_deltas; ++i )
        deltas[i] = 0;
      return 1;
    }

    for ( i = 0; i < num_deltas; ++i )
    {
      loop_var_index = var_index_base + i;

      if ( colr->delta_set_idx_map.innerIndex )
      {
        if ( loop_var_index >= colr->delta_set_idx_map.mapCount )
          loop_var_index = colr->delta_set_idx_map.mapCount - 1;

        outer_index = colr->delta_set_idx_map.outerIndex[loop_var_index];
        inner_index = colr->delta_set_idx_map.innerIndex[loop_var_index];
      }
      else
      {
        outer_index = 0;
        inner_index = loop_var_index;
      }

      deltas[i] = mm->get_item_delta( FT_FACE( face ), &colr->var_store,
                                      outer_index, inner_index );
    }

    return 1;
  }

#endif /* TT_CONFIG_OPTION_GX_VAR_SUPPORT */


  static FT_Bool
  read_paint( TT_Face         face,
              Colr*           colr,
              FT_Byte*        p,
              FT_COLR_Paint*  apaint )
  {
    FT_Byte*  paint_base    = p;
    FT_Byte*  child_table_p = NULL;
    FT_Bool   do_read_var   = FALSE;

#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT
    FT_ULong         var_index_base = 0;
    /* Longest varIndexBase offset is 5 in the spec. */
    FT_ItemVarDelta  item_deltas[6] = { 0, 0, 0, 0, 0, 0 };
#else
    FT_UNUSED( face );
#endif


    if ( !p || !colr || !colr->table )
      return 0;

    /* The last byte of the 'COLR' table is at 'size-1'; subtract 1 of    */
    /* that to account for the expected format byte we are going to read. */
    if ( p < colr->paints_start_v1                        ||
         p > (FT_Byte*)colr->table + colr->table_size - 2 )
      return 0;

    apaint->format = (FT_PaintFormat)FT_NEXT_BYTE( p );

    if ( apaint->format >= FT_COLR_PAINT_FORMAT_MAX )
      return 0;

    if ( apaint->format == FT_COLR_PAINTFORMAT_COLR_LAYERS )
    {
      /* Initialize layer iterator/ */
      FT_Byte    num_layers;
      FT_UInt32  first_layer_index;


      num_layers = FT_NEXT_BYTE( p );
      if ( num_layers > colr->num_layers_v1 )
        return 0;

      first_layer_index = FT_NEXT_ULONG( p );
      if ( first_layer_index + num_layers > colr->num_layers_v1 )
        return 0;

      apaint->u.colr_layers.layer_iterator.num_layers = num_layers;
      apaint->u.colr_layers.layer_iterator.layer      = 0;
      /* TODO: Check whether pointer is outside colr? */
      apaint->u.colr_layers.layer_iterator.p =
        colr->layers_v1 +
        LAYER_V1_LIST_NUM_LAYERS_SIZE +
        LAYER_V1_LIST_PAINT_OFFSET_SIZE * first_layer_index;

      return 1;
    }

    else if ( apaint->format == FT_COLR_PAINTFORMAT_SOLID ||
              (FT_PaintFormat_Internal)apaint->format ==
                 FT_COLR_PAINTFORMAT_INTERNAL_VAR_SOLID   )
    {
      ENSURE_READ_BYTES( 4 );
      apaint->u.solid.color.palette_index = FT_NEXT_USHORT( p );
      apaint->u.solid.color.alpha         = FT_NEXT_SHORT( p );

#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT
      if ( (FT_PaintFormat_Internal)apaint->format ==
              FT_COLR_PAINTFORMAT_INTERNAL_VAR_SOLID )
      {
        ENSURE_READ_BYTES( 4 );
        var_index_base = FT_NEXT_ULONG( p );

        if ( !get_deltas_for_var_index_base( face, colr, var_index_base, 1,
                                             item_deltas ) )
          return 0;

        apaint->u.solid.color.alpha += (FT_F2Dot14)item_deltas[0];
      }
#endif

      apaint->format = FT_COLR_PAINTFORMAT_SOLID;

      return 1;
    }

    else if ( apaint->format == FT_COLR_PAINTFORMAT_COLR_GLYPH )
    {
      ENSURE_READ_BYTES(2);
      apaint->u.colr_glyph.glyphID = FT_NEXT_USHORT( p );

      return 1;
    }

    /*
     * Grouped below here are all paint formats that have an offset to a
     * child paint table as the first entry (for example, a color line or a
     * child paint table).  Retrieve that and determine whether that paint
     * offset is valid first.
     */

    if ( !get_child_table_pointer( colr, paint_base, &p, &child_table_p ) )
      return 0;

    if ( apaint->format == FT_COLR_PAINTFORMAT_LINEAR_GRADIENT      ||
         ( do_read_var =
             ( (FT_PaintFormat_Internal)apaint->format ==
               FT_COLR_PAINTFORMAT_INTERNAL_VAR_LINEAR_GRADIENT ) ) )
    {
      if ( !read_color_line( colr,
                             child_table_p,
                             &apaint->u.linear_gradient.colorline,
                             do_read_var ) )
        return 0;

      /*
       * In order to support variations expose these as FT_Fixed 16.16
       * values so that we can support fractional values after
       * interpolation.
       */
      ENSURE_READ_BYTES( 12 );
      apaint->u.linear_gradient.p0.x = INT_TO_FIXED( FT_NEXT_SHORT( p ) );
      apaint->u.linear_gradient.p0.y = INT_TO_FIXED( FT_NEXT_SHORT( p ) );
      apaint->u.linear_gradient.p1.x = INT_TO_FIXED( FT_NEXT_SHORT( p ) );
      apaint->u.linear_gradient.p1.y = INT_TO_FIXED( FT_NEXT_SHORT( p ) );
      apaint->u.linear_gradient.p2.x = INT_TO_FIXED( FT_NEXT_SHORT( p ) );
      apaint->u.linear_gradient.p2.y = INT_TO_FIXED( FT_NEXT_SHORT( p ) );

#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT
      if ( do_read_var )
      {
        ENSURE_READ_BYTES( 4 );
        var_index_base = FT_NEXT_ULONG ( p );

        if ( !get_deltas_for_var_index_base( face, colr, var_index_base, 6,
                                             item_deltas ) )
          return 0;

        apaint->u.linear_gradient.p0.x += INT_TO_FIXED( item_deltas[0] );
        apaint->u.linear_gradient.p0.y += INT_TO_FIXED( item_deltas[1] );
        apaint->u.linear_gradient.p1.x += INT_TO_FIXED( item_deltas[2] );
        apaint->u.linear_gradient.p1.y += INT_TO_FIXED( item_deltas[3] );
        apaint->u.linear_gradient.p2.x += INT_TO_FIXED( item_deltas[4] );
        apaint->u.linear_gradient.p2.y += INT_TO_FIXED( item_deltas[5] );
      }
#endif

      apaint->format = FT_COLR_PAINTFORMAT_LINEAR_GRADIENT;

      return 1;
    }

    else if ( apaint->format == FT_COLR_PAINTFORMAT_RADIAL_GRADIENT      ||
              ( do_read_var =
                  ( (FT_PaintFormat_Internal)apaint->format ==
                    FT_COLR_PAINTFORMAT_INTERNAL_VAR_RADIAL_GRADIENT ) ) )
    {
      FT_Pos  tmp;


      if ( !read_color_line( colr,
                             child_table_p,
                             &apaint->u.radial_gradient.colorline,
                             do_read_var ) )
        return 0;


      /* In the OpenType specification, `r0` and `r1` are defined as   */
      /* `UFWORD`.  Since FreeType doesn't have a corresponding 16.16  */
      /* format we convert to `FWORD` and replace negative values with */
      /* (32bit) `FT_INT_MAX`.                                         */

      ENSURE_READ_BYTES( 12 );

      apaint->u.radial_gradient.c0.x = INT_TO_FIXED( FT_NEXT_SHORT( p ) );
      apaint->u.radial_gradient.c0.y = INT_TO_FIXED( FT_NEXT_SHORT( p ) );

      tmp                          = INT_TO_FIXED( FT_NEXT_SHORT( p ) );
      apaint->u.radial_gradient.r0 = tmp < 0 ? FT_INT_MAX : tmp;

      apaint->u.radial_gradient.c1.x = INT_TO_FIXED( FT_NEXT_SHORT( p ) );
      apaint->u.radial_gradient.c1.y = INT_TO_FIXED( FT_NEXT_SHORT( p ) );

      tmp                          = INT_TO_FIXED( FT_NEXT_SHORT( p ) );
      apaint->u.radial_gradient.r1 = tmp < 0 ? FT_INT_MAX : tmp;

#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT
      if ( do_read_var )
      {
        ENSURE_READ_BYTES( 4 );
        var_index_base = FT_NEXT_ULONG ( p );

        if ( !get_deltas_for_var_index_base( face, colr, var_index_base, 6,
                                             item_deltas ) )
          return 0;

        apaint->u.radial_gradient.c0.x += INT_TO_FIXED( item_deltas[0] );
        apaint->u.radial_gradient.c0.y += INT_TO_FIXED( item_deltas[1] );

        // TODO: Anything to be done about UFWORD deltas here?
        apaint->u.radial_gradient.r0 += INT_TO_FIXED( item_deltas[2] );

        apaint->u.radial_gradient.c1.x += INT_TO_FIXED( item_deltas[3] );
        apaint->u.radial_gradient.c1.y += INT_TO_FIXED( item_deltas[4] );

        apaint->u.radial_gradient.r1 += INT_TO_FIXED( item_deltas[5] );
      }
#endif

      apaint->format = FT_COLR_PAINTFORMAT_RADIAL_GRADIENT;

      return 1;
    }

    else if ( apaint->format == FT_COLR_PAINTFORMAT_SWEEP_GRADIENT      ||
              ( do_read_var =
                  ( (FT_PaintFormat_Internal)apaint->format ==
                    FT_COLR_PAINTFORMAT_INTERNAL_VAR_SWEEP_GRADIENT ) ) )
    {
      if ( !read_color_line( colr,
                             child_table_p,
                             &apaint->u.sweep_gradient.colorline,
                             do_read_var) )
        return 0;

      ENSURE_READ_BYTES( 8 );

      apaint->u.sweep_gradient.center.x =
          INT_TO_FIXED( FT_NEXT_SHORT( p ) );
      apaint->u.sweep_gradient.center.y =
          INT_TO_FIXED( FT_NEXT_SHORT( p ) );

      apaint->u.sweep_gradient.start_angle =
          F2DOT14_TO_FIXED( FT_NEXT_SHORT( p ) );
      apaint->u.sweep_gradient.end_angle =
          F2DOT14_TO_FIXED( FT_NEXT_SHORT( p ) );

#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT
      if ( do_read_var )
      {
        ENSURE_READ_BYTES( 4 );
        var_index_base = FT_NEXT_ULONG ( p );

        if ( !get_deltas_for_var_index_base( face, colr, var_index_base, 4,
                                             item_deltas ) )
          return 0;

        // TODO: Handle overflow?
        apaint->u.sweep_gradient.center.x += INT_TO_FIXED( item_deltas[0] );
        apaint->u.sweep_gradient.center.y += INT_TO_FIXED( item_deltas[1] );

        apaint->u.sweep_gradient.start_angle +=
          F2DOT14_TO_FIXED( item_deltas[2] );
        apaint->u.sweep_gradient.end_angle +=
          F2DOT14_TO_FIXED( item_deltas[3] );
      }
#endif
      apaint->format = FT_COLR_PAINTFORMAT_SWEEP_GRADIENT;

      return 1;
    }

    if ( apaint->format == FT_COLR_PAINTFORMAT_GLYPH )
    {
      ENSURE_READ_BYTES( 2 );
      apaint->u.glyph.paint.p                     = child_table_p;
      apaint->u.glyph.paint.insert_root_transform = 0;
      apaint->u.glyph.glyphID                     = FT_NEXT_USHORT( p );

      return 1;
    }

    else if ( apaint->format == FT_COLR_PAINTFORMAT_TRANSFORM ||
              (FT_PaintFormat_Internal)apaint->format ==
                FT_COLR_PAINTFORMAT_INTERNAL_VAR_TRANSFORM    )
    {
      apaint->u.transform.paint.p                     = child_table_p;
      apaint->u.transform.paint.insert_root_transform = 0;

      if ( !get_child_table_pointer( colr, paint_base, &p, &child_table_p ) )
         return 0;

      p = child_table_p;

      /*
       * The following matrix coefficients are encoded as
       * OpenType 16.16 fixed-point values.
       */
      ENSURE_READ_BYTES( 24 );
      apaint->u.transform.affine.xx = FT_NEXT_LONG( p );
      apaint->u.transform.affine.yx = FT_NEXT_LONG( p );
      apaint->u.transform.affine.xy = FT_NEXT_LONG( p );
      apaint->u.transform.affine.yy = FT_NEXT_LONG( p );
      apaint->u.transform.affine.dx = FT_NEXT_LONG( p );
      apaint->u.transform.affine.dy = FT_NEXT_LONG( p );

#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT
      if ( (FT_PaintFormat_Internal)apaint->format ==
             FT_COLR_PAINTFORMAT_INTERNAL_VAR_TRANSFORM )
      {
        ENSURE_READ_BYTES( 4 );
        var_index_base = FT_NEXT_ULONG( p );

        if ( !get_deltas_for_var_index_base( face, colr, var_index_base, 6,
                                             item_deltas ) )
          return 0;

        apaint->u.transform.affine.xx += (FT_Fixed)item_deltas[0];
        apaint->u.transform.affine.yx += (FT_Fixed)item_deltas[1];
        apaint->u.transform.affine.xy += (FT_Fixed)item_deltas[2];
        apaint->u.transform.affine.yy += (FT_Fixed)item_deltas[3];
        apaint->u.transform.affine.dx += (FT_Fixed)item_deltas[4];
        apaint->u.transform.affine.dy += (FT_Fixed)item_deltas[5];
      }
#endif

      apaint->format = FT_COLR_PAINTFORMAT_TRANSFORM;

      return 1;
    }

    else if ( apaint->format == FT_COLR_PAINTFORMAT_TRANSLATE ||
              (FT_PaintFormat_Internal)apaint->format ==
                FT_COLR_PAINTFORMAT_INTERNAL_VAR_TRANSLATE    )
    {
      apaint->u.translate.paint.p                     = child_table_p;
      apaint->u.translate.paint.insert_root_transform = 0;

      ENSURE_READ_BYTES( 4 );
      apaint->u.translate.dx = INT_TO_FIXED( FT_NEXT_SHORT( p ) );
      apaint->u.translate.dy = INT_TO_FIXED( FT_NEXT_SHORT( p ) );

#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT
      if ( (FT_PaintFormat_Internal)apaint->format ==
             FT_COLR_PAINTFORMAT_INTERNAL_VAR_TRANSLATE )
      {
        ENSURE_READ_BYTES( 4 );
        var_index_base = FT_NEXT_ULONG( p );

        if ( !get_deltas_for_var_index_base( face, colr, var_index_base, 2,
                                             item_deltas ) )
          return 0;

        apaint->u.translate.dx += INT_TO_FIXED( item_deltas[0] );
        apaint->u.translate.dy += INT_TO_FIXED( item_deltas[1] );
      }
#endif

      apaint->format = FT_COLR_PAINTFORMAT_TRANSLATE;

      return 1;
    }

    else if ( apaint->format >= FT_COLR_PAINTFORMAT_SCALE             &&
              (FT_PaintFormat_Internal)apaint->format <=
                FT_COLR_PAINTFORMAT_INTERNAL_VAR_SCALE_UNIFORM_CENTER )
    {
      apaint->u.scale.paint.p                     = child_table_p;
      apaint->u.scale.paint.insert_root_transform = 0;

      /* All scale paints get at least one scale value. */
      ENSURE_READ_BYTES( 2 );
      apaint->u.scale.scale_x = F2DOT14_TO_FIXED( FT_NEXT_SHORT( p ) );

      /* Non-uniform ones read an extra y value. */
      if ( apaint->format == FT_COLR_PAINTFORMAT_SCALE     ||
           (FT_PaintFormat_Internal)apaint->format ==
             FT_COLR_PAINTFORMAT_INTERNAL_VAR_SCALE        ||
           (FT_PaintFormat_Internal)apaint->format ==
             FT_COLR_PAINTFORMAT_INTERNAL_SCALE_CENTER     ||
           (FT_PaintFormat_Internal)apaint->format ==
             FT_COLR_PAINTFORMAT_INTERNAL_VAR_SCALE_CENTER )
      {
        ENSURE_READ_BYTES( 2 );
        apaint->u.scale.scale_y = F2DOT14_TO_FIXED( FT_NEXT_SHORT( p ) );
      }
      else
        apaint->u.scale.scale_y = apaint->u.scale.scale_x;

      /* Scale paints that have a center read center coordinates, */
      /* otherwise the center is (0,0).                           */
      if ( (FT_PaintFormat_Internal)apaint->format ==
             FT_COLR_PAINTFORMAT_INTERNAL_SCALE_CENTER             ||
           (FT_PaintFormat_Internal)apaint->format ==
             FT_COLR_PAINTFORMAT_INTERNAL_VAR_SCALE_CENTER         ||
           (FT_PaintFormat_Internal)apaint->format ==
             FT_COLR_PAINTFORMAT_INTERNAL_SCALE_UNIFORM_CENTER     ||
           (FT_PaintFormat_Internal)apaint->format ==
             FT_COLR_PAINTFORMAT_INTERNAL_VAR_SCALE_UNIFORM_CENTER )
      {
        ENSURE_READ_BYTES( 4 );
        apaint->u.scale.center_x = INT_TO_FIXED( FT_NEXT_SHORT ( p ) );
        apaint->u.scale.center_y = INT_TO_FIXED( FT_NEXT_SHORT ( p ) );
      }
      else
      {
        apaint->u.scale.center_x = 0;
        apaint->u.scale.center_y = 0;
      }

      /* Base values set, now handle variations. */

#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT
      if ( (FT_PaintFormat_Internal)apaint->format ==
             FT_COLR_PAINTFORMAT_INTERNAL_VAR_SCALE                ||
           (FT_PaintFormat_Internal)apaint->format ==
             FT_COLR_PAINTFORMAT_INTERNAL_VAR_SCALE_CENTER         ||
           (FT_PaintFormat_Internal)apaint->format ==
             FT_COLR_PAINTFORMAT_INTERNAL_VAR_SCALE_UNIFORM        ||
           (FT_PaintFormat_Internal)apaint->format ==
             FT_COLR_PAINTFORMAT_INTERNAL_VAR_SCALE_UNIFORM_CENTER )
      {
        ENSURE_READ_BYTES( 4 );
        var_index_base = FT_NEXT_ULONG( p );

        if ( (FT_PaintFormat_Internal)apaint->format ==
               FT_COLR_PAINTFORMAT_INTERNAL_VAR_SCALE )
        {
          if ( !get_deltas_for_var_index_base( face, colr, var_index_base, 2,
                                               item_deltas ) )
            return 0;

          apaint->u.scale.scale_x += F2DOT14_TO_FIXED( item_deltas[0] );
          apaint->u.scale.scale_y += F2DOT14_TO_FIXED( item_deltas[1] );
        }

        if ( (FT_PaintFormat_Internal)apaint->format ==
               FT_COLR_PAINTFORMAT_INTERNAL_VAR_SCALE_CENTER )
        {
          if ( !get_deltas_for_var_index_base( face, colr, var_index_base, 4,
                                               item_deltas ) )
            return 0;

          apaint->u.scale.scale_x  += F2DOT14_TO_FIXED( item_deltas[0] );
          apaint->u.scale.scale_y  += F2DOT14_TO_FIXED( item_deltas[1] );
          apaint->u.scale.center_x += INT_TO_FIXED( item_deltas[2] );
          apaint->u.scale.center_y += INT_TO_FIXED( item_deltas[3] );
        }

        if ( (FT_PaintFormat_Internal)apaint->format ==
               FT_COLR_PAINTFORMAT_INTERNAL_VAR_SCALE_UNIFORM )
        {
          if ( !get_deltas_for_var_index_base( face, colr, var_index_base, 1,
                                               item_deltas ) )
            return 0;

          apaint->u.scale.scale_x += F2DOT14_TO_FIXED( item_deltas[0] );
          apaint->u.scale.scale_y += F2DOT14_TO_FIXED( item_deltas[0] );
        }

        if ( (FT_PaintFormat_Internal)apaint->format ==
               FT_COLR_PAINTFORMAT_INTERNAL_VAR_SCALE_UNIFORM_CENTER )
        {
          if ( !get_deltas_for_var_index_base( face, colr, var_index_base, 3,
                                               item_deltas ) )
            return 0;

          apaint->u.scale.scale_x  += F2DOT14_TO_FIXED( item_deltas[0] );
          apaint->u.scale.scale_y  += F2DOT14_TO_FIXED( item_deltas[0] );
          apaint->u.scale.center_x += INT_TO_FIXED( item_deltas[1] );
          apaint->u.scale.center_y += INT_TO_FIXED( item_deltas[2] );
        }
      }
#endif

      /* FT 'COLR' v1 API output format always returns fully defined */
      /* structs; we thus set the format to the public API value.    */
      apaint->format = FT_COLR_PAINTFORMAT_SCALE;

      return 1;
    }

    else if ( apaint->format == FT_COLR_PAINTFORMAT_ROTATE     ||
              (FT_PaintFormat_Internal)apaint->format ==
                FT_COLR_PAINTFORMAT_INTERNAL_ROTATE_CENTER     ||
              (FT_PaintFormat_Internal)apaint->format ==
                FT_COLR_PAINTFORMAT_INTERNAL_VAR_ROTATE        ||
              (FT_PaintFormat_Internal)apaint->format ==
                FT_COLR_PAINTFORMAT_INTERNAL_VAR_ROTATE_CENTER )
    {
      apaint->u.rotate.paint.p                     = child_table_p;
      apaint->u.rotate.paint.insert_root_transform = 0;

      ENSURE_READ_BYTES( 2 );
      apaint->u.rotate.angle = F2DOT14_TO_FIXED( FT_NEXT_SHORT( p ) );

      if ( (FT_PaintFormat_Internal)apaint->format ==
             FT_COLR_PAINTFORMAT_INTERNAL_ROTATE_CENTER     ||
           (FT_PaintFormat_Internal)apaint->format ==
             FT_COLR_PAINTFORMAT_INTERNAL_VAR_ROTATE_CENTER )
      {
        ENSURE_READ_BYTES( 4 );
        apaint->u.rotate.center_x = INT_TO_FIXED( FT_NEXT_SHORT( p ) );
        apaint->u.rotate.center_y = INT_TO_FIXED( FT_NEXT_SHORT( p ) );
      }
      else
      {
        apaint->u.rotate.center_x = 0;
        apaint->u.rotate.center_y = 0;
      }

#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT
      if ( (FT_PaintFormat_Internal)apaint->format ==
             FT_COLR_PAINTFORMAT_INTERNAL_VAR_ROTATE        ||
           (FT_PaintFormat_Internal)apaint->format ==
             FT_COLR_PAINTFORMAT_INTERNAL_VAR_ROTATE_CENTER )
      {
        FT_UInt  num_deltas = 0;


        ENSURE_READ_BYTES( 4 );
        var_index_base = FT_NEXT_ULONG( p );

        if ( (FT_PaintFormat_Internal)apaint->format ==
               FT_COLR_PAINTFORMAT_INTERNAL_VAR_ROTATE_CENTER )
          num_deltas = 3;
        if ( (FT_PaintFormat_Internal)apaint->format ==
               FT_COLR_PAINTFORMAT_INTERNAL_VAR_ROTATE )
          num_deltas = 1;

        if ( num_deltas > 0 )
        {
          if ( !get_deltas_for_var_index_base( face, colr, var_index_base,
                                               num_deltas, item_deltas ) )
            return 0;

          apaint->u.rotate.angle += F2DOT14_TO_FIXED( item_deltas[0] );

          if ( num_deltas == 3 )
          {
            apaint->u.rotate.center_x += INT_TO_FIXED( item_deltas[1] );
            apaint->u.rotate.center_y += INT_TO_FIXED( item_deltas[2] );
          }
        }
      }
#endif

      apaint->format = FT_COLR_PAINTFORMAT_ROTATE;


      return 1;
    }

    else if ( apaint->format == FT_COLR_PAINTFORMAT_SKEW     ||
              (FT_PaintFormat_Internal)apaint->format ==
                FT_COLR_PAINTFORMAT_INTERNAL_VAR_SKEW        ||
              (FT_PaintFormat_Internal)apaint->format ==
                FT_COLR_PAINTFORMAT_INTERNAL_SKEW_CENTER     ||
              (FT_PaintFormat_Internal)apaint->format ==
                FT_COLR_PAINTFORMAT_INTERNAL_VAR_SKEW_CENTER )
    {
      apaint->u.skew.paint.p                     = child_table_p;
      apaint->u.skew.paint.insert_root_transform = 0;

      ENSURE_READ_BYTES( 4 );
      apaint->u.skew.x_skew_angle = F2DOT14_TO_FIXED( FT_NEXT_SHORT( p ) );
      apaint->u.skew.y_skew_angle = F2DOT14_TO_FIXED( FT_NEXT_SHORT( p ) );

      if ( (FT_PaintFormat_Internal)apaint->format ==
             FT_COLR_PAINTFORMAT_INTERNAL_SKEW_CENTER     ||
           (FT_PaintFormat_Internal)apaint->format ==
             FT_COLR_PAINTFORMAT_INTERNAL_VAR_SKEW_CENTER )
      {
        ENSURE_READ_BYTES( 4 );
        apaint->u.skew.center_x = INT_TO_FIXED( FT_NEXT_SHORT( p ) );
        apaint->u.skew.center_y = INT_TO_FIXED( FT_NEXT_SHORT( p ) );
      }
      else
      {
        apaint->u.skew.center_x = 0;
        apaint->u.skew.center_y = 0;
      }


#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT
      if ( (FT_PaintFormat_Internal)apaint->format ==
             FT_COLR_PAINTFORMAT_INTERNAL_VAR_SKEW        ||
           (FT_PaintFormat_Internal)apaint->format ==
             FT_COLR_PAINTFORMAT_INTERNAL_VAR_SKEW_CENTER )
      {
        ENSURE_READ_BYTES( 4 );
        var_index_base = FT_NEXT_ULONG( p );

        if ( (FT_PaintFormat_Internal)apaint->format ==
               FT_COLR_PAINTFORMAT_INTERNAL_VAR_SKEW )
        {
          if ( !get_deltas_for_var_index_base( face, colr, var_index_base, 2,
                                               item_deltas ) )
            return 0;

          apaint->u.skew.x_skew_angle += F2DOT14_TO_FIXED( item_deltas[0] );
          apaint->u.skew.y_skew_angle += F2DOT14_TO_FIXED( item_deltas[1] );
        }

        if ( (FT_PaintFormat_Internal)apaint->format ==
               FT_COLR_PAINTFORMAT_INTERNAL_VAR_SKEW_CENTER )
        {
          if ( !get_deltas_for_var_index_base( face, colr, var_index_base, 4,
                                               item_deltas ) )
            return 0;

          apaint->u.skew.x_skew_angle += F2DOT14_TO_FIXED( item_deltas[0] );
          apaint->u.skew.y_skew_angle += F2DOT14_TO_FIXED( item_deltas[1] );
          apaint->u.skew.center_x     += INT_TO_FIXED( item_deltas[2] );
          apaint->u.skew.center_y     += INT_TO_FIXED( item_deltas[3] );
        }
      }
#endif

      apaint->format = FT_COLR_PAINTFORMAT_SKEW;

      return 1;
    }

    else if ( apaint->format == FT_COLR_PAINTFORMAT_COMPOSITE )
    {
      FT_UInt  composite_mode;


      apaint->u.composite.source_paint.p                     = child_table_p;
      apaint->u.composite.source_paint.insert_root_transform = 0;

      ENSURE_READ_BYTES( 1 );
      composite_mode = FT_NEXT_BYTE( p );
      if ( composite_mode >= FT_COLR_COMPOSITE_MAX )
        return 0;

      apaint->u.composite.composite_mode = (FT_Composite_Mode)composite_mode;

      if ( !get_child_table_pointer( colr, paint_base, &p, &child_table_p ) )
         return 0;

      apaint->u.composite.backdrop_paint.p =
        child_table_p;
      apaint->u.composite.backdrop_paint.insert_root_transform =
        0;

      return 1;
    }

    return 0;
  }


  static FT_Bool
  find_base_glyph_v1_record( FT_Byte *           base_glyph_begin,
                             FT_UInt             num_base_glyph,
                             FT_UInt             glyph_id,
                             BaseGlyphV1Record  *record )
  {
    FT_UInt  min = 0;
    FT_UInt  max = num_base_glyph;


    while ( min < max )
    {
      FT_UInt  mid = min + ( max - min ) / 2;

      /*
       * `base_glyph_begin` is the beginning of `BaseGlyphV1List`;
       * skip `numBaseGlyphV1Records` by adding 4 to start binary search
       * in the array of `BaseGlyphV1Record`.
       */
      FT_Byte  *p = base_glyph_begin + 4 + mid * BASE_GLYPH_PAINT_RECORD_SIZE;

      FT_UShort  gid = FT_NEXT_USHORT( p );


      if ( gid < glyph_id )
        min = mid + 1;
      else if (gid > glyph_id )
        max = mid;
      else
      {
        record->gid          = gid;
        record->paint_offset = FT_NEXT_ULONG ( p );
        return 1;
      }
    }

    return 0;
  }


  FT_LOCAL_DEF( FT_Bool )
  tt_face_get_colr_glyph_paint( TT_Face                  face,
                                FT_UInt                  base_glyph,
                                FT_Color_Root_Transform  root_transform,
                                FT_OpaquePaint*          opaque_paint )
  {
    Colr*              colr = (Colr*)face->colr;
    BaseGlyphV1Record  base_glyph_v1_record;
    FT_Byte*           p;

    if ( !colr || !colr->table )
      return 0;

    if ( colr->version < 1 || !colr->num_base_glyphs_v1 ||
         !colr->base_glyphs_v1 )
      return 0;

    if ( opaque_paint->p )
      return 0;

    if ( !find_base_glyph_v1_record( colr->base_glyphs_v1,
                                     colr->num_base_glyphs_v1,
                                     base_glyph,
                                     &base_glyph_v1_record ) )
      return 0;

    if ( !base_glyph_v1_record.paint_offset                   ||
         base_glyph_v1_record.paint_offset > colr->table_size )
      return 0;

    p = (FT_Byte*)( colr->base_glyphs_v1 +
                    base_glyph_v1_record.paint_offset );
    if ( p >= ( (FT_Byte*)colr->table + colr->table_size ) )
      return 0;

    opaque_paint->p = p;

    if ( root_transform == FT_COLOR_INCLUDE_ROOT_TRANSFORM )
      opaque_paint->insert_root_transform = 1;
    else
      opaque_paint->insert_root_transform = 0;

    return 1;
  }


  FT_LOCAL_DEF( FT_Bool )
  tt_face_get_color_glyph_clipbox( TT_Face      face,
                                   FT_UInt      base_glyph,
                                   FT_ClipBox*  clip_box )
  {
    Colr*  colr;

    FT_Byte  *p, *p1, *clip_base, *limit;

    FT_Byte    clip_list_format;
    FT_ULong   num_clip_boxes, i;
    FT_UShort  gid_start, gid_end;
    FT_UInt32  clip_box_offset;
    FT_Byte    format;

    const FT_Byte  num_corners = 4;
    FT_Vector      corners[4];
    FT_Byte        j;
    FT_BBox        font_clip_box;


    colr = (Colr*)face->colr;
    if ( !colr )
      return 0;

    if ( !colr->clip_list )
      return 0;

    p = colr->clip_list;

    /* Limit points to the first byte after the end of the color table.    */
    /* Thus, in subsequent limit checks below we need to check whether the */
    /* read pointer is strictly greater than a position offset by certain  */
    /* field sizes to the left of that position.                           */
    limit = (FT_Byte*)colr->table + colr->table_size;

    /* Check whether we can extract one `uint8` and one `uint32`. */
    if ( p > limit - ( 1 + 4 ) )
      return 0;

    clip_base        = p;
    clip_list_format = FT_NEXT_BYTE ( p );

    /* Format byte used here to be able to upgrade ClipList for >16bit */
    /* glyph ids; for now we can expect it to be 1.                    */
    if ( !( clip_list_format == 1 ) )
      return 0;

    num_clip_boxes = FT_NEXT_ULONG( p );

    /* Check whether we can extract two `uint16` and one `Offset24`, */
    /* `num_clip_boxes` times.                                       */
    if ( colr->table_size / ( 2 + 2 + 3 ) < num_clip_boxes ||
         p > limit - ( 2 + 2 + 3 ) * num_clip_boxes        )
      return 0;

    for ( i = 0; i < num_clip_boxes; ++i )
    {
      gid_start       = FT_NEXT_USHORT( p );
      gid_end         = FT_NEXT_USHORT( p );
      clip_box_offset = FT_NEXT_UOFF3( p );

      if ( base_glyph >= gid_start && base_glyph <= gid_end )
      {
        p1 = (FT_Byte*)( clip_base + clip_box_offset );

        /* Check whether we can extract one `uint8`. */
        if ( p1 > limit - 1 )
          return 0;

        format = FT_NEXT_BYTE( p1 );

        if ( format > 2 )
          return 0;

        /* Check whether we can extract four `FWORD`. */
        if ( p1 > limit - ( 2 + 2 + 2 + 2 ) )
          return 0;

        /* `face->root.size->metrics.x_scale` and `y_scale` are factors   */
        /* that scale a font unit value in integers to a 26.6 fixed value */
        /* according to the requested size, see for example               */
        /* `ft_recompute_scaled_metrics`.                                 */
        font_clip_box.xMin = FT_MulFix( FT_NEXT_SHORT( p1 ),
                                        face->root.size->metrics.x_scale );
        font_clip_box.yMin = FT_MulFix( FT_NEXT_SHORT( p1 ),
                                        face->root.size->metrics.y_scale );
        font_clip_box.xMax = FT_MulFix( FT_NEXT_SHORT( p1 ),
                                        face->root.size->metrics.x_scale );
        font_clip_box.yMax = FT_MulFix( FT_NEXT_SHORT( p1 ),
                                        face->root.size->metrics.y_scale );

#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT
        if ( format == 2 )
        {
          FT_ULong         var_index_base = 0;
          /* varIndexBase offset for clipbox is 3 at most. */
          FT_ItemVarDelta  item_deltas[4] = { 0, 0, 0, 0 };


          /* Check whether we can extract a 32-bit varIndexBase now. */
          if ( p1 > limit - 4 )
            return 0;

          var_index_base = FT_NEXT_ULONG( p1 );

          if ( !get_deltas_for_var_index_base( face, colr, var_index_base, 4,
                                               item_deltas ) )
            return 0;

          font_clip_box.xMin +=
            FT_MulFix( item_deltas[0], face->root.size->metrics.x_scale );
          font_clip_box.yMin +=
            FT_MulFix( item_deltas[1], face->root.size->metrics.y_scale );
          font_clip_box.xMax +=
            FT_MulFix( item_deltas[2], face->root.size->metrics.x_scale );
          font_clip_box.yMax +=
            FT_MulFix( item_deltas[3], face->root.size->metrics.y_scale );
        }
#endif

        /* Make 4 corner points (xMin, yMin), (xMax, yMax) and transform */
        /* them.  If we we would only transform two corner points and    */
        /* span a rectangle based on those, the rectangle may become too */
        /* small to cover the glyph.                                     */
        corners[0].x = font_clip_box.xMin;
        corners[1].x = font_clip_box.xMin;
        corners[2].x = font_clip_box.xMax;
        corners[3].x = font_clip_box.xMax;

        corners[0].y = font_clip_box.yMin;
        corners[1].y = font_clip_box.yMax;
        corners[2].y = font_clip_box.yMax;
        corners[3].y = font_clip_box.yMin;

        for ( j = 0; j < num_corners; ++j )
        {
          if ( face->root.internal->transform_flags & 1 )
            FT_Vector_Transform( &corners[j],
                                 &face->root.internal->transform_matrix );

          if ( face->root.internal->transform_flags & 2 )
          {
            corners[j].x += face->root.internal->transform_delta.x;
            corners[j].y += face->root.internal->transform_delta.y;
          }
        }

        clip_box->bottom_left  = corners[0];
        clip_box->top_left     = corners[1];
        clip_box->top_right    = corners[2];
        clip_box->bottom_right = corners[3];

        return 1;
      }
    }

    return 0;
  }


  FT_LOCAL_DEF( FT_Bool )
  tt_face_get_paint_layers( TT_Face            face,
                            FT_LayerIterator*  iterator,
                            FT_OpaquePaint*    opaque_paint )
  {
    FT_Byte*   p             = NULL;
    FT_Byte*   p_first_layer = NULL;
    FT_Byte*   p_paint       = NULL;
    FT_UInt32  paint_offset;

    Colr*  colr;


    if ( iterator->layer == iterator->num_layers )
      return 0;

    colr = (Colr*)face->colr;
    if ( !colr )
      return 0;

    /*
     * We have an iterator pointing at a paint offset as part of the
     * `paintOffset` array in `LayerV1List`.
     */
    p = iterator->p;

    /*
     * Do a cursor sanity check of the iterator.  Counting backwards from
     * where it stands, we need to end up at a position after the beginning
     * of the `LayerV1List` table and not after the end of the
     * `LayerV1List`.
     */
    p_first_layer = p -
                      iterator->layer * LAYER_V1_LIST_PAINT_OFFSET_SIZE -
                      LAYER_V1_LIST_NUM_LAYERS_SIZE;
    if ( p_first_layer < (FT_Byte*)colr->layers_v1 )
      return 0;
    if ( p_first_layer >= (FT_Byte*)(
           colr->layers_v1 + LAYER_V1_LIST_NUM_LAYERS_SIZE +
           colr->num_layers_v1 * LAYER_V1_LIST_PAINT_OFFSET_SIZE ) )
      return 0;

    /*
     * Before reading, ensure that `p` is within 'COLR' v1 and we can read a
     * 4-byte ULONG.
     */
    if ( p < colr->layers_v1                              ||
         p > (FT_Byte*)colr->table + colr->table_size - 4 )
      return 0;

    paint_offset =
      FT_NEXT_ULONG( p );
    opaque_paint->insert_root_transform =
      0;

    p_paint = (FT_Byte*)( colr->layers_v1 + paint_offset );

    if ( p_paint < colr->paints_start_v1                         ||
         p_paint >= ( (FT_Byte*)colr->table + colr->table_size ) )
      return 0;

    opaque_paint->p = p_paint;

    iterator->p = p;

    iterator->layer++;

    return 1;
  }


  FT_LOCAL_DEF( FT_Bool )
  tt_face_get_colorline_stops( TT_Face                face,
                               FT_ColorStop*          color_stop,
                               FT_ColorStopIterator  *iterator )
  {
    Colr*  colr = (Colr*)face->colr;

    FT_Byte*  p;
    FT_ULong  var_index_base;
    FT_Byte*  last_entry_p = NULL;
    FT_UInt   entry_size   = COLOR_STOP_SIZE;


    if ( !colr || !colr->table || !iterator )
      return 0;

    if ( iterator->current_color_stop >= iterator->num_color_stops )
      return 0;

    if ( iterator->read_variable )
      entry_size += VAR_IDX_BASE_SIZE;

    /* Calculate the start pointer for the last to-be-read (Var)ColorStop */
    /* and check whether we can read a full (Var)ColorStop at that        */
    /* position by comparing it to the position that is the size of one   */
    /* (Var)ColorStop before the end of the 'COLR' table.                 */
    last_entry_p =
      iterator->p + ( iterator->num_color_stops - 1 -
                      iterator->current_color_stop ) * entry_size;
    if ( iterator->p < colr->paints_start_v1          ||
         last_entry_p > (FT_Byte*)colr->table +
                        colr->table_size - entry_size )
      return 0;

    /* Iterator points at first `ColorStop` of `ColorLine`. */
    p = iterator->p;

    color_stop->stop_offset = F2DOT14_TO_FIXED( FT_NEXT_SHORT( p ) );

    color_stop->color.palette_index = FT_NEXT_USHORT( p );

    color_stop->color.alpha = FT_NEXT_SHORT( p );

    if ( iterator->read_variable )
    {
      /* Pointer p needs to be advanced independently of whether we intend */
      /* to take variable deltas into account or not.  Otherwise iteration */
      /* would fail due to wrong offsets.                                  */
      var_index_base = FT_NEXT_ULONG( p );

#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT
      {
        FT_Int  item_deltas[2];


        if ( !get_deltas_for_var_index_base( face, colr,
                                             var_index_base,
                                             2,
                                             item_deltas ) )
          return 0;

        color_stop->stop_offset += F2DOT14_TO_FIXED( item_deltas[0] );
        color_stop->color.alpha += (FT_F2Dot14)item_deltas[1];
      }
#else
      FT_UNUSED( var_index_base );
#endif
    }

    iterator->p = p;
    iterator->current_color_stop++;

    return 1;
  }


  FT_LOCAL_DEF( FT_Bool )
  tt_face_get_paint( TT_Face         face,
                     FT_OpaquePaint  opaque_paint,
                     FT_COLR_Paint*  paint )
  {
    Colr*           colr = (Colr*)face->colr;
    FT_OpaquePaint  next_paint;
    FT_Matrix       ft_root_scale;

    if ( !colr || !colr->base_glyphs_v1 || !colr->table )
      return 0;

    if ( opaque_paint.insert_root_transform )
    {
      /* 'COLR' v1 glyph information is returned in unscaled coordinates,
       * i.e., `FT_Size` is not applied or multiplied into the values.  When
       * client applications draw color glyphs, they can request to include
       * a top-level transform, which includes the active `x_scale` and
       * `y_scale` information for scaling the glyph, as well the additional
       * transform and translate configured through `FT_Set_Transform`.
       * This allows client applications to apply this top-level transform
       * to the graphics context first and only once, then have gradient and
       * contour scaling applied correctly when performing the additional
       * drawing operations for subsequenct paints.  Prepare this initial
       * transform here.
       */
      paint->format = FT_COLR_PAINTFORMAT_TRANSFORM;

      next_paint.p                     = opaque_paint.p;
      next_paint.insert_root_transform = 0;
      paint->u.transform.paint         = next_paint;

      /* `x_scale` and `y_scale` are in 26.6 format, representing the scale
       * factor to get from font units to requested size.  However, expected
       * return values are in 16.16, so we shift accordingly with rounding.
       */
      ft_root_scale.xx = ( face->root.size->metrics.x_scale + 32 ) >> 6;
      ft_root_scale.xy = 0;
      ft_root_scale.yx = 0;
      ft_root_scale.yy = ( face->root.size->metrics.y_scale + 32 ) >> 6;

      if ( face->root.internal->transform_flags & 1 )
        FT_Matrix_Multiply( &face->root.internal->transform_matrix,
                            &ft_root_scale );

      paint->u.transform.affine.xx = ft_root_scale.xx;
      paint->u.transform.affine.xy = ft_root_scale.xy;
      paint->u.transform.affine.yx = ft_root_scale.yx;
      paint->u.transform.affine.yy = ft_root_scale.yy;

      /* The translation is specified in 26.6 format and, according to the
       * documentation of `FT_Set_Translate`, is performed on the character
       * size given in the last call to `FT_Set_Char_Size`.  The
       * 'PaintTransform' paint table's `FT_Affine23` format expects
       * values in 16.16 format, thus we need to shift by 10 bits.
       */
      if ( face->root.internal->transform_flags & 2 )
      {
        paint->u.transform.affine.dx =
          face->root.internal->transform_delta.x * ( 1 << 10 );
        paint->u.transform.affine.dy =
          face->root.internal->transform_delta.y * ( 1 << 10 );
      }
      else
      {
        paint->u.transform.affine.dx = 0;
        paint->u.transform.affine.dy = 0;
      }

      return 1;
    }

    return read_paint( face, colr, opaque_paint.p, paint );
  }


  FT_LOCAL_DEF( FT_Error )
  tt_face_colr_blend_layer( TT_Face       face,
                            FT_UInt       color_index,
                            FT_GlyphSlot  dstSlot,
                            FT_GlyphSlot  srcSlot )
  {
    FT_Error  error;

    FT_UInt  x, y;
    FT_Byte  b, g, r, alpha;

    FT_ULong  size;
    FT_Byte*  src;
    FT_Byte*  dst;


    if ( !dstSlot->bitmap.buffer )
    {
      /* Initialize destination of color bitmap */
      /* with the size of first component.      */
      dstSlot->bitmap_left = srcSlot->bitmap_left;
      dstSlot->bitmap_top  = srcSlot->bitmap_top;

      dstSlot->bitmap.width      = srcSlot->bitmap.width;
      dstSlot->bitmap.rows       = srcSlot->bitmap.rows;
      dstSlot->bitmap.pixel_mode = FT_PIXEL_MODE_BGRA;
      dstSlot->bitmap.pitch      = (int)dstSlot->bitmap.width * 4;
      dstSlot->bitmap.num_grays  = 256;

      size = dstSlot->bitmap.rows * (unsigned int)dstSlot->bitmap.pitch;

      error = ft_glyphslot_alloc_bitmap( dstSlot, size );
      if ( error )
        return error;

      FT_MEM_ZERO( dstSlot->bitmap.buffer, size );
    }
    else
    {
      /* Resize destination if needed such that new component fits. */
      FT_Int  x_min, x_max, y_min, y_max;


      x_min = FT_MIN( dstSlot->bitmap_left, srcSlot->bitmap_left );
      x_max = FT_MAX( dstSlot->bitmap_left + (FT_Int)dstSlot->bitmap.width,
                      srcSlot->bitmap_left + (FT_Int)srcSlot->bitmap.width );

      y_min = FT_MIN( dstSlot->bitmap_top - (FT_Int)dstSlot->bitmap.rows,
                      srcSlot->bitmap_top - (FT_Int)srcSlot->bitmap.rows );
      y_max = FT_MAX( dstSlot->bitmap_top, srcSlot->bitmap_top );

      if ( x_min != dstSlot->bitmap_left                                 ||
           x_max != dstSlot->bitmap_left + (FT_Int)dstSlot->bitmap.width ||
           y_min != dstSlot->bitmap_top - (FT_Int)dstSlot->bitmap.rows   ||
           y_max != dstSlot->bitmap_top                                  )
      {
        FT_Memory  memory = face->root.memory;

        FT_UInt  width = (FT_UInt)( x_max - x_min );
        FT_UInt  rows  = (FT_UInt)( y_max - y_min );
        FT_UInt  pitch = width * 4;

        FT_Byte*  buf = NULL;
        FT_Byte*  p;
        FT_Byte*  q;


        size  = rows * pitch;
        if ( FT_ALLOC( buf, size ) )
          return error;

        p = dstSlot->bitmap.buffer;
        q = buf +
            (int)pitch * ( y_max - dstSlot->bitmap_top ) +
            4 * ( dstSlot->bitmap_left - x_min );

        for ( y = 0; y < dstSlot->bitmap.rows; y++ )
        {
          FT_MEM_COPY( q, p, dstSlot->bitmap.width * 4 );

          p += dstSlot->bitmap.pitch;
          q += pitch;
        }

        ft_glyphslot_set_bitmap( dstSlot, buf );

        dstSlot->bitmap_top  = y_max;
        dstSlot->bitmap_left = x_min;

        dstSlot->bitmap.width = width;
        dstSlot->bitmap.rows  = rows;
        dstSlot->bitmap.pitch = (int)pitch;

        dstSlot->internal->flags |= FT_GLYPH_OWN_BITMAP;
        dstSlot->format           = FT_GLYPH_FORMAT_BITMAP;
      }
    }

    if ( color_index == 0xFFFF )
    {
      if ( face->have_foreground_color )
      {
        b     = face->foreground_color.blue;
        g     = face->foreground_color.green;
        r     = face->foreground_color.red;
        alpha = face->foreground_color.alpha;
      }
      else
      {
        if ( face->palette_data.palette_flags                          &&
             ( face->palette_data.palette_flags[face->palette_index] &
                 FT_PALETTE_FOR_DARK_BACKGROUND                      ) )
        {
          /* white opaque */
          b     = 0xFF;
          g     = 0xFF;
          r     = 0xFF;
          alpha = 0xFF;
        }
        else
        {
          /* black opaque */
          b     = 0x00;
          g     = 0x00;
          r     = 0x00;
          alpha = 0xFF;
        }
      }
    }
    else
    {
      b     = face->palette[color_index].blue;
      g     = face->palette[color_index].green;
      r     = face->palette[color_index].red;
      alpha = face->palette[color_index].alpha;
    }

    /* XXX Convert if srcSlot.bitmap is not grey? */
    src = srcSlot->bitmap.buffer;
    dst = dstSlot->bitmap.buffer +
          dstSlot->bitmap.pitch * ( dstSlot->bitmap_top - srcSlot->bitmap_top ) +
          4 * ( srcSlot->bitmap_left - dstSlot->bitmap_left );

    for ( y = 0; y < srcSlot->bitmap.rows; y++ )
    {
      for ( x = 0; x < srcSlot->bitmap.width; x++ )
      {
        int  aa = src[x];
        int  fa = alpha * aa / 255;

        int  fb = b * fa / 255;
        int  fg = g * fa / 255;
        int  fr = r * fa / 255;

        int  ba2 = 255 - fa;

        int  bb = dst[4 * x + 0];
        int  bg = dst[4 * x + 1];
        int  br = dst[4 * x + 2];
        int  ba = dst[4 * x + 3];


        dst[4 * x + 0] = (FT_Byte)( bb * ba2 / 255 + fb );
        dst[4 * x + 1] = (FT_Byte)( bg * ba2 / 255 + fg );
        dst[4 * x + 2] = (FT_Byte)( br * ba2 / 255 + fr );
        dst[4 * x + 3] = (FT_Byte)( ba * ba2 / 255 + fa );
      }

      src += srcSlot->bitmap.pitch;
      dst += dstSlot->bitmap.pitch;
    }

    return FT_Err_Ok;
  }

#else /* !TT_CONFIG_OPTION_COLOR_LAYERS */

  /* ANSI C doesn't like empty source files */
  typedef int  tt_colr_dummy_;

#endif /* !TT_CONFIG_OPTION_COLOR_LAYERS */

/* EOF */
