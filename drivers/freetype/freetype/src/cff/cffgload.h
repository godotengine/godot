/***************************************************************************/
/*                                                                         */
/*  cffgload.h                                                             */
/*                                                                         */
/*    OpenType Glyph Loader (specification).                               */
/*                                                                         */
/*  Copyright 1996-2004, 2006-2009, 2013 by                                */
/*  David Turner, Robert Wilhelm, and Werner Lemberg.                      */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/


#ifndef __CFFGLOAD_H__
#define __CFFGLOAD_H__


#include <ft2build.h>
#include FT_FREETYPE_H
#include "cffobjs.h"


FT_BEGIN_HEADER


#define CFF_MAX_OPERANDS        48
#define CFF_MAX_SUBRS_CALLS     32
#define CFF_MAX_TRANS_ELEMENTS  32


  /*************************************************************************/
  /*                                                                       */
  /* <Structure>                                                           */
  /*    CFF_Builder                                                        */
  /*                                                                       */
  /* <Description>                                                         */
  /*     A structure used during glyph loading to store its outline.       */
  /*                                                                       */
  /* <Fields>                                                              */
  /*    memory        :: The current memory object.                        */
  /*                                                                       */
  /*    face          :: The current face object.                          */
  /*                                                                       */
  /*    glyph         :: The current glyph slot.                           */
  /*                                                                       */
  /*    loader        :: The current glyph loader.                         */
  /*                                                                       */
  /*    base          :: The base glyph outline.                           */
  /*                                                                       */
  /*    current       :: The current glyph outline.                        */
  /*                                                                       */
  /*    pos_x         :: The horizontal translation (if composite glyph).  */
  /*                                                                       */
  /*    pos_y         :: The vertical translation (if composite glyph).    */
  /*                                                                       */
  /*    left_bearing  :: The left side bearing point.                      */
  /*                                                                       */
  /*    advance       :: The horizontal advance vector.                    */
  /*                                                                       */
  /*    bbox          :: Unused.                                           */
  /*                                                                       */
  /*    path_begun    :: A flag which indicates that a new path has begun. */
  /*                                                                       */
  /*    load_points   :: If this flag is not set, no points are loaded.    */
  /*                                                                       */
  /*    no_recurse    :: Set but not used.                                 */
  /*                                                                       */
  /*    metrics_only  :: A boolean indicating that we only want to compute */
  /*                     the metrics of a given glyph, not load all of its */
  /*                     points.                                           */
  /*                                                                       */
  /*    hints_funcs   :: Auxiliary pointer for hinting.                    */
  /*                                                                       */
  /*    hints_globals :: Auxiliary pointer for hinting.                    */
  /*                                                                       */
  typedef struct  CFF_Builder_
  {
    FT_Memory       memory;
    TT_Face         face;
    CFF_GlyphSlot   glyph;
    FT_GlyphLoader  loader;
    FT_Outline*     base;
    FT_Outline*     current;

    FT_Pos          pos_x;
    FT_Pos          pos_y;

    FT_Vector       left_bearing;
    FT_Vector       advance;

    FT_BBox         bbox;          /* bounding box */
    FT_Bool         path_begun;
    FT_Bool         load_points;
    FT_Bool         no_recurse;

    FT_Bool         metrics_only;

    void*           hints_funcs;    /* hinter-specific */
    void*           hints_globals;  /* hinter-specific */

  } CFF_Builder;


  FT_LOCAL( FT_Error )
  cff_check_points( CFF_Builder*  builder,
                    FT_Int        count );

  FT_LOCAL( void )
  cff_builder_add_point( CFF_Builder*  builder,
                         FT_Pos        x,
                         FT_Pos        y,
                         FT_Byte       flag );
  FT_LOCAL( FT_Error )
  cff_builder_add_point1( CFF_Builder*  builder,
                          FT_Pos        x,
                          FT_Pos        y );
  FT_LOCAL( FT_Error )
  cff_builder_start_point( CFF_Builder*  builder,
                           FT_Pos        x,
                           FT_Pos        y );
  FT_LOCAL( void )
  cff_builder_close_contour( CFF_Builder*  builder );


  FT_LOCAL( FT_Int )
  cff_lookup_glyph_by_stdcharcode( CFF_Font  cff,
                                   FT_Int    charcode );
  FT_LOCAL( FT_Error )
  cff_get_glyph_data( TT_Face    face,
                      FT_UInt    glyph_index,
                      FT_Byte**  pointer,
                      FT_ULong*  length );
  FT_LOCAL( void )
  cff_free_glyph_data( TT_Face    face,
                       FT_Byte**  pointer,
                       FT_ULong   length );


  /* execution context charstring zone */

  typedef struct  CFF_Decoder_Zone_
  {
    FT_Byte*  base;
    FT_Byte*  limit;
    FT_Byte*  cursor;

  } CFF_Decoder_Zone;


  typedef struct  CFF_Decoder_
  {
    CFF_Builder        builder;
    CFF_Font           cff;

    FT_Fixed           stack[CFF_MAX_OPERANDS + 1];
    FT_Fixed*          top;

    CFF_Decoder_Zone   zones[CFF_MAX_SUBRS_CALLS + 1];
    CFF_Decoder_Zone*  zone;

    FT_Int             flex_state;
    FT_Int             num_flex_vectors;
    FT_Vector          flex_vectors[7];

    FT_Pos             glyph_width;
    FT_Pos             nominal_width;

    FT_Bool            read_width;
    FT_Bool            width_only;
    FT_Int             num_hints;
    FT_Fixed           buildchar[CFF_MAX_TRANS_ELEMENTS];

    FT_UInt            num_locals;
    FT_UInt            num_globals;

    FT_Int             locals_bias;
    FT_Int             globals_bias;

    FT_Byte**          locals;
    FT_Byte**          globals;

    FT_Byte**          glyph_names;   /* for pure CFF fonts only  */
    FT_UInt            num_glyphs;    /* number of glyphs in font */

    FT_Render_Mode     hint_mode;

    FT_Bool            seac;

    CFF_SubFont        current_subfont; /* for current glyph_index */

  } CFF_Decoder;


  FT_LOCAL( void )
  cff_decoder_init( CFF_Decoder*    decoder,
                    TT_Face         face,
                    CFF_Size        size,
                    CFF_GlyphSlot   slot,
                    FT_Bool         hinting,
                    FT_Render_Mode  hint_mode );

  FT_LOCAL( FT_Error )
  cff_decoder_prepare( CFF_Decoder*  decoder,
                       CFF_Size      size,
                       FT_UInt       glyph_index );

#if 0  /* unused until we support pure CFF fonts */

  /* Compute the maximum advance width of a font through quick parsing */
  FT_LOCAL( FT_Error )
  cff_compute_max_advance( TT_Face  face,
                           FT_Int*  max_advance );

#endif /* 0 */

#ifdef CFF_CONFIG_OPTION_OLD_ENGINE
  FT_LOCAL( FT_Error )
  cff_decoder_parse_charstrings( CFF_Decoder*  decoder,
                                 FT_Byte*      charstring_base,
                                 FT_ULong      charstring_len );
#endif

  FT_LOCAL( FT_Error )
  cff_slot_load( CFF_GlyphSlot  glyph,
                 CFF_Size       size,
                 FT_UInt        glyph_index,
                 FT_Int32       load_flags );


FT_END_HEADER

#endif /* __CFFGLOAD_H__ */


/* END */
