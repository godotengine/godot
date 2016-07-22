/***************************************************************************/
/*                                                                         */
/*  pfrobjs.c                                                              */
/*                                                                         */
/*    FreeType PFR object methods (body).                                  */
/*                                                                         */
/*  Copyright 2002-2016 by                                                 */
/*  David Turner, Robert Wilhelm, and Werner Lemberg.                      */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/


#include "pfrobjs.h"
#include "pfrload.h"
#include "pfrgload.h"
#include "pfrcmap.h"
#include "pfrsbit.h"
#include FT_OUTLINE_H
#include FT_INTERNAL_DEBUG_H
#include FT_INTERNAL_CALC_H
#include FT_TRUETYPE_IDS_H

#include "pfrerror.h"

#undef  FT_COMPONENT
#define FT_COMPONENT  trace_pfr


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                     FACE OBJECT METHODS                       *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  FT_LOCAL_DEF( void )
  pfr_face_done( FT_Face  pfrface )     /* PFR_Face */
  {
    PFR_Face   face = (PFR_Face)pfrface;
    FT_Memory  memory;


    if ( !face )
      return;

    memory = pfrface->driver->root.memory;

    /* we don't want dangling pointers */
    pfrface->family_name = NULL;
    pfrface->style_name  = NULL;

    /* finalize the physical font record */
    pfr_phy_font_done( &face->phy_font, FT_FACE_MEMORY( face ) );

    /* no need to finalize the logical font or the header */
    FT_FREE( pfrface->available_sizes );
  }


  FT_LOCAL_DEF( FT_Error )
  pfr_face_init( FT_Stream      stream,
                 FT_Face        pfrface,
                 FT_Int         face_index,
                 FT_Int         num_params,
                 FT_Parameter*  params )
  {
    PFR_Face  face = (PFR_Face)pfrface;
    FT_Error  error;

    FT_UNUSED( num_params );
    FT_UNUSED( params );


    FT_TRACE2(( "PFR driver\n" ));

    /* load the header and check it */
    error = pfr_header_load( &face->header, stream );
    if ( error )
      goto Exit;

    if ( !pfr_header_check( &face->header ) )
    {
      FT_TRACE2(( "  not a PFR font\n" ));
      error = FT_THROW( Unknown_File_Format );
      goto Exit;
    }

    /* check face index */
    {
      FT_Long  num_faces;


      error = pfr_log_font_count( stream,
                                  face->header.log_dir_offset,
                                  &num_faces );
      if ( error )
        goto Exit;

      pfrface->num_faces = num_faces;
    }

    if ( face_index < 0 )
      goto Exit;

    if ( ( face_index & 0xFFFF ) >= pfrface->num_faces )
    {
      FT_ERROR(( "pfr_face_init: invalid face index\n" ));
      error = FT_THROW( Invalid_Argument );
      goto Exit;
    }

    /* load the face */
    error = pfr_log_font_load(
              &face->log_font,
              stream,
              (FT_UInt)( face_index & 0xFFFF ),
              face->header.log_dir_offset,
              FT_BOOL( face->header.phy_font_max_size_high != 0 ) );
    if ( error )
      goto Exit;

    /* now load the physical font descriptor */
    error = pfr_phy_font_load( &face->phy_font, stream,
                               face->log_font.phys_offset,
                               face->log_font.phys_size );
    if ( error )
      goto Exit;

    /* now set up all root face fields */
    {
      PFR_PhyFont  phy_font = &face->phy_font;


      pfrface->face_index = face_index & 0xFFFF;
      pfrface->num_glyphs = (FT_Long)phy_font->num_chars + 1;

      pfrface->face_flags |= FT_FACE_FLAG_SCALABLE;

      /* if gps_offset == 0 for all characters, we  */
      /* assume that the font only contains bitmaps */
      {
        FT_UInt  nn;


        for ( nn = 0; nn < phy_font->num_chars; nn++ )
          if ( phy_font->chars[nn].gps_offset != 0 )
            break;

        if ( nn == phy_font->num_chars )
        {
          if ( phy_font->num_strikes > 0 )
            pfrface->face_flags = 0;        /* not scalable */
          else
          {
            FT_ERROR(( "pfr_face_init: font doesn't contain glyphs\n" ));
            error = FT_THROW( Invalid_File_Format );
            goto Exit;
          }
        }
      }

      if ( ( phy_font->flags & PFR_PHY_PROPORTIONAL ) == 0 )
        pfrface->face_flags |= FT_FACE_FLAG_FIXED_WIDTH;

      if ( phy_font->flags & PFR_PHY_VERTICAL )
        pfrface->face_flags |= FT_FACE_FLAG_VERTICAL;
      else
        pfrface->face_flags |= FT_FACE_FLAG_HORIZONTAL;

      if ( phy_font->num_strikes > 0 )
        pfrface->face_flags |= FT_FACE_FLAG_FIXED_SIZES;

      if ( phy_font->num_kern_pairs > 0 )
        pfrface->face_flags |= FT_FACE_FLAG_KERNING;

      /* If no family name was found in the `undocumented' auxiliary
       * data, use the font ID instead.  This sucks but is better than
       * nothing.
       */
      pfrface->family_name = phy_font->family_name;
      if ( pfrface->family_name == NULL )
        pfrface->family_name = phy_font->font_id;

      /* note that the style name can be NULL in certain PFR fonts,
       * probably meaning `Regular'
       */
      pfrface->style_name = phy_font->style_name;

      pfrface->num_fixed_sizes = 0;
      pfrface->available_sizes = NULL;

      pfrface->bbox         = phy_font->bbox;
      pfrface->units_per_EM = (FT_UShort)phy_font->outline_resolution;
      pfrface->ascender     = (FT_Short) phy_font->bbox.yMax;
      pfrface->descender    = (FT_Short) phy_font->bbox.yMin;

      pfrface->height = (FT_Short)( ( pfrface->units_per_EM * 12 ) / 10 );
      if ( pfrface->height < pfrface->ascender - pfrface->descender )
        pfrface->height = (FT_Short)(pfrface->ascender - pfrface->descender);

      if ( phy_font->num_strikes > 0 )
      {
        FT_UInt          n, count = phy_font->num_strikes;
        FT_Bitmap_Size*  size;
        PFR_Strike       strike;
        FT_Memory        memory = pfrface->stream->memory;


        if ( FT_NEW_ARRAY( pfrface->available_sizes, count ) )
          goto Exit;

        size   = pfrface->available_sizes;
        strike = phy_font->strikes;
        for ( n = 0; n < count; n++, size++, strike++ )
        {
          size->height = (FT_Short)strike->y_ppm;
          size->width  = (FT_Short)strike->x_ppm;
          size->size   = (FT_Pos)( strike->y_ppm << 6 );
          size->x_ppem = (FT_Pos)( strike->x_ppm << 6 );
          size->y_ppem = (FT_Pos)( strike->y_ppm << 6 );
        }
        pfrface->num_fixed_sizes = (FT_Int)count;
      }

      /* now compute maximum advance width */
      if ( ( phy_font->flags & PFR_PHY_PROPORTIONAL ) == 0 )
        pfrface->max_advance_width = (FT_Short)phy_font->standard_advance;
      else
      {
        FT_Int    max = 0;
        FT_UInt   count = phy_font->num_chars;
        PFR_Char  gchar = phy_font->chars;


        for ( ; count > 0; count--, gchar++ )
        {
          if ( max < gchar->advance )
            max = gchar->advance;
        }

        pfrface->max_advance_width = (FT_Short)max;
      }

      pfrface->max_advance_height = pfrface->height;

      pfrface->underline_position  = (FT_Short)( -pfrface->units_per_EM / 10 );
      pfrface->underline_thickness = (FT_Short)(  pfrface->units_per_EM / 30 );

      /* create charmap */
      {
        FT_CharMapRec  charmap;


        charmap.face        = pfrface;
        charmap.platform_id = TT_PLATFORM_MICROSOFT;
        charmap.encoding_id = TT_MS_ID_UNICODE_CS;
        charmap.encoding    = FT_ENCODING_UNICODE;

        error = FT_CMap_New( &pfr_cmap_class_rec, NULL, &charmap, NULL );

#if 0
        /* select default charmap */
        if ( pfrface->num_charmaps )
          pfrface->charmap = pfrface->charmaps[0];
#endif
      }

      /* check whether we have loaded any kerning pairs */
      if ( phy_font->num_kern_pairs )
        pfrface->face_flags |= FT_FACE_FLAG_KERNING;
    }

  Exit:
    return error;
  }


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                    SLOT OBJECT METHOD                         *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  FT_LOCAL_DEF( FT_Error )
  pfr_slot_init( FT_GlyphSlot  pfrslot )        /* PFR_Slot */
  {
    PFR_Slot        slot   = (PFR_Slot)pfrslot;
    FT_GlyphLoader  loader = pfrslot->internal->loader;


    pfr_glyph_init( &slot->glyph, loader );

    return 0;
  }


  FT_LOCAL_DEF( void )
  pfr_slot_done( FT_GlyphSlot  pfrslot )        /* PFR_Slot */
  {
    PFR_Slot  slot = (PFR_Slot)pfrslot;


    pfr_glyph_done( &slot->glyph );
  }


  FT_LOCAL_DEF( FT_Error )
  pfr_slot_load( FT_GlyphSlot  pfrslot,         /* PFR_Slot */
                 FT_Size       pfrsize,         /* PFR_Size */
                 FT_UInt       gindex,
                 FT_Int32      load_flags )
  {
    PFR_Slot     slot    = (PFR_Slot)pfrslot;
    PFR_Size     size    = (PFR_Size)pfrsize;
    FT_Error     error;
    PFR_Face     face    = (PFR_Face)pfrslot->face;
    PFR_Char     gchar;
    FT_Outline*  outline = &pfrslot->outline;
    FT_ULong     gps_offset;


    FT_TRACE1(( "pfr_slot_load: glyph index %d\n", gindex ));

    if ( gindex > 0 )
      gindex--;

    if ( !face || gindex >= face->phy_font.num_chars )
    {
      error = FT_THROW( Invalid_Argument );
      goto Exit;
    }

    /* try to load an embedded bitmap */
    if ( ( load_flags & ( FT_LOAD_NO_SCALE | FT_LOAD_NO_BITMAP ) ) == 0 )
    {
      error = pfr_slot_load_bitmap( slot, size, gindex );
      if ( error == 0 )
        goto Exit;
    }

    if ( load_flags & FT_LOAD_SBITS_ONLY )
    {
      error = FT_THROW( Invalid_Argument );
      goto Exit;
    }

    gchar               = face->phy_font.chars + gindex;
    pfrslot->format     = FT_GLYPH_FORMAT_OUTLINE;
    outline->n_points   = 0;
    outline->n_contours = 0;
    gps_offset          = face->header.gps_section_offset;

    /* load the glyph outline (FT_LOAD_NO_RECURSE isn't supported) */
    error = pfr_glyph_load( &slot->glyph, face->root.stream,
                            gps_offset, gchar->gps_offset, gchar->gps_size );

    if ( !error )
    {
      FT_BBox            cbox;
      FT_Glyph_Metrics*  metrics = &pfrslot->metrics;
      FT_Pos             advance;
      FT_UInt            em_metrics, em_outline;
      FT_Bool            scaling;


      scaling = FT_BOOL( ( load_flags & FT_LOAD_NO_SCALE ) == 0 );

      /* copy outline data */
      *outline = slot->glyph.loader->base.outline;

      outline->flags &= ~FT_OUTLINE_OWNER;
      outline->flags |= FT_OUTLINE_REVERSE_FILL;

      if ( size && pfrsize->metrics.y_ppem < 24 )
        outline->flags |= FT_OUTLINE_HIGH_PRECISION;

      /* compute the advance vector */
      metrics->horiAdvance = 0;
      metrics->vertAdvance = 0;

      advance    = gchar->advance;
      em_metrics = face->phy_font.metrics_resolution;
      em_outline = face->phy_font.outline_resolution;

      if ( em_metrics != em_outline )
        advance = FT_MulDiv( advance,
                             (FT_Long)em_outline,
                             (FT_Long)em_metrics );

      if ( face->phy_font.flags & PFR_PHY_VERTICAL )
        metrics->vertAdvance = advance;
      else
        metrics->horiAdvance = advance;

      pfrslot->linearHoriAdvance = metrics->horiAdvance;
      pfrslot->linearVertAdvance = metrics->vertAdvance;

      /* make up vertical metrics(?) */
      metrics->vertBearingX = 0;
      metrics->vertBearingY = 0;

#if 0 /* some fonts seem to be broken here! */

      /* Apply the font matrix, if any.                 */
      /* TODO: Test existing fonts with unusual matrix  */
      /* whether we have to adjust Units per EM.        */
      {
        FT_Matrix font_matrix;


        font_matrix.xx = face->log_font.matrix[0] << 8;
        font_matrix.yx = face->log_font.matrix[1] << 8;
        font_matrix.xy = face->log_font.matrix[2] << 8;
        font_matrix.yy = face->log_font.matrix[3] << 8;

        FT_Outline_Transform( outline, &font_matrix );
      }
#endif

      /* scale when needed */
      if ( scaling )
      {
        FT_Int      n;
        FT_Fixed    x_scale = pfrsize->metrics.x_scale;
        FT_Fixed    y_scale = pfrsize->metrics.y_scale;
        FT_Vector*  vec     = outline->points;


        /* scale outline points */
        for ( n = 0; n < outline->n_points; n++, vec++ )
        {
          vec->x = FT_MulFix( vec->x, x_scale );
          vec->y = FT_MulFix( vec->y, y_scale );
        }

        /* scale the advance */
        metrics->horiAdvance = FT_MulFix( metrics->horiAdvance, x_scale );
        metrics->vertAdvance = FT_MulFix( metrics->vertAdvance, y_scale );
      }

      /* compute the rest of the metrics */
      FT_Outline_Get_CBox( outline, &cbox );

      metrics->width        = cbox.xMax - cbox.xMin;
      metrics->height       = cbox.yMax - cbox.yMin;
      metrics->horiBearingX = cbox.xMin;
      metrics->horiBearingY = cbox.yMax - metrics->height;
    }

  Exit:
    return error;
  }


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                      KERNING METHOD                           *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  FT_LOCAL_DEF( FT_Error )
  pfr_face_get_kerning( FT_Face     pfrface,        /* PFR_Face */
                        FT_UInt     glyph1,
                        FT_UInt     glyph2,
                        FT_Vector*  kerning )
  {
    PFR_Face     face     = (PFR_Face)pfrface;
    FT_Error     error    = FT_Err_Ok;
    PFR_PhyFont  phy_font = &face->phy_font;
    FT_UInt32    code1, code2, pair;


    kerning->x = 0;
    kerning->y = 0;

    if ( glyph1 > 0 )
      glyph1--;

    if ( glyph2 > 0 )
      glyph2--;

    /* convert glyph indices to character codes */
    if ( glyph1 > phy_font->num_chars ||
         glyph2 > phy_font->num_chars )
      goto Exit;

    code1 = phy_font->chars[glyph1].char_code;
    code2 = phy_font->chars[glyph2].char_code;
    pair  = PFR_KERN_INDEX( code1, code2 );

    /* now search the list of kerning items */
    {
      PFR_KernItem  item   = phy_font->kern_items;
      FT_Stream     stream = pfrface->stream;


      for ( ; item; item = item->next )
      {
        if ( pair >= item->pair1 && pair <= item->pair2 )
          goto FoundPair;
      }
      goto Exit;

    FoundPair: /* we found an item, now parse it and find the value if any */
      if ( FT_STREAM_SEEK( item->offset )                       ||
           FT_FRAME_ENTER( item->pair_count * item->pair_size ) )
        goto Exit;

      {
        FT_UInt    count       = item->pair_count;
        FT_UInt    size        = item->pair_size;
        FT_UInt    power       = 1 << FT_MSB( count );
        FT_UInt    probe       = power * size;
        FT_UInt    extra       = count - power;
        FT_Byte*   base        = stream->cursor;
        FT_Bool    twobytes    = FT_BOOL( item->flags & PFR_KERN_2BYTE_CHAR );
        FT_Bool    twobyte_adj = FT_BOOL( item->flags & PFR_KERN_2BYTE_ADJ  );
        FT_Byte*   p;
        FT_UInt32  cpair;


        if ( extra > 0 )
        {
          p = base + extra * size;

          if ( twobytes )
            cpair = FT_NEXT_ULONG( p );
          else
            cpair = PFR_NEXT_KPAIR( p );

          if ( cpair == pair )
            goto Found;

          if ( cpair < pair )
          {
            if ( twobyte_adj )
              p += 2;
            else
              p++;
            base = p;
          }
        }

        while ( probe > size )
        {
          probe >>= 1;
          p       = base + probe;

          if ( twobytes )
            cpair = FT_NEXT_ULONG( p );
          else
            cpair = PFR_NEXT_KPAIR( p );

          if ( cpair == pair )
            goto Found;

          if ( cpair < pair )
            base += probe;
        }

        p = base;

        if ( twobytes )
          cpair = FT_NEXT_ULONG( p );
        else
          cpair = PFR_NEXT_KPAIR( p );

        if ( cpair == pair )
        {
          FT_Int  value;


        Found:
          if ( twobyte_adj )
            value = FT_PEEK_SHORT( p );
          else
            value = p[0];

          kerning->x = item->base_adj + value;
        }
      }

      FT_FRAME_EXIT();
    }

  Exit:
    return error;
  }


/* END */
