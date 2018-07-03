/***************************************************************************/
/*                                                                         */
/*  ftgloadr.h                                                             */
/*                                                                         */
/*    The FreeType glyph loader (specification).                           */
/*                                                                         */
/*  Copyright 2002-2018 by                                                 */
/*  David Turner, Robert Wilhelm, and Werner Lemberg                       */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/


#ifndef FTGLOADR_H_
#define FTGLOADR_H_


#include <ft2build.h>
#include FT_FREETYPE_H


FT_BEGIN_HEADER


  /*************************************************************************/
  /*                                                                       */
  /* <Struct>                                                              */
  /*    FT_GlyphLoader                                                     */
  /*                                                                       */
  /* <Description>                                                         */
  /*    The glyph loader is an internal object used to load several glyphs */
  /*    together (for example, in the case of composites).                 */
  /*                                                                       */
  typedef struct  FT_SubGlyphRec_
  {
    FT_Int     index;
    FT_UShort  flags;
    FT_Int     arg1;
    FT_Int     arg2;
    FT_Matrix  transform;

  } FT_SubGlyphRec;


  typedef struct  FT_GlyphLoadRec_
  {
    FT_Outline   outline;       /* outline                   */
    FT_Vector*   extra_points;  /* extra points table        */
    FT_Vector*   extra_points2; /* second extra points table */
    FT_UInt      num_subglyphs; /* number of subglyphs       */
    FT_SubGlyph  subglyphs;     /* subglyphs                 */

  } FT_GlyphLoadRec, *FT_GlyphLoad;


  typedef struct  FT_GlyphLoaderRec_
  {
    FT_Memory        memory;
    FT_UInt          max_points;
    FT_UInt          max_contours;
    FT_UInt          max_subglyphs;
    FT_Bool          use_extra;

    FT_GlyphLoadRec  base;
    FT_GlyphLoadRec  current;

    void*            other;            /* for possible future extension? */

  } FT_GlyphLoaderRec, *FT_GlyphLoader;


  /* create new empty glyph loader */
  FT_BASE( FT_Error )
  FT_GlyphLoader_New( FT_Memory        memory,
                      FT_GlyphLoader  *aloader );

  /* add an extra points table to a glyph loader */
  FT_BASE( FT_Error )
  FT_GlyphLoader_CreateExtra( FT_GlyphLoader  loader );

  /* destroy a glyph loader */
  FT_BASE( void )
  FT_GlyphLoader_Done( FT_GlyphLoader  loader );

  /* reset a glyph loader (frees everything int it) */
  FT_BASE( void )
  FT_GlyphLoader_Reset( FT_GlyphLoader  loader );

  /* rewind a glyph loader */
  FT_BASE( void )
  FT_GlyphLoader_Rewind( FT_GlyphLoader  loader );

  /* check that there is enough space to add `n_points' and `n_contours' */
  /* to the glyph loader                                                 */
  FT_BASE( FT_Error )
  FT_GlyphLoader_CheckPoints( FT_GlyphLoader  loader,
                              FT_UInt         n_points,
                              FT_UInt         n_contours );


#define FT_GLYPHLOADER_CHECK_P( _loader, _count )       \
  ( (_count) == 0                                    || \
    ( (FT_UInt)(_loader)->base.outline.n_points    +    \
      (FT_UInt)(_loader)->current.outline.n_points +    \
      (FT_UInt)(_count) ) <= (_loader)->max_points   )

#define FT_GLYPHLOADER_CHECK_C( _loader, _count )         \
  ( (_count) == 0                                      || \
    ( (FT_UInt)(_loader)->base.outline.n_contours    +    \
      (FT_UInt)(_loader)->current.outline.n_contours +    \
      (FT_UInt)(_count) ) <= (_loader)->max_contours   )

#define FT_GLYPHLOADER_CHECK_POINTS( _loader, _points, _contours ) \
  ( ( FT_GLYPHLOADER_CHECK_P( _loader, _points )   &&              \
      FT_GLYPHLOADER_CHECK_C( _loader, _contours ) )               \
    ? 0                                                            \
    : FT_GlyphLoader_CheckPoints( (_loader),                       \
                                  (FT_UInt)(_points),              \
                                  (FT_UInt)(_contours) ) )


  /* check that there is enough space to add `n_subs' sub-glyphs to */
  /* a glyph loader                                                 */
  FT_BASE( FT_Error )
  FT_GlyphLoader_CheckSubGlyphs( FT_GlyphLoader  loader,
                                 FT_UInt         n_subs );

  /* prepare a glyph loader, i.e. empty the current glyph */
  FT_BASE( void )
  FT_GlyphLoader_Prepare( FT_GlyphLoader  loader );

  /* add the current glyph to the base glyph */
  FT_BASE( void )
  FT_GlyphLoader_Add( FT_GlyphLoader  loader );

  /* copy points from one glyph loader to another */
  FT_BASE( FT_Error )
  FT_GlyphLoader_CopyPoints( FT_GlyphLoader  target,
                             FT_GlyphLoader  source );

 /* */


FT_END_HEADER

#endif /* FTGLOADR_H_ */


/* END */
