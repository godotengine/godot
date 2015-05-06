/***************************************************************************/
/*                                                                         */
/*  ftgloadr.h                                                             */
/*                                                                         */
/*    The FreeType glyph loader (specification).                           */
/*                                                                         */
/*  Copyright 2002, 2003, 2005, 2006 by                                    */
/*  David Turner, Robert Wilhelm, and Werner Lemberg                       */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/


#ifndef __FTGLOADR_H__
#define __FTGLOADR_H__


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
  /* <Note>                                                                */
  /*    The glyph loader implementation is not part of the high-level API, */
  /*    hence the forward structure declaration.                           */
  /*                                                                       */
  typedef struct FT_GlyphLoaderRec_*  FT_GlyphLoader ;


#if 0  /* moved to freetype.h in version 2.2 */
#define FT_SUBGLYPH_FLAG_ARGS_ARE_WORDS          1
#define FT_SUBGLYPH_FLAG_ARGS_ARE_XY_VALUES      2
#define FT_SUBGLYPH_FLAG_ROUND_XY_TO_GRID        4
#define FT_SUBGLYPH_FLAG_SCALE                   8
#define FT_SUBGLYPH_FLAG_XY_SCALE             0x40
#define FT_SUBGLYPH_FLAG_2X2                  0x80
#define FT_SUBGLYPH_FLAG_USE_MY_METRICS      0x200
#endif


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

  } FT_GlyphLoaderRec;


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


#define FT_GLYPHLOADER_CHECK_P( _loader, _count )                         \
   ( (_count) == 0 || ((_loader)->base.outline.n_points    +              \
                       (_loader)->current.outline.n_points +              \
                       (unsigned long)(_count)) <= (_loader)->max_points )

#define FT_GLYPHLOADER_CHECK_C( _loader, _count )                          \
  ( (_count) == 0 || ((_loader)->base.outline.n_contours    +              \
                      (_loader)->current.outline.n_contours +              \
                      (unsigned long)(_count)) <= (_loader)->max_contours )

#define FT_GLYPHLOADER_CHECK_POINTS( _loader, _points,_contours )      \
  ( ( FT_GLYPHLOADER_CHECK_P( _loader, _points )   &&                  \
      FT_GLYPHLOADER_CHECK_C( _loader, _contours ) )                   \
    ? 0                                                                \
    : FT_GlyphLoader_CheckPoints( (_loader), (_points), (_contours) ) )


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

#endif /* __FTGLOADR_H__ */


/* END */
