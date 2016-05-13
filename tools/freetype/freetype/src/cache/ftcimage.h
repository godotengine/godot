/***************************************************************************/
/*                                                                         */
/*  ftcimage.h                                                             */
/*                                                                         */
/*    FreeType Generic Image cache (specification)                         */
/*                                                                         */
/*  Copyright 2000-2001, 2002, 2003, 2006 by                               */
/*  David Turner, Robert Wilhelm, and Werner Lemberg.                      */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/


 /*
  *  FTC_ICache is an _abstract_ cache used to store a single FT_Glyph
  *  image per cache node.
  *
  *  FTC_ICache extends FTC_GCache.  For an implementation example,
  *  see FTC_ImageCache in `src/cache/ftbasic.c'.
  */


  /*************************************************************************/
  /*                                                                       */
  /* Each image cache really manages FT_Glyph objects.                     */
  /*                                                                       */
  /*************************************************************************/


#ifndef __FTCIMAGE_H__
#define __FTCIMAGE_H__


#include <ft2build.h>
#include FT_CACHE_H
#include "ftcglyph.h"

FT_BEGIN_HEADER


  /* the FT_Glyph image node type - we store only 1 glyph per node */
  typedef struct  FTC_INodeRec_
  {
    FTC_GNodeRec  gnode;
    FT_Glyph      glyph;

  } FTC_INodeRec, *FTC_INode;

#define FTC_INODE( x )         ( (FTC_INode)( x ) )
#define FTC_INODE_GINDEX( x )  FTC_GNODE(x)->gindex
#define FTC_INODE_FAMILY( x )  FTC_GNODE(x)->family

  typedef FT_Error
  (*FTC_IFamily_LoadGlyphFunc)( FTC_Family  family,
                                FT_UInt     gindex,
                                FTC_Cache   cache,
                                FT_Glyph   *aglyph );

  typedef struct  FTC_IFamilyClassRec_
  {
    FTC_MruListClassRec        clazz;
    FTC_IFamily_LoadGlyphFunc  family_load_glyph;

  } FTC_IFamilyClassRec;

  typedef const FTC_IFamilyClassRec*  FTC_IFamilyClass;

#define FTC_IFAMILY_CLASS( x )  ((FTC_IFamilyClass)(x))

#define FTC_CACHE__IFAMILY_CLASS( x ) \
          FTC_IFAMILY_CLASS( FTC_CACHE__GCACHE_CLASS(x)->family_class )


  /* can be used as a @FTC_Node_FreeFunc */
  FT_LOCAL( void )
  FTC_INode_Free( FTC_INode  inode,
                  FTC_Cache  cache );

  /* Can be used as @FTC_Node_NewFunc.  `gquery.index' and `gquery.family'
   * must be set correctly.  This function will call the `family_load_glyph'
   * method to load the FT_Glyph into the cache node.
   */
  FT_LOCAL( FT_Error )
  FTC_INode_New( FTC_INode   *pinode,
                 FTC_GQuery   gquery,
                 FTC_Cache    cache );

#if 0
  /* can be used as @FTC_Node_WeightFunc */
  FT_LOCAL( FT_ULong )
  FTC_INode_Weight( FTC_INode  inode );
#endif


 /* */

FT_END_HEADER

#endif /* __FTCIMAGE_H__ */


/* END */
