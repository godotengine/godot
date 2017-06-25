/***************************************************************************/
/*                                                                         */
/*  ftcglyph.h                                                             */
/*                                                                         */
/*    FreeType abstract glyph cache (specification).                       */
/*                                                                         */
/*  Copyright 2000-2017 by                                                 */
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
   *
   *  FTC_GCache is an _abstract_ cache object optimized to store glyph
   *  data.  It works as follows:
   *
   *   - It manages FTC_GNode objects. Each one of them can hold one or more
   *     glyph `items'.  Item types are not specified in the FTC_GCache but
   *     in classes that extend it.
   *
   *   - Glyph attributes, like face ID, character size, render mode, etc.,
   *     can be grouped into abstract `glyph families'.  This avoids storing
   *     the attributes within the FTC_GCache, since it is likely that many
   *     FTC_GNodes will belong to the same family in typical uses.
   *
   *   - Each FTC_GNode is thus an FTC_Node with two additional fields:
   *
   *       * gindex: A glyph index, or the first index in a glyph range.
   *       * family: A pointer to a glyph `family'.
   *
   *   - Family types are not fully specific in the FTC_Family type, but
   *     by classes that extend it.
   *
   *  Note that both FTC_ImageCache and FTC_SBitCache extend FTC_GCache.
   *  They share an FTC_Family sub-class called FTC_BasicFamily which is
   *  used to store the following data: face ID, pixel/point sizes, load
   *  flags.  For more details see the file `src/cache/ftcbasic.c'.
   *
   *  Client applications can extend FTC_GNode with their own FTC_GNode
   *  and FTC_Family sub-classes to implement more complex caches (e.g.,
   *  handling automatic synthesis, like obliquing & emboldening, colored
   *  glyphs, etc.).
   *
   *  See also the FTC_ICache & FTC_SCache classes in `ftcimage.h' and
   *  `ftcsbits.h', which both extend FTC_GCache with additional
   *  optimizations.
   *
   *  A typical FTC_GCache implementation must provide at least the
   *  following:
   *
   *  - FTC_GNode sub-class, e.g. MyNode, with relevant methods:
   *        my_node_new            (must call FTC_GNode_Init)
   *        my_node_free           (must call FTC_GNode_Done)
   *        my_node_compare        (must call FTC_GNode_Compare)
   *        my_node_remove_faceid  (must call ftc_gnode_unselect in case
   *                                of match)
   *
   *  - FTC_Family sub-class, e.g. MyFamily, with relevant methods:
   *        my_family_compare
   *        my_family_init
   *        my_family_reset (optional)
   *        my_family_done
   *
   *  - FTC_GQuery sub-class, e.g. MyQuery, to hold cache-specific query
   *    data.
   *
   *  - Constant structures for a FTC_GNodeClass.
   *
   *  - MyCacheNew() can be implemented easily as a call to the convenience
   *    function FTC_GCache_New.
   *
   *  - MyCacheLookup with a call to FTC_GCache_Lookup.  This function will
   *    automatically:
   *
   *    - Search for the corresponding family in the cache, or create
   *      a new one if necessary.  Put it in FTC_GQUERY(myquery).family
   *
   *    - Call FTC_Cache_Lookup.
   *
   *    If it returns NULL, you should create a new node, then call
   *    ftc_cache_add as usual.
   */


  /*************************************************************************/
  /*                                                                       */
  /* Important: The functions defined in this file are only used to        */
  /*            implement an abstract glyph cache class.  You need to      */
  /*            provide additional logic to implement a complete cache.    */
  /*                                                                       */
  /*************************************************************************/


  /*************************************************************************/
  /*************************************************************************/
  /*************************************************************************/
  /*************************************************************************/
  /*************************************************************************/
  /*********                                                       *********/
  /*********             WARNING, THIS IS BETA CODE.               *********/
  /*********                                                       *********/
  /*************************************************************************/
  /*************************************************************************/
  /*************************************************************************/
  /*************************************************************************/
  /*************************************************************************/


#ifndef FTCGLYPH_H_
#define FTCGLYPH_H_


#include <ft2build.h>
#include "ftcmanag.h"


FT_BEGIN_HEADER


 /*
  *  We can group glyphs into `families'.  Each family correspond to a
  *  given face ID, character size, transform, etc.
  *
  *  Families are implemented as MRU list nodes.  They are
  *  reference-counted.
  */

  typedef struct  FTC_FamilyRec_
  {
    FTC_MruNodeRec    mrunode;
    FT_UInt           num_nodes; /* current number of nodes in this family */
    FTC_Cache         cache;
    FTC_MruListClass  clazz;

  } FTC_FamilyRec, *FTC_Family;

#define  FTC_FAMILY(x)    ( (FTC_Family)(x) )
#define  FTC_FAMILY_P(x)  ( (FTC_Family*)(x) )


  typedef struct  FTC_GNodeRec_
  {
    FTC_NodeRec      node;
    FTC_Family       family;
    FT_UInt          gindex;

  } FTC_GNodeRec, *FTC_GNode;

#define FTC_GNODE( x )    ( (FTC_GNode)(x) )
#define FTC_GNODE_P( x )  ( (FTC_GNode*)(x) )


  typedef struct  FTC_GQueryRec_
  {
    FT_UInt      gindex;
    FTC_Family   family;

  } FTC_GQueryRec, *FTC_GQuery;

#define FTC_GQUERY( x )  ( (FTC_GQuery)(x) )


  /*************************************************************************/
  /*                                                                       */
  /* These functions are exported so that they can be called from          */
  /* user-provided cache classes; otherwise, they are really part of the   */
  /* cache sub-system internals.                                           */
  /*                                                                       */

  /* must be called by derived FTC_Node_InitFunc routines */
  FT_LOCAL( void )
  FTC_GNode_Init( FTC_GNode   node,
                  FT_UInt     gindex,  /* glyph index for node */
                  FTC_Family  family );

#ifdef FTC_INLINE

  /* returns TRUE iff the query's glyph index correspond to the node;  */
  /* this assumes that the `family' and `hash' fields of the query are */
  /* already correctly set                                             */
  FT_LOCAL( FT_Bool )
  FTC_GNode_Compare( FTC_GNode   gnode,
                     FTC_GQuery  gquery,
                     FTC_Cache   cache,
                     FT_Bool*    list_changed );

#endif

  /* call this function to clear a node's family -- this is necessary */
  /* to implement the `node_remove_faceid' cache method correctly     */
  FT_LOCAL( void )
  FTC_GNode_UnselectFamily( FTC_GNode  gnode,
                            FTC_Cache  cache );

  /* must be called by derived FTC_Node_DoneFunc routines */
  FT_LOCAL( void )
  FTC_GNode_Done( FTC_GNode  node,
                  FTC_Cache  cache );


  FT_LOCAL( void )
  FTC_Family_Init( FTC_Family  family,
                   FTC_Cache   cache );

  typedef struct FTC_GCacheRec_
  {
    FTC_CacheRec    cache;
    FTC_MruListRec  families;

  } FTC_GCacheRec, *FTC_GCache;

#define FTC_GCACHE( x )  ((FTC_GCache)(x))


#if 0
  /* can be used as @FTC_Cache_InitFunc */
  FT_LOCAL( FT_Error )
  FTC_GCache_Init( FTC_GCache  cache );
#endif


#if 0
  /* can be used as @FTC_Cache_DoneFunc */
  FT_LOCAL( void )
  FTC_GCache_Done( FTC_GCache  cache );
#endif


  /* the glyph cache class adds fields for the family implementation */
  typedef struct  FTC_GCacheClassRec_
  {
    FTC_CacheClassRec  clazz;
    FTC_MruListClass   family_class;

  } FTC_GCacheClassRec;

  typedef const FTC_GCacheClassRec*   FTC_GCacheClass;

#define FTC_GCACHE_CLASS( x )  ((FTC_GCacheClass)(x))

#define FTC_CACHE_GCACHE_CLASS( x ) \
          FTC_GCACHE_CLASS( FTC_CACHE(x)->org_class )
#define FTC_CACHE_FAMILY_CLASS( x ) \
          ( (FTC_MruListClass)FTC_CACHE_GCACHE_CLASS( x )->family_class )


  /* convenience function; use it instead of FTC_Manager_Register_Cache */
  FT_LOCAL( FT_Error )
  FTC_GCache_New( FTC_Manager       manager,
                  FTC_GCacheClass   clazz,
                  FTC_GCache       *acache );

#ifndef FTC_INLINE
  FT_LOCAL( FT_Error )
  FTC_GCache_Lookup( FTC_GCache   cache,
                     FT_Offset    hash,
                     FT_UInt      gindex,
                     FTC_GQuery   query,
                     FTC_Node    *anode );
#endif


  /* */


#define FTC_FAMILY_FREE( family, cache )                      \
          FTC_MruList_Remove( &FTC_GCACHE((cache))->families, \
                              (FTC_MruNode)(family) )


#ifdef FTC_INLINE

#define FTC_GCACHE_LOOKUP_CMP( cache, famcmp, nodecmp, hash,                \
                               gindex, query, node, error )                 \
  FT_BEGIN_STMNT                                                            \
    FTC_GCache               _gcache   = FTC_GCACHE( cache );               \
    FTC_GQuery               _gquery   = (FTC_GQuery)( query );             \
    FTC_MruNode_CompareFunc  _fcompare = (FTC_MruNode_CompareFunc)(famcmp); \
    FTC_MruNode              _mrunode;                                      \
                                                                            \
                                                                            \
    _gquery->gindex = (gindex);                                             \
                                                                            \
    FTC_MRULIST_LOOKUP_CMP( &_gcache->families, _gquery, _fcompare,         \
                            _mrunode, error );                              \
    _gquery->family = FTC_FAMILY( _mrunode );                               \
    if ( !error )                                                           \
    {                                                                       \
      FTC_Family  _gqfamily = _gquery->family;                              \
                                                                            \
                                                                            \
      _gqfamily->num_nodes++;                                               \
                                                                            \
      FTC_CACHE_LOOKUP_CMP( cache, nodecmp, hash, query, node, error );     \
                                                                            \
      if ( --_gqfamily->num_nodes == 0 )                                    \
        FTC_FAMILY_FREE( _gqfamily, _gcache );                              \
    }                                                                       \
  FT_END_STMNT
  /* */

#else /* !FTC_INLINE */

#define FTC_GCACHE_LOOKUP_CMP( cache, famcmp, nodecmp, hash,          \
                               gindex, query, node, error )           \
   FT_BEGIN_STMNT                                                     \
                                                                      \
     error = FTC_GCache_Lookup( FTC_GCACHE( cache ), hash, gindex,    \
                                FTC_GQUERY( query ), &node );         \
                                                                      \
   FT_END_STMNT

#endif /* !FTC_INLINE */


FT_END_HEADER


#endif /* FTCGLYPH_H_ */


/* END */
