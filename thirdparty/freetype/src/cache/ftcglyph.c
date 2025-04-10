/****************************************************************************
 *
 * ftcglyph.c
 *
 *   FreeType Glyph Image (FT_Glyph) cache (body).
 *
 * Copyright (C) 2000-2024 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#include <freetype/internal/ftobjs.h>
#include <freetype/ftcache.h>
#include "ftcglyph.h"
#include <freetype/fterrors.h>

#include "ftccback.h"
#include "ftcerror.h"


  /* create a new chunk node, setting its cache index and ref count */
  FT_LOCAL_DEF( void )
  FTC_GNode_Init( FTC_GNode   gnode,
                  FT_UInt     gindex,
                  FTC_Family  family )
  {
    gnode->family = family;
    gnode->gindex = gindex;
    family->num_nodes++;
  }


  FT_LOCAL_DEF( void )
  FTC_GNode_UnselectFamily( FTC_GNode  gnode,
                            FTC_Cache  cache )
  {
    FTC_Family  family = gnode->family;


    gnode->family = NULL;
    if ( family && --family->num_nodes == 0 )
      FTC_FAMILY_FREE( family, cache );
  }


  FT_LOCAL_DEF( void )
  FTC_GNode_Done( FTC_GNode  gnode,
                  FTC_Cache  cache )
  {
    /* finalize the node */
    gnode->gindex = 0;

    FTC_GNode_UnselectFamily( gnode, cache );
  }


  FT_LOCAL_DEF( FT_Bool )
  ftc_gnode_compare( FTC_Node    ftcgnode,
                     FT_Pointer  ftcgquery,
                     FTC_Cache   cache,
                     FT_Bool*    list_changed )
  {
    FTC_GNode   gnode  = (FTC_GNode)ftcgnode;
    FTC_GQuery  gquery = (FTC_GQuery)ftcgquery;
    FT_UNUSED( cache );


    if ( list_changed )
      *list_changed = FALSE;
    return FT_BOOL( gnode->family == gquery->family &&
                    gnode->gindex == gquery->gindex );
  }


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                      CHUNK SETS                               *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  FT_LOCAL_DEF( void )
  FTC_Family_Init( FTC_Family  family,
                   FTC_Cache   cache )
  {
    FTC_GCacheClass  clazz = FTC_CACHE_GCACHE_CLASS( cache );


    family->clazz     = clazz->family_class;
    family->num_nodes = 0;
    family->cache     = cache;
  }


  FT_LOCAL_DEF( FT_Error )
  ftc_gcache_init( FTC_Cache  cache )
  {
    FTC_GCache  gcache = (FTC_GCache)cache;
    FT_Error    error;


    error = FTC_Cache_Init( cache );
    if ( !error )
    {
      FTC_GCacheClass   clazz = (FTC_GCacheClass)cache->org_class;

      FTC_MruList_Init( &gcache->families,
                        clazz->family_class,
                        0,  /* no maximum here! */
                        cache,
                        cache->memory );
    }

    return error;
  }


#if 0

  FT_LOCAL_DEF( FT_Error )
  FTC_GCache_Init( FTC_GCache  gcache )
  {
    return ftc_gcache_init( FTC_CACHE( gcache ) );
  }

#endif /* 0 */


  FT_LOCAL_DEF( void )
  ftc_gcache_done( FTC_Cache  cache )
  {
    FTC_GCache  gcache = (FTC_GCache)cache;


    FTC_Cache_Done( cache );
    FTC_MruList_Done( &gcache->families );
  }


#if 0

  FT_LOCAL_DEF( void )
  FTC_GCache_Done( FTC_GCache  gcache )
  {
    ftc_gcache_done( FTC_CACHE( gcache ) );
  }

#endif /* 0 */


  FT_LOCAL_DEF( FT_Error )
  FTC_GCache_New( FTC_Manager       manager,
                  FTC_GCacheClass   clazz,
                  FTC_GCache       *acache )
  {
    return FTC_Manager_RegisterCache( manager, (FTC_CacheClass)clazz,
                                      (FTC_Cache*)acache );
  }


#ifndef FTC_INLINE

  FT_LOCAL_DEF( FT_Error )
  FTC_GCache_Lookup( FTC_GCache   gcache,
                     FT_Offset    hash,
                     FT_UInt      gindex,
                     FTC_GQuery   query,
                     FTC_Node    *anode )
  {
    FT_Error  error;


    query->gindex = gindex;

    FTC_MRULIST_LOOKUP( &gcache->families, query, query->family, error );
    if ( !error )
    {
      FTC_Family  family = query->family;


      /* prevent the family from being destroyed too early when an        */
      /* out-of-memory condition occurs during glyph node initialization. */
      family->num_nodes++;

      error = FTC_Cache_Lookup( FTC_CACHE( gcache ), hash, query, anode );

      if ( --family->num_nodes == 0 )
        FTC_FAMILY_FREE( family, FTC_CACHE( gcache ) );
    }
    return error;
  }

#endif /* !FTC_INLINE */


/* END */
