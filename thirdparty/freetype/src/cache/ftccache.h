/***************************************************************************/
/*                                                                         */
/*  ftccache.h                                                             */
/*                                                                         */
/*    FreeType internal cache interface (specification).                   */
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


#ifndef FTCCACHE_H_
#define FTCCACHE_H_


#include "ftcmru.h"

FT_BEGIN_HEADER

#define FTC_FACE_ID_HASH( i )                                  \
         ( ( (FT_Offset)(i) >> 3 ) ^ ( (FT_Offset)(i) << 7 ) )

  /* handle to cache object */
  typedef struct FTC_CacheRec_*  FTC_Cache;

  /* handle to cache class */
  typedef const struct FTC_CacheClassRec_*  FTC_CacheClass;


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                   CACHE NODE DEFINITIONS                      *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  /*************************************************************************/
  /*                                                                       */
  /* Each cache controls one or more cache nodes.  Each node is part of    */
  /* the global_lru list of the manager.  Its `data' field however is used */
  /* as a reference count for now.                                         */
  /*                                                                       */
  /* A node can be anything, depending on the type of information held by  */
  /* the cache.  It can be an individual glyph image, a set of bitmaps     */
  /* glyphs for a given size, some metrics, etc.                           */
  /*                                                                       */
  /*************************************************************************/

  /* structure size should be 20 bytes on 32-bits machines */
  typedef struct  FTC_NodeRec_
  {
    FTC_MruNodeRec  mru;          /* circular mru list pointer           */
    FTC_Node        link;         /* used for hashing                    */
    FT_Offset       hash;         /* used for hashing too                */
    FT_UShort       cache_index;  /* index of cache the node belongs to  */
    FT_Short        ref_count;    /* reference count for this node       */

  } FTC_NodeRec;


#define FTC_NODE( x )    ( (FTC_Node)(x) )
#define FTC_NODE_P( x )  ( (FTC_Node*)(x) )

#define FTC_NODE_NEXT( x )  FTC_NODE( (x)->mru.next )
#define FTC_NODE_PREV( x )  FTC_NODE( (x)->mru.prev )

#ifdef FTC_INLINE
#define FTC_NODE_TOP_FOR_HASH( cache, hash )                      \
        ( ( cache )->buckets +                                    \
            ( ( ( ( hash ) &   ( cache )->mask ) < ( cache )->p ) \
              ? ( ( hash ) & ( ( cache )->mask * 2 + 1 ) )        \
              : ( ( hash ) &   ( cache )->mask ) ) )
#else
  FT_LOCAL( FTC_Node* )
  ftc_get_top_node_for_hash( FTC_Cache  cache,
                             FT_Offset  hash );
#define FTC_NODE_TOP_FOR_HASH( cache, hash )             \
        ftc_get_top_node_for_hash( ( cache ), ( hash ) )
#endif


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                       CACHE DEFINITIONS                       *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  /* initialize a new cache node */
  typedef FT_Error
  (*FTC_Node_NewFunc)( FTC_Node    *pnode,
                       FT_Pointer   query,
                       FTC_Cache    cache );

  typedef FT_Offset
  (*FTC_Node_WeightFunc)( FTC_Node   node,
                          FTC_Cache  cache );

  /* compare a node to a given key pair */
  typedef FT_Bool
  (*FTC_Node_CompareFunc)( FTC_Node    node,
                           FT_Pointer  key,
                           FTC_Cache   cache,
                           FT_Bool*    list_changed );


  typedef void
  (*FTC_Node_FreeFunc)( FTC_Node   node,
                        FTC_Cache  cache );

  typedef FT_Error
  (*FTC_Cache_InitFunc)( FTC_Cache  cache );

  typedef void
  (*FTC_Cache_DoneFunc)( FTC_Cache  cache );


  typedef struct  FTC_CacheClassRec_
  {
    FTC_Node_NewFunc      node_new;
    FTC_Node_WeightFunc   node_weight;
    FTC_Node_CompareFunc  node_compare;
    FTC_Node_CompareFunc  node_remove_faceid;
    FTC_Node_FreeFunc     node_free;

    FT_Offset             cache_size;
    FTC_Cache_InitFunc    cache_init;
    FTC_Cache_DoneFunc    cache_done;

  } FTC_CacheClassRec;


  /* each cache really implements a dynamic hash table to manage its nodes */
  typedef struct  FTC_CacheRec_
  {
    FT_UFast           p;
    FT_UFast           mask;
    FT_Long            slack;
    FTC_Node*          buckets;

    FTC_CacheClassRec  clazz;       /* local copy, for speed  */

    FTC_Manager        manager;
    FT_Memory          memory;
    FT_UInt            index;       /* in manager's table     */

    FTC_CacheClass     org_class;   /* original class pointer */

  } FTC_CacheRec;


#define FTC_CACHE( x )    ( (FTC_Cache)(x) )
#define FTC_CACHE_P( x )  ( (FTC_Cache*)(x) )


  /* default cache initialize */
  FT_LOCAL( FT_Error )
  FTC_Cache_Init( FTC_Cache  cache );

  /* default cache finalizer */
  FT_LOCAL( void )
  FTC_Cache_Done( FTC_Cache  cache );

  /* Call this function to look up the cache.  If no corresponding
   * node is found, a new one is automatically created.  This function
   * is capable of flushing the cache adequately to make room for the
   * new cache object.
   */

#ifndef FTC_INLINE
  FT_LOCAL( FT_Error )
  FTC_Cache_Lookup( FTC_Cache   cache,
                    FT_Offset   hash,
                    FT_Pointer  query,
                    FTC_Node   *anode );
#endif

  FT_LOCAL( FT_Error )
  FTC_Cache_NewNode( FTC_Cache   cache,
                     FT_Offset   hash,
                     FT_Pointer  query,
                     FTC_Node   *anode );

  /* Remove all nodes that relate to a given face_id.  This is useful
   * when un-installing fonts.  Note that if a cache node relates to
   * the face_id but is locked (i.e., has `ref_count > 0'), the node
   * will _not_ be destroyed, but its internal face_id reference will
   * be modified.
   *
   * The final result will be that the node will never come back
   * in further lookup requests, and will be flushed on demand from
   * the cache normally when its reference count reaches 0.
   */
  FT_LOCAL( void )
  FTC_Cache_RemoveFaceID( FTC_Cache   cache,
                          FTC_FaceID  face_id );


#ifdef FTC_INLINE

#define FTC_CACHE_LOOKUP_CMP( cache, nodecmp, hash, query, node, error ) \
  FT_BEGIN_STMNT                                                         \
    FTC_Node             *_bucket, *_pnode, _node;                       \
    FTC_Cache             _cache   = FTC_CACHE(cache);                   \
    FT_Offset             _hash    = (FT_Offset)(hash);                  \
    FTC_Node_CompareFunc  _nodcomp = (FTC_Node_CompareFunc)(nodecmp);    \
    FT_Bool               _list_changed = FALSE;                         \
                                                                         \
                                                                         \
    error = FT_Err_Ok;                                                   \
    node  = NULL;                                                        \
                                                                         \
    /* Go to the `top' node of the list sharing same masked hash */      \
    _bucket = _pnode = FTC_NODE_TOP_FOR_HASH( _cache, _hash );           \
                                                                         \
    /* Look up a node with identical hash and queried properties.    */  \
    /* NOTE: _nodcomp() may change the linked list to reduce memory. */  \
    for (;;)                                                             \
    {                                                                    \
      _node = *_pnode;                                                   \
      if ( !_node )                                                      \
        goto NewNode_;                                                   \
                                                                         \
      if ( _node->hash == _hash                             &&           \
           _nodcomp( _node, query, _cache, &_list_changed ) )            \
        break;                                                           \
                                                                         \
      _pnode = &_node->link;                                             \
    }                                                                    \
                                                                         \
    if ( _list_changed )                                                 \
    {                                                                    \
      /* Update _bucket by possibly modified linked list */              \
      _bucket = _pnode = FTC_NODE_TOP_FOR_HASH( _cache, _hash );         \
                                                                         \
      /* Update _pnode by possibly modified linked list */               \
      while ( *_pnode != _node )                                         \
      {                                                                  \
        if ( !*_pnode )                                                  \
        {                                                                \
          FT_ERROR(( "FTC_CACHE_LOOKUP_CMP: oops!!! node missing\n" ));  \
          goto NewNode_;                                                 \
        }                                                                \
        else                                                             \
          _pnode = &((*_pnode)->link);                                   \
      }                                                                  \
    }                                                                    \
                                                                         \
    /* Reorder the list to move the found node to the `top' */           \
    if ( _node != *_bucket )                                             \
    {                                                                    \
      *_pnode     = _node->link;                                         \
      _node->link = *_bucket;                                            \
      *_bucket    = _node;                                               \
    }                                                                    \
                                                                         \
    /* Update MRU list */                                                \
    {                                                                    \
      FTC_Manager  _manager = _cache->manager;                           \
      void*        _nl      = &_manager->nodes_list;                     \
                                                                         \
                                                                         \
      if ( _node != _manager->nodes_list )                               \
        FTC_MruNode_Up( (FTC_MruNode*)_nl,                               \
                        (FTC_MruNode)_node );                            \
    }                                                                    \
    goto Ok_;                                                            \
                                                                         \
  NewNode_:                                                              \
    error = FTC_Cache_NewNode( _cache, _hash, query, &_node );           \
                                                                         \
  Ok_:                                                                   \
    node = _node;                                                        \
  FT_END_STMNT

#else /* !FTC_INLINE */

#define FTC_CACHE_LOOKUP_CMP( cache, nodecmp, hash, query, node, error ) \
  FT_BEGIN_STMNT                                                         \
    error = FTC_Cache_Lookup( FTC_CACHE( cache ), hash, query,           \
                              (FTC_Node*)&(node) );                      \
  FT_END_STMNT

#endif /* !FTC_INLINE */


  /*
   * This macro, together with FTC_CACHE_TRYLOOP_END, defines a retry
   * loop to flush the cache repeatedly in case of memory overflows.
   *
   * It is used when creating a new cache node, or within a lookup
   * that needs to allocate data (e.g. the sbit cache lookup).
   *
   * Example:
   *
   *   {
   *     FTC_CACHE_TRYLOOP( cache )
   *       error = load_data( ... );
   *     FTC_CACHE_TRYLOOP_END()
   *   }
   *
   */
#define FTC_CACHE_TRYLOOP( cache )                           \
  {                                                          \
    FTC_Manager  _try_manager = FTC_CACHE( cache )->manager; \
    FT_UInt      _try_count   = 4;                           \
                                                             \
                                                             \
    for (;;)                                                 \
    {                                                        \
      FT_UInt  _try_done;


#define FTC_CACHE_TRYLOOP_END( list_changed )                     \
      if ( !error || FT_ERR_NEQ( error, Out_Of_Memory ) )         \
        break;                                                    \
                                                                  \
      _try_done = FTC_Manager_FlushN( _try_manager, _try_count ); \
      if ( _try_done > 0 && list_changed != NULL )                \
        *(FT_Bool*)( list_changed ) = TRUE;                       \
                                                                  \
      if ( _try_done == 0 )                                       \
        break;                                                    \
                                                                  \
      if ( _try_done == _try_count )                              \
      {                                                           \
        _try_count *= 2;                                          \
        if ( _try_count < _try_done              ||               \
            _try_count > _try_manager->num_nodes )                \
          _try_count = _try_manager->num_nodes;                   \
      }                                                           \
    }                                                             \
  }

 /* */

FT_END_HEADER


#endif /* FTCCACHE_H_ */


/* END */
