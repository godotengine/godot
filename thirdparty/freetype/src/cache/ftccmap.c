/***************************************************************************/
/*                                                                         */
/*  ftccmap.c                                                              */
/*                                                                         */
/*    FreeType CharMap cache (body)                                        */
/*                                                                         */
/*  Copyright 2000-2018 by                                                 */
/*  David Turner, Robert Wilhelm, and Werner Lemberg.                      */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/


#include <ft2build.h>
#include FT_FREETYPE_H
#include FT_CACHE_H
#include "ftcmanag.h"
#include FT_INTERNAL_MEMORY_H
#include FT_INTERNAL_OBJECTS_H
#include FT_INTERNAL_DEBUG_H

#include "ftccback.h"
#include "ftcerror.h"

#undef  FT_COMPONENT
#define FT_COMPONENT  trace_cache


  /*************************************************************************/
  /*                                                                       */
  /* Each FTC_CMapNode contains a simple array to map a range of character */
  /* codes to equivalent glyph indices.                                    */
  /*                                                                       */
  /* For now, the implementation is very basic: Each node maps a range of  */
  /* 128 consecutive character codes to their corresponding glyph indices. */
  /*                                                                       */
  /* We could do more complex things, but I don't think it is really very  */
  /* useful.                                                               */
  /*                                                                       */
  /*************************************************************************/


  /* number of glyph indices / character code per node */
#define FTC_CMAP_INDICES_MAX  128

  /* compute a query/node hash */
#define FTC_CMAP_HASH( faceid, index, charcode )         \
          ( FTC_FACE_ID_HASH( faceid ) + 211 * (index) + \
            ( (charcode) / FTC_CMAP_INDICES_MAX )      )

  /* the charmap query */
  typedef struct  FTC_CMapQueryRec_
  {
    FTC_FaceID  face_id;
    FT_UInt     cmap_index;
    FT_UInt32   char_code;

  } FTC_CMapQueryRec, *FTC_CMapQuery;

#define FTC_CMAP_QUERY( x )  ((FTC_CMapQuery)(x))

  /* the cmap cache node */
  typedef struct  FTC_CMapNodeRec_
  {
    FTC_NodeRec  node;
    FTC_FaceID   face_id;
    FT_UInt      cmap_index;
    FT_UInt32    first;                         /* first character in node */
    FT_UInt16    indices[FTC_CMAP_INDICES_MAX]; /* array of glyph indices  */

  } FTC_CMapNodeRec, *FTC_CMapNode;

#define FTC_CMAP_NODE( x ) ( (FTC_CMapNode)( x ) )

  /* if (indices[n] == FTC_CMAP_UNKNOWN), we assume that the corresponding */
  /* glyph indices haven't been queried through FT_Get_Glyph_Index() yet   */
#define FTC_CMAP_UNKNOWN  (FT_UInt16)~0


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                        CHARMAP NODES                          *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/


  FT_CALLBACK_DEF( void )
  ftc_cmap_node_free( FTC_Node   ftcnode,
                      FTC_Cache  cache )
  {
    FTC_CMapNode  node   = (FTC_CMapNode)ftcnode;
    FT_Memory     memory = cache->memory;


    FT_FREE( node );
  }


  /* initialize a new cmap node */
  FT_CALLBACK_DEF( FT_Error )
  ftc_cmap_node_new( FTC_Node   *ftcanode,
                     FT_Pointer  ftcquery,
                     FTC_Cache   cache )
  {
    FTC_CMapNode  *anode  = (FTC_CMapNode*)ftcanode;
    FTC_CMapQuery  query  = (FTC_CMapQuery)ftcquery;
    FT_Error       error;
    FT_Memory      memory = cache->memory;
    FTC_CMapNode   node   = NULL;
    FT_UInt        nn;


    if ( !FT_NEW( node ) )
    {
      node->face_id    = query->face_id;
      node->cmap_index = query->cmap_index;
      node->first      = (query->char_code / FTC_CMAP_INDICES_MAX) *
                         FTC_CMAP_INDICES_MAX;

      for ( nn = 0; nn < FTC_CMAP_INDICES_MAX; nn++ )
        node->indices[nn] = FTC_CMAP_UNKNOWN;
    }

    *anode = node;
    return error;
  }


  /* compute the weight of a given cmap node */
  FT_CALLBACK_DEF( FT_Offset )
  ftc_cmap_node_weight( FTC_Node   cnode,
                        FTC_Cache  cache )
  {
    FT_UNUSED( cnode );
    FT_UNUSED( cache );

    return sizeof ( *cnode );
  }


  /* compare a cmap node to a given query */
  FT_CALLBACK_DEF( FT_Bool )
  ftc_cmap_node_compare( FTC_Node    ftcnode,
                         FT_Pointer  ftcquery,
                         FTC_Cache   cache,
                         FT_Bool*    list_changed )
  {
    FTC_CMapNode   node  = (FTC_CMapNode)ftcnode;
    FTC_CMapQuery  query = (FTC_CMapQuery)ftcquery;
    FT_UNUSED( cache );


    if ( list_changed )
      *list_changed = FALSE;
    if ( node->face_id    == query->face_id    &&
         node->cmap_index == query->cmap_index )
    {
      FT_UInt32  offset = (FT_UInt32)( query->char_code - node->first );


      return FT_BOOL( offset < FTC_CMAP_INDICES_MAX );
    }

    return 0;
  }


  FT_CALLBACK_DEF( FT_Bool )
  ftc_cmap_node_remove_faceid( FTC_Node    ftcnode,
                               FT_Pointer  ftcface_id,
                               FTC_Cache   cache,
                               FT_Bool*    list_changed )
  {
    FTC_CMapNode  node    = (FTC_CMapNode)ftcnode;
    FTC_FaceID    face_id = (FTC_FaceID)ftcface_id;
    FT_UNUSED( cache );


    if ( list_changed )
      *list_changed = FALSE;
    return FT_BOOL( node->face_id == face_id );
  }


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                    GLYPH IMAGE CACHE                          *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/


  static
  const FTC_CacheClassRec  ftc_cmap_cache_class =
  {
    ftc_cmap_node_new,           /* FTC_Node_NewFunc      node_new           */
    ftc_cmap_node_weight,        /* FTC_Node_WeightFunc   node_weight        */
    ftc_cmap_node_compare,       /* FTC_Node_CompareFunc  node_compare       */
    ftc_cmap_node_remove_faceid, /* FTC_Node_CompareFunc  node_remove_faceid */
    ftc_cmap_node_free,          /* FTC_Node_FreeFunc     node_free          */

    sizeof ( FTC_CacheRec ),
    ftc_cache_init,              /* FTC_Cache_InitFunc    cache_init         */
    ftc_cache_done,              /* FTC_Cache_DoneFunc    cache_done         */
  };


  /* documentation is in ftcache.h */

  FT_EXPORT_DEF( FT_Error )
  FTC_CMapCache_New( FTC_Manager     manager,
                     FTC_CMapCache  *acache )
  {
    return FTC_Manager_RegisterCache( manager,
                                      &ftc_cmap_cache_class,
                                      FTC_CACHE_P( acache ) );
  }


  /* documentation is in ftcache.h */

  FT_EXPORT_DEF( FT_UInt )
  FTC_CMapCache_Lookup( FTC_CMapCache  cmap_cache,
                        FTC_FaceID     face_id,
                        FT_Int         cmap_index,
                        FT_UInt32      char_code )
  {
    FTC_Cache         cache = FTC_CACHE( cmap_cache );
    FTC_CMapQueryRec  query;
    FTC_Node          node;
    FT_Error          error;
    FT_UInt           gindex = 0;
    FT_Offset         hash;
    FT_Int            no_cmap_change = 0;


    if ( cmap_index < 0 )
    {
      /* Treat a negative cmap index as a special value, meaning that you */
      /* don't want to change the FT_Face's character map through this    */
      /* call.  This can be useful if the face requester callback already */
      /* sets the face's charmap to the appropriate value.                */

      no_cmap_change = 1;
      cmap_index     = 0;
    }

    if ( !cache )
    {
      FT_TRACE0(( "FTC_CMapCache_Lookup: bad arguments, returning 0\n" ));
      return 0;
    }

    query.face_id    = face_id;
    query.cmap_index = (FT_UInt)cmap_index;
    query.char_code  = char_code;

    hash = FTC_CMAP_HASH( face_id, (FT_UInt)cmap_index, char_code );

#if 1
    FTC_CACHE_LOOKUP_CMP( cache, ftc_cmap_node_compare, hash, &query,
                          node, error );
#else
    error = FTC_Cache_Lookup( cache, hash, &query, &node );
#endif
    if ( error )
      goto Exit;

    FT_ASSERT( (FT_UInt)( char_code - FTC_CMAP_NODE( node )->first ) <
                FTC_CMAP_INDICES_MAX );

    /* something rotten can happen with rogue clients */
    if ( (FT_UInt)( char_code - FTC_CMAP_NODE( node )->first >=
                    FTC_CMAP_INDICES_MAX ) )
      return 0; /* XXX: should return appropriate error */

    gindex = FTC_CMAP_NODE( node )->indices[char_code -
                                            FTC_CMAP_NODE( node )->first];
    if ( gindex == FTC_CMAP_UNKNOWN )
    {
      FT_Face  face;


      gindex = 0;

      error = FTC_Manager_LookupFace( cache->manager,
                                      FTC_CMAP_NODE( node )->face_id,
                                      &face );
      if ( error )
        goto Exit;

      if ( (FT_UInt)cmap_index < (FT_UInt)face->num_charmaps )
      {
        FT_CharMap  old, cmap  = NULL;


        old  = face->charmap;
        cmap = face->charmaps[cmap_index];

        if ( old != cmap && !no_cmap_change )
          FT_Set_Charmap( face, cmap );

        gindex = FT_Get_Char_Index( face, char_code );

        if ( old != cmap && !no_cmap_change )
          FT_Set_Charmap( face, old );
      }

      FTC_CMAP_NODE( node )->indices[char_code -
                                     FTC_CMAP_NODE( node )->first]
        = (FT_UShort)gindex;
    }

  Exit:
    return gindex;
  }


/* END */
