/****************************************************************************
 *
 * ftcmanag.h
 *
 *   FreeType Cache Manager (specification).
 *
 * Copyright (C) 2000-2025 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
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
   * A cache manager is in charge of the following:
   *
   * - Maintain a mapping between generic FTC_FaceIDs and live FT_Face
   *   objects.  The mapping itself is performed through a user-provided
   *   callback.  However, the manager maintains a small cache of FT_Face
   *   and FT_Size objects in order to speed up things considerably.
   *
   * - Manage one or more cache objects.  Each cache is in charge of
   *   holding a varying number of `cache nodes'.  Each cache node
   *   represents a minimal amount of individually accessible cached
   *   data.  For example, a cache node can be an FT_Glyph image
   *   containing a vector outline, or some glyph metrics, or anything
   *   else.
   *
   *   Each cache node has a certain size in bytes that is added to the
   *   total amount of `cache memory' within the manager.
   *
   *   All cache nodes are located in a global LRU list, where the oldest
   *   node is at the tail of the list.
   *
   *   Each node belongs to a single cache, and includes a reference
   *   count to avoid destroying it (due to caching).
   *
   */


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


#ifndef FTCMANAG_H_
#define FTCMANAG_H_


#include <freetype/ftcache.h>
#include "ftcmru.h"
#include "ftccache.h"


FT_BEGIN_HEADER


  /**************************************************************************
   *
   * @Section:
   *   cache_subsystem
   *
   */


#define FTC_MAX_FACES_DEFAULT  2
#define FTC_MAX_SIZES_DEFAULT  4
#define FTC_MAX_BYTES_DEFAULT  200000L  /* ~200kByte by default */

  /* maximum number of caches registered in a single manager */
#define FTC_MAX_CACHES         16


  typedef struct  FTC_ManagerRec_
  {
    FT_Library          library;
    FT_Memory           memory;

    FTC_Node            nodes_list;
    FT_Offset           max_weight;
    FT_Offset           cur_weight;
    FT_UInt             num_nodes;

    FTC_Cache           caches[FTC_MAX_CACHES];
    FT_UInt             num_caches;

    FTC_MruListRec      faces;
    FTC_MruListRec      sizes;

    FT_Pointer          request_data;
    FTC_Face_Requester  request_face;

  } FTC_ManagerRec;


  /**************************************************************************
   *
   * @Function:
   *   FTC_Manager_Compress
   *
   * @Description:
   *   This function is used to check the state of the cache manager if
   *   its `num_bytes' field is greater than its `max_bytes' field.  It
   *   will flush as many old cache nodes as possible (ignoring cache
   *   nodes with a non-zero reference count).
   *
   * @InOut:
   *   manager ::
   *     A handle to the cache manager.
   *
   * @Note:
   *   Client applications should not call this function directly.  It is
   *   normally invoked by specific cache implementations.
   *
   *   The reason this function is exported is to allow client-specific
   *   cache classes.
   */
  FT_LOCAL( void )
  FTC_Manager_Compress( FTC_Manager  manager );


  /* try to flush `count' old nodes from the cache; return the number
   * of really flushed nodes
   */
  FT_LOCAL( FT_UInt )
  FTC_Manager_FlushN( FTC_Manager  manager,
                      FT_UInt      count );


  /* this must be used internally for the moment */
  FT_LOCAL( FT_Error )
  FTC_Manager_RegisterCache( FTC_Manager      manager,
                             FTC_CacheClass   clazz,
                             FTC_Cache       *acache );

 /* */

#define FTC_SCALER_COMPARE( a, b )                \
    ( (a)->face_id      == (b)->face_id      &&   \
      (a)->width        == (b)->width        &&   \
      (a)->height       == (b)->height       &&   \
      ((a)->pixel != 0) == ((b)->pixel != 0) &&   \
      ( (a)->pixel ||                             \
        ( (a)->x_res == (b)->x_res &&             \
          (a)->y_res == (b)->y_res ) ) )

#define FTC_SCALER_HASH( q )                                 \
    ( FTC_FACE_ID_HASH( (q)->face_id ) +                     \
      (q)->width + (q)->height*7 +                           \
      ( (q)->pixel ? 0 : ( (q)->x_res*33 ^ (q)->y_res*61 ) ) )

 /* */

FT_END_HEADER

#endif /* FTCMANAG_H_ */


/* END */
