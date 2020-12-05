/****************************************************************************
 *
 * ftcsbits.c
 *
 *   FreeType sbits manager (body).
 *
 * Copyright (C) 2000-2020 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#include <freetype/ftcache.h>
#include "ftcsbits.h"
#include <freetype/internal/ftobjs.h>
#include <freetype/internal/ftdebug.h>
#include <freetype/fterrors.h>

#include "ftccback.h"
#include "ftcerror.h"

#undef  FT_COMPONENT
#define FT_COMPONENT  cache


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                     SBIT CACHE NODES                          *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/


  static FT_Error
  ftc_sbit_copy_bitmap( FTC_SBit    sbit,
                        FT_Bitmap*  bitmap,
                        FT_Memory   memory )
  {
    FT_Error  error;
    FT_Int    pitch = bitmap->pitch;
    FT_ULong  size;


    if ( pitch < 0 )
      pitch = -pitch;

    size = (FT_ULong)pitch * bitmap->rows;
    if ( !size )
      return FT_Err_Ok;

    if ( !FT_ALLOC( sbit->buffer, size ) )
      FT_MEM_COPY( sbit->buffer, bitmap->buffer, size );

    return error;
  }


  FT_LOCAL_DEF( void )
  ftc_snode_free( FTC_Node   ftcsnode,
                  FTC_Cache  cache )
  {
    FTC_SNode  snode  = (FTC_SNode)ftcsnode;
    FTC_SBit   sbit   = snode->sbits;
    FT_UInt    count  = snode->count;
    FT_Memory  memory = cache->memory;


    for ( ; count > 0; sbit++, count-- )
      FT_FREE( sbit->buffer );

    FTC_GNode_Done( FTC_GNODE( snode ), cache );

    FT_FREE( snode );
  }


  FT_LOCAL_DEF( void )
  FTC_SNode_Free( FTC_SNode  snode,
                  FTC_Cache  cache )
  {
    ftc_snode_free( FTC_NODE( snode ), cache );
  }


  /*
   * This function tries to load a small bitmap within a given FTC_SNode.
   * Note that it returns a non-zero error code _only_ in the case of
   * out-of-memory condition.  For all other errors (e.g., corresponding
   * to a bad font file), this function will mark the sbit as `unavailable'
   * and return a value of 0.
   *
   * You should also read the comment within the @ftc_snode_compare
   * function below to see how out-of-memory is handled during a lookup.
   */
  static FT_Error
  ftc_snode_load( FTC_SNode    snode,
                  FTC_Manager  manager,
                  FT_UInt      gindex,
                  FT_ULong    *asize )
  {
    FT_Error          error;
    FTC_GNode         gnode  = FTC_GNODE( snode );
    FTC_Family        family = gnode->family;
    FT_Memory         memory = manager->memory;
    FT_Face           face;
    FTC_SBit          sbit;
    FTC_SFamilyClass  clazz;


    if ( (FT_UInt)(gindex - gnode->gindex) >= snode->count )
    {
      FT_ERROR(( "ftc_snode_load: invalid glyph index" ));
      return FT_THROW( Invalid_Argument );
    }

    sbit  = snode->sbits + ( gindex - gnode->gindex );
    clazz = (FTC_SFamilyClass)family->clazz;

    sbit->buffer = 0;

    error = clazz->family_load_glyph( family, gindex, manager, &face );
    if ( error )
      goto BadGlyph;

    {
      FT_Int        temp;
      FT_GlyphSlot  slot   = face->glyph;
      FT_Bitmap*    bitmap = &slot->bitmap;
      FT_Pos        xadvance, yadvance; /* FT_GlyphSlot->advance.{x|y} */


      if ( slot->format != FT_GLYPH_FORMAT_BITMAP )
      {
        FT_TRACE0(( "ftc_snode_load:"
                    " glyph loaded didn't return a bitmap\n" ));
        goto BadGlyph;
      }

      /* Check whether our values fit into 8-bit containers!    */
      /* If this is not the case, our bitmap is too large       */
      /* and we will leave it as `missing' with sbit.buffer = 0 */

#define CHECK_CHAR( d )  ( temp = (FT_Char)d, (FT_Int) temp == (FT_Int) d )
#define CHECK_BYTE( d )  ( temp = (FT_Byte)d, (FT_UInt)temp == (FT_UInt)d )

      /* horizontal advance in pixels */
      xadvance = ( slot->advance.x + 32 ) >> 6;
      yadvance = ( slot->advance.y + 32 ) >> 6;

      if ( !CHECK_BYTE( bitmap->rows  )     ||
           !CHECK_BYTE( bitmap->width )     ||
           !CHECK_CHAR( bitmap->pitch )     ||
           !CHECK_CHAR( slot->bitmap_left ) ||
           !CHECK_CHAR( slot->bitmap_top  ) ||
           !CHECK_CHAR( xadvance )          ||
           !CHECK_CHAR( yadvance )          )
      {
        FT_TRACE2(( "ftc_snode_load:"
                    " glyph too large for small bitmap cache\n"));
        goto BadGlyph;
      }

      sbit->width     = (FT_Byte)bitmap->width;
      sbit->height    = (FT_Byte)bitmap->rows;
      sbit->pitch     = (FT_Char)bitmap->pitch;
      sbit->left      = (FT_Char)slot->bitmap_left;
      sbit->top       = (FT_Char)slot->bitmap_top;
      sbit->xadvance  = (FT_Char)xadvance;
      sbit->yadvance  = (FT_Char)yadvance;
      sbit->format    = (FT_Byte)bitmap->pixel_mode;
      sbit->max_grays = (FT_Byte)(bitmap->num_grays - 1);

      /* copy the bitmap into a new buffer -- ignore error */
      error = ftc_sbit_copy_bitmap( sbit, bitmap, memory );

      /* now, compute size */
      if ( asize )
        *asize = (FT_ULong)FT_ABS( sbit->pitch ) * sbit->height;

    } /* glyph loading successful */

    /* ignore the errors that might have occurred --   */
    /* we mark unloaded glyphs with `sbit.buffer == 0' */
    /* and `width == 255', `height == 0'               */
    /*                                                 */
    if ( error && FT_ERR_NEQ( error, Out_Of_Memory ) )
    {
    BadGlyph:
      sbit->width  = 255;
      sbit->height = 0;
      sbit->buffer = NULL;
      error        = FT_Err_Ok;
      if ( asize )
        *asize = 0;
    }

    return error;
  }


  FT_LOCAL_DEF( FT_Error )
  FTC_SNode_New( FTC_SNode  *psnode,
                 FTC_GQuery  gquery,
                 FTC_Cache   cache )
  {
    FT_Memory   memory = cache->memory;
    FT_Error    error;
    FTC_SNode   snode  = NULL;
    FT_UInt     gindex = gquery->gindex;
    FTC_Family  family = gquery->family;

    FTC_SFamilyClass  clazz = FTC_CACHE_SFAMILY_CLASS( cache );
    FT_UInt           total;
    FT_UInt           node_count;


    total = clazz->family_get_count( family, cache->manager );
    if ( total == 0 || gindex >= total )
    {
      error = FT_THROW( Invalid_Argument );
      goto Exit;
    }

    if ( !FT_NEW( snode ) )
    {
      FT_UInt  count, start;


      start = gindex - ( gindex % FTC_SBIT_ITEMS_PER_NODE );
      count = total - start;
      if ( count > FTC_SBIT_ITEMS_PER_NODE )
        count = FTC_SBIT_ITEMS_PER_NODE;

      FTC_GNode_Init( FTC_GNODE( snode ), start, family );

      snode->count = count;
      for ( node_count = 0; node_count < count; node_count++ )
      {
        snode->sbits[node_count].width = 255;
      }

      error = ftc_snode_load( snode,
                              cache->manager,
                              gindex,
                              NULL );
      if ( error )
      {
        FTC_SNode_Free( snode, cache );
        snode = NULL;
      }
    }

  Exit:
    *psnode = snode;
    return error;
  }


  FT_LOCAL_DEF( FT_Error )
  ftc_snode_new( FTC_Node   *ftcpsnode,
                 FT_Pointer  ftcgquery,
                 FTC_Cache   cache )
  {
    FTC_SNode  *psnode = (FTC_SNode*)ftcpsnode;
    FTC_GQuery  gquery = (FTC_GQuery)ftcgquery;


    return FTC_SNode_New( psnode, gquery, cache );
  }


  FT_LOCAL_DEF( FT_Offset )
  ftc_snode_weight( FTC_Node   ftcsnode,
                    FTC_Cache  cache )
  {
    FTC_SNode  snode = (FTC_SNode)ftcsnode;
    FT_UInt    count = snode->count;
    FTC_SBit   sbit  = snode->sbits;
    FT_Int     pitch;
    FT_Offset  size;

    FT_UNUSED( cache );


    FT_ASSERT( snode->count <= FTC_SBIT_ITEMS_PER_NODE );

    /* the node itself */
    size = sizeof ( *snode );

    for ( ; count > 0; count--, sbit++ )
    {
      if ( sbit->buffer )
      {
        pitch = sbit->pitch;
        if ( pitch < 0 )
          pitch = -pitch;

        /* add the size of a given glyph image */
        size += (FT_Offset)pitch * sbit->height;
      }
    }

    return size;
  }


#if 0

  FT_LOCAL_DEF( FT_Offset )
  FTC_SNode_Weight( FTC_SNode  snode )
  {
    return ftc_snode_weight( FTC_NODE( snode ), NULL );
  }

#endif /* 0 */


  FT_LOCAL_DEF( FT_Bool )
  ftc_snode_compare( FTC_Node    ftcsnode,
                     FT_Pointer  ftcgquery,
                     FTC_Cache   cache,
                     FT_Bool*    list_changed )
  {
    FTC_SNode   snode  = (FTC_SNode)ftcsnode;
    FTC_GQuery  gquery = (FTC_GQuery)ftcgquery;
    FTC_GNode   gnode  = FTC_GNODE( snode );
    FT_UInt     gindex = gquery->gindex;
    FT_Bool     result;


    if (list_changed)
      *list_changed = FALSE;
    result = FT_BOOL( gnode->family == gquery->family                    &&
                      (FT_UInt)( gindex - gnode->gindex ) < snode->count );
    if ( result )
    {
      /* check if we need to load the glyph bitmap now */
      FTC_SBit  sbit = snode->sbits + ( gindex - gnode->gindex );


      /*
       * The following code illustrates what to do when you want to
       * perform operations that may fail within a lookup function.
       *
       * Here, we want to load a small bitmap on-demand; we thus
       * need to call the `ftc_snode_load' function which may return
       * a non-zero error code only when we are out of memory (OOM).
       *
       * The correct thing to do is to use @FTC_CACHE_TRYLOOP and
       * @FTC_CACHE_TRYLOOP_END in order to implement a retry loop
       * that is capable of flushing the cache incrementally when
       * an OOM errors occur.
       *
       * However, we need to `lock' the node before this operation to
       * prevent it from being flushed within the loop.
       *
       * When we exit the loop, we unlock the node, then check the `error'
       * variable.  If it is non-zero, this means that the cache was
       * completely flushed and that no usable memory was found to load
       * the bitmap.
       *
       * We then prefer to return a value of 0 (i.e., NO MATCH).  This
       * ensures that the caller will try to allocate a new node.
       * This operation consequently _fail_ and the lookup function
       * returns the appropriate OOM error code.
       *
       * Note that `buffer == NULL && width == 255' is a hack used to
       * tag `unavailable' bitmaps in the array.  We should never try
       * to load these.
       *
       */

      if ( !sbit->buffer && sbit->width == 255 )
      {
        FT_ULong  size;
        FT_Error  error;


        ftcsnode->ref_count++;  /* lock node to prevent flushing */
                                /* in retry loop                 */

        FTC_CACHE_TRYLOOP( cache )
        {
          error = ftc_snode_load( snode, cache->manager, gindex, &size );
        }
        FTC_CACHE_TRYLOOP_END( list_changed );

        ftcsnode->ref_count--;  /* unlock the node */

        if ( error )
          result = 0;
        else
          cache->manager->cur_weight += size;
      }
    }

    return result;
  }


#ifdef FTC_INLINE

  FT_LOCAL_DEF( FT_Bool )
  FTC_SNode_Compare( FTC_SNode   snode,
                     FTC_GQuery  gquery,
                     FTC_Cache   cache,
                     FT_Bool*    list_changed )
  {
    return ftc_snode_compare( FTC_NODE( snode ), gquery,
                              cache, list_changed );
  }

#endif

/* END */
