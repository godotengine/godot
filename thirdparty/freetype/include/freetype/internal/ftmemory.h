/****************************************************************************
 *
 * ftmemory.h
 *
 *   The FreeType memory management macros (specification).
 *
 * Copyright (C) 1996-2023 by
 * David Turner, Robert Wilhelm, and Werner Lemberg
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#ifndef FTMEMORY_H_
#define FTMEMORY_H_


#include <ft2build.h>
#include FT_CONFIG_CONFIG_H
#include <freetype/fttypes.h>

#include "compiler-macros.h"

FT_BEGIN_HEADER


  /**************************************************************************
   *
   * @macro:
   *   FT_SET_ERROR
   *
   * @description:
   *   This macro is used to set an implicit 'error' variable to a given
   *   expression's value (usually a function call), and convert it to a
   *   boolean which is set whenever the value is != 0.
   */
#undef  FT_SET_ERROR
#define FT_SET_ERROR( expression ) \
          ( ( error = (expression) ) != 0 )



  /*************************************************************************/
  /*************************************************************************/
  /*************************************************************************/
  /****                                                                 ****/
  /****                                                                 ****/
  /****                           M E M O R Y                           ****/
  /****                                                                 ****/
  /****                                                                 ****/
  /*************************************************************************/
  /*************************************************************************/
  /*************************************************************************/


  /* The calculation `NULL + n' is undefined in C.  Even if the resulting */
  /* pointer doesn't get dereferenced, this causes warnings with          */
  /* sanitizers.                                                          */
  /*                                                                      */
  /* We thus provide a macro that should be used if `base' can be NULL.   */
#define FT_OFFSET( base, count )  ( (base) ? (base) + (count) : NULL )


  /*
   * C++ refuses to handle statements like p = (void*)anything, with `p' a
   * typed pointer.  Since we don't have a `typeof' operator in standard C++,
   * we have to use a template to emulate it.
   */

#ifdef __cplusplus

extern "C++"
{
  template <typename T> inline T*
  cplusplus_typeof(        T*,
                    void  *v )
  {
    return static_cast <T*> ( v );
  }
}

#define FT_ASSIGNP( p, val )  (p) = cplusplus_typeof( (p), (val) )

#else

#define FT_ASSIGNP( p, val )  (p) = (val)

#endif



#ifdef FT_DEBUG_MEMORY

  FT_BASE( const char* )  ft_debug_file_;
  FT_BASE( long )         ft_debug_lineno_;

#define FT_DEBUG_INNER( exp )  ( ft_debug_file_   = __FILE__, \
                                 ft_debug_lineno_ = __LINE__, \
                                 (exp) )

#define FT_ASSIGNP_INNER( p, exp )  ( ft_debug_file_   = __FILE__, \
                                      ft_debug_lineno_ = __LINE__, \
                                      FT_ASSIGNP( p, exp ) )

#else /* !FT_DEBUG_MEMORY */

#define FT_DEBUG_INNER( exp )       (exp)
#define FT_ASSIGNP_INNER( p, exp )  FT_ASSIGNP( p, exp )

#endif /* !FT_DEBUG_MEMORY */


  /*
   * The allocation functions return a pointer, and the error code is written
   * to through the `p_error' parameter.
   */

  /* The `q' variants of the functions below (`q' for `quick') don't fill */
  /* the allocated or reallocated memory with zero bytes.                 */

  FT_BASE( FT_Pointer )
  ft_mem_alloc( FT_Memory  memory,
                FT_Long    size,
                FT_Error  *p_error );

  FT_BASE( FT_Pointer )
  ft_mem_qalloc( FT_Memory  memory,
                 FT_Long    size,
                 FT_Error  *p_error );

  FT_BASE( FT_Pointer )
  ft_mem_realloc( FT_Memory  memory,
                  FT_Long    item_size,
                  FT_Long    cur_count,
                  FT_Long    new_count,
                  void*      block,
                  FT_Error  *p_error );

  FT_BASE( FT_Pointer )
  ft_mem_qrealloc( FT_Memory  memory,
                   FT_Long    item_size,
                   FT_Long    cur_count,
                   FT_Long    new_count,
                   void*      block,
                   FT_Error  *p_error );

  FT_BASE( void )
  ft_mem_free( FT_Memory    memory,
               const void*  P );


  /* The `Q' variants of the macros below (`Q' for `quick') don't fill */
  /* the allocated or reallocated memory with zero bytes.              */

#define FT_MEM_ALLOC( ptr, size )                               \
          FT_ASSIGNP_INNER( ptr, ft_mem_alloc( memory,          \
                                               (FT_Long)(size), \
                                               &error ) )

#define FT_MEM_FREE( ptr )                                  \
          FT_BEGIN_STMNT                                    \
            FT_DEBUG_INNER( ft_mem_free( memory, (ptr) ) ); \
            (ptr) = NULL;                                   \
          FT_END_STMNT

#define FT_MEM_NEW( ptr )                        \
          FT_MEM_ALLOC( ptr, sizeof ( *(ptr) ) )

#define FT_MEM_REALLOC( ptr, cursz, newsz )                        \
          FT_ASSIGNP_INNER( ptr, ft_mem_realloc( memory,           \
                                                 1,                \
                                                 (FT_Long)(cursz), \
                                                 (FT_Long)(newsz), \
                                                 (ptr),            \
                                                 &error ) )

#define FT_MEM_QALLOC( ptr, size )                               \
          FT_ASSIGNP_INNER( ptr, ft_mem_qalloc( memory,          \
                                                (FT_Long)(size), \
                                                &error ) )

#define FT_MEM_QNEW( ptr )                        \
          FT_MEM_QALLOC( ptr, sizeof ( *(ptr) ) )

#define FT_MEM_QREALLOC( ptr, cursz, newsz )                        \
          FT_ASSIGNP_INNER( ptr, ft_mem_qrealloc( memory,           \
                                                  1,                \
                                                  (FT_Long)(cursz), \
                                                  (FT_Long)(newsz), \
                                                  (ptr),            \
                                                  &error ) )

#define FT_MEM_ALLOC_MULT( ptr, count, item_size )                     \
          FT_ASSIGNP_INNER( ptr, ft_mem_realloc( memory,               \
                                                 (FT_Long)(item_size), \
                                                 0,                    \
                                                 (FT_Long)(count),     \
                                                 NULL,                 \
                                                 &error ) )

#define FT_MEM_REALLOC_MULT( ptr, oldcnt, newcnt, itmsz )           \
          FT_ASSIGNP_INNER( ptr, ft_mem_realloc( memory,            \
                                                 (FT_Long)(itmsz),  \
                                                 (FT_Long)(oldcnt), \
                                                 (FT_Long)(newcnt), \
                                                 (ptr),             \
                                                 &error ) )

#define FT_MEM_QALLOC_MULT( ptr, count, item_size )                     \
          FT_ASSIGNP_INNER( ptr, ft_mem_qrealloc( memory,               \
                                                  (FT_Long)(item_size), \
                                                  0,                    \
                                                  (FT_Long)(count),     \
                                                  NULL,                 \
                                                  &error ) )

#define FT_MEM_QREALLOC_MULT( ptr, oldcnt, newcnt, itmsz )           \
          FT_ASSIGNP_INNER( ptr, ft_mem_qrealloc( memory,            \
                                                  (FT_Long)(itmsz),  \
                                                  (FT_Long)(oldcnt), \
                                                  (FT_Long)(newcnt), \
                                                  (ptr),             \
                                                  &error ) )


#define FT_MEM_SET_ERROR( cond )  ( (cond), error != 0 )


#define FT_MEM_SET( dest, byte, count )               \
          ft_memset( dest, byte, (FT_Offset)(count) )

#define FT_MEM_COPY( dest, source, count )              \
          ft_memcpy( dest, source, (FT_Offset)(count) )

#define FT_MEM_MOVE( dest, source, count )               \
          ft_memmove( dest, source, (FT_Offset)(count) )


#define FT_MEM_ZERO( dest, count )  FT_MEM_SET( dest, 0, count )

#define FT_ZERO( p )                FT_MEM_ZERO( p, sizeof ( *(p) ) )


#define FT_ARRAY_ZERO( dest, count )                             \
          FT_MEM_ZERO( dest,                                     \
                       (FT_Offset)(count) * sizeof ( *(dest) ) )

#define FT_ARRAY_COPY( dest, source, count )                     \
          FT_MEM_COPY( dest,                                     \
                       source,                                   \
                       (FT_Offset)(count) * sizeof ( *(dest) ) )

#define FT_ARRAY_MOVE( dest, source, count )                     \
          FT_MEM_MOVE( dest,                                     \
                       source,                                   \
                       (FT_Offset)(count) * sizeof ( *(dest) ) )


  /*
   * Return the maximum number of addressable elements in an array.  We limit
   * ourselves to INT_MAX, rather than UINT_MAX, to avoid any problems.
   */
#define FT_ARRAY_MAX( ptr )           ( FT_INT_MAX / sizeof ( *(ptr) ) )

#define FT_ARRAY_CHECK( ptr, count )  ( (count) <= FT_ARRAY_MAX( ptr ) )


  /**************************************************************************
   *
   * The following functions macros expect that their pointer argument is
   * _typed_ in order to automatically compute array element sizes.
   */

#define FT_MEM_NEW_ARRAY( ptr, count )                              \
          FT_ASSIGNP_INNER( ptr, ft_mem_realloc( memory,            \
                                                 sizeof ( *(ptr) ), \
                                                 0,                 \
                                                 (FT_Long)(count),  \
                                                 NULL,              \
                                                 &error ) )

#define FT_MEM_RENEW_ARRAY( ptr, cursz, newsz )                     \
          FT_ASSIGNP_INNER( ptr, ft_mem_realloc( memory,            \
                                                 sizeof ( *(ptr) ), \
                                                 (FT_Long)(cursz),  \
                                                 (FT_Long)(newsz),  \
                                                 (ptr),             \
                                                 &error ) )

#define FT_MEM_QNEW_ARRAY( ptr, count )                              \
          FT_ASSIGNP_INNER( ptr, ft_mem_qrealloc( memory,            \
                                                  sizeof ( *(ptr) ), \
                                                  0,                 \
                                                  (FT_Long)(count),  \
                                                  NULL,              \
                                                  &error ) )

#define FT_MEM_QRENEW_ARRAY( ptr, cursz, newsz )                     \
          FT_ASSIGNP_INNER( ptr, ft_mem_qrealloc( memory,            \
                                                  sizeof ( *(ptr) ), \
                                                  (FT_Long)(cursz),  \
                                                  (FT_Long)(newsz),  \
                                                  (ptr),             \
                                                  &error ) )

#define FT_ALLOC( ptr, size )                           \
          FT_MEM_SET_ERROR( FT_MEM_ALLOC( ptr, size ) )

#define FT_REALLOC( ptr, cursz, newsz )                           \
          FT_MEM_SET_ERROR( FT_MEM_REALLOC( ptr, cursz, newsz ) )

#define FT_ALLOC_MULT( ptr, count, item_size )                           \
          FT_MEM_SET_ERROR( FT_MEM_ALLOC_MULT( ptr, count, item_size ) )

#define FT_REALLOC_MULT( ptr, oldcnt, newcnt, itmsz )              \
          FT_MEM_SET_ERROR( FT_MEM_REALLOC_MULT( ptr, oldcnt,      \
                                                 newcnt, itmsz ) )

#define FT_QALLOC( ptr, size )                           \
          FT_MEM_SET_ERROR( FT_MEM_QALLOC( ptr, size ) )

#define FT_QREALLOC( ptr, cursz, newsz )                           \
          FT_MEM_SET_ERROR( FT_MEM_QREALLOC( ptr, cursz, newsz ) )

#define FT_QALLOC_MULT( ptr, count, item_size )                           \
          FT_MEM_SET_ERROR( FT_MEM_QALLOC_MULT( ptr, count, item_size ) )

#define FT_QREALLOC_MULT( ptr, oldcnt, newcnt, itmsz )              \
          FT_MEM_SET_ERROR( FT_MEM_QREALLOC_MULT( ptr, oldcnt,      \
                                                  newcnt, itmsz ) )

#define FT_FREE( ptr )  FT_MEM_FREE( ptr )

#define FT_NEW( ptr )  FT_MEM_SET_ERROR( FT_MEM_NEW( ptr ) )

#define FT_NEW_ARRAY( ptr, count )                           \
          FT_MEM_SET_ERROR( FT_MEM_NEW_ARRAY( ptr, count ) )

#define FT_RENEW_ARRAY( ptr, curcnt, newcnt )                           \
          FT_MEM_SET_ERROR( FT_MEM_RENEW_ARRAY( ptr, curcnt, newcnt ) )

#define FT_QNEW( ptr )  FT_MEM_SET_ERROR( FT_MEM_QNEW( ptr ) )

#define FT_QNEW_ARRAY( ptr, count )                           \
          FT_MEM_SET_ERROR( FT_MEM_QNEW_ARRAY( ptr, count ) )

#define FT_QRENEW_ARRAY( ptr, curcnt, newcnt )                           \
          FT_MEM_SET_ERROR( FT_MEM_QRENEW_ARRAY( ptr, curcnt, newcnt ) )


  FT_BASE( FT_Pointer )
  ft_mem_strdup( FT_Memory    memory,
                 const char*  str,
                 FT_Error    *p_error );

  FT_BASE( FT_Pointer )
  ft_mem_dup( FT_Memory    memory,
              const void*  address,
              FT_ULong     size,
              FT_Error    *p_error );


#define FT_MEM_STRDUP( dst, str )                                            \
          (dst) = (char*)ft_mem_strdup( memory, (const char*)(str), &error )

#define FT_STRDUP( dst, str )                           \
          FT_MEM_SET_ERROR( FT_MEM_STRDUP( dst, str ) )

#define FT_MEM_DUP( dst, address, size )                                    \
          (dst) = ft_mem_dup( memory, (address), (FT_ULong)(size), &error )

#define FT_DUP( dst, address, size )                           \
          FT_MEM_SET_ERROR( FT_MEM_DUP( dst, address, size ) )


  /* Return >= 1 if a truncation occurs.            */
  /* Return 0 if the source string fits the buffer. */
  /* This is *not* the same as strlcpy().           */
  FT_BASE( FT_Int )
  ft_mem_strcpyn( char*        dst,
                  const char*  src,
                  FT_ULong     size );

#define FT_STRCPYN( dst, src, size )                                         \
          ft_mem_strcpyn( (char*)dst, (const char*)(src), (FT_ULong)(size) )


FT_END_HEADER

#endif /* FTMEMORY_H_ */


/* END */
