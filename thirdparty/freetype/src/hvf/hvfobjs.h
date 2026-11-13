/****************************************************************************
 *
 * hvfobjs.h
 *
 *   HVF objects manager (specification).
 *
 * Copyright (C) 2025-2026 by
 * Apple Inc.
 * written by Deborah Goldsmith <goldsmit@apple.com>
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#ifndef HVFOBJS_H_
#define HVFOBJS_H_

#include <freetype/freetype.h>
#include <freetype/ftglyph.h>
#include <freetype/internal/ftobjs.h>
#include <freetype/internal/tttypes.h>  /* For TT_FaceRec. */


FT_BEGIN_HEADER


  /**************************************************************************
   *
   * @type:
   *   HVF_Face
   *
   * @description:
   *   A handle to an HVF face object.  Always available for compatibility.
   */
  typedef struct HVF_FaceRec_*  HVF_Face;


  /**************************************************************************
   *
   * @struct:
   *   HVF_FaceRec
   *
   * @description:

   *   HVF face record.  This structure inherits from `TT_FaceRec` instead of
   *   `FT_FaceRec` because HVF fonts are TrueType/OpenType fonts with
   *   additional HVF tables, and the HVF driver calls into SFNT functions
   *   that require `TT_Face` access.
   *
   *   NOTE: Structure is always available with all fields always present.
   *         Only the implementation functions are conditionally compiled.
   */
  typedef struct  HVF_FaceRec_
  {
    TT_FaceRec  root;      /* Inherit from `TT_FaceRec`. */

    void*  renderer;       /* `HVFPartRenderer` storage or NULL. */

    /* HVF-specific table data (memory-mapped). */
    FT_Byte*  hvgl_data;   /* 'hvgl' table data                       */
    FT_ULong  hvgl_size;
    FT_Byte*  hvpm_data;   /* 'hvpm' table data, NULL if not present. */
    FT_ULong  hvpm_size;

    /* Cache management. */
    FT_UInt  cache_count;  /* Clear cache every N glyphs. */

    /* Variation axis data (allocated once in `hvf_face_init`). */
    FT_UInt  num_axes;     /* Number of variation axes.          */
    void*    axis_coords;  /* Pre-converted HVF axis coordinates */
                           /* (HVFAxisValue* when configured).   */

  } HVF_FaceRec;


  /* Conditional declarations - only when HVF is enabled. */
#ifdef FT_CONFIG_OPTION_HVF

#include <hvf/Scaler.h>

  /* Runtime availability checking for Apple platforms. */
#ifdef HVF_RUNTIME_AVAILABLE

  /* Single macro that encapsulates the `__builtin_available` */
  /* check for C/C++.                                         */
  /*                                                          */
  /* BEWARE: Only use this as `if ( HVS_IS_AVAILABLE ) ...`   */
  /*         Any deviation from this form (also including     */
  /*         negation) causes `clang` to emit a warning.      */
#define HVF_IS_AVAILABLE  __builtin_available( macOS 15.4, iOS 18.4, * )

#else

  /* Non-Apple platforms: No runtime check needed. */
#define HVF_IS_AVAILABLE  ( 1 )

#endif /* HVF_RUNTIME_AVAILABLE */


  /**************************************************************************
   *
   * @struct:
   *   HVF_RenderContext
   *
   * @description:
   *   Context for HVF rendering callbacks.
   */
  typedef struct  HVF_RenderContext_
  {
    FT_GlyphLoader  loader;     /* Standard FreeType loader.            */
    FT_Outline*     outline;    /* Points to `loader->current.outline`. */
    FT_Bool         path_begun; /* Path state tracking.                 */

    /* Pre-calculated scaling factors for efficient coordinate conversion. */
    HVFXYCoord  x_scale_fixed; /* x_scale * 65536.0 (or just 65536.0) */
    HVFXYCoord  y_scale_fixed; /* y_scale * 65536.0 (or just 65536.0) */

  } HVF_RenderContext;


  /* Function declarations. */
  FT_LOCAL( void )
  hvf_face_done( FT_Face  face );

  FT_LOCAL( FT_Error )
  hvf_refresh_axis_coordinates( HVF_Face  face );


  /* Convert FreeType normalized coordinates to HVF axis values. */
#define FT_COORD_TO_HVF_AXIS( coord )         \
          ( (HVFAxisValue)(coord) / 65536.0 )

#endif /* FT_CONFIG_OPTION_HVF */


  FT_LOCAL( FT_Error )
  hvf_face_init( FT_Stream      stream,
                 FT_Face        face,
                 FT_Int         typeface_index,
                 FT_Int         num_params,
                 FT_Parameter*  parameters );


FT_END_HEADER

#endif /* HVFOBJS_H_ */


/* END */
