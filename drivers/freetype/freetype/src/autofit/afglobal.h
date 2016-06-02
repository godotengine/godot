/***************************************************************************/
/*                                                                         */
/*  afglobal.h                                                             */
/*                                                                         */
/*    Auto-fitter routines to compute global hinting values                */
/*    (specification).                                                     */
/*                                                                         */
/*  Copyright 2003-2005, 2007, 2009, 2011-2012 by                          */
/*  David Turner, Robert Wilhelm, and Werner Lemberg.                      */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/


#ifndef __AFGLOBAL_H__
#define __AFGLOBAL_H__


#include "aftypes.h"
#include "afmodule.h"


FT_BEGIN_HEADER


  /*
   *  Default values and flags for both autofitter globals (found in
   *  AF_ModuleRec) and face globals (in AF_FaceGlobalsRec).
   */

  /* index of fallback script in `af_script_classes' */
#define AF_SCRIPT_FALLBACK  2
  /* a bit mask indicating an uncovered glyph        */
#define AF_SCRIPT_NONE      0x7F
  /* if this flag is set, we have an ASCII digit     */
#define AF_DIGIT            0x80

  /* `increase-x-height' property */
#define AF_PROP_INCREASE_X_HEIGHT_MIN  6
#define AF_PROP_INCREASE_X_HEIGHT_MAX  0


  /************************************************************************/
  /************************************************************************/
  /*****                                                              *****/
  /*****                  F A C E   G L O B A L S                     *****/
  /*****                                                              *****/
  /************************************************************************/
  /************************************************************************/


  /*
   *  Note that glyph_scripts[] is used to map each glyph into
   *  an index into the `af_script_classes' array.
   *
   */
  typedef struct  AF_FaceGlobalsRec_
  {
    FT_Face           face;
    FT_Long           glyph_count;    /* same as face->num_glyphs */
    FT_Byte*          glyph_scripts;

    /* per-face auto-hinter properties */
    FT_UInt           increase_x_height;

    AF_ScriptMetrics  metrics[AF_SCRIPT_MAX];

    AF_Module         module;         /* to access global properties */

  } AF_FaceGlobalsRec;


  /*
   *  model the global hints data for a given face, decomposed into
   *  script-specific items
   */

  FT_LOCAL( FT_Error )
  af_face_globals_new( FT_Face          face,
                       AF_FaceGlobals  *aglobals,
                       AF_Module        module );

  FT_LOCAL( FT_Error )
  af_face_globals_get_metrics( AF_FaceGlobals     globals,
                               FT_UInt            gindex,
                               FT_UInt            options,
                               AF_ScriptMetrics  *ametrics );

  FT_LOCAL( void )
  af_face_globals_free( AF_FaceGlobals  globals );

  FT_LOCAL_DEF( FT_Bool )
  af_face_globals_is_digit( AF_FaceGlobals  globals,
                            FT_UInt         gindex );

  /* */


FT_END_HEADER

#endif /* __AFGLOBAL_H__ */


/* END */
