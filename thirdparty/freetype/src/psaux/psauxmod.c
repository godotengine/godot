/****************************************************************************
 *
 * psauxmod.c
 *
 *   FreeType auxiliary PostScript module implementation (body).
 *
 * Copyright (C) 2000-2022 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#include "psauxmod.h"
#include "psobjs.h"
#include "t1decode.h"
#include "t1cmap.h"
#include "psft.h"
#include "cffdecode.h"

#ifndef T1_CONFIG_OPTION_NO_AFM
#include "afmparse.h"
#endif


  FT_CALLBACK_TABLE_DEF
  const PS_Table_FuncsRec  ps_table_funcs =
  {
    ps_table_new,     /* init    */
    ps_table_done,    /* done    */
    ps_table_add,     /* add     */
    ps_table_release  /* release */
  };


  FT_CALLBACK_TABLE_DEF
  const PS_Parser_FuncsRec  ps_parser_funcs =
  {
    ps_parser_init,             /* init             */
    ps_parser_done,             /* done             */

    ps_parser_skip_spaces,      /* skip_spaces      */
    ps_parser_skip_PS_token,    /* skip_PS_token    */

    ps_parser_to_int,           /* to_int           */
    ps_parser_to_fixed,         /* to_fixed         */
    ps_parser_to_bytes,         /* to_bytes         */
    ps_parser_to_coord_array,   /* to_coord_array   */
    ps_parser_to_fixed_array,   /* to_fixed_array   */
    ps_parser_to_token,         /* to_token         */
    ps_parser_to_token_array,   /* to_token_array   */

    ps_parser_load_field,       /* load_field       */
    ps_parser_load_field_table  /* load_field_table */
  };


  FT_CALLBACK_TABLE_DEF
  const PS_Builder_FuncsRec  ps_builder_funcs =
  {
    ps_builder_init,          /* init */
    ps_builder_done           /* done */
  };


  FT_CALLBACK_TABLE_DEF
  const T1_Builder_FuncsRec  t1_builder_funcs =
  {
    t1_builder_init,          /* init */
    t1_builder_done,          /* done */

    t1_builder_check_points,  /* check_points  */
    t1_builder_add_point,     /* add_point     */
    t1_builder_add_point1,    /* add_point1    */
    t1_builder_add_contour,   /* add_contour   */
    t1_builder_start_point,   /* start_point   */
    t1_builder_close_contour  /* close_contour */
  };


  FT_CALLBACK_TABLE_DEF
  const T1_Decoder_FuncsRec  t1_decoder_funcs =
  {
    t1_decoder_init,               /* init                  */
    t1_decoder_done,               /* done                  */
#ifdef T1_CONFIG_OPTION_OLD_ENGINE
    t1_decoder_parse_charstrings,  /* parse_charstrings_old */
#else
    t1_decoder_parse_metrics,      /* parse_metrics         */
#endif
    cf2_decoder_parse_charstrings  /* parse_charstrings     */
  };


#ifndef T1_CONFIG_OPTION_NO_AFM
  FT_CALLBACK_TABLE_DEF
  const AFM_Parser_FuncsRec  afm_parser_funcs =
  {
    afm_parser_init,  /* init  */
    afm_parser_done,  /* done  */
    afm_parser_parse  /* parse */
  };
#endif


  FT_CALLBACK_TABLE_DEF
  const T1_CMap_ClassesRec  t1_cmap_classes =
  {
    &t1_cmap_standard_class_rec,
    &t1_cmap_expert_class_rec,
    &t1_cmap_custom_class_rec,
    &t1_cmap_unicode_class_rec
  };


  FT_CALLBACK_TABLE_DEF
  const CFF_Builder_FuncsRec  cff_builder_funcs =
  {
    cff_builder_init,          /* init */
    cff_builder_done,          /* done */

    cff_check_points,          /* check_points  */
    cff_builder_add_point,     /* add_point     */
    cff_builder_add_point1,    /* add_point1    */
    cff_builder_add_contour,   /* add_contour   */
    cff_builder_start_point,   /* start_point   */
    cff_builder_close_contour  /* close_contour */
  };


  FT_CALLBACK_TABLE_DEF
  const CFF_Decoder_FuncsRec  cff_decoder_funcs =
  {
    cff_decoder_init,              /* init    */
    cff_decoder_prepare,           /* prepare */

#ifdef CFF_CONFIG_OPTION_OLD_ENGINE
    cff_decoder_parse_charstrings, /* parse_charstrings_old */
#endif
    cf2_decoder_parse_charstrings  /* parse_charstrings     */
  };


  static
  const PSAux_Interface  psaux_interface =
  {
    &ps_table_funcs,
    &ps_parser_funcs,
    &t1_builder_funcs,
    &t1_decoder_funcs,
    t1_decrypt,
    cff_random,
    ps_decoder_init,
    t1_make_subfont,

    (const T1_CMap_ClassesRec*) &t1_cmap_classes,

#ifndef T1_CONFIG_OPTION_NO_AFM
    &afm_parser_funcs,
#else
    0,
#endif

    &cff_decoder_funcs,
  };


  FT_DEFINE_MODULE(
    psaux_module_class,

    0,
    sizeof ( FT_ModuleRec ),
    "psaux",
    0x20000L,
    0x20000L,

    &psaux_interface,  /* module-specific interface */

    (FT_Module_Constructor)NULL,  /* module_init   */
    (FT_Module_Destructor) NULL,  /* module_done   */
    (FT_Module_Requester)  NULL   /* get_interface */
  )


/* END */
