/***************************************************************************/
/*                                                                         */
/*  basepic.h                                                              */
/*                                                                         */
/*    The FreeType position independent code services for base.            */
/*                                                                         */
/*  Copyright 2009-2018 by                                                 */
/*  Oran Agra and Mickey Gabel.                                            */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/


#ifndef BASEPIC_H_
#define BASEPIC_H_


#include FT_INTERNAL_PIC_H


#ifndef FT_CONFIG_OPTION_PIC

#define FT_OUTLINE_GLYPH_CLASS_GET  &ft_outline_glyph_class
#define FT_BITMAP_GLYPH_CLASS_GET   &ft_bitmap_glyph_class
#define FT_DEFAULT_MODULES_GET      ft_default_modules

#ifdef FT_CONFIG_OPTION_GUESSING_EMBEDDED_RFORK
#define FT_RACCESS_GUESS_TABLE_GET  ft_raccess_guess_table
#endif

#else /* FT_CONFIG_OPTION_PIC */

#include FT_GLYPH_H

#ifdef FT_CONFIG_OPTION_GUESSING_EMBEDDED_RFORK
#include FT_INTERNAL_RFORK_H
#endif


FT_BEGIN_HEADER

  typedef struct  BasePIC_
  {
    FT_Module_Class**  default_module_classes;
    FT_Glyph_Class     ft_outline_glyph_class;
    FT_Glyph_Class     ft_bitmap_glyph_class;

#ifdef FT_CONFIG_OPTION_GUESSING_EMBEDDED_RFORK
    ft_raccess_guess_rec  ft_raccess_guess_table[FT_RACCESS_N_RULES];
#endif

  } BasePIC;


#define GET_PIC( lib )  ( (BasePIC*)( (lib)->pic_container.base ) )

#define FT_OUTLINE_GLYPH_CLASS_GET                      \
          ( &GET_PIC( library )->ft_outline_glyph_class )
#define FT_BITMAP_GLYPH_CLASS_GET                        \
          ( &GET_PIC( library )->ft_bitmap_glyph_class )
#define FT_DEFAULT_MODULES_GET                           \
          ( GET_PIC( library )->default_module_classes )

#ifdef FT_CONFIG_OPTION_GUESSING_EMBEDDED_RFORK
#define FT_RACCESS_GUESS_TABLE_GET                       \
          ( GET_PIC( library )->ft_raccess_guess_table )
#endif


  /* see basepic.c for the implementation */
  void
  ft_base_pic_free( FT_Library  library );

  FT_Error
  ft_base_pic_init( FT_Library  library );

FT_END_HEADER

#endif /* FT_CONFIG_OPTION_PIC */

  /* */

#endif /* BASEPIC_H_ */


/* END */
