/***************************************************************************/
/*                                                                         */
/*  basepic.c                                                              */
/*                                                                         */
/*    The FreeType position independent code services for base.            */
/*                                                                         */
/*  Copyright 2009-2016 by                                                 */
/*  Oran Agra and Mickey Gabel.                                            */
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
#include FT_INTERNAL_OBJECTS_H
#include "basepic.h"


#ifdef FT_CONFIG_OPTION_PIC

  /* forward declaration of PIC init functions from ftglyph.c */
  void
  FT_Init_Class_ft_outline_glyph_class( FT_Glyph_Class*  clazz );

  void
  FT_Init_Class_ft_bitmap_glyph_class( FT_Glyph_Class*  clazz );

#ifdef FT_CONFIG_OPTION_MAC_FONTS
  /* forward declaration of PIC init function from ftrfork.c */
  /* (not modularized)                                       */
  void
  FT_Init_Table_ft_raccess_guess_table( ft_raccess_guess_rec*  record );
#endif

  /* forward declaration of PIC init functions from ftinit.c */
  FT_Error
  ft_create_default_module_classes( FT_Library  library );

  void
  ft_destroy_default_module_classes( FT_Library  library );


  void
  ft_base_pic_free( FT_Library  library )
  {
    FT_PIC_Container*  pic_container = &library->pic_container;
    FT_Memory          memory        = library->memory;


    if ( pic_container->base )
    {
      /* destroy default module classes            */
      /* (in case FT_Add_Default_Modules was used) */
      ft_destroy_default_module_classes( library );

      FT_FREE( pic_container->base );
      pic_container->base = NULL;
    }
  }


  FT_Error
  ft_base_pic_init( FT_Library  library )
  {
    FT_PIC_Container*  pic_container = &library->pic_container;
    FT_Error           error         = FT_Err_Ok;
    BasePIC*           container     = NULL;
    FT_Memory          memory        = library->memory;


    /* allocate pointer, clear and set global container pointer */
    if ( FT_ALLOC( container, sizeof ( *container ) ) )
      return error;
    FT_MEM_SET( container, 0, sizeof ( *container ) );
    pic_container->base = container;

    /* initialize default modules list and pointers */
    error = ft_create_default_module_classes( library );
    if ( error )
      goto Exit;

    /* initialize pointer table -                       */
    /* this is how the module usually expects this data */
    FT_Init_Class_ft_outline_glyph_class(
      &container->ft_outline_glyph_class );
    FT_Init_Class_ft_bitmap_glyph_class(
      &container->ft_bitmap_glyph_class );
#ifdef FT_CONFIG_OPTION_MAC_FONTS
    FT_Init_Table_ft_raccess_guess_table(
      (ft_raccess_guess_rec*)&container->ft_raccess_guess_table );
#endif

  Exit:
    if ( error )
      ft_base_pic_free( library );
    return error;
  }

#endif /* FT_CONFIG_OPTION_PIC */


/* END */
