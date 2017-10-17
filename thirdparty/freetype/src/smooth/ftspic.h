/***************************************************************************/
/*                                                                         */
/*  ftspic.h                                                               */
/*                                                                         */
/*    The FreeType position independent code services for smooth module.   */
/*                                                                         */
/*  Copyright 2009-2017 by                                                 */
/*  Oran Agra and Mickey Gabel.                                            */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/


#ifndef FTSPIC_H_
#define FTSPIC_H_


#include FT_INTERNAL_PIC_H


FT_BEGIN_HEADER

#ifndef FT_CONFIG_OPTION_PIC

#define FT_GRAYS_RASTER_GET  ft_grays_raster

#else /* FT_CONFIG_OPTION_PIC */

  typedef struct  SmoothPIC_
  {
    int              ref_count;
    FT_Raster_Funcs  ft_grays_raster;

  } SmoothPIC;


#define GET_PIC( lib ) \
          ( (SmoothPIC*)( (lib)->pic_container.smooth ) )
#define FT_GRAYS_RASTER_GET  ( GET_PIC( library )->ft_grays_raster )


  /* see ftspic.c for the implementation */
  void
  ft_smooth_renderer_class_pic_free( FT_Library  library );

  void
  ft_smooth_lcd_renderer_class_pic_free( FT_Library  library );

  void
  ft_smooth_lcdv_renderer_class_pic_free( FT_Library  library );

  FT_Error
  ft_smooth_renderer_class_pic_init( FT_Library  library );

  FT_Error
  ft_smooth_lcd_renderer_class_pic_init( FT_Library  library );

  FT_Error
  ft_smooth_lcdv_renderer_class_pic_init( FT_Library  library );

#endif /* FT_CONFIG_OPTION_PIC */

 /* */

FT_END_HEADER

#endif /* FTSPIC_H_ */


/* END */
