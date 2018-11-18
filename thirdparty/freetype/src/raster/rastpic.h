/***************************************************************************/
/*                                                                         */
/*  rastpic.h                                                              */
/*                                                                         */
/*    The FreeType position independent code services for raster module.   */
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


#ifndef RASTPIC_H_
#define RASTPIC_H_


#include FT_INTERNAL_PIC_H


FT_BEGIN_HEADER

#ifndef FT_CONFIG_OPTION_PIC

#define FT_STANDARD_RASTER_GET  ft_standard_raster

#else /* FT_CONFIG_OPTION_PIC */

  typedef struct  RasterPIC_
  {
    int              ref_count;
    FT_Raster_Funcs  ft_standard_raster;

  } RasterPIC;


#define GET_PIC( lib )                                    \
          ( (RasterPIC*)( (lib)->pic_container.raster ) )
#define FT_STANDARD_RASTER_GET  ( GET_PIC( library )->ft_standard_raster )


  /* see rastpic.c for the implementation */
  void
  ft_raster1_renderer_class_pic_free( FT_Library  library );

  FT_Error
  ft_raster1_renderer_class_pic_init( FT_Library  library );

#endif /* FT_CONFIG_OPTION_PIC */

 /* */

FT_END_HEADER

#endif /* RASTPIC_H_ */


/* END */
