/***************************************************************************/
/*                                                                         */
/*  ftpic.h                                                                */
/*                                                                         */
/*    The FreeType position independent code services (declaration).       */
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

  /*************************************************************************/
  /*                                                                       */
  /*  Modules that ordinarily have const global data that need address     */
  /*  can instead define pointers here.                                    */
  /*                                                                       */
  /*************************************************************************/


#ifndef FTPIC_H_
#define FTPIC_H_


FT_BEGIN_HEADER

#ifdef FT_CONFIG_OPTION_PIC

  typedef struct  FT_PIC_Container_
  {
    /* pic containers for base */
    void*  base;

    /* pic containers for modules */
    void*  autofit;
    void*  cff;
    void*  pshinter;
    void*  psnames;
    void*  raster;
    void*  sfnt;
    void*  smooth;
    void*  truetype;

  } FT_PIC_Container;


  /* Initialize the various function tables, structs, etc. */
  /* stored in the container.                              */
  FT_BASE( FT_Error )
  ft_pic_container_init( FT_Library  library );


  /* Destroy the contents of the container. */
  FT_BASE( void )
  ft_pic_container_destroy( FT_Library  library );

#endif /* FT_CONFIG_OPTION_PIC */

 /* */

FT_END_HEADER

#endif /* FTPIC_H_ */


/* END */
