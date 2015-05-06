/***************************************************************************/
/*                                                                         */
/*  pshpic.h                                                               */
/*                                                                         */
/*    The FreeType position independent code services for pshinter module. */
/*                                                                         */
/*  Copyright 2009, 2012, 2013 by                                          */
/*  Oran Agra and Mickey Gabel.                                            */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/


#ifndef __PSHPIC_H__
#define __PSHPIC_H__


FT_BEGIN_HEADER

#include FT_INTERNAL_PIC_H


#ifndef FT_CONFIG_OPTION_PIC

#define PSHINTER_INTERFACE_GET  pshinter_interface

#else /* FT_CONFIG_OPTION_PIC */

#include FT_INTERNAL_POSTSCRIPT_HINTS_H

  typedef struct  PSHinterPIC_
  {
    PSHinter_Interface  pshinter_interface;

  } PSHinterPIC;


#define GET_PIC( lib )  ( (PSHinterPIC*)( (lib)->pic_container.pshinter ) )

#define PSHINTER_INTERFACE_GET  ( GET_PIC( library )->pshinter_interface )

  /* see pshpic.c for the implementation */
  void
  pshinter_module_class_pic_free( FT_Library  library );

  FT_Error
  pshinter_module_class_pic_init( FT_Library  library );

#endif /* FT_CONFIG_OPTION_PIC */

 /* */

FT_END_HEADER

#endif /* __PSHPIC_H__ */


/* END */
