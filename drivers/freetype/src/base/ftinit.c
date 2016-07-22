/***************************************************************************/
/*                                                                         */
/*  ftinit.c                                                               */
/*                                                                         */
/*    FreeType initialization layer (body).                                */
/*                                                                         */
/*  Copyright 1996-2016 by                                                 */
/*  David Turner, Robert Wilhelm, and Werner Lemberg.                      */
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
  /*  The purpose of this file is to implement the following two           */
  /*  functions:                                                           */
  /*                                                                       */
  /*  FT_Add_Default_Modules():                                            */
  /*     This function is used to add the set of default modules to a      */
  /*     fresh new library object.  The set is taken from the header file  */
  /*     `freetype/config/ftmodule.h'.  See the document `FreeType 2.0     */
  /*     Build System' for more information.                               */
  /*                                                                       */
  /*  FT_Init_FreeType():                                                  */
  /*     This function creates a system object for the current platform,   */
  /*     builds a library out of it, then calls FT_Default_Drivers().      */
  /*                                                                       */
  /*  Note that even if FT_Init_FreeType() uses the implementation of the  */
  /*  system object defined at build time, client applications are still   */
  /*  able to provide their own `ftsystem.c'.                              */
  /*                                                                       */
  /*************************************************************************/


#include <ft2build.h>
#include FT_CONFIG_CONFIG_H
#include FT_INTERNAL_OBJECTS_H
#include FT_INTERNAL_DEBUG_H
#include FT_MODULE_H
#include "basepic.h"


  /*************************************************************************/
  /*                                                                       */
  /* The macro FT_COMPONENT is used in trace mode.  It is an implicit      */
  /* parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log  */
  /* messages during execution.                                            */
  /*                                                                       */
#undef  FT_COMPONENT
#define FT_COMPONENT  trace_init


#ifndef FT_CONFIG_OPTION_PIC


#undef  FT_USE_MODULE
#ifdef __cplusplus
#define FT_USE_MODULE( type, x )  extern "C" const type  x;
#else
#define FT_USE_MODULE( type, x )  extern const type  x;
#endif

#include FT_CONFIG_MODULES_H

#undef  FT_USE_MODULE
#define FT_USE_MODULE( type, x )  (const FT_Module_Class*)&(x),

  static
  const FT_Module_Class*  const ft_default_modules[] =
  {
#include FT_CONFIG_MODULES_H
    0
  };


#else /* FT_CONFIG_OPTION_PIC */


#ifdef __cplusplus
#define FT_EXTERNC  extern "C"
#else
#define FT_EXTERNC  extern
#endif

  /* declare the module's class creation/destruction functions */
#undef  FT_USE_MODULE
#define FT_USE_MODULE( type, x )                            \
  FT_EXTERNC FT_Error                                       \
  FT_Create_Class_ ## x( FT_Library         library,        \
                         FT_Module_Class*  *output_class ); \
  FT_EXTERNC void                                           \
  FT_Destroy_Class_ ## x( FT_Library        library,        \
                          FT_Module_Class*  clazz );

#include FT_CONFIG_MODULES_H

  /* count all module classes */
#undef  FT_USE_MODULE
#define FT_USE_MODULE( type, x )  MODULE_CLASS_ ## x,

  enum
  {
#include FT_CONFIG_MODULES_H
    FT_NUM_MODULE_CLASSES
  };

  /* destroy all module classes */
#undef  FT_USE_MODULE
#define FT_USE_MODULE( type, x )                   \
  if ( classes[i] )                                \
  {                                                \
    FT_Destroy_Class_ ## x( library, classes[i] ); \
  }                                                \
  i++;


  FT_BASE_DEF( void )
  ft_destroy_default_module_classes( FT_Library  library )
  {
    FT_Module_Class*  *classes;
    FT_Memory          memory;
    FT_UInt            i;
    BasePIC*           pic_container = (BasePIC*)library->pic_container.base;


    if ( !pic_container->default_module_classes )
      return;

    memory  = library->memory;
    classes = pic_container->default_module_classes;
    i       = 0;

#include FT_CONFIG_MODULES_H

    FT_FREE( classes );
    pic_container->default_module_classes = NULL;
  }


  /* initialize all module classes and the pointer table */
#undef  FT_USE_MODULE
#define FT_USE_MODULE( type, x )                     \
  error = FT_Create_Class_ ## x( library, &clazz );  \
  if ( error )                                       \
    goto Exit;                                       \
  classes[i++] = clazz;


  FT_BASE_DEF( FT_Error )
  ft_create_default_module_classes( FT_Library  library )
  {
    FT_Error           error;
    FT_Memory          memory;
    FT_Module_Class*  *classes = NULL;
    FT_Module_Class*   clazz;
    FT_UInt            i;
    BasePIC*           pic_container = (BasePIC*)library->pic_container.base;


    memory = library->memory;

    pic_container->default_module_classes = NULL;

    if ( FT_ALLOC( classes, sizeof ( FT_Module_Class* ) *
                              ( FT_NUM_MODULE_CLASSES + 1 ) ) )
      return error;

    /* initialize all pointers to 0, especially the last one */
    for ( i = 0; i < FT_NUM_MODULE_CLASSES; i++ )
      classes[i] = NULL;
    classes[FT_NUM_MODULE_CLASSES] = NULL;

    i = 0;

#include FT_CONFIG_MODULES_H

  Exit:
    if ( error )
      ft_destroy_default_module_classes( library );
    else
      pic_container->default_module_classes = classes;

    return error;
  }


#endif /* FT_CONFIG_OPTION_PIC */


  /* documentation is in ftmodapi.h */

  FT_EXPORT_DEF( void )
  FT_Add_Default_Modules( FT_Library  library )
  {
    FT_Error                       error;
    const FT_Module_Class* const*  cur;


    /* FT_DEFAULT_MODULES_GET dereferences `library' in PIC mode */
#ifdef FT_CONFIG_OPTION_PIC
    if ( !library )
      return;
#endif

    /* GCC 4.6 warns the type difference:
     *   FT_Module_Class** != const FT_Module_Class* const*
     */
    cur = (const FT_Module_Class* const*)FT_DEFAULT_MODULES_GET;

    /* test for valid `library' delayed to FT_Add_Module() */
    while ( *cur )
    {
      error = FT_Add_Module( library, *cur );
      /* notify errors, but don't stop */
      if ( error )
        FT_TRACE0(( "FT_Add_Default_Module:"
                    " Cannot install `%s', error = 0x%x\n",
                    (*cur)->module_name, error ));
      cur++;
    }
  }


  /* documentation is in freetype.h */

  FT_EXPORT_DEF( FT_Error )
  FT_Init_FreeType( FT_Library  *alibrary )
  {
    FT_Error   error;
    FT_Memory  memory;


    /* check of `alibrary' delayed to `FT_New_Library' */

    /* First of all, allocate a new system object -- this function is part */
    /* of the system-specific component, i.e. `ftsystem.c'.                */

    memory = FT_New_Memory();
    if ( !memory )
    {
      FT_ERROR(( "FT_Init_FreeType: cannot find memory manager\n" ));
      return FT_THROW( Unimplemented_Feature );
    }

    /* build a library out of it, then fill it with the set of */
    /* default drivers.                                        */

    error = FT_New_Library( memory, alibrary );
    if ( error )
      FT_Done_Memory( memory );
    else
      FT_Add_Default_Modules( *alibrary );

    return error;
  }


  /* documentation is in freetype.h */

  FT_EXPORT_DEF( FT_Error )
  FT_Done_FreeType( FT_Library  library )
  {
    FT_Memory  memory;


    if ( !library )
      return FT_THROW( Invalid_Library_Handle );

    memory = library->memory;

    /* Discard the library object */
    FT_Done_Library( library );

    /* discard memory manager */
    FT_Done_Memory( memory );

    return FT_Err_Ok;
  }


/* END */
