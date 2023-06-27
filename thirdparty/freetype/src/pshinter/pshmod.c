/****************************************************************************
 *
 * pshmod.c
 *
 *   FreeType PostScript hinter module implementation (body).
 *
 * Copyright (C) 2001-2023 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#include <freetype/internal/ftobjs.h>
#include "pshrec.h"
#include "pshalgo.h"
#include "pshmod.h"


  /* the Postscript Hinter module structure */
  typedef struct  PS_Hinter_Module_Rec_
  {
    FT_ModuleRec          root;
    PS_HintsRec           ps_hints;

    PSH_Globals_FuncsRec  globals_funcs;
    T1_Hints_FuncsRec     t1_funcs;
    T2_Hints_FuncsRec     t2_funcs;

  } PS_Hinter_ModuleRec, *PS_Hinter_Module;


  /* finalize module */
  FT_CALLBACK_DEF( void )
  ps_hinter_done( PS_Hinter_Module  module )
  {
    module->t1_funcs.hints = NULL;
    module->t2_funcs.hints = NULL;

    ps_hints_done( &module->ps_hints );
  }


  /* initialize module, create hints recorder and the interface */
  FT_CALLBACK_DEF( FT_Error )
  ps_hinter_init( PS_Hinter_Module  module )
  {
    FT_Memory  memory = module->root.memory;
    void*      ph     = &module->ps_hints;


    ps_hints_init( &module->ps_hints, memory );

    psh_globals_funcs_init( &module->globals_funcs );

    t1_hints_funcs_init( &module->t1_funcs );
    module->t1_funcs.hints = (T1_Hints)ph;

    t2_hints_funcs_init( &module->t2_funcs );
    module->t2_funcs.hints = (T2_Hints)ph;

    return 0;
  }


  /* returns global hints interface */
  FT_CALLBACK_DEF( PSH_Globals_Funcs )
  pshinter_get_globals_funcs( FT_Module  module )
  {
    return &((PS_Hinter_Module)module)->globals_funcs;
  }


  /* return Type 1 hints interface */
  FT_CALLBACK_DEF( T1_Hints_Funcs )
  pshinter_get_t1_funcs( FT_Module  module )
  {
    return &((PS_Hinter_Module)module)->t1_funcs;
  }


  /* return Type 2 hints interface */
  FT_CALLBACK_DEF( T2_Hints_Funcs )
  pshinter_get_t2_funcs( FT_Module  module )
  {
    return &((PS_Hinter_Module)module)->t2_funcs;
  }


  FT_DEFINE_PSHINTER_INTERFACE(
    pshinter_interface,

    pshinter_get_globals_funcs,
    pshinter_get_t1_funcs,
    pshinter_get_t2_funcs
  )


  FT_DEFINE_MODULE(
    pshinter_module_class,

    0,
    sizeof ( PS_Hinter_ModuleRec ),
    "pshinter",
    0x10000L,
    0x20000L,

    &pshinter_interface,        /* module-specific interface */

    (FT_Module_Constructor)ps_hinter_init,  /* module_init   */
    (FT_Module_Destructor) ps_hinter_done,  /* module_done   */
    (FT_Module_Requester)  NULL             /* get_interface */
  )

/* END */
