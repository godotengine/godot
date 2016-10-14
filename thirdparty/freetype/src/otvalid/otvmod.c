/***************************************************************************/
/*                                                                         */
/*  otvmod.c                                                               */
/*                                                                         */
/*    FreeType's OpenType validation module implementation (body).         */
/*                                                                         */
/*  Copyright 2004-2016 by                                                 */
/*  David Turner, Robert Wilhelm, and Werner Lemberg.                      */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/


#include <ft2build.h>
#include FT_TRUETYPE_TABLES_H
#include FT_TRUETYPE_TAGS_H
#include FT_OPENTYPE_VALIDATE_H
#include FT_INTERNAL_OBJECTS_H
#include FT_SERVICE_OPENTYPE_VALIDATE_H

#include "otvmod.h"
#include "otvalid.h"
#include "otvcommn.h"


  /*************************************************************************/
  /*                                                                       */
  /* The macro FT_COMPONENT is used in trace mode.  It is an implicit      */
  /* parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log  */
  /* messages during execution.                                            */
  /*                                                                       */
#undef  FT_COMPONENT
#define FT_COMPONENT  trace_otvmodule


  static FT_Error
  otv_load_table( FT_Face             face,
                  FT_Tag              tag,
                  FT_Byte* volatile*  table,
                  FT_ULong*           table_len )
  {
    FT_Error   error;
    FT_Memory  memory = FT_FACE_MEMORY( face );


    error = FT_Load_Sfnt_Table( face, tag, 0, NULL, table_len );
    if ( FT_ERR_EQ( error, Table_Missing ) )
      return FT_Err_Ok;
    if ( error )
      goto Exit;

    if ( FT_ALLOC( *table, *table_len ) )
      goto Exit;

    error = FT_Load_Sfnt_Table( face, tag, 0, *table, table_len );

  Exit:
    return error;
  }


  static FT_Error
  otv_validate( FT_Face volatile   face,
                FT_UInt            ot_flags,
                FT_Bytes          *ot_base,
                FT_Bytes          *ot_gdef,
                FT_Bytes          *ot_gpos,
                FT_Bytes          *ot_gsub,
                FT_Bytes          *ot_jstf )
  {
    FT_Error                  error = FT_Err_Ok;
    FT_Byte* volatile         base;
    FT_Byte* volatile         gdef;
    FT_Byte* volatile         gpos;
    FT_Byte* volatile         gsub;
    FT_Byte* volatile         jstf;
    FT_Byte* volatile         math;
    FT_ULong                  len_base, len_gdef, len_gpos, len_gsub, len_jstf;
    FT_ULong                  len_math;
    FT_UInt                   num_glyphs = (FT_UInt)face->num_glyphs;
    FT_ValidatorRec volatile  valid;


    base     = gdef     = gpos     = gsub     = jstf     = math     = NULL;
    len_base = len_gdef = len_gpos = len_gsub = len_jstf = len_math = 0;

    /*
     * XXX: OpenType tables cannot handle 32-bit glyph index,
     *      although broken TrueType can have 32-bit glyph index.
     */
    if ( face->num_glyphs > 0xFFFFL )
    {
      FT_TRACE1(( "otv_validate: Invalid glyphs index (0x0000FFFF - 0x%08x) ",
                  face->num_glyphs ));
      FT_TRACE1(( "are not handled by OpenType tables\n" ));
      num_glyphs = 0xFFFF;
    }

    /* load tables */

    if ( ot_flags & FT_VALIDATE_BASE )
    {
      error = otv_load_table( face, TTAG_BASE, &base, &len_base );
      if ( error )
        goto Exit;
    }

    if ( ot_flags & FT_VALIDATE_GDEF )
    {
      error = otv_load_table( face, TTAG_GDEF, &gdef, &len_gdef );
      if ( error )
        goto Exit;
    }

    if ( ot_flags & FT_VALIDATE_GPOS )
    {
      error = otv_load_table( face, TTAG_GPOS, &gpos, &len_gpos );
      if ( error )
        goto Exit;
    }

    if ( ot_flags & FT_VALIDATE_GSUB )
    {
      error = otv_load_table( face, TTAG_GSUB, &gsub, &len_gsub );
      if ( error )
        goto Exit;
    }

    if ( ot_flags & FT_VALIDATE_JSTF )
    {
      error = otv_load_table( face, TTAG_JSTF, &jstf, &len_jstf );
      if ( error )
        goto Exit;
    }

    if ( ot_flags & FT_VALIDATE_MATH )
    {
      error = otv_load_table( face, TTAG_MATH, &math, &len_math );
      if ( error )
        goto Exit;
    }

    /* validate tables */

    if ( base )
    {
      ft_validator_init( &valid, base, base + len_base, FT_VALIDATE_DEFAULT );
      if ( ft_setjmp( valid.jump_buffer ) == 0 )
        otv_BASE_validate( base, &valid );
      error = valid.error;
      if ( error )
        goto Exit;
    }

    if ( gpos )
    {
      ft_validator_init( &valid, gpos, gpos + len_gpos, FT_VALIDATE_DEFAULT );
      if ( ft_setjmp( valid.jump_buffer ) == 0 )
        otv_GPOS_validate( gpos, num_glyphs, &valid );
      error = valid.error;
      if ( error )
        goto Exit;
    }

    if ( gsub )
    {
      ft_validator_init( &valid, gsub, gsub + len_gsub, FT_VALIDATE_DEFAULT );
      if ( ft_setjmp( valid.jump_buffer ) == 0 )
        otv_GSUB_validate( gsub, num_glyphs, &valid );
      error = valid.error;
      if ( error )
        goto Exit;
    }

    if ( gdef )
    {
      ft_validator_init( &valid, gdef, gdef + len_gdef, FT_VALIDATE_DEFAULT );
      if ( ft_setjmp( valid.jump_buffer ) == 0 )
        otv_GDEF_validate( gdef, gsub, gpos, num_glyphs, &valid );
      error = valid.error;
      if ( error )
        goto Exit;
    }

    if ( jstf )
    {
      ft_validator_init( &valid, jstf, jstf + len_jstf, FT_VALIDATE_DEFAULT );
      if ( ft_setjmp( valid.jump_buffer ) == 0 )
        otv_JSTF_validate( jstf, gsub, gpos, num_glyphs, &valid );
      error = valid.error;
      if ( error )
        goto Exit;
    }

    if ( math )
    {
      ft_validator_init( &valid, math, math + len_math, FT_VALIDATE_DEFAULT );
      if ( ft_setjmp( valid.jump_buffer ) == 0 )
        otv_MATH_validate( math, num_glyphs, &valid );
      error = valid.error;
      if ( error )
        goto Exit;
    }

    *ot_base = (FT_Bytes)base;
    *ot_gdef = (FT_Bytes)gdef;
    *ot_gpos = (FT_Bytes)gpos;
    *ot_gsub = (FT_Bytes)gsub;
    *ot_jstf = (FT_Bytes)jstf;

  Exit:
    if ( error )
    {
      FT_Memory  memory = FT_FACE_MEMORY( face );


      FT_FREE( base );
      FT_FREE( gdef );
      FT_FREE( gpos );
      FT_FREE( gsub );
      FT_FREE( jstf );
    }

    {
      FT_Memory  memory = FT_FACE_MEMORY( face );


      FT_FREE( math );                 /* Can't return this as API is frozen */
    }

    return error;
  }


  static
  const FT_Service_OTvalidateRec  otvalid_interface =
  {
    otv_validate        /* validate */
  };


  static
  const FT_ServiceDescRec  otvalid_services[] =
  {
    { FT_SERVICE_ID_OPENTYPE_VALIDATE, &otvalid_interface },
    { NULL, NULL }
  };


  static FT_Pointer
  otvalid_get_service( FT_Module    module,
                       const char*  service_id )
  {
    FT_UNUSED( module );

    return ft_service_list_lookup( otvalid_services, service_id );
  }


  FT_CALLBACK_TABLE_DEF
  const FT_Module_Class  otv_module_class =
  {
    0,
    sizeof ( FT_ModuleRec ),
    "otvalid",
    0x10000L,
    0x20000L,

    0,              /* module-specific interface */

    (FT_Module_Constructor)0,
    (FT_Module_Destructor) 0,
    (FT_Module_Requester)  otvalid_get_service
  };


/* END */
