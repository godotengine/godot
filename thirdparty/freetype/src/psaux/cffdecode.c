/****************************************************************************
 *
 * cffdecode.c
 *
 *   PostScript CFF (Type 2) decoding routines (body).
 *
 * Copyright (C) 2017-2023 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#include <freetype/freetype.h>
#include <freetype/internal/ftdebug.h>
#include <freetype/internal/ftserv.h>
#include <freetype/internal/services/svcfftl.h>

#include "cffdecode.h"
#include "psobjs.h"

#include "psauxerr.h"


  /**************************************************************************
   *
   * The macro FT_COMPONENT is used in trace mode.  It is an implicit
   * parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log
   * messages during execution.
   */
#undef  FT_COMPONENT
#define FT_COMPONENT  cffdecode


#ifdef CFF_CONFIG_OPTION_OLD_ENGINE

  typedef enum  CFF_Operator_
  {
    cff_op_unknown = 0,

    cff_op_rmoveto,
    cff_op_hmoveto,
    cff_op_vmoveto,

    cff_op_rlineto,
    cff_op_hlineto,
    cff_op_vlineto,

    cff_op_rrcurveto,
    cff_op_hhcurveto,
    cff_op_hvcurveto,
    cff_op_rcurveline,
    cff_op_rlinecurve,
    cff_op_vhcurveto,
    cff_op_vvcurveto,

    cff_op_flex,
    cff_op_hflex,
    cff_op_hflex1,
    cff_op_flex1,

    cff_op_endchar,

    cff_op_hstem,
    cff_op_vstem,
    cff_op_hstemhm,
    cff_op_vstemhm,

    cff_op_hintmask,
    cff_op_cntrmask,
    cff_op_dotsection,  /* deprecated, acts as no-op */

    cff_op_abs,
    cff_op_add,
    cff_op_sub,
    cff_op_div,
    cff_op_neg,
    cff_op_random,
    cff_op_mul,
    cff_op_sqrt,

    cff_op_blend,

    cff_op_drop,
    cff_op_exch,
    cff_op_index,
    cff_op_roll,
    cff_op_dup,

    cff_op_put,
    cff_op_get,
    cff_op_store,
    cff_op_load,

    cff_op_and,
    cff_op_or,
    cff_op_not,
    cff_op_eq,
    cff_op_ifelse,

    cff_op_callsubr,
    cff_op_callgsubr,
    cff_op_return,

    /* Type 1 opcodes: invalid but seen in real life */
    cff_op_hsbw,
    cff_op_closepath,
    cff_op_callothersubr,
    cff_op_pop,
    cff_op_seac,
    cff_op_sbw,
    cff_op_setcurrentpoint,

    /* do not remove */
    cff_op_max

  } CFF_Operator;


#define CFF_COUNT_CHECK_WIDTH  0x80
#define CFF_COUNT_EXACT        0x40
#define CFF_COUNT_CLEAR_STACK  0x20

  /* count values which have the `CFF_COUNT_CHECK_WIDTH' flag set are  */
  /* used for checking the width and requested numbers of arguments    */
  /* only; they are set to zero afterwards                             */

  /* the other two flags are informative only and unused currently     */

  static const FT_Byte  cff_argument_counts[] =
  {
    0,  /* unknown */

    2 | CFF_COUNT_CHECK_WIDTH | CFF_COUNT_EXACT, /* rmoveto */
    1 | CFF_COUNT_CHECK_WIDTH | CFF_COUNT_EXACT,
    1 | CFF_COUNT_CHECK_WIDTH | CFF_COUNT_EXACT,

    0 | CFF_COUNT_CLEAR_STACK, /* rlineto */
    0 | CFF_COUNT_CLEAR_STACK,
    0 | CFF_COUNT_CLEAR_STACK,

    0 | CFF_COUNT_CLEAR_STACK, /* rrcurveto */
    0 | CFF_COUNT_CLEAR_STACK,
    0 | CFF_COUNT_CLEAR_STACK,
    0 | CFF_COUNT_CLEAR_STACK,
    0 | CFF_COUNT_CLEAR_STACK,
    0 | CFF_COUNT_CLEAR_STACK,
    0 | CFF_COUNT_CLEAR_STACK,

    13, /* flex */
    7,
    9,
    11,

    0 | CFF_COUNT_CHECK_WIDTH, /* endchar */

    2 | CFF_COUNT_CHECK_WIDTH, /* hstem */
    2 | CFF_COUNT_CHECK_WIDTH,
    2 | CFF_COUNT_CHECK_WIDTH,
    2 | CFF_COUNT_CHECK_WIDTH,

    0 | CFF_COUNT_CHECK_WIDTH, /* hintmask */
    0 | CFF_COUNT_CHECK_WIDTH, /* cntrmask */
    0, /* dotsection */

    1, /* abs */
    2,
    2,
    2,
    1,
    0,
    2,
    1,

    1, /* blend */

    1, /* drop */
    2,
    1,
    2,
    1,

    2, /* put */
    1,
    4,
    3,

    2, /* and */
    2,
    1,
    2,
    4,

    1, /* callsubr */
    1,
    0,

    2, /* hsbw */
    0,
    0,
    0,
    5, /* seac */
    4, /* sbw */
    2  /* setcurrentpoint */
  };


  static FT_Error
  cff_operator_seac( CFF_Decoder*  decoder,
                     FT_Pos        asb,
                     FT_Pos        adx,
                     FT_Pos        ady,
                     FT_Int        bchar,
                     FT_Int        achar )
  {
    FT_Error      error;
    CFF_Builder*  builder = &decoder->builder;
    FT_Int        bchar_index, achar_index;
    TT_Face       face    = decoder->builder.face;
    FT_Vector     left_bearing, advance;
    FT_Byte*      charstring;
    FT_ULong      charstring_len;
    FT_Pos        glyph_width;


    if ( decoder->seac )
    {
      FT_ERROR(( "cff_operator_seac: invalid nested seac\n" ));
      return FT_THROW( Syntax_Error );
    }

    adx = ADD_LONG( adx, decoder->builder.left_bearing.x );
    ady = ADD_LONG( ady, decoder->builder.left_bearing.y );

#ifdef FT_CONFIG_OPTION_INCREMENTAL
    /* Incremental fonts don't necessarily have valid charsets.        */
    /* They use the character code, not the glyph index, in this case. */
    if ( face->root.internal->incremental_interface )
    {
      bchar_index = bchar;
      achar_index = achar;
    }
    else
#endif /* FT_CONFIG_OPTION_INCREMENTAL */
    {
      CFF_Font cff = (CFF_Font)( face->extra.data );


      bchar_index = cff_lookup_glyph_by_stdcharcode( cff, bchar );
      achar_index = cff_lookup_glyph_by_stdcharcode( cff, achar );
    }

    if ( bchar_index < 0 || achar_index < 0 )
    {
      FT_ERROR(( "cff_operator_seac:"
                 " invalid seac character code arguments\n" ));
      return FT_THROW( Syntax_Error );
    }

    /* If we are trying to load a composite glyph, do not load the */
    /* accent character and return the array of subglyphs.         */
    if ( builder->no_recurse )
    {
      FT_GlyphSlot    glyph  = (FT_GlyphSlot)builder->glyph;
      FT_GlyphLoader  loader = glyph->internal->loader;
      FT_SubGlyph     subg;


      /* reallocate subglyph array if necessary */
      error = FT_GlyphLoader_CheckSubGlyphs( loader, 2 );
      if ( error )
        goto Exit;

      subg = loader->current.subglyphs;

      /* subglyph 0 = base character */
      subg->index = bchar_index;
      subg->flags = FT_SUBGLYPH_FLAG_ARGS_ARE_XY_VALUES |
                    FT_SUBGLYPH_FLAG_USE_MY_METRICS;
      subg->arg1  = 0;
      subg->arg2  = 0;
      subg++;

      /* subglyph 1 = accent character */
      subg->index = achar_index;
      subg->flags = FT_SUBGLYPH_FLAG_ARGS_ARE_XY_VALUES;
      subg->arg1  = (FT_Int)( adx >> 16 );
      subg->arg2  = (FT_Int)( ady >> 16 );

      /* set up remaining glyph fields */
      glyph->num_subglyphs = 2;
      glyph->subglyphs     = loader->base.subglyphs;
      glyph->format        = FT_GLYPH_FORMAT_COMPOSITE;

      loader->current.num_subglyphs = 2;
    }

    FT_GlyphLoader_Prepare( builder->loader );

    /* First load `bchar' in builder */
    error = decoder->get_glyph_callback( face, (FT_UInt)bchar_index,
                                         &charstring, &charstring_len );
    if ( !error )
    {
      /* the seac operator must not be nested */
      decoder->seac = TRUE;
      error = cff_decoder_parse_charstrings( decoder, charstring,
                                             charstring_len, 0 );
      decoder->seac = FALSE;

      decoder->free_glyph_callback( face, &charstring, charstring_len );

      if ( error )
        goto Exit;
    }

    /* Save the left bearing, advance and glyph width of the base */
    /* character as they will be erased by the next load.         */

    left_bearing = builder->left_bearing;
    advance      = builder->advance;
    glyph_width  = decoder->glyph_width;

    builder->left_bearing.x = 0;
    builder->left_bearing.y = 0;

    builder->pos_x = SUB_LONG( adx, asb );
    builder->pos_y = ady;

    /* Now load `achar' on top of the base outline. */
    error = decoder->get_glyph_callback( face, (FT_UInt)achar_index,
                                         &charstring, &charstring_len );
    if ( !error )
    {
      /* the seac operator must not be nested */
      decoder->seac = TRUE;
      error = cff_decoder_parse_charstrings( decoder, charstring,
                                             charstring_len, 0 );
      decoder->seac = FALSE;

      decoder->free_glyph_callback( face, &charstring, charstring_len );

      if ( error )
        goto Exit;
    }

    /* Restore the left side bearing, advance and glyph width */
    /* of the base character.                                 */
    builder->left_bearing = left_bearing;
    builder->advance      = advance;
    decoder->glyph_width  = glyph_width;

    builder->pos_x = 0;
    builder->pos_y = 0;

  Exit:
    return error;
  }

#endif /* CFF_CONFIG_OPTION_OLD_ENGINE */


  /*************************************************************************/
  /*************************************************************************/
  /*************************************************************************/
  /**********                                                      *********/
  /**********                                                      *********/
  /**********             GENERIC CHARSTRING PARSING               *********/
  /**********                                                      *********/
  /**********                                                      *********/
  /*************************************************************************/
  /*************************************************************************/
  /*************************************************************************/

  /**************************************************************************
   *
   * @Function:
   *   cff_compute_bias
   *
   * @Description:
   *   Computes the bias value in dependence of the number of glyph
   *   subroutines.
   *
   * @Input:
   *   in_charstring_type ::
   *     The `CharstringType' value of the top DICT
   *     dictionary.
   *
   *   num_subrs ::
   *     The number of glyph subroutines.
   *
   * @Return:
   *   The bias value.
   */
  static FT_Int
  cff_compute_bias( FT_Int   in_charstring_type,
                    FT_UInt  num_subrs )
  {
    FT_Int  result;


    if ( in_charstring_type == 1 )
      result = 0;
    else if ( num_subrs < 1240 )
      result = 107;
    else if ( num_subrs < 33900U )
      result = 1131;
    else
      result = 32768U;

    return result;
  }


  FT_LOCAL_DEF( FT_Int )
  cff_lookup_glyph_by_stdcharcode( CFF_Font  cff,
                                   FT_Int    charcode )
  {
    FT_UInt    n;
    FT_UShort  glyph_sid;

    FT_Service_CFFLoad  cffload;


    /* CID-keyed fonts don't have glyph names */
    if ( !cff->charset.sids )
      return -1;

    /* check range of standard char code */
    if ( charcode < 0 || charcode > 255 )
      return -1;

#if 0
    /* retrieve cffload from list of current modules */
    FT_Service_CFFLoad  cffload;


    FT_FACE_FIND_GLOBAL_SERVICE( face, cffload, CFF_LOAD );
    if ( !cffload )
    {
      FT_ERROR(( "cff_lookup_glyph_by_stdcharcode:"
                 " the `cffload' module is not available\n" ));
      return FT_THROW( Unimplemented_Feature );
    }
#endif

    cffload = (FT_Service_CFFLoad)cff->cffload;

    /* Get code to SID mapping from `cff_standard_encoding'. */
    glyph_sid = cffload->get_standard_encoding( (FT_UInt)charcode );

    for ( n = 0; n < cff->num_glyphs; n++ )
    {
      if ( cff->charset.sids[n] == glyph_sid )
        return (FT_Int)n;
    }

    return -1;
  }


#ifdef CFF_CONFIG_OPTION_OLD_ENGINE

  /**************************************************************************
   *
   * @Function:
   *   cff_decoder_parse_charstrings
   *
   * @Description:
   *   Parses a given Type 2 charstrings program.
   *
   * @InOut:
   *   decoder ::
   *     The current Type 1 decoder.
   *
   * @Input:
   *   charstring_base ::
   *     The base of the charstring stream.
   *
   *   charstring_len ::
   *     The length in bytes of the charstring stream.
   *
   *   in_dict ::
   *     Set to 1 if function is called from top or
   *     private DICT (needed for Multiple Master CFFs).
   *
   * @Return:
   *   FreeType error code.  0 means success.
   */
  FT_LOCAL_DEF( FT_Error )
  cff_decoder_parse_charstrings( CFF_Decoder*  decoder,
                                 FT_Byte*      charstring_base,
                                 FT_ULong      charstring_len,
                                 FT_Bool       in_dict )
  {
    FT_Error           error;
    CFF_Decoder_Zone*  zone;
    FT_Byte*           ip;
    FT_Byte*           limit;
    CFF_Builder*       builder = &decoder->builder;
    FT_Pos             x, y;
    FT_Fixed*          stack;
    FT_Int             charstring_type =
                         decoder->cff->top_font.font_dict.charstring_type;
    FT_UShort          num_designs =
                         decoder->cff->top_font.font_dict.num_designs;
    FT_UShort          num_axes =
                         decoder->cff->top_font.font_dict.num_axes;

    T2_Hints_Funcs  hinter;


    /* set default width */
    decoder->num_hints  = 0;
    decoder->read_width = 1;

    /* initialize the decoder */
    decoder->top  = decoder->stack;
    decoder->zone = decoder->zones;
    zone          = decoder->zones;
    stack         = decoder->top;

    hinter = (T2_Hints_Funcs)builder->hints_funcs;

    builder->path_begun = 0;

    if ( !charstring_base )
      return FT_Err_Ok;

    zone->base           = charstring_base;
    limit = zone->limit  = charstring_base + charstring_len;
    ip    = zone->cursor = zone->base;

    error = FT_Err_Ok;

    x = builder->pos_x;
    y = builder->pos_y;

    /* begin hints recording session, if any */
    if ( hinter )
      hinter->open( hinter->hints );

    /* now execute loop */
    while ( ip < limit )
    {
      CFF_Operator  op;
      FT_Byte       v;


      /*********************************************************************
       *
       * Decode operator or operand
       */
      v = *ip++;
      if ( v >= 32 || v == 28 )
      {
        FT_Int    shift = 16;
        FT_Int32  val;


        /* this is an operand, push it on the stack */

        /* if we use shifts, all computations are done with unsigned */
        /* values; the conversion to a signed value is the last step */
        if ( v == 28 )
        {
          if ( ip + 1 >= limit )
            goto Syntax_Error;
          val = (FT_Short)( ( (FT_UShort)ip[0] << 8 ) | ip[1] );
          ip += 2;
        }
        else if ( v < 247 )
          val = (FT_Int32)v - 139;
        else if ( v < 251 )
        {
          if ( ip >= limit )
            goto Syntax_Error;
          val = ( (FT_Int32)v - 247 ) * 256 + *ip++ + 108;
        }
        else if ( v < 255 )
        {
          if ( ip >= limit )
            goto Syntax_Error;
          val = -( (FT_Int32)v - 251 ) * 256 - *ip++ - 108;
        }
        else
        {
          if ( ip + 3 >= limit )
            goto Syntax_Error;
          val = (FT_Int32)( ( (FT_UInt32)ip[0] << 24 ) |
                            ( (FT_UInt32)ip[1] << 16 ) |
                            ( (FT_UInt32)ip[2] <<  8 ) |
                              (FT_UInt32)ip[3]         );
          ip += 4;
          if ( charstring_type == 2 )
            shift = 0;
        }
        if ( decoder->top - stack >= CFF_MAX_OPERANDS )
          goto Stack_Overflow;

        val             = (FT_Int32)( (FT_UInt32)val << shift );
        *decoder->top++ = val;

#ifdef FT_DEBUG_LEVEL_TRACE
        if ( !( val & 0xFFFFL ) )
          FT_TRACE4(( " %hd", (FT_Short)( (FT_UInt32)val >> 16 ) ));
        else
          FT_TRACE4(( " %.5f", val / 65536.0 ));
#endif

      }
      else
      {
        /* The specification says that normally arguments are to be taken */
        /* from the bottom of the stack.  However, this seems not to be   */
        /* correct, at least for Acroread 7.0.8 on GNU/Linux: It pops the */
        /* arguments similar to a PS interpreter.                         */

        FT_Fixed*  args     = decoder->top;
        FT_Int     num_args = (FT_Int)( args - decoder->stack );
        FT_Int     req_args;


        /* find operator */
        op = cff_op_unknown;

        switch ( v )
        {
        case 1:
          op = cff_op_hstem;
          break;
        case 3:
          op = cff_op_vstem;
          break;
        case 4:
          op = cff_op_vmoveto;
          break;
        case 5:
          op = cff_op_rlineto;
          break;
        case 6:
          op = cff_op_hlineto;
          break;
        case 7:
          op = cff_op_vlineto;
          break;
        case 8:
          op = cff_op_rrcurveto;
          break;
        case 9:
          op = cff_op_closepath;
          break;
        case 10:
          op = cff_op_callsubr;
          break;
        case 11:
          op = cff_op_return;
          break;
        case 12:
          if ( ip >= limit )
            goto Syntax_Error;
          v = *ip++;

          switch ( v )
          {
          case 0:
            op = cff_op_dotsection;
            break;
          case 1: /* this is actually the Type1 vstem3 operator */
            op = cff_op_vstem;
            break;
          case 2: /* this is actually the Type1 hstem3 operator */
            op = cff_op_hstem;
            break;
          case 3:
            op = cff_op_and;
            break;
          case 4:
            op = cff_op_or;
            break;
          case 5:
            op = cff_op_not;
            break;
          case 6:
            op = cff_op_seac;
            break;
          case 7:
            op = cff_op_sbw;
            break;
          case 8:
            op = cff_op_store;
            break;
          case 9:
            op = cff_op_abs;
            break;
          case 10:
            op = cff_op_add;
            break;
          case 11:
            op = cff_op_sub;
            break;
          case 12:
            op = cff_op_div;
            break;
          case 13:
            op = cff_op_load;
            break;
          case 14:
            op = cff_op_neg;
            break;
          case 15:
            op = cff_op_eq;
            break;
          case 16:
            op = cff_op_callothersubr;
            break;
          case 17:
            op = cff_op_pop;
            break;
          case 18:
            op = cff_op_drop;
            break;
          case 20:
            op = cff_op_put;
            break;
          case 21:
            op = cff_op_get;
            break;
          case 22:
            op = cff_op_ifelse;
            break;
          case 23:
            op = cff_op_random;
            break;
          case 24:
            op = cff_op_mul;
            break;
          case 26:
            op = cff_op_sqrt;
            break;
          case 27:
            op = cff_op_dup;
            break;
          case 28:
            op = cff_op_exch;
            break;
          case 29:
            op = cff_op_index;
            break;
          case 30:
            op = cff_op_roll;
            break;
          case 33:
            op = cff_op_setcurrentpoint;
            break;
          case 34:
            op = cff_op_hflex;
            break;
          case 35:
            op = cff_op_flex;
            break;
          case 36:
            op = cff_op_hflex1;
            break;
          case 37:
            op = cff_op_flex1;
            break;
          default:
            FT_TRACE4(( " unknown op (12, %d)\n", v ));
            break;
          }
          break;
        case 13:
          op = cff_op_hsbw;
          break;
        case 14:
          op = cff_op_endchar;
          break;
        case 16:
          op = cff_op_blend;
          break;
        case 18:
          op = cff_op_hstemhm;
          break;
        case 19:
          op = cff_op_hintmask;
          break;
        case 20:
          op = cff_op_cntrmask;
          break;
        case 21:
          op = cff_op_rmoveto;
          break;
        case 22:
          op = cff_op_hmoveto;
          break;
        case 23:
          op = cff_op_vstemhm;
          break;
        case 24:
          op = cff_op_rcurveline;
          break;
        case 25:
          op = cff_op_rlinecurve;
          break;
        case 26:
          op = cff_op_vvcurveto;
          break;
        case 27:
          op = cff_op_hhcurveto;
          break;
        case 29:
          op = cff_op_callgsubr;
          break;
        case 30:
          op = cff_op_vhcurveto;
          break;
        case 31:
          op = cff_op_hvcurveto;
          break;
        default:
          FT_TRACE4(( " unknown op (%d)\n", v ));
          break;
        }

        if ( op == cff_op_unknown )
          continue;

        /* in Multiple Master CFFs, T2 charstrings can appear in */
        /* dictionaries, but some operators are prohibited       */
        if ( in_dict )
        {
          switch ( op )
          {
          case cff_op_hstem:
          case cff_op_vstem:
          case cff_op_vmoveto:
          case cff_op_rlineto:
          case cff_op_hlineto:
          case cff_op_vlineto:
          case cff_op_rrcurveto:
          case cff_op_hstemhm:
          case cff_op_hintmask:
          case cff_op_cntrmask:
          case cff_op_rmoveto:
          case cff_op_hmoveto:
          case cff_op_vstemhm:
          case cff_op_rcurveline:
          case cff_op_rlinecurve:
          case cff_op_vvcurveto:
          case cff_op_hhcurveto:
          case cff_op_vhcurveto:
          case cff_op_hvcurveto:
          case cff_op_hflex:
          case cff_op_flex:
          case cff_op_hflex1:
          case cff_op_flex1:
          case cff_op_callsubr:
          case cff_op_callgsubr:
            /* deprecated opcodes */
          case cff_op_dotsection:
            /* invalid Type 1 opcodes */
          case cff_op_hsbw:
          case cff_op_closepath:
          case cff_op_callothersubr:
          case cff_op_seac:
          case cff_op_sbw:
          case cff_op_setcurrentpoint:
            goto MM_Error;

          default:
            break;
          }
        }

        /* check arguments */
        req_args = cff_argument_counts[op];
        if ( req_args & CFF_COUNT_CHECK_WIDTH )
        {
          if ( num_args > 0 && decoder->read_width )
          {
            /* If `nominal_width' is non-zero, the number is really a      */
            /* difference against `nominal_width'.  Else, the number here  */
            /* is truly a width, not a difference against `nominal_width'. */
            /* If the font does not set `nominal_width', then              */
            /* `nominal_width' defaults to zero, and so we can set         */
            /* `glyph_width' to `nominal_width' plus number on the stack   */
            /* -- for either case.                                         */

            FT_Int  set_width_ok;


            switch ( op )
            {
            case cff_op_hmoveto:
            case cff_op_vmoveto:
              set_width_ok = num_args & 2;
              break;

            case cff_op_hstem:
            case cff_op_vstem:
            case cff_op_hstemhm:
            case cff_op_vstemhm:
            case cff_op_rmoveto:
            case cff_op_hintmask:
            case cff_op_cntrmask:
              set_width_ok = num_args & 1;
              break;

            case cff_op_endchar:
              /* If there is a width specified for endchar, we either have */
              /* 1 argument or 5 arguments.  We like to argue.             */
              set_width_ok = in_dict
                               ? 0
                               : ( ( num_args == 5 ) || ( num_args == 1 ) );
              break;

            default:
              set_width_ok = 0;
              break;
            }

            if ( set_width_ok )
            {
              decoder->glyph_width = decoder->nominal_width +
                                       ( stack[0] >> 16 );

              if ( decoder->width_only )
              {
                /* we only want the advance width; stop here */
                break;
              }

              /* Consumed an argument. */
              num_args--;
            }
          }

          decoder->read_width = 0;
          req_args            = 0;
        }

        req_args &= 0x000F;
        if ( num_args < req_args )
          goto Stack_Underflow;
        args     -= req_args;
        num_args -= req_args;

        /* At this point, `args' points to the first argument of the  */
        /* operand in case `req_args' isn't zero.  Otherwise, we have */
        /* to adjust `args' manually.                                 */

        /* Note that we only pop arguments from the stack which we    */
        /* really need and can digest so that we can continue in case */
        /* of superfluous stack elements.                             */

        switch ( op )
        {
        case cff_op_hstem:
        case cff_op_vstem:
        case cff_op_hstemhm:
        case cff_op_vstemhm:
          /* the number of arguments is always even here */
          FT_TRACE4(( "%s\n",
              op == cff_op_hstem   ? " hstem"   :
            ( op == cff_op_vstem   ? " vstem"   :
            ( op == cff_op_hstemhm ? " hstemhm" : " vstemhm" ) ) ));

          if ( hinter )
            hinter->stems( hinter->hints,
                           ( op == cff_op_hstem || op == cff_op_hstemhm ),
                           num_args / 2,
                           args - ( num_args & ~1 ) );

          decoder->num_hints += num_args / 2;
          args = stack;
          break;

        case cff_op_hintmask:
        case cff_op_cntrmask:
          FT_TRACE4(( "%s", op == cff_op_hintmask ? " hintmask"
                                                  : " cntrmask" ));

          /* implement vstem when needed --                        */
          /* the specification doesn't say it, but this also works */
          /* with the 'cntrmask' operator                          */
          /*                                                       */
          if ( num_args > 0 )
          {
            if ( hinter )
              hinter->stems( hinter->hints,
                             0,
                             num_args / 2,
                             args - ( num_args & ~1 ) );

            decoder->num_hints += num_args / 2;
          }

          /* In a valid charstring there must be at least one byte */
          /* after `hintmask' or `cntrmask' (e.g., for a `return'  */
          /* instruction).  Additionally, there must be space for  */
          /* `num_hints' bits.                                     */

          if ( ( ip + ( ( decoder->num_hints + 7 ) >> 3 ) ) >= limit )
            goto Syntax_Error;

          if ( hinter )
          {
            if ( op == cff_op_hintmask )
              hinter->hintmask( hinter->hints,
                                (FT_UInt)builder->current->n_points,
                                (FT_UInt)decoder->num_hints,
                                ip );
            else
              hinter->counter( hinter->hints,
                               (FT_UInt)decoder->num_hints,
                               ip );
          }

#ifdef FT_DEBUG_LEVEL_TRACE
          {
            FT_UInt  maskbyte;


            FT_TRACE4(( " (maskbytes:" ));

            for ( maskbyte = 0;
                  maskbyte < (FT_UInt)( ( decoder->num_hints + 7 ) >> 3 );
                  maskbyte++, ip++ )
              FT_TRACE4(( " 0x%02X", *ip ));

            FT_TRACE4(( ")\n" ));
          }
#else
          ip += ( decoder->num_hints + 7 ) >> 3;
#endif
          args = stack;
          break;

        case cff_op_rmoveto:
          FT_TRACE4(( " rmoveto\n" ));

          cff_builder_close_contour( builder );
          builder->path_begun = 0;
          x    = ADD_LONG( x, args[-2] );
          y    = ADD_LONG( y, args[-1] );
          args = stack;
          break;

        case cff_op_vmoveto:
          FT_TRACE4(( " vmoveto\n" ));

          cff_builder_close_contour( builder );
          builder->path_begun = 0;
          y    = ADD_LONG( y, args[-1] );
          args = stack;
          break;

        case cff_op_hmoveto:
          FT_TRACE4(( " hmoveto\n" ));

          cff_builder_close_contour( builder );
          builder->path_begun = 0;
          x    = ADD_LONG( x, args[-1] );
          args = stack;
          break;

        case cff_op_rlineto:
          FT_TRACE4(( " rlineto\n" ));

          if ( cff_builder_start_point( builder, x, y )  ||
               cff_check_points( builder, num_args / 2 ) )
            goto Fail;

          if ( num_args < 2 )
            goto Stack_Underflow;

          args -= num_args & ~1;
          while ( args < decoder->top )
          {
            x = ADD_LONG( x, args[0] );
            y = ADD_LONG( y, args[1] );
            cff_builder_add_point( builder, x, y, 1 );
            args += 2;
          }
          args = stack;
          break;

        case cff_op_hlineto:
        case cff_op_vlineto:
          {
            FT_Int  phase = ( op == cff_op_hlineto );


            FT_TRACE4(( "%s\n", op == cff_op_hlineto ? " hlineto"
                                                     : " vlineto" ));

            if ( num_args < 0 )
              goto Stack_Underflow;

            /* there exist subsetted fonts (found in PDFs) */
            /* which call `hlineto' without arguments      */
            if ( num_args == 0 )
              break;

            if ( cff_builder_start_point( builder, x, y ) ||
                 cff_check_points( builder, num_args )    )
              goto Fail;

            args = stack;
            while ( args < decoder->top )
            {
              if ( phase )
                x = ADD_LONG( x, args[0] );
              else
                y = ADD_LONG( y, args[0] );

              if ( cff_builder_add_point1( builder, x, y ) )
                goto Fail;

              args++;
              phase ^= 1;
            }
            args = stack;
          }
          break;

        case cff_op_rrcurveto:
          {
            FT_Int  nargs;


            FT_TRACE4(( " rrcurveto\n" ));

            if ( num_args < 6 )
              goto Stack_Underflow;

            nargs = num_args - num_args % 6;

            if ( cff_builder_start_point( builder, x, y ) ||
                 cff_check_points( builder, nargs / 2 )   )
              goto Fail;

            args -= nargs;
            while ( args < decoder->top )
            {
              x = ADD_LONG( x, args[0] );
              y = ADD_LONG( y, args[1] );
              cff_builder_add_point( builder, x, y, 0 );

              x = ADD_LONG( x, args[2] );
              y = ADD_LONG( y, args[3] );
              cff_builder_add_point( builder, x, y, 0 );

              x = ADD_LONG( x, args[4] );
              y = ADD_LONG( y, args[5] );
              cff_builder_add_point( builder, x, y, 1 );

              args += 6;
            }
            args = stack;
          }
          break;

        case cff_op_vvcurveto:
          {
            FT_Int  nargs;


            FT_TRACE4(( " vvcurveto\n" ));

            if ( num_args < 4 )
              goto Stack_Underflow;

            /* if num_args isn't of the form 4n or 4n+1, */
            /* we enforce it by clearing the second bit  */

            nargs = num_args & ~2;

            if ( cff_builder_start_point( builder, x, y ) )
              goto Fail;

            args -= nargs;

            if ( nargs & 1 )
            {
              x = ADD_LONG( x, args[0] );
              args++;
              nargs--;
            }

            if ( cff_check_points( builder, 3 * ( nargs / 4 ) ) )
              goto Fail;

            while ( args < decoder->top )
            {
              y = ADD_LONG( y, args[0] );
              cff_builder_add_point( builder, x, y, 0 );

              x = ADD_LONG( x, args[1] );
              y = ADD_LONG( y, args[2] );
              cff_builder_add_point( builder, x, y, 0 );

              y = ADD_LONG( y, args[3] );
              cff_builder_add_point( builder, x, y, 1 );

              args += 4;
            }
            args = stack;
          }
          break;

        case cff_op_hhcurveto:
          {
            FT_Int  nargs;


            FT_TRACE4(( " hhcurveto\n" ));

            if ( num_args < 4 )
              goto Stack_Underflow;

            /* if num_args isn't of the form 4n or 4n+1, */
            /* we enforce it by clearing the second bit  */

            nargs = num_args & ~2;

            if ( cff_builder_start_point( builder, x, y ) )
              goto Fail;

            args -= nargs;
            if ( nargs & 1 )
            {
              y = ADD_LONG( y, args[0] );
              args++;
              nargs--;
            }

            if ( cff_check_points( builder, 3 * ( nargs / 4 ) ) )
              goto Fail;

            while ( args < decoder->top )
            {
              x = ADD_LONG( x, args[0] );
              cff_builder_add_point( builder, x, y, 0 );

              x = ADD_LONG( x, args[1] );
              y = ADD_LONG( y, args[2] );
              cff_builder_add_point( builder, x, y, 0 );

              x = ADD_LONG( x, args[3] );
              cff_builder_add_point( builder, x, y, 1 );

              args += 4;
            }
            args = stack;
          }
          break;

        case cff_op_vhcurveto:
        case cff_op_hvcurveto:
          {
            FT_Int  phase;
            FT_Int  nargs;


            FT_TRACE4(( "%s\n", op == cff_op_vhcurveto ? " vhcurveto"
                                                       : " hvcurveto" ));

            if ( cff_builder_start_point( builder, x, y ) )
              goto Fail;

            if ( num_args < 4 )
              goto Stack_Underflow;

            /* if num_args isn't of the form 8n, 8n+1, 8n+4, or 8n+5, */
            /* we enforce it by clearing the second bit               */

            nargs = num_args & ~2;

            args -= nargs;
            if ( cff_check_points( builder, ( nargs / 4 ) * 3 ) )
              goto Stack_Underflow;

            phase = ( op == cff_op_hvcurveto );

            while ( nargs >= 4 )
            {
              nargs -= 4;
              if ( phase )
              {
                x = ADD_LONG( x, args[0] );
                cff_builder_add_point( builder, x, y, 0 );

                x = ADD_LONG( x, args[1] );
                y = ADD_LONG( y, args[2] );
                cff_builder_add_point( builder, x, y, 0 );

                y = ADD_LONG( y, args[3] );
                if ( nargs == 1 )
                  x = ADD_LONG( x, args[4] );
                cff_builder_add_point( builder, x, y, 1 );
              }
              else
              {
                y = ADD_LONG( y, args[0] );
                cff_builder_add_point( builder, x, y, 0 );

                x = ADD_LONG( x, args[1] );
                y = ADD_LONG( y, args[2] );
                cff_builder_add_point( builder, x, y, 0 );

                x = ADD_LONG( x, args[3] );
                if ( nargs == 1 )
                  y = ADD_LONG( y, args[4] );
                cff_builder_add_point( builder, x, y, 1 );
              }
              args  += 4;
              phase ^= 1;
            }
            args = stack;
          }
          break;

        case cff_op_rlinecurve:
          {
            FT_Int  num_lines;
            FT_Int  nargs;


            FT_TRACE4(( " rlinecurve\n" ));

            if ( num_args < 8 )
              goto Stack_Underflow;

            nargs     = num_args & ~1;
            num_lines = ( nargs - 6 ) / 2;

            if ( cff_builder_start_point( builder, x, y )   ||
                 cff_check_points( builder, num_lines + 3 ) )
              goto Fail;

            args -= nargs;

            /* first, add the line segments */
            while ( num_lines > 0 )
            {
              x = ADD_LONG( x, args[0] );
              y = ADD_LONG( y, args[1] );
              cff_builder_add_point( builder, x, y, 1 );

              args += 2;
              num_lines--;
            }

            /* then the curve */
            x = ADD_LONG( x, args[0] );
            y = ADD_LONG( y, args[1] );
            cff_builder_add_point( builder, x, y, 0 );

            x = ADD_LONG( x, args[2] );
            y = ADD_LONG( y, args[3] );
            cff_builder_add_point( builder, x, y, 0 );

            x = ADD_LONG( x, args[4] );
            y = ADD_LONG( y, args[5] );
            cff_builder_add_point( builder, x, y, 1 );

            args = stack;
          }
          break;

        case cff_op_rcurveline:
          {
            FT_Int  num_curves;
            FT_Int  nargs;


            FT_TRACE4(( " rcurveline\n" ));

            if ( num_args < 8 )
              goto Stack_Underflow;

            nargs      = num_args - 2;
            nargs      = nargs - nargs % 6 + 2;
            num_curves = ( nargs - 2 ) / 6;

            if ( cff_builder_start_point( builder, x, y )        ||
                 cff_check_points( builder, num_curves * 3 + 2 ) )
              goto Fail;

            args -= nargs;

            /* first, add the curves */
            while ( num_curves > 0 )
            {
              x = ADD_LONG( x, args[0] );
              y = ADD_LONG( y, args[1] );
              cff_builder_add_point( builder, x, y, 0 );

              x = ADD_LONG( x, args[2] );
              y = ADD_LONG( y, args[3] );
              cff_builder_add_point( builder, x, y, 0 );

              x = ADD_LONG( x, args[4] );
              y = ADD_LONG( y, args[5] );
              cff_builder_add_point( builder, x, y, 1 );

              args += 6;
              num_curves--;
            }

            /* then the final line */
            x = ADD_LONG( x, args[0] );
            y = ADD_LONG( y, args[1] );
            cff_builder_add_point( builder, x, y, 1 );

            args = stack;
          }
          break;

        case cff_op_hflex1:
          {
            FT_Pos  start_y;


            FT_TRACE4(( " hflex1\n" ));

            /* adding five more points: 4 control points, 1 on-curve point */
            /* -- make sure we have enough space for the start point if it */
            /* needs to be added                                           */
            if ( cff_builder_start_point( builder, x, y ) ||
                 cff_check_points( builder, 6 )           )
              goto Fail;

            /* record the starting point's y position for later use */
            start_y = y;

            /* first control point */
            x = ADD_LONG( x, args[0] );
            y = ADD_LONG( y, args[1] );
            cff_builder_add_point( builder, x, y, 0 );

            /* second control point */
            x = ADD_LONG( x, args[2] );
            y = ADD_LONG( y, args[3] );
            cff_builder_add_point( builder, x, y, 0 );

            /* join point; on curve, with y-value the same as the last */
            /* control point's y-value                                 */
            x = ADD_LONG( x, args[4] );
            cff_builder_add_point( builder, x, y, 1 );

            /* third control point, with y-value the same as the join */
            /* point's y-value                                        */
            x = ADD_LONG( x, args[5] );
            cff_builder_add_point( builder, x, y, 0 );

            /* fourth control point */
            x = ADD_LONG( x, args[6] );
            y = ADD_LONG( y, args[7] );
            cff_builder_add_point( builder, x, y, 0 );

            /* ending point, with y-value the same as the start   */
            x = ADD_LONG( x, args[8] );
            y = start_y;
            cff_builder_add_point( builder, x, y, 1 );

            args = stack;
            break;
          }

        case cff_op_hflex:
          {
            FT_Pos  start_y;


            FT_TRACE4(( " hflex\n" ));

            /* adding six more points; 4 control points, 2 on-curve points */
            if ( cff_builder_start_point( builder, x, y ) ||
                 cff_check_points( builder, 6 )           )
              goto Fail;

            /* record the starting point's y-position for later use */
            start_y = y;

            /* first control point */
            x = ADD_LONG( x, args[0] );
            cff_builder_add_point( builder, x, y, 0 );

            /* second control point */
            x = ADD_LONG( x, args[1] );
            y = ADD_LONG( y, args[2] );
            cff_builder_add_point( builder, x, y, 0 );

            /* join point; on curve, with y-value the same as the last */
            /* control point's y-value                                 */
            x = ADD_LONG( x, args[3] );
            cff_builder_add_point( builder, x, y, 1 );

            /* third control point, with y-value the same as the join */
            /* point's y-value                                        */
            x = ADD_LONG( x, args[4] );
            cff_builder_add_point( builder, x, y, 0 );

            /* fourth control point */
            x = ADD_LONG( x, args[5] );
            y = start_y;
            cff_builder_add_point( builder, x, y, 0 );

            /* ending point, with y-value the same as the start point's */
            /* y-value -- we don't add this point, though               */
            x = ADD_LONG( x, args[6] );
            cff_builder_add_point( builder, x, y, 1 );

            args = stack;
            break;
          }

        case cff_op_flex1:
          {
            FT_Pos     start_x, start_y; /* record start x, y values for */
                                         /* alter use                    */
            FT_Fixed   dx = 0, dy = 0;   /* used in horizontal/vertical  */
                                         /* algorithm below              */
            FT_Int     horizontal, count;
            FT_Fixed*  temp;


            FT_TRACE4(( " flex1\n" ));

            /* adding six more points; 4 control points, 2 on-curve points */
            if ( cff_builder_start_point( builder, x, y ) ||
                 cff_check_points( builder, 6 )           )
              goto Fail;

            /* record the starting point's x, y position for later use */
            start_x = x;
            start_y = y;

            /* XXX: figure out whether this is supposed to be a horizontal */
            /*      or vertical flex; the Type 2 specification is vague... */

            temp = args;

            /* grab up to the last argument */
            for ( count = 5; count > 0; count-- )
            {
              dx    = ADD_LONG( dx, temp[0] );
              dy    = ADD_LONG( dy, temp[1] );
              temp += 2;
            }

            if ( dx < 0 )
              dx = NEG_LONG( dx );
            if ( dy < 0 )
              dy = NEG_LONG( dy );

            /* strange test, but here it is... */
            horizontal = ( dx > dy );

            for ( count = 5; count > 0; count-- )
            {
              x = ADD_LONG( x, args[0] );
              y = ADD_LONG( y, args[1] );
              cff_builder_add_point( builder, x, y,
                                     FT_BOOL( count == 3 ) );
              args += 2;
            }

            /* is last operand an x- or y-delta? */
            if ( horizontal )
            {
              x = ADD_LONG( x, args[0] );
              y = start_y;
            }
            else
            {
              x = start_x;
              y = ADD_LONG( y, args[0] );
            }

            cff_builder_add_point( builder, x, y, 1 );

            args = stack;
            break;
           }

        case cff_op_flex:
          {
            FT_UInt  count;


            FT_TRACE4(( " flex\n" ));

            if ( cff_builder_start_point( builder, x, y ) ||
                 cff_check_points( builder, 6 )           )
              goto Fail;

            for ( count = 6; count > 0; count-- )
            {
              x = ADD_LONG( x, args[0] );
              y = ADD_LONG( y, args[1] );
              cff_builder_add_point( builder, x, y,
                                     FT_BOOL( count == 4 || count == 1 ) );
              args += 2;
            }

            args = stack;
          }
          break;

        case cff_op_seac:
          FT_TRACE4(( " seac\n" ));

          error = cff_operator_seac( decoder,
                                     args[0], args[1], args[2],
                                     (FT_Int)( args[3] >> 16 ),
                                     (FT_Int)( args[4] >> 16 ) );

          /* add current outline to the glyph slot */
          FT_GlyphLoader_Add( builder->loader );

          /* return now! */
          FT_TRACE4(( "\n" ));
          return error;

        case cff_op_endchar:
          /* in dictionaries, `endchar' simply indicates end of data */
          if ( in_dict )
            return error;

          FT_TRACE4(( " endchar\n" ));

          /* We are going to emulate the seac operator. */
          if ( num_args >= 4 )
          {
            /* Save glyph width so that the subglyphs don't overwrite it. */
            FT_Pos  glyph_width = decoder->glyph_width;


            error = cff_operator_seac( decoder,
                                       0L, args[-4], args[-3],
                                       (FT_Int)( args[-2] >> 16 ),
                                       (FT_Int)( args[-1] >> 16 ) );

            decoder->glyph_width = glyph_width;
          }
          else
          {
            cff_builder_close_contour( builder );

            /* close hints recording session */
            if ( hinter )
            {
              if ( hinter->close( hinter->hints,
                                  (FT_UInt)builder->current->n_points ) )
                goto Syntax_Error;

              /* apply hints to the loaded glyph outline now */
              error = hinter->apply( hinter->hints,
                                     builder->current,
                                     (PSH_Globals)builder->hints_globals,
                                     decoder->hint_mode );
              if ( error )
                goto Fail;
            }

            /* add current outline to the glyph slot */
            FT_GlyphLoader_Add( builder->loader );
          }

          /* return now! */
          FT_TRACE4(( "\n" ));
          return error;

        case cff_op_abs:
          FT_TRACE4(( " abs\n" ));

          if ( args[0] < 0 )
          {
            if ( args[0] == FT_LONG_MIN )
              args[0] = FT_LONG_MAX;
            else
              args[0] = -args[0];
          }
          args++;
          break;

        case cff_op_add:
          FT_TRACE4(( " add\n" ));

          args[0] = ADD_LONG( args[0], args[1] );
          args++;
          break;

        case cff_op_sub:
          FT_TRACE4(( " sub\n" ));

          args[0] = SUB_LONG( args[0], args[1] );
          args++;
          break;

        case cff_op_div:
          FT_TRACE4(( " div\n" ));

          args[0] = FT_DivFix( args[0], args[1] );
          args++;
          break;

        case cff_op_neg:
          FT_TRACE4(( " neg\n" ));

          if ( args[0] == FT_LONG_MIN )
            args[0] = FT_LONG_MAX;
          args[0] = -args[0];
          args++;
          break;

        case cff_op_random:
          {
            FT_UInt32*  randval = in_dict ? &decoder->cff->top_font.random
                                          : &decoder->current_subfont->random;


            FT_TRACE4(( " random\n" ));

            /* only use the lower 16 bits of `random'  */
            /* to generate a number in the range (0;1] */
            args[0] = (FT_Fixed)( ( *randval & 0xFFFF ) + 1 );
            args++;

            *randval = cff_random( *randval );
          }
          break;

        case cff_op_mul:
          FT_TRACE4(( " mul\n" ));

          args[0] = FT_MulFix( args[0], args[1] );
          args++;
          break;

        case cff_op_sqrt:
          FT_TRACE4(( " sqrt\n" ));

          /* without upper limit the loop below might not finish */
          if ( args[0] > 0x7FFFFFFFL )
            args[0] = 46341;
          else if ( args[0] > 0 )
          {
            FT_Fixed  root = args[0];
            FT_Fixed  new_root;


            for (;;)
            {
              new_root = ( root + FT_DivFix( args[0], root ) + 1 ) >> 1;
              if ( new_root == root )
                break;
              root = new_root;
            }
            args[0] = new_root;
          }
          else
            args[0] = 0;
          args++;
          break;

        case cff_op_drop:
          /* nothing */
          FT_TRACE4(( " drop\n" ));

          break;

        case cff_op_exch:
          {
            FT_Fixed  tmp;


            FT_TRACE4(( " exch\n" ));

            tmp     = args[0];
            args[0] = args[1];
            args[1] = tmp;
            args   += 2;
          }
          break;

        case cff_op_index:
          {
            FT_Int  idx = (FT_Int)( args[0] >> 16 );


            FT_TRACE4(( " index\n" ));

            if ( idx < 0 )
              idx = 0;
            else if ( idx > num_args - 2 )
              idx = num_args - 2;
            args[0] = args[-( idx + 1 )];
            args++;
          }
          break;

        case cff_op_roll:
          {
            FT_Int  count = (FT_Int)( args[0] >> 16 );
            FT_Int  idx   = (FT_Int)( args[1] >> 16 );


            FT_TRACE4(( " roll\n" ));

            if ( count <= 0 )
              count = 1;

            args -= count;
            if ( args < stack )
              goto Stack_Underflow;

            if ( idx >= 0 )
            {
              idx = idx % count;
              while ( idx > 0 )
              {
                FT_Fixed  tmp = args[count - 1];
                FT_Int    i;


                for ( i = count - 2; i >= 0; i-- )
                  args[i + 1] = args[i];
                args[0] = tmp;
                idx--;
              }
            }
            else
            {
              /* before C99 it is implementation-defined whether    */
              /* the result of `%' is negative if the first operand */
              /* is negative                                        */
              idx = -( NEG_INT( idx ) % count );
              while ( idx < 0 )
              {
                FT_Fixed  tmp = args[0];
                FT_Int    i;


                for ( i = 0; i < count - 1; i++ )
                  args[i] = args[i + 1];
                args[count - 1] = tmp;
                idx++;
              }
            }
            args += count;
          }
          break;

        case cff_op_dup:
          FT_TRACE4(( " dup\n" ));

          args[1] = args[0];
          args   += 2;
          break;

        case cff_op_put:
          {
            FT_Fixed  val = args[0];
            FT_UInt   idx = (FT_UInt)( args[1] >> 16 );


            FT_TRACE4(( " put\n" ));

            /* the Type2 specification before version 16-March-2000 */
            /* didn't give a hard-coded size limit of the temporary */
            /* storage array; instead, an argument of the           */
            /* `MultipleMaster' operator set the size               */
            if ( idx < CFF_MAX_TRANS_ELEMENTS )
              decoder->buildchar[idx] = val;
          }
          break;

        case cff_op_get:
          {
            FT_UInt   idx = (FT_UInt)( args[0] >> 16 );
            FT_Fixed  val = 0;


            FT_TRACE4(( " get\n" ));

            if ( idx < CFF_MAX_TRANS_ELEMENTS )
              val = decoder->buildchar[idx];

            args[0] = val;
            args++;
          }
          break;

        case cff_op_store:
          /* this operator was removed from the Type2 specification */
          /* in version 16-March-2000                               */

          /* since we currently don't handle interpolation of multiple */
          /* master fonts, this is a no-op                             */
          FT_TRACE4(( " store\n" ));
          break;

        case cff_op_load:
          /* this operator was removed from the Type2 specification */
          /* in version 16-March-2000                               */
          {
            FT_UInt  reg_idx = (FT_UInt)args[0];
            FT_UInt  idx     = (FT_UInt)args[1];
            FT_UInt  count   = (FT_UInt)args[2];


            FT_TRACE4(( " load\n" ));

            /* since we currently don't handle interpolation of multiple */
            /* master fonts, we store a vector [1 0 0 ...] in the        */
            /* temporary storage array regardless of the Registry index  */
            if ( reg_idx <= 2                 &&
                 idx < CFF_MAX_TRANS_ELEMENTS &&
                 count <= num_axes            )
            {
              FT_UInt  end, i;


              end = FT_MIN( idx + count, CFF_MAX_TRANS_ELEMENTS );

              if ( idx < end )
                decoder->buildchar[idx] = 1 << 16;

              for ( i = idx + 1; i < end; i++ )
                decoder->buildchar[i] = 0;
            }
          }
          break;

        case cff_op_blend:
          /* this operator was removed from the Type2 specification */
          /* in version 16-March-2000                               */
          if ( num_designs )
          {
            FT_Int  num_results = (FT_Int)( args[0] >> 16 );


            FT_TRACE4(( " blend\n" ));

            if ( num_results < 0 )
              goto Syntax_Error;

            if ( num_results > num_args                       ||
                 num_results * (FT_Int)num_designs > num_args )
              goto Stack_Underflow;

            /* since we currently don't handle interpolation of multiple */
            /* master fonts, return the `num_results' values of the      */
            /* first master                                              */
            args     -= num_results * ( num_designs - 1 );
            num_args -= num_results * ( num_designs - 1 );
          }
          else
            goto Syntax_Error;
          break;

        case cff_op_dotsection:
          /* this operator is deprecated and ignored by the parser */
          FT_TRACE4(( " dotsection\n" ));
          break;

        case cff_op_closepath:
          /* this is an invalid Type 2 operator; however, there        */
          /* exist fonts which are incorrectly converted from probably */
          /* Type 1 to CFF, and some parsers seem to accept it         */

          FT_TRACE4(( " closepath (invalid op)\n" ));

          args = stack;
          break;

        case cff_op_hsbw:
          /* this is an invalid Type 2 operator; however, there        */
          /* exist fonts which are incorrectly converted from probably */
          /* Type 1 to CFF, and some parsers seem to accept it         */

          FT_TRACE4(( " hsbw (invalid op)\n" ));

          decoder->glyph_width =
            ADD_LONG( decoder->nominal_width, ( args[1] >> 16 ) );

          decoder->builder.left_bearing.x = args[0];
          decoder->builder.left_bearing.y = 0;

          x    = ADD_LONG( decoder->builder.pos_x, args[0] );
          y    = decoder->builder.pos_y;
          args = stack;
          break;

        case cff_op_sbw:
          /* this is an invalid Type 2 operator; however, there        */
          /* exist fonts which are incorrectly converted from probably */
          /* Type 1 to CFF, and some parsers seem to accept it         */

          FT_TRACE4(( " sbw (invalid op)\n" ));

          decoder->glyph_width =
            ADD_LONG( decoder->nominal_width, ( args[2] >> 16 ) );

          decoder->builder.left_bearing.x = args[0];
          decoder->builder.left_bearing.y = args[1];

          x    = ADD_LONG( decoder->builder.pos_x, args[0] );
          y    = ADD_LONG( decoder->builder.pos_y, args[1] );
          args = stack;
          break;

        case cff_op_setcurrentpoint:
          /* this is an invalid Type 2 operator; however, there        */
          /* exist fonts which are incorrectly converted from probably */
          /* Type 1 to CFF, and some parsers seem to accept it         */

          FT_TRACE4(( " setcurrentpoint (invalid op)\n" ));

          x    = ADD_LONG( decoder->builder.pos_x, args[0] );
          y    = ADD_LONG( decoder->builder.pos_y, args[1] );
          args = stack;
          break;

        case cff_op_callothersubr:
          {
            FT_Fixed  arg;


            /* this is an invalid Type 2 operator; however, there      */
            /* exist fonts which are incorrectly converted from        */
            /* probably Type 1 to CFF, and some parsers seem to accept */
            /* it                                                      */

            FT_TRACE4(( " callothersubr (invalid op)\n" ));

            /* subsequent `pop' operands should add the arguments,     */
            /* this is the implementation described for `unknown'      */
            /* other subroutines in the Type1 spec.                    */
            /*                                                         */
            /* XXX Fix return arguments (see discussion below).        */

            arg = 2 + ( args[-2] >> 16 );
            if ( arg >= CFF_MAX_OPERANDS )
              goto Stack_Underflow;

            args -= arg;
            if ( args < stack )
              goto Stack_Underflow;
          }
          break;

        case cff_op_pop:
          /* this is an invalid Type 2 operator; however, there        */
          /* exist fonts which are incorrectly converted from probably */
          /* Type 1 to CFF, and some parsers seem to accept it         */

          FT_TRACE4(( " pop (invalid op)\n" ));

          /* XXX Increasing `args' is wrong: After a certain number of */
          /* `pop's we get a stack overflow.  Reason for doing it is   */
          /* code like this (actually found in a CFF font):            */
          /*                                                           */
          /*   17 1 3 callothersubr                                    */
          /*   pop                                                     */
          /*   callsubr                                                */
          /*                                                           */
          /* Since we handle `callothersubr' as a no-op, and           */
          /* `callsubr' needs at least one argument, `pop' can't be a  */
          /* no-op too as it basically should be.                      */
          /*                                                           */
          /* The right solution would be to provide real support for   */
          /* `callothersubr' as done in `t1decode.c', however, given   */
          /* the fact that CFF fonts with `pop' are invalid, it is     */
          /* questionable whether it is worth the time.                */
          args++;
          break;

        case cff_op_and:
          {
            FT_Fixed  cond = ( args[0] && args[1] );


            FT_TRACE4(( " and\n" ));

            args[0] = cond ? 0x10000L : 0;
            args++;
          }
          break;

        case cff_op_or:
          {
            FT_Fixed  cond = ( args[0] || args[1] );


            FT_TRACE4(( " or\n" ));

            args[0] = cond ? 0x10000L : 0;
            args++;
          }
          break;

        case cff_op_not:
          {
            FT_Fixed  cond = !args[0];


            FT_TRACE4(( " not\n" ));

            args[0] = cond ? 0x10000L : 0;
            args++;
          }
          break;

        case cff_op_eq:
          {
            FT_Fixed  cond = ( args[0] == args[1] );


            FT_TRACE4(( " eq\n" ));

            args[0] = cond ? 0x10000L : 0;
            args++;
          }
          break;

        case cff_op_ifelse:
          {
            FT_Fixed  cond = ( args[2] <= args[3] );


            FT_TRACE4(( " ifelse\n" ));

            if ( !cond )
              args[0] = args[1];
            args++;
          }
          break;

        case cff_op_callsubr:
          {
            FT_UInt  idx = (FT_UInt)( ( args[0] >> 16 ) +
                                      decoder->locals_bias );


            FT_TRACE4(( " callsubr (idx %d, entering level %td)\n",
                        idx,
                        zone - decoder->zones + 1 ));

            if ( idx >= decoder->num_locals )
            {
              FT_ERROR(( "cff_decoder_parse_charstrings:"
                         " invalid local subr index\n" ));
              goto Syntax_Error;
            }

            if ( zone - decoder->zones >= CFF_MAX_SUBRS_CALLS )
            {
              FT_ERROR(( "cff_decoder_parse_charstrings:"
                         " too many nested subrs\n" ));
              goto Syntax_Error;
            }

            zone->cursor = ip;  /* save current instruction pointer */

            zone++;
            zone->base   = decoder->locals[idx];
            zone->limit  = decoder->locals[idx + 1];
            zone->cursor = zone->base;

            if ( !zone->base || zone->limit == zone->base )
            {
              FT_ERROR(( "cff_decoder_parse_charstrings:"
                         " invoking empty subrs\n" ));
              goto Syntax_Error;
            }

            decoder->zone = zone;
            ip            = zone->base;
            limit         = zone->limit;
          }
          break;

        case cff_op_callgsubr:
          {
            FT_UInt  idx = (FT_UInt)( ( args[0] >> 16 ) +
                                      decoder->globals_bias );


            FT_TRACE4(( " callgsubr (idx %d, entering level %td)\n",
                        idx,
                        zone - decoder->zones + 1 ));

            if ( idx >= decoder->num_globals )
            {
              FT_ERROR(( "cff_decoder_parse_charstrings:"
                         " invalid global subr index\n" ));
              goto Syntax_Error;
            }

            if ( zone - decoder->zones >= CFF_MAX_SUBRS_CALLS )
            {
              FT_ERROR(( "cff_decoder_parse_charstrings:"
                         " too many nested subrs\n" ));
              goto Syntax_Error;
            }

            zone->cursor = ip;  /* save current instruction pointer */

            zone++;
            zone->base   = decoder->globals[idx];
            zone->limit  = decoder->globals[idx + 1];
            zone->cursor = zone->base;

            if ( !zone->base || zone->limit == zone->base )
            {
              FT_ERROR(( "cff_decoder_parse_charstrings:"
                         " invoking empty subrs\n" ));
              goto Syntax_Error;
            }

            decoder->zone = zone;
            ip            = zone->base;
            limit         = zone->limit;
          }
          break;

        case cff_op_return:
          FT_TRACE4(( " return (leaving level %td)\n",
                      decoder->zone - decoder->zones ));

          if ( decoder->zone <= decoder->zones )
          {
            FT_ERROR(( "cff_decoder_parse_charstrings:"
                       " unexpected return\n" ));
            goto Syntax_Error;
          }

          decoder->zone--;
          zone  = decoder->zone;
          ip    = zone->cursor;
          limit = zone->limit;
          break;

        default:
          FT_ERROR(( "Unimplemented opcode: %d", ip[-1] ));

          if ( ip[-1] == 12 )
            FT_ERROR(( " %d", ip[0] ));
          FT_ERROR(( "\n" ));

          return FT_THROW( Unimplemented_Feature );
        }

        decoder->top = args;

        if ( decoder->top - stack >= CFF_MAX_OPERANDS )
          goto Stack_Overflow;

      } /* general operator processing */

    } /* while ip < limit */

    FT_TRACE4(( "..end..\n" ));
    FT_TRACE4(( "\n" ));

  Fail:
    return error;

  MM_Error:
    FT_TRACE4(( "cff_decoder_parse_charstrings:"
                " invalid opcode found in top DICT charstring\n"));
    return FT_THROW( Invalid_File_Format );

  Syntax_Error:
    FT_TRACE4(( "cff_decoder_parse_charstrings: syntax error\n" ));
    return FT_THROW( Invalid_File_Format );

  Stack_Underflow:
    FT_TRACE4(( "cff_decoder_parse_charstrings: stack underflow\n" ));
    return FT_THROW( Too_Few_Arguments );

  Stack_Overflow:
    FT_TRACE4(( "cff_decoder_parse_charstrings: stack overflow\n" ));
    return FT_THROW( Stack_Overflow );
  }

#endif /* CFF_CONFIG_OPTION_OLD_ENGINE */


  /**************************************************************************
   *
   * @Function:
   *   cff_decoder_init
   *
   * @Description:
   *   Initializes a given glyph decoder.
   *
   * @InOut:
   *   decoder ::
   *     A pointer to the glyph builder to initialize.
   *
   * @Input:
   *   face ::
   *     The current face object.
   *
   *   size ::
   *     The current size object.
   *
   *   slot ::
   *     The current glyph object.
   *
   *   hinting ::
   *     Whether hinting is active.
   *
   *   hint_mode ::
   *     The hinting mode.
   */
  FT_LOCAL_DEF( void )
  cff_decoder_init( CFF_Decoder*                     decoder,
                    TT_Face                          face,
                    CFF_Size                         size,
                    CFF_GlyphSlot                    slot,
                    FT_Bool                          hinting,
                    FT_Render_Mode                   hint_mode,
                    CFF_Decoder_Get_Glyph_Callback   get_callback,
                    CFF_Decoder_Free_Glyph_Callback  free_callback )
  {
    CFF_Font  cff = (CFF_Font)face->extra.data;


    /* clear everything */
    FT_ZERO( decoder );

    /* initialize builder */
    cff_builder_init( &decoder->builder, face, size, slot, hinting );

    /* initialize Type2 decoder */
    decoder->cff          = cff;
    decoder->num_globals  = cff->global_subrs_index.count;
    decoder->globals      = cff->global_subrs;
    decoder->globals_bias = cff_compute_bias(
                              cff->top_font.font_dict.charstring_type,
                              decoder->num_globals );

    decoder->hint_mode = hint_mode;

    decoder->get_glyph_callback  = get_callback;
    decoder->free_glyph_callback = free_callback;
  }


  /* this function is used to select the subfont */
  /* and the locals subrs array                  */
  FT_LOCAL_DEF( FT_Error )
  cff_decoder_prepare( CFF_Decoder*  decoder,
                       CFF_Size      size,
                       FT_UInt       glyph_index )
  {
    CFF_Builder  *builder = &decoder->builder;
    CFF_Font      cff     = (CFF_Font)builder->face->extra.data;
    CFF_SubFont   sub     = &cff->top_font;
    FT_Error      error   = FT_Err_Ok;

    FT_Service_CFFLoad  cffload = (FT_Service_CFFLoad)cff->cffload;


    /* manage CID fonts */
    if ( cff->num_subfonts )
    {
      FT_Byte  fd_index = cffload->fd_select_get( &cff->fd_select,
                                                  glyph_index );


      if ( fd_index >= cff->num_subfonts )
      {
        FT_TRACE4(( "cff_decoder_prepare: invalid CID subfont index\n" ));
        error = FT_THROW( Invalid_File_Format );
        goto Exit;
      }

      FT_TRACE3(( "  in subfont %d:\n", fd_index ));

      sub = cff->subfonts[fd_index];

      if ( builder->hints_funcs && size )
      {
        FT_Size       ftsize   = FT_SIZE( size );
        CFF_Internal  internal = (CFF_Internal)ftsize->internal->module_data;


        /* for CFFs without subfonts, this value has already been set */
        builder->hints_globals = (void *)internal->subfonts[fd_index];
      }
    }

    decoder->num_locals  = sub->local_subrs_index.count;
    decoder->locals      = sub->local_subrs;
    decoder->locals_bias = cff_compute_bias(
                             decoder->cff->top_font.font_dict.charstring_type,
                             decoder->num_locals );

    decoder->glyph_width   = sub->private_dict.default_width;
    decoder->nominal_width = sub->private_dict.nominal_width;

    decoder->current_subfont = sub;

  Exit:
    return error;
  }


/* END */
