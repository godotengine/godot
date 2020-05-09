/****************************************************************************
 *
 * psintrp.c
 *
 *   Adobe's CFF Interpreter (body).
 *
 * Copyright 2007-2014 Adobe Systems Incorporated.
 *
 * This software, and all works of authorship, whether in source or
 * object code form as indicated by the copyright notice(s) included
 * herein (collectively, the "Work") is made available, and may only be
 * used, modified, and distributed under the FreeType Project License,
 * LICENSE.TXT.  Additionally, subject to the terms and conditions of the
 * FreeType Project License, each contributor to the Work hereby grants
 * to any individual or legal entity exercising permissions granted by
 * the FreeType Project License and this section (hereafter, "You" or
 * "Your") a perpetual, worldwide, non-exclusive, no-charge,
 * royalty-free, irrevocable (except as stated in this section) patent
 * license to make, have made, use, offer to sell, sell, import, and
 * otherwise transfer the Work, where such license applies only to those
 * patent claims licensable by such contributor that are necessarily
 * infringed by their contribution(s) alone or by combination of their
 * contribution(s) with the Work to which such contribution(s) was
 * submitted.  If You institute patent litigation against any entity
 * (including a cross-claim or counterclaim in a lawsuit) alleging that
 * the Work or a contribution incorporated within the Work constitutes
 * direct or contributory patent infringement, then any patent licenses
 * granted to You under this License for that Work shall terminate as of
 * the date such litigation is filed.
 *
 * By using, modifying, or distributing the Work you indicate that you
 * have read and understood the terms and conditions of the
 * FreeType Project License as well as those provided in this section,
 * and you accept them fully.
 *
 */


#include "psft.h"
#include FT_INTERNAL_DEBUG_H
#include FT_SERVICE_CFF_TABLE_LOAD_H

#include "psglue.h"
#include "psfont.h"
#include "psstack.h"
#include "pshints.h"
#include "psintrp.h"

#include "pserror.h"

#include "psobjs.h"   /* for cff_random */
#include "t1decode.h" /* for t1 seac    */


  /**************************************************************************
   *
   * The macro FT_COMPONENT is used in trace mode.  It is an implicit
   * parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log
   * messages during execution.
   */
#undef  FT_COMPONENT
#define FT_COMPONENT  cf2interp


  FT_LOCAL_DEF( void )
  cf2_hintmask_init( CF2_HintMask  hintmask,
                     FT_Error*     error )
  {
    FT_ZERO( hintmask );

    hintmask->error = error;
  }


  FT_LOCAL_DEF( FT_Bool )
  cf2_hintmask_isValid( const CF2_HintMask  hintmask )
  {
    return hintmask->isValid;
  }


  FT_LOCAL_DEF( FT_Bool )
  cf2_hintmask_isNew( const CF2_HintMask  hintmask )
  {
    return hintmask->isNew;
  }


  FT_LOCAL_DEF( void )
  cf2_hintmask_setNew( CF2_HintMask  hintmask,
                       FT_Bool       val )
  {
    hintmask->isNew = val;
  }


  /* clients call `getMaskPtr' in order to iterate */
  /* through hint mask                             */

  FT_LOCAL_DEF( FT_Byte* )
  cf2_hintmask_getMaskPtr( CF2_HintMask  hintmask )
  {
    return hintmask->mask;
  }


  static size_t
  cf2_hintmask_setCounts( CF2_HintMask  hintmask,
                          size_t        bitCount )
  {
    if ( bitCount > CF2_MAX_HINTS )
    {
      /* total of h and v stems must be <= 96 */
      CF2_SET_ERROR( hintmask->error, Invalid_Glyph_Format );
      return 0;
    }

    hintmask->bitCount  = bitCount;
    hintmask->byteCount = ( hintmask->bitCount + 7 ) / 8;

    hintmask->isValid = TRUE;
    hintmask->isNew   = TRUE;

    return bitCount;
  }


  /* consume the hintmask bytes from the charstring, advancing the src */
  /* pointer                                                           */
  static void
  cf2_hintmask_read( CF2_HintMask  hintmask,
                     CF2_Buffer    charstring,
                     size_t        bitCount )
  {
    size_t  i;

#ifndef CF2_NDEBUG
    /* these are the bits in the final mask byte that should be zero  */
    /* Note: this variable is only used in an assert expression below */
    /* and then only if CF2_NDEBUG is not defined                     */
    CF2_UInt  mask = ( 1 << ( -(CF2_Int)bitCount & 7 ) ) - 1;
#endif


    /* initialize counts and isValid */
    if ( cf2_hintmask_setCounts( hintmask, bitCount ) == 0 )
      return;

    FT_ASSERT( hintmask->byteCount > 0 );

    FT_TRACE4(( " (maskbytes:" ));

    /* set mask and advance interpreter's charstring pointer */
    for ( i = 0; i < hintmask->byteCount; i++ )
    {
      hintmask->mask[i] = (FT_Byte)cf2_buf_readByte( charstring );
      FT_TRACE4(( " 0x%02X", hintmask->mask[i] ));
    }

    FT_TRACE4(( ")\n" ));

    /* assert any unused bits in last byte are zero unless there's a prior */
    /* error                                                               */
    /* bitCount -> mask, 0 -> 0, 1 -> 7f, 2 -> 3f, ... 6 -> 3, 7 -> 1      */
#ifndef CF2_NDEBUG
    FT_ASSERT( ( hintmask->mask[hintmask->byteCount - 1] & mask ) == 0 ||
               *hintmask->error                                        );
#endif
  }


  FT_LOCAL_DEF( void )
  cf2_hintmask_setAll( CF2_HintMask  hintmask,
                       size_t        bitCount )
  {
    size_t    i;
    CF2_UInt  mask = ( 1 << ( -(CF2_Int)bitCount & 7 ) ) - 1;


    /* initialize counts and isValid */
    if ( cf2_hintmask_setCounts( hintmask, bitCount ) == 0 )
      return;

    FT_ASSERT( hintmask->byteCount > 0 );
    FT_ASSERT( hintmask->byteCount <=
                 sizeof ( hintmask->mask ) / sizeof ( hintmask->mask[0] ) );

    /* set mask to all ones */
    for ( i = 0; i < hintmask->byteCount; i++ )
      hintmask->mask[i] = 0xFF;

    /* clear unused bits                                              */
    /* bitCount -> mask, 0 -> 0, 1 -> 7f, 2 -> 3f, ... 6 -> 3, 7 -> 1 */
    hintmask->mask[hintmask->byteCount - 1] &= ~mask;
  }


  /* Type2 charstring opcodes */
  enum
  {
    cf2_cmdRESERVED_0,   /* 0 */
    cf2_cmdHSTEM,        /* 1 */
    cf2_cmdRESERVED_2,   /* 2 */
    cf2_cmdVSTEM,        /* 3 */
    cf2_cmdVMOVETO,      /* 4 */
    cf2_cmdRLINETO,      /* 5 */
    cf2_cmdHLINETO,      /* 6 */
    cf2_cmdVLINETO,      /* 7 */
    cf2_cmdRRCURVETO,    /* 8 */
    cf2_cmdCLOSEPATH,    /* 9      T1 only */
    cf2_cmdCALLSUBR,     /* 10 */
    cf2_cmdRETURN,       /* 11 */
    cf2_cmdESC,          /* 12 */
    cf2_cmdHSBW,         /* 13     T1 only */
    cf2_cmdENDCHAR,      /* 14 */
    cf2_cmdVSINDEX,      /* 15 */
    cf2_cmdBLEND,        /* 16 */
    cf2_cmdRESERVED_17,  /* 17 */
    cf2_cmdHSTEMHM,      /* 18 */
    cf2_cmdHINTMASK,     /* 19 */
    cf2_cmdCNTRMASK,     /* 20 */
    cf2_cmdRMOVETO,      /* 21 */
    cf2_cmdHMOVETO,      /* 22 */
    cf2_cmdVSTEMHM,      /* 23 */
    cf2_cmdRCURVELINE,   /* 24 */
    cf2_cmdRLINECURVE,   /* 25 */
    cf2_cmdVVCURVETO,    /* 26 */
    cf2_cmdHHCURVETO,    /* 27 */
    cf2_cmdEXTENDEDNMBR, /* 28 */
    cf2_cmdCALLGSUBR,    /* 29 */
    cf2_cmdVHCURVETO,    /* 30 */
    cf2_cmdHVCURVETO     /* 31 */
  };

  enum
  {
    cf2_escDOTSECTION,   /* 0 */
    cf2_escVSTEM3,       /* 1      T1 only */
    cf2_escHSTEM3,       /* 2      T1 only */
    cf2_escAND,          /* 3 */
    cf2_escOR,           /* 4 */
    cf2_escNOT,          /* 5 */
    cf2_escSEAC,         /* 6      T1 only */
    cf2_escSBW,          /* 7      T1 only */
    cf2_escRESERVED_8,   /* 8 */
    cf2_escABS,          /* 9 */
    cf2_escADD,          /* 10     like otherADD */
    cf2_escSUB,          /* 11     like otherSUB */
    cf2_escDIV,          /* 12 */
    cf2_escRESERVED_13,  /* 13 */
    cf2_escNEG,          /* 14 */
    cf2_escEQ,           /* 15 */
    cf2_escCALLOTHERSUBR,/* 16     T1 only */
    cf2_escPOP,          /* 17     T1 only */
    cf2_escDROP,         /* 18 */
    cf2_escRESERVED_19,  /* 19 */
    cf2_escPUT,          /* 20     like otherPUT    */
    cf2_escGET,          /* 21     like otherGET    */
    cf2_escIFELSE,       /* 22     like otherIFELSE */
    cf2_escRANDOM,       /* 23     like otherRANDOM */
    cf2_escMUL,          /* 24     like otherMUL    */
    cf2_escRESERVED_25,  /* 25 */
    cf2_escSQRT,         /* 26 */
    cf2_escDUP,          /* 27     like otherDUP    */
    cf2_escEXCH,         /* 28     like otherEXCH   */
    cf2_escINDEX,        /* 29 */
    cf2_escROLL,         /* 30 */
    cf2_escRESERVED_31,  /* 31 */
    cf2_escRESERVED_32,  /* 32 */
    cf2_escSETCURRENTPT, /* 33     T1 only */
    cf2_escHFLEX,        /* 34 */
    cf2_escFLEX,         /* 35 */
    cf2_escHFLEX1,       /* 36 */
    cf2_escFLEX1,        /* 37 */
    cf2_escRESERVED_38   /* 38     & all higher     */
  };


  /* `stemHintArray' does not change once we start drawing the outline. */
  static void
  cf2_doStems( const CF2_Font  font,
               CF2_Stack       opStack,
               CF2_ArrStack    stemHintArray,
               CF2_Fixed*      width,
               FT_Bool*        haveWidth,
               CF2_Fixed       hintOffset )
  {
    CF2_UInt  i;
    CF2_UInt  count       = cf2_stack_count( opStack );
    FT_Bool   hasWidthArg = FT_BOOL( count & 1 );

    /* variable accumulates delta values from operand stack */
    CF2_Fixed  position = hintOffset;

    if ( font->isT1 && !font->decoder->flex_state && !*haveWidth )
      FT_ERROR(( "cf2_doStems (Type 1 mode):"
                 " No width. Use hsbw/sbw as first op\n" ));

    if ( !font->isT1 && hasWidthArg && !*haveWidth )
      *width = ADD_INT32( cf2_stack_getReal( opStack, 0 ),
                          cf2_getNominalWidthX( font->decoder ) );

    if ( font->decoder->width_only )
      goto exit;

    for ( i = hasWidthArg ? 1 : 0; i < count; i += 2 )
    {
      /* construct a CF2_StemHint and push it onto the list */
      CF2_StemHintRec  stemhint;


      stemhint.min =
      position     = ADD_INT32( position,
                                cf2_stack_getReal( opStack, i ) );
      stemhint.max =
      position     = ADD_INT32( position,
                                cf2_stack_getReal( opStack, i + 1 ) );

      stemhint.used  = FALSE;
      stemhint.maxDS =
      stemhint.minDS = 0;

      cf2_arrstack_push( stemHintArray, &stemhint ); /* defer error check */
    }

    cf2_stack_clear( opStack );

  exit:
    /* cf2_doStems must define a width (may be default) */
    *haveWidth = TRUE;
  }


  static void
  cf2_doFlex( CF2_Stack       opStack,
              CF2_Fixed*      curX,
              CF2_Fixed*      curY,
              CF2_GlyphPath   glyphPath,
              const FT_Bool*  readFromStack,
              FT_Bool         doConditionalLastRead )
  {
    CF2_Fixed  vals[14];
    CF2_UInt   idx;
    FT_Bool    isHFlex;
    CF2_Int    top, i, j;


    vals[0] = *curX;
    vals[1] = *curY;
    idx     = 0;
    isHFlex = FT_BOOL( readFromStack[9] == FALSE );
    top     = isHFlex ? 9 : 10;

    for ( i = 0; i < top; i++ )
    {
      vals[i + 2] = vals[i];
      if ( readFromStack[i] )
        vals[i + 2] = ADD_INT32( vals[i + 2], cf2_stack_getReal( opStack,
                                                                 idx++ ) );
    }

    if ( isHFlex )
      vals[9 + 2] = *curY;

    if ( doConditionalLastRead )
    {
      FT_Bool    lastIsX = FT_BOOL(
                             cf2_fixedAbs( SUB_INT32( vals[10], *curX ) ) >
                             cf2_fixedAbs( SUB_INT32( vals[11], *curY ) ) );
      CF2_Fixed  lastVal = cf2_stack_getReal( opStack, idx );


      if ( lastIsX )
      {
        vals[12] = ADD_INT32( vals[10], lastVal );
        vals[13] = *curY;
      }
      else
      {
        vals[12] = *curX;
        vals[13] = ADD_INT32( vals[11], lastVal );
      }
    }
    else
    {
      if ( readFromStack[10] )
        vals[12] = ADD_INT32( vals[10],
                              cf2_stack_getReal( opStack, idx++ ) );
      else
        vals[12] = *curX;

      if ( readFromStack[11] )
        vals[13] = ADD_INT32( vals[11],
                              cf2_stack_getReal( opStack, idx ) );
      else
        vals[13] = *curY;
    }

    for ( j = 0; j < 2; j++ )
      cf2_glyphpath_curveTo( glyphPath, vals[j * 6 + 2],
                                        vals[j * 6 + 3],
                                        vals[j * 6 + 4],
                                        vals[j * 6 + 5],
                                        vals[j * 6 + 6],
                                        vals[j * 6 + 7] );

    cf2_stack_clear( opStack );

    *curX = vals[12];
    *curY = vals[13];
  }


  /* Blend numOperands on the stack,                */
  /* store results into the first numBlends values, */
  /* then pop remaining arguments.                  */
  static void
  cf2_doBlend( const CFF_Blend  blend,
               CF2_Stack        opStack,
               CF2_UInt         numBlends )
  {
    CF2_UInt  delta;
    CF2_UInt  base;
    CF2_UInt  i, j;
    CF2_UInt  numOperands = (CF2_UInt)( numBlends * blend->lenBV );


    base  = cf2_stack_count( opStack ) - numOperands;
    delta = base + numBlends;

    for ( i = 0; i < numBlends; i++ )
    {
      const CF2_Fixed*  weight = &blend->BV[1];

      /* start with first term */
      CF2_Fixed  sum = cf2_stack_getReal( opStack, i + base );


      for ( j = 1; j < blend->lenBV; j++ )
        sum = ADD_INT32( sum,
                         FT_MulFix( *weight++,
                                    cf2_stack_getReal( opStack,
                                                       delta++ ) ) );

      /* store blended result  */
      cf2_stack_setReal( opStack, i + base, sum );
    }

    /* leave only `numBlends' results on stack */
    cf2_stack_pop( opStack, numOperands - numBlends );
  }


  /*
   * `error' is a shared error code used by many objects in this
   * routine.  Before the code continues from an error, it must check and
   * record the error in `*error'.  The idea is that this shared
   * error code will record the first error encountered.  If testing
   * for an error anyway, the cost of `goto exit' is small, so we do it,
   * even if continuing would be safe.  In this case, `lastError' is
   * set, so the testing and storing can be done in one place, at `exit'.
   *
   * Continuing after an error is intended for objects which do their own
   * testing of `*error', e.g., array stack functions.  This allows us to
   * avoid an extra test after the call.
   *
   * Unimplemented opcodes are ignored.
   *
   */
  FT_LOCAL_DEF( void )
  cf2_interpT2CharString( CF2_Font              font,
                          CF2_Buffer            buf,
                          CF2_OutlineCallbacks  callbacks,
                          const FT_Vector*      translation,
                          FT_Bool               doingSeac,
                          CF2_Fixed             curX,
                          CF2_Fixed             curY,
                          CF2_Fixed*            width )
  {
    /* lastError is used for errors that are immediately tested */
    FT_Error  lastError = FT_Err_Ok;

    /* pointer to parsed font object */
    PS_Decoder*  decoder = font->decoder;

    FT_Error*  error  = &font->error;
    FT_Memory  memory = font->memory;

    CF2_Fixed  scaleY        = font->innerTransform.d;
    CF2_Fixed  nominalWidthX = cf2_getNominalWidthX( decoder );

    /* stuff for Type 1 */
    FT_Int   known_othersubr_result_cnt = 0;
    FT_Bool  large_int                  = FALSE;
    FT_Bool  initial_map_ready          = FALSE;

#define PS_STORAGE_SIZE 3
    CF2_F16Dot16  results[PS_STORAGE_SIZE];   /* for othersubr results */
    FT_Int        result_cnt = 0;

    /* save this for hinting seac accents */
    CF2_Fixed  hintOriginY = curY;

    CF2_Stack  opStack = NULL;
    FT_UInt    stackSize;
    FT_Byte    op1;                       /* first opcode byte */

    CF2_F16Dot16  storage[CF2_STORAGE_SIZE];    /* for `put' and `get' */
    CF2_F16Dot16  flexStore[6];                 /* for Type 1 flex     */

    /* instruction limit; 20,000,000 matches Avalon */
    FT_UInt32  instructionLimit = 20000000UL;

    CF2_ArrStackRec  subrStack;

    FT_Bool     haveWidth;
    CF2_Buffer  charstring = NULL;

    CF2_Int  charstringIndex = -1;       /* initialize to empty */

    /* TODO: placeholders for hint structures */

    /* objects used for hinting */
    CF2_ArrStackRec  hStemHintArray;
    CF2_ArrStackRec  vStemHintArray;

    CF2_HintMaskRec   hintMask;
    CF2_GlyphPathRec  glyphPath;


    FT_ZERO( &storage );
    FT_ZERO( &results );
    FT_ZERO( &flexStore );

    /* initialize the remaining objects */
    cf2_arrstack_init( &subrStack,
                       memory,
                       error,
                       sizeof ( CF2_BufferRec ) );
    cf2_arrstack_init( &hStemHintArray,
                       memory,
                       error,
                       sizeof ( CF2_StemHintRec ) );
    cf2_arrstack_init( &vStemHintArray,
                       memory,
                       error,
                       sizeof ( CF2_StemHintRec ) );

    /* initialize CF2_StemHint arrays */
    cf2_hintmask_init( &hintMask, error );

    /* initialize path map to manage drawing operations */

    /* Note: last 4 params are used to handle `MoveToPermissive', which */
    /*       may need to call `hintMap.Build'                           */
    /* TODO: MoveToPermissive is gone; are these still needed?          */
    cf2_glyphpath_init( &glyphPath,
                        font,
                        callbacks,
                        scaleY,
                        /* hShift, */
                        &hStemHintArray,
                        &vStemHintArray,
                        &hintMask,
                        hintOriginY,
                        &font->blues,
                        translation );

    /*
     * Initialize state for width parsing.  From the CFF Spec:
     *
     *   The first stack-clearing operator, which must be one of hstem,
     *   hstemhm, vstem, vstemhm, cntrmask, hintmask, hmoveto, vmoveto,
     *   rmoveto, or endchar, takes an additional argument - the width (as
     *   described earlier), which may be expressed as zero or one numeric
     *   argument.
     *
     * What we implement here uses the first validly specified width, but
     * does not detect errors for specifying more than one width.
     *
     * If one of the above operators occurs without explicitly specifying
     * a width, we assume the default width.
     *
     * CFF2 charstrings always return the default width (0).
     *
     */
    haveWidth = font->isCFF2 ? TRUE : FALSE;
    *width    = cf2_getDefaultWidthX( decoder );

    /*
     * Note: At this point, all pointers to resources must be NULL
     *       and all local objects must be initialized.
     *       There must be no branches to `exit:' above this point.
     *
     */

    /* allocate an operand stack */
    stackSize = font->isCFF2 ? cf2_getMaxstack( decoder )
                             : CF2_OPERAND_STACK_SIZE;
    opStack   = cf2_stack_init( memory, error, stackSize );

    if ( !opStack )
    {
      lastError = FT_THROW( Out_Of_Memory );
      goto exit;
    }

    /* initialize subroutine stack by placing top level charstring as */
    /* first element (max depth plus one for the charstring)          */
    /* Note: Caller owns and must finalize the first charstring.      */
    /*       Our copy of it does not change that requirement.         */
    cf2_arrstack_setCount( &subrStack, CF2_MAX_SUBR + 1 );

    charstring  = (CF2_Buffer)cf2_arrstack_getBuffer( &subrStack );

    /* catch errors so far */
    if ( *error )
      goto exit;

    *charstring     = *buf;    /* structure copy     */
    charstringIndex = 0;       /* entry is valid now */

    /* main interpreter loop */
    while ( 1 )
    {
      if ( font->isT1 )
        FT_ASSERT( known_othersubr_result_cnt == 0 ||
                   result_cnt == 0                 );

      if ( cf2_buf_isEnd( charstring ) )
      {
        /* If we've reached the end of the charstring, simulate a */
        /* cf2_cmdRETURN or cf2_cmdENDCHAR.                       */
        /* We do this for both CFF and CFF2.                      */
        if ( charstringIndex )
          op1 = cf2_cmdRETURN;  /* end of buffer for subroutine */
        else
          op1 = cf2_cmdENDCHAR; /* end of buffer for top level charstring */
      }
      else
      {
        op1 = (FT_Byte)cf2_buf_readByte( charstring );

        /* Explicit RETURN and ENDCHAR in CFF2 should be ignored. */
        /* Note: Trace message will report 0 instead of 11 or 14. */
        if ( ( op1 == cf2_cmdRETURN || op1 == cf2_cmdENDCHAR ) &&
             font->isCFF2                                      )
          op1 = cf2_cmdRESERVED_0;
      }

      if ( font->isT1 )
      {
        if ( !initial_map_ready           &&
             !( op1 == cf2_cmdHSTEM     ||
                op1 == cf2_cmdVSTEM     ||
                op1 == cf2_cmdHSBW      ||
                op1 == cf2_cmdCALLSUBR  ||
                op1 == cf2_cmdRETURN    ||
                op1 == cf2_cmdESC       ||
                op1 == cf2_cmdENDCHAR   ||
                op1 >= 32 /* Numbers */ ) )
        {
          /* Skip outline commands first time round.       */
          /* `endchar' will trigger initial hintmap build  */
          /* and rewind the charstring.                    */
          FT_TRACE4(( " <outline command skipped>\n" ));
          cf2_stack_clear( opStack );
          continue;
        }

        if ( result_cnt > 0               &&
             !( op1 == cf2_cmdCALLSUBR  ||
                op1 == cf2_cmdRETURN    ||
                op1 == cf2_cmdESC       ||
                op1 >= 32 /* Numbers */ ) )
        {
          /* all operands have been transferred by previous pops */
          result_cnt = 0;
        }

        if ( large_int && !( op1 >= 32 || op1 == cf2_escDIV ) )
        {
          FT_ERROR(( "cf2_interpT2CharString (Type 1 mode):"
                     " no `div' after large integer\n" ));

          large_int = FALSE;
        }
      }

      /* check for errors once per loop */
      if ( *error )
        goto exit;

      instructionLimit--;
      if ( instructionLimit == 0 )
      {
        lastError = FT_THROW( Invalid_Glyph_Format );
        goto exit;
      }

      switch( op1 )
      {
      case cf2_cmdRESERVED_0:
      case cf2_cmdRESERVED_2:
      case cf2_cmdRESERVED_17:
        /* we may get here if we have a prior error */
        FT_TRACE4(( " unknown op (%d)\n", op1 ));
        break;

      case cf2_cmdVSINDEX:
        FT_TRACE4(( " vsindex\n" ));

        if ( !font->isCFF2 )
          break;    /* clear stack & ignore */

        if ( font->blend.usedBV )
        {
          /* vsindex not allowed after blend */
          lastError = FT_THROW( Invalid_Glyph_Format );
          goto exit;
        }

        {
          FT_Int  temp = cf2_stack_popInt( opStack );


          if ( temp >= 0 )
            font->vsindex = (FT_UInt)temp;
        }
        break;

      case cf2_cmdBLEND:
        {
          FT_UInt  numBlends;


          FT_TRACE4(( " blend\n" ));

          if ( !font->isCFF2 )
            break;    /* clear stack & ignore */

          /* do we have a `blend' op in a non-variant font? */
          if ( !font->blend.font )
          {
            lastError = FT_THROW( Invalid_Glyph_Format );
            goto exit;
          }

          /* check cached blend vector */
          if ( font->cffload->blend_check_vector( &font->blend,
                                                  font->vsindex,
                                                  font->lenNDV,
                                                  font->NDV ) )
          {
            lastError = font->cffload->blend_build_vector( &font->blend,
                                                           font->vsindex,
                                                           font->lenNDV,
                                                           font->NDV );
            if ( lastError )
              goto exit;
          }

          /* do the blend */
          numBlends = (FT_UInt)cf2_stack_popInt( opStack );
          if ( numBlends > stackSize )
          {
            lastError = FT_THROW( Invalid_Glyph_Format );
            goto exit;
          }

          cf2_doBlend( &font->blend, opStack, numBlends );

          font->blend.usedBV = TRUE;
        }
        continue;     /* do not clear the stack */

      case cf2_cmdHSTEMHM:
      case cf2_cmdHSTEM:
        FT_TRACE4(( "%s\n", op1 == cf2_cmdHSTEMHM ? " hstemhm"
                                                  : " hstem" ));

        if ( !font->isT1 )
        {
          /* never add hints after the mask is computed */
          /* except if in Type 1 mode (no hintmask op)  */
          if ( cf2_hintmask_isValid( &hintMask ) )
          {
            FT_TRACE4(( "cf2_interpT2CharString:"
                        " invalid horizontal hint mask\n" ));
            break;
          }
        }

        /* add left-sidebearing correction in Type 1 mode */
        cf2_doStems( font,
                     opStack,
                     &hStemHintArray,
                     width,
                     &haveWidth,
                     font->isT1 ? decoder->builder.left_bearing->y
                                : 0 );

        if ( decoder->width_only )
          goto exit;

        break;

      case cf2_cmdVSTEMHM:
      case cf2_cmdVSTEM:
        FT_TRACE4(( "%s\n", op1 == cf2_cmdVSTEMHM ? " vstemhm"
                                                  : " vstem" ));

        if ( !font->isT1 )
        {
          /* never add hints after the mask is computed */
          /* except if in Type 1 mode (no hintmask op)  */
          if ( cf2_hintmask_isValid( &hintMask ) )
          {
            FT_TRACE4(( "cf2_interpT2CharString:"
                        " invalid vertical hint mask\n" ));
            break;
          }
        }

        /* add left-sidebearing correction in Type 1 mode */
        cf2_doStems( font,
                     opStack,
                     &vStemHintArray,
                     width,
                     &haveWidth,
                     font->isT1 ? decoder->builder.left_bearing->x
                                : 0 );

        if ( decoder->width_only )
          goto exit;

        break;

      case cf2_cmdVMOVETO:
        FT_TRACE4(( " vmoveto\n" ));

        if ( font->isT1 && !decoder->flex_state && !haveWidth )
          FT_ERROR(( "cf2_interpT2CharString (Type 1 mode):"
                     " No width. Use hsbw/sbw as first op\n" ));

        if ( cf2_stack_count( opStack ) > 1 && !haveWidth )
          *width = ADD_INT32( cf2_stack_getReal( opStack, 0 ),
                              nominalWidthX );

        /* width is defined or default after this */
        haveWidth = TRUE;

        if ( decoder->width_only )
          goto exit;

        curY = ADD_INT32( curY, cf2_stack_popFixed( opStack ) );

        if ( !decoder->flex_state )
          cf2_glyphpath_moveTo( &glyphPath, curX, curY );

        break;

      case cf2_cmdRLINETO:
        {
          CF2_UInt  idx;
          CF2_UInt  count = cf2_stack_count( opStack );


          FT_TRACE4(( " rlineto\n" ));

          for ( idx = 0; idx < count; idx += 2 )
          {
            curX = ADD_INT32( curX, cf2_stack_getReal( opStack,
                                                       idx + 0 ) );
            curY = ADD_INT32( curY, cf2_stack_getReal( opStack,
                                                       idx + 1 ) );

            cf2_glyphpath_lineTo( &glyphPath, curX, curY );
          }

          cf2_stack_clear( opStack );
        }
        continue; /* no need to clear stack again */

      case cf2_cmdHLINETO:
      case cf2_cmdVLINETO:
        {
          CF2_UInt  idx;
          CF2_UInt  count = cf2_stack_count( opStack );

          FT_Bool  isX = FT_BOOL( op1 == cf2_cmdHLINETO );


          FT_TRACE4(( "%s\n", isX ? " hlineto" : " vlineto" ));

          for ( idx = 0; idx < count; idx++ )
          {
            CF2_Fixed  v = cf2_stack_getReal( opStack, idx );


            if ( isX )
              curX = ADD_INT32( curX, v );
            else
              curY = ADD_INT32( curY, v );

            isX = !isX;

            cf2_glyphpath_lineTo( &glyphPath, curX, curY );
          }

          cf2_stack_clear( opStack );
        }
        continue;

      case cf2_cmdRCURVELINE:
      case cf2_cmdRRCURVETO:
        {
          CF2_UInt  count = cf2_stack_count( opStack );
          CF2_UInt  idx   = 0;


          FT_TRACE4(( "%s\n", op1 == cf2_cmdRCURVELINE ? " rcurveline"
                                                       : " rrcurveto" ));

          while ( idx + 6 <= count )
          {
            CF2_Fixed  x1, y1, x2, y2, x3, y3;


            x1 = ADD_INT32( cf2_stack_getReal( opStack, idx + 0 ), curX );
            y1 = ADD_INT32( cf2_stack_getReal( opStack, idx + 1 ), curY );
            x2 = ADD_INT32( cf2_stack_getReal( opStack, idx + 2 ), x1 );
            y2 = ADD_INT32( cf2_stack_getReal( opStack, idx + 3 ), y1 );
            x3 = ADD_INT32( cf2_stack_getReal( opStack, idx + 4 ), x2 );
            y3 = ADD_INT32( cf2_stack_getReal( opStack, idx + 5 ), y2 );

            cf2_glyphpath_curveTo( &glyphPath, x1, y1, x2, y2, x3, y3 );

            curX  = x3;
            curY  = y3;
            idx  += 6;
          }

          if ( op1 == cf2_cmdRCURVELINE )
          {
            curX = ADD_INT32( curX, cf2_stack_getReal( opStack,
                                                       idx + 0 ) );
            curY = ADD_INT32( curY, cf2_stack_getReal( opStack,
                                                       idx + 1 ) );

            cf2_glyphpath_lineTo( &glyphPath, curX, curY );
          }

          cf2_stack_clear( opStack );
        }
        continue; /* no need to clear stack again */

      case cf2_cmdCLOSEPATH:
        if ( !font->isT1 )
          FT_TRACE4(( " unknown op (%d)\n", op1 ));
        else
        {
          FT_TRACE4(( " closepath\n" ));

          /* if there is no path, `closepath' is a no-op */
          cf2_glyphpath_closeOpenPath( &glyphPath );

          haveWidth = TRUE;
        }
        break;

      case cf2_cmdCALLGSUBR:
      case cf2_cmdCALLSUBR:
        {
          CF2_Int  subrNum;


          FT_TRACE4(( "%s", op1 == cf2_cmdCALLGSUBR ? " callgsubr"
                                                    : " callsubr" ));

          if ( ( !font->isT1 && charstringIndex > CF2_MAX_SUBR )       ||
               (  font->isT1 && charstringIndex > T1_MAX_SUBRS_CALLS ) )
          {
            /* max subr plus one for charstring */
            lastError = FT_THROW( Invalid_Glyph_Format );
            goto exit;                      /* overflow of stack */
          }

          /* push our current CFF charstring region on subrStack */
          charstring = (CF2_Buffer)
                         cf2_arrstack_getPointer(
                           &subrStack,
                           (size_t)charstringIndex + 1 );

          /* set up the new CFF region and pointer */
          subrNum = cf2_stack_popInt( opStack );

          if ( font->isT1 && decoder->locals_hash )
          {
            size_t*  val = ft_hash_num_lookup( subrNum,
                                               decoder->locals_hash );


            if ( val )
              subrNum = *val;
            else
              subrNum = -1;
          }

          switch ( op1 )
          {
          case cf2_cmdCALLGSUBR:
            FT_TRACE4(( " (idx %d, entering level %d)\n",
                        subrNum + decoder->globals_bias,
                        charstringIndex + 1 ));

            if ( cf2_initGlobalRegionBuffer( decoder,
                                             subrNum,
                                             charstring ) )
            {
              lastError = FT_THROW( Invalid_Glyph_Format );
              goto exit;  /* subroutine lookup or stream error */
            }
            break;

          default:
            /* cf2_cmdCALLSUBR */
            FT_TRACE4(( " (idx %d, entering level %d)\n",
                        subrNum + decoder->locals_bias,
                        charstringIndex + 1 ));

            if ( cf2_initLocalRegionBuffer( decoder,
                                            subrNum,
                                            charstring ) )
            {
              lastError = FT_THROW( Invalid_Glyph_Format );
              goto exit;  /* subroutine lookup or stream error */
            }
          }

          charstringIndex += 1;       /* entry is valid now */
        }
        continue; /* do not clear the stack */

      case cf2_cmdRETURN:
        FT_TRACE4(( " return (leaving level %d)\n", charstringIndex ));

        if ( charstringIndex < 1 )
        {
          /* Note: cannot return from top charstring */
          lastError = FT_THROW( Invalid_Glyph_Format );
          goto exit;                      /* underflow of stack */
        }

        /* restore position in previous charstring */
        charstring = (CF2_Buffer)
                       cf2_arrstack_getPointer(
                         &subrStack,
                         (CF2_UInt)--charstringIndex );
        continue;     /* do not clear the stack */

      case cf2_cmdESC:
        {
          FT_Byte  op2 = (FT_Byte)cf2_buf_readByte( charstring );


          /* first switch for 2-byte operators handles CFF2      */
          /* and opcodes that are reserved for both CFF and CFF2 */
          switch ( op2 )
          {
          case cf2_escHFLEX:
            {
              static const FT_Bool  readFromStack[12] =
              {
                TRUE /* dx1 */, FALSE /* dy1 */,
                TRUE /* dx2 */, TRUE  /* dy2 */,
                TRUE /* dx3 */, FALSE /* dy3 */,
                TRUE /* dx4 */, FALSE /* dy4 */,
                TRUE /* dx5 */, FALSE /* dy5 */,
                TRUE /* dx6 */, FALSE /* dy6 */
              };


              FT_TRACE4(( " hflex\n" ));

              cf2_doFlex( opStack,
                          &curX,
                          &curY,
                          &glyphPath,
                          readFromStack,
                          FALSE /* doConditionalLastRead */ );
            }
            continue;

          case cf2_escFLEX:
            {
              static const FT_Bool  readFromStack[12] =
              {
                TRUE /* dx1 */, TRUE /* dy1 */,
                TRUE /* dx2 */, TRUE /* dy2 */,
                TRUE /* dx3 */, TRUE /* dy3 */,
                TRUE /* dx4 */, TRUE /* dy4 */,
                TRUE /* dx5 */, TRUE /* dy5 */,
                TRUE /* dx6 */, TRUE /* dy6 */
              };


              FT_TRACE4(( " flex\n" ));

              cf2_doFlex( opStack,
                          &curX,
                          &curY,
                          &glyphPath,
                          readFromStack,
                          FALSE /* doConditionalLastRead */ );
            }
            break;      /* TODO: why is this not a continue? */

          case cf2_escHFLEX1:
            {
              static const FT_Bool  readFromStack[12] =
              {
                TRUE /* dx1 */, TRUE  /* dy1 */,
                TRUE /* dx2 */, TRUE  /* dy2 */,
                TRUE /* dx3 */, FALSE /* dy3 */,
                TRUE /* dx4 */, FALSE /* dy4 */,
                TRUE /* dx5 */, TRUE  /* dy5 */,
                TRUE /* dx6 */, FALSE /* dy6 */
              };


              FT_TRACE4(( " hflex1\n" ));

              cf2_doFlex( opStack,
                          &curX,
                          &curY,
                          &glyphPath,
                          readFromStack,
                          FALSE /* doConditionalLastRead */ );
            }
            continue;

          case cf2_escFLEX1:
            {
              static const FT_Bool  readFromStack[12] =
              {
                TRUE  /* dx1 */, TRUE  /* dy1 */,
                TRUE  /* dx2 */, TRUE  /* dy2 */,
                TRUE  /* dx3 */, TRUE  /* dy3 */,
                TRUE  /* dx4 */, TRUE  /* dy4 */,
                TRUE  /* dx5 */, TRUE  /* dy5 */,
                FALSE /* dx6 */, FALSE /* dy6 */
              };


              FT_TRACE4(( " flex1\n" ));

              cf2_doFlex( opStack,
                          &curX,
                          &curY,
                          &glyphPath,
                          readFromStack,
                          TRUE /* doConditionalLastRead */ );
            }
            continue;

          /* these opcodes are always reserved */
          case cf2_escRESERVED_8:
          case cf2_escRESERVED_13:
          case cf2_escRESERVED_19:
          case cf2_escRESERVED_25:
          case cf2_escRESERVED_31:
          case cf2_escRESERVED_32:
            FT_TRACE4(( " unknown op (12, %d)\n", op2 ));
            break;

          default:
            {
              if ( font->isCFF2 || op2 >= cf2_escRESERVED_38 )
                FT_TRACE4(( " unknown op (12, %d)\n", op2 ));
              else if ( font->isT1 && result_cnt > 0 && op2 != cf2_escPOP )
              {
                /* all operands have been transferred by previous pops */
                result_cnt = 0;
              }
              else
              {
                /* second switch for 2-byte operators handles */
                /* CFF and Type 1                             */
                switch ( op2 )
                {

                case cf2_escDOTSECTION:
                  /* something about `flip type of locking' -- ignore it */
                  FT_TRACE4(( " dotsection\n" ));

                  break;

                case cf2_escVSTEM3:
                case cf2_escHSTEM3:
                  /*
                   * Type 1:                          Type 2:
                   *   x0 dx0 x1 dx1 x2 dx2 vstem3      x dx {dxa dxb}* vstem
                   *   y0 dy0 y1 dy1 y2 dy2 hstem3      y dy {dya dyb}* hstem
                   *   relative to lsb point            relative to zero
                   *
                   */
                  {
                    if ( !font->isT1 )
                      FT_TRACE4(( " unknown op (12, %d)\n", op2 ));
                    else
                    {
                      CF2_F16Dot16  v0, v1, v2;

                      FT_Bool  isV = FT_BOOL( op2 == cf2_escVSTEM3 );


                      FT_TRACE4(( "%s\n", isV ? " vstem3"
                                              : " hstem3" ));

                      FT_ASSERT( cf2_stack_count( opStack ) == 6 );

                      v0 = cf2_stack_getReal( opStack, 0 );
                      v1 = cf2_stack_getReal( opStack, 2 );
                      v2 = cf2_stack_getReal( opStack, 4 );

                      cf2_stack_setReal(
                        opStack, 2,
                        SUB_INT32( SUB_INT32( v1, v0 ),
                                   cf2_stack_getReal( opStack, 1 ) ) );
                      cf2_stack_setReal(
                        opStack, 4,
                        SUB_INT32( SUB_INT32( v2, v1 ),
                                   cf2_stack_getReal( opStack, 3 ) ) );

                      /* add left-sidebearing correction */
                      cf2_doStems( font,
                                   opStack,
                                   isV ? &vStemHintArray : &hStemHintArray,
                                   width,
                                   &haveWidth,
                                   isV ? decoder->builder.left_bearing->x
                                       : decoder->builder.left_bearing->y );

                      if ( decoder->width_only )
                        goto exit;
                    }
                  }
                  break;

                case cf2_escAND:
                  {
                    CF2_F16Dot16  arg1;
                    CF2_F16Dot16  arg2;


                    FT_TRACE4(( " and\n" ));

                    arg2 = cf2_stack_popFixed( opStack );
                    arg1 = cf2_stack_popFixed( opStack );

                    cf2_stack_pushInt( opStack, arg1 && arg2 );
                  }
                  continue; /* do not clear the stack */

                case cf2_escOR:
                  {
                    CF2_F16Dot16  arg1;
                    CF2_F16Dot16  arg2;


                    FT_TRACE4(( " or\n" ));

                    arg2 = cf2_stack_popFixed( opStack );
                    arg1 = cf2_stack_popFixed( opStack );

                    cf2_stack_pushInt( opStack, arg1 || arg2 );
                  }
                  continue; /* do not clear the stack */

                case cf2_escNOT:
                  {
                    CF2_F16Dot16  arg;


                    FT_TRACE4(( " not\n" ));

                    arg = cf2_stack_popFixed( opStack );

                    cf2_stack_pushInt( opStack, !arg );
                  }
                  continue; /* do not clear the stack */

                case cf2_escSEAC:
                  if ( !font->isT1 )
                    FT_TRACE4(( " unknown op (12, %d)\n", op2 ));
                  else
                  {
                    FT_Error   error2;
                    CF2_Int    bchar_index, achar_index;
                    FT_Vector  left_bearing, advance;

#ifdef FT_CONFIG_OPTION_INCREMENTAL
                    T1_Face  face = (T1_Face)decoder->builder.face;
#endif
                    CF2_BufferRec  component;
                    CF2_Fixed      dummyWidth;

                    CF2_Int  achar = cf2_stack_popInt( opStack );
                    CF2_Int  bchar = cf2_stack_popInt( opStack );

                    FT_Pos  ady = cf2_stack_popFixed ( opStack );
                    FT_Pos  adx = cf2_stack_popFixed ( opStack );
                    FT_Pos  asb = cf2_stack_popFixed ( opStack );


                    FT_TRACE4(( " seac\n" ));

                    if ( doingSeac )
                    {
                      FT_ERROR(( " nested seac\n" ));
                      lastError = FT_THROW( Invalid_Glyph_Format );
                      goto exit;      /* nested seac */
                    }

                    if ( decoder->builder.metrics_only )
                    {
                      FT_ERROR(( " unexpected seac\n" ));
                      lastError = FT_THROW( Invalid_Glyph_Format );
                      goto exit;      /* unexpected seac */
                    }

                    /* `glyph_names' is set to 0 for CID fonts which do */
                    /* not include an encoding.  How can we deal with   */
                    /* these?                                           */
#ifdef FT_CONFIG_OPTION_INCREMENTAL
                    if ( decoder->glyph_names == 0                   &&
                         !face->root.internal->incremental_interface )
#else
                      if ( decoder->glyph_names == 0 )
#endif /* FT_CONFIG_OPTION_INCREMENTAL */
                      {
                        FT_ERROR((
                          "cf2_interpT2CharString: (Type 1 seac)"
                          " glyph names table not available in this font\n" ));
                        lastError = FT_THROW( Invalid_Glyph_Format );
                        goto exit;
                      }

                    /* seac weirdness */
                    adx += decoder->builder.left_bearing->x;

#ifdef FT_CONFIG_OPTION_INCREMENTAL
                    if ( face->root.internal->incremental_interface )
                    {
                      /* the caller must handle the font encoding also */
                      bchar_index = bchar;
                      achar_index = achar;
                    }
                    else
#endif
                    {
                      bchar_index = t1_lookup_glyph_by_stdcharcode_ps(
                                      decoder, bchar );
                      achar_index = t1_lookup_glyph_by_stdcharcode_ps(
                                      decoder, achar );
                    }

                    if ( bchar_index < 0 || achar_index < 0 )
                    {
                      FT_ERROR((
                        "cf2_interpT2CharString: (Type 1 seac)"
                        " invalid seac character code arguments\n" ));
                      lastError = FT_THROW( Invalid_Glyph_Format );
                      goto exit;
                    }

                    /* if we are trying to load a composite glyph, */
                    /* do not load the accent character and return */
                    /* the array of subglyphs.                     */
                    if ( decoder->builder.no_recurse )
                    {
                      FT_GlyphSlot    glyph  = (FT_GlyphSlot)decoder->builder.glyph;
                      FT_GlyphLoader  loader = glyph->internal->loader;
                      FT_SubGlyph     subg;


                      /* reallocate subglyph array if necessary */
                      error2 = FT_GlyphLoader_CheckSubGlyphs( loader, 2 );
                      if ( error2 )
                      {
                        lastError = error2; /* pass FreeType error through */
                        goto exit;
                      }

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
                      subg->arg1  = (FT_Int)FIXED_TO_INT( adx - asb );
                      subg->arg2  = (FT_Int)FIXED_TO_INT( ady );

                      /* set up remaining glyph fields */
                      glyph->num_subglyphs = 2;
                      glyph->subglyphs     = loader->base.subglyphs;
                      glyph->format        = FT_GLYPH_FORMAT_COMPOSITE;

                      loader->current.num_subglyphs = 2;

                      goto exit;
                    }

                    /* First load `bchar' in builder */
                    /* now load the unscaled outline */

                    /* prepare loader */
                    FT_GlyphLoader_Prepare( decoder->builder.loader );

                    error2 = cf2_getT1SeacComponent( decoder,
                                                     (FT_UInt)bchar_index,
                                                     &component );
                    if ( error2 )
                    {
                      lastError = error2; /* pass FreeType error through */
                      goto exit;
                    }

                    /* save the left bearing and width of the SEAC   */
                    /* glyph as they will be erased by the next load */

                    left_bearing = *decoder->builder.left_bearing;
                    advance      = *decoder->builder.advance;

                    cf2_interpT2CharString( font,
                                            &component,
                                            callbacks,
                                            translation,
                                            TRUE,
                                            0,
                                            0,
                                            &dummyWidth );
                    cf2_freeT1SeacComponent( decoder, &component );

                    /* If the SEAC glyph doesn't have a (H)SBW of its */
                    /* own use the values from the base glyph.        */

                    if ( !haveWidth )
                    {
                      left_bearing = *decoder->builder.left_bearing;
                      advance      = *decoder->builder.advance;
                    }

                    decoder->builder.left_bearing->x = 0;
                    decoder->builder.left_bearing->y = 0;

                    /* Now load `achar' on top of */
                    /* the base outline           */

                    error2 = cf2_getT1SeacComponent( decoder,
                                                     (FT_UInt)achar_index,
                                                     &component );
                    if ( error2 )
                    {
                      lastError = error2; /* pass FreeType error through */
                      goto exit;
                    }
                    cf2_interpT2CharString( font,
                                            &component,
                                            callbacks,
                                            translation,
                                            TRUE,
                                            adx - asb,
                                            ady,
                                            &dummyWidth );
                    cf2_freeT1SeacComponent( decoder, &component );

                    /* restore the left side bearing and advance width   */
                    /* of the SEAC glyph or base character (saved above) */

                    *decoder->builder.left_bearing = left_bearing;
                    *decoder->builder.advance      = advance;

                    goto exit;
                  }
                  break;

                case cf2_escSBW:
                  if ( !font->isT1 )
                    FT_TRACE4(( " unknown op (12, %d)\n", op2 ));
                  else
                  {
                    CF2_Fixed    lsb_x, lsb_y;
                    PS_Builder*  builder;


                    FT_TRACE4(( " sbw" ));

                    builder = &decoder->builder;

                    builder->advance->y = cf2_stack_popFixed( opStack );
                    builder->advance->x = cf2_stack_popFixed( opStack );

                    lsb_y = cf2_stack_popFixed( opStack );
                    lsb_x = cf2_stack_popFixed( opStack );

                    builder->left_bearing->x =
                      ADD_INT32( builder->left_bearing->x, lsb_x );
                    builder->left_bearing->y =
                      ADD_INT32( builder->left_bearing->y, lsb_y );

                    haveWidth = TRUE;

                    /* the `metrics_only' indicates that we only want */
                    /* to compute the glyph's metrics (lsb + advance  */
                    /* width), not load the  rest of it; so exit      */
                    /* immediately                                    */
                    if ( builder->metrics_only )
                      goto exit;

                    if ( initial_map_ready )
                    {
                      curX = ADD_INT32( curX, lsb_x );
                      curY = ADD_INT32( curY, lsb_y );
                    }
                  }
                  break;

                case cf2_escABS:
                  {
                    CF2_F16Dot16  arg;


                    FT_TRACE4(( " abs\n" ));

                    arg = cf2_stack_popFixed( opStack );

                    if ( arg < -CF2_FIXED_MAX )
                      cf2_stack_pushFixed( opStack, CF2_FIXED_MAX );
                    else
                      cf2_stack_pushFixed( opStack, FT_ABS( arg ) );
                  }
                  continue; /* do not clear the stack */

                case cf2_escADD:
                  {
                    CF2_F16Dot16  summand1;
                    CF2_F16Dot16  summand2;


                    FT_TRACE4(( " add\n" ));

                    summand2 = cf2_stack_popFixed( opStack );
                    summand1 = cf2_stack_popFixed( opStack );

                    cf2_stack_pushFixed( opStack,
                                         ADD_INT32( summand1,
                                                    summand2 ) );
                  }
                  continue; /* do not clear the stack */

                case cf2_escSUB:
                  {
                    CF2_F16Dot16  minuend;
                    CF2_F16Dot16  subtrahend;


                    FT_TRACE4(( " sub\n" ));

                    subtrahend = cf2_stack_popFixed( opStack );
                    minuend    = cf2_stack_popFixed( opStack );

                    cf2_stack_pushFixed( opStack,
                                         SUB_INT32( minuend, subtrahend ) );
                  }
                  continue; /* do not clear the stack */

                case cf2_escDIV:
                  {
                    CF2_F16Dot16  dividend;
                    CF2_F16Dot16  divisor;


                    FT_TRACE4(( " div\n" ));

                    if ( font->isT1 && large_int )
                    {
                      divisor  = (CF2_F16Dot16)cf2_stack_popInt( opStack );
                      dividend = (CF2_F16Dot16)cf2_stack_popInt( opStack );

                      large_int = FALSE;
                    }
                    else
                    {
                      divisor  = cf2_stack_popFixed( opStack );
                      dividend = cf2_stack_popFixed( opStack );
                    }

                    cf2_stack_pushFixed( opStack,
                                         FT_DivFix( dividend, divisor ) );

                  }
                  continue; /* do not clear the stack */

                case cf2_escNEG:
                  {
                    CF2_F16Dot16  arg;


                    FT_TRACE4(( " neg\n" ));

                    arg = cf2_stack_popFixed( opStack );

                    if ( arg < -CF2_FIXED_MAX )
                      cf2_stack_pushFixed( opStack, CF2_FIXED_MAX );
                    else
                      cf2_stack_pushFixed( opStack, -arg );
                  }
                  continue; /* do not clear the stack */

                case cf2_escEQ:
                  {
                    CF2_F16Dot16  arg1;
                    CF2_F16Dot16  arg2;


                    FT_TRACE4(( " eq\n" ));

                    arg2 = cf2_stack_popFixed( opStack );
                    arg1 = cf2_stack_popFixed( opStack );

                    cf2_stack_pushInt( opStack, arg1 == arg2 );
                  }
                  continue; /* do not clear the stack */

                case cf2_escCALLOTHERSUBR:
                  if ( !font->isT1 )
                    FT_TRACE4(( " unknown op (12, %d)\n", op2 ));
                  else
                  {
                    CF2_Int   subr_no;
                    CF2_Int   arg_cnt;
                    CF2_UInt  count;
                    CF2_UInt  opIdx = 0;


                    FT_TRACE4(( " callothersubr\n" ));

                    subr_no = cf2_stack_popInt( opStack );
                    arg_cnt = cf2_stack_popInt( opStack );

                    /********************************************************
                     *
                     * remove all operands to callothersubr from the stack
                     *
                     * for handled othersubrs, where we know the number of
                     * arguments, we increase the stack by the value of
                     * known_othersubr_result_cnt
                     *
                     * for unhandled othersubrs the following pops adjust
                     * the stack pointer as necessary
                     */

                    count = cf2_stack_count( opStack );
                    FT_ASSERT( (CF2_UInt)arg_cnt <= count );

                    opIdx += count - (CF2_UInt)arg_cnt;

                    known_othersubr_result_cnt = 0;
                    result_cnt                 = 0;

                    /* XXX TODO: The checks to `arg_count == <whatever>'   */
                    /* might not be correct; an othersubr expects a        */
                    /* certain number of operands on the PostScript stack  */
                    /* (as opposed to the T1 stack) but it doesn't have to */
                    /* put them there by itself; previous othersubrs might */
                    /* have left the operands there if they were not       */
                    /* followed by an appropriate number of pops           */
                    /*                                                     */
                    /* On the other hand, Adobe Reader 7.0.8 for Linux     */
                    /* doesn't accept a font that contains charstrings     */
                    /* like                                                */
                    /*                                                     */
                    /*     100 200 2 20 callothersubr                      */
                    /*     300 1 20 callothersubr pop                      */
                    /*                                                     */
                    /* Perhaps this is the reason why BuildCharArray       */
                    /* exists.                                             */

                    switch ( subr_no )
                    {
                    case 0:                     /* end flex feature */
                      if ( arg_cnt != 3 )
                        goto Unexpected_OtherSubr;

                      if ( initial_map_ready &&
                           ( !decoder->flex_state           ||
                             decoder->num_flex_vectors != 7 ) )
                      {
                        FT_ERROR(( "cf2_interpT2CharString (Type 1 mode):"
                                   " unexpected flex end\n" ));
                        lastError = FT_THROW( Invalid_Glyph_Format );
                        goto exit;
                      }

                      /* the two `results' are popped     */
                      /* by the following setcurrentpoint */
                      cf2_stack_pushFixed( opStack, curX );
                      cf2_stack_pushFixed( opStack, curY );
                      known_othersubr_result_cnt = 2;
                      break;

                    case 1:                     /* start flex feature */
                      if ( arg_cnt != 0 )
                        goto Unexpected_OtherSubr;

                      if ( !initial_map_ready )
                        break;

                      if ( ps_builder_check_points( &decoder->builder, 6 ) )
                        goto exit;

                      decoder->flex_state        = 1;
                      decoder->num_flex_vectors  = 0;
                      break;

                    case 2:                     /* add flex vectors */
                      {
                        FT_Int  idx;
                        FT_Int  idx2;


                        if ( arg_cnt != 0 )
                          goto Unexpected_OtherSubr;

                        if ( !initial_map_ready )
                          break;

                        if ( !decoder->flex_state )
                        {
                          FT_ERROR(( "cf2_interpT2CharString (Type 1 mode):"
                                     " missing flex start\n" ));
                          lastError = FT_THROW( Invalid_Glyph_Format );
                          goto exit;
                        }

                        /* note that we should not add a point for      */
                        /* index 0; this will move our current position */
                        /* to the flex point without adding any point   */
                        /* to the outline                               */
                        idx = decoder->num_flex_vectors++;
                        if ( idx > 0 && idx < 7 )
                        {
                          /* in malformed fonts it is possible to have    */
                          /* other opcodes in the middle of a flex (which */
                          /* don't increase `num_flex_vectors'); we thus  */
                          /* have to check whether we can add a point     */

                          if ( ps_builder_check_points( &decoder->builder,
                                                        1 ) )
                          {
                            lastError = FT_THROW( Invalid_Glyph_Format );
                            goto exit;
                          }

                          /* map: 1->2 2->4 3->6 4->2 5->4 6->6 */
                          idx2 = ( idx > 3 ? idx - 3 : idx ) * 2;

                          flexStore[idx2 - 2] = curX;
                          flexStore[idx2 - 1] = curY;

                          if ( idx == 3 || idx == 6 )
                            cf2_glyphpath_curveTo( &glyphPath,
                                                   flexStore[0],
                                                   flexStore[1],
                                                   flexStore[2],
                                                   flexStore[3],
                                                   flexStore[4],
                                                   flexStore[5] );
                        }
                      }
                      break;

                    case 3:                     /* change hints */
                      if ( arg_cnt != 1 )
                        goto Unexpected_OtherSubr;

                      if ( initial_map_ready )
                      {
                        /* do not clear hints if initial hintmap */
                        /* is not ready - we need to collate all */
                        cf2_arrstack_clear( &vStemHintArray );
                        cf2_arrstack_clear( &hStemHintArray );

                        cf2_hintmask_init( &hintMask, error );
                        hintMask.isValid = FALSE;
                        hintMask.isNew   = TRUE;
                      }

                      known_othersubr_result_cnt = 1;
                      break;

                    case 12:
                    case 13:
                      /* counter control hints, clear stack */
                      cf2_stack_clear( opStack );
                      break;

                    case 14:
                    case 15:
                    case 16:
                    case 17:
                    case 18:                    /* multiple masters */
                      {
                        PS_Blend  blend = decoder->blend;
                        FT_UInt   num_points, nn, mm;
                        CF2_UInt  delta;
                        CF2_UInt  values;


                        if ( !blend )
                        {
                          FT_ERROR((
                            "cf2_interpT2CharString:"
                            " unexpected multiple masters operator\n" ));
                          lastError = FT_THROW( Invalid_Glyph_Format );
                          goto exit;
                        }

                        num_points = (FT_UInt)subr_no - 13 +
                                       ( subr_no == 18 );
                        if ( arg_cnt != (FT_Int)( num_points *
                                                  blend->num_designs ) )
                        {
                          FT_ERROR((
                            "cf2_interpT2CharString:"
                            " incorrect number of multiple masters arguments\n" ));
                          lastError = FT_THROW( Invalid_Glyph_Format );
                          goto exit;
                        }

                        /* We want to compute                                */
                        /*                                                   */
                        /*   a0*w0 + a1*w1 + ... + ak*wk                     */
                        /*                                                   */
                        /* but we only have a0, a1-a0, a2-a0, ..., ak-a0.    */
                        /*                                                   */
                        /* However, given that w0 + w1 + ... + wk == 1, we   */
                        /* can rewrite it easily as                          */
                        /*                                                   */
                        /*   a0 + (a1-a0)*w1 + (a2-a0)*w2 + ... + (ak-a0)*wk */
                        /*                                                   */
                        /* where k == num_designs-1.                         */
                        /*                                                   */
                        /* I guess that's why it's written in this `compact' */
                        /* form.                                             */
                        /*                                                   */
                        delta  = opIdx + num_points;
                        values = opIdx;
                        for ( nn = 0; nn < num_points; nn++ )
                        {
                          CF2_Fixed  tmp = cf2_stack_getReal( opStack,
                                                              values );


                          for ( mm = 1; mm < blend->num_designs; mm++ )
                            tmp = ADD_INT32( tmp,
                                             FT_MulFix(
                                               cf2_stack_getReal( opStack,
                                                                  delta++ ),
                                               blend->weight_vector[mm] ) );

                          cf2_stack_setReal( opStack, values++, tmp );
                        }
                        cf2_stack_pop( opStack,
                                       (CF2_UInt)arg_cnt - num_points );

                        known_othersubr_result_cnt = (FT_Int)num_points;
                        break;
                      }

                    case 19:
                      /* <idx> 1 19 callothersubr                 */
                      /* ==> replace elements starting from index */
                      /*     cvi( <idx> ) of BuildCharArray with  */
                      /*     WeightVector                         */
                      {
                        FT_Int    idx;
                        PS_Blend  blend = decoder->blend;


                        if ( arg_cnt != 1 || !blend )
                          goto Unexpected_OtherSubr;

                        idx = cf2_stack_popInt( opStack );

                        if ( idx < 0                             ||
                             (FT_UInt)idx + blend->num_designs >
                               decoder->len_buildchar            )
                          goto Unexpected_OtherSubr;

                        ft_memcpy( &decoder->buildchar[idx],
                                   blend->weight_vector,
                                   blend->num_designs *
                                   sizeof ( blend->weight_vector[0] ) );
                      }
                      break;

                    case 20:
                      /* <arg1> <arg2> 2 20 callothersubr pop   */
                      /* ==> push <arg1> + <arg2> onto T1 stack */
                      {
                        CF2_F16Dot16  summand1;
                        CF2_F16Dot16  summand2;


                        if ( arg_cnt != 2 )
                          goto Unexpected_OtherSubr;

                        summand2 = cf2_stack_popFixed( opStack );
                        summand1 = cf2_stack_popFixed( opStack );

                        cf2_stack_pushFixed( opStack,
                                             ADD_INT32( summand1,
                                                        summand2 ) );
                        known_othersubr_result_cnt = 1;
                      }
                      break;

                    case 21:
                      /* <arg1> <arg2> 2 21 callothersubr pop   */
                      /* ==> push <arg1> - <arg2> onto T1 stack */
                      {
                        CF2_F16Dot16  minuend;
                        CF2_F16Dot16  subtrahend;


                        if ( arg_cnt != 2 )
                          goto Unexpected_OtherSubr;

                        subtrahend = cf2_stack_popFixed( opStack );
                        minuend    = cf2_stack_popFixed( opStack );

                        cf2_stack_pushFixed( opStack,
                                             SUB_INT32( minuend,
                                                        subtrahend ) );
                        known_othersubr_result_cnt = 1;
                      }
                      break;

                    case 22:
                      /* <arg1> <arg2> 2 22 callothersubr pop   */
                      /* ==> push <arg1> * <arg2> onto T1 stack */
                      {
                        CF2_F16Dot16  factor1;
                        CF2_F16Dot16  factor2;


                        if ( arg_cnt != 2 )
                          goto Unexpected_OtherSubr;

                        factor2 = cf2_stack_popFixed( opStack );
                        factor1 = cf2_stack_popFixed( opStack );

                        cf2_stack_pushFixed( opStack,
                                             FT_MulFix( factor1, factor2 ) );
                        known_othersubr_result_cnt = 1;
                      }
                      break;

                    case 23:
                      /* <arg1> <arg2> 2 23 callothersubr pop   */
                      /* ==> push <arg1> / <arg2> onto T1 stack */
                      {
                        CF2_F16Dot16  dividend;
                        CF2_F16Dot16  divisor;


                        if ( arg_cnt != 2 )
                          goto Unexpected_OtherSubr;

                        divisor  = cf2_stack_popFixed( opStack );
                        dividend = cf2_stack_popFixed( opStack );

                        if ( divisor == 0 )
                          goto Unexpected_OtherSubr;

                        cf2_stack_pushFixed( opStack,
                                             FT_DivFix( dividend,
                                                        divisor ) );
                        known_othersubr_result_cnt = 1;
                      }
                      break;

                    case 24:
                      /* <val> <idx> 2 24 callothersubr               */
                      /* ==> set BuildCharArray[cvi( <idx> )] = <val> */
                      {
                        CF2_Int   idx;
                        PS_Blend  blend = decoder->blend;


                        if ( arg_cnt != 2 || !blend )
                          goto Unexpected_OtherSubr;

                        idx = cf2_stack_popInt( opStack );

                        if ( idx < 0                                ||
                             (FT_UInt)idx >= decoder->len_buildchar )
                          goto Unexpected_OtherSubr;

                        decoder->buildchar[idx] =
                          cf2_stack_popFixed( opStack );
                      }
                      break;

                    case 25:
                      /* <idx> 1 25 callothersubr pop        */
                      /* ==> push BuildCharArray[cvi( idx )] */
                      /*     onto T1 stack                   */
                      {
                        CF2_Int   idx;
                        PS_Blend  blend = decoder->blend;


                        if ( arg_cnt != 1 || !blend )
                          goto Unexpected_OtherSubr;

                        idx = cf2_stack_popInt( opStack );

                        if ( idx < 0                                ||
                             (FT_UInt)idx >= decoder->len_buildchar )
                          goto Unexpected_OtherSubr;

                        cf2_stack_pushFixed( opStack,
                                             decoder->buildchar[idx] );
                        known_othersubr_result_cnt = 1;
                      }
                      break;

#if 0
                    case 26:
                      /* <val> mark <idx>                              */
                      /* ==> set BuildCharArray[cvi( <idx> )] = <val>, */
                      /*     leave mark on T1 stack                    */
                      /* <val> <idx>                                   */
                      /* ==> set BuildCharArray[cvi( <idx> )] = <val>  */
                      XXX which routine has left its mark on the
                      XXX (PostScript) stack?;
                      break;
#endif

                    case 27:
                      /* <res1> <res2> <val1> <val2> 4 27 callothersubr pop */
                      /* ==> push <res1> onto T1 stack if <val1> <= <val2>, */
                      /*     otherwise push <res2>                          */
                      {
                        CF2_F16Dot16  arg1;
                        CF2_F16Dot16  arg2;
                        CF2_F16Dot16  cond1;
                        CF2_F16Dot16  cond2;


                        if ( arg_cnt != 4 )
                          goto Unexpected_OtherSubr;

                        cond2 = cf2_stack_popFixed( opStack );
                        cond1 = cf2_stack_popFixed( opStack );
                        arg2  = cf2_stack_popFixed( opStack );
                        arg1  = cf2_stack_popFixed( opStack );

                        cf2_stack_pushFixed( opStack,
                                             cond1 <= cond2 ? arg1 : arg2 );
                        known_othersubr_result_cnt = 1;
                      }
                      break;

                    case 28:
                      /* 0 28 callothersubr pop                     */
                      /* ==> push random value from interval [0, 1) */
                      /*     onto stack                             */
                      {
                        CF2_F16Dot16  r;


                        if ( arg_cnt != 0 )
                          goto Unexpected_OtherSubr;

                        /* only use the lower 16 bits of `random'  */
                        /* to generate a number in the range (0;1] */
                        r = (CF2_F16Dot16)
                              ( ( decoder->current_subfont->random & 0xFFFF ) + 1 );

                        decoder->current_subfont->random =
                          cff_random( decoder->current_subfont->random );

                        cf2_stack_pushFixed( opStack, r );
                        known_othersubr_result_cnt = 1;
                      }
                      break;

                    default:
                      if ( arg_cnt >= 0 && subr_no >= 0 )
                      {
                        FT_Int  i;


                        FT_ERROR((
                          "cf2_interpT2CharString (Type 1 mode):"
                          " unknown othersubr [%d %d], wish me luck\n",
                          arg_cnt, subr_no ));

                        /* store the unused args        */
                        /* for this unhandled OtherSubr */

                        if ( arg_cnt > PS_STORAGE_SIZE )
                          arg_cnt = PS_STORAGE_SIZE;
                        result_cnt = arg_cnt;

                        for ( i = 1; i <= arg_cnt; i++ )
                          results[result_cnt - i] =
                            cf2_stack_popFixed( opStack );

                        break;
                      }
                      /* fall through */

                    Unexpected_OtherSubr:
                      FT_ERROR(( "cf2_interpT2CharString (Type 1 mode):"
                                 " invalid othersubr [%d %d]\n",
                                 arg_cnt, subr_no ));
                      lastError = FT_THROW( Invalid_Glyph_Format );
                      goto exit;
                    }
                  }
                  continue; /* do not clear the stack */

                case cf2_escPOP:
                  if ( !font->isT1 )
                    FT_TRACE4(( " unknown op (12, %d)\n", op2 ));
                  else
                  {
                    FT_TRACE4(( " pop" ));

                    if ( known_othersubr_result_cnt > 0 )
                    {
                      known_othersubr_result_cnt--;
                      /* ignore, we pushed the operands ourselves */
                      continue;
                    }

                    if ( result_cnt == 0 )
                    {
                      FT_ERROR(( "cf2_interpT2CharString (Type 1 mode):"
                                 " no more operands for othersubr\n" ));
                      lastError = FT_THROW( Invalid_Glyph_Format );
                      goto exit;
                    }

                    result_cnt--;
                    cf2_stack_pushFixed( opStack, results[result_cnt] );
                  }
                  continue; /* do not clear the stack */

                case cf2_escDROP:
                  FT_TRACE4(( " drop\n" ));

                  (void)cf2_stack_popFixed( opStack );
                  continue; /* do not clear the stack */

                case cf2_escPUT:
                  {
                    CF2_F16Dot16  val;
                    CF2_Int       idx;


                    FT_TRACE4(( " put\n" ));

                    idx = cf2_stack_popInt( opStack );
                    val = cf2_stack_popFixed( opStack );

                    if ( idx >= 0 && idx < CF2_STORAGE_SIZE )
                      storage[idx] = val;
                  }
                  continue; /* do not clear the stack */

                case cf2_escGET:
                  {
                    CF2_Int  idx;


                    FT_TRACE4(( " get\n" ));

                    idx = cf2_stack_popInt( opStack );

                    if ( idx >= 0 && idx < CF2_STORAGE_SIZE )
                      cf2_stack_pushFixed( opStack, storage[idx] );
                  }
                  continue; /* do not clear the stack */

                case cf2_escIFELSE:
                  {
                    CF2_F16Dot16  arg1;
                    CF2_F16Dot16  arg2;
                    CF2_F16Dot16  cond1;
                    CF2_F16Dot16  cond2;


                    FT_TRACE4(( " ifelse\n" ));

                    cond2 = cf2_stack_popFixed( opStack );
                    cond1 = cf2_stack_popFixed( opStack );
                    arg2  = cf2_stack_popFixed( opStack );
                    arg1  = cf2_stack_popFixed( opStack );

                    cf2_stack_pushFixed( opStack,
                                         cond1 <= cond2 ? arg1 : arg2 );
                  }
                  continue; /* do not clear the stack */

                case cf2_escRANDOM: /* in spec */
                  {
                    CF2_F16Dot16  r;


                    FT_TRACE4(( " random\n" ));

                    /* only use the lower 16 bits of `random'  */
                    /* to generate a number in the range (0;1] */
                    r = (CF2_F16Dot16)
                          ( ( decoder->current_subfont->random & 0xFFFF ) + 1 );

                    decoder->current_subfont->random =
                      cff_random( decoder->current_subfont->random );

                    cf2_stack_pushFixed( opStack, r );
                  }
                  continue; /* do not clear the stack */

                case cf2_escMUL:
                  {
                    CF2_F16Dot16  factor1;
                    CF2_F16Dot16  factor2;


                    FT_TRACE4(( " mul\n" ));

                    factor2 = cf2_stack_popFixed( opStack );
                    factor1 = cf2_stack_popFixed( opStack );

                    cf2_stack_pushFixed( opStack,
                                         FT_MulFix( factor1, factor2 ) );
                  }
                  continue; /* do not clear the stack */

                case cf2_escSQRT:
                  {
                    CF2_F16Dot16  arg;


                    FT_TRACE4(( " sqrt\n" ));

                    arg = cf2_stack_popFixed( opStack );
                    if ( arg > 0 )
                    {
                      /* use a start value that doesn't make */
                      /* the algorithm's addition overflow   */
                      FT_Fixed  root = arg < 10 ? arg : arg >> 1;
                      FT_Fixed  new_root;


                      /* Babylonian method */
                      for (;;)
                      {
                        new_root = ( root + FT_DivFix( arg, root ) + 1 ) >> 1;
                        if ( new_root == root )
                          break;
                        root = new_root;
                      }
                      arg = new_root;
                    }
                    else
                      arg = 0;

                    cf2_stack_pushFixed( opStack, arg );
                  }
                  continue; /* do not clear the stack */

                case cf2_escDUP:
                  {
                    CF2_F16Dot16  arg;


                    FT_TRACE4(( " dup\n" ));

                    arg = cf2_stack_popFixed( opStack );

                    cf2_stack_pushFixed( opStack, arg );
                    cf2_stack_pushFixed( opStack, arg );
                  }
                  continue; /* do not clear the stack */

                case cf2_escEXCH:
                  {
                    CF2_F16Dot16  arg1;
                    CF2_F16Dot16  arg2;


                    FT_TRACE4(( " exch\n" ));

                    arg2 = cf2_stack_popFixed( opStack );
                    arg1 = cf2_stack_popFixed( opStack );

                    cf2_stack_pushFixed( opStack, arg2 );
                    cf2_stack_pushFixed( opStack, arg1 );
                  }
                  continue; /* do not clear the stack */

                case cf2_escINDEX:
                  {
                    CF2_Int   idx;
                    CF2_UInt  size;


                    FT_TRACE4(( " index\n" ));

                    idx  = cf2_stack_popInt( opStack );
                    size = cf2_stack_count( opStack );

                    if ( size > 0 )
                    {
                      /* for `cf2_stack_getReal',   */
                      /* index 0 is bottom of stack */
                      CF2_UInt  gr_idx;


                      if ( idx < 0 )
                        gr_idx = size - 1;
                      else if ( (CF2_UInt)idx >= size )
                        gr_idx = 0;
                      else
                        gr_idx = size - 1 - (CF2_UInt)idx;

                      cf2_stack_pushFixed( opStack,
                                           cf2_stack_getReal( opStack,
                                                              gr_idx ) );
                    }
                  }
                  continue; /* do not clear the stack */

                case cf2_escROLL:
                  {
                    CF2_Int  idx;
                    CF2_Int  count;


                    FT_TRACE4(( " roll\n" ));

                    idx   = cf2_stack_popInt( opStack );
                    count = cf2_stack_popInt( opStack );

                    cf2_stack_roll( opStack, count, idx );
                  }
                  continue; /* do not clear the stack */

                case cf2_escSETCURRENTPT:
                  if ( !font->isT1 )
                    FT_TRACE4(( " unknown op (12, %d)\n", op2 ));
                  else
                  {
                    FT_TRACE4(( " setcurrentpoint" ));

                    if ( !initial_map_ready )
                      break;

                    /* From the T1 specification, section 6.4:            */
                    /*                                                    */
                    /*   The setcurrentpoint command is used only in      */
                    /*   conjunction with results from OtherSubrs         */
                    /*   procedures.                                      */

                    /* known_othersubr_result_cnt != 0 is already handled */
                    /* above.                                             */

                    /* Note, however, that both Ghostscript and Adobe     */
                    /* Distiller handle this situation by silently        */
                    /* ignoring the inappropriate `setcurrentpoint'       */
                    /* instruction.  So we do the same.                   */
#if 0

                    if ( decoder->flex_state != 1 )
                    {
                      FT_ERROR(( "cf2_interpT2CharString:"
                                 " unexpected `setcurrentpoint'\n" ));
                      goto Syntax_Error;
                    }
                    else
                      ...
#endif

                    curY = cf2_stack_popFixed( opStack );
                    curX = cf2_stack_popFixed( opStack );

                    decoder->flex_state = 0;
                  }
                  break;

                } /* end of 2nd switch checking op2 */
              }
            }
          } /* end of 1st switch checking op2 */
        } /* case cf2_cmdESC */

        break;

      case cf2_cmdHSBW:
        if ( !font->isT1 )
          FT_TRACE4(( " unknown op (%d)\n", op1 ));
        else
        {
          CF2_Fixed    lsb_x;
          PS_Builder*  builder;


          FT_TRACE4(( " hsbw\n" ));

          builder = &decoder->builder;

          builder->advance->x = cf2_stack_popFixed( opStack );
          builder->advance->y = 0;

          lsb_x = cf2_stack_popFixed( opStack );

          builder->left_bearing->x = ADD_INT32( builder->left_bearing->x,
                                                lsb_x );

          haveWidth = TRUE;

          /* the `metrics_only' indicates that we only want to compute */
          /* the glyph's metrics (lsb + advance width), not load the   */
          /* rest of it; so exit immediately                           */
          if ( builder->metrics_only )
            goto exit;

          if ( initial_map_ready )
            curX = ADD_INT32( curX, lsb_x );
        }
        break;

      case cf2_cmdENDCHAR:
        FT_TRACE4(( " endchar\n" ));

        if ( font->isT1 && !initial_map_ready )
        {
          FT_TRACE5(( "cf2_interpT2CharString (Type 1 mode): "
                      "Build initial hintmap, rewinding...\n" ));

          /* trigger initial hintmap build */
          cf2_glyphpath_moveTo( &glyphPath, curX, curY );

          initial_map_ready = TRUE;

          /* change hints routine - clear for rewind */
          cf2_arrstack_clear( &vStemHintArray );
          cf2_arrstack_clear( &hStemHintArray );

          cf2_hintmask_init( &hintMask, error );
          hintMask.isValid = FALSE;
          hintMask.isNew   = TRUE;

          /* rewind charstring */
          /* some charstrings use endchar from a final subroutine call */
          /* without returning, detect these and exit to the top level */
          /* charstring                                                */
          while ( charstringIndex > 0 )
          {
            FT_TRACE4(( " return (leaving level %d)\n", charstringIndex ));

            /* restore position in previous charstring */
            charstring = (CF2_Buffer)
                           cf2_arrstack_getPointer(
                             &subrStack,
                             (CF2_UInt)--charstringIndex );
          }
          charstring->ptr = charstring->start;

          break;
        }

        if ( cf2_stack_count( opStack ) == 1 ||
             cf2_stack_count( opStack ) == 5 )
        {
          if ( !haveWidth )
            *width = ADD_INT32( cf2_stack_getReal( opStack, 0 ),
                                nominalWidthX );
        }

        /* width is defined or default after this */
        haveWidth = TRUE;

        if ( decoder->width_only )
          goto exit;

        /* close path if still open */
        cf2_glyphpath_closeOpenPath( &glyphPath );

        /* disable seac for CFF2 and Type1        */
        /* (charstring ending with args on stack) */
        if ( !font->isCFF2 && !font->isT1 && cf2_stack_count( opStack ) > 1 )
        {
          /* must be either 4 or 5 --                       */
          /* this is a (deprecated) implied `seac' operator */

          CF2_Int        achar;
          CF2_Int        bchar;
          CF2_BufferRec  component;
          CF2_Fixed      dummyWidth;   /* ignore component width */
          FT_Error       error2;


          if ( doingSeac )
          {
            lastError = FT_THROW( Invalid_Glyph_Format );
            goto exit;      /* nested seac */
          }

          achar = cf2_stack_popInt( opStack );
          bchar = cf2_stack_popInt( opStack );

          curY = cf2_stack_popFixed( opStack );
          curX = cf2_stack_popFixed( opStack );

          error2 = cf2_getSeacComponent( decoder, achar, &component );
          if ( error2 )
          {
            lastError = error2;      /* pass FreeType error through */
            goto exit;
          }
          cf2_interpT2CharString( font,
                                  &component,
                                  callbacks,
                                  translation,
                                  TRUE,
                                  curX,
                                  curY,
                                  &dummyWidth );
          cf2_freeSeacComponent( decoder, &component );

          error2 = cf2_getSeacComponent( decoder, bchar, &component );
          if ( error2 )
          {
            lastError = error2;      /* pass FreeType error through */
            goto exit;
          }
          cf2_interpT2CharString( font,
                                  &component,
                                  callbacks,
                                  translation,
                                  TRUE,
                                  0,
                                  0,
                                  &dummyWidth );
          cf2_freeSeacComponent( decoder, &component );
        }
        goto exit;

      case cf2_cmdCNTRMASK:
      case cf2_cmdHINTMASK:
        /* the final \n in the tracing message gets added in      */
        /* `cf2_hintmask_read' (which also traces the mask bytes) */
        FT_TRACE4(( "%s", op1 == cf2_cmdCNTRMASK ? " cntrmask" : " hintmask" ));

        /* never add hints after the mask is computed */
        if ( cf2_stack_count( opStack ) > 1    &&
             cf2_hintmask_isValid( &hintMask ) )
        {
          FT_TRACE4(( "cf2_interpT2CharString: invalid hint mask\n" ));
          break;
        }

        /* if there are arguments on the stack, there this is an */
        /* implied cf2_cmdVSTEMHM                                */
        cf2_doStems( font,
                     opStack,
                     &vStemHintArray,
                     width,
                     &haveWidth,
                     0 );

        if ( decoder->width_only )
          goto exit;

        if ( op1 == cf2_cmdHINTMASK )
        {
          /* consume the hint mask bytes which follow the operator */
          cf2_hintmask_read( &hintMask,
                             charstring,
                             cf2_arrstack_size( &hStemHintArray ) +
                               cf2_arrstack_size( &vStemHintArray ) );
        }
        else
        {
          /*
           * Consume the counter mask bytes which follow the operator:
           * Build a temporary hint map, just to place and lock those
           * stems participating in the counter mask.  These are most
           * likely the dominant hstems, and are grouped together in a
           * few counter groups, not necessarily in correspondence
           * with the hint groups.  This reduces the chances of
           * conflicts between hstems that are initially placed in
           * separate hint groups and then brought together.  The
           * positions are copied back to `hStemHintArray', so we can
           * discard `counterMask' and `counterHintMap'.
           *
           */
          CF2_HintMapRec   counterHintMap;
          CF2_HintMaskRec  counterMask;


          cf2_hintmap_init( &counterHintMap,
                            font,
                            &glyphPath.initialHintMap,
                            &glyphPath.hintMoves,
                            scaleY );
          cf2_hintmask_init( &counterMask, error );

          cf2_hintmask_read( &counterMask,
                             charstring,
                             cf2_arrstack_size( &hStemHintArray ) +
                               cf2_arrstack_size( &vStemHintArray ) );
          cf2_hintmap_build( &counterHintMap,
                             &hStemHintArray,
                             &vStemHintArray,
                             &counterMask,
                             0,
                             FALSE );
        }
        break;

      case cf2_cmdRMOVETO:
        FT_TRACE4(( " rmoveto\n" ));

        if ( font->isT1 && !decoder->flex_state && !haveWidth )
          FT_ERROR(( "cf2_interpT2CharString (Type 1 mode):"
                     " No width. Use hsbw/sbw as first op\n" ));

        if ( cf2_stack_count( opStack ) > 2 && !haveWidth )
          *width = ADD_INT32( cf2_stack_getReal( opStack, 0 ),
                              nominalWidthX );

        /* width is defined or default after this */
        haveWidth = TRUE;

        if ( decoder->width_only )
          goto exit;

        curY = ADD_INT32( curY, cf2_stack_popFixed( opStack ) );
        curX = ADD_INT32( curX, cf2_stack_popFixed( opStack ) );

        if ( !decoder->flex_state )
          cf2_glyphpath_moveTo( &glyphPath, curX, curY );

        break;

      case cf2_cmdHMOVETO:
        FT_TRACE4(( " hmoveto\n" ));

        if ( font->isT1 && !decoder->flex_state && !haveWidth )
          FT_ERROR(( "cf2_interpT2CharString (Type 1 mode):"
                     " No width. Use hsbw/sbw as first op\n" ));

        if ( cf2_stack_count( opStack ) > 1 && !haveWidth )
          *width = ADD_INT32( cf2_stack_getReal( opStack, 0 ),
                              nominalWidthX );

        /* width is defined or default after this */
        haveWidth = TRUE;

        if ( decoder->width_only )
          goto exit;

        curX = ADD_INT32( curX, cf2_stack_popFixed( opStack ) );

        if ( !decoder->flex_state )
          cf2_glyphpath_moveTo( &glyphPath, curX, curY );

        break;

      case cf2_cmdRLINECURVE:
        {
          CF2_UInt  count = cf2_stack_count( opStack );
          CF2_UInt  idx   = 0;


          FT_TRACE4(( " rlinecurve\n" ));

          while ( idx + 6 < count )
          {
            curX = ADD_INT32( curX, cf2_stack_getReal( opStack,
                                                       idx + 0 ) );
            curY = ADD_INT32( curY, cf2_stack_getReal( opStack,
                                                       idx + 1 ) );

            cf2_glyphpath_lineTo( &glyphPath, curX, curY );
            idx += 2;
          }

          while ( idx < count )
          {
            CF2_Fixed  x1, y1, x2, y2, x3, y3;


            x1 = ADD_INT32( cf2_stack_getReal( opStack, idx + 0 ), curX );
            y1 = ADD_INT32( cf2_stack_getReal( opStack, idx + 1 ), curY );
            x2 = ADD_INT32( cf2_stack_getReal( opStack, idx + 2 ), x1 );
            y2 = ADD_INT32( cf2_stack_getReal( opStack, idx + 3 ), y1 );
            x3 = ADD_INT32( cf2_stack_getReal( opStack, idx + 4 ), x2 );
            y3 = ADD_INT32( cf2_stack_getReal( opStack, idx + 5 ), y2 );

            cf2_glyphpath_curveTo( &glyphPath, x1, y1, x2, y2, x3, y3 );

            curX  = x3;
            curY  = y3;
            idx  += 6;
          }

          cf2_stack_clear( opStack );
        }
        continue; /* no need to clear stack again */

      case cf2_cmdVVCURVETO:
        {
          CF2_UInt  count, count1 = cf2_stack_count( opStack );
          CF2_UInt  idx = 0;


          /* if `cf2_stack_count' isn't of the form 4n or 4n+1, */
          /* we enforce it by clearing the second bit           */
          /* (and sorting the stack indexing to suit)           */
          count = count1 & ~2U;
          idx  += count1 - count;

          FT_TRACE4(( " vvcurveto\n" ));

          while ( idx < count )
          {
            CF2_Fixed  x1, y1, x2, y2, x3, y3;


            if ( ( count - idx ) & 1 )
            {
              x1 = ADD_INT32( cf2_stack_getReal( opStack, idx ), curX );

              idx++;
            }
            else
              x1 = curX;

            y1 = ADD_INT32( cf2_stack_getReal( opStack, idx + 0 ), curY );
            x2 = ADD_INT32( cf2_stack_getReal( opStack, idx + 1 ), x1 );
            y2 = ADD_INT32( cf2_stack_getReal( opStack, idx + 2 ), y1 );
            x3 = x2;
            y3 = ADD_INT32( cf2_stack_getReal( opStack, idx + 3 ), y2 );

            cf2_glyphpath_curveTo( &glyphPath, x1, y1, x2, y2, x3, y3 );

            curX  = x3;
            curY  = y3;
            idx  += 4;
          }

          cf2_stack_clear( opStack );
        }
        continue; /* no need to clear stack again */

      case cf2_cmdHHCURVETO:
        {
          CF2_UInt  count, count1 = cf2_stack_count( opStack );
          CF2_UInt  idx = 0;


          /* if `cf2_stack_count' isn't of the form 4n or 4n+1, */
          /* we enforce it by clearing the second bit           */
          /* (and sorting the stack indexing to suit)           */
          count = count1 & ~2U;
          idx  += count1 - count;

          FT_TRACE4(( " hhcurveto\n" ));

          while ( idx < count )
          {
            CF2_Fixed  x1, y1, x2, y2, x3, y3;


            if ( ( count - idx ) & 1 )
            {
              y1 = ADD_INT32( cf2_stack_getReal( opStack, idx ), curY );

              idx++;
            }
            else
              y1 = curY;

            x1 = ADD_INT32( cf2_stack_getReal( opStack, idx + 0 ), curX );
            x2 = ADD_INT32( cf2_stack_getReal( opStack, idx + 1 ), x1 );
            y2 = ADD_INT32( cf2_stack_getReal( opStack, idx + 2 ), y1 );
            x3 = ADD_INT32( cf2_stack_getReal( opStack, idx + 3 ), x2 );
            y3 = y2;

            cf2_glyphpath_curveTo( &glyphPath, x1, y1, x2, y2, x3, y3 );

            curX  = x3;
            curY  = y3;
            idx  += 4;
          }

          cf2_stack_clear( opStack );
        }
        continue; /* no need to clear stack again */

      case cf2_cmdVHCURVETO:
      case cf2_cmdHVCURVETO:
        {
          CF2_UInt  count, count1 = cf2_stack_count( opStack );
          CF2_UInt  idx = 0;

          FT_Bool  alternate = FT_BOOL( op1 == cf2_cmdHVCURVETO );


          /* if `cf2_stack_count' isn't of the form 8n, 8n+1, */
          /* 8n+4, or 8n+5, we enforce it by clearing the     */
          /* second bit                                       */
          /* (and sorting the stack indexing to suit)         */
          count = count1 & ~2U;
          idx  += count1 - count;

          FT_TRACE4(( "%s\n", alternate ? " hvcurveto" : " vhcurveto" ));

          while ( idx < count )
          {
            CF2_Fixed x1, x2, x3, y1, y2, y3;


            if ( alternate )
            {
              x1 = ADD_INT32( cf2_stack_getReal( opStack, idx + 0 ), curX );
              y1 = curY;
              x2 = ADD_INT32( cf2_stack_getReal( opStack, idx + 1 ), x1 );
              y2 = ADD_INT32( cf2_stack_getReal( opStack, idx + 2 ), y1 );
              y3 = ADD_INT32( cf2_stack_getReal( opStack, idx + 3 ), y2 );

              if ( count - idx == 5 )
              {
                x3 = ADD_INT32( cf2_stack_getReal( opStack, idx + 4 ), x2 );

                idx++;
              }
              else
                x3 = x2;

              alternate = FALSE;
            }
            else
            {
              x1 = curX;
              y1 = ADD_INT32( cf2_stack_getReal( opStack, idx + 0 ), curY );
              x2 = ADD_INT32( cf2_stack_getReal( opStack, idx + 1 ), x1 );
              y2 = ADD_INT32( cf2_stack_getReal( opStack, idx + 2 ), y1 );
              x3 = ADD_INT32( cf2_stack_getReal( opStack, idx + 3 ), x2 );

              if ( count - idx == 5 )
              {
                y3 = ADD_INT32( cf2_stack_getReal( opStack, idx + 4 ), y2 );

                idx++;
              }
              else
                y3 = y2;

              alternate = TRUE;
            }

            cf2_glyphpath_curveTo( &glyphPath, x1, y1, x2, y2, x3, y3 );

            curX  = x3;
            curY  = y3;
            idx  += 4;
          }

          cf2_stack_clear( opStack );
        }
        continue;     /* no need to clear stack again */

      case cf2_cmdEXTENDEDNMBR:
        {
          CF2_Int  v;

          CF2_Int  byte1 = cf2_buf_readByte( charstring );
          CF2_Int  byte2 = cf2_buf_readByte( charstring );


          v = (FT_Short)( ( byte1 << 8 ) |
                            byte2        );

          FT_TRACE4(( " %d", v ));

          cf2_stack_pushInt( opStack, v );
        }
        continue;

      default:
        /* numbers */
        {
          if ( /* op1 >= 32 && */ op1 <= 246 )
          {
            CF2_Int  v;


            v = op1 - 139;

            FT_TRACE4(( " %d", v ));

            /* -107 .. 107 */
            cf2_stack_pushInt( opStack, v );
          }

          else if ( /* op1 >= 247 && */ op1 <= 250 )
          {
            CF2_Int  v;


            v  = op1;
            v -= 247;
            v *= 256;
            v += cf2_buf_readByte( charstring );
            v += 108;

            FT_TRACE4(( " %d", v ));

            /* 108 .. 1131 */
            cf2_stack_pushInt( opStack, v );
          }

          else if ( /* op1 >= 251 && */ op1 <= 254 )
          {
            CF2_Int  v;


            v  = op1;
            v -= 251;
            v *= 256;
            v += cf2_buf_readByte( charstring );
            v  = -v - 108;

            FT_TRACE4(( " %d", v ));

            /* -1131 .. -108 */
            cf2_stack_pushInt( opStack, v );
          }

          else /* op1 == 255 */
          {
            CF2_Fixed  v;

            FT_UInt32  byte1 = (FT_UInt32)cf2_buf_readByte( charstring );
            FT_UInt32  byte2 = (FT_UInt32)cf2_buf_readByte( charstring );
            FT_UInt32  byte3 = (FT_UInt32)cf2_buf_readByte( charstring );
            FT_UInt32  byte4 = (FT_UInt32)cf2_buf_readByte( charstring );


            v = (CF2_Fixed)( ( byte1 << 24 ) |
                             ( byte2 << 16 ) |
                             ( byte3 <<  8 ) |
                               byte4         );

            /*
             * For Type 1:
             *
             * According to the specification, values > 32000 or < -32000
             * must be followed by a `div' operator to make the result be
             * in the range [-32000;32000].  We expect that the second
             * argument of `div' is not a large number.  Additionally, we
             * don't handle stuff like `<large1> <large2> <num> div <num>
             * div' or <large1> <large2> <num> div div'.  This is probably
             * not allowed anyway.
             *
             * <large> <num> <num>+ div is not checked but should not be
             * allowed as the large value remains untouched.
             *
             */
            if ( font->isT1 )
            {
              if ( v > 32000 || v < -32000 )
              {
                if ( large_int )
                  FT_ERROR(( "cf2_interpT2CharString (Type 1 mode):"
                             " no `div' after large integer\n" ));
                else
                  large_int = TRUE;
              }

              FT_TRACE4(( " %d", v ));

              cf2_stack_pushInt( opStack, (CF2_Int)v );
            }
            else
            {
              FT_TRACE4(( " %.5fF", v / 65536.0 ));

              cf2_stack_pushFixed( opStack, v );
            }
          }
        }
        continue;   /* don't clear stack */

      } /* end of switch statement checking `op1' */

      cf2_stack_clear( opStack );

    } /* end of main interpreter loop */

    /* we get here if the charstring ends without cf2_cmdENDCHAR */
    FT_TRACE4(( "cf2_interpT2CharString:"
                "  charstring ends without ENDCHAR\n" ));

  exit:
    /* check whether last error seen is also the first one */
    cf2_setError( error, lastError );

    if ( *error )
      FT_TRACE4(( "charstring error %d\n", *error ));

    /* free resources from objects we've used */
    cf2_glyphpath_finalize( &glyphPath );
    cf2_arrstack_finalize( &vStemHintArray );
    cf2_arrstack_finalize( &hStemHintArray );
    cf2_arrstack_finalize( &subrStack );
    cf2_stack_free( opStack );

    FT_TRACE4(( "\n" ));

    return;
  }


/* END */
