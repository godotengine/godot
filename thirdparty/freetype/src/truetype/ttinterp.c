/****************************************************************************
 *
 * ttinterp.c
 *
 *   TrueType bytecode interpreter (body).
 *
 * Copyright (C) 1996-2022 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


/* Greg Hitchcock from Microsoft has helped a lot in resolving unclear */
/* issues; many thanks!                                                */


#include <freetype/internal/ftdebug.h>
#include <freetype/internal/ftcalc.h>
#include <freetype/fttrigon.h>
#include <freetype/ftsystem.h>
#include <freetype/ftdriver.h>
#include <freetype/ftmm.h>

#include "ttinterp.h"
#include "tterrors.h"
#include "ttsubpix.h"
#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT
#include "ttgxvar.h"
#endif


#ifdef TT_USE_BYTECODE_INTERPRETER


  /**************************************************************************
   *
   * The macro FT_COMPONENT is used in trace mode.  It is an implicit
   * parameter of the FT_TRACE() and FT_ERROR() macros, used to print/log
   * messages during execution.
   */
#undef  FT_COMPONENT
#define FT_COMPONENT  ttinterp


#define NO_SUBPIXEL_HINTING                                                  \
          ( ((TT_Driver)FT_FACE_DRIVER( exc->face ))->interpreter_version == \
            TT_INTERPRETER_VERSION_35 )

#ifdef TT_SUPPORT_SUBPIXEL_HINTING_INFINALITY
#define SUBPIXEL_HINTING_INFINALITY                                          \
          ( ((TT_Driver)FT_FACE_DRIVER( exc->face ))->interpreter_version == \
            TT_INTERPRETER_VERSION_38 )
#endif

#ifdef TT_SUPPORT_SUBPIXEL_HINTING_MINIMAL
#define SUBPIXEL_HINTING_MINIMAL                                             \
          ( ((TT_Driver)FT_FACE_DRIVER( exc->face ))->interpreter_version == \
            TT_INTERPRETER_VERSION_40 )
#endif

#define PROJECT( v1, v2 )                                   \
          exc->func_project( exc,                           \
                             SUB_LONG( (v1)->x, (v2)->x ),  \
                             SUB_LONG( (v1)->y, (v2)->y ) )

#define DUALPROJ( v1, v2 )                                   \
          exc->func_dualproj( exc,                           \
                              SUB_LONG( (v1)->x, (v2)->x ),  \
                              SUB_LONG( (v1)->y, (v2)->y ) )

#define FAST_PROJECT( v )                          \
          exc->func_project( exc, (v)->x, (v)->y )

#define FAST_DUALPROJ( v )                          \
          exc->func_dualproj( exc, (v)->x, (v)->y )


  /**************************************************************************
   *
   * Two simple bounds-checking macros.
   */
#define BOUNDS( x, n )   ( (FT_UInt)(x)  >= (FT_UInt)(n)  )
#define BOUNDSL( x, n )  ( (FT_ULong)(x) >= (FT_ULong)(n) )


#undef  SUCCESS
#define SUCCESS  0

#undef  FAILURE
#define FAILURE  1


  /**************************************************************************
   *
   *                       CODERANGE FUNCTIONS
   *
   */


  /**************************************************************************
   *
   * @Function:
   *   TT_Goto_CodeRange
   *
   * @Description:
   *   Switches to a new code range (updates the code related elements in
   *   `exec', and `IP').
   *
   * @Input:
   *   range ::
   *     The new execution code range.
   *
   *   IP ::
   *     The new IP in the new code range.
   *
   * @InOut:
   *   exec ::
   *     The target execution context.
   */
  FT_LOCAL_DEF( void )
  TT_Goto_CodeRange( TT_ExecContext  exec,
                     FT_Int          range,
                     FT_Long         IP )
  {
    TT_CodeRange*  coderange;


    FT_ASSERT( range >= 1 && range <= 3 );

    coderange = &exec->codeRangeTable[range - 1];

    FT_ASSERT( coderange->base );

    /* NOTE: Because the last instruction of a program may be a CALL */
    /*       which will return to the first byte *after* the code    */
    /*       range, we test for IP <= Size instead of IP < Size.     */
    /*                                                               */
    FT_ASSERT( IP <= coderange->size );

    exec->code     = coderange->base;
    exec->codeSize = coderange->size;
    exec->IP       = IP;
    exec->curRange = range;
  }


  /**************************************************************************
   *
   * @Function:
   *   TT_Set_CodeRange
   *
   * @Description:
   *   Sets a code range.
   *
   * @Input:
   *   range ::
   *     The code range index.
   *
   *   base ::
   *     The new code base.
   *
   *   length ::
   *     The range size in bytes.
   *
   * @InOut:
   *   exec ::
   *     The target execution context.
   */
  FT_LOCAL_DEF( void )
  TT_Set_CodeRange( TT_ExecContext  exec,
                    FT_Int          range,
                    void*           base,
                    FT_Long         length )
  {
    FT_ASSERT( range >= 1 && range <= 3 );

    exec->codeRangeTable[range - 1].base = (FT_Byte*)base;
    exec->codeRangeTable[range - 1].size = length;
  }


  /**************************************************************************
   *
   * @Function:
   *   TT_Clear_CodeRange
   *
   * @Description:
   *   Clears a code range.
   *
   * @Input:
   *   range ::
   *     The code range index.
   *
   * @InOut:
   *   exec ::
   *     The target execution context.
   */
  FT_LOCAL_DEF( void )
  TT_Clear_CodeRange( TT_ExecContext  exec,
                      FT_Int          range )
  {
    FT_ASSERT( range >= 1 && range <= 3 );

    exec->codeRangeTable[range - 1].base = NULL;
    exec->codeRangeTable[range - 1].size = 0;
  }


  /**************************************************************************
   *
   *                  EXECUTION CONTEXT ROUTINES
   *
   */


  /**************************************************************************
   *
   * @Function:
   *   TT_Done_Context
   *
   * @Description:
   *   Destroys a given context.
   *
   * @Input:
   *   exec ::
   *     A handle to the target execution context.
   *
   *   memory ::
   *     A handle to the parent memory object.
   *
   * @Note:
   *   Only the glyph loader and debugger should call this function.
   */
  FT_LOCAL_DEF( void )
  TT_Done_Context( TT_ExecContext  exec )
  {
    FT_Memory  memory = exec->memory;


    /* points zone */
    exec->maxPoints   = 0;
    exec->maxContours = 0;

    /* free stack */
    FT_FREE( exec->stack );
    exec->stackSize = 0;

    /* free glyf cvt working area */
    FT_FREE( exec->glyfCvt );
    exec->glyfCvtSize = 0;

    /* free glyf storage working area */
    FT_FREE( exec->glyfStorage );
    exec->glyfStoreSize = 0;

    /* free call stack */
    FT_FREE( exec->callStack );
    exec->callSize = 0;
    exec->callTop  = 0;

    /* free glyph code range */
    FT_FREE( exec->glyphIns );
    exec->glyphSize = 0;

    exec->size = NULL;
    exec->face = NULL;

    FT_FREE( exec );
  }


  /**************************************************************************
   *
   * @Function:
   *   Update_Max
   *
   * @Description:
   *   Checks the size of a buffer and reallocates it if necessary.
   *
   * @Input:
   *   memory ::
   *     A handle to the parent memory object.
   *
   *   multiplier ::
   *     The size in bytes of each element in the buffer.
   *
   *   new_max ::
   *     The new capacity (size) of the buffer.
   *
   * @InOut:
   *   size ::
   *     The address of the buffer's current size expressed
   *     in elements.
   *
   *   buff ::
   *     The address of the buffer base pointer.
   *
   * @Return:
   *   FreeType error code.  0 means success.
   */
  FT_LOCAL_DEF( FT_Error )
  Update_Max( FT_Memory  memory,
              FT_ULong*  size,
              FT_ULong   multiplier,
              void*      _pbuff,
              FT_ULong   new_max )
  {
    FT_Error  error;
    void**    pbuff = (void**)_pbuff;


    if ( *size < new_max )
    {
      if ( FT_QREALLOC( *pbuff, *size * multiplier, new_max * multiplier ) )
        return error;
      *size = new_max;
    }

    return FT_Err_Ok;
  }


  /**************************************************************************
   *
   * @Function:
   *   TT_Load_Context
   *
   * @Description:
   *   Prepare an execution context for glyph hinting.
   *
   * @Input:
   *   face ::
   *     A handle to the source face object.
   *
   *   size ::
   *     A handle to the source size object.
   *
   * @InOut:
   *   exec ::
   *     A handle to the target execution context.
   *
   * @Return:
   *   FreeType error code.  0 means success.
   *
   * @Note:
   *   Only the glyph loader and debugger should call this function.
   *
   *   Note that not all members of `TT_ExecContext` get initialized.
   */
  FT_LOCAL_DEF( FT_Error )
  TT_Load_Context( TT_ExecContext  exec,
                   TT_Face         face,
                   TT_Size         size )
  {
    FT_Int          i;
    FT_ULong        tmp;
    TT_MaxProfile*  maxp;
    FT_Error        error;


    exec->face = face;
    maxp       = &face->max_profile;
    exec->size = size;

    if ( size )
    {
      exec->numFDefs   = size->num_function_defs;
      exec->maxFDefs   = size->max_function_defs;
      exec->numIDefs   = size->num_instruction_defs;
      exec->maxIDefs   = size->max_instruction_defs;
      exec->FDefs      = size->function_defs;
      exec->IDefs      = size->instruction_defs;
      exec->pointSize  = size->point_size;
      exec->tt_metrics = size->ttmetrics;
      exec->metrics    = *size->metrics;

      exec->maxFunc    = size->max_func;
      exec->maxIns     = size->max_ins;

      for ( i = 0; i < TT_MAX_CODE_RANGES; i++ )
        exec->codeRangeTable[i] = size->codeRangeTable[i];

      /* set graphics state */
      exec->GS = size->GS;

      exec->cvtSize = size->cvt_size;
      exec->cvt     = size->cvt;

      exec->storeSize = size->storage_size;
      exec->storage   = size->storage;

      exec->twilight  = size->twilight;

      /* In case of multi-threading it can happen that the old size object */
      /* no longer exists, thus we must clear all glyph zone references.   */
      FT_ZERO( &exec->zp0 );
      exec->zp1 = exec->zp0;
      exec->zp2 = exec->zp0;
    }

    /* XXX: We reserve a little more elements on the stack to deal safely */
    /*      with broken fonts like arialbs, courbs, timesbs, etc.         */
    tmp = (FT_ULong)exec->stackSize;
    error = Update_Max( exec->memory,
                        &tmp,
                        sizeof ( FT_F26Dot6 ),
                        (void*)&exec->stack,
                        maxp->maxStackElements + 32 );
    exec->stackSize = (FT_Long)tmp;
    if ( error )
      return error;

    tmp = (FT_ULong)exec->glyphSize;
    error = Update_Max( exec->memory,
                        &tmp,
                        sizeof ( FT_Byte ),
                        (void*)&exec->glyphIns,
                        maxp->maxSizeOfInstructions );
    exec->glyphSize = (FT_UInt)tmp;
    if ( error )
      return error;

    exec->pts.n_points   = 0;
    exec->pts.n_contours = 0;

    exec->zp1 = exec->pts;
    exec->zp2 = exec->pts;
    exec->zp0 = exec->pts;

    exec->instruction_trap = FALSE;

    return FT_Err_Ok;
  }


  /**************************************************************************
   *
   * @Function:
   *   TT_Save_Context
   *
   * @Description:
   *   Saves the code ranges in a `size' object.
   *
   * @Input:
   *   exec ::
   *     A handle to the source execution context.
   *
   * @InOut:
   *   size ::
   *     A handle to the target size object.
   *
   * @Note:
   *   Only the glyph loader and debugger should call this function.
   */
  FT_LOCAL_DEF( void )
  TT_Save_Context( TT_ExecContext  exec,
                   TT_Size         size )
  {
    FT_Int  i;


    /* XXX: Will probably disappear soon with all the code range */
    /*      management, which is now rather obsolete.            */
    /*                                                           */
    size->num_function_defs    = exec->numFDefs;
    size->num_instruction_defs = exec->numIDefs;

    size->max_func = exec->maxFunc;
    size->max_ins  = exec->maxIns;

    for ( i = 0; i < TT_MAX_CODE_RANGES; i++ )
      size->codeRangeTable[i] = exec->codeRangeTable[i];
  }


  /**************************************************************************
   *
   * @Function:
   *   TT_Run_Context
   *
   * @Description:
   *   Executes one or more instructions in the execution context.
   *
   * @Input:
   *   exec ::
   *     A handle to the target execution context.
   *
   * @Return:
   *   TrueType error code.  0 means success.
   */
  FT_LOCAL_DEF( FT_Error )
  TT_Run_Context( TT_ExecContext  exec )
  {
    TT_Goto_CodeRange( exec, tt_coderange_glyph, 0 );

    exec->zp0 = exec->pts;
    exec->zp1 = exec->pts;
    exec->zp2 = exec->pts;

    exec->GS.gep0 = 1;
    exec->GS.gep1 = 1;
    exec->GS.gep2 = 1;

    exec->GS.projVector.x = 0x4000;
    exec->GS.projVector.y = 0x0000;

    exec->GS.freeVector = exec->GS.projVector;
    exec->GS.dualVector = exec->GS.projVector;

    exec->GS.round_state = 1;
    exec->GS.loop        = 1;

    /* some glyphs leave something on the stack. so we clean it */
    /* before a new execution.                                  */
    exec->top     = 0;
    exec->callTop = 0;

    return exec->face->interpreter( exec );
  }


  /* The default value for `scan_control' is documented as FALSE in the */
  /* TrueType specification.  This is confusing since it implies a      */
  /* Boolean value.  However, this is not the case, thus both the       */
  /* default values of our `scan_type' and `scan_control' fields (which */
  /* the documentation's `scan_control' variable is split into) are     */
  /* zero.                                                              */

  const TT_GraphicsState  tt_default_graphics_state =
  {
    0, 0, 0,
    { 0x4000, 0 },
    { 0x4000, 0 },
    { 0x4000, 0 },

    1, 64, 1,
    TRUE, 68, 0, 0, 9, 3,
    0, FALSE, 0, 1, 1, 1
  };


  /* documentation is in ttinterp.h */

  FT_EXPORT_DEF( TT_ExecContext )
  TT_New_Context( TT_Driver  driver )
  {
    FT_Memory  memory;
    FT_Error   error;

    TT_ExecContext  exec = NULL;


    if ( !driver )
      goto Fail;

    memory = driver->root.root.memory;

    /* allocate object and zero everything inside */
    if ( FT_NEW( exec ) )
      goto Fail;

    /* create callStack here, other allocations delayed */
    exec->memory   = memory;
    exec->callSize = 32;

    if ( FT_QNEW_ARRAY( exec->callStack, exec->callSize ) )
      FT_FREE( exec );

  Fail:
    return exec;
  }


  /**************************************************************************
   *
   * Before an opcode is executed, the interpreter verifies that there are
   * enough arguments on the stack, with the help of the `Pop_Push_Count'
   * table.
   *
   * For each opcode, the first column gives the number of arguments that
   * are popped from the stack; the second one gives the number of those
   * that are pushed in result.
   *
   * Opcodes which have a varying number of parameters in the data stream
   * (NPUSHB, NPUSHW) are handled specially; they have a negative value in
   * the `opcode_length' table, and the value in `Pop_Push_Count' is set
   * to zero.
   *
   */


#undef  PACK
#define PACK( x, y )  ( ( x << 4 ) | y )


  static
  const FT_Byte  Pop_Push_Count[256] =
  {
    /* opcodes are gathered in groups of 16 */
    /* please keep the spaces as they are   */

    /* 0x00 */
    /*  SVTCA[0]  */  PACK( 0, 0 ),
    /*  SVTCA[1]  */  PACK( 0, 0 ),
    /*  SPVTCA[0] */  PACK( 0, 0 ),
    /*  SPVTCA[1] */  PACK( 0, 0 ),
    /*  SFVTCA[0] */  PACK( 0, 0 ),
    /*  SFVTCA[1] */  PACK( 0, 0 ),
    /*  SPVTL[0]  */  PACK( 2, 0 ),
    /*  SPVTL[1]  */  PACK( 2, 0 ),
    /*  SFVTL[0]  */  PACK( 2, 0 ),
    /*  SFVTL[1]  */  PACK( 2, 0 ),
    /*  SPVFS     */  PACK( 2, 0 ),
    /*  SFVFS     */  PACK( 2, 0 ),
    /*  GPV       */  PACK( 0, 2 ),
    /*  GFV       */  PACK( 0, 2 ),
    /*  SFVTPV    */  PACK( 0, 0 ),
    /*  ISECT     */  PACK( 5, 0 ),

    /* 0x10 */
    /*  SRP0      */  PACK( 1, 0 ),
    /*  SRP1      */  PACK( 1, 0 ),
    /*  SRP2      */  PACK( 1, 0 ),
    /*  SZP0      */  PACK( 1, 0 ),
    /*  SZP1      */  PACK( 1, 0 ),
    /*  SZP2      */  PACK( 1, 0 ),
    /*  SZPS      */  PACK( 1, 0 ),
    /*  SLOOP     */  PACK( 1, 0 ),
    /*  RTG       */  PACK( 0, 0 ),
    /*  RTHG      */  PACK( 0, 0 ),
    /*  SMD       */  PACK( 1, 0 ),
    /*  ELSE      */  PACK( 0, 0 ),
    /*  JMPR      */  PACK( 1, 0 ),
    /*  SCVTCI    */  PACK( 1, 0 ),
    /*  SSWCI     */  PACK( 1, 0 ),
    /*  SSW       */  PACK( 1, 0 ),

    /* 0x20 */
    /*  DUP       */  PACK( 1, 2 ),
    /*  POP       */  PACK( 1, 0 ),
    /*  CLEAR     */  PACK( 0, 0 ),
    /*  SWAP      */  PACK( 2, 2 ),
    /*  DEPTH     */  PACK( 0, 1 ),
    /*  CINDEX    */  PACK( 1, 1 ),
    /*  MINDEX    */  PACK( 1, 0 ),
    /*  ALIGNPTS  */  PACK( 2, 0 ),
    /*  INS_$28   */  PACK( 0, 0 ),
    /*  UTP       */  PACK( 1, 0 ),
    /*  LOOPCALL  */  PACK( 2, 0 ),
    /*  CALL      */  PACK( 1, 0 ),
    /*  FDEF      */  PACK( 1, 0 ),
    /*  ENDF      */  PACK( 0, 0 ),
    /*  MDAP[0]   */  PACK( 1, 0 ),
    /*  MDAP[1]   */  PACK( 1, 0 ),

    /* 0x30 */
    /*  IUP[0]    */  PACK( 0, 0 ),
    /*  IUP[1]    */  PACK( 0, 0 ),
    /*  SHP[0]    */  PACK( 0, 0 ), /* loops */
    /*  SHP[1]    */  PACK( 0, 0 ), /* loops */
    /*  SHC[0]    */  PACK( 1, 0 ),
    /*  SHC[1]    */  PACK( 1, 0 ),
    /*  SHZ[0]    */  PACK( 1, 0 ),
    /*  SHZ[1]    */  PACK( 1, 0 ),
    /*  SHPIX     */  PACK( 1, 0 ), /* loops */
    /*  IP        */  PACK( 0, 0 ), /* loops */
    /*  MSIRP[0]  */  PACK( 2, 0 ),
    /*  MSIRP[1]  */  PACK( 2, 0 ),
    /*  ALIGNRP   */  PACK( 0, 0 ), /* loops */
    /*  RTDG      */  PACK( 0, 0 ),
    /*  MIAP[0]   */  PACK( 2, 0 ),
    /*  MIAP[1]   */  PACK( 2, 0 ),

    /* 0x40 */
    /*  NPUSHB    */  PACK( 0, 0 ),
    /*  NPUSHW    */  PACK( 0, 0 ),
    /*  WS        */  PACK( 2, 0 ),
    /*  RS        */  PACK( 1, 1 ),
    /*  WCVTP     */  PACK( 2, 0 ),
    /*  RCVT      */  PACK( 1, 1 ),
    /*  GC[0]     */  PACK( 1, 1 ),
    /*  GC[1]     */  PACK( 1, 1 ),
    /*  SCFS      */  PACK( 2, 0 ),
    /*  MD[0]     */  PACK( 2, 1 ),
    /*  MD[1]     */  PACK( 2, 1 ),
    /*  MPPEM     */  PACK( 0, 1 ),
    /*  MPS       */  PACK( 0, 1 ),
    /*  FLIPON    */  PACK( 0, 0 ),
    /*  FLIPOFF   */  PACK( 0, 0 ),
    /*  DEBUG     */  PACK( 1, 0 ),

    /* 0x50 */
    /*  LT        */  PACK( 2, 1 ),
    /*  LTEQ      */  PACK( 2, 1 ),
    /*  GT        */  PACK( 2, 1 ),
    /*  GTEQ      */  PACK( 2, 1 ),
    /*  EQ        */  PACK( 2, 1 ),
    /*  NEQ       */  PACK( 2, 1 ),
    /*  ODD       */  PACK( 1, 1 ),
    /*  EVEN      */  PACK( 1, 1 ),
    /*  IF        */  PACK( 1, 0 ),
    /*  EIF       */  PACK( 0, 0 ),
    /*  AND       */  PACK( 2, 1 ),
    /*  OR        */  PACK( 2, 1 ),
    /*  NOT       */  PACK( 1, 1 ),
    /*  DELTAP1   */  PACK( 1, 0 ),
    /*  SDB       */  PACK( 1, 0 ),
    /*  SDS       */  PACK( 1, 0 ),

    /* 0x60 */
    /*  ADD       */  PACK( 2, 1 ),
    /*  SUB       */  PACK( 2, 1 ),
    /*  DIV       */  PACK( 2, 1 ),
    /*  MUL       */  PACK( 2, 1 ),
    /*  ABS       */  PACK( 1, 1 ),
    /*  NEG       */  PACK( 1, 1 ),
    /*  FLOOR     */  PACK( 1, 1 ),
    /*  CEILING   */  PACK( 1, 1 ),
    /*  ROUND[0]  */  PACK( 1, 1 ),
    /*  ROUND[1]  */  PACK( 1, 1 ),
    /*  ROUND[2]  */  PACK( 1, 1 ),
    /*  ROUND[3]  */  PACK( 1, 1 ),
    /*  NROUND[0] */  PACK( 1, 1 ),
    /*  NROUND[1] */  PACK( 1, 1 ),
    /*  NROUND[2] */  PACK( 1, 1 ),
    /*  NROUND[3] */  PACK( 1, 1 ),

    /* 0x70 */
    /*  WCVTF     */  PACK( 2, 0 ),
    /*  DELTAP2   */  PACK( 1, 0 ),
    /*  DELTAP3   */  PACK( 1, 0 ),
    /*  DELTAC1   */  PACK( 1, 0 ),
    /*  DELTAC2   */  PACK( 1, 0 ),
    /*  DELTAC3   */  PACK( 1, 0 ),
    /*  SROUND    */  PACK( 1, 0 ),
    /*  S45ROUND  */  PACK( 1, 0 ),
    /*  JROT      */  PACK( 2, 0 ),
    /*  JROF      */  PACK( 2, 0 ),
    /*  ROFF      */  PACK( 0, 0 ),
    /*  INS_$7B   */  PACK( 0, 0 ),
    /*  RUTG      */  PACK( 0, 0 ),
    /*  RDTG      */  PACK( 0, 0 ),
    /*  SANGW     */  PACK( 1, 0 ),
    /*  AA        */  PACK( 1, 0 ),

    /* 0x80 */
    /*  FLIPPT    */  PACK( 0, 0 ), /* loops */
    /*  FLIPRGON  */  PACK( 2, 0 ),
    /*  FLIPRGOFF */  PACK( 2, 0 ),
    /*  INS_$83   */  PACK( 0, 0 ),
    /*  INS_$84   */  PACK( 0, 0 ),
    /*  SCANCTRL  */  PACK( 1, 0 ),
    /*  SDPVTL[0] */  PACK( 2, 0 ),
    /*  SDPVTL[1] */  PACK( 2, 0 ),
    /*  GETINFO   */  PACK( 1, 1 ),
    /*  IDEF      */  PACK( 1, 0 ),
    /*  ROLL      */  PACK( 3, 3 ),
    /*  MAX       */  PACK( 2, 1 ),
    /*  MIN       */  PACK( 2, 1 ),
    /*  SCANTYPE  */  PACK( 1, 0 ),
    /*  INSTCTRL  */  PACK( 2, 0 ),
    /*  INS_$8F   */  PACK( 0, 0 ),

    /* 0x90 */
    /*  INS_$90  */   PACK( 0, 0 ),
    /*  GETVAR   */   PACK( 0, 0 ), /* will be handled specially */
    /*  GETDATA  */   PACK( 0, 1 ),
    /*  INS_$93  */   PACK( 0, 0 ),
    /*  INS_$94  */   PACK( 0, 0 ),
    /*  INS_$95  */   PACK( 0, 0 ),
    /*  INS_$96  */   PACK( 0, 0 ),
    /*  INS_$97  */   PACK( 0, 0 ),
    /*  INS_$98  */   PACK( 0, 0 ),
    /*  INS_$99  */   PACK( 0, 0 ),
    /*  INS_$9A  */   PACK( 0, 0 ),
    /*  INS_$9B  */   PACK( 0, 0 ),
    /*  INS_$9C  */   PACK( 0, 0 ),
    /*  INS_$9D  */   PACK( 0, 0 ),
    /*  INS_$9E  */   PACK( 0, 0 ),
    /*  INS_$9F  */   PACK( 0, 0 ),

    /* 0xA0 */
    /*  INS_$A0  */   PACK( 0, 0 ),
    /*  INS_$A1  */   PACK( 0, 0 ),
    /*  INS_$A2  */   PACK( 0, 0 ),
    /*  INS_$A3  */   PACK( 0, 0 ),
    /*  INS_$A4  */   PACK( 0, 0 ),
    /*  INS_$A5  */   PACK( 0, 0 ),
    /*  INS_$A6  */   PACK( 0, 0 ),
    /*  INS_$A7  */   PACK( 0, 0 ),
    /*  INS_$A8  */   PACK( 0, 0 ),
    /*  INS_$A9  */   PACK( 0, 0 ),
    /*  INS_$AA  */   PACK( 0, 0 ),
    /*  INS_$AB  */   PACK( 0, 0 ),
    /*  INS_$AC  */   PACK( 0, 0 ),
    /*  INS_$AD  */   PACK( 0, 0 ),
    /*  INS_$AE  */   PACK( 0, 0 ),
    /*  INS_$AF  */   PACK( 0, 0 ),

    /* 0xB0 */
    /*  PUSHB[0]  */  PACK( 0, 1 ),
    /*  PUSHB[1]  */  PACK( 0, 2 ),
    /*  PUSHB[2]  */  PACK( 0, 3 ),
    /*  PUSHB[3]  */  PACK( 0, 4 ),
    /*  PUSHB[4]  */  PACK( 0, 5 ),
    /*  PUSHB[5]  */  PACK( 0, 6 ),
    /*  PUSHB[6]  */  PACK( 0, 7 ),
    /*  PUSHB[7]  */  PACK( 0, 8 ),
    /*  PUSHW[0]  */  PACK( 0, 1 ),
    /*  PUSHW[1]  */  PACK( 0, 2 ),
    /*  PUSHW[2]  */  PACK( 0, 3 ),
    /*  PUSHW[3]  */  PACK( 0, 4 ),
    /*  PUSHW[4]  */  PACK( 0, 5 ),
    /*  PUSHW[5]  */  PACK( 0, 6 ),
    /*  PUSHW[6]  */  PACK( 0, 7 ),
    /*  PUSHW[7]  */  PACK( 0, 8 ),

    /* 0xC0 */
    /*  MDRP[00]  */  PACK( 1, 0 ),
    /*  MDRP[01]  */  PACK( 1, 0 ),
    /*  MDRP[02]  */  PACK( 1, 0 ),
    /*  MDRP[03]  */  PACK( 1, 0 ),
    /*  MDRP[04]  */  PACK( 1, 0 ),
    /*  MDRP[05]  */  PACK( 1, 0 ),
    /*  MDRP[06]  */  PACK( 1, 0 ),
    /*  MDRP[07]  */  PACK( 1, 0 ),
    /*  MDRP[08]  */  PACK( 1, 0 ),
    /*  MDRP[09]  */  PACK( 1, 0 ),
    /*  MDRP[10]  */  PACK( 1, 0 ),
    /*  MDRP[11]  */  PACK( 1, 0 ),
    /*  MDRP[12]  */  PACK( 1, 0 ),
    /*  MDRP[13]  */  PACK( 1, 0 ),
    /*  MDRP[14]  */  PACK( 1, 0 ),
    /*  MDRP[15]  */  PACK( 1, 0 ),

    /* 0xD0 */
    /*  MDRP[16]  */  PACK( 1, 0 ),
    /*  MDRP[17]  */  PACK( 1, 0 ),
    /*  MDRP[18]  */  PACK( 1, 0 ),
    /*  MDRP[19]  */  PACK( 1, 0 ),
    /*  MDRP[20]  */  PACK( 1, 0 ),
    /*  MDRP[21]  */  PACK( 1, 0 ),
    /*  MDRP[22]  */  PACK( 1, 0 ),
    /*  MDRP[23]  */  PACK( 1, 0 ),
    /*  MDRP[24]  */  PACK( 1, 0 ),
    /*  MDRP[25]  */  PACK( 1, 0 ),
    /*  MDRP[26]  */  PACK( 1, 0 ),
    /*  MDRP[27]  */  PACK( 1, 0 ),
    /*  MDRP[28]  */  PACK( 1, 0 ),
    /*  MDRP[29]  */  PACK( 1, 0 ),
    /*  MDRP[30]  */  PACK( 1, 0 ),
    /*  MDRP[31]  */  PACK( 1, 0 ),

    /* 0xE0 */
    /*  MIRP[00]  */  PACK( 2, 0 ),
    /*  MIRP[01]  */  PACK( 2, 0 ),
    /*  MIRP[02]  */  PACK( 2, 0 ),
    /*  MIRP[03]  */  PACK( 2, 0 ),
    /*  MIRP[04]  */  PACK( 2, 0 ),
    /*  MIRP[05]  */  PACK( 2, 0 ),
    /*  MIRP[06]  */  PACK( 2, 0 ),
    /*  MIRP[07]  */  PACK( 2, 0 ),
    /*  MIRP[08]  */  PACK( 2, 0 ),
    /*  MIRP[09]  */  PACK( 2, 0 ),
    /*  MIRP[10]  */  PACK( 2, 0 ),
    /*  MIRP[11]  */  PACK( 2, 0 ),
    /*  MIRP[12]  */  PACK( 2, 0 ),
    /*  MIRP[13]  */  PACK( 2, 0 ),
    /*  MIRP[14]  */  PACK( 2, 0 ),
    /*  MIRP[15]  */  PACK( 2, 0 ),

    /* 0xF0 */
    /*  MIRP[16]  */  PACK( 2, 0 ),
    /*  MIRP[17]  */  PACK( 2, 0 ),
    /*  MIRP[18]  */  PACK( 2, 0 ),
    /*  MIRP[19]  */  PACK( 2, 0 ),
    /*  MIRP[20]  */  PACK( 2, 0 ),
    /*  MIRP[21]  */  PACK( 2, 0 ),
    /*  MIRP[22]  */  PACK( 2, 0 ),
    /*  MIRP[23]  */  PACK( 2, 0 ),
    /*  MIRP[24]  */  PACK( 2, 0 ),
    /*  MIRP[25]  */  PACK( 2, 0 ),
    /*  MIRP[26]  */  PACK( 2, 0 ),
    /*  MIRP[27]  */  PACK( 2, 0 ),
    /*  MIRP[28]  */  PACK( 2, 0 ),
    /*  MIRP[29]  */  PACK( 2, 0 ),
    /*  MIRP[30]  */  PACK( 2, 0 ),
    /*  MIRP[31]  */  PACK( 2, 0 )
  };


#ifdef FT_DEBUG_LEVEL_TRACE

  /* the first hex digit gives the length of the opcode name; the space */
  /* after the digit is here just to increase readability of the source */
  /* code                                                               */

  static
  const char*  const opcode_name[256] =
  {
    /* 0x00 */
    "8 SVTCA[y]",
    "8 SVTCA[x]",
    "9 SPVTCA[y]",
    "9 SPVTCA[x]",
    "9 SFVTCA[y]",
    "9 SFVTCA[x]",
    "9 SPVTL[||]",
    "8 SPVTL[+]",
    "9 SFVTL[||]",
    "8 SFVTL[+]",
    "5 SPVFS",
    "5 SFVFS",
    "3 GPV",
    "3 GFV",
    "6 SFVTPV",
    "5 ISECT",

    /* 0x10 */
    "4 SRP0",
    "4 SRP1",
    "4 SRP2",
    "4 SZP0",
    "4 SZP1",
    "4 SZP2",
    "4 SZPS",
    "5 SLOOP",
    "3 RTG",
    "4 RTHG",
    "3 SMD",
    "4 ELSE",
    "4 JMPR",
    "6 SCVTCI",
    "5 SSWCI",
    "3 SSW",

    /* 0x20 */
    "3 DUP",
    "3 POP",
    "5 CLEAR",
    "4 SWAP",
    "5 DEPTH",
    "6 CINDEX",
    "6 MINDEX",
    "8 ALIGNPTS",
    "7 INS_$28",
    "3 UTP",
    "8 LOOPCALL",
    "4 CALL",
    "4 FDEF",
    "4 ENDF",
    "6 MDAP[]",
    "9 MDAP[rnd]",

    /* 0x30 */
    "6 IUP[y]",
    "6 IUP[x]",
    "8 SHP[rp2]",
    "8 SHP[rp1]",
    "8 SHC[rp2]",
    "8 SHC[rp1]",
    "8 SHZ[rp2]",
    "8 SHZ[rp1]",
    "5 SHPIX",
    "2 IP",
    "7 MSIRP[]",
    "A MSIRP[rp0]",
    "7 ALIGNRP",
    "4 RTDG",
    "6 MIAP[]",
    "9 MIAP[rnd]",

    /* 0x40 */
    "6 NPUSHB",
    "6 NPUSHW",
    "2 WS",
    "2 RS",
    "5 WCVTP",
    "4 RCVT",
    "8 GC[curr]",
    "8 GC[orig]",
    "4 SCFS",
    "8 MD[curr]",
    "8 MD[orig]",
    "5 MPPEM",
    "3 MPS",
    "6 FLIPON",
    "7 FLIPOFF",
    "5 DEBUG",

    /* 0x50 */
    "2 LT",
    "4 LTEQ",
    "2 GT",
    "4 GTEQ",
    "2 EQ",
    "3 NEQ",
    "3 ODD",
    "4 EVEN",
    "2 IF",
    "3 EIF",
    "3 AND",
    "2 OR",
    "3 NOT",
    "7 DELTAP1",
    "3 SDB",
    "3 SDS",

    /* 0x60 */
    "3 ADD",
    "3 SUB",
    "3 DIV",
    "3 MUL",
    "3 ABS",
    "3 NEG",
    "5 FLOOR",
    "7 CEILING",
    "8 ROUND[G]",
    "8 ROUND[B]",
    "8 ROUND[W]",
    "7 ROUND[]",
    "9 NROUND[G]",
    "9 NROUND[B]",
    "9 NROUND[W]",
    "8 NROUND[]",

    /* 0x70 */
    "5 WCVTF",
    "7 DELTAP2",
    "7 DELTAP3",
    "7 DELTAC1",
    "7 DELTAC2",
    "7 DELTAC3",
    "6 SROUND",
    "8 S45ROUND",
    "4 JROT",
    "4 JROF",
    "4 ROFF",
    "7 INS_$7B",
    "4 RUTG",
    "4 RDTG",
    "5 SANGW",
    "2 AA",

    /* 0x80 */
    "6 FLIPPT",
    "8 FLIPRGON",
    "9 FLIPRGOFF",
    "7 INS_$83",
    "7 INS_$84",
    "8 SCANCTRL",
    "A SDPVTL[||]",
    "9 SDPVTL[+]",
    "7 GETINFO",
    "4 IDEF",
    "4 ROLL",
    "3 MAX",
    "3 MIN",
    "8 SCANTYPE",
    "8 INSTCTRL",
    "7 INS_$8F",

    /* 0x90 */
    "7 INS_$90",
#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT
    "C GETVARIATION",
    "7 GETDATA",
#else
    "7 INS_$91",
    "7 INS_$92",
#endif
    "7 INS_$93",
    "7 INS_$94",
    "7 INS_$95",
    "7 INS_$96",
    "7 INS_$97",
    "7 INS_$98",
    "7 INS_$99",
    "7 INS_$9A",
    "7 INS_$9B",
    "7 INS_$9C",
    "7 INS_$9D",
    "7 INS_$9E",
    "7 INS_$9F",

    /* 0xA0 */
    "7 INS_$A0",
    "7 INS_$A1",
    "7 INS_$A2",
    "7 INS_$A3",
    "7 INS_$A4",
    "7 INS_$A5",
    "7 INS_$A6",
    "7 INS_$A7",
    "7 INS_$A8",
    "7 INS_$A9",
    "7 INS_$AA",
    "7 INS_$AB",
    "7 INS_$AC",
    "7 INS_$AD",
    "7 INS_$AE",
    "7 INS_$AF",

    /* 0xB0 */
    "8 PUSHB[0]",
    "8 PUSHB[1]",
    "8 PUSHB[2]",
    "8 PUSHB[3]",
    "8 PUSHB[4]",
    "8 PUSHB[5]",
    "8 PUSHB[6]",
    "8 PUSHB[7]",
    "8 PUSHW[0]",
    "8 PUSHW[1]",
    "8 PUSHW[2]",
    "8 PUSHW[3]",
    "8 PUSHW[4]",
    "8 PUSHW[5]",
    "8 PUSHW[6]",
    "8 PUSHW[7]",

    /* 0xC0 */
    "7 MDRP[G]",
    "7 MDRP[B]",
    "7 MDRP[W]",
    "6 MDRP[]",
    "8 MDRP[rG]",
    "8 MDRP[rB]",
    "8 MDRP[rW]",
    "7 MDRP[r]",
    "8 MDRP[mG]",
    "8 MDRP[mB]",
    "8 MDRP[mW]",
    "7 MDRP[m]",
    "9 MDRP[mrG]",
    "9 MDRP[mrB]",
    "9 MDRP[mrW]",
    "8 MDRP[mr]",

    /* 0xD0 */
    "8 MDRP[pG]",
    "8 MDRP[pB]",
    "8 MDRP[pW]",
    "7 MDRP[p]",
    "9 MDRP[prG]",
    "9 MDRP[prB]",
    "9 MDRP[prW]",
    "8 MDRP[pr]",
    "9 MDRP[pmG]",
    "9 MDRP[pmB]",
    "9 MDRP[pmW]",
    "8 MDRP[pm]",
    "A MDRP[pmrG]",
    "A MDRP[pmrB]",
    "A MDRP[pmrW]",
    "9 MDRP[pmr]",

    /* 0xE0 */
    "7 MIRP[G]",
    "7 MIRP[B]",
    "7 MIRP[W]",
    "6 MIRP[]",
    "8 MIRP[rG]",
    "8 MIRP[rB]",
    "8 MIRP[rW]",
    "7 MIRP[r]",
    "8 MIRP[mG]",
    "8 MIRP[mB]",
    "8 MIRP[mW]",
    "7 MIRP[m]",
    "9 MIRP[mrG]",
    "9 MIRP[mrB]",
    "9 MIRP[mrW]",
    "8 MIRP[mr]",

    /* 0xF0 */
    "8 MIRP[pG]",
    "8 MIRP[pB]",
    "8 MIRP[pW]",
    "7 MIRP[p]",
    "9 MIRP[prG]",
    "9 MIRP[prB]",
    "9 MIRP[prW]",
    "8 MIRP[pr]",
    "9 MIRP[pmG]",
    "9 MIRP[pmB]",
    "9 MIRP[pmW]",
    "8 MIRP[pm]",
    "A MIRP[pmrG]",
    "A MIRP[pmrB]",
    "A MIRP[pmrW]",
    "9 MIRP[pmr]"
  };

#endif /* FT_DEBUG_LEVEL_TRACE */


  static
  const FT_Char  opcode_length[256] =
  {
    1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,
    1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,
    1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,
    1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,

   -1,-2, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,
    1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,
    1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,
    1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,

    1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,
    1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,
    1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,
    2, 3, 4, 5,  6, 7, 8, 9,  3, 5, 7, 9, 11,13,15,17,

    1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,
    1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,
    1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,
    1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1
  };

#undef PACK


#ifndef FT_CONFIG_OPTION_NO_ASSEMBLER

#if defined( __arm__ )                                 && \
    ( defined( __thumb2__ ) || !defined( __thumb__ ) )

#define TT_MulFix14  TT_MulFix14_arm

  static FT_Int32
  TT_MulFix14_arm( FT_Int32  a,
                   FT_Int    b )
  {
    FT_Int32  t, t2;


#if defined( __CC_ARM ) || defined( __ARMCC__ )

    __asm
    {
      smull t2, t,  b,  a           /* (lo=t2,hi=t) = a*b */
      mov   a,  t,  asr #31         /* a   = (hi >> 31) */
      add   a,  a,  #0x2000         /* a  += 0x2000 */
      adds  t2, t2, a               /* t2 += a */
      adc   t,  t,  #0              /* t  += carry */
      mov   a,  t2, lsr #14         /* a   = t2 >> 14 */
      orr   a,  a,  t,  lsl #18     /* a  |= t << 18 */
    }

#elif defined( __GNUC__ )

    __asm__ __volatile__ (
      "smull  %1, %2, %4, %3\n\t"       /* (lo=%1,hi=%2) = a*b */
      "mov    %0, %2, asr #31\n\t"      /* %0  = (hi >> 31) */
#if defined( __clang__ ) && defined( __thumb2__ )
      "add.w  %0, %0, #0x2000\n\t"      /* %0 += 0x2000 */
#else
      "add    %0, %0, #0x2000\n\t"      /* %0 += 0x2000 */
#endif
      "adds   %1, %1, %0\n\t"           /* %1 += %0 */
      "adc    %2, %2, #0\n\t"           /* %2 += carry */
      "mov    %0, %1, lsr #14\n\t"      /* %0  = %1 >> 16 */
      "orr    %0, %0, %2, lsl #18\n\t"  /* %0 |= %2 << 16 */
      : "=r"(a), "=&r"(t2), "=&r"(t)
      : "r"(a), "r"(b)
      : "cc" );

#endif

    return a;
  }

#endif /* __arm__ && ( __thumb2__ || !__thumb__ ) */

#endif /* !FT_CONFIG_OPTION_NO_ASSEMBLER */


#if defined( __GNUC__ )                              && \
    ( defined( __i386__ ) || defined( __x86_64__ ) )

#define TT_MulFix14  TT_MulFix14_long_long

  /* Temporarily disable the warning that C90 doesn't support `long long'. */
#if ( __GNUC__ * 100 + __GNUC_MINOR__ ) >= 406
#pragma GCC diagnostic push
#endif
#pragma GCC diagnostic ignored "-Wlong-long"

  /* This is declared `noinline' because inlining the function results */
  /* in slower code.  The `pure' attribute indicates that the result   */
  /* only depends on the parameters.                                   */
  static __attribute__(( noinline ))
         __attribute__(( pure )) FT_Int32
  TT_MulFix14_long_long( FT_Int32  a,
                         FT_Int    b )
  {

    long long  ret = (long long)a * b;

    /* The following line assumes that right shifting of signed values */
    /* will actually preserve the sign bit.  The exact behaviour is    */
    /* undefined, but this is true on x86 and x86_64.                  */
    long long  tmp = ret >> 63;


    ret += 0x2000 + tmp;

    return (FT_Int32)( ret >> 14 );
  }

#if ( __GNUC__ * 100 + __GNUC_MINOR__ ) >= 406
#pragma GCC diagnostic pop
#endif

#endif /* __GNUC__ && ( __i386__ || __x86_64__ ) */


#ifndef TT_MulFix14

  /* Compute (a*b)/2^14 with maximum accuracy and rounding.  */
  /* This is optimized to be faster than calling FT_MulFix() */
  /* for platforms where sizeof(int) == 2.                   */
  static FT_Int32
  TT_MulFix14( FT_Int32  a,
               FT_Int    b )
  {
    FT_Int32   sign;
    FT_UInt32  ah, al, mid, lo, hi;


    sign = a ^ b;

    if ( a < 0 )
      a = -a;
    if ( b < 0 )
      b = -b;

    ah = (FT_UInt32)( ( a >> 16 ) & 0xFFFFU );
    al = (FT_UInt32)( a & 0xFFFFU );

    lo    = al * b;
    mid   = ah * b;
    hi    = mid >> 16;
    mid   = ( mid << 16 ) + ( 1 << 13 ); /* rounding */
    lo   += mid;
    if ( lo < mid )
      hi += 1;

    mid = ( lo >> 14 ) | ( hi << 18 );

    return sign >= 0 ? (FT_Int32)mid : -(FT_Int32)mid;
  }

#endif  /* !TT_MulFix14 */


#if defined( __GNUC__ )        && \
    ( defined( __i386__ )   ||    \
      defined( __x86_64__ ) ||    \
      defined( __arm__ )    )

#define TT_DotFix14  TT_DotFix14_long_long

#if ( __GNUC__ * 100 + __GNUC_MINOR__ ) >= 406
#pragma GCC diagnostic push
#endif
#pragma GCC diagnostic ignored "-Wlong-long"

  static __attribute__(( pure )) FT_Int32
  TT_DotFix14_long_long( FT_Int32  ax,
                         FT_Int32  ay,
                         FT_Int    bx,
                         FT_Int    by )
  {
    /* Temporarily disable the warning that C90 doesn't support */
    /* `long long'.                                             */

    long long  temp1 = (long long)ax * bx;
    long long  temp2 = (long long)ay * by;


    temp1 += temp2;
    temp2  = temp1 >> 63;
    temp1 += 0x2000 + temp2;

    return (FT_Int32)( temp1 >> 14 );

  }

#if ( __GNUC__ * 100 + __GNUC_MINOR__ ) >= 406
#pragma GCC diagnostic pop
#endif

#endif /* __GNUC__ && (__arm__ || __i386__ || __x86_64__) */


#ifndef TT_DotFix14

  /* compute (ax*bx+ay*by)/2^14 with maximum accuracy and rounding */
  static FT_Int32
  TT_DotFix14( FT_Int32  ax,
               FT_Int32  ay,
               FT_Int    bx,
               FT_Int    by )
  {
    FT_Int32   m, s, hi1, hi2, hi;
    FT_UInt32  l, lo1, lo2, lo;


    /* compute ax*bx as 64-bit value */
    l = (FT_UInt32)( ( ax & 0xFFFFU ) * bx );
    m = ( ax >> 16 ) * bx;

    lo1 = l + ( (FT_UInt32)m << 16 );
    hi1 = ( m >> 16 ) + ( (FT_Int32)l >> 31 ) + ( lo1 < l );

    /* compute ay*by as 64-bit value */
    l = (FT_UInt32)( ( ay & 0xFFFFU ) * by );
    m = ( ay >> 16 ) * by;

    lo2 = l + ( (FT_UInt32)m << 16 );
    hi2 = ( m >> 16 ) + ( (FT_Int32)l >> 31 ) + ( lo2 < l );

    /* add them */
    lo = lo1 + lo2;
    hi = hi1 + hi2 + ( lo < lo1 );

    /* divide the result by 2^14 with rounding */
    s   = hi >> 31;
    l   = lo + (FT_UInt32)s;
    hi += s + ( l < lo );
    lo  = l;

    l   = lo + 0x2000U;
    hi += ( l < lo );

    return (FT_Int32)( ( (FT_UInt32)hi << 18 ) | ( l >> 14 ) );
  }

#endif /* TT_DotFix14 */


  /**************************************************************************
   *
   * @Function:
   *   Current_Ratio
   *
   * @Description:
   *   Returns the current aspect ratio scaling factor depending on the
   *   projection vector's state and device resolutions.
   *
   * @Return:
   *   The aspect ratio in 16.16 format, always <= 1.0 .
   */
  static FT_Long
  Current_Ratio( TT_ExecContext  exc )
  {
    if ( !exc->tt_metrics.ratio )
    {
      if ( exc->GS.projVector.y == 0 )
        exc->tt_metrics.ratio = exc->tt_metrics.x_ratio;

      else if ( exc->GS.projVector.x == 0 )
        exc->tt_metrics.ratio = exc->tt_metrics.y_ratio;

      else
      {
        FT_F26Dot6  x, y;


        x = TT_MulFix14( exc->tt_metrics.x_ratio,
                         exc->GS.projVector.x );
        y = TT_MulFix14( exc->tt_metrics.y_ratio,
                         exc->GS.projVector.y );
        exc->tt_metrics.ratio = FT_Hypot( x, y );
      }
    }
    return exc->tt_metrics.ratio;
  }


  FT_CALLBACK_DEF( FT_Long )
  Current_Ppem( TT_ExecContext  exc )
  {
    return exc->tt_metrics.ppem;
  }


  FT_CALLBACK_DEF( FT_Long )
  Current_Ppem_Stretched( TT_ExecContext  exc )
  {
    return FT_MulFix( exc->tt_metrics.ppem, Current_Ratio( exc ) );
  }


  /**************************************************************************
   *
   * Functions related to the control value table (CVT).
   *
   */


  FT_CALLBACK_DEF( FT_F26Dot6 )
  Read_CVT( TT_ExecContext  exc,
            FT_ULong        idx )
  {
    return exc->cvt[idx];
  }


  FT_CALLBACK_DEF( FT_F26Dot6 )
  Read_CVT_Stretched( TT_ExecContext  exc,
                      FT_ULong        idx )
  {
    return FT_MulFix( exc->cvt[idx], Current_Ratio( exc ) );
  }


  static void
  Modify_CVT_Check( TT_ExecContext  exc )
  {
    /* TT_RunIns sets origCvt and restores cvt to origCvt when done. */
    if ( exc->iniRange == tt_coderange_glyph &&
         exc->cvt == exc->origCvt            )
    {
      exc->error = Update_Max( exc->memory,
                               &exc->glyfCvtSize,
                               sizeof ( FT_Long ),
                               (void*)&exc->glyfCvt,
                               exc->cvtSize );
      if ( exc->error )
        return;

      FT_ARRAY_COPY( exc->glyfCvt, exc->cvt, exc->glyfCvtSize );
      exc->cvt = exc->glyfCvt;
    }
  }


  FT_CALLBACK_DEF( void )
  Write_CVT( TT_ExecContext  exc,
             FT_ULong        idx,
             FT_F26Dot6      value )
  {
    Modify_CVT_Check( exc );
    if ( exc->error )
      return;

    exc->cvt[idx] = value;
  }


  FT_CALLBACK_DEF( void )
  Write_CVT_Stretched( TT_ExecContext  exc,
                       FT_ULong        idx,
                       FT_F26Dot6      value )
  {
    Modify_CVT_Check( exc );
    if ( exc->error )
      return;

    exc->cvt[idx] = FT_DivFix( value, Current_Ratio( exc ) );
  }


  FT_CALLBACK_DEF( void )
  Move_CVT( TT_ExecContext  exc,
            FT_ULong        idx,
            FT_F26Dot6      value )
  {
    Modify_CVT_Check( exc );
    if ( exc->error )
      return;

    exc->cvt[idx] = ADD_LONG( exc->cvt[idx], value );
  }


  FT_CALLBACK_DEF( void )
  Move_CVT_Stretched( TT_ExecContext  exc,
                      FT_ULong        idx,
                      FT_F26Dot6      value )
  {
    Modify_CVT_Check( exc );
    if ( exc->error )
      return;

    exc->cvt[idx] = ADD_LONG( exc->cvt[idx],
                              FT_DivFix( value, Current_Ratio( exc ) ) );
  }


  /**************************************************************************
   *
   * @Function:
   *   GetShortIns
   *
   * @Description:
   *   Returns a short integer taken from the instruction stream at
   *   address IP.
   *
   * @Return:
   *   Short read at code[IP].
   *
   * @Note:
   *   This one could become a macro.
   */
  static FT_Short
  GetShortIns( TT_ExecContext  exc )
  {
    /* Reading a byte stream so there is no endianness (DaveP) */
    exc->IP += 2;
    return (FT_Short)( ( exc->code[exc->IP - 2] << 8 ) +
                         exc->code[exc->IP - 1]      );
  }


  /**************************************************************************
   *
   * @Function:
   *   Ins_Goto_CodeRange
   *
   * @Description:
   *   Goes to a certain code range in the instruction stream.
   *
   * @Input:
   *   aRange ::
   *     The index of the code range.
   *
   *   aIP ::
   *     The new IP address in the code range.
   *
   * @Return:
   *   SUCCESS or FAILURE.
   */
  static FT_Bool
  Ins_Goto_CodeRange( TT_ExecContext  exc,
                      FT_Int          aRange,
                      FT_Long         aIP )
  {
    TT_CodeRange*  range;


    if ( aRange < 1 || aRange > 3 )
    {
      exc->error = FT_THROW( Bad_Argument );
      return FAILURE;
    }

    range = &exc->codeRangeTable[aRange - 1];

    if ( !range->base )     /* invalid coderange */
    {
      exc->error = FT_THROW( Invalid_CodeRange );
      return FAILURE;
    }

    /* NOTE: Because the last instruction of a program may be a CALL */
    /*       which will return to the first byte *after* the code    */
    /*       range, we test for aIP <= Size, instead of aIP < Size.  */

    if ( aIP > range->size )
    {
      exc->error = FT_THROW( Code_Overflow );
      return FAILURE;
    }

    exc->code     = range->base;
    exc->codeSize = range->size;
    exc->IP       = aIP;
    exc->curRange = aRange;

    return SUCCESS;
  }


  /*
   *
   * Apple's TrueType specification at
   *
   *   https://developer.apple.com/fonts/TrueType-Reference-Manual/RM02/Chap2.html#order
   *
   * gives the following order of operations in instructions that move
   * points.
   *
   *   - check single width cut-in (MIRP, MDRP)
   *
   *   - check control value cut-in (MIRP, MIAP)
   *
   *   - apply engine compensation (MIRP, MDRP)
   *
   *   - round distance (MIRP, MDRP) or value (MIAP, MDAP)
   *
   *   - check minimum distance (MIRP,MDRP)
   *
   *   - move point (MIRP, MDRP, MIAP, MSIRP, MDAP)
   *
   * For rounding instructions, engine compensation happens before rounding.
   *
   */


  /**************************************************************************
   *
   * @Function:
   *   Direct_Move
   *
   * @Description:
   *   Moves a point by a given distance along the freedom vector.  The
   *   point will be `touched'.
   *
   * @Input:
   *   point ::
   *     The index of the point to move.
   *
   *   distance ::
   *     The distance to apply.
   *
   * @InOut:
   *   zone ::
   *     The affected glyph zone.
   *
   * @Note:
   *   See `ttinterp.h' for details on backward compatibility mode.
   *   `Touches' the point.
   */
  static void
  Direct_Move( TT_ExecContext  exc,
               TT_GlyphZone    zone,
               FT_UShort       point,
               FT_F26Dot6      distance )
  {
    FT_F26Dot6  v;


    v = exc->GS.freeVector.x;

    if ( v != 0 )
    {
#ifdef TT_SUPPORT_SUBPIXEL_HINTING_INFINALITY
      if ( SUBPIXEL_HINTING_INFINALITY                            &&
           ( !exc->ignore_x_mode                                ||
             ( exc->sph_tweak_flags & SPH_TWEAK_ALLOW_X_DMOVE ) ) )
        zone->cur[point].x = ADD_LONG( zone->cur[point].x,
                                       FT_MulDiv( distance,
                                                  v,
                                                  exc->F_dot_P ) );
      else
#endif /* TT_SUPPORT_SUBPIXEL_HINTING_INFINALITY */

#ifdef TT_SUPPORT_SUBPIXEL_HINTING_MINIMAL
      /* Exception to the post-IUP curfew: Allow the x component of */
      /* diagonal moves, but only post-IUP.  DejaVu tries to adjust */
      /* diagonal stems like on `Z' and `z' post-IUP.               */
      if ( SUBPIXEL_HINTING_MINIMAL && !exc->backward_compatibility )
        zone->cur[point].x = ADD_LONG( zone->cur[point].x,
                                       FT_MulDiv( distance,
                                                  v,
                                                  exc->F_dot_P ) );
      else
#endif

      if ( NO_SUBPIXEL_HINTING )
        zone->cur[point].x = ADD_LONG( zone->cur[point].x,
                                       FT_MulDiv( distance,
                                                  v,
                                                  exc->F_dot_P ) );

      zone->tags[point] |= FT_CURVE_TAG_TOUCH_X;
    }

    v = exc->GS.freeVector.y;

    if ( v != 0 )
    {
#ifdef TT_SUPPORT_SUBPIXEL_HINTING_MINIMAL
      if ( !( SUBPIXEL_HINTING_MINIMAL    &&
              exc->backward_compatibility &&
              exc->iupx_called            &&
              exc->iupy_called            ) )
#endif
        zone->cur[point].y = ADD_LONG( zone->cur[point].y,
                                       FT_MulDiv( distance,
                                                  v,
                                                  exc->F_dot_P ) );

      zone->tags[point] |= FT_CURVE_TAG_TOUCH_Y;
    }
  }


  /**************************************************************************
   *
   * @Function:
   *   Direct_Move_Orig
   *
   * @Description:
   *   Moves the *original* position of a point by a given distance along
   *   the freedom vector.  Obviously, the point will not be `touched'.
   *
   * @Input:
   *   point ::
   *     The index of the point to move.
   *
   *   distance ::
   *     The distance to apply.
   *
   * @InOut:
   *   zone ::
   *     The affected glyph zone.
   */
  static void
  Direct_Move_Orig( TT_ExecContext  exc,
                    TT_GlyphZone    zone,
                    FT_UShort       point,
                    FT_F26Dot6      distance )
  {
    FT_F26Dot6  v;


    v = exc->GS.freeVector.x;

    if ( v != 0 )
      zone->org[point].x = ADD_LONG( zone->org[point].x,
                                     FT_MulDiv( distance,
                                                v,
                                                exc->F_dot_P ) );

    v = exc->GS.freeVector.y;

    if ( v != 0 )
      zone->org[point].y = ADD_LONG( zone->org[point].y,
                                     FT_MulDiv( distance,
                                                v,
                                                exc->F_dot_P ) );
  }


  /**************************************************************************
   *
   * Special versions of Direct_Move()
   *
   *   The following versions are used whenever both vectors are both
   *   along one of the coordinate unit vectors, i.e. in 90% of the cases.
   *   See `ttinterp.h' for details on backward compatibility mode.
   *
   */


  static void
  Direct_Move_X( TT_ExecContext  exc,
                 TT_GlyphZone    zone,
                 FT_UShort       point,
                 FT_F26Dot6      distance )
  {
#ifdef TT_SUPPORT_SUBPIXEL_HINTING_INFINALITY
    if ( SUBPIXEL_HINTING_INFINALITY && !exc->ignore_x_mode )
      zone->cur[point].x = ADD_LONG( zone->cur[point].x, distance );
    else
#endif /* TT_SUPPORT_SUBPIXEL_HINTING_INFINALITY */

#ifdef TT_SUPPORT_SUBPIXEL_HINTING_MINIMAL
    if ( SUBPIXEL_HINTING_MINIMAL && !exc->backward_compatibility )
      zone->cur[point].x = ADD_LONG( zone->cur[point].x, distance );
    else
#endif

    if ( NO_SUBPIXEL_HINTING )
      zone->cur[point].x = ADD_LONG( zone->cur[point].x, distance );

    zone->tags[point]  |= FT_CURVE_TAG_TOUCH_X;
  }


  static void
  Direct_Move_Y( TT_ExecContext  exc,
                 TT_GlyphZone    zone,
                 FT_UShort       point,
                 FT_F26Dot6      distance )
  {
    FT_UNUSED( exc );

#ifdef TT_SUPPORT_SUBPIXEL_HINTING_MINIMAL
    if ( !( SUBPIXEL_HINTING_MINIMAL             &&
            exc->backward_compatibility          &&
            exc->iupx_called && exc->iupy_called ) )
#endif
      zone->cur[point].y = ADD_LONG( zone->cur[point].y, distance );

    zone->tags[point] |= FT_CURVE_TAG_TOUCH_Y;
  }


  /**************************************************************************
   *
   * Special versions of Direct_Move_Orig()
   *
   *   The following versions are used whenever both vectors are both
   *   along one of the coordinate unit vectors, i.e. in 90% of the cases.
   *
   */


  static void
  Direct_Move_Orig_X( TT_ExecContext  exc,
                      TT_GlyphZone    zone,
                      FT_UShort       point,
                      FT_F26Dot6      distance )
  {
    FT_UNUSED( exc );

    zone->org[point].x = ADD_LONG( zone->org[point].x, distance );
  }


  static void
  Direct_Move_Orig_Y( TT_ExecContext  exc,
                      TT_GlyphZone    zone,
                      FT_UShort       point,
                      FT_F26Dot6      distance )
  {
    FT_UNUSED( exc );

    zone->org[point].y = ADD_LONG( zone->org[point].y, distance );
  }

  /**************************************************************************
   *
   * @Function:
   *   Round_None
   *
   * @Description:
   *   Does not round, but adds engine compensation.
   *
   * @Input:
   *   distance ::
   *     The distance (not) to round.
   *
   *   color ::
   *     The engine compensation color.
   *
   * @Return:
   *   The compensated distance.
   */
  static FT_F26Dot6
  Round_None( TT_ExecContext  exc,
              FT_F26Dot6      distance,
              FT_Int          color )
  {
    FT_F26Dot6  compensation = exc->tt_metrics.compensations[color];
    FT_F26Dot6  val;


    if ( distance >= 0 )
    {
      val = ADD_LONG( distance, compensation );
      if ( val < 0 )
        val = 0;
    }
    else
    {
      val = SUB_LONG( distance, compensation );
      if ( val > 0 )
        val = 0;
    }
    return val;
  }


  /**************************************************************************
   *
   * @Function:
   *   Round_To_Grid
   *
   * @Description:
   *   Rounds value to grid after adding engine compensation.
   *
   * @Input:
   *   distance ::
   *     The distance to round.
   *
   *   color ::
   *     The engine compensation color.
   *
   * @Return:
   *   Rounded distance.
   */
  static FT_F26Dot6
  Round_To_Grid( TT_ExecContext  exc,
                 FT_F26Dot6      distance,
                 FT_Int          color )
  {
    FT_F26Dot6  compensation = exc->tt_metrics.compensations[color];
    FT_F26Dot6  val;


    if ( distance >= 0 )
    {
      val = FT_PIX_ROUND_LONG( ADD_LONG( distance, compensation ) );
      if ( val < 0 )
        val = 0;
    }
    else
    {
      val = NEG_LONG( FT_PIX_ROUND_LONG( SUB_LONG( compensation,
                                                   distance ) ) );
      if ( val > 0 )
        val = 0;
    }

    return val;
  }


  /**************************************************************************
   *
   * @Function:
   *   Round_To_Half_Grid
   *
   * @Description:
   *   Rounds value to half grid after adding engine compensation.
   *
   * @Input:
   *   distance ::
   *     The distance to round.
   *
   *   color ::
   *     The engine compensation color.
   *
   * @Return:
   *   Rounded distance.
   */
  static FT_F26Dot6
  Round_To_Half_Grid( TT_ExecContext  exc,
                      FT_F26Dot6      distance,
                      FT_Int          color )
  {
    FT_F26Dot6  compensation = exc->tt_metrics.compensations[color];
    FT_F26Dot6  val;


    if ( distance >= 0 )
    {
      val = ADD_LONG( FT_PIX_FLOOR( ADD_LONG( distance, compensation ) ),
                      32 );
      if ( val < 0 )
        val = 32;
    }
    else
    {
      val = NEG_LONG( ADD_LONG( FT_PIX_FLOOR( SUB_LONG( compensation,
                                                        distance ) ),
                                32 ) );
      if ( val > 0 )
        val = -32;
    }

    return val;
  }


  /**************************************************************************
   *
   * @Function:
   *   Round_Down_To_Grid
   *
   * @Description:
   *   Rounds value down to grid after adding engine compensation.
   *
   * @Input:
   *   distance ::
   *     The distance to round.
   *
   *   color ::
   *     The engine compensation color.
   *
   * @Return:
   *   Rounded distance.
   */
  static FT_F26Dot6
  Round_Down_To_Grid( TT_ExecContext  exc,
                      FT_F26Dot6      distance,
                      FT_Int          color )
  {
    FT_F26Dot6  compensation = exc->tt_metrics.compensations[color];
    FT_F26Dot6  val;


    if ( distance >= 0 )
    {
      val = FT_PIX_FLOOR( ADD_LONG( distance, compensation ) );
      if ( val < 0 )
        val = 0;
    }
    else
    {
      val = NEG_LONG( FT_PIX_FLOOR( SUB_LONG( compensation, distance ) ) );
      if ( val > 0 )
        val = 0;
    }

    return val;
  }


  /**************************************************************************
   *
   * @Function:
   *   Round_Up_To_Grid
   *
   * @Description:
   *   Rounds value up to grid after adding engine compensation.
   *
   * @Input:
   *   distance ::
   *     The distance to round.
   *
   *   color ::
   *     The engine compensation color.
   *
   * @Return:
   *   Rounded distance.
   */
  static FT_F26Dot6
  Round_Up_To_Grid( TT_ExecContext  exc,
                    FT_F26Dot6      distance,
                    FT_Int          color )
  {
    FT_F26Dot6  compensation = exc->tt_metrics.compensations[color];
    FT_F26Dot6  val;


    if ( distance >= 0 )
    {
      val = FT_PIX_CEIL_LONG( ADD_LONG( distance, compensation ) );
      if ( val < 0 )
        val = 0;
    }
    else
    {
      val = NEG_LONG( FT_PIX_CEIL_LONG( SUB_LONG( compensation,
                                                  distance ) ) );
      if ( val > 0 )
        val = 0;
    }

    return val;
  }


  /**************************************************************************
   *
   * @Function:
   *   Round_To_Double_Grid
   *
   * @Description:
   *   Rounds value to double grid after adding engine compensation.
   *
   * @Input:
   *   distance ::
   *     The distance to round.
   *
   *   color ::
   *     The engine compensation color.
   *
   * @Return:
   *   Rounded distance.
   */
  static FT_F26Dot6
  Round_To_Double_Grid( TT_ExecContext  exc,
                        FT_F26Dot6      distance,
                        FT_Int          color )
  {
    FT_F26Dot6  compensation = exc->tt_metrics.compensations[color];
    FT_F26Dot6  val;


    if ( distance >= 0 )
    {
      val = FT_PAD_ROUND_LONG( ADD_LONG( distance, compensation ), 32 );
      if ( val < 0 )
        val = 0;
    }
    else
    {
      val = NEG_LONG( FT_PAD_ROUND_LONG( SUB_LONG( compensation, distance ),
                                         32 ) );
      if ( val > 0 )
        val = 0;
    }

    return val;
  }


  /**************************************************************************
   *
   * @Function:
   *   Round_Super
   *
   * @Description:
   *   Super-rounds value to grid after adding engine compensation.
   *
   * @Input:
   *   distance ::
   *     The distance to round.
   *
   *   color ::
   *     The engine compensation color.
   *
   * @Return:
   *   Rounded distance.
   *
   * @Note:
   *   The TrueType specification says very little about the relationship
   *   between rounding and engine compensation.  However, it seems from
   *   the description of super round that we should add the compensation
   *   before rounding.
   */
  static FT_F26Dot6
  Round_Super( TT_ExecContext  exc,
               FT_F26Dot6      distance,
               FT_Int          color )
  {
    FT_F26Dot6  compensation = exc->tt_metrics.compensations[color];
    FT_F26Dot6  val;


    if ( distance >= 0 )
    {
      val = ADD_LONG( distance,
                      exc->threshold - exc->phase + compensation ) &
              -exc->period;
      val = ADD_LONG( val, exc->phase );
      if ( val < 0 )
        val = exc->phase;
    }
    else
    {
      val = NEG_LONG( SUB_LONG( exc->threshold - exc->phase + compensation,
                                distance ) &
                        -exc->period );
      val = SUB_LONG( val, exc->phase );
      if ( val > 0 )
        val = -exc->phase;
    }

    return val;
  }


  /**************************************************************************
   *
   * @Function:
   *   Round_Super_45
   *
   * @Description:
   *   Super-rounds value to grid after adding engine compensation.
   *
   * @Input:
   *   distance ::
   *     The distance to round.
   *
   *   color ::
   *     The engine compensation color.
   *
   * @Return:
   *   Rounded distance.
   *
   * @Note:
   *   There is a separate function for Round_Super_45() as we may need
   *   greater precision.
   */
  static FT_F26Dot6
  Round_Super_45( TT_ExecContext  exc,
                  FT_F26Dot6      distance,
                  FT_Int          color )
  {
    FT_F26Dot6  compensation = exc->tt_metrics.compensations[color];
    FT_F26Dot6  val;


    if ( distance >= 0 )
    {
      val = ( ADD_LONG( distance,
                        exc->threshold - exc->phase + compensation ) /
                exc->period ) * exc->period;
      val = ADD_LONG( val, exc->phase );
      if ( val < 0 )
        val = exc->phase;
    }
    else
    {
      val = NEG_LONG( ( SUB_LONG( exc->threshold - exc->phase + compensation,
                                  distance ) /
                          exc->period ) * exc->period );
      val = SUB_LONG( val, exc->phase );
      if ( val > 0 )
        val = -exc->phase;
    }

    return val;
  }


  /**************************************************************************
   *
   * @Function:
   *   Compute_Round
   *
   * @Description:
   *   Sets the rounding mode.
   *
   * @Input:
   *   round_mode ::
   *     The rounding mode to be used.
   */
  static void
  Compute_Round( TT_ExecContext  exc,
                 FT_Byte         round_mode )
  {
    switch ( round_mode )
    {
    case TT_Round_Off:
      exc->func_round = (TT_Round_Func)Round_None;
      break;

    case TT_Round_To_Grid:
      exc->func_round = (TT_Round_Func)Round_To_Grid;
      break;

    case TT_Round_Up_To_Grid:
      exc->func_round = (TT_Round_Func)Round_Up_To_Grid;
      break;

    case TT_Round_Down_To_Grid:
      exc->func_round = (TT_Round_Func)Round_Down_To_Grid;
      break;

    case TT_Round_To_Half_Grid:
      exc->func_round = (TT_Round_Func)Round_To_Half_Grid;
      break;

    case TT_Round_To_Double_Grid:
      exc->func_round = (TT_Round_Func)Round_To_Double_Grid;
      break;

    case TT_Round_Super:
      exc->func_round = (TT_Round_Func)Round_Super;
      break;

    case TT_Round_Super_45:
      exc->func_round = (TT_Round_Func)Round_Super_45;
      break;
    }
  }


  /**************************************************************************
   *
   * @Function:
   *   SetSuperRound
   *
   * @Description:
   *   Sets Super Round parameters.
   *
   * @Input:
   *   GridPeriod ::
   *     The grid period.
   *
   *   selector ::
   *     The SROUND opcode.
   */
  static void
  SetSuperRound( TT_ExecContext  exc,
                 FT_F2Dot14      GridPeriod,
                 FT_Long         selector )
  {
    switch ( (FT_Int)( selector & 0xC0 ) )
    {
      case 0:
        exc->period = GridPeriod / 2;
        break;

      case 0x40:
        exc->period = GridPeriod;
        break;

      case 0x80:
        exc->period = GridPeriod * 2;
        break;

      /* This opcode is reserved, but... */
      case 0xC0:
        exc->period = GridPeriod;
        break;
    }

    switch ( (FT_Int)( selector & 0x30 ) )
    {
    case 0:
      exc->phase = 0;
      break;

    case 0x10:
      exc->phase = exc->period / 4;
      break;

    case 0x20:
      exc->phase = exc->period / 2;
      break;

    case 0x30:
      exc->phase = exc->period * 3 / 4;
      break;
    }

    if ( ( selector & 0x0F ) == 0 )
      exc->threshold = exc->period - 1;
    else
      exc->threshold = ( (FT_Int)( selector & 0x0F ) - 4 ) * exc->period / 8;

    /* convert to F26Dot6 format */
    exc->period    >>= 8;
    exc->phase     >>= 8;
    exc->threshold >>= 8;
  }


  /**************************************************************************
   *
   * @Function:
   *   Project
   *
   * @Description:
   *   Computes the projection of vector given by (v2-v1) along the
   *   current projection vector.
   *
   * @Input:
   *   v1 ::
   *     First input vector.
   *   v2 ::
   *     Second input vector.
   *
   * @Return:
   *   The distance in F26dot6 format.
   */
  static FT_F26Dot6
  Project( TT_ExecContext  exc,
           FT_Pos          dx,
           FT_Pos          dy )
  {
    return TT_DotFix14( dx, dy,
                        exc->GS.projVector.x,
                        exc->GS.projVector.y );
  }


  /**************************************************************************
   *
   * @Function:
   *   Dual_Project
   *
   * @Description:
   *   Computes the projection of the vector given by (v2-v1) along the
   *   current dual vector.
   *
   * @Input:
   *   v1 ::
   *     First input vector.
   *   v2 ::
   *     Second input vector.
   *
   * @Return:
   *   The distance in F26dot6 format.
   */
  static FT_F26Dot6
  Dual_Project( TT_ExecContext  exc,
                FT_Pos          dx,
                FT_Pos          dy )
  {
    return TT_DotFix14( dx, dy,
                        exc->GS.dualVector.x,
                        exc->GS.dualVector.y );
  }


  /**************************************************************************
   *
   * @Function:
   *   Project_x
   *
   * @Description:
   *   Computes the projection of the vector given by (v2-v1) along the
   *   horizontal axis.
   *
   * @Input:
   *   v1 ::
   *     First input vector.
   *   v2 ::
   *     Second input vector.
   *
   * @Return:
   *   The distance in F26dot6 format.
   */
  static FT_F26Dot6
  Project_x( TT_ExecContext  exc,
             FT_Pos          dx,
             FT_Pos          dy )
  {
    FT_UNUSED( exc );
    FT_UNUSED( dy );

    return dx;
  }


  /**************************************************************************
   *
   * @Function:
   *   Project_y
   *
   * @Description:
   *   Computes the projection of the vector given by (v2-v1) along the
   *   vertical axis.
   *
   * @Input:
   *   v1 ::
   *     First input vector.
   *   v2 ::
   *     Second input vector.
   *
   * @Return:
   *   The distance in F26dot6 format.
   */
  static FT_F26Dot6
  Project_y( TT_ExecContext  exc,
             FT_Pos          dx,
             FT_Pos          dy )
  {
    FT_UNUSED( exc );
    FT_UNUSED( dx );

    return dy;
  }


  /**************************************************************************
   *
   * @Function:
   *   Compute_Funcs
   *
   * @Description:
   *   Computes the projection and movement function pointers according
   *   to the current graphics state.
   */
  static void
  Compute_Funcs( TT_ExecContext  exc )
  {
    if ( exc->GS.freeVector.x == 0x4000 )
      exc->F_dot_P = exc->GS.projVector.x;
    else if ( exc->GS.freeVector.y == 0x4000 )
      exc->F_dot_P = exc->GS.projVector.y;
    else
      exc->F_dot_P =
        ( (FT_Long)exc->GS.projVector.x * exc->GS.freeVector.x +
          (FT_Long)exc->GS.projVector.y * exc->GS.freeVector.y ) >> 14;

    if ( exc->GS.projVector.x == 0x4000 )
      exc->func_project = (TT_Project_Func)Project_x;
    else if ( exc->GS.projVector.y == 0x4000 )
      exc->func_project = (TT_Project_Func)Project_y;
    else
      exc->func_project = (TT_Project_Func)Project;

    if ( exc->GS.dualVector.x == 0x4000 )
      exc->func_dualproj = (TT_Project_Func)Project_x;
    else if ( exc->GS.dualVector.y == 0x4000 )
      exc->func_dualproj = (TT_Project_Func)Project_y;
    else
      exc->func_dualproj = (TT_Project_Func)Dual_Project;

    exc->func_move      = (TT_Move_Func)Direct_Move;
    exc->func_move_orig = (TT_Move_Func)Direct_Move_Orig;

    if ( exc->F_dot_P == 0x4000L )
    {
      if ( exc->GS.freeVector.x == 0x4000 )
      {
        exc->func_move      = (TT_Move_Func)Direct_Move_X;
        exc->func_move_orig = (TT_Move_Func)Direct_Move_Orig_X;
      }
      else if ( exc->GS.freeVector.y == 0x4000 )
      {
        exc->func_move      = (TT_Move_Func)Direct_Move_Y;
        exc->func_move_orig = (TT_Move_Func)Direct_Move_Orig_Y;
      }
    }

    /* at small sizes, F_dot_P can become too small, resulting   */
    /* in overflows and `spikes' in a number of glyphs like `w'. */

    if ( FT_ABS( exc->F_dot_P ) < 0x400L )
      exc->F_dot_P = 0x4000L;

    /* Disable cached aspect ratio */
    exc->tt_metrics.ratio = 0;
  }


  /**************************************************************************
   *
   * @Function:
   *   Normalize
   *
   * @Description:
   *   Norms a vector.
   *
   * @Input:
   *   Vx ::
   *     The horizontal input vector coordinate.
   *   Vy ::
   *     The vertical input vector coordinate.
   *
   * @Output:
   *   R ::
   *     The normed unit vector.
   *
   * @Return:
   *   Returns FAILURE if a vector parameter is zero.
   *
   * @Note:
   *   In case Vx and Vy are both zero, `Normalize' returns SUCCESS, and
   *   R is undefined.
   */
  static FT_Bool
  Normalize( FT_F26Dot6      Vx,
             FT_F26Dot6      Vy,
             FT_UnitVector*  R )
  {
    FT_Vector V;


    if ( Vx == 0 && Vy == 0 )
    {
      /* XXX: UNDOCUMENTED! It seems that it is possible to try   */
      /*      to normalize the vector (0,0).  Return immediately. */
      return SUCCESS;
    }

    V.x = Vx;
    V.y = Vy;

    FT_Vector_NormLen( &V );

    R->x = (FT_F2Dot14)( V.x / 4 );
    R->y = (FT_F2Dot14)( V.y / 4 );

    return SUCCESS;
  }


  /**************************************************************************
   *
   * Here we start with the implementation of the various opcodes.
   *
   */


#define ARRAY_BOUND_ERROR                         \
    do                                            \
    {                                             \
      exc->error = FT_THROW( Invalid_Reference ); \
      return;                                     \
    } while (0)


  /**************************************************************************
   *
   * MPPEM[]:      Measure Pixel Per EM
   * Opcode range: 0x4B
   * Stack:        --> Euint16
   */
  static void
  Ins_MPPEM( TT_ExecContext  exc,
             FT_Long*        args )
  {
    args[0] = exc->func_cur_ppem( exc );
  }


  /**************************************************************************
   *
   * MPS[]:        Measure Point Size
   * Opcode range: 0x4C
   * Stack:        --> Euint16
   */
  static void
  Ins_MPS( TT_ExecContext  exc,
           FT_Long*        args )
  {
    if ( NO_SUBPIXEL_HINTING )
    {
      /* Microsoft's GDI bytecode interpreter always returns value 12; */
      /* we return the current PPEM value instead.                     */
      args[0] = exc->func_cur_ppem( exc );
    }
    else
    {
      /* A possible practical application of the MPS instruction is to   */
      /* implement optical scaling and similar features, which should be */
      /* based on perceptual attributes, thus independent of the         */
      /* resolution.                                                     */
      args[0] = exc->pointSize;
    }
  }


  /**************************************************************************
   *
   * DUP[]:        DUPlicate the stack's top element
   * Opcode range: 0x20
   * Stack:        StkElt --> StkElt StkElt
   */
  static void
  Ins_DUP( FT_Long*  args )
  {
    args[1] = args[0];
  }


  /**************************************************************************
   *
   * POP[]:        POP the stack's top element
   * Opcode range: 0x21
   * Stack:        StkElt -->
   */
  static void
  Ins_POP( void )
  {
    /* nothing to do */
  }


  /**************************************************************************
   *
   * CLEAR[]:      CLEAR the entire stack
   * Opcode range: 0x22
   * Stack:        StkElt... -->
   */
  static void
  Ins_CLEAR( TT_ExecContext  exc )
  {
    exc->new_top = 0;
  }


  /**************************************************************************
   *
   * SWAP[]:       SWAP the stack's top two elements
   * Opcode range: 0x23
   * Stack:        2 * StkElt --> 2 * StkElt
   */
  static void
  Ins_SWAP( FT_Long*  args )
  {
    FT_Long  L;


    L       = args[0];
    args[0] = args[1];
    args[1] = L;
  }


  /**************************************************************************
   *
   * DEPTH[]:      return the stack DEPTH
   * Opcode range: 0x24
   * Stack:        --> uint32
   */
  static void
  Ins_DEPTH( TT_ExecContext  exc,
             FT_Long*        args )
  {
    args[0] = exc->top;
  }


  /**************************************************************************
   *
   * LT[]:         Less Than
   * Opcode range: 0x50
   * Stack:        int32? int32? --> bool
   */
  static void
  Ins_LT( FT_Long*  args )
  {
    args[0] = ( args[0] < args[1] );
  }


  /**************************************************************************
   *
   * LTEQ[]:       Less Than or EQual
   * Opcode range: 0x51
   * Stack:        int32? int32? --> bool
   */
  static void
  Ins_LTEQ( FT_Long*  args )
  {
    args[0] = ( args[0] <= args[1] );
  }


  /**************************************************************************
   *
   * GT[]:         Greater Than
   * Opcode range: 0x52
   * Stack:        int32? int32? --> bool
   */
  static void
  Ins_GT( FT_Long*  args )
  {
    args[0] = ( args[0] > args[1] );
  }


  /**************************************************************************
   *
   * GTEQ[]:       Greater Than or EQual
   * Opcode range: 0x53
   * Stack:        int32? int32? --> bool
   */
  static void
  Ins_GTEQ( FT_Long*  args )
  {
    args[0] = ( args[0] >= args[1] );
  }


  /**************************************************************************
   *
   * EQ[]:         EQual
   * Opcode range: 0x54
   * Stack:        StkElt StkElt --> bool
   */
  static void
  Ins_EQ( FT_Long*  args )
  {
    args[0] = ( args[0] == args[1] );
  }


  /**************************************************************************
   *
   * NEQ[]:        Not EQual
   * Opcode range: 0x55
   * Stack:        StkElt StkElt --> bool
   */
  static void
  Ins_NEQ( FT_Long*  args )
  {
    args[0] = ( args[0] != args[1] );
  }


  /**************************************************************************
   *
   * ODD[]:        Is ODD
   * Opcode range: 0x56
   * Stack:        f26.6 --> bool
   */
  static void
  Ins_ODD( TT_ExecContext  exc,
           FT_Long*        args )
  {
    args[0] = ( ( exc->func_round( exc, args[0], 3 ) & 127 ) == 64 );
  }


  /**************************************************************************
   *
   * EVEN[]:       Is EVEN
   * Opcode range: 0x57
   * Stack:        f26.6 --> bool
   */
  static void
  Ins_EVEN( TT_ExecContext  exc,
            FT_Long*        args )
  {
    args[0] = ( ( exc->func_round( exc, args[0], 3 ) & 127 ) == 0 );
  }


  /**************************************************************************
   *
   * AND[]:        logical AND
   * Opcode range: 0x5A
   * Stack:        uint32 uint32 --> uint32
   */
  static void
  Ins_AND( FT_Long*  args )
  {
    args[0] = ( args[0] && args[1] );
  }


  /**************************************************************************
   *
   * OR[]:         logical OR
   * Opcode range: 0x5B
   * Stack:        uint32 uint32 --> uint32
   */
  static void
  Ins_OR( FT_Long*  args )
  {
    args[0] = ( args[0] || args[1] );
  }


  /**************************************************************************
   *
   * NOT[]:        logical NOT
   * Opcode range: 0x5C
   * Stack:        StkElt --> uint32
   */
  static void
  Ins_NOT( FT_Long*  args )
  {
    args[0] = !args[0];
  }


  /**************************************************************************
   *
   * ADD[]:        ADD
   * Opcode range: 0x60
   * Stack:        f26.6 f26.6 --> f26.6
   */
  static void
  Ins_ADD( FT_Long*  args )
  {
    args[0] = ADD_LONG( args[0], args[1] );
  }


  /**************************************************************************
   *
   * SUB[]:        SUBtract
   * Opcode range: 0x61
   * Stack:        f26.6 f26.6 --> f26.6
   */
  static void
  Ins_SUB( FT_Long*  args )
  {
    args[0] = SUB_LONG( args[0], args[1] );
  }


  /**************************************************************************
   *
   * DIV[]:        DIVide
   * Opcode range: 0x62
   * Stack:        f26.6 f26.6 --> f26.6
   */
  static void
  Ins_DIV( TT_ExecContext  exc,
           FT_Long*        args )
  {
    if ( args[1] == 0 )
      exc->error = FT_THROW( Divide_By_Zero );
    else
      args[0] = FT_MulDiv_No_Round( args[0], 64L, args[1] );
  }


  /**************************************************************************
   *
   * MUL[]:        MULtiply
   * Opcode range: 0x63
   * Stack:        f26.6 f26.6 --> f26.6
   */
  static void
  Ins_MUL( FT_Long*  args )
  {
    args[0] = FT_MulDiv( args[0], args[1], 64L );
  }


  /**************************************************************************
   *
   * ABS[]:        ABSolute value
   * Opcode range: 0x64
   * Stack:        f26.6 --> f26.6
   */
  static void
  Ins_ABS( FT_Long*  args )
  {
    if ( args[0] < 0 )
      args[0] = NEG_LONG( args[0] );
  }


  /**************************************************************************
   *
   * NEG[]:        NEGate
   * Opcode range: 0x65
   * Stack:        f26.6 --> f26.6
   */
  static void
  Ins_NEG( FT_Long*  args )
  {
    args[0] = NEG_LONG( args[0] );
  }


  /**************************************************************************
   *
   * FLOOR[]:      FLOOR
   * Opcode range: 0x66
   * Stack:        f26.6 --> f26.6
   */
  static void
  Ins_FLOOR( FT_Long*  args )
  {
    args[0] = FT_PIX_FLOOR( args[0] );
  }


  /**************************************************************************
   *
   * CEILING[]:    CEILING
   * Opcode range: 0x67
   * Stack:        f26.6 --> f26.6
   */
  static void
  Ins_CEILING( FT_Long*  args )
  {
    args[0] = FT_PIX_CEIL_LONG( args[0] );
  }


  /**************************************************************************
   *
   * RS[]:         Read Store
   * Opcode range: 0x43
   * Stack:        uint32 --> uint32
   */
  static void
  Ins_RS( TT_ExecContext  exc,
          FT_Long*        args )
  {
    FT_ULong  I = (FT_ULong)args[0];


    if ( BOUNDSL( I, exc->storeSize ) )
    {
      if ( exc->pedantic_hinting )
        ARRAY_BOUND_ERROR;
      else
        args[0] = 0;
    }
    else
    {
#ifdef TT_SUPPORT_SUBPIXEL_HINTING_INFINALITY
      /* subpixel hinting - avoid Typeman Dstroke and */
      /* IStroke and Vacuform rounds                  */
      if ( SUBPIXEL_HINTING_INFINALITY                 &&
           exc->ignore_x_mode                          &&
           ( ( I == 24                             &&
               ( exc->face->sph_found_func_flags &
                 ( SPH_FDEF_SPACING_1 |
                   SPH_FDEF_SPACING_2 )          ) ) ||
             ( I == 22                      &&
               ( exc->sph_in_func_flags   &
                 SPH_FDEF_TYPEMAN_STROKES ) )        ||
             ( I == 8                              &&
               ( exc->face->sph_found_func_flags &
                 SPH_FDEF_VACUFORM_ROUND_1       ) &&
               exc->iup_called                     ) ) )
        args[0] = 0;
      else
#endif /* TT_SUPPORT_SUBPIXEL_HINTING_INFINALITY */
        args[0] = exc->storage[I];
    }
  }


  /**************************************************************************
   *
   * WS[]:         Write Store
   * Opcode range: 0x42
   * Stack:        uint32 uint32 -->
   */
  static void
  Ins_WS( TT_ExecContext  exc,
          FT_Long*        args )
  {
    FT_ULong  I = (FT_ULong)args[0];


    if ( BOUNDSL( I, exc->storeSize ) )
    {
      if ( exc->pedantic_hinting )
        ARRAY_BOUND_ERROR;
    }
    else
    {
      /* TT_RunIns sets origStorage and restores storage to origStorage */
      /* when done.                                                     */
      if ( exc->iniRange == tt_coderange_glyph &&
           exc->storage == exc->origStorage    )
      {
        FT_ULong  tmp = (FT_ULong)exc->glyfStoreSize;


        exc->error = Update_Max( exc->memory,
                                 &tmp,
                                 sizeof ( FT_Long ),
                                 (void*)&exc->glyfStorage,
                                 exc->storeSize );
        exc->glyfStoreSize = (FT_UShort)tmp;
        if ( exc->error )
          return;

        FT_ARRAY_COPY( exc->glyfStorage, exc->storage, exc->glyfStoreSize );
        exc->storage = exc->glyfStorage;
      }

      exc->storage[I] = args[1];
    }
  }


  /**************************************************************************
   *
   * WCVTP[]:      Write CVT in Pixel units
   * Opcode range: 0x44
   * Stack:        f26.6 uint32 -->
   */
  static void
  Ins_WCVTP( TT_ExecContext  exc,
             FT_Long*        args )
  {
    FT_ULong  I = (FT_ULong)args[0];


    if ( BOUNDSL( I, exc->cvtSize ) )
    {
      if ( exc->pedantic_hinting )
        ARRAY_BOUND_ERROR;
    }
    else
      exc->func_write_cvt( exc, I, args[1] );
  }


  /**************************************************************************
   *
   * WCVTF[]:      Write CVT in Funits
   * Opcode range: 0x70
   * Stack:        uint32 uint32 -->
   */
  static void
  Ins_WCVTF( TT_ExecContext  exc,
             FT_Long*        args )
  {
    FT_ULong  I = (FT_ULong)args[0];


    if ( BOUNDSL( I, exc->cvtSize ) )
    {
      if ( exc->pedantic_hinting )
        ARRAY_BOUND_ERROR;
    }
    else
      exc->cvt[I] = FT_MulFix( args[1], exc->tt_metrics.scale );
  }


  /**************************************************************************
   *
   * RCVT[]:       Read CVT
   * Opcode range: 0x45
   * Stack:        uint32 --> f26.6
   */
  static void
  Ins_RCVT( TT_ExecContext  exc,
            FT_Long*        args )
  {
    FT_ULong  I = (FT_ULong)args[0];


    if ( BOUNDSL( I, exc->cvtSize ) )
    {
      if ( exc->pedantic_hinting )
        ARRAY_BOUND_ERROR;
      else
        args[0] = 0;
    }
    else
      args[0] = exc->func_read_cvt( exc, I );
  }


  /**************************************************************************
   *
   * AA[]:         Adjust Angle
   * Opcode range: 0x7F
   * Stack:        uint32 -->
   */
  static void
  Ins_AA( void )
  {
    /* intentionally no longer supported */
  }


  /**************************************************************************
   *
   * DEBUG[]:      DEBUG.  Unsupported.
   * Opcode range: 0x4F
   * Stack:        uint32 -->
   *
   * Note: The original instruction pops a value from the stack.
   */
  static void
  Ins_DEBUG( TT_ExecContext  exc )
  {
    exc->error = FT_THROW( Debug_OpCode );
  }


  /**************************************************************************
   *
   * ROUND[ab]:    ROUND value
   * Opcode range: 0x68-0x6B
   * Stack:        f26.6 --> f26.6
   */
  static void
  Ins_ROUND( TT_ExecContext  exc,
             FT_Long*        args )
  {
    args[0] = exc->func_round( exc, args[0], exc->opcode & 3 );
  }


  /**************************************************************************
   *
   * NROUND[ab]:   No ROUNDing of value
   * Opcode range: 0x6C-0x6F
   * Stack:        f26.6 --> f26.6
   */
  static void
  Ins_NROUND( TT_ExecContext  exc,
              FT_Long*        args )
  {
    args[0] = Round_None( exc, args[0], exc->opcode & 3 );
  }


  /**************************************************************************
   *
   * MAX[]:        MAXimum
   * Opcode range: 0x8B
   * Stack:        int32? int32? --> int32
   */
  static void
  Ins_MAX( FT_Long*  args )
  {
    if ( args[1] > args[0] )
      args[0] = args[1];
  }


  /**************************************************************************
   *
   * MIN[]:        MINimum
   * Opcode range: 0x8C
   * Stack:        int32? int32? --> int32
   */
  static void
  Ins_MIN( FT_Long*  args )
  {
    if ( args[1] < args[0] )
      args[0] = args[1];
  }


  /**************************************************************************
   *
   * MINDEX[]:     Move INDEXed element
   * Opcode range: 0x26
   * Stack:        int32? --> StkElt
   */
  static void
  Ins_MINDEX( TT_ExecContext  exc,
              FT_Long*        args )
  {
    FT_Long  L, K;


    L = args[0];

    if ( L <= 0 || L > exc->args )
    {
      if ( exc->pedantic_hinting )
        exc->error = FT_THROW( Invalid_Reference );
    }
    else
    {
      K = exc->stack[exc->args - L];

      FT_ARRAY_MOVE( &exc->stack[exc->args - L    ],
                     &exc->stack[exc->args - L + 1],
                     ( L - 1 ) );

      exc->stack[exc->args - 1] = K;
    }
  }


  /**************************************************************************
   *
   * CINDEX[]:     Copy INDEXed element
   * Opcode range: 0x25
   * Stack:        int32 --> StkElt
   */
  static void
  Ins_CINDEX( TT_ExecContext  exc,
              FT_Long*        args )
  {
    FT_Long  L;


    L = args[0];

    if ( L <= 0 || L > exc->args )
    {
      if ( exc->pedantic_hinting )
        exc->error = FT_THROW( Invalid_Reference );
      args[0] = 0;
    }
    else
      args[0] = exc->stack[exc->args - L];
  }


  /**************************************************************************
   *
   * ROLL[]:       ROLL top three elements
   * Opcode range: 0x8A
   * Stack:        3 * StkElt --> 3 * StkElt
   */
  static void
  Ins_ROLL( FT_Long*  args )
  {
    FT_Long  A, B, C;


    A = args[2];
    B = args[1];
    C = args[0];

    args[2] = C;
    args[1] = A;
    args[0] = B;
  }


  /**************************************************************************
   *
   * MANAGING THE FLOW OF CONTROL
   *
   */


  /**************************************************************************
   *
   * SLOOP[]:      Set LOOP variable
   * Opcode range: 0x17
   * Stack:        int32? -->
   */
  static void
  Ins_SLOOP( TT_ExecContext  exc,
             FT_Long*        args )
  {
    if ( args[0] < 0 )
      exc->error = FT_THROW( Bad_Argument );
    else
    {
      /* we heuristically limit the number of loops to 16 bits */
      exc->GS.loop = args[0] > 0xFFFFL ? 0xFFFFL : args[0];
    }
  }


  static FT_Bool
  SkipCode( TT_ExecContext  exc )
  {
    exc->IP += exc->length;

    if ( exc->IP < exc->codeSize )
    {
      exc->opcode = exc->code[exc->IP];

      exc->length = opcode_length[exc->opcode];
      if ( exc->length < 0 )
      {
        if ( exc->IP + 1 >= exc->codeSize )
          goto Fail_Overflow;
        exc->length = 2 - exc->length * exc->code[exc->IP + 1];
      }

      if ( exc->IP + exc->length <= exc->codeSize )
        return SUCCESS;
    }

  Fail_Overflow:
    exc->error = FT_THROW( Code_Overflow );
    return FAILURE;
  }


  /**************************************************************************
   *
   * IF[]:         IF test
   * Opcode range: 0x58
   * Stack:        StkElt -->
   */
  static void
  Ins_IF( TT_ExecContext  exc,
          FT_Long*        args )
  {
    FT_Int   nIfs;
    FT_Bool  Out;


    if ( args[0] != 0 )
      return;

    nIfs = 1;
    Out = 0;

    do
    {
      if ( SkipCode( exc ) == FAILURE )
        return;

      switch ( exc->opcode )
      {
      case 0x58:      /* IF */
        nIfs++;
        break;

      case 0x1B:      /* ELSE */
        Out = FT_BOOL( nIfs == 1 );
        break;

      case 0x59:      /* EIF */
        nIfs--;
        Out = FT_BOOL( nIfs == 0 );
        break;
      }
    } while ( Out == 0 );
  }


  /**************************************************************************
   *
   * ELSE[]:       ELSE
   * Opcode range: 0x1B
   * Stack:        -->
   */
  static void
  Ins_ELSE( TT_ExecContext  exc )
  {
    FT_Int  nIfs;


    nIfs = 1;

    do
    {
      if ( SkipCode( exc ) == FAILURE )
        return;

      switch ( exc->opcode )
      {
      case 0x58:    /* IF */
        nIfs++;
        break;

      case 0x59:    /* EIF */
        nIfs--;
        break;
      }
    } while ( nIfs != 0 );
  }


  /**************************************************************************
   *
   * EIF[]:        End IF
   * Opcode range: 0x59
   * Stack:        -->
   */
  static void
  Ins_EIF( void )
  {
    /* nothing to do */
  }


  /**************************************************************************
   *
   * JMPR[]:       JuMP Relative
   * Opcode range: 0x1C
   * Stack:        int32 -->
   */
  static void
  Ins_JMPR( TT_ExecContext  exc,
            FT_Long*        args )
  {
    if ( args[0] == 0 && exc->args == 0 )
    {
      exc->error = FT_THROW( Bad_Argument );
      return;
    }

    exc->IP = ADD_LONG( exc->IP, args[0] );
    if ( exc->IP < 0                                             ||
         ( exc->callTop > 0                                    &&
           exc->IP > exc->callStack[exc->callTop - 1].Def->end ) )
    {
      exc->error = FT_THROW( Bad_Argument );
      return;
    }

    exc->step_ins = FALSE;

    if ( args[0] < 0 )
    {
      if ( ++exc->neg_jump_counter > exc->neg_jump_counter_max )
        exc->error = FT_THROW( Execution_Too_Long );
    }
  }


  /**************************************************************************
   *
   * JROT[]:       Jump Relative On True
   * Opcode range: 0x78
   * Stack:        StkElt int32 -->
   */
  static void
  Ins_JROT( TT_ExecContext  exc,
            FT_Long*        args )
  {
    if ( args[1] != 0 )
      Ins_JMPR( exc, args );
  }


  /**************************************************************************
   *
   * JROF[]:       Jump Relative On False
   * Opcode range: 0x79
   * Stack:        StkElt int32 -->
   */
  static void
  Ins_JROF( TT_ExecContext  exc,
            FT_Long*        args )
  {
    if ( args[1] == 0 )
      Ins_JMPR( exc, args );
  }


  /**************************************************************************
   *
   * DEFINING AND USING FUNCTIONS AND INSTRUCTIONS
   *
   */


  /**************************************************************************
   *
   * FDEF[]:       Function DEFinition
   * Opcode range: 0x2C
   * Stack:        uint32 -->
   */
  static void
  Ins_FDEF( TT_ExecContext  exc,
            FT_Long*        args )
  {
    FT_ULong       n;
    TT_DefRecord*  rec;
    TT_DefRecord*  limit;

#ifdef TT_SUPPORT_SUBPIXEL_HINTING_INFINALITY
    /* arguments to opcodes are skipped by `SKIP_Code' */
    FT_Byte    opcode_pattern[9][12] = {
                 /* #0 inline delta function 1 */
                 {
                   0x4B, /* PPEM    */
                   0x53, /* GTEQ    */
                   0x23, /* SWAP    */
                   0x4B, /* PPEM    */
                   0x51, /* LTEQ    */
                   0x5A, /* AND     */
                   0x58, /* IF      */
                   0x38, /*   SHPIX */
                   0x1B, /* ELSE    */
                   0x21, /*   POP   */
                   0x21, /*   POP   */
                   0x59  /* EIF     */
                 },
                 /* #1 inline delta function 2 */
                 {
                   0x4B, /* PPEM    */
                   0x54, /* EQ      */
                   0x58, /* IF      */
                   0x38, /*   SHPIX */
                   0x1B, /* ELSE    */
                   0x21, /*   POP   */
                   0x21, /*   POP   */
                   0x59  /* EIF     */
                 },
                 /* #2 diagonal stroke function */
                 {
                   0x20, /* DUP     */
                   0x20, /* DUP     */
                   0xB0, /* PUSHB_1 */
                         /*   1     */
                   0x60, /* ADD     */
                   0x46, /* GC_cur  */
                   0xB0, /* PUSHB_1 */
                         /*   64    */
                   0x23, /* SWAP    */
                   0x42  /* WS      */
                 },
                 /* #3 VacuFormRound function */
                 {
                   0x45, /* RCVT    */
                   0x23, /* SWAP    */
                   0x46, /* GC_cur  */
                   0x60, /* ADD     */
                   0x20, /* DUP     */
                   0xB0  /* PUSHB_1 */
                         /*   38    */
                 },
                 /* #4 TTFautohint bytecode (old) */
                 {
                   0x20, /* DUP     */
                   0x64, /* ABS     */
                   0xB0, /* PUSHB_1 */
                         /*   32    */
                   0x60, /* ADD     */
                   0x66, /* FLOOR   */
                   0x23, /* SWAP    */
                   0xB0  /* PUSHB_1 */
                 },
                 /* #5 spacing function 1 */
                 {
                   0x01, /* SVTCA_x */
                   0xB0, /* PUSHB_1 */
                         /*   24    */
                   0x43, /* RS      */
                   0x58  /* IF      */
                 },
                 /* #6 spacing function 2 */
                 {
                   0x01, /* SVTCA_x */
                   0x18, /* RTG     */
                   0xB0, /* PUSHB_1 */
                         /*   24    */
                   0x43, /* RS      */
                   0x58  /* IF      */
                 },
                 /* #7 TypeMan Talk DiagEndCtrl function */
                 {
                   0x01, /* SVTCA_x */
                   0x20, /* DUP     */
                   0xB0, /* PUSHB_1 */
                         /*   3     */
                   0x25, /* CINDEX  */
                 },
                 /* #8 TypeMan Talk Align */
                 {
                   0x06, /* SPVTL   */
                   0x7D, /* RDTG    */
                 },
               };
    FT_UShort  opcode_patterns   = 9;
    FT_UShort  opcode_pointer[9] = {  0, 0, 0, 0, 0, 0, 0, 0, 0 };
    FT_UShort  opcode_size[9]    = { 12, 8, 8, 6, 7, 4, 5, 4, 2 };
    FT_UShort  i;
#endif /* TT_SUPPORT_SUBPIXEL_HINTING_INFINALITY */


    /* FDEF is only allowed in `prep' or `fpgm' */
    if ( exc->iniRange == tt_coderange_glyph )
    {
      exc->error = FT_THROW( DEF_In_Glyf_Bytecode );
      return;
    }

    /* some font programs are broken enough to redefine functions! */
    /* We will then parse the current table.                       */

    rec   = exc->FDefs;
    limit = FT_OFFSET( rec, exc->numFDefs );
    n     = (FT_ULong)args[0];

    for ( ; rec < limit; rec++ )
    {
      if ( rec->opc == n )
        break;
    }

    if ( rec == limit )
    {
      /* check that there is enough room for new functions */
      if ( exc->numFDefs >= exc->maxFDefs )
      {
        exc->error = FT_THROW( Too_Many_Function_Defs );
        return;
      }
      exc->numFDefs++;
    }

    /* Although FDEF takes unsigned 32-bit integer,  */
    /* func # must be within unsigned 16-bit integer */
    if ( n > 0xFFFFU )
    {
      exc->error = FT_THROW( Too_Many_Function_Defs );
      return;
    }

    rec->range          = exc->curRange;
    rec->opc            = (FT_UInt16)n;
    rec->start          = exc->IP + 1;
    rec->active         = TRUE;
    rec->inline_delta   = FALSE;
    rec->sph_fdef_flags = 0x0000;

    if ( n > exc->maxFunc )
      exc->maxFunc = (FT_UInt16)n;

#ifdef TT_SUPPORT_SUBPIXEL_HINTING_INFINALITY
    /* We don't know for sure these are typeman functions, */
    /* however they are only active when RS 22 is called   */
    if ( n >= 64 && n <= 66 )
      rec->sph_fdef_flags |= SPH_FDEF_TYPEMAN_STROKES;
#endif

    /* Now skip the whole function definition. */
    /* We don't allow nested IDEFS & FDEFs.    */

    while ( SkipCode( exc ) == SUCCESS )
    {

#ifdef TT_SUPPORT_SUBPIXEL_HINTING_INFINALITY

      if ( SUBPIXEL_HINTING_INFINALITY )
      {
        for ( i = 0; i < opcode_patterns; i++ )
        {
          if ( opcode_pointer[i] < opcode_size[i]                  &&
               exc->opcode == opcode_pattern[i][opcode_pointer[i]] )
          {
            opcode_pointer[i] += 1;

            if ( opcode_pointer[i] == opcode_size[i] )
            {
              FT_TRACE6(( "sph: Function %d, opcode ptrn: %ld, %s %s\n",
                          i, n,
                          exc->face->root.family_name,
                          exc->face->root.style_name ));

              switch ( i )
              {
              case 0:
                rec->sph_fdef_flags             |= SPH_FDEF_INLINE_DELTA_1;
                exc->face->sph_found_func_flags |= SPH_FDEF_INLINE_DELTA_1;
                break;

              case 1:
                rec->sph_fdef_flags             |= SPH_FDEF_INLINE_DELTA_2;
                exc->face->sph_found_func_flags |= SPH_FDEF_INLINE_DELTA_2;
                break;

              case 2:
                switch ( n )
                {
                  /* needs to be implemented still */
                case 58:
                  rec->sph_fdef_flags             |= SPH_FDEF_DIAGONAL_STROKE;
                  exc->face->sph_found_func_flags |= SPH_FDEF_DIAGONAL_STROKE;
                }
                break;

              case 3:
                switch ( n )
                {
                case 0:
                  rec->sph_fdef_flags             |= SPH_FDEF_VACUFORM_ROUND_1;
                  exc->face->sph_found_func_flags |= SPH_FDEF_VACUFORM_ROUND_1;
                }
                break;

              case 4:
                /* probably not necessary to detect anymore */
                rec->sph_fdef_flags             |= SPH_FDEF_TTFAUTOHINT_1;
                exc->face->sph_found_func_flags |= SPH_FDEF_TTFAUTOHINT_1;
                break;

              case 5:
                switch ( n )
                {
                case 0:
                case 1:
                case 2:
                case 4:
                case 7:
                case 8:
                  rec->sph_fdef_flags             |= SPH_FDEF_SPACING_1;
                  exc->face->sph_found_func_flags |= SPH_FDEF_SPACING_1;
                }
                break;

              case 6:
                switch ( n )
                {
                case 0:
                case 1:
                case 2:
                case 4:
                case 7:
                case 8:
                  rec->sph_fdef_flags             |= SPH_FDEF_SPACING_2;
                  exc->face->sph_found_func_flags |= SPH_FDEF_SPACING_2;
                }
                break;

               case 7:
                 rec->sph_fdef_flags             |= SPH_FDEF_TYPEMAN_DIAGENDCTRL;
                 exc->face->sph_found_func_flags |= SPH_FDEF_TYPEMAN_DIAGENDCTRL;
                 break;

               case 8:
#if 0
                 rec->sph_fdef_flags             |= SPH_FDEF_TYPEMAN_DIAGENDCTRL;
                 exc->face->sph_found_func_flags |= SPH_FDEF_TYPEMAN_DIAGENDCTRL;
#endif
                 break;
              }
              opcode_pointer[i] = 0;
            }
          }

          else
            opcode_pointer[i] = 0;
        }

        /* Set sph_compatibility_mode only when deltas are detected */
        exc->face->sph_compatibility_mode =
          ( ( exc->face->sph_found_func_flags & SPH_FDEF_INLINE_DELTA_1 ) |
            ( exc->face->sph_found_func_flags & SPH_FDEF_INLINE_DELTA_2 ) );
      }

#endif /* TT_SUPPORT_SUBPIXEL_HINTING_INFINALITY */

      switch ( exc->opcode )
      {
      case 0x89:    /* IDEF */
      case 0x2C:    /* FDEF */
        exc->error = FT_THROW( Nested_DEFS );
        return;

      case 0x2D:   /* ENDF */
        rec->end = exc->IP;
        return;
      }
    }
  }


  /**************************************************************************
   *
   * ENDF[]:       END Function definition
   * Opcode range: 0x2D
   * Stack:        -->
   */
  static void
  Ins_ENDF( TT_ExecContext  exc )
  {
    TT_CallRec*  pRec;


#ifdef TT_SUPPORT_SUBPIXEL_HINTING_INFINALITY
    exc->sph_in_func_flags = 0x0000;
#endif /* TT_SUPPORT_SUBPIXEL_HINTING_INFINALITY */

    if ( exc->callTop <= 0 )     /* We encountered an ENDF without a call */
    {
      exc->error = FT_THROW( ENDF_In_Exec_Stream );
      return;
    }

    exc->callTop--;

    pRec = &exc->callStack[exc->callTop];

    pRec->Cur_Count--;

    exc->step_ins = FALSE;

    if ( pRec->Cur_Count > 0 )
    {
      exc->callTop++;
      exc->IP = pRec->Def->start;
    }
    else
      /* Loop through the current function */
      Ins_Goto_CodeRange( exc, pRec->Caller_Range, pRec->Caller_IP );

    /* Exit the current call frame.                      */

    /* NOTE: If the last instruction of a program is a   */
    /*       CALL or LOOPCALL, the return address is     */
    /*       always out of the code range.  This is a    */
    /*       valid address, and it is why we do not test */
    /*       the result of Ins_Goto_CodeRange() here!    */
  }


  /**************************************************************************
   *
   * CALL[]:       CALL function
   * Opcode range: 0x2B
   * Stack:        uint32? -->
   */
  static void
  Ins_CALL( TT_ExecContext  exc,
            FT_Long*        args )
  {
    FT_ULong       F;
    TT_CallRec*    pCrec;
    TT_DefRecord*  def;


    /* first of all, check the index */

    F = (FT_ULong)args[0];
    if ( BOUNDSL( F, exc->maxFunc + 1 ) )
      goto Fail;

    if ( !exc->FDefs )
      goto Fail;

    /* Except for some old Apple fonts, all functions in a TrueType */
    /* font are defined in increasing order, starting from 0.  This */
    /* means that we normally have                                  */
    /*                                                              */
    /*    exc->maxFunc+1 == exc->numFDefs                           */
    /*    exc->FDefs[n].opc == n for n in 0..exc->maxFunc           */
    /*                                                              */
    /* If this isn't true, we need to look up the function table.   */

    def = exc->FDefs + F;
    if ( exc->maxFunc + 1 != exc->numFDefs || def->opc != F )
    {
      /* look up the FDefs table */
      TT_DefRecord*  limit;


      def   = exc->FDefs;
      limit = def + exc->numFDefs;

      while ( def < limit && def->opc != F )
        def++;

      if ( def == limit )
        goto Fail;
    }

    /* check that the function is active */
    if ( !def->active )
      goto Fail;

#ifdef TT_SUPPORT_SUBPIXEL_HINTING_INFINALITY
    if ( SUBPIXEL_HINTING_INFINALITY                                    &&
         exc->ignore_x_mode                                             &&
         ( ( exc->iup_called                                        &&
             ( exc->sph_tweak_flags & SPH_TWEAK_NO_CALL_AFTER_IUP ) ) ||
           ( def->sph_fdef_flags & SPH_FDEF_VACUFORM_ROUND_1 )        ) )
      goto Fail;
    else
      exc->sph_in_func_flags = def->sph_fdef_flags;
#endif /* TT_SUPPORT_SUBPIXEL_HINTING_INFINALITY */

    /* check the call stack */
    if ( exc->callTop >= exc->callSize )
    {
      exc->error = FT_THROW( Stack_Overflow );
      return;
    }

    pCrec = exc->callStack + exc->callTop;

    pCrec->Caller_Range = exc->curRange;
    pCrec->Caller_IP    = exc->IP + 1;
    pCrec->Cur_Count    = 1;
    pCrec->Def          = def;

    exc->callTop++;

    Ins_Goto_CodeRange( exc, def->range, def->start );

    exc->step_ins = FALSE;

    return;

  Fail:
    exc->error = FT_THROW( Invalid_Reference );
  }


  /**************************************************************************
   *
   * LOOPCALL[]:   LOOP and CALL function
   * Opcode range: 0x2A
   * Stack:        uint32? Eint16? -->
   */
  static void
  Ins_LOOPCALL( TT_ExecContext  exc,
                FT_Long*        args )
  {
    FT_ULong       F;
    TT_CallRec*    pCrec;
    TT_DefRecord*  def;


    /* first of all, check the index */
    F = (FT_ULong)args[1];
    if ( BOUNDSL( F, exc->maxFunc + 1 ) )
      goto Fail;

    /* Except for some old Apple fonts, all functions in a TrueType */
    /* font are defined in increasing order, starting from 0.  This */
    /* means that we normally have                                  */
    /*                                                              */
    /*    exc->maxFunc+1 == exc->numFDefs                           */
    /*    exc->FDefs[n].opc == n for n in 0..exc->maxFunc           */
    /*                                                              */
    /* If this isn't true, we need to look up the function table.   */

    def = FT_OFFSET( exc->FDefs, F );
    if ( exc->maxFunc + 1 != exc->numFDefs || def->opc != F )
    {
      /* look up the FDefs table */
      TT_DefRecord*  limit;


      def   = exc->FDefs;
      limit = FT_OFFSET( def, exc->numFDefs );

      while ( def < limit && def->opc != F )
        def++;

      if ( def == limit )
        goto Fail;
    }

    /* check that the function is active */
    if ( !def->active )
      goto Fail;

#ifdef TT_SUPPORT_SUBPIXEL_HINTING_INFINALITY
    if ( SUBPIXEL_HINTING_INFINALITY                         &&
         exc->ignore_x_mode                                  &&
         ( def->sph_fdef_flags & SPH_FDEF_VACUFORM_ROUND_1 ) )
      goto Fail;
    else
      exc->sph_in_func_flags = def->sph_fdef_flags;
#endif /* TT_SUPPORT_SUBPIXEL_HINTING_INFINALITY */

    /* check stack */
    if ( exc->callTop >= exc->callSize )
    {
      exc->error = FT_THROW( Stack_Overflow );
      return;
    }

    if ( args[0] > 0 )
    {
      pCrec = exc->callStack + exc->callTop;

      pCrec->Caller_Range = exc->curRange;
      pCrec->Caller_IP    = exc->IP + 1;
      pCrec->Cur_Count    = (FT_Int)args[0];
      pCrec->Def          = def;

      exc->callTop++;

      Ins_Goto_CodeRange( exc, def->range, def->start );

      exc->step_ins = FALSE;

      exc->loopcall_counter += (FT_ULong)args[0];
      if ( exc->loopcall_counter > exc->loopcall_counter_max )
        exc->error = FT_THROW( Execution_Too_Long );
    }

    return;

  Fail:
    exc->error = FT_THROW( Invalid_Reference );
  }


  /**************************************************************************
   *
   * IDEF[]:       Instruction DEFinition
   * Opcode range: 0x89
   * Stack:        Eint8 -->
   */
  static void
  Ins_IDEF( TT_ExecContext  exc,
            FT_Long*        args )
  {
    TT_DefRecord*  def;
    TT_DefRecord*  limit;


    /* we enable IDEF only in `prep' or `fpgm' */
    if ( exc->iniRange == tt_coderange_glyph )
    {
      exc->error = FT_THROW( DEF_In_Glyf_Bytecode );
      return;
    }

    /*  First of all, look for the same function in our table */

    def   = exc->IDefs;
    limit = FT_OFFSET( def, exc->numIDefs );

    for ( ; def < limit; def++ )
      if ( def->opc == (FT_ULong)args[0] )
        break;

    if ( def == limit )
    {
      /* check that there is enough room for a new instruction */
      if ( exc->numIDefs >= exc->maxIDefs )
      {
        exc->error = FT_THROW( Too_Many_Instruction_Defs );
        return;
      }
      exc->numIDefs++;
    }

    /* opcode must be unsigned 8-bit integer */
    if ( 0 > args[0] || args[0] > 0x00FF )
    {
      exc->error = FT_THROW( Too_Many_Instruction_Defs );
      return;
    }

    def->opc    = (FT_Byte)args[0];
    def->start  = exc->IP + 1;
    def->range  = exc->curRange;
    def->active = TRUE;

    if ( (FT_ULong)args[0] > exc->maxIns )
      exc->maxIns = (FT_Byte)args[0];

    /* Now skip the whole function definition. */
    /* We don't allow nested IDEFs & FDEFs.    */

    while ( SkipCode( exc ) == SUCCESS )
    {
      switch ( exc->opcode )
      {
      case 0x89:   /* IDEF */
      case 0x2C:   /* FDEF */
        exc->error = FT_THROW( Nested_DEFS );
        return;
      case 0x2D:   /* ENDF */
        def->end = exc->IP;
        return;
      }
    }
  }


  /**************************************************************************
   *
   * PUSHING DATA ONTO THE INTERPRETER STACK
   *
   */


  /**************************************************************************
   *
   * NPUSHB[]:     PUSH N Bytes
   * Opcode range: 0x40
   * Stack:        --> uint32...
   */
  static void
  Ins_NPUSHB( TT_ExecContext  exc,
              FT_Long*        args )
  {
    FT_UShort  L, K;


    L = (FT_UShort)exc->code[exc->IP + 1];

    if ( BOUNDS( L, exc->stackSize + 1 - exc->top ) )
    {
      exc->error = FT_THROW( Stack_Overflow );
      return;
    }

    for ( K = 1; K <= L; K++ )
      args[K - 1] = exc->code[exc->IP + K + 1];

    exc->new_top += L;
  }


  /**************************************************************************
   *
   * NPUSHW[]:     PUSH N Words
   * Opcode range: 0x41
   * Stack:        --> int32...
   */
  static void
  Ins_NPUSHW( TT_ExecContext  exc,
              FT_Long*        args )
  {
    FT_UShort  L, K;


    L = (FT_UShort)exc->code[exc->IP + 1];

    if ( BOUNDS( L, exc->stackSize + 1 - exc->top ) )
    {
      exc->error = FT_THROW( Stack_Overflow );
      return;
    }

    exc->IP += 2;

    for ( K = 0; K < L; K++ )
      args[K] = GetShortIns( exc );

    exc->step_ins = FALSE;
    exc->new_top += L;
  }


  /**************************************************************************
   *
   * PUSHB[abc]:   PUSH Bytes
   * Opcode range: 0xB0-0xB7
   * Stack:        --> uint32...
   */
  static void
  Ins_PUSHB( TT_ExecContext  exc,
             FT_Long*        args )
  {
    FT_UShort  L, K;


    L = (FT_UShort)( exc->opcode - 0xB0 + 1 );

    if ( BOUNDS( L, exc->stackSize + 1 - exc->top ) )
    {
      exc->error = FT_THROW( Stack_Overflow );
      return;
    }

    for ( K = 1; K <= L; K++ )
      args[K - 1] = exc->code[exc->IP + K];
  }


  /**************************************************************************
   *
   * PUSHW[abc]:   PUSH Words
   * Opcode range: 0xB8-0xBF
   * Stack:        --> int32...
   */
  static void
  Ins_PUSHW( TT_ExecContext  exc,
             FT_Long*        args )
  {
    FT_UShort  L, K;


    L = (FT_UShort)( exc->opcode - 0xB8 + 1 );

    if ( BOUNDS( L, exc->stackSize + 1 - exc->top ) )
    {
      exc->error = FT_THROW( Stack_Overflow );
      return;
    }

    exc->IP++;

    for ( K = 0; K < L; K++ )
      args[K] = GetShortIns( exc );

    exc->step_ins = FALSE;
  }


  /**************************************************************************
   *
   * MANAGING THE GRAPHICS STATE
   *
   */


  static FT_Bool
  Ins_SxVTL( TT_ExecContext  exc,
             FT_UShort       aIdx1,
             FT_UShort       aIdx2,
             FT_UnitVector*  Vec )
  {
    FT_Long     A, B, C;
    FT_Vector*  p1;
    FT_Vector*  p2;

    FT_Byte  opcode = exc->opcode;


    if ( BOUNDS( aIdx1, exc->zp2.n_points ) ||
         BOUNDS( aIdx2, exc->zp1.n_points ) )
    {
      if ( exc->pedantic_hinting )
        exc->error = FT_THROW( Invalid_Reference );
      return FAILURE;
    }

    p1 = exc->zp1.cur + aIdx2;
    p2 = exc->zp2.cur + aIdx1;

    A = SUB_LONG( p1->x, p2->x );
    B = SUB_LONG( p1->y, p2->y );

    /* If p1 == p2, SPvTL and SFvTL behave the same as */
    /* SPvTCA[X] and SFvTCA[X], respectively.          */
    /*                                                 */
    /* Confirmed by Greg Hitchcock.                    */

    if ( A == 0 && B == 0 )
    {
      A      = 0x4000;
      opcode = 0;
    }

    if ( ( opcode & 1 ) != 0 )
    {
      C = B;   /* counter-clockwise rotation */
      B = A;
      A = NEG_LONG( C );
    }

    Normalize( A, B, Vec );

    return SUCCESS;
  }


  /**************************************************************************
   *
   * SVTCA[a]:     Set (F and P) Vectors to Coordinate Axis
   * Opcode range: 0x00-0x01
   * Stack:        -->
   *
   * SPvTCA[a]:    Set PVector to Coordinate Axis
   * Opcode range: 0x02-0x03
   * Stack:        -->
   *
   * SFvTCA[a]:    Set FVector to Coordinate Axis
   * Opcode range: 0x04-0x05
   * Stack:        -->
   */
  static void
  Ins_SxyTCA( TT_ExecContext  exc )
  {
    FT_Short  AA, BB;

    FT_Byte  opcode = exc->opcode;


    AA = (FT_Short)( ( opcode & 1 ) << 14 );
    BB = (FT_Short)( AA ^ 0x4000 );

    if ( opcode < 4 )
    {
      exc->GS.projVector.x = AA;
      exc->GS.projVector.y = BB;

      exc->GS.dualVector.x = AA;
      exc->GS.dualVector.y = BB;
    }

    if ( ( opcode & 2 ) == 0 )
    {
      exc->GS.freeVector.x = AA;
      exc->GS.freeVector.y = BB;
    }

    Compute_Funcs( exc );
  }


  /**************************************************************************
   *
   * SPvTL[a]:     Set PVector To Line
   * Opcode range: 0x06-0x07
   * Stack:        uint32 uint32 -->
   */
  static void
  Ins_SPVTL( TT_ExecContext  exc,
             FT_Long*        args )
  {
    if ( Ins_SxVTL( exc,
                    (FT_UShort)args[1],
                    (FT_UShort)args[0],
                    &exc->GS.projVector ) == SUCCESS )
    {
      exc->GS.dualVector = exc->GS.projVector;
      Compute_Funcs( exc );
    }
  }


  /**************************************************************************
   *
   * SFvTL[a]:     Set FVector To Line
   * Opcode range: 0x08-0x09
   * Stack:        uint32 uint32 -->
   */
  static void
  Ins_SFVTL( TT_ExecContext  exc,
             FT_Long*        args )
  {
    if ( Ins_SxVTL( exc,
                    (FT_UShort)args[1],
                    (FT_UShort)args[0],
                    &exc->GS.freeVector ) == SUCCESS )
    {
      Compute_Funcs( exc );
    }
  }


  /**************************************************************************
   *
   * SFvTPv[]:     Set FVector To PVector
   * Opcode range: 0x0E
   * Stack:        -->
   */
  static void
  Ins_SFVTPV( TT_ExecContext  exc )
  {
    exc->GS.freeVector = exc->GS.projVector;
    Compute_Funcs( exc );
  }


  /**************************************************************************
   *
   * SPvFS[]:      Set PVector From Stack
   * Opcode range: 0x0A
   * Stack:        f2.14 f2.14 -->
   */
  static void
  Ins_SPVFS( TT_ExecContext  exc,
             FT_Long*        args )
  {
    FT_Short  S;
    FT_Long   X, Y;


    /* Only use low 16bits, then sign extend */
    S = (FT_Short)args[1];
    Y = (FT_Long)S;
    S = (FT_Short)args[0];
    X = (FT_Long)S;

    Normalize( X, Y, &exc->GS.projVector );

    exc->GS.dualVector = exc->GS.projVector;
    Compute_Funcs( exc );
  }


  /**************************************************************************
   *
   * SFvFS[]:      Set FVector From Stack
   * Opcode range: 0x0B
   * Stack:        f2.14 f2.14 -->
   */
  static void
  Ins_SFVFS( TT_ExecContext  exc,
             FT_Long*        args )
  {
    FT_Short  S;
    FT_Long   X, Y;


    /* Only use low 16bits, then sign extend */
    S = (FT_Short)args[1];
    Y = (FT_Long)S;
    S = (FT_Short)args[0];
    X = S;

    Normalize( X, Y, &exc->GS.freeVector );
    Compute_Funcs( exc );
  }


  /**************************************************************************
   *
   * GPv[]:        Get Projection Vector
   * Opcode range: 0x0C
   * Stack:        ef2.14 --> ef2.14
   */
  static void
  Ins_GPV( TT_ExecContext  exc,
           FT_Long*        args )
  {
    args[0] = exc->GS.projVector.x;
    args[1] = exc->GS.projVector.y;
  }


  /**************************************************************************
   *
   * GFv[]:        Get Freedom Vector
   * Opcode range: 0x0D
   * Stack:        ef2.14 --> ef2.14
   */
  static void
  Ins_GFV( TT_ExecContext  exc,
           FT_Long*        args )
  {
    args[0] = exc->GS.freeVector.x;
    args[1] = exc->GS.freeVector.y;
  }


  /**************************************************************************
   *
   * SRP0[]:       Set Reference Point 0
   * Opcode range: 0x10
   * Stack:        uint32 -->
   */
  static void
  Ins_SRP0( TT_ExecContext  exc,
            FT_Long*        args )
  {
    exc->GS.rp0 = (FT_UShort)args[0];
  }


  /**************************************************************************
   *
   * SRP1[]:       Set Reference Point 1
   * Opcode range: 0x11
   * Stack:        uint32 -->
   */
  static void
  Ins_SRP1( TT_ExecContext  exc,
            FT_Long*        args )
  {
    exc->GS.rp1 = (FT_UShort)args[0];
  }


  /**************************************************************************
   *
   * SRP2[]:       Set Reference Point 2
   * Opcode range: 0x12
   * Stack:        uint32 -->
   */
  static void
  Ins_SRP2( TT_ExecContext  exc,
            FT_Long*        args )
  {
    exc->GS.rp2 = (FT_UShort)args[0];
  }


  /**************************************************************************
   *
   * SMD[]:        Set Minimum Distance
   * Opcode range: 0x1A
   * Stack:        f26.6 -->
   */
  static void
  Ins_SMD( TT_ExecContext  exc,
           FT_Long*        args )
  {
    exc->GS.minimum_distance = args[0];
  }


  /**************************************************************************
   *
   * SCVTCI[]:     Set Control Value Table Cut In
   * Opcode range: 0x1D
   * Stack:        f26.6 -->
   */
  static void
  Ins_SCVTCI( TT_ExecContext  exc,
              FT_Long*        args )
  {
    exc->GS.control_value_cutin = (FT_F26Dot6)args[0];
  }


  /**************************************************************************
   *
   * SSWCI[]:      Set Single Width Cut In
   * Opcode range: 0x1E
   * Stack:        f26.6 -->
   */
  static void
  Ins_SSWCI( TT_ExecContext  exc,
             FT_Long*        args )
  {
    exc->GS.single_width_cutin = (FT_F26Dot6)args[0];
  }


  /**************************************************************************
   *
   * SSW[]:        Set Single Width
   * Opcode range: 0x1F
   * Stack:        int32? -->
   */
  static void
  Ins_SSW( TT_ExecContext  exc,
           FT_Long*        args )
  {
    exc->GS.single_width_value = FT_MulFix( args[0],
                                            exc->tt_metrics.scale );
  }


  /**************************************************************************
   *
   * FLIPON[]:     Set auto-FLIP to ON
   * Opcode range: 0x4D
   * Stack:        -->
   */
  static void
  Ins_FLIPON( TT_ExecContext  exc )
  {
    exc->GS.auto_flip = TRUE;
  }


  /**************************************************************************
   *
   * FLIPOFF[]:    Set auto-FLIP to OFF
   * Opcode range: 0x4E
   * Stack:        -->
   */
  static void
  Ins_FLIPOFF( TT_ExecContext  exc )
  {
    exc->GS.auto_flip = FALSE;
  }


  /**************************************************************************
   *
   * SANGW[]:      Set ANGle Weight
   * Opcode range: 0x7E
   * Stack:        uint32 -->
   */
  static void
  Ins_SANGW( void )
  {
    /* instruction not supported anymore */
  }


  /**************************************************************************
   *
   * SDB[]:        Set Delta Base
   * Opcode range: 0x5E
   * Stack:        uint32 -->
   */
  static void
  Ins_SDB( TT_ExecContext  exc,
           FT_Long*        args )
  {
    exc->GS.delta_base = (FT_UShort)args[0];
  }


  /**************************************************************************
   *
   * SDS[]:        Set Delta Shift
   * Opcode range: 0x5F
   * Stack:        uint32 -->
   */
  static void
  Ins_SDS( TT_ExecContext  exc,
           FT_Long*        args )
  {
    if ( (FT_ULong)args[0] > 6UL )
      exc->error = FT_THROW( Bad_Argument );
    else
      exc->GS.delta_shift = (FT_UShort)args[0];
  }


  /**************************************************************************
   *
   * RTHG[]:       Round To Half Grid
   * Opcode range: 0x19
   * Stack:        -->
   */
  static void
  Ins_RTHG( TT_ExecContext  exc )
  {
    exc->GS.round_state = TT_Round_To_Half_Grid;
    exc->func_round     = (TT_Round_Func)Round_To_Half_Grid;
  }


  /**************************************************************************
   *
   * RTG[]:        Round To Grid
   * Opcode range: 0x18
   * Stack:        -->
   */
  static void
  Ins_RTG( TT_ExecContext  exc )
  {
    exc->GS.round_state = TT_Round_To_Grid;
    exc->func_round     = (TT_Round_Func)Round_To_Grid;
  }


  /**************************************************************************
   * RTDG[]:       Round To Double Grid
   * Opcode range: 0x3D
   * Stack:        -->
   */
  static void
  Ins_RTDG( TT_ExecContext  exc )
  {
    exc->GS.round_state = TT_Round_To_Double_Grid;
    exc->func_round     = (TT_Round_Func)Round_To_Double_Grid;
  }


  /**************************************************************************
   * RUTG[]:       Round Up To Grid
   * Opcode range: 0x7C
   * Stack:        -->
   */
  static void
  Ins_RUTG( TT_ExecContext  exc )
  {
    exc->GS.round_state = TT_Round_Up_To_Grid;
    exc->func_round     = (TT_Round_Func)Round_Up_To_Grid;
  }


  /**************************************************************************
   *
   * RDTG[]:       Round Down To Grid
   * Opcode range: 0x7D
   * Stack:        -->
   */
  static void
  Ins_RDTG( TT_ExecContext  exc )
  {
    exc->GS.round_state = TT_Round_Down_To_Grid;
    exc->func_round     = (TT_Round_Func)Round_Down_To_Grid;
  }


  /**************************************************************************
   *
   * ROFF[]:       Round OFF
   * Opcode range: 0x7A
   * Stack:        -->
   */
  static void
  Ins_ROFF( TT_ExecContext  exc )
  {
    exc->GS.round_state = TT_Round_Off;
    exc->func_round     = (TT_Round_Func)Round_None;
  }


  /**************************************************************************
   *
   * SROUND[]:     Super ROUND
   * Opcode range: 0x76
   * Stack:        Eint8 -->
   */
  static void
  Ins_SROUND( TT_ExecContext  exc,
              FT_Long*        args )
  {
    SetSuperRound( exc, 0x4000, args[0] );

    exc->GS.round_state = TT_Round_Super;
    exc->func_round     = (TT_Round_Func)Round_Super;
  }


  /**************************************************************************
   *
   * S45ROUND[]:   Super ROUND 45 degrees
   * Opcode range: 0x77
   * Stack:        uint32 -->
   */
  static void
  Ins_S45ROUND( TT_ExecContext  exc,
                FT_Long*        args )
  {
    SetSuperRound( exc, 0x2D41, args[0] );

    exc->GS.round_state = TT_Round_Super_45;
    exc->func_round     = (TT_Round_Func)Round_Super_45;
  }


  /**************************************************************************
   *
   * GC[a]:        Get Coordinate projected onto
   * Opcode range: 0x46-0x47
   * Stack:        uint32 --> f26.6
   *
   * XXX: UNDOCUMENTED: Measures from the original glyph must be taken
   *      along the dual projection vector!
   */
  static void
  Ins_GC( TT_ExecContext  exc,
          FT_Long*        args )
  {
    FT_ULong    L;
    FT_F26Dot6  R;


    L = (FT_ULong)args[0];

    if ( BOUNDSL( L, exc->zp2.n_points ) )
    {
      if ( exc->pedantic_hinting )
        exc->error = FT_THROW( Invalid_Reference );
      R = 0;
    }
    else
    {
      if ( exc->opcode & 1 )
        R = FAST_DUALPROJ( &exc->zp2.org[L] );
      else
        R = FAST_PROJECT( &exc->zp2.cur[L] );
    }

    args[0] = R;
  }


  /**************************************************************************
   *
   * SCFS[]:       Set Coordinate From Stack
   * Opcode range: 0x48
   * Stack:        f26.6 uint32 -->
   *
   * Formula:
   *
   *   OA := OA + ( value - OA.p )/( f.p ) * f
   */
  static void
  Ins_SCFS( TT_ExecContext  exc,
            FT_Long*        args )
  {
    FT_Long    K;
    FT_UShort  L;


    L = (FT_UShort)args[0];

    if ( BOUNDS( L, exc->zp2.n_points ) )
    {
      if ( exc->pedantic_hinting )
        exc->error = FT_THROW( Invalid_Reference );
      return;
    }

    K = FAST_PROJECT( &exc->zp2.cur[L] );

    exc->func_move( exc, &exc->zp2, L, SUB_LONG( args[1], K ) );

    /* UNDOCUMENTED!  The MS rasterizer does that with */
    /* twilight points (confirmed by Greg Hitchcock)   */
    if ( exc->GS.gep2 == 0 )
      exc->zp2.org[L] = exc->zp2.cur[L];
  }


  /**************************************************************************
   *
   * MD[a]:        Measure Distance
   * Opcode range: 0x49-0x4A
   * Stack:        uint32 uint32 --> f26.6
   *
   * XXX: UNDOCUMENTED: Measure taken in the original glyph must be along
   *                    the dual projection vector.
   *
   * XXX: UNDOCUMENTED: Flag attributes are inverted!
   *                      0 => measure distance in original outline
   *                      1 => measure distance in grid-fitted outline
   *
   * XXX: UNDOCUMENTED: `zp0 - zp1', and not `zp2 - zp1!
   */
  static void
  Ins_MD( TT_ExecContext  exc,
          FT_Long*        args )
  {
    FT_UShort   K, L;
    FT_F26Dot6  D;


    K = (FT_UShort)args[1];
    L = (FT_UShort)args[0];

    if ( BOUNDS( L, exc->zp0.n_points ) ||
         BOUNDS( K, exc->zp1.n_points ) )
    {
      if ( exc->pedantic_hinting )
        exc->error = FT_THROW( Invalid_Reference );
      D = 0;
    }
    else
    {
      if ( exc->opcode & 1 )
        D = PROJECT( exc->zp0.cur + L, exc->zp1.cur + K );
      else
      {
        /* XXX: UNDOCUMENTED: twilight zone special case */

        if ( exc->GS.gep0 == 0 || exc->GS.gep1 == 0 )
        {
          FT_Vector*  vec1 = exc->zp0.org + L;
          FT_Vector*  vec2 = exc->zp1.org + K;


          D = DUALPROJ( vec1, vec2 );
        }
        else
        {
          FT_Vector*  vec1 = exc->zp0.orus + L;
          FT_Vector*  vec2 = exc->zp1.orus + K;


          if ( exc->metrics.x_scale == exc->metrics.y_scale )
          {
            /* this should be faster */
            D = DUALPROJ( vec1, vec2 );
            D = FT_MulFix( D, exc->metrics.x_scale );
          }
          else
          {
            FT_Vector  vec;


            vec.x = FT_MulFix( vec1->x - vec2->x, exc->metrics.x_scale );
            vec.y = FT_MulFix( vec1->y - vec2->y, exc->metrics.y_scale );

            D = FAST_DUALPROJ( &vec );
          }
        }
      }
    }

#ifdef TT_SUPPORT_SUBPIXEL_HINTING_INFINALITY
    /* Disable Type 2 Vacuform Rounds - e.g. Arial Narrow */
    if ( SUBPIXEL_HINTING_INFINALITY         &&
         exc->ignore_x_mode                  &&
         ( D < 0 ? NEG_LONG( D ) : D ) == 64 )
      D += 1;
#endif /* TT_SUPPORT_SUBPIXEL_HINTING_INFINALITY */

    args[0] = D;
  }


  /**************************************************************************
   *
   * SDPvTL[a]:    Set Dual PVector to Line
   * Opcode range: 0x86-0x87
   * Stack:        uint32 uint32 -->
   */
  static void
  Ins_SDPVTL( TT_ExecContext  exc,
              FT_Long*        args )
  {
    FT_Long    A, B, C;
    FT_UShort  p1, p2;            /* was FT_Int in pas type ERROR */

    FT_Byte  opcode = exc->opcode;


    p1 = (FT_UShort)args[1];
    p2 = (FT_UShort)args[0];

    if ( BOUNDS( p2, exc->zp1.n_points ) ||
         BOUNDS( p1, exc->zp2.n_points ) )
    {
      if ( exc->pedantic_hinting )
        exc->error = FT_THROW( Invalid_Reference );
      return;
    }

    {
      FT_Vector*  v1 = exc->zp1.org + p2;
      FT_Vector*  v2 = exc->zp2.org + p1;


      A = SUB_LONG( v1->x, v2->x );
      B = SUB_LONG( v1->y, v2->y );

      /* If v1 == v2, SDPvTL behaves the same as */
      /* SVTCA[X], respectively.                 */
      /*                                         */
      /* Confirmed by Greg Hitchcock.            */

      if ( A == 0 && B == 0 )
      {
        A      = 0x4000;
        opcode = 0;
      }
    }

    if ( ( opcode & 1 ) != 0 )
    {
      C = B;   /* counter-clockwise rotation */
      B = A;
      A = NEG_LONG( C );
    }

    Normalize( A, B, &exc->GS.dualVector );

    {
      FT_Vector*  v1 = exc->zp1.cur + p2;
      FT_Vector*  v2 = exc->zp2.cur + p1;


      A = SUB_LONG( v1->x, v2->x );
      B = SUB_LONG( v1->y, v2->y );

      if ( A == 0 && B == 0 )
      {
        A      = 0x4000;
        opcode = 0;
      }
    }

    if ( ( opcode & 1 ) != 0 )
    {
      C = B;   /* counter-clockwise rotation */
      B = A;
      A = NEG_LONG( C );
    }

    Normalize( A, B, &exc->GS.projVector );
    Compute_Funcs( exc );
  }


  /**************************************************************************
   *
   * SZP0[]:       Set Zone Pointer 0
   * Opcode range: 0x13
   * Stack:        uint32 -->
   */
  static void
  Ins_SZP0( TT_ExecContext  exc,
            FT_Long*        args )
  {
    switch ( (FT_Int)args[0] )
    {
    case 0:
      exc->zp0 = exc->twilight;
      break;

    case 1:
      exc->zp0 = exc->pts;
      break;

    default:
      if ( exc->pedantic_hinting )
        exc->error = FT_THROW( Invalid_Reference );
      return;
    }

    exc->GS.gep0 = (FT_UShort)args[0];
  }


  /**************************************************************************
   *
   * SZP1[]:       Set Zone Pointer 1
   * Opcode range: 0x14
   * Stack:        uint32 -->
   */
  static void
  Ins_SZP1( TT_ExecContext  exc,
            FT_Long*        args )
  {
    switch ( (FT_Int)args[0] )
    {
    case 0:
      exc->zp1 = exc->twilight;
      break;

    case 1:
      exc->zp1 = exc->pts;
      break;

    default:
      if ( exc->pedantic_hinting )
        exc->error = FT_THROW( Invalid_Reference );
      return;
    }

    exc->GS.gep1 = (FT_UShort)args[0];
  }


  /**************************************************************************
   *
   * SZP2[]:       Set Zone Pointer 2
   * Opcode range: 0x15
   * Stack:        uint32 -->
   */
  static void
  Ins_SZP2( TT_ExecContext  exc,
            FT_Long*        args )
  {
    switch ( (FT_Int)args[0] )
    {
    case 0:
      exc->zp2 = exc->twilight;
      break;

    case 1:
      exc->zp2 = exc->pts;
      break;

    default:
      if ( exc->pedantic_hinting )
        exc->error = FT_THROW( Invalid_Reference );
      return;
    }

    exc->GS.gep2 = (FT_UShort)args[0];
  }


  /**************************************************************************
   *
   * SZPS[]:       Set Zone PointerS
   * Opcode range: 0x16
   * Stack:        uint32 -->
   */
  static void
  Ins_SZPS( TT_ExecContext  exc,
            FT_Long*        args )
  {
    switch ( (FT_Int)args[0] )
    {
    case 0:
      exc->zp0 = exc->twilight;
      break;

    case 1:
      exc->zp0 = exc->pts;
      break;

    default:
      if ( exc->pedantic_hinting )
        exc->error = FT_THROW( Invalid_Reference );
      return;
    }

    exc->zp1 = exc->zp0;
    exc->zp2 = exc->zp0;

    exc->GS.gep0 = (FT_UShort)args[0];
    exc->GS.gep1 = (FT_UShort)args[0];
    exc->GS.gep2 = (FT_UShort)args[0];
  }


  /**************************************************************************
   *
   * INSTCTRL[]:   INSTruction ConTRoL
   * Opcode range: 0x8E
   * Stack:        int32 int32 -->
   */
  static void
  Ins_INSTCTRL( TT_ExecContext  exc,
                FT_Long*        args )
  {
    FT_ULong  K, L, Kf;


    K = (FT_ULong)args[1];
    L = (FT_ULong)args[0];

    /* selector values cannot be `OR'ed;                 */
    /* they are indices starting with index 1, not flags */
    if ( K < 1 || K > 3 )
    {
      if ( exc->pedantic_hinting )
        exc->error = FT_THROW( Invalid_Reference );
      return;
    }

    /* convert index to flag value */
    Kf = 1 << ( K - 1 );

    if ( L != 0 )
    {
      /* arguments to selectors look like flag values */
      if ( L != Kf )
      {
        if ( exc->pedantic_hinting )
          exc->error = FT_THROW( Invalid_Reference );
        return;
      }
    }

    /* INSTCTRL should only be used in the CVT program */
    if ( exc->iniRange == tt_coderange_cvt )
    {
      exc->GS.instruct_control &= ~(FT_Byte)Kf;
      exc->GS.instruct_control |= (FT_Byte)L;
    }

    /* except to change the subpixel flags temporarily */
    else if ( exc->iniRange == tt_coderange_glyph && K == 3 )
    {
#ifdef TT_SUPPORT_SUBPIXEL_HINTING_INFINALITY
      /* INSTCTRL modifying flag 3 also has an effect */
      /* outside of the CVT program                   */
      if ( SUBPIXEL_HINTING_INFINALITY )
        exc->ignore_x_mode = !FT_BOOL( L == 4 );
#endif

#ifdef TT_SUPPORT_SUBPIXEL_HINTING_MINIMAL
      /* Native ClearType fonts sign a waiver that turns off all backward  */
      /* compatibility hacks and lets them program points to the grid like */
      /* it's 1996.  They might sign a waiver for just one glyph, though.  */
      if ( SUBPIXEL_HINTING_MINIMAL )
        exc->backward_compatibility = !FT_BOOL( L == 4 );
#endif
    }
    else if ( exc->pedantic_hinting )
      exc->error = FT_THROW( Invalid_Reference );
  }


  /**************************************************************************
   *
   * SCANCTRL[]:   SCAN ConTRoL
   * Opcode range: 0x85
   * Stack:        uint32? -->
   */
  static void
  Ins_SCANCTRL( TT_ExecContext  exc,
                FT_Long*        args )
  {
    FT_Int  A;


    /* Get Threshold */
    A = (FT_Int)( args[0] & 0xFF );

    if ( A == 0xFF )
    {
      exc->GS.scan_control = TRUE;
      return;
    }
    else if ( A == 0 )
    {
      exc->GS.scan_control = FALSE;
      return;
    }

    if ( ( args[0] & 0x100 ) != 0 && exc->tt_metrics.ppem <= A )
      exc->GS.scan_control = TRUE;

    if ( ( args[0] & 0x200 ) != 0 && exc->tt_metrics.rotated )
      exc->GS.scan_control = TRUE;

    if ( ( args[0] & 0x400 ) != 0 && exc->tt_metrics.stretched )
      exc->GS.scan_control = TRUE;

    if ( ( args[0] & 0x800 ) != 0 && exc->tt_metrics.ppem > A )
      exc->GS.scan_control = FALSE;

    if ( ( args[0] & 0x1000 ) != 0 && exc->tt_metrics.rotated )
      exc->GS.scan_control = FALSE;

    if ( ( args[0] & 0x2000 ) != 0 && exc->tt_metrics.stretched )
      exc->GS.scan_control = FALSE;
  }


  /**************************************************************************
   *
   * SCANTYPE[]:   SCAN TYPE
   * Opcode range: 0x8D
   * Stack:        uint16 -->
   */
  static void
  Ins_SCANTYPE( TT_ExecContext  exc,
                FT_Long*        args )
  {
    if ( args[0] >= 0 )
      exc->GS.scan_type = (FT_Int)args[0] & 0xFFFF;
  }


  /**************************************************************************
   *
   * MANAGING OUTLINES
   *
   */


  /**************************************************************************
   *
   * FLIPPT[]:     FLIP PoinT
   * Opcode range: 0x80
   * Stack:        uint32... -->
   */
  static void
  Ins_FLIPPT( TT_ExecContext  exc )
  {
    FT_UShort  point;


#ifdef TT_SUPPORT_SUBPIXEL_HINTING_MINIMAL
    /* See `ttinterp.h' for details on backward compatibility mode. */
    if ( SUBPIXEL_HINTING_MINIMAL    &&
         exc->backward_compatibility &&
         exc->iupx_called            &&
         exc->iupy_called            )
      goto Fail;
#endif

    if ( exc->top < exc->GS.loop )
    {
      if ( exc->pedantic_hinting )
        exc->error = FT_THROW( Too_Few_Arguments );
      goto Fail;
    }

    while ( exc->GS.loop > 0 )
    {
      exc->args--;

      point = (FT_UShort)exc->stack[exc->args];

      if ( BOUNDS( point, exc->pts.n_points ) )
      {
        if ( exc->pedantic_hinting )
        {
          exc->error = FT_THROW( Invalid_Reference );
          return;
        }
      }
      else
        exc->pts.tags[point] ^= FT_CURVE_TAG_ON;

      exc->GS.loop--;
    }

  Fail:
    exc->GS.loop = 1;
    exc->new_top = exc->args;
  }


  /**************************************************************************
   *
   * FLIPRGON[]:   FLIP RanGe ON
   * Opcode range: 0x81
   * Stack:        uint32 uint32 -->
   */
  static void
  Ins_FLIPRGON( TT_ExecContext  exc,
                FT_Long*        args )
  {
    FT_UShort  I, K, L;


#ifdef TT_SUPPORT_SUBPIXEL_HINTING_MINIMAL
    /* See `ttinterp.h' for details on backward compatibility mode. */
    if ( SUBPIXEL_HINTING_MINIMAL    &&
         exc->backward_compatibility &&
         exc->iupx_called            &&
         exc->iupy_called            )
      return;
#endif

    K = (FT_UShort)args[1];
    L = (FT_UShort)args[0];

    if ( BOUNDS( K, exc->pts.n_points ) ||
         BOUNDS( L, exc->pts.n_points ) )
    {
      if ( exc->pedantic_hinting )
        exc->error = FT_THROW( Invalid_Reference );
      return;
    }

    for ( I = L; I <= K; I++ )
      exc->pts.tags[I] |= FT_CURVE_TAG_ON;
  }


  /**************************************************************************
   *
   * FLIPRGOFF:    FLIP RanGe OFF
   * Opcode range: 0x82
   * Stack:        uint32 uint32 -->
   */
  static void
  Ins_FLIPRGOFF( TT_ExecContext  exc,
                 FT_Long*        args )
  {
    FT_UShort  I, K, L;


#ifdef TT_SUPPORT_SUBPIXEL_HINTING_MINIMAL
    /* See `ttinterp.h' for details on backward compatibility mode. */
    if ( SUBPIXEL_HINTING_MINIMAL    &&
         exc->backward_compatibility &&
         exc->iupx_called            &&
         exc->iupy_called            )
      return;
#endif

    K = (FT_UShort)args[1];
    L = (FT_UShort)args[0];

    if ( BOUNDS( K, exc->pts.n_points ) ||
         BOUNDS( L, exc->pts.n_points ) )
    {
      if ( exc->pedantic_hinting )
        exc->error = FT_THROW( Invalid_Reference );
      return;
    }

    for ( I = L; I <= K; I++ )
      exc->pts.tags[I] &= ~FT_CURVE_TAG_ON;
  }


  static FT_Bool
  Compute_Point_Displacement( TT_ExecContext  exc,
                              FT_F26Dot6*     x,
                              FT_F26Dot6*     y,
                              TT_GlyphZone    zone,
                              FT_UShort*      refp )
  {
    TT_GlyphZoneRec  zp;
    FT_UShort        p;
    FT_F26Dot6       d;


    if ( exc->opcode & 1 )
    {
      zp = exc->zp0;
      p  = exc->GS.rp1;
    }
    else
    {
      zp = exc->zp1;
      p  = exc->GS.rp2;
    }

    if ( BOUNDS( p, zp.n_points ) )
    {
      if ( exc->pedantic_hinting )
        exc->error = FT_THROW( Invalid_Reference );
      *refp = 0;
      return FAILURE;
    }

    *zone = zp;
    *refp = p;

    d = PROJECT( zp.cur + p, zp.org + p );

    *x = FT_MulDiv( d, (FT_Long)exc->GS.freeVector.x, exc->F_dot_P );
    *y = FT_MulDiv( d, (FT_Long)exc->GS.freeVector.y, exc->F_dot_P );

    return SUCCESS;
  }


  /* See `ttinterp.h' for details on backward compatibility mode. */
  static void
  Move_Zp2_Point( TT_ExecContext  exc,
                  FT_UShort       point,
                  FT_F26Dot6      dx,
                  FT_F26Dot6      dy,
                  FT_Bool         touch )
  {
    if ( exc->GS.freeVector.x != 0 )
    {
#ifdef TT_SUPPORT_SUBPIXEL_HINTING_MINIMAL
      if ( !( SUBPIXEL_HINTING_MINIMAL    &&
              exc->backward_compatibility ) )
#endif
        exc->zp2.cur[point].x = ADD_LONG( exc->zp2.cur[point].x, dx );

      if ( touch )
        exc->zp2.tags[point] |= FT_CURVE_TAG_TOUCH_X;
    }

    if ( exc->GS.freeVector.y != 0 )
    {
#ifdef TT_SUPPORT_SUBPIXEL_HINTING_MINIMAL
      if ( !( SUBPIXEL_HINTING_MINIMAL    &&
              exc->backward_compatibility &&
              exc->iupx_called            &&
              exc->iupy_called            ) )
#endif
        exc->zp2.cur[point].y = ADD_LONG( exc->zp2.cur[point].y, dy );

      if ( touch )
        exc->zp2.tags[point] |= FT_CURVE_TAG_TOUCH_Y;
    }
  }


  /**************************************************************************
   *
   * SHP[a]:       SHift Point by the last point
   * Opcode range: 0x32-0x33
   * Stack:        uint32... -->
   */
  static void
  Ins_SHP( TT_ExecContext  exc )
  {
    TT_GlyphZoneRec  zp;
    FT_UShort        refp;

    FT_F26Dot6       dx, dy;
    FT_UShort        point;


    if ( exc->top < exc->GS.loop )
    {
      if ( exc->pedantic_hinting )
        exc->error = FT_THROW( Invalid_Reference );
      goto Fail;
    }

    if ( Compute_Point_Displacement( exc, &dx, &dy, &zp, &refp ) )
      return;

    while ( exc->GS.loop > 0 )
    {
      exc->args--;
      point = (FT_UShort)exc->stack[exc->args];

      if ( BOUNDS( point, exc->zp2.n_points ) )
      {
        if ( exc->pedantic_hinting )
        {
          exc->error = FT_THROW( Invalid_Reference );
          return;
        }
      }
      else
#ifdef TT_SUPPORT_SUBPIXEL_HINTING_INFINALITY
      /* doesn't follow Cleartype spec but produces better result */
      if ( SUBPIXEL_HINTING_INFINALITY && exc->ignore_x_mode )
        Move_Zp2_Point( exc, point, 0, dy, TRUE );
      else
#endif /* TT_SUPPORT_SUBPIXEL_HINTING_INFINALITY */
        Move_Zp2_Point( exc, point, dx, dy, TRUE );

      exc->GS.loop--;
    }

  Fail:
    exc->GS.loop = 1;
    exc->new_top = exc->args;
  }


  /**************************************************************************
   *
   * SHC[a]:       SHift Contour
   * Opcode range: 0x34-35
   * Stack:        uint32 -->
   *
   * UNDOCUMENTED: According to Greg Hitchcock, there is one (virtual)
   *               contour in the twilight zone, namely contour number
   *               zero which includes all points of it.
   */
  static void
  Ins_SHC( TT_ExecContext  exc,
           FT_Long*        args )
  {
    TT_GlyphZoneRec  zp;
    FT_UShort        refp;
    FT_F26Dot6       dx, dy;

    FT_Short         contour, bounds;
    FT_UShort        start, limit, i;


    contour = (FT_Short)args[0];
    bounds  = ( exc->GS.gep2 == 0 ) ? 1 : exc->zp2.n_contours;

    if ( BOUNDS( contour, bounds ) )
    {
      if ( exc->pedantic_hinting )
        exc->error = FT_THROW( Invalid_Reference );
      return;
    }

    if ( Compute_Point_Displacement( exc, &dx, &dy, &zp, &refp ) )
      return;

    if ( contour == 0 )
      start = 0;
    else
      start = (FT_UShort)( exc->zp2.contours[contour - 1] + 1 -
                           exc->zp2.first_point );

    /* we use the number of points if in the twilight zone */
    if ( exc->GS.gep2 == 0 )
      limit = exc->zp2.n_points;
    else
      limit = (FT_UShort)( exc->zp2.contours[contour] -
                           exc->zp2.first_point + 1 );

    for ( i = start; i < limit; i++ )
    {
      if ( zp.cur != exc->zp2.cur || refp != i )
        Move_Zp2_Point( exc, i, dx, dy, TRUE );
    }
  }


  /**************************************************************************
   *
   * SHZ[a]:       SHift Zone
   * Opcode range: 0x36-37
   * Stack:        uint32 -->
   */
  static void
  Ins_SHZ( TT_ExecContext  exc,
           FT_Long*        args )
  {
    TT_GlyphZoneRec  zp;
    FT_UShort        refp;
    FT_F26Dot6       dx,
                     dy;

    FT_UShort        limit, i;


    if ( BOUNDS( args[0], 2 ) )
    {
      if ( exc->pedantic_hinting )
        exc->error = FT_THROW( Invalid_Reference );
      return;
    }

    if ( Compute_Point_Displacement( exc, &dx, &dy, &zp, &refp ) )
      return;

    /* XXX: UNDOCUMENTED! SHZ doesn't move the phantom points.     */
    /*      Twilight zone has no real contours, so use `n_points'. */
    /*      Normal zone's `n_points' includes phantoms, so must    */
    /*      use end of last contour.                               */
    if ( exc->GS.gep2 == 0 )
      limit = (FT_UShort)exc->zp2.n_points;
    else if ( exc->GS.gep2 == 1 && exc->zp2.n_contours > 0 )
      limit = (FT_UShort)( exc->zp2.contours[exc->zp2.n_contours - 1] + 1 );
    else
      limit = 0;

    /* XXX: UNDOCUMENTED! SHZ doesn't touch the points */
    for ( i = 0; i < limit; i++ )
    {
      if ( zp.cur != exc->zp2.cur || refp != i )
        Move_Zp2_Point( exc, i, dx, dy, FALSE );
    }
  }


  /**************************************************************************
   *
   * SHPIX[]:      SHift points by a PIXel amount
   * Opcode range: 0x38
   * Stack:        f26.6 uint32... -->
   */
  static void
  Ins_SHPIX( TT_ExecContext  exc,
             FT_Long*        args )
  {
    FT_F26Dot6  dx, dy;
    FT_UShort   point;
#ifdef TT_SUPPORT_SUBPIXEL_HINTING_MINIMAL
    FT_Bool     in_twilight = FT_BOOL( exc->GS.gep0 == 0 ||
                                       exc->GS.gep1 == 0 ||
                                       exc->GS.gep2 == 0 );
#endif



    if ( exc->top < exc->GS.loop + 1 )
    {
      if ( exc->pedantic_hinting )
        exc->error = FT_THROW( Invalid_Reference );
      goto Fail;
    }

    dx = TT_MulFix14( args[0], exc->GS.freeVector.x );
    dy = TT_MulFix14( args[0], exc->GS.freeVector.y );

    while ( exc->GS.loop > 0 )
    {
      exc->args--;

      point = (FT_UShort)exc->stack[exc->args];

      if ( BOUNDS( point, exc->zp2.n_points ) )
      {
        if ( exc->pedantic_hinting )
        {
          exc->error = FT_THROW( Invalid_Reference );
          return;
        }
      }
      else
#ifdef TT_SUPPORT_SUBPIXEL_HINTING_INFINALITY
      if ( SUBPIXEL_HINTING_INFINALITY &&
           exc->ignore_x_mode          )
      {
        FT_Int  B1, B2;


        /*  If not using ignore_x_mode rendering, allow ZP2 move.        */
        /*  If inline deltas aren't allowed, skip ZP2 move.              */
        /*  If using ignore_x_mode rendering, allow ZP2 point move if:   */
        /*   - freedom vector is y and sph_compatibility_mode is off     */
        /*   - the glyph is composite and the move is in the Y direction */
        /*   - the glyph is specifically set to allow SHPIX moves        */
        /*   - the move is on a previously Y-touched point               */

        /* save point for later comparison */
        B1 = exc->zp2.cur[point].y;

        if ( exc->face->sph_compatibility_mode )
        {
          if ( exc->sph_tweak_flags & SPH_TWEAK_ROUND_NONPIXEL_Y_MOVES )
            dy = FT_PIX_ROUND( B1 + dy ) - B1;

          /* skip post-iup deltas */
          if ( exc->iup_called                                          &&
               ( ( exc->sph_in_func_flags & SPH_FDEF_INLINE_DELTA_1 ) ||
                 ( exc->sph_in_func_flags & SPH_FDEF_INLINE_DELTA_2 ) ) )
            goto Skip;

          if ( !( exc->sph_tweak_flags & SPH_TWEAK_ALWAYS_SKIP_DELTAP ) &&
                ( ( exc->is_composite && exc->GS.freeVector.y != 0 ) ||
                  ( exc->zp2.tags[point] & FT_CURVE_TAG_TOUCH_Y )    ||
                  ( exc->sph_tweak_flags & SPH_TWEAK_DO_SHPIX )      )  )
            Move_Zp2_Point( exc, point, 0, dy, TRUE );

          /* save new point */
          if ( exc->GS.freeVector.y != 0 )
          {
            B2 = exc->zp2.cur[point].y;

            /* reverse any disallowed moves */
            if ( ( B1 & 63 ) == 0 &&
                 ( B2 & 63 ) != 0 &&
                 B1 != B2         )
              Move_Zp2_Point( exc, point, 0, NEG_LONG( dy ), TRUE );
          }
        }
        else if ( exc->GS.freeVector.y != 0 )
        {
          Move_Zp2_Point( exc, point, dx, dy, TRUE );

          /* save new point */
          B2 = exc->zp2.cur[point].y;

          /* reverse any disallowed moves */
          if ( ( exc->sph_tweak_flags & SPH_TWEAK_SKIP_NONPIXEL_Y_MOVES ) &&
               ( B1 & 63 ) != 0                                           &&
               ( B2 & 63 ) != 0                                           &&
               B1 != B2                                                   )
            Move_Zp2_Point( exc,
                            point,
                            NEG_LONG( dx ),
                            NEG_LONG( dy ),
                            TRUE );
        }
        else if ( exc->sph_in_func_flags & SPH_FDEF_TYPEMAN_DIAGENDCTRL )
          Move_Zp2_Point( exc, point, dx, dy, TRUE );
      }
      else
#endif
#ifdef TT_SUPPORT_SUBPIXEL_HINTING_MINIMAL
      if ( SUBPIXEL_HINTING_MINIMAL    &&
           exc->backward_compatibility )
      {
        /* Special case: allow SHPIX to move points in the twilight zone.  */
        /* Otherwise, treat SHPIX the same as DELTAP.  Unbreaks various    */
        /* fonts such as older versions of Rokkitt and DTL Argo T Light    */
        /* that would glitch severely after calling ALIGNRP after a        */
        /* blocked SHPIX.                                                  */
        if ( in_twilight                                                ||
             ( !( exc->iupx_called && exc->iupy_called )              &&
               ( ( exc->is_composite && exc->GS.freeVector.y != 0 ) ||
                 ( exc->zp2.tags[point] & FT_CURVE_TAG_TOUCH_Y )    ) ) )
          Move_Zp2_Point( exc, point, 0, dy, TRUE );
      }
      else
#endif
        Move_Zp2_Point( exc, point, dx, dy, TRUE );

#ifdef TT_SUPPORT_SUBPIXEL_HINTING_INFINALITY
    Skip:
#endif
      exc->GS.loop--;
    }

  Fail:
    exc->GS.loop = 1;
    exc->new_top = exc->args;
  }


  /**************************************************************************
   *
   * MSIRP[a]:     Move Stack Indirect Relative Position
   * Opcode range: 0x3A-0x3B
   * Stack:        f26.6 uint32 -->
   */
  static void
  Ins_MSIRP( TT_ExecContext  exc,
             FT_Long*        args )
  {
    FT_UShort   point = 0;
    FT_F26Dot6  distance;


    point = (FT_UShort)args[0];

    if ( BOUNDS( point,       exc->zp1.n_points ) ||
         BOUNDS( exc->GS.rp0, exc->zp0.n_points ) )
    {
      if ( exc->pedantic_hinting )
        exc->error = FT_THROW( Invalid_Reference );
      return;
    }

    /* UNDOCUMENTED!  The MS rasterizer does that with */
    /* twilight points (confirmed by Greg Hitchcock)   */
    if ( exc->GS.gep1 == 0 )
    {
      exc->zp1.org[point] = exc->zp0.org[exc->GS.rp0];
      exc->func_move_orig( exc, &exc->zp1, point, args[1] );
      exc->zp1.cur[point] = exc->zp1.org[point];
    }

    distance = PROJECT( exc->zp1.cur + point, exc->zp0.cur + exc->GS.rp0 );

#ifdef TT_SUPPORT_SUBPIXEL_HINTING_INFINALITY
    /* subpixel hinting - make MSIRP respect CVT cut-in; */
    if ( SUBPIXEL_HINTING_INFINALITY &&
         exc->ignore_x_mode          &&
         exc->GS.freeVector.x != 0   )
    {
      FT_F26Dot6  control_value_cutin = exc->GS.control_value_cutin;
      FT_F26Dot6  delta;


      if ( !( exc->sph_tweak_flags & SPH_TWEAK_NORMAL_ROUND ) )
        control_value_cutin = 0;

      delta = SUB_LONG( distance, args[1] );
      if ( delta < 0 )
        delta = NEG_LONG( delta );

      if ( delta >= control_value_cutin )
        distance = args[1];
    }
#endif /* TT_SUPPORT_SUBPIXEL_HINTING_INFINALITY */

    exc->func_move( exc,
                    &exc->zp1,
                    point,
                    SUB_LONG( args[1], distance ) );

    exc->GS.rp1 = exc->GS.rp0;
    exc->GS.rp2 = point;

    if ( ( exc->opcode & 1 ) != 0 )
      exc->GS.rp0 = point;
  }


  /**************************************************************************
   *
   * MDAP[a]:      Move Direct Absolute Point
   * Opcode range: 0x2E-0x2F
   * Stack:        uint32 -->
   */
  static void
  Ins_MDAP( TT_ExecContext  exc,
            FT_Long*        args )
  {
    FT_UShort   point;
    FT_F26Dot6  cur_dist;
    FT_F26Dot6  distance;


    point = (FT_UShort)args[0];

    if ( BOUNDS( point, exc->zp0.n_points ) )
    {
      if ( exc->pedantic_hinting )
        exc->error = FT_THROW( Invalid_Reference );
      return;
    }

    if ( ( exc->opcode & 1 ) != 0 )
    {
      cur_dist = FAST_PROJECT( &exc->zp0.cur[point] );
#ifdef TT_SUPPORT_SUBPIXEL_HINTING_INFINALITY
      if ( SUBPIXEL_HINTING_INFINALITY &&
           exc->ignore_x_mode          &&
           exc->GS.freeVector.x != 0   )
        distance = SUB_LONG( Round_None( exc, cur_dist, 3 ), cur_dist );
      else
#endif
        distance = SUB_LONG( exc->func_round( exc, cur_dist, 3 ), cur_dist );
    }
    else
      distance = 0;

    exc->func_move( exc, &exc->zp0, point, distance );

    exc->GS.rp0 = point;
    exc->GS.rp1 = point;
  }


  /**************************************************************************
   *
   * MIAP[a]:      Move Indirect Absolute Point
   * Opcode range: 0x3E-0x3F
   * Stack:        uint32 uint32 -->
   */
  static void
  Ins_MIAP( TT_ExecContext  exc,
            FT_Long*        args )
  {
    FT_ULong    cvtEntry;
    FT_UShort   point;
    FT_F26Dot6  distance;
    FT_F26Dot6  org_dist;


    cvtEntry = (FT_ULong)args[1];
    point    = (FT_UShort)args[0];

    if ( BOUNDS( point,     exc->zp0.n_points ) ||
         BOUNDSL( cvtEntry, exc->cvtSize )      )
    {
      if ( exc->pedantic_hinting )
        exc->error = FT_THROW( Invalid_Reference );
      goto Fail;
    }

    /* UNDOCUMENTED!                                                      */
    /*                                                                    */
    /* The behaviour of an MIAP instruction is quite different when used  */
    /* in the twilight zone.                                              */
    /*                                                                    */
    /* First, no control value cut-in test is performed as it would fail  */
    /* anyway.  Second, the original point, i.e. (org_x,org_y) of         */
    /* zp0.point, is set to the absolute, unrounded distance found in the */
    /* CVT.                                                               */
    /*                                                                    */
    /* This is used in the CVT programs of the Microsoft fonts Arial,     */
    /* Times, etc., in order to re-adjust some key font heights.  It      */
    /* allows the use of the IP instruction in the twilight zone, which   */
    /* otherwise would be invalid according to the specification.         */
    /*                                                                    */
    /* We implement it with a special sequence for the twilight zone.     */
    /* This is a bad hack, but it seems to work.                          */
    /*                                                                    */
    /* Confirmed by Greg Hitchcock.                                       */

    distance = exc->func_read_cvt( exc, cvtEntry );

    if ( exc->GS.gep0 == 0 )   /* If in twilight zone */
    {
#ifdef TT_SUPPORT_SUBPIXEL_HINTING_INFINALITY
      /* Only adjust if not in sph_compatibility_mode or ignore_x_mode. */
      /* Determined via experimentation and may be incorrect...         */
      if ( !( SUBPIXEL_HINTING_INFINALITY           &&
              ( exc->ignore_x_mode                &&
                exc->face->sph_compatibility_mode ) ) )
#endif /* TT_SUPPORT_SUBPIXEL_HINTING_INFINALITY */
        exc->zp0.org[point].x = TT_MulFix14( distance,
                                             exc->GS.freeVector.x );
      exc->zp0.org[point].y = TT_MulFix14( distance,
                                           exc->GS.freeVector.y );
      exc->zp0.cur[point]   = exc->zp0.org[point];
    }
#ifdef TT_SUPPORT_SUBPIXEL_HINTING_INFINALITY
    if ( SUBPIXEL_HINTING_INFINALITY                    &&
         exc->ignore_x_mode                             &&
         ( exc->sph_tweak_flags & SPH_TWEAK_MIAP_HACK ) &&
         distance > 0                                   &&
         exc->GS.freeVector.y != 0                      )
      distance = 0;
#endif /* TT_SUPPORT_SUBPIXEL_HINTING_INFINALITY */

    org_dist = FAST_PROJECT( &exc->zp0.cur[point] );

    if ( ( exc->opcode & 1 ) != 0 )   /* rounding and control cut-in flag */
    {
      FT_F26Dot6  control_value_cutin = exc->GS.control_value_cutin;
      FT_F26Dot6  delta;


#ifdef TT_SUPPORT_SUBPIXEL_HINTING_INFINALITY
      if ( SUBPIXEL_HINTING_INFINALITY                        &&
           exc->ignore_x_mode                                 &&
           exc->GS.freeVector.x != 0                          &&
           exc->GS.freeVector.y == 0                          &&
           !( exc->sph_tweak_flags & SPH_TWEAK_NORMAL_ROUND ) )
        control_value_cutin = 0;
#endif /* TT_SUPPORT_SUBPIXEL_HINTING_INFINALITY */

      delta = SUB_LONG( distance, org_dist );
      if ( delta < 0 )
        delta = NEG_LONG( delta );

      if ( delta > control_value_cutin )
        distance = org_dist;

#ifdef TT_SUPPORT_SUBPIXEL_HINTING_INFINALITY
      if ( SUBPIXEL_HINTING_INFINALITY &&
           exc->ignore_x_mode          &&
           exc->GS.freeVector.x != 0   )
        distance = Round_None( exc, distance, 3 );
      else
#endif
        distance = exc->func_round( exc, distance, 3 );
    }

    exc->func_move( exc, &exc->zp0, point, SUB_LONG( distance, org_dist ) );

  Fail:
    exc->GS.rp0 = point;
    exc->GS.rp1 = point;
  }


  /**************************************************************************
   *
   * MDRP[abcde]:  Move Direct Relative Point
   * Opcode range: 0xC0-0xDF
   * Stack:        uint32 -->
   */
  static void
  Ins_MDRP( TT_ExecContext  exc,
            FT_Long*        args )
  {
    FT_UShort   point = 0;
    FT_F26Dot6  org_dist, distance;


    point = (FT_UShort)args[0];

    if ( BOUNDS( point,       exc->zp1.n_points ) ||
         BOUNDS( exc->GS.rp0, exc->zp0.n_points ) )
    {
      if ( exc->pedantic_hinting )
        exc->error = FT_THROW( Invalid_Reference );
      goto Fail;
    }

    /* XXX: Is there some undocumented feature while in the */
    /*      twilight zone?                                  */

    /* XXX: UNDOCUMENTED: twilight zone special case */

    if ( exc->GS.gep0 == 0 || exc->GS.gep1 == 0 )
    {
      FT_Vector*  vec1 = &exc->zp1.org[point];
      FT_Vector*  vec2 = &exc->zp0.org[exc->GS.rp0];


      org_dist = DUALPROJ( vec1, vec2 );
    }
    else
    {
      FT_Vector*  vec1 = &exc->zp1.orus[point];
      FT_Vector*  vec2 = &exc->zp0.orus[exc->GS.rp0];


      if ( exc->metrics.x_scale == exc->metrics.y_scale )
      {
        /* this should be faster */
        org_dist = DUALPROJ( vec1, vec2 );
        org_dist = FT_MulFix( org_dist, exc->metrics.x_scale );
      }
      else
      {
        FT_Vector  vec;


        vec.x = FT_MulFix( SUB_LONG( vec1->x, vec2->x ),
                           exc->metrics.x_scale );
        vec.y = FT_MulFix( SUB_LONG( vec1->y, vec2->y ),
                           exc->metrics.y_scale );

        org_dist = FAST_DUALPROJ( &vec );
      }
    }

    /* single width cut-in test */

    /* |org_dist - single_width_value| < single_width_cutin */
    if ( exc->GS.single_width_cutin > 0          &&
         org_dist < exc->GS.single_width_value +
                      exc->GS.single_width_cutin &&
         org_dist > exc->GS.single_width_value -
                      exc->GS.single_width_cutin )
    {
      if ( org_dist >= 0 )
        org_dist = exc->GS.single_width_value;
      else
        org_dist = -exc->GS.single_width_value;
    }

    /* round flag */

    if ( ( exc->opcode & 4 ) != 0 )
    {
#ifdef TT_SUPPORT_SUBPIXEL_HINTING_INFINALITY
      if ( SUBPIXEL_HINTING_INFINALITY &&
           exc->ignore_x_mode          &&
           exc->GS.freeVector.x != 0   )
        distance = Round_None( exc, org_dist, exc->opcode & 3 );
      else
#endif
        distance = exc->func_round( exc, org_dist, exc->opcode & 3 );
    }
    else
      distance = Round_None( exc, org_dist, exc->opcode & 3 );

    /* minimum distance flag */

    if ( ( exc->opcode & 8 ) != 0 )
    {
      FT_F26Dot6  minimum_distance = exc->GS.minimum_distance;


#ifdef TT_SUPPORT_SUBPIXEL_HINTING_INFINALITY
      if ( SUBPIXEL_HINTING_INFINALITY                        &&
           exc->ignore_x_mode                                 &&
           exc->GS.freeVector.x != 0                          &&
           !( exc->sph_tweak_flags & SPH_TWEAK_NORMAL_ROUND ) )
        minimum_distance = 0;
#endif /* TT_SUPPORT_SUBPIXEL_HINTING_INFINALITY */

      if ( org_dist >= 0 )
      {
        if ( distance < minimum_distance )
          distance = minimum_distance;
      }
      else
      {
        if ( distance > NEG_LONG( minimum_distance ) )
          distance = NEG_LONG( minimum_distance );
      }
    }

    /* now move the point */

    org_dist = PROJECT( exc->zp1.cur + point, exc->zp0.cur + exc->GS.rp0 );

    exc->func_move( exc, &exc->zp1, point, SUB_LONG( distance, org_dist ) );

  Fail:
    exc->GS.rp1 = exc->GS.rp0;
    exc->GS.rp2 = point;

    if ( ( exc->opcode & 16 ) != 0 )
      exc->GS.rp0 = point;
  }


  /**************************************************************************
   *
   * MIRP[abcde]:  Move Indirect Relative Point
   * Opcode range: 0xE0-0xFF
   * Stack:        int32? uint32 -->
   */
  static void
  Ins_MIRP( TT_ExecContext  exc,
            FT_Long*        args )
  {
    FT_UShort   point;
    FT_ULong    cvtEntry;

    FT_F26Dot6  cvt_dist,
                distance,
                cur_dist,
                org_dist;

    FT_F26Dot6  delta;


    point    = (FT_UShort)args[0];
    cvtEntry = (FT_ULong)( ADD_LONG( args[1], 1 ) );

    /* XXX: UNDOCUMENTED! cvt[-1] = 0 always */

    if ( BOUNDS( point,       exc->zp1.n_points ) ||
         BOUNDSL( cvtEntry,   exc->cvtSize + 1 )  ||
         BOUNDS( exc->GS.rp0, exc->zp0.n_points ) )
    {
      if ( exc->pedantic_hinting )
        exc->error = FT_THROW( Invalid_Reference );
      goto Fail;
    }

    if ( !cvtEntry )
      cvt_dist = 0;
    else
      cvt_dist = exc->func_read_cvt( exc, cvtEntry - 1 );

    /* single width test */

    delta = SUB_LONG( cvt_dist, exc->GS.single_width_value );
    if ( delta < 0 )
      delta = NEG_LONG( delta );

    if ( delta < exc->GS.single_width_cutin )
    {
      if ( cvt_dist >= 0 )
        cvt_dist =  exc->GS.single_width_value;
      else
        cvt_dist = -exc->GS.single_width_value;
    }

    /* UNDOCUMENTED!  The MS rasterizer does that with */
    /* twilight points (confirmed by Greg Hitchcock)   */
    if ( exc->GS.gep1 == 0 )
    {
      exc->zp1.org[point].x = ADD_LONG(
                                exc->zp0.org[exc->GS.rp0].x,
                                TT_MulFix14( cvt_dist,
                                             exc->GS.freeVector.x ) );
      exc->zp1.org[point].y = ADD_LONG(
                                exc->zp0.org[exc->GS.rp0].y,
                                TT_MulFix14( cvt_dist,
                                             exc->GS.freeVector.y ) );
      exc->zp1.cur[point]   = exc->zp1.org[point];
    }

    org_dist = DUALPROJ( &exc->zp1.org[point], &exc->zp0.org[exc->GS.rp0] );
    cur_dist = PROJECT ( &exc->zp1.cur[point], &exc->zp0.cur[exc->GS.rp0] );

    /* auto-flip test */

    if ( exc->GS.auto_flip )
    {
      if ( ( org_dist ^ cvt_dist ) < 0 )
        cvt_dist = NEG_LONG( cvt_dist );
    }

    /* control value cut-in and round */

    if ( ( exc->opcode & 4 ) != 0 )
    {
      /* XXX: UNDOCUMENTED!  Only perform cut-in test when both points */
      /*      refer to the same zone.                                  */

      if ( exc->GS.gep0 == exc->GS.gep1 )
      {
        FT_F26Dot6  control_value_cutin = exc->GS.control_value_cutin;


        /* XXX: According to Greg Hitchcock, the following wording is */
        /*      the right one:                                        */
        /*                                                            */
        /*        When the absolute difference between the value in   */
        /*        the table [CVT] and the measurement directly from   */
        /*        the outline is _greater_ than the cut_in value, the */
        /*        outline measurement is used.                        */
        /*                                                            */
        /*      This is from `instgly.doc'.  The description in       */
        /*      `ttinst2.doc', version 1.66, is thus incorrect since  */
        /*      it implies `>=' instead of `>'.                       */

        delta = SUB_LONG( cvt_dist, org_dist );
        if ( delta < 0 )
          delta = NEG_LONG( delta );

        if ( delta > control_value_cutin )
          cvt_dist = org_dist;
      }

      distance = exc->func_round( exc, cvt_dist, exc->opcode & 3 );
    }
    else
    {

#ifdef TT_SUPPORT_SUBPIXEL_HINTING_INFINALITY
      /* do cvt cut-in always in MIRP for sph */
      if ( SUBPIXEL_HINTING_INFINALITY  &&
           exc->ignore_x_mode           &&
           exc->GS.gep0 == exc->GS.gep1 )
      {
        FT_F26Dot6  control_value_cutin = exc->GS.control_value_cutin;


        if ( exc->GS.freeVector.x != 0                          &&
             !( exc->sph_tweak_flags & SPH_TWEAK_NORMAL_ROUND ) )
          control_value_cutin = 0;

        if ( exc->GS.freeVector.y != 0                                 &&
             ( exc->sph_tweak_flags & SPH_TWEAK_TIMES_NEW_ROMAN_HACK ) )
        {
          if ( cur_dist < -64 )
            cvt_dist -= 16;
          else if ( cur_dist > 64 && cur_dist < 84 )
            cvt_dist += 32;
        }

        delta = SUB_LONG( cvt_dist, org_dist );
        if ( delta < 0 )
          delta = NEG_LONG( delta );

        if ( delta > control_value_cutin )
          cvt_dist = org_dist;
      }
#endif /* TT_SUPPORT_SUBPIXEL_HINTING_INFINALITY */

      distance = Round_None( exc, cvt_dist, exc->opcode & 3 );
    }

    /* minimum distance test */

    if ( ( exc->opcode & 8 ) != 0 )
    {
      FT_F26Dot6  minimum_distance    = exc->GS.minimum_distance;


#ifdef TT_SUPPORT_SUBPIXEL_HINTING_INFINALITY
      if ( SUBPIXEL_HINTING_INFINALITY                        &&
           exc->ignore_x_mode                                 &&
           exc->GS.freeVector.x != 0                          &&
           !( exc->sph_tweak_flags & SPH_TWEAK_NORMAL_ROUND ) )
        minimum_distance = 0;
#endif /* TT_SUPPORT_SUBPIXEL_HINTING_INFINALITY */

      if ( org_dist >= 0 )
      {
        if ( distance < minimum_distance )
          distance = minimum_distance;
      }
      else
      {
        if ( distance > NEG_LONG( minimum_distance ) )
          distance = NEG_LONG( minimum_distance );
      }
    }

#ifdef TT_SUPPORT_SUBPIXEL_HINTING_INFINALITY
    if ( SUBPIXEL_HINTING_INFINALITY &&
         exc->ignore_x_mode          &&
         exc->GS.freeVector.y != 0   )
    {
      FT_Int   B1, B2;


      B1 = exc->zp1.cur[point].y;

      /* Round moves if necessary */
      if ( exc->sph_tweak_flags & SPH_TWEAK_ROUND_NONPIXEL_Y_MOVES )
        distance = FT_PIX_ROUND( B1 + distance - cur_dist ) - B1 + cur_dist;

      if ( ( exc->opcode & 16 ) == 0                               &&
           ( exc->opcode & 8 ) == 0                                &&
           ( exc->sph_tweak_flags & SPH_TWEAK_COURIER_NEW_2_HACK ) )
        distance += 64;

      exc->func_move( exc,
                      &exc->zp1,
                      point,
                      SUB_LONG( distance, cur_dist ) );

      B2 = exc->zp1.cur[point].y;

      /* Reverse move if necessary */
      if ( ( exc->face->sph_compatibility_mode &&
             ( B1 & 63 ) == 0                  &&
             ( B2 & 63 ) != 0                  )                          ||
           ( ( exc->sph_tweak_flags & SPH_TWEAK_SKIP_NONPIXEL_Y_MOVES ) &&
             ( B1 & 63 ) != 0                                           &&
             ( B2 & 63 ) != 0                                           ) )
        exc->func_move( exc,
                        &exc->zp1,
                        point,
                        SUB_LONG( cur_dist, distance ) );
    }
    else
#endif /* TT_SUPPORT_SUBPIXEL_HINTING_INFINALITY */

      exc->func_move( exc,
                      &exc->zp1,
                      point,
                      SUB_LONG( distance, cur_dist ) );

  Fail:
    exc->GS.rp1 = exc->GS.rp0;

    if ( ( exc->opcode & 16 ) != 0 )
      exc->GS.rp0 = point;

    exc->GS.rp2 = point;
  }


  /**************************************************************************
   *
   * ALIGNRP[]:    ALIGN Relative Point
   * Opcode range: 0x3C
   * Stack:        uint32 uint32... -->
   */
  static void
  Ins_ALIGNRP( TT_ExecContext  exc )
  {
    FT_UShort   point;
    FT_F26Dot6  distance;


#ifdef TT_SUPPORT_SUBPIXEL_HINTING_INFINALITY
    if ( SUBPIXEL_HINTING_INFINALITY                               &&
         exc->ignore_x_mode                                        &&
         exc->iup_called                                           &&
         ( exc->sph_tweak_flags & SPH_TWEAK_NO_ALIGNRP_AFTER_IUP ) )
    {
      exc->error = FT_THROW( Invalid_Reference );
      goto Fail;
    }
#endif /* TT_SUPPORT_SUBPIXEL_HINTING_INFINALITY */

    if ( exc->top < exc->GS.loop                  ||
         BOUNDS( exc->GS.rp0, exc->zp0.n_points ) )
    {
      if ( exc->pedantic_hinting )
        exc->error = FT_THROW( Invalid_Reference );
      goto Fail;
    }

    while ( exc->GS.loop > 0 )
    {
      exc->args--;

      point = (FT_UShort)exc->stack[exc->args];

      if ( BOUNDS( point, exc->zp1.n_points ) )
      {
        if ( exc->pedantic_hinting )
        {
          exc->error = FT_THROW( Invalid_Reference );
          return;
        }
      }
      else
      {
        distance = PROJECT( exc->zp1.cur + point,
                            exc->zp0.cur + exc->GS.rp0 );

        exc->func_move( exc, &exc->zp1, point, NEG_LONG( distance ) );
      }

      exc->GS.loop--;
    }

  Fail:
    exc->GS.loop = 1;
    exc->new_top = exc->args;
  }


  /**************************************************************************
   *
   * ISECT[]:      moves point to InterSECTion
   * Opcode range: 0x0F
   * Stack:        5 * uint32 -->
   */
  static void
  Ins_ISECT( TT_ExecContext  exc,
             FT_Long*        args )
  {
    FT_UShort   point,
                a0, a1,
                b0, b1;

    FT_F26Dot6  discriminant, dotproduct;

    FT_F26Dot6  dx,  dy,
                dax, day,
                dbx, dby;

    FT_F26Dot6  val;

    FT_Vector   R;


    point = (FT_UShort)args[0];

    a0 = (FT_UShort)args[1];
    a1 = (FT_UShort)args[2];
    b0 = (FT_UShort)args[3];
    b1 = (FT_UShort)args[4];

    if ( BOUNDS( b0,    exc->zp0.n_points ) ||
         BOUNDS( b1,    exc->zp0.n_points ) ||
         BOUNDS( a0,    exc->zp1.n_points ) ||
         BOUNDS( a1,    exc->zp1.n_points ) ||
         BOUNDS( point, exc->zp2.n_points ) )
    {
      if ( exc->pedantic_hinting )
        exc->error = FT_THROW( Invalid_Reference );
      return;
    }

    /* Cramer's rule */

    dbx = SUB_LONG( exc->zp0.cur[b1].x, exc->zp0.cur[b0].x );
    dby = SUB_LONG( exc->zp0.cur[b1].y, exc->zp0.cur[b0].y );

    dax = SUB_LONG( exc->zp1.cur[a1].x, exc->zp1.cur[a0].x );
    day = SUB_LONG( exc->zp1.cur[a1].y, exc->zp1.cur[a0].y );

    dx = SUB_LONG( exc->zp0.cur[b0].x, exc->zp1.cur[a0].x );
    dy = SUB_LONG( exc->zp0.cur[b0].y, exc->zp1.cur[a0].y );

    discriminant = ADD_LONG( FT_MulDiv( dax, NEG_LONG( dby ), 0x40 ),
                             FT_MulDiv( day, dbx, 0x40 ) );
    dotproduct   = ADD_LONG( FT_MulDiv( dax, dbx, 0x40 ),
                             FT_MulDiv( day, dby, 0x40 ) );

    /* The discriminant above is actually a cross product of vectors     */
    /* da and db. Together with the dot product, they can be used as     */
    /* surrogates for sine and cosine of the angle between the vectors.  */
    /* Indeed,                                                           */
    /*       dotproduct   = |da||db|cos(angle)                           */
    /*       discriminant = |da||db|sin(angle)     .                     */
    /* We use these equations to reject grazing intersections by         */
    /* thresholding abs(tan(angle)) at 1/19, corresponding to 3 degrees. */
    if ( MUL_LONG( 19, FT_ABS( discriminant ) ) > FT_ABS( dotproduct ) )
    {
      val = ADD_LONG( FT_MulDiv( dx, NEG_LONG( dby ), 0x40 ),
                      FT_MulDiv( dy, dbx, 0x40 ) );

      R.x = FT_MulDiv( val, dax, discriminant );
      R.y = FT_MulDiv( val, day, discriminant );

      /* XXX: Block in backward_compatibility and/or post-IUP? */
      exc->zp2.cur[point].x = ADD_LONG( exc->zp1.cur[a0].x, R.x );
      exc->zp2.cur[point].y = ADD_LONG( exc->zp1.cur[a0].y, R.y );
    }
    else
    {
      /* else, take the middle of the middles of A and B */

      /* XXX: Block in backward_compatibility and/or post-IUP? */
      exc->zp2.cur[point].x =
        ADD_LONG( ADD_LONG( exc->zp1.cur[a0].x, exc->zp1.cur[a1].x ),
                  ADD_LONG( exc->zp0.cur[b0].x, exc->zp0.cur[b1].x ) ) / 4;
      exc->zp2.cur[point].y =
        ADD_LONG( ADD_LONG( exc->zp1.cur[a0].y, exc->zp1.cur[a1].y ),
                  ADD_LONG( exc->zp0.cur[b0].y, exc->zp0.cur[b1].y ) ) / 4;
    }

    exc->zp2.tags[point] |= FT_CURVE_TAG_TOUCH_BOTH;
  }


  /**************************************************************************
   *
   * ALIGNPTS[]:   ALIGN PoinTS
   * Opcode range: 0x27
   * Stack:        uint32 uint32 -->
   */
  static void
  Ins_ALIGNPTS( TT_ExecContext  exc,
                FT_Long*        args )
  {
    FT_UShort   p1, p2;
    FT_F26Dot6  distance;


    p1 = (FT_UShort)args[0];
    p2 = (FT_UShort)args[1];

    if ( BOUNDS( p1, exc->zp1.n_points ) ||
         BOUNDS( p2, exc->zp0.n_points ) )
    {
      if ( exc->pedantic_hinting )
        exc->error = FT_THROW( Invalid_Reference );
      return;
    }

    distance = PROJECT( exc->zp0.cur + p2, exc->zp1.cur + p1 ) / 2;

    exc->func_move( exc, &exc->zp1, p1, distance );
    exc->func_move( exc, &exc->zp0, p2, NEG_LONG( distance ) );
  }


  /**************************************************************************
   *
   * IP[]:         Interpolate Point
   * Opcode range: 0x39
   * Stack:        uint32... -->
   */

  /* SOMETIMES, DUMBER CODE IS BETTER CODE */

  static void
  Ins_IP( TT_ExecContext  exc )
  {
    FT_F26Dot6  old_range, cur_range;
    FT_Vector*  orus_base;
    FT_Vector*  cur_base;
    FT_Int      twilight;


    if ( exc->top < exc->GS.loop )
    {
      if ( exc->pedantic_hinting )
        exc->error = FT_THROW( Invalid_Reference );
      goto Fail;
    }

    /*
     * We need to deal in a special way with the twilight zone.
     * Otherwise, by definition, the value of exc->twilight.orus[n] is (0,0),
     * for every n.
     */
    twilight = ( exc->GS.gep0 == 0 ||
                 exc->GS.gep1 == 0 ||
                 exc->GS.gep2 == 0 );

    if ( BOUNDS( exc->GS.rp1, exc->zp0.n_points ) )
    {
      if ( exc->pedantic_hinting )
        exc->error = FT_THROW( Invalid_Reference );
      goto Fail;
    }

    if ( twilight )
      orus_base = &exc->zp0.org[exc->GS.rp1];
    else
      orus_base = &exc->zp0.orus[exc->GS.rp1];

    cur_base = &exc->zp0.cur[exc->GS.rp1];

    /* XXX: There are some glyphs in some braindead but popular */
    /*      fonts out there (e.g. [aeu]grave in monotype.ttf)   */
    /*      calling IP[] with bad values of rp[12].             */
    /*      Do something sane when this odd thing happens.      */
    if ( BOUNDS( exc->GS.rp1, exc->zp0.n_points ) ||
         BOUNDS( exc->GS.rp2, exc->zp1.n_points ) )
    {
      old_range = 0;
      cur_range = 0;
    }
    else
    {
      if ( twilight )
        old_range = DUALPROJ( &exc->zp1.org[exc->GS.rp2], orus_base );
      else if ( exc->metrics.x_scale == exc->metrics.y_scale )
        old_range = DUALPROJ( &exc->zp1.orus[exc->GS.rp2], orus_base );
      else
      {
        FT_Vector  vec;


        vec.x = FT_MulFix( SUB_LONG( exc->zp1.orus[exc->GS.rp2].x,
                                     orus_base->x ),
                           exc->metrics.x_scale );
        vec.y = FT_MulFix( SUB_LONG( exc->zp1.orus[exc->GS.rp2].y,
                                     orus_base->y ),
                           exc->metrics.y_scale );

        old_range = FAST_DUALPROJ( &vec );
      }

      cur_range = PROJECT( &exc->zp1.cur[exc->GS.rp2], cur_base );
    }

    for ( ; exc->GS.loop > 0; exc->GS.loop-- )
    {
      FT_UInt     point = (FT_UInt)exc->stack[--exc->args];
      FT_F26Dot6  org_dist, cur_dist, new_dist;


      /* check point bounds */
      if ( BOUNDS( point, exc->zp2.n_points ) )
      {
        if ( exc->pedantic_hinting )
        {
          exc->error = FT_THROW( Invalid_Reference );
          return;
        }
        continue;
      }

      if ( twilight )
        org_dist = DUALPROJ( &exc->zp2.org[point], orus_base );
      else if ( exc->metrics.x_scale == exc->metrics.y_scale )
        org_dist = DUALPROJ( &exc->zp2.orus[point], orus_base );
      else
      {
        FT_Vector  vec;


        vec.x = FT_MulFix( SUB_LONG( exc->zp2.orus[point].x,
                                     orus_base->x ),
                           exc->metrics.x_scale );
        vec.y = FT_MulFix( SUB_LONG( exc->zp2.orus[point].y,
                                     orus_base->y ),
                           exc->metrics.y_scale );

        org_dist = FAST_DUALPROJ( &vec );
      }

      cur_dist = PROJECT( &exc->zp2.cur[point], cur_base );

      if ( org_dist )
      {
        if ( old_range )
          new_dist = FT_MulDiv( org_dist, cur_range, old_range );
        else
        {
          /* This is the same as what MS does for the invalid case:  */
          /*                                                         */
          /*   delta = (Original_Pt - Original_RP1) -                */
          /*           (Current_Pt - Current_RP1)         ;          */
          /*                                                         */
          /* In FreeType speak:                                      */
          /*                                                         */
          /*   delta = org_dist - cur_dist          .                */
          /*                                                         */
          /* We move `point' by `new_dist - cur_dist' after leaving  */
          /* this block, thus we have                                */
          /*                                                         */
          /*   new_dist - cur_dist = delta                   ,       */
          /*   new_dist - cur_dist = org_dist - cur_dist     ,       */
          /*              new_dist = org_dist                .       */

          new_dist = org_dist;
        }
      }
      else
        new_dist = 0;

      exc->func_move( exc,
                      &exc->zp2,
                      (FT_UShort)point,
                      SUB_LONG( new_dist, cur_dist ) );
    }

  Fail:
    exc->GS.loop = 1;
    exc->new_top = exc->args;
  }


  /**************************************************************************
   *
   * UTP[a]:       UnTouch Point
   * Opcode range: 0x29
   * Stack:        uint32 -->
   */
  static void
  Ins_UTP( TT_ExecContext  exc,
           FT_Long*        args )
  {
    FT_UShort  point;
    FT_Byte    mask;


    point = (FT_UShort)args[0];

    if ( BOUNDS( point, exc->zp0.n_points ) )
    {
      if ( exc->pedantic_hinting )
        exc->error = FT_THROW( Invalid_Reference );
      return;
    }

    mask = 0xFF;

    if ( exc->GS.freeVector.x != 0 )
      mask &= ~FT_CURVE_TAG_TOUCH_X;

    if ( exc->GS.freeVector.y != 0 )
      mask &= ~FT_CURVE_TAG_TOUCH_Y;

    exc->zp0.tags[point] &= mask;
  }


  /* Local variables for Ins_IUP: */
  typedef struct  IUP_WorkerRec_
  {
    FT_Vector*  orgs;   /* original and current coordinate */
    FT_Vector*  curs;   /* arrays                          */
    FT_Vector*  orus;
    FT_UInt     max_points;

  } IUP_WorkerRec, *IUP_Worker;


  static void
  _iup_worker_shift( IUP_Worker  worker,
                     FT_UInt     p1,
                     FT_UInt     p2,
                     FT_UInt     p )
  {
    FT_UInt     i;
    FT_F26Dot6  dx;


    dx = SUB_LONG( worker->curs[p].x, worker->orgs[p].x );
    if ( dx != 0 )
    {
      for ( i = p1; i < p; i++ )
        worker->curs[i].x = ADD_LONG( worker->curs[i].x, dx );

      for ( i = p + 1; i <= p2; i++ )
        worker->curs[i].x = ADD_LONG( worker->curs[i].x, dx );
    }
  }


  static void
  _iup_worker_interpolate( IUP_Worker  worker,
                           FT_UInt     p1,
                           FT_UInt     p2,
                           FT_UInt     ref1,
                           FT_UInt     ref2 )
  {
    FT_UInt     i;
    FT_F26Dot6  orus1, orus2, org1, org2, cur1, cur2, delta1, delta2;


    if ( p1 > p2 )
      return;

    if ( BOUNDS( ref1, worker->max_points ) ||
         BOUNDS( ref2, worker->max_points ) )
      return;

    orus1 = worker->orus[ref1].x;
    orus2 = worker->orus[ref2].x;

    if ( orus1 > orus2 )
    {
      FT_F26Dot6  tmp_o;
      FT_UInt     tmp_r;


      tmp_o = orus1;
      orus1 = orus2;
      orus2 = tmp_o;

      tmp_r = ref1;
      ref1  = ref2;
      ref2  = tmp_r;
    }

    org1   = worker->orgs[ref1].x;
    org2   = worker->orgs[ref2].x;
    cur1   = worker->curs[ref1].x;
    cur2   = worker->curs[ref2].x;
    delta1 = SUB_LONG( cur1, org1 );
    delta2 = SUB_LONG( cur2, org2 );

    if ( cur1 == cur2 || orus1 == orus2 )
    {

      /* trivial snap or shift of untouched points */
      for ( i = p1; i <= p2; i++ )
      {
        FT_F26Dot6  x = worker->orgs[i].x;


        if ( x <= org1 )
          x = ADD_LONG( x, delta1 );

        else if ( x >= org2 )
          x = ADD_LONG( x, delta2 );

        else
          x = cur1;

        worker->curs[i].x = x;
      }
    }
    else
    {
      FT_Fixed  scale       = 0;
      FT_Bool   scale_valid = 0;


      /* interpolation */
      for ( i = p1; i <= p2; i++ )
      {
        FT_F26Dot6  x = worker->orgs[i].x;


        if ( x <= org1 )
          x = ADD_LONG( x, delta1 );

        else if ( x >= org2 )
          x = ADD_LONG( x, delta2 );

        else
        {
          if ( !scale_valid )
          {
            scale_valid = 1;
            scale       = FT_DivFix( SUB_LONG( cur2, cur1 ),
                                     SUB_LONG( orus2, orus1 ) );
          }

          x = ADD_LONG( cur1,
                        FT_MulFix( SUB_LONG( worker->orus[i].x, orus1 ),
                                   scale ) );
        }
        worker->curs[i].x = x;
      }
    }
  }


  /**************************************************************************
   *
   * IUP[a]:       Interpolate Untouched Points
   * Opcode range: 0x30-0x31
   * Stack:        -->
   */
  static void
  Ins_IUP( TT_ExecContext  exc )
  {
    IUP_WorkerRec  V;
    FT_Byte        mask;

    FT_UInt   first_point;   /* first point of contour        */
    FT_UInt   end_point;     /* end point (last+1) of contour */

    FT_UInt   first_touched; /* first touched point in contour   */
    FT_UInt   cur_touched;   /* current touched point in contour */

    FT_UInt   point;         /* current point   */
    FT_Short  contour;       /* current contour */


#ifdef TT_SUPPORT_SUBPIXEL_HINTING_MINIMAL
    /* See `ttinterp.h' for details on backward compatibility mode.  */
    /* Allow IUP until it has been called on both axes.  Immediately */
    /* return on subsequent ones.                                    */
    if ( SUBPIXEL_HINTING_MINIMAL    &&
         exc->backward_compatibility )
    {
      if ( exc->iupx_called && exc->iupy_called )
        return;

      if ( exc->opcode & 1 )
        exc->iupx_called = TRUE;
      else
        exc->iupy_called = TRUE;
    }
#endif

    /* ignore empty outlines */
    if ( exc->pts.n_contours == 0 )
      return;

    if ( exc->opcode & 1 )
    {
      mask   = FT_CURVE_TAG_TOUCH_X;
      V.orgs = exc->pts.org;
      V.curs = exc->pts.cur;
      V.orus = exc->pts.orus;
    }
    else
    {
      mask   = FT_CURVE_TAG_TOUCH_Y;
      V.orgs = (FT_Vector*)( (FT_Pos*)exc->pts.org + 1 );
      V.curs = (FT_Vector*)( (FT_Pos*)exc->pts.cur + 1 );
      V.orus = (FT_Vector*)( (FT_Pos*)exc->pts.orus + 1 );
    }
    V.max_points = exc->pts.n_points;

    contour = 0;
    point   = 0;

#ifdef TT_SUPPORT_SUBPIXEL_HINTING_INFINALITY
    if ( SUBPIXEL_HINTING_INFINALITY &&
         exc->ignore_x_mode          )
    {
      exc->iup_called = TRUE;
      if ( exc->sph_tweak_flags & SPH_TWEAK_SKIP_IUP )
        return;
    }
#endif /* TT_SUPPORT_SUBPIXEL_HINTING_INFINALITY */

    do
    {
      end_point   = exc->pts.contours[contour] - exc->pts.first_point;
      first_point = point;

      if ( BOUNDS( end_point, exc->pts.n_points ) )
        end_point = exc->pts.n_points - 1;

      while ( point <= end_point && ( exc->pts.tags[point] & mask ) == 0 )
        point++;

      if ( point <= end_point )
      {
        first_touched = point;
        cur_touched   = point;

        point++;

        while ( point <= end_point )
        {
          if ( ( exc->pts.tags[point] & mask ) != 0 )
          {
            _iup_worker_interpolate( &V,
                                     cur_touched + 1,
                                     point - 1,
                                     cur_touched,
                                     point );
            cur_touched = point;
          }

          point++;
        }

        if ( cur_touched == first_touched )
          _iup_worker_shift( &V, first_point, end_point, cur_touched );
        else
        {
          _iup_worker_interpolate( &V,
                                   (FT_UShort)( cur_touched + 1 ),
                                   end_point,
                                   cur_touched,
                                   first_touched );

          if ( first_touched > 0 )
            _iup_worker_interpolate( &V,
                                     first_point,
                                     first_touched - 1,
                                     cur_touched,
                                     first_touched );
        }
      }
      contour++;
    } while ( contour < exc->pts.n_contours );
  }


  /**************************************************************************
   *
   * DELTAPn[]:    DELTA exceptions P1, P2, P3
   * Opcode range: 0x5D,0x71,0x72
   * Stack:        uint32 (2 * uint32)... -->
   */
  static void
  Ins_DELTAP( TT_ExecContext  exc,
              FT_Long*        args )
  {
    FT_ULong   nump, k;
    FT_UShort  A;
    FT_ULong   C, P;
    FT_Long    B;


#ifdef TT_SUPPORT_SUBPIXEL_HINTING_INFINALITY
    if ( SUBPIXEL_HINTING_INFINALITY                              &&
         exc->ignore_x_mode                                       &&
         exc->iup_called                                          &&
         ( exc->sph_tweak_flags & SPH_TWEAK_NO_DELTAP_AFTER_IUP ) )
      goto Fail;
#endif /* TT_SUPPORT_SUBPIXEL_HINTING_INFINALITY */

    P    = (FT_ULong)exc->func_cur_ppem( exc );
    nump = (FT_ULong)args[0];   /* some points theoretically may occur more
                                   than once, thus UShort isn't enough */

    for ( k = 1; k <= nump; k++ )
    {
      if ( exc->args < 2 )
      {
        if ( exc->pedantic_hinting )
          exc->error = FT_THROW( Too_Few_Arguments );
        exc->args = 0;
        goto Fail;
      }

      exc->args -= 2;

      A = (FT_UShort)exc->stack[exc->args + 1];
      B = exc->stack[exc->args];

      /* XXX: Because some popular fonts contain some invalid DeltaP */
      /*      instructions, we simply ignore them when the stacked   */
      /*      point reference is off limit, rather than returning an */
      /*      error.  As a delta instruction doesn't change a glyph  */
      /*      in great ways, this shouldn't be a problem.            */

      if ( !BOUNDS( A, exc->zp0.n_points ) )
      {
        C = ( (FT_ULong)B & 0xF0 ) >> 4;

        switch ( exc->opcode )
        {
        case 0x5D:
          break;

        case 0x71:
          C += 16;
          break;

        case 0x72:
          C += 32;
          break;
        }

        C += exc->GS.delta_base;

        if ( P == C )
        {
          B = ( (FT_ULong)B & 0xF ) - 8;
          if ( B >= 0 )
            B++;
          B *= 1L << ( 6 - exc->GS.delta_shift );

#ifdef TT_SUPPORT_SUBPIXEL_HINTING_INFINALITY

          if ( SUBPIXEL_HINTING_INFINALITY )
          {
            /*
             * Allow delta move if
             *
             * - not using ignore_x_mode rendering,
             * - glyph is specifically set to allow it, or
             * - glyph is composite and freedom vector is not in subpixel
             *   direction.
             */
            if ( !exc->ignore_x_mode                                   ||
                 ( exc->sph_tweak_flags & SPH_TWEAK_ALWAYS_DO_DELTAP ) ||
                 ( exc->is_composite && exc->GS.freeVector.y != 0 )    )
              exc->func_move( exc, &exc->zp0, A, B );

            /* Otherwise, apply subpixel hinting and compatibility mode */
            /* rules, always skipping deltas in subpixel direction.     */
            else if ( exc->ignore_x_mode && exc->GS.freeVector.y != 0 )
            {
              FT_UShort  B1, B2;


              /* save the y value of the point now; compare after move */
              B1 = (FT_UShort)exc->zp0.cur[A].y;

              /* Standard subpixel hinting: Allow y move for y-touched */
              /* points.  This messes up DejaVu ...                    */
              if ( !exc->face->sph_compatibility_mode          &&
                   ( exc->zp0.tags[A] & FT_CURVE_TAG_TOUCH_Y ) )
                exc->func_move( exc, &exc->zp0, A, B );

              /* compatibility mode */
              else if ( exc->face->sph_compatibility_mode                        &&
                        !( exc->sph_tweak_flags & SPH_TWEAK_ALWAYS_SKIP_DELTAP ) )
              {
                if ( exc->sph_tweak_flags & SPH_TWEAK_ROUND_NONPIXEL_Y_MOVES )
                  B = FT_PIX_ROUND( B1 + B ) - B1;

                /* Allow delta move if using sph_compatibility_mode,   */
                /* IUP has not been called, and point is touched on Y. */
                if ( !exc->iup_called                            &&
                     ( exc->zp0.tags[A] & FT_CURVE_TAG_TOUCH_Y ) )
                  exc->func_move( exc, &exc->zp0, A, B );
              }

              B2 = (FT_UShort)exc->zp0.cur[A].y;

              /* Reverse this move if it results in a disallowed move */
              if ( exc->GS.freeVector.y != 0                          &&
                   ( ( exc->face->sph_compatibility_mode          &&
                       ( B1 & 63 ) == 0                           &&
                       ( B2 & 63 ) != 0                           ) ||
                     ( ( exc->sph_tweak_flags                   &
                         SPH_TWEAK_SKIP_NONPIXEL_Y_MOVES_DELTAP ) &&
                       ( B1 & 63 ) != 0                           &&
                       ( B2 & 63 ) != 0                           ) ) )
                exc->func_move( exc, &exc->zp0, A, NEG_LONG( B ) );
            }
          }
          else
#endif /* TT_SUPPORT_SUBPIXEL_HINTING_INFINALITY */

          {

#ifdef TT_SUPPORT_SUBPIXEL_HINTING_MINIMAL
            /* See `ttinterp.h' for details on backward compatibility */
            /* mode.                                                  */
            if ( SUBPIXEL_HINTING_MINIMAL    &&
                 exc->backward_compatibility )
            {
              if ( !( exc->iupx_called && exc->iupy_called )              &&
                   ( ( exc->is_composite && exc->GS.freeVector.y != 0 ) ||
                     ( exc->zp0.tags[A] & FT_CURVE_TAG_TOUCH_Y )        ) )
                exc->func_move( exc, &exc->zp0, A, B );
            }
            else
#endif
              exc->func_move( exc, &exc->zp0, A, B );
          }
        }
      }
      else
        if ( exc->pedantic_hinting )
          exc->error = FT_THROW( Invalid_Reference );
    }

  Fail:
    exc->new_top = exc->args;
  }


  /**************************************************************************
   *
   * DELTACn[]:    DELTA exceptions C1, C2, C3
   * Opcode range: 0x73,0x74,0x75
   * Stack:        uint32 (2 * uint32)... -->
   */
  static void
  Ins_DELTAC( TT_ExecContext  exc,
              FT_Long*        args )
  {
    FT_ULong  nump, k;
    FT_ULong  A, C, P;
    FT_Long   B;


    P    = (FT_ULong)exc->func_cur_ppem( exc );
    nump = (FT_ULong)args[0];

    for ( k = 1; k <= nump; k++ )
    {
      if ( exc->args < 2 )
      {
        if ( exc->pedantic_hinting )
          exc->error = FT_THROW( Too_Few_Arguments );
        exc->args = 0;
        goto Fail;
      }

      exc->args -= 2;

      A = (FT_ULong)exc->stack[exc->args + 1];
      B = exc->stack[exc->args];

      if ( BOUNDSL( A, exc->cvtSize ) )
      {
        if ( exc->pedantic_hinting )
        {
          exc->error = FT_THROW( Invalid_Reference );
          return;
        }
      }
      else
      {
        C = ( (FT_ULong)B & 0xF0 ) >> 4;

        switch ( exc->opcode )
        {
        case 0x73:
          break;

        case 0x74:
          C += 16;
          break;

        case 0x75:
          C += 32;
          break;
        }

        C += exc->GS.delta_base;

        if ( P == C )
        {
          B = ( (FT_ULong)B & 0xF ) - 8;
          if ( B >= 0 )
            B++;
          B *= 1L << ( 6 - exc->GS.delta_shift );

          exc->func_move_cvt( exc, A, B );
        }
      }
    }

  Fail:
    exc->new_top = exc->args;
  }


  /**************************************************************************
   *
   * MISC. INSTRUCTIONS
   *
   */


  /**************************************************************************
   *
   * GETINFO[]:    GET INFOrmation
   * Opcode range: 0x88
   * Stack:        uint32 --> uint32
   *
   * XXX: UNDOCUMENTED: Selector bits higher than 9 are currently (May
   *      2015) not documented in the OpenType specification.
   *
   *      Selector bit 11 is incorrectly described as bit 8, while the
   *      real meaning of bit 8 (vertical LCD subpixels) stays
   *      undocumented.  The same mistake can be found in Greg Hitchcock's
   *      whitepaper.
   */
  static void
  Ins_GETINFO( TT_ExecContext  exc,
               FT_Long*        args )
  {
    FT_Long    K;
    TT_Driver  driver = (TT_Driver)FT_FACE_DRIVER( exc->face );


    K = 0;

#ifdef TT_SUPPORT_SUBPIXEL_HINTING_INFINALITY
    /*********************************
     * RASTERIZER VERSION
     * Selector Bit:  0
     * Return Bit(s): 0-7
     */
    if ( SUBPIXEL_HINTING_INFINALITY &&
         ( args[0] & 1 ) != 0        &&
         exc->subpixel_hinting       )
    {
      if ( exc->ignore_x_mode )
      {
        /* if in ClearType backward compatibility mode,         */
        /* we sometimes change the TrueType version dynamically */
        K = exc->rasterizer_version;
        FT_TRACE6(( "Setting rasterizer version %d\n",
                    exc->rasterizer_version ));
      }
      else
        K = TT_INTERPRETER_VERSION_38;
    }
    else
#endif /* TT_SUPPORT_SUBPIXEL_HINTING_INFINALITY */
      if ( ( args[0] & 1 ) != 0 )
        K = driver->interpreter_version;

    /*********************************
     * GLYPH ROTATED
     * Selector Bit:  1
     * Return Bit(s): 8
     */
    if ( ( args[0] & 2 ) != 0 && exc->tt_metrics.rotated )
      K |= 1 << 8;

    /*********************************
     * GLYPH STRETCHED
     * Selector Bit:  2
     * Return Bit(s): 9
     */
    if ( ( args[0] & 4 ) != 0 && exc->tt_metrics.stretched )
      K |= 1 << 9;

#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT
    /*********************************
     * VARIATION GLYPH
     * Selector Bit:  3
     * Return Bit(s): 10
     *
     * XXX: UNDOCUMENTED!
     */
    if ( (args[0] & 8 ) != 0 && exc->face->blend )
      K |= 1 << 10;
#endif

    /*********************************
     * BI-LEVEL HINTING AND
     * GRAYSCALE RENDERING
     * Selector Bit:  5
     * Return Bit(s): 12
     */
    if ( ( args[0] & 32 ) != 0 && exc->grayscale )
      K |= 1 << 12;

#ifdef TT_SUPPORT_SUBPIXEL_HINTING_MINIMAL
    /* Toggle the following flags only outside of monochrome mode.      */
    /* Otherwise, instructions may behave weirdly and rendering results */
    /* may differ between v35 and v40 mode, e.g., in `Times New Roman   */
    /* Bold Italic'. */
    if ( SUBPIXEL_HINTING_MINIMAL && exc->subpixel_hinting_lean )
    {
      /*********************************
       * HINTING FOR SUBPIXEL
       * Selector Bit:  6
       * Return Bit(s): 13
       *
       * v40 does subpixel hinting by default.
       */
      if ( ( args[0] & 64 ) != 0 )
        K |= 1 << 13;

      /*********************************
       * VERTICAL LCD SUBPIXELS?
       * Selector Bit:  8
       * Return Bit(s): 15
       */
      if ( ( args[0] & 256 ) != 0 && exc->vertical_lcd_lean )
        K |= 1 << 15;

      /*********************************
       * SUBPIXEL POSITIONED?
       * Selector Bit:  10
       * Return Bit(s): 17
       *
       * XXX: FreeType supports it, dependent on what client does?
       */
      if ( ( args[0] & 1024 ) != 0 )
        K |= 1 << 17;

      /*********************************
       * SYMMETRICAL SMOOTHING
       * Selector Bit:  11
       * Return Bit(s): 18
       *
       * The only smoothing method FreeType supports unless someone sets
       * FT_LOAD_TARGET_MONO.
       */
      if ( ( args[0] & 2048 ) != 0 && exc->subpixel_hinting_lean )
        K |= 1 << 18;

      /*********************************
       * CLEARTYPE HINTING AND
       * GRAYSCALE RENDERING
       * Selector Bit:  12
       * Return Bit(s): 19
       *
       * Grayscale rendering is what FreeType does anyway unless someone
       * sets FT_LOAD_TARGET_MONO or FT_LOAD_TARGET_LCD(_V)
       */
      if ( ( args[0] & 4096 ) != 0 && exc->grayscale_cleartype )
        K |= 1 << 19;
    }
#endif

#ifdef TT_SUPPORT_SUBPIXEL_HINTING_INFINALITY

    if ( SUBPIXEL_HINTING_INFINALITY                          &&
         exc->rasterizer_version >= TT_INTERPRETER_VERSION_35 )
    {

      if ( exc->rasterizer_version >= 37 )
      {
        /*********************************
         * HINTING FOR SUBPIXEL
         * Selector Bit:  6
         * Return Bit(s): 13
         */
        if ( ( args[0] & 64 ) != 0 && exc->subpixel_hinting )
          K |= 1 << 13;

        /*********************************
         * COMPATIBLE WIDTHS ENABLED
         * Selector Bit:  7
         * Return Bit(s): 14
         *
         * Functionality still needs to be added
         */
        if ( ( args[0] & 128 ) != 0 && exc->compatible_widths )
          K |= 1 << 14;

        /*********************************
         * VERTICAL LCD SUBPIXELS?
         * Selector Bit:  8
         * Return Bit(s): 15
         *
         * Functionality still needs to be added
         */
        if ( ( args[0] & 256 ) != 0 && exc->vertical_lcd )
          K |= 1 << 15;

        /*********************************
         * HINTING FOR BGR?
         * Selector Bit:  9
         * Return Bit(s): 16
         *
         * Functionality still needs to be added
         */
        if ( ( args[0] & 512 ) != 0 && exc->bgr )
          K |= 1 << 16;

        if ( exc->rasterizer_version >= 38 )
        {
          /*********************************
           * SUBPIXEL POSITIONED?
           * Selector Bit:  10
           * Return Bit(s): 17
           *
           * Functionality still needs to be added
           */
          if ( ( args[0] & 1024 ) != 0 && exc->subpixel_positioned )
            K |= 1 << 17;

          /*********************************
           * SYMMETRICAL SMOOTHING
           * Selector Bit:  11
           * Return Bit(s): 18
           *
           * Functionality still needs to be added
           */
          if ( ( args[0] & 2048 ) != 0 && exc->symmetrical_smoothing )
            K |= 1 << 18;

          /*********************************
           * GRAY CLEARTYPE
           * Selector Bit:  12
           * Return Bit(s): 19
           *
           * Functionality still needs to be added
           */
          if ( ( args[0] & 4096 ) != 0 && exc->gray_cleartype )
            K |= 1 << 19;
        }
      }
    }

#endif /* TT_SUPPORT_SUBPIXEL_HINTING_INFINALITY */

    args[0] = K;
  }


#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT

  /**************************************************************************
   *
   * GETVARIATION[]: get normalized variation (blend) coordinates
   * Opcode range: 0x91
   * Stack:        --> f2.14...
   *
   * XXX: UNDOCUMENTED!  There is no official documentation from Apple for
   *      this bytecode instruction.  Active only if a font has GX
   *      variation axes.
   */
  static void
  Ins_GETVARIATION( TT_ExecContext  exc,
                    FT_Long*        args )
  {
    FT_UInt    num_axes = exc->face->blend->num_axis;
    FT_Fixed*  coords   = exc->face->blend->normalizedcoords;

    FT_UInt  i;


    if ( BOUNDS( num_axes, exc->stackSize + 1 - exc->top ) )
    {
      exc->error = FT_THROW( Stack_Overflow );
      return;
    }

    if ( coords )
    {
      for ( i = 0; i < num_axes; i++ )
        args[i] = coords[i] >> 2; /* convert 16.16 to 2.14 format */
    }
    else
    {
      for ( i = 0; i < num_axes; i++ )
        args[i] = 0;
    }
  }


  /**************************************************************************
   *
   * GETDATA[]:    no idea what this is good for
   * Opcode range: 0x92
   * Stack:        --> 17
   *
   * XXX: UNDOCUMENTED!  There is no documentation from Apple for this
   *      very weird bytecode instruction.
   */
  static void
  Ins_GETDATA( FT_Long*  args )
  {
    args[0] = 17;
  }

#endif /* TT_CONFIG_OPTION_GX_VAR_SUPPORT */


  static void
  Ins_UNKNOWN( TT_ExecContext  exc )
  {
    TT_DefRecord*  def   = exc->IDefs;
    TT_DefRecord*  limit = FT_OFFSET( def, exc->numIDefs );


    for ( ; def < limit; def++ )
    {
      if ( (FT_Byte)def->opc == exc->opcode && def->active )
      {
        TT_CallRec*  call;


        if ( exc->callTop >= exc->callSize )
        {
          exc->error = FT_THROW( Stack_Overflow );
          return;
        }

        call = exc->callStack + exc->callTop++;

        call->Caller_Range = exc->curRange;
        call->Caller_IP    = exc->IP + 1;
        call->Cur_Count    = 1;
        call->Def          = def;

        Ins_Goto_CodeRange( exc, def->range, def->start );

        exc->step_ins = FALSE;
        return;
      }
    }

    exc->error = FT_THROW( Invalid_Opcode );
  }


  /**************************************************************************
   *
   * RUN
   *
   * This function executes a run of opcodes.  It will exit in the
   * following cases:
   *
   * - Errors (in which case it returns FALSE).
   *
   * - Reaching the end of the main code range (returns TRUE).
   *   Reaching the end of a code range within a function call is an
   *   error.
   *
   * - After executing one single opcode, if the flag `Instruction_Trap'
   *   is set to TRUE (returns TRUE).
   *
   * On exit with TRUE, test IP < CodeSize to know whether it comes from
   * an instruction trap or a normal termination.
   *
   *
   * Note: The documented DEBUG opcode pops a value from the stack.  This
   *       behaviour is unsupported; here a DEBUG opcode is always an
   *       error.
   *
   *
   * THIS IS THE INTERPRETER'S MAIN LOOP.
   *
   */


  /* documentation is in ttinterp.h */

  FT_EXPORT_DEF( FT_Error )
  TT_RunIns( TT_ExecContext  exc )
  {
    FT_ULong   ins_counter = 0;  /* executed instructions counter */
    FT_ULong   num_twilight_points;
    FT_UShort  i;

#ifdef TT_SUPPORT_SUBPIXEL_HINTING_INFINALITY
    FT_Byte    opcode_pattern[1][2] = {
                  /* #8 TypeMan Talk Align */
                  {
                    0x06, /* SPVTL   */
                    0x7D, /* RDTG    */
                  },
                };
    FT_UShort  opcode_patterns   = 1;
    FT_UShort  opcode_pointer[1] = { 0 };
    FT_UShort  opcode_size[1]    = { 1 };
#endif /* TT_SUPPORT_SUBPIXEL_HINTING_INFINALITY */


    /* We restrict the number of twilight points to a reasonable,     */
    /* heuristic value to avoid slow execution of malformed bytecode. */
    num_twilight_points = FT_MAX( 30,
                                  2 * ( exc->pts.n_points + exc->cvtSize ) );
    if ( exc->twilight.n_points > num_twilight_points )
    {
      if ( num_twilight_points > 0xFFFFU )
        num_twilight_points = 0xFFFFU;

      FT_TRACE5(( "TT_RunIns: Resetting number of twilight points\n" ));
      FT_TRACE5(( "           from %d to the more reasonable value %ld\n",
                  exc->twilight.n_points,
                  num_twilight_points ));
      exc->twilight.n_points = (FT_UShort)num_twilight_points;
    }

    /* Set up loop detectors.  We restrict the number of LOOPCALL loops */
    /* and the number of JMPR, JROT, and JROF calls with a negative     */
    /* argument to values that depend on various parameters like the    */
    /* size of the CVT table or the number of points in the current     */
    /* glyph (if applicable).                                           */
    /*                                                                  */
    /* The idea is that in real-world bytecode you either iterate over  */
    /* all CVT entries (in the `prep' table), or over all points (or    */
    /* contours, in the `glyf' table) of a glyph, and such iterations   */
    /* don't happen very often.                                         */
    exc->loopcall_counter = 0;
    exc->neg_jump_counter = 0;

    /* The maximum values are heuristic. */
    if ( exc->pts.n_points )
      exc->loopcall_counter_max = FT_MAX( 50,
                                          10 * exc->pts.n_points ) +
                                  FT_MAX( 50,
                                          exc->cvtSize / 10 );
    else
      exc->loopcall_counter_max = 300 + 22 * exc->cvtSize;

    /* as a protection against an unreasonable number of CVT entries  */
    /* we assume at most 100 control values per glyph for the counter */
    if ( exc->loopcall_counter_max >
         100 * (FT_ULong)exc->face->root.num_glyphs )
      exc->loopcall_counter_max = 100 * (FT_ULong)exc->face->root.num_glyphs;

    FT_TRACE5(( "TT_RunIns: Limiting total number of loops in LOOPCALL"
                " to %ld\n", exc->loopcall_counter_max ));

    exc->neg_jump_counter_max = exc->loopcall_counter_max;
    FT_TRACE5(( "TT_RunIns: Limiting total number of backward jumps"
                " to %ld\n", exc->neg_jump_counter_max ));

    /* set PPEM and CVT functions */
    exc->tt_metrics.ratio = 0;
    if ( exc->metrics.x_ppem != exc->metrics.y_ppem )
    {
      /* non-square pixels, use the stretched routines */
      exc->func_cur_ppem  = Current_Ppem_Stretched;
      exc->func_read_cvt  = Read_CVT_Stretched;
      exc->func_write_cvt = Write_CVT_Stretched;
      exc->func_move_cvt  = Move_CVT_Stretched;
    }
    else
    {
      /* square pixels, use normal routines */
      exc->func_cur_ppem  = Current_Ppem;
      exc->func_read_cvt  = Read_CVT;
      exc->func_write_cvt = Write_CVT;
      exc->func_move_cvt  = Move_CVT;
    }

    exc->origCvt     = exc->cvt;
    exc->origStorage = exc->storage;
    exc->iniRange    = exc->curRange;

    Compute_Funcs( exc );
    Compute_Round( exc, (FT_Byte)exc->GS.round_state );

    /* These flags cancel execution of some opcodes after IUP is called */
#ifdef TT_SUPPORT_SUBPIXEL_HINTING_INFINALITY
    exc->iup_called  = FALSE;
#endif
#ifdef TT_SUPPORT_SUBPIXEL_HINTING_MINIMAL
    exc->iupx_called = FALSE;
    exc->iupy_called = FALSE;
#endif

    do
    {
      exc->opcode = exc->code[exc->IP];

#ifdef FT_DEBUG_LEVEL_TRACE
      if ( ft_trace_levels[trace_ttinterp] >= 6 )
      {
        FT_Long  cnt = FT_MIN( 8, exc->top );
        FT_Long  n;


        /* if tracing level is 7, show current code position */
        /* and the first few stack elements also             */
        FT_TRACE6(( "  " ));
        FT_TRACE7(( "%06ld ", exc->IP ));
        FT_TRACE6(( "%s", opcode_name[exc->opcode] + 2 ));
        FT_TRACE7(( "%*s", *opcode_name[exc->opcode] == 'A'
                              ? 2
                              : 12 - ( *opcode_name[exc->opcode] - '0' ),
                              "#" ));
        for ( n = 1; n <= cnt; n++ )
          FT_TRACE7(( " %ld", exc->stack[exc->top - n] ));
        FT_TRACE6(( "\n" ));
      }
#endif /* FT_DEBUG_LEVEL_TRACE */

      if ( ( exc->length = opcode_length[exc->opcode] ) < 0 )
      {
        if ( exc->IP + 1 >= exc->codeSize )
          goto LErrorCodeOverflow_;

        exc->length = 2 - exc->length * exc->code[exc->IP + 1];
      }

      if ( exc->IP + exc->length > exc->codeSize )
        goto LErrorCodeOverflow_;

      /* First, let's check for empty stack and overflow */
      exc->args = exc->top - ( Pop_Push_Count[exc->opcode] >> 4 );

      /* `args' is the top of the stack once arguments have been popped. */
      /* One can also interpret it as the index of the last argument.    */
      if ( exc->args < 0 )
      {
        if ( exc->pedantic_hinting )
        {
          exc->error = FT_THROW( Too_Few_Arguments );
          goto LErrorLabel_;
        }

        /* push zeroes onto the stack */
        for ( i = 0; i < Pop_Push_Count[exc->opcode] >> 4; i++ )
          exc->stack[i] = 0;
        exc->args = 0;
      }

#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT
      if ( exc->opcode == 0x91 )
      {
        /* this is very special: GETVARIATION returns */
        /* a variable number of arguments             */

        /* it is the job of the application to `activate' GX handling, */
        /* this is, calling any of the GX API functions on the current */
        /* font to select a variation instance                         */
        if ( exc->face->blend )
          exc->new_top = exc->args + exc->face->blend->num_axis;
      }
      else
#endif
        exc->new_top = exc->args + ( Pop_Push_Count[exc->opcode] & 15 );

      /* `new_top' is the new top of the stack, after the instruction's */
      /* execution.  `top' will be set to `new_top' after the `switch'  */
      /* statement.                                                     */
      if ( exc->new_top > exc->stackSize )
      {
        exc->error = FT_THROW( Stack_Overflow );
        goto LErrorLabel_;
      }

      exc->step_ins = TRUE;
      exc->error    = FT_Err_Ok;

#ifdef TT_SUPPORT_SUBPIXEL_HINTING_INFINALITY

      if ( SUBPIXEL_HINTING_INFINALITY )
      {
        for ( i = 0; i < opcode_patterns; i++ )
        {
          if ( opcode_pointer[i] < opcode_size[i]                  &&
               exc->opcode == opcode_pattern[i][opcode_pointer[i]] )
          {
            opcode_pointer[i] += 1;

            if ( opcode_pointer[i] == opcode_size[i] )
            {
              FT_TRACE6(( "sph: opcode ptrn: %d, %s %s\n",
                          i,
                          exc->face->root.family_name,
                          exc->face->root.style_name ));

              switch ( i )
              {
              case 0:
                break;
              }
              opcode_pointer[i] = 0;
            }
          }
          else
            opcode_pointer[i] = 0;
        }
      }

#endif /* TT_SUPPORT_SUBPIXEL_HINTING_INFINALITY */

      {
        FT_Long*  args   = exc->stack + exc->args;
        FT_Byte   opcode = exc->opcode;


        switch ( opcode )
        {
        case 0x00:  /* SVTCA y  */
        case 0x01:  /* SVTCA x  */
        case 0x02:  /* SPvTCA y */
        case 0x03:  /* SPvTCA x */
        case 0x04:  /* SFvTCA y */
        case 0x05:  /* SFvTCA x */
          Ins_SxyTCA( exc );
          break;

        case 0x06:  /* SPvTL // */
        case 0x07:  /* SPvTL +  */
          Ins_SPVTL( exc, args );
          break;

        case 0x08:  /* SFvTL // */
        case 0x09:  /* SFvTL +  */
          Ins_SFVTL( exc, args );
          break;

        case 0x0A:  /* SPvFS */
          Ins_SPVFS( exc, args );
          break;

        case 0x0B:  /* SFvFS */
          Ins_SFVFS( exc, args );
          break;

        case 0x0C:  /* GPv */
          Ins_GPV( exc, args );
          break;

        case 0x0D:  /* GFv */
          Ins_GFV( exc, args );
          break;

        case 0x0E:  /* SFvTPv */
          Ins_SFVTPV( exc );
          break;

        case 0x0F:  /* ISECT  */
          Ins_ISECT( exc, args );
          break;

        case 0x10:  /* SRP0 */
          Ins_SRP0( exc, args );
          break;

        case 0x11:  /* SRP1 */
          Ins_SRP1( exc, args );
          break;

        case 0x12:  /* SRP2 */
          Ins_SRP2( exc, args );
          break;

        case 0x13:  /* SZP0 */
          Ins_SZP0( exc, args );
          break;

        case 0x14:  /* SZP1 */
          Ins_SZP1( exc, args );
          break;

        case 0x15:  /* SZP2 */
          Ins_SZP2( exc, args );
          break;

        case 0x16:  /* SZPS */
          Ins_SZPS( exc, args );
          break;

        case 0x17:  /* SLOOP */
          Ins_SLOOP( exc, args );
          break;

        case 0x18:  /* RTG */
          Ins_RTG( exc );
          break;

        case 0x19:  /* RTHG */
          Ins_RTHG( exc );
          break;

        case 0x1A:  /* SMD */
          Ins_SMD( exc, args );
          break;

        case 0x1B:  /* ELSE */
          Ins_ELSE( exc );
          break;

        case 0x1C:  /* JMPR */
          Ins_JMPR( exc, args );
          break;

        case 0x1D:  /* SCVTCI */
          Ins_SCVTCI( exc, args );
          break;

        case 0x1E:  /* SSWCI */
          Ins_SSWCI( exc, args );
          break;

        case 0x1F:  /* SSW */
          Ins_SSW( exc, args );
          break;

        case 0x20:  /* DUP */
          Ins_DUP( args );
          break;

        case 0x21:  /* POP */
          Ins_POP();
          break;

        case 0x22:  /* CLEAR */
          Ins_CLEAR( exc );
          break;

        case 0x23:  /* SWAP */
          Ins_SWAP( args );
          break;

        case 0x24:  /* DEPTH */
          Ins_DEPTH( exc, args );
          break;

        case 0x25:  /* CINDEX */
          Ins_CINDEX( exc, args );
          break;

        case 0x26:  /* MINDEX */
          Ins_MINDEX( exc, args );
          break;

        case 0x27:  /* ALIGNPTS */
          Ins_ALIGNPTS( exc, args );
          break;

        case 0x28:  /* RAW */
          Ins_UNKNOWN( exc );
          break;

        case 0x29:  /* UTP */
          Ins_UTP( exc, args );
          break;

        case 0x2A:  /* LOOPCALL */
          Ins_LOOPCALL( exc, args );
          break;

        case 0x2B:  /* CALL */
          Ins_CALL( exc, args );
          break;

        case 0x2C:  /* FDEF */
          Ins_FDEF( exc, args );
          break;

        case 0x2D:  /* ENDF */
          Ins_ENDF( exc );
          break;

        case 0x2E:  /* MDAP */
        case 0x2F:  /* MDAP */
          Ins_MDAP( exc, args );
          break;

        case 0x30:  /* IUP */
        case 0x31:  /* IUP */
          Ins_IUP( exc );
          break;

        case 0x32:  /* SHP */
        case 0x33:  /* SHP */
          Ins_SHP( exc );
          break;

        case 0x34:  /* SHC */
        case 0x35:  /* SHC */
          Ins_SHC( exc, args );
          break;

        case 0x36:  /* SHZ */
        case 0x37:  /* SHZ */
          Ins_SHZ( exc, args );
          break;

        case 0x38:  /* SHPIX */
          Ins_SHPIX( exc, args );
          break;

        case 0x39:  /* IP    */
          Ins_IP( exc );
          break;

        case 0x3A:  /* MSIRP */
        case 0x3B:  /* MSIRP */
          Ins_MSIRP( exc, args );
          break;

        case 0x3C:  /* AlignRP */
          Ins_ALIGNRP( exc );
          break;

        case 0x3D:  /* RTDG */
          Ins_RTDG( exc );
          break;

        case 0x3E:  /* MIAP */
        case 0x3F:  /* MIAP */
          Ins_MIAP( exc, args );
          break;

        case 0x40:  /* NPUSHB */
          Ins_NPUSHB( exc, args );
          break;

        case 0x41:  /* NPUSHW */
          Ins_NPUSHW( exc, args );
          break;

        case 0x42:  /* WS */
          Ins_WS( exc, args );
          break;

        case 0x43:  /* RS */
          Ins_RS( exc, args );
          break;

        case 0x44:  /* WCVTP */
          Ins_WCVTP( exc, args );
          break;

        case 0x45:  /* RCVT */
          Ins_RCVT( exc, args );
          break;

        case 0x46:  /* GC */
        case 0x47:  /* GC */
          Ins_GC( exc, args );
          break;

        case 0x48:  /* SCFS */
          Ins_SCFS( exc, args );
          break;

        case 0x49:  /* MD */
        case 0x4A:  /* MD */
          Ins_MD( exc, args );
          break;

        case 0x4B:  /* MPPEM */
          Ins_MPPEM( exc, args );
          break;

        case 0x4C:  /* MPS */
          Ins_MPS( exc, args );
          break;

        case 0x4D:  /* FLIPON */
          Ins_FLIPON( exc );
          break;

        case 0x4E:  /* FLIPOFF */
          Ins_FLIPOFF( exc );
          break;

        case 0x4F:  /* DEBUG */
          Ins_DEBUG( exc );
          break;

        case 0x50:  /* LT */
          Ins_LT( args );
          break;

        case 0x51:  /* LTEQ */
          Ins_LTEQ( args );
          break;

        case 0x52:  /* GT */
          Ins_GT( args );
          break;

        case 0x53:  /* GTEQ */
          Ins_GTEQ( args );
          break;

        case 0x54:  /* EQ */
          Ins_EQ( args );
          break;

        case 0x55:  /* NEQ */
          Ins_NEQ( args );
          break;

        case 0x56:  /* ODD */
          Ins_ODD( exc, args );
          break;

        case 0x57:  /* EVEN */
          Ins_EVEN( exc, args );
          break;

        case 0x58:  /* IF */
          Ins_IF( exc, args );
          break;

        case 0x59:  /* EIF */
          Ins_EIF();
          break;

        case 0x5A:  /* AND */
          Ins_AND( args );
          break;

        case 0x5B:  /* OR */
          Ins_OR( args );
          break;

        case 0x5C:  /* NOT */
          Ins_NOT( args );
          break;

        case 0x5D:  /* DELTAP1 */
          Ins_DELTAP( exc, args );
          break;

        case 0x5E:  /* SDB */
          Ins_SDB( exc, args );
          break;

        case 0x5F:  /* SDS */
          Ins_SDS( exc, args );
          break;

        case 0x60:  /* ADD */
          Ins_ADD( args );
          break;

        case 0x61:  /* SUB */
          Ins_SUB( args );
          break;

        case 0x62:  /* DIV */
          Ins_DIV( exc, args );
          break;

        case 0x63:  /* MUL */
          Ins_MUL( args );
          break;

        case 0x64:  /* ABS */
          Ins_ABS( args );
          break;

        case 0x65:  /* NEG */
          Ins_NEG( args );
          break;

        case 0x66:  /* FLOOR */
          Ins_FLOOR( args );
          break;

        case 0x67:  /* CEILING */
          Ins_CEILING( args );
          break;

        case 0x68:  /* ROUND */
        case 0x69:  /* ROUND */
        case 0x6A:  /* ROUND */
        case 0x6B:  /* ROUND */
          Ins_ROUND( exc, args );
          break;

        case 0x6C:  /* NROUND */
        case 0x6D:  /* NROUND */
        case 0x6E:  /* NRRUND */
        case 0x6F:  /* NROUND */
          Ins_NROUND( exc, args );
          break;

        case 0x70:  /* WCVTF */
          Ins_WCVTF( exc, args );
          break;

        case 0x71:  /* DELTAP2 */
        case 0x72:  /* DELTAP3 */
          Ins_DELTAP( exc, args );
          break;

        case 0x73:  /* DELTAC0 */
        case 0x74:  /* DELTAC1 */
        case 0x75:  /* DELTAC2 */
          Ins_DELTAC( exc, args );
          break;

        case 0x76:  /* SROUND */
          Ins_SROUND( exc, args );
          break;

        case 0x77:  /* S45Round */
          Ins_S45ROUND( exc, args );
          break;

        case 0x78:  /* JROT */
          Ins_JROT( exc, args );
          break;

        case 0x79:  /* JROF */
          Ins_JROF( exc, args );
          break;

        case 0x7A:  /* ROFF */
          Ins_ROFF( exc );
          break;

        case 0x7B:  /* ???? */
          Ins_UNKNOWN( exc );
          break;

        case 0x7C:  /* RUTG */
          Ins_RUTG( exc );
          break;

        case 0x7D:  /* RDTG */
          Ins_RDTG( exc );
          break;

        case 0x7E:  /* SANGW */
          Ins_SANGW();
          break;

        case 0x7F:  /* AA */
          Ins_AA();
          break;

        case 0x80:  /* FLIPPT */
          Ins_FLIPPT( exc );
          break;

        case 0x81:  /* FLIPRGON */
          Ins_FLIPRGON( exc, args );
          break;

        case 0x82:  /* FLIPRGOFF */
          Ins_FLIPRGOFF( exc, args );
          break;

        case 0x83:  /* UNKNOWN */
        case 0x84:  /* UNKNOWN */
          Ins_UNKNOWN( exc );
          break;

        case 0x85:  /* SCANCTRL */
          Ins_SCANCTRL( exc, args );
          break;

        case 0x86:  /* SDPvTL */
        case 0x87:  /* SDPvTL */
          Ins_SDPVTL( exc, args );
          break;

        case 0x88:  /* GETINFO */
          Ins_GETINFO( exc, args );
          break;

        case 0x89:  /* IDEF */
          Ins_IDEF( exc, args );
          break;

        case 0x8A:  /* ROLL */
          Ins_ROLL( args );
          break;

        case 0x8B:  /* MAX */
          Ins_MAX( args );
          break;

        case 0x8C:  /* MIN */
          Ins_MIN( args );
          break;

        case 0x8D:  /* SCANTYPE */
          Ins_SCANTYPE( exc, args );
          break;

        case 0x8E:  /* INSTCTRL */
          Ins_INSTCTRL( exc, args );
          break;

        case 0x8F:  /* ADJUST */
        case 0x90:  /* ADJUST */
          Ins_UNKNOWN( exc );
          break;

#ifdef TT_CONFIG_OPTION_GX_VAR_SUPPORT
        case 0x91:
          /* it is the job of the application to `activate' GX handling, */
          /* this is, calling any of the GX API functions on the current */
          /* font to select a variation instance                         */
          if ( exc->face->blend )
            Ins_GETVARIATION( exc, args );
          else
            Ins_UNKNOWN( exc );
          break;

        case 0x92:
          /* there is at least one MS font (LaoUI.ttf version 5.01) that */
          /* uses IDEFs for 0x91 and 0x92; for this reason we activate   */
          /* GETDATA for GX fonts only, similar to GETVARIATION          */
          if ( exc->face->blend )
            Ins_GETDATA( args );
          else
            Ins_UNKNOWN( exc );
          break;
#endif

        default:
          if ( opcode >= 0xE0 )
            Ins_MIRP( exc, args );
          else if ( opcode >= 0xC0 )
            Ins_MDRP( exc, args );
          else if ( opcode >= 0xB8 )
            Ins_PUSHW( exc, args );
          else if ( opcode >= 0xB0 )
            Ins_PUSHB( exc, args );
          else
            Ins_UNKNOWN( exc );
        }
      }

      if ( exc->error )
      {
        switch ( exc->error )
        {
          /* looking for redefined instructions */
        case FT_ERR( Invalid_Opcode ):
          {
            TT_DefRecord*  def   = exc->IDefs;
            TT_DefRecord*  limit = FT_OFFSET( def, exc->numIDefs );


            for ( ; def < limit; def++ )
            {
              if ( def->active && exc->opcode == (FT_Byte)def->opc )
              {
                TT_CallRec*  callrec;


                if ( exc->callTop >= exc->callSize )
                {
                  exc->error = FT_THROW( Invalid_Reference );
                  goto LErrorLabel_;
                }

                callrec = &exc->callStack[exc->callTop];

                callrec->Caller_Range = exc->curRange;
                callrec->Caller_IP    = exc->IP + 1;
                callrec->Cur_Count    = 1;
                callrec->Def          = def;

                if ( Ins_Goto_CodeRange( exc,
                                         def->range,
                                         def->start ) == FAILURE )
                  goto LErrorLabel_;

                goto LSuiteLabel_;
              }
            }
          }

          exc->error = FT_THROW( Invalid_Opcode );
          goto LErrorLabel_;

#if 0
          break;   /* Unreachable code warning suppression.             */
                   /* Leave to remind in case a later change the editor */
                   /* to consider break;                                */
#endif

        default:
          goto LErrorLabel_;

#if 0
        break;
#endif
        }
      }

      exc->top = exc->new_top;

      if ( exc->step_ins )
        exc->IP += exc->length;

      /* increment instruction counter and check if we didn't */
      /* run this program for too long (e.g. infinite loops). */
      if ( ++ins_counter > TT_CONFIG_OPTION_MAX_RUNNABLE_OPCODES ) {
        exc->error = FT_THROW( Execution_Too_Long );
        goto LErrorLabel_;
      }

    LSuiteLabel_:
      if ( exc->IP >= exc->codeSize )
      {
        if ( exc->callTop > 0 )
        {
          exc->error = FT_THROW( Code_Overflow );
          goto LErrorLabel_;
        }
        else
          goto LNo_Error_;
      }
    } while ( !exc->instruction_trap );

  LNo_Error_:
    FT_TRACE4(( "  %ld instruction%s executed\n",
                ins_counter,
                ins_counter == 1 ? "" : "s" ));

    exc->cvt     = exc->origCvt;
    exc->storage = exc->origStorage;

    return FT_Err_Ok;

  LErrorCodeOverflow_:
    exc->error = FT_THROW( Code_Overflow );

  LErrorLabel_:
    if ( exc->error && !exc->instruction_trap )
      FT_TRACE1(( "  The interpreter returned error 0x%x\n", exc->error ));

    exc->cvt     = exc->origCvt;
    exc->storage = exc->origStorage;

    return exc->error;
  }

#else /* !TT_USE_BYTECODE_INTERPRETER */

  /* ANSI C doesn't like empty source files */
  typedef int  _tt_interp_dummy;

#endif /* !TT_USE_BYTECODE_INTERPRETER */


/* END */
