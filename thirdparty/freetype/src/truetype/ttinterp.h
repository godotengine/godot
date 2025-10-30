/****************************************************************************
 *
 * ttinterp.h
 *
 *   TrueType bytecode interpreter (specification).
 *
 * Copyright (C) 1996-2024 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#ifndef TTINTERP_H_
#define TTINTERP_H_

#include "ttobjs.h"


FT_BEGIN_HEADER


  /**************************************************************************
   *
   * Rounding mode constants.
   */
#define TT_Round_Off             5
#define TT_Round_To_Half_Grid    0
#define TT_Round_To_Grid         1
#define TT_Round_To_Double_Grid  2
#define TT_Round_Up_To_Grid      4
#define TT_Round_Down_To_Grid    3
#define TT_Round_Super           6
#define TT_Round_Super_45        7


  /**************************************************************************
   *
   * Function types used by the interpreter, depending on various modes
   * (e.g. the rounding mode, whether to render a vertical or horizontal
   * line etc).
   *
   */

  /* Rounding function */
  typedef FT_F26Dot6
  (*TT_Round_Func)( TT_ExecContext  exc,
                    FT_F26Dot6      distance,
                    FT_Int          color );

  /* Point displacement along the freedom vector routine */
  typedef void
  (*TT_Move_Func)( TT_ExecContext  exc,
                   TT_GlyphZone    zone,
                   FT_UShort       point,
                   FT_F26Dot6      distance );

  /* Distance projection along one of the projection vectors */
  typedef FT_F26Dot6
  (*TT_Project_Func)( TT_ExecContext  exc,
                      FT_Pos          dx,
                      FT_Pos          dy );

  /* getting current ppem.  Take care of non-square pixels if necessary */
  typedef FT_Long
  (*TT_Cur_Ppem_Func)( TT_ExecContext  exc );

  /* reading a cvt value.  Take care of non-square pixels if necessary */
  typedef FT_F26Dot6
  (*TT_Get_CVT_Func)( TT_ExecContext  exc,
                      FT_ULong        idx );

  /* setting or moving a cvt value.  Take care of non-square pixels  */
  /* if necessary                                                    */
  typedef void
  (*TT_Set_CVT_Func)( TT_ExecContext  exc,
                      FT_ULong        idx,
                      FT_F26Dot6      value );


  /**************************************************************************
   *
   * This structure defines a call record, used to manage function calls.
   */
  typedef struct  TT_CallRec_
  {
    FT_Int   Caller_Range;
    FT_Long  Caller_IP;
    FT_Long  Cur_Count;

    TT_DefRecord  *Def; /* either FDEF or IDEF */

  } TT_CallRec, *TT_CallStack;


  /**************************************************************************
   *
   * The main structure for the interpreter which collects all necessary
   * variables and states.
   *
   * Members that are initialized by `TT_Load_Context` are marked with '!'.
   * Members that are initialized by `TT_Run_Context` are marked with '@'.
   */
  typedef struct  TT_ExecContextRec_
  {
    TT_Face            face;       /* ! */
    TT_Size            size;       /* ! */
    FT_Memory          memory;

    /* instructions state */

    FT_Error           error;      /* last execution error */

    FT_Long            top;        /* @ top of exec. stack */

    FT_Long            stackSize;  /* ! size of exec. stack */
    FT_Long*           stack;      /* ! current exec. stack */

    FT_Long            args;
    FT_Long            new_top;    /* new top after exec. */

    TT_GlyphZoneRec    zp0,        /* @! zone records */
                       zp1,        /* @!              */
                       zp2,        /* @!              */
                       pts,        /*  !              */
                       twilight;   /*  !              */

    FT_Long            pointSize;  /* ! in 26.6 format */
    FT_Size_Metrics    metrics;    /* !                */
    TT_Size_Metrics    tt_metrics; /* ! size metrics   */

    TT_GraphicsState   GS;         /* !@ current graphics state */

    FT_Int             iniRange;  /* initial code range number   */
    FT_Int             curRange;  /* current code range number   */
    FT_Byte*           code;      /* current code range          */
    FT_Long            IP;        /* current instruction pointer */
    FT_Long            codeSize;  /* size of current range       */

    FT_Byte            opcode;    /* current opcode              */
    FT_Int             length;    /* length of current opcode    */

    FT_Bool            step_ins;  /* true if the interpreter must */
                                  /* increment IP after ins. exec */
    FT_ULong           cvtSize;   /* ! */
    FT_Long*           cvt;       /* ! */
    FT_ULong           glyfCvtSize;
    FT_Long*           glyfCvt;   /* cvt working copy for glyph */

    FT_UInt            glyphSize; /* ! glyph instructions buffer size */
    FT_Byte*           glyphIns;  /* ! glyph instructions buffer      */

    FT_UInt            numFDefs;  /* ! number of function defs         */
    FT_UInt            maxFDefs;  /* ! maximum number of function defs */
    TT_DefArray        FDefs;     /*   table of FDefs entries          */

    FT_UInt            numIDefs;  /* ! number of instruction defs */
    FT_UInt            maxIDefs;  /* ! maximum number of ins defs */
    TT_DefArray        IDefs;     /*   table of IDefs entries     */

    FT_UInt            maxFunc;   /* ! maximum function index    */
    FT_UInt            maxIns;    /* ! maximum instruction index */

    FT_Int             callTop,    /* @ top of call stack during execution */
                       callSize;   /*   size of call stack                 */
    TT_CallStack       callStack;  /*   call stack                         */

    FT_UShort          maxPoints;    /* capacity of this context's `pts' */
    FT_Short           maxContours;  /* record, expressed in points and  */
                                     /* contours.                        */

    TT_CodeRangeTable  codeRangeTable;  /* ! table of valid code ranges */
                                        /*   useful for the debugger    */

    FT_UShort          storeSize;    /* ! size of current storage */
    FT_Long*           storage;      /* ! storage area            */
    FT_UShort          glyfStoreSize;
    FT_Long*           glyfStorage;  /* storage working copy for glyph */

    FT_F26Dot6         period;     /* values used for the */
    FT_F26Dot6         phase;      /* `SuperRounding'     */
    FT_F26Dot6         threshold;

    FT_Bool            instruction_trap; /* ! If `True', the interpreter   */
                                         /*   exits after each instruction */

    TT_GraphicsState   default_GS;       /* graphics state resulting from   */
                                         /* the prep program                */
    FT_Bool            is_composite;     /* true if the glyph is composite  */
    FT_Bool            pedantic_hinting; /* true if pedantic interpretation */

    /* latest interpreter additions */

    FT_Long            F_dot_P;    /* dot product of freedom and projection */
                                   /* vectors                               */
    TT_Round_Func      func_round; /* current rounding function             */

    TT_Project_Func    func_project,   /* current projection function */
                       func_dualproj,  /* current dual proj. function */
                       func_freeProj;  /* current freedom proj. func  */

    TT_Move_Func       func_move;      /* current point move function     */
    TT_Move_Func       func_move_orig; /* move original position function */

    TT_Cur_Ppem_Func   func_cur_ppem;  /* get current proj. ppem value  */

    TT_Get_CVT_Func    func_read_cvt;  /* read a cvt entry              */
    TT_Set_CVT_Func    func_write_cvt; /* write a cvt entry (in pixels) */
    TT_Set_CVT_Func    func_move_cvt;  /* incr a cvt entry (in pixels)  */

    FT_Bool            grayscale;      /* bi-level hinting and */
                                       /* grayscale rendering  */

#ifdef TT_SUPPORT_SUBPIXEL_HINTING_MINIMAL
    /*
     * FreeType supports ClearType-like hinting of TrueType fonts through
     * the version 40 interpreter.  This is achieved through several hacks
     * in the base (v35) interpreter, as detailed below.
     *
     * ClearType is an umbrella term for several rendering techniques
     * employed by Microsoft's various GUI and rendering toolkit
     * implementations, most importantly: subpixel rendering for using the
     * RGB subpixels of LCDs to approximately triple the perceived
     * resolution on the x-axis and subpixel hinting for positioning stems
     * on subpixel borders.  TrueType programming is explicit, i.e., fonts
     * must be programmed to take advantage of ClearType's possibilities.
     *
     * When ClearType was introduced, it seemed unlikely that all fonts
     * would be reprogrammed, so Microsoft decided to implement a backward
     * compatibility mode.  It employs several simple to complicated
     * assumptions and tricks, many of them font-dependent, that modify the
     * interpretation of the bytecode contained in these fonts to retrofit
     * them into a ClearType-y look.  The quality of the results varies.
     * Most (web)fonts that were released since then have come to rely on
     * these hacks to render correctly, even some of Microsoft's flagship
     * fonts (e.g., Calibri, Cambria, Segoe UI).
     *
     * FreeType's minimal subpixel hinting code (interpreter version 40)
     * employs a small list of font-agnostic hacks loosely based on the
     * public information available on Microsoft's compatibility mode[2].
     * The focus is on modern (web)fonts rather than legacy fonts that were
     * made for monochrome rendering.  It will not match ClearType rendering
     * exactly.  Unlike the `Infinality' code (interpreter version 38) that
     * came before, it will not try to toggle hacks for specific fonts for
     * performance and complexity reasons.  It will fall back to version 35
     * behavior for tricky fonts[1] or when monochrome rendering is
     * requested.
     *
     * Major hacks
     *
     * - Any point movement on the x axis is ignored (cf. `Direct_Move' and
     *   `Direct_Move_X').  This has the smallest code footprint and single
     *   biggest effect.  The ClearType way to increase resolution is
     *   supersampling the x axis, the FreeType way is ignoring instructions
     *   on the x axis, which gives the same result in the majority of
     *   cases.
     *
     * - Points are not moved post-IUP (neither on the x nor on the y axis),
     *   except the x component of diagonal moves post-IUP (cf.
     *   `Direct_Move', `Direct_Move_Y', `Move_Zp2_Point').  Post-IUP
     *   changes are commonly used to `fix' pixel patterns which has little
     *   use outside monochrome rendering.
     *
     * - SHPIX and DELTAP don't execute unless moving a composite on the
     *   y axis or moving a previously y touched point.  SHPIX additionally
     *   denies movement on the x axis (cf. `Ins_SHPIX' and `Ins_DELTAP').
     *   Both instructions are commonly used to `fix' pixel patterns for
     *   monochrome or Windows's GDI rendering but make little sense for
     *   FreeType rendering.  Both can distort the outline.  See [2] for
     *   details.
     *
     * - The hdmx table and modifications to phantom points are ignored.
     *   Bearings and advance widths remain unchanged (except rounding them
     *   outside the interpreter!), cf. `compute_glyph_metrics' and
     *   `TT_Hint_Glyph'.  Letting non-native-ClearType fonts modify spacing
     *   might mess up spacing.
     *
     * Minor hacks
     *
     * - FLIPRGON, FLIPRGOFF, and FLIPPT don't execute post-IUP.  This
     *   prevents dents in e.g. Arial-Regular's `D' and `G' glyphs at
     *   various sizes.
     *
     * (Post-IUP is the state after both IUP[x] and IUP[y] have been
     * executed.)
     *
     * The best results are achieved for fonts that were from the outset
     * designed with ClearType in mind, meaning they leave the x axis mostly
     * alone and don't mess with the `final' outline to produce more
     * pleasing pixel patterns.  The harder the designer tried to produce
     * very specific patterns (`superhinting') for pre-ClearType-displays,
     * the worse the results.
     *
     * Microsoft defines a way to turn off backward compatibility and
     * interpret instructions as before (called `native ClearType')[2][3].
     * The font designer then regains full control and is responsible for
     * making the font work correctly with ClearType without any
     * hand-holding by the interpreter or rasterizer[4].  The v40
     * interpreter assumes backward compatibility by default, which can be
     * turned off the same way by executing the following in the control
     * program (cf. `Ins_INSTCTRL').
     *
     *   #PUSH 4,3
     *   INSTCTRL[]
     *
     * [1] Tricky fonts as FreeType defines them rely on the bytecode
     *     interpreter to display correctly.  Hacks can interfere with them,
     *     so they get treated like native ClearType fonts (v40 with
     *     backward compatibility turned off).  Cf. `TT_RunIns'.
     *
     * [2] Proposed by Microsoft's Greg Hitchcock in
     *     https://www.microsoft.com/typography/cleartype/truetypecleartype.aspx
     *
     * [3] Beat Stamm describes it in more detail:
     *     http://rastertragedy.com/RTRCh4.htm#Sec12.
     *
     * [4] The list of `native ClearType' fonts is small at the time of this
     *     writing; I found the following on a Windows 10 Update 1511
     *     installation: Constantia, Corbel, Sitka, Malgun Gothic, Microsoft
     *     JhengHei (Bold and UI Bold), Microsoft YaHei (Bold and UI Bold),
     *     SimSun, NSimSun, and Yu Gothic.
     *
     */

    /* Using v40 implies subpixel hinting, unless FT_RENDER_MODE_MONO has been
     * requested.  Used to detect interpreter */
    /* version switches.  `_lean' to differentiate from the Infinality */
    /* `subpixel_hinting', which is managed differently.               */
    FT_Bool            subpixel_hinting_lean;

    /* Long side of a LCD subpixel is vertical (e.g., screen is rotated). */
    /* `_lean' to differentiate from the Infinality `vertical_lcd', which */
    /* is managed differently.                                            */
    FT_Bool            vertical_lcd_lean;

    /* Default to backward compatibility mode in v40 interpreter.  If   */
    /* this is false, it implies the interpreter is in v35 or in native */
    /* ClearType mode.                                                  */
    FT_Bool            backward_compatibility;

    /* Useful for detecting and denying post-IUP trickery that is usually */
    /* used to fix pixel patterns (`superhinting').                       */
    FT_Bool            iupx_called;
    FT_Bool            iupy_called;

    /* ClearType hinting and grayscale rendering, as used by Universal */
    /* Windows Platform apps (Windows 8 and above).  Like the standard */
    /* colorful ClearType mode, it utilizes a vastly increased virtual */
    /* resolution on the x axis.  Different from bi-level hinting and  */
    /* grayscale rendering, the old mode from Win9x days that roughly  */
    /* adheres to the physical pixel grid on both axes.                */
    FT_Bool            grayscale_cleartype;
#endif /* TT_SUPPORT_SUBPIXEL_HINTING_MINIMAL */

    /* We maintain two counters (in addition to the instruction counter) */
    /* that act as loop detectors for LOOPCALL and jump opcodes with     */
    /* negative arguments.                                               */
    FT_ULong           loopcall_counter;
    FT_ULong           loopcall_counter_max;
    FT_ULong           neg_jump_counter;
    FT_ULong           neg_jump_counter_max;

  } TT_ExecContextRec;


  extern const TT_GraphicsState  tt_default_graphics_state;


#ifdef TT_USE_BYTECODE_INTERPRETER
  FT_LOCAL( void )
  TT_Goto_CodeRange( TT_ExecContext  exec,
                     FT_Int          range,
                     FT_Long         IP );

  FT_LOCAL( void )
  TT_Set_CodeRange( TT_ExecContext  exec,
                    FT_Int          range,
                    void*           base,
                    FT_Long         length );

  FT_LOCAL( void )
  TT_Clear_CodeRange( TT_ExecContext  exec,
                      FT_Int          range );
#endif /* TT_USE_BYTECODE_INTERPRETER */


  /**************************************************************************
   *
   * @Function:
   *   TT_New_Context
   *
   * @Description:
   *   Create a `TT_ExecContext`.  Note that there is now an execution
   *   context per `TT_Size` that is not shared among faces.
   *
   * @Input:
   *   driver ::
   *     A handle to the driver, used for memory allocation.
   *
   * @Return:
   *   A handle to a new empty execution context.
   *
   * @Note:
   *   Only the glyph loader and debugger should call this function.
   *   (And right now only the glyph loader uses it.)
   */
  FT_EXPORT( TT_ExecContext )
  TT_New_Context( TT_Driver  driver );


#ifdef TT_USE_BYTECODE_INTERPRETER
  FT_LOCAL( void )
  TT_Done_Context( TT_ExecContext  exec );

  FT_LOCAL( FT_Error )
  TT_Load_Context( TT_ExecContext  exec,
                   TT_Face         face,
                   TT_Size         size );

  FT_LOCAL( void )
  TT_Save_Context( TT_ExecContext  exec,
                   TT_Size         ins );

  FT_LOCAL( FT_Error )
  TT_Run_Context( TT_ExecContext  exec );
#endif /* TT_USE_BYTECODE_INTERPRETER */


  /**************************************************************************
   *
   * @Function:
   *   TT_RunIns
   *
   * @Description:
   *   Executes one or more instruction in the execution context.  This
   *   is the main function of the TrueType opcode interpreter.
   *
   * @Input:
   *   exec ::
   *     A handle to the target execution context.
   *
   * @Return:
   *   FreeType error code.  0 means success.
   *
   * @Note:
   *   Only the object manager and debugger should call this function.
   *
   *   This function is publicly exported because it is directly
   *   invoked by the TrueType debugger.
   */
  FT_EXPORT( FT_Error )
  TT_RunIns( void*  exec );


FT_END_HEADER

#endif /* TTINTERP_H_ */


/* END */
