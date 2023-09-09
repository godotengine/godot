/****************************************************************************
 *
 * ttobjs.h
 *
 *   Objects manager (specification).
 *
 * Copyright (C) 1996-2023 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#ifndef TTOBJS_H_
#define TTOBJS_H_


#include <freetype/internal/ftobjs.h>
#include <freetype/internal/tttypes.h>


FT_BEGIN_HEADER


  /**************************************************************************
   *
   * @Type:
   *   TT_Driver
   *
   * @Description:
   *   A handle to a TrueType driver object.
   */
  typedef struct TT_DriverRec_*  TT_Driver;


  /**************************************************************************
   *
   * @Type:
   *   TT_GlyphSlot
   *
   * @Description:
   *   A handle to a TrueType glyph slot object.
   *
   * @Note:
   *   This is a direct typedef of FT_GlyphSlot, as there is nothing
   *   specific about the TrueType glyph slot.
   */
  typedef FT_GlyphSlot  TT_GlyphSlot;


  /**************************************************************************
   *
   * @Struct:
   *   TT_GraphicsState
   *
   * @Description:
   *   The TrueType graphics state used during bytecode interpretation.
   */
  typedef struct  TT_GraphicsState_
  {
    FT_UShort      rp0;
    FT_UShort      rp1;
    FT_UShort      rp2;

    FT_UnitVector  dualVector;
    FT_UnitVector  projVector;
    FT_UnitVector  freeVector;

    FT_Long        loop;
    FT_F26Dot6     minimum_distance;
    FT_Int         round_state;

    FT_Bool        auto_flip;
    FT_F26Dot6     control_value_cutin;
    FT_F26Dot6     single_width_cutin;
    FT_F26Dot6     single_width_value;
    FT_UShort      delta_base;
    FT_UShort      delta_shift;

    FT_Byte        instruct_control;
    /* According to Greg Hitchcock from Microsoft, the `scan_control'     */
    /* variable as documented in the TrueType specification is a 32-bit   */
    /* integer; the high-word part holds the SCANTYPE value, the low-word */
    /* part the SCANCTRL value.  We separate it into two fields.          */
    FT_Bool        scan_control;
    FT_Int         scan_type;

    FT_UShort      gep0;
    FT_UShort      gep1;
    FT_UShort      gep2;

  } TT_GraphicsState;


#ifdef TT_USE_BYTECODE_INTERPRETER

  FT_LOCAL( void )
  tt_glyphzone_done( TT_GlyphZone  zone );

  FT_LOCAL( FT_Error )
  tt_glyphzone_new( FT_Memory     memory,
                    FT_UShort     maxPoints,
                    FT_Short      maxContours,
                    TT_GlyphZone  zone );

#endif /* TT_USE_BYTECODE_INTERPRETER */



  /**************************************************************************
   *
   * EXECUTION SUBTABLES
   *
   * These sub-tables relate to instruction execution.
   *
   */


#define TT_MAX_CODE_RANGES  3


  /**************************************************************************
   *
   * There can only be 3 active code ranges at once:
   *   - the Font Program
   *   - the CVT Program
   *   - a glyph's instructions set
   */
  typedef enum  TT_CodeRange_Tag_
  {
    tt_coderange_none = 0,
    tt_coderange_font,
    tt_coderange_cvt,
    tt_coderange_glyph

  } TT_CodeRange_Tag;


  typedef struct  TT_CodeRange_
  {
    FT_Byte*  base;
    FT_Long   size;

  } TT_CodeRange;

  typedef TT_CodeRange  TT_CodeRangeTable[TT_MAX_CODE_RANGES];


  /**************************************************************************
   *
   * Defines a function/instruction definition record.
   */
  typedef struct  TT_DefRecord_
  {
    FT_Int    range;          /* in which code range is it located?     */
    FT_Long   start;          /* where does it start?                   */
    FT_Long   end;            /* where does it end?                     */
    FT_UInt   opc;            /* function #, or instruction code        */
    FT_Bool   active;         /* is it active?                          */

  } TT_DefRecord, *TT_DefArray;


  /**************************************************************************
   *
   * Subglyph transformation record.
   */
  typedef struct  TT_Transform_
  {
    FT_Fixed    xx, xy;     /* transformation matrix coefficients */
    FT_Fixed    yx, yy;
    FT_F26Dot6  ox, oy;     /* offsets                            */

  } TT_Transform;


  /**************************************************************************
   *
   * A note regarding non-squared pixels:
   *
   * (This text will probably go into some docs at some time; for now, it
   * is kept here to explain some definitions in the TT_Size_Metrics
   * record).
   *
   * The CVT is a one-dimensional array containing values that control
   * certain important characteristics in a font, like the height of all
   * capitals, all lowercase letter, default spacing or stem width/height.
   *
   * These values are found in FUnits in the font file, and must be scaled
   * to pixel coordinates before being used by the CVT and glyph programs.
   * Unfortunately, when using distinct x and y resolutions (or distinct x
   * and y pointsizes), there are two possible scalings.
   *
   * A first try was to implement a `lazy' scheme where all values were
   * scaled when first used.  However, while some values are always used
   * in the same direction, some others are used under many different
   * circumstances and orientations.
   *
   * I have found a simpler way to do the same, and it even seems to work
   * in most of the cases:
   *
   * - All CVT values are scaled to the maximum ppem size.
   *
   * - When performing a read or write in the CVT, a ratio factor is used
   *   to perform adequate scaling.  Example:
   *
   *     x_ppem = 14
   *     y_ppem = 10
   *
   *   We choose ppem = x_ppem = 14 as the CVT scaling size.  All cvt
   *   entries are scaled to it.
   *
   *     x_ratio = 1.0
   *     y_ratio = y_ppem/ppem (< 1.0)
   *
   *   We compute the current ratio like:
   *
   *   - If projVector is horizontal,
   *       ratio = x_ratio = 1.0
   *
   *   - if projVector is vertical,
   *       ratio = y_ratio
   *
   *   - else,
   *       ratio = sqrt( (proj.x * x_ratio) ^ 2 + (proj.y * y_ratio) ^ 2 )
   *
   *   Reading a cvt value returns
   *     ratio * cvt[index]
   *
   *   Writing a cvt value in pixels:
   *     cvt[index] / ratio
   *
   *   The current ppem is simply
   *     ratio * ppem
   *
   */


  /**************************************************************************
   *
   * Metrics used by the TrueType size and context objects.
   */
  typedef struct  TT_Size_Metrics_
  {
    /* for non-square pixels */
    FT_Long     x_ratio;
    FT_Long     y_ratio;

    FT_UShort   ppem;               /* maximum ppem size              */
    FT_Long     ratio;              /* current ratio                  */
    FT_Fixed    scale;

    FT_F26Dot6  compensations[4];   /* device-specific compensations  */

    FT_Bool     valid;

    FT_Bool     rotated;            /* `is the glyph rotated?'-flag   */
    FT_Bool     stretched;          /* `is the glyph stretched?'-flag */

  } TT_Size_Metrics;


  /**************************************************************************
   *
   * TrueType size class.
   */
  typedef struct  TT_SizeRec_
  {
    FT_SizeRec         root;

    /* we have our own copy of metrics so that we can modify */
    /* it without affecting auto-hinting (when used)         */
    FT_Size_Metrics*   metrics;        /* for the current rendering mode */
    FT_Size_Metrics    hinted_metrics; /* for the hinted rendering mode  */

    TT_Size_Metrics    ttmetrics;

    FT_Byte*           widthp;          /* glyph widths from the hdmx table */

    FT_ULong           strike_index;      /* 0xFFFFFFFF to indicate invalid */

#ifdef TT_USE_BYTECODE_INTERPRETER

    FT_Long            point_size;    /* for the `MPS' bytecode instruction */

    FT_UInt            num_function_defs; /* number of function definitions */
    FT_UInt            max_function_defs;
    TT_DefArray        function_defs;     /* table of function definitions  */

    FT_UInt            num_instruction_defs;  /* number of ins. definitions */
    FT_UInt            max_instruction_defs;
    TT_DefArray        instruction_defs;      /* table of ins. definitions  */

    FT_UInt            max_func;
    FT_UInt            max_ins;

    TT_CodeRangeTable  codeRangeTable;

    TT_GraphicsState   GS;

    FT_ULong           cvt_size;      /* the scaled control value table */
    FT_Long*           cvt;

    FT_UShort          storage_size; /* The storage area is now part of */
    FT_Long*           storage;      /* the instance                    */

    TT_GlyphZoneRec    twilight;     /* The instance's twilight zone    */

    TT_ExecContext     context;

    /* if negative, `fpgm' (resp. `prep'), wasn't executed yet; */
    /* otherwise it is the returned error code                  */
    FT_Error           bytecode_ready;
    FT_Error           cvt_ready;

#endif /* TT_USE_BYTECODE_INTERPRETER */

  } TT_SizeRec;


  /**************************************************************************
   *
   * TrueType driver class.
   */
  typedef struct  TT_DriverRec_
  {
    FT_DriverRec  root;

    TT_GlyphZoneRec  zone;     /* glyph loader points zone */

    FT_UInt  interpreter_version;

  } TT_DriverRec;


  /* Note: All of the functions below (except tt_size_reset()) are used    */
  /* as function pointers in a FT_Driver_ClassRec.  Therefore their        */
  /* parameters are of types FT_Face, FT_Size, etc., rather than TT_Face,  */
  /* TT_Size, etc., so that the compiler can confirm that the types and    */
  /* number of parameters are correct.  In all cases the FT_xxx types are  */
  /* cast to their TT_xxx counterparts inside the functions since FreeType */
  /* will always use the TT driver to create them.                         */


  /**************************************************************************
   *
   * Face functions
   */
  FT_LOCAL( FT_Error )
  tt_face_init( FT_Stream      stream,
                FT_Face        ttface,      /* TT_Face */
                FT_Int         face_index,
                FT_Int         num_params,
                FT_Parameter*  params );

  FT_LOCAL( void )
  tt_face_done( FT_Face  ttface );          /* TT_Face */


  /**************************************************************************
   *
   * Size functions
   */
  FT_LOCAL( FT_Error )
  tt_size_init( FT_Size  ttsize );          /* TT_Size */

  FT_LOCAL( void )
  tt_size_done( FT_Size  ttsize );          /* TT_Size */

#ifdef TT_USE_BYTECODE_INTERPRETER

  FT_LOCAL( FT_Error )
  tt_size_run_fpgm( TT_Size  size,
                    FT_Bool  pedantic );

  FT_LOCAL( FT_Error )
  tt_size_run_prep( TT_Size  size,
                    FT_Bool  pedantic );

  FT_LOCAL( FT_Error )
  tt_size_ready_bytecode( TT_Size  size,
                          FT_Bool  pedantic );

#endif /* TT_USE_BYTECODE_INTERPRETER */

  FT_LOCAL( FT_Error )
  tt_size_reset_height( FT_Size  size );

  FT_LOCAL( FT_Error )
  tt_size_reset( TT_Size  size );


  /**************************************************************************
   *
   * Driver functions
   */
  FT_LOCAL( FT_Error )
  tt_driver_init( FT_Module  ttdriver );    /* TT_Driver */

  FT_LOCAL( void )
  tt_driver_done( FT_Module  ttdriver );    /* TT_Driver */


  /**************************************************************************
   *
   * Slot functions
   */
  FT_LOCAL( FT_Error )
  tt_slot_init( FT_GlyphSlot  slot );


  /* auxiliary */
#define IS_HINTED( flags )  ( ( flags & FT_LOAD_NO_HINTING ) == 0 )


FT_END_HEADER

#endif /* TTOBJS_H_ */


/* END */
