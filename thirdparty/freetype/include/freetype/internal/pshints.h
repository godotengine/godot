/****************************************************************************
 *
 * pshints.h
 *
 *   Interface to Postscript-specific (Type 1 and Type 2) hints
 *   recorders (specification only).  These are used to support native
 *   T1/T2 hints in the 'type1', 'cid', and 'cff' font drivers.
 *
 * Copyright (C) 2001-2022 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#ifndef PSHINTS_H_
#define PSHINTS_H_


#include <freetype/freetype.h>
#include <freetype/t1tables.h>


FT_BEGIN_HEADER


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****               INTERNAL REPRESENTATION OF GLOBALS              *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  typedef struct PSH_GlobalsRec_*  PSH_Globals;

  typedef FT_Error
  (*PSH_Globals_NewFunc)( FT_Memory     memory,
                          T1_Private*   private_dict,
                          PSH_Globals*  aglobals );

  typedef void
  (*PSH_Globals_SetScaleFunc)( PSH_Globals  globals,
                               FT_Fixed     x_scale,
                               FT_Fixed     y_scale,
                               FT_Fixed     x_delta,
                               FT_Fixed     y_delta );

  typedef void
  (*PSH_Globals_DestroyFunc)( PSH_Globals  globals );


  typedef struct  PSH_Globals_FuncsRec_
  {
    PSH_Globals_NewFunc       create;
    PSH_Globals_SetScaleFunc  set_scale;
    PSH_Globals_DestroyFunc   destroy;

  } PSH_Globals_FuncsRec, *PSH_Globals_Funcs;


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                  PUBLIC TYPE 1 HINTS RECORDER                 *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  /**************************************************************************
   *
   * @type:
   *   T1_Hints
   *
   * @description:
   *   This is a handle to an opaque structure used to record glyph hints
   *   from a Type 1 character glyph character string.
   *
   *   The methods used to operate on this object are defined by the
   *   @T1_Hints_FuncsRec structure.  Recording glyph hints is normally
   *   achieved through the following scheme:
   *
   *   - Open a new hint recording session by calling the 'open' method.
   *     This rewinds the recorder and prepare it for new input.
   *
   *   - For each hint found in the glyph charstring, call the corresponding
   *     method ('stem', 'stem3', or 'reset').  Note that these functions do
   *     not return an error code.
   *
   *   - Close the recording session by calling the 'close' method.  It
   *     returns an error code if the hints were invalid or something strange
   *     happened (e.g., memory shortage).
   *
   *   The hints accumulated in the object can later be used by the
   *   PostScript hinter.
   *
   */
  typedef struct T1_HintsRec_*  T1_Hints;


  /**************************************************************************
   *
   * @type:
   *   T1_Hints_Funcs
   *
   * @description:
   *   A pointer to the @T1_Hints_FuncsRec structure that defines the API of
   *   a given @T1_Hints object.
   *
   */
  typedef const struct T1_Hints_FuncsRec_*  T1_Hints_Funcs;


  /**************************************************************************
   *
   * @functype:
   *   T1_Hints_OpenFunc
   *
   * @description:
   *   A method of the @T1_Hints class used to prepare it for a new Type 1
   *   hints recording session.
   *
   * @input:
   *   hints ::
   *     A handle to the Type 1 hints recorder.
   *
   * @note:
   *   You should always call the @T1_Hints_CloseFunc method in order to
   *   close an opened recording session.
   *
   */
  typedef void
  (*T1_Hints_OpenFunc)( T1_Hints  hints );


  /**************************************************************************
   *
   * @functype:
   *   T1_Hints_SetStemFunc
   *
   * @description:
   *   A method of the @T1_Hints class used to record a new horizontal or
   *   vertical stem.  This corresponds to the Type 1 'hstem' and 'vstem'
   *   operators.
   *
   * @input:
   *   hints ::
   *     A handle to the Type 1 hints recorder.
   *
   *   dimension ::
   *     0 for horizontal stems (hstem), 1 for vertical ones (vstem).
   *
   *   coords ::
   *     Array of 2 coordinates in 16.16 format, used as (position,length)
   *     stem descriptor.
   *
   * @note:
   *   Use vertical coordinates (y) for horizontal stems (dim=0).  Use
   *   horizontal coordinates (x) for vertical stems (dim=1).
   *
   *   'coords[0]' is the absolute stem position (lowest coordinate);
   *   'coords[1]' is the length.
   *
   *   The length can be negative, in which case it must be either -20 or
   *   -21.  It is interpreted as a 'ghost' stem, according to the Type 1
   *   specification.
   *
   *   If the length is -21 (corresponding to a bottom ghost stem), then the
   *   real stem position is 'coords[0]+coords[1]'.
   *
   */
  typedef void
  (*T1_Hints_SetStemFunc)( T1_Hints   hints,
                           FT_UInt    dimension,
                           FT_Fixed*  coords );


  /**************************************************************************
   *
   * @functype:
   *   T1_Hints_SetStem3Func
   *
   * @description:
   *   A method of the @T1_Hints class used to record three
   *   counter-controlled horizontal or vertical stems at once.
   *
   * @input:
   *   hints ::
   *     A handle to the Type 1 hints recorder.
   *
   *   dimension ::
   *     0 for horizontal stems, 1 for vertical ones.
   *
   *   coords ::
   *     An array of 6 values in 16.16 format, holding 3 (position,length)
   *     pairs for the counter-controlled stems.
   *
   * @note:
   *   Use vertical coordinates (y) for horizontal stems (dim=0).  Use
   *   horizontal coordinates (x) for vertical stems (dim=1).
   *
   *   The lengths cannot be negative (ghost stems are never
   *   counter-controlled).
   *
   */
  typedef void
  (*T1_Hints_SetStem3Func)( T1_Hints   hints,
                            FT_UInt    dimension,
                            FT_Fixed*  coords );


  /**************************************************************************
   *
   * @functype:
   *   T1_Hints_ResetFunc
   *
   * @description:
   *   A method of the @T1_Hints class used to reset the stems hints in a
   *   recording session.
   *
   * @input:
   *   hints ::
   *     A handle to the Type 1 hints recorder.
   *
   *   end_point ::
   *     The index of the last point in the input glyph in which the
   *     previously defined hints apply.
   *
   */
  typedef void
  (*T1_Hints_ResetFunc)( T1_Hints  hints,
                         FT_UInt   end_point );


  /**************************************************************************
   *
   * @functype:
   *   T1_Hints_CloseFunc
   *
   * @description:
   *   A method of the @T1_Hints class used to close a hint recording
   *   session.
   *
   * @input:
   *   hints ::
   *     A handle to the Type 1 hints recorder.
   *
   *   end_point ::
   *     The index of the last point in the input glyph.
   *
   * @return:
   *   FreeType error code.  0 means success.
   *
   * @note:
   *   The error code is set to indicate that an error occurred during the
   *   recording session.
   *
   */
  typedef FT_Error
  (*T1_Hints_CloseFunc)( T1_Hints  hints,
                         FT_UInt   end_point );


  /**************************************************************************
   *
   * @functype:
   *   T1_Hints_ApplyFunc
   *
   * @description:
   *   A method of the @T1_Hints class used to apply hints to the
   *   corresponding glyph outline.  Must be called once all hints have been
   *   recorded.
   *
   * @input:
   *   hints ::
   *     A handle to the Type 1 hints recorder.
   *
   *   outline ::
   *     A pointer to the target outline descriptor.
   *
   *   globals ::
   *     The hinter globals for this font.
   *
   *   hint_mode ::
   *     Hinting information.
   *
   * @return:
   *   FreeType error code.  0 means success.
   *
   * @note:
   *   On input, all points within the outline are in font coordinates. On
   *   output, they are in 1/64th of pixels.
   *
   *   The scaling transformation is taken from the 'globals' object which
   *   must correspond to the same font as the glyph.
   *
   */
  typedef FT_Error
  (*T1_Hints_ApplyFunc)( T1_Hints        hints,
                         FT_Outline*     outline,
                         PSH_Globals     globals,
                         FT_Render_Mode  hint_mode );


  /**************************************************************************
   *
   * @struct:
   *   T1_Hints_FuncsRec
   *
   * @description:
   *   The structure used to provide the API to @T1_Hints objects.
   *
   * @fields:
   *   hints ::
   *     A handle to the T1 Hints recorder.
   *
   *   open ::
   *     The function to open a recording session.
   *
   *   close ::
   *     The function to close a recording session.
   *
   *   stem ::
   *     The function to set a simple stem.
   *
   *   stem3 ::
   *     The function to set counter-controlled stems.
   *
   *   reset ::
   *     The function to reset stem hints.
   *
   *   apply ::
   *     The function to apply the hints to the corresponding glyph outline.
   *
   */
  typedef struct  T1_Hints_FuncsRec_
  {
    T1_Hints               hints;
    T1_Hints_OpenFunc      open;
    T1_Hints_CloseFunc     close;
    T1_Hints_SetStemFunc   stem;
    T1_Hints_SetStem3Func  stem3;
    T1_Hints_ResetFunc     reset;
    T1_Hints_ApplyFunc     apply;

  } T1_Hints_FuncsRec;


  /*************************************************************************/
  /*************************************************************************/
  /*****                                                               *****/
  /*****                  PUBLIC TYPE 2 HINTS RECORDER                 *****/
  /*****                                                               *****/
  /*************************************************************************/
  /*************************************************************************/

  /**************************************************************************
   *
   * @type:
   *   T2_Hints
   *
   * @description:
   *   This is a handle to an opaque structure used to record glyph hints
   *   from a Type 2 character glyph character string.
   *
   *   The methods used to operate on this object are defined by the
   *   @T2_Hints_FuncsRec structure.  Recording glyph hints is normally
   *   achieved through the following scheme:
   *
   *   - Open a new hint recording session by calling the 'open' method.
   *     This rewinds the recorder and prepare it for new input.
   *
   *   - For each hint found in the glyph charstring, call the corresponding
   *     method ('stems', 'hintmask', 'counters').  Note that these functions
   *     do not return an error code.
   *
   *   - Close the recording session by calling the 'close' method.  It
   *     returns an error code if the hints were invalid or something strange
   *     happened (e.g., memory shortage).
   *
   *   The hints accumulated in the object can later be used by the
   *   Postscript hinter.
   *
   */
  typedef struct T2_HintsRec_*  T2_Hints;


  /**************************************************************************
   *
   * @type:
   *   T2_Hints_Funcs
   *
   * @description:
   *   A pointer to the @T2_Hints_FuncsRec structure that defines the API of
   *   a given @T2_Hints object.
   *
   */
  typedef const struct T2_Hints_FuncsRec_*  T2_Hints_Funcs;


  /**************************************************************************
   *
   * @functype:
   *   T2_Hints_OpenFunc
   *
   * @description:
   *   A method of the @T2_Hints class used to prepare it for a new Type 2
   *   hints recording session.
   *
   * @input:
   *   hints ::
   *     A handle to the Type 2 hints recorder.
   *
   * @note:
   *   You should always call the @T2_Hints_CloseFunc method in order to
   *   close an opened recording session.
   *
   */
  typedef void
  (*T2_Hints_OpenFunc)( T2_Hints  hints );


  /**************************************************************************
   *
   * @functype:
   *   T2_Hints_StemsFunc
   *
   * @description:
   *   A method of the @T2_Hints class used to set the table of stems in
   *   either the vertical or horizontal dimension.  Equivalent to the
   *   'hstem', 'vstem', 'hstemhm', and 'vstemhm' Type 2 operators.
   *
   * @input:
   *   hints ::
   *     A handle to the Type 2 hints recorder.
   *
   *   dimension ::
   *     0 for horizontal stems (hstem), 1 for vertical ones (vstem).
   *
   *   count ::
   *     The number of stems.
   *
   *   coords ::
   *     An array of 'count' (position,length) pairs in 16.16 format.
   *
   * @note:
   *   Use vertical coordinates (y) for horizontal stems (dim=0).  Use
   *   horizontal coordinates (x) for vertical stems (dim=1).
   *
   *   There are '2*count' elements in the 'coords' array.  Each even element
   *   is an absolute position in font units, each odd element is a length in
   *   font units.
   *
   *   A length can be negative, in which case it must be either -20 or -21.
   *   It is interpreted as a 'ghost' stem, according to the Type 1
   *   specification.
   *
   */
  typedef void
  (*T2_Hints_StemsFunc)( T2_Hints   hints,
                         FT_UInt    dimension,
                         FT_Int     count,
                         FT_Fixed*  coordinates );


  /**************************************************************************
   *
   * @functype:
   *   T2_Hints_MaskFunc
   *
   * @description:
   *   A method of the @T2_Hints class used to set a given hintmask (this
   *   corresponds to the 'hintmask' Type 2 operator).
   *
   * @input:
   *   hints ::
   *     A handle to the Type 2 hints recorder.
   *
   *   end_point ::
   *     The glyph index of the last point to which the previously defined or
   *     activated hints apply.
   *
   *   bit_count ::
   *     The number of bits in the hint mask.
   *
   *   bytes ::
   *     An array of bytes modelling the hint mask.
   *
   * @note:
   *   If the hintmask starts the charstring (before any glyph point
   *   definition), the value of `end_point` should be 0.
   *
   *   `bit_count` is the number of meaningful bits in the 'bytes' array; it
   *   must be equal to the total number of hints defined so far (i.e.,
   *   horizontal+verticals).
   *
   *   The 'bytes' array can come directly from the Type 2 charstring and
   *   respects the same format.
   *
   */
  typedef void
  (*T2_Hints_MaskFunc)( T2_Hints        hints,
                        FT_UInt         end_point,
                        FT_UInt         bit_count,
                        const FT_Byte*  bytes );


  /**************************************************************************
   *
   * @functype:
   *   T2_Hints_CounterFunc
   *
   * @description:
   *   A method of the @T2_Hints class used to set a given counter mask (this
   *   corresponds to the 'hintmask' Type 2 operator).
   *
   * @input:
   *   hints ::
   *     A handle to the Type 2 hints recorder.
   *
   *   end_point ::
   *     A glyph index of the last point to which the previously defined or
   *     active hints apply.
   *
   *   bit_count ::
   *     The number of bits in the hint mask.
   *
   *   bytes ::
   *     An array of bytes modelling the hint mask.
   *
   * @note:
   *   If the hintmask starts the charstring (before any glyph point
   *   definition), the value of `end_point` should be 0.
   *
   *   `bit_count` is the number of meaningful bits in the 'bytes' array; it
   *   must be equal to the total number of hints defined so far (i.e.,
   *   horizontal+verticals).
   *
   *    The 'bytes' array can come directly from the Type 2 charstring and
   *    respects the same format.
   *
   */
  typedef void
  (*T2_Hints_CounterFunc)( T2_Hints        hints,
                           FT_UInt         bit_count,
                           const FT_Byte*  bytes );


  /**************************************************************************
   *
   * @functype:
   *   T2_Hints_CloseFunc
   *
   * @description:
   *   A method of the @T2_Hints class used to close a hint recording
   *   session.
   *
   * @input:
   *   hints ::
   *     A handle to the Type 2 hints recorder.
   *
   *   end_point ::
   *     The index of the last point in the input glyph.
   *
   * @return:
   *   FreeType error code.  0 means success.
   *
   * @note:
   *   The error code is set to indicate that an error occurred during the
   *   recording session.
   *
   */
  typedef FT_Error
  (*T2_Hints_CloseFunc)( T2_Hints  hints,
                         FT_UInt   end_point );


  /**************************************************************************
   *
   * @functype:
   *   T2_Hints_ApplyFunc
   *
   * @description:
   *   A method of the @T2_Hints class used to apply hints to the
   *   corresponding glyph outline.  Must be called after the 'close' method.
   *
   * @input:
   *   hints ::
   *     A handle to the Type 2 hints recorder.
   *
   *   outline ::
   *     A pointer to the target outline descriptor.
   *
   *   globals ::
   *     The hinter globals for this font.
   *
   *   hint_mode ::
   *     Hinting information.
   *
   * @return:
   *   FreeType error code.  0 means success.
   *
   * @note:
   *   On input, all points within the outline are in font coordinates. On
   *   output, they are in 1/64th of pixels.
   *
   *   The scaling transformation is taken from the 'globals' object which
   *   must correspond to the same font than the glyph.
   *
   */
  typedef FT_Error
  (*T2_Hints_ApplyFunc)( T2_Hints        hints,
                         FT_Outline*     outline,
                         PSH_Globals     globals,
                         FT_Render_Mode  hint_mode );


  /**************************************************************************
   *
   * @struct:
   *   T2_Hints_FuncsRec
   *
   * @description:
   *   The structure used to provide the API to @T2_Hints objects.
   *
   * @fields:
   *   hints ::
   *     A handle to the T2 hints recorder object.
   *
   *   open ::
   *     The function to open a recording session.
   *
   *   close ::
   *     The function to close a recording session.
   *
   *   stems ::
   *     The function to set the dimension's stems table.
   *
   *   hintmask ::
   *     The function to set hint masks.
   *
   *   counter ::
   *     The function to set counter masks.
   *
   *   apply ::
   *     The function to apply the hints on the corresponding glyph outline.
   *
   */
  typedef struct  T2_Hints_FuncsRec_
  {
    T2_Hints              hints;
    T2_Hints_OpenFunc     open;
    T2_Hints_CloseFunc    close;
    T2_Hints_StemsFunc    stems;
    T2_Hints_MaskFunc     hintmask;
    T2_Hints_CounterFunc  counter;
    T2_Hints_ApplyFunc    apply;

  } T2_Hints_FuncsRec;


  /* */


  typedef struct  PSHinter_Interface_
  {
    PSH_Globals_Funcs  (*get_globals_funcs)( FT_Module  module );
    T1_Hints_Funcs     (*get_t1_funcs)     ( FT_Module  module );
    T2_Hints_Funcs     (*get_t2_funcs)     ( FT_Module  module );

  } PSHinter_Interface;

  typedef PSHinter_Interface*  PSHinter_Service;


#define FT_DEFINE_PSHINTER_INTERFACE(        \
          class_,                            \
          get_globals_funcs_,                \
          get_t1_funcs_,                     \
          get_t2_funcs_ )                    \
  static const PSHinter_Interface  class_ =  \
  {                                          \
    get_globals_funcs_,                      \
    get_t1_funcs_,                           \
    get_t2_funcs_                            \
  };


FT_END_HEADER

#endif /* PSHINTS_H_ */


/* END */
