/****************************************************************************
 *
 * otsvg.h
 *
 *   Interface for OT-SVG support related things (specification).
 *
 * Copyright (C) 2022-2025 by
 * David Turner, Robert Wilhelm, Werner Lemberg, and Moazin Khatti.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, LICENSE.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */


#ifndef OTSVG_H_
#define OTSVG_H_

#include <freetype/freetype.h>

#ifdef FREETYPE_H
#error "freetype.h of FreeType 1 has been loaded!"
#error "Please fix the directory search order for header files"
#error "so that freetype.h of FreeType 2 is found first."
#endif


FT_BEGIN_HEADER


  /**************************************************************************
   *
   * @section:
   *   svg_fonts
   *
   * @title:
   *   OpenType SVG Fonts
   *
   * @abstract:
   *   OT-SVG API between FreeType and an external SVG rendering library.
   *
   * @description:
   *   This section describes the four hooks necessary to render SVG
   *   'documents' that are contained in an OpenType font's 'SVG~' table.
   *
   *   For more information on the implementation, see our standard hooks
   *   based on 'librsvg' in the [FreeType Demo
   *   Programs](https://gitlab.freedesktop.org/freetype/freetype-demos)
   *   repository.
   *
   */


  /**************************************************************************
   *
   * @functype:
   *   SVG_Lib_Init_Func
   *
   * @description:
   *   A callback that is called when the first OT-SVG glyph is rendered in
   *   the lifetime of an @FT_Library object.  In a typical implementation,
   *   one would want to allocate a structure and point the `data_pointer`
   *   to it and perform any library initializations that might be needed.
   *
   * @inout:
   *   data_pointer ::
   *     The SVG rendering module stores a pointer variable that can be used
   *     by clients to store any data that needs to be shared across
   *     different hooks.  `data_pointer` is essentially a pointer to that
   *     pointer such that it can be written to as well as read from.
   *
   * @return:
   *   FreeType error code.  0 means success.
   *
   * @since:
   *   2.12
   */
  typedef FT_Error
  (*SVG_Lib_Init_Func)( FT_Pointer  *data_pointer );


  /**************************************************************************
   *
   * @functype:
   *   SVG_Lib_Free_Func
   *
   * @description:
   *   A callback that is called when the `ot-svg` module is being freed.
   *   It is only called if the init hook was called earlier.  This means
   *   that neither the init nor the free hook is called if no OT-SVG glyph
   *   is rendered.
   *
   *   In a typical implementation, one would want to free any state
   *   structure that was allocated in the init hook and perform any
   *   library-related closure that might be needed.
   *
   * @inout:
   *   data_pointer ::
   *     The SVG rendering module stores a pointer variable that can be used
   *     by clients to store any data that needs to be shared across
   *     different hooks.  `data_pointer` is essentially a pointer to that
   *     pointer such that it can be written to as well as read from.
   *
   * @since:
   *   2.12
   */
  typedef void
  (*SVG_Lib_Free_Func)( FT_Pointer  *data_pointer );


  /**************************************************************************
   *
   * @functype:
   *   SVG_Lib_Render_Func
   *
   * @description:
   *   A callback that is called to render an OT-SVG glyph.  This callback
   *   hook is called right after the preset hook @SVG_Lib_Preset_Slot_Func
   *   has been called with `cache` set to `TRUE`.  The data necessary to
   *   render is available through the handle @FT_SVG_Document, which is set
   *   in the `other` field of @FT_GlyphSlotRec.
   *
   *   The render hook is expected to render the SVG glyph to the bitmap
   *   buffer that is allocated already at `slot->bitmap.buffer`.  It also
   *   sets the `num_grays` value as well as `slot->format`.
   *
   * @input:
   *   slot ::
   *     The slot to render.
   *
   * @inout:
   *   data_pointer ::
   *     The SVG rendering module stores a pointer variable that can be used
   *     by clients to store any data that needs to be shared across
   *     different hooks.  `data_pointer` is essentially a pointer to that
   *     pointer such that it can be written to as well as read from.
   *
   * @return:
   *   FreeType error code.  0 means success.
   *
   * @since:
   *   2.12
   */
  typedef FT_Error
  (*SVG_Lib_Render_Func)( FT_GlyphSlot  slot,
                          FT_Pointer   *data_pointer );


  /**************************************************************************
   *
   * @functype:
   *   SVG_Lib_Preset_Slot_Func
   *
   * @description:
   *   A callback that is called to preset the glyph slot.  It is called from
   *   two places.
   *
   *   1. When `FT_Load_Glyph` needs to preset the glyph slot.
   *
   *   2. Right before the `svg` module calls the render callback hook.
   *
   *   When it is the former, the argument `cache` is set to `FALSE`.  When
   *   it is the latter, the argument `cache` is set to `TRUE`.  This
   *   distinction has been made because many calculations that are necessary
   *   for presetting a glyph slot are the same needed later for the render
   *   callback hook.  Thus, if `cache` is `TRUE`, the hook can _cache_ those
   *   calculations in a memory block referenced by the state pointer.
   *
   *   This hook is expected to preset the slot by setting parameters such as
   *   `bitmap_left`, `bitmap_top`, `width`, `rows`, `pitch`, and
   *   `pixel_mode`.  It is also expected to set all the metrics for the slot
   *   including the vertical advance if it is not already set.  Typically,
   *   fonts have horizontal advances but not vertical ones.  If those are
   *   available, they had already been set, otherwise they have to be
   *   estimated and set manually.  The hook must take into account the
   *   transformations that have been set, and translate the transformation
   *   matrices into the SVG coordinate system, as the original matrix is
   *   intended for the TTF/CFF coordinate system.
   *
   * @input:
   *   slot ::
   *     The glyph slot that has the SVG document loaded.
   *
   *   cache ::
   *     See description.
   *
   * @inout:
   *   data_pointer ::
   *     The SVG rendering module stores a pointer variable that can be used
   *     by clients to store any data that needs to be shared across
   *     different hooks.  `data_pointer` is essentially a pointer to that
   *     pointer such that it can be written to as well as read from.
   *
   * @return:
   *   FreeType error code.  0 means success.
   *
   * @since:
   *   2.12
   */
  typedef FT_Error
  (*SVG_Lib_Preset_Slot_Func)( FT_GlyphSlot  slot,
                               FT_Bool       cache,
                               FT_Pointer   *state );


  /**************************************************************************
   *
   * @struct:
   *   SVG_RendererHooks
   *
   * @description:
   *   A structure that stores the four hooks needed to render OT-SVG glyphs
   *   properly.  The structure is publicly used to set the hooks via the
   *   @svg-hooks driver property.
   *
   *   The behavior of each hook is described in its documentation.  One
   *   thing to note is that the preset hook and the render hook often need
   *   to do the same operations; therefore, it's better to cache the
   *   intermediate data in a state structure to avoid calculating it twice.
   *   For example, in the preset hook one can draw the glyph on a recorder
   *   surface and later create a bitmap surface from it in the render hook.
   *
   *   All four hooks must be non-NULL.
   *
   * @fields:
   *   init_svg ::
   *     The initialization hook.
   *
   *   free_svg ::
   *     The cleanup hook.
   *
   *   render_hook ::
   *     The render hook.
   *
   *   preset_slot ::
   *     The preset hook.
   *
   * @since:
   *   2.12
   */
  typedef struct SVG_RendererHooks_
  {
    SVG_Lib_Init_Func    init_svg;
    SVG_Lib_Free_Func    free_svg;
    SVG_Lib_Render_Func  render_svg;

    SVG_Lib_Preset_Slot_Func  preset_slot;

  } SVG_RendererHooks;


  /**************************************************************************
   *
   * @struct:
   *   FT_SVG_DocumentRec
   *
   * @description:
   *   A structure that models one SVG document.
   *
   * @fields:
   *   svg_document ::
   *     A pointer to the SVG document.
   *
   *   svg_document_length ::
   *     The length of `svg_document`.
   *
   *   metrics ::
   *     A metrics object storing the size information.
   *
   *   units_per_EM ::
   *     The size of the EM square.
   *
   *   start_glyph_id ::
   *     The first glyph ID in the glyph range covered by this document.
   *
   *   end_glyph_id ::
   *     The last glyph ID in the glyph range covered by this document.
   *
   *   transform ::
   *     A 2x2 transformation matrix to apply to the glyph while rendering
   *     it.
   *
   *   delta ::
   *     The translation to apply to the glyph while rendering.
   *
   * @note:
   *   When an @FT_GlyphSlot object `slot` is passed down to a renderer, the
   *   renderer can only access the `metrics` and `units_per_EM` fields via
   *   `slot->face`.  However, when @FT_Glyph_To_Bitmap sets up a dummy
   *   object, it has no way to set a `face` object.  Thus, metrics
   *   information and `units_per_EM` (which is necessary for OT-SVG) has to
   *   be stored separately.
   *
   * @since:
   *   2.12
   */
  typedef struct  FT_SVG_DocumentRec_
  {
    FT_Byte*  svg_document;
    FT_ULong  svg_document_length;

    FT_Size_Metrics  metrics;
    FT_UShort        units_per_EM;

    FT_UShort  start_glyph_id;
    FT_UShort  end_glyph_id;

    FT_Matrix  transform;
    FT_Vector  delta;

  } FT_SVG_DocumentRec;


  /**************************************************************************
   *
   * @type:
   *   FT_SVG_Document
   *
   * @description:
   *   A handle to an @FT_SVG_DocumentRec object.
   *
   * @since:
   *   2.12
   */
  typedef struct FT_SVG_DocumentRec_*  FT_SVG_Document;


FT_END_HEADER

#endif /* OTSVG_H_ */


/* END */
