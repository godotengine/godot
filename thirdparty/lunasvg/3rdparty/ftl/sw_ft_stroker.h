#ifndef SW_FT_STROKER_H
#define SW_FT_STROKER_H
/***************************************************************************/
/*                                                                         */
/*  ftstroke.h                                                             */
/*                                                                         */
/*    FreeType path stroker (specification).                               */
/*                                                                         */
/*  Copyright 2002-2006, 2008, 2009, 2011-2012 by                          */
/*  David Turner, Robert Wilhelm, and Werner Lemberg.                      */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/

#include "sw_ft_raster.h"

 /**************************************************************
  *
  * @type:
  *   SW_FT_Stroker
  *
  * @description:
  *   Opaque handler to a path stroker object.
  */
  typedef struct SW_FT_StrokerRec_*  SW_FT_Stroker;


  /**************************************************************
   *
   * @enum:
   *   SW_FT_Stroker_LineJoin
   *
   * @description:
   *   These values determine how two joining lines are rendered
   *   in a stroker.
   *
   * @values:
   *   SW_FT_STROKER_LINEJOIN_ROUND ::
   *     Used to render rounded line joins.  Circular arcs are used
   *     to join two lines smoothly.
   *
   *   SW_FT_STROKER_LINEJOIN_BEVEL ::
   *     Used to render beveled line joins.  The outer corner of
   *     the joined lines is filled by enclosing the triangular
   *     region of the corner with a straight line between the
   *     outer corners of each stroke.
   *
   *   SW_FT_STROKER_LINEJOIN_MITER_FIXED ::
   *     Used to render mitered line joins, with fixed bevels if the
   *     miter limit is exceeded.  The outer edges of the strokes
   *     for the two segments are extended until they meet at an
   *     angle.  If the segments meet at too sharp an angle (such
   *     that the miter would extend from the intersection of the
   *     segments a distance greater than the product of the miter
   *     limit value and the border radius), then a bevel join (see
   *     above) is used instead.  This prevents long spikes being
   *     created.  SW_FT_STROKER_LINEJOIN_MITER_FIXED generates a miter
   *     line join as used in PostScript and PDF.
   *
   *   SW_FT_STROKER_LINEJOIN_MITER_VARIABLE ::
   *   SW_FT_STROKER_LINEJOIN_MITER ::
   *     Used to render mitered line joins, with variable bevels if
   *     the miter limit is exceeded.  The intersection of the
   *     strokes is clipped at a line perpendicular to the bisector
   *     of the angle between the strokes, at the distance from the
   *     intersection of the segments equal to the product of the
   *     miter limit value and the border radius.  This prevents
   *     long spikes being created.
   *     SW_FT_STROKER_LINEJOIN_MITER_VARIABLE generates a mitered line
   *     join as used in XPS.  SW_FT_STROKER_LINEJOIN_MITER is an alias
   *     for SW_FT_STROKER_LINEJOIN_MITER_VARIABLE, retained for
   *     backwards compatibility.
   */
  typedef enum  SW_FT_Stroker_LineJoin_
  {
    SW_FT_STROKER_LINEJOIN_ROUND          = 0,
    SW_FT_STROKER_LINEJOIN_BEVEL          = 1,
    SW_FT_STROKER_LINEJOIN_MITER_VARIABLE = 2,
    SW_FT_STROKER_LINEJOIN_MITER          = SW_FT_STROKER_LINEJOIN_MITER_VARIABLE,
    SW_FT_STROKER_LINEJOIN_MITER_FIXED    = 3

  } SW_FT_Stroker_LineJoin;


  /**************************************************************
   *
   * @enum:
   *   SW_FT_Stroker_LineCap
   *
   * @description:
   *   These values determine how the end of opened sub-paths are
   *   rendered in a stroke.
   *
   * @values:
   *   SW_FT_STROKER_LINECAP_BUTT ::
   *     The end of lines is rendered as a full stop on the last
   *     point itself.
   *
   *   SW_FT_STROKER_LINECAP_ROUND ::
   *     The end of lines is rendered as a half-circle around the
   *     last point.
   *
   *   SW_FT_STROKER_LINECAP_SQUARE ::
   *     The end of lines is rendered as a square around the
   *     last point.
   */
  typedef enum  SW_FT_Stroker_LineCap_
  {
    SW_FT_STROKER_LINECAP_BUTT = 0,
    SW_FT_STROKER_LINECAP_ROUND,
    SW_FT_STROKER_LINECAP_SQUARE

  } SW_FT_Stroker_LineCap;


  /**************************************************************
   *
   * @enum:
   *   SW_FT_StrokerBorder
   *
   * @description:
   *   These values are used to select a given stroke border
   *   in @SW_FT_Stroker_GetBorderCounts and @SW_FT_Stroker_ExportBorder.
   *
   * @values:
   *   SW_FT_STROKER_BORDER_LEFT ::
   *     Select the left border, relative to the drawing direction.
   *
   *   SW_FT_STROKER_BORDER_RIGHT ::
   *     Select the right border, relative to the drawing direction.
   *
   * @note:
   *   Applications are generally interested in the `inside' and `outside'
   *   borders.  However, there is no direct mapping between these and the
   *   `left' and `right' ones, since this really depends on the glyph's
   *   drawing orientation, which varies between font formats.
   *
   *   You can however use @SW_FT_Outline_GetInsideBorder and
   *   @SW_FT_Outline_GetOutsideBorder to get these.
   */
  typedef enum  SW_FT_StrokerBorder_
  {
    SW_FT_STROKER_BORDER_LEFT = 0,
    SW_FT_STROKER_BORDER_RIGHT

  } SW_FT_StrokerBorder;


  /**************************************************************
   *
   * @function:
   *   SW_FT_Stroker_New
   *
   * @description:
   *   Create a new stroker object.
   *
   * @input:
   *   library ::
   *     FreeType library handle.
   *
   * @output:
   *   astroker ::
   *     A new stroker object handle.  NULL in case of error.
   *
   * @return:
   *    FreeType error code.  0~means success.
   */
  SW_FT_Error
  SW_FT_Stroker_New( SW_FT_Stroker  *astroker );


  /**************************************************************
   *
   * @function:
   *   SW_FT_Stroker_Set
   *
   * @description:
   *   Reset a stroker object's attributes.
   *
   * @input:
   *   stroker ::
   *     The target stroker handle.
   *
   *   radius ::
   *     The border radius.
   *
   *   line_cap ::
   *     The line cap style.
   *
   *   line_join ::
   *     The line join style.
   *
   *   miter_limit ::
   *     The miter limit for the SW_FT_STROKER_LINEJOIN_MITER_FIXED and
   *     SW_FT_STROKER_LINEJOIN_MITER_VARIABLE line join styles,
   *     expressed as 16.16 fixed-point value.
   *
   * @note:
   *   The radius is expressed in the same units as the outline
   *   coordinates.
   */
  void
  SW_FT_Stroker_Set( SW_FT_Stroker           stroker,
                  SW_FT_Fixed             radius,
                  SW_FT_Stroker_LineCap   line_cap,
                  SW_FT_Stroker_LineJoin  line_join,
                  SW_FT_Fixed             miter_limit );

  /**************************************************************
   *
   * @function:
   *   SW_FT_Stroker_ParseOutline
   *
   * @description:
   *   A convenience function used to parse a whole outline with
   *   the stroker.  The resulting outline(s) can be retrieved
   *   later by functions like @SW_FT_Stroker_GetCounts and @SW_FT_Stroker_Export.
   *
   * @input:
   *   stroker ::
   *     The target stroker handle.
   *
   *   outline ::
   *     The source outline.
   *
   *
   * @return:
   *   FreeType error code.  0~means success.
   *
   * @note:
   *   If `opened' is~0 (the default), the outline is treated as a closed
   *   path, and the stroker generates two distinct `border' outlines.
   *
   *
   *   This function calls @SW_FT_Stroker_Rewind automatically.
   */
  SW_FT_Error
  SW_FT_Stroker_ParseOutline( SW_FT_Stroker   stroker,
                             const SW_FT_Outline*  outline);


  /**************************************************************
   *
   * @function:
   *   SW_FT_Stroker_GetCounts
   *
   * @description:
   *   Call this function once you have finished parsing your paths
   *   with the stroker.  It returns the number of points and
   *   contours necessary to export all points/borders from the stroked
   *   outline/path.
   *
   * @input:
   *   stroker ::
   *     The target stroker handle.
   *
   * @output:
   *   anum_points ::
   *     The number of points.
   *
   *   anum_contours ::
   *     The number of contours.
   *
   * @return:
   *   FreeType error code.  0~means success.
   */
  SW_FT_Error
  SW_FT_Stroker_GetCounts( SW_FT_Stroker  stroker,
                        SW_FT_UInt    *anum_points,
                        SW_FT_UInt    *anum_contours );


  /**************************************************************
   *
   * @function:
   *   SW_FT_Stroker_Export
   *
   * @description:
   *   Call this function after @SW_FT_Stroker_GetBorderCounts to
   *   export all borders to your own @SW_FT_Outline structure.
   *
   *   Note that this function appends the border points and
   *   contours to your outline, but does not try to resize its
   *   arrays.
   *
   * @input:
   *   stroker ::
   *     The target stroker handle.
   *
   *   outline ::
   *     The target outline handle.
   */
  void
  SW_FT_Stroker_Export( SW_FT_Stroker   stroker,
                     SW_FT_Outline*  outline );


  /**************************************************************
   *
   * @function:
   *   SW_FT_Stroker_Done
   *
   * @description:
   *   Destroy a stroker object.
   *
   * @input:
   *   stroker ::
   *     A stroker handle.  Can be NULL.
   */
  void
  SW_FT_Stroker_Done( SW_FT_Stroker  stroker );


#endif // SW_FT_STROKER_H
