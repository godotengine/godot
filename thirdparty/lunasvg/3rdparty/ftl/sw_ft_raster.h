#ifndef SW_FT_IMG_H
#define SW_FT_IMG_H
/***************************************************************************/
/*                                                                         */
/*  ftimage.h                                                              */
/*                                                                         */
/*    FreeType glyph image formats and default raster interface            */
/*    (specification).                                                     */
/*                                                                         */
/*  Copyright 1996-2010, 2013 by                                           */
/*  David Turner, Robert Wilhelm, and Werner Lemberg.                      */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/

  /*************************************************************************/
  /*                                                                       */
  /* Note: A `raster' is simply a scan-line converter, used to render      */
  /*       SW_FT_Outlines into SW_FT_Bitmaps.                                    */
  /*                                                                       */
  /*************************************************************************/

#include "sw_ft_types.h"

  /*************************************************************************/
  /*                                                                       */
  /* <Struct>                                                              */
  /*    FT_BBox                                                            */
  /*                                                                       */
  /* <Description>                                                         */
  /*    A structure used to hold an outline's bounding box, i.e., the      */
  /*    coordinates of its extrema in the horizontal and vertical          */
  /*    directions.                                                        */
  /*                                                                       */
  /* <Fields>                                                              */
  /*    xMin :: The horizontal minimum (left-most).                        */
  /*                                                                       */
  /*    yMin :: The vertical minimum (bottom-most).                        */
  /*                                                                       */
  /*    xMax :: The horizontal maximum (right-most).                       */
  /*                                                                       */
  /*    yMax :: The vertical maximum (top-most).                           */
  /*                                                                       */
  /* <Note>                                                                */
  /*    The bounding box is specified with the coordinates of the lower    */
  /*    left and the upper right corner.  In PostScript, those values are  */
  /*    often called (llx,lly) and (urx,ury), respectively.                */
  /*                                                                       */
  /*    If `yMin' is negative, this value gives the glyph's descender.     */
  /*    Otherwise, the glyph doesn't descend below the baseline.           */
  /*    Similarly, if `ymax' is positive, this value gives the glyph's     */
  /*    ascender.                                                          */
  /*                                                                       */
  /*    `xMin' gives the horizontal distance from the glyph's origin to    */
  /*    the left edge of the glyph's bounding box.  If `xMin' is negative, */
  /*    the glyph extends to the left of the origin.                       */
  /*                                                                       */
  typedef struct  SW_FT_BBox_
  {
    SW_FT_Pos  xMin, yMin;
    SW_FT_Pos  xMax, yMax;

  } SW_FT_BBox;

/*************************************************************************/
/*                                                                       */
/* <Struct>                                                              */
/*    SW_FT_Outline                                                      */
/*                                                                       */
/* <Description>                                                         */
/*    This structure is used to describe an outline to the scan-line     */
/*    converter.                                                         */
/*                                                                       */
/* <Fields>                                                              */
/*    n_contours :: The number of contours in the outline.               */
/*                                                                       */
/*    n_points   :: The number of points in the outline.                 */
/*                                                                       */
/*    points     :: A pointer to an array of `n_points' @SW_FT_Vector    */
/*                  elements, giving the outline's point coordinates.    */
/*                                                                       */
/*    tags       :: A pointer to an array of `n_points' chars, giving    */
/*                  each outline point's type.                           */
/*                                                                       */
/*                  If bit~0 is unset, the point is `off' the curve,     */
/*                  i.e., a Bézier control point, while it is `on' if    */
/*                  set.                                                 */
/*                                                                       */
/*                  Bit~1 is meaningful for `off' points only.  If set,  */
/*                  it indicates a third-order Bézier arc control point; */
/*                  and a second-order control point if unset.           */
/*                                                                       */
/*                  If bit~2 is set, bits 5-7 contain the drop-out mode  */
/*                  (as defined in the OpenType specification; the value */
/*                  is the same as the argument to the SCANMODE          */
/*                  instruction).                                        */
/*                                                                       */
/*                  Bits 3 and~4 are reserved for internal purposes.     */
/*                                                                       */
/*    contours   :: An array of `n_contours' shorts, giving the end      */
/*                  point of each contour within the outline.  For       */
/*                  example, the first contour is defined by the points  */
/*                  `0' to `contours[0]', the second one is defined by   */
/*                  the points `contours[0]+1' to `contours[1]', etc.    */
/*                                                                       */
/*    flags      :: A set of bit flags used to characterize the outline  */
/*                  and give hints to the scan-converter and hinter on   */
/*                  how to convert/grid-fit it.  See @SW_FT_OUTLINE_FLAGS.*/
/*                                                                       */
typedef struct  SW_FT_Outline_
{
  short       n_contours;      /* number of contours in glyph        */
  short       n_points;        /* number of points in the glyph      */

  SW_FT_Vector*  points;          /* the outline's points               */
  char*       tags;            /* the points flags                   */
  short*      contours;        /* the contour end points             */
  char*       contours_flag;   /* the contour open flags             */

  int         flags;           /* outline masks                      */

} SW_FT_Outline;


  /*************************************************************************/
  /*                                                                       */
  /* <Enum>                                                                */
  /*    SW_FT_OUTLINE_FLAGS                                                   */
  /*                                                                       */
  /* <Description>                                                         */
  /*    A list of bit-field constants use for the flags in an outline's    */
  /*    `flags' field.                                                     */
  /*                                                                       */
  /* <Values>                                                              */
  /*    SW_FT_OUTLINE_NONE ::                                                 */
  /*      Value~0 is reserved.                                             */
  /*                                                                       */
  /*    SW_FT_OUTLINE_OWNER ::                                                */
  /*      If set, this flag indicates that the outline's field arrays      */
  /*      (i.e., `points', `flags', and `contours') are `owned' by the     */
  /*      outline object, and should thus be freed when it is destroyed.   */
  /*                                                                       */
  /*    SW_FT_OUTLINE_EVEN_ODD_FILL ::                                        */
  /*      By default, outlines are filled using the non-zero winding rule. */
  /*      If set to 1, the outline will be filled using the even-odd fill  */
  /*      rule (only works with the smooth rasterizer).                    */
  /*                                                                       */
  /*    SW_FT_OUTLINE_REVERSE_FILL ::                                         */
  /*      By default, outside contours of an outline are oriented in       */
  /*      clock-wise direction, as defined in the TrueType specification.  */
  /*      This flag is set if the outline uses the opposite direction      */
  /*      (typically for Type~1 fonts).  This flag is ignored by the scan  */
  /*      converter.                                                       */
  /*                                                                       */
  /*                                                                       */
  /*                                                                       */
  /*    There exists a second mechanism to pass the drop-out mode to the   */
  /*    B/W rasterizer; see the `tags' field in @SW_FT_Outline.               */
  /*                                                                       */
  /*    Please refer to the description of the `SCANTYPE' instruction in   */
  /*    the OpenType specification (in file `ttinst1.doc') how simple      */
  /*    drop-outs, smart drop-outs, and stubs are defined.                 */
  /*                                                                       */
#define SW_FT_OUTLINE_NONE             0x0
#define SW_FT_OUTLINE_OWNER            0x1
#define SW_FT_OUTLINE_EVEN_ODD_FILL    0x2
#define SW_FT_OUTLINE_REVERSE_FILL     0x4

  /* */

#define SW_FT_CURVE_TAG( flag )  ( flag & 3 )

#define SW_FT_CURVE_TAG_ON            1
#define SW_FT_CURVE_TAG_CONIC         0
#define SW_FT_CURVE_TAG_CUBIC         2


#define SW_FT_Curve_Tag_On       SW_FT_CURVE_TAG_ON
#define SW_FT_Curve_Tag_Conic    SW_FT_CURVE_TAG_CONIC
#define SW_FT_Curve_Tag_Cubic    SW_FT_CURVE_TAG_CUBIC

  /*************************************************************************/
  /*                                                                       */
  /* A raster is a scan converter, in charge of rendering an outline into  */
  /* a a bitmap.  This section contains the public API for rasters.        */
  /*                                                                       */
  /* Note that in FreeType 2, all rasters are now encapsulated within      */
  /* specific modules called `renderers'.  See `ftrender.h' for more       */
  /* details on renderers.                                                 */
  /*                                                                       */
  /*************************************************************************/


  /*************************************************************************/
  /*                                                                       */
  /* <Type>                                                                */
  /*    SW_FT_Raster                                                          */
  /*                                                                       */
  /* <Description>                                                         */
  /*    A handle (pointer) to a raster object.  Each object can be used    */
  /*    independently to convert an outline into a bitmap or pixmap.       */
  /*                                                                       */
  typedef struct SW_FT_RasterRec_*  SW_FT_Raster;


  /*************************************************************************/
  /*                                                                       */
  /* <Struct>                                                              */
  /*    SW_FT_Span                                                            */
  /*                                                                       */
  /* <Description>                                                         */
  /*    A structure used to model a single span of gray (or black) pixels  */
  /*    when rendering a monochrome or anti-aliased bitmap.                */
  /*                                                                       */
  /* <Fields>                                                              */
  /*    x        :: The span's horizontal start position.                  */
  /*                                                                       */
  /*    len      :: The span's length in pixels.                           */
  /*                                                                       */
  /*    coverage :: The span color/coverage, ranging from 0 (background)   */
  /*                to 255 (foreground).  Only used for anti-aliased       */
  /*                rendering.                                             */
  /*                                                                       */
  /* <Note>                                                                */
  /*    This structure is used by the span drawing callback type named     */
  /*    @SW_FT_SpanFunc that takes the y~coordinate of the span as a          */
  /*    parameter.                                                         */
  /*                                                                       */
  /*    The coverage value is always between 0 and 255.  If you want less  */
  /*    gray values, the callback function has to reduce them.             */
  /*                                                                       */
  typedef struct  SW_FT_Span_
  {
    short           x;
    short           y;
    unsigned short  len;
    unsigned char   coverage;

  } SW_FT_Span;


  /*************************************************************************/
  /*                                                                       */
  /* <FuncType>                                                            */
  /*    SW_FT_SpanFunc                                                        */
  /*                                                                       */
  /* <Description>                                                         */
  /*    A function used as a call-back by the anti-aliased renderer in     */
  /*    order to let client applications draw themselves the gray pixel    */
  /*    spans on each scan line.                                           */
  /*                                                                       */
  /* <Input>                                                               */
  /*    y     :: The scanline's y~coordinate.                              */
  /*                                                                       */
  /*    count :: The number of spans to draw on this scanline.             */
  /*                                                                       */
  /*    spans :: A table of `count' spans to draw on the scanline.         */
  /*                                                                       */
  /*    user  :: User-supplied data that is passed to the callback.        */
  /*                                                                       */
  /* <Note>                                                                */
  /*    This callback allows client applications to directly render the    */
  /*    gray spans of the anti-aliased bitmap to any kind of surfaces.     */
  /*                                                                       */
  /*    This can be used to write anti-aliased outlines directly to a      */
  /*    given background bitmap, and even perform translucency.            */
  /*                                                                       */
  /*    Note that the `count' field cannot be greater than a fixed value   */
  /*    defined by the `SW_FT_MAX_GRAY_SPANS' configuration macro in          */
  /*    `ftoption.h'.  By default, this value is set to~32, which means    */
  /*    that if there are more than 32~spans on a given scanline, the      */
  /*    callback is called several times with the same `y' parameter in    */
  /*    order to draw all callbacks.                                       */
  /*                                                                       */
  /*    Otherwise, the callback is only called once per scan-line, and     */
  /*    only for those scanlines that do have `gray' pixels on them.       */
  /*                                                                       */
  typedef void
  (*SW_FT_SpanFunc)( int             count,
                  const SW_FT_Span*  spans,
                  void*           user );

 typedef void
 (*SW_FT_BboxFunc)( int x, int y, int w, int h,
                    void*  user);

#define SW_FT_Raster_Span_Func  SW_FT_SpanFunc



  /*************************************************************************/
  /*                                                                       */
  /* <Enum>                                                                */
  /*    SW_FT_RASTER_FLAG_XXX                                                 */
  /*                                                                       */
  /* <Description>                                                         */
  /*    A list of bit flag constants as used in the `flags' field of a     */
  /*    @SW_FT_Raster_Params structure.                                       */
  /*                                                                       */
  /* <Values>                                                              */
  /*    SW_FT_RASTER_FLAG_DEFAULT :: This value is 0.                         */
  /*                                                                       */
  /*    SW_FT_RASTER_FLAG_AA      :: This flag is set to indicate that an     */
  /*                              anti-aliased glyph image should be       */
  /*                              generated.  Otherwise, it will be        */
  /*                              monochrome (1-bit).                      */
  /*                                                                       */
  /*    SW_FT_RASTER_FLAG_DIRECT  :: This flag is set to indicate direct      */
  /*                              rendering.  In this mode, client         */
  /*                              applications must provide their own span */
  /*                              callback.  This lets them directly       */
  /*                              draw or compose over an existing bitmap. */
  /*                              If this bit is not set, the target       */
  /*                              pixmap's buffer _must_ be zeroed before  */
  /*                              rendering.                               */
  /*                                                                       */
  /*                              Note that for now, direct rendering is   */
  /*                              only possible with anti-aliased glyphs.  */
  /*                                                                       */
  /*    SW_FT_RASTER_FLAG_CLIP    :: This flag is only used in direct         */
  /*                              rendering mode.  If set, the output will */
  /*                              be clipped to a box specified in the     */
  /*                              `clip_box' field of the                  */
  /*                              @SW_FT_Raster_Params structure.             */
  /*                                                                       */
  /*                              Note that by default, the glyph bitmap   */
  /*                              is clipped to the target pixmap, except  */
  /*                              in direct rendering mode where all spans */
  /*                              are generated if no clipping box is set. */
  /*                                                                       */
#define SW_FT_RASTER_FLAG_DEFAULT  0x0
#define SW_FT_RASTER_FLAG_AA       0x1
#define SW_FT_RASTER_FLAG_DIRECT   0x2
#define SW_FT_RASTER_FLAG_CLIP     0x4


  /*************************************************************************/
  /*                                                                       */
  /* <Struct>                                                              */
  /*    SW_FT_Raster_Params                                                   */
  /*                                                                       */
  /* <Description>                                                         */
  /*    A structure to hold the arguments used by a raster's render        */
  /*    function.                                                          */
  /*                                                                       */
  /* <Fields>                                                              */
  /*    target      :: The target bitmap.                                  */
  /*                                                                       */
  /*    source      :: A pointer to the source glyph image (e.g., an       */
  /*                   @SW_FT_Outline).                                       */
  /*                                                                       */
  /*    flags       :: The rendering flags.                                */
  /*                                                                       */
  /*    gray_spans  :: The gray span drawing callback.                     */
  /*                                                                       */
  /*    black_spans :: The black span drawing callback.  UNIMPLEMENTED!    */
  /*                                                                       */
  /*    bit_test    :: The bit test callback.  UNIMPLEMENTED!              */
  /*                                                                       */
  /*    bit_set     :: The bit set callback.  UNIMPLEMENTED!               */
  /*                                                                       */
  /*    user        :: User-supplied data that is passed to each drawing   */
  /*                   callback.                                           */
  /*                                                                       */
  /*    clip_box    :: An optional clipping box.  It is only used in       */
  /*                   direct rendering mode.  Note that coordinates here  */
  /*                   should be expressed in _integer_ pixels (and not in */
  /*                   26.6 fixed-point units).                            */
  /*                                                                       */
  /* <Note>                                                                */
  /*    An anti-aliased glyph bitmap is drawn if the @SW_FT_RASTER_FLAG_AA    */
  /*    bit flag is set in the `flags' field, otherwise a monochrome       */
  /*    bitmap is generated.                                               */
  /*                                                                       */
  /*    If the @SW_FT_RASTER_FLAG_DIRECT bit flag is set in `flags', the      */
  /*    raster will call the `gray_spans' callback to draw gray pixel      */
  /*    spans, in the case of an aa glyph bitmap, it will call             */
  /*    `black_spans', and `bit_test' and `bit_set' in the case of a       */
  /*    monochrome bitmap.  This allows direct composition over a          */
  /*    pre-existing bitmap through user-provided callbacks to perform the */
  /*    span drawing/composition.                                          */
  /*                                                                       */
  /*    Note that the `bit_test' and `bit_set' callbacks are required when */
  /*    rendering a monochrome bitmap, as they are crucial to implement    */
  /*    correct drop-out control as defined in the TrueType specification. */
  /*                                                                       */
  typedef struct  SW_FT_Raster_Params_
  {
    const void*             source;
    int                     flags;
    SW_FT_SpanFunc          gray_spans;
    SW_FT_BboxFunc          bbox_cb;
    void*                   user;
    SW_FT_BBox              clip_box;

  } SW_FT_Raster_Params;


/*************************************************************************/
/*                                                                       */
/* <Function>                                                            */
/*    SW_FT_Outline_Check                                                   */
/*                                                                       */
/* <Description>                                                         */
/*    Check the contents of an outline descriptor.                       */
/*                                                                       */
/* <Input>                                                               */
/*    outline :: A handle to a source outline.                           */
/*                                                                       */
/* <Return>                                                              */
/*    FreeType error code.  0~means success.                             */
/*                                                                       */
SW_FT_Error
SW_FT_Outline_Check( SW_FT_Outline*  outline );


/*************************************************************************/
/*                                                                       */
/* <Function>                                                            */
/*    SW_FT_Outline_Get_CBox                                                */
/*                                                                       */
/* <Description>                                                         */
/*    Return an outline's `control box'.  The control box encloses all   */
/*    the outline's points, including Bézier control points.  Though it  */
/*    coincides with the exact bounding box for most glyphs, it can be   */
/*    slightly larger in some situations (like when rotating an outline  */
/*    that contains Bézier outside arcs).                                */
/*                                                                       */
/*    Computing the control box is very fast, while getting the bounding */
/*    box can take much more time as it needs to walk over all segments  */
/*    and arcs in the outline.  To get the latter, you can use the       */
/*    `ftbbox' component, which is dedicated to this single task.        */
/*                                                                       */
/* <Input>                                                               */
/*    outline :: A pointer to the source outline descriptor.             */
/*                                                                       */
/* <Output>                                                              */
/*    acbox   :: The outline's control box.                              */
/*                                                                       */
/* <Note>                                                                */
/*    See @SW_FT_Glyph_Get_CBox for a discussion of tricky fonts.           */
/*                                                                       */
void
SW_FT_Outline_Get_CBox( const SW_FT_Outline*  outline,
                     SW_FT_BBox           *acbox );


  /*************************************************************************/
  /*                                                                       */
  /* <FuncType>                                                            */
  /*    SW_FT_Raster_NewFunc                                                  */
  /*                                                                       */
  /* <Description>                                                         */
  /*    A function used to create a new raster object.                     */
  /*                                                                       */
  /* <Input>                                                               */
  /*    memory :: A handle to the memory allocator.                        */
  /*                                                                       */
  /* <Output>                                                              */
  /*    raster :: A handle to the new raster object.                       */
  /*                                                                       */
  /* <Return>                                                              */
  /*    Error code.  0~means success.                                      */
  /*                                                                       */
  /* <Note>                                                                */
  /*    The `memory' parameter is a typeless pointer in order to avoid     */
  /*    un-wanted dependencies on the rest of the FreeType code.  In       */
  /*    practice, it is an @SW_FT_Memory object, i.e., a handle to the        */
  /*    standard FreeType memory allocator.  However, this field can be    */
  /*    completely ignored by a given raster implementation.               */
  /*                                                                       */
  typedef int
  (*SW_FT_Raster_NewFunc)( SW_FT_Raster*  raster );

#define SW_FT_Raster_New_Func  SW_FT_Raster_NewFunc


  /*************************************************************************/
  /*                                                                       */
  /* <FuncType>                                                            */
  /*    SW_FT_Raster_DoneFunc                                                 */
  /*                                                                       */
  /* <Description>                                                         */
  /*    A function used to destroy a given raster object.                  */
  /*                                                                       */
  /* <Input>                                                               */
  /*    raster :: A handle to the raster object.                           */
  /*                                                                       */
  typedef void
  (*SW_FT_Raster_DoneFunc)( SW_FT_Raster  raster );

#define SW_FT_Raster_Done_Func  SW_FT_Raster_DoneFunc


  /*************************************************************************/
  /*                                                                       */
  /* <FuncType>                                                            */
  /*    SW_FT_Raster_ResetFunc                                                */
  /*                                                                       */
  /* <Description>                                                         */
  /*    FreeType provides an area of memory called the `render pool',      */
  /*    available to all registered rasters.  This pool can be freely used */
  /*    during a given scan-conversion but is shared by all rasters.  Its  */
  /*    content is thus transient.                                         */
  /*                                                                       */
  /*    This function is called each time the render pool changes, or just */
  /*    after a new raster object is created.                              */
  /*                                                                       */
  /* <Input>                                                               */
  /*    raster    :: A handle to the new raster object.                    */
  /*                                                                       */
  /*    pool_base :: The address in memory of the render pool.             */
  /*                                                                       */
  /*    pool_size :: The size in bytes of the render pool.                 */
  /*                                                                       */
  /* <Note>                                                                */
  /*    Rasters can ignore the render pool and rely on dynamic memory      */
  /*    allocation if they want to (a handle to the memory allocator is    */
  /*    passed to the raster constructor).  However, this is not           */
  /*    recommended for efficiency purposes.                               */
  /*                                                                       */
  typedef void
  (*SW_FT_Raster_ResetFunc)( SW_FT_Raster       raster,
                          unsigned char*  pool_base,
                          unsigned long   pool_size );

#define SW_FT_Raster_Reset_Func  SW_FT_Raster_ResetFunc


  /*************************************************************************/
  /*                                                                       */
  /* <FuncType>                                                            */
  /*    SW_FT_Raster_RenderFunc                                               */
  /*                                                                       */
  /* <Description>                                                         */
  /*    Invoke a given raster to scan-convert a given glyph image into a   */
  /*    target bitmap.                                                     */
  /*                                                                       */
  /* <Input>                                                               */
  /*    raster :: A handle to the raster object.                           */
  /*                                                                       */
  /*    params :: A pointer to an @SW_FT_Raster_Params structure used to      */
  /*              store the rendering parameters.                          */
  /*                                                                       */
  /* <Return>                                                              */
  /*    Error code.  0~means success.                                      */
  /*                                                                       */
  /* <Note>                                                                */
  /*    The exact format of the source image depends on the raster's glyph */
  /*    format defined in its @SW_FT_Raster_Funcs structure.  It can be an    */
  /*    @SW_FT_Outline or anything else in order to support a large array of  */
  /*    glyph formats.                                                     */
  /*                                                                       */
  /*    Note also that the render function can fail and return a           */
  /*    `SW_FT_Err_Unimplemented_Feature' error code if the raster used does  */
  /*    not support direct composition.                                    */
  /*                                                                       */
  /*    XXX: For now, the standard raster doesn't support direct           */
  /*         composition but this should change for the final release (see */
  /*         the files `demos/src/ftgrays.c' and `demos/src/ftgrays2.c'    */
  /*         for examples of distinct implementations that support direct  */
  /*         composition).                                                 */
  /*                                                                       */
  typedef int
  (*SW_FT_Raster_RenderFunc)( SW_FT_Raster                raster,
                           const SW_FT_Raster_Params*  params );

#define SW_FT_Raster_Render_Func  SW_FT_Raster_RenderFunc


  /*************************************************************************/
  /*                                                                       */
  /* <Struct>                                                              */
  /*    SW_FT_Raster_Funcs                                                    */
  /*                                                                       */
  /* <Description>                                                         */
  /*   A structure used to describe a given raster class to the library.   */
  /*                                                                       */
  /* <Fields>                                                              */
  /*    glyph_format  :: The supported glyph format for this raster.       */
  /*                                                                       */
  /*    raster_new    :: The raster constructor.                           */
  /*                                                                       */
  /*    raster_reset  :: Used to reset the render pool within the raster.  */
  /*                                                                       */
  /*    raster_render :: A function to render a glyph into a given bitmap. */
  /*                                                                       */
  /*    raster_done   :: The raster destructor.                            */
  /*                                                                       */
  typedef struct  SW_FT_Raster_Funcs_
  {
    SW_FT_Raster_NewFunc      raster_new;
    SW_FT_Raster_ResetFunc    raster_reset;
    SW_FT_Raster_RenderFunc   raster_render;
    SW_FT_Raster_DoneFunc     raster_done;

  } SW_FT_Raster_Funcs;


extern const SW_FT_Raster_Funcs   sw_ft_grays_raster;

#endif // SW_FT_IMG_H
