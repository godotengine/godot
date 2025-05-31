/*
 * Copyright © 2022 Matthias Clasen
 *
 *  This is part of HarfBuzz, a text shaping library.
 *
 * Permission is hereby granted, without written agreement and without
 * license or royalty fees, to use, copy, modify, and distribute this
 * software and its documentation for any purpose, provided that the
 * above copyright notice and the following two paragraphs appear in
 * all copies of this software.
 *
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE TO ANY PARTY FOR
 * DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES
 * ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN
 * IF THE COPYRIGHT HOLDER HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH
 * DAMAGE.
 *
 * THE COPYRIGHT HOLDER SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING,
 * BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE.  THE SOFTWARE PROVIDED HEREUNDER IS
 * ON AN "AS IS" BASIS, AND THE COPYRIGHT HOLDER HAS NO OBLIGATION TO
 * PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
 */

#if !defined(HB_H_IN) && !defined(HB_NO_SINGLE_HEADER_ERROR)
#error "Include <hb.h> instead."
#endif

#ifndef HB_PAINT_H
#define HB_PAINT_H

#include "hb-common.h"

HB_BEGIN_DECLS


/**
 * hb_paint_funcs_t:
 *
 * Glyph paint callbacks.
 *
 * The callbacks assume that the caller maintains a stack
 * of current transforms, clips and intermediate surfaces,
 * as evidenced by the pairs of push/pop callbacks. The
 * push/pop calls will be properly nested, so it is fine
 * to store the different kinds of object on a single stack.
 *
 * Not all callbacks are required for all kinds of glyphs.
 * For rendering COLRv0 or non-color outline glyphs, the
 * gradient callbacks are not needed, and the composite
 * callback only needs to handle simple alpha compositing
 * (#HB_PAINT_COMPOSITE_MODE_SRC_OVER).
 *
 * The paint-image callback is only needed for glyphs
 * with image blobs in the CBDT, sbix or SVG tables.
 *
 * The custom-palette-color callback is only necessary if
 * you want to override colors from the font palette with
 * custom colors.
 *
 * Since: 7.0.0
 **/
typedef struct hb_paint_funcs_t hb_paint_funcs_t;

HB_EXTERN hb_paint_funcs_t *
hb_paint_funcs_create (void);

HB_EXTERN hb_paint_funcs_t *
hb_paint_funcs_get_empty (void);

HB_EXTERN hb_paint_funcs_t *
hb_paint_funcs_reference (hb_paint_funcs_t *funcs);

HB_EXTERN void
hb_paint_funcs_destroy (hb_paint_funcs_t *funcs);

HB_EXTERN hb_bool_t
hb_paint_funcs_set_user_data (hb_paint_funcs_t *funcs,
			      hb_user_data_key_t *key,
			      void *              data,
			      hb_destroy_func_t   destroy,
			      hb_bool_t           replace);


HB_EXTERN void *
hb_paint_funcs_get_user_data (const hb_paint_funcs_t *funcs,
			      hb_user_data_key_t       *key);

HB_EXTERN void
hb_paint_funcs_make_immutable (hb_paint_funcs_t *funcs);

HB_EXTERN hb_bool_t
hb_paint_funcs_is_immutable (hb_paint_funcs_t *funcs);

/**
 * hb_paint_push_transform_func_t:
 * @funcs: paint functions object
 * @paint_data: The data accompanying the paint functions in hb_font_paint_glyph()
 * @xx: xx component of the transform matrix
 * @yx: yx component of the transform matrix
 * @xy: xy component of the transform matrix
 * @yy: yy component of the transform matrix
 * @dx: dx component of the transform matrix
 * @dy: dy component of the transform matrix
 * @user_data: User data pointer passed to hb_paint_funcs_set_push_transform_func()
 *
 * A virtual method for the #hb_paint_funcs_t to apply
 * a transform to subsequent paint calls.
 *
 * This transform is applied after the current transform,
 * and remains in effect until a matching call to
 * the #hb_paint_funcs_pop_transform_func_t vfunc.
 *
 * Since: 7.0.0
 */
typedef void (*hb_paint_push_transform_func_t) (hb_paint_funcs_t *funcs,
                                                void *paint_data,
                                                float xx, float yx,
                                                float xy, float yy,
                                                float dx, float dy,
                                                void *user_data);

/**
 * hb_paint_pop_transform_func_t:
 * @funcs: paint functions object
 * @paint_data: The data accompanying the paint functions in hb_font_paint_glyph()
 * @user_data: User data pointer passed to hb_paint_funcs_set_pop_transform_func()
 *
 * A virtual method for the #hb_paint_funcs_t to undo
 * the effect of a prior call to the #hb_paint_funcs_push_transform_func_t
 * vfunc.
 *
 * Since: 7.0.0
 */
typedef void (*hb_paint_pop_transform_func_t) (hb_paint_funcs_t *funcs,
                                               void *paint_data,
                                               void *user_data);

/**
 * hb_paint_color_glyph_func_t:
 * @funcs: paint functions object
 * @paint_data: The data accompanying the paint functions in hb_font_paint_glyph()
 * @glyph: the glyph ID
 * @font: the font
 * @user_data: User data pointer passed to hb_paint_funcs_set_color_glyph_func()
 *
 * A virtual method for the #hb_paint_funcs_t to render a color glyph by glyph index.
 *
 * Return value: `true` if the glyph was painted, `false` otherwise.
 *
 * Since: 8.2.0
 */
typedef hb_bool_t (*hb_paint_color_glyph_func_t) (hb_paint_funcs_t *funcs,
                                                  void *paint_data,
                                                  hb_codepoint_t glyph,
                                                  hb_font_t *font,
                                                  void *user_data);

/**
 * hb_paint_push_clip_glyph_func_t:
 * @funcs: paint functions object
 * @paint_data: The data accompanying the paint functions in hb_font_paint_glyph()
 * @glyph: the glyph ID
 * @font: the font
 * @user_data: User data pointer passed to hb_paint_funcs_set_push_clip_glyph_func()
 *
 * A virtual method for the #hb_paint_funcs_t to clip
 * subsequent paint calls to the outline of a glyph.
 *
 * The coordinates of the glyph outline are interpreted according
 * to the current transform.
 *
 * This clip is applied in addition to the current clip,
 * and remains in effect until a matching call to
 * the #hb_paint_funcs_pop_clip_func_t vfunc.
 *
 * Since: 7.0.0
 */
typedef void (*hb_paint_push_clip_glyph_func_t) (hb_paint_funcs_t *funcs,
                                                 void *paint_data,
                                                 hb_codepoint_t glyph,
                                                 hb_font_t *font,
                                                 void *user_data);

/**
 * hb_paint_push_clip_rectangle_func_t:
 * @funcs: paint functions object
 * @paint_data: The data accompanying the paint functions in hb_font_paint_glyph()
 * @xmin: min X for the rectangle
 * @ymin: min Y for the rectangle
 * @xmax: max X for the rectangle
 * @ymax: max Y for the rectangle
 * @user_data: User data pointer passed to hb_paint_funcs_set_push_clip_rectangle_func()
 *
 * A virtual method for the #hb_paint_funcs_t to clip
 * subsequent paint calls to a rectangle.
 *
 * The coordinates of the rectangle are interpreted according
 * to the current transform.
 *
 * This clip is applied in addition to the current clip,
 * and remains in effect until a matching call to
 * the #hb_paint_funcs_pop_clip_func_t vfunc.
 *
 * Since: 7.0.0
 */
typedef void (*hb_paint_push_clip_rectangle_func_t) (hb_paint_funcs_t *funcs,
                                                     void *paint_data,
                                                     float xmin, float ymin,
                                                     float xmax, float ymax,
                                                     void *user_data);

/**
 * hb_paint_pop_clip_func_t:
 * @funcs: paint functions object
 * @paint_data: The data accompanying the paint functions in hb_font_paint_glyph()
 * @user_data: User data pointer passed to hb_paint_funcs_set_pop_clip_func()
 *
 * A virtual method for the #hb_paint_funcs_t to undo
 * the effect of a prior call to the #hb_paint_funcs_push_clip_glyph_func_t
 * or #hb_paint_funcs_push_clip_rectangle_func_t vfuncs.
 *
 * Since: 7.0.0
 */
typedef void (*hb_paint_pop_clip_func_t) (hb_paint_funcs_t *funcs,
                                          void *paint_data,
                                          void *user_data);

/**
 * hb_paint_color_func_t:
 * @funcs: paint functions object
 * @paint_data: The data accompanying the paint functions in hb_font_paint_glyph()
 * @is_foreground: whether the color is the foreground
 * @color: The color to use, unpremultiplied
 * @user_data: User data pointer passed to hb_paint_funcs_set_color_func()
 *
 * A virtual method for the #hb_paint_funcs_t to paint a
 * color everywhere within the current clip.
 *
 * Since: 7.0.0
 */
typedef void (*hb_paint_color_func_t) (hb_paint_funcs_t *funcs,
                                       void *paint_data,
                                       hb_bool_t is_foreground,
                                       hb_color_t color,
                                       void *user_data);

/**
 * HB_PAINT_IMAGE_FORMAT_PNG:
 *
 * Tag identifying PNG images in #hb_paint_image_func_t callbacks.
 *
 * Since: 7.0.0
 */
#define HB_PAINT_IMAGE_FORMAT_PNG HB_TAG('p','n','g',' ')

/**
 * HB_PAINT_IMAGE_FORMAT_SVG:
 *
 * Tag identifying SVG images in #hb_paint_image_func_t callbacks.
 *
 * Since: 7.0.0
 */
#define HB_PAINT_IMAGE_FORMAT_SVG HB_TAG('s','v','g',' ')

/**
 * HB_PAINT_IMAGE_FORMAT_BGRA:
 *
 * Tag identifying raw pixel-data images in #hb_paint_image_func_t callbacks.
 * The data is in BGRA pre-multiplied sRGBA color-space format.
 *
 * Since: 7.0.0
 */
#define HB_PAINT_IMAGE_FORMAT_BGRA HB_TAG('B','G','R','A')

/**
 * hb_paint_image_func_t:
 * @funcs: paint functions object
 * @paint_data: The data accompanying the paint functions in hb_font_paint_glyph()
 * @image: the image data
 * @width: width of the raster image in pixels, or 0
 * @height: height of the raster image in pixels, or 0
 * @format: the image format as a tag
 * @slant: the synthetic slant ratio to be applied to the image during rendering
 * @extents: (nullable): glyph extents for desired rendering
 * @user_data: User data pointer passed to hb_paint_funcs_set_image_func()
 *
 * A virtual method for the #hb_paint_funcs_t to paint a glyph image.
 *
 * This method is called for glyphs with image blobs in the CBDT,
 * sbix or SVG tables. The @format identifies the kind of data that
 * is contained in @image. Possible values include #HB_PAINT_IMAGE_FORMAT_PNG,
 * #HB_PAINT_IMAGE_FORMAT_SVG and #HB_PAINT_IMAGE_FORMAT_BGRA.
 *
 * The image dimensions and glyph extents are provided if available,
 * and should be used to size and position the image.
 *
 * Return value: Whether the operation was successful.
 *
 * Since: 7.0.0
 */
typedef hb_bool_t (*hb_paint_image_func_t) (hb_paint_funcs_t *funcs,
					    void *paint_data,
					    hb_blob_t *image,
					    unsigned int width,
					    unsigned int height,
					    hb_tag_t format,
					    float slant,
					    hb_glyph_extents_t *extents,
					    void *user_data);

/**
 * hb_color_stop_t:
 * @offset: the offset of the color stop
 * @is_foreground: whether the color is the foreground
 * @color: the color, unpremultiplied
 *
 * Information about a color stop on a color line.
 *
 * Color lines typically have offsets ranging between 0 and 1,
 * but that is not required.
 *
 * Note: despite @color being unpremultiplied here, interpolation in
 * gradients shall happen in premultiplied space. See the OpenType spec
 * [COLR](https://learn.microsoft.com/en-us/typography/opentype/spec/colr)
 * section for details.
 *
 * Since: 7.0.0
 */
typedef struct {
  float offset;
  hb_bool_t is_foreground;
  hb_color_t color;
} hb_color_stop_t;

/**
 * hb_paint_extend_t:
 * @HB_PAINT_EXTEND_PAD: Outside the defined interval,
 *   the color of the closest color stop is used.
 * @HB_PAINT_EXTEND_REPEAT: The color line is repeated over
 *   repeated multiples of the defined interval
 * @HB_PAINT_EXTEND_REFLECT: The color line is repeated over
 *      repeated intervals, as for the repeat mode.
 *      However, in each repeated interval, the ordering of
 *      color stops is the reverse of the adjacent interval.
 *
 * The values of this enumeration determine how color values
 * outside the minimum and maximum defined offset on a #hb_color_line_t
 * are determined.
 *
 * See the OpenType spec [COLR](https://learn.microsoft.com/en-us/typography/opentype/spec/colr)
 * section for details.
 *
 * Since: 7.0.0
 */
typedef enum {
  HB_PAINT_EXTEND_PAD,
  HB_PAINT_EXTEND_REPEAT,
  HB_PAINT_EXTEND_REFLECT
} hb_paint_extend_t;

typedef struct hb_color_line_t hb_color_line_t;

/**
 * hb_color_line_get_color_stops_func_t:
 * @color_line: a #hb_color_line_t object
 * @color_line_data: the data accompanying @color_line
 * @start: the index of the first color stop to return
 * @count: (inout) (optional): Input = the maximum number of feature tags to return;
 *     Output = the actual number of feature tags returned (may be zero)
 * @color_stops: (out) (array length=count) (optional): Array of #hb_color_stop_t to populate
 * @user_data: the data accompanying this method
 *
 * A virtual method for the #hb_color_line_t to fetch color stops.
 *
 * Return value: the total number of color stops in @color_line
 *
 * Since: 7.0.0
 */
typedef unsigned int (*hb_color_line_get_color_stops_func_t) (hb_color_line_t *color_line,
							      void *color_line_data,
							      unsigned int start,
							      unsigned int *count,
							      hb_color_stop_t *color_stops,
							      void *user_data);

/**
 * hb_color_line_get_extend_func_t:
 * @color_line: a #hb_color_line_t object
 * @color_line_data: the data accompanying @color_line
 * @user_data: the data accompanying this method
 *
 * A virtual method for the @hb_color_line_t to fetches the extend mode.
 *
 * Return value: the extend mode of @color_line
 *
 * Since: 7.0.0
 */
typedef hb_paint_extend_t (*hb_color_line_get_extend_func_t) (hb_color_line_t *color_line,
							      void *color_line_data,
							      void *user_data);

/**
 * hb_color_line_t:
 *
 * A struct containing color information for a gradient.
 *
 * Since: 7.0.0
 */
struct hb_color_line_t {
  void *data;

  hb_color_line_get_color_stops_func_t get_color_stops;
  void *get_color_stops_user_data;

  hb_color_line_get_extend_func_t get_extend;
  void *get_extend_user_data;

  void *reserved0;
  void *reserved1;
  void *reserved2;
  void *reserved3;
  void *reserved5;
  void *reserved6;
  void *reserved7;
  void *reserved8;
};

HB_EXTERN unsigned int
hb_color_line_get_color_stops (hb_color_line_t *color_line,
                               unsigned int start,
                               unsigned int *count,
                               hb_color_stop_t *color_stops);

HB_EXTERN hb_paint_extend_t
hb_color_line_get_extend (hb_color_line_t *color_line);

/**
 * hb_paint_linear_gradient_func_t:
 * @funcs: paint functions object
 * @paint_data: The data accompanying the paint functions in hb_font_paint_glyph()
 * @color_line: Color information for the gradient
 * @x0: X coordinate of the first point
 * @y0: Y coordinate of the first point
 * @x1: X coordinate of the second point
 * @y1: Y coordinate of the second point
 * @x2: X coordinate of the third point
 * @y2: Y coordinate of the third point
 * @user_data: User data pointer passed to hb_paint_funcs_set_linear_gradient_func()
 *
 * A virtual method for the #hb_paint_funcs_t to paint a linear
 * gradient everywhere within the current clip.
 *
 * The @color_line object contains information about the colors of the gradients.
 * It is only valid for the duration of the callback, you cannot keep it around.
 *
 * The coordinates of the points are interpreted according
 * to the current transform.
 *
 * See the OpenType spec [COLR](https://learn.microsoft.com/en-us/typography/opentype/spec/colr)
 * section for details on how the points define the direction
 * of the gradient, and how to interpret the @color_line.
 *
 * Since: 7.0.0
 */
typedef void (*hb_paint_linear_gradient_func_t) (hb_paint_funcs_t *funcs,
                                                 void *paint_data,
                                                 hb_color_line_t *color_line,
                                                 float x0, float y0,
                                                 float x1, float y1,
                                                 float x2, float y2,
                                                 void *user_data);

/**
 * hb_paint_radial_gradient_func_t:
 * @funcs: paint functions object
 * @paint_data: The data accompanying the paint functions in hb_font_paint_glyph()
 * @color_line: Color information for the gradient
 * @x0: X coordinate of the first circle's center
 * @y0: Y coordinate of the first circle's center
 * @r0: radius of the first circle
 * @x1: X coordinate of the second circle's center
 * @y1: Y coordinate of the second circle's center
 * @r1: radius of the second circle
 * @user_data: User data pointer passed to hb_paint_funcs_set_radial_gradient_func()
 *
 * A virtual method for the #hb_paint_funcs_t to paint a radial
 * gradient everywhere within the current clip.
 *
 * The @color_line object contains information about the colors of the gradients.
 * It is only valid for the duration of the callback, you cannot keep it around.
 *
 * The coordinates of the points are interpreted according
 * to the current transform.
 *
 * See the OpenType spec [COLR](https://learn.microsoft.com/en-us/typography/opentype/spec/colr)
 * section for details on how the points define the direction
 * of the gradient, and how to interpret the @color_line.
 *
 * Since: 7.0.0
 */
typedef void (*hb_paint_radial_gradient_func_t) (hb_paint_funcs_t *funcs,
                                                 void *paint_data,
                                                 hb_color_line_t *color_line,
                                                 float x0, float y0, float r0,
                                                 float x1, float y1, float r1,
                                                 void *user_data);

/**
 * hb_paint_sweep_gradient_func_t:
 * @funcs: paint functions object
 * @paint_data: The data accompanying the paint functions in hb_font_paint_glyph()
 * @color_line: Color information for the gradient
 * @x0: X coordinate of the circle's center
 * @y0: Y coordinate of the circle's center
 * @start_angle: the start angle, in radians
 * @end_angle: the end angle, in radians
 * @user_data: User data pointer passed to hb_paint_funcs_set_sweep_gradient_func()
 *
 * A virtual method for the #hb_paint_funcs_t to paint a sweep
 * gradient everywhere within the current clip.
 *
 * The @color_line object contains information about the colors of the gradients.
 * It is only valid for the duration of the callback, you cannot keep it around.
 *
 * The coordinates of the points are interpreted according
 * to the current transform.
 *
 * See the OpenType spec [COLR](https://learn.microsoft.com/en-us/typography/opentype/spec/colr)
 * section for details on how the points define the direction
 * of the gradient, and how to interpret the @color_line.
 *
 * Since: 7.0.0
 */
typedef void (*hb_paint_sweep_gradient_func_t)  (hb_paint_funcs_t *funcs,
                                                 void *paint_data,
                                                 hb_color_line_t *color_line,
                                                 float x0, float y0,
                                                 float start_angle,
                                                 float end_angle,
                                                 void *user_data);

/**
 * hb_paint_composite_mode_t:
 * @HB_PAINT_COMPOSITE_MODE_CLEAR: clear destination layer (bounded)
 * @HB_PAINT_COMPOSITE_MODE_SRC: replace destination layer (bounded)
 * @HB_PAINT_COMPOSITE_MODE_SRC_OVER: draw source layer on top of destination layer
 * (bounded)
 * @HB_PAINT_COMPOSITE_MODE_SRC_IN: draw source where there was destination content
 * (unbounded)
 * @HB_PAINT_COMPOSITE_MODE_SRC_OUT: draw source where there was no destination
 * content (unbounded)
 * @HB_PAINT_COMPOSITE_MODE_SRC_ATOP: draw source on top of destination content and
 * only there
 * @HB_PAINT_COMPOSITE_MODE_DEST: ignore the source
 * @HB_PAINT_COMPOSITE_MODE_DEST_OVER: draw destination on top of source
 * @HB_PAINT_COMPOSITE_MODE_DEST_IN: leave destination only where there was
 * source content (unbounded)
 * @HB_PAINT_COMPOSITE_MODE_DEST_OUT: leave destination only where there was no
 * source content
 * @HB_PAINT_COMPOSITE_MODE_DEST_ATOP: leave destination on top of source content
 * and only there (unbounded)
 * @HB_PAINT_COMPOSITE_MODE_XOR: source and destination are shown where there is only
 * one of them
 * @HB_PAINT_COMPOSITE_MODE_PLUS: source and destination layers are accumulated
 * @HB_PAINT_COMPOSITE_MODE_MULTIPLY: source and destination layers are multiplied.
 * This causes the result to be at least as dark as the darker inputs.
 * @HB_PAINT_COMPOSITE_MODE_SCREEN: source and destination are complemented and
 * multiplied. This causes the result to be at least as light as the lighter
 * inputs.
 * @HB_PAINT_COMPOSITE_MODE_OVERLAY: multiplies or screens, depending on the
 * lightness of the destination color.
 * @HB_PAINT_COMPOSITE_MODE_DARKEN: replaces the destination with the source if it
 * is darker, otherwise keeps the source.
 * @HB_PAINT_COMPOSITE_MODE_LIGHTEN: replaces the destination with the source if it
 * is lighter, otherwise keeps the source.
 * @HB_PAINT_COMPOSITE_MODE_COLOR_DODGE: brightens the destination color to reflect
 * the source color.
 * @HB_PAINT_COMPOSITE_MODE_COLOR_BURN: darkens the destination color to reflect
 * the source color.
 * @HB_PAINT_COMPOSITE_MODE_HARD_LIGHT: Multiplies or screens, dependent on source
 * color.
 * @HB_PAINT_COMPOSITE_MODE_SOFT_LIGHT: Darkens or lightens, dependent on source
 * color.
 * @HB_PAINT_COMPOSITE_MODE_DIFFERENCE: Takes the difference of the source and
 * destination color.
 * @HB_PAINT_COMPOSITE_MODE_EXCLUSION: Produces an effect similar to difference, but
 * with lower contrast.
 * @HB_PAINT_COMPOSITE_MODE_HSL_HUE: Creates a color with the hue of the source
 * and the saturation and luminosity of the target.
 * @HB_PAINT_COMPOSITE_MODE_HSL_SATURATION: Creates a color with the saturation
 * of the source and the hue and luminosity of the target. Painting with
 * this mode onto a gray area produces no change.
 * @HB_PAINT_COMPOSITE_MODE_HSL_COLOR: Creates a color with the hue and saturation
 * of the source and the luminosity of the target. This preserves the gray
 * levels of the target and is useful for coloring monochrome images or
 * tinting color images.
 * @HB_PAINT_COMPOSITE_MODE_HSL_LUMINOSITY: Creates a color with the luminosity of
 * the source and the hue and saturation of the target. This produces an
 * inverse effect to @HB_PAINT_COMPOSITE_MODE_HSL_COLOR.
 *
 * The values of this enumeration describe the compositing modes
 * that can be used when combining temporary redirected drawing
 * with the backdrop.
 *
 * See the OpenType spec [COLR](https://learn.microsoft.com/en-us/typography/opentype/spec/colr)
 * section for details.
 *
 * Since: 7.0.0
 */
typedef enum {
  HB_PAINT_COMPOSITE_MODE_CLEAR,
  HB_PAINT_COMPOSITE_MODE_SRC,
  HB_PAINT_COMPOSITE_MODE_DEST,
  HB_PAINT_COMPOSITE_MODE_SRC_OVER,
  HB_PAINT_COMPOSITE_MODE_DEST_OVER,
  HB_PAINT_COMPOSITE_MODE_SRC_IN,
  HB_PAINT_COMPOSITE_MODE_DEST_IN,
  HB_PAINT_COMPOSITE_MODE_SRC_OUT,
  HB_PAINT_COMPOSITE_MODE_DEST_OUT,
  HB_PAINT_COMPOSITE_MODE_SRC_ATOP,
  HB_PAINT_COMPOSITE_MODE_DEST_ATOP,
  HB_PAINT_COMPOSITE_MODE_XOR,
  HB_PAINT_COMPOSITE_MODE_PLUS,
  HB_PAINT_COMPOSITE_MODE_SCREEN,
  HB_PAINT_COMPOSITE_MODE_OVERLAY,
  HB_PAINT_COMPOSITE_MODE_DARKEN,
  HB_PAINT_COMPOSITE_MODE_LIGHTEN,
  HB_PAINT_COMPOSITE_MODE_COLOR_DODGE,
  HB_PAINT_COMPOSITE_MODE_COLOR_BURN,
  HB_PAINT_COMPOSITE_MODE_HARD_LIGHT,
  HB_PAINT_COMPOSITE_MODE_SOFT_LIGHT,
  HB_PAINT_COMPOSITE_MODE_DIFFERENCE,
  HB_PAINT_COMPOSITE_MODE_EXCLUSION,
  HB_PAINT_COMPOSITE_MODE_MULTIPLY,
  HB_PAINT_COMPOSITE_MODE_HSL_HUE,
  HB_PAINT_COMPOSITE_MODE_HSL_SATURATION,
  HB_PAINT_COMPOSITE_MODE_HSL_COLOR,
  HB_PAINT_COMPOSITE_MODE_HSL_LUMINOSITY
} hb_paint_composite_mode_t;

/**
 * hb_paint_push_group_func_t:
 * @funcs: paint functions object
 * @paint_data: The data accompanying the paint functions in hb_font_paint_glyph()
 * @user_data: User data pointer passed to hb_paint_funcs_set_push_group_func()
 *
 * A virtual method for the #hb_paint_funcs_t to use
 * an intermediate surface for subsequent paint calls.
 *
 * The drawing will be redirected to an intermediate surface
 * until a matching call to the #hb_paint_funcs_pop_group_func_t
 * vfunc.
 *
 * Since: 7.0.0
 */
typedef void (*hb_paint_push_group_func_t) (hb_paint_funcs_t *funcs,
                                            void *paint_data,
                                            void *user_data);

/**
 * hb_paint_pop_group_func_t:
 * @funcs: paint functions object
 * @paint_data: The data accompanying the paint functions in hb_font_paint_glyph()
 * @mode: the compositing mode to use
 * @user_data: User data pointer passed to hb_paint_funcs_set_pop_group_func()
 *
 * A virtual method for the #hb_paint_funcs_t to undo
 * the effect of a prior call to the #hb_paint_funcs_push_group_func_t
 * vfunc.
 *
 * This call stops the redirection to the intermediate surface,
 * and then composites it on the previous surface, using the
 * compositing mode passed to this call.
 *
 * Since: 7.0.0
 */
typedef void (*hb_paint_pop_group_func_t) (hb_paint_funcs_t *funcs,
                                           void *paint_data,
                                           hb_paint_composite_mode_t mode,
                                           void *user_data);

/**
 * hb_paint_custom_palette_color_func_t:
 * @funcs: paint functions object
 * @paint_data: The data accompanying the paint functions in hb_font_paint_glyph()
 * @color_index: the color index
 * @color: (out): fetched color
 * @user_data: User data pointer passed to hb_paint_funcs_set_pop_group_func()
 *
 * A virtual method for the #hb_paint_funcs_t to fetch a color from the custom
 * color palette.
 *
 * Custom palette colors override the colors from the fonts selected color
 * palette. It is not necessary to override all palette entries; for entries
 * that should be taken from the font palette, return `false`.
 *
 * This function might get called multiple times, but the custom palette is
 * expected to remain unchanged for duration of a hb_font_paint_glyph() call.
 *
 * Return value: `true` if found, `false` otherwise
 *
 * Since: 7.0.0
 */
typedef hb_bool_t (*hb_paint_custom_palette_color_func_t) (hb_paint_funcs_t *funcs,
                                                           void *paint_data,
                                                           unsigned int color_index,
                                                           hb_color_t *color,
                                                           void *user_data);


/**
 * hb_paint_funcs_set_push_transform_func:
 * @funcs: A paint functions struct
 * @func: (closure user_data) (destroy destroy) (scope notified): The push-transform callback
 * @user_data: Data to pass to @func
 * @destroy: (nullable): Function to call when @user_data is no longer needed
 *
 * Sets the push-transform callback on the paint functions struct.
 *
 * Since: 7.0.0
 */
HB_EXTERN void
hb_paint_funcs_set_push_transform_func (hb_paint_funcs_t               *funcs,
                                        hb_paint_push_transform_func_t  func,
                                        void                           *user_data,
                                        hb_destroy_func_t               destroy);

/**
 * hb_paint_funcs_set_pop_transform_func:
 * @funcs: A paint functions struct
 * @func: (closure user_data) (destroy destroy) (scope notified): The pop-transform callback
 * @user_data: Data to pass to @func
 * @destroy: (nullable): Function to call when @user_data is no longer needed
 *
 * Sets the pop-transform callback on the paint functions struct.
 *
 * Since: 7.0.0
 */
HB_EXTERN void
hb_paint_funcs_set_pop_transform_func (hb_paint_funcs_t              *funcs,
                                       hb_paint_pop_transform_func_t  func,
                                       void                          *user_data,
                                       hb_destroy_func_t              destroy);

/**
 * hb_paint_funcs_set_color_glyph_func:
 * @funcs: A paint functions struct
 * @func: (closure user_data) (destroy destroy) (scope notified): The color-glyph callback
 * @user_data: Data to pass to @func
 * @destroy: (nullable): Function to call when @user_data is no longer needed
 *
 * Sets the color-glyph callback on the paint functions struct.
 *
 * Since: 8.2.0
 */
HB_EXTERN void
hb_paint_funcs_set_color_glyph_func (hb_paint_funcs_t                *funcs,
				     hb_paint_color_glyph_func_t     func,
				     void                            *user_data,
				     hb_destroy_func_t                destroy);

/**
 * hb_paint_funcs_set_push_clip_glyph_func:
 * @funcs: A paint functions struct
 * @func: (closure user_data) (destroy destroy) (scope notified): The push-clip-glyph callback
 * @user_data: Data to pass to @func
 * @destroy: (nullable): Function to call when @user_data is no longer needed
 *
 * Sets the push-clip-glyph callback on the paint functions struct.
 *
 * Since: 7.0.0
 */
HB_EXTERN void
hb_paint_funcs_set_push_clip_glyph_func (hb_paint_funcs_t                *funcs,
                                         hb_paint_push_clip_glyph_func_t  func,
                                         void                            *user_data,
                                         hb_destroy_func_t                destroy);

/**
 * hb_paint_funcs_set_push_clip_rectangle_func:
 * @funcs: A paint functions struct
 * @func: (closure user_data) (destroy destroy) (scope notified): The push-clip-rectangle callback
 * @user_data: Data to pass to @func
 * @destroy: (nullable): Function to call when @user_data is no longer needed
 *
 * Sets the push-clip-rect callback on the paint functions struct.
 *
 * Since: 7.0.0
 */
HB_EXTERN void
hb_paint_funcs_set_push_clip_rectangle_func (hb_paint_funcs_t                    *funcs,
                                             hb_paint_push_clip_rectangle_func_t  func,
                                             void                                *user_data,
                                             hb_destroy_func_t                    destroy);

/**
 * hb_paint_funcs_set_pop_clip_func:
 * @funcs: A paint functions struct
 * @func: (closure user_data) (destroy destroy) (scope notified): The pop-clip callback
 * @user_data: Data to pass to @func
 * @destroy: (nullable): Function to call when @user_data is no longer needed
 *
 * Sets the pop-clip callback on the paint functions struct.
 *
 * Since: 7.0.0
 */
HB_EXTERN void
hb_paint_funcs_set_pop_clip_func (hb_paint_funcs_t         *funcs,
                                  hb_paint_pop_clip_func_t  func,
                                  void                     *user_data,
                                  hb_destroy_func_t         destroy);

/**
 * hb_paint_funcs_set_color_func:
 * @funcs: A paint functions struct
 * @func: (closure user_data) (destroy destroy) (scope notified): The paint-color callback
 * @user_data: Data to pass to @func
 * @destroy: (nullable): Function to call when @user_data is no longer needed
 *
 * Sets the paint-color callback on the paint functions struct.
 *
 * Since: 7.0.0
 */
HB_EXTERN void
hb_paint_funcs_set_color_func (hb_paint_funcs_t      *funcs,
                               hb_paint_color_func_t  func,
                               void                  *user_data,
                               hb_destroy_func_t      destroy);

/**
 * hb_paint_funcs_set_image_func:
 * @funcs: A paint functions struct
 * @func: (closure user_data) (destroy destroy) (scope notified): The paint-image callback
 * @user_data: Data to pass to @func
 * @destroy: (nullable): Function to call when @user_data is no longer needed
 *
 * Sets the paint-image callback on the paint functions struct.
 *
 * Since: 7.0.0
 */
HB_EXTERN void
hb_paint_funcs_set_image_func (hb_paint_funcs_t      *funcs,
                               hb_paint_image_func_t  func,
                               void                  *user_data,
                               hb_destroy_func_t      destroy);

/**
 * hb_paint_funcs_set_linear_gradient_func:
 * @funcs: A paint functions struct
 * @func: (closure user_data) (destroy destroy) (scope notified): The linear-gradient callback
 * @user_data: Data to pass to @func
 * @destroy: (nullable): Function to call when @user_data is no longer needed
 *
 * Sets the linear-gradient callback on the paint functions struct.
 *
 * Since: 7.0.0
 */
HB_EXTERN void
hb_paint_funcs_set_linear_gradient_func (hb_paint_funcs_t                *funcs,
                                         hb_paint_linear_gradient_func_t  func,
                                         void                            *user_data,
                                         hb_destroy_func_t                destroy);

/**
 * hb_paint_funcs_set_radial_gradient_func:
 * @funcs: A paint functions struct
 * @func: (closure user_data) (destroy destroy) (scope notified): The radial-gradient callback
 * @user_data: Data to pass to @func
 * @destroy: (nullable): Function to call when @user_data is no longer needed
 *
 * Sets the radial-gradient callback on the paint functions struct.
 *
 * Since: 7.0.0
 */
HB_EXTERN void
hb_paint_funcs_set_radial_gradient_func (hb_paint_funcs_t                *funcs,
                                         hb_paint_radial_gradient_func_t  func,
                                         void                            *user_data,
                                         hb_destroy_func_t                destroy);

/**
 * hb_paint_funcs_set_sweep_gradient_func:
 * @funcs: A paint functions struct
 * @func: (closure user_data) (destroy destroy) (scope notified): The sweep-gradient callback
 * @user_data: Data to pass to @func
 * @destroy: (nullable): Function to call when @user_data is no longer needed
 *
 * Sets the sweep-gradient callback on the paint functions struct.
 *
 * Since: 7.0.0
 */
HB_EXTERN void
hb_paint_funcs_set_sweep_gradient_func (hb_paint_funcs_t               *funcs,
                                        hb_paint_sweep_gradient_func_t  func,
                                        void                           *user_data,
                                        hb_destroy_func_t               destroy);

/**
 * hb_paint_funcs_set_push_group_func:
 * @funcs: A paint functions struct
 * @func: (closure user_data) (destroy destroy) (scope notified): The push-group callback
 * @user_data: Data to pass to @func
 * @destroy: (nullable): Function to call when @user_data is no longer needed
 *
 * Sets the push-group callback on the paint functions struct.
 *
 * Since: 7.0.0
 */
HB_EXTERN void
hb_paint_funcs_set_push_group_func (hb_paint_funcs_t           *funcs,
                                    hb_paint_push_group_func_t  func,
                                    void                       *user_data,
                                    hb_destroy_func_t           destroy);

/**
 * hb_paint_funcs_set_pop_group_func:
 * @funcs: A paint functions struct
 * @func: (closure user_data) (destroy destroy) (scope notified): The pop-group callback
 * @user_data: Data to pass to @func
 * @destroy: (nullable): Function to call when @user_data is no longer needed
 *
 * Sets the pop-group callback on the paint functions struct.
 *
 * Since: 7.0.0
 */
HB_EXTERN void
hb_paint_funcs_set_pop_group_func (hb_paint_funcs_t          *funcs,
                                   hb_paint_pop_group_func_t  func,
                                   void                       *user_data,
                                   hb_destroy_func_t           destroy);

/**
 * hb_paint_funcs_set_custom_palette_color_func:
 * @funcs: A paint functions struct
 * @func: (closure user_data) (destroy destroy) (scope notified): The custom-palette-color callback
 * @user_data: Data to pass to @func
 * @destroy: (nullable): Function to call when @user_data is no longer needed
 *
 * Sets the custom-palette-color callback on the paint functions struct.
 *
 * Since: 7.0.0
 */
HB_EXTERN void
hb_paint_funcs_set_custom_palette_color_func (hb_paint_funcs_t                     *funcs,
                                              hb_paint_custom_palette_color_func_t  func,
                                              void                                 *user_data,
                                              hb_destroy_func_t                     destroy);
/*
 * Manual API
 */

HB_EXTERN void
hb_paint_push_transform (hb_paint_funcs_t *funcs, void *paint_data,
                         float xx, float yx,
                         float xy, float yy,
                         float dx, float dy);

HB_EXTERN void
hb_paint_pop_transform (hb_paint_funcs_t *funcs, void *paint_data);

HB_EXTERN hb_bool_t
hb_paint_color_glyph (hb_paint_funcs_t *funcs, void *paint_data,
                      hb_codepoint_t glyph,
                      hb_font_t *font);

HB_EXTERN void
hb_paint_push_clip_glyph (hb_paint_funcs_t *funcs, void *paint_data,
                          hb_codepoint_t glyph,
                          hb_font_t *font);

HB_EXTERN void
hb_paint_push_clip_rectangle (hb_paint_funcs_t *funcs, void *paint_data,
                              float xmin, float ymin,
                              float xmax, float ymax);

HB_EXTERN void
hb_paint_pop_clip (hb_paint_funcs_t *funcs, void *paint_data);

HB_EXTERN void
hb_paint_color (hb_paint_funcs_t *funcs, void *paint_data,
                hb_bool_t is_foreground,
                hb_color_t color);

HB_EXTERN void
hb_paint_image (hb_paint_funcs_t *funcs, void *paint_data,
                hb_blob_t *image,
                unsigned int width,
                unsigned int height,
                hb_tag_t format,
                float slant,
                hb_glyph_extents_t *extents);

HB_EXTERN void
hb_paint_linear_gradient (hb_paint_funcs_t *funcs, void *paint_data,
                          hb_color_line_t *color_line,
                          float x0, float y0,
                          float x1, float y1,
                          float x2, float y2);

HB_EXTERN void
hb_paint_radial_gradient (hb_paint_funcs_t *funcs, void *paint_data,
                          hb_color_line_t *color_line,
                          float x0, float y0,
                          float r0,
                          float x1, float y1,
                          float r1);

HB_EXTERN void
hb_paint_sweep_gradient (hb_paint_funcs_t *funcs, void *paint_data,
                         hb_color_line_t *color_line,
                         float x0, float y0,
                         float start_angle, float end_angle);

HB_EXTERN void
hb_paint_push_group (hb_paint_funcs_t *funcs, void *paint_data);

HB_EXTERN void
hb_paint_pop_group (hb_paint_funcs_t *funcs, void *paint_data,
                    hb_paint_composite_mode_t mode);

HB_EXTERN hb_bool_t
hb_paint_custom_palette_color (hb_paint_funcs_t *funcs, void *paint_data,
                               unsigned int color_index,
                               hb_color_t *color);

HB_END_DECLS

#endif  /* HB_PAINT_H */
