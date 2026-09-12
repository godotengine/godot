/*
 * Copyright (C) 2026  Behdad Esfahbod
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
 *
 * Author(s): Behdad Esfahbod
 */

#include "hb.hh"

#include "hb-gpu.h"
#include "hb-blob.hh"

/**
 * SECTION:hb-gpu
 * @title: hb-gpu
 * @short_description: GPU text rendering
 * @include: hb-gpu.h
 *
 * This module provides GPU-side text rendering.  Glyph data is
 * encoded on the CPU into compact blobs that the GPU decodes and
 * rasterizes directly in the fragment shader, with no intermediate
 * bitmap atlas.
 *
 * Two renderers share the same atlas, vertex stage, and pipeline:
 *
 * - Draw, an antialiased monochrome coverage mask for outline
 *   glyphs.
 * - Paint, a premultiplied-RGBA color accumulator for COLRv0 and
 *   COLRv1 glyphs.  Paint also renders plain monochrome outlines
 *   (as a single foreground-colored layer), at some cost over the
 *   Draw path.
 *
 * Applications typically pick one renderer per font, based on
 * whether the font has a color paint table (see
 * hb_ot_color_has_paint()).  The Draw path is meaningfully faster
 * than Paint on monochrome fonts, so using Paint unconditionally
 * leaves performance on the table.
 *
 * See the [live web demo](https://harfbuzz.github.io/hb-gpu-demo/).
 *
 * # Draw
 *
 * The draw renderer implements the
 * [Slug](https://github.com/EricLengyel/Slug) algorithm by Eric
 * Lengyel.  It produces an antialiased coverage mask in the
 * fragment shader, suitable for compositing over any background
 * or multiplying with any foreground color.
 *
 * ## Encoding
 *
 * Each glyph is encoded independently into an opaque blob of
 * RGBA16I texels (8 bytes each).  Typical encoding flow:
 *
 * |[<!-- language="plain" -->
 * hb_gpu_draw_t *draw = hb_gpu_draw_create_or_fail ();
 *
 * hb_gpu_draw_glyph (draw, font, gid);
 * hb_blob_t *blob = hb_gpu_draw_encode (draw);
 *
 * // Upload hb_blob_get_data(blob) to the atlas at some offset.
 * unsigned atlas_offset = upload_to_atlas (blob);
 *
 * hb_gpu_draw_recycle_blob (draw, blob);
 * // repeat for more glyphs...
 *
 * hb_gpu_draw_destroy (draw);
 * ]|
 *
 * Create the encoder once and reuse it across glyphs.
 * hb_gpu_draw_encode() auto-clears the accumulated outlines on
 * return, so the next call to hb_gpu_draw_glyph() starts fresh;
 * user configuration (font scale) is preserved.  Pass the
 * encoded blob back via hb_gpu_draw_recycle_blob() after upload
 * to recycle the buffer across encodes.
 *
 * ## Atlas setup
 *
 * All encoded blobs are uploaded into a single GL_TEXTURE_BUFFER
 * backed by a GL_RGBA16I format buffer.  Each glyph occupies a
 * contiguous range of texels at a known offset:
 *
 * |[<!-- language="plain" -->
 * GLuint buf, tex;
 * glGenBuffers (1, &buf);
 * glGenTextures (1, &tex);
 *
 * glBindBuffer (GL_TEXTURE_BUFFER, buf);
 * glBufferData (GL_TEXTURE_BUFFER, capacity * 8, NULL, GL_STATIC_DRAW);
 *
 * glBindTexture (GL_TEXTURE_BUFFER, tex);
 * glTexBuffer (GL_TEXTURE_BUFFER, GL_RGBA16I, buf);
 *
 * // Upload a glyph blob at 'offset' texels:
 * glBufferSubData (GL_TEXTURE_BUFFER,
 *                  offset * 8,
 *                  hb_blob_get_length (blob),
 *                  hb_blob_get_data (blob, NULL));
 * ]|
 *
 * ## Shader compilation
 *
 * The library provides GLSL source strings for shared helpers
 * (atlas access, vertex dilation) and for the draw-renderer stage
 * code.  Prepend a #version directive, concatenate shared and
 * draw sources in order, and append your own main():
 *
 * |[<!-- language="plain" -->
 * const char *vert_sources[] = {
 *     "#version 330\n",
 *     hb_gpu_shader_source      (HB_GPU_SHADER_STAGE_VERTEX,
 *                                HB_GPU_SHADER_LANG_GLSL),
 *     hb_gpu_draw_shader_source (HB_GPU_SHADER_STAGE_VERTEX,
 *                                HB_GPU_SHADER_LANG_GLSL),
 *     your_vertex_main
 * };
 * glShaderSource (vert_shader, 4, vert_sources, NULL);
 * // Same pattern for the fragment shader.
 * ]|
 *
 * hb_gpu_draw_shader_source() returns an empty string for the
 * vertex stage, so the assembly order above is safe.
 *
 * ## Vertex shader
 *
 * The vertex library provides one function:
 *
 * |[<!-- language="plain" -->
 * void hb_gpu_dilate (inout vec2 position, inout vec2 texcoord,
 *                     vec2 normal, vec4 jac,
 *                     mat4 m, vec2 viewport);
 * ]|
 *
 * This expands each glyph quad by approximately half a pixel on
 * screen to ensure proper antialiasing coverage at the edges.
 * It must be called in the vertex shader before computing
 * gl_Position.
 *
 * Per-vertex attributes for each glyph quad (2 triangles, 6
 * vertices, 4 unique corners):
 *
 * - position (vec2): object-space vertex position, computed from
 *   the glyph's bounding box.  For corner (cx, cy) where cx,cy
 *   are 0 or 1:
 *   |[<!-- language="plain" -->
 *   pos.x = pen_x + scale * lerp(extent_min_x, extent_max_x, cx)
 *   pos.y = pen_y - scale * lerp(extent_min_y, extent_max_y, cy)
 *   ]|
 *   where scale = font_size / upem.
 *
 * - texcoord (vec2): em-space texture coordinates, equal to the
 *   raw extent values: `(ex, ey)` where ex/ey are the glyph's
 *   bounding box corners in font design units.  The fragment shader
 *   samples the encoded blob in this coordinate space.
 *
 * - normal (vec2): outward normal at each corner.
 *   `(-1, +1)` for (0,0), `(+1, +1)` for (1,0),
 *   `(-1, -1)` for (0,1), `(+1, -1)` for (1,1).
 *   Note: y is negated relative to cx,cy because the common
 *   em-to-object transform flips y.
 *
 * - jac (vec4): maps object-space displacements back to
 *   em-space after dilation, stored row-major as
 *   (j00, j01, j10, j11).  For the common case of uniform
 *   scaling with y-flip:
 *   |[<!-- language="plain" -->
 *   jac = vec4(emPerPos, 0.0, 0.0, -emPerPos)
 *   ]|
 *   where emPerPos = upem / font_size.  For non-trivial
 *   transforms, see the hb_gpu_dilate source.
 *
 * - m (mat4): the model-view-projection matrix (uniform).
 *
 * - viewport (vec2): the viewport size in pixels (uniform).
 *
 * A typical vertex main:
 *
 * |[<!-- language="plain" -->
 * uniform mat4 u_matViewProjection;
 * uniform vec2 u_viewport;
 *
 * in vec2 a_position;
 * in vec2 a_texcoord;
 * in vec2 a_normal;
 * in float a_emPerPos;
 * in uint a_glyphLoc;
 *
 * out vec2 v_texcoord;
 * flat out uint v_glyphLoc;
 *
 * void main () {
 *   vec2 pos = a_position;
 *   vec2 tex = a_texcoord;
 *   vec4 jac = vec4 (a_emPerPos, 0.0, 0.0, -a_emPerPos);
 *   hb_gpu_dilate (pos, tex, a_normal, jac,
 *                  u_matViewProjection, u_viewport);
 *   gl_Position = u_matViewProjection * vec4 (pos, 0.0, 1.0);
 *   v_texcoord = tex;
 *   v_glyphLoc = a_glyphLoc;
 * }
 * ]|
 *
 * ## Font scale
 *
 * The encoder works in font design units (upem).  For best
 * results, do not set a scale on the #hb_font_t passed to
 * hb_gpu_draw_glyph() — leave it at the default (upem).
 * Scaling is applied later when computing vertex positions:
 *
 * |[<!-- language="plain" -->
 * float scale = font_size / upem;
 * pos.x = pen_x + scale * extent_x;
 * pos.y = pen_y - scale * extent_y;
 * ]|
 *
 * If you do set a font scale (e.g. for hinting), the encoded
 * blob and extents will be in the scaled coordinate space,
 * and you must adjust emPerPos accordingly.
 *
 * ## Dilation
 *
 * Each glyph is rendered on a screen-aligned quad whose
 * corners match the glyph's bounding box.  Without dilation,
 * the antialiased edges at the quad boundary get clipped — the
 * fragment shader needs at least half a pixel of padding
 * around the glyph to produce smooth coverage gradients.
 *
 * hb_gpu_dilate computes a perspective-correct half-pixel
 * expansion.  It adjusts both the vertex position (expanding
 * the quad) and the texcoord (so the fragment shader samples
 * the right location in em-space after expansion).
 *
 * The `jac` parameter maps object-space displacements back to
 * em-space.  For the common case of uniform scaling with
 * y-flip (`pos.y = pen_y - scale * ey`):
 *
 * |[<!-- language="plain" -->
 * float emPerPos = upem / font_size;
 * jac = vec4(emPerPos, 0.0, 0.0, -emPerPos);
 * ]|
 *
 * For non-uniform scaling or shearing transforms, jac is the
 * 2x2 inverse of the em-to-object linear transform, stored
 * row-major.
 *
 * ### Static dilation alternative
 *
 * Applications that do not need perspective correctness (e.g.
 * strictly 2D rendering at known sizes) can skip hb_gpu_dilate
 * and instead expand each quad vertex by a fixed amount along
 * its normal, based on the minimum expected font size:
 *
 * |[<!-- language="plain" -->
 * float min_ppem = 10.0;  // smallest expected size in pixels
 * float pad = 0.5 * upem / min_ppem;  // half-pixel in em units
 * pos += normal * pad * scale;
 * texcoord += normal * pad;
 * ]|
 *
 * This over-dilates at larger sizes (wasting fill rate on
 * transparent pixels) but is simpler to implement and avoids
 * the need for the MVP matrix and viewport size in the vertex
 * shader.
 *
 * ## Fragment shader
 *
 * The fragment library provides two functions:
 *
 * |[<!-- language="plain" -->
 * float hb_gpu_draw (vec2 renderCoord, uint glyphLoc);
 * ]|
 *
 * It requires the `hb_gpu_atlas` uniform to be bound to the
 * texture containing the encoded glyph data.  By default this
 * is an `isamplerBuffer` (GL_TEXTURE_BUFFER).  For platforms
 * without texture buffers (e.g. WebGL2), define
 * `HB_GPU_ATLAS_2D` before including the shader source; the
 * atlas then becomes an `isampler2D` and a
 * `uniform int hb_gpu_atlas_width` must also be set to the
 * texture width.
 *
 * Parameters:
 *
 * - renderCoord: the interpolated em-space coordinate from
 *   the vertex shader (v_texcoord).
 *
 * - glyphLoc: the texel offset of this glyph's encoded blob
 *   in the atlas buffer.  Passed as a flat varying from the
 *   vertex shader.
 *
 * Returns the coverage in [0, 1]: 1.0 inside the glyph, 0.0
 * outside, with antialiased edges.
 *
 * A typical fragment main:
 *
 * |[<!-- language="plain" -->
 * in vec2 v_texcoord;
 * flat in uint v_glyphLoc;
 * out vec4 fragColor;
 *
 * void main () {
 *   float coverage = hb_gpu_draw (v_texcoord, v_glyphLoc);
 *   fragColor = vec4 (0.0, 0.0, 0.0, coverage);
 * }
 * ]|
 *
 * The coverage value can be used with alpha blending
 * (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA) to composite
 * over a background, or multiplied with any color.
 *
 * ## Stem darkening
 *
 * |[<!-- language="plain" -->
 * float hb_gpu_stem_darken (float coverage, float brightness, float ppem);
 * ]|
 *
 * Optional stem darkening adjusts coverage at small sizes so
 * thin stems are not washed out by gamma correction.
 *
 * Parameters:
 *
 * - coverage: a coverage / alpha value in [0, 1].  For
 *   hb_gpu_draw() this is the returned coverage; for
 *   hb_gpu_paint() use the edge coverage out parameter.
 *
 * - brightness: brightness of the color that will end up on
 *   screen for this fragment, in [0, 1], computed as
 *   `dot (straight_color.rgb, vec3 (1.0 / 3.0))`.
 *
 * - ppem: pixels per em at this fragment, computed as
 *   `1.0 / max (fwidth (v_texcoord).x, fwidth (v_texcoord).y)`.
 *
 * Example — draw path:
 *
 * |[<!-- language="plain" -->
 * float coverage = hb_gpu_draw (v_texcoord, v_glyphLoc);
 * float brightness = dot (u_foreground.rgb, vec3 (1.0 / 3.0));
 * float ppem = 1.0 / max (fwidth (v_texcoord).x,
 *                          fwidth (v_texcoord).y);
 * coverage = hb_gpu_stem_darken (coverage, brightness, ppem);
 * ]|
 *
 * Example — paint path: hb_gpu_paint() returns a separate
 * edge coverage value that tracks the maximum antialiasing
 * coverage across paint layers.  Apply stem darkening and gamma to this
 * edge coverage, then scale the premultiplied color to match.
 * Interior fragments (coverage == 1) are left untouched.
 *
 * |[<!-- language="plain" -->
 * float cov;
 * vec4 c = hb_gpu_paint (v_texcoord, v_glyphLoc, u_foreground, cov);
 * if (cov > 0.0 && cov < 1.0) {
 *   float brightness = c.a > 0.0
 *     ? dot (c.rgb, vec3 (1.0 / 3.0)) / c.a : 0.0;
 *   float ppem = 1.0 / max (fwidth (v_texcoord).x,
 *                            fwidth (v_texcoord).y);
 *   float adj = hb_gpu_stem_darken (cov, brightness, ppem);
 *   c *= adj / cov;
 * }
 * ]|
 *
 * # Paint
 *
 * The paint renderer walks the font's paint tree (COLRv0 layers
 * or COLRv1 paint subtree) and records one layer op per solid or
 * gradient fill, each with its own clip outline encoded as a Slug
 * (so the paint renderer reuses the draw renderer internally for
 * clips).  Palette colors and custom overrides are resolved and
 * baked into the blob at encode time, so the shader does not need
 * a separate palette uniform.
 *
 * Monochrome (non-color) glyphs are handled transparently:
 * hb_gpu_paint_glyph() synthesizes a single foreground-colored
 * layer from the glyph's outline, so callers can use the paint
 * renderer uniformly for any font.  This costs more per fragment
 * than the Draw path, but the rendered result is the same.
 *
 * ## Encoding
 *
 * |[<!-- language="plain" -->
 * hb_gpu_paint_t *paint = hb_gpu_paint_create_or_fail ();
 *
 * // Optional: select a palette and install any custom-palette
 * // overrides before hb_gpu_paint_glyph — their values are baked
 * // into the encoded blob.  (The foreground color is NOT set here:
 * // it stays a shader uniform so dark-mode and color-change
 * // toggles take effect without re-encoding.)
 * hb_gpu_paint_set_palette (paint, 0);
 * hb_gpu_paint_set_custom_palette_color (paint, 2,
 *                                        HB_COLOR (0xff, 0, 0, 0xff));
 *
 * hb_gpu_paint_glyph (paint, font, gid);
 * hb_glyph_extents_t ext;
 * hb_blob_t *blob = hb_gpu_paint_encode (paint, &ext);
 *
 * // Upload hb_blob_get_data(blob) to the same atlas used by draw.
 * unsigned atlas_offset = upload_to_atlas (blob);
 *
 * hb_gpu_paint_recycle_blob (paint, blob);
 * // repeat for more glyphs...
 *
 * hb_gpu_paint_destroy (paint);
 * ]|
 *
 * As with the draw encoder, create the paint encoder once and
 * reuse it.  hb_gpu_paint_encode() auto-clears on return; user
 * configuration (palette, custom-palette overrides) is preserved
 * across encodes.  Because colors are baked, changing any of that
 * configuration afterwards requires re-encoding the affected
 * glyphs.  The encoded blob's texel format is the same RGBA16I
 * as the draw renderer's, so both kinds of blobs can coexist in
 * one atlas at different offsets.
 *
 * ## Shader composition
 *
 * Same pattern as the draw renderer but using
 * hb_gpu_paint_shader_source() for the paint-specific helpers.
 * The vertex stage is identical — hb_gpu_paint_shader_source()
 * returns an empty string for HB_GPU_SHADER_STAGE_VERTEX, so the
 * fixed assembly order still works:
 *
 * |[<!-- language="plain" -->
 * const char *frag_sources[] = {
 *     "#version 330\n",
 *     hb_gpu_shader_source       (HB_GPU_SHADER_STAGE_FRAGMENT,
 *                                 HB_GPU_SHADER_LANG_GLSL),
 *     hb_gpu_paint_shader_source (HB_GPU_SHADER_STAGE_FRAGMENT,
 *                                 HB_GPU_SHADER_LANG_GLSL),
 *     your_fragment_main
 * };
 * glShaderSource (frag_shader, 4, frag_sources, NULL);
 * ]|
 *
 * The atlas setup, vertex attributes, dilation, and font scaling
 * guidance from the Draw section all apply unchanged.
 *
 * ## Fragment shader
 *
 * The paint library provides one function:
 *
 * |[<!-- language="plain" -->
 * vec4 hb_gpu_paint (vec2 renderCoord, uint glyphLoc, vec4 foreground,
 *                    out float coverage);
 * ]|
 *
 * Parameters:
 *
 * - renderCoord: the interpolated em-space coordinate from the
 *   vertex shader (v_texcoord), as for hb_gpu_draw().
 *
 * - glyphLoc: the texel offset of this glyph's encoded paint blob
 *   in the atlas, passed as a flat varying from the vertex shader.
 *
 * - foreground: the foreground color used to resolve palette
 *   entries that COLRv1 marks as "foreground color".  The encoder
 *   only records the is-foreground flag per layer/stop; the actual
 *   color is substituted here at render time, so runtime
 *   foreground-color changes (e.g. a dark-mode toggle) take effect
 *   without re-encoding any glyphs.
 *
 * - coverage (out): the maximum antialiasing coverage across all
 *   paint layers.  Fractional values (between 0 and 1) indicate
 *   antialiased edge pixels; 0 means fully outside, 1 means fully
 *   inside.  Stem darkening and gamma correction should be applied
 *   to this value rather than to the returned color, so that
 *   interior paint colors are unaffected.  See the stem darkening
 *   notes in the Draw section.
 *
 * Returns premultiplied RGBA: fully transparent outside the glyph,
 * composited layers inside.  Use (GL_ONE, GL_ONE_MINUS_SRC_ALPHA)
 * blending to composite over a background.
 *
 * A typical fragment main:
 *
 * |[<!-- language="plain" -->
 * uniform vec4 u_foreground;
 * uniform float u_gamma;
 * uniform float u_stem_darkening;
 * in vec2 v_texcoord;
 * flat in uint v_glyphLoc;
 * out vec4 fragColor;
 *
 * void main () {
 *   float cov;
 *   vec4 c = hb_gpu_paint (v_texcoord, v_glyphLoc, u_foreground, cov);
 *   if (cov > 0.0 && cov < 1.0) {
 *     float adj = cov;
 *     if (u_stem_darkening > 0.0) {
 *       float brightness = c.a > 0.0
 *         ? dot (c.rgb, vec3 (1.0 / 3.0)) / c.a : 0.0;
 *       adj = hb_gpu_stem_darken (adj, brightness,
 *         1.0 / max (fwidth (v_texcoord).x, fwidth (v_texcoord).y));
 *     }
 *     if (u_gamma != 1.0)
 *       adj = pow (adj, u_gamma);
 *     c *= adj / cov;
 *   }
 *   fragColor = c;
 * }
 * ]|
 **/


#include "hb-gpu-fragment-glsl.hh"
#include "hb-gpu-fragment-msl.hh"
#include "hb-gpu-fragment-wgsl.hh"
#include "hb-gpu-fragment-hlsl.hh"
#include "hb-gpu-vertex-glsl.hh"
#include "hb-gpu-vertex-msl.hh"
#include "hb-gpu-vertex-wgsl.hh"
#include "hb-gpu-vertex-hlsl.hh"

/**
 * hb_gpu_shader_source:
 * @stage: pipeline stage (vertex or fragment)
 * @lang: shader language variant
 *
 * Returns the shared helper shader source used by the hb-gpu
 * renderers for the specified stage and language.  The returned
 * string is static and must not be freed.
 *
 * The shared source defines utilities such as the atlas sampler
 * and the `hb_gpu_fetch()` accessor for the fragment stage, and
 * the `hb_gpu_dilate()` helper for the vertex stage.  Each
 * renderer-specific source (for example,
 * hb_gpu_draw_shader_source()) assumes these helpers are already
 * in scope — callers should concatenate a `#version` directive,
 * then this shared source, then the renderer-specific source,
 * then their own `main()`.
 *
 * Return value: (transfer none):
 * A shader source string, or `NULL` if @stage or @lang is
 * unsupported.
 *
 * Since: 14.2.0
 **/
const char *
hb_gpu_shader_source (hb_gpu_shader_stage_t stage,
		      hb_gpu_shader_lang_t  lang)
{
  switch (stage) {
  case HB_GPU_SHADER_STAGE_FRAGMENT:
    switch (lang) {
    case HB_GPU_SHADER_LANG_GLSL: return hb_gpu_fragment_glsl;
    case HB_GPU_SHADER_LANG_MSL:  return hb_gpu_fragment_msl;
    case HB_GPU_SHADER_LANG_WGSL: return hb_gpu_fragment_wgsl;
    case HB_GPU_SHADER_LANG_HLSL: return hb_gpu_fragment_hlsl;
    case HB_GPU_SHADER_LANG_INVALID:
    default: return nullptr;
    }
  case HB_GPU_SHADER_STAGE_VERTEX:
    switch (lang) {
    case HB_GPU_SHADER_LANG_GLSL: return hb_gpu_vertex_glsl;
    case HB_GPU_SHADER_LANG_MSL:  return hb_gpu_vertex_msl;
    case HB_GPU_SHADER_LANG_WGSL: return hb_gpu_vertex_wgsl;
    case HB_GPU_SHADER_LANG_HLSL: return hb_gpu_vertex_hlsl;
    case HB_GPU_SHADER_LANG_INVALID:
    default: return nullptr;
    }
  default:
    return nullptr;
  }
}


