/*
 * Copyright © 2022  Behdad Esfahbod
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

#ifndef HB_LIMITS_HH
#define HB_LIMITS_HH

#include "hb.hh"


#ifndef HB_BUFFER_MAX_LEN_FACTOR
#define HB_BUFFER_MAX_LEN_FACTOR 256
#endif
#ifndef HB_BUFFER_MAX_LEN_MIN
#define HB_BUFFER_MAX_LEN_MIN 65536
#endif
#ifndef HB_BUFFER_MAX_LEN_DEFAULT
#define HB_BUFFER_MAX_LEN_DEFAULT 0x3FFFFFFF /* Shaping more than a billion chars? Let us know! */
#endif

#ifndef HB_BUFFER_MAX_OPS_FACTOR
#define HB_BUFFER_MAX_OPS_FACTOR 4096
#endif
#ifndef HB_BUFFER_MAX_OPS_MIN
#define HB_BUFFER_MAX_OPS_MIN 65536
#endif
#ifndef HB_BUFFER_MAX_OPS_DEFAULT
#define HB_BUFFER_MAX_OPS_DEFAULT 0x1FFFFFFF /* Shaping more than a billion operations? Let us know! */
#endif


#ifndef HB_MAX_NESTING_LEVEL
#define HB_MAX_NESTING_LEVEL 64
#endif


#ifndef HB_MAX_CONTEXT_LENGTH
#define HB_MAX_CONTEXT_LENGTH 64
#endif

#ifndef HB_MAX_SYLLABLE_LENGTH
#define HB_MAX_SYLLABLE_LENGTH 64
#endif

#ifndef HB_CLOSURE_MAX_STAGES
/*
 * The maximum number of times a lookup can be applied during shaping.
 * Used to limit the number of iterations of the closure algorithm.
 * This must be larger than the number of times add_gsub_pause() is
 * called in a collect_features call of any shaper.
 */
#define HB_CLOSURE_MAX_STAGES 12
#endif

#ifndef HB_MAX_SCRIPTS
#define HB_MAX_SCRIPTS 500
#endif

#ifndef HB_MAX_LANGSYS
#define HB_MAX_LANGSYS 2000
#endif

#ifndef HB_MAX_LANGSYS_FEATURE_COUNT
#define HB_MAX_LANGSYS_FEATURE_COUNT 50000
#endif

#ifndef HB_MAX_FEATURE_INDICES
#define HB_MAX_FEATURE_INDICES 8000
#endif

#ifndef HB_MAX_LOOKUP_VISIT_COUNT
#define HB_MAX_LOOKUP_VISIT_COUNT 35000
#endif

#ifndef HB_MAX_GRAPH_EDGE_COUNT
#define HB_MAX_GRAPH_EDGE_COUNT 16384
#endif

#ifndef HB_VAR_COMPOSITE_MAX_AXES
#define HB_VAR_COMPOSITE_MAX_AXES 4096
#endif

#ifndef HB_GLYF_MAX_POINTS
#define HB_GLYF_MAX_POINTS 200000
#endif

#ifndef HB_CFF_MAX_OPS
#define HB_CFF_MAX_OPS 200000
#endif

#ifndef HB_MAX_COMPOSITE_OPERATIONS_PER_GLYPH
#define HB_MAX_COMPOSITE_OPERATIONS_PER_GLYPH 64
#endif

#ifndef HB_SVG_MAX_PATH_SEGMENTS
#define HB_SVG_MAX_PATH_SEGMENTS 262144
#endif

#ifndef HB_GPU_DRAW_MAX_CURVES
#define HB_GPU_DRAW_MAX_CURVES 65536
#endif

/* Tiles emitted by one hb_paint_sweep_gradient_tiles() call.  Also
 * sets the angular resolution (2π over this) below which a repeating
 * color line is filled with its average color instead of tiled;
 * a repeat/reflect color line whose stops span a tiny angle could
 * otherwise emit millions of patches while covering 0..2π. */
#ifndef HB_PAINT_MAX_SWEEP_TILES
#define HB_PAINT_MAX_SWEEP_TILES 4096
#endif

#ifndef HB_SVG_MAX_DOCUMENT_SIZE
#define HB_SVG_MAX_DOCUMENT_SIZE ((size_t) 16 << 20)
#endif

#ifndef HB_RASTER_MAX_BUFFER_SIZE
#define HB_RASTER_MAX_BUFFER_SIZE ((size_t) 1 << 30)
#endif

/* Maximum surface dimension (pixels per side) when extents are derived
 * from font data (glyph extents or accumulated outline bounds).  Bounds
 * attacker-controlled allocations; extents set explicitly through
 * hb_raster_{draw,paint}_set_extents() are not limited. */
#ifndef HB_RASTER_MAX_AUTO_DIMENSION
#define HB_RASTER_MAX_AUTO_DIMENSION 4096
#endif

/*
 * Cumulative work budgets.
 *
 * The limits above bound work within one subsystem: points per glyf
 * glyph, ops per CFF charstring, nodes in a COLR or VARC graph,
 * pixels per raster surface.  When one subsystem drives another --
 * COLR driving a paint backend, a paint backend or VARC loading
 * glyf/CFF outlines -- those limits multiply.  Each driving session
 * therefore carries one cumulative budget, initialized once per
 * top-level entry and only ever decremented, shared by everything
 * the session consumes.  Once a budget is exhausted, further work
 * is skipped best-effort.
 */

/* One VARC glyph (draw or extents), shared by all leaf glyphs loaded
 * from glyf/CFF/CFF2, in units of glyf points / CFF charstring ops. */
#ifndef HB_VARC_MAX_WORK
#define HB_VARC_MAX_WORK ((int64_t) 1 << 20)
#endif

/* One paint-extents session, in outline points consumed by
 * clip-glyph draws. */
#ifndef HB_PAINT_EXTENTS_MAX_WORK
#define HB_PAINT_EXTENTS_MAX_WORK ((int64_t) 1 << 20)
#endif

/* One GPU paint walk, in curves consumed by clip encodes. */
#ifndef HB_GPU_PAINT_MAX_WORK
#define HB_GPU_PAINT_MAX_WORK ((int64_t) 1 << 20)
#endif

/* One vector (SVG/PDF) draw session, in bytes of generated outline
 * path data. */
#ifndef HB_VECTOR_MAX_DRAW_WORK
#define HB_VECTOR_MAX_DRAW_WORK ((int64_t) 16 << 20)
#endif

/* One vector (SVG/PDF) paint session, in bytes of generated outline
 * path and sweep-gradient patch data. */
#ifndef HB_VECTOR_MAX_PAINT_WORK
#define HB_VECTOR_MAX_PAINT_WORK ((int64_t) 16 << 20)
#endif

/* One raster paint session (everything painted between two
 * render/clear calls), in pixel-op units; pixel loops charge their
 * area, consumed outline segments are charged with a fixed weight.
 * The session budget is the larger of this flat value and
 * HB_RASTER_MAX_PAINT_WORK_PASSES full-surface passes, so very large
 * surfaces still get a few full-surface operations. */
#ifndef HB_RASTER_MAX_PAINT_WORK
#define HB_RASTER_MAX_PAINT_WORK ((int64_t) 1 << 26)
#endif

#ifndef HB_RASTER_MAX_PAINT_WORK_PASSES
#define HB_RASTER_MAX_PAINT_WORK_PASSES 4
#endif

/* One raster draw session (everything drawn between two render/clear
 * calls) through the standalone hb-raster-draw API, in Bézier
 * subdivision steps.  When driven by raster-paint, the paint session
 * budget above is charged instead. */
#ifndef HB_RASTER_MAX_DRAW_WORK
#define HB_RASTER_MAX_DRAW_WORK ((int64_t) 1 << 24)
#endif

/* One raster draw session, in accumulated non-horizontal edges. */
#ifndef HB_RASTER_MAX_DRAW_EDGES
#define HB_RASTER_MAX_DRAW_EDGES ((int64_t) 1 << 20)
#endif


#ifndef HB_REPACKER_MAX_ITERATIONS
#define HB_REPACKER_MAX_ITERATIONS 500
#endif

#ifndef HB_REPACKER_MAX_VERTICES
#define HB_REPACKER_MAX_VERTICES 800000
#endif

#ifndef HB_REPACKER_MAX_SPACES
#define HB_REPACKER_MAX_SPACES 8000
#endif


#endif /* HB_LIMITS_HH */
