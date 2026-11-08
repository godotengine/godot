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

/* Total bytes of sub-blobs (gradient color stops, clip outlines) in one
 * hb_gpu_paint_t walk; exceeding it makes the paint unsupported. */
#ifndef HB_GPU_PAINT_MAX_SUB_BYTES
#define HB_GPU_PAINT_MAX_SUB_BYTES ((unsigned) 64 << 20)
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

/* Maximum size of one serialized vector document (SVG or PDF) produced by
 * the vector backends.  Bounds output that is not outline-derived -- most
 * of a path, but especially sweep-gradient meshes and embedded bitmaps --
 * which the outline work budget cannot see.  vector is one glyph per
 * context, so a flat cap suffices.  Unsigned (not size_t like the SVG-table
 * limit above) to match hb_vector_buf_t's unsigned length arithmetic. */
#ifndef HB_VECTOR_MAX_DOCUMENT_SIZE
#define HB_VECTOR_MAX_DOCUMENT_SIZE ((unsigned) 16 << 20)
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

/* Fixed relative weights.  Keep the value in the name so charge sites are
 * easy to audit; tune the common session cap, not these values. */
#define HB_BUDGET_1	1u
#define HB_BUDGET_2	2u
#define HB_BUDGET_4	4u
#define HB_BUDGET_8	8u
#define HB_BUDGET_16	16u
#define HB_BUDGET_32	32u
#define HB_BUDGET_64	64u
#define HB_BUDGET_128	128u
#define HB_BUDGET_256	256u
#define HB_BUDGET_512	512u
#define HB_BUDGET_1024	1024u

/* The common finite default for one top-level glyph rendering session.
 * Nested outline and paint work shares the same live counter. */
#ifndef HB_BUDGET_GLYPH
#define HB_BUDGET_GLYPH ((int64_t) 1 << 24)
#endif

/* Precharge COST * MULT.  Callers bound COST and MULT structurally, and live
 * budgets are concrete non-negative values when a session starts. */
static HB_ALWAYS_INLINE bool
hb_budget_spend (int64_t &budget, unsigned int cost, unsigned int mult = 1)
{
  budget -= (int64_t) cost * mult;
  return budget >= 0;
}

/* The flat pixel-work floor for one raster paint session.  Pixel and
 * outline work use separate live counters, so keep this independent of
 * the common outline budget above. */
#ifndef HB_BUDGET_RASTER_PIXELS
#define HB_BUDGET_RASTER_PIXELS ((int64_t) 1 << 26)
#endif

/* Very large raster surfaces still get a few full-surface operations. */
#ifndef HB_BUDGET_RASTER_PAINT_PASSES
#define HB_BUDGET_RASTER_PAINT_PASSES 4
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
