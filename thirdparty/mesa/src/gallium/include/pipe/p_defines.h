/**************************************************************************
 *
 * Copyright 2007 VMware, Inc.
 * All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sub license, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice (including the
 * next paragraph) shall be included in all copies or substantial portions
 * of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
 * IN NO EVENT SHALL VMWARE AND/OR ITS SUPPLIERS BE LIABLE FOR
 * ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 **************************************************************************/

#ifndef PIPE_DEFINES_H
#define PIPE_DEFINES_H

#include "p_compiler.h"

#include "compiler/shader_enums.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Gallium error codes.
 *
 * - A zero value always means success.
 * - A negative value always means failure.
 * - The meaning of a positive value is function dependent.
 */
enum pipe_error
{
   PIPE_OK = 0,
   PIPE_ERROR = -1,    /**< Generic error */
   PIPE_ERROR_BAD_INPUT = -2,
   PIPE_ERROR_OUT_OF_MEMORY = -3,
   PIPE_ERROR_RETRY = -4
   /* TODO */
};

enum pipe_blendfactor {
   PIPE_BLENDFACTOR_ONE = 1,
   PIPE_BLENDFACTOR_SRC_COLOR,
   PIPE_BLENDFACTOR_SRC_ALPHA,
   PIPE_BLENDFACTOR_DST_ALPHA,
   PIPE_BLENDFACTOR_DST_COLOR,
   PIPE_BLENDFACTOR_SRC_ALPHA_SATURATE,
   PIPE_BLENDFACTOR_CONST_COLOR,
   PIPE_BLENDFACTOR_CONST_ALPHA,
   PIPE_BLENDFACTOR_SRC1_COLOR,
   PIPE_BLENDFACTOR_SRC1_ALPHA,

   PIPE_BLENDFACTOR_ZERO = 0x11,
   PIPE_BLENDFACTOR_INV_SRC_COLOR,
   PIPE_BLENDFACTOR_INV_SRC_ALPHA,
   PIPE_BLENDFACTOR_INV_DST_ALPHA,
   PIPE_BLENDFACTOR_INV_DST_COLOR,

   PIPE_BLENDFACTOR_INV_CONST_COLOR = 0x17,
   PIPE_BLENDFACTOR_INV_CONST_ALPHA,
   PIPE_BLENDFACTOR_INV_SRC1_COLOR,
   PIPE_BLENDFACTOR_INV_SRC1_ALPHA,
};

enum pipe_blend_func {
   PIPE_BLEND_ADD,
   PIPE_BLEND_SUBTRACT,
   PIPE_BLEND_REVERSE_SUBTRACT,
   PIPE_BLEND_MIN,
   PIPE_BLEND_MAX,
};

enum pipe_logicop {
   PIPE_LOGICOP_CLEAR,
   PIPE_LOGICOP_NOR,
   PIPE_LOGICOP_AND_INVERTED,
   PIPE_LOGICOP_COPY_INVERTED,
   PIPE_LOGICOP_AND_REVERSE,
   PIPE_LOGICOP_INVERT,
   PIPE_LOGICOP_XOR,
   PIPE_LOGICOP_NAND,
   PIPE_LOGICOP_AND,
   PIPE_LOGICOP_EQUIV,
   PIPE_LOGICOP_NOOP,
   PIPE_LOGICOP_OR_INVERTED,
   PIPE_LOGICOP_COPY,
   PIPE_LOGICOP_OR_REVERSE,
   PIPE_LOGICOP_OR,
   PIPE_LOGICOP_SET,
};

#define PIPE_MASK_R  0x1
#define PIPE_MASK_G  0x2
#define PIPE_MASK_B  0x4
#define PIPE_MASK_A  0x8
#define PIPE_MASK_RGBA 0xf
#define PIPE_MASK_Z  0x10
#define PIPE_MASK_S  0x20
#define PIPE_MASK_ZS 0x30
#define PIPE_MASK_RGBAZS (PIPE_MASK_RGBA|PIPE_MASK_ZS)


/**
 * Inequality functions.  Used for depth test, stencil compare, alpha
 * test, shadow compare, etc.
 */
enum pipe_compare_func {
   PIPE_FUNC_NEVER,
   PIPE_FUNC_LESS,
   PIPE_FUNC_EQUAL,
   PIPE_FUNC_LEQUAL,
   PIPE_FUNC_GREATER,
   PIPE_FUNC_NOTEQUAL,
   PIPE_FUNC_GEQUAL,
   PIPE_FUNC_ALWAYS,
};

/** Polygon fill mode */
enum {
   PIPE_POLYGON_MODE_FILL,
   PIPE_POLYGON_MODE_LINE,
   PIPE_POLYGON_MODE_POINT,
   PIPE_POLYGON_MODE_FILL_RECTANGLE,
};

/** Polygon face specification, eg for culling */
#define PIPE_FACE_NONE           0
#define PIPE_FACE_FRONT          1
#define PIPE_FACE_BACK           2
#define PIPE_FACE_FRONT_AND_BACK (PIPE_FACE_FRONT | PIPE_FACE_BACK)

/** Stencil ops */
enum pipe_stencil_op {
   PIPE_STENCIL_OP_KEEP,
   PIPE_STENCIL_OP_ZERO,
   PIPE_STENCIL_OP_REPLACE,
   PIPE_STENCIL_OP_INCR,
   PIPE_STENCIL_OP_DECR,
   PIPE_STENCIL_OP_INCR_WRAP,
   PIPE_STENCIL_OP_DECR_WRAP,
   PIPE_STENCIL_OP_INVERT,
};

/** Texture types.
 * See the documentation for info on PIPE_TEXTURE_RECT vs PIPE_TEXTURE_2D
 */
enum pipe_texture_target
{
   PIPE_BUFFER,
   PIPE_TEXTURE_1D,
   PIPE_TEXTURE_2D,
   PIPE_TEXTURE_3D,
   PIPE_TEXTURE_CUBE,
   PIPE_TEXTURE_RECT,
   PIPE_TEXTURE_1D_ARRAY,
   PIPE_TEXTURE_2D_ARRAY,
   PIPE_TEXTURE_CUBE_ARRAY,
   PIPE_MAX_TEXTURE_TYPES,
};

enum pipe_tex_face {
   PIPE_TEX_FACE_POS_X,
   PIPE_TEX_FACE_NEG_X,
   PIPE_TEX_FACE_POS_Y,
   PIPE_TEX_FACE_NEG_Y,
   PIPE_TEX_FACE_POS_Z,
   PIPE_TEX_FACE_NEG_Z,
   PIPE_TEX_FACE_MAX,
};

enum pipe_tex_wrap {
   PIPE_TEX_WRAP_REPEAT,
   PIPE_TEX_WRAP_CLAMP,
   PIPE_TEX_WRAP_CLAMP_TO_EDGE,
   PIPE_TEX_WRAP_CLAMP_TO_BORDER,
   PIPE_TEX_WRAP_MIRROR_REPEAT,
   PIPE_TEX_WRAP_MIRROR_CLAMP,
   PIPE_TEX_WRAP_MIRROR_CLAMP_TO_EDGE,
   PIPE_TEX_WRAP_MIRROR_CLAMP_TO_BORDER,
};

/** Between mipmaps, ie mipfilter */
enum pipe_tex_mipfilter {
   PIPE_TEX_MIPFILTER_NEAREST,
   PIPE_TEX_MIPFILTER_LINEAR,
   PIPE_TEX_MIPFILTER_NONE,
};

/** Within a mipmap, ie min/mag filter */
enum pipe_tex_filter {
   PIPE_TEX_FILTER_NEAREST,
   PIPE_TEX_FILTER_LINEAR,
};

enum pipe_tex_compare {
   PIPE_TEX_COMPARE_NONE,
   PIPE_TEX_COMPARE_R_TO_TEXTURE,
};

enum pipe_tex_reduction_mode {
   PIPE_TEX_REDUCTION_WEIGHTED_AVERAGE,
   PIPE_TEX_REDUCTION_MIN,
   PIPE_TEX_REDUCTION_MAX,
};

/**
 * Clear buffer bits
 */
#define PIPE_CLEAR_DEPTH        (1 << 0)
#define PIPE_CLEAR_STENCIL      (1 << 1)
#define PIPE_CLEAR_COLOR0       (1 << 2)
#define PIPE_CLEAR_COLOR1       (1 << 3)
#define PIPE_CLEAR_COLOR2       (1 << 4)
#define PIPE_CLEAR_COLOR3       (1 << 5)
#define PIPE_CLEAR_COLOR4       (1 << 6)
#define PIPE_CLEAR_COLOR5       (1 << 7)
#define PIPE_CLEAR_COLOR6       (1 << 8)
#define PIPE_CLEAR_COLOR7       (1 << 9)
/** Combined flags */
/** All color buffers currently bound */
#define PIPE_CLEAR_COLOR        (PIPE_CLEAR_COLOR0 | PIPE_CLEAR_COLOR1 | \
                                 PIPE_CLEAR_COLOR2 | PIPE_CLEAR_COLOR3 | \
                                 PIPE_CLEAR_COLOR4 | PIPE_CLEAR_COLOR5 | \
                                 PIPE_CLEAR_COLOR6 | PIPE_CLEAR_COLOR7)
#define PIPE_CLEAR_DEPTHSTENCIL (PIPE_CLEAR_DEPTH | PIPE_CLEAR_STENCIL)

/**
 * CPU access map flags
 */
enum pipe_map_flags
{
   /**
    * Resource contents read back (or accessed directly) at transfer
    * create time.
    */
   PIPE_MAP_READ = 1 << 0,

   /**
    * Resource contents will be written back at buffer/texture_unmap
    * time (or modified as a result of being accessed directly).
    */
   PIPE_MAP_WRITE = 1 << 1,

   /**
    * Read/modify/write
    */
   PIPE_MAP_READ_WRITE = PIPE_MAP_READ | PIPE_MAP_WRITE,

   /**
    * The transfer should map the texture storage directly. The driver may
    * return NULL if that isn't possible, and the gallium frontend needs to cope
    * with that and use an alternative path without this flag.
    *
    * E.g. the gallium frontend could have a simpler path which maps textures and
    * does read/modify/write cycles on them directly, and a more complicated
    * path which uses minimal read and write transfers.
    *
    * This flag supresses implicit "DISCARD" for buffer_subdata.
    */
   PIPE_MAP_DIRECTLY = 1 << 2,

   /**
    * Discards the memory within the mapped region.
    *
    * It should not be used with PIPE_MAP_READ.
    *
    * See also:
    * - OpenGL's ARB_map_buffer_range extension, MAP_INVALIDATE_RANGE_BIT flag.
    */
   PIPE_MAP_DISCARD_RANGE = 1 << 3,

   /**
    * Fail if the resource cannot be mapped immediately.
    *
    * See also:
    * - Direct3D's D3DLOCK_DONOTWAIT flag.
    * - Mesa's MESA_MAP_NOWAIT_BIT flag.
    * - WDDM's D3DDDICB_LOCKFLAGS.DonotWait flag.
    */
   PIPE_MAP_DONTBLOCK = 1 << 4,

   /**
    * Do not attempt to synchronize pending operations on the resource when mapping.
    *
    * It should not be used with PIPE_MAP_READ.
    *
    * See also:
    * - OpenGL's ARB_map_buffer_range extension, MAP_UNSYNCHRONIZED_BIT flag.
    * - Direct3D's D3DLOCK_NOOVERWRITE flag.
    * - WDDM's D3DDDICB_LOCKFLAGS.IgnoreSync flag.
    */
   PIPE_MAP_UNSYNCHRONIZED = 1 << 5,

   /**
    * Written ranges will be notified later with
    * pipe_context::transfer_flush_region.
    *
    * It should not be used with PIPE_MAP_READ.
    *
    * See also:
    * - pipe_context::transfer_flush_region
    * - OpenGL's ARB_map_buffer_range extension, MAP_FLUSH_EXPLICIT_BIT flag.
    */
   PIPE_MAP_FLUSH_EXPLICIT = 1 << 6,

   /**
    * Discards all memory backing the resource.
    *
    * It should not be used with PIPE_MAP_READ.
    *
    * This is equivalent to:
    * - OpenGL's ARB_map_buffer_range extension, MAP_INVALIDATE_BUFFER_BIT
    * - BufferData(NULL) on a GL buffer
    * - Direct3D's D3DLOCK_DISCARD flag.
    * - WDDM's D3DDDICB_LOCKFLAGS.Discard flag.
    * - D3D10 DDI's D3D10_DDI_MAP_WRITE_DISCARD flag
    * - D3D10's D3D10_MAP_WRITE_DISCARD flag.
    */
   PIPE_MAP_DISCARD_WHOLE_RESOURCE = 1 << 7,

   /**
    * Allows the resource to be used for rendering while mapped.
    *
    * PIPE_RESOURCE_FLAG_MAP_PERSISTENT must be set when creating
    * the resource.
    *
    * If COHERENT is not set, memory_barrier(PIPE_BARRIER_MAPPED_BUFFER)
    * must be called to ensure the device can see what the CPU has written.
    */
   PIPE_MAP_PERSISTENT = 1 << 8,

   /**
    * If PERSISTENT is set, this ensures any writes done by the device are
    * immediately visible to the CPU and vice versa.
    *
    * PIPE_RESOURCE_FLAG_MAP_COHERENT must be set when creating
    * the resource.
    */
   PIPE_MAP_COHERENT = 1 << 9,

   /**
    * Map a resource in a thread-safe manner, because the calling thread can
    * be any thread. It can only be used if both WRITE and UNSYNCHRONIZED are
    * set.
    */
   PIPE_MAP_THREAD_SAFE = 1 << 10,

   /**
    * Map only the depth aspect of a resource
    */
   PIPE_MAP_DEPTH_ONLY = 1 << 11,

   /**
    * Map only the stencil aspect of a resource
    */
   PIPE_MAP_STENCIL_ONLY = 1 << 12,

   /**
    * Mapping will be used only once (never remapped).
    */
   PIPE_MAP_ONCE = 1 << 13,

   /**
    * This and higher bits are reserved for private use by drivers. Drivers
    * should use this as (PIPE_MAP_DRV_PRV << i).
    */
   PIPE_MAP_DRV_PRV = 1 << 14,
};

/**
 * Flags for the flush function.
 */
enum pipe_flush_flags
{
   PIPE_FLUSH_END_OF_FRAME = (1 << 0),
   PIPE_FLUSH_DEFERRED = (1 << 1),
   PIPE_FLUSH_FENCE_FD = (1 << 2),
   PIPE_FLUSH_ASYNC = (1 << 3),
   PIPE_FLUSH_HINT_FINISH = (1 << 4),
   PIPE_FLUSH_TOP_OF_PIPE = (1 << 5),
   PIPE_FLUSH_BOTTOM_OF_PIPE = (1 << 6),
};

/**
 * Flags for pipe_context::dump_debug_state.
 */
#define PIPE_DUMP_DEVICE_STATUS_REGISTERS    (1 << 0)

/**
 * Create a compute-only context. Use in pipe_screen::context_create.
 * This disables draw, blit, and clear*, render_condition, and other graphics
 * functions. Interop with other graphics contexts is still allowed.
 * This allows scheduling jobs on a compute-only hardware command queue that
 * can run in parallel with graphics without stalling it.
 */
#define PIPE_CONTEXT_COMPUTE_ONLY      (1 << 0)

/**
 * Gather debug information and expect that pipe_context::dump_debug_state
 * will be called. Use in pipe_screen::context_create.
 */
#define PIPE_CONTEXT_DEBUG             (1 << 1)

/**
 * Whether out-of-bounds shader loads must return zero and out-of-bounds
 * shader stores must be dropped.
 */
#define PIPE_CONTEXT_ROBUST_BUFFER_ACCESS (1 << 2)

/**
 * Prefer threaded pipe_context. It also implies that video codec functions
 * will not be used. (they will be either no-ops or NULL when threading is
 * enabled)
 */
#define PIPE_CONTEXT_PREFER_THREADED   (1 << 3)

/**
 * Create a high priority context.
 */
#define PIPE_CONTEXT_HIGH_PRIORITY     (1 << 4)

/**
 * Create a low priority context.
 */
#define PIPE_CONTEXT_LOW_PRIORITY      (1 << 5)

/** Stop execution if the device is reset. */
#define PIPE_CONTEXT_LOSE_CONTEXT_ON_RESET (1 << 6)

/**
 * Create a protected context to access protected content (surfaces,
 * textures, ...)
 *
 * This is required to access protected images and surfaces if
 * EGL_EXT_protected_surface is not supported.
 */
#define PIPE_CONTEXT_PROTECTED         (1 << 7)

/**
 * Flags for pipe_context::memory_barrier.
 */
#define PIPE_BARRIER_MAPPED_BUFFER     (1 << 0)
#define PIPE_BARRIER_SHADER_BUFFER     (1 << 1)
#define PIPE_BARRIER_QUERY_BUFFER      (1 << 2)
#define PIPE_BARRIER_VERTEX_BUFFER     (1 << 3)
#define PIPE_BARRIER_INDEX_BUFFER      (1 << 4)
#define PIPE_BARRIER_CONSTANT_BUFFER   (1 << 5)
#define PIPE_BARRIER_INDIRECT_BUFFER   (1 << 6)
#define PIPE_BARRIER_TEXTURE           (1 << 7)
#define PIPE_BARRIER_IMAGE             (1 << 8)
#define PIPE_BARRIER_FRAMEBUFFER       (1 << 9)
#define PIPE_BARRIER_STREAMOUT_BUFFER  (1 << 10)
#define PIPE_BARRIER_GLOBAL_BUFFER     (1 << 11)
#define PIPE_BARRIER_UPDATE_BUFFER     (1 << 12)
#define PIPE_BARRIER_UPDATE_TEXTURE    (1 << 13)
#define PIPE_BARRIER_ALL               ((1 << 14) - 1)

#define PIPE_BARRIER_UPDATE \
   (PIPE_BARRIER_UPDATE_BUFFER | PIPE_BARRIER_UPDATE_TEXTURE)

/**
 * Flags for pipe_context::texture_barrier.
 */
#define PIPE_TEXTURE_BARRIER_SAMPLER      (1 << 0)
#define PIPE_TEXTURE_BARRIER_FRAMEBUFFER  (1 << 1)

/**
 * Resource binding flags -- gallium frontends must specify in advance all
 * the ways a resource might be used.
 */
#define PIPE_BIND_DEPTH_STENCIL        (1 << 0) /* create_surface */
#define PIPE_BIND_RENDER_TARGET        (1 << 1) /* create_surface */
#define PIPE_BIND_BLENDABLE            (1 << 2) /* create_surface */
#define PIPE_BIND_SAMPLER_VIEW         (1 << 3) /* create_sampler_view */
#define PIPE_BIND_VERTEX_BUFFER        (1 << 4) /* set_vertex_buffers */
#define PIPE_BIND_INDEX_BUFFER         (1 << 5) /* draw_elements */
#define PIPE_BIND_CONSTANT_BUFFER      (1 << 6) /* set_constant_buffer */
#define PIPE_BIND_DISPLAY_TARGET       (1 << 7) /* flush_front_buffer */
#define PIPE_BIND_VERTEX_STATE         (1 << 8) /* create_vertex_state */
/* gap */
#define PIPE_BIND_STREAM_OUTPUT        (1 << 10) /* set_stream_output_buffers */
#define PIPE_BIND_CURSOR               (1 << 11) /* mouse cursor */
#define PIPE_BIND_CUSTOM               (1 << 12) /* gallium frontend/winsys usages */
#define PIPE_BIND_GLOBAL               (1 << 13) /* set_global_binding */
#define PIPE_BIND_SHADER_BUFFER        (1 << 14) /* set_shader_buffers */
#define PIPE_BIND_SHADER_IMAGE         (1 << 15) /* set_shader_images */
#define PIPE_BIND_COMPUTE_RESOURCE     (1 << 16) /* set_compute_resources */
#define PIPE_BIND_COMMAND_ARGS_BUFFER  (1 << 17) /* pipe_draw_info.indirect */
#define PIPE_BIND_QUERY_BUFFER         (1 << 18) /* get_query_result_resource */

/**
 * The first two flags above were previously part of the amorphous
 * TEXTURE_USAGE, most of which are now descriptions of the ways a
 * particular texture can be bound to the gallium pipeline.  The two flags
 * below do not fit within that and probably need to be migrated to some
 * other place.
 *
 * Scanout is used to ask for a texture suitable for actual scanout (hence
 * the name), which implies extra layout constraints on some hardware.
 * It may also have some special meaning regarding mouse cursor images.
 *
 * The shared flag is quite underspecified, but certainly isn't a
 * binding flag - it seems more like a message to the winsys to create
 * a shareable allocation.
 *
 * The third flag has been added to be able to force textures to be created
 * in linear mode (no tiling).
 */
#define PIPE_BIND_SCANOUT     (1 << 19) /*  */
#define PIPE_BIND_SHARED      (1 << 20) /* get_texture_handle ??? */
#define PIPE_BIND_LINEAR      (1 << 21)
#define PIPE_BIND_PROTECTED   (1 << 22) /* Resource will be protected/encrypted */
#define PIPE_BIND_SAMPLER_REDUCTION_MINMAX (1 << 23) /* PIPE_CAP_SAMPLER_REDUCTION_MINMAX */
/* Resource is the DRI_PRIME blit destination. Only set on on the render GPU. */
#define PIPE_BIND_PRIME_BLIT_DST (1 << 24)
#define PIPE_BIND_USE_FRONT_RENDERING (1 << 25) /* Resource may be used for frontbuffer rendering */


/**
 * Flags for the driver about resource behaviour:
 */
#define PIPE_RESOURCE_FLAG_MAP_PERSISTENT (1 << 0)
#define PIPE_RESOURCE_FLAG_MAP_COHERENT   (1 << 1)
#define PIPE_RESOURCE_FLAG_TEXTURING_MORE_LIKELY (1 << 2)
#define PIPE_RESOURCE_FLAG_SPARSE                (1 << 3)
#define PIPE_RESOURCE_FLAG_SINGLE_THREAD_USE     (1 << 4)
#define PIPE_RESOURCE_FLAG_ENCRYPTED             (1 << 5)
#define PIPE_RESOURCE_FLAG_DONT_OVER_ALLOCATE    (1 << 6)
#define PIPE_RESOURCE_FLAG_DONT_MAP_DIRECTLY     (1 << 7) /* for small visible VRAM */
#define PIPE_RESOURCE_FLAG_UNMAPPABLE            (1 << 8) /* implies staging transfers due to VK interop */
#define PIPE_RESOURCE_FLAG_DRV_PRIV              (1 << 9) /* driver/winsys private */
#define PIPE_RESOURCE_FLAG_FRONTEND_PRIV         (1 << 24) /* gallium frontend private */

/**
 * Hint about the expected lifecycle of a resource.
 * Sorted according to GPU vs CPU access.
 */
enum pipe_resource_usage {
   PIPE_USAGE_DEFAULT,        /* fast GPU access */
   PIPE_USAGE_IMMUTABLE,      /* fast GPU access, immutable */
   PIPE_USAGE_DYNAMIC,        /* uploaded data is used multiple times */
   PIPE_USAGE_STREAM,         /* uploaded data is used once */
   PIPE_USAGE_STAGING,        /* fast CPU access */
};

/**
 * Primitive types:
 */
enum PACKED pipe_prim_type {
   PIPE_PRIM_POINTS,
   PIPE_PRIM_LINES,
   PIPE_PRIM_LINE_LOOP,
   PIPE_PRIM_LINE_STRIP,
   PIPE_PRIM_TRIANGLES,
   PIPE_PRIM_TRIANGLE_STRIP,
   PIPE_PRIM_TRIANGLE_FAN,
   PIPE_PRIM_QUADS,
   PIPE_PRIM_QUAD_STRIP,
   PIPE_PRIM_POLYGON,
   PIPE_PRIM_LINES_ADJACENCY,
   PIPE_PRIM_LINE_STRIP_ADJACENCY,
   PIPE_PRIM_TRIANGLES_ADJACENCY,
   PIPE_PRIM_TRIANGLE_STRIP_ADJACENCY,
   PIPE_PRIM_PATCHES,
   PIPE_PRIM_MAX,
};

/**
 * Tessellator spacing types
 */
enum pipe_tess_spacing {
   PIPE_TESS_SPACING_FRACTIONAL_ODD,
   PIPE_TESS_SPACING_FRACTIONAL_EVEN,
   PIPE_TESS_SPACING_EQUAL,
};

/**
 * Query object types
 */
enum pipe_query_type {
   PIPE_QUERY_OCCLUSION_COUNTER,
   PIPE_QUERY_OCCLUSION_PREDICATE,
   PIPE_QUERY_OCCLUSION_PREDICATE_CONSERVATIVE,
   PIPE_QUERY_TIMESTAMP,
   PIPE_QUERY_TIMESTAMP_DISJOINT,
   PIPE_QUERY_TIME_ELAPSED,
   PIPE_QUERY_PRIMITIVES_GENERATED,
   PIPE_QUERY_PRIMITIVES_EMITTED,
   PIPE_QUERY_SO_STATISTICS,
   PIPE_QUERY_SO_OVERFLOW_PREDICATE,
   PIPE_QUERY_SO_OVERFLOW_ANY_PREDICATE,
   PIPE_QUERY_GPU_FINISHED,
   PIPE_QUERY_PIPELINE_STATISTICS,
   PIPE_QUERY_PIPELINE_STATISTICS_SINGLE,
   PIPE_QUERY_TYPES,
   /* start of driver queries, see pipe_screen::get_driver_query_info */
   PIPE_QUERY_DRIVER_SPECIFIC = 256,
};

/**
 * Index for PIPE_QUERY_PIPELINE_STATISTICS subqueries.
 */
enum pipe_statistics_query_index {
   PIPE_STAT_QUERY_IA_VERTICES,
   PIPE_STAT_QUERY_IA_PRIMITIVES,
   PIPE_STAT_QUERY_VS_INVOCATIONS,
   PIPE_STAT_QUERY_GS_INVOCATIONS,
   PIPE_STAT_QUERY_GS_PRIMITIVES,
   PIPE_STAT_QUERY_C_INVOCATIONS,
   PIPE_STAT_QUERY_C_PRIMITIVES,
   PIPE_STAT_QUERY_PS_INVOCATIONS,
   PIPE_STAT_QUERY_HS_INVOCATIONS,
   PIPE_STAT_QUERY_DS_INVOCATIONS,
   PIPE_STAT_QUERY_CS_INVOCATIONS,
};

/**
 * Conditional rendering modes
 */
enum pipe_render_cond_flag {
   PIPE_RENDER_COND_WAIT,
   PIPE_RENDER_COND_NO_WAIT,
   PIPE_RENDER_COND_BY_REGION_WAIT,
   PIPE_RENDER_COND_BY_REGION_NO_WAIT,
};

/**
 * Point sprite coord modes
 */
enum pipe_sprite_coord_mode {
   PIPE_SPRITE_COORD_UPPER_LEFT,
   PIPE_SPRITE_COORD_LOWER_LEFT,
};

/**
 * Texture & format swizzles
 */
enum pipe_swizzle {
   PIPE_SWIZZLE_X,
   PIPE_SWIZZLE_Y,
   PIPE_SWIZZLE_Z,
   PIPE_SWIZZLE_W,
   PIPE_SWIZZLE_0,
   PIPE_SWIZZLE_1,
   PIPE_SWIZZLE_NONE,
   PIPE_SWIZZLE_MAX, /**< Number of enums counter (must be last) */
};

/**
 * Viewport swizzles
 */
enum pipe_viewport_swizzle {
   PIPE_VIEWPORT_SWIZZLE_POSITIVE_X,
   PIPE_VIEWPORT_SWIZZLE_NEGATIVE_X,
   PIPE_VIEWPORT_SWIZZLE_POSITIVE_Y,
   PIPE_VIEWPORT_SWIZZLE_NEGATIVE_Y,
   PIPE_VIEWPORT_SWIZZLE_POSITIVE_Z,
   PIPE_VIEWPORT_SWIZZLE_NEGATIVE_Z,
   PIPE_VIEWPORT_SWIZZLE_POSITIVE_W,
   PIPE_VIEWPORT_SWIZZLE_NEGATIVE_W,
};

#define PIPE_TIMEOUT_INFINITE 0xffffffffffffffffull


/**
 * Device reset status.
 */
enum pipe_reset_status
{
   PIPE_NO_RESET,
   PIPE_GUILTY_CONTEXT_RESET,
   PIPE_INNOCENT_CONTEXT_RESET,
   PIPE_UNKNOWN_CONTEXT_RESET,
};


/**
 * Conservative rasterization modes.
 */
enum pipe_conservative_raster_mode
{
   PIPE_CONSERVATIVE_RASTER_OFF,

   /**
    * The post-snap mode means the conservative rasterization occurs after
    * the conversion from floating-point to fixed-point coordinates
    * on the subpixel grid.
    */
   PIPE_CONSERVATIVE_RASTER_POST_SNAP,

   /**
    * The pre-snap mode means the conservative rasterization occurs before
    * the conversion from floating-point to fixed-point coordinates.
    */
   PIPE_CONSERVATIVE_RASTER_PRE_SNAP,
};


/**
 * resource_get_handle flags.
 */
/* Requires pipe_context::flush_resource before external use. */
#define PIPE_HANDLE_USAGE_EXPLICIT_FLUSH     (1 << 0)
/* Expected external use of the resource: */
#define PIPE_HANDLE_USAGE_FRAMEBUFFER_WRITE  (1 << 1)
#define PIPE_HANDLE_USAGE_SHADER_WRITE       (1 << 2)

/**
 * pipe_image_view access flags.
 */
#define PIPE_IMAGE_ACCESS_READ       (1 << 0)
#define PIPE_IMAGE_ACCESS_WRITE      (1 << 1)
#define PIPE_IMAGE_ACCESS_READ_WRITE (PIPE_IMAGE_ACCESS_READ | \
                                      PIPE_IMAGE_ACCESS_WRITE)
#define PIPE_IMAGE_ACCESS_COHERENT   (1 << 2)
#define PIPE_IMAGE_ACCESS_VOLATILE   (1 << 3)

/**
 * Implementation capabilities/limits which are queried through
 * pipe_screen::get_param()
 */
enum pipe_cap
{
   PIPE_CAP_GRAPHICS,
   PIPE_CAP_NPOT_TEXTURES,
   PIPE_CAP_MAX_DUAL_SOURCE_RENDER_TARGETS,
   PIPE_CAP_ANISOTROPIC_FILTER,
   PIPE_CAP_MAX_RENDER_TARGETS,
   PIPE_CAP_OCCLUSION_QUERY,
   PIPE_CAP_QUERY_TIME_ELAPSED,
   PIPE_CAP_TEXTURE_SHADOW_MAP,
   PIPE_CAP_TEXTURE_SWIZZLE,
   PIPE_CAP_MAX_TEXTURE_2D_SIZE,
   PIPE_CAP_MAX_TEXTURE_3D_LEVELS,
   PIPE_CAP_MAX_TEXTURE_CUBE_LEVELS,
   PIPE_CAP_TEXTURE_MIRROR_CLAMP,
   PIPE_CAP_BLEND_EQUATION_SEPARATE,
   PIPE_CAP_MAX_STREAM_OUTPUT_BUFFERS,
   PIPE_CAP_PRIMITIVE_RESTART,
   /** subset of PRIMITIVE_RESTART where the restart index is always the fixed
    * maximum value for the index type
    */
   PIPE_CAP_PRIMITIVE_RESTART_FIXED_INDEX,
   /** blend enables and write masks per rendertarget */
   PIPE_CAP_INDEP_BLEND_ENABLE,
   /** different blend funcs per rendertarget */
   PIPE_CAP_INDEP_BLEND_FUNC,
   PIPE_CAP_MAX_TEXTURE_ARRAY_LAYERS,
   PIPE_CAP_FS_COORD_ORIGIN_UPPER_LEFT,
   PIPE_CAP_FS_COORD_ORIGIN_LOWER_LEFT,
   PIPE_CAP_FS_COORD_PIXEL_CENTER_HALF_INTEGER,
   PIPE_CAP_FS_COORD_PIXEL_CENTER_INTEGER,
   PIPE_CAP_DEPTH_CLIP_DISABLE,
   PIPE_CAP_DEPTH_CLIP_DISABLE_SEPARATE,
   PIPE_CAP_DEPTH_CLAMP_ENABLE,
   PIPE_CAP_SHADER_STENCIL_EXPORT,
   PIPE_CAP_VS_INSTANCEID,
   PIPE_CAP_VERTEX_ELEMENT_INSTANCE_DIVISOR,
   PIPE_CAP_FRAGMENT_COLOR_CLAMPED,
   PIPE_CAP_MIXED_COLORBUFFER_FORMATS,
   PIPE_CAP_SEAMLESS_CUBE_MAP,
   PIPE_CAP_SEAMLESS_CUBE_MAP_PER_TEXTURE,
   PIPE_CAP_MIN_TEXEL_OFFSET,
   PIPE_CAP_MAX_TEXEL_OFFSET,
   PIPE_CAP_CONDITIONAL_RENDER,
   PIPE_CAP_TEXTURE_BARRIER,
   PIPE_CAP_MAX_STREAM_OUTPUT_SEPARATE_COMPONENTS,
   PIPE_CAP_MAX_STREAM_OUTPUT_INTERLEAVED_COMPONENTS,
   PIPE_CAP_STREAM_OUTPUT_PAUSE_RESUME,
   PIPE_CAP_TGSI_CAN_COMPACT_CONSTANTS,
   PIPE_CAP_VERTEX_COLOR_UNCLAMPED,
   PIPE_CAP_VERTEX_COLOR_CLAMPED,
   PIPE_CAP_GLSL_FEATURE_LEVEL,
   PIPE_CAP_GLSL_FEATURE_LEVEL_COMPATIBILITY,
   PIPE_CAP_ESSL_FEATURE_LEVEL,
   PIPE_CAP_QUADS_FOLLOW_PROVOKING_VERTEX_CONVENTION,
   PIPE_CAP_USER_VERTEX_BUFFERS,
   PIPE_CAP_VERTEX_BUFFER_OFFSET_4BYTE_ALIGNED_ONLY,
   PIPE_CAP_VERTEX_BUFFER_STRIDE_4BYTE_ALIGNED_ONLY,
   PIPE_CAP_VERTEX_ELEMENT_SRC_OFFSET_4BYTE_ALIGNED_ONLY,
   PIPE_CAP_VERTEX_ATTRIB_ELEMENT_ALIGNED_ONLY,
   PIPE_CAP_COMPUTE,
   PIPE_CAP_CONSTANT_BUFFER_OFFSET_ALIGNMENT,
   PIPE_CAP_START_INSTANCE,
   PIPE_CAP_QUERY_TIMESTAMP,
   PIPE_CAP_TEXTURE_MULTISAMPLE,
   PIPE_CAP_MIN_MAP_BUFFER_ALIGNMENT,
   PIPE_CAP_CUBE_MAP_ARRAY,
   PIPE_CAP_TEXTURE_BUFFER_OBJECTS,
   PIPE_CAP_TEXTURE_BUFFER_OFFSET_ALIGNMENT,
   PIPE_CAP_BUFFER_SAMPLER_VIEW_RGBA_ONLY,
   PIPE_CAP_TGSI_TEXCOORD,
   PIPE_CAP_TEXTURE_BUFFER_SAMPLER,
   PIPE_CAP_TEXTURE_TRANSFER_MODES,
   PIPE_CAP_QUERY_PIPELINE_STATISTICS,
   PIPE_CAP_TEXTURE_BORDER_COLOR_QUIRK,
   PIPE_CAP_MAX_TEXEL_BUFFER_ELEMENTS_UINT,
   PIPE_CAP_MAX_VIEWPORTS,
   PIPE_CAP_ENDIANNESS,
   PIPE_CAP_MIXED_FRAMEBUFFER_SIZES,
   PIPE_CAP_VS_LAYER_VIEWPORT,
   PIPE_CAP_MAX_GEOMETRY_OUTPUT_VERTICES,
   PIPE_CAP_MAX_GEOMETRY_TOTAL_OUTPUT_COMPONENTS,
   PIPE_CAP_MAX_TEXTURE_GATHER_COMPONENTS,
   PIPE_CAP_TEXTURE_GATHER_SM5,
   PIPE_CAP_BUFFER_MAP_PERSISTENT_COHERENT,
   PIPE_CAP_FAKE_SW_MSAA,
   PIPE_CAP_TEXTURE_QUERY_LOD,
   PIPE_CAP_MIN_TEXTURE_GATHER_OFFSET,
   PIPE_CAP_MAX_TEXTURE_GATHER_OFFSET,
   PIPE_CAP_SAMPLE_SHADING,
   PIPE_CAP_TEXTURE_GATHER_OFFSETS,
   PIPE_CAP_VS_WINDOW_SPACE_POSITION,
   PIPE_CAP_MAX_VERTEX_STREAMS,
   PIPE_CAP_DRAW_INDIRECT,
   PIPE_CAP_FS_FINE_DERIVATIVE,
   PIPE_CAP_VENDOR_ID,
   PIPE_CAP_DEVICE_ID,
   PIPE_CAP_ACCELERATED,
   PIPE_CAP_VIDEO_MEMORY,
   PIPE_CAP_UMA,
   PIPE_CAP_CONDITIONAL_RENDER_INVERTED,
   PIPE_CAP_MAX_VERTEX_ATTRIB_STRIDE,
   PIPE_CAP_SAMPLER_VIEW_TARGET,
   PIPE_CAP_CLIP_HALFZ,
   PIPE_CAP_POLYGON_OFFSET_CLAMP,
   PIPE_CAP_MULTISAMPLE_Z_RESOLVE,
   PIPE_CAP_RESOURCE_FROM_USER_MEMORY,
   PIPE_CAP_RESOURCE_FROM_USER_MEMORY_COMPUTE_ONLY,
   PIPE_CAP_DEVICE_RESET_STATUS_QUERY,
   PIPE_CAP_MAX_SHADER_PATCH_VARYINGS,
   PIPE_CAP_TEXTURE_FLOAT_LINEAR,
   PIPE_CAP_TEXTURE_HALF_FLOAT_LINEAR,
   PIPE_CAP_DEPTH_BOUNDS_TEST,
   PIPE_CAP_TEXTURE_QUERY_SAMPLES,
   PIPE_CAP_FORCE_PERSAMPLE_INTERP,
   PIPE_CAP_SHAREABLE_SHADERS,
   PIPE_CAP_COPY_BETWEEN_COMPRESSED_AND_PLAIN_FORMATS,
   PIPE_CAP_CLEAR_TEXTURE,
   PIPE_CAP_CLEAR_SCISSORED,
   PIPE_CAP_DRAW_PARAMETERS,
   PIPE_CAP_SHADER_PACK_HALF_FLOAT,
   PIPE_CAP_MULTI_DRAW_INDIRECT,
   PIPE_CAP_MULTI_DRAW_INDIRECT_PARAMS,
   PIPE_CAP_MULTI_DRAW_INDIRECT_PARTIAL_STRIDE,
   PIPE_CAP_FS_POSITION_IS_SYSVAL,
   PIPE_CAP_FS_POINT_IS_SYSVAL,
   PIPE_CAP_FS_FACE_IS_INTEGER_SYSVAL,
   PIPE_CAP_SHADER_BUFFER_OFFSET_ALIGNMENT,
   PIPE_CAP_INVALIDATE_BUFFER,
   PIPE_CAP_GENERATE_MIPMAP,
   PIPE_CAP_STRING_MARKER,
   PIPE_CAP_SURFACE_REINTERPRET_BLOCKS,
   PIPE_CAP_QUERY_BUFFER_OBJECT,
   PIPE_CAP_QUERY_MEMORY_INFO,
   PIPE_CAP_PCI_GROUP,
   PIPE_CAP_PCI_BUS,
   PIPE_CAP_PCI_DEVICE,
   PIPE_CAP_PCI_FUNCTION,
   PIPE_CAP_FRAMEBUFFER_NO_ATTACHMENT,
   PIPE_CAP_ROBUST_BUFFER_ACCESS_BEHAVIOR,
   PIPE_CAP_CULL_DISTANCE,
   PIPE_CAP_CULL_DISTANCE_NOCOMBINE,
   PIPE_CAP_SHADER_GROUP_VOTE,
   PIPE_CAP_MAX_WINDOW_RECTANGLES,
   PIPE_CAP_POLYGON_OFFSET_UNITS_UNSCALED,
   PIPE_CAP_VIEWPORT_SUBPIXEL_BITS,
   PIPE_CAP_RASTERIZER_SUBPIXEL_BITS,
   PIPE_CAP_MIXED_COLOR_DEPTH_BITS,
   PIPE_CAP_SHADER_ARRAY_COMPONENTS,
   PIPE_CAP_STREAM_OUTPUT_INTERLEAVE_BUFFERS,
   PIPE_CAP_SHADER_CAN_READ_OUTPUTS,
   PIPE_CAP_NATIVE_FENCE_FD,
   PIPE_CAP_GLSL_TESS_LEVELS_AS_INPUTS,
   PIPE_CAP_FBFETCH,
   PIPE_CAP_LEGACY_MATH_RULES,
   PIPE_CAP_DOUBLES,
   PIPE_CAP_INT64,
   PIPE_CAP_INT64_DIVMOD,
   PIPE_CAP_TGSI_TEX_TXF_LZ,
   PIPE_CAP_SHADER_CLOCK,
   PIPE_CAP_POLYGON_MODE_FILL_RECTANGLE,
   PIPE_CAP_SPARSE_BUFFER_PAGE_SIZE,
   PIPE_CAP_SHADER_BALLOT,
   PIPE_CAP_TES_LAYER_VIEWPORT,
   PIPE_CAP_CAN_BIND_CONST_BUFFER_AS_VERTEX,
   PIPE_CAP_ALLOW_MAPPED_BUFFERS_DURING_EXECUTION,
   PIPE_CAP_POST_DEPTH_COVERAGE,
   PIPE_CAP_BINDLESS_TEXTURE,
   PIPE_CAP_NIR_SAMPLERS_AS_DEREF,
   PIPE_CAP_QUERY_SO_OVERFLOW,
   PIPE_CAP_MEMOBJ,
   PIPE_CAP_LOAD_CONSTBUF,
   PIPE_CAP_TILE_RASTER_ORDER,
   PIPE_CAP_MAX_COMBINED_SHADER_OUTPUT_RESOURCES,
   PIPE_CAP_FRAMEBUFFER_MSAA_CONSTRAINTS,
   PIPE_CAP_SIGNED_VERTEX_BUFFER_OFFSET,
   PIPE_CAP_CONTEXT_PRIORITY_MASK,
   PIPE_CAP_FENCE_SIGNAL,
   PIPE_CAP_CONSTBUF0_FLAGS,
   PIPE_CAP_PACKED_UNIFORMS,
   PIPE_CAP_CONSERVATIVE_RASTER_POST_SNAP_TRIANGLES,
   PIPE_CAP_CONSERVATIVE_RASTER_POST_SNAP_POINTS_LINES,
   PIPE_CAP_CONSERVATIVE_RASTER_PRE_SNAP_TRIANGLES,
   PIPE_CAP_CONSERVATIVE_RASTER_PRE_SNAP_POINTS_LINES,
   PIPE_CAP_MAX_CONSERVATIVE_RASTER_SUBPIXEL_PRECISION_BIAS,
   PIPE_CAP_CONSERVATIVE_RASTER_POST_DEPTH_COVERAGE,
   PIPE_CAP_CONSERVATIVE_RASTER_INNER_COVERAGE,
   PIPE_CAP_PROGRAMMABLE_SAMPLE_LOCATIONS,
   PIPE_CAP_MAX_GS_INVOCATIONS,
   PIPE_CAP_MAX_SHADER_BUFFER_SIZE_UINT,
   PIPE_CAP_TEXTURE_MIRROR_CLAMP_TO_EDGE,
   PIPE_CAP_MAX_COMBINED_SHADER_BUFFERS,
   PIPE_CAP_MAX_COMBINED_HW_ATOMIC_COUNTERS,
   PIPE_CAP_MAX_COMBINED_HW_ATOMIC_COUNTER_BUFFERS,
   PIPE_CAP_MAX_TEXTURE_UPLOAD_MEMORY_BUDGET,
   PIPE_CAP_MAX_VERTEX_ELEMENT_SRC_OFFSET,
   PIPE_CAP_SURFACE_SAMPLE_COUNT,
   PIPE_CAP_IMAGE_ATOMIC_FLOAT_ADD,
   PIPE_CAP_QUERY_PIPELINE_STATISTICS_SINGLE,
   PIPE_CAP_RGB_OVERRIDE_DST_ALPHA_BLEND,
   PIPE_CAP_DEST_SURFACE_SRGB_CONTROL,
   PIPE_CAP_NIR_COMPACT_ARRAYS,
   PIPE_CAP_MAX_VARYINGS,
   PIPE_CAP_COMPUTE_GRID_INFO_LAST_BLOCK,
   PIPE_CAP_COMPUTE_SHADER_DERIVATIVES,
   PIPE_CAP_IMAGE_LOAD_FORMATTED,
   PIPE_CAP_IMAGE_STORE_FORMATTED,
   PIPE_CAP_THROTTLE,
   PIPE_CAP_DMABUF,
   PIPE_CAP_PREFER_COMPUTE_FOR_MULTIMEDIA,
   PIPE_CAP_FRAGMENT_SHADER_INTERLOCK,
   PIPE_CAP_FBFETCH_COHERENT,
   PIPE_CAP_ATOMIC_FLOAT_MINMAX,
   PIPE_CAP_TGSI_DIV,
   PIPE_CAP_FRAGMENT_SHADER_TEXTURE_LOD,
   PIPE_CAP_FRAGMENT_SHADER_DERIVATIVES,
   PIPE_CAP_TEXTURE_SHADOW_LOD,
   PIPE_CAP_SHADER_SAMPLES_IDENTICAL,
   PIPE_CAP_IMAGE_ATOMIC_INC_WRAP,
   PIPE_CAP_PREFER_IMM_ARRAYS_AS_CONSTBUF,
   PIPE_CAP_GL_SPIRV,
   PIPE_CAP_GL_SPIRV_VARIABLE_POINTERS,
   PIPE_CAP_DEMOTE_TO_HELPER_INVOCATION,
   PIPE_CAP_TGSI_TG4_COMPONENT_IN_SWIZZLE,
   PIPE_CAP_FLATSHADE,
   PIPE_CAP_ALPHA_TEST,
   PIPE_CAP_POINT_SIZE_FIXED,
   PIPE_CAP_TWO_SIDED_COLOR,
   PIPE_CAP_CLIP_PLANES,
   PIPE_CAP_MAX_VERTEX_BUFFERS,
   PIPE_CAP_OPENCL_INTEGER_FUNCTIONS,
   PIPE_CAP_INTEGER_MULTIPLY_32X16,
   /* Turn draw, dispatch, blit into NOOP */
   PIPE_CAP_FRONTEND_NOOP,
   PIPE_CAP_NIR_IMAGES_AS_DEREF,
   PIPE_CAP_PACKED_STREAM_OUTPUT,
   PIPE_CAP_VIEWPORT_TRANSFORM_LOWERED,
   PIPE_CAP_PSIZ_CLAMPED,
   PIPE_CAP_GL_BEGIN_END_BUFFER_SIZE,
   PIPE_CAP_VIEWPORT_SWIZZLE,
   PIPE_CAP_SYSTEM_SVM,
   PIPE_CAP_VIEWPORT_MASK,
   PIPE_CAP_ALPHA_TO_COVERAGE_DITHER_CONTROL,
   PIPE_CAP_MAP_UNSYNCHRONIZED_THREAD_SAFE,
   PIPE_CAP_GLSL_ZERO_INIT,
   PIPE_CAP_BLEND_EQUATION_ADVANCED,
   PIPE_CAP_NIR_ATOMICS_AS_DEREF,
   PIPE_CAP_NO_CLIP_ON_COPY_TEX,
   PIPE_CAP_MAX_TEXTURE_MB,
   PIPE_CAP_SHADER_ATOMIC_INT64,
   /** For EGL_EXT_protected_surface */
   PIPE_CAP_DEVICE_PROTECTED_SURFACE,
   PIPE_CAP_PREFER_REAL_BUFFER_IN_CONSTBUF0,
   PIPE_CAP_GL_CLAMP,
   PIPE_CAP_TEXRECT,
   PIPE_CAP_SAMPLER_REDUCTION_MINMAX,
   PIPE_CAP_SAMPLER_REDUCTION_MINMAX_ARB,
   PIPE_CAP_ALLOW_DYNAMIC_VAO_FASTPATH,
   PIPE_CAP_EMULATE_NONFIXED_PRIMITIVE_RESTART,
   PIPE_CAP_SUPPORTED_PRIM_MODES,
   PIPE_CAP_SUPPORTED_PRIM_MODES_WITH_RESTART,
   PIPE_CAP_PREFER_BACK_BUFFER_REUSE,
   PIPE_CAP_DRAW_VERTEX_STATE,
   PIPE_CAP_PREFER_POT_ALIGNED_VARYINGS,
   PIPE_CAP_MAX_SPARSE_TEXTURE_SIZE,
   PIPE_CAP_MAX_SPARSE_3D_TEXTURE_SIZE,
   PIPE_CAP_MAX_SPARSE_ARRAY_TEXTURE_LAYERS,
   PIPE_CAP_SPARSE_TEXTURE_FULL_ARRAY_CUBE_MIPMAPS,
   PIPE_CAP_QUERY_SPARSE_TEXTURE_RESIDENCY,
   PIPE_CAP_CLAMP_SPARSE_TEXTURE_LOD,
   PIPE_CAP_ALLOW_DRAW_OUT_OF_ORDER,
   PIPE_CAP_MAX_CONSTANT_BUFFER_SIZE_UINT,
   PIPE_CAP_HARDWARE_GL_SELECT,
   PIPE_CAP_DITHERING,
   PIPE_CAP_FBFETCH_ZS,
   PIPE_CAP_TIMELINE_SEMAPHORE_IMPORT,
   PIPE_CAP_QUERY_TIMESTAMP_BITS,
   /** For EGL_EXT_protected_content */
   PIPE_CAP_DEVICE_PROTECTED_CONTEXT,
   PIPE_CAP_ALLOW_GLTHREAD_BUFFER_SUBDATA_OPT,

   PIPE_CAP_VALIDATE_ALL_DIRTY_STATES,
   PIPE_CAP_LAST,
   /* XXX do not add caps after PIPE_CAP_LAST! */
};

enum pipe_texture_transfer_mode {
   PIPE_TEXTURE_TRANSFER_DEFAULT = 0,
   PIPE_TEXTURE_TRANSFER_BLIT = (1 << 0),
   PIPE_TEXTURE_TRANSFER_COMPUTE = (1 << 1),
};

/**
 * Possible bits for PIPE_CAP_CONTEXT_PRIORITY_MASK param, which should
 * return a bitmask of the supported priorities.  If the driver does not
 * support prioritized contexts, it can return 0.
 *
 * Note that these match __EGL_CONTEXT_PRIORITY_*_BIT.
 */
#define PIPE_CONTEXT_PRIORITY_LOW     (1 << 0)
#define PIPE_CONTEXT_PRIORITY_MEDIUM  (1 << 1)
#define PIPE_CONTEXT_PRIORITY_HIGH    (1 << 2)

enum pipe_quirk_texture_border_color_swizzle {
   PIPE_QUIRK_TEXTURE_BORDER_COLOR_SWIZZLE_NV50 = (1 << 0),
   PIPE_QUIRK_TEXTURE_BORDER_COLOR_SWIZZLE_R600 = (1 << 1),
   PIPE_QUIRK_TEXTURE_BORDER_COLOR_SWIZZLE_FREEDRENO = (1 << 2),
   PIPE_QUIRK_TEXTURE_BORDER_COLOR_SWIZZLE_ALPHA_NOT_W = (1 << 3),
};

enum pipe_endian
{
   PIPE_ENDIAN_LITTLE = 0,
   PIPE_ENDIAN_BIG = 1,
#if UTIL_ARCH_LITTLE_ENDIAN
   PIPE_ENDIAN_NATIVE = PIPE_ENDIAN_LITTLE
#elif UTIL_ARCH_BIG_ENDIAN
   PIPE_ENDIAN_NATIVE = PIPE_ENDIAN_BIG
#endif
};

/**
 * Implementation limits which are queried through
 * pipe_screen::get_paramf()
 */
enum pipe_capf
{
   PIPE_CAPF_MIN_LINE_WIDTH,
   PIPE_CAPF_MIN_LINE_WIDTH_AA,
   PIPE_CAPF_MAX_LINE_WIDTH,
   PIPE_CAPF_MAX_LINE_WIDTH_AA,
   PIPE_CAPF_LINE_WIDTH_GRANULARITY,
   PIPE_CAPF_MIN_POINT_SIZE,
   PIPE_CAPF_MIN_POINT_SIZE_AA,
   PIPE_CAPF_MAX_POINT_SIZE,
   PIPE_CAPF_MAX_POINT_SIZE_AA,
   PIPE_CAPF_POINT_SIZE_GRANULARITY,
   PIPE_CAPF_MAX_TEXTURE_ANISOTROPY,
   PIPE_CAPF_MAX_TEXTURE_LOD_BIAS,
   PIPE_CAPF_MIN_CONSERVATIVE_RASTER_DILATE,
   PIPE_CAPF_MAX_CONSERVATIVE_RASTER_DILATE,
   PIPE_CAPF_CONSERVATIVE_RASTER_DILATE_GRANULARITY,
};

/** Shader caps not specific to any single stage */
enum pipe_shader_cap
{
   PIPE_SHADER_CAP_MAX_INSTRUCTIONS, /* if 0, it means the stage is unsupported */
   PIPE_SHADER_CAP_MAX_ALU_INSTRUCTIONS,
   PIPE_SHADER_CAP_MAX_TEX_INSTRUCTIONS,
   PIPE_SHADER_CAP_MAX_TEX_INDIRECTIONS,
   PIPE_SHADER_CAP_MAX_CONTROL_FLOW_DEPTH,
   PIPE_SHADER_CAP_MAX_INPUTS,
   PIPE_SHADER_CAP_MAX_OUTPUTS,
   PIPE_SHADER_CAP_MAX_CONST_BUFFER0_SIZE,
   PIPE_SHADER_CAP_MAX_CONST_BUFFERS,
   PIPE_SHADER_CAP_MAX_TEMPS,
   /* boolean caps */
   PIPE_SHADER_CAP_CONT_SUPPORTED,
   PIPE_SHADER_CAP_INDIRECT_INPUT_ADDR,
   PIPE_SHADER_CAP_INDIRECT_OUTPUT_ADDR,
   PIPE_SHADER_CAP_INDIRECT_TEMP_ADDR,
   PIPE_SHADER_CAP_INDIRECT_CONST_ADDR,
   PIPE_SHADER_CAP_SUBROUTINES, /* BGNSUB, ENDSUB, CAL, RET */
   PIPE_SHADER_CAP_INTEGERS,
   PIPE_SHADER_CAP_INT64_ATOMICS,
   PIPE_SHADER_CAP_FP16,
   PIPE_SHADER_CAP_FP16_DERIVATIVES,
   PIPE_SHADER_CAP_FP16_CONST_BUFFERS,
   PIPE_SHADER_CAP_INT16,
   PIPE_SHADER_CAP_GLSL_16BIT_CONSTS,
   PIPE_SHADER_CAP_MAX_TEXTURE_SAMPLERS,
   PIPE_SHADER_CAP_PREFERRED_IR,
   PIPE_SHADER_CAP_TGSI_SQRT_SUPPORTED,
   PIPE_SHADER_CAP_MAX_SAMPLER_VIEWS,
   PIPE_SHADER_CAP_DROUND_SUPPORTED, /* all rounding modes */
   PIPE_SHADER_CAP_DFRACEXP_DLDEXP_SUPPORTED,
   PIPE_SHADER_CAP_TGSI_ANY_INOUT_DECL_RANGE,
   PIPE_SHADER_CAP_MAX_SHADER_BUFFERS,
   PIPE_SHADER_CAP_SUPPORTED_IRS,
   PIPE_SHADER_CAP_MAX_SHADER_IMAGES,
   PIPE_SHADER_CAP_LDEXP_SUPPORTED,
   PIPE_SHADER_CAP_MAX_HW_ATOMIC_COUNTERS,
   PIPE_SHADER_CAP_MAX_HW_ATOMIC_COUNTER_BUFFERS,
};

/**
 * Shader intermediate representation.
 *
 * Note that if the driver requests something other than TGSI, it must
 * always be prepared to receive TGSI in addition to its preferred IR.
 * If the driver requests TGSI as its preferred IR, it will *always*
 * get TGSI.
 *
 * Note that PIPE_SHADER_IR_TGSI should be zero for backwards compat with
 * gallium frontends that only understand TGSI.
 */
enum pipe_shader_ir
{
   PIPE_SHADER_IR_TGSI = 0,
   PIPE_SHADER_IR_NATIVE,
   PIPE_SHADER_IR_NIR,
   PIPE_SHADER_IR_NIR_SERIALIZED,
};

/**
 * Compute-specific implementation capability.  They can be queried
 * using pipe_screen::get_compute_param.
 */
enum pipe_compute_cap
{
   PIPE_COMPUTE_CAP_ADDRESS_BITS,
   PIPE_COMPUTE_CAP_IR_TARGET,
   PIPE_COMPUTE_CAP_GRID_DIMENSION,
   PIPE_COMPUTE_CAP_MAX_GRID_SIZE,
   PIPE_COMPUTE_CAP_MAX_BLOCK_SIZE,
   PIPE_COMPUTE_CAP_MAX_THREADS_PER_BLOCK,
   PIPE_COMPUTE_CAP_MAX_GLOBAL_SIZE,
   PIPE_COMPUTE_CAP_MAX_LOCAL_SIZE,
   PIPE_COMPUTE_CAP_MAX_PRIVATE_SIZE,
   PIPE_COMPUTE_CAP_MAX_INPUT_SIZE,
   PIPE_COMPUTE_CAP_MAX_MEM_ALLOC_SIZE,
   PIPE_COMPUTE_CAP_MAX_CLOCK_FREQUENCY,
   PIPE_COMPUTE_CAP_MAX_COMPUTE_UNITS,
   PIPE_COMPUTE_CAP_IMAGES_SUPPORTED,
   PIPE_COMPUTE_CAP_SUBGROUP_SIZE,
   PIPE_COMPUTE_CAP_MAX_VARIABLE_THREADS_PER_BLOCK,
};

/**
 * Resource parameters. They can be queried using
 * pipe_screen::get_resource_param.
 */
enum pipe_resource_param
{
   PIPE_RESOURCE_PARAM_NPLANES,
   PIPE_RESOURCE_PARAM_STRIDE,
   PIPE_RESOURCE_PARAM_OFFSET,
   PIPE_RESOURCE_PARAM_MODIFIER,
   PIPE_RESOURCE_PARAM_HANDLE_TYPE_SHARED,
   PIPE_RESOURCE_PARAM_HANDLE_TYPE_KMS,
   PIPE_RESOURCE_PARAM_HANDLE_TYPE_FD,
   PIPE_RESOURCE_PARAM_LAYER_STRIDE,
};

/**
 * Types of parameters for pipe_context::set_context_param.
 */
enum pipe_context_param
{
   /* A hint for the driver that it should pin its execution threads to
    * a group of cores sharing a specific L3 cache if the CPU has multiple
    * L3 caches. This is needed for good multithreading performance on
    * AMD Zen CPUs. "value" is the L3 cache index. Drivers that don't have
    * any internal threads or don't run on affected CPUs can ignore this.
    */
   PIPE_CONTEXT_PARAM_PIN_THREADS_TO_L3_CACHE,
};

/**
 * Composite query types
 */

/**
 * Query result for PIPE_QUERY_SO_STATISTICS.
 */
struct pipe_query_data_so_statistics
{
   uint64_t num_primitives_written;
   uint64_t primitives_storage_needed;
};

/**
 * Query result for PIPE_QUERY_TIMESTAMP_DISJOINT.
 */
struct pipe_query_data_timestamp_disjoint
{
   uint64_t frequency;
   bool     disjoint;
};

/**
 * Query result for PIPE_QUERY_PIPELINE_STATISTICS.
 */
struct pipe_query_data_pipeline_statistics
{
   union {
      struct {
         uint64_t ia_vertices;    /**< Num vertices read by the vertex fetcher. */
         uint64_t ia_primitives;  /**< Num primitives read by the vertex fetcher. */
         uint64_t vs_invocations; /**< Num vertex shader invocations. */
         uint64_t gs_invocations; /**< Num geometry shader invocations. */
         uint64_t gs_primitives;  /**< Num primitives output by a geometry shader. */
         uint64_t c_invocations;  /**< Num primitives sent to the rasterizer. */
         uint64_t c_primitives;   /**< Num primitives that were rendered. */
         uint64_t ps_invocations; /**< Num pixel shader invocations. */
         uint64_t hs_invocations; /**< Num hull shader invocations. */
         uint64_t ds_invocations; /**< Num domain shader invocations. */
         uint64_t cs_invocations; /**< Num compute shader invocations. */
      };
      uint64_t counters[11];
   };
};

/**
 * For batch queries.
 */
union pipe_numeric_type_union
{
   uint64_t u64;
   uint32_t u32;
   float f;
};

/**
 * Query result (returned by pipe_context::get_query_result).
 */
union pipe_query_result
{
   /* PIPE_QUERY_OCCLUSION_PREDICATE */
   /* PIPE_QUERY_OCCLUSION_PREDICATE_CONSERVATIVE */
   /* PIPE_QUERY_SO_OVERFLOW_PREDICATE */
   /* PIPE_QUERY_SO_OVERFLOW_ANY_PREDICATE */
   /* PIPE_QUERY_GPU_FINISHED */
   bool b;

   /* PIPE_QUERY_OCCLUSION_COUNTER */
   /* PIPE_QUERY_TIMESTAMP */
   /* PIPE_QUERY_TIME_ELAPSED */
   /* PIPE_QUERY_PRIMITIVES_GENERATED */
   /* PIPE_QUERY_PRIMITIVES_EMITTED */
   /* PIPE_DRIVER_QUERY_TYPE_UINT64 */
   /* PIPE_DRIVER_QUERY_TYPE_BYTES */
   /* PIPE_DRIVER_QUERY_TYPE_MICROSECONDS */
   /* PIPE_DRIVER_QUERY_TYPE_HZ */
   uint64_t u64;

   /* PIPE_DRIVER_QUERY_TYPE_UINT */
   uint32_t u32;

   /* PIPE_DRIVER_QUERY_TYPE_FLOAT */
   /* PIPE_DRIVER_QUERY_TYPE_PERCENTAGE */
   float f;

   /* PIPE_QUERY_SO_STATISTICS */
   struct pipe_query_data_so_statistics so_statistics;

   /* PIPE_QUERY_TIMESTAMP_DISJOINT */
   struct pipe_query_data_timestamp_disjoint timestamp_disjoint;

   /* PIPE_QUERY_PIPELINE_STATISTICS */
   struct pipe_query_data_pipeline_statistics pipeline_statistics;

   /* batch queries (variable length) */
   union pipe_numeric_type_union batch[1];
};

enum pipe_query_value_type
{
   PIPE_QUERY_TYPE_I32,
   PIPE_QUERY_TYPE_U32,
   PIPE_QUERY_TYPE_I64,
   PIPE_QUERY_TYPE_U64,
};

enum pipe_query_flags
{
   PIPE_QUERY_WAIT = (1 << 0),
   PIPE_QUERY_PARTIAL = (1 << 1),
};

union pipe_color_union
{
   float f[4];
   int i[4];
   unsigned int ui[4];
};

enum pipe_driver_query_type
{
   PIPE_DRIVER_QUERY_TYPE_UINT64,
   PIPE_DRIVER_QUERY_TYPE_UINT,
   PIPE_DRIVER_QUERY_TYPE_FLOAT,
   PIPE_DRIVER_QUERY_TYPE_PERCENTAGE,
   PIPE_DRIVER_QUERY_TYPE_BYTES,
   PIPE_DRIVER_QUERY_TYPE_MICROSECONDS,
   PIPE_DRIVER_QUERY_TYPE_HZ,
   PIPE_DRIVER_QUERY_TYPE_DBM,
   PIPE_DRIVER_QUERY_TYPE_TEMPERATURE,
   PIPE_DRIVER_QUERY_TYPE_VOLTS,
   PIPE_DRIVER_QUERY_TYPE_AMPS,
   PIPE_DRIVER_QUERY_TYPE_WATTS,
};

/* Whether an average value per frame or a cumulative value should be
 * displayed.
 */
enum pipe_driver_query_result_type
{
   PIPE_DRIVER_QUERY_RESULT_TYPE_AVERAGE,
   PIPE_DRIVER_QUERY_RESULT_TYPE_CUMULATIVE,
};

/**
 * Some hardware requires some hardware-specific queries to be submitted
 * as batched queries. The corresponding query objects are created using
 * create_batch_query, and at most one such query may be active at
 * any time.
 */
#define PIPE_DRIVER_QUERY_FLAG_BATCH     (1 << 0)

/* Do not list this query in the HUD. */
#define PIPE_DRIVER_QUERY_FLAG_DONT_LIST (1 << 1)

struct pipe_driver_query_info
{
   const char *name;
   unsigned query_type; /* PIPE_QUERY_DRIVER_SPECIFIC + i */
   union pipe_numeric_type_union max_value; /* max value that can be returned */
   enum pipe_driver_query_type type;
   enum pipe_driver_query_result_type result_type;
   unsigned group_id;
   unsigned flags;
};

struct pipe_driver_query_group_info
{
   const char *name;
   unsigned max_active_queries;
   unsigned num_queries;
};

enum pipe_fd_type
{
   PIPE_FD_TYPE_NATIVE_SYNC,
   PIPE_FD_TYPE_SYNCOBJ,
   PIPE_FD_TYPE_TIMELINE_SEMAPHORE,
};

/**
 * counter type and counter data type enums used by INTEL_performance_query
 * APIs in gallium drivers.
 */
enum pipe_perf_counter_type
{
   PIPE_PERF_COUNTER_TYPE_EVENT,
   PIPE_PERF_COUNTER_TYPE_DURATION_NORM,
   PIPE_PERF_COUNTER_TYPE_DURATION_RAW,
   PIPE_PERF_COUNTER_TYPE_THROUGHPUT,
   PIPE_PERF_COUNTER_TYPE_RAW,
   PIPE_PERF_COUNTER_TYPE_TIMESTAMP,
};

enum pipe_perf_counter_data_type
{
   PIPE_PERF_COUNTER_DATA_TYPE_BOOL32,
   PIPE_PERF_COUNTER_DATA_TYPE_UINT32,
   PIPE_PERF_COUNTER_DATA_TYPE_UINT64,
   PIPE_PERF_COUNTER_DATA_TYPE_FLOAT,
   PIPE_PERF_COUNTER_DATA_TYPE_DOUBLE,
};

#define PIPE_UUID_SIZE 16
#define PIPE_LUID_SIZE 8

#if DETECT_OS_UNIX
#define PIPE_MEMORY_FD
#endif

#ifdef __cplusplus
}
#endif

#endif
