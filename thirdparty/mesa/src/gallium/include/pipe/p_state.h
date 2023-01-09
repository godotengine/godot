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


/**
 * @file
 *
 * Abstract graphics pipe state objects.
 *
 * Basic notes:
 *   1. Want compact representations, so we use bitfields.
 *   2. Put bitfields before other (GLfloat) fields.
 *   3. enum bitfields need to be at least one bit extra in size so the most
 *      significant bit is zero.  MSVC treats enums as signed so if the high
 *      bit is set, the value will be interpreted as a negative number.
 *      That causes trouble in various places.
 */


#ifndef PIPE_STATE_H
#define PIPE_STATE_H

#include "util/u_memory.h"

#include "p_compiler.h"
#include "p_defines.h"
#include "util/format/u_formats.h"


#ifdef __cplusplus
extern "C" {
#endif

/**
 * Implementation limits
 */
#define PIPE_MAX_ATTRIBS          32
#define PIPE_MAX_CLIP_PLANES       8
#define PIPE_MAX_COLOR_BUFS        8
#define PIPE_MAX_CONSTANT_BUFFERS 32
#define PIPE_MAX_SAMPLERS         32
#define PIPE_MAX_SHADER_INPUTS    80 /* 32 GENERIC + 32 PATCH + 16 others */
#define PIPE_MAX_SHADER_OUTPUTS   80 /* 32 GENERIC + 32 PATCH + 16 others */
#define PIPE_MAX_SHADER_SAMPLER_VIEWS 128
#define PIPE_MAX_SHADER_BUFFERS   32
#define PIPE_MAX_SHADER_IMAGES    64
#define PIPE_MAX_TEXTURE_LEVELS   16
#define PIPE_MAX_SO_BUFFERS        4
#define PIPE_MAX_SO_OUTPUTS       64
#define PIPE_MAX_VIEWPORTS        16
#define PIPE_MAX_CLIP_OR_CULL_DISTANCE_COUNT 8
#define PIPE_MAX_CLIP_OR_CULL_DISTANCE_ELEMENT_COUNT 2
#define PIPE_MAX_WINDOW_RECTANGLES 8
#define PIPE_MAX_SAMPLE_LOCATION_GRID_SIZE 4

#define PIPE_MAX_HW_ATOMIC_BUFFERS 32
#define PIPE_MAX_VERTEX_STREAMS   4

struct pipe_reference
{
   int32_t count; /* atomic */
};



/**
 * Primitive (point/line/tri) rasterization info
 */
struct pipe_rasterizer_state
{
   unsigned flatshade:1;
   unsigned light_twoside:1;
   unsigned clamp_vertex_color:1;
   unsigned clamp_fragment_color:1;
   unsigned front_ccw:1;
   unsigned cull_face:2;      /**< PIPE_FACE_x */
   unsigned fill_front:2;     /**< PIPE_POLYGON_MODE_x */
   unsigned fill_back:2;      /**< PIPE_POLYGON_MODE_x */
   unsigned offset_point:1;
   unsigned offset_line:1;
   unsigned offset_tri:1;
   unsigned scissor:1;
   unsigned poly_smooth:1;
   unsigned poly_stipple_enable:1;
   unsigned point_smooth:1;
   unsigned sprite_coord_mode:1;     /**< PIPE_SPRITE_COORD_ */
   unsigned point_quad_rasterization:1; /** points rasterized as quads or points */
   unsigned point_tri_clip:1; /** large points clipped as tris or points */
   unsigned point_size_per_vertex:1; /**< size computed in vertex shader */
   unsigned multisample:1;         /* XXX maybe more ms state in future */
   unsigned no_ms_sample_mask_out:1;
   unsigned force_persample_interp:1;
   unsigned line_smooth:1;
   unsigned line_stipple_enable:1;
   unsigned line_last_pixel:1;
   unsigned line_rectangular:1; /** lines rasterized as rectangles or parallelograms */
   unsigned conservative_raster_mode:2; /**< PIPE_CONSERVATIVE_RASTER_x */

   /**
    * Use the first vertex of a primitive as the provoking vertex for
    * flat shading.
    */
   unsigned flatshade_first:1;

   unsigned half_pixel_center:1;
   unsigned bottom_edge_rule:1;

   /*
    * Conservative rasterization subpixel precision bias in bits
    */
   unsigned subpixel_precision_x:4;
   unsigned subpixel_precision_y:4;

   /**
    * When true, rasterization is disabled and no pixels are written.
    * This only makes sense with the Stream Out functionality.
    */
   unsigned rasterizer_discard:1;

   /**
    * Exposed by PIPE_CAP_TILE_RASTER_ORDER.  When true,
    * tile_raster_order_increasing_* indicate the order that the rasterizer
    * should render tiles, to meet the requirements of
    * GL_MESA_tile_raster_order.
    */
   unsigned tile_raster_order_fixed:1;
   unsigned tile_raster_order_increasing_x:1;
   unsigned tile_raster_order_increasing_y:1;

   /**
    * When false, depth clipping is disabled and the depth value will be
    * clamped later at the per-pixel level before depth testing.
    * This depends on PIPE_CAP_DEPTH_CLIP_DISABLE.
    *
    * If PIPE_CAP_DEPTH_CLIP_DISABLE_SEPARATE is unsupported, depth_clip_near
    * is equal to depth_clip_far.
    */
   unsigned depth_clip_near:1;
   unsigned depth_clip_far:1;

   /**
    * When true, depth clamp is enabled.
    * If PIPE_CAP_DEPTH_CLAMP_ENABLE is unsupported, this is always the inverse
    * of depth_clip_far.
    */
   unsigned depth_clamp:1;

   /**
    * When true clip space in the z axis goes from [0..1] (D3D).  When false
    * [-1, 1] (GL).
    *
    * NOTE: D3D will always use depth clamping.
    */
   unsigned clip_halfz:1;

   /**
    * When true do not scale offset_units and use same rules for unorm and
    * float depth buffers (D3D9). When false use GL/D3D1X behaviour.
    * This depends on PIPE_CAP_POLYGON_OFFSET_UNITS_UNSCALED.
    */
   unsigned offset_units_unscaled:1;

   /**
    * Depth values output from fragment shader may be outside 0..1.
    * These have to be clamped for use with UNORM buffers.
    * Vulkan can allow this with an extension,
    * GL could with NV_depth_buffer_float, but GLES doesn't.
    */
   unsigned unclamped_fragment_depth_values:1;

   /**
    * Enable bits for clipping half-spaces.
    * This applies to both user clip planes and shader clip distances.
    * Note that if the bound shader exports any clip distances, these
    * replace all user clip planes, and clip half-spaces enabled here
    * but not written by the shader count as disabled.
    */
   unsigned clip_plane_enable:PIPE_MAX_CLIP_PLANES;

   unsigned line_stipple_factor:8;  /**< [1..256] actually */
   unsigned line_stipple_pattern:16;

   /**
    * Replace the given TEXCOORD inputs with point coordinates, max. 8 inputs.
    * If TEXCOORD (including PCOORD) are unsupported, replace GENERIC inputs
    * instead. Max. 9 inputs: 8x GENERIC to emulate TEXCOORD, and 1x GENERIC
    * to emulate PCOORD.
    */
   uint16_t sprite_coord_enable; /* 0-7: TEXCOORD/GENERIC, 8: PCOORD */

   float line_width;
   float point_size;           /**< used when no per-vertex size */
   float offset_units;
   float offset_scale;
   float offset_clamp;
   float conservative_raster_dilate;
};


struct pipe_poly_stipple
{
   unsigned stipple[32];
};


struct pipe_viewport_state
{
   float scale[3];
   float translate[3];
   enum pipe_viewport_swizzle swizzle_x:8;
   enum pipe_viewport_swizzle swizzle_y:8;
   enum pipe_viewport_swizzle swizzle_z:8;
   enum pipe_viewport_swizzle swizzle_w:8;
};


struct pipe_scissor_state
{
   unsigned minx:16;
   unsigned miny:16;
   unsigned maxx:16;
   unsigned maxy:16;
};


struct pipe_clip_state
{
   float ucp[PIPE_MAX_CLIP_PLANES][4];
};

/**
 * A single output for vertex transform feedback.
 */
struct pipe_stream_output
{
   unsigned register_index:6;  /**< 0 to 63 (OUT index) */
   unsigned start_component:2; /** 0 to 3 */
   unsigned num_components:3;  /** 1 to 4 */
   unsigned output_buffer:3;   /**< 0 to PIPE_MAX_SO_BUFFERS */
   unsigned dst_offset:16;     /**< offset into the buffer in dwords */
   unsigned stream:2;          /**< 0 to 3 */
};

/**
 * Stream output for vertex transform feedback.
 */
struct pipe_stream_output_info
{
   unsigned num_outputs;
   /** stride for an entire vertex for each buffer in dwords */
   uint16_t stride[PIPE_MAX_SO_BUFFERS];

   /**
    * Array of stream outputs, in the order they are to be written in.
    * Selected components are tightly packed into the output buffer.
    */
   struct pipe_stream_output output[PIPE_MAX_SO_OUTPUTS];
};

/**
 * The 'type' parameter identifies whether the shader state contains TGSI
 * tokens, etc.  If the driver returns 'PIPE_SHADER_IR_TGSI' for the
 * 'PIPE_SHADER_CAP_PREFERRED_IR' shader param, the ir will *always* be
 * 'PIPE_SHADER_IR_TGSI' and the tokens ptr will be valid.  If the driver
 * requests a different 'pipe_shader_ir' type, then it must check the 'type'
 * enum to see if it is getting TGSI tokens or its preferred IR.
 *
 * TODO pipe_compute_state should probably get similar treatment to handle
 * multiple IR's in a cleaner way..
 *
 * NOTE: since it is expected that the consumer will want to perform
 * additional passes on the nir_shader, the driver takes ownership of
 * the nir_shader.  If gallium frontends need to hang on to the IR (for
 * example, variant management), it should use nir_shader_clone().
 */
struct pipe_shader_state
{
   enum pipe_shader_ir type;
   /* TODO move tokens into union. */
   const struct tgsi_token *tokens;
   union {
      void *native;
      void *nir;
   } ir;
   struct pipe_stream_output_info stream_output;
};

static inline void
pipe_shader_state_from_tgsi(struct pipe_shader_state *state,
                            const struct tgsi_token *tokens)
{
   state->type = PIPE_SHADER_IR_TGSI;
   state->tokens = tokens;
   memset(&state->stream_output, 0, sizeof(state->stream_output));
}


struct pipe_stencil_state
{
   unsigned enabled:1;  /**< stencil[0]: stencil enabled, stencil[1]: two-side enabled */
   unsigned func:3;     /**< PIPE_FUNC_x */
   unsigned fail_op:3;  /**< PIPE_STENCIL_OP_x */
   unsigned zpass_op:3; /**< PIPE_STENCIL_OP_x */
   unsigned zfail_op:3; /**< PIPE_STENCIL_OP_x */
   unsigned valuemask:8;
   unsigned writemask:8;
};


struct pipe_depth_stencil_alpha_state
{
   struct pipe_stencil_state stencil[2]; /**< [0] = front, [1] = back */

   unsigned alpha_enabled:1;         /**< alpha test enabled? */
   unsigned alpha_func:3;            /**< PIPE_FUNC_x */

   unsigned depth_enabled:1;         /**< depth test enabled? */
   unsigned depth_writemask:1;       /**< allow depth buffer writes? */
   unsigned depth_func:3;            /**< depth test func (PIPE_FUNC_x) */
   unsigned depth_bounds_test:1;     /**< depth bounds test enabled? */

   float alpha_ref_value;            /**< reference value */
   double depth_bounds_min;          /**< minimum depth bound */
   double depth_bounds_max;          /**< maximum depth bound */
};


struct pipe_rt_blend_state
{
   unsigned blend_enable:1;

   unsigned rgb_func:3;          /**< PIPE_BLEND_x */
   unsigned rgb_src_factor:5;    /**< PIPE_BLENDFACTOR_x */
   unsigned rgb_dst_factor:5;    /**< PIPE_BLENDFACTOR_x */

   unsigned alpha_func:3;        /**< PIPE_BLEND_x */
   unsigned alpha_src_factor:5;  /**< PIPE_BLENDFACTOR_x */
   unsigned alpha_dst_factor:5;  /**< PIPE_BLENDFACTOR_x */

   unsigned colormask:4;         /**< bitmask of PIPE_MASK_R/G/B/A */
};


struct pipe_blend_state
{
   unsigned independent_blend_enable:1;
   unsigned logicop_enable:1;
   unsigned logicop_func:4;      /**< PIPE_LOGICOP_x */
   unsigned dither:1;
   unsigned alpha_to_coverage:1;
   unsigned alpha_to_coverage_dither:1;
   unsigned alpha_to_one:1;
   unsigned max_rt:3;            /* index of max rt, Ie. # of cbufs minus 1 */
   unsigned advanced_blend_func:4;
   struct pipe_rt_blend_state rt[PIPE_MAX_COLOR_BUFS];
};


struct pipe_blend_color
{
   float color[4];
};


struct pipe_stencil_ref
{
   ubyte ref_value[2];
};


/**
 * Note that pipe_surfaces are "texture views for rendering"
 * and so in the case of ARB_framebuffer_no_attachment there
 * is no pipe_surface state available such that we may
 * extract the number of samples and layers.
 */
struct pipe_framebuffer_state
{
   uint16_t width, height;
   uint16_t layers;  /**< Number of layers  in a no-attachment framebuffer */
   ubyte samples; /**< Number of samples in a no-attachment framebuffer */

   /** multiple color buffers for multiple render targets */
   ubyte nr_cbufs;
   struct pipe_surface *cbufs[PIPE_MAX_COLOR_BUFS];

   struct pipe_surface *zsbuf;      /**< Z/stencil buffer */
};


/**
 * Texture sampler state.
 */
struct pipe_sampler_state
{
   unsigned wrap_s:3;            /**< PIPE_TEX_WRAP_x */
   unsigned wrap_t:3;            /**< PIPE_TEX_WRAP_x */
   unsigned wrap_r:3;            /**< PIPE_TEX_WRAP_x */
   unsigned min_img_filter:1;    /**< PIPE_TEX_FILTER_x */
   unsigned min_mip_filter:2;    /**< PIPE_TEX_MIPFILTER_x */
   unsigned mag_img_filter:1;    /**< PIPE_TEX_FILTER_x */
   unsigned compare_mode:1;      /**< PIPE_TEX_COMPARE_x */
   unsigned compare_func:3;      /**< PIPE_FUNC_x */
   unsigned unnormalized_coords:1; /**< Are coords normalized to [0,1]? */
   unsigned max_anisotropy:5;
   unsigned seamless_cube_map:1;
   unsigned border_color_is_integer:1;
   unsigned reduction_mode:2;    /**< PIPE_TEX_REDUCTION_x */
   unsigned pad:5;               /**< take bits from this for new members */
   float lod_bias;               /**< LOD/lambda bias */
   float min_lod, max_lod;       /**< LOD clamp range, after bias */
   union pipe_color_union border_color;
   enum pipe_format border_color_format;      /**< only with PIPE_QUIRK_TEXTURE_BORDER_COLOR_SWIZZLE_FREEDRENO, must be last */
};

union pipe_surface_desc {
   struct {
      unsigned level;
      unsigned first_layer:16;
      unsigned last_layer:16;
   } tex;
   struct {
      unsigned first_element;
      unsigned last_element;
   } buf;
};

/**
 * A view into a texture that can be bound to a color render target /
 * depth stencil attachment point.
 */
struct pipe_surface
{
   struct pipe_reference reference;
   enum pipe_format format:16;
   unsigned writable:1;          /**< writable shader resource */
   struct pipe_resource *texture; /**< resource into which this is a view  */
   struct pipe_context *context; /**< context this surface belongs to */

   /* XXX width/height should be removed */
   uint16_t width;               /**< logical width in pixels */
   uint16_t height;              /**< logical height in pixels */

   /**
    * Number of samples for the surface.  This will be 0 if rendering
    * should use the resource's nr_samples, or another value if the resource
    * is bound using FramebufferTexture2DMultisampleEXT.
    */
   unsigned nr_samples:8;

   union pipe_surface_desc u;
};


/**
 * A view into a texture that can be bound to a shader stage.
 */
struct pipe_sampler_view
{
   /* Put the refcount on its own cache line to prevent "False sharing". */
   EXCLUSIVE_CACHELINE(struct pipe_reference reference);

   enum pipe_format format:15;      /**< typed PIPE_FORMAT_x */
   enum pipe_texture_target target:5; /**< PIPE_TEXTURE_x */
   unsigned swizzle_r:3;         /**< PIPE_SWIZZLE_x for red component */
   unsigned swizzle_g:3;         /**< PIPE_SWIZZLE_x for green component */
   unsigned swizzle_b:3;         /**< PIPE_SWIZZLE_x for blue component */
   unsigned swizzle_a:3;         /**< PIPE_SWIZZLE_x for alpha component */
   struct pipe_resource *texture; /**< texture into which this is a view  */
   struct pipe_context *context; /**< context this view belongs to */
   union {
      struct {
         unsigned first_layer:16;  /**< first layer to use for array textures */
         unsigned last_layer:16;   /**< last layer to use for array textures */
         unsigned first_level:8;   /**< first mipmap level to use */
         unsigned last_level:8;    /**< last mipmap level to use */
      } tex;
      struct {
         unsigned offset;   /**< offset in bytes */
         unsigned size;     /**< size of the readable sub-range in bytes */
      } buf;
   } u;
};


/**
 * A description of a buffer or texture image that can be bound to a shader
 * stage.
 *
 * Note that pipe_image_view::access comes from the frontend API, while
 * shader_access comes from the shader and may contain additional information
 * (ie. coherent/volatile may be set on shader_access but not on access)
 */
struct pipe_image_view
{
   struct pipe_resource *resource; /**< resource into which this is a view  */
   enum pipe_format format;      /**< typed PIPE_FORMAT_x */
   uint16_t access;              /**< PIPE_IMAGE_ACCESS_x */
   uint16_t shader_access;       /**< PIPE_IMAGE_ACCESS_x */

   union {
      struct {
         unsigned first_layer:16;     /**< first layer to use for array textures */
         unsigned last_layer:16;      /**< last layer to use for array textures */
         unsigned level:8;            /**< mipmap level to use */
      } tex;
      struct {
         unsigned offset;   /**< offset in bytes */
         unsigned size;     /**< size of the accessible sub-range in bytes */
      } buf;
   } u;
};


/**
 * Subregion of 1D/2D/3D image resource.
 */
struct pipe_box
{
   /* Fields only used by textures use int16_t instead of int.
    * x and width are used by buffers, so they need the full 32-bit range.
    */
   int x;
   int16_t y;
   int16_t z;
   int width;
   int16_t height;
   int16_t depth;
};


/**
 * A memory object/resource such as a vertex buffer or texture.
 */
struct pipe_resource
{
   /* Put the refcount on its own cache line to prevent "False sharing". */
   EXCLUSIVE_CACHELINE(struct pipe_reference reference);

   unsigned width0; /**< Used by both buffers and textures. */
   uint16_t height0; /* Textures: The maximum height/depth/array_size is 16k. */
   uint16_t depth0;
   uint16_t array_size;

   enum pipe_format format:16;         /**< PIPE_FORMAT_x */
   enum pipe_texture_target target:8; /**< PIPE_TEXTURE_x */
   unsigned last_level:8;    /**< Index of last mipmap level present/defined */

   /** Number of samples determining quality, driving rasterizer, shading,
    *  and framebuffer.
    */
   unsigned nr_samples:8;

   /** Multiple samples within a pixel can have the same value.
    *  nr_storage_samples determines how many slots for different values
    *  there are per pixel. Only color buffers can set this lower than
    *  nr_samples.
    */
   unsigned nr_storage_samples:8;

   unsigned nr_sparse_levels:8; /**< Mipmap levels support partial resident */

   unsigned usage:8;         /**< PIPE_USAGE_x (not a bitmask) */
   unsigned bind;            /**< bitmask of PIPE_BIND_x */
   unsigned flags;           /**< bitmask of PIPE_RESOURCE_FLAG_x */

   /**
    * For planar images, ie. YUV EGLImage external, etc, pointer to the
    * next plane.
    */
   struct pipe_resource *next;
   /* The screen pointer should be last for optimal structure packing. */
   struct pipe_screen *screen; /**< screen that this texture belongs to */
};

/**
 * Opaque object used for separate resource/memory allocations.
 */
struct pipe_memory_allocation;

/**
 * Transfer object.  For data transfer to/from a resource.
 */
struct pipe_transfer
{
   struct pipe_resource *resource; /**< resource to transfer to/from  */
   enum pipe_map_flags usage:24;
   unsigned level:8;               /**< texture mipmap level */
   struct pipe_box box;            /**< region of the resource to access */
   unsigned stride;                /**< row stride in bytes */
   unsigned layer_stride;          /**< image/layer stride in bytes */

   /* Offset into a driver-internal staging buffer to make use of unused
    * padding in this structure.
    */
   unsigned offset;
};


/**
 * A vertex buffer.  Typically, all the vertex data/attributes for
 * drawing something will be in one buffer.  But it's also possible, for
 * example, to put colors in one buffer and texcoords in another.
 */
struct pipe_vertex_buffer
{
   uint16_t stride;    /**< stride to same attrib in next vertex, in bytes */
   bool is_user_buffer;
   unsigned buffer_offset;  /**< offset to start of data in buffer, in bytes */

   union {
      struct pipe_resource *resource;  /**< the actual buffer */
      const void *user;  /**< pointer to a user buffer */
   } buffer;
};


/**
 * A constant buffer.  A subrange of an existing buffer can be set
 * as a constant buffer.
 */
struct pipe_constant_buffer
{
   struct pipe_resource *buffer; /**< the actual buffer */
   unsigned buffer_offset; /**< offset to start of data in buffer, in bytes */
   unsigned buffer_size;   /**< how much data can be read in shader */
   const void *user_buffer;  /**< pointer to a user buffer if buffer == NULL */
};


/**
 * An untyped shader buffer supporting loads, stores, and atomics.
 */
struct pipe_shader_buffer {
   struct pipe_resource *buffer; /**< the actual buffer */
   unsigned buffer_offset; /**< offset to start of data in buffer, in bytes */
   unsigned buffer_size;   /**< how much data can be read in shader */
};


/**
 * A stream output target. The structure specifies the range vertices can
 * be written to.
 *
 * In addition to that, the structure should internally maintain the offset
 * into the buffer, which should be incremented everytime something is written
 * (appended) to it. The internal offset is buffer_offset + how many bytes
 * have been written. The internal offset can be stored on the device
 * and the CPU actually doesn't have to query it.
 *
 * Note that the buffer_size variable is actually specifying the available
 * space in the buffer, not the size of the attached buffer.
 * In other words in majority of cases buffer_size would simply be
 * 'buffer->width0 - buffer_offset', so buffer_size refers to the size
 * of the buffer left, after accounting for buffer offset, for stream output
 * to write to.
 *
 * Use PIPE_QUERY_SO_STATISTICS to know how many primitives have
 * actually been written.
 */
struct pipe_stream_output_target
{
   struct pipe_reference reference;
   struct pipe_resource *buffer; /**< the output buffer */
   struct pipe_context *context; /**< context this SO target belongs to */

   unsigned buffer_offset;  /**< offset where data should be written, in bytes */
   unsigned buffer_size;    /**< how much data is allowed to be written */
};


/**
 * Information to describe a vertex attribute (position, color, etc)
 */
struct pipe_vertex_element
{
   /** Offset of this attribute, in bytes, from the start of the vertex */
   uint16_t src_offset;

   /** Which vertex_buffer (as given to pipe->set_vertex_buffer()) does
    * this attribute live in?
    */
   uint8_t vertex_buffer_index:7;

   /**
    * Whether this element refers to a dual-slot vertex shader input.
    * The purpose of this field is to do dual-slot lowering when the CSO is
    * created instead of during every state change.
    *
    * It's lowered by util_lower_uint64_vertex_elements.
    */
   bool dual_slot:1;

   /**
    * This has only 8 bits because all vertex formats should be <= 255.
    */
   uint8_t src_format; /* low 8 bits of enum pipe_format. */

   /** Instance data rate divisor. 0 means this is per-vertex data,
    *  n means per-instance data used for n consecutive instances (n > 0).
    */
   unsigned instance_divisor;
};

/**
 * Opaque refcounted constant state object encapsulating a vertex buffer,
 * index buffer, and vertex elements. Used by display lists to bind those
 * states and pass buffer references quickly.
 *
 * The state contains 1 index buffer, 0 or 1 vertex buffer, and 0 or more
 * vertex elements.
 *
 * Constraints on the buffers to get the fastest codepath:
 * - All buffer contents are considered immutable and read-only after
 *   initialization. This implies the following things.
 * - No place is required to track whether these buffers are busy.
 * - All CPU mappings of these buffers can be forced to UNSYNCHRONIZED by
 *   both drivers and common code unconditionally.
 * - Buffer invalidation can be skipped by both drivers and common code
 *   unconditionally.
 */
struct pipe_vertex_state {
   struct pipe_reference reference;
   struct pipe_screen *screen;

   /* The following structure is used as a key for util_vertex_state_cache
    * to deduplicate identical state objects and thus enable more
    * opportunities for draw merging.
    */
   struct {
      struct pipe_resource *indexbuf;
      struct pipe_vertex_buffer vbuffer;
      unsigned num_elements;
      struct pipe_vertex_element elements[PIPE_MAX_ATTRIBS];
      uint32_t full_velem_mask;
   } input;
};

struct pipe_draw_indirect_info
{
   unsigned offset; /**< must be 4 byte aligned */
   unsigned stride; /**< must be 4 byte aligned */
   unsigned draw_count; /**< number of indirect draws */
   unsigned indirect_draw_count_offset; /**< must be 4 byte aligned */

   /* Indirect draw parameters resource is laid out as follows:
    *
    * if using indexed drawing:
    *  struct {
    *     uint32_t count;
    *     uint32_t instance_count;
    *     uint32_t start;
    *     int32_t index_bias;
    *     uint32_t start_instance;
    *  };
    * otherwise:
    *  struct {
    *     uint32_t count;
    *     uint32_t instance_count;
    *     uint32_t start;
    *     uint32_t start_instance;
    *  };
    *
    * If NULL, count_from_stream_output != NULL.
    */
   struct pipe_resource *buffer;

   /* Indirect draw count resource: If not NULL, contains a 32-bit value which
    * is to be used as the real draw_count.
    */
   struct pipe_resource *indirect_draw_count;

   /**
    * Stream output target. If not NULL, it's used to provide the 'count'
    * parameter based on the number vertices captured by the stream output
    * stage. (or generally, based on the number of bytes captured)
    *
    * Only 'mode', 'start_instance', and 'instance_count' are taken into
    * account, all the other variables from pipe_draw_info are ignored.
    *
    * 'start' is implicitly 0 and 'count' is set as discussed above.
    * The draw command is non-indexed.
    *
    * Note that this only provides the count. The vertex buffers must
    * be set via set_vertex_buffers manually.
    */
   struct pipe_stream_output_target *count_from_stream_output;
};

struct pipe_draw_start_count_bias {
   unsigned start;
   unsigned count;
   int index_bias; /**< a bias to be added to each index */
};

/**
 * Draw vertex state description. It's translated to pipe_draw_info as follows:
 * - mode comes from this structure
 * - index_size is 4
 * - instance_count is 1
 * - index.resource comes from pipe_vertex_state
 * - everything else is 0
 */
struct pipe_draw_vertex_state_info {
#if defined(__GNUC__)
   /* sizeof(mode) == 1 because it's a packed enum. */
   enum pipe_prim_type mode;  /**< the mode of the primitive */
#else
   /* sizeof(mode) == 1 is required by draw merging in u_threaded_context. */
   uint8_t mode;              /**< the mode of the primitive */
#endif
   bool take_vertex_state_ownership; /**< for skipping reference counting */
};

/**
 * Information to describe a draw_vbo call.
 */
struct pipe_draw_info
{
#if defined(__GNUC__)
   /* sizeof(mode) == 1 because it's a packed enum. */
   enum pipe_prim_type mode;  /**< the mode of the primitive */
#else
   /* sizeof(mode) == 1 is required by draw merging in u_threaded_context. */
   uint8_t mode;              /**< the mode of the primitive */
#endif
   uint8_t index_size;        /**< if 0, the draw is not indexed. */
   uint8_t view_mask;         /**< mask of multiviews for this draw */
   bool primitive_restart:1;
   bool has_user_indices:1;   /**< if true, use index.user_buffer */
   bool index_bounds_valid:1; /**< whether min_index and max_index are valid;
                                   they're always invalid if index_size == 0 */
   bool increment_draw_id:1;  /**< whether drawid increments for direct draws */
   bool take_index_buffer_ownership:1; /**< callee inherits caller's refcount
         (no need to reference indexbuf, but still needs to unreference it) */
   bool index_bias_varies:1;   /**< true if index_bias varies between draws */
   bool was_line_loop:1; /**< true if pipe_prim_type was LINE_LOOP before translation */
   uint8_t _pad:1;

   unsigned start_instance; /**< first instance id */
   unsigned instance_count; /**< number of instances */

   /**
    * Primitive restart enable/index (only applies to indexed drawing)
    */
   unsigned restart_index;

   /* Pointers must be placed appropriately for optimal structure packing on
    * 64-bit CPUs.
    */

   /**
    * An index buffer.  When an index buffer is bound, all indices to vertices
    * will be looked up from the buffer.
    *
    * If has_user_indices, use index.user, else use index.resource.
    */
   union {
      struct pipe_resource *resource;  /**< real buffer */
      const void *user;  /**< pointer to a user buffer */
   } index;

   /* These must be last for better packing in u_threaded_context. */
   unsigned min_index; /**< the min index */
   unsigned max_index; /**< the max index */
};


/**
 * Information to describe a blit call.
 */
struct pipe_blit_info
{
   struct {
      struct pipe_resource *resource;
      unsigned level;
      struct pipe_box box; /**< negative width, height only legal for src */
      /* For pipe_surface-like format casting: */
      enum pipe_format format; /**< must be supported for sampling (src)
                               or rendering (dst), ZS is always supported */
   } dst, src;

   unsigned mask; /**< bitmask of PIPE_MASK_R/G/B/A/Z/S */
   unsigned filter; /**< PIPE_TEX_FILTER_* */
   uint8_t dst_sample; /**< if non-zero, set sample_mask to (1 << (dst_sample - 1)) */
   bool sample0_only;
   bool scissor_enable;
   struct pipe_scissor_state scissor;

   /* Window rectangles can either be inclusive or exclusive. */
   bool window_rectangle_include;
   unsigned num_window_rectangles;
   struct pipe_scissor_state window_rectangles[PIPE_MAX_WINDOW_RECTANGLES];

   bool render_condition_enable; /**< whether the blit should honor the
                                 current render condition */
   bool alpha_blend; /* dst.rgb = src.rgb * src.a + dst.rgb * (1 - src.a) */
};

/**
 * Information to describe a launch_grid call.
 */
struct pipe_grid_info
{
   /**
    * For drivers that use PIPE_SHADER_IR_NATIVE as their prefered IR, this
    * value will be the index of the kernel in the opencl.kernels metadata
    * list.
    */
   uint32_t pc;

   /**
    * Will be used to initialize the INPUT resource, and it should point to a
    * buffer of at least pipe_compute_state::req_input_mem bytes.
    */
   const void *input;

   /**
    * Variable shared memory used by this invocation.
    *
    * This comes on top of shader declared shared memory.
    */
   uint32_t variable_shared_mem;

   /**
    * Grid number of dimensions, 1-3, e.g. the work_dim parameter passed to
    * clEnqueueNDRangeKernel. Note block[] and grid[] must be padded with
    * 1 for non-used dimensions.
    */
   uint work_dim;

   /**
    * Determine the layout of the working block (in thread units) to be used.
    */
   uint block[3];

   /**
    * last_block allows disabling threads at the farthermost grid boundary.
    * Full blocks as specified by "block" are launched, but the threads
    * outside of "last_block" dimensions are disabled.
    *
    * If a block touches the grid boundary in the i-th axis, threads with
    * THREAD_ID[i] >= last_block[i] are disabled.
    *
    * If last_block[i] is 0, it has the same behavior as last_block[i] = block[i],
    * meaning no effect.
    *
    * It's equivalent to doing this at the beginning of the compute shader:
    *
    *   for (i = 0; i < 3; i++) {
    *      if (block_id[i] == grid[i] - 1 &&
    *          last_block[i] && thread_id[i] >= last_block[i])
    *         return;
    *   }
    */
   uint last_block[3];

   /**
    * Determine the layout of the grid (in block units) to be used.
    */
   uint grid[3];

   /**
    * Base offsets to launch grids from
    */
   uint grid_base[3];

   /* Indirect compute parameters resource: If not NULL, block sizes are taken
    * from this buffer instead, which is laid out as follows:
    *
    *  struct {
    *     uint32_t num_blocks_x;
    *     uint32_t num_blocks_y;
    *     uint32_t num_blocks_z;
    *  };
    */
   struct pipe_resource *indirect;
   unsigned indirect_offset; /**< must be 4 byte aligned */
};

/**
 * Structure used as a header for serialized compute programs.
 */
struct pipe_binary_program_header
{
   uint32_t num_bytes; /**< Number of bytes in the LLVM bytecode program. */
   char blob[];
};

struct pipe_compute_state
{
   enum pipe_shader_ir ir_type; /**< IR type contained in prog. */
   const void *prog; /**< Compute program to be executed. */
   unsigned static_shared_mem; /**< equal to info.shared_size, used for shaders passed as TGSI */
   unsigned req_input_mem; /**< Required size of the INPUT resource. */
};

/**
 * Structure that contains a callback for device reset messages from the driver
 * back to the gallium frontend.
 *
 * The callback must not be called from driver-created threads.
 */
struct pipe_device_reset_callback
{
   /**
    * Callback for the driver to report when a device reset is detected.
    *
    * \param data   user-supplied data pointer
    * \param status PIPE_*_RESET
    */
   void (*reset)(void *data, enum pipe_reset_status status);

   void *data;
};

/**
 * Information about memory usage. All sizes are in kilobytes.
 */
struct pipe_memory_info
{
   unsigned total_device_memory; /**< size of device memory, e.g. VRAM */
   unsigned avail_device_memory; /**< free device memory at the moment */
   unsigned total_staging_memory; /**< size of staging memory, e.g. GART */
   unsigned avail_staging_memory; /**< free staging memory at the moment */
   unsigned device_memory_evicted; /**< size of memory evicted (monotonic counter) */
   unsigned nr_device_memory_evictions; /**< # of evictions (monotonic counter) */
};

/**
 * Structure that contains information about external memory
 */
struct pipe_memory_object
{
   bool dedicated;
};

#ifdef __cplusplus
}
#endif

#endif
