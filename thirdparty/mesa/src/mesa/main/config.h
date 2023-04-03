/*
 * Mesa 3-D graphics library
 *
 * Copyright (C) 1999-2007  Brian Paul   All Rights Reserved.
 * Copyright (C) 2008  VMware, Inc.  All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

/**
 * \file config.h
 * Tunable configuration parameters.
 */

#ifndef MESA_CONFIG_H_INCLUDED
#define MESA_CONFIG_H_INCLUDED

#include "compiler/shader_enums.h"

/**
 * \name OpenGL implementation limits
 */
/*@{*/

/** Maximum modelview matrix stack depth */
#define MAX_MODELVIEW_STACK_DEPTH 32

/** Maximum projection matrix stack depth */
#define MAX_PROJECTION_STACK_DEPTH 32

/** Maximum texture matrix stack depth */
#define MAX_TEXTURE_STACK_DEPTH 10

/** Maximum attribute stack depth */
#define MAX_ATTRIB_STACK_DEPTH 16

/** Maximum client attribute stack depth */
#define MAX_CLIENT_ATTRIB_STACK_DEPTH 16

/** Maximum recursion depth of display list calls */
#define MAX_LIST_NESTING 64

/** Maximum number of lights */
#define MAX_LIGHTS 8

/**
 * Maximum number of user-defined clipping planes supported by any driver in
 * Mesa.  This is used to size arrays.
 */
#define MAX_CLIP_PLANES 8

/** Maximum pixel map lookup table size */
#define MAX_PIXEL_MAP_TABLE 256

/** Maximum number of auxillary color buffers */
#define MAX_AUX_BUFFERS 1

/** Maximum order (degree) of curves */
#define MAX_EVAL_ORDER 30

/** Maximum Name stack depth */
#define MAX_NAME_STACK_DEPTH 64
/** Name stack buffer size */
#define NAME_STACK_BUFFER_SIZE 2048
/** Maximum name stack result number */
#define MAX_NAME_STACK_RESULT_NUM 256

/** Minimum point size */
#define MIN_POINT_SIZE 1.0
/** Maximum point size */
#define MAX_POINT_SIZE 60.0
/** Point size granularity */
#define POINT_SIZE_GRANULARITY 0.1

/** Minimum line width */
#define MIN_LINE_WIDTH 1.0
/** Maximum line width */
#define MAX_LINE_WIDTH 10.0
/** Line width granularity */
#define LINE_WIDTH_GRANULARITY 0.1

/** Max memory to allow for a single texture image (in megabytes) */
#define MAX_TEXTURE_MBYTES 1024

/** Number of texture mipmap levels */
#define MAX_TEXTURE_LEVELS 15

/** Maximum rectangular texture size - GL_NV_texture_rectangle */
#define MAX_TEXTURE_RECT_SIZE 16384

/**
 * Maximum number of layers in a 1D or 2D array texture - GL_MESA_texture_array
 */
#define MAX_ARRAY_TEXTURE_LAYERS 64

/**
 * Max number of texture image units.  Also determines number of texture
 * samplers in shaders.
 */
#define MAX_TEXTURE_IMAGE_UNITS 32

/**
 * Larger of MAX_TEXTURE_COORD_UNITS and MAX_TEXTURE_IMAGE_UNITS.
 * This value is only used for dimensioning arrays.
 * Either MAX_TEXTURE_COORD_UNITS or MAX_TEXTURE_IMAGE_UNITS (or the
 * corresponding ctx->Const.MaxTextureCoord/ImageUnits fields) should be
 * used almost everywhere else.
 */
#define MAX_TEXTURE_UNITS ((MAX_TEXTURE_COORD_UNITS > MAX_TEXTURE_IMAGE_UNITS) ? MAX_TEXTURE_COORD_UNITS : MAX_TEXTURE_IMAGE_UNITS)

/** Maximum number of viewports supported with ARB_viewport_array */
#define MAX_VIEWPORTS 16

/** Maximum number of window rectangles supported with EXT_window_rectangles */
#define MAX_WINDOW_RECTANGLES 8

/** Maximum size for CVA.  May be overridden by the drivers.  */
#define MAX_ARRAY_LOCK_SIZE 3000

/** Subpixel precision for antialiasing, window coordinate snapping */
#define SUB_PIXEL_BITS 4

/** For GL_ARB_texture_compression */
#define MAX_COMPRESSED_TEXTURE_FORMATS 25

/** For GL_EXT_texture_filter_anisotropic */
#define MAX_TEXTURE_MAX_ANISOTROPY 16.0

/** For GL_EXT_texture_lod_bias (typically MAX_TEXTURE_LEVELS - 1) */
#define MAX_TEXTURE_LOD_BIAS 14.0

/** For any program target/extension */
/*@{*/
#define MAX_PROGRAM_INSTRUCTIONS       (16 * 1024)

/**
 * Per-program constants (power of two)
 *
 * \c MAX_PROGRAM_LOCAL_PARAMS and \c MAX_UNIFORMS are just the assembly shader
 * and GLSL shader names for the same thing.  They should \b always have the
 * same value.  Each refers to the number of vec4 values supplied as
 * per-program parameters.
 */
/*@{*/
#define MAX_PROGRAM_LOCAL_PARAMS       4096
#define MAX_UNIFORMS                   4096
#define MAX_UNIFORM_BUFFERS            15 /* + 1 default uniform buffer */
#define MAX_SHADER_STORAGE_BUFFERS     16
/* 6 is for vertex, hull, domain, geometry, fragment, and compute shader. */
#define MAX_COMBINED_UNIFORM_BUFFERS   (MAX_UNIFORM_BUFFERS * 6)
#define MAX_COMBINED_SHADER_STORAGE_BUFFERS   (MAX_SHADER_STORAGE_BUFFERS * 6)
#define MAX_ATOMIC_COUNTERS            4096
/* 6 is for vertex, hull, domain, geometry, fragment, and compute shader. */
#define MAX_COMBINED_ATOMIC_BUFFERS    (MAX_UNIFORM_BUFFERS * 6)
/* Size of an atomic counter in bytes according to ARB_shader_atomic_counters */
#define ATOMIC_COUNTER_SIZE            4
#define MAX_IMAGE_UNIFORMS             32
/* 6 is for vertex, hull, domain, geometry, fragment, and compute shader. */
#define MAX_IMAGE_UNITS                (MAX_IMAGE_UNIFORMS * 6)
/*@}*/

/**
 * Per-context constants (power of two)
 *
 * \note
 * This value should always be less than or equal to \c MAX_PROGRAM_LOCAL_PARAMS
 * and \c MAX_VERTEX_PROGRAM_PARAMS.  Otherwise some applications will make
 * incorrect assumptions.
 */
#define MAX_PROGRAM_ENV_PARAMS         256

#define MAX_PROGRAM_MATRICES           8
#define MAX_PROGRAM_MATRIX_STACK_DEPTH 4
#define MAX_PROGRAM_CALL_DEPTH         8
#define MAX_PROGRAM_TEMPS              256
#define MAX_PROGRAM_ADDRESS_REGS       1
#define MAX_SAMPLERS                   MAX_TEXTURE_IMAGE_UNITS
#define MAX_PROGRAM_INPUTS             32
#define MAX_PROGRAM_OUTPUTS            64
/*@}*/

/** For GL_ARB_vertex_program */
/*@{*/
#define MAX_VERTEX_PROGRAM_ADDRESS_REGS 1
#define MAX_VERTEX_PROGRAM_PARAMS       MAX_UNIFORMS
/*@}*/

/** For GL_ARB_fragment_program */
/*@{*/
#define MAX_FRAGMENT_PROGRAM_ADDRESS_REGS 0
#define MAX_FRAGMENT_PROGRAM_PARAMS       64
#define MAX_FRAGMENT_PROGRAM_INPUTS       12
/*@}*/

/** For GL_ARB_vertex_shader */
/*@{*/
/* 6 is for vertex, hull, domain, geometry, fragment, and compute shader. */
#define MAX_COMBINED_TEXTURE_IMAGE_UNITS (MAX_TEXTURE_IMAGE_UNITS * 6)
/*@}*/


/** For GL_EXT_framebuffer_object */
/*@{*/
#define MAX_COLOR_ATTACHMENTS 8
#define MAX_RENDERBUFFER_SIZE 16384
/*@}*/

/** For GL_ATI_envmap_bump - support bump mapping on first 8 units */
#define SUPPORTED_ATI_BUMP_UNITS 0xff

/** For GL_EXT_transform_feedback */
#define MAX_FEEDBACK_BUFFERS 4
#define MAX_FEEDBACK_ATTRIBS 32

/** For geometry shader */
/*@{*/
#define MAX_GEOMETRY_UNIFORM_COMPONENTS              512
#define MAX_GEOMETRY_OUTPUT_VERTICES                 256
#define MAX_GEOMETRY_TOTAL_OUTPUT_COMPONENTS         1024
/*@}*/

/** For GL_ARB_debug_output and GL_KHR_debug */
/*@{*/
#define MAX_DEBUG_LOGGED_MESSAGES   10
#define MAX_DEBUG_MESSAGE_LENGTH    4096
/*@}*/

/** For GL_KHR_debug */
/*@{*/
#define MAX_LABEL_LENGTH 256
#define MAX_DEBUG_GROUP_STACK_DEPTH 64
/*@}*/

/** For GL_ARB_gpu_shader5 */
/*@{*/
#define MAX_GEOMETRY_SHADER_INVOCATIONS     32
#define MIN_FRAGMENT_INTERPOLATION_OFFSET   -0.5
#define MAX_FRAGMENT_INTERPOLATION_OFFSET   0.5
#define FRAGMENT_INTERPOLATION_OFFSET_BITS  4
#define MAX_VERTEX_STREAMS                  4
/*@}*/

/** For GL_ARB_shader_subroutine */
/*@{*/
#define MAX_SUBROUTINES                   256
#define MAX_SUBROUTINE_UNIFORM_LOCATIONS  1024
/*@}*/

/** For GL_INTEL_performance_query */
/*@{*/
#define MAX_PERFQUERY_QUERY_NAME_LENGTH     256
#define MAX_PERFQUERY_COUNTER_NAME_LENGTH   256
#define MAX_PERFQUERY_COUNTER_DESC_LENGTH   1024
#define PERFQUERY_HAVE_GPA_EXTENDED_COUNTERS 0
/*@}*/

/** For GL_ARB_pipeline_statistics_query */
#define MAX_PIPELINE_STATISTICS             11

/** For GL_ARB_tessellation_shader */
/*@{*/
#define MAX_TESS_GEN_LEVEL 64
#define MAX_PATCH_VERTICES 32
#define MAX_TESS_PATCH_COMPONENTS 120
#define MAX_TESS_CONTROL_TOTAL_OUTPUT_COMPONENTS 4096
/*@}*/

/*
 * Color channel component order
 * 
 * \note Changes will almost certainly cause problems at this time.
 */
#define RCOMP 0
#define GCOMP 1
#define BCOMP 2
#define ACOMP 3


/**
 * Maximum number of temporary vertices required for clipping.  
 *
 * Used in array_cache and tnl modules.
 */
#define MAX_CLIPPED_VERTICES ((2 * (6 + MAX_CLIP_PLANES))+1)


/** For GL_ARB_sample_locations - maximum of SAMPLE_LOCATION_PIXEL_GRID_*_ARB */
#define MAX_SAMPLE_LOCATION_GRID_SIZE 4

/* It is theoretically possible for Consts.MaxSamples to be >32 but
 * other code seems to assume that is not the case.
 */
#define MAX_SAMPLE_LOCATION_TABLE_SIZE \
   (MAX_SAMPLE_LOCATION_GRID_SIZE * MAX_SAMPLE_LOCATION_GRID_SIZE * 32)

#endif /* MESA_CONFIG_H_INCLUDED */
