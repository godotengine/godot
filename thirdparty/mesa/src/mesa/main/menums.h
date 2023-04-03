/*
 * Mesa 3-D graphics library
 *
 * Copyright (C) 1999-2008  Brian Paul   All Rights Reserved.
 * Copyright (C) 2009  VMware, Inc.  All Rights Reserved.
 * Copyright (C) 2018 Advanced Micro Devices, Inc.  All Rights Reserved.
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
 * \file menums.h
 * Often used definitions and enums.
 */

#ifndef MENUMS_H
#define MENUMS_H

#include "util/macros.h"

/**
 * Enum for the OpenGL APIs we know about and may support.
 *
 * NOTE: This must match the api_enum table in
 * src/mesa/main/get_hash_generator.py
 */
typedef enum
{
   API_OPENGL_COMPAT,      /* legacy / compatibility contexts */
   API_OPENGLES,
   API_OPENGLES2,
   API_OPENGL_CORE,
   API_OPENGL_LAST = API_OPENGL_CORE
} gl_api;

/**
 * An index for each type of texture object.  These correspond to the GL
 * texture target enums, such as GL_TEXTURE_2D, GL_TEXTURE_CUBE_MAP, etc.
 * Note: the order is from highest priority to lowest priority.
 */
typedef enum
{
   TEXTURE_2D_MULTISAMPLE_INDEX,
   TEXTURE_2D_MULTISAMPLE_ARRAY_INDEX,
   TEXTURE_CUBE_ARRAY_INDEX,
   TEXTURE_BUFFER_INDEX,
   TEXTURE_2D_ARRAY_INDEX,
   TEXTURE_1D_ARRAY_INDEX,
   TEXTURE_EXTERNAL_INDEX,
   TEXTURE_CUBE_INDEX,
   TEXTURE_3D_INDEX,
   TEXTURE_RECT_INDEX,
   TEXTURE_2D_INDEX,
   TEXTURE_1D_INDEX,
   NUM_TEXTURE_TARGETS
} gl_texture_index;

/**
 * Remapped color logical operations
 *
 * With the exception of NVIDIA hardware, which consumes the OpenGL enumerants
 * directly, everything wants this mapping of color logical operations.
 *
 * Fun fact: These values are just the bit-reverse of the low-nibble of the GL
 * enumerant values (i.e., `GL_NOOP & 0x0f` is `b0101' while
 * \c COLOR_LOGICOP_NOOP is `b1010`).
 *
 * Fun fact #2: These values are just an encoding of the operation as a table
 * of bit values. The result of the logic op is:
 *
 *    result_bit = (logic_op >> (2 * src_bit + dst_bit)) & 1
 *
 * For the GL enums, the result is:
 *
 *    result_bit = logic_op & (1 << (2 * src_bit + dst_bit))
 */
enum PACKED gl_logicop_mode {
   COLOR_LOGICOP_CLEAR = 0,
   COLOR_LOGICOP_NOR = 1,
   COLOR_LOGICOP_AND_INVERTED = 2,
   COLOR_LOGICOP_COPY_INVERTED = 3,
   COLOR_LOGICOP_AND_REVERSE = 4,
   COLOR_LOGICOP_INVERT = 5,
   COLOR_LOGICOP_XOR = 6,
   COLOR_LOGICOP_NAND = 7,
   COLOR_LOGICOP_AND = 8,
   COLOR_LOGICOP_EQUIV = 9,
   COLOR_LOGICOP_NOOP = 10,
   COLOR_LOGICOP_OR_INVERTED = 11,
   COLOR_LOGICOP_COPY = 12,
   COLOR_LOGICOP_OR_REVERSE = 13,
   COLOR_LOGICOP_OR = 14,
   COLOR_LOGICOP_SET = 15
};

/**
 * Indexes for all renderbuffers
 */
typedef enum
{
   /* the four standard color buffers */
   BUFFER_FRONT_LEFT,
   BUFFER_BACK_LEFT,
   BUFFER_FRONT_RIGHT,
   BUFFER_BACK_RIGHT,
   BUFFER_DEPTH,
   BUFFER_STENCIL,
   BUFFER_ACCUM,
   /* generic renderbuffers */
   BUFFER_COLOR0,
   BUFFER_COLOR1,
   BUFFER_COLOR2,
   BUFFER_COLOR3,
   BUFFER_COLOR4,
   BUFFER_COLOR5,
   BUFFER_COLOR6,
   BUFFER_COLOR7,
   BUFFER_COUNT,
   BUFFER_NONE = -1,
} gl_buffer_index;

typedef enum
{
   MAP_USER,
   MAP_INTERNAL,
   MAP_GLTHREAD,
   MAP_COUNT
} gl_map_buffer_index;

/** @{
 *
 * These are a mapping of the GL_ARB_debug_output/GL_KHR_debug enums
 * to small enums suitable for use as an array index.
 */

enum mesa_debug_source
{
   MESA_DEBUG_SOURCE_API,
   MESA_DEBUG_SOURCE_WINDOW_SYSTEM,
   MESA_DEBUG_SOURCE_SHADER_COMPILER,
   MESA_DEBUG_SOURCE_THIRD_PARTY,
   MESA_DEBUG_SOURCE_APPLICATION,
   MESA_DEBUG_SOURCE_OTHER,
   MESA_DEBUG_SOURCE_COUNT
};

enum mesa_debug_type
{
   MESA_DEBUG_TYPE_ERROR,
   MESA_DEBUG_TYPE_DEPRECATED,
   MESA_DEBUG_TYPE_UNDEFINED,
   MESA_DEBUG_TYPE_PORTABILITY,
   MESA_DEBUG_TYPE_PERFORMANCE,
   MESA_DEBUG_TYPE_OTHER,
   MESA_DEBUG_TYPE_MARKER,
   MESA_DEBUG_TYPE_PUSH_GROUP,
   MESA_DEBUG_TYPE_POP_GROUP,
   MESA_DEBUG_TYPE_COUNT
};

enum mesa_debug_severity
{
   MESA_DEBUG_SEVERITY_LOW,
   MESA_DEBUG_SEVERITY_MEDIUM,
   MESA_DEBUG_SEVERITY_HIGH,
   MESA_DEBUG_SEVERITY_NOTIFICATION,
   MESA_DEBUG_SEVERITY_COUNT
};

/** @} */

#endif
