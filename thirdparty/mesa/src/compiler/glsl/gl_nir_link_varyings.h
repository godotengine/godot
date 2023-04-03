/*
 * Copyright © 2012 Intel Corporation
 * Copyright © 2022 Valve Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef GLSL_LINK_VARYINGS_H
#define GLSL_LINK_VARYINGS_H

/**
 * Linker functions related specifically to linking varyings between shader
 * stages.
 */


#include "util/glheader.h"
#include "program/prog_parameter.h"
#include "util/bitset.h"

#include "nir.h"

struct gl_shader_program;
struct gl_shader_stage;
struct gl_shader;
struct gl_type;


/**
 * Data structure describing a varying which is available for use in transform
 * feedback.
 *
 * For example, if the vertex shader contains:
 *
 *     struct S {
 *       vec4 foo;
 *       float[3] bar;
 *     };
 *
 *     varying S[2] v;
 *
 * Then there would be tfeedback_candidate objects corresponding to the
 * following varyings:
 *
 *     v[0].foo
 *     v[0].bar
 *     v[1].foo
 *     v[1].bar
 */
struct tfeedback_candidate
{
   /**
    * Toplevel variable containing this varying.  In the above example, this
    * would point to the declaration of the varying v.
    */
   nir_variable *toplevel_var;

   /**
    * Type of this varying.  In the above example, this would point to the
    * glsl_type for "vec4" or "float[3]".
    */
   const struct glsl_type *type;

   /**
    * Offset within the toplevel variable where this varying occurs.
    * Counted in floats.
    */
   unsigned struct_offset_floats;

   /**
    * Offset within the xfb with respect to alignment requirements.
    * Counted in floats.
    */
   unsigned xfb_offset_floats;

   /* Used to match varyings and update toplevel_var pointer after NIR
    * optimisations have been performed.
    */
   unsigned initial_location;
   unsigned initial_location_frac;
};

enum lowered_builtin_array_var {
   none,
   clip_distance,
   cull_distance,
   tess_level_outer,
   tess_level_inner,
};

/**
 * Data structure tracking information about a transform feedback declaration
 * during linking.
 */
struct xfb_decl
{
   /**
    * The name that was supplied to glTransformFeedbackVaryings.  Used for
    * error reporting and glGetTransformFeedbackVarying().
    */
   const char *orig_name;

   /**
    * The name of the variable, parsed from orig_name.
    */
   const char *var_name;

   /**
    * True if the declaration in orig_name represents an array.
    */
   bool is_subscripted;

   /**
    * If is_subscripted is true, the subscript that was specified in orig_name.
    */
   unsigned array_subscript;

   /**
    * Non-zero if the variable is gl_ClipDistance, glTessLevelOuter or
    * gl_TessLevelInner and the driver lowers it to gl_*MESA.
    */
   enum lowered_builtin_array_var lowered_builtin_array_variable;

   /**
    * The vertex shader output location that the linker assigned for this
    * variable.  -1 if a location hasn't been assigned yet.
    */
   int location;

   /**
    * Used to store the buffer assigned by xfb_buffer.
    */
   unsigned buffer;

   /**
    * Used to store the offset assigned by xfb_offset.
    */
   unsigned offset;

   /**
    * If non-zero, then this variable may be packed along with other variables
    * into a single varying slot, so this offset should be applied when
    * accessing components.  For example, an offset of 1 means that the x
    * component of this variable is actually stored in component y of the
    * location specified by \c location.
    *
    * Only valid if location != -1.
    */
   unsigned location_frac;

   /**
    * If location != -1, the number of vector elements in this variable, or 1
    * if this variable is a scalar.
    */
   unsigned vector_elements;

   /**
    * If location != -1, the number of matrix columns in this variable, or 1
    * if this variable is not a matrix.
    */
   unsigned matrix_columns;

   /** Type of the varying returned by glGetTransformFeedbackVarying() */
   GLenum type;

   /**
    * If location != -1, the size that should be returned by
    * glGetTransformFeedbackVarying().
    */
   unsigned size;

   /**
    * How many components to skip. If non-zero, this is
    * gl_SkipComponents{1,2,3,4} from ARB_transform_feedback3.
    */
   unsigned skip_components;

   /**
    * Whether this is gl_NextBuffer from ARB_transform_feedback3.
    */
   bool next_buffer_separator;

   /**
    * If find_candidate() has been called, pointer to the tfeedback_candidate
    * data structure that was found.  Otherwise NULL.
    */
   struct tfeedback_candidate *matched_candidate;

   /**
    * StreamId assigned to this varying (defaults to 0). Can only be set to
    * values other than 0 in geometry shaders that use the stream layout
    * modifier. Accepted values must be in the range [0, MAX_VERTEX_STREAMS-1].
    */
   unsigned stream_id;
};

static inline bool
xfb_decl_is_varying(const struct xfb_decl *xfb_decl)
{
   return !xfb_decl->next_buffer_separator && !xfb_decl->skip_components;
}

#endif /* GLSL_LINK_VARYINGS_H */
