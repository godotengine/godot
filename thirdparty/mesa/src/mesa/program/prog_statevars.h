/*
 * Mesa 3-D graphics library
 *
 * Copyright (C) 1999-2007  Brian Paul   All Rights Reserved.
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

#ifndef PROG_STATEVARS_H
#define PROG_STATEVARS_H


#include "util/glheader.h"
#include "compiler/shader_enums.h"
#include <stdint.h>


#ifdef __cplusplus
extern "C" {
#endif


struct gl_context;
struct gl_constants;
struct gl_program_parameter_list;


/**
 * Used for describing GL state referenced from inside ARB vertex and
 * fragment programs.
 * A string such as "state.light[0].ambient" gets translated into a
 * sequence of tokens such as [ STATE_LIGHT, 0, STATE_AMBIENT ].
 *
 * For state that's an array, like STATE_CLIPPLANE, the 2nd token [1] should
 * always be the array index.
 */
typedef enum gl_state_index_ {
   STATE_NOT_STATE_VAR = 0,

   STATE_MATERIAL,

   STATE_LIGHT,         /* One gl_light attribute. */
   STATE_LIGHT_ARRAY, /* Multiple gl_light attributes loaded at once. */
   STATE_LIGHT_ATTENUATION_ARRAY,
   STATE_LIGHTMODEL_AMBIENT,
   STATE_LIGHTMODEL_SCENECOLOR,
   STATE_LIGHTPROD,
   STATE_LIGHTPROD_ARRAY_FRONT,   /* multiple lights, only front faces */
   STATE_LIGHTPROD_ARRAY_BACK,    /* multiple lights, only back faces */
   STATE_LIGHTPROD_ARRAY_TWOSIDE, /* multiple lights, both sides */

   STATE_TEXGEN,
   STATE_TEXENV_COLOR,

   STATE_FOG_COLOR,
   STATE_FOG_PARAMS,

   STATE_CLIPPLANE,

   STATE_POINT_SIZE,
   STATE_POINT_ATTENUATION,

   STATE_MODELVIEW_MATRIX,
   STATE_MODELVIEW_MATRIX_INVERSE,
   STATE_MODELVIEW_MATRIX_TRANSPOSE,
   STATE_MODELVIEW_MATRIX_INVTRANS,

   STATE_PROJECTION_MATRIX,
   STATE_PROJECTION_MATRIX_INVERSE,
   STATE_PROJECTION_MATRIX_TRANSPOSE,
   STATE_PROJECTION_MATRIX_INVTRANS,

   STATE_MVP_MATRIX,
   STATE_MVP_MATRIX_INVERSE,
   STATE_MVP_MATRIX_TRANSPOSE,
   STATE_MVP_MATRIX_INVTRANS,

   STATE_TEXTURE_MATRIX,
   STATE_TEXTURE_MATRIX_INVERSE,
   STATE_TEXTURE_MATRIX_TRANSPOSE,
   STATE_TEXTURE_MATRIX_INVTRANS,

   STATE_PROGRAM_MATRIX,
   STATE_PROGRAM_MATRIX_INVERSE,
   STATE_PROGRAM_MATRIX_TRANSPOSE,
   STATE_PROGRAM_MATRIX_INVTRANS,

   STATE_NUM_SAMPLES,    /* An integer, not a float like the other state vars */

   STATE_DEPTH_RANGE,

   STATE_FRAGMENT_PROGRAM_ENV,
   STATE_FRAGMENT_PROGRAM_ENV_ARRAY,
   STATE_FRAGMENT_PROGRAM_LOCAL,
   STATE_FRAGMENT_PROGRAM_LOCAL_ARRAY,
   STATE_VERTEX_PROGRAM_ENV,
   STATE_VERTEX_PROGRAM_ENV_ARRAY,
   STATE_VERTEX_PROGRAM_LOCAL,
   STATE_VERTEX_PROGRAM_LOCAL_ARRAY,

   STATE_NORMAL_SCALE_EYESPACE,
   STATE_CURRENT_ATTRIB,        /* ctx->Current vertex attrib value */
   STATE_CURRENT_ATTRIB_MAYBE_VP_CLAMPED,        /* ctx->Current vertex attrib value after passthrough vertex processing */
   STATE_NORMAL_SCALE,
   STATE_FOG_PARAMS_OPTIMIZED,  /* for faster fog calc */
   STATE_POINT_SIZE_CLAMPED,    /* includes implementation dependent size clamp */
   STATE_LIGHT_SPOT_DIR_NORMALIZED,   /* pre-normalized spot dir */
   STATE_LIGHT_POSITION,              /* object vs eye space */
   STATE_LIGHT_POSITION_ARRAY,
   STATE_LIGHT_POSITION_NORMALIZED,   /* object vs eye space */
   STATE_LIGHT_POSITION_NORMALIZED_ARRAY,
   STATE_LIGHT_HALF_VECTOR,           /* object vs eye space */
   STATE_PT_SCALE,              /**< Pixel transfer RGBA scale */
   STATE_PT_BIAS,               /**< Pixel transfer RGBA bias */
   STATE_FB_SIZE,               /**< (width-1, height-1, 0, 0) */
   STATE_FB_WPOS_Y_TRANSFORM,   /**< (1, 0, -1, height) if a FBO is bound, (-1, height, 1, 0) otherwise */
   STATE_FB_PNTC_Y_TRANSFORM,   /**< (1, 0, 0, 0) if point origin is upper left, (-1, 1, 0, 0) otherwise */
   STATE_TCS_PATCH_VERTICES_IN, /**< gl_PatchVerticesIn for TCS (integer) */
   STATE_TES_PATCH_VERTICES_IN, /**< gl_PatchVerticesIn for TES (integer) */
   /**
    * A single enum gl_blend_support_qualifier value representing the
    * currently active advanced blending equation, or zero if disabled.
    */
   STATE_ADVANCED_BLENDING_MODE,
   STATE_ALPHA_REF,        /* alpha-test reference value */
   STATE_CLIP_INTERNAL,    /* similar to STATE_CLIPPLANE, but in clip-space */
   STATE_ATOMIC_COUNTER_OFFSET,    /* the byte offset to add to atomic counter bindings */

   STATE_INTERNAL_DRIVER,	/* first available state index for drivers (must be last) */


   /** All enums below don't occur in state[0]. **/

   /* These 8 enums must be in the same order as the gl_light union members,
    * which should also match the order of gl_LightSource members.
    */
   STATE_AMBIENT,
   STATE_DIFFUSE,
   STATE_SPECULAR,
   STATE_POSITION,       /**< xyzw = position */
   STATE_HALF_VECTOR,
   STATE_SPOT_DIRECTION, /**< xyz = direction, w = cos(cutoff) */
   STATE_ATTENUATION,    /**< xyz = attenuation, w = spot exponent */
   STATE_SPOT_CUTOFF,    /**< x = cutoff, yzw = undefined */

   STATE_EMISSION,
   STATE_SHININESS,

   /* These 8 enums must be in the same order as the memory layout of
    * gl_fixedfunc_texture_unit::EyePlane/ObjectPlane.
    */
   STATE_TEXGEN_EYE_S,
   STATE_TEXGEN_EYE_T,
   STATE_TEXGEN_EYE_R,
   STATE_TEXGEN_EYE_Q,
   STATE_TEXGEN_OBJECT_S,
   STATE_TEXGEN_OBJECT_T,
   STATE_TEXGEN_OBJECT_R,
   STATE_TEXGEN_OBJECT_Q,
} gl_state_index;


extern void
_mesa_load_state_parameters(struct gl_context *ctx,
                            struct gl_program_parameter_list *paramList);

extern void
_mesa_upload_state_parameters(struct gl_context *ctx,
                              struct gl_program_parameter_list *paramList,
                              uint32_t *dst);

extern void
_mesa_optimize_state_parameters(struct gl_constants *consts,
                                struct gl_program_parameter_list *list);

extern unsigned
_mesa_program_state_value_size(const gl_state_index16 state[STATE_LENGTH]);

extern GLbitfield
_mesa_program_state_flags(const gl_state_index16 state[STATE_LENGTH]);


extern char *
_mesa_program_state_string(const gl_state_index16 state[STATE_LENGTH]);



#ifdef __cplusplus
}
#endif

#endif /* PROG_STATEVARS_H */
