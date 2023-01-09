/*
 * Mesa 3-D graphics library
 *
 * Copyright (C) 1999-2008  Brian Paul   All Rights Reserved.
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
 * \file prog_parameter.c
 * Program parameter lists and functions.
 * \author Brian Paul
 */

#ifndef PROG_PARAMETER_H
#define PROG_PARAMETER_H

#include <stdbool.h>
#include <stdint.h>
#include "prog_statevars.h"

#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Names of the various vertex/fragment program register files, etc.
 *
 * NOTE: first four tokens must fit into 2 bits (see t_vb_arbprogram.c)
 * All values should fit in a 4-bit field.
 *
 * NOTE: PROGRAM_STATE_VAR, PROGRAM_CONSTANT, and PROGRAM_UNIFORM can all be
 * considered to be "uniform" variables since they can only be set outside
 * glBegin/End.  They're also all stored in the same Parameters array.
 */
typedef enum
{
   PROGRAM_TEMPORARY,   /**< machine->Temporary[] */
   PROGRAM_INPUT,       /**< machine->Inputs[] */
   PROGRAM_OUTPUT,      /**< machine->Outputs[] */
   PROGRAM_STATE_VAR,   /**< gl_program->Parameters[] */
   PROGRAM_CONSTANT,    /**< gl_program->Parameters[] */
   PROGRAM_UNIFORM,     /**< gl_program->Parameters[] */
   PROGRAM_WRITE_ONLY,  /**< A dummy, write-only register */
   PROGRAM_ADDRESS,     /**< machine->AddressReg */
   PROGRAM_SYSTEM_VALUE,/**< InstanceId, PrimitiveID, etc. */
   PROGRAM_UNDEFINED,   /**< Invalid/TBD value */
   PROGRAM_FILE_MAX
} gl_register_file;


/**
 * Actual data for constant values of parameters.
 */
typedef union gl_constant_value
{
   GLfloat f;
   GLint b;
   GLint i;
   GLuint u;
} gl_constant_value;


/**
 * Program parameter.
 * Used by shaders/programs for uniforms, constants, varying vars, etc.
 */
struct gl_program_parameter
{
   const char *Name;        /**< Null-terminated string */
   gl_register_file Type:5;  /**< PROGRAM_CONSTANT or STATE_VAR */

   /**
    * We need to keep track of whether the param is padded for use in the
    * shader cache.
    */
   bool Padded:1;

   GLenum16 DataType;         /**< GL_FLOAT, GL_FLOAT_VEC2, etc */

   /**
    * Number of components (1..4), or more.
    * If the number of components is greater than 4,
    * this parameter is part of a larger uniform like a GLSL matrix or array.
    */
   GLushort Size;
   /**
    * A sequence of STATE_* tokens and integers to identify GL state.
    */
   gl_state_index16 StateIndexes[STATE_LENGTH];

   /**
    * Offset within ParameterValues where this parameter is stored.
    */
   unsigned ValueOffset;

   /**
    * Index of this parameter's uniform storage.
    */
   uint32_t UniformStorageIndex;

   /**
    * Index of the first uniform storage that is associated with the same
    * variable as this parameter.
    */
   uint32_t MainUniformStorageIndex;
};


/**
 * List of gl_program_parameter instances.
 */
struct gl_program_parameter_list
{
   unsigned Size;           /**< allocated size of Parameters */
   unsigned SizeValues;     /**< alllocate size of ParameterValues */
   GLuint NumParameters;  /**< number of used parameters in array */
   unsigned NumParameterValues;  /**< number of used parameter values array */
   struct gl_program_parameter *Parameters; /**< Array [Size] */
   gl_constant_value *ParameterValues; /**< Array [Size] of gl_constant_value */
   GLbitfield StateFlags; /**< _NEW_* flags indicating which state changes
                               might invalidate ParameterValues[] */
   bool DisallowRealloc;

   /* Parameters are optionally sorted as follows. Uniforms and constants
    * are first, then state vars. This should be true in all cases except
    * ir_to_mesa, which adds constants at the end, and ARB_vp with ARL,
    * which can't sort parameters.
    */
   int UniformBytes;
   int FirstStateVarIndex;
   int LastStateVarIndex;
};


extern struct gl_program_parameter_list *
_mesa_new_parameter_list(void);

extern struct gl_program_parameter_list *
_mesa_new_parameter_list_sized(unsigned size);

extern void
_mesa_free_parameter_list(struct gl_program_parameter_list *paramList);

extern void
_mesa_reserve_parameter_storage(struct gl_program_parameter_list *paramList,
                                unsigned reserve_params,
                                unsigned reserve_values);

extern void
_mesa_disallow_parameter_storage_realloc(struct gl_program_parameter_list *paramList);

extern GLint
_mesa_add_parameter(struct gl_program_parameter_list *paramList,
                    gl_register_file type, const char *name,
                    GLuint size, GLenum datatype,
                    const gl_constant_value *values,
                    const gl_state_index16 state[STATE_LENGTH],
                    bool pad_and_align);

extern GLint
_mesa_add_typed_unnamed_constant(struct gl_program_parameter_list *paramList,
                           const gl_constant_value *values, GLuint size,
                           GLenum datatype, GLuint *swizzleOut);

static inline GLint
_mesa_add_unnamed_constant(struct gl_program_parameter_list *paramList,
                           const gl_constant_value *values, GLuint size,
                           GLuint *swizzleOut)
{
   return _mesa_add_typed_unnamed_constant(paramList, values, size, GL_NONE,
                                           swizzleOut);
}

extern GLint
_mesa_add_sized_state_reference(struct gl_program_parameter_list *paramList,
                                const gl_state_index16 stateTokens[STATE_LENGTH],
                                const unsigned size, bool pad_and_align);

extern GLint
_mesa_add_state_reference(struct gl_program_parameter_list *paramList,
                          const gl_state_index16 stateTokens[STATE_LENGTH]);


static inline GLint
_mesa_lookup_parameter_index(const struct gl_program_parameter_list *paramList,
                             const char *name)
{
   if (!paramList)
      return -1;

   /* name must be null-terminated */
   for (GLint i = 0; i < (GLint) paramList->NumParameters; i++) {
      if (paramList->Parameters[i].Name &&
         strcmp(paramList->Parameters[i].Name, name) == 0)
         return i;
   }

   return -1;
}

static inline bool
_mesa_gl_datatype_is_64bit(GLenum datatype)
{
   switch (datatype) {
   case GL_DOUBLE:
   case GL_DOUBLE_VEC2:
   case GL_DOUBLE_VEC3:
   case GL_DOUBLE_VEC4:
   case GL_DOUBLE_MAT2:
   case GL_DOUBLE_MAT2x3:
   case GL_DOUBLE_MAT2x4:
   case GL_DOUBLE_MAT3:
   case GL_DOUBLE_MAT3x2:
   case GL_DOUBLE_MAT3x4:
   case GL_DOUBLE_MAT4:
   case GL_DOUBLE_MAT4x2:
   case GL_DOUBLE_MAT4x3:
   case GL_INT64_ARB:
   case GL_INT64_VEC2_ARB:
   case GL_INT64_VEC3_ARB:
   case GL_INT64_VEC4_ARB:
   case GL_UNSIGNED_INT64_ARB:
   case GL_UNSIGNED_INT64_VEC2_ARB:
   case GL_UNSIGNED_INT64_VEC3_ARB:
   case GL_UNSIGNED_INT64_VEC4_ARB:
      return true;
   default:
      return false;
   }
}

void
_mesa_recompute_parameter_bounds(struct gl_program_parameter_list *list);

#ifdef __cplusplus
}
#endif

#endif /* PROG_PARAMETER_H */
