/*
 * Copyright © 2016 Intel Corporation
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

#ifndef BULITIN_FUNCTIONS_H
#define BULITIN_FUNCTIONS_H

struct gl_shader;

#ifdef __cplusplus
extern "C" {
#endif

void
_mesa_glsl_builtin_functions_init_or_ref();

void
_mesa_glsl_builtin_functions_decref(void);

#ifdef __cplusplus

} /* extern "C" */

extern ir_function_signature *
_mesa_glsl_find_builtin_function(_mesa_glsl_parse_state *state,
                                 const char *name, exec_list *actual_parameters);

extern bool
_mesa_glsl_has_builtin_function(_mesa_glsl_parse_state *state,
                                const char *name);

extern gl_shader *
_mesa_glsl_get_builtin_function_shader(void);

extern ir_function_signature *
_mesa_get_main_function_signature(glsl_symbol_table *symbols);

namespace generate_ir {

ir_function_signature *
udiv64(void *mem_ctx, builtin_available_predicate avail);

ir_function_signature *
idiv64(void *mem_ctx, builtin_available_predicate avail);

ir_function_signature *
umod64(void *mem_ctx, builtin_available_predicate avail);

ir_function_signature *
imod64(void *mem_ctx, builtin_available_predicate avail);

ir_function_signature *
udivmod64(void *mem_ctx, builtin_available_predicate avail);

}

#endif /* __cplusplus */

#endif /* BULITIN_FUNCTIONS_H */
