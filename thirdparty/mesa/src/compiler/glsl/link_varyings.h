/*
 * Copyright © 2012 Intel Corporation
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
 * \file link_varyings.h
 *
 * Linker functions related specifically to linking varyings between shader
 * stages.
 */

struct gl_shader_program;

void
validate_first_and_last_interface_explicit_locations(const struct gl_constants *consts,
                                                     struct gl_shader_program *prog,
                                                     gl_shader_stage first,
                                                     gl_shader_stage last);

void
cross_validate_outputs_to_inputs(const struct gl_constants *consts,
                                 struct gl_shader_program *prog,
                                 gl_linked_shader *producer,
                                 gl_linked_shader *consumer);

#endif /* GLSL_LINK_VARYINGS_H */
