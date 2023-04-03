/*
 * Copyright © 2018 Timothy Arceri
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
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef GL_NIR_H
#define GL_NIR_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>

struct nir_shader;
struct gl_constants;
struct gl_linked_shader;
struct gl_shader_program;

bool gl_nir_lower_atomics(nir_shader *shader,
                          const struct gl_shader_program *shader_program,
                          bool use_binding_as_idx);

bool gl_nir_lower_images(nir_shader *shader, bool bindless_only);
bool gl_nir_lower_samplers(nir_shader *shader,
                           const struct gl_shader_program *shader_program);
bool gl_nir_lower_samplers_as_deref(nir_shader *shader,
                                    const struct gl_shader_program *shader_program);

bool gl_nir_lower_buffers(nir_shader *shader,
                          const struct gl_shader_program *shader_program);

void gl_nir_lower_packed_varyings(const struct gl_constants *consts,
                                  struct gl_shader_program *prog,
                                  void *mem_ctx, unsigned locations_used,
                                  const uint8_t *components,
                                  nir_variable_mode mode,
                                  unsigned gs_input_vertices,
                                  struct gl_linked_shader *linked_shader,
                                  bool disable_varying_packing,
                                  bool disable_xfb_packing, bool xfb_enabled);

#ifdef __cplusplus
}
#endif

#endif /* GL_NIR_H */
