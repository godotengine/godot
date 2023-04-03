/*
 * Copyright © 2016 Red Hat
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

#ifndef GLSL_STANDALONE_H
#define GLSL_STANDALONE_H

#ifdef __cplusplus
extern "C" {
#endif

struct standalone_options {
   int glsl_version;
   int dump_ast;
   int dump_hir;
   int dump_lir;
   int dump_builder;
   int do_link;
   int just_log;
   int lower_precision;
};

struct gl_shader_program;

struct gl_shader_program * standalone_compile_shader(
      const struct standalone_options *options,
      unsigned num_files, char* const* files,
      struct gl_context *ctx);

void standalone_compiler_cleanup(struct gl_shader_program *prog);

#ifdef __cplusplus
}
#endif

#endif /* GLSL_STANDALONE_H */
