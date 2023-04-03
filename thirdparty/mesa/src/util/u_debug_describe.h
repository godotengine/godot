/**************************************************************************
 *
 * Copyright 2010 Luca Barbieri
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice (including the
 * next paragraph) shall be included in all copies or substantial
 * portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE COPYRIGHT OWNER(S) AND/OR ITS SUPPLIERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 **************************************************************************/

#ifndef U_DEBUG_DESCRIBE_H_
#define U_DEBUG_DESCRIBE_H_

#ifdef __cplusplus
extern "C" {
#endif

struct pipe_reference;
struct pipe_resource;
struct pipe_surface;
struct pipe_sampler_view;
struct pipe_image_view;

/* a 256-byte buffer is necessary and sufficient */
void debug_describe_reference(char* buf, const struct pipe_reference*ptr);
void debug_describe_resource(char* buf, const struct pipe_resource *ptr);
void debug_describe_surface(char* buf, const struct pipe_surface *ptr);
void debug_describe_sampler_view(char* buf, const struct pipe_sampler_view *ptr);
void debug_describe_image_view(char* buf, const struct pipe_image_view *ptr);
void debug_describe_so_target(char* buf,
                              const struct pipe_stream_output_target *ptr);

#ifdef __cplusplus
}
#endif

#endif /* U_DEBUG_DESCRIBE_H_ */
