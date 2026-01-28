/*
 * Copyright © 2011 Kristian Høgsberg
 * Copyright © 2011 Benjamin Franzke
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
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT.  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef WAYLAND_EGL_CORE_H
#define WAYLAND_EGL_CORE_H

#ifdef  __cplusplus
extern "C" {
#endif

#define WL_EGL_PLATFORM 1

struct wl_egl_window;
struct wl_surface;

struct wl_egl_window *
wl_egl_window_create(struct wl_surface *surface,
		     int width, int height);

void
wl_egl_window_destroy(struct wl_egl_window *egl_window);

void
wl_egl_window_resize(struct wl_egl_window *egl_window,
		     int width, int height,
		     int dx, int dy);

void
wl_egl_window_get_attached_size(struct wl_egl_window *egl_window,
				int *width, int *height);

#ifdef  __cplusplus
}
#endif

#endif
