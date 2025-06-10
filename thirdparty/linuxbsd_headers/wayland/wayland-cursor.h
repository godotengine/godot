/*
 * Copyright © 2012 Intel Corporation
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

#ifndef WAYLAND_CURSOR_H
#define WAYLAND_CURSOR_H

#include <stdint.h>

#ifdef  __cplusplus
extern "C" {
#endif

struct wl_cursor_theme;
struct wl_buffer;
struct wl_shm;

/** A still image part of a cursor
 *
 * Use `wl_cursor_image_get_buffer()` to get the corresponding `struct
 * wl_buffer` to attach to your `struct wl_surface`. */
struct wl_cursor_image {
	/** Actual width */
	uint32_t width;

	/** Actual height */
	uint32_t height;

	/** Hot spot x (must be inside image) */
	uint32_t hotspot_x;

	/** Hot spot y (must be inside image) */
	uint32_t hotspot_y;

	/** Animation delay to next frame (ms) */
	uint32_t delay;
};

/** A cursor, as returned by `wl_cursor_theme_get_cursor()` */
struct wl_cursor {
	/** How many images there are in this cursor’s animation */
	unsigned int image_count;

	/** The array of still images composing this animation */
	struct wl_cursor_image **images;

	/** The name of this cursor */
	char *name;
};

struct wl_cursor_theme *
wl_cursor_theme_load(const char *name, int size, struct wl_shm *shm);

void
wl_cursor_theme_destroy(struct wl_cursor_theme *theme);

struct wl_cursor *
wl_cursor_theme_get_cursor(struct wl_cursor_theme *theme,
			   const char *name);

struct wl_buffer *
wl_cursor_image_get_buffer(struct wl_cursor_image *image);

int
wl_cursor_frame(struct wl_cursor *cursor, uint32_t time);

int
wl_cursor_frame_and_duration(struct wl_cursor *cursor, uint32_t time,
			     uint32_t *duration);

#ifdef  __cplusplus
}
#endif

#endif
