/*
 * Copyright (c) 2011-2013 Research In Motion Limited.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _UTILITY_H_INCLUDED
#define _UTILITY_H_INCLUDED

#include <EGL/egl.h>
#include <GLES2/gl2.h>
#include <screen/screen.h>
#include <sys/platform.h>

#ifdef __cplusplus
extern "C" {
#endif

extern EGLDisplay egl_disp;
extern EGLSurface egl_surf;

enum RENDERING_API {
	GL_ES_1 = EGL_OPENGL_ES_BIT,
	GL_ES_2 = EGL_OPENGL_ES2_BIT,
	VG = EGL_OPENVG_BIT
};

/**
 * Initializes EGL, GL and loads a default font
 *
 * \param libscreen context that will be used for EGL setup
 * \return EXIT_SUCCESS if initialization succeeded otherwise EXIT_FAILURE
 */
int bbutil_init(screen_context_t ctx, enum RENDERING_API api);

/**
 * Initializes EGL
 *
 * \param libscreen context that will be used for EGL setup
 * \return EXIT_SUCCESS if initialization succeeded otherwise EXIT_FAILURE
 */
int bbutil_init_egl(screen_context_t ctx, enum RENDERING_API api);

/**
 * Initializes GL 1.1 for simple 2D rendering. GL2 initialization will be added at a later point.
 *
 * \return EXIT_SUCCESS if initialization succeeded otherwise EXIT_FAILURE
 */
int bbutil_init_gl2d();

int bbutil_is_flipped();
int bbutil_get_rotation();

char *get_window_group_id();

int bbutil_rotate_screen_surface(int angle);

/**
 * Terminates EGL
 */
void bbutil_terminate();

/**
 * Swaps default bbutil window surface to the screen
 */
void bbutil_swap();

/**
 * Clears the screen of any existing text.
 * NOTE: must be called after a successful return from bbutil_init() or bbutil_init_egl() call
 */
void bbutil_clear();

#ifdef __cplusplus
};
#endif

#endif
