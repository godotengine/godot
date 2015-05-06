/**************************************************************************
 *
 * Copyright 2008 Tungsten Graphics, Inc., Cedar Park, Texas.
 * All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sub license, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice (including the
 * next paragraph) shall be included in all copies or substantial portions
 * of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 **************************************************************************/

#ifndef __eglmesaext_h_
#define __eglmesaext_h_

#ifdef __cplusplus
extern "C" {
#endif

#include <EGL/eglplatform.h>

/* EGL_MESA_screen extension  >>> PRELIMINARY <<< */
#ifndef EGL_MESA_screen_surface
#define EGL_MESA_screen_surface 1

#define EGL_BAD_SCREEN_MESA                    0x4000
#define EGL_BAD_MODE_MESA                      0x4001
#define EGL_SCREEN_COUNT_MESA                  0x4002
#define EGL_SCREEN_POSITION_MESA               0x4003
#define EGL_SCREEN_POSITION_GRANULARITY_MESA   0x4004
#define EGL_MODE_ID_MESA                       0x4005
#define EGL_REFRESH_RATE_MESA                  0x4006
#define EGL_OPTIMAL_MESA                       0x4007
#define EGL_INTERLACED_MESA                    0x4008
#define EGL_SCREEN_BIT_MESA                    0x08

typedef khronos_uint32_t EGLScreenMESA;
typedef khronos_uint32_t EGLModeMESA;

#ifdef EGL_EGLEXT_PROTOTYPES
EGLAPI EGLBoolean EGLAPIENTRY eglChooseModeMESA(EGLDisplay dpy, EGLScreenMESA screen, const EGLint *attrib_list, EGLModeMESA *modes, EGLint modes_size, EGLint *num_modes);
EGLAPI EGLBoolean EGLAPIENTRY eglGetModesMESA(EGLDisplay dpy, EGLScreenMESA screen, EGLModeMESA *modes, EGLint modes_size, EGLint *num_modes);
EGLAPI EGLBoolean EGLAPIENTRY eglGetModeAttribMESA(EGLDisplay dpy, EGLModeMESA mode, EGLint attribute, EGLint *value);
EGLAPI EGLBoolean EGLAPIENTRY eglGetScreensMESA(EGLDisplay dpy, EGLScreenMESA *screens, EGLint max_screens, EGLint *num_screens);
EGLAPI EGLSurface EGLAPIENTRY eglCreateScreenSurfaceMESA(EGLDisplay dpy, EGLConfig config, const EGLint *attrib_list);
EGLAPI EGLBoolean EGLAPIENTRY eglShowScreenSurfaceMESA(EGLDisplay dpy, EGLint screen, EGLSurface surface, EGLModeMESA mode);
EGLAPI EGLBoolean EGLAPIENTRY eglScreenPositionMESA(EGLDisplay dpy, EGLScreenMESA screen, EGLint x, EGLint y);
EGLAPI EGLBoolean EGLAPIENTRY eglQueryScreenMESA(EGLDisplay dpy, EGLScreenMESA screen, EGLint attribute, EGLint *value);
EGLAPI EGLBoolean EGLAPIENTRY eglQueryScreenSurfaceMESA(EGLDisplay dpy, EGLScreenMESA screen, EGLSurface *surface);
EGLAPI EGLBoolean EGLAPIENTRY eglQueryScreenModeMESA(EGLDisplay dpy, EGLScreenMESA screen, EGLModeMESA *mode);
EGLAPI const char * EGLAPIENTRY eglQueryModeStringMESA(EGLDisplay dpy, EGLModeMESA mode);
#endif /* EGL_EGLEXT_PROTOTYPES */

typedef EGLBoolean (EGLAPIENTRYP PFNEGLCHOOSEMODEMESA) (EGLDisplay dpy, EGLScreenMESA screen, const EGLint *attrib_list, EGLModeMESA *modes, EGLint modes_size, EGLint *num_modes);
typedef EGLBoolean (EGLAPIENTRYP PFNEGLGETMODESMESA) (EGLDisplay dpy, EGLScreenMESA screen, EGLModeMESA *modes, EGLint modes_size, EGLint *num_modes);
typedef EGLBoolean (EGLAPIENTRYP PFNEGLGetModeATTRIBMESA) (EGLDisplay dpy, EGLModeMESA mode, EGLint attribute, EGLint *value);
typedef EGLBoolean (EGLAPIENTRYP PFNEGLGETSCRREENSMESA) (EGLDisplay dpy, EGLScreenMESA *screens, EGLint max_screens, EGLint *num_screens);
typedef EGLSurface (EGLAPIENTRYP PFNEGLCREATESCREENSURFACEMESA) (EGLDisplay dpy, EGLConfig config, const EGLint *attrib_list);
typedef EGLBoolean (EGLAPIENTRYP PFNEGLSHOWSCREENSURFACEMESA) (EGLDisplay dpy, EGLint screen, EGLSurface surface, EGLModeMESA mode);
typedef EGLBoolean (EGLAPIENTRYP PFNEGLSCREENPOSIITONMESA) (EGLDisplay dpy, EGLScreenMESA screen, EGLint x, EGLint y);
typedef EGLBoolean (EGLAPIENTRYP PFNEGLQUERYSCREENMESA) (EGLDisplay dpy, EGLScreenMESA screen, EGLint attribute, EGLint *value);
typedef EGLBoolean (EGLAPIENTRYP PFNEGLQUERYSCREENSURFACEMESA) (EGLDisplay dpy, EGLScreenMESA screen, EGLSurface *surface);
typedef EGLBoolean (EGLAPIENTRYP PFNEGLQUERYSCREENMODEMESA) (EGLDisplay dpy, EGLScreenMESA screen, EGLModeMESA *mode);
typedef const char * (EGLAPIENTRYP PFNEGLQUERYMODESTRINGMESA) (EGLDisplay dpy, EGLModeMESA mode);

#endif /* EGL_MESA_screen_surface */

#ifndef EGL_MESA_copy_context
#define EGL_MESA_copy_context 1

#ifdef EGL_EGLEXT_PROTOTYPES
EGLAPI EGLBoolean EGLAPIENTRY eglCopyContextMESA(EGLDisplay dpy, EGLContext source, EGLContext dest, EGLint mask);
#endif /* EGL_EGLEXT_PROTOTYPES */

typedef EGLBoolean (EGLAPIENTRYP PFNEGLCOPYCONTEXTMESA) (EGLDisplay dpy, EGLContext source, EGLContext dest, EGLint mask);

#endif /* EGL_MESA_copy_context */

#ifndef EGL_MESA_drm_display
#define EGL_MESA_drm_display 1

#ifdef EGL_EGLEXT_PROTOTYPES
EGLAPI EGLDisplay EGLAPIENTRY eglGetDRMDisplayMESA(int fd);
#endif /* EGL_EGLEXT_PROTOTYPES */

typedef EGLDisplay (EGLAPIENTRYP PFNEGLGETDRMDISPLAYMESA) (int fd);

#endif /* EGL_MESA_drm_display */

#ifdef EGL_MESA_drm_image
/* Mesa's extension to EGL_MESA_drm_image... */
#ifndef EGL_DRM_BUFFER_USE_CURSOR_MESA
#define EGL_DRM_BUFFER_USE_CURSOR_MESA		0x0004
#endif
#endif

#ifndef EGL_WL_bind_wayland_display
#define EGL_WL_bind_wayland_display 1

#define EGL_WAYLAND_BUFFER_WL			0x31D5 /* eglCreateImageKHR target */
struct wl_display;
#ifdef EGL_EGLEXT_PROTOTYPES
EGLAPI EGLBoolean EGLAPIENTRY eglBindWaylandDisplayWL(EGLDisplay dpy, struct wl_display *display);
EGLAPI EGLBoolean EGLAPIENTRY eglUnbindWaylandDisplayWL(EGLDisplay dpy, struct wl_display *display);
#endif
typedef EGLBoolean (EGLAPIENTRYP PFNEGLBINDWAYLANDDISPLAYWL) (EGLDisplay dpy, struct wl_display *display);
typedef EGLBoolean (EGLAPIENTRYP PFNEGLUNBINDWAYLANDDISPLAYWL) (EGLDisplay dpy, struct wl_display *display);
#endif

#ifndef EGL_NOK_swap_region
#define EGL_NOK_swap_region 1

#ifdef EGL_EGLEXT_PROTOTYPES
EGLAPI EGLBoolean EGLAPIENTRY eglSwapBuffersRegionNOK(EGLDisplay dpy, EGLSurface surface, EGLint numRects, const EGLint* rects);
#endif

typedef EGLBoolean (EGLAPIENTRYP PFNEGLSWAPBUFFERSREGIONNOK) (EGLDisplay dpy, EGLSurface surface, EGLint numRects, const EGLint* rects);
#endif

#ifndef EGL_NOK_texture_from_pixmap
#define EGL_NOK_texture_from_pixmap 1

#define EGL_Y_INVERTED_NOK			0x307F
#endif /* EGL_NOK_texture_from_pixmap */

#ifndef EGL_ANDROID_image_native_buffer
#define EGL_ANDROID_image_native_buffer 1
#define EGL_NATIVE_BUFFER_ANDROID       0x3140  /* eglCreateImageKHR target */
#endif

#ifdef __cplusplus
}
#endif

#endif