//
// Copyright 2014 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// entry_points_egl.h : Defines the EGL entry points.

#ifndef LIBGLESV2_ENTRYPOINTSEGL_H_
#define LIBGLESV2_ENTRYPOINTSEGL_H_

#include <EGL/egl.h>
#include <export.h>

extern "C" {

// EGL 1.0
ANGLE_EXPORT EGLBoolean EGLAPIENTRY EGL_ChooseConfig(EGLDisplay dpy,
                                                     const EGLint *attrib_list,
                                                     EGLConfig *configs,
                                                     EGLint config_size,
                                                     EGLint *num_config);
ANGLE_EXPORT EGLBoolean EGLAPIENTRY EGL_CopyBuffers(EGLDisplay dpy,
                                                    EGLSurface surface,
                                                    EGLNativePixmapType target);
ANGLE_EXPORT EGLContext EGLAPIENTRY EGL_CreateContext(EGLDisplay dpy,
                                                      EGLConfig config,
                                                      EGLContext share_context,
                                                      const EGLint *attrib_list);
ANGLE_EXPORT EGLSurface EGLAPIENTRY EGL_CreatePbufferSurface(EGLDisplay dpy,
                                                             EGLConfig config,
                                                             const EGLint *attrib_list);
ANGLE_EXPORT EGLSurface EGLAPIENTRY EGL_CreatePixmapSurface(EGLDisplay dpy,
                                                            EGLConfig config,
                                                            EGLNativePixmapType pixmap,
                                                            const EGLint *attrib_list);
ANGLE_EXPORT EGLSurface EGLAPIENTRY EGL_CreateWindowSurface(EGLDisplay dpy,
                                                            EGLConfig config,
                                                            EGLNativeWindowType win,
                                                            const EGLint *attrib_list);
ANGLE_EXPORT EGLBoolean EGLAPIENTRY EGL_DestroyContext(EGLDisplay dpy, EGLContext ctx);
ANGLE_EXPORT EGLBoolean EGLAPIENTRY EGL_DestroySurface(EGLDisplay dpy, EGLSurface surface);
ANGLE_EXPORT EGLBoolean EGLAPIENTRY EGL_GetConfigAttrib(EGLDisplay dpy,
                                                        EGLConfig config,
                                                        EGLint attribute,
                                                        EGLint *value);
ANGLE_EXPORT EGLBoolean EGLAPIENTRY EGL_GetConfigs(EGLDisplay dpy,
                                                   EGLConfig *configs,
                                                   EGLint config_size,
                                                   EGLint *num_config);
ANGLE_EXPORT EGLDisplay EGLAPIENTRY EGL_GetCurrentDisplay(void);
ANGLE_EXPORT EGLSurface EGLAPIENTRY EGL_GetCurrentSurface(EGLint readdraw);
ANGLE_EXPORT EGLDisplay EGLAPIENTRY EGL_GetDisplay(EGLNativeDisplayType display_id);
ANGLE_EXPORT EGLint EGLAPIENTRY EGL_GetError(void);
ANGLE_EXPORT __eglMustCastToProperFunctionPointerType EGLAPIENTRY
EGL_GetProcAddress(const char *procname);
ANGLE_EXPORT EGLBoolean EGLAPIENTRY EGL_Initialize(EGLDisplay dpy, EGLint *major, EGLint *minor);
ANGLE_EXPORT EGLBoolean EGLAPIENTRY EGL_MakeCurrent(EGLDisplay dpy,
                                                    EGLSurface draw,
                                                    EGLSurface read,
                                                    EGLContext ctx);
ANGLE_EXPORT EGLBoolean EGLAPIENTRY EGL_QueryContext(EGLDisplay dpy,
                                                     EGLContext ctx,
                                                     EGLint attribute,
                                                     EGLint *value);
ANGLE_EXPORT const char *EGLAPIENTRY EGL_QueryString(EGLDisplay dpy, EGLint name);
ANGLE_EXPORT EGLBoolean EGLAPIENTRY EGL_QuerySurface(EGLDisplay dpy,
                                                     EGLSurface surface,
                                                     EGLint attribute,
                                                     EGLint *value);
ANGLE_EXPORT EGLBoolean EGLAPIENTRY EGL_SwapBuffers(EGLDisplay dpy, EGLSurface surface);
ANGLE_EXPORT EGLBoolean EGLAPIENTRY EGL_Terminate(EGLDisplay dpy);
ANGLE_EXPORT EGLBoolean EGLAPIENTRY EGL_WaitGL(void);
ANGLE_EXPORT EGLBoolean EGLAPIENTRY EGL_WaitNative(EGLint engine);

// EGL 1.1
ANGLE_EXPORT EGLBoolean EGLAPIENTRY EGL_BindTexImage(EGLDisplay dpy,
                                                     EGLSurface surface,
                                                     EGLint buffer);
ANGLE_EXPORT EGLBoolean EGLAPIENTRY EGL_ReleaseTexImage(EGLDisplay dpy,
                                                        EGLSurface surface,
                                                        EGLint buffer);
ANGLE_EXPORT EGLBoolean EGLAPIENTRY EGL_SurfaceAttrib(EGLDisplay dpy,
                                                      EGLSurface surface,
                                                      EGLint attribute,
                                                      EGLint value);
ANGLE_EXPORT EGLBoolean EGLAPIENTRY EGL_SwapInterval(EGLDisplay dpy, EGLint interval);

// EGL 1.2
ANGLE_EXPORT EGLBoolean EGLAPIENTRY EGL_BindAPI(EGLenum api);
ANGLE_EXPORT EGLenum EGLAPIENTRY EGL_QueryAPI(void);
ANGLE_EXPORT EGLSurface EGLAPIENTRY EGL_CreatePbufferFromClientBuffer(EGLDisplay dpy,
                                                                      EGLenum buftype,
                                                                      EGLClientBuffer buffer,
                                                                      EGLConfig config,
                                                                      const EGLint *attrib_list);
ANGLE_EXPORT EGLBoolean EGLAPIENTRY EGL_ReleaseThread(void);
ANGLE_EXPORT EGLBoolean EGLAPIENTRY EGL_WaitClient(void);

// EGL 1.4
ANGLE_EXPORT EGLContext EGLAPIENTRY EGL_GetCurrentContext(void);

// EGL 1.5
ANGLE_EXPORT EGLSync EGLAPIENTRY EGL_CreateSync(EGLDisplay dpy,
                                                EGLenum type,
                                                const EGLAttrib *attrib_list);
ANGLE_EXPORT EGLBoolean EGLAPIENTRY EGL_DestroySync(EGLDisplay dpy, EGLSync sync);
ANGLE_EXPORT EGLint EGLAPIENTRY EGL_ClientWaitSync(EGLDisplay dpy,
                                                   EGLSync sync,
                                                   EGLint flags,
                                                   EGLTime timeout);
ANGLE_EXPORT EGLBoolean EGLAPIENTRY EGL_GetSyncAttrib(EGLDisplay dpy,
                                                      EGLSync sync,
                                                      EGLint attribute,
                                                      EGLAttrib *value);
ANGLE_EXPORT EGLImage EGLAPIENTRY EGL_CreateImage(EGLDisplay dpy,
                                                  EGLContext ctx,
                                                  EGLenum target,
                                                  EGLClientBuffer buffer,
                                                  const EGLAttrib *attrib_list);
ANGLE_EXPORT EGLBoolean EGLAPIENTRY EGL_DestroyImage(EGLDisplay dpy, EGLImage image);
ANGLE_EXPORT EGLDisplay EGLAPIENTRY EGL_GetPlatformDisplay(EGLenum platform,
                                                           void *native_display,
                                                           const EGLAttrib *attrib_list);
ANGLE_EXPORT EGLSurface EGLAPIENTRY EGL_CreatePlatformWindowSurface(EGLDisplay dpy,
                                                                    EGLConfig config,
                                                                    void *native_window,
                                                                    const EGLAttrib *attrib_list);
ANGLE_EXPORT EGLSurface EGLAPIENTRY EGL_CreatePlatformPixmapSurface(EGLDisplay dpy,
                                                                    EGLConfig config,
                                                                    void *native_pixmap,
                                                                    const EGLAttrib *attrib_list);
ANGLE_EXPORT EGLBoolean EGLAPIENTRY EGL_WaitSync(EGLDisplay dpy, EGLSync sync, EGLint flags);
}  // extern "C"

#endif  // LIBGLESV2_ENTRYPOINTSEGL_H_
