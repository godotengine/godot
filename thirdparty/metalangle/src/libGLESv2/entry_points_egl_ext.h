//
// Copyright 2014 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// entry_points_egl_ext.h : Defines the EGL extension entry points.

#ifndef LIBGLESV2_ENTRYPOINTSEGLEXT_H_
#define LIBGLESV2_ENTRYPOINTSEGLEXT_H_

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <export.h>

extern "C" {

// EGL_ANGLE_query_surface_pointer
ANGLE_EXPORT EGLBoolean EGLAPIENTRY EGL_QuerySurfacePointerANGLE(EGLDisplay dpy,
                                                                 EGLSurface surface,
                                                                 EGLint attribute,
                                                                 void **value);

// EGL_NV_post_sub_buffer
ANGLE_EXPORT EGLBoolean EGLAPIENTRY EGL_PostSubBufferNV(EGLDisplay dpy,
                                                        EGLSurface surface,
                                                        EGLint x,
                                                        EGLint y,
                                                        EGLint width,
                                                        EGLint height);

// EGL_EXT_platform_base
ANGLE_EXPORT EGLDisplay EGLAPIENTRY EGL_GetPlatformDisplayEXT(EGLenum platform,
                                                              void *native_display,
                                                              const EGLint *attrib_list);
ANGLE_EXPORT EGLSurface EGLAPIENTRY EGL_CreatePlatformWindowSurfaceEXT(EGLDisplay dpy,
                                                                       EGLConfig config,
                                                                       void *native_window,
                                                                       const EGLint *attrib_list);
ANGLE_EXPORT EGLSurface EGLAPIENTRY EGL_CreatePlatformPixmapSurfaceEXT(EGLDisplay dpy,
                                                                       EGLConfig config,
                                                                       void *native_pixmap,
                                                                       const EGLint *attrib_list);

// EGL_EXT_device_query
ANGLE_EXPORT EGLBoolean EGLAPIENTRY EGL_QueryDisplayAttribEXT(EGLDisplay dpy,
                                                              EGLint attribute,
                                                              EGLAttrib *value);

// EGL_ANGLE_feature_control
ANGLE_EXPORT EGLBoolean EGLAPIENTRY EGL_QueryDisplayAttribANGLE(EGLDisplay dpy,
                                                                EGLint attribute,
                                                                EGLAttrib *value);

// EGL_EXT_device_query
ANGLE_EXPORT EGLBoolean EGLAPIENTRY EGL_QueryDeviceAttribEXT(EGLDeviceEXT device,
                                                             EGLint attribute,
                                                             EGLAttrib *value);
ANGLE_EXPORT const char *EGLAPIENTRY EGL_QueryDeviceStringEXT(EGLDeviceEXT device, EGLint name);

// EGL_KHR_image_base/EGL_KHR_image
ANGLE_EXPORT EGLImageKHR EGLAPIENTRY EGL_CreateImageKHR(EGLDisplay dpy,
                                                        EGLContext ctx,
                                                        EGLenum target,
                                                        EGLClientBuffer buffer,
                                                        const EGLint *attrib_list);
ANGLE_EXPORT EGLBoolean EGLAPIENTRY EGL_DestroyImageKHR(EGLDisplay dpy, EGLImageKHR image);

// EGL_ANGLE_device_creation
ANGLE_EXPORT EGLDeviceEXT EGLAPIENTRY EGL_CreateDeviceANGLE(EGLint device_type,
                                                            void *native_device,
                                                            const EGLAttrib *attrib_list);
ANGLE_EXPORT EGLBoolean EGLAPIENTRY EGL_ReleaseDeviceANGLE(EGLDeviceEXT device);

// EGL_KHR_stream
ANGLE_EXPORT EGLStreamKHR EGLAPIENTRY EGL_CreateStreamKHR(EGLDisplay dpy,
                                                          const EGLint *attrib_list);
ANGLE_EXPORT EGLBoolean EGLAPIENTRY EGL_DestroyStreamKHR(EGLDisplay dpy, EGLStreamKHR stream);
ANGLE_EXPORT EGLBoolean EGLAPIENTRY EGL_StreamAttribKHR(EGLDisplay dpy,
                                                        EGLStreamKHR stream,
                                                        EGLenum attribute,
                                                        EGLint value);
ANGLE_EXPORT EGLBoolean EGLAPIENTRY EGL_QueryStreamKHR(EGLDisplay dpy,
                                                       EGLStreamKHR stream,
                                                       EGLenum attribute,
                                                       EGLint *value);
ANGLE_EXPORT EGLBoolean EGLAPIENTRY EGL_QueryStreamu64KHR(EGLDisplay dpy,
                                                          EGLStreamKHR stream,
                                                          EGLenum attribute,
                                                          EGLuint64KHR *value);

// EGL_KHR_stream_consumer_gltexture
ANGLE_EXPORT EGLBoolean EGLAPIENTRY EGL_StreamConsumerGLTextureExternalKHR(EGLDisplay dpy,
                                                                           EGLStreamKHR stream);
ANGLE_EXPORT EGLBoolean EGLAPIENTRY EGL_StreamConsumerAcquireKHR(EGLDisplay dpy,
                                                                 EGLStreamKHR stream);
ANGLE_EXPORT EGLBoolean EGLAPIENTRY EGL_StreamConsumerReleaseKHR(EGLDisplay dpy,
                                                                 EGLStreamKHR stream);
ANGLE_EXPORT EGLBoolean EGLAPIENTRY
EGL_StreamConsumerGLTextureExternalAttribsNV(EGLDisplay dpy,
                                             EGLStreamKHR stream,
                                             const EGLAttrib *attrib_list);

// EGL_ANGLE_stream_producer_d3d_texture
ANGLE_EXPORT EGLBoolean EGLAPIENTRY
EGL_CreateStreamProducerD3DTextureANGLE(EGLDisplay dpy,
                                        EGLStreamKHR stream,
                                        const EGLAttrib *attrib_list);
ANGLE_EXPORT EGLBoolean EGLAPIENTRY EGL_StreamPostD3DTextureANGLE(EGLDisplay dpy,
                                                                  EGLStreamKHR stream,
                                                                  void *texture,
                                                                  const EGLAttrib *attrib_list);

// EGL_KHR_fence_sync
ANGLE_EXPORT EGLSync EGLAPIENTRY EGL_CreateSyncKHR(EGLDisplay dpy,
                                                   EGLenum type,
                                                   const EGLint *attrib_list);
ANGLE_EXPORT EGLBoolean EGLAPIENTRY EGL_DestroySyncKHR(EGLDisplay dpy, EGLSync sync);
ANGLE_EXPORT EGLint EGLAPIENTRY EGL_ClientWaitSyncKHR(EGLDisplay dpy,
                                                      EGLSync sync,
                                                      EGLint flags,
                                                      EGLTime timeout);
ANGLE_EXPORT EGLBoolean EGLAPIENTRY EGL_GetSyncAttribKHR(EGLDisplay dpy,
                                                         EGLSync sync,
                                                         EGLint attribute,
                                                         EGLint *value);

// EGL_KHR_wait_sync
ANGLE_EXPORT EGLBoolean EGLAPIENTRY EGL_WaitSyncKHR(EGLDisplay dpy, EGLSync sync, EGLint flags);

// EGL_CHROMIUM_get_sync_values
ANGLE_EXPORT EGLBoolean EGLAPIENTRY EGL_GetSyncValuesCHROMIUM(EGLDisplay dpy,
                                                              EGLSurface surface,
                                                              EGLuint64KHR *ust,
                                                              EGLuint64KHR *msc,
                                                              EGLuint64KHR *sbc);

// EGL_KHR_swap_buffers_with_damage
ANGLE_EXPORT EGLBoolean EGLAPIENTRY EGL_SwapBuffersWithDamageKHR(EGLDisplay dpy,
                                                                 EGLSurface surface,
                                                                 EGLint *rects,
                                                                 EGLint n_rects);

// EGL_ANDROID_presentation_time
ANGLE_EXPORT EGLBoolean EGLAPIENTRY EGL_PresentationTimeANDROID(EGLDisplay dpy,
                                                                EGLSurface surface,
                                                                EGLnsecsANDROID time);

// EGL_ANDRIOD_blob_cache
ANGLE_EXPORT void EGLAPIENTRY EGL_SetBlobCacheFuncsANDROID(EGLDisplay dpy,
                                                           EGLSetBlobFuncANDROID set,
                                                           EGLGetBlobFuncANDROID get);

// EGL_ANGLE_program_cache_control
ANGLE_EXPORT EGLint EGLAPIENTRY EGL_ProgramCacheGetAttribANGLE(EGLDisplay dpy, EGLenum attrib);
ANGLE_EXPORT void EGLAPIENTRY EGL_ProgramCacheQueryANGLE(EGLDisplay dpy,
                                                         EGLint index,
                                                         void *key,
                                                         EGLint *keysize,
                                                         void *binary,
                                                         EGLint *binarysize);
ANGLE_EXPORT void EGLAPIENTRY EGL_ProgramCachePopulateANGLE(EGLDisplay dpy,
                                                            const void *key,
                                                            EGLint keysize,
                                                            const void *binary,
                                                            EGLint binarysize);
ANGLE_EXPORT EGLint EGLAPIENTRY EGL_ProgramCacheResizeANGLE(EGLDisplay dpy,
                                                            EGLint limit,
                                                            EGLenum mode);

// EGL_KHR_debug
ANGLE_EXPORT EGLint EGLAPIENTRY EGL_DebugMessageControlKHR(EGLDEBUGPROCKHR callback,
                                                           const EGLAttrib *attrib_list);

ANGLE_EXPORT EGLBoolean EGLAPIENTRY EGL_QueryDebugKHR(EGLint attribute, EGLAttrib *value);

ANGLE_EXPORT EGLint EGLAPIENTRY EGL_LabelObjectKHR(EGLDisplay display,
                                                   EGLenum objectType,
                                                   EGLObjectKHR object,
                                                   EGLLabelKHR label);

// ANDROID_get_frame_timestamps
ANGLE_EXPORT EGLBoolean EGLAPIENTRY EGL_GetCompositorTimingSupportedANDROID(EGLDisplay dpy,
                                                                            EGLSurface surface,
                                                                            EGLint name);

ANGLE_EXPORT EGLBoolean EGLAPIENTRY EGL_GetCompositorTimingANDROID(EGLDisplay dpy,
                                                                   EGLSurface surface,
                                                                   EGLint numTimestamps,
                                                                   const EGLint *names,
                                                                   EGLnsecsANDROID *values);

ANGLE_EXPORT EGLBoolean EGLAPIENTRY EGL_GetNextFrameIdANDROID(EGLDisplay dpy,
                                                              EGLSurface surface,
                                                              EGLuint64KHR *frameId);

ANGLE_EXPORT EGLBoolean EGLAPIENTRY EGL_GetFrameTimestampSupportedANDROID(EGLDisplay dpy,
                                                                          EGLSurface surface,
                                                                          EGLint timestamp);

ANGLE_EXPORT EGLBoolean EGLAPIENTRY EGL_GetFrameTimestampsANDROID(EGLDisplay dpy,
                                                                  EGLSurface surface,
                                                                  EGLuint64KHR frameId,
                                                                  EGLint numTimestamps,
                                                                  const EGLint *timestamps,
                                                                  EGLnsecsANDROID *values);

// EGL_ANGLE_feature_control
ANGLE_EXPORT const char *EGLAPIENTRY EGL_QueryStringiANGLE(EGLDisplay dpy,
                                                           EGLint name,
                                                           EGLint index);

// EGL_ANDROID_get_native_client_buffer
ANGLE_EXPORT EGLClientBuffer EGLAPIENTRY
EGL_GetNativeClientBufferANDROID(const struct AHardwareBuffer *buffer);

// EGL_ANDROID_native_fence_sync
ANGLE_EXPORT EGLint EGLAPIENTRY EGL_DupNativeFenceFDANDROID(EGLDisplay dpy, EGLSyncKHR sync);

}  // extern "C"

#endif  // LIBGLESV2_ENTRYPOINTSEGLEXT_H_
