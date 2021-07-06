//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// libEGL.cpp: Implements the exported EGL functions.

#include "anglebase/no_destructor.h"
#include "common/system_utils.h"

#include <memory>

#if defined(ANGLE_USE_EGL_LOADER)
#    include "libEGL/egl_loader_autogen.h"
#else
#    include "libGLESv2/entry_points_egl.h"
#    include "libGLESv2/entry_points_egl_ext.h"
#endif  // defined(ANGLE_USE_EGL_LOADER)

namespace
{
#if defined(ANGLE_USE_EGL_LOADER)
bool gLoaded = false;

std::unique_ptr<angle::Library> &EntryPointsLib()
{
    static angle::base::NoDestructor<std::unique_ptr<angle::Library>> entryPointsLib;
    return *entryPointsLib;
}

angle::GenericProc KHRONOS_APIENTRY GlobalLoad(const char *symbol)
{
    return reinterpret_cast<angle::GenericProc>(EntryPointsLib()->getSymbol(symbol));
}

void EnsureEGLLoaded()
{
    if (gLoaded)
        return;

    EntryPointsLib().reset(
        angle::OpenSharedLibrary(ANGLE_GLESV2_LIBRARY_NAME, angle::SearchType::ApplicationDir));
    angle::LoadEGL_EGL(GlobalLoad);
    if (!EGL_GetPlatformDisplay)
    {
        fprintf(stderr, "Error loading EGL entry points.\n");
    }
    else
    {
        gLoaded = true;
    }
}
#else
void EnsureEGLLoaded() {}
#endif  // defined(ANGLE_USE_EGL_LOADER)
}  // anonymous namespace

extern "C" {

EGLBoolean EGLAPIENTRY eglChooseConfig(EGLDisplay dpy,
                                       const EGLint *attrib_list,
                                       EGLConfig *configs,
                                       EGLint config_size,
                                       EGLint *num_config)
{
    EnsureEGLLoaded();
    return EGL_ChooseConfig(dpy, attrib_list, configs, config_size, num_config);
}

EGLBoolean EGLAPIENTRY eglCopyBuffers(EGLDisplay dpy,
                                      EGLSurface surface,
                                      EGLNativePixmapType target)
{
    EnsureEGLLoaded();
    return EGL_CopyBuffers(dpy, surface, target);
}

EGLContext EGLAPIENTRY eglCreateContext(EGLDisplay dpy,
                                        EGLConfig config,
                                        EGLContext share_context,
                                        const EGLint *attrib_list)
{
    EnsureEGLLoaded();
    return EGL_CreateContext(dpy, config, share_context, attrib_list);
}

EGLSurface EGLAPIENTRY eglCreatePbufferSurface(EGLDisplay dpy,
                                               EGLConfig config,
                                               const EGLint *attrib_list)
{
    EnsureEGLLoaded();
    return EGL_CreatePbufferSurface(dpy, config, attrib_list);
}

EGLSurface EGLAPIENTRY eglCreatePixmapSurface(EGLDisplay dpy,
                                              EGLConfig config,
                                              EGLNativePixmapType pixmap,
                                              const EGLint *attrib_list)
{
    EnsureEGLLoaded();
    return EGL_CreatePixmapSurface(dpy, config, pixmap, attrib_list);
}

EGLSurface EGLAPIENTRY eglCreateWindowSurface(EGLDisplay dpy,
                                              EGLConfig config,
                                              EGLNativeWindowType win,
                                              const EGLint *attrib_list)
{
    EnsureEGLLoaded();
    return EGL_CreateWindowSurface(dpy, config, win, attrib_list);
}

EGLBoolean EGLAPIENTRY eglDestroyContext(EGLDisplay dpy, EGLContext ctx)
{
    EnsureEGLLoaded();
    return EGL_DestroyContext(dpy, ctx);
}

EGLBoolean EGLAPIENTRY eglDestroySurface(EGLDisplay dpy, EGLSurface surface)
{
    EnsureEGLLoaded();
    return EGL_DestroySurface(dpy, surface);
}

EGLBoolean EGLAPIENTRY eglGetConfigAttrib(EGLDisplay dpy,
                                          EGLConfig config,
                                          EGLint attribute,
                                          EGLint *value)
{
    EnsureEGLLoaded();
    return EGL_GetConfigAttrib(dpy, config, attribute, value);
}

EGLBoolean EGLAPIENTRY eglGetConfigs(EGLDisplay dpy,
                                     EGLConfig *configs,
                                     EGLint config_size,
                                     EGLint *num_config)
{
    EnsureEGLLoaded();
    return EGL_GetConfigs(dpy, configs, config_size, num_config);
}

EGLDisplay EGLAPIENTRY eglGetCurrentDisplay(void)
{
    EnsureEGLLoaded();
    return EGL_GetCurrentDisplay();
}

EGLSurface EGLAPIENTRY eglGetCurrentSurface(EGLint readdraw)
{
    EnsureEGLLoaded();
    return EGL_GetCurrentSurface(readdraw);
}

EGLDisplay EGLAPIENTRY eglGetDisplay(EGLNativeDisplayType display_id)
{
    EnsureEGLLoaded();
    return EGL_GetDisplay(display_id);
}

EGLint EGLAPIENTRY eglGetError(void)
{
    EnsureEGLLoaded();
    return EGL_GetError();
}

EGLBoolean EGLAPIENTRY eglInitialize(EGLDisplay dpy, EGLint *major, EGLint *minor)
{
    EnsureEGLLoaded();
    return EGL_Initialize(dpy, major, minor);
}

EGLBoolean EGLAPIENTRY eglMakeCurrent(EGLDisplay dpy,
                                      EGLSurface draw,
                                      EGLSurface read,
                                      EGLContext ctx)
{
    EnsureEGLLoaded();
    return EGL_MakeCurrent(dpy, draw, read, ctx);
}

EGLBoolean EGLAPIENTRY eglQueryContext(EGLDisplay dpy,
                                       EGLContext ctx,
                                       EGLint attribute,
                                       EGLint *value)
{
    EnsureEGLLoaded();
    return EGL_QueryContext(dpy, ctx, attribute, value);
}

const char *EGLAPIENTRY eglQueryString(EGLDisplay dpy, EGLint name)
{
    EnsureEGLLoaded();
    return EGL_QueryString(dpy, name);
}

EGLBoolean EGLAPIENTRY eglQuerySurface(EGLDisplay dpy,
                                       EGLSurface surface,
                                       EGLint attribute,
                                       EGLint *value)
{
    EnsureEGLLoaded();
    return EGL_QuerySurface(dpy, surface, attribute, value);
}

EGLBoolean EGLAPIENTRY eglSwapBuffers(EGLDisplay dpy, EGLSurface surface)
{
    EnsureEGLLoaded();
    return EGL_SwapBuffers(dpy, surface);
}

EGLBoolean EGLAPIENTRY eglTerminate(EGLDisplay dpy)
{
    EnsureEGLLoaded();
    return EGL_Terminate(dpy);
}

EGLBoolean EGLAPIENTRY eglWaitGL(void)
{
    EnsureEGLLoaded();
    return EGL_WaitGL();
}

EGLBoolean EGLAPIENTRY eglWaitNative(EGLint engine)
{
    EnsureEGLLoaded();
    return EGL_WaitNative(engine);
}

EGLBoolean EGLAPIENTRY eglBindTexImage(EGLDisplay dpy, EGLSurface surface, EGLint buffer)
{
    EnsureEGLLoaded();
    return EGL_BindTexImage(dpy, surface, buffer);
}

EGLBoolean EGLAPIENTRY eglReleaseTexImage(EGLDisplay dpy, EGLSurface surface, EGLint buffer)
{
    EnsureEGLLoaded();
    return EGL_ReleaseTexImage(dpy, surface, buffer);
}

EGLBoolean EGLAPIENTRY eglSurfaceAttrib(EGLDisplay dpy,
                                        EGLSurface surface,
                                        EGLint attribute,
                                        EGLint value)
{
    EnsureEGLLoaded();
    return EGL_SurfaceAttrib(dpy, surface, attribute, value);
}

EGLBoolean EGLAPIENTRY eglSwapInterval(EGLDisplay dpy, EGLint interval)
{
    EnsureEGLLoaded();
    return EGL_SwapInterval(dpy, interval);
}

EGLBoolean EGLAPIENTRY eglBindAPI(EGLenum api)
{
    EnsureEGLLoaded();
    return EGL_BindAPI(api);
}

EGLenum EGLAPIENTRY eglQueryAPI(void)
{
    EnsureEGLLoaded();
    return EGL_QueryAPI();
}

EGLSurface EGLAPIENTRY eglCreatePbufferFromClientBuffer(EGLDisplay dpy,
                                                        EGLenum buftype,
                                                        EGLClientBuffer buffer,
                                                        EGLConfig config,
                                                        const EGLint *attrib_list)
{
    EnsureEGLLoaded();
    return EGL_CreatePbufferFromClientBuffer(dpy, buftype, buffer, config, attrib_list);
}

EGLBoolean EGLAPIENTRY eglReleaseThread(void)
{
    EnsureEGLLoaded();
    return EGL_ReleaseThread();
}

EGLBoolean EGLAPIENTRY eglWaitClient(void)
{
    EnsureEGLLoaded();
    return EGL_WaitClient();
}

EGLContext EGLAPIENTRY eglGetCurrentContext(void)
{
    EnsureEGLLoaded();
    return EGL_GetCurrentContext();
}

EGLSync EGLAPIENTRY eglCreateSync(EGLDisplay dpy, EGLenum type, const EGLAttrib *attrib_list)
{
    EnsureEGLLoaded();
    return EGL_CreateSync(dpy, type, attrib_list);
}

EGLBoolean EGLAPIENTRY eglDestroySync(EGLDisplay dpy, EGLSync sync)
{
    EnsureEGLLoaded();
    return EGL_DestroySync(dpy, sync);
}

EGLint EGLAPIENTRY eglClientWaitSync(EGLDisplay dpy, EGLSync sync, EGLint flags, EGLTime timeout)
{
    EnsureEGLLoaded();
    return EGL_ClientWaitSync(dpy, sync, flags, timeout);
}

EGLBoolean EGLAPIENTRY eglGetSyncAttrib(EGLDisplay dpy,
                                        EGLSync sync,
                                        EGLint attribute,
                                        EGLAttrib *value)
{
    EnsureEGLLoaded();
    return EGL_GetSyncAttrib(dpy, sync, attribute, value);
}

EGLImage EGLAPIENTRY eglCreateImage(EGLDisplay dpy,
                                    EGLContext ctx,
                                    EGLenum target,
                                    EGLClientBuffer buffer,
                                    const EGLAttrib *attrib_list)
{
    EnsureEGLLoaded();
    return EGL_CreateImage(dpy, ctx, target, buffer, attrib_list);
}

EGLBoolean EGLAPIENTRY eglDestroyImage(EGLDisplay dpy, EGLImage image)
{
    EnsureEGLLoaded();
    return EGL_DestroyImage(dpy, image);
}

EGLDisplay EGLAPIENTRY eglGetPlatformDisplay(EGLenum platform,
                                             void *native_display,
                                             const EGLAttrib *attrib_list)
{
    EnsureEGLLoaded();
    return EGL_GetPlatformDisplay(platform, native_display, attrib_list);
}

EGLSurface EGLAPIENTRY eglCreatePlatformWindowSurface(EGLDisplay dpy,
                                                      EGLConfig config,
                                                      void *native_window,
                                                      const EGLAttrib *attrib_list)
{
    EnsureEGLLoaded();
    return EGL_CreatePlatformWindowSurface(dpy, config, native_window, attrib_list);
}

EGLSurface EGLAPIENTRY eglCreatePlatformPixmapSurface(EGLDisplay dpy,
                                                      EGLConfig config,
                                                      void *native_pixmap,
                                                      const EGLAttrib *attrib_list)
{
    EnsureEGLLoaded();
    return EGL_CreatePlatformPixmapSurface(dpy, config, native_pixmap, attrib_list);
}

EGLBoolean EGLAPIENTRY eglWaitSync(EGLDisplay dpy, EGLSync sync, EGLint flags)
{
    EnsureEGLLoaded();
    return EGL_WaitSync(dpy, sync, flags);
}

EGLBoolean EGLAPIENTRY eglQuerySurfacePointerANGLE(EGLDisplay dpy,
                                                   EGLSurface surface,
                                                   EGLint attribute,
                                                   void **value)
{
    EnsureEGLLoaded();
    return EGL_QuerySurfacePointerANGLE(dpy, surface, attribute, value);
}

EGLBoolean EGLAPIENTRY eglPostSubBufferNV(EGLDisplay dpy,
                                          EGLSurface surface,
                                          EGLint x,
                                          EGLint y,
                                          EGLint width,
                                          EGLint height)
{
    EnsureEGLLoaded();
    return EGL_PostSubBufferNV(dpy, surface, x, y, width, height);
}

EGLDisplay EGLAPIENTRY eglGetPlatformDisplayEXT(EGLenum platform,
                                                void *native_display,
                                                const EGLint *attrib_list)
{
    EnsureEGLLoaded();
    return EGL_GetPlatformDisplayEXT(platform, native_display, attrib_list);
}

EGLSurface EGLAPIENTRY eglCreatePlatformWindowSurfaceEXT(EGLDisplay dpy,
                                                         EGLConfig config,
                                                         void *native_window,
                                                         const EGLint *attrib_list)
{
    EnsureEGLLoaded();
    return EGL_CreatePlatformWindowSurfaceEXT(dpy, config, native_window, attrib_list);
}

EGLSurface EGLAPIENTRY eglCreatePlatformPixmapSurfaceEXT(EGLDisplay dpy,
                                                         EGLConfig config,
                                                         void *native_pixmap,
                                                         const EGLint *attrib_list)
{
    EnsureEGLLoaded();
    return EGL_CreatePlatformPixmapSurfaceEXT(dpy, config, native_pixmap, attrib_list);
}

EGLBoolean EGLAPIENTRY eglQueryDisplayAttribEXT(EGLDisplay dpy, EGLint attribute, EGLAttrib *value)
{
    EnsureEGLLoaded();
    return EGL_QueryDisplayAttribEXT(dpy, attribute, value);
}

EGLBoolean EGLAPIENTRY eglQueryDisplayAttribANGLE(EGLDisplay dpy,
                                                  EGLint attribute,
                                                  EGLAttrib *value)
{
    EnsureEGLLoaded();
    return EGL_QueryDisplayAttribANGLE(dpy, attribute, value);
}

EGLBoolean EGLAPIENTRY eglQueryDeviceAttribEXT(EGLDeviceEXT device,
                                               EGLint attribute,
                                               EGLAttrib *value)
{
    EnsureEGLLoaded();
    return EGL_QueryDeviceAttribEXT(device, attribute, value);
}

const char *EGLAPIENTRY eglQueryDeviceStringEXT(EGLDeviceEXT device, EGLint name)
{
    EnsureEGLLoaded();
    return EGL_QueryDeviceStringEXT(device, name);
}

EGLImageKHR EGLAPIENTRY eglCreateImageKHR(EGLDisplay dpy,
                                          EGLContext ctx,
                                          EGLenum target,
                                          EGLClientBuffer buffer,
                                          const EGLint *attrib_list)
{
    EnsureEGLLoaded();
    return EGL_CreateImageKHR(dpy, ctx, target, buffer, attrib_list);
}

EGLBoolean EGLAPIENTRY eglDestroyImageKHR(EGLDisplay dpy, EGLImageKHR image)
{
    EnsureEGLLoaded();
    return EGL_DestroyImageKHR(dpy, image);
}

EGLDeviceEXT EGLAPIENTRY eglCreateDeviceANGLE(EGLint device_type,
                                              void *native_device,
                                              const EGLAttrib *attrib_list)
{
    EnsureEGLLoaded();
    return EGL_CreateDeviceANGLE(device_type, native_device, attrib_list);
}

EGLBoolean EGLAPIENTRY eglReleaseDeviceANGLE(EGLDeviceEXT device)
{
    EnsureEGLLoaded();
    return EGL_ReleaseDeviceANGLE(device);
}

__eglMustCastToProperFunctionPointerType EGLAPIENTRY eglGetProcAddress(const char *procname)
{
    EnsureEGLLoaded();
    return EGL_GetProcAddress(procname);
}

EGLStreamKHR EGLAPIENTRY eglCreateStreamKHR(EGLDisplay dpy, const EGLint *attrib_list)
{
    EnsureEGLLoaded();
    return EGL_CreateStreamKHR(dpy, attrib_list);
}

EGLBoolean EGLAPIENTRY eglDestroyStreamKHR(EGLDisplay dpy, EGLStreamKHR stream)
{
    EnsureEGLLoaded();
    return EGL_DestroyStreamKHR(dpy, stream);
}

EGLBoolean EGLAPIENTRY eglStreamAttribKHR(EGLDisplay dpy,
                                          EGLStreamKHR stream,
                                          EGLenum attribute,
                                          EGLint value)
{
    EnsureEGLLoaded();
    return EGL_StreamAttribKHR(dpy, stream, attribute, value);
}

EGLBoolean EGLAPIENTRY eglQueryStreamKHR(EGLDisplay dpy,
                                         EGLStreamKHR stream,
                                         EGLenum attribute,
                                         EGLint *value)
{
    EnsureEGLLoaded();
    return EGL_QueryStreamKHR(dpy, stream, attribute, value);
}

EGLBoolean EGLAPIENTRY eglQueryStreamu64KHR(EGLDisplay dpy,
                                            EGLStreamKHR stream,
                                            EGLenum attribute,
                                            EGLuint64KHR *value)
{
    EnsureEGLLoaded();
    return EGL_QueryStreamu64KHR(dpy, stream, attribute, value);
}

EGLBoolean EGLAPIENTRY eglStreamConsumerGLTextureExternalKHR(EGLDisplay dpy, EGLStreamKHR stream)
{
    EnsureEGLLoaded();
    return EGL_StreamConsumerGLTextureExternalKHR(dpy, stream);
}

EGLBoolean EGLAPIENTRY eglStreamConsumerAcquireKHR(EGLDisplay dpy, EGLStreamKHR stream)
{
    EnsureEGLLoaded();
    return EGL_StreamConsumerAcquireKHR(dpy, stream);
}

EGLBoolean EGLAPIENTRY eglStreamConsumerReleaseKHR(EGLDisplay dpy, EGLStreamKHR stream)
{
    EnsureEGLLoaded();
    return EGL_StreamConsumerReleaseKHR(dpy, stream);
}

EGLBoolean EGLAPIENTRY eglStreamConsumerGLTextureExternalAttribsNV(EGLDisplay dpy,
                                                                   EGLStreamKHR stream,
                                                                   const EGLAttrib *attrib_list)
{
    EnsureEGLLoaded();
    return EGL_StreamConsumerGLTextureExternalAttribsNV(dpy, stream, attrib_list);
}

EGLBoolean EGLAPIENTRY eglCreateStreamProducerD3DTextureANGLE(EGLDisplay dpy,
                                                              EGLStreamKHR stream,
                                                              const EGLAttrib *attrib_list)
{
    EnsureEGLLoaded();
    return EGL_CreateStreamProducerD3DTextureANGLE(dpy, stream, attrib_list);
}

EGLBoolean EGLAPIENTRY eglStreamPostD3DTextureANGLE(EGLDisplay dpy,
                                                    EGLStreamKHR stream,
                                                    void *texture,
                                                    const EGLAttrib *attrib_list)
{
    EnsureEGLLoaded();
    return EGL_StreamPostD3DTextureANGLE(dpy, stream, texture, attrib_list);
}

EGLBoolean EGLAPIENTRY eglGetSyncValuesCHROMIUM(EGLDisplay dpy,
                                                EGLSurface surface,
                                                EGLuint64KHR *ust,
                                                EGLuint64KHR *msc,
                                                EGLuint64KHR *sbc)
{
    EnsureEGLLoaded();
    return EGL_GetSyncValuesCHROMIUM(dpy, surface, ust, msc, sbc);
}

EGLBoolean EGLAPIENTRY eglSwapBuffersWithDamageKHR(EGLDisplay dpy,
                                                   EGLSurface surface,
                                                   EGLint *rects,
                                                   EGLint n_rects)
{
    EnsureEGLLoaded();
    return EGL_SwapBuffersWithDamageKHR(dpy, surface, rects, n_rects);
}

EGLBoolean EGLAPIENTRY eglPresentationTimeANDROID(EGLDisplay dpy,
                                                  EGLSurface surface,
                                                  EGLnsecsANDROID time)
{
    EnsureEGLLoaded();
    return EGL_PresentationTimeANDROID(dpy, surface, time);
}

EGLint EGLAPIENTRY eglProgramCacheGetAttribANGLE(EGLDisplay dpy, EGLenum attrib)
{
    EnsureEGLLoaded();
    return EGL_ProgramCacheGetAttribANGLE(dpy, attrib);
}

void EGLAPIENTRY eglProgramCacheQueryANGLE(EGLDisplay dpy,
                                           EGLint index,
                                           void *key,
                                           EGLint *keysize,
                                           void *binary,
                                           EGLint *binarysize)
{
    EGL_ProgramCacheQueryANGLE(dpy, index, key, keysize, binary, binarysize);
}

void EGLAPIENTRY eglProgramCachePopulateANGLE(EGLDisplay dpy,
                                              const void *key,
                                              EGLint keysize,
                                              const void *binary,
                                              EGLint binarysize)
{
    EGL_ProgramCachePopulateANGLE(dpy, key, keysize, binary, binarysize);
}

EGLint EGLAPIENTRY eglProgramCacheResizeANGLE(EGLDisplay dpy, EGLint limit, EGLenum mode)
{
    EnsureEGLLoaded();
    return EGL_ProgramCacheResizeANGLE(dpy, limit, mode);
}

EGLint EGLAPIENTRY eglDebugMessageControlKHR(EGLDEBUGPROCKHR callback, const EGLAttrib *attrib_list)
{
    EnsureEGLLoaded();
    return EGL_DebugMessageControlKHR(callback, attrib_list);
}

EGLBoolean EGLAPIENTRY eglQueryDebugKHR(EGLint attribute, EGLAttrib *value)
{
    EnsureEGLLoaded();
    return EGL_QueryDebugKHR(attribute, value);
}

EGLint EGLAPIENTRY eglLabelObjectKHR(EGLDisplay dpy,
                                     EGLenum objectType,
                                     EGLObjectKHR object,
                                     EGLLabelKHR label)
{
    EnsureEGLLoaded();
    return EGL_LabelObjectKHR(dpy, objectType, object, label);
}

void EGLAPIENTRY eglSetBlobCacheFuncsANDROID(EGLDisplay dpy,
                                             EGLSetBlobFuncANDROID set,
                                             EGLGetBlobFuncANDROID get)
{
    EnsureEGLLoaded();
    return EGL_SetBlobCacheFuncsANDROID(dpy, set, get);
}

EGLBoolean EGLAPIENTRY eglGetCompositorTimingSupportedANDROID(EGLDisplay dpy,
                                                              EGLSurface surface,
                                                              EGLint name)
{
    EnsureEGLLoaded();
    return EGL_GetCompositorTimingSupportedANDROID(dpy, surface, name);
}

EGLBoolean EGLAPIENTRY eglGetCompositorTimingANDROID(EGLDisplay dpy,
                                                     EGLSurface surface,
                                                     EGLint numTimestamps,
                                                     const EGLint *names,
                                                     EGLnsecsANDROID *values)
{
    EnsureEGLLoaded();
    return EGL_GetCompositorTimingANDROID(dpy, surface, numTimestamps, names, values);
}

EGLBoolean EGLAPIENTRY eglGetNextFrameIdANDROID(EGLDisplay dpy,
                                                EGLSurface surface,
                                                EGLuint64KHR *frameId)
{
    EnsureEGLLoaded();
    return EGL_GetNextFrameIdANDROID(dpy, surface, frameId);
}

EGLBoolean EGLAPIENTRY eglGetFrameTimestampSupportedANDROID(EGLDisplay dpy,
                                                            EGLSurface surface,
                                                            EGLint timestamp)
{
    EnsureEGLLoaded();
    return EGL_GetFrameTimestampSupportedANDROID(dpy, surface, timestamp);
}

EGLBoolean EGLAPIENTRY eglGetFrameTimestampsANDROID(EGLDisplay dpy,
                                                    EGLSurface surface,
                                                    EGLuint64KHR frameId,
                                                    EGLint numTimestamps,
                                                    const EGLint *timestamps,
                                                    EGLnsecsANDROID *values)
{
    EnsureEGLLoaded();
    return EGL_GetFrameTimestampsANDROID(dpy, surface, frameId, numTimestamps, timestamps, values);
}

const char *EGLAPIENTRY eglQueryStringiANGLE(EGLDisplay dpy, EGLint name, EGLint index)
{
    EnsureEGLLoaded();
    return EGL_QueryStringiANGLE(dpy, name, index);
}

EGLClientBuffer EGLAPIENTRY eglGetNativeClientBufferANDROID(const struct AHardwareBuffer *buffer)
{
    EnsureEGLLoaded();
    return EGL_GetNativeClientBufferANDROID(buffer);
}

EGLint EGLAPIENTRY eglDupNativeFenceFDANDROID(EGLDisplay dpy, EGLSyncKHR sync)
{
    EnsureEGLLoaded();
    return EGL_DupNativeFenceFDANDROID(dpy, sync);
}

}  // extern "C"
