//
// Copyright 2015 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// validationEGL.h: Validation functions for generic EGL entry point parameters

#ifndef LIBANGLE_VALIDATIONEGL_H_
#define LIBANGLE_VALIDATIONEGL_H_

#include "common/PackedEnums.h"
#include "libANGLE/Error.h"
#include "libANGLE/Texture.h"

#include <EGL/egl.h>
#include <EGL/eglext.h>

namespace gl
{
class Context;
}

namespace egl
{

class AttributeMap;
struct ClientExtensions;
struct Config;
class Device;
class Display;
class Image;
class Stream;
class Surface;
class Sync;
class Thread;
class LabeledObject;

// Object validation
Error ValidateDisplay(const Display *display);
Error ValidateSurface(const Display *display, const Surface *surface);
Error ValidateConfig(const Display *display, const Config *config);
Error ValidateContext(const Display *display, const gl::Context *context);
Error ValidateImage(const Display *display, const Image *image);
Error ValidateDevice(const Device *device);
Error ValidateSync(const Display *display, const Sync *sync);

// Return the requested object only if it is valid (otherwise nullptr)
const Thread *GetThreadIfValid(const Thread *thread);
const Display *GetDisplayIfValid(const Display *display);
const Surface *GetSurfaceIfValid(const Display *display, const Surface *surface);
const Image *GetImageIfValid(const Display *display, const Image *image);
const Stream *GetStreamIfValid(const Display *display, const Stream *stream);
const gl::Context *GetContextIfValid(const Display *display, const gl::Context *context);
const Device *GetDeviceIfValid(const Device *device);
const Sync *GetSyncIfValid(const Display *display, const Sync *sync);
LabeledObject *GetLabeledObjectIfValid(Thread *thread,
                                       const Display *display,
                                       ObjectType objectType,
                                       EGLObjectKHR object);

// Entry point validation
Error ValidateInitialize(const Display *display);

Error ValidateTerminate(const Display *display);

Error ValidateCreateContext(Display *display,
                            Config *configuration,
                            gl::Context *shareContext,
                            const AttributeMap &attributes);

Error ValidateCreateWindowSurface(Display *display,
                                  Config *config,
                                  EGLNativeWindowType window,
                                  const AttributeMap &attributes);

Error ValidateCreatePbufferSurface(Display *display,
                                   Config *config,
                                   const AttributeMap &attributes);
Error ValidateCreatePbufferFromClientBuffer(Display *display,
                                            EGLenum buftype,
                                            EGLClientBuffer buffer,
                                            Config *config,
                                            const AttributeMap &attributes);

Error ValidateMakeCurrent(Display *display, Surface *draw, Surface *read, gl::Context *context);

Error ValidateCreateImage(const Display *display,
                          gl::Context *context,
                          EGLenum target,
                          EGLClientBuffer buffer,
                          const AttributeMap &attributes);
Error ValidateDestroyImage(const Display *display, const Image *image);
Error ValidateCreateImageKHR(const Display *display,
                             gl::Context *context,
                             EGLenum target,
                             EGLClientBuffer buffer,
                             const AttributeMap &attributes);
Error ValidateDestroyImageKHR(const Display *display, const Image *image);

Error ValidateCreateDeviceANGLE(EGLint device_type,
                                void *native_device,
                                const EGLAttrib *attrib_list);
Error ValidateReleaseDeviceANGLE(Device *device);

Error ValidateCreateSyncBase(const Display *display,
                             EGLenum type,
                             const AttributeMap &attribs,
                             const Display *currentDisplay,
                             const gl::Context *currentContext,
                             bool isExt);
Error ValidateGetSyncAttribBase(const Display *display, const Sync *sync, EGLint attribute);

Error ValidateCreateSyncKHR(const Display *display,
                            EGLenum type,
                            const AttributeMap &attribs,
                            const Display *currentDisplay,
                            const gl::Context *currentContext);
Error ValidateCreateSync(const Display *display,
                         EGLenum type,
                         const AttributeMap &attribs,
                         const Display *currentDisplay,
                         const gl::Context *currentContext);
Error ValidateDestroySync(const Display *display, const Sync *sync);
Error ValidateClientWaitSync(const Display *display,
                             const Sync *sync,
                             EGLint flags,
                             EGLTime timeout);
Error ValidateWaitSync(const Display *display,
                       const gl::Context *context,
                       const Sync *sync,
                       EGLint flags);
Error ValidateGetSyncAttribKHR(const Display *display,
                               const Sync *sync,
                               EGLint attribute,
                               EGLint *value);
Error ValidateGetSyncAttrib(const Display *display,
                            const Sync *sync,
                            EGLint attribute,
                            EGLAttrib *value);

Error ValidateCreateStreamKHR(const Display *display, const AttributeMap &attributes);
Error ValidateDestroyStreamKHR(const Display *display, const Stream *stream);
Error ValidateStreamAttribKHR(const Display *display,
                              const Stream *stream,
                              EGLint attribute,
                              EGLint value);
Error ValidateQueryStreamKHR(const Display *display,
                             const Stream *stream,
                             EGLenum attribute,
                             EGLint *value);
Error ValidateQueryStreamu64KHR(const Display *display,
                                const Stream *stream,
                                EGLenum attribute,
                                EGLuint64KHR *value);
Error ValidateStreamConsumerGLTextureExternalKHR(const Display *display,
                                                 gl::Context *context,
                                                 const Stream *stream);
Error ValidateStreamConsumerAcquireKHR(const Display *display,
                                       gl::Context *context,
                                       const Stream *stream);
Error ValidateStreamConsumerReleaseKHR(const Display *display,
                                       gl::Context *context,
                                       const Stream *stream);
Error ValidateStreamConsumerGLTextureExternalAttribsNV(const Display *display,
                                                       gl::Context *context,
                                                       const Stream *stream,
                                                       const AttributeMap &attribs);
Error ValidateCreateStreamProducerD3DTextureANGLE(const Display *display,
                                                  const Stream *stream,
                                                  const AttributeMap &attribs);
Error ValidateStreamPostD3DTextureANGLE(const Display *display,
                                        const Stream *stream,
                                        void *texture,
                                        const AttributeMap &attribs);

Error ValidateGetSyncValuesCHROMIUM(const Display *display,
                                    const Surface *surface,
                                    const EGLuint64KHR *ust,
                                    const EGLuint64KHR *msc,
                                    const EGLuint64KHR *sbc);

Error ValidateDestroySurface(const Display *display,
                             const Surface *surface,
                             const EGLSurface eglSurface);

Error ValidateDestroyContext(const Display *display,
                             const gl::Context *glCtx,
                             const EGLContext eglCtx);

Error ValidateSwapBuffers(Thread *thread, const Display *display, const Surface *surface);

Error ValidateWaitNative(const Display *display, const EGLint engine);

Error ValidateCopyBuffers(Display *display, const Surface *surface);

Error ValidateSwapBuffersWithDamageKHR(const Display *display,
                                       const Surface *surface,
                                       EGLint *rects,
                                       EGLint n_rects);

Error ValidateBindTexImage(const Display *display,
                           const Surface *surface,
                           const EGLSurface eglSurface,
                           const EGLint buffer,
                           const gl::Context *context,
                           gl::Texture **textureObject);

Error ValidateReleaseTexImage(const Display *display,
                              const Surface *surface,
                              const EGLSurface eglSurface,
                              const EGLint buffer);

Error ValidateSwapInterval(const Display *display,
                           const Surface *draw_surface,
                           const gl::Context *context);

Error ValidateBindAPI(const EGLenum api);

Error ValidatePresentationTimeANDROID(const Display *display,
                                      const Surface *surface,
                                      EGLnsecsANDROID time);

Error ValidateSetBlobCacheANDROID(const Display *display,
                                  EGLSetBlobFuncANDROID set,
                                  EGLGetBlobFuncANDROID get);

Error ValidateGetConfigAttrib(const Display *display, const Config *config, EGLint attribute);
Error ValidateChooseConfig(const Display *display,
                           const AttributeMap &attribs,
                           EGLint configSize,
                           EGLint *numConfig);
Error ValidateGetConfigs(const Display *display, EGLint configSize, EGLint *numConfig);

// Other validation
Error ValidateCompatibleSurface(const Display *display,
                                gl::Context *context,
                                const Surface *surface);

Error ValidateGetPlatformDisplay(EGLenum platform,
                                 void *native_display,
                                 const EGLAttrib *attrib_list);
Error ValidateGetPlatformDisplayEXT(EGLenum platform,
                                    void *native_display,
                                    const EGLint *attrib_list);
Error ValidateCreatePlatformWindowSurfaceEXT(const Display *display,
                                             const Config *configuration,
                                             void *nativeWindow,
                                             const AttributeMap &attributes);
Error ValidateCreatePlatformPixmapSurfaceEXT(const Display *display,
                                             const Config *configuration,
                                             void *nativePixmap,
                                             const AttributeMap &attributes);

Error ValidateProgramCacheGetAttribANGLE(const Display *display, EGLenum attrib);

Error ValidateProgramCacheQueryANGLE(const Display *display,
                                     EGLint index,
                                     void *key,
                                     EGLint *keysize,
                                     void *binary,
                                     EGLint *binarysize);

Error ValidateProgramCachePopulateANGLE(const Display *display,
                                        const void *key,
                                        EGLint keysize,
                                        const void *binary,
                                        EGLint binarysize);

Error ValidateProgramCacheResizeANGLE(const Display *display, EGLint limit, EGLenum mode);

Error ValidateSurfaceAttrib(const Display *display,
                            const Surface *surface,
                            EGLint attribute,
                            EGLint value);
Error ValidateQuerySurface(const Display *display,
                           const Surface *surface,
                           EGLint attribute,
                           EGLint *value);
Error ValidateQueryContext(const Display *display,
                           const gl::Context *context,
                           EGLint attribute,
                           EGLint *value);

// EGL_KHR_debug
Error ValidateDebugMessageControlKHR(EGLDEBUGPROCKHR callback, const AttributeMap &attribs);

Error ValidateQueryDebugKHR(EGLint attribute, EGLAttrib *value);

Error ValidateLabelObjectKHR(Thread *thread,
                             const Display *display,
                             ObjectType objectType,
                             EGLObjectKHR object,
                             EGLLabelKHR label);

// ANDROID_get_frame_timestamps
Error ValidateGetCompositorTimingSupportedANDROID(const Display *display,
                                                  const Surface *surface,
                                                  CompositorTiming name);

Error ValidateGetCompositorTimingANDROID(const Display *display,
                                         const Surface *surface,
                                         EGLint numTimestamps,
                                         const EGLint *names,
                                         EGLnsecsANDROID *values);

Error ValidateGetNextFrameIdANDROID(const Display *display,
                                    const Surface *surface,
                                    EGLuint64KHR *frameId);

Error ValidateGetFrameTimestampSupportedANDROID(const Display *display,
                                                const Surface *surface,
                                                Timestamp timestamp);

Error ValidateGetFrameTimestampsANDROID(const Display *display,
                                        const Surface *surface,
                                        EGLuint64KHR frameId,
                                        EGLint numTimestamps,
                                        const EGLint *timestamps,
                                        EGLnsecsANDROID *values);

Error ValidateQueryStringiANGLE(const Display *display, EGLint name, EGLint index);

Error ValidateQueryDisplayAttribEXT(const Display *display, const EGLint attribute);
Error ValidateQueryDisplayAttribANGLE(const Display *display, const EGLint attribute);

// EGL_ANDROID_get_native_client_buffer
Error ValidateGetNativeClientBufferANDROID(const struct AHardwareBuffer *buffer);

// EGL_ANDROID_native_fence_sync
Error ValidateDupNativeFenceFDANDROID(const Display *display, const Sync *sync);

}  // namespace egl

#define ANGLE_EGL_TRY(THREAD, EXPR, FUNCNAME, LABELOBJECT)                               \
    do                                                                                   \
    {                                                                                    \
        auto ANGLE_LOCAL_VAR = (EXPR);                                                   \
        if (ANGLE_LOCAL_VAR.isError())                                                   \
            return THREAD->setError(ANGLE_LOCAL_VAR, GetDebug(), FUNCNAME, LABELOBJECT); \
    } while (0)

#define ANGLE_EGL_TRY_RETURN(THREAD, EXPR, FUNCNAME, LABELOBJECT, RETVAL)         \
    do                                                                            \
    {                                                                             \
        auto ANGLE_LOCAL_VAR = (EXPR);                                            \
        if (ANGLE_LOCAL_VAR.isError())                                            \
        {                                                                         \
            THREAD->setError(ANGLE_LOCAL_VAR, GetDebug(), FUNCNAME, LABELOBJECT); \
            return RETVAL;                                                        \
        }                                                                         \
    } while (0)

#endif  // LIBANGLE_VALIDATIONEGL_H_
