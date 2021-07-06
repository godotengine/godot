//
// Copyright 2014 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// entry_points_ext.cpp : Implements the EGL extension entry points.

#include "libGLESv2/entry_points_egl_ext.h"

#include "common/debug.h"
#include "libANGLE/Context.h"
#include "libANGLE/Device.h"
#include "libANGLE/Display.h"
#include "libANGLE/EGLSync.h"
#include "libANGLE/Stream.h"
#include "libANGLE/Surface.h"
#include "libANGLE/Thread.h"
#include "libANGLE/entry_points_utils.h"
#include "libANGLE/queryutils.h"
#include "libANGLE/validationEGL.h"
#include "libGLESv2/global_state.h"

using namespace egl;

extern "C" {

// EGL_ANGLE_query_surface_pointer
EGLBoolean EGLAPIENTRY EGL_QuerySurfacePointerANGLE(EGLDisplay dpy,
                                                    EGLSurface surface,
                                                    EGLint attribute,
                                                    void **value)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    FUNC_EVENT("EGLDisplay dpy = 0x%016" PRIxPTR ", EGLSurface surface = 0x%016" PRIxPTR
               ", EGLint attribute = %d, void "
               "**value = 0x%016" PRIxPTR,
               (uintptr_t)dpy, (uintptr_t)surface, attribute, (uintptr_t)value);
    Thread *thread = egl::GetCurrentThread();

    egl::Display *display = static_cast<egl::Display *>(dpy);
    Surface *eglSurface   = static_cast<Surface *>(surface);

    Error error = ValidateSurface(display, eglSurface);
    if (error.isError())
    {
        thread->setError(error, GetDebug(), "eglQuerySurfacePointerANGLE",
                         GetSurfaceIfValid(display, eglSurface));
        return EGL_FALSE;
    }

    if (!display->getExtensions().querySurfacePointer)
    {
        thread->setSuccess();
        return EGL_FALSE;
    }

    if (surface == EGL_NO_SURFACE)
    {
        thread->setError(EglBadSurface(), GetDebug(), "eglQuerySurfacePointerANGLE",
                         GetSurfaceIfValid(display, eglSurface));
        return EGL_FALSE;
    }

    // validate the attribute parameter
    switch (attribute)
    {
        case EGL_D3D_TEXTURE_2D_SHARE_HANDLE_ANGLE:
            if (!display->getExtensions().surfaceD3DTexture2DShareHandle)
            {
                thread->setError(EglBadAttribute(), GetDebug(), "eglQuerySurfacePointerANGLE",
                                 GetSurfaceIfValid(display, eglSurface));
                return EGL_FALSE;
            }
            break;
        case EGL_DXGI_KEYED_MUTEX_ANGLE:
            if (!display->getExtensions().keyedMutex)
            {
                thread->setError(EglBadAttribute(), GetDebug(), "eglQuerySurfacePointerANGLE",
                                 GetSurfaceIfValid(display, eglSurface));
                return EGL_FALSE;
            }
            break;
        default:
            thread->setError(EglBadAttribute(), GetDebug(), "eglQuerySurfacePointerANGLE",
                             GetSurfaceIfValid(display, eglSurface));
            return EGL_FALSE;
    }

    error = eglSurface->querySurfacePointerANGLE(attribute, value);
    if (error.isError())
    {
        thread->setError(error, GetDebug(), "eglQuerySurfacePointerANGLE",
                         GetSurfaceIfValid(display, eglSurface));
        return EGL_FALSE;
    }

    thread->setSuccess();
    return EGL_TRUE;
}

// EGL_NV_post_sub_buffer
EGLBoolean EGLAPIENTRY EGL_PostSubBufferNV(EGLDisplay dpy,
                                           EGLSurface surface,
                                           EGLint x,
                                           EGLint y,
                                           EGLint width,
                                           EGLint height)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    FUNC_EVENT("EGLDisplay dpy = 0x%016" PRIxPTR ", EGLSurface surface = 0x%016" PRIxPTR
               ", EGLint x = %d, EGLint y = %d, "
               "EGLint width = %d, EGLint height = %d",
               (uintptr_t)dpy, (uintptr_t)surface, x, y, width, height);
    Thread *thread        = egl::GetCurrentThread();
    egl::Display *display = static_cast<egl::Display *>(dpy);
    Surface *eglSurface   = static_cast<Surface *>(surface);

    if (x < 0 || y < 0 || width < 0 || height < 0)
    {
        thread->setError(EglBadParameter(), GetDebug(), "eglPostSubBufferNV",
                         GetSurfaceIfValid(display, eglSurface));
        return EGL_FALSE;
    }

    Error error = ValidateSurface(display, eglSurface);
    if (error.isError())
    {
        thread->setError(error, GetDebug(), "eglPostSubBufferNV",
                         GetSurfaceIfValid(display, eglSurface));
        return EGL_FALSE;
    }

    if (display->testDeviceLost())
    {
        thread->setError(EglContextLost(), GetDebug(), "eglPostSubBufferNV",
                         GetSurfaceIfValid(display, eglSurface));
        return EGL_FALSE;
    }

    if (surface == EGL_NO_SURFACE)
    {
        thread->setError(EglBadSurface(), GetDebug(), "eglPostSubBufferNV",
                         GetSurfaceIfValid(display, eglSurface));
        return EGL_FALSE;
    }

    if (!display->getExtensions().postSubBuffer)
    {
        // Spec is not clear about how this should be handled.
        thread->setSuccess();
        return EGL_TRUE;
    }

    // TODO(jmadill): Validate Surface is bound to the thread.
    error = eglSurface->postSubBuffer(thread->getContext(), x, y, width, height);
    if (error.isError())
    {
        thread->setError(error, GetDebug(), "eglPostSubBufferNV",
                         GetSurfaceIfValid(display, eglSurface));
        return EGL_FALSE;
    }

    thread->setSuccess();
    return EGL_TRUE;
}

// EGL_EXT_platform_base
EGLDisplay EGLAPIENTRY EGL_GetPlatformDisplayEXT(EGLenum platform,
                                                 void *native_display,
                                                 const EGLint *attrib_list)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    FUNC_EVENT("EGLenum platform = %d, void* native_display = 0x%016" PRIxPTR
               ", const EGLint* attrib_list = "
               "0x%016" PRIxPTR,
               platform, (uintptr_t)native_display, (uintptr_t)attrib_list);
    Thread *thread = egl::GetCurrentThread();

    Error err = ValidateGetPlatformDisplayEXT(platform, native_display, attrib_list);
    thread->setError(err, GetDebug(), "eglGetPlatformDisplayEXT", GetThreadIfValid(thread));
    if (err.isError())
    {
        return EGL_NO_DISPLAY;
    }

    const auto &attribMap = AttributeMap::CreateFromIntArray(attrib_list);
    if (platform == EGL_PLATFORM_ANGLE_ANGLE)
    {
        return egl::Display::GetDisplayFromNativeDisplay(
            gl::bitCast<EGLNativeDisplayType>(native_display), attribMap);
    }
    else if (platform == EGL_PLATFORM_DEVICE_EXT)
    {
        Device *eglDevice = static_cast<Device *>(native_display);
        return egl::Display::GetDisplayFromDevice(eglDevice, attribMap);
    }
    else
    {
        UNREACHABLE();
        return EGL_NO_DISPLAY;
    }
}

EGLSurface EGLAPIENTRY EGL_CreatePlatformWindowSurfaceEXT(EGLDisplay dpy,
                                                          EGLConfig config,
                                                          void *native_window,
                                                          const EGLint *attrib_list)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    FUNC_EVENT("EGLDisplay dpy = 0x%016" PRIxPTR ", EGLConfig config = 0x%016" PRIxPTR
               ", void *native_window = 0x%016" PRIxPTR
               ", "
               "const EGLint *attrib_list = 0x%016" PRIxPTR,
               (uintptr_t)dpy, (uintptr_t)config, (uintptr_t)native_window, (uintptr_t)attrib_list);
    Thread *thread = egl::GetCurrentThread();

    egl::Display *display   = static_cast<egl::Display *>(dpy);
    Config *configuration   = static_cast<Config *>(config);
    AttributeMap attributes = AttributeMap::CreateFromIntArray(attrib_list);

    ANGLE_EGL_TRY_RETURN(
        thread,
        ValidateCreatePlatformWindowSurfaceEXT(display, configuration, native_window, attributes),
        "eglCreatePlatformWindowSurfaceEXT", GetDisplayIfValid(display), EGL_NO_SURFACE);

    thread->setError(EglBadDisplay() << "CreatePlatformWindowSurfaceEXT unimplemented.", GetDebug(),
                     "eglCreatePlatformWindowSurfaceEXT", GetDisplayIfValid(display));
    return EGL_NO_SURFACE;
}

EGLSurface EGLAPIENTRY EGL_CreatePlatformPixmapSurfaceEXT(EGLDisplay dpy,
                                                          EGLConfig config,
                                                          void *native_pixmap,
                                                          const EGLint *attrib_list)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    FUNC_EVENT("EGLDisplay dpy = 0x%016" PRIxPTR ", EGLConfig config = 0x%016" PRIxPTR
               ", void *native_pixmap = 0x%016" PRIxPTR
               ", "
               "const EGLint *attrib_list = 0x%016" PRIxPTR,
               (uintptr_t)dpy, (uintptr_t)config, (uintptr_t)native_pixmap, (uintptr_t)attrib_list);
    Thread *thread = egl::GetCurrentThread();

    egl::Display *display   = static_cast<egl::Display *>(dpy);
    Config *configuration   = static_cast<Config *>(config);
    AttributeMap attributes = AttributeMap::CreateFromIntArray(attrib_list);

    ANGLE_EGL_TRY_RETURN(
        thread,
        ValidateCreatePlatformPixmapSurfaceEXT(display, configuration, native_pixmap, attributes),
        "eglCreatePlatformPixmapSurfaceEXT", GetDisplayIfValid(display), EGL_NO_SURFACE);

    thread->setError(EglBadDisplay() << "CreatePlatformPixmapSurfaceEXT unimplemented.", GetDebug(),
                     "eglCreatePlatformPixmapSurfaceEXT", GetDisplayIfValid(display));
    return EGL_NO_SURFACE;
}

// EGL_EXT_device_query
EGLBoolean EGLAPIENTRY EGL_QueryDeviceAttribEXT(EGLDeviceEXT device,
                                                EGLint attribute,
                                                EGLAttrib *value)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    FUNC_EVENT("EGLDeviceEXT device = 0x%016" PRIxPTR
               ", EGLint attribute = %d, EGLAttrib *value = 0x%016" PRIxPTR,
               (uintptr_t)device, attribute, (uintptr_t)value);
    Thread *thread = egl::GetCurrentThread();

    Device *dev = static_cast<Device *>(device);

    Error error = ValidateDevice(dev);
    if (error.isError())
    {
        thread->setError(error, GetDebug(), "eglQueryDeviceAttribEXT", GetDeviceIfValid(dev));
        return EGL_FALSE;
    }

    // If the device was created by (and is owned by) a display, and that display doesn't support
    // device querying, then this call should fail
    egl::Display *owningDisplay = dev->getOwningDisplay();
    if (owningDisplay != nullptr && !owningDisplay->getExtensions().deviceQuery)
    {
        thread->setError(EglBadAccess() << "Device wasn't created using eglCreateDeviceANGLE, "
                                           "and the egl::Display that created it doesn't support "
                                           "device querying",
                         GetDebug(), "eglQueryDeviceAttribEXT", GetDeviceIfValid(dev));
        return EGL_FALSE;
    }

    // validate the attribute parameter
    switch (attribute)
    {
        case EGL_D3D11_DEVICE_ANGLE:
        case EGL_D3D9_DEVICE_ANGLE:
            if (!dev->getExtensions().deviceD3D || dev->getType() != attribute)
            {
                thread->setError(EglBadAttribute(), GetDebug(), "eglQueryDeviceAttribEXT",
                                 GetDeviceIfValid(dev));
                return EGL_FALSE;
            }
            error = dev->getAttribute(attribute, value);
            break;
        case EGL_MTL_DEVICE_ANGLE:
            if (!dev->getExtensions().deviceMTL)
            {
                thread->setError(EglBadAttribute(), GetDebug(), "eglQueryDeviceAttribEXT",
                                 GetDeviceIfValid(dev));
                return EGL_FALSE;
            }
            error = dev->getAttribute(attribute, value);
            break;
        case EGL_CGL_CONTEXT_ANGLE:
        case EGL_CGL_PIXEL_FORMAT_ANGLE:
            if (!dev->getExtensions().deviceCGL)
            {
                thread->setError(EglBadAttribute(), GetDebug(), "eglQueryDeviceAttribEXT",
                                 GetDeviceIfValid(dev));
                return EGL_FALSE;
            }
            error = dev->getAttribute(attribute, value);
            break;
        default:
            thread->setError(EglBadAttribute(), GetDebug(), "eglQueryDeviceAttribEXT",
                             GetDeviceIfValid(dev));
            return EGL_FALSE;
    }

    if (error.isError())
    {
        thread->setError(error, GetDebug(), "eglQueryDeviceAttribEXT", GetDeviceIfValid(dev));
        return EGL_FALSE;
    }
    thread->setSuccess();
    return EGL_TRUE;
}

// EGL_EXT_device_query
const char *EGLAPIENTRY EGL_QueryDeviceStringEXT(EGLDeviceEXT device, EGLint name)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    FUNC_EVENT("EGLDeviceEXT device = 0x%016" PRIxPTR ", EGLint name = %d", (uintptr_t)device,
               name);
    Thread *thread = egl::GetCurrentThread();

    Device *dev = static_cast<Device *>(device);

    Error error = ValidateDevice(dev);
    if (error.isError())
    {
        thread->setError(error, GetDebug(), "eglQueryDeviceStringEXT", GetDeviceIfValid(dev));
        return EGL_FALSE;
    }

    const char *result;
    switch (name)
    {
        case EGL_EXTENSIONS:
            result = dev->getExtensionString().c_str();
            break;
        default:
            thread->setError(EglBadDevice(), GetDebug(), "eglQueryDeviceStringEXT",
                             GetDeviceIfValid(dev));
            return nullptr;
    }

    thread->setSuccess();
    return result;
}

// EGL_EXT_device_query
EGLBoolean EGLAPIENTRY EGL_QueryDisplayAttribEXT(EGLDisplay dpy, EGLint attribute, EGLAttrib *value)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    FUNC_EVENT("EGLDisplay dpy = 0x%016" PRIxPTR
               ", EGLint attribute = %d, EGLAttrib *value = 0x%016" PRIxPTR,
               (uintptr_t)dpy, attribute, (uintptr_t)value);

    egl::Display *display = static_cast<egl::Display *>(dpy);
    Thread *thread        = egl::GetCurrentThread();

    ANGLE_EGL_TRY_RETURN(thread, ValidateQueryDisplayAttribEXT(display, attribute),
                         "eglQueryDisplayAttribEXT", GetDisplayIfValid(display), EGL_FALSE);

    *value = display->queryAttrib(attribute);
    thread->setSuccess();
    return EGL_TRUE;
}

// EGL_ANGLE_feature_control
EGLBoolean EGLAPIENTRY EGL_QueryDisplayAttribANGLE(EGLDisplay dpy,
                                                   EGLint attribute,
                                                   EGLAttrib *value)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    FUNC_EVENT("EGLDisplay dpy = 0x%016" PRIxPTR
               ", EGLint attribute = %d, EGLAttrib *value = 0x%016" PRIxPTR,
               (uintptr_t)dpy, attribute, (uintptr_t)value);

    egl::Display *display = static_cast<egl::Display *>(dpy);
    Thread *thread        = egl::GetCurrentThread();

    ANGLE_EGL_TRY_RETURN(thread, ValidateQueryDisplayAttribANGLE(display, attribute),
                         "eglQueryDisplayAttribANGLE", GetDisplayIfValid(display), EGL_FALSE);

    *value = display->queryAttrib(attribute);
    thread->setSuccess();
    return EGL_TRUE;
}

ANGLE_EXPORT EGLImageKHR EGLAPIENTRY EGL_CreateImageKHR(EGLDisplay dpy,
                                                        EGLContext ctx,
                                                        EGLenum target,
                                                        EGLClientBuffer buffer,
                                                        const EGLint *attrib_list)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    FUNC_EVENT("EGLDisplay dpy = 0x%016" PRIxPTR
               ", EGLContext ctx = %d"
               ", EGLenum target = 0x%X, "
               "EGLClientBuffer buffer = 0x%016" PRIxPTR
               ", const EGLAttrib *attrib_list = 0x%016" PRIxPTR,
               (uintptr_t)dpy, CID(dpy, ctx), target, (uintptr_t)buffer, (uintptr_t)attrib_list);
    Thread *thread = egl::GetCurrentThread();

    egl::Display *display   = static_cast<egl::Display *>(dpy);
    gl::Context *context    = static_cast<gl::Context *>(ctx);
    AttributeMap attributes = AttributeMap::CreateFromIntArray(attrib_list);

    Error error = ValidateCreateImageKHR(display, context, target, buffer, attributes);
    if (error.isError())
    {
        thread->setError(error, GetDebug(), "eglCreateImageKHR", GetDisplayIfValid(display));
        return EGL_NO_IMAGE;
    }

    Image *image = nullptr;
    error        = display->createImage(context, target, buffer, attributes, &image);
    if (error.isError())
    {
        thread->setError(error, GetDebug(), "eglCreateImageKHR", GetDisplayIfValid(display));
        return EGL_NO_IMAGE;
    }

    thread->setSuccess();
    return static_cast<EGLImage>(image);
}

ANGLE_EXPORT EGLBoolean EGLAPIENTRY EGL_DestroyImageKHR(EGLDisplay dpy, EGLImageKHR image)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    FUNC_EVENT("EGLDisplay dpy = 0x%016" PRIxPTR ", EGLImage image = 0x%016" PRIxPTR,
               (uintptr_t)dpy, (uintptr_t)image);
    Thread *thread = egl::GetCurrentThread();

    egl::Display *display = static_cast<egl::Display *>(dpy);
    Image *img            = static_cast<Image *>(image);

    Error error = ValidateDestroyImageKHR(display, img);
    if (error.isError())
    {
        thread->setError(error, GetDebug(), "eglDestroyImageKHR", GetImageIfValid(display, img));
        return EGL_FALSE;
    }

    display->destroyImage(img);

    thread->setSuccess();
    return EGL_TRUE;
}

ANGLE_EXPORT EGLDeviceEXT EGLAPIENTRY EGL_CreateDeviceANGLE(EGLint device_type,
                                                            void *native_device,
                                                            const EGLAttrib *attrib_list)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    FUNC_EVENT("EGLint device_type = %d, void* native_device = 0x%016" PRIxPTR
               ", const EGLAttrib* attrib_list = "
               "0x%016" PRIxPTR,
               device_type, (uintptr_t)native_device, (uintptr_t)attrib_list);
    Thread *thread = egl::GetCurrentThread();

    Error error = ValidateCreateDeviceANGLE(device_type, native_device, attrib_list);
    if (error.isError())
    {
        thread->setError(error, GetDebug(), "eglCreateDeviceANGLE", GetThreadIfValid(thread));
        return EGL_NO_DEVICE_EXT;
    }

    Device *device = nullptr;
    error          = Device::CreateDevice(device_type, native_device, &device);
    if (error.isError())
    {
        ASSERT(device == nullptr);
        thread->setError(error, GetDebug(), "eglCreateDeviceANGLE", GetThreadIfValid(thread));
        return EGL_NO_DEVICE_EXT;
    }

    thread->setSuccess();
    return device;
}

ANGLE_EXPORT EGLBoolean EGLAPIENTRY EGL_ReleaseDeviceANGLE(EGLDeviceEXT device)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    FUNC_EVENT("EGLDeviceEXT device = 0x%016" PRIxPTR, (uintptr_t)device);
    Thread *thread = egl::GetCurrentThread();

    Device *dev = static_cast<Device *>(device);

    Error error = ValidateReleaseDeviceANGLE(dev);
    if (error.isError())
    {
        thread->setError(error, GetDebug(), "eglReleaseDeviceANGLE", GetDeviceIfValid(dev));
        return EGL_FALSE;
    }

    SafeDelete(dev);

    thread->setSuccess();
    return EGL_TRUE;
}

// EGL_KHR_stream
EGLStreamKHR EGLAPIENTRY EGL_CreateStreamKHR(EGLDisplay dpy, const EGLint *attrib_list)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    FUNC_EVENT("EGLDisplay dpy = 0x%016" PRIxPTR ", const EGLAttrib* attrib_list = 0x%016" PRIxPTR,
               (uintptr_t)dpy, (uintptr_t)attrib_list);
    Thread *thread = egl::GetCurrentThread();

    egl::Display *display   = static_cast<egl::Display *>(dpy);
    AttributeMap attributes = AttributeMap::CreateFromIntArray(attrib_list);

    Error error = ValidateCreateStreamKHR(display, attributes);
    if (error.isError())
    {
        thread->setError(error, GetDebug(), "eglCreateStreamKHR", GetDisplayIfValid(display));
        return EGL_NO_STREAM_KHR;
    }

    Stream *stream;
    error = display->createStream(attributes, &stream);
    if (error.isError())
    {
        thread->setError(error, GetDebug(), "eglCreateStreamKHR", GetDisplayIfValid(display));
        return EGL_NO_STREAM_KHR;
    }

    thread->setSuccess();
    return static_cast<EGLStreamKHR>(stream);
}

EGLBoolean EGLAPIENTRY EGL_DestroyStreamKHR(EGLDisplay dpy, EGLStreamKHR stream)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    FUNC_EVENT("EGLDisplay dpy = 0x%016" PRIxPTR ", EGLStreamKHR = 0x%016" PRIxPTR, (uintptr_t)dpy,
               (uintptr_t)stream);
    Thread *thread = egl::GetCurrentThread();

    egl::Display *display = static_cast<egl::Display *>(dpy);
    Stream *streamObject  = static_cast<Stream *>(stream);

    Error error = ValidateDestroyStreamKHR(display, streamObject);
    if (error.isError())
    {
        thread->setError(error, GetDebug(), "eglDestroyStreamKHR",
                         GetStreamIfValid(display, streamObject));
        return EGL_FALSE;
    }

    display->destroyStream(streamObject);

    thread->setSuccess();
    return EGL_TRUE;
}

EGLBoolean EGLAPIENTRY EGL_StreamAttribKHR(EGLDisplay dpy,
                                           EGLStreamKHR stream,
                                           EGLenum attribute,
                                           EGLint value)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    FUNC_EVENT("EGLDisplay dpy = 0x%016" PRIxPTR ", EGLStreamKHR stream = 0x%016" PRIxPTR
               ", EGLenum attribute = 0x%X, "
               "EGLint value = 0x%X",
               (uintptr_t)dpy, (uintptr_t)stream, attribute, value);
    Thread *thread = egl::GetCurrentThread();

    egl::Display *display = static_cast<egl::Display *>(dpy);
    Stream *streamObject  = static_cast<Stream *>(stream);

    Error error = ValidateStreamAttribKHR(display, streamObject, attribute, value);
    if (error.isError())
    {
        thread->setError(error, GetDebug(), "eglStreamAttribKHR",
                         GetStreamIfValid(display, streamObject));
        return EGL_FALSE;
    }

    switch (attribute)
    {
        case EGL_CONSUMER_LATENCY_USEC_KHR:
            streamObject->setConsumerLatency(value);
            break;
        case EGL_CONSUMER_ACQUIRE_TIMEOUT_USEC_KHR:
            streamObject->setConsumerAcquireTimeout(value);
            break;
        default:
            UNREACHABLE();
    }

    thread->setSuccess();
    return EGL_TRUE;
}

EGLBoolean EGLAPIENTRY EGL_QueryStreamKHR(EGLDisplay dpy,
                                          EGLStreamKHR stream,
                                          EGLenum attribute,
                                          EGLint *value)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    FUNC_EVENT("EGLDisplay dpy = 0x%016" PRIxPTR ", EGLStreamKHR stream = 0x%016" PRIxPTR
               ", EGLenum attribute = 0x%X, "
               "EGLint value = 0x%016" PRIxPTR,
               (uintptr_t)dpy, (uintptr_t)stream, attribute, (uintptr_t)value);
    Thread *thread = egl::GetCurrentThread();

    egl::Display *display = static_cast<egl::Display *>(dpy);
    Stream *streamObject  = static_cast<Stream *>(stream);

    Error error = ValidateQueryStreamKHR(display, streamObject, attribute, value);
    if (error.isError())
    {
        thread->setError(error, GetDebug(), "eglQueryStreamKHR",
                         GetStreamIfValid(display, streamObject));
        return EGL_FALSE;
    }

    switch (attribute)
    {
        case EGL_STREAM_STATE_KHR:
            *value = streamObject->getState();
            break;
        case EGL_CONSUMER_LATENCY_USEC_KHR:
            *value = streamObject->getConsumerLatency();
            break;
        case EGL_CONSUMER_ACQUIRE_TIMEOUT_USEC_KHR:
            *value = streamObject->getConsumerAcquireTimeout();
            break;
        default:
            UNREACHABLE();
    }

    thread->setSuccess();
    return EGL_TRUE;
}

EGLBoolean EGLAPIENTRY EGL_QueryStreamu64KHR(EGLDisplay dpy,
                                             EGLStreamKHR stream,
                                             EGLenum attribute,
                                             EGLuint64KHR *value)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    FUNC_EVENT("EGLDisplay dpy = 0x%016" PRIxPTR ", EGLStreamKHR stream = 0x%016" PRIxPTR
               ", EGLenum attribute = 0x%X, "
               "EGLuint64KHR value = 0x%016" PRIxPTR,
               (uintptr_t)dpy, (uintptr_t)stream, attribute, (uintptr_t)value);
    Thread *thread = egl::GetCurrentThread();

    egl::Display *display = static_cast<egl::Display *>(dpy);
    Stream *streamObject  = static_cast<Stream *>(stream);

    Error error = ValidateQueryStreamu64KHR(display, streamObject, attribute, value);
    if (error.isError())
    {
        thread->setError(error, GetDebug(), "eglQueryStreamu64KHR",
                         GetStreamIfValid(display, streamObject));
        return EGL_FALSE;
    }

    switch (attribute)
    {
        case EGL_PRODUCER_FRAME_KHR:
            *value = streamObject->getProducerFrame();
            break;
        case EGL_CONSUMER_FRAME_KHR:
            *value = streamObject->getConsumerFrame();
            break;
        default:
            UNREACHABLE();
    }

    thread->setSuccess();
    return EGL_TRUE;
}

EGLBoolean EGLAPIENTRY EGL_StreamConsumerGLTextureExternalKHR(EGLDisplay dpy, EGLStreamKHR stream)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    FUNC_EVENT("EGLDisplay dpy = 0x%016" PRIxPTR ", EGLStreamKHR = 0x%016" PRIxPTR, (uintptr_t)dpy,
               (uintptr_t)stream);
    Thread *thread = egl::GetCurrentThread();

    egl::Display *display = static_cast<egl::Display *>(dpy);
    Stream *streamObject  = static_cast<Stream *>(stream);
    gl::Context *context  = gl::GetValidGlobalContext();

    Error error = ValidateStreamConsumerGLTextureExternalKHR(display, context, streamObject);
    if (error.isError())
    {
        thread->setError(error, GetDebug(), "eglStreamConsumerGLTextureExternalKHR",
                         GetStreamIfValid(display, streamObject));
        return EGL_FALSE;
    }

    error = streamObject->createConsumerGLTextureExternal(AttributeMap(), context);
    if (error.isError())
    {
        thread->setError(error, GetDebug(), "eglStreamConsumerGLTextureExternalKHR",
                         GetStreamIfValid(display, streamObject));
        return EGL_FALSE;
    }

    thread->setSuccess();
    return EGL_TRUE;
}

EGLBoolean EGLAPIENTRY EGL_StreamConsumerAcquireKHR(EGLDisplay dpy, EGLStreamKHR stream)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    FUNC_EVENT("EGLDisplay dpy = 0x%016" PRIxPTR ", EGLStreamKHR = 0x%016" PRIxPTR, (uintptr_t)dpy,
               (uintptr_t)stream);
    Thread *thread = egl::GetCurrentThread();

    egl::Display *display = static_cast<egl::Display *>(dpy);
    Stream *streamObject  = static_cast<Stream *>(stream);
    gl::Context *context  = gl::GetValidGlobalContext();

    Error error = ValidateStreamConsumerAcquireKHR(display, context, streamObject);
    if (error.isError())
    {
        thread->setError(error, GetDebug(), "eglStreamConsumerAcquireKHR",
                         GetStreamIfValid(display, streamObject));
        return EGL_FALSE;
    }

    error = streamObject->consumerAcquire(context);
    if (error.isError())
    {
        thread->setError(error, GetDebug(), "eglStreamConsumerAcquireKHR",
                         GetStreamIfValid(display, streamObject));
        return EGL_FALSE;
    }

    thread->setSuccess();
    return EGL_TRUE;
}

EGLBoolean EGLAPIENTRY EGL_StreamConsumerReleaseKHR(EGLDisplay dpy, EGLStreamKHR stream)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    FUNC_EVENT("EGLDisplay dpy = 0x%016" PRIxPTR ", EGLStreamKHR = 0x%016" PRIxPTR, (uintptr_t)dpy,
               (uintptr_t)stream);
    Thread *thread = egl::GetCurrentThread();

    egl::Display *display = static_cast<egl::Display *>(dpy);
    Stream *streamObject  = static_cast<Stream *>(stream);
    gl::Context *context  = gl::GetValidGlobalContext();

    Error error = ValidateStreamConsumerReleaseKHR(display, context, streamObject);
    if (error.isError())
    {
        thread->setError(error, GetDebug(), "eglSStreamConsumerReleaseKHR",
                         GetStreamIfValid(display, streamObject));
        return EGL_FALSE;
    }

    error = streamObject->consumerRelease(context);
    if (error.isError())
    {
        thread->setError(error, GetDebug(), "eglStreamConsumerReleaseKHR",
                         GetStreamIfValid(display, streamObject));
        return EGL_FALSE;
    }

    thread->setSuccess();
    return EGL_TRUE;
}

EGLBoolean EGLAPIENTRY EGL_StreamConsumerGLTextureExternalAttribsNV(EGLDisplay dpy,
                                                                    EGLStreamKHR stream,
                                                                    const EGLAttrib *attrib_list)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    FUNC_EVENT("EGLDisplay dpy = 0x%016" PRIxPTR ", EGLStreamKHR stream = 0x%016" PRIxPTR
               ", EGLAttrib attrib_list = 0x%016" PRIxPTR "",
               (uintptr_t)dpy, (uintptr_t)stream, (uintptr_t)attrib_list);
    Thread *thread = egl::GetCurrentThread();

    egl::Display *display   = static_cast<egl::Display *>(dpy);
    Stream *streamObject    = static_cast<Stream *>(stream);
    gl::Context *context    = gl::GetValidGlobalContext();
    AttributeMap attributes = AttributeMap::CreateFromAttribArray(attrib_list);

    Error error = ValidateStreamConsumerGLTextureExternalAttribsNV(display, context, streamObject,
                                                                   attributes);
    if (error.isError())
    {
        thread->setError(error, GetDebug(), "eglStreamConsumerGLTextureExternalAttribsNV",
                         GetStreamIfValid(display, streamObject));
        return EGL_FALSE;
    }

    error = streamObject->createConsumerGLTextureExternal(attributes, context);
    if (error.isError())
    {
        thread->setError(error, GetDebug(), "eglStreamConsumerGLTextureExternalAttribsNV",
                         GetStreamIfValid(display, streamObject));
        return EGL_FALSE;
    }

    thread->setSuccess();
    return EGL_TRUE;
}

EGLBoolean EGLAPIENTRY EGL_CreateStreamProducerD3DTextureANGLE(EGLDisplay dpy,
                                                               EGLStreamKHR stream,
                                                               const EGLAttrib *attrib_list)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    FUNC_EVENT("EGLDisplay dpy = 0x%016" PRIxPTR ", EGLStreamKHR stream = 0x%016" PRIxPTR
               ", EGLAttrib attrib_list = 0x%016" PRIxPTR "",
               (uintptr_t)dpy, (uintptr_t)stream, (uintptr_t)attrib_list);
    Thread *thread = egl::GetCurrentThread();

    egl::Display *display   = static_cast<egl::Display *>(dpy);
    Stream *streamObject    = static_cast<Stream *>(stream);
    AttributeMap attributes = AttributeMap::CreateFromAttribArray(attrib_list);

    Error error = ValidateCreateStreamProducerD3DTextureANGLE(display, streamObject, attributes);
    if (error.isError())
    {
        thread->setError(error, GetDebug(), "eglCreateStreamProducerD3DTextureANGLE",
                         GetStreamIfValid(display, streamObject));
        return EGL_FALSE;
    }

    error = streamObject->createProducerD3D11Texture(attributes);
    if (error.isError())
    {
        thread->setError(error, GetDebug(), "eglCreateStreamProducerD3DTextureANGLE",
                         GetStreamIfValid(display, streamObject));
        return EGL_FALSE;
    }

    thread->setSuccess();
    return EGL_TRUE;
}

EGLBoolean EGLAPIENTRY EGL_StreamPostD3DTextureANGLE(EGLDisplay dpy,
                                                     EGLStreamKHR stream,
                                                     void *texture,
                                                     const EGLAttrib *attrib_list)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    FUNC_EVENT("EGLDisplay dpy = 0x%016" PRIxPTR ", EGLStreamKHR stream = 0x%016" PRIxPTR
               ", void* texture = 0x%016" PRIxPTR
               ", "
               "EGLAttrib attrib_list = 0x%016" PRIxPTR "",
               (uintptr_t)dpy, (uintptr_t)stream, (uintptr_t)texture, (uintptr_t)attrib_list);
    Thread *thread = egl::GetCurrentThread();

    egl::Display *display   = static_cast<egl::Display *>(dpy);
    Stream *streamObject    = static_cast<Stream *>(stream);
    AttributeMap attributes = AttributeMap::CreateFromAttribArray(attrib_list);

    Error error = ValidateStreamPostD3DTextureANGLE(display, streamObject, texture, attributes);
    if (error.isError())
    {
        thread->setError(error, GetDebug(), "eglStreamPostD3DTextureANGLE",
                         GetStreamIfValid(display, streamObject));
        return EGL_FALSE;
    }

    error = streamObject->postD3D11Texture(texture, attributes);
    if (error.isError())
    {
        thread->setError(error, GetDebug(), "eglStreamPostD3DTextureANGLE",
                         GetStreamIfValid(display, streamObject));
        return EGL_FALSE;
    }

    thread->setSuccess();
    return EGL_TRUE;
}

// EGL_KHR_fence_sync
ANGLE_EXPORT EGLSync EGLAPIENTRY EGL_CreateSyncKHR(EGLDisplay dpy,
                                                   EGLenum type,
                                                   const EGLint *attrib_list)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    FUNC_EVENT("EGLDisplay dpy = 0x%016" PRIxPTR
               ", EGLenum type = 0x%X, const EGLint* attrib_list = 0x%016" PRIxPTR,
               (uintptr_t)dpy, type, (uintptr_t)attrib_list);

    Thread *thread          = egl::GetCurrentThread();
    egl::Display *display   = static_cast<egl::Display *>(dpy);
    AttributeMap attributes = AttributeMap::CreateFromIntArray(attrib_list);

    gl::Context *currentContext  = thread->getContext();
    egl::Display *currentDisplay = currentContext ? currentContext->getDisplay() : nullptr;

    ANGLE_EGL_TRY_RETURN(
        thread, ValidateCreateSyncKHR(display, type, attributes, currentDisplay, currentContext),
        "eglCreateSync", GetDisplayIfValid(display), EGL_NO_SYNC);

    egl::Sync *syncObject = nullptr;
    ANGLE_EGL_TRY_RETURN(thread, display->createSync(currentContext, type, attributes, &syncObject),
                         "eglCreateSync", GetDisplayIfValid(display), EGL_NO_SYNC);

    thread->setSuccess();
    return static_cast<EGLSync>(syncObject);
}

ANGLE_EXPORT EGLBoolean EGLAPIENTRY EGL_DestroySyncKHR(EGLDisplay dpy, EGLSync sync)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    FUNC_EVENT("EGLDisplay dpy = 0x%016" PRIxPTR ", EGLSync sync = 0x%016" PRIxPTR, (uintptr_t)dpy,
               (uintptr_t)sync);

    Thread *thread        = egl::GetCurrentThread();
    egl::Display *display = static_cast<egl::Display *>(dpy);
    egl::Sync *syncObject = static_cast<Sync *>(sync);

    ANGLE_EGL_TRY_RETURN(thread, ValidateDestroySync(display, syncObject), "eglDestroySync",
                         GetDisplayIfValid(display), EGL_FALSE);

    display->destroySync(syncObject);

    thread->setSuccess();
    return EGL_TRUE;
}

ANGLE_EXPORT EGLint EGLAPIENTRY EGL_ClientWaitSyncKHR(EGLDisplay dpy,
                                                      EGLSync sync,
                                                      EGLint flags,
                                                      EGLTime timeout)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    FUNC_EVENT("EGLDisplay dpy = 0x%016" PRIxPTR ", EGLSync sync = 0x%016" PRIxPTR
               ", EGLint flags = 0x%X, EGLTime timeout = "
               "%llu",
               (uintptr_t)dpy, (uintptr_t)sync, flags, static_cast<unsigned long long>(timeout));

    Thread *thread        = egl::GetCurrentThread();
    egl::Display *display = static_cast<egl::Display *>(dpy);
    egl::Sync *syncObject = static_cast<Sync *>(sync);

    ANGLE_EGL_TRY_RETURN(thread, ValidateClientWaitSync(display, syncObject, flags, timeout),
                         "eglClientWaitSync", GetSyncIfValid(display, syncObject), EGL_FALSE);

    gl::Context *currentContext = thread->getContext();
    EGLint syncStatus           = EGL_FALSE;
    ANGLE_EGL_TRY_RETURN(
        thread, syncObject->clientWait(display, currentContext, flags, timeout, &syncStatus),
        "eglClientWaitSync", GetSyncIfValid(display, syncObject), EGL_FALSE);

    thread->setSuccess();
    return syncStatus;
}

ANGLE_EXPORT EGLBoolean EGLAPIENTRY EGL_GetSyncAttribKHR(EGLDisplay dpy,
                                                         EGLSync sync,
                                                         EGLint attribute,
                                                         EGLint *value)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    FUNC_EVENT("EGLDisplay dpy = 0x%016" PRIxPTR ", EGLSync sync = 0x%016" PRIxPTR
               ", EGLint attribute = 0x%X, EGLAttrib "
               "*value = 0x%016" PRIxPTR,
               (uintptr_t)dpy, (uintptr_t)sync, attribute, (uintptr_t)value);

    Thread *thread        = egl::GetCurrentThread();
    egl::Display *display = static_cast<egl::Display *>(dpy);
    egl::Sync *syncObject = static_cast<Sync *>(sync);

    ANGLE_EGL_TRY_RETURN(thread, ValidateGetSyncAttribKHR(display, syncObject, attribute, value),
                         "eglGetSyncAttrib", GetSyncIfValid(display, syncObject), EGL_FALSE);

    ANGLE_EGL_TRY_RETURN(thread, GetSyncAttrib(display, syncObject, attribute, value),
                         "eglGetSyncAttrib", GetSyncIfValid(display, syncObject), EGL_FALSE);

    thread->setSuccess();
    return EGL_TRUE;
}

// EGL_KHR_wait_sync
ANGLE_EXPORT EGLBoolean EGLAPIENTRY EGL_WaitSyncKHR(EGLDisplay dpy, EGLSync sync, EGLint flags)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    FUNC_EVENT("EGLDisplay dpy = 0x%016" PRIxPTR "p, EGLSync sync = 0x%016" PRIxPTR
               ", EGLint flags = 0x%X",
               (uintptr_t)dpy, (uintptr_t)sync, flags);

    Thread *thread        = egl::GetCurrentThread();
    egl::Display *display = static_cast<egl::Display *>(dpy);
    gl::Context *context  = thread->getContext();
    egl::Sync *syncObject = static_cast<Sync *>(sync);

    ANGLE_EGL_TRY_RETURN(thread, ValidateWaitSync(display, context, syncObject, flags),
                         "eglWaitSync", GetSyncIfValid(display, syncObject), EGL_FALSE);

    gl::Context *currentContext = thread->getContext();
    ANGLE_EGL_TRY_RETURN(thread, syncObject->serverWait(display, currentContext, flags),
                         "eglWaitSync", GetSyncIfValid(display, syncObject), EGL_FALSE);

    thread->setSuccess();
    return EGL_TRUE;
}

EGLBoolean EGLAPIENTRY EGL_GetSyncValuesCHROMIUM(EGLDisplay dpy,
                                                 EGLSurface surface,
                                                 EGLuint64KHR *ust,
                                                 EGLuint64KHR *msc,
                                                 EGLuint64KHR *sbc)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    FUNC_EVENT("EGLDisplay dpy = 0x%016" PRIxPTR ", EGLSurface surface = 0x%016" PRIxPTR
               ", EGLuint64KHR* ust = 0x%016" PRIxPTR
               ", "
               "EGLuint64KHR* msc = 0x%016" PRIxPTR ", EGLuint64KHR* sbc = 0x%016" PRIxPTR "",
               (uintptr_t)dpy, (uintptr_t)surface, (uintptr_t)ust, (uintptr_t)msc, (uintptr_t)sbc);
    Thread *thread = egl::GetCurrentThread();

    egl::Display *display = static_cast<egl::Display *>(dpy);
    Surface *eglSurface   = static_cast<Surface *>(surface);

    Error error = ValidateGetSyncValuesCHROMIUM(display, eglSurface, ust, msc, sbc);
    if (error.isError())
    {
        thread->setError(error, GetDebug(), "eglGetSyncValuesCHROMIUM",
                         GetSurfaceIfValid(display, eglSurface));
        return EGL_FALSE;
    }

    error = eglSurface->getSyncValues(ust, msc, sbc);
    if (error.isError())
    {
        thread->setError(error, GetDebug(), "eglGetSyncValuesCHROMIUM",
                         GetSurfaceIfValid(display, eglSurface));
        return EGL_FALSE;
    }

    thread->setSuccess();
    return EGL_TRUE;
}

EGLBoolean EGLAPIENTRY EGL_SwapBuffersWithDamageKHR(EGLDisplay dpy,
                                                    EGLSurface surface,
                                                    EGLint *rects,
                                                    EGLint n_rects)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    FUNC_EVENT("EGLDisplay dpy = 0x%016" PRIxPTR ", EGLSurface surface = 0x%016" PRIxPTR
               ", EGLint *rects = 0x%016" PRIxPTR
               ", EGLint "
               "n_rects = %d",
               (uintptr_t)dpy, (uintptr_t)surface, (uintptr_t)rects, n_rects);
    Thread *thread = egl::GetCurrentThread();

    egl::Display *display = static_cast<egl::Display *>(dpy);
    Surface *eglSurface   = static_cast<Surface *>(surface);

    Error error = ValidateSwapBuffersWithDamageKHR(display, eglSurface, rects, n_rects);
    if (error.isError())
    {
        thread->setError(error, GetDebug(), "eglSwapBuffersWithDamageEXT",
                         GetSurfaceIfValid(display, eglSurface));
        return EGL_FALSE;
    }

    error = eglSurface->swapWithDamage(thread->getContext(), rects, n_rects);
    if (error.isError())
    {
        thread->setError(error, GetDebug(), "eglSwapBuffersWithDamageEXT",
                         GetSurfaceIfValid(display, eglSurface));
        return EGL_FALSE;
    }

    thread->setSuccess();
    return EGL_TRUE;
}

EGLBoolean EGLAPIENTRY EGL_PresentationTimeANDROID(EGLDisplay dpy,
                                                   EGLSurface surface,
                                                   EGLnsecsANDROID time)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    FUNC_EVENT("EGLDisplay dpy = 0x%016" PRIxPTR ", EGLSurface surface = 0x%016" PRIxPTR
               ", EGLnsecsANDROID time = %llu",
               (uintptr_t)dpy, (uintptr_t)surface, static_cast<unsigned long long>(time));
    Thread *thread = egl::GetCurrentThread();

    egl::Display *display = static_cast<egl::Display *>(dpy);
    Surface *eglSurface   = static_cast<Surface *>(surface);

    ANGLE_EGL_TRY_RETURN(thread, ValidatePresentationTimeANDROID(display, eglSurface, time),
                         "eglPresentationTimeANDROID", GetSurfaceIfValid(display, eglSurface),
                         EGL_FALSE);
    ANGLE_EGL_TRY_RETURN(thread, eglSurface->setPresentationTime(time),
                         "eglPresentationTimeANDROID", GetSurfaceIfValid(display, eglSurface),
                         EGL_FALSE);

    return EGL_TRUE;
}

ANGLE_EXPORT void EGLAPIENTRY EGL_SetBlobCacheFuncsANDROID(EGLDisplay dpy,
                                                           EGLSetBlobFuncANDROID set,
                                                           EGLGetBlobFuncANDROID get)
{
    FUNC_EVENT("EGLDisplay dpy = 0x%016" PRIxPTR ", EGLSetBlobFuncANDROID set = 0x%016" PRIxPTR
               ", EGLGetBlobFuncANDROID get "
               "= 0x%016" PRIxPTR,
               (uintptr_t)dpy, (uintptr_t)set, (uintptr_t)get);
    Thread *thread = egl::GetCurrentThread();

    egl::Display *display = static_cast<egl::Display *>(dpy);

    ANGLE_EGL_TRY(thread, ValidateSetBlobCacheANDROID(display, set, get),
                  "eglSetBlobCacheFuncsANDROID", GetDisplayIfValid(display));

    thread->setSuccess();
    display->setBlobCacheFuncs(set, get);
}

EGLint EGLAPIENTRY EGL_ProgramCacheGetAttribANGLE(EGLDisplay dpy, EGLenum attrib)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    FUNC_EVENT("EGLDisplay dpy = 0x%016" PRIxPTR ", EGLenum attrib = 0x%X", (uintptr_t)dpy, attrib);

    egl::Display *display = static_cast<egl::Display *>(dpy);
    Thread *thread        = egl::GetCurrentThread();

    ANGLE_EGL_TRY_RETURN(thread, ValidateProgramCacheGetAttribANGLE(display, attrib),
                         "eglProgramCacheGetAttribANGLE", GetDisplayIfValid(display), 0);

    thread->setSuccess();
    return display->programCacheGetAttrib(attrib);
}

void EGLAPIENTRY EGL_ProgramCacheQueryANGLE(EGLDisplay dpy,
                                            EGLint index,
                                            void *key,
                                            EGLint *keysize,
                                            void *binary,
                                            EGLint *binarysize)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    FUNC_EVENT("EGLDisplay dpy = 0x%016" PRIxPTR ", EGLint index = %d, void *key = 0x%016" PRIxPTR
               ", EGLint *keysize = "
               "0x%016" PRIxPTR ", void *binary = 0x%016" PRIxPTR ", EGLint *size = 0x%016" PRIxPTR,
               (uintptr_t)dpy, index, (uintptr_t)key, (uintptr_t)keysize, (uintptr_t)binary,
               (uintptr_t)binarysize);

    egl::Display *display = static_cast<egl::Display *>(dpy);
    Thread *thread        = egl::GetCurrentThread();

    ANGLE_EGL_TRY(thread,
                  ValidateProgramCacheQueryANGLE(display, index, key, keysize, binary, binarysize),
                  "eglProgramCacheQueryANGLE", GetDisplayIfValid(display));

    ANGLE_EGL_TRY(thread, display->programCacheQuery(index, key, keysize, binary, binarysize),
                  "eglProgramCacheQueryANGLE", GetDisplayIfValid(display));

    thread->setSuccess();
}

void EGLAPIENTRY EGL_ProgramCachePopulateANGLE(EGLDisplay dpy,
                                               const void *key,
                                               EGLint keysize,
                                               const void *binary,
                                               EGLint binarysize)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    FUNC_EVENT("EGLDisplay dpy = 0x%016" PRIxPTR ", void *key = 0x%016" PRIxPTR
               ", EGLint keysize = %d, void *binary = "
               "0x%016" PRIxPTR ", EGLint size = %d",
               (uintptr_t)dpy, (uintptr_t)key, keysize, (uintptr_t)binary, binarysize);

    egl::Display *display = static_cast<egl::Display *>(dpy);
    Thread *thread        = egl::GetCurrentThread();

    ANGLE_EGL_TRY(thread,
                  ValidateProgramCachePopulateANGLE(display, key, keysize, binary, binarysize),
                  "eglProgramCachePopulateANGLE", GetDisplayIfValid(display));

    ANGLE_EGL_TRY(thread, display->programCachePopulate(key, keysize, binary, binarysize),
                  "eglProgramCachePopulateANGLE", GetDisplayIfValid(display));

    thread->setSuccess();
}

EGLint EGLAPIENTRY EGL_ProgramCacheResizeANGLE(EGLDisplay dpy, EGLint limit, EGLenum mode)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    FUNC_EVENT("EGLDisplay dpy = 0x%016" PRIxPTR ", EGLint limit = %d, EGLenum mode = 0x%X",
               (uintptr_t)dpy, limit, mode);

    egl::Display *display = static_cast<egl::Display *>(dpy);
    Thread *thread        = egl::GetCurrentThread();

    ANGLE_EGL_TRY_RETURN(thread, ValidateProgramCacheResizeANGLE(display, limit, mode),
                         "eglProgramCacheResizeANGLE", GetDisplayIfValid(display), 0);

    thread->setSuccess();
    return display->programCacheResize(limit, mode);
}

EGLint EGLAPIENTRY EGL_DebugMessageControlKHR(EGLDEBUGPROCKHR callback,
                                              const EGLAttrib *attrib_list)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    FUNC_EVENT("EGLDEBUGPROCKHR callback = 0x%016" PRIxPTR
               ", EGLAttrib attrib_list = 0x%016" PRIxPTR,
               (uintptr_t)callback, (uintptr_t)attrib_list);

    Thread *thread = egl::GetCurrentThread();

    AttributeMap attributes = AttributeMap::CreateFromAttribArray(attrib_list);

    Error error = ValidateDebugMessageControlKHR(callback, attributes);
    if (error.isError())
    {
        thread->setError(error, GetDebug(), "eglDebugMessageControlKHR", nullptr);
        return error.getCode();
    }

    Debug *debug = GetDebug();
    debug->setCallback(callback, attributes);

    thread->setSuccess();
    return EGL_SUCCESS;
}

EGLBoolean EGLAPIENTRY EGL_QueryDebugKHR(EGLint attribute, EGLAttrib *value)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    FUNC_EVENT("EGLint attribute = 0x%X, EGLAttrib* value = 0x%016" PRIxPTR, attribute,
               (uintptr_t)value);

    Thread *thread = egl::GetCurrentThread();

    Error error = ValidateQueryDebugKHR(attribute, value);
    if (error.isError())
    {
        thread->setError(error, GetDebug(), "eglQueryDebugKHR", nullptr);
        return EGL_FALSE;
    }

    Debug *debug = GetDebug();
    switch (attribute)
    {
        case EGL_DEBUG_MSG_CRITICAL_KHR:
        case EGL_DEBUG_MSG_ERROR_KHR:
        case EGL_DEBUG_MSG_WARN_KHR:
        case EGL_DEBUG_MSG_INFO_KHR:
            *value = debug->isMessageTypeEnabled(FromEGLenum<MessageType>(attribute)) ? EGL_TRUE
                                                                                      : EGL_FALSE;
            break;
        case EGL_DEBUG_CALLBACK_KHR:
            *value = reinterpret_cast<EGLAttrib>(debug->getCallback());
            break;

        default:
            UNREACHABLE();
    }

    thread->setSuccess();
    return EGL_TRUE;
}

EGLint EGLAPIENTRY EGL_LabelObjectKHR(EGLDisplay dpy,
                                      EGLenum objectType,
                                      EGLObjectKHR object,
                                      EGLLabelKHR label)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    FUNC_EVENT("EGLDisplay dpy = 0x%016" PRIxPTR
               ", EGLenum objectType = 0x%X, EGLObjectKHR object = 0x%016" PRIxPTR
               ", "
               "EGLLabelKHR label = 0x%016" PRIxPTR,
               (uintptr_t)dpy, objectType, (uintptr_t)object, (uintptr_t)label);

    egl::Display *display = static_cast<egl::Display *>(dpy);
    Thread *thread        = egl::GetCurrentThread();

    ObjectType objectTypePacked = FromEGLenum<ObjectType>(objectType);
    Error error = ValidateLabelObjectKHR(thread, display, objectTypePacked, object, label);
    if (error.isError())
    {
        thread->setError(error, GetDebug(), "eglLabelObjectKHR",
                         GetLabeledObjectIfValid(thread, display, objectTypePacked, object));
        return error.getCode();
    }

    LabeledObject *labeledObject =
        GetLabeledObjectIfValid(thread, display, objectTypePacked, object);
    ASSERT(labeledObject != nullptr);
    labeledObject->setLabel(label);

    thread->setSuccess();
    return EGL_SUCCESS;
}

ANGLE_EXPORT EGLBoolean EGLAPIENTRY EGL_GetCompositorTimingSupportedANDROID(EGLDisplay dpy,
                                                                            EGLSurface surface,
                                                                            EGLint name)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    FUNC_EVENT("EGLDisplay dpy = 0x%016" PRIxPTR ", EGLSurface surface = 0x%016" PRIxPTR
               ", EGLint name = 0x%X",
               (uintptr_t)dpy, (uintptr_t)surface, name);

    egl::Display *display = static_cast<egl::Display *>(dpy);
    Surface *eglSurface   = static_cast<Surface *>(surface);
    Thread *thread        = egl::GetCurrentThread();

    CompositorTiming nameInternal = FromEGLenum<CompositorTiming>(name);

    ANGLE_EGL_TRY_RETURN(
        thread, ValidateGetCompositorTimingSupportedANDROID(display, eglSurface, nameInternal),
        "eglQueryTimestampSupportedANDROID", GetSurfaceIfValid(display, eglSurface), EGL_FALSE);

    thread->setSuccess();
    return eglSurface->getSupportedCompositorTimings().test(nameInternal);
}

ANGLE_EXPORT EGLBoolean EGLAPIENTRY EGL_GetCompositorTimingANDROID(EGLDisplay dpy,
                                                                   EGLSurface surface,
                                                                   EGLint numTimestamps,
                                                                   const EGLint *names,
                                                                   EGLnsecsANDROID *values)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    FUNC_EVENT("EGLDisplay dpy = 0x%016" PRIxPTR ", EGLSurface surface = 0x%016" PRIxPTR
               ", EGLint numTimestamps = %d, const EGLint *names = 0x%016" PRIxPTR
               ", EGLnsecsANDROID *values = 0x%016" PRIxPTR,
               (uintptr_t)dpy, (uintptr_t)surface, numTimestamps, (uintptr_t)names,
               (uintptr_t)values);

    egl::Display *display = static_cast<egl::Display *>(dpy);
    Surface *eglSurface   = static_cast<Surface *>(surface);
    Thread *thread        = egl::GetCurrentThread();

    ANGLE_EGL_TRY_RETURN(
        thread,
        ValidateGetCompositorTimingANDROID(display, eglSurface, numTimestamps, names, values),
        "eglGetCompositorTimingANDROIDD", GetSurfaceIfValid(display, eglSurface), EGL_FALSE);
    ANGLE_EGL_TRY_RETURN(thread, eglSurface->getCompositorTiming(numTimestamps, names, values),
                         "eglGetCompositorTimingANDROIDD", GetSurfaceIfValid(display, eglSurface),
                         EGL_FALSE);

    thread->setSuccess();
    return EGL_TRUE;
}

ANGLE_EXPORT EGLBoolean EGLAPIENTRY EGL_GetNextFrameIdANDROID(EGLDisplay dpy,
                                                              EGLSurface surface,
                                                              EGLuint64KHR *frameId)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    FUNC_EVENT("EGLDisplay dpy = 0x%016" PRIxPTR ", EGLSurface surface = 0x%016" PRIxPTR
               ", EGLuint64KHR *frameId = 0x%016" PRIxPTR,
               (uintptr_t)dpy, (uintptr_t)surface, (uintptr_t)frameId);

    egl::Display *display = static_cast<egl::Display *>(dpy);
    Surface *eglSurface   = static_cast<Surface *>(surface);
    Thread *thread        = egl::GetCurrentThread();

    ANGLE_EGL_TRY_RETURN(thread, ValidateGetNextFrameIdANDROID(display, eglSurface, frameId),
                         "eglGetNextFrameIdANDROID", GetSurfaceIfValid(display, eglSurface),
                         EGL_FALSE);
    ANGLE_EGL_TRY_RETURN(thread, eglSurface->getNextFrameId(frameId), "eglGetNextFrameIdANDROID",
                         GetSurfaceIfValid(display, eglSurface), EGL_FALSE);

    thread->setSuccess();
    return EGL_TRUE;
}

ANGLE_EXPORT EGLBoolean EGLAPIENTRY EGL_GetFrameTimestampSupportedANDROID(EGLDisplay dpy,
                                                                          EGLSurface surface,
                                                                          EGLint timestamp)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    FUNC_EVENT("EGLDisplay dpy = 0x%016" PRIxPTR ", EGLSurface surface = 0x%016" PRIxPTR
               ", EGLint timestamp = 0x%X",
               (uintptr_t)dpy, (uintptr_t)surface, timestamp);

    egl::Display *display = static_cast<egl::Display *>(dpy);
    Surface *eglSurface   = static_cast<Surface *>(surface);
    Thread *thread        = egl::GetCurrentThread();

    Timestamp timestampInternal = FromEGLenum<Timestamp>(timestamp);

    ANGLE_EGL_TRY_RETURN(
        thread, ValidateGetFrameTimestampSupportedANDROID(display, eglSurface, timestampInternal),
        "eglQueryTimestampSupportedANDROID", GetSurfaceIfValid(display, eglSurface), EGL_FALSE);

    thread->setSuccess();
    return eglSurface->getSupportedTimestamps().test(timestampInternal);
}

ANGLE_EXPORT EGLBoolean EGLAPIENTRY EGL_GetFrameTimestampsANDROID(EGLDisplay dpy,
                                                                  EGLSurface surface,
                                                                  EGLuint64KHR frameId,
                                                                  EGLint numTimestamps,
                                                                  const EGLint *timestamps,
                                                                  EGLnsecsANDROID *values)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    FUNC_EVENT(
        "EGLDisplay dpy = 0x%016" PRIxPTR ", EGLSurface surface = 0x%016" PRIxPTR
        ", EGLuint64KHR frameId = %llu, EGLint numTimestamps = %d, const EGLint *timestamps = "
        "0x%016" PRIxPTR ", EGLnsecsANDROID *values = 0x%016" PRIxPTR,
        (uintptr_t)dpy, (uintptr_t)surface, (unsigned long long)frameId, numTimestamps,
        (uintptr_t)timestamps, (uintptr_t)values);

    egl::Display *display = static_cast<egl::Display *>(dpy);
    Surface *eglSurface   = static_cast<Surface *>(surface);
    Thread *thread        = egl::GetCurrentThread();

    ANGLE_EGL_TRY_RETURN(thread,
                         ValidateGetFrameTimestampsANDROID(display, eglSurface, frameId,
                                                           numTimestamps, timestamps, values),
                         "eglGetFrameTimestampsANDROID", GetSurfaceIfValid(display, eglSurface),
                         EGL_FALSE);
    ANGLE_EGL_TRY_RETURN(
        thread, eglSurface->getFrameTimestamps(frameId, numTimestamps, timestamps, values),
        "eglGetFrameTimestampsANDROID", GetSurfaceIfValid(display, eglSurface), EGL_FALSE);

    thread->setSuccess();
    return EGL_TRUE;
}

// EGL_ANGLE_feature_control
ANGLE_EXPORT const char *EGLAPIENTRY EGL_QueryStringiANGLE(EGLDisplay dpy,
                                                           EGLint name,
                                                           EGLint index)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    FUNC_EVENT("EGLDisplay dpy = 0x%016" PRIxPTR ", EGLint name = %d, EGLint index = %d",
               (uintptr_t)dpy, name, index);

    egl::Display *display = static_cast<egl::Display *>(dpy);
    Thread *thread        = egl::GetCurrentThread();

    ANGLE_EGL_TRY_RETURN(thread, ValidateQueryStringiANGLE(display, name, index),
                         "eglQueryStringiANGLE", GetDisplayIfValid(display), nullptr);

    thread->setSuccess();
    return display->queryStringi(name, index);
}

EGLClientBuffer EGLAPIENTRY EGL_GetNativeClientBufferANDROID(const struct AHardwareBuffer *buffer)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    FUNC_EVENT("const struct AHardwareBuffer *buffer = 0x%016" PRIxPTR, (uintptr_t)buffer);

    Thread *thread = egl::GetCurrentThread();

    ANGLE_EGL_TRY_RETURN(thread, ValidateGetNativeClientBufferANDROID(buffer),
                         "eglGetNativeClientBufferANDROID", nullptr, nullptr);

    thread->setSuccess();
    return egl::Display::GetNativeClientBuffer(buffer);
}

EGLint EGLAPIENTRY EGL_DupNativeFenceFDANDROID(EGLDisplay dpy, EGLSyncKHR sync)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    FUNC_EVENT("EGLDisplay dpy = 0x%016" PRIxPTR ", EGLSyncKHR sync = 0x%016" PRIxPTR,
               (uintptr_t)dpy, (uintptr_t)sync);

    egl::Display *display = static_cast<egl::Display *>(dpy);
    Sync *syncObject      = static_cast<Sync *>(sync);
    Thread *thread        = egl::GetCurrentThread();

    ANGLE_EGL_TRY_RETURN(thread, ValidateDupNativeFenceFDANDROID(display, syncObject),
                         "eglDupNativeFenceFDANDROID", GetSyncIfValid(display, syncObject),
                         EGL_NO_NATIVE_FENCE_FD_ANDROID);

    EGLint result = EGL_NO_NATIVE_FENCE_FD_ANDROID;
    ANGLE_EGL_TRY_RETURN(thread, syncObject->dupNativeFenceFD(display, &result),
                         "eglDupNativeFenceFDANDROID", GetSyncIfValid(display, syncObject),
                         EGL_NO_NATIVE_FENCE_FD_ANDROID);

    thread->setSuccess();
    return result;
}

}  // extern "C"
