//
// Copyright 2014 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// entry_points_egl.cpp : Implements the EGL entry points.

#include "libGLESv2/entry_points_egl.h"

#include "common/debug.h"
#include "common/utilities.h"
#include "common/version.h"
#include "libANGLE/Context.h"
#include "libANGLE/Display.h"
#include "libANGLE/EGLSync.h"
#include "libANGLE/Surface.h"
#include "libANGLE/Texture.h"
#include "libANGLE/Thread.h"
#include "libANGLE/entry_points_utils.h"
#include "libANGLE/queryutils.h"
#include "libANGLE/validationEGL.h"
#include "libGLESv2/global_state.h"
#include "libGLESv2/proc_table_egl.h"

using namespace egl;

namespace
{

bool CompareProc(const ProcEntry &a, const char *b)
{
    return strcmp(a.first, b) < 0;
}

void ClipConfigs(const std::vector<const Config *> &filteredConfigs,
                 EGLConfig *output_configs,
                 EGLint config_size,
                 EGLint *num_config)
{
    EGLint result_size = static_cast<EGLint>(filteredConfigs.size());
    if (output_configs)
    {
        result_size = std::max(std::min(result_size, config_size), 0);
        for (EGLint i = 0; i < result_size; i++)
        {
            output_configs[i] = const_cast<Config *>(filteredConfigs[i]);
        }
    }
    *num_config = result_size;
}
}  // anonymous namespace

extern "C" {
// EGL 1.0
EGLint EGLAPIENTRY EGL_GetError(void)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    EVENT(__FUNCTION__, "");
    Thread *thread = egl::GetCurrentThread();

    EGLint error = thread->getError();
    thread->setSuccess();
    return error;
}

EGLDisplay EGLAPIENTRY EGL_GetDisplay(EGLNativeDisplayType display_id)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    FUNC_EVENT("EGLNativeDisplayType display_id = 0x%016" PRIxPTR, (uintptr_t)display_id);

    return egl::Display::GetDisplayFromNativeDisplay(display_id, AttributeMap());
}

EGLBoolean EGLAPIENTRY EGL_Initialize(EGLDisplay dpy, EGLint *major, EGLint *minor)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    FUNC_EVENT("EGLDisplay dpy = 0x%016" PRIxPTR ", EGLint *major = 0x%016" PRIxPTR
               ", EGLint *minor = 0x%016" PRIxPTR,
               (uintptr_t)dpy, (uintptr_t)major, (uintptr_t)minor);
    Thread *thread = egl::GetCurrentThread();

    egl::Display *display = static_cast<egl::Display *>(dpy);
    ANGLE_EGL_TRY_RETURN(thread, ValidateInitialize(display), "eglInitialize",
                         GetDisplayIfValid(display), EGL_FALSE);

    ANGLE_EGL_TRY_RETURN(thread, display->initialize(), "eglInitialize", GetDisplayIfValid(display),
                         EGL_FALSE);

    if (major)
        *major = 1;
    if (minor)
        *minor = 4;

    thread->setSuccess();
    return EGL_TRUE;
}

EGLBoolean EGLAPIENTRY EGL_Terminate(EGLDisplay dpy)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    FUNC_EVENT("EGLDisplay dpy = 0x%016" PRIxPTR, (uintptr_t)dpy);
    Thread *thread = egl::GetCurrentThread();

    egl::Display *display = static_cast<egl::Display *>(dpy);
    ANGLE_EGL_TRY_RETURN(thread, ValidateTerminate(display), "eglTerminate",
                         GetDisplayIfValid(display), EGL_FALSE);

    ANGLE_EGL_TRY_RETURN(thread, display->makeCurrent(thread, nullptr, nullptr, nullptr),
                         "eglTerminate", GetDisplayIfValid(display), EGL_FALSE);
    SetContextCurrent(thread, nullptr);
    ANGLE_EGL_TRY_RETURN(thread, display->terminate(thread), "eglTerminate",
                         GetDisplayIfValid(display), EGL_FALSE);

    thread->setSuccess();
    return EGL_TRUE;
}

const char *EGLAPIENTRY EGL_QueryString(EGLDisplay dpy, EGLint name)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    FUNC_EVENT("EGLDisplay dpy = 0x%016" PRIxPTR ", EGLint name = %d", (uintptr_t)dpy, name);
    Thread *thread = egl::GetCurrentThread();

    egl::Display *display = static_cast<egl::Display *>(dpy);
    if (!(display == EGL_NO_DISPLAY && name == EGL_EXTENSIONS))
    {
        ANGLE_EGL_TRY_RETURN(thread, ValidateDisplay(display), "eglQueryString",
                             GetDisplayIfValid(display), nullptr);
    }

    const char *result;
    switch (name)
    {
        case EGL_CLIENT_APIS:
            result = "OpenGL_ES";
            break;
        case EGL_EXTENSIONS:
            if (display == EGL_NO_DISPLAY)
            {
                result = egl::Display::GetClientExtensionString().c_str();
            }
            else
            {
                result = display->getExtensionString().c_str();
            }
            break;
        case EGL_VENDOR:
            result = display->getVendorString().c_str();
            break;
        case EGL_VERSION:
            result = "1.4 (ANGLE " ANGLE_VERSION_STRING ")";
            break;
        default:
            thread->setError(EglBadParameter(), GetDebug(), "eglQueryString",
                             GetDisplayIfValid(display));
            return nullptr;
    }

    thread->setSuccess();
    return result;
}

EGLBoolean EGLAPIENTRY EGL_GetConfigs(EGLDisplay dpy,
                                      EGLConfig *configs,
                                      EGLint config_size,
                                      EGLint *num_config)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    FUNC_EVENT("EGLDisplay dpy = 0x%016" PRIxPTR ", EGLConfig *configs = 0x%016" PRIxPTR
               ", "
               "EGLint config_size = %d, EGLint *num_config = 0x%016" PRIxPTR,
               (uintptr_t)dpy, (uintptr_t)configs, config_size, (uintptr_t)num_config);
    Thread *thread = egl::GetCurrentThread();

    egl::Display *display = static_cast<egl::Display *>(dpy);

    ANGLE_EGL_TRY_RETURN(thread, ValidateGetConfigs(display, config_size, num_config),
                         "eglGetConfigs", GetDisplayIfValid(display), EGL_FALSE);

    ClipConfigs(display->getConfigs(AttributeMap()), configs, config_size, num_config);

    thread->setSuccess();
    return EGL_TRUE;
}

EGLBoolean EGLAPIENTRY EGL_ChooseConfig(EGLDisplay dpy,
                                        const EGLint *attrib_list,
                                        EGLConfig *configs,
                                        EGLint config_size,
                                        EGLint *num_config)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    FUNC_EVENT("EGLDisplay dpy = 0x%016" PRIxPTR ", const EGLint *attrib_list = 0x%016" PRIxPTR
               ", "
               "EGLConfig *configs = 0x%016" PRIxPTR
               ", EGLint config_size = %d, EGLint *num_config = 0x%016" PRIxPTR,
               (uintptr_t)dpy, (uintptr_t)attrib_list, (uintptr_t)configs, config_size,
               (uintptr_t)num_config);
    Thread *thread = egl::GetCurrentThread();

    egl::Display *display  = static_cast<egl::Display *>(dpy);
    AttributeMap attribMap = AttributeMap::CreateFromIntArray(attrib_list);

    ANGLE_EGL_TRY_RETURN(thread, ValidateChooseConfig(display, attribMap, config_size, num_config),
                         "eglChooseConfig", GetDisplayIfValid(display), EGL_FALSE);

    ClipConfigs(display->chooseConfig(attribMap), configs, config_size, num_config);

    thread->setSuccess();
    return EGL_TRUE;
}

EGLBoolean EGLAPIENTRY EGL_GetConfigAttrib(EGLDisplay dpy,
                                           EGLConfig config,
                                           EGLint attribute,
                                           EGLint *value)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    FUNC_EVENT("EGLDisplay dpy = 0x%016" PRIxPTR ", EGLConfig config = 0x%016" PRIxPTR
               ", EGLint attribute = %d, EGLint "
               "*value = 0x%016" PRIxPTR,
               (uintptr_t)dpy, (uintptr_t)config, attribute, (uintptr_t)value);
    Thread *thread = egl::GetCurrentThread();

    egl::Display *display = static_cast<egl::Display *>(dpy);
    Config *configuration = static_cast<Config *>(config);

    ANGLE_EGL_TRY_RETURN(thread, ValidateGetConfigAttrib(display, configuration, attribute),
                         "eglGetConfigAttrib", GetDisplayIfValid(display), EGL_FALSE);

    QueryConfigAttrib(configuration, attribute, value);

    thread->setSuccess();
    return EGL_TRUE;
}

EGLSurface EGLAPIENTRY EGL_CreateWindowSurface(EGLDisplay dpy,
                                               EGLConfig config,
                                               EGLNativeWindowType win,
                                               const EGLint *attrib_list)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    FUNC_EVENT("EGLDisplay dpy = 0x%016" PRIxPTR ", EGLConfig config = 0x%016" PRIxPTR
               ", EGLNativeWindowType win = 0x%016" PRIxPTR
               ", "
               "const EGLint *attrib_list = 0x%016" PRIxPTR,
               (uintptr_t)dpy, (uintptr_t)config, (uintptr_t)win, (uintptr_t)attrib_list);
    Thread *thread = egl::GetCurrentThread();

    egl::Display *display   = static_cast<egl::Display *>(dpy);
    Config *configuration   = static_cast<Config *>(config);
    AttributeMap attributes = AttributeMap::CreateFromIntArray(attrib_list);

    ANGLE_EGL_TRY_RETURN(thread,
                         ValidateCreateWindowSurface(display, configuration, win, attributes),
                         "eglCreateWindowSurface", GetDisplayIfValid(display), EGL_NO_SURFACE);

    egl::Surface *surface = nullptr;
    ANGLE_EGL_TRY_RETURN(thread,
                         display->createWindowSurface(configuration, win, attributes, &surface),
                         "eglCreateWindowSurface", GetDisplayIfValid(display), EGL_NO_SURFACE);

    return static_cast<EGLSurface>(surface);
}

EGLSurface EGLAPIENTRY EGL_CreatePbufferSurface(EGLDisplay dpy,
                                                EGLConfig config,
                                                const EGLint *attrib_list)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    FUNC_EVENT("EGLDisplay dpy = 0x%016" PRIxPTR ", EGLConfig config = 0x%016" PRIxPTR
               ", const EGLint *attrib_list = "
               "0x%016" PRIxPTR,
               (uintptr_t)dpy, (uintptr_t)config, (uintptr_t)attrib_list);
    Thread *thread = egl::GetCurrentThread();

    egl::Display *display   = static_cast<egl::Display *>(dpy);
    Config *configuration   = static_cast<Config *>(config);
    AttributeMap attributes = AttributeMap::CreateFromIntArray(attrib_list);

    ANGLE_EGL_TRY_RETURN(thread, ValidateCreatePbufferSurface(display, configuration, attributes),
                         "eglCreatePbufferSurface", GetDisplayIfValid(display), EGL_NO_SURFACE);

    egl::Surface *surface = nullptr;
    ANGLE_EGL_TRY_RETURN(thread, display->createPbufferSurface(configuration, attributes, &surface),
                         "eglCreatePbufferSurface", GetDisplayIfValid(display), EGL_NO_SURFACE);

    return static_cast<EGLSurface>(surface);
}

EGLSurface EGLAPIENTRY EGL_CreatePixmapSurface(EGLDisplay dpy,
                                               EGLConfig config,
                                               EGLNativePixmapType pixmap,
                                               const EGLint *attrib_list)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    FUNC_EVENT("EGLDisplay dpy = 0x%016" PRIxPTR ", EGLConfig config = 0x%016" PRIxPTR
               ", EGLNativePixmapType pixmap = "
               "0x%016" PRIxPTR
               ", "
               "const EGLint *attrib_list = 0x%016" PRIxPTR,
               (uintptr_t)dpy, (uintptr_t)config, (uintptr_t)pixmap, (uintptr_t)attrib_list);
    Thread *thread = egl::GetCurrentThread();

    egl::Display *display = static_cast<egl::Display *>(dpy);
    Config *configuration = static_cast<Config *>(config);

    ANGLE_EGL_TRY_RETURN(thread, ValidateConfig(display, configuration), "eglCreatePixmapSurface",
                         GetDisplayIfValid(display), EGL_NO_SURFACE);

    UNIMPLEMENTED();  // FIXME

    thread->setSuccess();
    return EGL_NO_SURFACE;
}

EGLBoolean EGLAPIENTRY EGL_DestroySurface(EGLDisplay dpy, EGLSurface surface)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    FUNC_EVENT("EGLDisplay dpy = 0x%016" PRIxPTR ", EGLSurface surface = 0x%016" PRIxPTR,
               (uintptr_t)dpy, (uintptr_t)surface);
    Thread *thread = egl::GetCurrentThread();

    egl::Display *display = static_cast<egl::Display *>(dpy);
    Surface *eglSurface   = static_cast<Surface *>(surface);

    ANGLE_EGL_TRY_RETURN(thread, ValidateDestroySurface(display, eglSurface, surface),
                         "eglDestroySurface", GetSurfaceIfValid(display, eglSurface), EGL_FALSE);

    ANGLE_EGL_TRY_RETURN(thread, display->destroySurface(eglSurface), "eglDestroySurface",
                         GetSurfaceIfValid(display, eglSurface), EGL_FALSE);

    thread->setSuccess();
    return EGL_TRUE;
}

EGLBoolean EGLAPIENTRY EGL_QuerySurface(EGLDisplay dpy,
                                        EGLSurface surface,
                                        EGLint attribute,
                                        EGLint *value)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    FUNC_EVENT("EGLDisplay dpy = 0x%016" PRIxPTR ", EGLSurface surface = 0x%016" PRIxPTR
               ", EGLint attribute = %d, EGLint "
               "*value = 0x%016" PRIxPTR,
               (uintptr_t)dpy, (uintptr_t)surface, attribute, (uintptr_t)value);
    Thread *thread = egl::GetCurrentThread();

    const egl::Display *display = static_cast<const egl::Display *>(dpy);
    const Surface *eglSurface   = static_cast<const Surface *>(surface);

    ANGLE_EGL_TRY_RETURN(thread, ValidateQuerySurface(display, eglSurface, attribute, value),
                         "eglQuerySurface", GetSurfaceIfValid(display, eglSurface), EGL_FALSE);

    QuerySurfaceAttrib(eglSurface, attribute, value);

    thread->setSuccess();
    return EGL_TRUE;
}

EGLContext EGLAPIENTRY EGL_CreateContext(EGLDisplay dpy,
                                         EGLConfig config,
                                         EGLContext share_context,
                                         const EGLint *attrib_list)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    FUNC_EVENT("EGLDisplay dpy = 0x%016" PRIxPTR ", EGLConfig config = 0x%016" PRIxPTR
               ", EGLContext share_context = %d, "
               "const EGLint *attrib_list = 0x%016" PRIxPTR,
               (uintptr_t)dpy, (uintptr_t)config, CID(dpy, share_context), (uintptr_t)attrib_list);
    Thread *thread = egl::GetCurrentThread();

    egl::Display *display        = static_cast<egl::Display *>(dpy);
    Config *configuration        = static_cast<Config *>(config);
    gl::Context *sharedGLContext = static_cast<gl::Context *>(share_context);
    AttributeMap attributes      = AttributeMap::CreateFromIntArray(attrib_list);

    ANGLE_EGL_TRY_RETURN(thread,
                         ValidateCreateContext(display, configuration, sharedGLContext, attributes),
                         "eglCreateContext", GetDisplayIfValid(display), EGL_NO_CONTEXT);

    gl::Context *context = nullptr;
    ANGLE_EGL_TRY_RETURN(thread,
                         display->createContext(configuration, sharedGLContext, thread->getAPI(),
                                                attributes, &context),
                         "eglCreateContext", GetDisplayIfValid(display), EGL_NO_CONTEXT);

    thread->setSuccess();
    return static_cast<EGLContext>(context);
}

EGLBoolean EGLAPIENTRY EGL_DestroyContext(EGLDisplay dpy, EGLContext ctx)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    FUNC_EVENT("EGLDisplay dpy = 0x%016" PRIxPTR ", EGLContext ctx = %d", (uintptr_t)dpy,
               CID(dpy, ctx));
    Thread *thread = egl::GetCurrentThread();

    egl::Display *display = static_cast<egl::Display *>(dpy);
    gl::Context *context  = static_cast<gl::Context *>(ctx);

    ANGLE_EGL_TRY_RETURN(thread, ValidateDestroyContext(display, context, ctx), "eglDestroyContext",
                         GetContextIfValid(display, context), EGL_FALSE);

    bool contextWasCurrent = context == thread->getContext();

    ANGLE_EGL_TRY_RETURN(thread, display->destroyContext(thread, context), "eglDestroyContext",
                         GetContextIfValid(display, context), EGL_FALSE);

    if (contextWasCurrent)
    {
        SetContextCurrent(thread, nullptr);
    }

    thread->setSuccess();
    return EGL_TRUE;
}

EGLBoolean EGLAPIENTRY EGL_MakeCurrent(EGLDisplay dpy,
                                       EGLSurface draw,
                                       EGLSurface read,
                                       EGLContext ctx)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    FUNC_EVENT("EGLDisplay dpy = 0x%016" PRIxPTR ", EGLSurface draw = 0x%016" PRIxPTR
               ", EGLSurface read = 0x%016" PRIxPTR
               ", "
               "EGLContext ctx = %d",
               (uintptr_t)dpy, (uintptr_t)draw, (uintptr_t)read, CID(dpy, ctx));
    Thread *thread = egl::GetCurrentThread();

    egl::Display *display = static_cast<egl::Display *>(dpy);
    Surface *drawSurface  = static_cast<Surface *>(draw);
    Surface *readSurface  = static_cast<Surface *>(read);
    gl::Context *context  = static_cast<gl::Context *>(ctx);

    ANGLE_EGL_TRY_RETURN(thread, ValidateMakeCurrent(display, drawSurface, readSurface, context),
                         "eglMakeCurrent", GetContextIfValid(display, context), EGL_FALSE);

    Surface *previousDraw        = thread->getCurrentDrawSurface();
    Surface *previousRead        = thread->getCurrentReadSurface();
    gl::Context *previousContext = thread->getContext();

    // Only call makeCurrent if the context or surfaces have changed.
    if (previousDraw != drawSurface || previousRead != readSurface || previousContext != context)
    {
        ANGLE_EGL_TRY_RETURN(thread,
                             display->makeCurrent(thread, drawSurface, readSurface, context),
                             "eglMakeCurrent", GetContextIfValid(display, context), EGL_FALSE);

        SetContextCurrent(thread, context);
    }

    thread->setSuccess();
    return EGL_TRUE;
}

EGLSurface EGLAPIENTRY EGL_GetCurrentSurface(EGLint readdraw)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    FUNC_EVENT("EGLint readdraw = %d", readdraw);
    Thread *thread = egl::GetCurrentThread();

    if (readdraw == EGL_READ)
    {
        thread->setSuccess();
        return thread->getCurrentReadSurface();
    }
    else if (readdraw == EGL_DRAW)
    {
        thread->setSuccess();
        return thread->getCurrentDrawSurface();
    }
    else
    {
        thread->setError(EglBadParameter(), GetDebug(), "eglGetCurrentSurface", nullptr);
        return EGL_NO_SURFACE;
    }
}

EGLDisplay EGLAPIENTRY EGL_GetCurrentDisplay(void)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    EVENT(__FUNCTION__, "");
    Thread *thread = egl::GetCurrentThread();

    thread->setSuccess();
    if (thread->getContext() != nullptr)
    {
        return thread->getContext()->getDisplay();
    }
    return EGL_NO_DISPLAY;
}

EGLBoolean EGLAPIENTRY EGL_QueryContext(EGLDisplay dpy,
                                        EGLContext ctx,
                                        EGLint attribute,
                                        EGLint *value)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    FUNC_EVENT("EGLDisplay dpy = 0x%016" PRIxPTR
               ", EGLContext ctx = %d"
               ", EGLint attribute = %d, EGLint *value "
               "= 0x%016" PRIxPTR,
               (uintptr_t)dpy, CID(dpy, ctx), attribute, (uintptr_t)value);
    Thread *thread = egl::GetCurrentThread();

    egl::Display *display = static_cast<egl::Display *>(dpy);
    gl::Context *context  = static_cast<gl::Context *>(ctx);

    ANGLE_EGL_TRY_RETURN(thread, ValidateQueryContext(display, context, attribute, value),
                         "eglQueryContext", GetContextIfValid(display, context), EGL_FALSE);

    QueryContextAttrib(context, attribute, value);

    thread->setSuccess();
    return EGL_TRUE;
}

EGLBoolean EGLAPIENTRY EGL_WaitGL(void)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    EVENT(__FUNCTION__, "");
    Thread *thread = egl::GetCurrentThread();

    egl::Display *display = thread->getDisplay();

    ANGLE_EGL_TRY_RETURN(thread, ValidateDisplay(display), "eglWaitGL", GetDisplayIfValid(display),
                         EGL_FALSE);

    // eglWaitGL like calling eglWaitClient with the OpenGL ES API bound. Since we only implement
    // OpenGL ES we can do the call directly.
    ANGLE_EGL_TRY_RETURN(thread, display->waitClient(thread->getContext()), "eglWaitGL",
                         GetDisplayIfValid(display), EGL_FALSE);

    thread->setSuccess();
    return EGL_TRUE;
}

EGLBoolean EGLAPIENTRY EGL_WaitNative(EGLint engine)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    FUNC_EVENT("EGLint engine = %d", engine);
    Thread *thread = egl::GetCurrentThread();

    egl::Display *display = thread->getDisplay();

    ANGLE_EGL_TRY_RETURN(thread, ValidateWaitNative(display, engine), "eglWaitNative",
                         GetThreadIfValid(thread), EGL_FALSE);

    ANGLE_EGL_TRY_RETURN(thread, display->waitNative(thread->getContext(), engine), "eglWaitNative",
                         GetThreadIfValid(thread), EGL_FALSE);

    thread->setSuccess();
    return EGL_TRUE;
}

EGLBoolean EGLAPIENTRY EGL_SwapBuffers(EGLDisplay dpy, EGLSurface surface)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    FUNC_EVENT("EGLDisplay dpy = 0x%016" PRIxPTR ", EGLSurface surface = 0x%016" PRIxPTR,
               (uintptr_t)dpy, (uintptr_t)surface);
    Thread *thread = egl::GetCurrentThread();

    egl::Display *display = static_cast<egl::Display *>(dpy);
    Surface *eglSurface   = (Surface *)surface;

    ANGLE_EGL_TRY_RETURN(thread, ValidateSwapBuffers(thread, display, eglSurface), "eglSwapBuffers",
                         GetSurfaceIfValid(display, eglSurface), EGL_FALSE);

    ANGLE_EGL_TRY_RETURN(thread, eglSurface->swap(thread->getContext()), "eglSwapBuffers",
                         GetSurfaceIfValid(display, eglSurface), EGL_FALSE);

    thread->setSuccess();
    return EGL_TRUE;
}

EGLBoolean EGLAPIENTRY EGL_CopyBuffers(EGLDisplay dpy,
                                       EGLSurface surface,
                                       EGLNativePixmapType target)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    FUNC_EVENT("EGLDisplay dpy = 0x%016" PRIxPTR ", EGLSurface surface = 0x%016" PRIxPTR
               ", EGLNativePixmapType target = "
               "0x%016" PRIxPTR,
               (uintptr_t)dpy, (uintptr_t)surface, (uintptr_t)target);
    Thread *thread = egl::GetCurrentThread();

    egl::Display *display = static_cast<egl::Display *>(dpy);
    Surface *eglSurface   = static_cast<Surface *>(surface);

    ANGLE_EGL_TRY_RETURN(thread, ValidateCopyBuffers(display, eglSurface), "eglCopyBuffers",
                         GetSurfaceIfValid(display, eglSurface), EGL_FALSE);

    UNIMPLEMENTED();  // FIXME

    thread->setSuccess();
    return 0;
}

// EGL 1.1
EGLBoolean EGLAPIENTRY EGL_BindTexImage(EGLDisplay dpy, EGLSurface surface, EGLint buffer)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    FUNC_EVENT("EGLDisplay dpy = 0x%016" PRIxPTR ", EGLSurface surface = 0x%016" PRIxPTR
               ", EGLint buffer = %d",
               (uintptr_t)dpy, (uintptr_t)surface, buffer);
    Thread *thread = egl::GetCurrentThread();

    egl::Display *display      = static_cast<egl::Display *>(dpy);
    Surface *eglSurface        = static_cast<Surface *>(surface);
    gl::Context *context       = thread->getContext();
    gl::Texture *textureObject = nullptr;

    ANGLE_EGL_TRY_RETURN(
        thread, ValidateBindTexImage(display, eglSurface, surface, buffer, context, &textureObject),
        "eglBindTexImage", GetSurfaceIfValid(display, eglSurface), EGL_FALSE);

    if (context)
    {
        ANGLE_EGL_TRY_RETURN(thread, eglSurface->bindTexImage(context, textureObject, buffer),
                             "eglBindTexImage", GetSurfaceIfValid(display, eglSurface), EGL_FALSE);
    }

    thread->setSuccess();
    return EGL_TRUE;
}

EGLBoolean EGLAPIENTRY EGL_SurfaceAttrib(EGLDisplay dpy,
                                         EGLSurface surface,
                                         EGLint attribute,
                                         EGLint value)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    FUNC_EVENT("EGLDisplay dpy = 0x%016" PRIxPTR ", EGLSurface surface = 0x%016" PRIxPTR
               ", EGLint attribute = %d, EGLint "
               "value = %d",
               (uintptr_t)dpy, (uintptr_t)surface, attribute, value);
    Thread *thread = egl::GetCurrentThread();

    egl::Display *display = static_cast<egl::Display *>(dpy);
    Surface *eglSurface   = static_cast<Surface *>(surface);

    ANGLE_EGL_TRY_RETURN(thread, ValidateSurfaceAttrib(display, eglSurface, attribute, value),
                         "eglSurfaceAttrib", GetSurfaceIfValid(display, eglSurface), EGL_FALSE);

    SetSurfaceAttrib(eglSurface, attribute, value);

    thread->setSuccess();
    return EGL_TRUE;
}

EGLBoolean EGLAPIENTRY EGL_ReleaseTexImage(EGLDisplay dpy, EGLSurface surface, EGLint buffer)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    FUNC_EVENT("EGLDisplay dpy = 0x%016" PRIxPTR ", EGLSurface surface = 0x%016" PRIxPTR
               ", EGLint buffer = %d",
               (uintptr_t)dpy, (uintptr_t)surface, buffer);
    Thread *thread = egl::GetCurrentThread();

    egl::Display *display = static_cast<egl::Display *>(dpy);
    Surface *eglSurface   = static_cast<Surface *>(surface);

    ANGLE_EGL_TRY_RETURN(thread, ValidateReleaseTexImage(display, eglSurface, surface, buffer),
                         "eglReleaseTexImage", GetSurfaceIfValid(display, eglSurface), EGL_FALSE);

    gl::Texture *texture = eglSurface->getBoundTexture();

    if (texture)
    {
        ANGLE_EGL_TRY_RETURN(thread, eglSurface->releaseTexImage(thread->getContext(), buffer),
                             "eglReleaseTexImage", GetSurfaceIfValid(display, eglSurface),
                             EGL_FALSE);
    }

    thread->setSuccess();
    return EGL_TRUE;
}

EGLBoolean EGLAPIENTRY EGL_SwapInterval(EGLDisplay dpy, EGLint interval)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    FUNC_EVENT("EGLDisplay dpy = 0x%016" PRIxPTR ", EGLint interval = %d", (uintptr_t)dpy,
               interval);
    Thread *thread       = egl::GetCurrentThread();
    gl::Context *context = thread->getContext();

    egl::Display *display = static_cast<egl::Display *>(dpy);
    Surface *draw_surface = static_cast<Surface *>(thread->getCurrentDrawSurface());

    ANGLE_EGL_TRY_RETURN(thread, ValidateSwapInterval(display, draw_surface, context),
                         "eglSwapInterval", GetDisplayIfValid(display), EGL_FALSE);

    const egl::Config *surfaceConfig = draw_surface->getConfig();
    EGLint clampedInterval           = std::min(std::max(interval, surfaceConfig->minSwapInterval),
                                      surfaceConfig->maxSwapInterval);

    draw_surface->setSwapInterval(clampedInterval);

    thread->setSuccess();
    return EGL_TRUE;
}

// EGL 1.2
EGLBoolean EGLAPIENTRY EGL_BindAPI(EGLenum api)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    FUNC_EVENT("EGLenum api = 0x%X", api);
    Thread *thread = egl::GetCurrentThread();

    ANGLE_EGL_TRY_RETURN(thread, ValidateBindAPI(api), "eglBindAPI", GetThreadIfValid(thread),
                         EGL_FALSE);

    thread->setAPI(api);

    thread->setSuccess();
    return EGL_TRUE;
}

EGLenum EGLAPIENTRY EGL_QueryAPI(void)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    EVENT(__FUNCTION__, "");
    Thread *thread = egl::GetCurrentThread();

    EGLenum API = thread->getAPI();

    thread->setSuccess();
    return API;
}

EGLSurface EGLAPIENTRY EGL_CreatePbufferFromClientBuffer(EGLDisplay dpy,
                                                         EGLenum buftype,
                                                         EGLClientBuffer buffer,
                                                         EGLConfig config,
                                                         const EGLint *attrib_list)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    FUNC_EVENT("EGLDisplay dpy = 0x%016" PRIxPTR
               ", EGLenum buftype = 0x%X, EGLClientBuffer buffer = 0x%016" PRIxPTR
               ", "
               "EGLConfig config = 0x%016" PRIxPTR ", const EGLint *attrib_list = 0x%016" PRIxPTR,
               (uintptr_t)dpy, buftype, (uintptr_t)buffer, (uintptr_t)config,
               (uintptr_t)attrib_list);
    Thread *thread = egl::GetCurrentThread();

    egl::Display *display   = static_cast<egl::Display *>(dpy);
    Config *configuration   = static_cast<Config *>(config);
    AttributeMap attributes = AttributeMap::CreateFromIntArray(attrib_list);

    ANGLE_EGL_TRY_RETURN(
        thread,
        ValidateCreatePbufferFromClientBuffer(display, buftype, buffer, configuration, attributes),
        "eglCreatePbufferFromClientBuffer", GetDisplayIfValid(display), EGL_NO_SURFACE);

    egl::Surface *surface = nullptr;
    ANGLE_EGL_TRY_RETURN(thread,
                         display->createPbufferFromClientBuffer(configuration, buftype, buffer,
                                                                attributes, &surface),
                         "eglCreatePbufferFromClientBuffer", GetDisplayIfValid(display),
                         EGL_NO_SURFACE);

    return static_cast<EGLSurface>(surface);
}

EGLBoolean EGLAPIENTRY EGL_ReleaseThread(void)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    EVENT(__FUNCTION__, "");
    Thread *thread = egl::GetCurrentThread();

    Surface *previousDraw         = thread->getCurrentDrawSurface();
    Surface *previousRead         = thread->getCurrentReadSurface();
    gl::Context *previousContext  = thread->getContext();
    egl::Display *previousDisplay = thread->getDisplay();

    // Only call makeCurrent if the context or surfaces have changed.
    if (previousDraw != EGL_NO_SURFACE || previousRead != EGL_NO_SURFACE ||
        previousContext != EGL_NO_CONTEXT)
    {
        if (previousDisplay != EGL_NO_DISPLAY)
        {
            ANGLE_EGL_TRY_RETURN(thread,
                                 previousDisplay->makeCurrent(thread, nullptr, nullptr, nullptr),
                                 "eglReleaseThread", nullptr, EGL_FALSE);
        }

        SetContextCurrent(thread, nullptr);
    }

    thread->setSuccess();
    return EGL_TRUE;
}

EGLBoolean EGLAPIENTRY EGL_WaitClient(void)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    EVENT(__FUNCTION__, "");
    Thread *thread = egl::GetCurrentThread();

    egl::Display *display = thread->getDisplay();
    gl::Context *context  = thread->getContext();

    ANGLE_EGL_TRY_RETURN(thread, ValidateDisplay(display), "eglWaitClient",
                         GetContextIfValid(display, context), EGL_FALSE);

    ANGLE_EGL_TRY_RETURN(thread, display->waitClient(context), "eglWaitClient",
                         GetContextIfValid(display, context), EGL_FALSE);

    thread->setSuccess();
    return EGL_TRUE;
}

// EGL 1.4
EGLContext EGLAPIENTRY EGL_GetCurrentContext(void)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    EVENT(__FUNCTION__, "");
    Thread *thread = egl::GetCurrentThread();

    gl::Context *context = thread->getContext();

    thread->setSuccess();
    return static_cast<EGLContext>(context);
}

// EGL 1.5
EGLSync EGLAPIENTRY EGL_CreateSync(EGLDisplay dpy, EGLenum type, const EGLAttrib *attrib_list)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    FUNC_EVENT("EGLDisplay dpy = 0x%016" PRIxPTR
               ", EGLenum type = 0x%X, const EGLint* attrib_list = 0x%016" PRIxPTR,
               (uintptr_t)dpy, type, (uintptr_t)attrib_list);

    Thread *thread          = egl::GetCurrentThread();
    egl::Display *display   = static_cast<egl::Display *>(dpy);
    AttributeMap attributes = AttributeMap::CreateFromAttribArray(attrib_list);

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

EGLBoolean EGLAPIENTRY EGL_DestroySync(EGLDisplay dpy, EGLSync sync)
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

EGLint EGLAPIENTRY EGL_ClientWaitSync(EGLDisplay dpy, EGLSync sync, EGLint flags, EGLTime timeout)
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

EGLBoolean EGLAPIENTRY EGL_GetSyncAttrib(EGLDisplay dpy,
                                         EGLSync sync,
                                         EGLint attribute,
                                         EGLAttrib *value)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    FUNC_EVENT("EGLDisplay dpy = 0x%016" PRIxPTR ", EGLSync sync = 0x%016" PRIxPTR
               ", EGLint attribute = 0x%X, EGLAttrib "
               "*value = 0x%016" PRIxPTR,
               (uintptr_t)dpy, (uintptr_t)sync, attribute, (uintptr_t)value);

    Thread *thread        = egl::GetCurrentThread();
    egl::Display *display = static_cast<egl::Display *>(dpy);
    egl::Sync *syncObject = static_cast<Sync *>(sync);

    ANGLE_EGL_TRY_RETURN(thread, ValidateGetSyncAttrib(display, syncObject, attribute, value),
                         "eglGetSyncAttrib", GetSyncIfValid(display, syncObject), EGL_FALSE);

    EGLint valueExt;
    ANGLE_EGL_TRY_RETURN(thread, GetSyncAttrib(display, syncObject, attribute, &valueExt),
                         "eglGetSyncAttrib", GetSyncIfValid(display, syncObject), EGL_FALSE);
    *value = valueExt;

    thread->setSuccess();
    return EGL_TRUE;
}

EGLImage EGLAPIENTRY EGL_CreateImage(EGLDisplay dpy,
                                     EGLContext ctx,
                                     EGLenum target,
                                     EGLClientBuffer buffer,
                                     const EGLAttrib *attrib_list)
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
    AttributeMap attributes = AttributeMap::CreateFromIntArray((const EGLint *)attrib_list);

    Error error = ValidateCreateImage(display, context, target, buffer, attributes);
    if (error.isError())
    {
        thread->setError(error, GetDebug(), "eglCreateImage", GetDisplayIfValid(display));
        return EGL_NO_IMAGE;
    }

    Image *image = nullptr;
    error        = display->createImage(context, target, buffer, attributes, &image);
    if (error.isError())
    {
        thread->setError(error, GetDebug(), "eglCreateImage", GetDisplayIfValid(display));
        return EGL_NO_IMAGE;
    }

    thread->setSuccess();
    return static_cast<EGLImage>(image);
}

EGLBoolean EGLAPIENTRY EGL_DestroyImage(EGLDisplay dpy, EGLImage image)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    FUNC_EVENT("EGLDisplay dpy = 0x%016" PRIxPTR ", EGLImage image = 0x%016" PRIxPTR,
               (uintptr_t)dpy, (uintptr_t)image);
    Thread *thread        = egl::GetCurrentThread();
    egl::Display *display = static_cast<egl::Display *>(dpy);
    Image *img            = static_cast<Image *>(image);

    Error error = ValidateDestroyImage(display, img);
    if (error.isError())
    {
        thread->setError(error, GetDebug(), "eglDestroyImage", GetImageIfValid(display, img));
        return EGL_FALSE;
    }

    display->destroyImage(img);

    thread->setSuccess();
    return EGL_TRUE;
}

EGLDisplay EGLAPIENTRY EGL_GetPlatformDisplay(EGLenum platform,
                                              void *native_display,
                                              const EGLAttrib *attrib_list)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    FUNC_EVENT("EGLenum platform = %d, void* native_display = 0x%016" PRIxPTR
               ", const EGLint* attrib_list = "
               "0x%016" PRIxPTR,
               platform, (uintptr_t)native_display, (uintptr_t)attrib_list);
    Thread *thread = egl::GetCurrentThread();

    ANGLE_EGL_TRY_RETURN(thread, ValidateGetPlatformDisplay(platform, native_display, attrib_list),
                         "eglGetPlatformDisplay", GetThreadIfValid(thread), EGL_NO_DISPLAY);

    const auto &attribMap = AttributeMap::CreateFromAttribArray(attrib_list);
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

EGLSurface EGLAPIENTRY EGL_CreatePlatformWindowSurface(EGLDisplay dpy,
                                                       EGLConfig config,
                                                       void *native_window,
                                                       const EGLAttrib *attrib_list)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    FUNC_EVENT("EGLDisplay dpy = 0x%016" PRIxPTR ", EGLConfig config = 0x%016" PRIxPTR
               ", void* native_window = 0x%016" PRIxPTR
               ", "
               "const EGLint* attrib_list = 0x%016" PRIxPTR,
               (uintptr_t)dpy, (uintptr_t)config, (uintptr_t)native_window, (uintptr_t)attrib_list);
    Thread *thread        = egl::GetCurrentThread();
    egl::Display *display = static_cast<egl::Display *>(dpy);

    UNIMPLEMENTED();
    thread->setError(EglBadDisplay() << "eglCreatePlatformWindowSurface unimplemented.", GetDebug(),
                     "eglCreatePlatformWindowSurface", GetDisplayIfValid(display));
    return EGL_NO_SURFACE;
}

EGLSurface EGLAPIENTRY EGL_CreatePlatformPixmapSurface(EGLDisplay dpy,
                                                       EGLConfig config,
                                                       void *native_pixmap,
                                                       const EGLAttrib *attrib_list)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    FUNC_EVENT("EGLDisplay dpy = 0x%016" PRIxPTR ", EGLConfig config = 0x%016" PRIxPTR
               ", void* native_pixmap = 0x%016" PRIxPTR
               ", "
               "const EGLint* attrib_list = 0x%016" PRIxPTR,
               (uintptr_t)dpy, (uintptr_t)config, (uintptr_t)native_pixmap, (uintptr_t)attrib_list);
    Thread *thread        = egl::GetCurrentThread();
    egl::Display *display = static_cast<egl::Display *>(dpy);

    UNIMPLEMENTED();
    thread->setError(EglBadDisplay() << "eglCreatePlatformPixmapSurface unimplemented.", GetDebug(),
                     "eglCreatePlatformPixmapSurface", GetDisplayIfValid(display));
    return EGL_NO_SURFACE;
}

EGLBoolean EGLAPIENTRY EGL_WaitSync(EGLDisplay dpy, EGLSync sync, EGLint flags)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    FUNC_EVENT("EGLDisplay dpy =0x%016" PRIxPTR "p, EGLSync sync = 0x%016" PRIxPTR
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

__eglMustCastToProperFunctionPointerType EGLAPIENTRY EGL_GetProcAddress(const char *procname)
{
    ANGLE_SCOPED_GLOBAL_LOCK();
    FUNC_EVENT("const char *procname = \"%s\"", procname);
    Thread *thread = egl::GetCurrentThread();

    ProcEntry *entry =
        std::lower_bound(&g_procTable[0], &g_procTable[g_numProcs], procname, CompareProc);

    thread->setSuccess();

    if (entry == &g_procTable[g_numProcs] || strcmp(entry->first, procname) != 0)
    {
        return nullptr;
    }

    return entry->second;
}
}  // extern "C"
