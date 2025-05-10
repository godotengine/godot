/*
  Simple DirectMedia Layer
  Copyright (C) 1997-2025 Sam Lantinga <slouken@libsdl.org>
  Copyright (C) 2021 NVIDIA Corporation

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
*/
#include "SDL_internal.h"

#ifdef SDL_VIDEO_DRIVER_X11

#include "SDL_x11video.h"
#include "SDL_x11xsync.h"

// GLX implementation of SDL OpenGL support

#ifdef SDL_VIDEO_OPENGL_GLX
#include "SDL_x11opengles.h"

#if defined(SDL_PLATFORM_IRIX) || defined(SDL_PLATFORM_NETBSD) || defined(SDL_PLATFORM_OPENBSD)
/*
 * IRIX doesn't have a GL library versioning system.
 * NetBSD and OpenBSD have different GL library versions depending on how
 * the library was installed.
 */
#define DEFAULT_OPENGL "libGL.so"
#elif defined(SDL_PLATFORM_MACOS)
#define DEFAULT_OPENGL "/opt/X11/lib/libGL.1.dylib"
#else
#define DEFAULT_OPENGL "libGL.so.1"
#endif

#ifndef GLX_NONE_EXT
#define GLX_NONE_EXT 0x8000
#endif

#ifndef GLX_ARB_multisample
#define GLX_ARB_multisample
#define GLX_SAMPLE_BUFFERS_ARB 100000
#define GLX_SAMPLES_ARB        100001
#endif

#ifndef GLX_EXT_visual_rating
#define GLX_EXT_visual_rating
#define GLX_VISUAL_CAVEAT_EXT         0x20
#define GLX_NONE_EXT                  0x8000
#define GLX_SLOW_VISUAL_EXT           0x8001
#define GLX_NON_CONFORMANT_VISUAL_EXT 0x800D
#endif

#ifndef GLX_EXT_visual_info
#define GLX_EXT_visual_info
#define GLX_X_VISUAL_TYPE_EXT 0x22
#define GLX_DIRECT_COLOR_EXT  0x8003
#endif

#ifndef GLX_ARB_create_context
#define GLX_ARB_create_context
#define GLX_CONTEXT_MAJOR_VERSION_ARB          0x2091
#define GLX_CONTEXT_MINOR_VERSION_ARB          0x2092
#define GLX_CONTEXT_FLAGS_ARB                  0x2094
#define GLX_CONTEXT_DEBUG_BIT_ARB              0x0001
#define GLX_CONTEXT_FORWARD_COMPATIBLE_BIT_ARB 0x0002

// Typedef for the GL 3.0 context creation function
typedef GLXContext (*PFNGLXCREATECONTEXTATTRIBSARBPROC)(Display *dpy,
                                                        GLXFBConfig config,
                                                        GLXContext
                                                            share_context,
                                                        Bool direct,
                                                        const int
                                                            *attrib_list);
#endif

#ifndef GLX_ARB_create_context_profile
#define GLX_ARB_create_context_profile
#define GLX_CONTEXT_PROFILE_MASK_ARB              0x9126
#define GLX_CONTEXT_CORE_PROFILE_BIT_ARB          0x00000001
#define GLX_CONTEXT_COMPATIBILITY_PROFILE_BIT_ARB 0x00000002
#endif

#ifndef GLX_ARB_create_context_robustness
#define GLX_ARB_create_context_robustness
#define GLX_CONTEXT_ROBUST_ACCESS_BIT_ARB           0x00000004
#define GLX_CONTEXT_RESET_NOTIFICATION_STRATEGY_ARB 0x8256
#define GLX_NO_RESET_NOTIFICATION_ARB               0x8261
#define GLX_LOSE_CONTEXT_ON_RESET_ARB               0x8252
#endif

#ifndef GLX_EXT_create_context_es2_profile
#define GLX_EXT_create_context_es2_profile
#ifndef GLX_CONTEXT_ES2_PROFILE_BIT_EXT
#define GLX_CONTEXT_ES2_PROFILE_BIT_EXT 0x00000002
#endif
#endif

#ifndef GLX_ARB_framebuffer_sRGB
#define GLX_ARB_framebuffer_sRGB
#ifndef GLX_FRAMEBUFFER_SRGB_CAPABLE_ARB
#define GLX_FRAMEBUFFER_SRGB_CAPABLE_ARB 0x20B2
#endif
#endif

#ifndef GLX_ARB_fbconfig_float
#define GLX_ARB_fbconfig_float
#ifndef GLX_RGBA_FLOAT_TYPE_ARB
#define GLX_RGBA_FLOAT_TYPE_ARB 0x20B9
#endif
#ifndef GLX_RGBA_FLOAT_BIT_ARB
#define GLX_RGBA_FLOAT_BIT_ARB 0x00000004
#endif
#endif

#ifndef GLX_ARB_create_context_no_error
#define GLX_ARB_create_context_no_error
#ifndef GLX_CONTEXT_OPENGL_NO_ERROR_ARB
#define GLX_CONTEXT_OPENGL_NO_ERROR_ARB 0x31B3
#endif
#endif

#ifndef GLX_EXT_swap_control
#define GLX_SWAP_INTERVAL_EXT     0x20F1
#define GLX_MAX_SWAP_INTERVAL_EXT 0x20F2
#endif

#ifndef GLX_EXT_swap_control_tear
#define GLX_LATE_SWAPS_TEAR_EXT 0x20F3
#endif

#ifndef GLX_ARB_context_flush_control
#define GLX_ARB_context_flush_control
#define GLX_CONTEXT_RELEASE_BEHAVIOR_ARB       0x2097
#define GLX_CONTEXT_RELEASE_BEHAVIOR_NONE_ARB  0x0000
#define GLX_CONTEXT_RELEASE_BEHAVIOR_FLUSH_ARB 0x2098
#endif

#define OPENGL_REQUIRES_DLOPEN
#if defined(OPENGL_REQUIRES_DLOPEN) && defined(HAVE_DLOPEN)
#include <dlfcn.h>
#define GL_LoadObject(X) dlopen(X, (RTLD_NOW | RTLD_GLOBAL))
#define GL_LoadFunction  dlsym
#define GL_UnloadObject  dlclose
#else
#define GL_LoadObject   SDL_LoadObject
#define GL_LoadFunction SDL_LoadFunction
#define GL_UnloadObject SDL_UnloadObject
#endif

static void X11_GL_InitExtensions(SDL_VideoDevice *_this);

bool X11_GL_LoadLibrary(SDL_VideoDevice *_this, const char *path)
{
    Display *display;
    SDL_SharedObject *handle;

    if (_this->gl_data) {
        return SDL_SetError("OpenGL context already created");
    }

    // Load the OpenGL library
    if (path == NULL) {
        path = SDL_GetHint(SDL_HINT_OPENGL_LIBRARY);
    }
    if (path == NULL) {
        path = DEFAULT_OPENGL;
    }
    _this->gl_config.dll_handle = GL_LoadObject(path);
    if (!_this->gl_config.dll_handle) {
#if defined(OPENGL_REQUIRES_DLOPEN) && defined(HAVE_DLOPEN)
        SDL_SetError("Failed loading %s: %s", path, dlerror());
#endif
        return false;
    }
    SDL_strlcpy(_this->gl_config.driver_path, path,
                SDL_arraysize(_this->gl_config.driver_path));

    // Allocate OpenGL memory
    _this->gl_data =
        (struct SDL_GLDriverData *)SDL_calloc(1,
                                              sizeof(struct
                                                     SDL_GLDriverData));
    if (!_this->gl_data) {
        return false;
    }

    // Load function pointers
    handle = _this->gl_config.dll_handle;
    _this->gl_data->glXQueryExtension =
        (Bool(*)(Display *, int *, int *))
            GL_LoadFunction(handle, "glXQueryExtension");
    _this->gl_data->glXGetProcAddress =
        (__GLXextFuncPtr (*)(const GLubyte *))
            GL_LoadFunction(handle, "glXGetProcAddressARB");
    _this->gl_data->glXChooseVisual =
        (XVisualInfo * (*)(Display *, int, int *))
            X11_GL_GetProcAddress(_this, "glXChooseVisual");
    _this->gl_data->glXCreateContext =
        (GLXContext(*)(Display *, XVisualInfo *, GLXContext, int))
            X11_GL_GetProcAddress(_this, "glXCreateContext");
    _this->gl_data->glXDestroyContext =
        (void (*)(Display *, GLXContext))
            X11_GL_GetProcAddress(_this, "glXDestroyContext");
    _this->gl_data->glXMakeCurrent =
        (int (*)(Display *, GLXDrawable, GLXContext))
            X11_GL_GetProcAddress(_this, "glXMakeCurrent");
    _this->gl_data->glXSwapBuffers =
        (void (*)(Display *, GLXDrawable))
            X11_GL_GetProcAddress(_this, "glXSwapBuffers");
    _this->gl_data->glXQueryDrawable =
        (void (*)(Display *, GLXDrawable, int, unsigned int *))
            X11_GL_GetProcAddress(_this, "glXQueryDrawable");

    if (!_this->gl_data->glXQueryExtension ||
        !_this->gl_data->glXChooseVisual ||
        !_this->gl_data->glXCreateContext ||
        !_this->gl_data->glXDestroyContext ||
        !_this->gl_data->glXMakeCurrent ||
        !_this->gl_data->glXSwapBuffers) {
        return SDL_SetError("Could not retrieve OpenGL functions");
    }

    display = _this->internal->display;
    if (!_this->gl_data->glXQueryExtension(display, &_this->gl_data->errorBase, &_this->gl_data->eventBase)) {
        return SDL_SetError("GLX is not supported");
    }

    _this->gl_data->swap_interval_tear_behavior = SDL_SWAPINTERVALTEAR_UNTESTED;

    // Initialize extensions
    /* See lengthy comment about the inc/dec in
       ../windows/SDL_windowsopengl.c. */
    ++_this->gl_config.driver_loaded;
    X11_GL_InitExtensions(_this);
    --_this->gl_config.driver_loaded;

    /* If we need a GL ES context and there's no
     * GLX_EXT_create_context_es2_profile extension, switch over to X11_GLES functions
     */
    if (((_this->gl_config.profile_mask == SDL_GL_CONTEXT_PROFILE_ES) ||
         SDL_GetHintBoolean(SDL_HINT_VIDEO_FORCE_EGL, false)) &&
        X11_GL_UseEGL(_this)) {
#ifdef SDL_VIDEO_OPENGL_EGL
        X11_GL_UnloadLibrary(_this);
        _this->GL_LoadLibrary = X11_GLES_LoadLibrary;
        _this->GL_GetProcAddress = X11_GLES_GetProcAddress;
        _this->GL_UnloadLibrary = X11_GLES_UnloadLibrary;
        _this->GL_CreateContext = X11_GLES_CreateContext;
        _this->GL_MakeCurrent = X11_GLES_MakeCurrent;
        _this->GL_SetSwapInterval = X11_GLES_SetSwapInterval;
        _this->GL_GetSwapInterval = X11_GLES_GetSwapInterval;
        _this->GL_SwapWindow = X11_GLES_SwapWindow;
        _this->GL_DestroyContext = X11_GLES_DestroyContext;
        return X11_GLES_LoadLibrary(_this, NULL);
#else
        return SDL_SetError("SDL not configured with EGL support");
#endif
    }

    return true;
}

SDL_FunctionPointer X11_GL_GetProcAddress(SDL_VideoDevice *_this, const char *proc)
{
    if (_this->gl_data->glXGetProcAddress) {
        return _this->gl_data->glXGetProcAddress((const GLubyte *)proc);
    }
    return GL_LoadFunction(_this->gl_config.dll_handle, proc);
}

void X11_GL_UnloadLibrary(SDL_VideoDevice *_this)
{
    /* Don't actually unload the library, since it may have registered
     * X11 shutdown hooks, per the notes at:
     * http://dri.sourceforge.net/doc/DRIuserguide.html
     */
#if 0
    GL_UnloadObject(_this->gl_config.dll_handle);
    _this->gl_config.dll_handle = NULL;
#endif

    // Free OpenGL memory
    SDL_free(_this->gl_data);
    _this->gl_data = NULL;
}

static bool HasExtension(const char *extension, const char *extensions)
{
    const char *start;
    const char *where, *terminator;

    if (!extensions) {
        return false;
    }

    // Extension names should not have spaces.
    where = SDL_strchr(extension, ' ');
    if (where || *extension == '\0') {
        return false;
    }

    /* It takes a bit of care to be fool-proof about parsing the
     * OpenGL extensions string. Don't be fooled by sub-strings,
     * etc. */

    start = extensions;

    for (;;) {
        where = SDL_strstr(start, extension);
        if (!where) {
            break;
        }

        terminator = where + SDL_strlen(extension);
        if (where == start || *(where - 1) == ' ') {
            if (*terminator == ' ' || *terminator == '\0') {
                return true;
            }
        }

        start = terminator;
    }
    return false;
}

static void X11_GL_InitExtensions(SDL_VideoDevice *_this)
{
    Display *display = _this->internal->display;
    const int screen = DefaultScreen(display);
    XVisualInfo *vinfo = NULL;
    Window w = 0;
    GLXContext prev_ctx = 0;
    GLXDrawable prev_drawable = 0;
    GLXContext context = 0;
    const char *(*glXQueryExtensionsStringFunc)(Display *, int);
    const char *extensions;

    vinfo = X11_GL_GetVisual(_this, display, screen, false);
    if (vinfo) {
        GLXContext (*glXGetCurrentContextFunc)(void) =
            (GLXContext(*)(void))
                X11_GL_GetProcAddress(_this, "glXGetCurrentContext");

        GLXDrawable (*glXGetCurrentDrawableFunc)(void) =
            (GLXDrawable(*)(void))
                X11_GL_GetProcAddress(_this, "glXGetCurrentDrawable");

        if (glXGetCurrentContextFunc && glXGetCurrentDrawableFunc) {
            XSetWindowAttributes xattr;
            prev_ctx = glXGetCurrentContextFunc();
            prev_drawable = glXGetCurrentDrawableFunc();

            xattr.background_pixel = 0;
            xattr.border_pixel = 0;
            xattr.colormap =
                X11_XCreateColormap(display, RootWindow(display, screen),
                                    vinfo->visual, AllocNone);
            w = X11_XCreateWindow(display, RootWindow(display, screen), 0, 0,
                                  32, 32, 0, vinfo->depth, InputOutput, vinfo->visual,
                                  (CWBackPixel | CWBorderPixel | CWColormap), &xattr);

            context = _this->gl_data->glXCreateContext(display, vinfo,
                                                       NULL, True);
            if (context) {
                _this->gl_data->glXMakeCurrent(display, w, context);
            }
        }

        X11_XFree(vinfo);
    }

    glXQueryExtensionsStringFunc =
        (const char *(*)(Display *, int))X11_GL_GetProcAddress(_this,
                                                               "glXQueryExtensionsString");
    if (glXQueryExtensionsStringFunc) {
        extensions = glXQueryExtensionsStringFunc(display, screen);
    } else {
        extensions = NULL;
    }

    // Check for GLX_EXT_swap_control(_tear)
    _this->gl_data->HAS_GLX_EXT_swap_control_tear = false;
    if (HasExtension("GLX_EXT_swap_control", extensions)) {
        _this->gl_data->glXSwapIntervalEXT =
            (void (*)(Display *, GLXDrawable, int))
                X11_GL_GetProcAddress(_this, "glXSwapIntervalEXT");
        if (HasExtension("GLX_EXT_swap_control_tear", extensions)) {
            _this->gl_data->HAS_GLX_EXT_swap_control_tear = true;
        }
    }

    // Check for GLX_MESA_swap_control
    if (HasExtension("GLX_MESA_swap_control", extensions)) {
        _this->gl_data->glXSwapIntervalMESA =
            (int (*)(int))X11_GL_GetProcAddress(_this, "glXSwapIntervalMESA");
        _this->gl_data->glXGetSwapIntervalMESA =
            (int (*)(void))X11_GL_GetProcAddress(_this,
                                                 "glXGetSwapIntervalMESA");
    }

    // Check for GLX_SGI_swap_control
    if (HasExtension("GLX_SGI_swap_control", extensions)) {
        _this->gl_data->glXSwapIntervalSGI =
            (int (*)(int))X11_GL_GetProcAddress(_this, "glXSwapIntervalSGI");
    }

    // Check for GLX_ARB_create_context
    if (HasExtension("GLX_ARB_create_context", extensions)) {
        _this->gl_data->glXCreateContextAttribsARB =
            (GLXContext(*)(Display *, GLXFBConfig, GLXContext, Bool, const int *))
                X11_GL_GetProcAddress(_this, "glXCreateContextAttribsARB");
        _this->gl_data->glXChooseFBConfig =
            (GLXFBConfig * (*)(Display *, int, const int *, int *))
                X11_GL_GetProcAddress(_this, "glXChooseFBConfig");
        _this->gl_data->glXGetVisualFromFBConfig =
            (XVisualInfo * (*)(Display *, GLXFBConfig))
                X11_GL_GetProcAddress(_this, "glXGetVisualFromFBConfig");
    }

    // Check for GLX_EXT_visual_rating
    if (HasExtension("GLX_EXT_visual_rating", extensions)) {
        _this->gl_data->HAS_GLX_EXT_visual_rating = true;
    }

    // Check for GLX_EXT_visual_info
    if (HasExtension("GLX_EXT_visual_info", extensions)) {
        _this->gl_data->HAS_GLX_EXT_visual_info = true;
    }

    // Check for GLX_EXT_create_context_es2_profile
    if (HasExtension("GLX_EXT_create_context_es2_profile", extensions)) {
        // this wants to call glGetString(), so it needs a context.
        // !!! FIXME: it would be nice not to make a context here though!
        if (context) {
            SDL_GL_DeduceMaxSupportedESProfile(
                &_this->gl_data->es_profile_max_supported_version.major,
                &_this->gl_data->es_profile_max_supported_version.minor);
        }
    }

    // Check for GLX_ARB_context_flush_control
    if (HasExtension("GLX_ARB_context_flush_control", extensions)) {
        _this->gl_data->HAS_GLX_ARB_context_flush_control = true;
    }

    // Check for GLX_ARB_create_context_robustness
    if (HasExtension("GLX_ARB_create_context_robustness", extensions)) {
        _this->gl_data->HAS_GLX_ARB_create_context_robustness = true;
    }

    // Check for GLX_ARB_create_context_no_error
    if (HasExtension("GLX_ARB_create_context_no_error", extensions)) {
        _this->gl_data->HAS_GLX_ARB_create_context_no_error = true;
    }

    if (context) {
        _this->gl_data->glXMakeCurrent(display, None, NULL);
        _this->gl_data->glXDestroyContext(display, context);
        if (prev_ctx && prev_drawable) {
            _this->gl_data->glXMakeCurrent(display, prev_drawable, prev_ctx);
        }
    }

    if (w) {
        X11_XDestroyWindow(display, w);
    }
    X11_PumpEvents(_this);
}

/* glXChooseVisual and glXChooseFBConfig have some small differences in
 * the attribute encoding, it can be chosen with the for_FBConfig parameter.
 * Some targets fail if you use GLX_X_VISUAL_TYPE_EXT/GLX_DIRECT_COLOR_EXT,
 *  so it gets specified last if used and is pointed to by *_pvistypeattr.
 *  In case of failure, if that pointer is not NULL, set that pointer to None
 *  and try again.
 */
static int X11_GL_GetAttributes(SDL_VideoDevice *_this, Display *display, int screen, int *attribs, int size, Bool for_FBConfig, int **_pvistypeattr, bool transparent)
{
    int i = 0;
    const int MAX_ATTRIBUTES = 64;
    int *pvistypeattr = NULL;

    // assert buffer is large enough to hold all SDL attributes.
    SDL_assert(size >= MAX_ATTRIBUTES);

    // Setup our GLX attributes according to the gl_config.
    if (for_FBConfig) {
        attribs[i++] = GLX_RENDER_TYPE;
        if (_this->gl_config.floatbuffers) {
            attribs[i++] = GLX_RGBA_FLOAT_BIT_ARB;
        } else {
            attribs[i++] = GLX_RGBA_BIT;
        }
    } else {
        attribs[i++] = GLX_RGBA;
    }
    attribs[i++] = GLX_RED_SIZE;
    attribs[i++] = _this->gl_config.red_size;
    attribs[i++] = GLX_GREEN_SIZE;
    attribs[i++] = _this->gl_config.green_size;
    attribs[i++] = GLX_BLUE_SIZE;
    attribs[i++] = _this->gl_config.blue_size;

    if (_this->gl_config.alpha_size) {
        attribs[i++] = GLX_ALPHA_SIZE;
        attribs[i++] = _this->gl_config.alpha_size;
    }

    if (_this->gl_config.double_buffer) {
        attribs[i++] = GLX_DOUBLEBUFFER;
        if (for_FBConfig) {
            attribs[i++] = True;
        }
    }

    attribs[i++] = GLX_DEPTH_SIZE;
    attribs[i++] = _this->gl_config.depth_size;

    if (_this->gl_config.stencil_size) {
        attribs[i++] = GLX_STENCIL_SIZE;
        attribs[i++] = _this->gl_config.stencil_size;
    }

    if (_this->gl_config.accum_red_size) {
        attribs[i++] = GLX_ACCUM_RED_SIZE;
        attribs[i++] = _this->gl_config.accum_red_size;
    }

    if (_this->gl_config.accum_green_size) {
        attribs[i++] = GLX_ACCUM_GREEN_SIZE;
        attribs[i++] = _this->gl_config.accum_green_size;
    }

    if (_this->gl_config.accum_blue_size) {
        attribs[i++] = GLX_ACCUM_BLUE_SIZE;
        attribs[i++] = _this->gl_config.accum_blue_size;
    }

    if (_this->gl_config.accum_alpha_size) {
        attribs[i++] = GLX_ACCUM_ALPHA_SIZE;
        attribs[i++] = _this->gl_config.accum_alpha_size;
    }

    if (_this->gl_config.stereo) {
        attribs[i++] = GLX_STEREO;
        if (for_FBConfig) {
            attribs[i++] = True;
        }
    }

    if (_this->gl_config.multisamplebuffers) {
        attribs[i++] = GLX_SAMPLE_BUFFERS_ARB;
        attribs[i++] = _this->gl_config.multisamplebuffers;
    }

    if (_this->gl_config.multisamplesamples) {
        attribs[i++] = GLX_SAMPLES_ARB;
        attribs[i++] = _this->gl_config.multisamplesamples;
    }

    if (_this->gl_config.floatbuffers) {
        attribs[i++] = GLX_RENDER_TYPE;
        attribs[i++] = GLX_RGBA_FLOAT_TYPE_ARB;
    }

    if (_this->gl_config.framebuffer_srgb_capable) {
        attribs[i++] = GLX_FRAMEBUFFER_SRGB_CAPABLE_ARB;
        attribs[i++] = True; // always needed, for_FBConfig or not!
    }

    if (_this->gl_config.accelerated >= 0 &&
        _this->gl_data->HAS_GLX_EXT_visual_rating) {
        attribs[i++] = GLX_VISUAL_CAVEAT_EXT;
        attribs[i++] = _this->gl_config.accelerated ? GLX_NONE_EXT : GLX_SLOW_VISUAL_EXT;
    }

    // Un-wanted when we request a transparent buffer
    if (!transparent) {
        /* If we're supposed to use DirectColor visuals, and we've got the
           EXT_visual_info extension, then add GLX_X_VISUAL_TYPE_EXT. */
        if (X11_UseDirectColorVisuals() && _this->gl_data->HAS_GLX_EXT_visual_info) {
            pvistypeattr = &attribs[i];
            attribs[i++] = GLX_X_VISUAL_TYPE_EXT;
            attribs[i++] = GLX_DIRECT_COLOR_EXT;
        }
    }

    attribs[i++] = None;

    SDL_assert(i <= MAX_ATTRIBUTES);

    if (_pvistypeattr) {
        *_pvistypeattr = pvistypeattr;
    }

    return i;
}

//get the first transparent Visual
static XVisualInfo* X11_GL_GetTransparentVisualInfo(Display *display, int screen)
{
    XVisualInfo* visualinfo = NULL;
    XVisualInfo vi_in;
    int out_count = 0;

    vi_in.screen = screen;
    visualinfo = X11_XGetVisualInfo(display, VisualScreenMask, &vi_in, &out_count);
    if (visualinfo != NULL) {
        int i = 0;
        for (i = 0; i < out_count; i++) {
            XVisualInfo* v = &visualinfo[i];
            Uint32 format = X11_GetPixelFormatFromVisualInfo(display, v);
            if (SDL_ISPIXELFORMAT_ALPHA(format)) {
                vi_in.screen = screen;
                vi_in.visualid = v->visualid;
                X11_XFree(visualinfo);
                visualinfo = X11_XGetVisualInfo(display, VisualScreenMask | VisualIDMask, &vi_in, &out_count);
                break;
            }
        }
    }
    return visualinfo;
}

XVisualInfo *X11_GL_GetVisual(SDL_VideoDevice *_this, Display *display, int screen, bool transparent)
{
    // 64 seems nice.
    int attribs[64];
    XVisualInfo *vinfo = NULL;
    int *pvistypeattr = NULL;

    if (!_this->gl_data) {
        // The OpenGL library wasn't loaded, SDL_GetError() should have info
        return NULL;
    }

    if (_this->gl_data->glXChooseFBConfig &&
        _this->gl_data->glXGetVisualFromFBConfig) {
        GLXFBConfig *framebuffer_config = NULL;
        int fbcount = 0;

        X11_GL_GetAttributes(_this, display, screen, attribs, 64, true, &pvistypeattr, transparent);
        framebuffer_config = _this->gl_data->glXChooseFBConfig(display, screen, attribs, &fbcount);
        if (!framebuffer_config && (pvistypeattr != NULL)) {
            *pvistypeattr = None;
            framebuffer_config = _this->gl_data->glXChooseFBConfig(display, screen, attribs, &fbcount);
        }

        if (transparent) {
            // Return the first transparent Visual
            int i;
            for (i = 0; i < fbcount; i++) {
                Uint32 format;
                vinfo = _this->gl_data->glXGetVisualFromFBConfig(display, framebuffer_config[i]);
                format = X11_GetPixelFormatFromVisualInfo(display, vinfo);
                if (SDL_ISPIXELFORMAT_ALPHA(format)) { // found!
                    X11_XFree(framebuffer_config);
                    framebuffer_config = NULL;
                    break;
                }
                X11_XFree(vinfo);
                vinfo = NULL;
            }
        }

        if (framebuffer_config) {
            vinfo = _this->gl_data->glXGetVisualFromFBConfig(display, framebuffer_config[0]);
        }

        X11_XFree(framebuffer_config);
    }

    if (!vinfo) {
        X11_GL_GetAttributes(_this, display, screen, attribs, 64, false, &pvistypeattr, transparent);
        vinfo = _this->gl_data->glXChooseVisual(display, screen, attribs);

        if (!vinfo && (pvistypeattr != NULL)) {
            *pvistypeattr = None;
            vinfo = _this->gl_data->glXChooseVisual(display, screen, attribs);
        }
    }

    if (transparent && vinfo) {
        Uint32 format = X11_GetPixelFormatFromVisualInfo(display, vinfo);
        if (!SDL_ISPIXELFORMAT_ALPHA(format)) {
            // not transparent!
            XVisualInfo* visualinfo = X11_GL_GetTransparentVisualInfo(display, screen);
            if (visualinfo != NULL) {
                X11_XFree(vinfo);
                vinfo = visualinfo;
            }
        }
    }

    if (!vinfo) {
        SDL_SetError("Couldn't find matching GLX visual");
    }
    return vinfo;
}

static int (*handler)(Display *, XErrorEvent *) = NULL;
static const char *errorHandlerOperation = NULL;
static int errorBase = 0;
static int errorCode = 0;
static int X11_GL_ErrorHandler(Display *d, XErrorEvent *e)
{
    char *x11_error = NULL;
    char x11_error_locale[256];

    errorCode = e->error_code;
    if (X11_XGetErrorText(d, errorCode, x11_error_locale, sizeof(x11_error_locale)) == Success) {
        x11_error = SDL_iconv_string("UTF-8", "", x11_error_locale, SDL_strlen(x11_error_locale) + 1);
    }

    if (x11_error) {
        SDL_SetError("Could not %s: %s", errorHandlerOperation, x11_error);
        SDL_free(x11_error);
    } else {
        SDL_SetError("Could not %s: %i (Base %i)", errorHandlerOperation, errorCode, errorBase);
    }

    return 0;
}

bool X11_GL_UseEGL(SDL_VideoDevice *_this)
{
    SDL_assert(_this->gl_data != NULL);
    if (SDL_GetHintBoolean(SDL_HINT_VIDEO_FORCE_EGL, false)) {
        // use of EGL has been requested, even for desktop GL
        return true;
    }

    SDL_assert(_this->gl_config.profile_mask == SDL_GL_CONTEXT_PROFILE_ES);
    return (SDL_GetHintBoolean(SDL_HINT_OPENGL_ES_DRIVER, false) || _this->gl_config.major_version == 1 // No GLX extension for OpenGL ES 1.x profiles.
            || _this->gl_config.major_version > _this->gl_data->es_profile_max_supported_version.major || (_this->gl_config.major_version == _this->gl_data->es_profile_max_supported_version.major && _this->gl_config.minor_version > _this->gl_data->es_profile_max_supported_version.minor));
}

SDL_GLContext X11_GL_CreateContext(SDL_VideoDevice *_this, SDL_Window *window)
{
    SDL_WindowData *data = window->internal;
    Display *display = data->videodata->display;
    int screen = SDL_GetDisplayDriverDataForWindow(window)->screen;
    XWindowAttributes xattr;
    XVisualInfo v, *vinfo;
    int n;
    SDL_GLContext context = NULL;
    GLXContext share_context;
    const int transparent = (window->flags & SDL_WINDOW_TRANSPARENT) ? true : false;

    if (_this->gl_config.share_with_current_context) {
        share_context = (GLXContext)SDL_GL_GetCurrentContext();
    } else {
        share_context = NULL;
    }

    // We do this to create a clean separation between X and GLX errors.
    X11_XSync(display, False);
    errorHandlerOperation = "create GL context";
    errorBase = _this->gl_data->errorBase;
    errorCode = Success;
    handler = X11_XSetErrorHandler(X11_GL_ErrorHandler);
    X11_XGetWindowAttributes(display, data->xwindow, &xattr);
    v.screen = screen;
    v.visualid = X11_XVisualIDFromVisual(xattr.visual);
    vinfo = X11_XGetVisualInfo(display, VisualScreenMask | VisualIDMask, &v, &n);
    if (vinfo) {
        if (_this->gl_config.major_version < 3 &&
            _this->gl_config.profile_mask == 0 &&
            _this->gl_config.flags == 0 && !transparent) {
            // Create legacy context
            context =
                (SDL_GLContext)_this->gl_data->glXCreateContext(display, vinfo, share_context, True);
        } else {
            // max 14 attributes plus terminator
            int attribs[15] = {
                GLX_CONTEXT_MAJOR_VERSION_ARB,
                _this->gl_config.major_version,
                GLX_CONTEXT_MINOR_VERSION_ARB,
                _this->gl_config.minor_version,
                0
            };
            int iattr = 4;

            // SDL profile bits match GLX profile bits
            if (_this->gl_config.profile_mask != 0) {
                attribs[iattr++] = GLX_CONTEXT_PROFILE_MASK_ARB;
                attribs[iattr++] = _this->gl_config.profile_mask;
            }

            // SDL flags match GLX flags
            if (_this->gl_config.flags != 0) {
                attribs[iattr++] = GLX_CONTEXT_FLAGS_ARB;
                attribs[iattr++] = _this->gl_config.flags;
            }

            // only set if glx extension is available and not the default setting
            if ((_this->gl_data->HAS_GLX_ARB_context_flush_control) && (_this->gl_config.release_behavior == 0)) {
                attribs[iattr++] = GLX_CONTEXT_RELEASE_BEHAVIOR_ARB;
                attribs[iattr++] =
                    _this->gl_config.release_behavior ? GLX_CONTEXT_RELEASE_BEHAVIOR_FLUSH_ARB : GLX_CONTEXT_RELEASE_BEHAVIOR_NONE_ARB;
            }

            // only set if glx extension is available and not the default setting
            if ((_this->gl_data->HAS_GLX_ARB_create_context_robustness) && (_this->gl_config.reset_notification != 0)) {
                attribs[iattr++] = GLX_CONTEXT_RESET_NOTIFICATION_STRATEGY_ARB;
                attribs[iattr++] =
                    _this->gl_config.reset_notification ? GLX_LOSE_CONTEXT_ON_RESET_ARB : GLX_NO_RESET_NOTIFICATION_ARB;
            }

            // only set if glx extension is available and not the default setting
            if ((_this->gl_data->HAS_GLX_ARB_create_context_no_error) && (_this->gl_config.no_error != 0)) {
                attribs[iattr++] = GLX_CONTEXT_OPENGL_NO_ERROR_ARB;
                attribs[iattr++] = _this->gl_config.no_error;
            }

            attribs[iattr++] = 0;

            // Get a pointer to the context creation function for GL 3.0
            if (!_this->gl_data->glXCreateContextAttribsARB) {
                SDL_SetError("OpenGL 3.0 and later are not supported by this system");
            } else {
                int glxAttribs[64];

                // Create a GL 3.x context
                GLXFBConfig *framebuffer_config = NULL;
                int fbcount = 0;
                int *pvistypeattr = NULL;

                X11_GL_GetAttributes(_this, display, screen, glxAttribs, 64, true, &pvistypeattr, transparent);

                if (_this->gl_data->glXChooseFBConfig) {
                    framebuffer_config = _this->gl_data->glXChooseFBConfig(display,
                                                                           DefaultScreen(display), glxAttribs,
                                                                           &fbcount);

                    if (!framebuffer_config && (pvistypeattr != NULL)) {
                        *pvistypeattr = None;
                        framebuffer_config = _this->gl_data->glXChooseFBConfig(display,
                                                                               DefaultScreen(display), glxAttribs,
                                                                               &fbcount);
                    }

                    if (transparent && (framebuffer_config != NULL)) {
                        int i;
                        for (i = 0; i < fbcount; i++) {
                            XVisualInfo* vinfo_temp = _this->gl_data->glXGetVisualFromFBConfig(display, framebuffer_config[i]);
                            if ( vinfo_temp != NULL) {
                                Uint32 format = X11_GetPixelFormatFromVisualInfo(display, vinfo_temp);
                                if (SDL_ISPIXELFORMAT_ALPHA(format)) {
                                    // found!
                                    context = (SDL_GLContext)_this->gl_data->glXCreateContextAttribsARB(display,
                                                                                                        framebuffer_config[i],
                                                                                                        share_context, True, attribs);
                                    X11_XFree(framebuffer_config);
                                    framebuffer_config = NULL;
                                    X11_XFree(vinfo_temp);
                                    break;
                                }
                                X11_XFree(vinfo_temp);
                            }
                        }
                    }
                    if (framebuffer_config) {
                        context = (SDL_GLContext)_this->gl_data->glXCreateContextAttribsARB(display,
                                                                             framebuffer_config[0],
                                                                             share_context, True, attribs);
                        X11_XFree(framebuffer_config);
                    }
                }
            }
        }
        X11_XFree(vinfo);
    }
    X11_XSync(display, False);
    X11_XSetErrorHandler(handler);

    if (!context) {
        if (errorCode == Success) {
            SDL_SetError("Could not create GL context");
        }
        return NULL;
    }

    if (!X11_GL_MakeCurrent(_this, window, context)) {
        X11_GL_DestroyContext(_this, context);
        return NULL;
    }

    return context;
}

bool X11_GL_MakeCurrent(SDL_VideoDevice *_this, SDL_Window *window, SDL_GLContext context)
{
    Display *display = _this->internal->display;
    Window drawable =
        (context ? window->internal->xwindow : None);
    GLXContext glx_context = (GLXContext)context;
    int rc;

    if (!_this->gl_data) {
        return SDL_SetError("OpenGL not initialized");
    }

    // We do this to create a clean separation between X and GLX errors.
    X11_XSync(display, False);
    errorHandlerOperation = "make GL context current";
    errorBase = _this->gl_data->errorBase;
    errorCode = Success;
    handler = X11_XSetErrorHandler(X11_GL_ErrorHandler);
    rc = _this->gl_data->glXMakeCurrent(display, drawable, glx_context);
    X11_XSetErrorHandler(handler);

    if (errorCode != Success) { // uhoh, an X error was thrown!
        return false;              // the error handler called SDL_SetError() already.
    } else if (!rc) {           // glXMakeCurrent() failed without throwing an X error
        return SDL_SetError("Unable to make GL context current");
    }

    return true;
}

/*
   0 is a valid argument to glXSwapInterval(MESA|EXT) and setting it to 0
   will undo the effect of a previous call with a value that is greater
   than zero (or at least that is what the docs say). OTOH, 0 is an invalid
   argument to glXSwapIntervalSGI and it returns an error if you call it
   with 0 as an argument.
*/

static int swapinterval = 0;
bool X11_GL_SetSwapInterval(SDL_VideoDevice *_this, int interval)
{
    bool result = false;

    if ((interval < 0) && (!_this->gl_data->HAS_GLX_EXT_swap_control_tear)) {
        return SDL_SetError("Negative swap interval unsupported in this GL");
    } else if (_this->gl_data->glXSwapIntervalEXT) {
        Display *display = _this->internal->display;
        const SDL_WindowData *windowdata = SDL_GL_GetCurrentWindow()->internal;

        Window drawable = windowdata->xwindow;

        /*
         * This is a workaround for a bug in NVIDIA drivers. Bug has been reported
         * and will be fixed in a future release (probably 319.xx).
         *
         * There's a bug where glXSetSwapIntervalEXT ignores updates because
         * it has the wrong value cached. To work around it, we just run a no-op
         * update to the current value.
         */
        int currentInterval = 0;
        X11_GL_GetSwapInterval(_this, &currentInterval);
        _this->gl_data->glXSwapIntervalEXT(display, drawable, currentInterval);
        _this->gl_data->glXSwapIntervalEXT(display, drawable, interval);
        result = true;
        swapinterval = interval;
    } else if (_this->gl_data->glXSwapIntervalMESA) {
        const int rc = _this->gl_data->glXSwapIntervalMESA(interval);
        if (rc == 0) {
            swapinterval = interval;
            result = true;
        } else {
            result = SDL_SetError("glXSwapIntervalMESA failed");
        }
    } else if (_this->gl_data->glXSwapIntervalSGI) {
        const int rc = _this->gl_data->glXSwapIntervalSGI(interval);
        if (rc == 0) {
            swapinterval = interval;
            result = true;
        } else {
            result = SDL_SetError("glXSwapIntervalSGI failed");
        }
    } else {
        return SDL_Unsupported();
    }
    return result;
}

static SDL_GLSwapIntervalTearBehavior CheckSwapIntervalTearBehavior(SDL_VideoDevice *_this, Window drawable, unsigned int current_val, unsigned int current_allow_late)
{
    /* Mesa and Nvidia interpret GLX_EXT_swap_control_tear differently, as of this writing, so
        figure out which behavior we have.
       Technical details: https://github.com/libsdl-org/SDL/issues/8004#issuecomment-1819603282 */
    if (_this->gl_data->swap_interval_tear_behavior == SDL_SWAPINTERVALTEAR_UNTESTED) {
        if (!_this->gl_data->HAS_GLX_EXT_swap_control_tear) {
            _this->gl_data->swap_interval_tear_behavior = SDL_SWAPINTERVALTEAR_UNKNOWN;
        } else {
            Display *display = _this->internal->display;
            unsigned int allow_late_swap_tearing = 22;
            int original_val = (int) current_val;

            /*
             * This is a workaround for a bug in NVIDIA drivers. Bug has been reported
             * and will be fixed in a future release (probably 319.xx).
             *
             * There's a bug where glXSetSwapIntervalEXT ignores updates because
             * it has the wrong value cached. To work around it, we just run a no-op
             * update to the current value.
             */
            _this->gl_data->glXSwapIntervalEXT(display, drawable, current_val);

            // set it to no swap interval and see how it affects GLX_LATE_SWAPS_TEAR_EXT...
            _this->gl_data->glXSwapIntervalEXT(display, drawable, 0);
            _this->gl_data->glXQueryDrawable(display, drawable, GLX_LATE_SWAPS_TEAR_EXT, &allow_late_swap_tearing);

            if (allow_late_swap_tearing == 0) { // GLX_LATE_SWAPS_TEAR_EXT says whether late swapping is currently in use
                _this->gl_data->swap_interval_tear_behavior = SDL_SWAPINTERVALTEAR_NVIDIA;
                if (current_allow_late) {
                    original_val = -original_val;
                }
            } else if (allow_late_swap_tearing == 1) {  // GLX_LATE_SWAPS_TEAR_EXT says whether the Drawable can use late swapping at all
                _this->gl_data->swap_interval_tear_behavior = SDL_SWAPINTERVALTEAR_MESA;
            } else {  // unexpected outcome!
                _this->gl_data->swap_interval_tear_behavior = SDL_SWAPINTERVALTEAR_UNKNOWN;
            }

            // set us back to what it was originally...
            _this->gl_data->glXSwapIntervalEXT(display, drawable, original_val);
        }
    }

    return _this->gl_data->swap_interval_tear_behavior;
}


bool X11_GL_GetSwapInterval(SDL_VideoDevice *_this, int *interval)
{
    if (_this->gl_data->glXSwapIntervalEXT) {
        Display *display = _this->internal->display;
        const SDL_WindowData *windowdata = SDL_GL_GetCurrentWindow()->internal;
        Window drawable = windowdata->xwindow;
        unsigned int allow_late_swap_tearing = 0;
        unsigned int val = 0;

        if (_this->gl_data->HAS_GLX_EXT_swap_control_tear) {
            allow_late_swap_tearing = 22;  // set this to nonsense.
            _this->gl_data->glXQueryDrawable(display, drawable,
                                             GLX_LATE_SWAPS_TEAR_EXT,
                                             &allow_late_swap_tearing);
        }

        _this->gl_data->glXQueryDrawable(display, drawable,
                                         GLX_SWAP_INTERVAL_EXT, &val);

        *interval = (int)val;

        switch (CheckSwapIntervalTearBehavior(_this, drawable, val, allow_late_swap_tearing)) {
            case SDL_SWAPINTERVALTEAR_MESA:
                *interval = (int)val;  // unsigned int cast to signed that generates negative value if necessary.
                break;

            case SDL_SWAPINTERVALTEAR_NVIDIA:
            default:
                if ((allow_late_swap_tearing) && (val > 0)) {
                    *interval = -((int)val);
                }
                break;
        }

        return true;
    } else if (_this->gl_data->glXGetSwapIntervalMESA) {
        int val = _this->gl_data->glXGetSwapIntervalMESA();
        if (val == GLX_BAD_CONTEXT) {
            return SDL_SetError("GLX_BAD_CONTEXT");
        }
        *interval = val;
        return true;
    } else {
        *interval = swapinterval;
        return true;
    }
}

bool X11_GL_SwapWindow(SDL_VideoDevice *_this, SDL_Window *window)
{
    SDL_WindowData *data = window->internal;
    Display *display = data->videodata->display;

    _this->gl_data->glXSwapBuffers(display, data->xwindow);

#ifdef SDL_VIDEO_DRIVER_X11_XSYNC
    X11_HandlePresent(data->window);
#endif /* SDL_VIDEO_DRIVER_X11_XSYNC */

    return true;
}

bool X11_GL_DestroyContext(SDL_VideoDevice *_this, SDL_GLContext context)
{
    Display *display = _this->internal->display;
    GLXContext glx_context = (GLXContext)context;

    if (!_this->gl_data) {
        return true;
    }
    _this->gl_data->glXDestroyContext(display, glx_context);
    X11_XSync(display, False);
    return true;
}

#endif // SDL_VIDEO_OPENGL_GLX

#endif // SDL_VIDEO_DRIVER_X11
