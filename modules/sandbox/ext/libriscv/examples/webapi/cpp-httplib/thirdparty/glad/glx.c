/**
 * SPDX-License-Identifier: (WTFPL OR CC0-1.0) AND Apache-2.0
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <glad/glx.h>

#ifndef GLAD_IMPL_UTIL_C_
#define GLAD_IMPL_UTIL_C_

#ifdef _MSC_VER
#define GLAD_IMPL_UTIL_SSCANF sscanf_s
#else
#define GLAD_IMPL_UTIL_SSCANF sscanf
#endif

#endif /* GLAD_IMPL_UTIL_C_ */

#ifdef __cplusplus
extern "C" {
#endif



int GLAD_GLX_VERSION_1_0 = 0;
int GLAD_GLX_VERSION_1_1 = 0;
int GLAD_GLX_VERSION_1_2 = 0;
int GLAD_GLX_VERSION_1_3 = 0;
int GLAD_GLX_VERSION_1_4 = 0;
int GLAD_GLX_ARB_create_context = 0;
int GLAD_GLX_ARB_create_context_profile = 0;
int GLAD_GLX_ARB_get_proc_address = 0;
int GLAD_GLX_EXT_swap_control = 0;
int GLAD_GLX_MESA_swap_control = 0;
int GLAD_GLX_SGI_swap_control = 0;



PFNGLXCHOOSEFBCONFIGPROC glad_glXChooseFBConfig = NULL;
PFNGLXCHOOSEVISUALPROC glad_glXChooseVisual = NULL;
PFNGLXCOPYCONTEXTPROC glad_glXCopyContext = NULL;
PFNGLXCREATECONTEXTPROC glad_glXCreateContext = NULL;
PFNGLXCREATECONTEXTATTRIBSARBPROC glad_glXCreateContextAttribsARB = NULL;
PFNGLXCREATEGLXPIXMAPPROC glad_glXCreateGLXPixmap = NULL;
PFNGLXCREATENEWCONTEXTPROC glad_glXCreateNewContext = NULL;
PFNGLXCREATEPBUFFERPROC glad_glXCreatePbuffer = NULL;
PFNGLXCREATEPIXMAPPROC glad_glXCreatePixmap = NULL;
PFNGLXCREATEWINDOWPROC glad_glXCreateWindow = NULL;
PFNGLXDESTROYCONTEXTPROC glad_glXDestroyContext = NULL;
PFNGLXDESTROYGLXPIXMAPPROC glad_glXDestroyGLXPixmap = NULL;
PFNGLXDESTROYPBUFFERPROC glad_glXDestroyPbuffer = NULL;
PFNGLXDESTROYPIXMAPPROC glad_glXDestroyPixmap = NULL;
PFNGLXDESTROYWINDOWPROC glad_glXDestroyWindow = NULL;
PFNGLXGETCLIENTSTRINGPROC glad_glXGetClientString = NULL;
PFNGLXGETCONFIGPROC glad_glXGetConfig = NULL;
PFNGLXGETCURRENTCONTEXTPROC glad_glXGetCurrentContext = NULL;
PFNGLXGETCURRENTDISPLAYPROC glad_glXGetCurrentDisplay = NULL;
PFNGLXGETCURRENTDRAWABLEPROC glad_glXGetCurrentDrawable = NULL;
PFNGLXGETCURRENTREADDRAWABLEPROC glad_glXGetCurrentReadDrawable = NULL;
PFNGLXGETFBCONFIGATTRIBPROC glad_glXGetFBConfigAttrib = NULL;
PFNGLXGETFBCONFIGSPROC glad_glXGetFBConfigs = NULL;
PFNGLXGETPROCADDRESSPROC glad_glXGetProcAddress = NULL;
PFNGLXGETPROCADDRESSARBPROC glad_glXGetProcAddressARB = NULL;
PFNGLXGETSELECTEDEVENTPROC glad_glXGetSelectedEvent = NULL;
PFNGLXGETSWAPINTERVALMESAPROC glad_glXGetSwapIntervalMESA = NULL;
PFNGLXGETVISUALFROMFBCONFIGPROC glad_glXGetVisualFromFBConfig = NULL;
PFNGLXISDIRECTPROC glad_glXIsDirect = NULL;
PFNGLXMAKECONTEXTCURRENTPROC glad_glXMakeContextCurrent = NULL;
PFNGLXMAKECURRENTPROC glad_glXMakeCurrent = NULL;
PFNGLXQUERYCONTEXTPROC glad_glXQueryContext = NULL;
PFNGLXQUERYDRAWABLEPROC glad_glXQueryDrawable = NULL;
PFNGLXQUERYEXTENSIONPROC glad_glXQueryExtension = NULL;
PFNGLXQUERYEXTENSIONSSTRINGPROC glad_glXQueryExtensionsString = NULL;
PFNGLXQUERYSERVERSTRINGPROC glad_glXQueryServerString = NULL;
PFNGLXQUERYVERSIONPROC glad_glXQueryVersion = NULL;
PFNGLXSELECTEVENTPROC glad_glXSelectEvent = NULL;
PFNGLXSWAPBUFFERSPROC glad_glXSwapBuffers = NULL;
PFNGLXSWAPINTERVALEXTPROC glad_glXSwapIntervalEXT = NULL;
PFNGLXSWAPINTERVALMESAPROC glad_glXSwapIntervalMESA = NULL;
PFNGLXSWAPINTERVALSGIPROC glad_glXSwapIntervalSGI = NULL;
PFNGLXUSEXFONTPROC glad_glXUseXFont = NULL;
PFNGLXWAITGLPROC glad_glXWaitGL = NULL;
PFNGLXWAITXPROC glad_glXWaitX = NULL;


static void glad_glx_load_GLX_VERSION_1_0( GLADuserptrloadfunc load, void* userptr) {
    if(!GLAD_GLX_VERSION_1_0) return;
    glad_glXChooseVisual = (PFNGLXCHOOSEVISUALPROC) load(userptr, "glXChooseVisual");
    glad_glXCopyContext = (PFNGLXCOPYCONTEXTPROC) load(userptr, "glXCopyContext");
    glad_glXCreateContext = (PFNGLXCREATECONTEXTPROC) load(userptr, "glXCreateContext");
    glad_glXCreateGLXPixmap = (PFNGLXCREATEGLXPIXMAPPROC) load(userptr, "glXCreateGLXPixmap");
    glad_glXDestroyContext = (PFNGLXDESTROYCONTEXTPROC) load(userptr, "glXDestroyContext");
    glad_glXDestroyGLXPixmap = (PFNGLXDESTROYGLXPIXMAPPROC) load(userptr, "glXDestroyGLXPixmap");
    glad_glXGetConfig = (PFNGLXGETCONFIGPROC) load(userptr, "glXGetConfig");
    glad_glXGetCurrentContext = (PFNGLXGETCURRENTCONTEXTPROC) load(userptr, "glXGetCurrentContext");
    glad_glXGetCurrentDrawable = (PFNGLXGETCURRENTDRAWABLEPROC) load(userptr, "glXGetCurrentDrawable");
    glad_glXIsDirect = (PFNGLXISDIRECTPROC) load(userptr, "glXIsDirect");
    glad_glXMakeCurrent = (PFNGLXMAKECURRENTPROC) load(userptr, "glXMakeCurrent");
    glad_glXQueryExtension = (PFNGLXQUERYEXTENSIONPROC) load(userptr, "glXQueryExtension");
    glad_glXQueryVersion = (PFNGLXQUERYVERSIONPROC) load(userptr, "glXQueryVersion");
    glad_glXSwapBuffers = (PFNGLXSWAPBUFFERSPROC) load(userptr, "glXSwapBuffers");
    glad_glXUseXFont = (PFNGLXUSEXFONTPROC) load(userptr, "glXUseXFont");
    glad_glXWaitGL = (PFNGLXWAITGLPROC) load(userptr, "glXWaitGL");
    glad_glXWaitX = (PFNGLXWAITXPROC) load(userptr, "glXWaitX");
}
static void glad_glx_load_GLX_VERSION_1_1( GLADuserptrloadfunc load, void* userptr) {
    if(!GLAD_GLX_VERSION_1_1) return;
    glad_glXGetClientString = (PFNGLXGETCLIENTSTRINGPROC) load(userptr, "glXGetClientString");
    glad_glXQueryExtensionsString = (PFNGLXQUERYEXTENSIONSSTRINGPROC) load(userptr, "glXQueryExtensionsString");
    glad_glXQueryServerString = (PFNGLXQUERYSERVERSTRINGPROC) load(userptr, "glXQueryServerString");
}
static void glad_glx_load_GLX_VERSION_1_2( GLADuserptrloadfunc load, void* userptr) {
    if(!GLAD_GLX_VERSION_1_2) return;
    glad_glXGetCurrentDisplay = (PFNGLXGETCURRENTDISPLAYPROC) load(userptr, "glXGetCurrentDisplay");
}
static void glad_glx_load_GLX_VERSION_1_3( GLADuserptrloadfunc load, void* userptr) {
    if(!GLAD_GLX_VERSION_1_3) return;
    glad_glXChooseFBConfig = (PFNGLXCHOOSEFBCONFIGPROC) load(userptr, "glXChooseFBConfig");
    glad_glXCreateNewContext = (PFNGLXCREATENEWCONTEXTPROC) load(userptr, "glXCreateNewContext");
    glad_glXCreatePbuffer = (PFNGLXCREATEPBUFFERPROC) load(userptr, "glXCreatePbuffer");
    glad_glXCreatePixmap = (PFNGLXCREATEPIXMAPPROC) load(userptr, "glXCreatePixmap");
    glad_glXCreateWindow = (PFNGLXCREATEWINDOWPROC) load(userptr, "glXCreateWindow");
    glad_glXDestroyPbuffer = (PFNGLXDESTROYPBUFFERPROC) load(userptr, "glXDestroyPbuffer");
    glad_glXDestroyPixmap = (PFNGLXDESTROYPIXMAPPROC) load(userptr, "glXDestroyPixmap");
    glad_glXDestroyWindow = (PFNGLXDESTROYWINDOWPROC) load(userptr, "glXDestroyWindow");
    glad_glXGetCurrentReadDrawable = (PFNGLXGETCURRENTREADDRAWABLEPROC) load(userptr, "glXGetCurrentReadDrawable");
    glad_glXGetFBConfigAttrib = (PFNGLXGETFBCONFIGATTRIBPROC) load(userptr, "glXGetFBConfigAttrib");
    glad_glXGetFBConfigs = (PFNGLXGETFBCONFIGSPROC) load(userptr, "glXGetFBConfigs");
    glad_glXGetSelectedEvent = (PFNGLXGETSELECTEDEVENTPROC) load(userptr, "glXGetSelectedEvent");
    glad_glXGetVisualFromFBConfig = (PFNGLXGETVISUALFROMFBCONFIGPROC) load(userptr, "glXGetVisualFromFBConfig");
    glad_glXMakeContextCurrent = (PFNGLXMAKECONTEXTCURRENTPROC) load(userptr, "glXMakeContextCurrent");
    glad_glXQueryContext = (PFNGLXQUERYCONTEXTPROC) load(userptr, "glXQueryContext");
    glad_glXQueryDrawable = (PFNGLXQUERYDRAWABLEPROC) load(userptr, "glXQueryDrawable");
    glad_glXSelectEvent = (PFNGLXSELECTEVENTPROC) load(userptr, "glXSelectEvent");
}
static void glad_glx_load_GLX_VERSION_1_4( GLADuserptrloadfunc load, void* userptr) {
    if(!GLAD_GLX_VERSION_1_4) return;
    glad_glXGetProcAddress = (PFNGLXGETPROCADDRESSPROC) load(userptr, "glXGetProcAddress");
}
static void glad_glx_load_GLX_ARB_create_context( GLADuserptrloadfunc load, void* userptr) {
    if(!GLAD_GLX_ARB_create_context) return;
    glad_glXCreateContextAttribsARB = (PFNGLXCREATECONTEXTATTRIBSARBPROC) load(userptr, "glXCreateContextAttribsARB");
}
static void glad_glx_load_GLX_ARB_get_proc_address( GLADuserptrloadfunc load, void* userptr) {
    if(!GLAD_GLX_ARB_get_proc_address) return;
    glad_glXGetProcAddressARB = (PFNGLXGETPROCADDRESSARBPROC) load(userptr, "glXGetProcAddressARB");
}
static void glad_glx_load_GLX_EXT_swap_control( GLADuserptrloadfunc load, void* userptr) {
    if(!GLAD_GLX_EXT_swap_control) return;
    glad_glXSwapIntervalEXT = (PFNGLXSWAPINTERVALEXTPROC) load(userptr, "glXSwapIntervalEXT");
}
static void glad_glx_load_GLX_MESA_swap_control( GLADuserptrloadfunc load, void* userptr) {
    if(!GLAD_GLX_MESA_swap_control) return;
    glad_glXGetSwapIntervalMESA = (PFNGLXGETSWAPINTERVALMESAPROC) load(userptr, "glXGetSwapIntervalMESA");
    glad_glXSwapIntervalMESA = (PFNGLXSWAPINTERVALMESAPROC) load(userptr, "glXSwapIntervalMESA");
}
static void glad_glx_load_GLX_SGI_swap_control( GLADuserptrloadfunc load, void* userptr) {
    if(!GLAD_GLX_SGI_swap_control) return;
    glad_glXSwapIntervalSGI = (PFNGLXSWAPINTERVALSGIPROC) load(userptr, "glXSwapIntervalSGI");
}



static int glad_glx_has_extension(Display *display, int screen, const char *ext) {
#ifndef GLX_VERSION_1_1
    GLAD_UNUSED(display);
    GLAD_UNUSED(screen);
    GLAD_UNUSED(ext);
#else
    const char *terminator;
    const char *loc;
    const char *extensions;

    if (glXQueryExtensionsString == NULL) {
        return 0;
    }

    extensions = glXQueryExtensionsString(display, screen);

    if(extensions == NULL || ext == NULL) {
        return 0;
    }

    while(1) {
        loc = strstr(extensions, ext);
        if(loc == NULL)
            break;

        terminator = loc + strlen(ext);
        if((loc == extensions || *(loc - 1) == ' ') &&
            (*terminator == ' ' || *terminator == '\0')) {
            return 1;
        }
        extensions = terminator;
    }
#endif

    return 0;
}

static GLADapiproc glad_glx_get_proc_from_userptr(void *userptr, const char* name) {
    return (GLAD_GNUC_EXTENSION (GLADapiproc (*)(const char *name)) userptr)(name);
}

static int glad_glx_find_extensions(Display *display, int screen) {
    GLAD_GLX_ARB_create_context = glad_glx_has_extension(display, screen, "GLX_ARB_create_context");
    GLAD_GLX_ARB_create_context_profile = glad_glx_has_extension(display, screen, "GLX_ARB_create_context_profile");
    GLAD_GLX_ARB_get_proc_address = glad_glx_has_extension(display, screen, "GLX_ARB_get_proc_address");
    GLAD_GLX_EXT_swap_control = glad_glx_has_extension(display, screen, "GLX_EXT_swap_control");
    GLAD_GLX_MESA_swap_control = glad_glx_has_extension(display, screen, "GLX_MESA_swap_control");
    GLAD_GLX_SGI_swap_control = glad_glx_has_extension(display, screen, "GLX_SGI_swap_control");
    return 1;
}

static int glad_glx_find_core_glx(Display **display, int *screen) {
    int major = 0, minor = 0;
    if(*display == NULL) {
#ifdef GLAD_GLX_NO_X11
        GLAD_UNUSED(screen);
        return 0;
#else
        *display = XOpenDisplay(0);
        if (*display == NULL) {
            return 0;
        }
        *screen = XScreenNumberOfScreen(XDefaultScreenOfDisplay(*display));
#endif
    }
    glXQueryVersion(*display, &major, &minor);
    GLAD_GLX_VERSION_1_0 = (major == 1 && minor >= 0) || major > 1;
    GLAD_GLX_VERSION_1_1 = (major == 1 && minor >= 1) || major > 1;
    GLAD_GLX_VERSION_1_2 = (major == 1 && minor >= 2) || major > 1;
    GLAD_GLX_VERSION_1_3 = (major == 1 && minor >= 3) || major > 1;
    GLAD_GLX_VERSION_1_4 = (major == 1 && minor >= 4) || major > 1;
    return GLAD_MAKE_VERSION(major, minor);
}

int gladLoadGLXUserPtr(Display *display, int screen, GLADuserptrloadfunc load, void *userptr) {
    int version;
    glXQueryVersion = (PFNGLXQUERYVERSIONPROC) load(userptr, "glXQueryVersion");
    if(glXQueryVersion == NULL) return 0;
    version = glad_glx_find_core_glx(&display, &screen);

    glad_glx_load_GLX_VERSION_1_0(load, userptr);
    glad_glx_load_GLX_VERSION_1_1(load, userptr);
    glad_glx_load_GLX_VERSION_1_2(load, userptr);
    glad_glx_load_GLX_VERSION_1_3(load, userptr);
    glad_glx_load_GLX_VERSION_1_4(load, userptr);

    if (!glad_glx_find_extensions(display, screen)) return 0;
    glad_glx_load_GLX_ARB_create_context(load, userptr);
    glad_glx_load_GLX_ARB_get_proc_address(load, userptr);
    glad_glx_load_GLX_EXT_swap_control(load, userptr);
    glad_glx_load_GLX_MESA_swap_control(load, userptr);
    glad_glx_load_GLX_SGI_swap_control(load, userptr);


    return version;
}

int gladLoadGLX(Display *display, int screen, GLADloadfunc load) {
    return gladLoadGLXUserPtr(display, screen, glad_glx_get_proc_from_userptr, GLAD_GNUC_EXTENSION (void*) load);
}

 

#ifdef GLAD_GLX

#ifndef GLAD_LOADER_LIBRARY_C_
#define GLAD_LOADER_LIBRARY_C_

#include <stddef.h>
#include <stdlib.h>

#if GLAD_PLATFORM_WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif


static void* glad_get_dlopen_handle(const char *lib_names[], int length) {
    void *handle = NULL;
    int i;

    for (i = 0; i < length; ++i) {
#if GLAD_PLATFORM_WIN32
  #if GLAD_PLATFORM_UWP
        size_t buffer_size = (strlen(lib_names[i]) + 1) * sizeof(WCHAR);
        LPWSTR buffer = (LPWSTR) malloc(buffer_size);
        if (buffer != NULL) {
            int ret = MultiByteToWideChar(CP_ACP, 0, lib_names[i], -1, buffer, buffer_size);
            if (ret != 0) {
                handle = (void*) LoadPackagedLibrary(buffer, 0);
            }
            free((void*) buffer);
        }
  #else
        handle = (void*) LoadLibraryA(lib_names[i]);
  #endif
#else
        handle = dlopen(lib_names[i], RTLD_LAZY | RTLD_LOCAL);
#endif
        if (handle != NULL) {
            return handle;
        }
    }

    return NULL;
}

static void glad_close_dlopen_handle(void* handle) {
    if (handle != NULL) {
#if GLAD_PLATFORM_WIN32
        FreeLibrary((HMODULE) handle);
#else
        dlclose(handle);
#endif
    }
}

static GLADapiproc glad_dlsym_handle(void* handle, const char *name) {
    if (handle == NULL) {
        return NULL;
    }

#if GLAD_PLATFORM_WIN32
    return (GLADapiproc) GetProcAddress((HMODULE) handle, name);
#else
    return GLAD_GNUC_EXTENSION (GLADapiproc) dlsym(handle, name);
#endif
}

#endif /* GLAD_LOADER_LIBRARY_C_ */

typedef void* (GLAD_API_PTR *GLADglxprocaddrfunc)(const char*);

static GLADapiproc glad_glx_get_proc(void *userptr, const char *name) {
    return GLAD_GNUC_EXTENSION ((GLADapiproc (*)(const char *name)) userptr)(name);
}

static void* _glx_handle;

static void* glad_glx_dlopen_handle(void) {
    static const char *NAMES[] = {
#if defined __CYGWIN__
        "libGL-1.so",
#endif
        "libGL.so.1",
        "libGL.so"
    };

    if (_glx_handle == NULL) {
        _glx_handle = glad_get_dlopen_handle(NAMES, sizeof(NAMES) / sizeof(NAMES[0]));
    }

    return _glx_handle;
}

int gladLoaderLoadGLX(Display *display, int screen) {
    int version = 0;
    void *handle = NULL;
    int did_load = 0;
    GLADglxprocaddrfunc loader;

    did_load = _glx_handle == NULL;
    handle = glad_glx_dlopen_handle();
    if (handle != NULL) {
        loader = (GLADglxprocaddrfunc) glad_dlsym_handle(handle, "glXGetProcAddressARB");
        if (loader != NULL) {
            version = gladLoadGLXUserPtr(display, screen, glad_glx_get_proc, GLAD_GNUC_EXTENSION (void*) loader);
        }

        if (!version && did_load) {
            gladLoaderUnloadGLX();
        }
    }

    return version;
}


void gladLoaderUnloadGLX() {
    if (_glx_handle != NULL) {
        glad_close_dlopen_handle(_glx_handle);
        _glx_handle = NULL;
    }
}

#endif /* GLAD_GLX */

#ifdef __cplusplus
}
#endif
