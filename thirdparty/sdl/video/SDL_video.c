/*
  Simple DirectMedia Layer
  Copyright (C) 1997-2025 Sam Lantinga <slouken@libsdl.org>

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

// The high-level video driver subsystem

#include "SDL_sysvideo.h"
#include "SDL_clipboard_c.h"
#include "SDL_egl_c.h"
#include "SDL_surface_c.h"
#include "SDL_pixels_c.h"
#include "SDL_rect_c.h"
#include "SDL_video_c.h"
#include "../events/SDL_events_c.h"
#include "../SDL_hints_c.h"
#include "../SDL_properties_c.h"
#include "../timer/SDL_timer_c.h"
#include "../camera/SDL_camera_c.h"
#include "../render/SDL_sysrender.h"
#include "../main/SDL_main_callbacks.h"

#ifdef SDL_VIDEO_OPENGL
#include <SDL3/SDL_opengl.h>
#endif // SDL_VIDEO_OPENGL

#if defined(SDL_VIDEO_OPENGL_ES) && !defined(SDL_VIDEO_OPENGL)
#include <SDL3/SDL_opengles.h>
#endif // SDL_VIDEO_OPENGL_ES && !SDL_VIDEO_OPENGL

// GL and GLES2 headers conflict on Linux 32 bits
#if defined(SDL_VIDEO_OPENGL_ES2) && !defined(SDL_VIDEO_OPENGL)
#include <SDL3/SDL_opengles2.h>
#endif // SDL_VIDEO_OPENGL_ES2 && !SDL_VIDEO_OPENGL

// GL_CONTEXT_RELEASE_BEHAVIOR and GL_CONTEXT_RELEASE_BEHAVIOR_KHR have the same number.
#ifndef GL_CONTEXT_RELEASE_BEHAVIOR
#define GL_CONTEXT_RELEASE_BEHAVIOR 0x82FB
#endif

// GL_CONTEXT_RELEASE_BEHAVIOR_FLUSH and GL_CONTEXT_RELEASE_BEHAVIOR_FLUSH_KHR have the same number.
#ifndef GL_CONTEXT_RELEASE_BEHAVIOR_FLUSH
#define GL_CONTEXT_RELEASE_BEHAVIOR_FLUSH 0x82FC
#endif

#ifdef SDL_PLATFORM_EMSCRIPTEN
#include <emscripten.h>
#endif

#ifdef SDL_PLATFORM_3DS
#include <3ds.h>
#endif

#ifdef SDL_PLATFORM_LINUX
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

// Available video drivers
static VideoBootStrap *bootstrap[] = {
#ifdef SDL_VIDEO_DRIVER_PRIVATE
    &PRIVATE_bootstrap,
#endif
#ifdef SDL_VIDEO_DRIVER_COCOA
    &COCOA_bootstrap,
#endif
#ifdef SDL_VIDEO_DRIVER_X11
#ifdef SDL_VIDEO_DRIVER_WAYLAND
    &Wayland_preferred_bootstrap,
#endif
    &X11_bootstrap,
#endif
#ifdef SDL_VIDEO_DRIVER_WAYLAND
    &Wayland_bootstrap,
#endif
#ifdef SDL_VIDEO_DRIVER_VIVANTE
    &VIVANTE_bootstrap,
#endif
#ifdef SDL_VIDEO_DRIVER_WINDOWS
    &WINDOWS_bootstrap,
#endif
#ifdef SDL_VIDEO_DRIVER_HAIKU
    &HAIKU_bootstrap,
#endif
#ifdef SDL_VIDEO_DRIVER_UIKIT
    &UIKIT_bootstrap,
#endif
#ifdef SDL_VIDEO_DRIVER_ANDROID
    &Android_bootstrap,
#endif
#ifdef SDL_VIDEO_DRIVER_PS2
    &PS2_bootstrap,
#endif
#ifdef SDL_VIDEO_DRIVER_PSP
    &PSP_bootstrap,
#endif
#ifdef SDL_VIDEO_DRIVER_VITA
    &VITA_bootstrap,
#endif
#ifdef SDL_VIDEO_DRIVER_N3DS
    &N3DS_bootstrap,
#endif
#ifdef SDL_VIDEO_DRIVER_KMSDRM
    &KMSDRM_bootstrap,
#endif
#ifdef SDL_VIDEO_DRIVER_RISCOS
    &RISCOS_bootstrap,
#endif
#ifdef SDL_VIDEO_DRIVER_RPI
    &RPI_bootstrap,
#endif
#ifdef SDL_VIDEO_DRIVER_EMSCRIPTEN
    &Emscripten_bootstrap,
#endif
#ifdef SDL_VIDEO_DRIVER_QNX
    &QNX_bootstrap,
#endif
#ifdef SDL_VIDEO_DRIVER_OFFSCREEN
    &OFFSCREEN_bootstrap,
#endif
#ifdef SDL_VIDEO_DRIVER_DUMMY
    &DUMMY_bootstrap,
#ifdef SDL_INPUT_LINUXEV
    &DUMMY_evdev_bootstrap,
#endif
#endif
#ifdef SDL_VIDEO_DRIVER_OPENVR
    &OPENVR_bootstrap,
#endif
    NULL
};

#define CHECK_WINDOW_MAGIC(window, result)                              \
    if (!_this) {                                                       \
        SDL_UninitializedVideo();                                       \
        return result;                                                  \
    }                                                                   \
    if (!SDL_ObjectValid(window, SDL_OBJECT_TYPE_WINDOW)) {             \
        SDL_SetError("Invalid window");                                 \
        return result;                                                  \
    }

#define CHECK_DISPLAY_MAGIC(display, result)                            \
    if (!display) {                                                     \
        return result;                                                  \
    }                                                                   \

#define CHECK_WINDOW_NOT_POPUP(window, result)                          \
    if (SDL_WINDOW_IS_POPUP(window)) {                                  \
        SDL_SetError("Operation invalid on popup windows");             \
        return result;                                                  \
    }

#if defined(SDL_PLATFORM_MACOS) && defined(SDL_VIDEO_DRIVER_COCOA)
// Support for macOS fullscreen spaces
extern bool Cocoa_IsWindowInFullscreenSpace(SDL_Window *window);
extern bool Cocoa_SetWindowFullscreenSpace(SDL_Window *window, bool state, bool blocking);
#endif

#ifdef SDL_VIDEO_DRIVER_UIKIT
extern void SDL_UpdateLifecycleObserver(void);
#endif

static void SDL_CheckWindowDisplayChanged(SDL_Window *window);
static void SDL_CheckWindowDisplayScaleChanged(SDL_Window *window);
static void SDL_CheckWindowSafeAreaChanged(SDL_Window *window);

// Convenience functions for reading driver flags
static bool SDL_ModeSwitchingEmulated(SDL_VideoDevice *_this)
{
    if (_this->device_caps & VIDEO_DEVICE_CAPS_MODE_SWITCHING_EMULATED) {
        return true;
    }
    return false;
}

static bool SDL_SendsFullscreenDimensions(SDL_VideoDevice *_this)
{
    return !!(_this->device_caps & VIDEO_DEVICE_CAPS_SENDS_FULLSCREEN_DIMENSIONS);
}

static bool IsFullscreenOnly(SDL_VideoDevice *_this)
{
    return !!(_this->device_caps & VIDEO_DEVICE_CAPS_FULLSCREEN_ONLY);
}

static bool SDL_SendsDisplayChanges(SDL_VideoDevice *_this)
{
    return !!(_this->device_caps & VIDEO_DEVICE_CAPS_SENDS_DISPLAY_CHANGES);
}

static bool SDL_DisableMouseWarpOnFullscreenTransitions(SDL_VideoDevice *_this)
{
    return !!(_this->device_caps & VIDEO_DEVICE_CAPS_DISABLE_MOUSE_WARP_ON_FULLSCREEN_TRANSITIONS);
}

static bool SDL_DriverSendsHDRChanges(SDL_VideoDevice *_this)
{
    return !!(_this->device_caps & VIDEO_DEVICE_CAPS_SENDS_HDR_CHANGES);
}

// Hint to treat all window ops as synchronous
static bool syncHint;

static void SDL_SyncHintWatcher(void *userdata, const char *name, const char *oldValue, const char *newValue)
{
    syncHint = SDL_GetStringBoolean(newValue, false);
}

static void SDL_SyncIfRequired(SDL_Window *window)
{
    if (syncHint) {
        SDL_SyncWindow(window);
    }
}

static void SDL_UpdateWindowHierarchy(SDL_Window *window, SDL_Window *parent)
{
    // Unlink the window from the existing parent.
    if (window->parent) {
        if (window->next_sibling) {
            window->next_sibling->prev_sibling = window->prev_sibling;
        }
        if (window->prev_sibling) {
            window->prev_sibling->next_sibling = window->next_sibling;
        } else {
            window->parent->first_child = window->next_sibling;
        }

        window->parent = NULL;
    }

    if (parent) {
        window->parent = parent;

        window->next_sibling = parent->first_child;
        if (parent->first_child) {
            parent->first_child->prev_sibling = window;
        }
        parent->first_child = window;
    }
}

// Support for framebuffer emulation using an accelerated renderer

#define SDL_PROP_WINDOW_TEXTUREDATA_POINTER "SDL.internal.window.texturedata"

typedef struct
{
    SDL_Renderer *renderer;
    SDL_Texture *texture;
    void *pixels;
    int pitch;
    int bytes_per_pixel;
} SDL_WindowTextureData;

static Uint32 SDL_DefaultGraphicsBackends(SDL_VideoDevice *_this)
{
#if (defined(SDL_VIDEO_OPENGL) && defined(SDL_PLATFORM_MACOS)) || (defined(SDL_PLATFORM_IOS) && !TARGET_OS_MACCATALYST)
    if (_this->GL_CreateContext) {
        return SDL_WINDOW_OPENGL;
    }
#endif
#if defined(SDL_VIDEO_METAL) && (TARGET_OS_MACCATALYST || defined(SDL_PLATFORM_MACOS) || defined(SDL_PLATFORM_IOS))
    if (_this->Metal_CreateView) {
        return SDL_WINDOW_METAL;
    }
#endif
#if defined(SDL_VIDEO_OPENGL) && defined(SDL_VIDEO_DRIVER_OPENVR)
    if (SDL_strcmp(_this->name, "openvr") == 0) {
        return SDL_WINDOW_OPENGL;
    }
#endif
    return 0;
}

static void SDLCALL SDL_CleanupWindowTextureData(void *userdata, void *value)
{
    SDL_WindowTextureData *data = (SDL_WindowTextureData *)value;

    if (data->texture) {
        SDL_DestroyTexture(data->texture);
    }
    if (data->renderer) {
        SDL_DestroyRenderer(data->renderer);
    }
    SDL_free(data->pixels);
    SDL_free(data);
}

static bool SDL_CreateWindowTexture(SDL_VideoDevice *_this, SDL_Window *window, SDL_PixelFormat *format, void **pixels, int *pitch)
{
    SDL_PropertiesID props = SDL_GetWindowProperties(window);
    SDL_WindowTextureData *data = (SDL_WindowTextureData *)SDL_GetPointerProperty(props, SDL_PROP_WINDOW_TEXTUREDATA_POINTER, NULL);
    const bool transparent = (window->flags & SDL_WINDOW_TRANSPARENT) ? true : false;
    int i;
    int w, h;
    const SDL_PixelFormat *texture_formats;

    SDL_GetWindowSizeInPixels(window, &w, &h);

    if (!data) {
        SDL_Renderer *renderer = NULL;
        const char *render_driver = NULL;

        // See if there's a render driver being requested
        const char *hint = SDL_GetHint(SDL_HINT_FRAMEBUFFER_ACCELERATION);
        if (hint && *hint != '0' && *hint != '1' &&
            SDL_strcasecmp(hint, "true") != 0 &&
            SDL_strcasecmp(hint, "false") != 0 &&
            SDL_strcasecmp(hint, SDL_SOFTWARE_RENDERER) != 0) {
            render_driver = hint;
        }

        if (!render_driver) {
            render_driver = SDL_GetHint(SDL_HINT_RENDER_DRIVER);
        }
        if (render_driver && SDL_strcasecmp(render_driver, SDL_SOFTWARE_RENDERER) == 0) {
            render_driver = NULL;
        }

        char *render_driver_copy = NULL;
        if (render_driver && *render_driver) {
            render_driver_copy = SDL_strdup(render_driver);
            render_driver = render_driver_copy;
            if (render_driver_copy) {  // turn any "software" requests into "xxxxxxxx" so we don't end up in infinite recursion.
                char *prev = render_driver_copy;
                char *ptr = prev;
                while ((ptr = SDL_strchr(ptr, ',')) != NULL) {
                    *ptr = '\0';
                    const bool is_sw = (SDL_strcasecmp(prev, SDL_SOFTWARE_RENDERER) == 0);
                    *ptr = ',';
                    if (is_sw) {
                        SDL_memset(prev, 'x', SDL_strlen(SDL_SOFTWARE_RENDERER));
                        ptr = prev;
                    } else {
                        ptr++;
                        prev = ptr;
                    }
                }

                if (SDL_strcasecmp(prev, SDL_SOFTWARE_RENDERER) == 0) {
                    SDL_memset(prev, 'x', SDL_strlen(SDL_SOFTWARE_RENDERER));
                }
            }
        }

        // Check to see if there's a specific driver requested
        if (render_driver) {
            renderer = SDL_CreateRenderer(window, render_driver);
            SDL_free(render_driver_copy);
            if (!renderer) {
                // The error for this specific renderer has already been set
                return false;
            }
        } else {
            SDL_assert(render_driver_copy == NULL);
            const int total = SDL_GetNumRenderDrivers();
            for (i = 0; i < total; ++i) {
                const char *name = SDL_GetRenderDriver(i);
                if (name && SDL_strcmp(name, SDL_SOFTWARE_RENDERER) != 0) {
                    renderer = SDL_CreateRenderer(window, name);
                    if (renderer) {
                        break; // this will work.
                    }
                }
            }
            if (!renderer) {
                return SDL_SetError("No hardware accelerated renderers available");
            }
        }

        SDL_assert(renderer != NULL); // should have explicitly checked this above.

        // Create the data after we successfully create the renderer (bug #1116)
        data = (SDL_WindowTextureData *)SDL_calloc(1, sizeof(*data));
        if (!data) {
            SDL_DestroyRenderer(renderer);
            return false;
        }
        if (!SDL_SetPointerPropertyWithCleanup(props, SDL_PROP_WINDOW_TEXTUREDATA_POINTER, data, SDL_CleanupWindowTextureData, NULL)) {
            SDL_DestroyRenderer(renderer);
            return false;
        }

        data->renderer = renderer;
    }

    texture_formats = (const SDL_PixelFormat *)SDL_GetPointerProperty(SDL_GetRendererProperties(data->renderer), SDL_PROP_RENDERER_TEXTURE_FORMATS_POINTER, NULL);
    if (!texture_formats) {
        return false;
    }

    // Free any old texture and pixel data
    if (data->texture) {
        SDL_DestroyTexture(data->texture);
        data->texture = NULL;
    }
    SDL_free(data->pixels);
    data->pixels = NULL;

    // Find the first format with or without an alpha channel
    *format = texture_formats[0];

    for (i = 0; texture_formats[i] != SDL_PIXELFORMAT_UNKNOWN; ++i) {
        SDL_PixelFormat texture_format = texture_formats[i];
        if (!SDL_ISPIXELFORMAT_FOURCC(texture_format) &&
            !SDL_ISPIXELFORMAT_10BIT(texture_format) &&
            !SDL_ISPIXELFORMAT_FLOAT(texture_format) &&
            transparent == SDL_ISPIXELFORMAT_ALPHA(texture_format)) {
            *format = texture_format;
            break;
        }
    }

    data->texture = SDL_CreateTexture(data->renderer, *format,
                                      SDL_TEXTUREACCESS_STREAMING,
                                      w, h);
    if (!data->texture) {
        // codechecker_false_positive [Malloc] Static analyzer doesn't realize allocated `data` is saved to SDL_PROP_WINDOW_TEXTUREDATA_POINTER and not leaked here.
        return false; // NOLINT(clang-analyzer-unix.Malloc)
    }

    // Create framebuffer data
    data->bytes_per_pixel = SDL_BYTESPERPIXEL(*format);
    data->pitch = (((w * data->bytes_per_pixel) + 3) & ~3);

    {
        // Make static analysis happy about potential SDL_malloc(0) calls.
        const size_t allocsize = (size_t)h * data->pitch;
        data->pixels = SDL_malloc((allocsize > 0) ? allocsize : 1);
        if (!data->pixels) {
            return false;
        }
    }

    *pixels = data->pixels;
    *pitch = data->pitch;

    // Make sure we're not double-scaling the viewport
    SDL_SetRenderViewport(data->renderer, NULL);

    return true;
}

bool SDL_SetWindowTextureVSync(SDL_VideoDevice *_this, SDL_Window *window, int vsync)
{
    SDL_WindowTextureData *data;

    data = (SDL_WindowTextureData *)SDL_GetPointerProperty(SDL_GetWindowProperties(window), SDL_PROP_WINDOW_TEXTUREDATA_POINTER, NULL);
    if (!data) {
        return false;
    }
    if (!data->renderer) {
        return false;
    }
    return SDL_SetRenderVSync(data->renderer, vsync);
}

static bool SDL_GetWindowTextureVSync(SDL_VideoDevice *_this, SDL_Window *window, int *vsync)
{
    SDL_WindowTextureData *data;

    data = (SDL_WindowTextureData *)SDL_GetPointerProperty(SDL_GetWindowProperties(window), SDL_PROP_WINDOW_TEXTUREDATA_POINTER, NULL);
    if (!data) {
        return false;
    }
    if (!data->renderer) {
        return false;
    }
    return SDL_GetRenderVSync(data->renderer, vsync);
}

static bool SDL_UpdateWindowTexture(SDL_VideoDevice *_this, SDL_Window *window, const SDL_Rect *rects, int numrects)
{
    SDL_WindowTextureData *data;
    SDL_Rect rect;
    void *src;
    int w, h;

    SDL_GetWindowSizeInPixels(window, &w, &h);

    data = (SDL_WindowTextureData *)SDL_GetPointerProperty(SDL_GetWindowProperties(window), SDL_PROP_WINDOW_TEXTUREDATA_POINTER, NULL);
    if (!data || !data->texture) {
        return SDL_SetError("No window texture data");
    }

    // Update a single rect that contains subrects for best DMA performance
    if (SDL_GetSpanEnclosingRect(w, h, numrects, rects, &rect)) {
        src = (void *)((Uint8 *)data->pixels +
                       rect.y * data->pitch +
                       rect.x * data->bytes_per_pixel);
        if (!SDL_UpdateTexture(data->texture, &rect, src, data->pitch)) {
            return false;
        }

        if (!SDL_RenderTexture(data->renderer, data->texture, NULL, NULL)) {
            return false;
        }

        SDL_RenderPresent(data->renderer);
    }
    return true;
}

static void SDL_DestroyWindowTexture(SDL_VideoDevice *_this, SDL_Window *window)
{
    SDL_ClearProperty(SDL_GetWindowProperties(window), SDL_PROP_WINDOW_TEXTUREDATA_POINTER);
}

static SDL_VideoDevice *_this = NULL;
static SDL_AtomicInt SDL_messagebox_count;

static int SDLCALL cmpmodes(const void *A, const void *B)
{
    const SDL_DisplayMode *a = (const SDL_DisplayMode *)A;
    const SDL_DisplayMode *b = (const SDL_DisplayMode *)B;
    int a_refresh_rate = (int)(a->refresh_rate * 100);
    int b_refresh_rate = (int)(b->refresh_rate * 100);
    int a_pixel_density = (int)(a->pixel_density * 100);
    int b_pixel_density = (int)(b->pixel_density * 100);

    if (a->w != b->w) {
        return b->w - a->w;
    } else if (a->h != b->h) {
        return b->h - a->h;
    } else if (SDL_BITSPERPIXEL(a->format) != SDL_BITSPERPIXEL(b->format)) {
        return SDL_BITSPERPIXEL(b->format) - SDL_BITSPERPIXEL(a->format);
    } else if (SDL_PIXELLAYOUT(a->format) != SDL_PIXELLAYOUT(b->format)) {
        return SDL_PIXELLAYOUT(b->format) - SDL_PIXELLAYOUT(a->format);
    } else if (a_refresh_rate != b_refresh_rate) {
        return b_refresh_rate - a_refresh_rate;
    } else if (a_pixel_density != b_pixel_density) {
        return a_pixel_density - b_pixel_density;
    }
    return 0;
}

bool SDL_UninitializedVideo(void)
{
    return SDL_SetError("Video subsystem has not been initialized");
}

// Deduplicated list of video bootstrap drivers.
static const VideoBootStrap *deduped_bootstrap[SDL_arraysize(bootstrap) - 1];

int SDL_GetNumVideoDrivers(void)
{
    static int num_drivers = -1;

    if (num_drivers >= 0) {
        return num_drivers;
    }

    num_drivers = 0;

    // Build a list of unique video drivers.
    for (int i = 0; bootstrap[i] != NULL; ++i) {
        bool duplicate = false;
        for (int j = 0; j < i; ++j) {
            if (SDL_strcmp(bootstrap[i]->name, bootstrap[j]->name) == 0) {
                duplicate = true;
                break;
            }
        }

        if (!duplicate) {
            deduped_bootstrap[num_drivers++] = bootstrap[i];
        }
    }

    return num_drivers;
}

const char *SDL_GetVideoDriver(int index)
{
    if (index >= 0 && index < SDL_GetNumVideoDrivers()) {
        return deduped_bootstrap[index]->name;
    }
    SDL_InvalidParamError("index");
    return NULL;
}

/*
 * Initialize the video and event subsystems -- determine native pixel format
 */
bool SDL_VideoInit(const char *driver_name)
{
    SDL_VideoDevice *video;
    bool init_events = false;
    bool init_keyboard = false;
    bool init_mouse = false;
    bool init_touch = false;
    bool init_pen = false;
    int i = 0;

    // Check to make sure we don't overwrite '_this'
    if (_this) {
        SDL_VideoQuit();
    }

    SDL_InitTicks();

    // Start the event loop
    if (!SDL_InitSubSystem(SDL_INIT_EVENTS)) {
        goto pre_driver_error;
    }
    init_events = true;
    if (!SDL_InitKeyboard()) {
        goto pre_driver_error;
    }
    init_keyboard = true;
    if (!SDL_PreInitMouse()) {
        goto pre_driver_error;
    }
    init_mouse = true;
    if (!SDL_InitTouch()) {
        goto pre_driver_error;
    }
    init_touch = true;
    if (!SDL_InitPen()) {
        goto pre_driver_error;
    }
    init_pen = true;

    // Select the proper video driver
    video = NULL;
    if (!driver_name) {
        driver_name = SDL_GetHint(SDL_HINT_VIDEO_DRIVER);
    }
    if (driver_name && *driver_name != 0) {
        const char *driver_attempt = driver_name;
        while (driver_attempt && *driver_attempt != 0 && !video) {
            const char *driver_attempt_end = SDL_strchr(driver_attempt, ',');
            size_t driver_attempt_len = (driver_attempt_end) ? (driver_attempt_end - driver_attempt)
                                                                     : SDL_strlen(driver_attempt);

            for (i = 0; bootstrap[i]; ++i) {
                if (!bootstrap[i]->is_preferred &&
                    (driver_attempt_len == SDL_strlen(bootstrap[i]->name)) &&
                    (SDL_strncasecmp(bootstrap[i]->name, driver_attempt, driver_attempt_len) == 0)) {
                    video = bootstrap[i]->create();
                    if (video) {
                        break;
                    }
                }
            }

            driver_attempt = (driver_attempt_end) ? (driver_attempt_end + 1) : NULL;
        }
    } else {
        for (i = 0; bootstrap[i]; ++i) {
            video = bootstrap[i]->create();
            if (video) {
                break;
            }
        }
    }
    if (!video) {
        if (driver_name) {
            SDL_SetError("%s not available", driver_name);
            goto pre_driver_error;
        }
        SDL_SetError("No available video device");
        goto pre_driver_error;
    }

    /* From this point on, use SDL_VideoQuit to cleanup on error, rather than
    pre_driver_error. */
    _this = video;
    _this->name = bootstrap[i]->name;
    _this->thread = SDL_GetCurrentThreadID();

    // Set some very sane GL defaults
    _this->gl_config.driver_loaded = 0;
    _this->gl_config.dll_handle = NULL;
    SDL_GL_ResetAttributes();

    // Initialize the video subsystem
    if (!_this->VideoInit(_this)) {
        SDL_VideoQuit();
        return false;
    }

    // Make sure some displays were added
    if (_this->num_displays == 0) {
        SDL_VideoQuit();
        return SDL_SetError("The video driver did not add any displays");
    }

    SDL_AddHintCallback(SDL_HINT_VIDEO_SYNC_WINDOW_OPERATIONS, SDL_SyncHintWatcher, NULL);

    /* Disable the screen saver by default. This is a change from <= 2.0.1,
       but most things using SDL are games or media players; you wouldn't
       want a screensaver to trigger if you're playing exclusively with a
       joystick, or passively watching a movie. Things that use SDL but
       function more like a normal desktop app should explicitly reenable the
       screensaver. */
    if (!SDL_GetHintBoolean(SDL_HINT_VIDEO_ALLOW_SCREENSAVER, false)) {
        SDL_DisableScreenSaver();
    }

    SDL_PostInitMouse();

    // We're ready to go!
    return true;

pre_driver_error:
    SDL_assert(_this == NULL);
    if (init_pen) {
        SDL_QuitPen();
    }
    if (init_touch) {
        SDL_QuitTouch();
    }
    if (init_mouse) {
        SDL_QuitMouse();
    }
    if (init_keyboard) {
        SDL_QuitKeyboard();
    }
    if (init_events) {
        SDL_QuitSubSystem(SDL_INIT_EVENTS);
    }
    return false;
}

const char *SDL_GetCurrentVideoDriver(void)
{
    if (!_this) {
        SDL_UninitializedVideo();
        return NULL;
    }
    return _this->name;
}

SDL_VideoDevice *SDL_GetVideoDevice(void)
{
    return _this;
}

bool SDL_OnVideoThread(void)
{
    return (_this && SDL_GetCurrentThreadID() == _this->thread);
}

void SDL_SetSystemTheme(SDL_SystemTheme theme)
{
    if (_this && theme != _this->system_theme) {
        _this->system_theme = theme;
        SDL_SendSystemThemeChangedEvent();
    }
}

SDL_SystemTheme SDL_GetSystemTheme(void)
{
    if (_this) {
        return _this->system_theme;
    } else {
        return SDL_SYSTEM_THEME_UNKNOWN;
    }
}

void SDL_UpdateDesktopBounds(void)
{
    SDL_Rect rect;
    SDL_zero(rect);

    SDL_DisplayID *displays = SDL_GetDisplays(NULL);
    if (displays) {
        for (int i = 0; displays[i]; ++i) {
            SDL_Rect bounds;
            if (SDL_GetDisplayBounds(displays[i], &bounds)) {
                if (i == 0) {
                    SDL_copyp(&rect, &bounds);
                } else {
                    SDL_GetRectUnion(&rect, &bounds, &rect);
                }
            }
        }
        SDL_free(displays);
    }
    SDL_copyp(&_this->desktop_bounds, &rect);
}

static void SDL_FinalizeDisplayMode(SDL_DisplayMode *mode)
{
    // Make sure all the fields are set up correctly
    if (mode->pixel_density <= 0.0f) {
        mode->pixel_density = 1.0f;
    }

    if (mode->refresh_rate_numerator > 0) {
        if (mode->refresh_rate_denominator <= 0) {
            mode->refresh_rate_denominator = 1;
        }
        mode->refresh_rate = ((100 * (Sint64)mode->refresh_rate_numerator) / mode->refresh_rate_denominator) / 100.0f;
    } else {
        SDL_CalculateFraction(mode->refresh_rate, &mode->refresh_rate_numerator, &mode->refresh_rate_denominator);
        mode->refresh_rate = (int)(mode->refresh_rate * 100) / 100.0f;
    }
}

SDL_DisplayID SDL_AddBasicVideoDisplay(const SDL_DisplayMode *desktop_mode)
{
    SDL_VideoDisplay display;

    SDL_zero(display);
    if (desktop_mode) {
        SDL_memcpy(&display.desktop_mode, desktop_mode, sizeof(display.desktop_mode));
    }
    return SDL_AddVideoDisplay(&display, false);
}

SDL_DisplayID SDL_AddVideoDisplay(const SDL_VideoDisplay *display, bool send_event)
{
    SDL_VideoDisplay **displays, *new_display;
    SDL_DisplayID id;
    SDL_PropertiesID props;
    int i;

    new_display = (SDL_VideoDisplay *)SDL_malloc(sizeof(*new_display));
    if (!new_display) {
        return true;
    }

    displays = (SDL_VideoDisplay **)SDL_realloc(_this->displays, (_this->num_displays + 1) * sizeof(*displays));
    if (!displays) {
        SDL_free(new_display);
        return true;
    }
    _this->displays = displays;
    _this->displays[_this->num_displays++] = new_display;

    id = SDL_GetNextObjectID();
    SDL_copyp(new_display, display);
    new_display->id = id;
    new_display->device = _this;
    if (display->name) {
        new_display->name = SDL_strdup(display->name);
    } else {
        char name[32];

        SDL_itoa(id, name, 10);
        new_display->name = SDL_strdup(name);
    }
    if (new_display->content_scale == 0.0f) {
        new_display->content_scale = 1.0f;
    }

    new_display->desktop_mode.displayID = id;
    new_display->current_mode = &new_display->desktop_mode;
    SDL_FinalizeDisplayMode(&new_display->desktop_mode);

    for (i = 0; i < new_display->num_fullscreen_modes; ++i) {
        new_display->fullscreen_modes[i].displayID = id;
    }

    new_display->HDR.HDR_headroom = SDL_max(display->HDR.HDR_headroom, 1.0f);
    new_display->HDR.SDR_white_level = SDL_max(display->HDR.SDR_white_level, 1.0f);

    props = SDL_GetDisplayProperties(id);
    SDL_SetBooleanProperty(props, SDL_PROP_DISPLAY_HDR_ENABLED_BOOLEAN, new_display->HDR.HDR_headroom > 1.0f);

    SDL_UpdateDesktopBounds();

    if (send_event) {
        SDL_SendDisplayEvent(new_display, SDL_EVENT_DISPLAY_ADDED, 0, 0);
    }

    return id;
}

void SDL_OnDisplayAdded(SDL_VideoDisplay *display)
{
    SDL_Window *window;

    // See if any windows have changed to the new display
    for (window = _this->windows; window; window = window->next) {
        SDL_CheckWindowDisplayChanged(window);
    }
}

void SDL_OnDisplayMoved(SDL_VideoDisplay *display)
{
    SDL_UpdateDesktopBounds();
}

void SDL_DelVideoDisplay(SDL_DisplayID displayID, bool send_event)
{
    SDL_VideoDisplay *display;
    int display_index = SDL_GetDisplayIndex(displayID);
    if (display_index < 0) {
        return;
    }

    display = _this->displays[display_index];

    if (send_event) {
        SDL_SendDisplayEvent(display, SDL_EVENT_DISPLAY_REMOVED, 0, 0);
    }

    SDL_DestroyProperties(display->props);
    SDL_free(display->name);
    SDL_ResetFullscreenDisplayModes(display);
    SDL_free(display->desktop_mode.internal);
    display->desktop_mode.internal = NULL;
    SDL_free(display->internal);
    display->internal = NULL;
    SDL_free(display);

    if (display_index < (_this->num_displays - 1)) {
        SDL_memmove(&_this->displays[display_index], &_this->displays[display_index + 1], (_this->num_displays - display_index - 1) * sizeof(_this->displays[display_index]));
    }
    --_this->num_displays;

    SDL_UpdateDesktopBounds();
}

SDL_DisplayID *SDL_GetDisplays(int *count)
{
    int i;
    SDL_DisplayID *displays;

    if (!_this) {
        if (count) {
            *count = 0;
        }

        SDL_UninitializedVideo();
        return NULL;
    }

    displays = (SDL_DisplayID *)SDL_malloc((_this->num_displays + 1) * sizeof(*displays));
    if (displays) {
        if (count) {
            *count = _this->num_displays;
        }

        for (i = 0; i < _this->num_displays; ++i) {
            displays[i] = _this->displays[i]->id;
        }
        displays[i] = 0;
    } else {
        if (count) {
            *count = 0;
        }
    }
    return displays;
}

SDL_VideoDisplay *SDL_GetVideoDisplay(SDL_DisplayID displayID)
{
    int display_index;

    display_index = SDL_GetDisplayIndex(displayID);
    if (display_index < 0) {
        return NULL;
    }
    return _this->displays[display_index];
}

SDL_VideoDisplay *SDL_GetVideoDisplayForWindow(SDL_Window *window)
{
    return SDL_GetVideoDisplay(SDL_GetDisplayForWindow(window));
}

SDL_DisplayID SDL_GetPrimaryDisplay(void)
{
    if (!_this || _this->num_displays == 0) {
        SDL_UninitializedVideo();
        return 0;
    }
    return _this->displays[0]->id;
}

int SDL_GetDisplayIndex(SDL_DisplayID displayID)
{
    int display_index;

    if (!_this) {
        SDL_UninitializedVideo();
        return -1;
    }

    for (display_index = 0; display_index < _this->num_displays; ++display_index) {
        if (displayID == _this->displays[display_index]->id) {
            return display_index;
        }
    }
    SDL_SetError("Invalid display");
    return -1;
}

SDL_DisplayData *SDL_GetDisplayDriverData(SDL_DisplayID displayID)
{
    SDL_VideoDisplay *display = SDL_GetVideoDisplay(displayID);

    CHECK_DISPLAY_MAGIC(display, NULL);

    return display->internal;
}

SDL_DisplayData *SDL_GetDisplayDriverDataForWindow(SDL_Window *window)
{
    return SDL_GetDisplayDriverData(SDL_GetDisplayForWindow(window));
}

SDL_PropertiesID SDL_GetDisplayProperties(SDL_DisplayID displayID)
{
    SDL_VideoDisplay *display = SDL_GetVideoDisplay(displayID);

    CHECK_DISPLAY_MAGIC(display, 0);

    if (display->props == 0) {
        display->props = SDL_CreateProperties();
    }
    return display->props;
}

const char *SDL_GetDisplayName(SDL_DisplayID displayID)
{
    SDL_VideoDisplay *display = SDL_GetVideoDisplay(displayID);

    CHECK_DISPLAY_MAGIC(display, NULL);

    return display->name;
}

bool SDL_GetDisplayBounds(SDL_DisplayID displayID, SDL_Rect *rect)
{
    SDL_VideoDisplay *display = SDL_GetVideoDisplay(displayID);

    CHECK_DISPLAY_MAGIC(display, false);

    if (!rect) {
        return SDL_InvalidParamError("rect");
    }

    if (_this->GetDisplayBounds) {
        if (_this->GetDisplayBounds(_this, display, rect)) {
            return true;
        }
    }

    // Assume that the displays are left to right
    if (displayID == SDL_GetPrimaryDisplay()) {
        rect->x = 0;
        rect->y = 0;
    } else {
        SDL_GetDisplayBounds(_this->displays[SDL_GetDisplayIndex(displayID) - 1]->id, rect);
        rect->x += rect->w;
    }
    rect->w = display->current_mode->w;
    rect->h = display->current_mode->h;
    return true;
}

static int ParseDisplayUsableBoundsHint(SDL_Rect *rect)
{
    const char *hint = SDL_GetHint(SDL_HINT_DISPLAY_USABLE_BOUNDS);
    return hint && (SDL_sscanf(hint, "%d,%d,%d,%d", &rect->x, &rect->y, &rect->w, &rect->h) == 4);
}

bool SDL_GetDisplayUsableBounds(SDL_DisplayID displayID, SDL_Rect *rect)
{
    SDL_VideoDisplay *display = SDL_GetVideoDisplay(displayID);

    CHECK_DISPLAY_MAGIC(display, false);

    if (!rect) {
        return SDL_InvalidParamError("rect");
    }

    if (displayID == SDL_GetPrimaryDisplay() && ParseDisplayUsableBoundsHint(rect)) {
        return true;
    }

    if (_this->GetDisplayUsableBounds) {
        if (_this->GetDisplayUsableBounds(_this, display, rect)) {
            return true;
        }
    }

    // Oh well, just give the entire display bounds.
    return SDL_GetDisplayBounds(displayID, rect);
}

SDL_DisplayOrientation SDL_GetNaturalDisplayOrientation(SDL_DisplayID displayID)
{
    SDL_VideoDisplay *display = SDL_GetVideoDisplay(displayID);

    CHECK_DISPLAY_MAGIC(display, SDL_ORIENTATION_UNKNOWN);

    if (display->natural_orientation != SDL_ORIENTATION_UNKNOWN) {
        return display->natural_orientation;
    } else {
        // Default to landscape if the driver hasn't set it
        return SDL_ORIENTATION_LANDSCAPE;
    }
}

SDL_DisplayOrientation SDL_GetCurrentDisplayOrientation(SDL_DisplayID displayID)
{
    SDL_VideoDisplay *display = SDL_GetVideoDisplay(displayID);

    CHECK_DISPLAY_MAGIC(display, SDL_ORIENTATION_UNKNOWN);

    if (display->current_orientation != SDL_ORIENTATION_UNKNOWN) {
        return display->current_orientation;
    } else {
        // Default to landscape if the driver hasn't set it
        return SDL_ORIENTATION_LANDSCAPE;
    }
}

void SDL_SetDisplayContentScale(SDL_VideoDisplay *display, float scale)
{
    if (scale != display->content_scale) {
        SDL_Window *window;

        display->content_scale = scale;
        SDL_SendDisplayEvent(display, SDL_EVENT_DISPLAY_CONTENT_SCALE_CHANGED, 0, 0);

        // Check the windows on this display
        for (window = _this->windows; window; window = window->next) {
            if (display->id == window->last_displayID) {
                SDL_CheckWindowDisplayScaleChanged(window);
            }
        }
    }
}

float SDL_GetDisplayContentScale(SDL_DisplayID displayID)
{
    SDL_VideoDisplay *display = SDL_GetVideoDisplay(displayID);

    CHECK_DISPLAY_MAGIC(display, 0.0f);

    return display->content_scale;
}

void SDL_SetWindowHDRProperties(SDL_Window *window, const SDL_HDROutputProperties *HDR, bool send_event)
{
    if (window->HDR.HDR_headroom != HDR->HDR_headroom || window->HDR.SDR_white_level != window->HDR.SDR_white_level) {
        SDL_PropertiesID window_props = SDL_GetWindowProperties(window);

        SDL_SetFloatProperty(window_props, SDL_PROP_WINDOW_HDR_HEADROOM_FLOAT, SDL_max(HDR->HDR_headroom, 1.0f));
        SDL_SetFloatProperty(window_props, SDL_PROP_WINDOW_SDR_WHITE_LEVEL_FLOAT, SDL_max(HDR->SDR_white_level, 1.0f));
        SDL_SetBooleanProperty(window_props, SDL_PROP_WINDOW_HDR_ENABLED_BOOLEAN, HDR->HDR_headroom > 1.0f);
        SDL_copyp(&window->HDR, HDR);

        if (send_event) {
            SDL_SendWindowEvent(window, SDL_EVENT_WINDOW_HDR_STATE_CHANGED, HDR->HDR_headroom > 1.0f, 0);
        }
    }
}

void SDL_SetDisplayHDRProperties(SDL_VideoDisplay *display, const SDL_HDROutputProperties *HDR)
{
    bool changed = false;

    if (HDR->SDR_white_level != display->HDR.SDR_white_level) {
        display->HDR.SDR_white_level = SDL_max(HDR->SDR_white_level, 1.0f);
        changed = true;
    }
    if (HDR->HDR_headroom != display->HDR.HDR_headroom) {
        display->HDR.HDR_headroom = SDL_max(HDR->HDR_headroom, 1.0f);
        changed = true;
    }
    SDL_copyp(&display->HDR, HDR);

    if (changed && !SDL_DriverSendsHDRChanges(_this)) {
        for (SDL_Window *w = display->device->windows; w; w = w->next) {
            if (SDL_GetDisplayForWindow(w) == display->id) {
                SDL_SetWindowHDRProperties(w, &display->HDR, true);
            }
        }
    }
}

static void SDL_UpdateFullscreenDisplayModes(SDL_VideoDisplay *display)
{
    if (display->num_fullscreen_modes == 0 && _this->GetDisplayModes) {
        _this->GetDisplayModes(_this, display);
    }
}

// Return the matching mode as a pointer into our current mode list
static const SDL_DisplayMode *SDL_GetFullscreenModeMatch(const SDL_DisplayMode *mode)
{
    SDL_VideoDisplay *display;
    SDL_DisplayMode fullscreen_mode;

    if (mode->w <= 0 || mode->h <= 0) {
        // Use the desktop mode
        return NULL;
    }

    SDL_memcpy(&fullscreen_mode, mode, sizeof(fullscreen_mode));
    if (fullscreen_mode.displayID == 0) {
        fullscreen_mode.displayID = SDL_GetPrimaryDisplay();
    }
    SDL_FinalizeDisplayMode(&fullscreen_mode);

    mode = NULL;

    display = SDL_GetVideoDisplay(fullscreen_mode.displayID);
    if (display) {
        SDL_UpdateFullscreenDisplayModes(display);

        // Search for an exact match
        if (!mode) {
            for (int i = 0; i < display->num_fullscreen_modes; ++i) {
                if (SDL_memcmp(&fullscreen_mode, &display->fullscreen_modes[i], sizeof(fullscreen_mode)) == 0) {
                    mode = &display->fullscreen_modes[i];
                    break;
                }
            }
        }

        // Search for a mode with the same characteristics
        if (!mode) {
            for (int i = 0; i < display->num_fullscreen_modes; ++i) {
                if (cmpmodes(&fullscreen_mode, &display->fullscreen_modes[i]) == 0) {
                    mode = &display->fullscreen_modes[i];
                    break;
                }
            }
        }
    }
    return mode;
}

bool SDL_AddFullscreenDisplayMode(SDL_VideoDisplay *display, const SDL_DisplayMode *mode)
{
    SDL_DisplayMode *modes;
    SDL_DisplayMode new_mode;
    int i, nmodes;

    // Finalize the mode for the display
    SDL_memcpy(&new_mode, mode, sizeof(new_mode));
    new_mode.displayID = display->id;
    SDL_FinalizeDisplayMode(&new_mode);

    // Make sure we don't already have the mode in the list
    modes = display->fullscreen_modes;
    nmodes = display->num_fullscreen_modes;
    for (i = 0; i < nmodes; ++i) {
        if (cmpmodes(&new_mode, &modes[i]) == 0) {
            return false;
        }
    }

    // Go ahead and add the new mode
    if (nmodes == display->max_fullscreen_modes) {
        modes = (SDL_DisplayMode *)SDL_malloc((display->max_fullscreen_modes + 32) * sizeof(*modes));
        if (!modes) {
            return false;
        }

        if (display->fullscreen_modes) {
            // Copy the list and update the current mode pointer, if necessary.
            SDL_memcpy(modes, display->fullscreen_modes, nmodes * sizeof(*modes));
            for (i = 0; i < nmodes; ++i) {
                if (display->current_mode == &display->fullscreen_modes[i]) {
                    display->current_mode = &modes[i];
                }
            }

            SDL_free(display->fullscreen_modes);
        }

        display->fullscreen_modes = modes;
        display->max_fullscreen_modes += 32;
    }
    SDL_memcpy(&modes[display->num_fullscreen_modes++], &new_mode, sizeof(new_mode));

    // Re-sort video modes
    SDL_qsort(display->fullscreen_modes, display->num_fullscreen_modes,
              sizeof(SDL_DisplayMode), cmpmodes);

    return true;
}

void SDL_ResetFullscreenDisplayModes(SDL_VideoDisplay *display)
{
    int i;

    for (i = display->num_fullscreen_modes; i--;) {
        SDL_free(display->fullscreen_modes[i].internal);
        display->fullscreen_modes[i].internal = NULL;
    }
    SDL_free(display->fullscreen_modes);
    display->fullscreen_modes = NULL;
    display->num_fullscreen_modes = 0;
    display->max_fullscreen_modes = 0;
    display->current_mode = &display->desktop_mode;
}

SDL_DisplayMode **SDL_GetFullscreenDisplayModes(SDL_DisplayID displayID, int *count)
{
    int i;
    int num_modes;
    SDL_DisplayMode **result;
    SDL_VideoDisplay *display = SDL_GetVideoDisplay(displayID);

    if (count) {
        *count = 0;
    }

    CHECK_DISPLAY_MAGIC(display, NULL);

    SDL_UpdateFullscreenDisplayModes(display);

    num_modes = display->num_fullscreen_modes;
    result = (SDL_DisplayMode **)SDL_malloc((num_modes + 1) * sizeof(*result) + num_modes * sizeof(**result));
    if (result) {
        SDL_DisplayMode *modes = (SDL_DisplayMode *)((Uint8 *)result + ((num_modes + 1) * sizeof(*result)));
        SDL_memcpy(modes, display->fullscreen_modes, num_modes * sizeof(*modes));
        for (i = 0; i < num_modes; ++i) {
            result[i] = modes++;
        }
        result[i] = NULL;

        if (count) {
            *count = num_modes;
        }
    } else {
        if (count) {
            *count = 0;
        }
    }
    return result;
}

bool SDL_GetClosestFullscreenDisplayMode(SDL_DisplayID displayID, int w, int h, float refresh_rate, bool include_high_density_modes, SDL_DisplayMode *result)
{
    if (!result) {
        return SDL_InvalidParamError("closest"); // Parameter `result` is called `closest` in the header.
    }

    const SDL_DisplayMode *mode, *closest = NULL;
    float aspect_ratio;
    int i;
    SDL_VideoDisplay *display = SDL_GetVideoDisplay(displayID);

    SDL_zerop(result);

    CHECK_DISPLAY_MAGIC(display, false);

    if (h > 0) {
        aspect_ratio = (float)w / h;
    } else {
        aspect_ratio = 1.0f;
    }

    if (refresh_rate == 0.0f) {
        refresh_rate = display->desktop_mode.refresh_rate;
    }

    SDL_UpdateFullscreenDisplayModes(display);

    for (i = 0; i < display->num_fullscreen_modes; ++i) {
        mode = &display->fullscreen_modes[i];

        if (w > mode->w) {
            // Out of sorted modes large enough here
            break;
        }
        if (h > mode->h) {
            /* Wider, but not tall enough, due to a different aspect ratio.
             * This mode must be skipped, but closer modes may still follow */
            continue;
        }
        if (mode->pixel_density > 1.0f && !include_high_density_modes) {
            continue;
        }
        if (closest) {
            float current_aspect_ratio = (float)mode->w / mode->h;
            float closest_aspect_ratio = (float)closest->w / closest->h;
            if (SDL_fabsf(aspect_ratio - closest_aspect_ratio) < SDL_fabsf(aspect_ratio - current_aspect_ratio)) {
                // The mode we already found has a better aspect ratio match
                continue;
            }

            if (mode->w == closest->w && mode->h == closest->h &&
                SDL_fabsf(closest->refresh_rate - refresh_rate) < SDL_fabsf(mode->refresh_rate - refresh_rate)) {
                /* We already found a mode and the new mode is further from our
                 * refresh rate target */
                continue;
            }
        }

        closest = mode;
    }
    if (!closest) {
        return SDL_SetError("Couldn't find any matching video modes");
    }

    SDL_copyp(result, closest);

    return true;
}

static bool DisplayModeChanged(const SDL_DisplayMode *old_mode, const SDL_DisplayMode *new_mode)
{
    return ((old_mode->displayID && old_mode->displayID != new_mode->displayID) ||
            (old_mode->format && old_mode->format != new_mode->format) ||
            (old_mode->w && old_mode->h && (old_mode->w != new_mode->w ||old_mode->h != new_mode->h)) ||
            (old_mode->pixel_density != 0.0f && old_mode->pixel_density != new_mode->pixel_density) ||
            (old_mode->refresh_rate != 0.0f && old_mode->refresh_rate != new_mode->refresh_rate));
}

void SDL_SetDesktopDisplayMode(SDL_VideoDisplay *display, const SDL_DisplayMode *mode)
{
    SDL_DisplayMode last_mode;

    if (display->fullscreen_active) {
        // This is a temporary mode change, don't save the desktop mode
        return;
    }

    SDL_copyp(&last_mode, &display->desktop_mode);

    if (display->desktop_mode.internal) {
        SDL_free(display->desktop_mode.internal);
    }
    SDL_copyp(&display->desktop_mode, mode);
    display->desktop_mode.displayID = display->id;
    SDL_FinalizeDisplayMode(&display->desktop_mode);

    if (DisplayModeChanged(&last_mode, &display->desktop_mode)) {
        SDL_SendDisplayEvent(display, SDL_EVENT_DISPLAY_DESKTOP_MODE_CHANGED, mode->w, mode->h);
        if (display->current_mode == &display->desktop_mode) {
            SDL_SendDisplayEvent(display, SDL_EVENT_DISPLAY_CURRENT_MODE_CHANGED, mode->w, mode->h);
        }
    }
}

const SDL_DisplayMode *SDL_GetDesktopDisplayMode(SDL_DisplayID displayID)
{
    SDL_VideoDisplay *display = SDL_GetVideoDisplay(displayID);

    CHECK_DISPLAY_MAGIC(display, NULL);

    return &display->desktop_mode;
}

void SDL_SetCurrentDisplayMode(SDL_VideoDisplay *display, const SDL_DisplayMode *mode)
{
    SDL_DisplayMode last_mode;

    if (display->current_mode) {
        SDL_copyp(&last_mode, display->current_mode);
    } else {
        SDL_zero(last_mode);
    }

    display->current_mode = mode;

    if (DisplayModeChanged(&last_mode, mode)) {
        SDL_SendDisplayEvent(display, SDL_EVENT_DISPLAY_CURRENT_MODE_CHANGED, mode->w, mode->h);
    }
}

const SDL_DisplayMode *SDL_GetCurrentDisplayMode(SDL_DisplayID displayID)
{
    SDL_VideoDisplay *display = SDL_GetVideoDisplay(displayID);

    CHECK_DISPLAY_MAGIC(display, NULL);

    // Make sure our mode list is updated
    SDL_UpdateFullscreenDisplayModes(display);

    return display->current_mode;
}

bool SDL_SetDisplayModeForDisplay(SDL_VideoDisplay *display, SDL_DisplayMode *mode)
{
    /* Mode switching is being emulated per-window; nothing to do and cannot fail,
     * except for XWayland, which still needs the actual mode setting call since
     * it's emulated via the XRandR interface.
     */
    if (SDL_ModeSwitchingEmulated(_this) && SDL_strcmp(_this->name, "x11") != 0) {
        return true;
    }

    if (!mode) {
        mode = &display->desktop_mode;
    }

    if (mode == display->current_mode) {
        return true;
    }

    // Actually change the display mode
    if (_this->SetDisplayMode) {
        bool result;

        _this->setting_display_mode = true;
        result = _this->SetDisplayMode(_this, display, mode);
        _this->setting_display_mode = false;
        if (!result) {
            return false;
        }
    }

    SDL_SetCurrentDisplayMode(display, mode);

    return true;
}

/**
 * If x, y are outside of rect, snaps them to the closest point inside rect
 * (between rect->x, rect->y, inclusive, and rect->x + w, rect->y + h, exclusive)
 */
static void SDL_GetClosestPointOnRect(const SDL_Rect *rect, SDL_Point *point)
{
    const int right = rect->x + rect->w - 1;
    const int bottom = rect->y + rect->h - 1;

    if (point->x < rect->x) {
        point->x = rect->x;
    } else if (point->x > right) {
        point->x = right;
    }

    if (point->y < rect->y) {
        point->y = rect->y;
    } else if (point->y > bottom) {
        point->y = bottom;
    }
}

static SDL_DisplayID GetDisplayForRect(int x, int y, int w, int h)
{
    int i, dist;
    SDL_DisplayID closest = 0;
    int closest_dist = 0x7FFFFFFF;
    SDL_Point closest_point_on_display;
    SDL_Point delta;
    SDL_Point center;
    center.x = x + w / 2;
    center.y = y + h / 2;

    if (_this) {
        for (i = 0; i < _this->num_displays; ++i) {
            SDL_VideoDisplay *display = _this->displays[i];
            SDL_Rect display_rect;
            SDL_GetDisplayBounds(display->id, &display_rect);

            // Check if the window is fully enclosed
            if (SDL_GetRectEnclosingPoints(&center, 1, &display_rect, NULL)) {
                return display->id;
            }

            // Snap window center to the display rect
            closest_point_on_display = center;
            SDL_GetClosestPointOnRect(&display_rect, &closest_point_on_display);

            delta.x = center.x - closest_point_on_display.x;
            delta.y = center.y - closest_point_on_display.y;
            dist = (delta.x * delta.x + delta.y * delta.y);
            if (dist < closest_dist) {
                closest = display->id;
                closest_dist = dist;
            }
        }
    }

    if (closest == 0) {
        SDL_SetError("Couldn't find any displays");
    }

    return closest;
}

void SDL_RelativeToGlobalForWindow(SDL_Window *window, int rel_x, int rel_y, int *abs_x, int *abs_y)
{
    SDL_Window *w;

    if (SDL_WINDOW_IS_POPUP(window)) {
        // Calculate the total offset of the popup from the parents
        for (w = window->parent; w; w = w->parent) {
            rel_x += w->x;
            rel_y += w->y;

            if (!SDL_WINDOW_IS_POPUP(w)) {
                break;
            }
        }
    }

    if (abs_x) {
        *abs_x = rel_x;
    }
    if (abs_y) {
        *abs_y = rel_y;
    }
}

void SDL_GlobalToRelativeForWindow(SDL_Window *window, int abs_x, int abs_y, int *rel_x, int *rel_y)
{
    SDL_Window *w;

    if (SDL_WINDOW_IS_POPUP(window)) {
        // Convert absolute window coordinates to relative for a popup
        for (w = window->parent; w; w = w->parent) {
            abs_x -= w->x;
            abs_y -= w->y;

            if (!SDL_WINDOW_IS_POPUP(w)) {
                break;
            }
        }
    }

    if (rel_x) {
        *rel_x = abs_x;
    }
    if (rel_y) {
        *rel_y = abs_y;
    }
}

SDL_DisplayID SDL_GetDisplayForPoint(const SDL_Point *point)
{
    if (!point) {
        SDL_InvalidParamError("point");
        return 0;
    }

    return GetDisplayForRect(point->x, point->y, 1, 1);
}

SDL_DisplayID SDL_GetDisplayForRect(const SDL_Rect *rect)
{
    if (!rect) {
        SDL_InvalidParamError("rect");
        return 0;
    }

    return GetDisplayForRect(rect->x, rect->y, rect->w, rect->h);
}

SDL_DisplayID SDL_GetDisplayForWindowPosition(SDL_Window *window)
{
    int x, y;
    SDL_DisplayID displayID = 0;

    CHECK_WINDOW_MAGIC(window, 0);

    if (_this->GetDisplayForWindow) {
        displayID = _this->GetDisplayForWindow(_this, window);
    }

    /* A backend implementation may fail to get a display for the window
     * (for example if the window is off-screen), but other code may expect it
     * to succeed in that situation, so we fall back to a generic position-
     * based implementation in that case. */
    SDL_RelativeToGlobalForWindow(window, window->x, window->y, &x, &y);

    if (!displayID) {
        /* Fullscreen windows may be larger than the display if they were moved between differently sized
         * displays and the new position was received before the new size or vice versa. Using the center
         * of the window rect in this case can report the wrong display, so use the origin.
         */
        if (window->flags & SDL_WINDOW_FULLSCREEN) {
            displayID = GetDisplayForRect(x, y, 1, 1);
        } else {
            displayID = GetDisplayForRect(x, y, window->w, window->h);
        }
    }
    if (!displayID) {
        // Use the primary display for a window if we can't find it anywhere else
        displayID = SDL_GetPrimaryDisplay();
    }
    return displayID;
}

SDL_VideoDisplay *SDL_GetVideoDisplayForFullscreenWindow(SDL_Window *window)
{
    SDL_DisplayID displayID = 0;

    CHECK_WINDOW_MAGIC(window, 0);

    // An explicit fullscreen display overrides all
    if (window->current_fullscreen_mode.displayID) {
        displayID = window->current_fullscreen_mode.displayID;
    }

    /* This is used to handle the very common pattern of SDL_SetWindowPosition()
     * followed immediately by SDL_SetWindowFullscreen() to make the window fullscreen
     * desktop on a specific display. If the backend doesn't support changing the
     * window position, or an async window manager hasn't yet actually moved the window,
     * the current position won't be updated at the time of the fullscreen call.
     */
    if (!displayID) {
        // Use the pending position and dimensions, if available, otherwise, use the current.
        const int x = window->last_position_pending ? window->pending.x : window->x;
        const int y = window->last_position_pending ? window->pending.y : window->y;
        const int w = window->last_size_pending ? window->pending.w : window->w;
        const int h = window->last_size_pending ? window->pending.h : window->h;

        displayID = GetDisplayForRect(x, y, w, h);
    }
    if (!displayID) {
        // Use the primary display for a window if we can't find it anywhere else
        displayID = SDL_GetPrimaryDisplay();
    }
    return SDL_GetVideoDisplay(displayID);
}

SDL_DisplayID SDL_GetDisplayForWindow(SDL_Window *window)
{
    SDL_DisplayID displayID = 0;

    CHECK_WINDOW_MAGIC(window, 0);

    // An explicit fullscreen display overrides all
    if (window->flags & SDL_WINDOW_FULLSCREEN) {
        displayID = window->current_fullscreen_mode.displayID;
    }

    if (!displayID) {
        displayID = SDL_GetDisplayForWindowPosition(window);
    }
    return displayID;
}

static void SDL_CheckWindowDisplayChanged(SDL_Window *window)
{
    if (SDL_SendsDisplayChanges(_this)) {
        return;
    }

    SDL_DisplayID displayID = SDL_GetDisplayForWindowPosition(window);

    if (displayID != window->last_displayID) {
        int i, display_index;

        // Sanity check our fullscreen windows
        display_index = SDL_GetDisplayIndex(displayID);
        for (i = 0; i < _this->num_displays; ++i) {
            SDL_VideoDisplay *display = _this->displays[i];

            if (display->fullscreen_window == window) {
                if (display_index != i) {
                    if (display_index < 0) {
                        display_index = i;
                    } else {
                        SDL_VideoDisplay *new_display = _this->displays[display_index];

                        // The window was moved to a different display
                        if (new_display->fullscreen_window &&
                            new_display->fullscreen_window != window) {
                            // Uh oh, there's already a fullscreen window here; minimize it
                            SDL_MinimizeWindow(new_display->fullscreen_window);
                        }
                        new_display->fullscreen_window = window;
                        display->fullscreen_window = NULL;
                    }
                }
                break;
            }
        }

        SDL_SendWindowEvent(window, SDL_EVENT_WINDOW_DISPLAY_CHANGED, (int)displayID, 0);
    }
}

float SDL_GetWindowPixelDensity(SDL_Window *window)
{
    int window_w, window_h, pixel_w, pixel_h;
    float pixel_density = 1.0f;

    CHECK_WINDOW_MAGIC(window, 0.0f);

    if (SDL_GetWindowSize(window, &window_w, &window_h) &&
        SDL_GetWindowSizeInPixels(window, &pixel_w, &pixel_h)) {
        pixel_density = (float)pixel_w / window_w;
    }
    return pixel_density;
}

float SDL_GetWindowDisplayScale(SDL_Window *window)
{
    CHECK_WINDOW_MAGIC(window, 0.0f);

    return window->display_scale;
}

static void SDL_CheckWindowDisplayScaleChanged(SDL_Window *window)
{
    float display_scale;

    if (_this->GetWindowContentScale) {
        display_scale = _this->GetWindowContentScale(_this, window);
    } else {
        const float pixel_density = SDL_GetWindowPixelDensity(window);
        const float content_scale = SDL_GetDisplayContentScale(SDL_GetDisplayForWindowPosition(window));

        display_scale = pixel_density * content_scale;
    }

    if (display_scale != window->display_scale) {
        window->display_scale = display_scale;
        SDL_SendWindowEvent(window, SDL_EVENT_WINDOW_DISPLAY_SCALE_CHANGED, 0, 0);
    }
}

static void SDL_RestoreMousePosition(SDL_Window *window)
{
    float x, y;
    SDL_Mouse *mouse = SDL_GetMouse();

    if (window == SDL_GetMouseFocus()) {
        const bool prev_warp_val = mouse->warp_emulation_prohibited;
        SDL_GetMouseState(&x, &y);

        // Disable the warp emulation so it isn't accidentally activated on a fullscreen transitions.
        mouse->warp_emulation_prohibited = true;
        SDL_WarpMouseInWindow(window, x, y);
        mouse->warp_emulation_prohibited = prev_warp_val;
    }
}

bool SDL_UpdateFullscreenMode(SDL_Window *window, SDL_FullscreenOp fullscreen, bool commit)
{
    SDL_VideoDisplay *display = NULL;
    SDL_DisplayMode *mode = NULL;
    int i;

    CHECK_WINDOW_MAGIC(window, false);

    window->fullscreen_exclusive = false;
    window->update_fullscreen_on_display_changed = false;

    // If we are in the process of hiding don't go back to fullscreen
    if (window->is_destroying || window->is_hiding) {
        fullscreen = SDL_FULLSCREEN_OP_LEAVE;
    }

    // Get the correct display for this operation
    if (fullscreen) {
        display = SDL_GetVideoDisplayForFullscreenWindow(window);
        if (!display) {
            // This should never happen, but it did...
            goto done;
        }
    } else {
        for (i = 0; i < _this->num_displays; ++i) {
            display = _this->displays[i];
            if (display->fullscreen_window == window) {
                break;
            }
        }
        if (!display || i == _this->num_displays) {
            // Already not fullscreen on any display
            display = NULL;
        }
    }

    if (fullscreen) {
        mode = (SDL_DisplayMode *)SDL_GetWindowFullscreenMode(window);
        if (mode) {
            window->fullscreen_exclusive = true;
        } else {
            // Make sure the current mode is zeroed for fullscreen desktop.
            SDL_zero(window->current_fullscreen_mode);
        }
    }

#if defined(SDL_PLATFORM_MACOS) && defined(SDL_VIDEO_DRIVER_COCOA)
    /* if the window is going away and no resolution change is necessary,
       do nothing, or else we may trigger an ugly double-transition
     */
    if (SDL_strcmp(_this->name, "cocoa") == 0) { // don't do this for X11, etc
        if (window->is_destroying && !window->last_fullscreen_exclusive_display) {
            window->fullscreen_exclusive = false;
            if (display) {
                display->fullscreen_window = NULL;
            }
            goto done;
        }
        if (commit) {
            // If we're switching between a fullscreen Space and exclusive fullscreen, we need to get back to normal first.
            if (fullscreen && Cocoa_IsWindowInFullscreenSpace(window) && !window->last_fullscreen_exclusive_display && window->fullscreen_exclusive) {
                if (!Cocoa_SetWindowFullscreenSpace(window, false, true)) {
                    goto error;
                }
            } else if (fullscreen && window->last_fullscreen_exclusive_display && !window->fullscreen_exclusive) {
                for (i = 0; i < _this->num_displays; ++i) {
                    SDL_VideoDisplay *last_display = _this->displays[i];
                    if (last_display->fullscreen_window == window) {
                        SDL_SetDisplayModeForDisplay(last_display, NULL);
                        if (_this->SetWindowFullscreen) {
                            _this->SetWindowFullscreen(_this, window, last_display, false);
                        }
                        last_display->fullscreen_window = NULL;
                    }
                }
            }

            if (Cocoa_SetWindowFullscreenSpace(window, !!fullscreen, syncHint)) {
                goto done;
            }
        }
    }
#endif

    if (display) {
        // Restore the video mode on other displays if needed
        for (i = 0; i < _this->num_displays; ++i) {
            SDL_VideoDisplay *other = _this->displays[i];
            if (other != display && other->fullscreen_window == window) {
                SDL_SetDisplayModeForDisplay(other, NULL);
                other->fullscreen_window = NULL;
            }
        }
    }

    if (fullscreen) {
        int mode_w = 0, mode_h = 0;
        bool resized = false;

        // Hide any other fullscreen window on this display
        if (display->fullscreen_window &&
            display->fullscreen_window != window) {
            SDL_MinimizeWindow(display->fullscreen_window);
        }

        display->fullscreen_active = window->fullscreen_exclusive;

        if (!SDL_SetDisplayModeForDisplay(display, mode)) {
            goto error;
        }
        if (commit) {
            SDL_FullscreenResult ret = SDL_FULLSCREEN_SUCCEEDED;
            if (_this->SetWindowFullscreen) {
                ret = _this->SetWindowFullscreen(_this, window, display, fullscreen);
            } else {
                resized = true;
            }

            if (ret == SDL_FULLSCREEN_SUCCEEDED) {
                // Window is fullscreen immediately upon return. If the driver hasn't already sent the event, do so now.
                if (!(window->flags & SDL_WINDOW_FULLSCREEN)) {
                    SDL_SendWindowEvent(window, SDL_EVENT_WINDOW_ENTER_FULLSCREEN, 0, 0);
                }
            } else if (ret == SDL_FULLSCREEN_FAILED) {
                display->fullscreen_active = false;
                goto error;
            }
        }

        if (window->flags & SDL_WINDOW_FULLSCREEN) {
            display->fullscreen_window = window;

            /* Android may not resize the window to exactly what our fullscreen mode is,
             * especially on windowed Android environments like the Chromebook or Samsung DeX.
             * Given this, we shouldn't use the mode size. Android's SetWindowFullscreen
             * will generate the window event for us with the proper final size.
             *
             * This is also unnecessary on Cocoa, Wayland, Win32, and X11 (will send SDL_EVENT_WINDOW_RESIZED).
             */
            if (!SDL_SendsFullscreenDimensions(_this)) {
                SDL_Rect displayRect;

                if (mode) {
                    mode_w = mode->w;
                    mode_h = mode->h;
                    SDL_GetDisplayBounds(mode->displayID, &displayRect);
                } else {
                    mode_w = display->desktop_mode.w;
                    mode_h = display->desktop_mode.h;
                    SDL_GetDisplayBounds(display->id, &displayRect);
                }

                if (window->w != mode_w || window->h != mode_h) {
                    resized = true;
                }

                SDL_SendWindowEvent(window, SDL_EVENT_WINDOW_MOVED, displayRect.x, displayRect.y);

                if (resized) {
                    SDL_SendWindowEvent(window, SDL_EVENT_WINDOW_RESIZED, mode_w, mode_h);
                } else {
                    SDL_OnWindowResized(window);
                }
            }

            // Restore the cursor position
            if (!SDL_DisableMouseWarpOnFullscreenTransitions(_this)) {
                SDL_RestoreMousePosition(window);
            }
        }
    } else {
        bool resized = false;

        // Restore the desktop mode
        if (display) {
            display->fullscreen_active = false;

            SDL_SetDisplayModeForDisplay(display, NULL);
        }
        if (commit) {
            SDL_FullscreenResult ret = SDL_FULLSCREEN_SUCCEEDED;
            if (_this->SetWindowFullscreen) {
                SDL_VideoDisplay *full_screen_display = display ? display : SDL_GetVideoDisplayForFullscreenWindow(window);
                if (full_screen_display) {
                    ret = _this->SetWindowFullscreen(_this, window, full_screen_display, SDL_FULLSCREEN_OP_LEAVE);
                }
            } else {
                resized = true;
            }

            if (ret == SDL_FULLSCREEN_SUCCEEDED) {
                // Window left fullscreen immediately upon return. If the driver hasn't already sent the event, do so now.
                if (window->flags & SDL_WINDOW_FULLSCREEN) {
                    SDL_SendWindowEvent(window, SDL_EVENT_WINDOW_LEAVE_FULLSCREEN, 0, 0);
                }
            } else if (ret == SDL_FULLSCREEN_FAILED) {
                goto error;
            }
        }

        if (!(window->flags & SDL_WINDOW_FULLSCREEN)) {
            if (display) {
                display->fullscreen_window = NULL;
            }

            if (!SDL_SendsFullscreenDimensions(_this)) {
                SDL_SendWindowEvent(window, SDL_EVENT_WINDOW_MOVED, window->windowed.x, window->windowed.y);
                if (resized) {
                    SDL_SendWindowEvent(window, SDL_EVENT_WINDOW_RESIZED, window->windowed.w, window->windowed.h);
                } else {
                    SDL_OnWindowResized(window);
                }
            }

            // Restore the cursor position if we've exited fullscreen on a display
            if (display && !SDL_DisableMouseWarpOnFullscreenTransitions(_this)) {
                SDL_RestoreMousePosition(window);
            }
        }
    }

done:
    window->last_fullscreen_exclusive_display = display && (window->flags & SDL_WINDOW_FULLSCREEN) && window->fullscreen_exclusive ? display->id : 0;
    return true;

error:
    if (fullscreen) {
        // Something went wrong and the window is no longer fullscreen.
        SDL_UpdateFullscreenMode(window, SDL_FULLSCREEN_OP_LEAVE, commit);
    }
    return false;
}

bool SDL_SetWindowFullscreenMode(SDL_Window *window, const SDL_DisplayMode *mode)
{
    CHECK_WINDOW_MAGIC(window, false);
    CHECK_WINDOW_NOT_POPUP(window, false);

    if (mode) {
        if (!SDL_GetFullscreenModeMatch(mode)) {
            return SDL_SetError("Invalid fullscreen display mode");
        }

        // Save the mode so we can look up the closest match later
        SDL_copyp(&window->requested_fullscreen_mode, mode);
    } else {
        SDL_zero(window->requested_fullscreen_mode);
    }

    /* Copy to the current mode now, in case an asynchronous fullscreen window request
     * is in progress. It will be overwritten if a new request is made.
     */
    SDL_copyp(&window->current_fullscreen_mode, &window->requested_fullscreen_mode);
    if (SDL_WINDOW_FULLSCREEN_VISIBLE(window)) {
        SDL_UpdateFullscreenMode(window, SDL_FULLSCREEN_OP_UPDATE, true);
        SDL_SyncIfRequired(window);
    }

    return true;
}

const SDL_DisplayMode *SDL_GetWindowFullscreenMode(SDL_Window *window)
{
    CHECK_WINDOW_MAGIC(window, NULL);
    CHECK_WINDOW_NOT_POPUP(window, NULL);

    if (window->flags & SDL_WINDOW_FULLSCREEN) {
        return SDL_GetFullscreenModeMatch(&window->current_fullscreen_mode);
    } else {
        return SDL_GetFullscreenModeMatch(&window->requested_fullscreen_mode);
    }
}

void *SDL_GetWindowICCProfile(SDL_Window *window, size_t *size)
{
    if (!_this->GetWindowICCProfile) {
        SDL_Unsupported();
        return NULL;
    }
    return _this->GetWindowICCProfile(_this, window, size);
}

SDL_PixelFormat SDL_GetWindowPixelFormat(SDL_Window *window)
{
    SDL_DisplayID displayID;
    const SDL_DisplayMode *mode;

    CHECK_WINDOW_MAGIC(window, SDL_PIXELFORMAT_UNKNOWN);

    displayID = SDL_GetDisplayForWindow(window);
    mode = SDL_GetCurrentDisplayMode(displayID);
    if (mode) {
        return mode->format;
    } else {
        return SDL_PIXELFORMAT_UNKNOWN;
    }
}

#define CREATE_FLAGS \
    (SDL_WINDOW_OPENGL | SDL_WINDOW_BORDERLESS | SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY | SDL_WINDOW_ALWAYS_ON_TOP | SDL_WINDOW_POPUP_MENU | SDL_WINDOW_UTILITY | SDL_WINDOW_TOOLTIP | SDL_WINDOW_VULKAN | SDL_WINDOW_MINIMIZED | SDL_WINDOW_METAL | SDL_WINDOW_TRANSPARENT | SDL_WINDOW_NOT_FOCUSABLE)

static SDL_INLINE bool IsAcceptingDragAndDrop(void)
{
    if (SDL_EventEnabled(SDL_EVENT_DROP_FILE) || SDL_EventEnabled(SDL_EVENT_DROP_TEXT)) {
        return true;
    }
    return false;
}

// prepare a newly-created window
static SDL_INLINE void PrepareDragAndDropSupport(SDL_Window *window)
{
    if (_this->AcceptDragAndDrop) {
        _this->AcceptDragAndDrop(window, IsAcceptingDragAndDrop());
    }
}

// toggle d'n'd for all existing windows.
void SDL_ToggleDragAndDropSupport(void)
{
    if (_this && _this->AcceptDragAndDrop) {
        const bool enable = IsAcceptingDragAndDrop();
        SDL_Window *window;
        for (window = _this->windows; window; window = window->next) {
            _this->AcceptDragAndDrop(window, enable);
        }
    }
}

SDL_Window ** SDLCALL SDL_GetWindows(int *count)
{
    if (count) {
        *count = 0;
    }

    if (!_this) {
        SDL_UninitializedVideo();
        return NULL;
    }

    SDL_Window *window;
    int num_added = 0;
    int num_windows = 0;
    for (window = _this->windows; window; window = window->next) {
        ++num_windows;
    }

    SDL_Window **windows = (SDL_Window **)SDL_malloc((num_windows + 1) * sizeof(*windows));
    if (!windows) {
        return NULL;
    }

    for (window = _this->windows; window; window = window->next) {
        windows[num_added++] = window;
        if (num_added == num_windows) {
            // Race condition? Multi-threading not supported, ignore it
            break;
        }
    }
    windows[num_added] = NULL;

    if (count) {
        *count = num_added;
    }
    return windows;
}

static void ApplyWindowFlags(SDL_Window *window, SDL_WindowFlags flags)
{
    if (!SDL_WINDOW_IS_POPUP(window)) {
        if (!(flags & (SDL_WINDOW_MINIMIZED | SDL_WINDOW_MAXIMIZED))) {
            SDL_RestoreWindow(window);
        }
        if (flags & SDL_WINDOW_MAXIMIZED) {
            SDL_MaximizeWindow(window);
        }

        SDL_SetWindowFullscreen(window, (flags & SDL_WINDOW_FULLSCREEN) != 0);

        if (flags & SDL_WINDOW_MINIMIZED) {
            SDL_MinimizeWindow(window);
        }

        if (flags & SDL_WINDOW_MODAL) {
            SDL_SetWindowModal(window, true);
        }

        if (flags & SDL_WINDOW_MOUSE_GRABBED) {
            SDL_SetWindowMouseGrab(window, true);
        }
        if (flags & SDL_WINDOW_KEYBOARD_GRABBED) {
            SDL_SetWindowKeyboardGrab(window, true);
        }
    }
}

static void SDL_FinishWindowCreation(SDL_Window *window, SDL_WindowFlags flags)
{
    PrepareDragAndDropSupport(window);

    if (window->flags & SDL_WINDOW_EXTERNAL) {
        // Whoever has created the window has already applied whatever flags are needed
    } else {
        ApplyWindowFlags(window, flags);
        if (!(flags & SDL_WINDOW_HIDDEN)) {
            SDL_ShowWindow(window);
        }
    }
}

static bool SDL_ContextNotSupported(const char *name)
{
    return SDL_SetError("%s support is either not configured in SDL "
                        "or not available in current SDL video driver "
                        "(%s) or platform",
                        name,
                        _this->name);
}

static bool SDL_DllNotSupported(const char *name)
{
    return SDL_SetError("No dynamic %s support in current SDL video driver (%s)", name, _this->name);
}

static struct {
    const char *property_name;
    SDL_WindowFlags flag;
    bool invert_value;
} SDL_WindowFlagProperties[] = {
    { SDL_PROP_WINDOW_CREATE_ALWAYS_ON_TOP_BOOLEAN,      SDL_WINDOW_ALWAYS_ON_TOP,       false },
    { SDL_PROP_WINDOW_CREATE_BORDERLESS_BOOLEAN,         SDL_WINDOW_BORDERLESS,          false },
    { SDL_PROP_WINDOW_CREATE_FOCUSABLE_BOOLEAN,          SDL_WINDOW_NOT_FOCUSABLE,       true },
    { SDL_PROP_WINDOW_CREATE_FULLSCREEN_BOOLEAN,         SDL_WINDOW_FULLSCREEN,          false },
    { SDL_PROP_WINDOW_CREATE_HIDDEN_BOOLEAN,             SDL_WINDOW_HIDDEN,              false },
    { SDL_PROP_WINDOW_CREATE_HIGH_PIXEL_DENSITY_BOOLEAN, SDL_WINDOW_HIGH_PIXEL_DENSITY,  false },
    { SDL_PROP_WINDOW_CREATE_MAXIMIZED_BOOLEAN,          SDL_WINDOW_MAXIMIZED,           false },
    { SDL_PROP_WINDOW_CREATE_MENU_BOOLEAN,               SDL_WINDOW_POPUP_MENU,          false },
    { SDL_PROP_WINDOW_CREATE_METAL_BOOLEAN,              SDL_WINDOW_METAL,               false },
    { SDL_PROP_WINDOW_CREATE_MINIMIZED_BOOLEAN,          SDL_WINDOW_MINIMIZED,           false },
    { SDL_PROP_WINDOW_CREATE_MODAL_BOOLEAN,              SDL_WINDOW_MODAL,               false },
    { SDL_PROP_WINDOW_CREATE_MOUSE_GRABBED_BOOLEAN,      SDL_WINDOW_MOUSE_GRABBED,       false },
    { SDL_PROP_WINDOW_CREATE_OPENGL_BOOLEAN,             SDL_WINDOW_OPENGL,              false },
    { SDL_PROP_WINDOW_CREATE_RESIZABLE_BOOLEAN,          SDL_WINDOW_RESIZABLE,           false },
    { SDL_PROP_WINDOW_CREATE_TRANSPARENT_BOOLEAN,        SDL_WINDOW_TRANSPARENT,         false },
    { SDL_PROP_WINDOW_CREATE_TOOLTIP_BOOLEAN,            SDL_WINDOW_TOOLTIP,             false },
    { SDL_PROP_WINDOW_CREATE_UTILITY_BOOLEAN,            SDL_WINDOW_UTILITY,             false },
    { SDL_PROP_WINDOW_CREATE_VULKAN_BOOLEAN,             SDL_WINDOW_VULKAN,              false }
};

static SDL_WindowFlags SDL_GetWindowFlagProperties(SDL_PropertiesID props)
{
    unsigned i;
    SDL_WindowFlags flags = (SDL_WindowFlags)SDL_GetNumberProperty(props, SDL_PROP_WINDOW_CREATE_FLAGS_NUMBER, 0);

    for (i = 0; i < SDL_arraysize(SDL_WindowFlagProperties); ++i) {
        if (SDL_WindowFlagProperties[i].invert_value) {
            if (!SDL_GetBooleanProperty(props, SDL_WindowFlagProperties[i].property_name, true)) {
                flags |= SDL_WindowFlagProperties[i].flag;
            }
        } else {
            if (SDL_GetBooleanProperty(props, SDL_WindowFlagProperties[i].property_name, false)) {
                flags |= SDL_WindowFlagProperties[i].flag;
            }
        }
    }
    return flags;
}

SDL_Window *SDL_CreateWindowWithProperties(SDL_PropertiesID props)
{
    SDL_Window *window;
    const char *title = SDL_GetStringProperty(props, SDL_PROP_WINDOW_CREATE_TITLE_STRING, NULL);
    int x = (int)SDL_GetNumberProperty(props, SDL_PROP_WINDOW_CREATE_X_NUMBER, SDL_WINDOWPOS_UNDEFINED);
    int y = (int)SDL_GetNumberProperty(props, SDL_PROP_WINDOW_CREATE_Y_NUMBER, SDL_WINDOWPOS_UNDEFINED);
    int w = (int)SDL_GetNumberProperty(props, SDL_PROP_WINDOW_CREATE_WIDTH_NUMBER, 0);
    int h = (int)SDL_GetNumberProperty(props, SDL_PROP_WINDOW_CREATE_HEIGHT_NUMBER, 0);
    SDL_Window *parent = (SDL_Window *)SDL_GetPointerProperty(props, SDL_PROP_WINDOW_CREATE_PARENT_POINTER, NULL);
    SDL_WindowFlags flags = SDL_GetWindowFlagProperties(props);
    SDL_WindowFlags type_flags, graphics_flags;
    bool undefined_x = false;
    bool undefined_y = false;
    bool external_graphics_context = SDL_GetBooleanProperty(props, SDL_PROP_WINDOW_CREATE_EXTERNAL_GRAPHICS_CONTEXT_BOOLEAN, false);

    if (!_this) {
        // Initialize the video system if needed
        if (!SDL_Init(SDL_INIT_VIDEO)) {
            return NULL;
        }

        // Make clang-tidy happy
        if (!_this) {
            return NULL;
        }
    }

    if ((flags & SDL_WINDOW_MODAL) && !SDL_ObjectValid(parent, SDL_OBJECT_TYPE_WINDOW)) {
        SDL_SetError("Modal windows must specify a parent window");
        return NULL;
    }

    if ((flags & (SDL_WINDOW_TOOLTIP | SDL_WINDOW_POPUP_MENU)) != 0) {
        if (!(_this->device_caps & VIDEO_DEVICE_CAPS_HAS_POPUP_WINDOW_SUPPORT)) {
            SDL_Unsupported();
            return NULL;
        }

        // Tooltip and popup menu window must specify a parent window
        if (!SDL_ObjectValid(parent, SDL_OBJECT_TYPE_WINDOW)) {
            SDL_SetError("Tooltip and popup menu windows must specify a parent window");
            return NULL;
        }

        // Remove invalid flags
        flags &= ~(SDL_WINDOW_MINIMIZED | SDL_WINDOW_MAXIMIZED | SDL_WINDOW_FULLSCREEN | SDL_WINDOW_BORDERLESS);
    }

    // Ensure no more than one of these flags is set
    type_flags = flags & (SDL_WINDOW_UTILITY | SDL_WINDOW_TOOLTIP | SDL_WINDOW_POPUP_MENU | SDL_WINDOW_MODAL);
    if (type_flags & (type_flags - 1)) {
        SDL_SetError("Conflicting window type flags specified: 0x%.8x", (unsigned int)type_flags);
        return NULL;
    }

    // Make sure the display list is up to date for window placement
    if (_this->RefreshDisplays) {
        _this->RefreshDisplays(_this);
    }

    // Some platforms can't create zero-sized windows
    if (w < 1) {
        w = 1;
    }
    if (h < 1) {
        h = 1;
    }

    if (SDL_WINDOWPOS_ISUNDEFINED(x) || SDL_WINDOWPOS_ISUNDEFINED(y) ||
        SDL_WINDOWPOS_ISCENTERED(x) || SDL_WINDOWPOS_ISCENTERED(y)) {
        SDL_DisplayID displayID = 0;
        SDL_Rect bounds;

        if ((SDL_WINDOWPOS_ISUNDEFINED(x) || SDL_WINDOWPOS_ISCENTERED(x)) && (x & 0xFFFF)) {
            displayID = (x & 0xFFFF);
        } else if ((SDL_WINDOWPOS_ISUNDEFINED(y) || SDL_WINDOWPOS_ISCENTERED(y)) && (y & 0xFFFF)) {
            displayID = (y & 0xFFFF);
        }
        if (displayID == 0 || SDL_GetDisplayIndex(displayID) < 0) {
            displayID = SDL_GetPrimaryDisplay();
        }

        SDL_zero(bounds);
        SDL_GetDisplayUsableBounds(displayID, &bounds);
        if (w > bounds.w || h > bounds.h) {
            // This window is larger than the usable bounds, just center on the display
            SDL_GetDisplayBounds(displayID, &bounds);
        }
        if (SDL_WINDOWPOS_ISCENTERED(x) || SDL_WINDOWPOS_ISUNDEFINED(x)) {
            if (SDL_WINDOWPOS_ISUNDEFINED(x)) {
                undefined_x = true;
            }
            x = bounds.x + (bounds.w - w) / 2;
        }
        if (SDL_WINDOWPOS_ISCENTERED(y) || SDL_WINDOWPOS_ISUNDEFINED(y)) {
            if (SDL_WINDOWPOS_ISUNDEFINED(y)) {
                undefined_y = true;
            }
            y = bounds.y + (bounds.h - h) / 2;
        }
    }

    // ensure no more than one of these flags is set
    graphics_flags = flags & (SDL_WINDOW_OPENGL | SDL_WINDOW_METAL | SDL_WINDOW_VULKAN);
    if (graphics_flags & (graphics_flags - 1)) {
        SDL_SetError("Conflicting window graphics flags specified: 0x%.8x", (unsigned int)graphics_flags);
        return NULL;
    }

    // Some platforms have certain graphics backends enabled by default
    if (!graphics_flags && !external_graphics_context) {
        flags |= SDL_DefaultGraphicsBackends(_this);
    }

    if (flags & SDL_WINDOW_OPENGL) {
        if (!_this->GL_CreateContext) {
            SDL_ContextNotSupported("OpenGL");
            return NULL;
        }
        if (!SDL_GL_LoadLibrary(NULL)) {
            return NULL;
        }
    }

    if (flags & SDL_WINDOW_VULKAN) {
        if (!_this->Vulkan_CreateSurface) {
            SDL_ContextNotSupported("Vulkan");
            return NULL;
        }
        if (!SDL_Vulkan_LoadLibrary(NULL)) {
            return NULL;
        }
    }

    if (flags & SDL_WINDOW_METAL) {
        if (!_this->Metal_CreateView) {
            SDL_ContextNotSupported("Metal");
            return NULL;
        }
    }

    window = (SDL_Window *)SDL_calloc(1, sizeof(*window));
    if (!window) {
        return NULL;
    }
    SDL_SetObjectValid(window, SDL_OBJECT_TYPE_WINDOW, true);
    window->id = SDL_GetNextObjectID();
    window->floating.x = window->windowed.x = window->x = x;
    window->floating.y = window->windowed.y = window->y = y;
    window->floating.w = window->windowed.w = window->w = w;
    window->floating.h = window->windowed.h = window->h = h;
    window->undefined_x = undefined_x;
    window->undefined_y = undefined_y;

    SDL_VideoDisplay *display = SDL_GetVideoDisplayForWindow(window);
    if (display) {
        SDL_SetWindowHDRProperties(window, &display->HDR, false);
    }

    if (flags & SDL_WINDOW_FULLSCREEN || IsFullscreenOnly(_this)) {
        SDL_Rect bounds;

        SDL_GetDisplayBounds(display ? display->id : SDL_GetPrimaryDisplay(), &bounds);
        window->x = bounds.x;
        window->y = bounds.y;
        window->w = bounds.w;
        window->h = bounds.h;
        window->pending_flags |= SDL_WINDOW_FULLSCREEN;
        flags |= SDL_WINDOW_FULLSCREEN;
    }

    window->flags = ((flags & CREATE_FLAGS) | SDL_WINDOW_HIDDEN);
    window->display_scale = 1.0f;
    window->opacity = 1.0f;
    window->next = _this->windows;
    window->is_destroying = false;
    window->last_displayID = SDL_GetDisplayForWindow(window);
    window->external_graphics_context = external_graphics_context;
    window->constrain_popup = SDL_GetBooleanProperty(props, SDL_PROP_WINDOW_CREATE_CONSTRAIN_POPUP_BOOLEAN, true);

    if (_this->windows) {
        _this->windows->prev = window;
    }
    _this->windows = window;

    // Set the parent before creation.
    SDL_UpdateWindowHierarchy(window, parent);

    if (_this->CreateSDLWindow && !_this->CreateSDLWindow(_this, window, props)) {
        SDL_DestroyWindow(window);
        return NULL;
    }

    /* Clear minimized if not on windows, only windows handles it at create rather than FinishWindowCreation,
     * but it's important or window focus will get broken on windows!
     */
#if !defined(SDL_PLATFORM_WINDOWS)
    if (window->flags & SDL_WINDOW_MINIMIZED) {
        window->flags &= ~SDL_WINDOW_MINIMIZED;
    }
#endif

    if (title) {
        SDL_SetWindowTitle(window, title);
    }
    SDL_FinishWindowCreation(window, flags);

    // Make sure window pixel size is up to date
    SDL_CheckWindowPixelSizeChanged(window);

#ifdef SDL_VIDEO_DRIVER_UIKIT
    SDL_UpdateLifecycleObserver();
#endif

    SDL_ClearError();

    return window;
}

SDL_Window *SDL_CreateWindow(const char *title, int w, int h, SDL_WindowFlags flags)
{
    SDL_Window *window;
    SDL_PropertiesID props = SDL_CreateProperties();
    if (title && *title) {
        SDL_SetStringProperty(props, SDL_PROP_WINDOW_CREATE_TITLE_STRING, title);
    }
    SDL_SetNumberProperty(props, SDL_PROP_WINDOW_CREATE_WIDTH_NUMBER, w);
    SDL_SetNumberProperty(props, SDL_PROP_WINDOW_CREATE_HEIGHT_NUMBER, h);
    SDL_SetNumberProperty(props, SDL_PROP_WINDOW_CREATE_FLAGS_NUMBER, flags);
    window = SDL_CreateWindowWithProperties(props);
    SDL_DestroyProperties(props);
    return window;
}

SDL_Window *SDL_CreatePopupWindow(SDL_Window *parent, int offset_x, int offset_y, int w, int h, SDL_WindowFlags flags)
{
    SDL_Window *window;
    SDL_PropertiesID props = SDL_CreateProperties();

    // Popups must specify either the tooltip or popup menu window flags
    if (!(flags & (SDL_WINDOW_TOOLTIP | SDL_WINDOW_POPUP_MENU))) {
        SDL_SetError("Popup windows must specify either the 'SDL_WINDOW_TOOLTIP' or the 'SDL_WINDOW_POPUP_MENU' flag");
        return NULL;
    }

    SDL_SetPointerProperty(props, SDL_PROP_WINDOW_CREATE_PARENT_POINTER, parent);
    SDL_SetNumberProperty(props, SDL_PROP_WINDOW_CREATE_X_NUMBER, offset_x);
    SDL_SetNumberProperty(props, SDL_PROP_WINDOW_CREATE_Y_NUMBER, offset_y);
    SDL_SetNumberProperty(props, SDL_PROP_WINDOW_CREATE_WIDTH_NUMBER, w);
    SDL_SetNumberProperty(props, SDL_PROP_WINDOW_CREATE_HEIGHT_NUMBER, h);
    SDL_SetNumberProperty(props, SDL_PROP_WINDOW_CREATE_FLAGS_NUMBER, flags);
    window = SDL_CreateWindowWithProperties(props);
    SDL_DestroyProperties(props);
    return window;
}

bool SDL_RecreateWindow(SDL_Window *window, SDL_WindowFlags flags)
{
    bool loaded_opengl = false;
    bool need_gl_unload = false;
    bool need_gl_load = false;
    bool loaded_vulkan = false;
    bool need_vulkan_unload = false;
    bool need_vulkan_load = false;
    SDL_WindowFlags graphics_flags;

    // ensure no more than one of these flags is set
    graphics_flags = flags & (SDL_WINDOW_OPENGL | SDL_WINDOW_METAL | SDL_WINDOW_VULKAN);
    if (graphics_flags & (graphics_flags - 1)) {
        return SDL_SetError("Conflicting window flags specified");
    }

    if ((flags & SDL_WINDOW_OPENGL) && !_this->GL_CreateContext) {
        return SDL_ContextNotSupported("OpenGL");
    }
    if ((flags & SDL_WINDOW_VULKAN) && !_this->Vulkan_CreateSurface) {
        return SDL_ContextNotSupported("Vulkan");
    }
    if ((flags & SDL_WINDOW_METAL) && !_this->Metal_CreateView) {
        return SDL_ContextNotSupported("Metal");
    }

    if (window->flags & SDL_WINDOW_EXTERNAL) {
        // Can't destroy and re-create external windows, hrm
        flags |= SDL_WINDOW_EXTERNAL;
    } else {
        flags &= ~SDL_WINDOW_EXTERNAL;
    }

    // If this is a modal dialog, clear the modal status.
    if (window->flags & SDL_WINDOW_MODAL) {
        SDL_SetWindowModal(window, false);
    }

    // Restore video mode, etc.
    if (!(window->flags & SDL_WINDOW_EXTERNAL)) {
        const bool restore_on_show = window->restore_on_show;
        SDL_HideWindow(window);
        window->restore_on_show = restore_on_show;
    }

    // Tear down the old native window
    SDL_DestroyWindowSurface(window);

    if ((window->flags & SDL_WINDOW_OPENGL) != (flags & SDL_WINDOW_OPENGL)) {
        if (flags & SDL_WINDOW_OPENGL) {
            need_gl_load = true;
        } else {
            need_gl_unload = true;
        }
    } else if (window->flags & SDL_WINDOW_OPENGL) {
        need_gl_unload = true;
        need_gl_load = true;
    }

    if ((window->flags & SDL_WINDOW_VULKAN) != (flags & SDL_WINDOW_VULKAN)) {
        if (flags & SDL_WINDOW_VULKAN) {
            need_vulkan_load = true;
        } else {
            need_vulkan_unload = true;
        }
    } else if (window->flags & SDL_WINDOW_VULKAN) {
        need_vulkan_unload = true;
        need_vulkan_load = true;
    }

    if (need_gl_unload) {
        SDL_GL_UnloadLibrary();
    }

    if (need_vulkan_unload) {
        SDL_Vulkan_UnloadLibrary();
    }

    if (_this->DestroyWindow && !(flags & SDL_WINDOW_EXTERNAL)) {
        _this->DestroyWindow(_this, window);
    }

    if (need_gl_load) {
        if (!SDL_GL_LoadLibrary(NULL)) {
            return false;
        }
        loaded_opengl = true;
    }

    if (need_vulkan_load) {
        if (!SDL_Vulkan_LoadLibrary(NULL)) {
            return false;
        }
        loaded_vulkan = true;
    }

    window->flags = ((flags & CREATE_FLAGS) | SDL_WINDOW_HIDDEN);
    window->is_destroying = false;

    if (_this->CreateSDLWindow && !(flags & SDL_WINDOW_EXTERNAL)) {
        /* Reset the window size to the original floating value, so the
         * recreated window has the proper base size.
         */
        window->x = window->windowed.x = window->floating.x;
        window->y = window->windowed.y = window->floating.y;
        window->w = window->windowed.w = window->floating.w;
        window->h = window->windowed.h = window->floating.h;

        if (!_this->CreateSDLWindow(_this, window, 0)) {
            if (loaded_opengl) {
                SDL_GL_UnloadLibrary();
                window->flags &= ~SDL_WINDOW_OPENGL;
            }
            if (loaded_vulkan) {
                SDL_Vulkan_UnloadLibrary();
                window->flags &= ~SDL_WINDOW_VULKAN;
            }
            return false;
        }
    }

    if (flags & SDL_WINDOW_EXTERNAL) {
        window->flags |= SDL_WINDOW_EXTERNAL;
    }

    if (_this->SetWindowTitle && window->title) {
        _this->SetWindowTitle(_this, window);
    }

    if (_this->SetWindowIcon && window->icon) {
        _this->SetWindowIcon(_this, window, window->icon);
    }

    if (_this->SetWindowMinimumSize && (window->min_w || window->min_h)) {
        _this->SetWindowMinimumSize(_this, window);
    }

    if (_this->SetWindowMaximumSize && (window->max_w || window->max_h)) {
        _this->SetWindowMaximumSize(_this, window);
    }

    if (_this->SetWindowAspectRatio && (window->min_aspect > 0.0f || window->max_aspect > 0.0f)) {
        _this->SetWindowAspectRatio(_this, window);
    }

    if (window->hit_test) {
        _this->SetWindowHitTest(window, true);
    }

    SDL_FinishWindowCreation(window, flags);

    return true;
}

bool SDL_HasWindows(void)
{
    return _this && _this->windows;
}

SDL_WindowID SDL_GetWindowID(SDL_Window *window)
{
    CHECK_WINDOW_MAGIC(window, 0);

    return window->id;
}

SDL_Window *SDL_GetWindowFromID(SDL_WindowID id)
{
    SDL_Window *window;

    if (!_this) {
        SDL_UninitializedVideo();
        return NULL;
    }
    if (id) {
        for (window = _this->windows; window; window = window->next) {
            if (window->id == id) {
                return window;
            }
        }
    }
    SDL_SetError("Invalid window ID");                                 \
    return NULL;
}

SDL_Window *SDL_GetWindowParent(SDL_Window *window)
{
    CHECK_WINDOW_MAGIC(window, NULL);

    return window->parent;
}

SDL_PropertiesID SDL_GetWindowProperties(SDL_Window *window)
{
    CHECK_WINDOW_MAGIC(window, 0);

    if (window->props == 0) {
        window->props = SDL_CreateProperties();
    }
    return window->props;
}

SDL_WindowFlags SDL_GetWindowFlags(SDL_Window *window)
{
    CHECK_WINDOW_MAGIC(window, 0);

    return window->flags | window->pending_flags;
}

bool SDL_SetWindowTitle(SDL_Window *window, const char *title)
{
    CHECK_WINDOW_MAGIC(window, false);
    CHECK_WINDOW_NOT_POPUP(window, false);

    if (title == window->title) {
        return true;
    }
    if (!title) {
        title = "";
    }
    if (window->title && SDL_strcmp(title, window->title) == 0) {
        return true;
    }

    SDL_free(window->title);

    window->title = SDL_strdup(title);

    if (_this->SetWindowTitle) {
        _this->SetWindowTitle(_this, window);
    }
    return true;
}

const char *SDL_GetWindowTitle(SDL_Window *window)
{
    CHECK_WINDOW_MAGIC(window, "");

    return window->title ? window->title : "";
}

bool SDL_SetWindowIcon(SDL_Window *window, SDL_Surface *icon)
{
    CHECK_WINDOW_MAGIC(window, false);

    if (!icon) {
        return SDL_InvalidParamError("icon");
    }

    SDL_DestroySurface(window->icon);

    // Convert the icon into ARGB8888
    window->icon = SDL_ConvertSurface(icon, SDL_PIXELFORMAT_ARGB8888);
    if (!window->icon) {
        return false;
    }

    if (!_this->SetWindowIcon) {
        return SDL_Unsupported();
    }

    return _this->SetWindowIcon(_this, window, window->icon);
}

bool SDL_SetWindowPosition(SDL_Window *window, int x, int y)
{
    SDL_DisplayID original_displayID;

    CHECK_WINDOW_MAGIC(window, false);

    const int w = window->last_size_pending ? window->pending.w : window->windowed.w;
    const int h = window->last_size_pending ? window->pending.h : window->windowed.h;

    original_displayID = SDL_GetDisplayForWindow(window);

    if (SDL_WINDOWPOS_ISUNDEFINED(x)) {
        x = window->windowed.x;
    }
    if (SDL_WINDOWPOS_ISUNDEFINED(y)) {
        y = window->windowed.y;
    }
    if (SDL_WINDOWPOS_ISCENTERED(x) || SDL_WINDOWPOS_ISCENTERED(y)) {
        SDL_DisplayID displayID = original_displayID;
        SDL_Rect bounds;

        if (SDL_WINDOWPOS_ISCENTERED(x) && (x & 0xFFFF)) {
            displayID = (x & 0xFFFF);
        } else if (SDL_WINDOWPOS_ISCENTERED(y) && (y & 0xFFFF)) {
            displayID = (y & 0xFFFF);
        }
        if (displayID == 0 || SDL_GetDisplayIndex(displayID) < 0) {
            displayID = SDL_GetPrimaryDisplay();
        }

        SDL_zero(bounds);
        if (!SDL_GetDisplayUsableBounds(displayID, &bounds) || w > bounds.w || h > bounds.h) {
            if (!SDL_GetDisplayBounds(displayID, &bounds)) {
                return false;
            }
        }
        if (SDL_WINDOWPOS_ISCENTERED(x)) {
            x = bounds.x + (bounds.w - w) / 2;
        }
        if (SDL_WINDOWPOS_ISCENTERED(y)) {
            y = bounds.y + (bounds.h - h) / 2;
        }
    }

    window->pending.x = x;
    window->pending.y = y;
    window->undefined_x = false;
    window->undefined_y = false;
    window->last_position_pending = true;

    if (_this->SetWindowPosition) {
        const bool result = _this->SetWindowPosition(_this, window);
        if (result) {
            SDL_SyncIfRequired(window);
        }
        return result;
    }

    return SDL_Unsupported();
}

bool SDL_GetWindowPosition(SDL_Window *window, int *x, int *y)
{
    CHECK_WINDOW_MAGIC(window, false);

    // Fullscreen windows are always at their display's origin
    if (window->flags & SDL_WINDOW_FULLSCREEN) {
        SDL_DisplayID displayID;

        if (x) {
            *x = 0;
        }
        if (y) {
            *y = 0;
        }

        /* Find the window's monitor and update to the
           monitor offset. */
        displayID = SDL_GetDisplayForWindow(window);
        if (displayID != 0) {
            SDL_Rect bounds;

            SDL_zero(bounds);

            SDL_GetDisplayBounds(displayID, &bounds);
            if (x) {
                *x = bounds.x;
            }
            if (y) {
                *y = bounds.y;
            }
        }
    } else {
        const bool use_pending = (window->flags & SDL_WINDOW_HIDDEN) && window->last_position_pending;
        if (x) {
            *x = use_pending ? window->pending.x : window->x;
        }
        if (y) {
            *y = use_pending ? window->pending.y : window->y;
        }
    }
    return true;
}

bool SDL_SetWindowBordered(SDL_Window *window, bool bordered)
{
    CHECK_WINDOW_MAGIC(window, false);
    CHECK_WINDOW_NOT_POPUP(window, false);

    const bool want = (bordered != false); // normalize the flag.
    const bool have = !(window->flags & SDL_WINDOW_BORDERLESS);
    if ((want != have) && (_this->SetWindowBordered)) {
        if (want) {
            window->flags &= ~SDL_WINDOW_BORDERLESS;
        } else {
            window->flags |= SDL_WINDOW_BORDERLESS;
        }
        _this->SetWindowBordered(_this, window, want);
    }

    return true;
}

bool SDL_SetWindowResizable(SDL_Window *window, bool resizable)
{
    CHECK_WINDOW_MAGIC(window, false);
    CHECK_WINDOW_NOT_POPUP(window, false);

    const bool want = (resizable != false); // normalize the flag.
    const bool have = ((window->flags & SDL_WINDOW_RESIZABLE) != 0);
    if ((want != have) && (_this->SetWindowResizable)) {
        if (want) {
            window->flags |= SDL_WINDOW_RESIZABLE;
        } else {
            window->flags &= ~SDL_WINDOW_RESIZABLE;
            SDL_copyp(&window->windowed, &window->floating);
        }
        _this->SetWindowResizable(_this, window, want);
    }

    return true;
}

bool SDL_SetWindowAlwaysOnTop(SDL_Window *window, bool on_top)
{
    CHECK_WINDOW_MAGIC(window, false);
    CHECK_WINDOW_NOT_POPUP(window, false);

    const bool want = (on_top != false); // normalize the flag.
    const bool have = ((window->flags & SDL_WINDOW_ALWAYS_ON_TOP) != 0);
    if ((want != have) && (_this->SetWindowAlwaysOnTop)) {
        if (want) {
            window->flags |= SDL_WINDOW_ALWAYS_ON_TOP;
        } else {
            window->flags &= ~SDL_WINDOW_ALWAYS_ON_TOP;
        }
        _this->SetWindowAlwaysOnTop(_this, window, want);
    }

    return true;
}

bool SDL_SetWindowSize(SDL_Window *window, int w, int h)
{
    CHECK_WINDOW_MAGIC(window, false);

    if (w <= 0) {
        return SDL_InvalidParamError("w");
    }
    if (h <= 0) {
        return SDL_InvalidParamError("h");
    }

    // It is possible for the aspect ratio constraints to not satisfy the size constraints.
    // The size constraints will override the aspect ratio constraints so we will apply the
    // the aspect ratio constraints first
    float new_aspect = w / (float)h;
    if (window->max_aspect > 0.0f && new_aspect > window->max_aspect) {
        w = (int)SDL_roundf(h * window->max_aspect);
    } else if (window->min_aspect > 0.0f && new_aspect < window->min_aspect) {
        h = (int)SDL_roundf(w / window->min_aspect);
    }

    // Make sure we don't exceed any window size limits
    if (window->min_w && w < window->min_w) {
        w = window->min_w;
    }
    if (window->max_w && w > window->max_w) {
        w = window->max_w;
    }
    if (window->min_h && h < window->min_h) {
        h = window->min_h;
    }
    if (window->max_h && h > window->max_h) {
        h = window->max_h;
    }

    window->last_size_pending = true;
    window->pending.w = w;
    window->pending.h = h;

    if (_this->SetWindowSize) {
        _this->SetWindowSize(_this, window);
        SDL_SyncIfRequired(window);
    } else {
        return SDL_Unsupported();
    }
    return true;
}

bool SDL_GetWindowSize(SDL_Window *window, int *w, int *h)
{
    CHECK_WINDOW_MAGIC(window, false);
    if (w) {
        *w = window->w;
    }
    if (h) {
        *h = window->h;
    }
    return true;
}

bool SDL_SetWindowAspectRatio(SDL_Window *window, float min_aspect, float max_aspect)
{
    CHECK_WINDOW_MAGIC(window, false);

    window->min_aspect = min_aspect;
    window->max_aspect = max_aspect;
    if (_this->SetWindowAspectRatio) {
        _this->SetWindowAspectRatio(_this, window);
    }
    return SDL_SetWindowSize(window, window->floating.w, window->floating.h);
}

bool SDL_GetWindowAspectRatio(SDL_Window *window, float *min_aspect, float *max_aspect)
{
    CHECK_WINDOW_MAGIC(window, false);

    if (min_aspect) {
        *min_aspect = window->min_aspect;
    }
    if (max_aspect) {
        *max_aspect = window->max_aspect;
    }
    return true;
}

bool SDL_GetWindowBordersSize(SDL_Window *window, int *top, int *left, int *bottom, int *right)
{
    int dummy = 0;

    if (!top) {
        top = &dummy;
    }
    if (!left) {
        left = &dummy;
    }
    if (!right) {
        right = &dummy;
    }
    if (!bottom) {
        bottom = &dummy;
    }

    // Always initialize, so applications don't have to care
    *top = *left = *bottom = *right = 0;

    CHECK_WINDOW_MAGIC(window, false);

    if (!_this->GetWindowBordersSize) {
        return SDL_Unsupported();
    }

    return _this->GetWindowBordersSize(_this, window, top, left, bottom, right);
}

bool SDL_GetWindowSizeInPixels(SDL_Window *window, int *w, int *h)
{
    int filter;

    CHECK_WINDOW_MAGIC(window, false);

    if (!w) {
        w = &filter;
    }

    if (!h) {
        h = &filter;
    }

    if (_this->GetWindowSizeInPixels) {
        _this->GetWindowSizeInPixels(_this, window, w, h);
    } else {
        SDL_DisplayID displayID = SDL_GetDisplayForWindow(window);
        const SDL_DisplayMode *mode;

        SDL_GetWindowSize(window, w, h);

        if ((window->flags & SDL_WINDOW_FULLSCREEN) && SDL_GetWindowFullscreenMode(window)) {
            mode = SDL_GetCurrentDisplayMode(displayID);
        } else {
            mode = SDL_GetDesktopDisplayMode(displayID);
        }
        if (mode) {
            *w = (int)SDL_ceilf(*w * mode->pixel_density);
            *h = (int)SDL_ceilf(*h * mode->pixel_density);
        }
    }
    return true;
}

bool SDL_SetWindowMinimumSize(SDL_Window *window, int min_w, int min_h)
{
    CHECK_WINDOW_MAGIC(window, false);
    if (min_w < 0) {
        return SDL_InvalidParamError("min_w");
    }
    if (min_h < 0) {
        return SDL_InvalidParamError("min_h");
    }

    if ((window->max_w && min_w > window->max_w) ||
        (window->max_h && min_h > window->max_h)) {
        return SDL_SetError("SDL_SetWindowMinimumSize(): Tried to set minimum size larger than maximum size");
    }

    window->min_w = min_w;
    window->min_h = min_h;

    if (_this->SetWindowMinimumSize) {
        _this->SetWindowMinimumSize(_this, window);
    }

    // Ensure that window is not smaller than minimal size
    int w = window->last_size_pending ? window->pending.w : window->floating.w;
    int h = window->last_size_pending ? window->pending.h : window->floating.h;
    w = window->min_w ? SDL_max(w, window->min_w) : w;
    h = window->min_h ? SDL_max(h, window->min_h) : h;
    return SDL_SetWindowSize(window, w, h);
}

bool SDL_GetWindowMinimumSize(SDL_Window *window, int *min_w, int *min_h)
{
    CHECK_WINDOW_MAGIC(window, false);
    if (min_w) {
        *min_w = window->min_w;
    }
    if (min_h) {
        *min_h = window->min_h;
    }
    return true;
}

bool SDL_SetWindowMaximumSize(SDL_Window *window, int max_w, int max_h)
{
    CHECK_WINDOW_MAGIC(window, false);
    if (max_w < 0) {
        return SDL_InvalidParamError("max_w");
    }
    if (max_h < 0) {
        return SDL_InvalidParamError("max_h");
    }

    if ((max_w && max_w < window->min_w) ||
        (max_h && max_h < window->min_h)) {
        return SDL_SetError("SDL_SetWindowMaximumSize(): Tried to set maximum size smaller than minimum size");
    }

    window->max_w = max_w;
    window->max_h = max_h;

    if (_this->SetWindowMaximumSize) {
        _this->SetWindowMaximumSize(_this, window);
    }

    // Ensure that window is not larger than maximal size
    int w = window->last_size_pending ? window->pending.w : window->floating.w;
    int h = window->last_size_pending ? window->pending.h : window->floating.h;
    w = window->max_w ? SDL_min(w, window->max_w) : w;
    h = window->max_h ? SDL_min(h, window->max_h) : h;
    return SDL_SetWindowSize(window, w, h);
}

bool SDL_GetWindowMaximumSize(SDL_Window *window, int *max_w, int *max_h)
{
    CHECK_WINDOW_MAGIC(window, false);
    if (max_w) {
        *max_w = window->max_w;
    }
    if (max_h) {
        *max_h = window->max_h;
    }
    return true;
}

bool SDL_ShowWindow(SDL_Window *window)
{
    SDL_Window *child;
    CHECK_WINDOW_MAGIC(window, false);

    if (!(window->flags & SDL_WINDOW_HIDDEN)) {
        return true;
    }

    // If the parent is hidden, set the flag to restore this when the parent is shown
    if (window->parent && (window->parent->flags & SDL_WINDOW_HIDDEN)) {
        window->restore_on_show = true;
        return true;
    }

    if (_this->ShowWindow) {
        _this->ShowWindow(_this, window);
    } else {
        SDL_SetMouseFocus(window);
        SDL_SetKeyboardFocus(window);
    }
    SDL_SendWindowEvent(window, SDL_EVENT_WINDOW_SHOWN, 0, 0);

    // Restore child windows
    for (child = window->first_child; child; child = child->next_sibling) {
        if (!child->restore_on_show && (child->flags & SDL_WINDOW_HIDDEN)) {
            break;
        }
        SDL_ShowWindow(child);
        child->restore_on_show = false;
    }
    return true;
}

bool SDL_HideWindow(SDL_Window *window)
{
    SDL_Window *child;
    CHECK_WINDOW_MAGIC(window, false);

    if (window->flags & SDL_WINDOW_HIDDEN) {
        window->restore_on_show = false;
        return true;
    }

    // Hide all child windows
    for (child = window->first_child; child; child = child->next_sibling) {
        if (child->flags & SDL_WINDOW_HIDDEN) {
            break;
        }
        SDL_HideWindow(child);
        child->restore_on_show = true;
    }

    // Store the flags for restoration later.
    const SDL_WindowFlags pending_mask = (SDL_WINDOW_MAXIMIZED | SDL_WINDOW_MINIMIZED | SDL_WINDOW_FULLSCREEN | SDL_WINDOW_KEYBOARD_GRABBED | SDL_WINDOW_MOUSE_GRABBED);
    window->pending_flags = (window->flags & pending_mask);

    window->is_hiding = true;
    if (_this->HideWindow) {
        _this->HideWindow(_this, window);
    } else {
        SDL_SetMouseFocus(NULL);
        SDL_SetKeyboardFocus(NULL);
    }
    window->is_hiding = false;
    SDL_SendWindowEvent(window, SDL_EVENT_WINDOW_HIDDEN, 0, 0);
    return true;
}

bool SDL_RaiseWindow(SDL_Window *window)
{
    CHECK_WINDOW_MAGIC(window, false);

    if (window->flags & SDL_WINDOW_HIDDEN) {
        return true;
    }
    if (_this->RaiseWindow) {
        _this->RaiseWindow(_this, window);
    }
    return true;
}

bool SDL_MaximizeWindow(SDL_Window *window)
{
    CHECK_WINDOW_MAGIC(window, false);
    CHECK_WINDOW_NOT_POPUP(window, false);

    if (!_this->MaximizeWindow) {
        return SDL_Unsupported();
    }

    if (!(window->flags & SDL_WINDOW_RESIZABLE)) {
        return SDL_SetError("A window without the 'SDL_WINDOW_RESIZABLE' flag can't be maximized");
    }

    if (window->flags & SDL_WINDOW_HIDDEN) {
        window->pending_flags |= SDL_WINDOW_MAXIMIZED;
        return true;
    }

    _this->MaximizeWindow(_this, window);
    SDL_SyncIfRequired(window);
    return true;
}

bool SDL_MinimizeWindow(SDL_Window *window)
{
    CHECK_WINDOW_MAGIC(window, false);
    CHECK_WINDOW_NOT_POPUP(window, false);

    if (!_this->MinimizeWindow) {
        return SDL_Unsupported();
    }

    if (window->flags & SDL_WINDOW_HIDDEN) {
        window->pending_flags |= SDL_WINDOW_MINIMIZED;
        return true;
    }

    _this->MinimizeWindow(_this, window);
    SDL_SyncIfRequired(window);
    return true;
}

bool SDL_RestoreWindow(SDL_Window *window)
{
    CHECK_WINDOW_MAGIC(window, false);
    CHECK_WINDOW_NOT_POPUP(window, false);

    if (!_this->RestoreWindow) {
        return SDL_Unsupported();
    }

    if (window->flags & SDL_WINDOW_HIDDEN) {
        window->pending_flags &= ~(SDL_WINDOW_MAXIMIZED | SDL_WINDOW_MINIMIZED);
        return true;
    }

    _this->RestoreWindow(_this, window);
    SDL_SyncIfRequired(window);
    return true;
}

bool SDL_SetWindowFullscreen(SDL_Window *window, bool fullscreen)
{
    bool result;

    CHECK_WINDOW_MAGIC(window, false);
    CHECK_WINDOW_NOT_POPUP(window, false);

    if (window->flags & SDL_WINDOW_HIDDEN) {
        if (fullscreen) {
            window->pending_flags |= SDL_WINDOW_FULLSCREEN;
        } else {
            window->pending_flags &= ~SDL_WINDOW_FULLSCREEN;
        }
        return true;
    }

    if (fullscreen) {
        // Set the current fullscreen mode to the desired mode
        SDL_copyp(&window->current_fullscreen_mode, &window->requested_fullscreen_mode);
    }

    result = SDL_UpdateFullscreenMode(window, fullscreen ? SDL_FULLSCREEN_OP_ENTER : SDL_FULLSCREEN_OP_LEAVE, true);

    if (!fullscreen || !result) {
        // Clear the current fullscreen mode.
        SDL_zero(window->current_fullscreen_mode);
    }

    if (result) {
        SDL_SyncIfRequired(window);
    }

    return result;
}

bool SDL_SyncWindow(SDL_Window *window)
{
    CHECK_WINDOW_MAGIC(window, false)

    if (_this->SyncWindow) {
        return _this->SyncWindow(_this, window);
    } else {
        return true;
    }
}

static bool ShouldAttemptTextureFramebuffer(void)
{
    const char *hint;
    bool attempt_texture_framebuffer = true;

    // The dummy driver never has GPU support, of course.
    if (_this->is_dummy) {
        return false;
    }

    // See if there's a hint override
    hint = SDL_GetHint(SDL_HINT_FRAMEBUFFER_ACCELERATION);
    if (hint && *hint) {
        if (*hint == '0' || SDL_strcasecmp(hint, "false") == 0 || SDL_strcasecmp(hint, SDL_SOFTWARE_RENDERER) == 0) {
            attempt_texture_framebuffer = false;
        } else {
            attempt_texture_framebuffer = true;
        }
    } else {
        // Check for platform specific defaults
#ifdef SDL_PLATFORM_LINUX
        // On WSL, direct X11 is faster than using OpenGL for window framebuffers, so try to detect WSL and avoid texture framebuffer.
        if ((_this->CreateWindowFramebuffer) && (SDL_strcmp(_this->name, "x11") == 0)) {
            struct stat sb;
            if ((stat("/proc/sys/fs/binfmt_misc/WSLInterop", &sb) == 0) || (stat("/run/WSL", &sb) == 0)) { // if either of these exist, we're on WSL.
                attempt_texture_framebuffer = false;
            }
        }
#endif
#if defined(SDL_PLATFORM_WIN32) || defined(SDL_PLATFORM_WINGDK) // GDI BitBlt() is way faster than Direct3D dynamic textures right now. (!!! FIXME: is this still true?)
        if (_this->CreateWindowFramebuffer && (SDL_strcmp(_this->name, "windows") == 0)) {
            attempt_texture_framebuffer = false;
        }
#endif
#ifdef SDL_PLATFORM_EMSCRIPTEN
        attempt_texture_framebuffer = false;
#endif
    }
    return attempt_texture_framebuffer;
}

static SDL_Surface *SDL_CreateWindowFramebuffer(SDL_Window *window)
{
    SDL_PixelFormat format = SDL_PIXELFORMAT_UNKNOWN;
    void *pixels = NULL;
    int pitch = 0;
    bool created_framebuffer = false;
    int w, h;

    SDL_GetWindowSizeInPixels(window, &w, &h);

    /* This will switch the video backend from using a software surface to
       using a GPU texture through the 2D render API, if we think this would
       be more efficient. This only checks once, on demand. */
    if (!_this->checked_texture_framebuffer) {
        if (ShouldAttemptTextureFramebuffer()) {
            if (!SDL_CreateWindowTexture(_this, window, &format, &pixels, &pitch)) {
                /* !!! FIXME: if this failed halfway (made renderer, failed to make texture, etc),
                   !!! FIXME:  we probably need to clean this up so it doesn't interfere with
                   !!! FIXME:  a software fallback at the system level (can we blit to an
                   !!! FIXME:  OpenGL window? etc). */
            } else {
                // future attempts will just try to use a texture framebuffer.
                /* !!! FIXME:  maybe we shouldn't override these but check if we used a texture
                   !!! FIXME:  framebuffer at the right places; is it feasible we could have an
                   !!! FIXME:  accelerated OpenGL window and a second ends up in software? */
                _this->CreateWindowFramebuffer = SDL_CreateWindowTexture;
                _this->SetWindowFramebufferVSync = SDL_SetWindowTextureVSync;
                _this->GetWindowFramebufferVSync = SDL_GetWindowTextureVSync;
                _this->UpdateWindowFramebuffer = SDL_UpdateWindowTexture;
                _this->DestroyWindowFramebuffer = SDL_DestroyWindowTexture;
                created_framebuffer = true;
            }
        }

        _this->checked_texture_framebuffer = true; // don't check this again.
    }

    if (!created_framebuffer) {
        if (!_this->CreateWindowFramebuffer || !_this->UpdateWindowFramebuffer) {
            SDL_SetError("Window framebuffer support not available");
            return NULL;
        }

        if (!_this->CreateWindowFramebuffer(_this, window, &format, &pixels, &pitch)) {
            return NULL;
        }
    }

    if (window->surface) {
        // We may have gone recursive and already created the surface
        return window->surface;
    }

    return SDL_CreateSurfaceFrom(w, h, format, pixels, pitch);
}

bool SDL_WindowHasSurface(SDL_Window *window)
{
    CHECK_WINDOW_MAGIC(window, false);

    return window->surface ? true : false;
}

SDL_Surface *SDL_GetWindowSurface(SDL_Window *window)
{
    CHECK_WINDOW_MAGIC(window, NULL);

    if (!window->surface_valid) {
        if (window->surface) {
            window->surface->internal_flags &= ~SDL_INTERNAL_SURFACE_DONTFREE;
            SDL_DestroySurface(window->surface);
            window->surface = NULL;
        }

        window->surface = SDL_CreateWindowFramebuffer(window);
        if (window->surface) {
            window->surface_valid = true;
            window->surface->internal_flags |= SDL_INTERNAL_SURFACE_DONTFREE;
        }
    }
    return window->surface;
}

bool SDL_SetWindowSurfaceVSync(SDL_Window *window, int vsync)
{
    CHECK_WINDOW_MAGIC(window, false);

    if (!_this->SetWindowFramebufferVSync) {
        return SDL_Unsupported();
    }
    return _this->SetWindowFramebufferVSync(_this, window, vsync);
}

bool SDL_GetWindowSurfaceVSync(SDL_Window *window, int *vsync)
{
    CHECK_WINDOW_MAGIC(window, false);

    if (!_this->GetWindowFramebufferVSync) {
        return SDL_Unsupported();
    }
    return _this->GetWindowFramebufferVSync(_this, window, vsync);
}

bool SDL_UpdateWindowSurface(SDL_Window *window)
{
    SDL_Rect full_rect;

    CHECK_WINDOW_MAGIC(window, false);

    full_rect.x = 0;
    full_rect.y = 0;
    SDL_GetWindowSizeInPixels(window, &full_rect.w, &full_rect.h);

    return SDL_UpdateWindowSurfaceRects(window, &full_rect, 1);
}

bool SDL_UpdateWindowSurfaceRects(SDL_Window *window, const SDL_Rect *rects,
                                 int numrects)
{
    CHECK_WINDOW_MAGIC(window, false);

    if (!window->surface_valid) {
        return SDL_SetError("Window surface is invalid, please call SDL_GetWindowSurface() to get a new surface");
    }

    SDL_assert(_this->checked_texture_framebuffer); // we should have done this before we had a valid surface.

    return _this->UpdateWindowFramebuffer(_this, window, rects, numrects);
}

bool SDL_DestroyWindowSurface(SDL_Window *window)
{
    CHECK_WINDOW_MAGIC(window, false);

    if (window->surface) {
        window->surface->internal_flags &= ~SDL_INTERNAL_SURFACE_DONTFREE;
        SDL_DestroySurface(window->surface);
        window->surface = NULL;
        window->surface_valid = false;
    }

    if (_this->checked_texture_framebuffer) { // never checked? No framebuffer to destroy. Don't risk calling the wrong implementation.
        if (_this->DestroyWindowFramebuffer) {
            _this->DestroyWindowFramebuffer(_this, window);
        }
    }
    return true;
}

bool SDL_SetWindowOpacity(SDL_Window *window, float opacity)
{
    bool result;

    CHECK_WINDOW_MAGIC(window, false);

    if (!_this->SetWindowOpacity) {
        return SDL_Unsupported();
    }

    if (opacity < 0.0f) {
        opacity = 0.0f;
    } else if (opacity > 1.0f) {
        opacity = 1.0f;
    }

    result = _this->SetWindowOpacity(_this, window, opacity);
    if (result) {
        window->opacity = opacity;
    }

    return result;
}

float SDL_GetWindowOpacity(SDL_Window *window)
{
    CHECK_WINDOW_MAGIC(window, -1.0f);

    return window->opacity;
}

bool SDL_SetWindowParent(SDL_Window *window, SDL_Window *parent)
{
    CHECK_WINDOW_MAGIC(window, false);
    CHECK_WINDOW_NOT_POPUP(window, false);

    if (parent) {
        CHECK_WINDOW_MAGIC(parent, false);
        CHECK_WINDOW_NOT_POPUP(parent, false);
    }

    if (!_this->SetWindowParent) {
        return SDL_Unsupported();
    }

    if (window->flags & SDL_WINDOW_MODAL) {
        return SDL_SetError("Modal windows cannot change parents; call SDL_SetWindowModal() to clear modal status first.");
    }

    if (window->parent == parent) {
        return true;
    }

    const bool ret = _this->SetWindowParent(_this, window, parent);
    SDL_UpdateWindowHierarchy(window, ret ? parent : NULL);

    return ret;
}

bool SDL_SetWindowModal(SDL_Window *window, bool modal)
{
    CHECK_WINDOW_MAGIC(window, false);
    CHECK_WINDOW_NOT_POPUP(window, false);

    if (!_this->SetWindowModal) {
        return SDL_Unsupported();
    }

    if (modal) {
        if (!window->parent) {
            return SDL_SetError("Window must have a parent to enable the modal state; use SDL_SetWindowParent() to set the parent first.");
        }
        window->flags |= SDL_WINDOW_MODAL;
    } else if (window->flags & SDL_WINDOW_MODAL) {
        window->flags &= ~SDL_WINDOW_MODAL;
    } else {
        return true; // Already not modal, so nothing to do.
    }

    if (window->flags & SDL_WINDOW_HIDDEN) {
        return true;
    }

    return _this->SetWindowModal(_this, window, modal);
}

bool SDL_ShouldRelinquishPopupFocus(SDL_Window *window, SDL_Window **new_focus)
{
    SDL_Window *focus = window->parent;
    bool set_focus = !!(window->flags & SDL_WINDOW_INPUT_FOCUS);

    // Find the highest level window, up to the toplevel parent, that isn't being hidden or destroyed, and can grab the keyboard focus.
    while (SDL_WINDOW_IS_POPUP(focus) && ((focus->flags & SDL_WINDOW_NOT_FOCUSABLE) || focus->is_hiding || focus->is_destroying)) {
        focus = focus->parent;

        // If some window in the chain currently had focus, set it to the new lowest-level window.
        if (!set_focus) {
            set_focus = !!(focus->flags & SDL_WINDOW_INPUT_FOCUS);
        }
    }

    *new_focus = focus;
    return set_focus;
}

bool SDL_ShouldFocusPopup(SDL_Window *window)
{
    SDL_Window *toplevel_parent;
    for (toplevel_parent = window->parent; SDL_WINDOW_IS_POPUP(toplevel_parent); toplevel_parent = toplevel_parent->parent) {
    }

    SDL_Window *current_focus = toplevel_parent->keyboard_focus;
    bool found_higher_focus = false;

    /* Traverse the window tree from the currently focused window to the toplevel parent and see if we encounter
     * the new focus request. If the new window is found, a higher-level window already has focus.
     */
    SDL_Window *w;
    for (w = current_focus; w != toplevel_parent; w = w->parent) {
        if (w == window) {
            found_higher_focus = true;
            break;
        }
    }

    return !found_higher_focus || w == toplevel_parent;
}

bool SDL_SetWindowFocusable(SDL_Window *window, bool focusable)
{
    CHECK_WINDOW_MAGIC(window, false);

    const bool want = (focusable != false); // normalize the flag.
    const bool have = !(window->flags & SDL_WINDOW_NOT_FOCUSABLE);
    if ((want != have) && (_this->SetWindowFocusable)) {
        if (want) {
            window->flags &= ~SDL_WINDOW_NOT_FOCUSABLE;
        } else {
            window->flags |= SDL_WINDOW_NOT_FOCUSABLE;
        }
        if (!_this->SetWindowFocusable(_this, window, want)) {
            return false;
        }
    }

    return true;
}

void SDL_UpdateWindowGrab(SDL_Window *window)
{
    bool keyboard_grabbed, mouse_grabbed;

    if (window->flags & SDL_WINDOW_INPUT_FOCUS) {
        if (SDL_GetMouse()->relative_mode || (window->flags & SDL_WINDOW_MOUSE_GRABBED)) {
            mouse_grabbed = true;
        } else {
            mouse_grabbed = false;
        }

        if (window->flags & SDL_WINDOW_KEYBOARD_GRABBED) {
            keyboard_grabbed = true;
        } else {
            keyboard_grabbed = false;
        }
    } else {
        mouse_grabbed = false;
        keyboard_grabbed = false;
    }

    if (mouse_grabbed || keyboard_grabbed) {
        if (_this->grabbed_window && (_this->grabbed_window != window)) {
            // stealing a grab from another window!
            _this->grabbed_window->flags &= ~(SDL_WINDOW_MOUSE_GRABBED | SDL_WINDOW_KEYBOARD_GRABBED);
            if (_this->SetWindowMouseGrab) {
                _this->SetWindowMouseGrab(_this, _this->grabbed_window, false);
            }
            if (_this->SetWindowKeyboardGrab) {
                _this->SetWindowKeyboardGrab(_this, _this->grabbed_window, false);
            }
        }
        _this->grabbed_window = window;
    } else if (_this->grabbed_window == window) {
        _this->grabbed_window = NULL; // ungrabbing input.
    }

    if (_this->SetWindowMouseGrab) {
        if (!_this->SetWindowMouseGrab(_this, window, mouse_grabbed)) {
            window->flags &= ~SDL_WINDOW_MOUSE_GRABBED;
        }
    }
    if (_this->SetWindowKeyboardGrab) {
        if (!_this->SetWindowKeyboardGrab(_this, window, keyboard_grabbed)) {
            window->flags &= ~SDL_WINDOW_KEYBOARD_GRABBED;
        }
    }

    if (_this->grabbed_window && !(_this->grabbed_window->flags & (SDL_WINDOW_MOUSE_GRABBED | SDL_WINDOW_KEYBOARD_GRABBED))) {
        _this->grabbed_window = NULL;
    }
}

bool SDL_SetWindowKeyboardGrab(SDL_Window *window, bool grabbed)
{
    CHECK_WINDOW_MAGIC(window, false);
    CHECK_WINDOW_NOT_POPUP(window, false);

    if (window->flags & SDL_WINDOW_HIDDEN) {
        if (grabbed) {
            window->pending_flags |= SDL_WINDOW_KEYBOARD_GRABBED;
        } else {
            window->pending_flags &= ~SDL_WINDOW_KEYBOARD_GRABBED;
        }
        return true;
    }

    if (!!grabbed == !!(window->flags & SDL_WINDOW_KEYBOARD_GRABBED)) {
        return true;
    }
    if (grabbed) {
        window->flags |= SDL_WINDOW_KEYBOARD_GRABBED;
    } else {
        window->flags &= ~SDL_WINDOW_KEYBOARD_GRABBED;
    }
    SDL_UpdateWindowGrab(window);

    if (grabbed && !(window->flags & SDL_WINDOW_KEYBOARD_GRABBED)) {
        return false;
    }
    return true;
}

bool SDL_SetWindowMouseGrab(SDL_Window *window, bool grabbed)
{
    CHECK_WINDOW_MAGIC(window, false);
    CHECK_WINDOW_NOT_POPUP(window, false);

    if (window->flags & SDL_WINDOW_HIDDEN) {
        if (grabbed) {
            window->pending_flags |= SDL_WINDOW_MOUSE_GRABBED;
        } else {
            window->pending_flags &= ~SDL_WINDOW_MOUSE_GRABBED;
        }
        return true;
    }

    if (!!grabbed == !!(window->flags & SDL_WINDOW_MOUSE_GRABBED)) {
        return true;
    }
    if (grabbed) {
        window->flags |= SDL_WINDOW_MOUSE_GRABBED;
    } else {
        window->flags &= ~SDL_WINDOW_MOUSE_GRABBED;
    }
    SDL_UpdateWindowGrab(window);

    if (grabbed && !(window->flags & SDL_WINDOW_MOUSE_GRABBED)) {
        return false;
    }
    return true;
}

bool SDL_GetWindowKeyboardGrab(SDL_Window *window)
{
    CHECK_WINDOW_MAGIC(window, false);
    return window == _this->grabbed_window && (_this->grabbed_window->flags & SDL_WINDOW_KEYBOARD_GRABBED);
}

bool SDL_GetWindowMouseGrab(SDL_Window *window)
{
    CHECK_WINDOW_MAGIC(window, false);
    return window == _this->grabbed_window && (_this->grabbed_window->flags & SDL_WINDOW_MOUSE_GRABBED);
}

SDL_Window *SDL_GetGrabbedWindow(void)
{
    if (_this->grabbed_window &&
        (_this->grabbed_window->flags & (SDL_WINDOW_MOUSE_GRABBED | SDL_WINDOW_KEYBOARD_GRABBED)) != 0) {
        return _this->grabbed_window;
    } else {
        return NULL;
    }
}

bool SDL_SetWindowMouseRect(SDL_Window *window, const SDL_Rect *rect)
{
    CHECK_WINDOW_MAGIC(window, false);

    if (rect) {
        SDL_memcpy(&window->mouse_rect, rect, sizeof(*rect));
    } else {
        SDL_zero(window->mouse_rect);
    }

    if (_this->SetWindowMouseRect) {
        return _this->SetWindowMouseRect(_this, window);
    }
    return true;
}

const SDL_Rect *SDL_GetWindowMouseRect(SDL_Window *window)
{
    CHECK_WINDOW_MAGIC(window, NULL);

    if (SDL_RectEmpty(&window->mouse_rect)) {
        return NULL;
    } else {
        return &window->mouse_rect;
    }
}

bool SDL_SetWindowRelativeMouseMode(SDL_Window *window, bool enabled)
{
    CHECK_WINDOW_MAGIC(window, false);

    /* If the app toggles relative mode directly, it probably shouldn't
     * also be emulating it using repeated mouse warps, so disable
     * mouse warp emulation by default.
     */
    SDL_DisableMouseWarpEmulation();

    if (enabled == SDL_GetWindowRelativeMouseMode(window)) {
        return true;
    }

    if (enabled) {
        window->flags |= SDL_WINDOW_MOUSE_RELATIVE_MODE;
    } else {
        window->flags &= ~SDL_WINDOW_MOUSE_RELATIVE_MODE;
    }
    SDL_UpdateRelativeMouseMode();

    return true;
}

bool SDL_GetWindowRelativeMouseMode(SDL_Window *window)
{
    CHECK_WINDOW_MAGIC(window, false);

    if (window->flags & SDL_WINDOW_MOUSE_RELATIVE_MODE) {
        return true;
    } else {
        return false;
    }
}

bool SDL_FlashWindow(SDL_Window *window, SDL_FlashOperation operation)
{
    CHECK_WINDOW_MAGIC(window, false);
    CHECK_WINDOW_NOT_POPUP(window, false);

    if (_this->FlashWindow) {
        return _this->FlashWindow(_this, window, operation);
    }

    return SDL_Unsupported();
}

bool SDL_SetWindowProgressState(SDL_Window *window, SDL_ProgressState state)
{
    CHECK_WINDOW_MAGIC(window, false);
    CHECK_WINDOW_NOT_POPUP(window, false);

    if (state < SDL_PROGRESS_STATE_NONE || state > SDL_PROGRESS_STATE_ERROR) {
        return SDL_InvalidParamError("state");
    }

    window->progress_state = state;

    if (_this->ApplyWindowProgress) {
        if (!_this->ApplyWindowProgress(_this, window)) {
            return false;
        }
    }

    return true;
}

SDL_ProgressState SDL_GetWindowProgressState(SDL_Window *window)
{
    CHECK_WINDOW_MAGIC(window, SDL_PROGRESS_STATE_INVALID);
    CHECK_WINDOW_NOT_POPUP(window, SDL_PROGRESS_STATE_INVALID);

    return window->progress_state;
}

bool SDL_SetWindowProgressValue(SDL_Window *window, float value)
{
    CHECK_WINDOW_MAGIC(window, false);
    CHECK_WINDOW_NOT_POPUP(window, false);

    value = SDL_clamp(value, 0.0f, 1.f);

    window->progress_value = value;

    if (_this->ApplyWindowProgress) {
        if (!_this->ApplyWindowProgress(_this, window)) {
            return false;
        }
    }

    return true;
}

float SDL_GetWindowProgressValue(SDL_Window *window)
{
    CHECK_WINDOW_MAGIC(window, -1.0f);
    CHECK_WINDOW_NOT_POPUP(window, -1.0f);

    return window->progress_value;
}

void SDL_OnWindowShown(SDL_Window *window)
{
    // Set window state if we have pending window flags cached
    ApplyWindowFlags(window, window->pending_flags);
    window->pending_flags = 0;
}

void SDL_OnWindowHidden(SDL_Window *window)
{
    /* Store the maximized and fullscreen flags for restoration later, in case
     * this was initiated by the window manager due to the window being unmapped
     * when minimized.
     */
    window->pending_flags |= (window->flags & (SDL_WINDOW_FULLSCREEN | SDL_WINDOW_MAXIMIZED));

    // The window is already hidden at this point, so just change the mode back if necessary.
    SDL_UpdateFullscreenMode(window, SDL_FULLSCREEN_OP_LEAVE, false);
}

void SDL_OnWindowDisplayChanged(SDL_Window *window)
{
    // Don't run this if a fullscreen change was made in an event watcher callback in response to a display changed event.
    if (window->update_fullscreen_on_display_changed && (window->flags & SDL_WINDOW_FULLSCREEN)) {
        const bool auto_mode_switch = SDL_GetHintBoolean(SDL_HINT_VIDEO_MATCH_EXCLUSIVE_MODE_ON_MOVE, true);

        if (auto_mode_switch && (window->requested_fullscreen_mode.w != 0 || window->requested_fullscreen_mode.h != 0)) {
            SDL_DisplayID displayID = SDL_GetDisplayForWindowPosition(window);
            bool include_high_density_modes = false;

            if (window->requested_fullscreen_mode.pixel_density > 1.0f) {
                include_high_density_modes = true;
            }
            const bool found_match = SDL_GetClosestFullscreenDisplayMode(displayID, window->requested_fullscreen_mode.w, window->requested_fullscreen_mode.h,
                                                                         window->requested_fullscreen_mode.refresh_rate, include_high_density_modes, &window->current_fullscreen_mode);

            // If a mode without matching dimensions was not found, just go to fullscreen desktop.
            if (!found_match ||
                window->requested_fullscreen_mode.w != window->current_fullscreen_mode.w ||
                window->requested_fullscreen_mode.h != window->current_fullscreen_mode.h) {
                SDL_zero(window->current_fullscreen_mode);
            }
        } else {
            SDL_zero(window->current_fullscreen_mode);
        }

        if (SDL_WINDOW_FULLSCREEN_VISIBLE(window)) {
            SDL_UpdateFullscreenMode(window, SDL_FULLSCREEN_OP_UPDATE, true);
        }
    }

    SDL_CheckWindowPixelSizeChanged(window);
}

void SDL_OnWindowMoved(SDL_Window *window)
{
    SDL_CheckWindowDisplayChanged(window);
}

void SDL_OnWindowResized(SDL_Window *window)
{
    SDL_CheckWindowDisplayChanged(window);
    SDL_CheckWindowPixelSizeChanged(window);
    SDL_CheckWindowSafeAreaChanged(window);

    if ((window->flags & SDL_WINDOW_TRANSPARENT) && _this->UpdateWindowShape) {
        SDL_Surface *surface = (SDL_Surface *)SDL_GetPointerProperty(window->props, SDL_PROP_WINDOW_SHAPE_POINTER, NULL);
        if (surface) {
            _this->UpdateWindowShape(_this, window, surface);
        }
    }
}

void SDL_CheckWindowPixelSizeChanged(SDL_Window *window)
{
    int pixel_w = 0, pixel_h = 0;

    SDL_GetWindowSizeInPixels(window, &pixel_w, &pixel_h);
    SDL_SendWindowEvent(window, SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED, pixel_w, pixel_h);

    SDL_CheckWindowDisplayScaleChanged(window);
}

void SDL_OnWindowPixelSizeChanged(SDL_Window *window)
{
    window->surface_valid = false;
}

void SDL_OnWindowLiveResizeUpdate(SDL_Window *window)
{
    if (SDL_HasMainCallbacks()) {
        SDL_IterateMainCallbacks(false);
    } else {
        // Send an expose event so the application can redraw
        SDL_SendWindowEvent(window, SDL_EVENT_WINDOW_EXPOSED, 0, 0);
    }

    SDL_PumpEventMaintenance();
}

static void SDL_CheckWindowSafeAreaChanged(SDL_Window *window)
{
    SDL_Rect rect;

    rect.x = window->safe_inset_left;
    rect.y = window->safe_inset_top;
    rect.w = window->w - (window->safe_inset_right + window->safe_inset_left);
    rect.h = window->h - (window->safe_inset_top + window->safe_inset_bottom);
    if (SDL_memcmp(&rect, &window->safe_rect, sizeof(rect)) != 0) {
        SDL_copyp(&window->safe_rect, &rect);
        SDL_SendWindowEvent(window, SDL_EVENT_WINDOW_SAFE_AREA_CHANGED, 0, 0);
    }
}

void SDL_SetWindowSafeAreaInsets(SDL_Window *window, int left, int right, int top, int bottom)
{
    window->safe_inset_left = left;
    window->safe_inset_right = right;
    window->safe_inset_top = top;
    window->safe_inset_bottom = bottom;
    SDL_CheckWindowSafeAreaChanged(window);
}

bool SDL_GetWindowSafeArea(SDL_Window *window, SDL_Rect *rect)
{
    if (rect) {
        SDL_zerop(rect);
    }

    CHECK_WINDOW_MAGIC(window, false);

    if (rect) {
        if (SDL_RectEmpty(&window->safe_rect)) {
            rect->w = window->w;
            rect->h = window->h;
        } else {
            SDL_copyp(rect, &window->safe_rect);
        }
    }
    return true;
}

void SDL_OnWindowMinimized(SDL_Window *window)
{
    if (window->flags & SDL_WINDOW_FULLSCREEN) {
        SDL_UpdateFullscreenMode(window, SDL_FULLSCREEN_OP_LEAVE, false);
    }
}

void SDL_OnWindowMaximized(SDL_Window *window)
{
}

void SDL_OnWindowRestored(SDL_Window *window)
{
    /*
     * FIXME: Is this fine to just remove this, or should it be preserved just
     * for the fullscreen case? In principle it seems like just hiding/showing
     * windows shouldn't affect the stacking order; maybe the right fix is to
     * re-decouple OnWindowShown and OnWindowRestored.
     */
    // SDL_RaiseWindow(window);

    if (window->flags & SDL_WINDOW_FULLSCREEN) {
        SDL_UpdateFullscreenMode(window, SDL_FULLSCREEN_OP_ENTER, false);
    }
}

void SDL_OnWindowEnter(SDL_Window *window)
{
    if (_this->OnWindowEnter) {
        _this->OnWindowEnter(_this, window);
    }
}

void SDL_OnWindowLeave(SDL_Window *window)
{
}

void SDL_OnWindowFocusGained(SDL_Window *window)
{
    SDL_Mouse *mouse = SDL_GetMouse();

    if (mouse && mouse->relative_mode) {
        SDL_SetMouseFocus(window);
    }

    SDL_UpdateWindowGrab(window);
}

static bool SDL_ShouldMinimizeOnFocusLoss(SDL_Window *window)
{
    const char *hint;

    if (!(window->flags & SDL_WINDOW_FULLSCREEN) || window->is_destroying) {
        return false;
    }

#if defined(SDL_PLATFORM_MACOS) && defined(SDL_VIDEO_DRIVER_COCOA)
    if (SDL_strcmp(_this->name, "cocoa") == 0) { // don't do this for X11, etc
        if (Cocoa_IsWindowInFullscreenSpace(window)) {
            return false;
        }
    }
#endif

#ifdef SDL_PLATFORM_ANDROID
    {
        extern bool Android_JNI_ShouldMinimizeOnFocusLoss(void);
        if (!Android_JNI_ShouldMinimizeOnFocusLoss()) {
            return false;
        }
    }
#endif

    // Real fullscreen windows should minimize on focus loss so the desktop video mode is restored
    hint = SDL_GetHint(SDL_HINT_VIDEO_MINIMIZE_ON_FOCUS_LOSS);
    if (!hint || !*hint || SDL_strcasecmp(hint, "auto") == 0) {
        if (window->fullscreen_exclusive && !SDL_ModeSwitchingEmulated(_this)) {
            return true;
        } else {
            return false;
        }
    }
    return SDL_GetHintBoolean(SDL_HINT_VIDEO_MINIMIZE_ON_FOCUS_LOSS, false);
}

void SDL_OnWindowFocusLost(SDL_Window *window)
{
    SDL_UpdateWindowGrab(window);

    if (SDL_ShouldMinimizeOnFocusLoss(window)) {
        SDL_MinimizeWindow(window);
    }
}

SDL_Window *SDL_GetToplevelForKeyboardFocus(void)
{
    SDL_Window *focus = SDL_GetKeyboardFocus();

    if (focus) {
        // Get the toplevel parent window.
        while (focus->parent) {
            focus = focus->parent;
        }
    }

    return focus;
}

bool SDL_AddWindowRenderer(SDL_Window *window, SDL_Renderer *renderer)
{
    SDL_Renderer **renderers = (SDL_Renderer **)SDL_realloc(window->renderers, (window->num_renderers + 1) * sizeof(*renderers));
    if (!renderers) {
        return false;
    }

    window->renderers = renderers;
    window->renderers[window->num_renderers++] = renderer;
    return true;
}

void SDL_RemoveWindowRenderer(SDL_Window *window, SDL_Renderer *renderer)
{
    for (int i = 0; i < window->num_renderers; ++i) {
        if (window->renderers[i] == renderer) {
            if (i < (window->num_renderers - 1)) {
                SDL_memmove(&window->renderers[i], &window->renderers[i + 1], (window->num_renderers - i - 1) * sizeof(window->renderers[i]));
            }
            --window->num_renderers;
            break;
        }
    }
}

void SDL_DestroyWindow(SDL_Window *window)
{
    CHECK_WINDOW_MAGIC(window,);

    window->is_destroying = true;

    // Destroy any child windows of this window
    while (window->first_child) {
        SDL_DestroyWindow(window->first_child);
    }

    SDL_SendWindowEvent(window, SDL_EVENT_WINDOW_DESTROYED, 0, 0);

    SDL_Renderer *renderer = SDL_GetRenderer(window);
    if (renderer) {
        SDL_DestroyRendererWithoutFreeing(renderer);
    }

    // Restore video mode, etc.
    SDL_UpdateFullscreenMode(window, SDL_FULLSCREEN_OP_LEAVE, true);
    if (!(window->flags & SDL_WINDOW_EXTERNAL)) {
        SDL_HideWindow(window);
    }

    SDL_DestroyProperties(window->text_input_props);
    SDL_DestroyProperties(window->props);

    /* Clear the modal status, but don't unset the parent just yet, as it
     * may be needed later in the destruction process if a backend needs
     * to update the input focus.
     */
    if (_this->SetWindowModal && (window->flags & SDL_WINDOW_MODAL)) {
        _this->SetWindowModal(_this, window, false);
    }

    // Make sure the destroyed window isn't referenced by any display as a fullscreen window.
    for (int i = 0; i < _this->num_displays; ++i) {
        if (_this->displays[i]->fullscreen_window == window) {
            _this->displays[i]->fullscreen_window = NULL;
        }
    }

    // Make sure this window no longer has focus
    if (SDL_GetKeyboardFocus() == window) {
        SDL_SetKeyboardFocus(NULL);
    }
    if ((window->flags & SDL_WINDOW_MOUSE_CAPTURE)) {
        SDL_UpdateMouseCapture(true);
    }
    if (SDL_GetMouseFocus() == window) {
        SDL_SetMouseFocus(NULL);
    }

    SDL_DestroyWindowSurface(window);

    // Make no context current if this is the current context window
    if (window->flags & SDL_WINDOW_OPENGL) {
        if (_this->current_glwin == window) {
            SDL_GL_MakeCurrent(window, NULL);
        }
    }

    if (_this->DestroyWindow) {
        _this->DestroyWindow(_this, window);
    }

    // Unload the graphics libraries after the window is destroyed, which may clean up EGL surfaces
    if (window->flags & SDL_WINDOW_OPENGL) {
        SDL_GL_UnloadLibrary();
    }
    if (window->flags & SDL_WINDOW_VULKAN) {
        SDL_Vulkan_UnloadLibrary();
    }

    if (_this->grabbed_window == window) {
        _this->grabbed_window = NULL; // ungrabbing input.
    }

    if (_this->current_glwin == window) {
        _this->current_glwin = NULL;
    }

    if (_this->wakeup_window == window) {
        _this->wakeup_window = NULL;
    }

    // Now invalidate magic
    SDL_SetObjectValid(window, SDL_OBJECT_TYPE_WINDOW, false);

    // Free memory associated with the window
    SDL_free(window->title);
    SDL_DestroySurface(window->icon);

    // Unlink the window from its siblings.
    SDL_UpdateWindowHierarchy(window, NULL);

    // Unlink the window from the global window list
    if (window->next) {
        window->next->prev = window->prev;
    }
    if (window->prev) {
        window->prev->next = window->next;
    } else {
        _this->windows = window->next;
    }

    SDL_free(window->renderers);
    SDL_free(window);

#ifdef SDL_VIDEO_DRIVER_UIKIT
    SDL_UpdateLifecycleObserver();
#endif
}

bool SDL_ScreenSaverEnabled(void)
{
    if (!_this) {
        return true;
    }
    return !_this->suspend_screensaver;
}

bool SDL_EnableScreenSaver(void)
{
    if (!_this) {
        return SDL_UninitializedVideo();
    }
    if (!_this->suspend_screensaver) {
        return true;
    }
    _this->suspend_screensaver = false;
    if (_this->SuspendScreenSaver) {
        return _this->SuspendScreenSaver(_this);
    }

    return SDL_Unsupported();
}

bool SDL_DisableScreenSaver(void)
{
    if (!_this) {
        return SDL_UninitializedVideo();
    }
    if (_this->suspend_screensaver) {
        return true;
    }
    _this->suspend_screensaver = true;
    if (_this->SuspendScreenSaver) {
        return _this->SuspendScreenSaver(_this);
    }

    return SDL_Unsupported();
}

void SDL_VideoQuit(void)
{
    int i;

    if (!_this) {
        return;
    }

    // Halt event processing before doing anything else
#if 0 // This was moved to the end to fix a memory leak
    SDL_QuitPen();
#endif
    SDL_QuitTouch();
    SDL_QuitMouse();
    SDL_QuitKeyboard();
    SDL_QuitSubSystem(SDL_INIT_EVENTS);

    SDL_EnableScreenSaver();

    // Clean up the system video
    while (_this->windows) {
        SDL_DestroyWindow(_this->windows);
    }
    _this->VideoQuit(_this);

    for (i = _this->num_displays; i--; ) {
        SDL_VideoDisplay *display = _this->displays[i];
        SDL_DelVideoDisplay(display->id, false);
    }

    SDL_assert(_this->num_displays == 0);
    SDL_free(_this->displays);
    _this->displays = NULL;

    SDL_CancelClipboardData(0);

    if (_this->primary_selection_text) {
        SDL_free(_this->primary_selection_text);
        _this->primary_selection_text = NULL;
    }
    _this->free(_this);
    _this = NULL;

    // This needs to happen after the video subsystem has removed pen data
    SDL_QuitPen();
}

bool SDL_GL_LoadLibrary(const char *path)
{
    bool result;

    if (!_this) {
        return SDL_UninitializedVideo();
    }
    if (_this->gl_config.driver_loaded) {
        if (path && SDL_strcmp(path, _this->gl_config.driver_path) != 0) {
            return SDL_SetError("OpenGL library already loaded");
        }
        result = true;
    } else {
        if (!_this->GL_LoadLibrary) {
            return SDL_DllNotSupported("OpenGL");
        }
        result = _this->GL_LoadLibrary(_this, path);
    }
    if (result) {
        ++_this->gl_config.driver_loaded;
    } else {
        if (_this->GL_UnloadLibrary) {
            _this->GL_UnloadLibrary(_this);
        }
    }
    return result;
}

SDL_FunctionPointer SDL_GL_GetProcAddress(const char *proc)
{
    SDL_FunctionPointer func;

    if (!_this) {
        SDL_UninitializedVideo();
        return NULL;
    }
    func = NULL;
    if (_this->GL_GetProcAddress) {
        if (_this->gl_config.driver_loaded) {
            func = _this->GL_GetProcAddress(_this, proc);
        } else {
            SDL_SetError("No GL driver has been loaded");
        }
    } else {
        SDL_SetError("No dynamic GL support in current SDL video driver (%s)", _this->name);
    }
    return func;
}

SDL_FunctionPointer SDL_EGL_GetProcAddress(const char *proc)
{
#ifdef SDL_VIDEO_OPENGL_EGL
    SDL_FunctionPointer func;

    if (!_this) {
        SDL_UninitializedVideo();
        return NULL;
    }
    func = NULL;

    if (_this->egl_data) {
        func = SDL_EGL_GetProcAddressInternal(_this, proc);
    } else {
        SDL_SetError("No EGL library has been loaded");
    }

    return func;
#else
    SDL_SetError("SDL was not built with EGL support");
    return NULL;
#endif
}

void SDL_GL_UnloadLibrary(void)
{
    if (!_this) {
        SDL_UninitializedVideo();
        return;
    }
    if (_this->gl_config.driver_loaded > 0) {
        if (--_this->gl_config.driver_loaded > 0) {
            return;
        }
        if (_this->GL_UnloadLibrary) {
            _this->GL_UnloadLibrary(_this);
        }
    }
}

#if defined(SDL_VIDEO_OPENGL) || defined(SDL_VIDEO_OPENGL_ES) || defined(SDL_VIDEO_OPENGL_ES2)
typedef GLenum (APIENTRY* PFNGLGETERRORPROC) (void);
typedef void (APIENTRY* PFNGLGETINTEGERVPROC) (GLenum pname, GLint *params);
typedef const GLubyte *(APIENTRY* PFNGLGETSTRINGPROC) (GLenum name);
#ifndef SDL_VIDEO_OPENGL
typedef const GLubyte *(APIENTRY* PFNGLGETSTRINGIPROC) (GLenum name, GLuint index);
#endif

static SDL_INLINE bool isAtLeastGL3(const char *verstr)
{
    return verstr && (SDL_atoi(verstr) >= 3);
}
#endif // SDL_VIDEO_OPENGL || SDL_VIDEO_OPENGL_ES || SDL_VIDEO_OPENGL_ES2

bool SDL_GL_ExtensionSupported(const char *extension)
{
#if defined(SDL_VIDEO_OPENGL) || defined(SDL_VIDEO_OPENGL_ES) || defined(SDL_VIDEO_OPENGL_ES2)
    PFNGLGETSTRINGPROC glGetStringFunc;
    const char *extensions;
    const char *start;
    const char *where, *terminator;

    // Extension names should not have spaces.
    where = SDL_strchr(extension, ' ');
    if (where || *extension == '\0') {
        return false;
    }
    // See if there's a hint or environment variable override
    start = SDL_GetHint(extension);
    if (start && *start == '0') {
        return false;
    }

    // Lookup the available extensions

    glGetStringFunc = (PFNGLGETSTRINGPROC)SDL_GL_GetProcAddress("glGetString");
    if (!glGetStringFunc) {
        return false;
    }

    if (isAtLeastGL3((const char *)glGetStringFunc(GL_VERSION))) {
        PFNGLGETSTRINGIPROC glGetStringiFunc;
        PFNGLGETINTEGERVPROC glGetIntegervFunc;
        GLint num_exts = 0;
        GLint i;

        glGetStringiFunc = (PFNGLGETSTRINGIPROC)SDL_GL_GetProcAddress("glGetStringi");
        glGetIntegervFunc = (PFNGLGETINTEGERVPROC)SDL_GL_GetProcAddress("glGetIntegerv");
        if ((!glGetStringiFunc) || (!glGetIntegervFunc)) {
            return false;
        }

#ifndef GL_NUM_EXTENSIONS
#define GL_NUM_EXTENSIONS 0x821D
#endif
        glGetIntegervFunc(GL_NUM_EXTENSIONS, &num_exts);
        for (i = 0; i < num_exts; i++) {
            const char *thisext = (const char *)glGetStringiFunc(GL_EXTENSIONS, i);
            if (SDL_strcmp(thisext, extension) == 0) {
                return true;
            }
        }

        return false;
    }

    // Try the old way with glGetString(GL_EXTENSIONS) ...

    extensions = (const char *)glGetStringFunc(GL_EXTENSIONS);
    if (!extensions) {
        return false;
    }
    /*
     * It takes a bit of care to be fool-proof about parsing the OpenGL
     * extensions string. Don't be fooled by sub-strings, etc.
     */

    start = extensions;

    for (;;) {
        where = SDL_strstr(start, extension);
        if (!where) {
            break;
        }

        terminator = where + SDL_strlen(extension);
        if (where == extensions || *(where - 1) == ' ') {
            if (*terminator == ' ' || *terminator == '\0') {
                return true;
            }
        }

        start = terminator;
    }
    return false;
#else
    return false;
#endif
}

/* Deduce supported ES profile versions from the supported
   ARB_ES*_compatibility extensions. There is no direct query.

   This is normally only called when the OpenGL driver supports
   {GLX,WGL}_EXT_create_context_es2_profile.
 */
void SDL_GL_DeduceMaxSupportedESProfile(int *major, int *minor)
{
// THIS REQUIRES AN EXISTING GL CONTEXT THAT HAS BEEN MADE CURRENT.
// Please refer to https://bugzilla.libsdl.org/show_bug.cgi?id=3725 for discussion.
#if defined(SDL_VIDEO_OPENGL) || defined(SDL_VIDEO_OPENGL_ES) || defined(SDL_VIDEO_OPENGL_ES2)
    /* XXX This is fragile; it will break in the event of release of
     * new versions of OpenGL ES.
     */
    if (SDL_GL_ExtensionSupported("GL_ARB_ES3_2_compatibility")) {
        *major = 3;
        *minor = 2;
    } else if (SDL_GL_ExtensionSupported("GL_ARB_ES3_1_compatibility")) {
        *major = 3;
        *minor = 1;
    } else if (SDL_GL_ExtensionSupported("GL_ARB_ES3_compatibility")) {
        *major = 3;
        *minor = 0;
    } else {
        *major = 2;
        *minor = 0;
    }
#endif
}

void SDL_EGL_SetAttributeCallbacks(SDL_EGLAttribArrayCallback platformAttribCallback,
                                   SDL_EGLIntArrayCallback surfaceAttribCallback,
                                   SDL_EGLIntArrayCallback contextAttribCallback,
                                   void *userdata)
{
    if (!_this) {
        return;
    }
    _this->egl_platformattrib_callback = platformAttribCallback;
    _this->egl_surfaceattrib_callback = surfaceAttribCallback;
    _this->egl_contextattrib_callback = contextAttribCallback;
    _this->egl_attrib_callback_userdata = userdata;
}

void SDL_GL_ResetAttributes(void)
{
    if (!_this) {
        return;
    }

    _this->egl_platformattrib_callback = NULL;
    _this->egl_surfaceattrib_callback = NULL;
    _this->egl_contextattrib_callback = NULL;
    _this->egl_attrib_callback_userdata = NULL;

    _this->gl_config.red_size = 8;
    _this->gl_config.green_size = 8;
    _this->gl_config.blue_size = 8;
    _this->gl_config.alpha_size = 8;
    _this->gl_config.buffer_size = 0;
    _this->gl_config.depth_size = 16;
    _this->gl_config.stencil_size = 0;
    _this->gl_config.double_buffer = 1;
    _this->gl_config.accum_red_size = 0;
    _this->gl_config.accum_green_size = 0;
    _this->gl_config.accum_blue_size = 0;
    _this->gl_config.accum_alpha_size = 0;
    _this->gl_config.stereo = 0;
    _this->gl_config.multisamplebuffers = 0;
    _this->gl_config.multisamplesamples = 0;
    _this->gl_config.floatbuffers = 0;
    _this->gl_config.retained_backing = 1;
    _this->gl_config.accelerated = -1; // accelerated or not, both are fine

#ifdef SDL_VIDEO_OPENGL
    _this->gl_config.major_version = 2;
    _this->gl_config.minor_version = 1;
    _this->gl_config.profile_mask = 0;
#elif defined(SDL_VIDEO_OPENGL_ES2)
    _this->gl_config.major_version = 2;
    _this->gl_config.minor_version = 0;
    _this->gl_config.profile_mask = SDL_GL_CONTEXT_PROFILE_ES;
#elif defined(SDL_VIDEO_OPENGL_ES)
    _this->gl_config.major_version = 1;
    _this->gl_config.minor_version = 1;
    _this->gl_config.profile_mask = SDL_GL_CONTEXT_PROFILE_ES;
#endif

    if (_this->GL_DefaultProfileConfig) {
        _this->GL_DefaultProfileConfig(_this, &_this->gl_config.profile_mask,
                                       &_this->gl_config.major_version,
                                       &_this->gl_config.minor_version);
    }

    _this->gl_config.flags = 0;
    _this->gl_config.framebuffer_srgb_capable = 0;
    _this->gl_config.no_error = 0;
    _this->gl_config.release_behavior = SDL_GL_CONTEXT_RELEASE_BEHAVIOR_FLUSH;
    _this->gl_config.reset_notification = SDL_GL_CONTEXT_RESET_NO_NOTIFICATION;

    _this->gl_config.share_with_current_context = 0;

    _this->gl_config.egl_platform = 0;
}

bool SDL_GL_SetAttribute(SDL_GLAttr attr, int value)
{
#if defined(SDL_VIDEO_OPENGL) || defined(SDL_VIDEO_OPENGL_ES) || defined(SDL_VIDEO_OPENGL_ES2)
    bool result;

    if (!_this) {
        return SDL_UninitializedVideo();
    }
    result = true;
    switch (attr) {
    case SDL_GL_RED_SIZE:
        _this->gl_config.red_size = value;
        break;
    case SDL_GL_GREEN_SIZE:
        _this->gl_config.green_size = value;
        break;
    case SDL_GL_BLUE_SIZE:
        _this->gl_config.blue_size = value;
        break;
    case SDL_GL_ALPHA_SIZE:
        _this->gl_config.alpha_size = value;
        break;
    case SDL_GL_DOUBLEBUFFER:
        _this->gl_config.double_buffer = value;
        break;
    case SDL_GL_BUFFER_SIZE:
        _this->gl_config.buffer_size = value;
        break;
    case SDL_GL_DEPTH_SIZE:
        _this->gl_config.depth_size = value;
        break;
    case SDL_GL_STENCIL_SIZE:
        _this->gl_config.stencil_size = value;
        break;
    case SDL_GL_ACCUM_RED_SIZE:
        _this->gl_config.accum_red_size = value;
        break;
    case SDL_GL_ACCUM_GREEN_SIZE:
        _this->gl_config.accum_green_size = value;
        break;
    case SDL_GL_ACCUM_BLUE_SIZE:
        _this->gl_config.accum_blue_size = value;
        break;
    case SDL_GL_ACCUM_ALPHA_SIZE:
        _this->gl_config.accum_alpha_size = value;
        break;
    case SDL_GL_STEREO:
        _this->gl_config.stereo = value;
        break;
    case SDL_GL_MULTISAMPLEBUFFERS:
        _this->gl_config.multisamplebuffers = value;
        break;
    case SDL_GL_MULTISAMPLESAMPLES:
        _this->gl_config.multisamplesamples = value;
        break;
    case SDL_GL_FLOATBUFFERS:
        _this->gl_config.floatbuffers = value;
        break;
    case SDL_GL_ACCELERATED_VISUAL:
        _this->gl_config.accelerated = value;
        break;
    case SDL_GL_RETAINED_BACKING:
        _this->gl_config.retained_backing = value;
        break;
    case SDL_GL_CONTEXT_MAJOR_VERSION:
        _this->gl_config.major_version = value;
        break;
    case SDL_GL_CONTEXT_MINOR_VERSION:
        _this->gl_config.minor_version = value;
        break;
    case SDL_GL_CONTEXT_FLAGS:
        if (value & ~(SDL_GL_CONTEXT_DEBUG_FLAG |
                      SDL_GL_CONTEXT_FORWARD_COMPATIBLE_FLAG |
                      SDL_GL_CONTEXT_ROBUST_ACCESS_FLAG |
                      SDL_GL_CONTEXT_RESET_ISOLATION_FLAG)) {
            result = SDL_SetError("Unknown OpenGL context flag %d", value);
            break;
        }
        _this->gl_config.flags = value;
        break;
    case SDL_GL_CONTEXT_PROFILE_MASK:
        if (value != 0 &&
            value != SDL_GL_CONTEXT_PROFILE_CORE &&
            value != SDL_GL_CONTEXT_PROFILE_COMPATIBILITY &&
            value != SDL_GL_CONTEXT_PROFILE_ES) {
            result = SDL_SetError("Unknown OpenGL context profile %d", value);
            break;
        }
        _this->gl_config.profile_mask = value;
        break;
    case SDL_GL_SHARE_WITH_CURRENT_CONTEXT:
        _this->gl_config.share_with_current_context = value;
        break;
    case SDL_GL_FRAMEBUFFER_SRGB_CAPABLE:
        _this->gl_config.framebuffer_srgb_capable = value;
        break;
    case SDL_GL_CONTEXT_RELEASE_BEHAVIOR:
        _this->gl_config.release_behavior = value;
        break;
    case SDL_GL_CONTEXT_RESET_NOTIFICATION:
        _this->gl_config.reset_notification = value;
        break;
    case SDL_GL_CONTEXT_NO_ERROR:
        _this->gl_config.no_error = value;
        break;
    case SDL_GL_EGL_PLATFORM:
        _this->gl_config.egl_platform = value;
        break;
    default:
        result = SDL_SetError("Unknown OpenGL attribute");
        break;
    }
    return result;
#else
    return SDL_Unsupported();
#endif // SDL_VIDEO_OPENGL
}

bool SDL_GL_GetAttribute(SDL_GLAttr attr, int *value)
{
#if defined(SDL_VIDEO_OPENGL) || defined(SDL_VIDEO_OPENGL_ES) || defined(SDL_VIDEO_OPENGL_ES2)
    PFNGLGETERRORPROC glGetErrorFunc;
    GLenum attrib = 0;
    GLenum error = 0;

    /*
     * Some queries in Core Profile desktop OpenGL 3+ contexts require
     * glGetFramebufferAttachmentParameteriv instead of glGetIntegerv. Note that
     * the enums we use for the former function don't exist in OpenGL ES 2, and
     * the function itself doesn't exist prior to OpenGL 3 and OpenGL ES 2.
     */
#ifdef SDL_VIDEO_OPENGL
    PFNGLGETSTRINGPROC glGetStringFunc;
    PFNGLGETFRAMEBUFFERATTACHMENTPARAMETERIVPROC glGetFramebufferAttachmentParameterivFunc;
    GLenum attachment = GL_BACK_LEFT;
    GLenum attachmentattrib = 0;
#endif

    if (!value) {
        return SDL_InvalidParamError("value");
    }

    // Clear value in any case
    *value = 0;

    if (!_this) {
        return SDL_UninitializedVideo();
    }

    switch (attr) {
    case SDL_GL_RED_SIZE:
#ifdef SDL_VIDEO_OPENGL
        attachmentattrib = GL_FRAMEBUFFER_ATTACHMENT_RED_SIZE;
#endif
        attrib = GL_RED_BITS;
        break;
    case SDL_GL_BLUE_SIZE:
#ifdef SDL_VIDEO_OPENGL
        attachmentattrib = GL_FRAMEBUFFER_ATTACHMENT_BLUE_SIZE;
#endif
        attrib = GL_BLUE_BITS;
        break;
    case SDL_GL_GREEN_SIZE:
#ifdef SDL_VIDEO_OPENGL
        attachmentattrib = GL_FRAMEBUFFER_ATTACHMENT_GREEN_SIZE;
#endif
        attrib = GL_GREEN_BITS;
        break;
    case SDL_GL_ALPHA_SIZE:
#ifdef SDL_VIDEO_OPENGL
        attachmentattrib = GL_FRAMEBUFFER_ATTACHMENT_ALPHA_SIZE;
#endif
        attrib = GL_ALPHA_BITS;
        break;
    case SDL_GL_DOUBLEBUFFER:
#ifdef SDL_VIDEO_OPENGL
        attrib = GL_DOUBLEBUFFER;
        break;
#else
        // OpenGL ES 1.0 and above specifications have EGL_SINGLE_BUFFER
        // parameter which switches double buffer to single buffer. OpenGL ES
        // SDL driver must set proper value after initialization
        *value = _this->gl_config.double_buffer;
        return true;
#endif
    case SDL_GL_DEPTH_SIZE:
#ifdef SDL_VIDEO_OPENGL
        attachment = GL_DEPTH;
        attachmentattrib = GL_FRAMEBUFFER_ATTACHMENT_DEPTH_SIZE;
#endif
        attrib = GL_DEPTH_BITS;
        break;
    case SDL_GL_STENCIL_SIZE:
#ifdef SDL_VIDEO_OPENGL
        attachment = GL_STENCIL;
        attachmentattrib = GL_FRAMEBUFFER_ATTACHMENT_STENCIL_SIZE;
#endif
        attrib = GL_STENCIL_BITS;
        break;
#ifdef SDL_VIDEO_OPENGL
    case SDL_GL_ACCUM_RED_SIZE:
        attrib = GL_ACCUM_RED_BITS;
        break;
    case SDL_GL_ACCUM_GREEN_SIZE:
        attrib = GL_ACCUM_GREEN_BITS;
        break;
    case SDL_GL_ACCUM_BLUE_SIZE:
        attrib = GL_ACCUM_BLUE_BITS;
        break;
    case SDL_GL_ACCUM_ALPHA_SIZE:
        attrib = GL_ACCUM_ALPHA_BITS;
        break;
    case SDL_GL_STEREO:
        attrib = GL_STEREO;
        break;
#else
    case SDL_GL_ACCUM_RED_SIZE:
    case SDL_GL_ACCUM_GREEN_SIZE:
    case SDL_GL_ACCUM_BLUE_SIZE:
    case SDL_GL_ACCUM_ALPHA_SIZE:
    case SDL_GL_STEREO:
        // none of these are supported in OpenGL ES
        *value = 0;
        return true;
#endif
    case SDL_GL_MULTISAMPLEBUFFERS:
        attrib = GL_SAMPLE_BUFFERS;
        break;
    case SDL_GL_MULTISAMPLESAMPLES:
        attrib = GL_SAMPLES;
        break;
    case SDL_GL_CONTEXT_RELEASE_BEHAVIOR:
        attrib = GL_CONTEXT_RELEASE_BEHAVIOR;
        break;
    case SDL_GL_BUFFER_SIZE:
    {
        int rsize = 0, gsize = 0, bsize = 0, asize = 0;

        // There doesn't seem to be a single flag in OpenGL for this!
        if (!SDL_GL_GetAttribute(SDL_GL_RED_SIZE, &rsize)) {
            return false;
        }
        if (!SDL_GL_GetAttribute(SDL_GL_GREEN_SIZE, &gsize)) {
            return false;
        }
        if (!SDL_GL_GetAttribute(SDL_GL_BLUE_SIZE, &bsize)) {
            return false;
        }
        if (!SDL_GL_GetAttribute(SDL_GL_ALPHA_SIZE, &asize)) {
            return false;
        }

        *value = rsize + gsize + bsize + asize;
        return true;
    }
    case SDL_GL_ACCELERATED_VISUAL:
    {
        // FIXME: How do we get this information?
        *value = (_this->gl_config.accelerated != 0);
        return true;
    }
    case SDL_GL_RETAINED_BACKING:
    {
        *value = _this->gl_config.retained_backing;
        return true;
    }
    case SDL_GL_CONTEXT_MAJOR_VERSION:
    {
        *value = _this->gl_config.major_version;
        return true;
    }
    case SDL_GL_CONTEXT_MINOR_VERSION:
    {
        *value = _this->gl_config.minor_version;
        return true;
    }
    case SDL_GL_CONTEXT_FLAGS:
    {
        *value = _this->gl_config.flags;
        return true;
    }
    case SDL_GL_CONTEXT_PROFILE_MASK:
    {
        *value = _this->gl_config.profile_mask;
        return true;
    }
    case SDL_GL_SHARE_WITH_CURRENT_CONTEXT:
    {
        *value = _this->gl_config.share_with_current_context;
        return true;
    }
    case SDL_GL_FRAMEBUFFER_SRGB_CAPABLE:
    {
        *value = _this->gl_config.framebuffer_srgb_capable;
        return true;
    }
    case SDL_GL_CONTEXT_NO_ERROR:
    {
        *value = _this->gl_config.no_error;
        return true;
    }
    case SDL_GL_EGL_PLATFORM:
    {
        *value = _this->gl_config.egl_platform;
        return true;
    }
    default:
        return SDL_SetError("Unknown OpenGL attribute");
    }

#ifdef SDL_VIDEO_OPENGL
    glGetStringFunc = (PFNGLGETSTRINGPROC)SDL_GL_GetProcAddress("glGetString");
    if (!glGetStringFunc) {
        return false;
    }

    if (attachmentattrib && isAtLeastGL3((const char *)glGetStringFunc(GL_VERSION))) {
        // glGetFramebufferAttachmentParameteriv needs to operate on the window framebuffer for this, so bind FBO 0 if necessary.
        GLint current_fbo = 0;
        PFNGLGETINTEGERVPROC glGetIntegervFunc = (PFNGLGETINTEGERVPROC) SDL_GL_GetProcAddress("glGetIntegerv");
        PFNGLBINDFRAMEBUFFERPROC glBindFramebufferFunc = (PFNGLBINDFRAMEBUFFERPROC)SDL_GL_GetProcAddress("glBindFramebuffer");
        if (glGetIntegervFunc && glBindFramebufferFunc) {
            glGetIntegervFunc(GL_DRAW_FRAMEBUFFER_BINDING, &current_fbo);
        }

        glGetFramebufferAttachmentParameterivFunc = (PFNGLGETFRAMEBUFFERATTACHMENTPARAMETERIVPROC)SDL_GL_GetProcAddress("glGetFramebufferAttachmentParameteriv");
        if (glGetFramebufferAttachmentParameterivFunc) {
            if (glBindFramebufferFunc && (current_fbo != 0)) {
                glBindFramebufferFunc(GL_DRAW_FRAMEBUFFER, 0);
            }
            // glGetFramebufferAttachmentParameterivFunc may cause GL_INVALID_OPERATION when querying depth/stencil size if the
            // bits is 0. From the GL docs:
            //      If the value of GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE is GL_NONE, then either no framebuffer is bound to target;
            //      or a default framebuffer is queried, attachment is GL_DEPTH or GL_STENCIL, and the number of depth or stencil bits,
            //      respectively, is zero. In this case querying pname GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME will return zero, and all
            //      other queries will generate an error.
            GLint fbo_type = GL_FRAMEBUFFER_DEFAULT;
            if (attachment == GL_DEPTH || attachment == GL_STENCIL) {
                glGetFramebufferAttachmentParameterivFunc(GL_FRAMEBUFFER, attachment, GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE, &fbo_type);
            }
            if (fbo_type != GL_NONE) {
                glGetFramebufferAttachmentParameterivFunc(GL_FRAMEBUFFER, attachment, attachmentattrib, (GLint *)value);
            }
            else {
                *value = 0;
            }
            if (glBindFramebufferFunc && (current_fbo != 0)) {
                glBindFramebufferFunc(GL_DRAW_FRAMEBUFFER, current_fbo);
            }
        } else {
            return false;
        }
    } else
#endif
    {
        PFNGLGETINTEGERVPROC glGetIntegervFunc = (PFNGLGETINTEGERVPROC)SDL_GL_GetProcAddress("glGetIntegerv");
        if (glGetIntegervFunc) {
            glGetIntegervFunc(attrib, (GLint *)value);
        } else {
            return false;
        }
    }

    glGetErrorFunc = (PFNGLGETERRORPROC)SDL_GL_GetProcAddress("glGetError");
    if (!glGetErrorFunc) {
        return false;
    }

    error = glGetErrorFunc();
    if (error != GL_NO_ERROR) {
        if (error == GL_INVALID_ENUM) {
            return SDL_SetError("OpenGL error: GL_INVALID_ENUM");
        } else if (error == GL_INVALID_VALUE) {
            return SDL_SetError("OpenGL error: GL_INVALID_VALUE");
        }
        return SDL_SetError("OpenGL error: %08X", error);
    }

    // convert GL_CONTEXT_RELEASE_BEHAVIOR values back to SDL_GL_CONTEXT_RELEASE_BEHAVIOR values
    if (attr == SDL_GL_CONTEXT_RELEASE_BEHAVIOR) {
        *value = (*value == GL_CONTEXT_RELEASE_BEHAVIOR_FLUSH) ? SDL_GL_CONTEXT_RELEASE_BEHAVIOR_FLUSH : SDL_GL_CONTEXT_RELEASE_BEHAVIOR_NONE;
    }

    return true;
#else
    return SDL_Unsupported();
#endif // SDL_VIDEO_OPENGL
}

#define NOT_AN_OPENGL_WINDOW "The specified window isn't an OpenGL window"

SDL_GLContext SDL_GL_CreateContext(SDL_Window *window)
{
    SDL_GLContext ctx = NULL;
    CHECK_WINDOW_MAGIC(window, NULL);

    if (!(window->flags & SDL_WINDOW_OPENGL)) {
        SDL_SetError(NOT_AN_OPENGL_WINDOW);
        return NULL;
    }

    ctx = _this->GL_CreateContext(_this, window);

    // Creating a context is assumed to make it current in the SDL driver.
    if (ctx) {
        _this->current_glwin = window;
        _this->current_glctx = ctx;
        SDL_SetTLS(&_this->current_glwin_tls, window, NULL);
        SDL_SetTLS(&_this->current_glctx_tls, ctx, NULL);
    }
    return ctx;
}

bool SDL_GL_MakeCurrent(SDL_Window *window, SDL_GLContext context)
{
    bool result;

    if (!_this) {
        return SDL_UninitializedVideo();
    }

    if (window == SDL_GL_GetCurrentWindow() &&
        context == SDL_GL_GetCurrentContext()) {
        // We're already current.
        return true;
    }

    if (!context) {
        window = NULL;
    } else if (window) {
        CHECK_WINDOW_MAGIC(window, false);

        if (!(window->flags & SDL_WINDOW_OPENGL)) {
            return SDL_SetError(NOT_AN_OPENGL_WINDOW);
        }
    } else if (!_this->gl_allow_no_surface) {
        return SDL_SetError("Use of OpenGL without a window is not supported on this platform");
    }

    result = _this->GL_MakeCurrent(_this, window, context);
    if (result) {
        _this->current_glwin = window;
        _this->current_glctx = context;
        SDL_SetTLS(&_this->current_glwin_tls, window, NULL);
        SDL_SetTLS(&_this->current_glctx_tls, context, NULL);
    }
    return result;
}

SDL_Window *SDL_GL_GetCurrentWindow(void)
{
    if (!_this) {
        SDL_UninitializedVideo();
        return NULL;
    }
    return (SDL_Window *)SDL_GetTLS(&_this->current_glwin_tls);
}

SDL_GLContext SDL_GL_GetCurrentContext(void)
{
    if (!_this) {
        SDL_UninitializedVideo();
        return NULL;
    }
    return (SDL_GLContext)SDL_GetTLS(&_this->current_glctx_tls);
}

SDL_EGLDisplay SDL_EGL_GetCurrentDisplay(void)
{
#ifdef SDL_VIDEO_OPENGL_EGL
    if (!_this) {
        SDL_UninitializedVideo();
        return EGL_NO_DISPLAY;
    }
    if (!_this->egl_data) {
        SDL_SetError("There is no current EGL display");
        return EGL_NO_DISPLAY;
    }
    return _this->egl_data->egl_display;
#else
    SDL_SetError("SDL was not built with EGL support");
    return NULL;
#endif
}

SDL_EGLConfig SDL_EGL_GetCurrentConfig(void)
{
#ifdef SDL_VIDEO_OPENGL_EGL
    if (!_this) {
        SDL_UninitializedVideo();
        return NULL;
    }
    if (!_this->egl_data) {
        SDL_SetError("There is no current EGL display");
        return NULL;
    }
    return _this->egl_data->egl_config;
#else
    SDL_SetError("SDL was not built with EGL support");
    return NULL;
#endif
}

SDL_EGLConfig SDL_EGL_GetWindowSurface(SDL_Window *window)
{
#ifdef SDL_VIDEO_OPENGL_EGL
    if (!_this) {
        SDL_UninitializedVideo();
        return NULL;
    }
    if (!_this->egl_data) {
        SDL_SetError("There is no current EGL display");
        return NULL;
    }
    if (_this->GL_GetEGLSurface) {
        return _this->GL_GetEGLSurface(_this, window);
    }
    return NULL;
#else
    SDL_SetError("SDL was not built with EGL support");
    return NULL;
#endif
}

bool SDL_GL_SetSwapInterval(int interval)
{
    if (!_this) {
        return SDL_UninitializedVideo();
    } else if (SDL_GL_GetCurrentContext() == NULL) {
        return SDL_SetError("No OpenGL context has been made current");
    } else if (_this->GL_SetSwapInterval) {
        return _this->GL_SetSwapInterval(_this, interval);
    } else {
        return SDL_SetError("Setting the swap interval is not supported");
    }
}

bool SDL_GL_GetSwapInterval(int *interval)
{
    if (!interval) {
       return SDL_InvalidParamError("interval");
    }

    *interval = 0;

    if (!_this) {
        return SDL_SetError("no video driver");
    } else if (SDL_GL_GetCurrentContext() == NULL) {
        return SDL_SetError("no current context");
    } else if (_this->GL_GetSwapInterval) {
        return _this->GL_GetSwapInterval(_this, interval);
    } else {
        return SDL_SetError("not implemented");
    }
}

bool SDL_GL_SwapWindow(SDL_Window *window)
{
    CHECK_WINDOW_MAGIC(window, false);

    if (!(window->flags & SDL_WINDOW_OPENGL)) {
        return SDL_SetError(NOT_AN_OPENGL_WINDOW);
    }

    if (SDL_GL_GetCurrentWindow() != window) {
        return SDL_SetError("The specified window has not been made current");
    }

    return _this->GL_SwapWindow(_this, window);
}

bool SDL_GL_DestroyContext(SDL_GLContext context)
{
    if (!_this) {
        return SDL_UninitializedVideo();                                       \
    }
    if (!context) {
        return SDL_InvalidParamError("context");
    }

    if (SDL_GL_GetCurrentContext() == context) {
        SDL_GL_MakeCurrent(NULL, NULL);
    }

    return _this->GL_DestroyContext(_this, context);
}

#if 0 // FIXME
/*
 * Utility function used by SDL_WM_SetIcon(); flags & 1 for color key, flags
 * & 2 for alpha channel.
 */
static void CreateMaskFromColorKeyOrAlpha(SDL_Surface *icon, Uint8 *mask, int flags)
{
    int x, y;
    Uint32 colorkey;
#define SET_MASKBIT(icon, x, y, mask) \
    mask[(y * ((icon->w + 7) / 8)) + (x / 8)] &= ~(0x01 << (7 - (x % 8)))

    colorkey = icon->format->colorkey;
    switch (SDL_BYTESPERPIXEL(icon->format)) {
    case 1:
        {
            Uint8 *pixels;
            for (y = 0; y < icon->h; ++y) {
                pixels = (Uint8 *) icon->pixels + y * icon->pitch;
                for (x = 0; x < icon->w; ++x) {
                    if (*pixels++ == colorkey) {
                        SET_MASKBIT(icon, x, y, mask);
                    }
                }
            }
        }
        break;

    case 2:
        {
            Uint16 *pixels;
            for (y = 0; y < icon->h; ++y) {
                pixels = (Uint16 *) icon->pixels + y * icon->pitch / 2;
                for (x = 0; x < icon->w; ++x) {
                    if ((flags & 1) && *pixels == colorkey) {
                        SET_MASKBIT(icon, x, y, mask);
                    } else if ((flags & 2)
                               && (*pixels & icon->format->Amask) == 0) {
                        SET_MASKBIT(icon, x, y, mask);
                    }
                    pixels++;
                }
            }
        }
        break;

    case 4:
        {
            Uint32 *pixels;
            for (y = 0; y < icon->h; ++y) {
                pixels = (Uint32 *) icon->pixels + y * icon->pitch / 4;
                for (x = 0; x < icon->w; ++x) {
                    if ((flags & 1) && *pixels == colorkey) {
                        SET_MASKBIT(icon, x, y, mask);
                    } else if ((flags & 2)
                               && (*pixels & icon->format->Amask) == 0) {
                        SET_MASKBIT(icon, x, y, mask);
                    }
                    pixels++;
                }
            }
        }
        break;
    }
}

/*
 * Sets the window manager icon for the display window.
 */
void SDL_WM_SetIcon(SDL_Surface *icon, Uint8 *mask)
{
    if (icon && _this->SetIcon) {
        // Generate a mask if necessary, and create the icon!
        if (mask == NULL) {
            int mask_len = icon->h * (icon->w + 7) / 8;
            int flags = 0;
            mask = (Uint8 *) SDL_malloc(mask_len);
            if (mask == NULL) {
                return;
            }
            SDL_memset(mask, ~0, mask_len);
            if (icon->flags & SDL_SRCCOLORKEY)
                flags |= 1;
            if (icon->flags & SDL_SRCALPHA)
                flags |= 2;
            if (flags) {
                CreateMaskFromColorKeyOrAlpha(icon, mask, flags);
            }
            _this->SetIcon(_this, icon, mask);
            SDL_free(mask);
        } else {
            _this->SetIcon(_this, icon, mask);
        }
    }
}
#endif

SDL_TextInputType SDL_GetTextInputType(SDL_PropertiesID props)
{
    return (SDL_TextInputType)SDL_GetNumberProperty(props, SDL_PROP_TEXTINPUT_TYPE_NUMBER, SDL_TEXTINPUT_TYPE_TEXT);
}

SDL_Capitalization SDL_GetTextInputCapitalization(SDL_PropertiesID props)
{
    if (SDL_HasProperty(props, SDL_PROP_TEXTINPUT_CAPITALIZATION_NUMBER)) {
        return (SDL_Capitalization)SDL_GetNumberProperty(props, SDL_PROP_TEXTINPUT_CAPITALIZATION_NUMBER, SDL_CAPITALIZE_NONE);
    }

    switch (SDL_GetTextInputType(props)) {
    case SDL_TEXTINPUT_TYPE_TEXT:
        return SDL_CAPITALIZE_SENTENCES;
    case SDL_TEXTINPUT_TYPE_TEXT_NAME:
        return SDL_CAPITALIZE_WORDS;
    default:
        return SDL_CAPITALIZE_NONE;
    }
}

bool SDL_GetTextInputAutocorrect(SDL_PropertiesID props)
{
    return SDL_GetBooleanProperty(props, SDL_PROP_TEXTINPUT_AUTOCORRECT_BOOLEAN, true);
}

bool SDL_GetTextInputMultiline(SDL_PropertiesID props)
{
    if (SDL_HasProperty(props, SDL_PROP_TEXTINPUT_MULTILINE_BOOLEAN)) {
        return SDL_GetBooleanProperty(props, SDL_PROP_TEXTINPUT_MULTILINE_BOOLEAN, false);
    }

    if (SDL_GetHintBoolean(SDL_HINT_RETURN_KEY_HIDES_IME, false)) {
        return false;
    } else {
        return true;
    }
}

static bool AutoShowingScreenKeyboard(void)
{
    const char *hint = SDL_GetHint(SDL_HINT_ENABLE_SCREEN_KEYBOARD);
    if (!hint) {
        // Steam will eventually have smarts about whether a keyboard is active, so always request the on-screen keyboard on Steam Deck
        hint = SDL_GetHint("SteamDeck");
    }
    if (((!hint || SDL_strcasecmp(hint, "auto") == 0) && !SDL_HasKeyboard()) ||
        SDL_GetStringBoolean(hint, false)) {
        return true;
    } else {
        return false;
    }
}

bool SDL_StartTextInput(SDL_Window *window)
{
    return SDL_StartTextInputWithProperties(window, 0);
}

bool SDL_StartTextInputWithProperties(SDL_Window *window, SDL_PropertiesID props)
{
    CHECK_WINDOW_MAGIC(window, false);

    if (window->text_input_props) {
        SDL_DestroyProperties(window->text_input_props);
        window->text_input_props = 0;
    }

    if (props) {
        window->text_input_props = SDL_CreateProperties();
        if (!window->text_input_props) {
            return false;
        }
        if (!SDL_CopyProperties(props, window->text_input_props)) {
            return false;
        }
    }

    if (_this->SetTextInputProperties) {
        _this->SetTextInputProperties(_this, window, props);
    }

    // Show the on-screen keyboard, if desired
    if (AutoShowingScreenKeyboard() && !SDL_ScreenKeyboardShown(window)) {
        if (_this->ShowScreenKeyboard) {
            _this->ShowScreenKeyboard(_this, window, props);
        }
    }

    if (!window->text_input_active) {
        // Finally start the text input system
        if (_this->StartTextInput) {
            if (!_this->StartTextInput(_this, window, props)) {
                return false;
            }
        }
        window->text_input_active = true;
    }
    return true;
}

bool SDL_TextInputActive(SDL_Window *window)
{
    CHECK_WINDOW_MAGIC(window, false);

    return window->text_input_active;
}

bool SDL_StopTextInput(SDL_Window *window)
{
    CHECK_WINDOW_MAGIC(window, false);

    if (window->text_input_active) {
        // Stop the text input system
        if (_this->StopTextInput) {
            _this->StopTextInput(_this, window);
        }
        window->text_input_active = false;
    }

    // Hide the on-screen keyboard, if desired
    if (AutoShowingScreenKeyboard() && SDL_ScreenKeyboardShown(window)) {
        if (_this->HideScreenKeyboard) {
            _this->HideScreenKeyboard(_this, window);
        }
    }
    return true;
}

bool SDL_SetTextInputArea(SDL_Window *window, const SDL_Rect *rect, int cursor)
{
    CHECK_WINDOW_MAGIC(window, false);

    if (rect) {
        SDL_copyp(&window->text_input_rect, rect);
        window->text_input_cursor = cursor;
    } else {
        SDL_zero(window->text_input_rect);
        window->text_input_cursor = 0;
    }

    if (_this && _this->UpdateTextInputArea) {
        if (!_this->UpdateTextInputArea(_this, window)) {
            return false;
        }
    }
    return true;
}

bool SDL_GetTextInputArea(SDL_Window *window, SDL_Rect *rect, int *cursor)
{
    CHECK_WINDOW_MAGIC(window, false);

    if (rect) {
        SDL_copyp(rect, &window->text_input_rect);
    }
    if (cursor) {
        *cursor = window->text_input_cursor;
    }
    return true;
}

bool SDL_ClearComposition(SDL_Window *window)
{
    CHECK_WINDOW_MAGIC(window, false);

    if (_this->ClearComposition) {
        return _this->ClearComposition(_this, window);
    }
    return true;
}

bool SDL_HasScreenKeyboardSupport(void)
{
    if (_this && _this->HasScreenKeyboardSupport) {
        return _this->HasScreenKeyboardSupport(_this);
    }
    return false;
}

bool SDL_ScreenKeyboardShown(SDL_Window *window)
{
    CHECK_WINDOW_MAGIC(window, false);

    if (_this->IsScreenKeyboardShown) {
        return _this->IsScreenKeyboardShown(_this, window);
    }
    return false;
}

int SDL_GetMessageBoxCount(void)
{
    return SDL_GetAtomicInt(&SDL_messagebox_count);
}

bool SDL_ShowMessageBox(const SDL_MessageBoxData *messageboxdata, int *buttonID)
{
    int dummybutton;
    bool result = false;
    bool show_cursor_prev;
    SDL_Window *current_window;
    SDL_MessageBoxData mbdata;

    if (!messageboxdata) {
        return SDL_InvalidParamError("messageboxdata");
    } else if (messageboxdata->numbuttons < 0) {
        return SDL_SetError("Invalid number of buttons");
    }

    // in case either the title or message was a pointer from SDL_GetError(), make a copy
    // now, as we'll likely overwrite error state in here.
    bool titleisstack = false, msgisstack = false;
    char *titlecpy = NULL;
    char *msgcpy = NULL;
    if (messageboxdata->title) {
        const size_t slen = SDL_strlen(messageboxdata->title) + 1;
        titlecpy = SDL_small_alloc(char, slen, &titleisstack);
        if (!titlecpy) {
            return false;
        }
        SDL_memcpy(titlecpy, messageboxdata->title, slen);
    }

    if (messageboxdata->message) {
        const size_t slen = SDL_strlen(messageboxdata->message) + 1;
        msgcpy = SDL_small_alloc(char, slen, &msgisstack);
        if (!msgcpy) {
            SDL_small_free(titlecpy, titleisstack);
            return false;
        }
        SDL_memcpy(msgcpy, messageboxdata->message, slen);
    }

    (void)SDL_AtomicIncRef(&SDL_messagebox_count);

    current_window = SDL_GetKeyboardFocus();
    SDL_UpdateMouseCapture(false);
    SDL_SetRelativeMouseMode(false);
    show_cursor_prev = SDL_CursorVisible();
    SDL_ShowCursor();
    SDL_ResetKeyboard();

    if (!buttonID) {
        buttonID = &dummybutton;
    }

    SDL_memcpy(&mbdata, messageboxdata, sizeof(*messageboxdata));
    mbdata.title = titlecpy;
    if (!mbdata.title) {
        mbdata.title = "";
    }
    mbdata.message = msgcpy;
    if (!mbdata.message) {
        mbdata.message = "";
    }
    messageboxdata = &mbdata;

    SDL_ClearError();

    if (_this && _this->ShowMessageBox) {
        result = _this->ShowMessageBox(_this, messageboxdata, buttonID);
    } else {
        // It's completely fine to call this function before video is initialized
        const char *driver_name = SDL_GetHint(SDL_HINT_VIDEO_DRIVER);
        if (driver_name && *driver_name != 0) {
            const char *driver_attempt = driver_name;
            while (driver_attempt && (*driver_attempt != 0) && !result) {
                const char *driver_attempt_end = SDL_strchr(driver_attempt, ',');
                size_t driver_attempt_len = (driver_attempt_end) ? (driver_attempt_end - driver_attempt)
                                                                     : SDL_strlen(driver_attempt);
                for (int i = 0; bootstrap[i]; ++i) {
                    if (bootstrap[i]->ShowMessageBox && (driver_attempt_len == SDL_strlen(bootstrap[i]->name)) &&
                        (SDL_strncasecmp(bootstrap[i]->name, driver_attempt, driver_attempt_len) == 0)) {
                        if (bootstrap[i]->ShowMessageBox(messageboxdata, buttonID)) {
                            result = true;
                        }
                        break;
                    }
                }

                driver_attempt = (driver_attempt_end) ? (driver_attempt_end + 1) : NULL;
            }
        } else {
            for (int i = 0; bootstrap[i]; ++i) {
                if (bootstrap[i]->ShowMessageBox && bootstrap[i]->ShowMessageBox(messageboxdata, buttonID)) {
                    result = true;
                    break;
                }
            }
        }
    }

    if (!result) {
        const char *error = SDL_GetError();

        if (!*error) {
            SDL_SetError("No message system available");
        }
    } else {
        SDL_ClearError();
    }

    (void)SDL_AtomicDecRef(&SDL_messagebox_count);

    if (current_window) {
        SDL_RaiseWindow(current_window);
    }

    if (!show_cursor_prev) {
        SDL_HideCursor();
    }
    SDL_UpdateRelativeMouseMode();
    SDL_UpdateMouseCapture(false);

    SDL_small_free(msgcpy, msgisstack);
    SDL_small_free(titlecpy, titleisstack);

    return result;
}

bool SDL_ShowSimpleMessageBox(SDL_MessageBoxFlags flags, const char *title, const char *message, SDL_Window *window)
{
#if defined(SDL_PLATFORM_3DS)
    errorConf errCnf;
    bool hasGpuRight;

    // If the video subsystem has not been initialised, set up graphics temporarily
    hasGpuRight = gspHasGpuRight();
    if (!hasGpuRight)
        gfxInitDefault();

    errorInit(&errCnf, ERROR_TEXT_WORD_WRAP, CFG_LANGUAGE_EN);
    errorText(&errCnf, message);
    errorDisp(&errCnf);

    if (!hasGpuRight)
        gfxExit();

    return true;
#else
    SDL_MessageBoxData data;
    SDL_MessageBoxButtonData button;

    SDL_zero(data);
    data.flags = flags;
    data.title = title;
    data.message = message;
    data.numbuttons = 1;
    data.buttons = &button;
    data.window = window;

    SDL_zero(button);
    button.flags |= SDL_MESSAGEBOX_BUTTON_RETURNKEY_DEFAULT;
    button.flags |= SDL_MESSAGEBOX_BUTTON_ESCAPEKEY_DEFAULT;
    button.text = "OK";

    return SDL_ShowMessageBox(&data, NULL);
#endif
}

bool SDL_ShouldAllowTopmost(void)
{
    return SDL_GetHintBoolean(SDL_HINT_WINDOW_ALLOW_TOPMOST, true);
}

bool SDL_ShowWindowSystemMenu(SDL_Window *window, int x, int y)
{
    CHECK_WINDOW_MAGIC(window, false)
    CHECK_WINDOW_NOT_POPUP(window, false)

    if (_this->ShowWindowSystemMenu) {
        _this->ShowWindowSystemMenu(window, x, y);
        return true;
    }

    return SDL_Unsupported();
}

bool SDL_SetWindowHitTest(SDL_Window *window, SDL_HitTest callback, void *callback_data)
{
    CHECK_WINDOW_MAGIC(window, false);

    if (!_this->SetWindowHitTest) {
        return SDL_Unsupported();
    }

    window->hit_test = callback;
    window->hit_test_data = callback_data;

    return _this->SetWindowHitTest(window, callback != NULL);
}

bool SDL_SetWindowShape(SDL_Window *window, SDL_Surface *shape)
{
    SDL_PropertiesID props;
    SDL_Surface *surface;

    CHECK_WINDOW_MAGIC(window, false);

    if (!(window->flags & SDL_WINDOW_TRANSPARENT)) {
        return SDL_SetError("Window must be created with SDL_WINDOW_TRANSPARENT");
    }

    props = SDL_GetWindowProperties(window);
    if (!props) {
        return false;
    }

    surface = SDL_ConvertSurface(shape, SDL_PIXELFORMAT_ARGB32);
    if (!surface) {
        return false;
    }

    if (!SDL_SetSurfaceProperty(props, SDL_PROP_WINDOW_SHAPE_POINTER, surface)) {
        return false;
    }

    if (_this->UpdateWindowShape) {
        if (!_this->UpdateWindowShape(_this, window, surface)) {
            return false;
        }
    }
    return true;
}

/*
 * Functions used by iOS application delegates
 */
void SDL_OnApplicationWillTerminate(void)
{
    SDL_SendAppEvent(SDL_EVENT_TERMINATING);
}

void SDL_OnApplicationDidReceiveMemoryWarning(void)
{
    SDL_SendAppEvent(SDL_EVENT_LOW_MEMORY);
}

void SDL_OnApplicationWillEnterBackground(void)
{
    if (_this) {
        SDL_Window *window;
        for (window = _this->windows; window; window = window->next) {
            SDL_SendWindowEvent(window, SDL_EVENT_WINDOW_MINIMIZED, 0, 0);
        }
        SDL_SetKeyboardFocus(NULL);
    }
    SDL_SendAppEvent(SDL_EVENT_WILL_ENTER_BACKGROUND);
}

void SDL_OnApplicationDidEnterBackground(void)
{
    SDL_SendAppEvent(SDL_EVENT_DID_ENTER_BACKGROUND);
}

void SDL_OnApplicationWillEnterForeground(void)
{
    SDL_SendAppEvent(SDL_EVENT_WILL_ENTER_FOREGROUND);
}

void SDL_OnApplicationDidEnterForeground(void)
{
    SDL_SendAppEvent(SDL_EVENT_DID_ENTER_FOREGROUND);

    if (_this) {
        SDL_Window *window;
        for (window = _this->windows; window; window = window->next) {
            SDL_SetKeyboardFocus(window);
            SDL_SendWindowEvent(window, SDL_EVENT_WINDOW_RESTORED, 0, 0);
        }
    }
}

#define NOT_A_VULKAN_WINDOW "The specified window isn't a Vulkan window"

bool SDL_Vulkan_LoadLibrary(const char *path)
{
    bool result;

    if (!_this) {
        return SDL_UninitializedVideo();
    }
    if (_this->vulkan_config.loader_loaded) {
        if (path && SDL_strcmp(path, _this->vulkan_config.loader_path) != 0) {
            return SDL_SetError("Vulkan loader library already loaded");
        }
        result = true;
    } else {
        if (!_this->Vulkan_LoadLibrary) {
            return SDL_DllNotSupported("Vulkan");
        }
        result = _this->Vulkan_LoadLibrary(_this, path);
    }
    if (result) {
        _this->vulkan_config.loader_loaded++;
    }
    return result;
}

SDL_FunctionPointer SDL_Vulkan_GetVkGetInstanceProcAddr(void)
{
    if (!_this) {
        SDL_UninitializedVideo();
        return NULL;
    }
    if (!_this->vulkan_config.loader_loaded) {
        SDL_SetError("No Vulkan loader has been loaded");
        return NULL;
    }
    return (SDL_FunctionPointer)_this->vulkan_config.vkGetInstanceProcAddr;
}

void SDL_Vulkan_UnloadLibrary(void)
{
    if (!_this) {
        SDL_UninitializedVideo();
        return;
    }
    if (_this->vulkan_config.loader_loaded > 0) {
        if (--_this->vulkan_config.loader_loaded > 0) {
            return;
        }
        if (_this->Vulkan_UnloadLibrary) {
            _this->Vulkan_UnloadLibrary(_this);
        }
    }
}

char const* const* SDL_Vulkan_GetInstanceExtensions(Uint32 *count)
{
    return _this->Vulkan_GetInstanceExtensions(_this, count);
}

bool SDL_Vulkan_CreateSurface(SDL_Window *window,
                                  VkInstance instance,
                                  const struct VkAllocationCallbacks *allocator,
                                  VkSurfaceKHR *surface)
{
    CHECK_WINDOW_MAGIC(window, false);

    if (!(window->flags & SDL_WINDOW_VULKAN)) {
        return SDL_SetError(NOT_A_VULKAN_WINDOW);
    }

    if (!instance) {
        return SDL_InvalidParamError("instance");
    }

    if (!surface) {
        return SDL_InvalidParamError("surface");
    }

    return _this->Vulkan_CreateSurface(_this, window, instance, allocator, surface);
}

void SDL_Vulkan_DestroySurface(VkInstance instance,
                               VkSurfaceKHR surface,
                               const struct VkAllocationCallbacks *allocator)
{
    if (_this && instance && surface && _this->Vulkan_DestroySurface) {
        _this->Vulkan_DestroySurface(_this, instance, surface, allocator);
    }
}

bool SDL_Vulkan_GetPresentationSupport(VkInstance instance,
                                           VkPhysicalDevice physicalDevice,
                                           Uint32 queueFamilyIndex)
{
    if (!_this) {
        SDL_UninitializedVideo();
        return false;
    }

    if (!instance) {
        SDL_InvalidParamError("instance");
        return false;
    }

    if (!physicalDevice) {
        SDL_InvalidParamError("physicalDevice");
        return false;
    }

    if (_this->Vulkan_GetPresentationSupport) {
        return _this->Vulkan_GetPresentationSupport(_this, instance, physicalDevice, queueFamilyIndex);
    }

    /* If the backend does not have this function then it does not have a
     * WSI function to query it; in other words it's not necessary to check
     * as it is always supported.
     */
    return true;
}

SDL_MetalView SDL_Metal_CreateView(SDL_Window *window)
{
    CHECK_WINDOW_MAGIC(window, NULL);

    if (!_this->Metal_CreateView) {
        SDL_Unsupported();
        return NULL;
    }

    if (!(window->flags & SDL_WINDOW_METAL)) {
        // No problem, we can convert to Metal
        if (window->flags & SDL_WINDOW_OPENGL) {
            window->flags &= ~SDL_WINDOW_OPENGL;
            SDL_GL_UnloadLibrary();
        }
        if (window->flags & SDL_WINDOW_VULKAN) {
            window->flags &= ~SDL_WINDOW_VULKAN;
            SDL_Vulkan_UnloadLibrary();
        }
        window->flags |= SDL_WINDOW_METAL;
    }

    return _this->Metal_CreateView(_this, window);
}

void SDL_Metal_DestroyView(SDL_MetalView view)
{
    if (_this && view && _this->Metal_DestroyView) {
        _this->Metal_DestroyView(_this, view);
    }
}

void *SDL_Metal_GetLayer(SDL_MetalView view)
{
    if (_this && _this->Metal_GetLayer) {
        if (view) {
            return _this->Metal_GetLayer(_this, view);
        } else {
            SDL_InvalidParamError("view");
            return NULL;
        }
    } else {
        SDL_SetError("Metal is not supported.");
        return NULL;
    }
}

#if defined(SDL_VIDEO_DRIVER_X11) || defined(SDL_VIDEO_DRIVER_WAYLAND) || defined(SDL_VIDEO_DRIVER_EMSCRIPTEN)
const char *SDL_GetCSSCursorName(SDL_SystemCursor id, const char **fallback_name)
{
    // Reference: https://www.w3.org/TR/css-ui-4/#cursor
    // Also in: https://www.freedesktop.org/wiki/Specifications/cursor-spec/
    switch (id) {
    case SDL_SYSTEM_CURSOR_DEFAULT:
        return "default";

    case SDL_SYSTEM_CURSOR_TEXT:
        return "text";

    case SDL_SYSTEM_CURSOR_WAIT:
        return "wait";

    case SDL_SYSTEM_CURSOR_CROSSHAIR:
        return "crosshair";

    case SDL_SYSTEM_CURSOR_PROGRESS:
        return "progress";

    case SDL_SYSTEM_CURSOR_NWSE_RESIZE:
        if (fallback_name) {
            // only a single arrow
            *fallback_name = "nw-resize";
        }
        return "nwse-resize";

    case SDL_SYSTEM_CURSOR_NESW_RESIZE:
        if (fallback_name) {
            // only a single arrow
            *fallback_name = "ne-resize";
        }
        return "nesw-resize";

    case SDL_SYSTEM_CURSOR_EW_RESIZE:
        if (fallback_name) {
            *fallback_name = "col-resize";
        }
        return "ew-resize";

    case SDL_SYSTEM_CURSOR_NS_RESIZE:
        if (fallback_name) {
            *fallback_name = "row-resize";
        }
        return "ns-resize";

    case SDL_SYSTEM_CURSOR_MOVE:
        return "all-scroll";

    case SDL_SYSTEM_CURSOR_NOT_ALLOWED:
        return "not-allowed";

    case SDL_SYSTEM_CURSOR_POINTER:
        return "pointer";

    case SDL_SYSTEM_CURSOR_NW_RESIZE:
        return "nw-resize";

    case SDL_SYSTEM_CURSOR_N_RESIZE:
        return "n-resize";

    case SDL_SYSTEM_CURSOR_NE_RESIZE:
        return "ne-resize";

    case SDL_SYSTEM_CURSOR_E_RESIZE:
        return "e-resize";

    case SDL_SYSTEM_CURSOR_SE_RESIZE:
        return "se-resize";

    case SDL_SYSTEM_CURSOR_S_RESIZE:
        return "s-resize";

    case SDL_SYSTEM_CURSOR_SW_RESIZE:
        return "sw-resize";

    case SDL_SYSTEM_CURSOR_W_RESIZE:
        return "w-resize";

    default:
        return "default";
    }
}
#endif
