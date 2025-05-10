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

#ifdef SDL_VIDEO_DRIVER_EMSCRIPTEN

#include "../SDL_sysvideo.h"
#include "../SDL_pixels_c.h"
#include "../../events/SDL_events_c.h"

#include "SDL_emscriptenvideo.h"
#include "SDL_emscriptenopengles.h"
#include "SDL_emscriptenframebuffer.h"
#include "SDL_emscriptenevents.h"
#include "SDL_emscriptenmouse.h"

#define EMSCRIPTENVID_DRIVER_NAME "emscripten"

// Initialization/Query functions
static bool Emscripten_VideoInit(SDL_VideoDevice *_this);
static bool Emscripten_SetDisplayMode(SDL_VideoDevice *_this, SDL_VideoDisplay *display, SDL_DisplayMode *mode);
static void Emscripten_VideoQuit(SDL_VideoDevice *_this);
static bool Emscripten_GetDisplayUsableBounds(SDL_VideoDevice *_this, SDL_VideoDisplay *display, SDL_Rect *rect);

static bool Emscripten_CreateWindow(SDL_VideoDevice *_this, SDL_Window *window, SDL_PropertiesID create_props);
static void Emscripten_SetWindowSize(SDL_VideoDevice *_this, SDL_Window *window);
static void Emscripten_GetWindowSizeInPixels(SDL_VideoDevice *_this, SDL_Window *window, int *w, int *h);
static void Emscripten_DestroyWindow(SDL_VideoDevice *_this, SDL_Window *window);
static SDL_FullscreenResult Emscripten_SetWindowFullscreen(SDL_VideoDevice *_this, SDL_Window *window, SDL_VideoDisplay *display, SDL_FullscreenOp fullscreen);
static void Emscripten_PumpEvents(SDL_VideoDevice *_this);
static void Emscripten_SetWindowTitle(SDL_VideoDevice *_this, SDL_Window *window);

static bool pumpevents_has_run = false;
static int pending_swap_interval = -1;


// Emscripten driver bootstrap functions

static void Emscripten_DeleteDevice(SDL_VideoDevice *device)
{
    SDL_free(device);
}

static SDL_SystemTheme Emscripten_GetSystemTheme(void)
{
    /* Technically, light theme can mean explicit light theme or no preference.
       https://developer.mozilla.org/en-US/docs/Web/CSS/@media/prefers-color-scheme#syntax */

    int theme_code = EM_ASM_INT({
        if (!window.matchMedia) {
            return -1;
        }

        if (window.matchMedia('(prefers-color-scheme: light)').matches) {
            return 0;
        }

        if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
            return 1;
        }

        return -1;
    });

    switch (theme_code) {
    case 0:
        return SDL_SYSTEM_THEME_LIGHT;

    case 1:
        return SDL_SYSTEM_THEME_DARK;

    default:
        return SDL_SYSTEM_THEME_UNKNOWN;
    }
}

static void Emscripten_ListenSystemTheme(void)
{
    MAIN_THREAD_EM_ASM({
        if (window.matchMedia) {
            if (typeof(Module['SDL3']) === 'undefined') {
                Module['SDL3'] = {};
            }

            var SDL3 = Module['SDL3'];

            SDL3.eventHandlerThemeChanged = function(event) {
                _Emscripten_SendSystemThemeChangedEvent();
            };

            SDL3.themeChangedMatchMedia = window.matchMedia('(prefers-color-scheme: dark)');
            SDL3.themeChangedMatchMedia.addEventListener('change', SDL3.eventHandlerThemeChanged);
        }
    });
}

static void Emscripten_UnlistenSystemTheme(void)
{
    MAIN_THREAD_EM_ASM({
        if (typeof(Module['SDL3']) !== 'undefined') {
            var SDL3 = Module['SDL3'];

            SDL3.themeChangedMatchMedia.removeEventListener('change', SDL3.eventHandlerThemeChanged);
            SDL3.themeChangedMatchMedia = undefined;
            SDL3.eventHandlerThemeChanged = undefined;
        }
    });
}

EMSCRIPTEN_KEEPALIVE void Emscripten_SendSystemThemeChangedEvent(void)
{
    SDL_SetSystemTheme(Emscripten_GetSystemTheme());
}

static SDL_VideoDevice *Emscripten_CreateDevice(void)
{
    SDL_VideoDevice *device;

    // Initialize all variables that we clean on shutdown
    device = (SDL_VideoDevice *)SDL_calloc(1, sizeof(SDL_VideoDevice));
    if (!device) {
        return NULL;
    }

    /* Firefox sends blur event which would otherwise prevent full screen
     * when the user clicks to allow full screen.
     * See https://bugzilla.mozilla.org/show_bug.cgi?id=1144964
     */
    SDL_SetHint(SDL_HINT_VIDEO_MINIMIZE_ON_FOCUS_LOSS, "0");

    // Set the function pointers
    device->VideoInit = Emscripten_VideoInit;
    device->VideoQuit = Emscripten_VideoQuit;
    device->GetDisplayUsableBounds = Emscripten_GetDisplayUsableBounds;
    device->SetDisplayMode = Emscripten_SetDisplayMode;

    device->PumpEvents = Emscripten_PumpEvents;

    device->CreateSDLWindow = Emscripten_CreateWindow;
    device->SetWindowTitle = Emscripten_SetWindowTitle;
    /*device->SetWindowIcon = Emscripten_SetWindowIcon;
    device->SetWindowPosition = Emscripten_SetWindowPosition;*/
    device->SetWindowSize = Emscripten_SetWindowSize;
    /*device->ShowWindow = Emscripten_ShowWindow;
    device->HideWindow = Emscripten_HideWindow;
    device->RaiseWindow = Emscripten_RaiseWindow;
    device->MaximizeWindow = Emscripten_MaximizeWindow;
    device->MinimizeWindow = Emscripten_MinimizeWindow;
    device->RestoreWindow = Emscripten_RestoreWindow;
    device->SetWindowMouseGrab = Emscripten_SetWindowMouseGrab;*/
    device->GetWindowSizeInPixels = Emscripten_GetWindowSizeInPixels;
    device->DestroyWindow = Emscripten_DestroyWindow;
    device->SetWindowFullscreen = Emscripten_SetWindowFullscreen;

    device->CreateWindowFramebuffer = Emscripten_CreateWindowFramebuffer;
    device->UpdateWindowFramebuffer = Emscripten_UpdateWindowFramebuffer;
    device->DestroyWindowFramebuffer = Emscripten_DestroyWindowFramebuffer;

    device->GL_LoadLibrary = Emscripten_GLES_LoadLibrary;
    device->GL_GetProcAddress = Emscripten_GLES_GetProcAddress;
    device->GL_UnloadLibrary = Emscripten_GLES_UnloadLibrary;
    device->GL_CreateContext = Emscripten_GLES_CreateContext;
    device->GL_MakeCurrent = Emscripten_GLES_MakeCurrent;
    device->GL_SetSwapInterval = Emscripten_GLES_SetSwapInterval;
    device->GL_GetSwapInterval = Emscripten_GLES_GetSwapInterval;
    device->GL_SwapWindow = Emscripten_GLES_SwapWindow;
    device->GL_DestroyContext = Emscripten_GLES_DestroyContext;

    device->free = Emscripten_DeleteDevice;

    Emscripten_ListenSystemTheme();
    device->system_theme = Emscripten_GetSystemTheme();

    return device;
}

static bool Emscripten_ShowMessagebox(const SDL_MessageBoxData *messageboxdata, int *buttonID) {
    if (emscripten_has_asyncify() && SDL_GetHintBoolean(SDL_HINT_EMSCRIPTEN_ASYNCIFY, true)) {
        char dialog_background[32];
        char dialog_color[32];
        char button_border[32];
        char button_background[32];
        char button_hovered[32];

        if (messageboxdata->colorScheme) {
            SDL_MessageBoxColor color = messageboxdata->colorScheme->colors[SDL_MESSAGEBOX_COLOR_BACKGROUND];
            SDL_snprintf(dialog_background, sizeof(dialog_background), "rgb(%u, %u, %u)", color.r, color.g, color.b);

            color = messageboxdata->colorScheme->colors[SDL_MESSAGEBOX_COLOR_TEXT];
            SDL_snprintf(dialog_color, sizeof(dialog_color), "rgb(%u, %u, %u)", color.r, color.g, color.b);

            color = messageboxdata->colorScheme->colors[SDL_MESSAGEBOX_COLOR_BUTTON_BORDER];
            SDL_snprintf(button_border, sizeof(button_border), "rgb(%u, %u, %u)", color.r, color.g, color.b);

            color = messageboxdata->colorScheme->colors[SDL_MESSAGEBOX_COLOR_BUTTON_BACKGROUND];
            SDL_snprintf(button_background, sizeof(button_background), "rgb(%u, %u, %u)", color.r, color.g, color.b);

            color = messageboxdata->colorScheme->colors[SDL_MESSAGEBOX_COLOR_BUTTON_SELECTED];
            SDL_snprintf(button_hovered, sizeof(button_hovered), "rgb(%u, %u, %u)", color.r, color.g, color.b);
        } else {
            SDL_zero(dialog_background);
            SDL_zero(dialog_color);
            SDL_zero(button_border);
            SDL_zero(button_background);
            SDL_zero(button_hovered);
        }

        // TODO: Handle parent window when multiple windows can be added in Emscripten builds
        char dialog_id[64];
        SDL_snprintf(dialog_id, sizeof(dialog_id), "SDL3_messagebox_%u", SDL_rand_bits());
        EM_ASM({
            var title = UTF8ToString($0);
            var message = UTF8ToString($1);
            var background = UTF8ToString($2);
            var color = UTF8ToString($3);
            var id = UTF8ToString($4);

            // Dialogs are always put in the front of the DOM
            var dialog = document.createElement("dialog");
            // Set class to allow for CSS selectors
            dialog.classList.add("SDL3_messagebox");
            dialog.id = id;
            dialog.style.color = color;
            dialog.style.backgroundColor = background;
            document.body.append(dialog);

            var h1 = document.createElement("h1");
            h1.innerText = title;
            dialog.append(h1);

            var p = document.createElement("p");
            p.innerText = message;
            dialog.append(p);

            dialog.showModal();
        }, messageboxdata->title, messageboxdata->message, dialog_background, dialog_color, dialog_id);

        int i;
        for (i = 0; i < messageboxdata->numbuttons; ++i) {
            SDL_MessageBoxButtonData button = messageboxdata->buttons[i];

            const int created = EM_ASM_INT({
                    var dialog_id = UTF8ToString($0);
                    var text = UTF8ToString($1);
                    var responseId = $2;
                    var clickOnReturn = $3;
                    var clickOnEscape = $4;
                    var border = UTF8ToString($5);
                    var background = UTF8ToString($6);
                    var hovered = UTF8ToString($7);

                    var dialog = document.getElementById(dialog_id);
                    if (!dialog) {
                        return false;
                    }

                    var button = document.createElement("button");
                    button.innerText = text;
                    button.style.borderColor = border;
                    button.style.backgroundColor = background;

                    dialog.addEventListener('keydown', function(e) {
                        if (clickOnReturn && e.key === "Enter") {
                            e.preventDefault();
                            button.click();
                        } else if (clickOnEscape && e.key === "Escape") {
                            e.preventDefault();
                            button.click();
                        }
                    });
                    dialog.addEventListener('cancel', function(e){
                        e.preventDefault();
                    });

                    button.onmouseenter = function(e){
                        button.style.backgroundColor = hovered;
                    };
                    button.onmouseleave = function(e){
                        button.style.backgroundColor = background;
                    };
                    button.onclick = function(e) {
                        dialog.close(responseId);
                    };

                    dialog.append(button);
                    return true;
                },
                dialog_id,
                button.text,
                button.buttonID,
                button.flags & SDL_MESSAGEBOX_BUTTON_RETURNKEY_DEFAULT,
                button.flags & SDL_MESSAGEBOX_BUTTON_ESCAPEKEY_DEFAULT,
                button_border,
                button_background,
                button_hovered
            );

            if (!created) {
                return false;
            }
        }

        while (true) {
            // give back control to browser for screen refresh
            emscripten_sleep(0);

            const int dialog_open = EM_ASM_INT({
                var dialog_id = UTF8ToString($0);

                var dialog = document.getElementById(dialog_id);
                if (!dialog) {
                    return false;
                }
                return dialog.open;
            }, dialog_id);

            if (dialog_open) {
                continue;
            }

            *buttonID = EM_ASM_INT({
                var dialog_id = UTF8ToString($0);
                var dialog = document.getElementById(dialog_id);
                if (!dialog) {
                    return 0;
                }
                try
                {
                    return parseInt(dialog.returnValue);
                }
                catch(e)
                {
                    return 0;
                }
            }, dialog_id);
            break;
        }

    } else {
        // Cannot add elements to DOM and block without Asyncify. So, fall back to the alert function.
        EM_ASM({
            alert(UTF8ToString($0) + "\n\n" + UTF8ToString($1));
        }, messageboxdata->title, messageboxdata->message);
    }
    return true;
}

VideoBootStrap Emscripten_bootstrap = {
    EMSCRIPTENVID_DRIVER_NAME, "SDL emscripten video driver",
    Emscripten_CreateDevice,
    Emscripten_ShowMessagebox,
    false
};

bool Emscripten_VideoInit(SDL_VideoDevice *_this)
{
    SDL_DisplayMode mode;

    // Use a fake 32-bpp desktop mode
    SDL_zero(mode);
    mode.format = SDL_PIXELFORMAT_XRGB8888;
    emscripten_get_screen_size(&mode.w, &mode.h);
    mode.pixel_density = emscripten_get_device_pixel_ratio();

    if (SDL_AddBasicVideoDisplay(&mode) == 0) {
        return false;
    }

    Emscripten_InitMouse();

    // Assume we have a mouse and keyboard
    SDL_AddKeyboard(SDL_DEFAULT_KEYBOARD_ID, NULL, false);
    SDL_AddMouse(SDL_DEFAULT_MOUSE_ID, NULL, false);

    Emscripten_RegisterGlobalEventHandlers(_this);

    // We're done!
    return true;
}

static bool Emscripten_SetDisplayMode(SDL_VideoDevice *_this, SDL_VideoDisplay *display, SDL_DisplayMode *mode)
{
    // can't do this
    return true;
}

static void Emscripten_VideoQuit(SDL_VideoDevice *_this)
{
    Emscripten_UnregisterGlobalEventHandlers(_this);
    Emscripten_QuitMouse();
    Emscripten_UnlistenSystemTheme();
    pumpevents_has_run = false;
    pending_swap_interval = -1;
}

static bool Emscripten_GetDisplayUsableBounds(SDL_VideoDevice *_this, SDL_VideoDisplay *display, SDL_Rect *rect)
{
    if (rect) {
        rect->x = 0;
        rect->y = 0;
        rect->w = MAIN_THREAD_EM_ASM_INT({
            return window.innerWidth;
        });
        rect->h = MAIN_THREAD_EM_ASM_INT({
            return window.innerHeight;
        });
    }
    return true;
}

bool Emscripten_ShouldSetSwapInterval(int interval)
{
    if (!pumpevents_has_run) {
        pending_swap_interval = interval;
        return false;
    }
    return true;
}

static void Emscripten_PumpEvents(SDL_VideoDevice *_this)
{
    if (!pumpevents_has_run) {
        // we assume you've set a mainloop by the time you've called pumpevents, so we delay initial SetInterval changes until then.
        // otherwise you'll get a warning on the javascript console.
        pumpevents_has_run = true;
        if (pending_swap_interval >= 0) {
            Emscripten_GLES_SetSwapInterval(_this, pending_swap_interval);
            pending_swap_interval = -1;
        }
    }
}

EMSCRIPTEN_KEEPALIVE void requestFullscreenThroughSDL(SDL_Window *window)
{
    SDL_SetWindowFullscreen(window, true);
}

static bool Emscripten_CreateWindow(SDL_VideoDevice *_this, SDL_Window *window, SDL_PropertiesID props)
{
    SDL_WindowData *wdata;
    double scaled_w, scaled_h;
    double css_w, css_h;
    const char *selector;

    // Allocate window internal data
    wdata = (SDL_WindowData *)SDL_calloc(1, sizeof(SDL_WindowData));
    if (!wdata) {
        return false;
    }

    selector = SDL_GetHint(SDL_HINT_EMSCRIPTEN_CANVAS_SELECTOR);
    if (!selector || !*selector) {
        selector = SDL_GetStringProperty(props, SDL_PROP_WINDOW_CREATE_EMSCRIPTEN_CANVAS_ID_STRING, "#canvas");
    }
    wdata->canvas_id = SDL_strdup(selector);

    selector = SDL_GetHint(SDL_HINT_EMSCRIPTEN_KEYBOARD_ELEMENT);
    if (!selector || !*selector) {
        selector = SDL_GetStringProperty(props, SDL_PROP_WINDOW_CREATE_EMSCRIPTEN_KEYBOARD_ELEMENT_STRING, "#window");
    }
    wdata->keyboard_element = SDL_strdup(selector);

    if (window->flags & SDL_WINDOW_HIGH_PIXEL_DENSITY) {
        wdata->pixel_ratio = emscripten_get_device_pixel_ratio();
    } else {
        wdata->pixel_ratio = 1.0f;
    }

    scaled_w = SDL_floor(window->w * wdata->pixel_ratio);
    scaled_h = SDL_floor(window->h * wdata->pixel_ratio);

    // set a fake size to check if there is any CSS sizing the canvas
    emscripten_set_canvas_element_size(wdata->canvas_id, 1, 1);
    emscripten_get_element_css_size(wdata->canvas_id, &css_w, &css_h);

    wdata->external_size = SDL_floor(css_w) != 1 || SDL_floor(css_h) != 1;

    if ((window->flags & SDL_WINDOW_RESIZABLE) && wdata->external_size) {
        // external css has resized us
        scaled_w = css_w * wdata->pixel_ratio;
        scaled_h = css_h * wdata->pixel_ratio;

        SDL_SendWindowEvent(window, SDL_EVENT_WINDOW_RESIZED, SDL_lroundf(css_w), SDL_lroundf(css_h));
    }
    emscripten_set_canvas_element_size(wdata->canvas_id, SDL_lroundf(scaled_w), SDL_lroundf(scaled_h));

    // if the size is not being controlled by css, we need to scale down for hidpi
    if (!wdata->external_size) {
        if (wdata->pixel_ratio != 1.0f) {
            // scale canvas down
            emscripten_set_element_css_size(wdata->canvas_id, window->w, window->h);
        }
    }

    wdata->window = window;

    // Setup driver data for this window
    window->internal = wdata;

    // One window, it always has focus
    SDL_SetMouseFocus(window);
    SDL_SetKeyboardFocus(window);

    Emscripten_RegisterEventHandlers(wdata);

    // disable the emscripten "fullscreen" button.
    MAIN_THREAD_EM_ASM({
        Module['requestFullscreen'] = function(lockPointer, resizeCanvas) {
            _requestFullscreenThroughSDL($0);
        };
    }, window);

    // Ensure canvas_id and keyboard_element are added to the window's properties
    SDL_SetStringProperty(window->props, SDL_PROP_WINDOW_EMSCRIPTEN_CANVAS_ID_STRING, wdata->canvas_id);
    SDL_SetStringProperty(window->props, SDL_PROP_WINDOW_EMSCRIPTEN_KEYBOARD_ELEMENT_STRING, wdata->keyboard_element);

    // Window has been successfully created
    return true;
}

static void Emscripten_SetWindowSize(SDL_VideoDevice *_this, SDL_Window *window)
{
    SDL_WindowData *data;

    if (window->internal) {
        data = window->internal;
        // update pixel ratio
        if (window->flags & SDL_WINDOW_HIGH_PIXEL_DENSITY) {
            data->pixel_ratio = emscripten_get_device_pixel_ratio();
        }
        emscripten_set_canvas_element_size(data->canvas_id, SDL_lroundf(window->pending.w * data->pixel_ratio), SDL_lroundf(window->pending.h * data->pixel_ratio));

        // scale canvas down
        if (!data->external_size && data->pixel_ratio != 1.0f) {
            emscripten_set_element_css_size(data->canvas_id, window->pending.w, window->pending.h);
        }

        SDL_SendWindowEvent(window, SDL_EVENT_WINDOW_RESIZED, window->pending.w, window->pending.h);
    }
}

static void Emscripten_GetWindowSizeInPixels(SDL_VideoDevice *_this, SDL_Window *window, int *w, int *h)
{
    SDL_WindowData *data;
    if (window->internal) {
        data = window->internal;
        *w = SDL_lroundf(window->w * data->pixel_ratio);
        *h = SDL_lroundf(window->h * data->pixel_ratio);
    }
}

static void Emscripten_DestroyWindow(SDL_VideoDevice *_this, SDL_Window *window)
{
    SDL_WindowData *data;

    if (window->internal) {
        data = window->internal;

        Emscripten_UnregisterEventHandlers(data);

        // We can't destroy the canvas, so resize it to zero instead
        emscripten_set_canvas_element_size(data->canvas_id, 0, 0);
        SDL_free(data->canvas_id);

        SDL_free(data->keyboard_element);

        SDL_free(window->internal);
        window->internal = NULL;
    }

    // just ignore clicks on the fullscreen button while there's no SDL window.
    MAIN_THREAD_EM_ASM({ Module['requestFullscreen'] = function(lockPointer, resizeCanvas) {}; });
}

static SDL_FullscreenResult Emscripten_SetWindowFullscreen(SDL_VideoDevice *_this, SDL_Window *window, SDL_VideoDisplay *display, SDL_FullscreenOp fullscreen)
{
    SDL_WindowData *data;
    int res = -1;

    if (window->internal) {
        data = window->internal;

        if (fullscreen) {
            EmscriptenFullscreenStrategy strategy;
            bool is_fullscreen_desktop = !window->fullscreen_exclusive;

            SDL_zero(strategy);
            strategy.scaleMode = is_fullscreen_desktop ? EMSCRIPTEN_FULLSCREEN_SCALE_STRETCH : EMSCRIPTEN_FULLSCREEN_SCALE_ASPECT;

            if (!is_fullscreen_desktop) {
                strategy.canvasResolutionScaleMode = EMSCRIPTEN_FULLSCREEN_CANVAS_SCALE_NONE;
            } else if (window->flags & SDL_WINDOW_HIGH_PIXEL_DENSITY) {
                strategy.canvasResolutionScaleMode = EMSCRIPTEN_FULLSCREEN_CANVAS_SCALE_HIDEF;
            } else {
                strategy.canvasResolutionScaleMode = EMSCRIPTEN_FULLSCREEN_CANVAS_SCALE_STDDEF;
            }

            strategy.filteringMode = EMSCRIPTEN_FULLSCREEN_FILTERING_DEFAULT;

            strategy.canvasResizedCallback = Emscripten_HandleCanvasResize;
            strategy.canvasResizedCallbackUserData = data;

            data->fullscreen_mode_flags = (window->flags & SDL_WINDOW_FULLSCREEN);
            data->fullscreen_resize = is_fullscreen_desktop;

            res = emscripten_request_fullscreen_strategy(data->canvas_id, 1, &strategy);
        } else {
            res = emscripten_exit_fullscreen();
        }
    }

    if (res == EMSCRIPTEN_RESULT_SUCCESS) {
        return SDL_FULLSCREEN_SUCCEEDED;
    } else if (res == EMSCRIPTEN_RESULT_DEFERRED) {
        return SDL_FULLSCREEN_PENDING;
    } else {
        return SDL_FULLSCREEN_FAILED;
    }
}

static void Emscripten_SetWindowTitle(SDL_VideoDevice *_this, SDL_Window *window)
{
    emscripten_set_window_title(window->title);
}

#endif // SDL_VIDEO_DRIVER_EMSCRIPTEN
