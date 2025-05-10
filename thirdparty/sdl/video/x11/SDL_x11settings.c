/*
  Simple DirectMedia Layer
  Copyright 2024 Igalia S.L.

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

#if defined(SDL_VIDEO_DRIVER_X11)

#include "SDL_x11video.h"
#include "SDL_x11settings.h"

#define SDL_XSETTINGS_GDK_WINDOW_SCALING_FACTOR "Gdk/WindowScalingFactor"
#define SDL_XSETTINGS_XFT_DPI "Xft/DPI"

static void X11_XsettingsNotify(const char *name, XSettingsAction action, XSettingsSetting *setting, void *data)
{
    SDL_VideoDevice *_this = data;
    float scale_factor = 1.0;
    int i;

    if (SDL_strcmp(name, SDL_XSETTINGS_GDK_WINDOW_SCALING_FACTOR) != 0 ||
        SDL_strcmp(name, SDL_XSETTINGS_XFT_DPI) != 0) {
        return;
    }

    if (setting->type != XSETTINGS_TYPE_INT) {
        return;
    }

    switch (action) {
    case XSETTINGS_ACTION_NEW:
        SDL_FALLTHROUGH;
    case XSETTINGS_ACTION_CHANGED:
        scale_factor = setting->data.v_int;
        if (SDL_strcmp(name, SDL_XSETTINGS_XFT_DPI) == 0) {
            scale_factor = scale_factor / 1024.0f / 96.0f;
        }
        break;
    case XSETTINGS_ACTION_DELETED:
        scale_factor = 1.0;
        break;
    }

    if (_this) {
        for (i = 0; i < _this->num_displays; ++i) {
            SDL_SetDisplayContentScale(_this->displays[i], scale_factor);
        }
    }
}

void X11_InitXsettings(SDL_VideoDevice *_this)
{
    SDL_VideoData *data = _this->internal;
    SDLX11_SettingsData *xsettings_data = &data->xsettings_data;

    xsettings_data->xsettings = xsettings_client_new(data->display,
        DefaultScreen(data->display), X11_XsettingsNotify, NULL, _this);

}

void X11_QuitXsettings(SDL_VideoDevice *_this)
{
    SDL_VideoData *data = _this->internal;
    SDLX11_SettingsData *xsettings_data = &data->xsettings_data;

    if (xsettings_data->xsettings) {
        xsettings_client_destroy(xsettings_data->xsettings);
        xsettings_data->xsettings = NULL;
    }
}

void X11_HandleXsettings(SDL_VideoDevice *_this, const XEvent *xevent)
{
    SDL_VideoData *data = _this->internal;
    SDLX11_SettingsData *xsettings_data = &data->xsettings_data;

    if (xsettings_data->xsettings) {
        if (!xsettings_client_process_event(xsettings_data->xsettings, xevent)) {
            xsettings_client_destroy(xsettings_data->xsettings);
            xsettings_data->xsettings = NULL;
        }
    }
}

int X11_GetXsettingsIntKey(SDL_VideoDevice *_this, const char *key, int fallback_value) {
    SDL_VideoData *data = _this->internal;
    SDLX11_SettingsData *xsettings_data = &data->xsettings_data;
    XSettingsSetting *setting = NULL;
    int res = fallback_value;


    if (xsettings_data->xsettings) {
        if (xsettings_client_get_setting(xsettings_data->xsettings, key, &setting) != XSETTINGS_SUCCESS) {
            goto no_key;
        }

        if (setting->type != XSETTINGS_TYPE_INT) {
            goto no_key;
        }

        res = setting->data.v_int;
    }

no_key:
    if (setting) {
        xsettings_setting_free(setting);
    }

    return res;
}

#endif // SDL_VIDEO_DRIVER_X11
