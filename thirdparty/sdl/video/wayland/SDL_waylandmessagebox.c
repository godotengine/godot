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

#ifdef SDL_VIDEO_DRIVER_WAYLAND

#include "SDL_waylandmessagebox.h"

#define ZENITY_VERSION_LEN 32 // Number of bytes to read from zenity --version (including NUL)

#define MAX_BUTTONS 8 // Maximum number of buttons supported

static bool parse_zenity_version(const char *version, int *major, int *minor)
{
    /* We expect the version string is in the form of MAJOR.MINOR.MICRO
     * as described in meson.build. We'll ignore everything after that.
     */
    const char *version_ptr = version;
    char *end_ptr = NULL;
    int tmp = (int) SDL_strtol(version_ptr, &end_ptr, 10);
    if (tmp == 0 && end_ptr == version_ptr) {
        return SDL_SetError("failed to get zenity major version number");
    }
    *major = tmp;

    if (*end_ptr == '.') {
        version_ptr = end_ptr + 1; // skip the dot
        tmp = (int) SDL_strtol(version_ptr, &end_ptr, 10);
        if (tmp == 0 && end_ptr == version_ptr) {
            return SDL_SetError("failed to get zenity minor version number");
        }
        *minor = tmp;
    } else {
        *minor = 0;
    }
    return true;
}

static bool get_zenity_version(int *major, int *minor)
{
    const char *argv[] = { "zenity", "--version", NULL };
    bool result = false;

    SDL_Process *process = SDL_CreateProcess(argv, true);
    if (!process) {
        return false;
    }

    char *output = SDL_ReadProcess(process, NULL, NULL);
    if (output) {
        result = parse_zenity_version(output, major, minor);
        SDL_free(output);
    }
    SDL_DestroyProcess(process);

    return result;
}

bool Wayland_ShowMessageBox(const SDL_MessageBoxData *messageboxdata, int *buttonID)
{
    int zenity_major = 0, zenity_minor = 0, output_len = 0;
    int argc = 5, i;
    const char *argv[5 + 2 /* icon name */ + 2 /* title */ + 2 /* message */ + 2 * MAX_BUTTONS + 1 /* NULL */] = {
        "zenity", "--question", "--switch", "--no-wrap", "--no-markup"
    };
    SDL_Process *process;

    // Are we trying to connect to or are currently in a Wayland session?
    if (!SDL_getenv("WAYLAND_DISPLAY")) {
        const char *session = SDL_getenv("XDG_SESSION_TYPE");
        if (session && SDL_strcasecmp(session, "wayland") != 0) {
            return SDL_SetError("Not on a wayland display");
        }
    }

    if (messageboxdata->numbuttons > MAX_BUTTONS) {
        return SDL_SetError("Too many buttons (%d max allowed)", MAX_BUTTONS);
    }

    // get zenity version so we know which arg to use
    if (!get_zenity_version(&zenity_major, &zenity_minor)) {
        return false; // get_zenity_version() calls SDL_SetError(), so message is already set
    }

    /* https://gitlab.gnome.org/GNOME/zenity/-/commit/c686bdb1b45e95acf010efd9ca0c75527fbb4dea
     * This commit removed --icon-name without adding a deprecation notice.
     * We need to handle it gracefully, otherwise no message box will be shown.
     */
    argv[argc++] = zenity_major > 3 || (zenity_major == 3 && zenity_minor >= 90) ? "--icon" : "--icon-name";
    switch (messageboxdata->flags & (SDL_MESSAGEBOX_ERROR | SDL_MESSAGEBOX_WARNING | SDL_MESSAGEBOX_INFORMATION)) {
    case SDL_MESSAGEBOX_ERROR:
        argv[argc++] = "dialog-error";
        break;
    case SDL_MESSAGEBOX_WARNING:
        argv[argc++] = "dialog-warning";
        break;
    case SDL_MESSAGEBOX_INFORMATION:
    default:
        argv[argc++] = "dialog-information";
        break;
    }

    if (messageboxdata->title && messageboxdata->title[0]) {
        argv[argc++] = "--title";
        argv[argc++] = messageboxdata->title;
    } else {
        argv[argc++] = "--title=";
    }

    if (messageboxdata->message && messageboxdata->message[0]) {
        argv[argc++] = "--text";
        argv[argc++] = messageboxdata->message;
    } else {
        argv[argc++] = "--text=";
    }

    for (i = 0; i < messageboxdata->numbuttons; ++i) {
        if (messageboxdata->buttons[i].text && messageboxdata->buttons[i].text[0]) {
            int len = SDL_strlen(messageboxdata->buttons[i].text);
            if (len > output_len) {
                output_len = len;
            }

            argv[argc++] = "--extra-button";
            argv[argc++] = messageboxdata->buttons[i].text;
        } else {
            argv[argc++] = "--extra-button=";
        }
    }
    if (messageboxdata->numbuttons == 0) {
        argv[argc++] = "--extra-button=OK";
    }
    argv[argc] = NULL;

    SDL_PropertiesID props = SDL_CreateProperties();
    if (!props) {
        return false;
    }
    SDL_SetPointerProperty(props, SDL_PROP_PROCESS_CREATE_ARGS_POINTER, argv);
    // If buttonID is set we need to wait and read the results
    if (buttonID) {
        SDL_SetNumberProperty(props, SDL_PROP_PROCESS_CREATE_STDOUT_NUMBER, SDL_PROCESS_STDIO_APP);
    } else {
        SDL_SetNumberProperty(props, SDL_PROP_PROCESS_CREATE_STDOUT_NUMBER, SDL_PROCESS_STDIO_NULL);
    }
    process = SDL_CreateProcessWithProperties(props);
    SDL_DestroyProperties(props);
    if (!process) {
        return false;
    }
    if (buttonID) {
        char *output = SDL_ReadProcess(process, NULL, NULL);
        if (output) {
            // It likes to add a newline...
            char *tmp = SDL_strrchr(output, '\n');
            if (tmp) {
                *tmp = '\0';
            }

            // Check which button got pressed
            for (i = 0; i < messageboxdata->numbuttons; i += 1) {
                if (messageboxdata->buttons[i].text) {
                    if (SDL_strcmp(output, messageboxdata->buttons[i].text) == 0) {
                        *buttonID = messageboxdata->buttons[i].buttonID;
                        break;
                    }
                }
            }
            SDL_free(output);
        }
    }
    SDL_DestroyProcess(process);

    return true;
}

#endif // SDL_VIDEO_DRIVER_WAYLAND
