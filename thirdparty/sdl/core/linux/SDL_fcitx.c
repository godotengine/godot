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

#include <unistd.h>

#include "SDL_fcitx.h"
#include "../../video/SDL_sysvideo.h"
#include "../../events/SDL_keyboard_c.h"
#include "SDL_dbus.h"

#ifdef SDL_VIDEO_DRIVER_X11
#include "../../video/x11/SDL_x11video.h"
#endif

#define FCITX_DBUS_SERVICE "org.freedesktop.portal.Fcitx"

#define FCITX_IM_DBUS_PATH "/org/freedesktop/portal/inputmethod"

#define FCITX_IM_DBUS_INTERFACE "org.fcitx.Fcitx.InputMethod1"
#define FCITX_IC_DBUS_INTERFACE "org.fcitx.Fcitx.InputContext1"

#define DBUS_TIMEOUT 500

typedef struct FcitxClient
{
    SDL_DBusContext *dbus;

    char *ic_path;

    int id;

    SDL_Rect cursor_rect;
} FcitxClient;

static FcitxClient fcitx_client;

static char *GetAppName(void)
{
#if defined(SDL_PLATFORM_LINUX) || defined(SDL_PLATFORM_FREEBSD)
    char *spot;
    char procfile[1024];
    char linkfile[1024];
    int linksize;

#ifdef SDL_PLATFORM_LINUX
    (void)SDL_snprintf(procfile, sizeof(procfile), "/proc/%d/exe", getpid());
#elif defined(SDL_PLATFORM_FREEBSD)
    (void)SDL_snprintf(procfile, sizeof(procfile), "/proc/%d/file", getpid());
#endif
    linksize = readlink(procfile, linkfile, sizeof(linkfile) - 1);
    if (linksize > 0) {
        linkfile[linksize] = '\0';
        spot = SDL_strrchr(linkfile, '/');
        if (spot) {
            return SDL_strdup(spot + 1);
        } else {
            return SDL_strdup(linkfile);
        }
    }
#endif // SDL_PLATFORM_LINUX || SDL_PLATFORM_FREEBSD

    return SDL_strdup("SDL_App");
}

static size_t Fcitx_GetPreeditString(SDL_DBusContext *dbus,
                       DBusMessage *msg,
                       char **ret,
                       Sint32 *start_pos,
                       Sint32 *end_pos)
{
    char *text = NULL, *subtext;
    size_t text_bytes = 0;
    DBusMessageIter iter, array, sub;
    Sint32 p_start_pos = -1;
    Sint32 p_end_pos = -1;

    dbus->message_iter_init(msg, &iter);
    // Message type is a(si)i, we only need string part
    if (dbus->message_iter_get_arg_type(&iter) == DBUS_TYPE_ARRAY) {
        size_t pos = 0;
        // First pass: calculate string length
        dbus->message_iter_recurse(&iter, &array);
        while (dbus->message_iter_get_arg_type(&array) == DBUS_TYPE_STRUCT) {
            dbus->message_iter_recurse(&array, &sub);
            subtext = NULL;
            if (dbus->message_iter_get_arg_type(&sub) == DBUS_TYPE_STRING) {
                dbus->message_iter_get_basic(&sub, &subtext);
                if (subtext && *subtext) {
                    text_bytes += SDL_strlen(subtext);
                }
            }
            dbus->message_iter_next(&sub);
            if (dbus->message_iter_get_arg_type(&sub) == DBUS_TYPE_INT32 && p_end_pos == -1) {
                // Type is a bit field defined as follows:
                // bit 3: Underline, bit 4: HighLight, bit 5: DontCommit,
                // bit 6: Bold,      bit 7: Strike,    bit 8: Italic
                Sint32 type;
                dbus->message_iter_get_basic(&sub, &type);
                // We only consider highlight
                if (type & (1 << 4)) {
                    if (p_start_pos == -1) {
                        p_start_pos = pos;
                    }
                } else if (p_start_pos != -1 && p_end_pos == -1) {
                    p_end_pos = pos;
                }
            }
            dbus->message_iter_next(&array);
            if (subtext && *subtext) {
                pos += SDL_utf8strlen(subtext);
            }
        }
        if (p_start_pos != -1 && p_end_pos == -1) {
            p_end_pos = pos;
        }
        if (text_bytes) {
            text = SDL_malloc(text_bytes + 1);
        }

        if (text) {
            char *pivot = text;
            // Second pass: join all the sub string
            dbus->message_iter_recurse(&iter, &array);
            while (dbus->message_iter_get_arg_type(&array) == DBUS_TYPE_STRUCT) {
                dbus->message_iter_recurse(&array, &sub);
                if (dbus->message_iter_get_arg_type(&sub) == DBUS_TYPE_STRING) {
                    dbus->message_iter_get_basic(&sub, &subtext);
                    if (subtext && *subtext) {
                        size_t length = SDL_strlen(subtext);
                        SDL_strlcpy(pivot, subtext, length + 1);
                        pivot += length;
                    }
                }
                dbus->message_iter_next(&array);
            }
        } else {
            text_bytes = 0;
        }
    }

    *ret = text;
    *start_pos = p_start_pos;
    *end_pos = p_end_pos;
    return text_bytes;
}

static Sint32 Fcitx_GetPreeditCursorByte(SDL_DBusContext *dbus, DBusMessage *msg)
{
    Sint32 byte = -1;
    DBusMessageIter iter;

    dbus->message_iter_init(msg, &iter);

    dbus->message_iter_next(&iter);

    if (dbus->message_iter_get_arg_type(&iter) != DBUS_TYPE_INT32) {
        return -1;
    }

    dbus->message_iter_get_basic(&iter, &byte);

    return byte;
}

static DBusHandlerResult DBus_MessageFilter(DBusConnection *conn, DBusMessage *msg, void *data)
{
    SDL_DBusContext *dbus = (SDL_DBusContext *)data;

    if (dbus->message_is_signal(msg, FCITX_IC_DBUS_INTERFACE, "CommitString")) {
        DBusMessageIter iter;
        const char *text = NULL;

        dbus->message_iter_init(msg, &iter);
        dbus->message_iter_get_basic(&iter, &text);

        SDL_SendKeyboardText(text);

        return DBUS_HANDLER_RESULT_HANDLED;
    }

    if (dbus->message_is_signal(msg, FCITX_IC_DBUS_INTERFACE, "UpdateFormattedPreedit")) {
        char *text = NULL;
        Sint32 start_pos, end_pos;
        size_t text_bytes = Fcitx_GetPreeditString(dbus, msg, &text, &start_pos, &end_pos);
        if (text_bytes) {
            if (start_pos == -1) {
                Sint32 byte_pos = Fcitx_GetPreeditCursorByte(dbus, msg);
                start_pos = byte_pos >= 0 ? SDL_utf8strnlen(text, byte_pos) : -1;
            }
            SDL_SendEditingText(text, start_pos, end_pos >= 0 ? end_pos - start_pos : -1);
            SDL_free(text);
        } else {
            SDL_SendEditingText("", 0, 0);
        }

        SDL_Fcitx_UpdateTextInputArea(SDL_GetKeyboardFocus());
        return DBUS_HANDLER_RESULT_HANDLED;
    }

    return DBUS_HANDLER_RESULT_NOT_YET_HANDLED;
}

static void FcitxClientICCallMethod(FcitxClient *client, const char *method)
{
    if (!client->ic_path) {
        return;
    }
    SDL_DBus_CallVoidMethod(FCITX_DBUS_SERVICE, client->ic_path, FCITX_IC_DBUS_INTERFACE, method, DBUS_TYPE_INVALID);
}

static void SDLCALL Fcitx_SetCapabilities(void *data,
                                          const char *name,
                                          const char *old_val,
                                          const char *hint)
{
    FcitxClient *client = (FcitxClient *)data;
    Uint64 caps = 0;
    if (!client->ic_path) {
        return;
    }

    if (hint && SDL_strstr(hint, "composition")) {
        caps |= (1 << 1); // Preedit Flag
        caps |= (1 << 4); // Formatted Preedit Flag
    }
    if (hint && SDL_strstr(hint, "candidates")) {
        // FIXME, turn off native candidate rendering
    }

    SDL_DBus_CallVoidMethod(FCITX_DBUS_SERVICE, client->ic_path, FCITX_IC_DBUS_INTERFACE, "SetCapability", DBUS_TYPE_UINT64, &caps, DBUS_TYPE_INVALID);
}

static bool FcitxCreateInputContext(SDL_DBusContext *dbus, const char *appname, char **ic_path)
{
    const char *program = "program";
    bool result = false;

    if (dbus && dbus->session_conn) {
        DBusMessage *msg = dbus->message_new_method_call(FCITX_DBUS_SERVICE, FCITX_IM_DBUS_PATH, FCITX_IM_DBUS_INTERFACE, "CreateInputContext");
        if (msg) {
            DBusMessage *reply = NULL;
            DBusMessageIter args, array, sub;
            dbus->message_iter_init_append(msg, &args);
            dbus->message_iter_open_container(&args, DBUS_TYPE_ARRAY, "(ss)", &array);
            dbus->message_iter_open_container(&array, DBUS_TYPE_STRUCT, 0, &sub);
            dbus->message_iter_append_basic(&sub, DBUS_TYPE_STRING, &program);
            dbus->message_iter_append_basic(&sub, DBUS_TYPE_STRING, &appname);
            dbus->message_iter_close_container(&array, &sub);
            dbus->message_iter_close_container(&args, &array);
            reply = dbus->connection_send_with_reply_and_block(dbus->session_conn, msg, 300, NULL);
            if (reply) {
                if (dbus->message_get_args(reply, NULL, DBUS_TYPE_OBJECT_PATH, ic_path, DBUS_TYPE_INVALID)) {
                    result = true;
                }
                dbus->message_unref(reply);
            }
            dbus->message_unref(msg);
        }
    }
    return result;
}

static bool FcitxClientCreateIC(FcitxClient *client)
{
    char *appname = GetAppName();
    char *ic_path = NULL;
    SDL_DBusContext *dbus = client->dbus;

    // SDL_DBus_CallMethod cannot handle a(ss) type, call dbus function directly
    if (!FcitxCreateInputContext(dbus, appname, &ic_path)) {
        ic_path = NULL; // just in case.
    }

    SDL_free(appname);

    if (ic_path) {
        SDL_free(client->ic_path);
        client->ic_path = SDL_strdup(ic_path);

        dbus->bus_add_match(dbus->session_conn,
                            "type='signal', interface='org.fcitx.Fcitx.InputContext1'",
                            NULL);
        dbus->connection_add_filter(dbus->session_conn,
                                    &DBus_MessageFilter, dbus,
                                    NULL);
        dbus->connection_flush(dbus->session_conn);

        SDL_AddHintCallback(SDL_HINT_IME_IMPLEMENTED_UI, Fcitx_SetCapabilities, client);
        return true;
    }

    return false;
}

static Uint32 Fcitx_ModState(void)
{
    Uint32 fcitx_mods = 0;
    SDL_Keymod sdl_mods = SDL_GetModState();

    if (sdl_mods & SDL_KMOD_SHIFT) {
        fcitx_mods |= (1 << 0);
    }
    if (sdl_mods & SDL_KMOD_CAPS) {
        fcitx_mods |= (1 << 1);
    }
    if (sdl_mods & SDL_KMOD_CTRL) {
        fcitx_mods |= (1 << 2);
    }
    if (sdl_mods & SDL_KMOD_ALT) {
        fcitx_mods |= (1 << 3);
    }
    if (sdl_mods & SDL_KMOD_NUM) {
        fcitx_mods |= (1 << 4);
    }
    if (sdl_mods & SDL_KMOD_MODE) {
        fcitx_mods |= (1 << 7);
    }
    if (sdl_mods & SDL_KMOD_LGUI) {
        fcitx_mods |= (1 << 6);
    }
    if (sdl_mods & SDL_KMOD_RGUI) {
        fcitx_mods |= (1 << 28);
    }

    return fcitx_mods;
}

bool SDL_Fcitx_Init(void)
{
    fcitx_client.dbus = SDL_DBus_GetContext();

    fcitx_client.cursor_rect.x = -1;
    fcitx_client.cursor_rect.y = -1;
    fcitx_client.cursor_rect.w = 0;
    fcitx_client.cursor_rect.h = 0;

    return FcitxClientCreateIC(&fcitx_client);
}

void SDL_Fcitx_Quit(void)
{
    FcitxClientICCallMethod(&fcitx_client, "DestroyIC");
    if (fcitx_client.ic_path) {
        SDL_free(fcitx_client.ic_path);
        fcitx_client.ic_path = NULL;
    }
}

void SDL_Fcitx_SetFocus(bool focused)
{
    if (focused) {
        FcitxClientICCallMethod(&fcitx_client, "FocusIn");
    } else {
        FcitxClientICCallMethod(&fcitx_client, "FocusOut");
    }
}

void SDL_Fcitx_Reset(void)
{
    FcitxClientICCallMethod(&fcitx_client, "Reset");
}

bool SDL_Fcitx_ProcessKeyEvent(Uint32 keysym, Uint32 keycode, bool down)
{
    Uint32 mod_state = Fcitx_ModState();
    Uint32 handled = false;
    Uint32 is_release = !down;
    Uint32 event_time = 0;

    if (!fcitx_client.ic_path) {
        return false;
    }

    if (SDL_DBus_CallMethod(FCITX_DBUS_SERVICE, fcitx_client.ic_path, FCITX_IC_DBUS_INTERFACE, "ProcessKeyEvent",
                            DBUS_TYPE_UINT32, &keysym, DBUS_TYPE_UINT32, &keycode, DBUS_TYPE_UINT32, &mod_state, DBUS_TYPE_BOOLEAN, &is_release, DBUS_TYPE_UINT32, &event_time, DBUS_TYPE_INVALID,
                            DBUS_TYPE_BOOLEAN, &handled, DBUS_TYPE_INVALID)) {
        if (handled) {
            SDL_Fcitx_UpdateTextInputArea(SDL_GetKeyboardFocus());
            return true;
        }
    }

    return false;
}

void SDL_Fcitx_UpdateTextInputArea(SDL_Window *window)
{
    int x = 0, y = 0;
    SDL_Rect *cursor = &fcitx_client.cursor_rect;

    if (!window) {
        return;
    }

    // We'll use a square at the text input cursor location for the cursor_rect
    cursor->x = window->text_input_rect.x + window->text_input_cursor;
    cursor->y = window->text_input_rect.y;
    cursor->w = window->text_input_rect.h;
    cursor->h = window->text_input_rect.h;

    SDL_GetWindowPosition(window, &x, &y);

#ifdef SDL_VIDEO_DRIVER_X11
    {
        SDL_PropertiesID props = SDL_GetWindowProperties(window);
        Display *x_disp = (Display *)SDL_GetPointerProperty(props, SDL_PROP_WINDOW_X11_DISPLAY_POINTER, NULL);
        int x_screen = SDL_GetNumberProperty(props, SDL_PROP_WINDOW_X11_SCREEN_NUMBER, 0);
        Window x_win = SDL_GetNumberProperty(props, SDL_PROP_WINDOW_X11_WINDOW_NUMBER, 0);
        Window unused;
        if (x_disp && x_win) {
            X11_XTranslateCoordinates(x_disp, x_win, RootWindow(x_disp, x_screen), 0, 0, &x, &y, &unused);
        }
    }
#endif

    if (cursor->x == -1 && cursor->y == -1 && cursor->w == 0 && cursor->h == 0) {
        // move to bottom left
        int w = 0, h = 0;
        SDL_GetWindowSize(window, &w, &h);
        cursor->x = 0;
        cursor->y = h;
    }

    x += cursor->x;
    y += cursor->y;

    SDL_DBus_CallVoidMethod(FCITX_DBUS_SERVICE, fcitx_client.ic_path, FCITX_IC_DBUS_INTERFACE, "SetCursorRect",
                            DBUS_TYPE_INT32, &x, DBUS_TYPE_INT32, &y, DBUS_TYPE_INT32, &cursor->w, DBUS_TYPE_INT32, &cursor->h, DBUS_TYPE_INVALID);
}

void SDL_Fcitx_PumpEvents(void)
{
    SDL_DBusContext *dbus = fcitx_client.dbus;
    DBusConnection *conn = dbus->session_conn;

    dbus->connection_read_write(conn, 0);

    while (dbus->connection_dispatch(conn) == DBUS_DISPATCH_DATA_REMAINS) {
        // Do nothing, actual work happens in DBus_MessageFilter
    }
}
