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

#include "SDL_dbus.h"
#include "SDL_system_theme.h"
#include "../../video/SDL_sysvideo.h"

#include <unistd.h>

#define PORTAL_DESTINATION "org.freedesktop.portal.Desktop"
#define PORTAL_PATH "/org/freedesktop/portal/desktop"
#define PORTAL_INTERFACE "org.freedesktop.portal.Settings"
#define PORTAL_METHOD "Read"

#define SIGNAL_INTERFACE "org.freedesktop.portal.Settings"
#define SIGNAL_NAMESPACE "org.freedesktop.appearance"
#define SIGNAL_NAME "SettingChanged"
#define SIGNAL_KEY "color-scheme"

typedef struct SystemThemeData
{
    SDL_DBusContext *dbus;
    SDL_SystemTheme theme;
} SystemThemeData;

static SystemThemeData system_theme_data;

static bool DBus_ExtractThemeVariant(DBusMessageIter *iter, SDL_SystemTheme *theme) {
    SDL_DBusContext *dbus = system_theme_data.dbus;
    Uint32 color_scheme;
    DBusMessageIter variant_iter;

    if (dbus->message_iter_get_arg_type(iter) != DBUS_TYPE_VARIANT)
        return false;
    dbus->message_iter_recurse(iter, &variant_iter);
    if (dbus->message_iter_get_arg_type(&variant_iter) != DBUS_TYPE_UINT32)
        return false;
    dbus->message_iter_get_basic(&variant_iter, &color_scheme);
    switch (color_scheme) {
        case 0:
            *theme = SDL_SYSTEM_THEME_UNKNOWN;
            break;
        case 1:
            *theme = SDL_SYSTEM_THEME_DARK;
            break;
        case 2:
            *theme = SDL_SYSTEM_THEME_LIGHT;
            break;
    }
    return true;
}

static DBusHandlerResult DBus_MessageFilter(DBusConnection *conn, DBusMessage *msg, void *data) {
    SDL_DBusContext *dbus = (SDL_DBusContext *)data;

    if (dbus->message_is_signal(msg, SIGNAL_INTERFACE, SIGNAL_NAME)) {
        DBusMessageIter signal_iter;
        const char *namespace, *key;

        dbus->message_iter_init(msg, &signal_iter);
        // Check if the parameters are what we expect
        if (dbus->message_iter_get_arg_type(&signal_iter) != DBUS_TYPE_STRING)
            goto not_our_signal;
        dbus->message_iter_get_basic(&signal_iter, &namespace);
        if (SDL_strcmp(SIGNAL_NAMESPACE, namespace) != 0)
            goto not_our_signal;

        if (!dbus->message_iter_next(&signal_iter))
            goto not_our_signal;

        if (dbus->message_iter_get_arg_type(&signal_iter) != DBUS_TYPE_STRING)
            goto not_our_signal;
        dbus->message_iter_get_basic(&signal_iter, &key);
        if (SDL_strcmp(SIGNAL_KEY, key) != 0)
            goto not_our_signal;

        if (!dbus->message_iter_next(&signal_iter))
            goto not_our_signal;

        if (!DBus_ExtractThemeVariant(&signal_iter, &system_theme_data.theme))
            goto not_our_signal;

        SDL_SetSystemTheme(system_theme_data.theme);
        return DBUS_HANDLER_RESULT_HANDLED;
    }
not_our_signal:
    return DBUS_HANDLER_RESULT_NOT_YET_HANDLED;
}

bool SDL_SystemTheme_Init(void)
{
    SDL_DBusContext *dbus = SDL_DBus_GetContext();
    DBusMessage *msg;
    static const char *namespace = SIGNAL_NAMESPACE;
    static const char *key = SIGNAL_KEY;

    system_theme_data.theme = SDL_SYSTEM_THEME_UNKNOWN;
    system_theme_data.dbus = dbus;
    if (!dbus) {
        return false;
    }

    msg = dbus->message_new_method_call(PORTAL_DESTINATION, PORTAL_PATH, PORTAL_INTERFACE, PORTAL_METHOD);
    if (msg) {
        if (dbus->message_append_args(msg, DBUS_TYPE_STRING, &namespace, DBUS_TYPE_STRING, &key, DBUS_TYPE_INVALID)) {
            DBusMessage *reply = dbus->connection_send_with_reply_and_block(dbus->session_conn, msg, 300, NULL);
            if (reply) {
                DBusMessageIter reply_iter, variant_outer_iter;

                dbus->message_iter_init(reply, &reply_iter);
                // The response has signature <<u>>
                if (dbus->message_iter_get_arg_type(&reply_iter) != DBUS_TYPE_VARIANT)
                    goto incorrect_type;
                dbus->message_iter_recurse(&reply_iter, &variant_outer_iter);
                if (!DBus_ExtractThemeVariant(&variant_outer_iter, &system_theme_data.theme))
                    goto incorrect_type;
incorrect_type:
                dbus->message_unref(reply);
            }
        }
        dbus->message_unref(msg);
    }

    dbus->bus_add_match(dbus->session_conn,
                        "type='signal', interface='"SIGNAL_INTERFACE"',"
                        "member='"SIGNAL_NAME"', arg0='"SIGNAL_NAMESPACE"',"
                        "arg1='"SIGNAL_KEY"'", NULL);
    dbus->connection_add_filter(dbus->session_conn,
                                &DBus_MessageFilter, dbus, NULL);
    dbus->connection_flush(dbus->session_conn);
    return true;
}

SDL_SystemTheme SDL_SystemTheme_Get(void)
{
    return system_theme_data.theme;
}
