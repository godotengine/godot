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

#ifndef SDL_POWER_DISABLED
#ifdef SDL_POWER_LINUX

#include <stdio.h>
#include <unistd.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <fcntl.h>

#include "../SDL_syspower.h"

#include "../../core/linux/SDL_dbus.h"

static const char *proc_apm_path = "/proc/apm";
static const char *proc_acpi_battery_path = "/proc/acpi/battery";
static const char *proc_acpi_ac_adapter_path = "/proc/acpi/ac_adapter";
static const char *sys_class_power_supply_path = "/sys/class/power_supply";

static int open_power_file(const char *base, const char *node, const char *key)
{
    int fd;
    const size_t pathlen = SDL_strlen(base) + SDL_strlen(node) + SDL_strlen(key) + 3;
    char *path = SDL_stack_alloc(char, pathlen);
    if (!path) {
        return -1; // oh well.
    }

    (void)SDL_snprintf(path, pathlen, "%s/%s/%s", base, node, key);
    fd = open(path, O_RDONLY | O_CLOEXEC);
    SDL_stack_free(path);
    return fd;
}

static bool read_power_file(const char *base, const char *node, const char *key,
                                char *buf, size_t buflen)
{
    ssize_t br = 0;
    const int fd = open_power_file(base, node, key);
    if (fd == -1) {
        return false;
    }
    br = read(fd, buf, buflen - 1);
    close(fd);
    if (br < 0) {
        return false;
    }
    buf[br] = '\0'; // null-terminate the string.
    return true;
}

static bool make_proc_acpi_key_val(char **_ptr, char **_key, char **_val)
{
    char *ptr = *_ptr;

    while (*ptr == ' ') {
        ptr++; // skip whitespace.
    }

    if (*ptr == '\0') {
        return false; // EOF.
    }

    *_key = ptr;

    while ((*ptr != ':') && (*ptr != '\0')) {
        ptr++;
    }

    if (*ptr == '\0') {
        return false; // (unexpected) EOF.
    }

    *(ptr++) = '\0'; // terminate the key.

    while (*ptr == ' ') {
        ptr++; // skip whitespace.
    }

    if (*ptr == '\0') {
        return false; // (unexpected) EOF.
    }

    *_val = ptr;

    while ((*ptr != '\n') && (*ptr != '\0')) {
        ptr++;
    }

    if (*ptr != '\0') {
        *(ptr++) = '\0'; // terminate the value.
    }

    *_ptr = ptr; // store for next time.
    return true;
}

static void check_proc_acpi_battery(const char *node, bool *have_battery,
                                    bool *charging, int *seconds, int *percent)
{
    const char *base = proc_acpi_battery_path;
    char info[1024];
    char state[1024];
    char *ptr = NULL;
    char *key = NULL;
    char *val = NULL;
    bool charge = false;
    bool choose = false;
    int maximum = -1;
    int remaining = -1;
    int secs = -1;
    int pct = -1;

    if (!read_power_file(base, node, "state", state, sizeof(state))) {
        return;
    } else if (!read_power_file(base, node, "info", info, sizeof(info))) {
        return;
    }

    ptr = &state[0];
    while (make_proc_acpi_key_val(&ptr, &key, &val)) {
        if (SDL_strcasecmp(key, "present") == 0) {
            if (SDL_strcasecmp(val, "yes") == 0) {
                *have_battery = true;
            }
        } else if (SDL_strcasecmp(key, "charging state") == 0) {
            // !!! FIXME: what exactly _does_ charging/discharging mean?
            if (SDL_strcasecmp(val, "charging/discharging") == 0) {
                charge = true;
            } else if (SDL_strcasecmp(val, "charging") == 0) {
                charge = true;
            }
        } else if (SDL_strcasecmp(key, "remaining capacity") == 0) {
            char *endptr = NULL;
            const int cvt = (int)SDL_strtol(val, &endptr, 10);
            if (*endptr == ' ') {
                remaining = cvt;
            }
        }
    }

    ptr = &info[0];
    while (make_proc_acpi_key_val(&ptr, &key, &val)) {
        if (SDL_strcasecmp(key, "design capacity") == 0) {
            char *endptr = NULL;
            const int cvt = (int)SDL_strtol(val, &endptr, 10);
            if (*endptr == ' ') {
                maximum = cvt;
            }
        }
    }

    if ((maximum >= 0) && (remaining >= 0)) {
        pct = (int)((((float)remaining) / ((float)maximum)) * 100.0f);
        if (pct < 0) {
            pct = 0;
        } else if (pct > 100) {
            pct = 100;
        }
    }

    // !!! FIXME: calculate (secs).

    /*
     * We pick the battery that claims to have the most minutes left.
     *  (failing a report of minutes, we'll take the highest percent.)
     */
    if ((secs < 0) && (*seconds < 0)) {
        if ((pct < 0) && (*percent < 0)) {
            choose = true; // at least we know there's a battery.
        }
        if (pct > *percent) {
            choose = true;
        }
    } else if (secs > *seconds) {
        choose = true;
    }

    if (choose) {
        *seconds = secs;
        *percent = pct;
        *charging = charge;
    }
}

static void check_proc_acpi_ac_adapter(const char *node, bool *have_ac)
{
    const char *base = proc_acpi_ac_adapter_path;
    char state[256];
    char *ptr = NULL;
    char *key = NULL;
    char *val = NULL;

    if (!read_power_file(base, node, "state", state, sizeof(state))) {
        return;
    }

    ptr = &state[0];
    while (make_proc_acpi_key_val(&ptr, &key, &val)) {
        if (SDL_strcasecmp(key, "state") == 0) {
            if (SDL_strcasecmp(val, "on-line") == 0) {
                *have_ac = true;
            }
        }
    }
}

bool SDL_GetPowerInfo_Linux_proc_acpi(SDL_PowerState *state, int *seconds, int *percent)
{
    struct dirent *dent = NULL;
    DIR *dirp = NULL;
    bool have_battery = false;
    bool have_ac = false;
    bool charging = false;

    *seconds = -1;
    *percent = -1;
    *state = SDL_POWERSTATE_UNKNOWN;

    dirp = opendir(proc_acpi_battery_path);
    if (!dirp) {
        return false; // can't use this interface.
    } else {
        while ((dent = readdir(dirp)) != NULL) {
            const char *node = dent->d_name;
            check_proc_acpi_battery(node, &have_battery, &charging,
                                    seconds, percent);
        }
        closedir(dirp);
    }

    dirp = opendir(proc_acpi_ac_adapter_path);
    if (!dirp) {
        return false; // can't use this interface.
    } else {
        while ((dent = readdir(dirp)) != NULL) {
            const char *node = dent->d_name;
            check_proc_acpi_ac_adapter(node, &have_ac);
        }
        closedir(dirp);
    }

    if (!have_battery) {
        *state = SDL_POWERSTATE_NO_BATTERY;
    } else if (charging) {
        *state = SDL_POWERSTATE_CHARGING;
    } else if (have_ac) {
        *state = SDL_POWERSTATE_CHARGED;
    } else {
        *state = SDL_POWERSTATE_ON_BATTERY;
    }

    return true; // definitive answer.
}

static bool next_string(char **_ptr, char **_str)
{
    char *ptr = *_ptr;
    char *str;

    while (*ptr == ' ') { // skip any spaces...
        ptr++;
    }

    if (*ptr == '\0') {
        return false;
    }

    str = ptr;
    while ((*ptr != ' ') && (*ptr != '\n') && (*ptr != '\0')) {
        ptr++;
    }

    if (*ptr != '\0') {
        *(ptr++) = '\0';
    }

    *_str = str;
    *_ptr = ptr;
    return true;
}

static bool int_string(char *str, int *val)
{
    char *endptr = NULL;
    *val = (int)SDL_strtol(str, &endptr, 0);
    return (*str != '\0') && (*endptr == '\0');
}

// http://lxr.linux.no/linux+v2.6.29/drivers/char/apm-emulation.c
bool SDL_GetPowerInfo_Linux_proc_apm(SDL_PowerState *state, int *seconds, int *percent)
{
    bool need_details = false;
    int ac_status = 0;
    int battery_status = 0;
    int battery_flag = 0;
    int battery_percent = 0;
    int battery_time = 0;
    const int fd = open(proc_apm_path, O_RDONLY | O_CLOEXEC);
    char buf[128];
    char *ptr = &buf[0];
    char *str = NULL;
    ssize_t br;

    if (fd == -1) {
        return false; // can't use this interface.
    }

    br = read(fd, buf, sizeof(buf) - 1);
    close(fd);

    if (br < 0) {
        return false;
    }

    buf[br] = '\0';                 // null-terminate the string.
    if (!next_string(&ptr, &str)) { // driver version
        return false;
    }
    if (!next_string(&ptr, &str)) { // BIOS version
        return false;
    }
    if (!next_string(&ptr, &str)) { // APM flags
        return false;
    }

    if (!next_string(&ptr, &str)) { // AC line status
        return false;
    } else if (!int_string(str, &ac_status)) {
        return false;
    }

    if (!next_string(&ptr, &str)) { // battery status
        return false;
    } else if (!int_string(str, &battery_status)) {
        return false;
    }
    if (!next_string(&ptr, &str)) { // battery flag
        return false;
    } else if (!int_string(str, &battery_flag)) {
        return false;
    }
    if (!next_string(&ptr, &str)) { // remaining battery life percent
        return false;
    }
    if (str[SDL_strlen(str) - 1] == '%') {
        str[SDL_strlen(str) - 1] = '\0';
    }
    if (!int_string(str, &battery_percent)) {
        return false;
    }

    if (!next_string(&ptr, &str)) { // remaining battery life time
        return false;
    } else if (!int_string(str, &battery_time)) {
        return false;
    }

    if (!next_string(&ptr, &str)) { // remaining battery life time units
        return false;
    } else if (SDL_strcasecmp(str, "min") == 0) {
        battery_time *= 60;
    }

    if (battery_flag == 0xFF) { // unknown state
        *state = SDL_POWERSTATE_UNKNOWN;
    } else if (battery_flag & (1 << 7)) { // no battery
        *state = SDL_POWERSTATE_NO_BATTERY;
    } else if (battery_flag & (1 << 3)) { // charging
        *state = SDL_POWERSTATE_CHARGING;
        need_details = true;
    } else if (ac_status == 1) {
        *state = SDL_POWERSTATE_CHARGED; // on AC, not charging.
        need_details = true;
    } else {
        *state = SDL_POWERSTATE_ON_BATTERY;
        need_details = true;
    }

    *percent = -1;
    *seconds = -1;
    if (need_details) {
        const int pct = battery_percent;
        const int secs = battery_time;

        if (pct >= 0) {                         // -1 == unknown
            *percent = (pct > 100) ? 100 : pct; // clamp between 0%, 100%
        }
        if (secs >= 0) { // -1 == unknown
            *seconds = secs;
        }
    }

    return true;
}

bool SDL_GetPowerInfo_Linux_sys_class_power_supply(SDL_PowerState *state, int *seconds, int *percent)
{
    const char *base = sys_class_power_supply_path;
    struct dirent *dent;
    DIR *dirp;

    dirp = opendir(base);
    if (!dirp) {
        return false;
    }

    *state = SDL_POWERSTATE_NO_BATTERY; // assume we're just plugged in.
    *seconds = -1;
    *percent = -1;

    while ((dent = readdir(dirp)) != NULL) {
        const char *name = dent->d_name;
        bool choose = false;
        char str[64];
        SDL_PowerState st;
        int secs;
        int pct;
        int energy;
        int power;

        if ((SDL_strcmp(name, ".") == 0) || (SDL_strcmp(name, "..") == 0)) {
            continue; // skip these, of course.
        } else if (!read_power_file(base, name, "type", str, sizeof(str))) {
            continue; // Don't know _what_ we're looking at. Give up on it.
        } else if (SDL_strcasecmp(str, "Battery\n") != 0) {
            continue; // we don't care about UPS and such.
        }

        /* if the scope is "device," it might be something like a PS4
           controller reporting its own battery, and not something that powers
           the system. Most system batteries don't list a scope at all; we
           assume it's a system battery if not specified. */
        if (read_power_file(base, name, "scope", str, sizeof(str))) {
            if (SDL_strcasecmp(str, "Device\n") == 0) {
                continue; // skip external devices with their own batteries.
            }
        }

        // some drivers don't offer this, so if it's not explicitly reported assume it's present.
        if (read_power_file(base, name, "present", str, sizeof(str)) && (SDL_strcmp(str, "0\n") == 0)) {
            st = SDL_POWERSTATE_NO_BATTERY;
        } else if (!read_power_file(base, name, "status", str, sizeof(str))) {
            st = SDL_POWERSTATE_UNKNOWN; // uh oh
        } else if (SDL_strcasecmp(str, "Charging\n") == 0) {
            st = SDL_POWERSTATE_CHARGING;
        } else if (SDL_strcasecmp(str, "Discharging\n") == 0) {
            st = SDL_POWERSTATE_ON_BATTERY;
        } else if ((SDL_strcasecmp(str, "Full\n") == 0) || (SDL_strcasecmp(str, "Not charging\n") == 0)) {
            st = SDL_POWERSTATE_CHARGED;
        } else {
            st = SDL_POWERSTATE_UNKNOWN; // uh oh
        }

        if (!read_power_file(base, name, "capacity", str, sizeof(str))) {
            pct = -1;
        } else {
            pct = SDL_atoi(str);
            pct = (pct > 100) ? 100 : pct; // clamp between 0%, 100%
        }

        if (read_power_file(base, name, "time_to_empty_now", str, sizeof(str))) {
            secs = SDL_atoi(str);
            secs = (secs <= 0) ? -1 : secs; // 0 == unknown
        } else if (st == SDL_POWERSTATE_ON_BATTERY) {
            /* energy is Watt*hours and power is Watts */
            energy = (read_power_file(base, name, "energy_now", str, sizeof(str))) ? SDL_atoi(str) : -1;
            power = (read_power_file(base, name, "power_now", str, sizeof(str))) ? SDL_atoi(str) : -1;
            secs = (energy >= 0 && power > 0) ? (3600LL * energy) / power : -1;
        } else {
            secs = -1;
        }

        /*
         * We pick the battery that claims to have the most minutes left.
         *  (failing a report of minutes, we'll take the highest percent.)
         */
        if ((secs < 0) && (*seconds < 0)) {
            if ((pct < 0) && (*percent < 0)) {
                choose = true; // at least we know there's a battery.
            } else if (pct > *percent) {
                choose = true;
            }
        } else if (secs > *seconds) {
            choose = true;
        }

        if (choose) {
            *seconds = secs;
            *percent = pct;
            *state = st;
        }
    }

    closedir(dirp);
    return true; // don't look any further.
}

// d-bus queries to org.freedesktop.UPower.
#ifdef SDL_USE_LIBDBUS
#define UPOWER_DBUS_NODE             "org.freedesktop.UPower"
#define UPOWER_DBUS_PATH             "/org/freedesktop/UPower"
#define UPOWER_DBUS_INTERFACE        "org.freedesktop.UPower"
#define UPOWER_DEVICE_DBUS_INTERFACE "org.freedesktop.UPower.Device"

static void check_upower_device(DBusConnection *conn, const char *path, SDL_PowerState *state, int *seconds, int *percent)
{
    bool choose = false;
    SDL_PowerState st;
    int secs;
    int pct;
    Uint32 ui32 = 0;
    Sint64 si64 = 0;
    double d = 0.0;

    if (!SDL_DBus_QueryPropertyOnConnection(conn, UPOWER_DBUS_NODE, path, UPOWER_DEVICE_DBUS_INTERFACE, "Type", DBUS_TYPE_UINT32, &ui32)) {
        return;             // Don't know _what_ we're looking at. Give up on it.
    } else if (ui32 != 2) { // 2==Battery
        return;             // we don't care about UPS and such.
    } else if (!SDL_DBus_QueryPropertyOnConnection(conn, UPOWER_DBUS_NODE, path, UPOWER_DEVICE_DBUS_INTERFACE, "PowerSupply", DBUS_TYPE_BOOLEAN, &ui32)) {
        return;
    } else if (!ui32) {
        return; // we don't care about random devices with batteries, like wireless controllers, etc
    }

    if (!SDL_DBus_QueryPropertyOnConnection(conn, UPOWER_DBUS_NODE, path, UPOWER_DEVICE_DBUS_INTERFACE, "IsPresent", DBUS_TYPE_BOOLEAN, &ui32)) {
        return;
    }
    if (!ui32) {
        st = SDL_POWERSTATE_NO_BATTERY;
    } else {
        /* Get updated information on the battery status
         * This can occasionally fail, and we'll just return slightly stale data in that case
         */
        SDL_DBus_CallMethodOnConnection(conn, UPOWER_DBUS_NODE, path, UPOWER_DEVICE_DBUS_INTERFACE, "Refresh", DBUS_TYPE_INVALID, DBUS_TYPE_INVALID);

        if (!SDL_DBus_QueryPropertyOnConnection(conn, UPOWER_DBUS_NODE, path, UPOWER_DEVICE_DBUS_INTERFACE, "State", DBUS_TYPE_UINT32, &ui32)) {
            st = SDL_POWERSTATE_UNKNOWN; // uh oh
        } else if (ui32 == 1) {          // 1 == charging
            st = SDL_POWERSTATE_CHARGING;
        } else if ((ui32 == 2) || (ui32 == 3) || (ui32 == 6)) {
            /* 2 == discharging;
             * 3 == empty;
             * 6 == "pending discharge" which GNOME interprets as equivalent
             * to discharging */
            st = SDL_POWERSTATE_ON_BATTERY;
        } else if ((ui32 == 4) || (ui32 == 5)) {
            /* 4 == full;
             * 5 == "pending charge" which GNOME shows as "Not charging",
             * used when a battery is configured to stop charging at a
             * lower than 100% threshold */
            st = SDL_POWERSTATE_CHARGED;
        } else {
            st = SDL_POWERSTATE_UNKNOWN; // uh oh
        }
    }

    if (!SDL_DBus_QueryPropertyOnConnection(conn, UPOWER_DBUS_NODE, path, UPOWER_DEVICE_DBUS_INTERFACE, "Percentage", DBUS_TYPE_DOUBLE, &d)) {
        pct = -1; // some old/cheap batteries don't set this property.
    } else {
        pct = (int)d;
        pct = (pct > 100) ? 100 : pct; // clamp between 0%, 100%
    }

    if (!SDL_DBus_QueryPropertyOnConnection(conn, UPOWER_DBUS_NODE, path, UPOWER_DEVICE_DBUS_INTERFACE, "TimeToEmpty", DBUS_TYPE_INT64, &si64)) {
        secs = -1;
    } else {
        secs = (int)si64;
        secs = (secs <= 0) ? -1 : secs; // 0 == unknown
    }

    /*
     * We pick the battery that claims to have the most minutes left.
     *  (failing a report of minutes, we'll take the highest percent.)
     */
    if ((secs < 0) && (*seconds < 0)) {
        if ((pct < 0) && (*percent < 0)) {
            choose = true; // at least we know there's a battery.
        } else if (pct > *percent) {
            choose = true;
        }
    } else if (secs > *seconds) {
        choose = true;
    }

    if (choose) {
        *seconds = secs;
        *percent = pct;
        *state = st;
    }
}
#endif

bool SDL_GetPowerInfo_Linux_org_freedesktop_upower(SDL_PowerState *state, int *seconds, int *percent)
{
    bool result = false;

#ifdef SDL_USE_LIBDBUS
    SDL_DBusContext *dbus = SDL_DBus_GetContext();
    char **paths = NULL;
    int i, numpaths = 0;

    if (!dbus || !SDL_DBus_CallMethodOnConnection(dbus->system_conn, UPOWER_DBUS_NODE, UPOWER_DBUS_PATH, UPOWER_DBUS_INTERFACE, "EnumerateDevices",
                                                         DBUS_TYPE_INVALID,
                                                         DBUS_TYPE_ARRAY, DBUS_TYPE_OBJECT_PATH, &paths, &numpaths, DBUS_TYPE_INVALID)) {
        return false; // try a different approach than UPower.
    }

    result = true;                  // Clearly we can use this interface.
    *state = SDL_POWERSTATE_NO_BATTERY; // assume we're just plugged in.
    *seconds = -1;
    *percent = -1;

    for (i = 0; i < numpaths; i++) {
        check_upower_device(dbus->system_conn, paths[i], state, seconds, percent);
    }

    dbus->free_string_array(paths);
#endif // SDL_USE_LIBDBUS

    return result;
}

#endif // SDL_POWER_LINUX
#endif // SDL_POWER_DISABLED
