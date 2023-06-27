#ifndef foopulseproplisthfoo
#define foopulseproplisthfoo

/***
  This file is part of PulseAudio.

  Copyright 2007 Lennart Poettering

  PulseAudio is free software; you can redistribute it and/or modify
  it under the terms of the GNU Lesser General Public License as
  published by the Free Software Foundation; either version 2.1 of the
  License, or (at your option) any later version.

  PulseAudio is distributed in the hope that it will be useful, but
  WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with PulseAudio; if not, see <http://www.gnu.org/licenses/>.
***/

#include <sys/types.h>

#include <pulse/cdecl.h>
#include <pulse/gccmacro.h>
#include <pulse/version.h>

/** \file
 * Property list constants and functions */

PA_C_DECL_BEGIN

/** For streams: localized media name, formatted as UTF-8. E.g. "Guns'N'Roses: Civil War".*/
#define PA_PROP_MEDIA_NAME                     "media.name"

/** For streams: localized media title if applicable, formatted as UTF-8. E.g. "Civil War" */
#define PA_PROP_MEDIA_TITLE                    "media.title"

/** For streams: localized media artist if applicable, formatted as UTF-8. E.g. "Guns'N'Roses" */
#define PA_PROP_MEDIA_ARTIST                   "media.artist"

/** For streams: localized media copyright string if applicable, formatted as UTF-8. E.g. "Evil Record Corp." */
#define PA_PROP_MEDIA_COPYRIGHT                "media.copyright"

/** For streams: localized media generator software string if applicable, formatted as UTF-8. E.g. "Foocrop AudioFrobnicator" */
#define PA_PROP_MEDIA_SOFTWARE                 "media.software"

/** For streams: media language if applicable, in standard POSIX format. E.g. "de_DE" */
#define PA_PROP_MEDIA_LANGUAGE                 "media.language"

/** For streams: source filename if applicable, in URI format or local path. E.g. "/home/lennart/music/foobar.ogg" */
#define PA_PROP_MEDIA_FILENAME                 "media.filename"

/** \cond fulldocs */
/** For streams: icon for the media. A binary blob containing PNG image data */
#define PA_PROP_MEDIA_ICON                     "media.icon"
/** \endcond */

/** For streams: an XDG icon name for the media. E.g. "audio-x-mp3" */
#define PA_PROP_MEDIA_ICON_NAME                "media.icon_name"

/** For streams: logic role of this media. One of the strings "video", "music", "game", "event", "phone", "animation", "production", "a11y", "test" */
#define PA_PROP_MEDIA_ROLE                     "media.role"

/** For streams: the name of a filter that is desired, e.g.\ "echo-cancel" or "equalizer-sink". PulseAudio may choose to not apply the filter if it does not make sense (for example, applying echo-cancellation on a Bluetooth headset probably does not make sense. \since 1.0 */
#define PA_PROP_FILTER_WANT                    "filter.want"

/** For streams: the name of a filter that is desired, e.g.\ "echo-cancel" or "equalizer-sink". Differs from PA_PROP_FILTER_WANT in that it forces PulseAudio to apply the filter, regardless of whether PulseAudio thinks it makes sense to do so or not. If this is set, PA_PROP_FILTER_WANT is ignored. In other words, you almost certainly do not want to use this. \since 1.0 */
#define PA_PROP_FILTER_APPLY                   "filter.apply"

/** For streams: the name of a filter that should specifically suppressed (i.e.\ overrides PA_PROP_FILTER_WANT). Useful for the times that PA_PROP_FILTER_WANT is automatically added (e.g. echo-cancellation for phone streams when $VOIP_APP does its own, internal AEC) \since 1.0 */
#define PA_PROP_FILTER_SUPPRESS                "filter.suppress"

/** For event sound streams: XDG event sound name. e.g.\ "message-new-email" (Event sound streams are those with media.role set to "event") */
#define PA_PROP_EVENT_ID                       "event.id"

/** For event sound streams: localized human readable one-line description of the event, formatted as UTF-8. E.g. "Email from lennart@example.com received." */
#define PA_PROP_EVENT_DESCRIPTION              "event.description"

/** For event sound streams: absolute horizontal mouse position on the screen if the event sound was triggered by a mouse click, integer formatted as text string. E.g. "865" */
#define PA_PROP_EVENT_MOUSE_X                  "event.mouse.x"

/** For event sound streams: absolute vertical mouse position on the screen if the event sound was triggered by a mouse click, integer formatted as text string. E.g. "432" */
#define PA_PROP_EVENT_MOUSE_Y                  "event.mouse.y"

/** For event sound streams: relative horizontal mouse position on the screen if the event sound was triggered by a mouse click, float formatted as text string, ranging from 0.0 (left side of the screen) to 1.0 (right side of the screen). E.g. "0.65" */
#define PA_PROP_EVENT_MOUSE_HPOS               "event.mouse.hpos"

/** For event sound streams: relative vertical mouse position on the screen if the event sound was triggered by a mouse click, float formatted as text string, ranging from 0.0 (top of the screen) to 1.0 (bottom of the screen). E.g. "0.43" */
#define PA_PROP_EVENT_MOUSE_VPOS               "event.mouse.vpos"

/** For event sound streams: mouse button that triggered the event if applicable, integer formatted as string with 0=left, 1=middle, 2=right. E.g. "0" */
#define PA_PROP_EVENT_MOUSE_BUTTON             "event.mouse.button"

/** For streams that belong to a window on the screen: localized window title. E.g. "Totem Music Player" */
#define PA_PROP_WINDOW_NAME                    "window.name"

/** For streams that belong to a window on the screen: a textual id for identifying a window logically. E.g. "org.gnome.Totem.MainWindow" */
#define PA_PROP_WINDOW_ID                      "window.id"

/** \cond fulldocs */
/** For streams that belong to a window on the screen: window icon. A binary blob containing PNG image data */
#define PA_PROP_WINDOW_ICON                    "window.icon"
/** \endcond */

/** For streams that belong to a window on the screen: an XDG icon name for the window. E.g. "totem" */
#define PA_PROP_WINDOW_ICON_NAME               "window.icon_name"

/** For streams that belong to a window on the screen: absolute horizontal window position on the screen, integer formatted as text string. E.g. "865". \since 0.9.17 */
#define PA_PROP_WINDOW_X                       "window.x"

/** For streams that belong to a window on the screen: absolute vertical window position on the screen, integer formatted as text string. E.g. "343". \since 0.9.17 */
#define PA_PROP_WINDOW_Y                       "window.y"

/** For streams that belong to a window on the screen: window width on the screen, integer formatted as text string. e.g. "365". \since 0.9.17 */
#define PA_PROP_WINDOW_WIDTH                   "window.width"

/** For streams that belong to a window on the screen: window height on the screen, integer formatted as text string. E.g. "643". \since 0.9.17 */
#define PA_PROP_WINDOW_HEIGHT                  "window.height"

/** For streams that belong to a window on the screen: relative position of the window center on the screen, float formatted as text string, ranging from 0.0 (left side of the screen) to 1.0 (right side of the screen). E.g. "0.65". \since 0.9.17 */
#define PA_PROP_WINDOW_HPOS                    "window.hpos"

/** For streams that belong to a window on the screen: relative position of the window center on the screen, float formatted as text string, ranging from 0.0 (top of the screen) to 1.0 (bottom of the screen). E.g. "0.43". \since 0.9.17 */
#define PA_PROP_WINDOW_VPOS                    "window.vpos"

/** For streams that belong to a window on the screen: if the windowing system supports multiple desktops, a comma separated list of indexes of the desktops this window is visible on. If this property is an empty string, it is visible on all desktops (i.e. 'sticky'). The first desktop is 0. E.g. "0,2,3" \since 0.9.18 */
#define PA_PROP_WINDOW_DESKTOP                 "window.desktop"

/** For streams that belong to an X11 window on the screen: the X11 display string. E.g. ":0.0" */
#define PA_PROP_WINDOW_X11_DISPLAY             "window.x11.display"

/** For streams that belong to an X11 window on the screen: the X11 screen the window is on, an integer formatted as string. E.g. "0" */
#define PA_PROP_WINDOW_X11_SCREEN              "window.x11.screen"

/** For streams that belong to an X11 window on the screen: the X11 monitor the window is on, an integer formatted as string. E.g. "0" */
#define PA_PROP_WINDOW_X11_MONITOR             "window.x11.monitor"

/** For streams that belong to an X11 window on the screen: the window XID, an integer formatted as string. E.g. "25632" */
#define PA_PROP_WINDOW_X11_XID                 "window.x11.xid"

/** For clients/streams: localized human readable application name. E.g. "Totem Music Player" */
#define PA_PROP_APPLICATION_NAME               "application.name"

/** For clients/streams: a textual id for identifying an application logically. E.g. "org.gnome.Totem" */
#define PA_PROP_APPLICATION_ID                 "application.id"

/** For clients/streams: a version string, e.g.\ "0.6.88" */
#define PA_PROP_APPLICATION_VERSION            "application.version"

/** \cond fulldocs */
/** For clients/streams: application icon. A binary blob containing PNG image data */
#define PA_PROP_APPLICATION_ICON               "application.icon"
/** \endcond */

/** For clients/streams: an XDG icon name for the application. E.g. "totem" */
#define PA_PROP_APPLICATION_ICON_NAME          "application.icon_name"

/** For clients/streams: application language if applicable, in standard POSIX format. E.g. "de_DE" */
#define PA_PROP_APPLICATION_LANGUAGE           "application.language"

/** For clients/streams on UNIX: application process PID, an integer formatted as string. E.g. "4711" */
#define PA_PROP_APPLICATION_PROCESS_ID         "application.process.id"

/** For clients/streams: application process name. E.g. "totem" */
#define PA_PROP_APPLICATION_PROCESS_BINARY     "application.process.binary"

/** For clients/streams: application user name. E.g. "lennart" */
#define PA_PROP_APPLICATION_PROCESS_USER       "application.process.user"

/** For clients/streams: host name the application runs on. E.g. "omega" */
#define PA_PROP_APPLICATION_PROCESS_HOST       "application.process.host"

/** For clients/streams: the D-Bus host id the application runs on. E.g. "543679e7b01393ed3e3e650047d78f6e" */
#define PA_PROP_APPLICATION_PROCESS_MACHINE_ID "application.process.machine_id"

/** For clients/streams: an id for the login session the application runs in. On Unix the value of $XDG_SESSION_ID. E.g. "5" */
#define PA_PROP_APPLICATION_PROCESS_SESSION_ID "application.process.session_id"

/** For devices: device string in the underlying audio layer's format. E.g. "surround51:0" */
#define PA_PROP_DEVICE_STRING                  "device.string"

/** For devices: API this device is access with. E.g. "alsa" */
#define PA_PROP_DEVICE_API                     "device.api"

/** For devices: localized human readable device one-line description. E.g. "Foobar Industries USB Headset 2000+ Ultra" */
#define PA_PROP_DEVICE_DESCRIPTION             "device.description"

/** For devices: bus path to the device in the OS' format. E.g. "/sys/bus/pci/devices/0000:00:1f.2" */
#define PA_PROP_DEVICE_BUS_PATH                "device.bus_path"

/** For devices: serial number if applicable. E.g. "4711-0815-1234" */
#define PA_PROP_DEVICE_SERIAL                  "device.serial"

/** For devices: vendor ID if applicable. E.g. 1274 */
#define PA_PROP_DEVICE_VENDOR_ID               "device.vendor.id"

/** For devices: vendor name if applicable. E.g. "Foocorp Heavy Industries" */
#define PA_PROP_DEVICE_VENDOR_NAME             "device.vendor.name"

/** For devices: product ID if applicable. E.g. 4565 */
#define PA_PROP_DEVICE_PRODUCT_ID              "device.product.id"

/** For devices: product name if applicable. E.g. "SuperSpeakers 2000 Pro" */
#define PA_PROP_DEVICE_PRODUCT_NAME            "device.product.name"

/** For devices: device class. One of "sound", "modem", "monitor", "filter" */
#define PA_PROP_DEVICE_CLASS                   "device.class"

/** For devices: form factor if applicable. One of "internal", "speaker", "handset", "tv", "webcam", "microphone", "headset", "headphone", "hands-free", "car", "hifi", "computer", "portable" */
#define PA_PROP_DEVICE_FORM_FACTOR             "device.form_factor"

/** For devices: bus of the device if applicable. One of "isa", "pci", "usb", "firewire", "bluetooth" */
#define PA_PROP_DEVICE_BUS                     "device.bus"

/** \cond fulldocs */
/** For devices: icon for the device. A binary blob containing PNG image data */
#define PA_PROP_DEVICE_ICON                    "device.icon"
/** \endcond */

/** For devices: an XDG icon name for the device. E.g. "sound-card-speakers-usb" */
#define PA_PROP_DEVICE_ICON_NAME               "device.icon_name"

/** For devices: access mode of the device if applicable. One of "mmap", "mmap_rewrite", "serial" */
#define PA_PROP_DEVICE_ACCESS_MODE             "device.access_mode"

/** For filter devices: master device id if applicable. */
#define PA_PROP_DEVICE_MASTER_DEVICE           "device.master_device"

/** For devices: buffer size in bytes, integer formatted as string. */
#define PA_PROP_DEVICE_BUFFERING_BUFFER_SIZE   "device.buffering.buffer_size"

/** For devices: fragment size in bytes, integer formatted as string. */
#define PA_PROP_DEVICE_BUFFERING_FRAGMENT_SIZE "device.buffering.fragment_size"

/** For devices: profile identifier for the profile this devices is in. E.g. "analog-stereo", "analog-surround-40", "iec958-stereo", ...*/
#define PA_PROP_DEVICE_PROFILE_NAME            "device.profile.name"

/** For devices: intended use. A space separated list of roles (see PA_PROP_MEDIA_ROLE) this device is particularly well suited for, due to latency, quality or form factor. \since 0.9.16 */
#define PA_PROP_DEVICE_INTENDED_ROLES          "device.intended_roles"

/** For devices: human readable one-line description of the profile this device is in. E.g. "Analog Stereo", ... */
#define PA_PROP_DEVICE_PROFILE_DESCRIPTION     "device.profile.description"

/** For modules: the author's name, formatted as UTF-8 string. E.g. "Lennart Poettering" */
#define PA_PROP_MODULE_AUTHOR                  "module.author"

/** For modules: a human readable one-line description of the module's purpose formatted as UTF-8. E.g. "Frobnicate sounds with a flux compensator" */
#define PA_PROP_MODULE_DESCRIPTION             "module.description"

/** For modules: a human readable usage description of the module's arguments formatted as UTF-8. */
#define PA_PROP_MODULE_USAGE                   "module.usage"

/** For modules: a version string for the module. E.g. "0.9.15" */
#define PA_PROP_MODULE_VERSION                 "module.version"

/** For PCM formats: the sample format used as returned by pa_sample_format_to_string() \since 1.0 */
#define PA_PROP_FORMAT_SAMPLE_FORMAT           "format.sample_format"

/** For all formats: the sample rate (unsigned integer) \since 1.0 */
#define PA_PROP_FORMAT_RATE                    "format.rate"

/** For all formats: the number of channels (unsigned integer) \since 1.0 */
#define PA_PROP_FORMAT_CHANNELS                "format.channels"

/** For PCM formats: the channel map of the stream as returned by pa_channel_map_snprint() \since 1.0 */
#define PA_PROP_FORMAT_CHANNEL_MAP             "format.channel_map"

/** A property list object. Basically a dictionary with ASCII strings
 * as keys and arbitrary data as values. \since 0.9.11 */
typedef struct pa_proplist pa_proplist;

/** Allocate a property list. \since 0.9.11 */
pa_proplist* pa_proplist_new(void);

/** Free the property list. \since 0.9.11 */
void pa_proplist_free(pa_proplist* p);

/** Returns a non-zero value if the key is valid. \since 3.0 */
int pa_proplist_key_valid(const char *key);

/** Append a new string entry to the property list, possibly
 * overwriting an already existing entry with the same key. An
 * internal copy of the data passed is made. Will accept only valid
 * UTF-8. \since 0.9.11 */
int pa_proplist_sets(pa_proplist *p, const char *key, const char *value);

/** Append a new string entry to the property list, possibly
 * overwriting an already existing entry with the same key. An
 * internal copy of the data passed is made. Will accept only valid
 * UTF-8. The string passed in must contain a '='. Left hand side of
 * the '=' is used as key name, the right hand side as string
 * data. \since 0.9.16 */
int pa_proplist_setp(pa_proplist *p, const char *pair);

/** Append a new string entry to the property list, possibly
 * overwriting an already existing entry with the same key. An
 * internal copy of the data passed is made. Will accept only valid
 * UTF-8. The data can be passed as printf()-style format string with
 * arguments. \since 0.9.11 */
int pa_proplist_setf(pa_proplist *p, const char *key, const char *format, ...) PA_GCC_PRINTF_ATTR(3,4);

/** Append a new arbitrary data entry to the property list, possibly
 * overwriting an already existing entry with the same key. An
 * internal copy of the data passed is made. \since 0.9.11 */
int pa_proplist_set(pa_proplist *p, const char *key, const void *data, size_t nbytes);

/** Return a string entry for the specified key. Will return NULL if
 * the data is not valid UTF-8. Will return a NUL-terminated string in
 * an internally allocated buffer. The caller should make a copy of
 * the data before accessing the property list again. \since 0.9.11 */
const char *pa_proplist_gets(pa_proplist *p, const char *key);

/** Store the value for the specified key in \a data. Will store a
 * NUL-terminated string for string entries. The \a data pointer returned will
 * point to an internally allocated buffer. The caller should make a
 * copy of the data before the property list is accessed again. \since
 * 0.9.11 */
int pa_proplist_get(pa_proplist *p, const char *key, const void **data, size_t *nbytes);

/** Update mode enum for pa_proplist_update(). \since 0.9.11 */
typedef enum pa_update_mode {
    PA_UPDATE_SET
    /**< Replace the entire property list with the new one. Don't keep
     *  any of the old data around. */,

    PA_UPDATE_MERGE
    /**< Merge new property list into the existing one, not replacing
     *  any old entries if they share a common key with the new
     *  property list. */,

    PA_UPDATE_REPLACE
    /**< Merge new property list into the existing one, replacing all
     *  old entries that share a common key with the new property
     *  list. */
} pa_update_mode_t;

/** \cond fulldocs */
#define PA_UPDATE_SET PA_UPDATE_SET
#define PA_UPDATE_MERGE PA_UPDATE_MERGE
#define PA_UPDATE_REPLACE PA_UPDATE_REPLACE
/** \endcond */

/** Merge property list "other" into "p", adhering the merge mode as
 * specified in "mode". \since 0.9.11 */
void pa_proplist_update(pa_proplist *p, pa_update_mode_t mode, const pa_proplist *other);

/** Removes a single entry from the property list, identified be the
 * specified key name. \since 0.9.11 */
int pa_proplist_unset(pa_proplist *p, const char *key);

/** Similar to pa_proplist_unset() but takes an array of keys to
 * remove. The array should be terminated by a NULL pointer. Returns -1
 * on failure, otherwise the number of entries actually removed (which
 * might even be 0, if there were no matching entries to
 * remove). \since 0.9.11 */
int pa_proplist_unset_many(pa_proplist *p, const char * const keys[]);

/** Iterate through the property list. The user should allocate a
 * state variable of type void* and initialize it with NULL. A pointer
 * to this variable should then be passed to pa_proplist_iterate()
 * which should be called in a loop until it returns NULL which
 * signifies EOL. The property list should not be modified during
 * iteration through the list -- with the exception of deleting the
 * current entry. On each invocation this function will return the
 * key string for the next entry. The keys in the property list do not
 * have any particular order. \since 0.9.11 */
const char *pa_proplist_iterate(pa_proplist *p, void **state);

/** Format the property list nicely as a human readable string. This
 * works very much like pa_proplist_to_string_sep() and uses a newline
 * as separator and appends one final one. Call pa_xfree() on the
 * result. \since 0.9.11 */
char *pa_proplist_to_string(pa_proplist *p);

/** Format the property list nicely as a human readable string and
 * choose the separator. Call pa_xfree() on the result. \since
 * 0.9.15 */
char *pa_proplist_to_string_sep(pa_proplist *p, const char *sep);

/** Allocate a new property list and assign key/value from a human
 * readable string. \since 0.9.15 */
pa_proplist *pa_proplist_from_string(const char *str);

/** Returns 1 if an entry for the specified key exists in the
 * property list. \since 0.9.11 */
int pa_proplist_contains(pa_proplist *p, const char *key);

/** Remove all entries from the property list object. \since 0.9.11 */
void pa_proplist_clear(pa_proplist *p);

/** Allocate a new property list and copy over every single entry from
 * the specified list. \since 0.9.11 */
pa_proplist* pa_proplist_copy(const pa_proplist *p);

/** Return the number of entries in the property list. \since 0.9.15 */
unsigned pa_proplist_size(pa_proplist *p);

/** Returns 0 when the proplist is empty, positive otherwise \since 0.9.15 */
int pa_proplist_isempty(pa_proplist *p);

/** Return non-zero when a and b have the same keys and values.
 * \since 0.9.16 */
int pa_proplist_equal(pa_proplist *a, pa_proplist *b);

PA_C_DECL_END

#endif
