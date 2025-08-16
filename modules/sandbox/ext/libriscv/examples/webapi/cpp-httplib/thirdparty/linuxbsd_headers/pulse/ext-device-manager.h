#ifndef foopulseextdevicemanagerhfoo
#define foopulseextdevicemanagerhfoo

/***
  This file is part of PulseAudio.

  Copyright 2008 Lennart Poettering
  Copyright 2009 Colin Guthrie

  PulseAudio is free software; you can redistribute it and/or modify
  it under the terms of the GNU Lesser General Public License as published
  by the Free Software Foundation; either version 2.1 of the License,
  or (at your option) any later version.

  PulseAudio is distributed in the hope that it will be useful, but
  WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
  General Public License for more details.

  You should have received a copy of the GNU Lesser General Public License
  along with PulseAudio; if not, see <http://www.gnu.org/licenses/>.
***/

#include <pulse/cdecl.h>
#include <pulse/context.h>
#include <pulse/version.h>

/** \file
 *
 * Routines for controlling module-device-manager
 */

PA_C_DECL_BEGIN

/* Don't extend this struct! It will break binary compatibility, because
 * pa_ext_device_manager_info.role_priorities points to an array of structs
 * instead of an array of pointers to structs. */
typedef struct pa_ext_device_manager_role_priority_info {
    const char *role;
    uint32_t priority;
} pa_ext_device_manager_role_priority_info;

/** Stores information about one device in the device database that is
 * maintained by module-device-manager. \since 0.9.21 */
typedef struct pa_ext_device_manager_info {
    const char *name;            /**< Identifier string of the device. A string like "sink:" or similar followed by the name of the device. */
    const char *description;     /**< The description of the device when it was last seen, if applicable and saved */
    const char *icon;            /**< The icon given to the device */
    uint32_t index;              /**< The device index if it is currently available or PA_INVALID_INDEX */
    uint32_t n_role_priorities;  /**< How many role priorities do we have? */
    pa_ext_device_manager_role_priority_info *role_priorities; /**< An array of role priority structures or NULL */
} pa_ext_device_manager_info;

/** Callback prototype for pa_ext_device_manager_test(). \since 0.9.21 */
typedef void (*pa_ext_device_manager_test_cb_t)(
        pa_context *c,
        uint32_t version,
        void *userdata);

/** Test if this extension module is available in the server. \since 0.9.21 */
pa_operation *pa_ext_device_manager_test(
        pa_context *c,
        pa_ext_device_manager_test_cb_t cb,
        void *userdata);

/** Callback prototype for pa_ext_device_manager_read(). \since 0.9.21 */
typedef void (*pa_ext_device_manager_read_cb_t)(
        pa_context *c,
        const pa_ext_device_manager_info *info,
        int eol,
        void *userdata);

/** Read all entries from the device database. \since 0.9.21 */
pa_operation *pa_ext_device_manager_read(
        pa_context *c,
        pa_ext_device_manager_read_cb_t cb,
        void *userdata);

/** Sets the description for a device. \since 0.9.21 */
pa_operation *pa_ext_device_manager_set_device_description(
        pa_context *c,
        const char* device,
        const char* description,
        pa_context_success_cb_t cb,
        void *userdata);

/** Delete entries from the device database. \since 0.9.21 */
pa_operation *pa_ext_device_manager_delete(
        pa_context *c,
        const char *const s[],
        pa_context_success_cb_t cb,
        void *userdata);

/** Enable the role-based device-priority routing mode. \since 0.9.21 */
pa_operation *pa_ext_device_manager_enable_role_device_priority_routing(
        pa_context *c,
        int enable,
        pa_context_success_cb_t cb,
        void *userdata);

/** Prefer a given device in the priority list. \since 0.9.21 */
pa_operation *pa_ext_device_manager_reorder_devices_for_role(
        pa_context *c,
        const char* role,
        const char** devices,
        pa_context_success_cb_t cb,
        void *userdata);

/** Subscribe to changes in the device database. \since 0.9.21 */
pa_operation *pa_ext_device_manager_subscribe(
        pa_context *c,
        int enable,
        pa_context_success_cb_t cb,
        void *userdata);

/** Callback prototype for pa_ext_device_manager_set_subscribe_cb(). \since 0.9.21 */
typedef void (*pa_ext_device_manager_subscribe_cb_t)(
        pa_context *c,
        void *userdata);

/** Set the subscription callback that is called when
 * pa_ext_device_manager_subscribe() was called. \since 0.9.21 */
void pa_ext_device_manager_set_subscribe_cb(
        pa_context *c,
        pa_ext_device_manager_subscribe_cb_t cb,
        void *userdata);

PA_C_DECL_END

#endif
