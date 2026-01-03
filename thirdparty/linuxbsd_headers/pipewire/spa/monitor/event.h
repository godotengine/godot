/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2020 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_EVENT_DEVICE_H
#define SPA_EVENT_DEVICE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <spa/pod/event.h>

/**
 * \addtogroup spa_device
 * \{
 */

/* object id of SPA_TYPE_EVENT_Device */
enum spa_device_event {
	SPA_DEVICE_EVENT_ObjectConfig,
};

#define SPA_DEVICE_EVENT_ID(ev)	SPA_EVENT_ID(ev, SPA_TYPE_EVENT_Device)
#define SPA_DEVICE_EVENT_INIT(id) SPA_EVENT_INIT(SPA_TYPE_EVENT_Device, id)

/* properties for SPA_TYPE_EVENT_Device */
enum spa_event_device {
	SPA_EVENT_DEVICE_START,

	SPA_EVENT_DEVICE_Object,	/* an object id (Int) */
	SPA_EVENT_DEVICE_Props,		/* properties for an object (SPA_TYPE_OBJECT_Props) */
};

/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SPA_EVENT_DEVICE */
