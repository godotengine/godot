/* PipeWire */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef PIPEWIRE_CONTROL_H
#define PIPEWIRE_CONTROL_H

#ifdef __cplusplus
extern "C" {
#endif

#include <spa/utils/hook.h>

/** \defgroup pw_control Control
 *
 * \brief A control can be used to control a port property.
 */

/**
 * \addtogroup pw_control
 * \{
 */
struct pw_control;

#include <pipewire/impl.h>

/** Port events, use \ref pw_control_add_listener */
struct pw_control_events {
#define PW_VERSION_CONTROL_EVENTS 0
	uint32_t version;

	/** The control is destroyed */
	void (*destroy) (void *data);

	/** The control is freed */
	void (*free) (void *data);

	/** control is linked to another control */
	void (*linked) (void *data, struct pw_control *other);
	/** control is unlinked from another control */
	void (*unlinked) (void *data, struct pw_control *other);

};

/** Get the control parent port or NULL when not set */
struct pw_impl_port *pw_control_get_port(struct pw_control *control);

/** Add an event listener on the control */
void pw_control_add_listener(struct pw_control *control,
			     struct spa_hook *listener,
			     const struct pw_control_events *events,
			     void *data);

/**
 * \}
 */

#ifdef __cplusplus
}
#endif

#endif /* PIPEWIRE_CONTROL_H */
