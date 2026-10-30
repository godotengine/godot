/* PipeWire */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef PIPEWIRE_MAIN_LOOP_H
#define PIPEWIRE_MAIN_LOOP_H

#ifdef __cplusplus
extern "C" {
#endif

/** \defgroup pw_main_loop Main Loop
 *
 * A main loop object
 */

/**
 * \addtogroup pw_main_loop
 * \{
 */

/** A main loop object */
struct pw_main_loop;

#include <pipewire/loop.h>

/** Events of the main loop */
struct pw_main_loop_events {
#define PW_VERSION_MAIN_LOOP_EVENTS	0
	uint32_t version;

	/** Emitted when the main loop is destroyed */
	void (*destroy) (void *data);
};

/** Create a new main loop. */
struct pw_main_loop *
pw_main_loop_new(const struct spa_dict *props);

/** Add an event listener */
void pw_main_loop_add_listener(struct pw_main_loop *loop,
			       struct spa_hook *listener,
			       const struct pw_main_loop_events *events,
			       void *data);

/** Get the loop implementation */
struct pw_loop * pw_main_loop_get_loop(struct pw_main_loop *loop);

/** Destroy a loop */
void pw_main_loop_destroy(struct pw_main_loop *loop);

/** Run a main loop. This blocks until \ref pw_main_loop_quit is called */
int pw_main_loop_run(struct pw_main_loop *loop);

/** Quit a main loop */
int pw_main_loop_quit(struct pw_main_loop *loop);

/**
 * \}
 */

#ifdef __cplusplus
}
#endif

#endif /* PIPEWIRE_MAIN_LOOP_H */
