/* PipeWire */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef PIPEWIRE_DATA_LOOP_H
#define PIPEWIRE_DATA_LOOP_H

#ifdef __cplusplus
extern "C" {
#endif

#include <spa/utils/hook.h>
#include <spa/support/thread.h>

/** \defgroup pw_data_loop Data Loop
 *
 * \brief PipeWire rt-loop object
 *
 * This loop starts a new real-time thread that
 * is designed to run the processing graph.
 */

/**
 * \addtogroup pw_data_loop
 * \{
 */
struct pw_data_loop;

#include <pipewire/loop.h>
#include <pipewire/properties.h>

/** Loop events, use \ref pw_data_loop_add_listener to add a listener */
struct pw_data_loop_events {
#define PW_VERSION_DATA_LOOP_EVENTS		0
	uint32_t version;
	/** The loop is destroyed */
	void (*destroy) (void *data);
};

/** Make a new loop. */
struct pw_data_loop *
pw_data_loop_new(const struct spa_dict *props);

/** Add an event listener to loop */
void pw_data_loop_add_listener(struct pw_data_loop *loop,
			       struct spa_hook *listener,
			       const struct pw_data_loop_events *events,
			       void *data);

/** wait for activity on the loop up to \a timeout milliseconds.
 * Should be called from the loop function */
int pw_data_loop_wait(struct pw_data_loop *loop, int timeout);

/** make sure the thread will exit. Can be called from a loop callback */
void pw_data_loop_exit(struct pw_data_loop *loop);

/** Get the loop implementation of this data loop */
struct pw_loop *
pw_data_loop_get_loop(struct pw_data_loop *loop);

/** Destroy the loop */
void pw_data_loop_destroy(struct pw_data_loop *loop);

/** Start the processing thread */
int pw_data_loop_start(struct pw_data_loop *loop);

/** Stop the processing thread */
int pw_data_loop_stop(struct pw_data_loop *loop);

/** Check if the current thread is the processing thread */
bool pw_data_loop_in_thread(struct pw_data_loop *loop);
/** Get the thread object */
struct spa_thread *pw_data_loop_get_thread(struct pw_data_loop *loop);

/** invoke func in the context of the thread or in the caller thread when
 * the loop is not running. Since 0.3.3 */
int pw_data_loop_invoke(struct pw_data_loop *loop,
		spa_invoke_func_t func, uint32_t seq, const void *data, size_t size,
		bool block, void *user_data);

/** Set a custom spa_thread_utils for this loop. Setting NULL restores the
 * system default implementation. Since 0.3.50 */
void pw_data_loop_set_thread_utils(struct pw_data_loop *loop,
		struct spa_thread_utils *impl);
/**
 * \}
 */

#ifdef __cplusplus
}
#endif

#endif /* PIPEWIRE_DATA_LOOP_H */
