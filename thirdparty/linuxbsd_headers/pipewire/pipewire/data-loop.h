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

/** Get the loop name. Since 1.1.0 */
const char * pw_data_loop_get_name(struct pw_data_loop *loop);
/** Get the loop class. Since 1.1.0 */
const char * pw_data_loop_get_class(struct pw_data_loop *loop);

/** Destroy the loop */
void pw_data_loop_destroy(struct pw_data_loop *loop);

/** Start the processing thread */
int pw_data_loop_start(struct pw_data_loop *loop);

/** Stop the processing thread */
int pw_data_loop_stop(struct pw_data_loop *loop);

/** Check if the current thread is the processing thread.
 * May be called from any thread. */
bool pw_data_loop_in_thread(struct pw_data_loop *loop);
/** Get the thread object */
struct spa_thread *pw_data_loop_get_thread(struct pw_data_loop *loop);

/** invoke func in the context of the thread or in the caller thread when
 * the loop is not running. May be called from the loop's thread, but otherwise
 * can only be called by a single thread at a time.
 * If called from the loop's thread, all callbacks previously queued with
 * pw_data_loop_invoke() will be run synchronously, which might cause
 * unexpected reentrancy problems.
 *
 * \param[in] loop The loop to invoke func on.
 * \param func The function to be invoked.
 * \param seq A sequence number, opaque to PipeWire. This will be made
 *            available to func.
 * \param[in] data Data that will be copied into the internal ring buffer and made
 *             available to func. Because this data is copied, it is okay to
 *             pass a pointer to a local variable, but do not pass a pointer to
 *             an object that has identity.
 * \param size The size of data to copy.
 * \param block If \true, do not return until func has been called. Otherwise,
 *              returns immediately. Passing \true does not risk a deadlock because
 *              the data thread is never allowed to wait on any other thread.
 * \param user_data An opaque pointer passed to func.
 * \return `-EPIPE` if the internal ring buffer filled up,
 *         if block is \false, 0 is returned when seq is SPA_ID_INVALID or the
 *         sequence number with the ASYNC bit set otherwise. When block is \true,
 *         the return value of func is returned.
 *
 * Since 0.3.3 */
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
