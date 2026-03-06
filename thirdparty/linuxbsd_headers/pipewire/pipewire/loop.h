/* PipeWire */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef PIPEWIRE_LOOP_H
#define PIPEWIRE_LOOP_H

#ifdef __cplusplus
extern "C" {
#endif

#include <spa/support/loop.h>
#include <spa/utils/dict.h>

/** \defgroup pw_loop Loop
 *
 * PipeWire loop object provides an implementation of
 * the spa loop interfaces. It can be used to implement various
 * event loops.
 */

/**
 * \addtogroup pw_loop
 * \{
 */

struct pw_loop {
	struct spa_system *system;		/**< system utils */
	struct spa_loop *loop;			/**< wrapped loop */
	struct spa_loop_control *control;	/**< loop control */
	struct spa_loop_utils *utils;		/**< loop utils */
};

struct pw_loop *
pw_loop_new(const struct spa_dict *props);

void
pw_loop_destroy(struct pw_loop *loop);

#define pw_loop_add_source(l,...)	spa_loop_add_source((l)->loop,__VA_ARGS__)
#define pw_loop_update_source(l,...)	spa_loop_update_source((l)->loop,__VA_ARGS__)
#define pw_loop_remove_source(l,...)	spa_loop_remove_source((l)->loop,__VA_ARGS__)
#define pw_loop_invoke(l,...)		spa_loop_invoke((l)->loop,__VA_ARGS__)

#define pw_loop_get_fd(l)		spa_loop_control_get_fd((l)->control)
#define pw_loop_add_hook(l,...)		spa_loop_control_add_hook((l)->control,__VA_ARGS__)
#define pw_loop_enter(l)		spa_loop_control_enter((l)->control)
#define pw_loop_leave(l)		spa_loop_control_leave((l)->control)
#define pw_loop_iterate(l,...)		spa_loop_control_iterate_fast((l)->control,__VA_ARGS__)

#define pw_loop_add_io(l,...)		spa_loop_utils_add_io((l)->utils,__VA_ARGS__)
#define pw_loop_update_io(l,...)	spa_loop_utils_update_io((l)->utils,__VA_ARGS__)
#define pw_loop_add_idle(l,...)		spa_loop_utils_add_idle((l)->utils,__VA_ARGS__)
#define pw_loop_enable_idle(l,...)	spa_loop_utils_enable_idle((l)->utils,__VA_ARGS__)
#define pw_loop_add_event(l,...)	spa_loop_utils_add_event((l)->utils,__VA_ARGS__)
#define pw_loop_signal_event(l,...)	spa_loop_utils_signal_event((l)->utils,__VA_ARGS__)
#define pw_loop_add_timer(l,...)	spa_loop_utils_add_timer((l)->utils,__VA_ARGS__)
#define pw_loop_update_timer(l,...)	spa_loop_utils_update_timer((l)->utils,__VA_ARGS__)
#define pw_loop_add_signal(l,...)	spa_loop_utils_add_signal((l)->utils,__VA_ARGS__)
#define pw_loop_destroy_source(l,...)	spa_loop_utils_destroy_source((l)->utils,__VA_ARGS__)

/**
 * \}
 */

#ifdef __cplusplus
}
#endif

#endif /* PIPEWIRE_LOOP_H */
