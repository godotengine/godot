/* PipeWire */
/* SPDX-FileCopyrightText: Copyright © 2021 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef PIPEWIRE_THREAD_H
#define PIPEWIRE_THREAD_H

#ifdef __cplusplus
extern "C" {
#endif

#include <string.h>
#include <errno.h>

#include <spa/support/thread.h>

/** \defgroup pw_thread Thread
 *
 * \brief functions to manipulate threads
 */

/**
 * \addtogroup pw_thread
 * \{
 */

SPA_DEPRECATED
void pw_thread_utils_set(struct spa_thread_utils *impl);
struct spa_thread_utils *pw_thread_utils_get(void);
void *pw_thread_fill_attr(const struct spa_dict *props, void *attr);

#define pw_thread_utils_create(...)		spa_thread_utils_create(pw_thread_utils_get(), ##__VA_ARGS__)
#define pw_thread_utils_join(...)		spa_thread_utils_join(pw_thread_utils_get(), ##__VA_ARGS__)
#define pw_thread_utils_get_rt_range(...)	spa_thread_utils_get_rt_range(pw_thread_utils_get(), ##__VA_ARGS__)
#define pw_thread_utils_acquire_rt(...)		spa_thread_utils_acquire_rt(pw_thread_utils_get(), ##__VA_ARGS__)
#define pw_thread_utils_drop_rt(...)		spa_thread_utils_drop_rt(pw_thread_utils_get(), ##__VA_ARGS__)

/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* PIPEWIRE_THREAD_H */
