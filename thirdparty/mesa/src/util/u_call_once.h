/*
 * Copyright 2022 Yonggang Luo
 * SPDX-License-Identifier: MIT
 *
 * Extend C11 call_once to support context parameter
 */

#ifndef U_CALL_ONCE_H_
#define U_CALL_ONCE_H_

#include <stdbool.h>

#include "c11/threads.h"
#include "macros.h"
#include "u_atomic.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The data can be mutable or immutable. */
typedef void (*util_call_once_data_func)(const void *data);

struct util_once_flag {
   bool called;
   once_flag flag;
};
typedef struct util_once_flag util_once_flag;

#define UTIL_ONCE_FLAG_INIT { false, ONCE_FLAG_INIT }

/**
 * This is used to optimize the call to call_once out when the func are
 * already called and finished, so when util_call_once are called in
 * hot path it's only incur an extra load instruction cost.
 */
static ALWAYS_INLINE void
util_call_once(util_once_flag *flag, void (*func)(void))
{
   if (unlikely(!p_atomic_read_relaxed(&flag->called))) {
      call_once(&flag->flag, func);
      p_atomic_set(&flag->called, true);
   }
}

/**
 * @brief Wrapper around call_once to pass data to func
 */
void
util_call_once_data_slow(once_flag *once, util_call_once_data_func func, const void *data);

/**
 * This is used to optimize the call to util_call_once_data_slow out when
 * the func function are already called and finished,
 * so when util_call_once_data are called in hot path it's only incur an extra
 * load instruction cost.
 */
static ALWAYS_INLINE void
util_call_once_data(util_once_flag *flag, util_call_once_data_func func, const void *data)
{
   if (unlikely(!p_atomic_read_relaxed(&flag->called))) {
      util_call_once_data_slow(&(flag->flag), func, data);
      p_atomic_set(&flag->called, true);
   }
}

#ifdef __cplusplus
}
#endif

#endif /* U_CALL_ONCE_H_ */
