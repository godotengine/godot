/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2020 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_PARAM_PROFILER_H
#define SPA_PARAM_PROFILER_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \addtogroup spa_param
 * \{
 */

#include <spa/param/param.h>

/** properties for SPA_TYPE_OBJECT_Profiler */
enum spa_profiler {
	SPA_PROFILER_START,

	SPA_PROFILER_START_Driver	= 0x10000,	/**< driver related profiler properties */
	SPA_PROFILER_info,				/**< Generic info, counter and CPU load,
							  * (Struct(
							  *      Long : counter,
							  *      Float : cpu_load fast,
							  *      Float : cpu_load medium,
							  *      Float : cpu_load slow),
							  *      Int : xrun-count))  */
	SPA_PROFILER_clock,				/**< clock information
							  *  (Struct(
							  *      Int : clock flags,
							  *      Int : clock id,
							  *      String: clock name,
							  *      Long : clock nsec,
							  *      Fraction : clock rate,
							  *      Long : clock position,
							  *      Long : clock duration,
							  *      Long : clock delay,
							  *      Double : clock rate_diff,
							  *      Long : clock next_nsec)) */
	SPA_PROFILER_driverBlock,			/**< generic driver info block
							  *  (Struct(
							  *      Int : driver_id,
							  *      String : name,
							  *      Long : driver prev_signal,
							  *      Long : driver signal,
							  *      Long : driver awake,
							  *      Long : driver finish,
							  *      Int : driver status),
							  *      Fraction : latency))  */

	SPA_PROFILER_START_Follower	= 0x20000,	/**< follower related profiler properties */
	SPA_PROFILER_followerBlock,			/**< generic follower info block
							  *  (Struct(
							  *      Int : id,
							  *      String : name,
							  *      Long : prev_signal,
							  *      Long : signal,
							  *      Long : awake,
							  *      Long : finish,
							  *      Int : status,
							  *      Fraction : latency))  */

	SPA_PROFILER_START_CUSTOM	= 0x1000000,
};

/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SPA_PARAM_PROFILER_H */
