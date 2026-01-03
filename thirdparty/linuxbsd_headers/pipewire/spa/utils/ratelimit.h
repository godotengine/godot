/* Ratelimit */
/* SPDX-FileCopyrightText: Copyright © 2023 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_RATELIMIT_H
#define SPA_RATELIMIT_H

#ifdef __cplusplus
extern "C" {
#endif

#include <inttypes.h>
#include <stddef.h>

struct spa_ratelimit {
	uint64_t interval;
	uint64_t begin;
	unsigned burst;
	unsigned n_printed;
	unsigned n_suppressed;
};

static inline int spa_ratelimit_test(struct spa_ratelimit *r, uint64_t now)
{
	unsigned suppressed = 0;
	if (r->begin + r->interval < now) {
		suppressed = r->n_suppressed;
		r->begin = now;
		r->n_printed = 0;
		r->n_suppressed = 0;
	} else if (r->n_printed >= r->burst) {
		r->n_suppressed++;
		return -1;
	}
	r->n_printed++;
	return suppressed;
}

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* SPA_RATELIMIT_H */
