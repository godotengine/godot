/* PipeWire */
/* SPDX-FileCopyrightText: Copyright © 2019 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef PIPEWIRE_BUFFERS_H
#define PIPEWIRE_BUFFERS_H

#include <spa/node/node.h>

#include <pipewire/context.h>
#include <pipewire/mem.h>

#ifdef __cplusplus
extern "C" {
#endif

/** \defgroup pw_buffers Buffers
 * Buffer handling
 */

/**
 * \addtogroup pw_buffers
 * \{
 */

#define PW_BUFFERS_FLAG_NONE		0
#define PW_BUFFERS_FLAG_NO_MEM		(1<<0)	/**< don't allocate buffer memory */
#define PW_BUFFERS_FLAG_SHARED		(1<<1)	/**< buffers can be shared */
#define PW_BUFFERS_FLAG_DYNAMIC		(1<<2)	/**< buffers have dynamic data */
#define PW_BUFFERS_FLAG_SHARED_MEM	(1<<3)	/**< buffers need shared memory */
#define PW_BUFFERS_FLAG_IN_PRIORITY	(1<<4)	/**< input parameters have priority */
#define PW_BUFFERS_FLAG_ASYNC		(1<<5)	/**< one of the nodes is async */

struct pw_buffers {
	struct pw_memblock *mem;	/**< allocated buffer memory */
	struct spa_buffer **buffers;	/**< port buffers */
	uint32_t n_buffers;		/**< number of port buffers */
	uint32_t flags;			/**< flags */
};

int pw_buffers_negotiate(struct pw_context *context, uint32_t flags,
		struct spa_node *outnode, uint32_t out_port_id,
		struct spa_node *innode, uint32_t in_port_id,
		struct pw_buffers *result);

void pw_buffers_clear(struct pw_buffers *buffers);

/**
 * \}
 */

#ifdef __cplusplus
}
#endif

#endif /* PIPEWIRE_BUFFERS_H */
