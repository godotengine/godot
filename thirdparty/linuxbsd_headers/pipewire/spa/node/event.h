/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_EVENT_NODE_H
#define SPA_EVENT_NODE_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \addtogroup spa_node
 * \{
 */

#include <spa/pod/event.h>

/* object id of SPA_TYPE_EVENT_Node */
enum spa_node_event {
	SPA_NODE_EVENT_Error,
	SPA_NODE_EVENT_Buffering,
	SPA_NODE_EVENT_RequestRefresh,
	SPA_NODE_EVENT_RequestProcess,		/*< Ask the driver to start processing
						 *  the graph */
};

#define SPA_NODE_EVENT_ID(ev)	SPA_EVENT_ID(ev, SPA_TYPE_EVENT_Node)
#define SPA_NODE_EVENT_INIT(id) SPA_EVENT_INIT(SPA_TYPE_EVENT_Node, id)

/* properties for SPA_TYPE_EVENT_Node */
enum spa_event_node {
	SPA_EVENT_NODE_START,
};

/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SPA_EVENT_NODE_H */
