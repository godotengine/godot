/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_GRAPH_H
#define SPA_GRAPH_H

#ifdef __cplusplus
extern "C" {
#endif

/** \defgroup spa_graph Graph
 * Node graph
 */

/**
 * \addtogroup spa_graph
 * \{
 */

#include <spa/utils/atomic.h>
#include <spa/utils/defs.h>
#include <spa/utils/list.h>
#include <spa/utils/hook.h>
#include <spa/node/node.h>
#include <spa/node/io.h>

#ifndef spa_debug
#define spa_debug(...)
#endif

struct spa_graph;
struct spa_graph_node;
struct spa_graph_link;
struct spa_graph_port;

struct spa_graph_state {
	int status;			/**< current status */
	int32_t required;		/**< required number of signals */
	int32_t pending;		/**< number of pending signals */
};

static inline void spa_graph_state_reset(struct spa_graph_state *state)
{
	state->pending = state->required;
}

struct spa_graph_link {
	struct spa_list link;
	struct spa_graph_state *state;
	int (*signal) (void *data);
	void *signal_data;
};

#define spa_graph_link_signal(l)	((l)->signal((l)->signal_data))

#define spa_graph_state_dec(s) (SPA_ATOMIC_DEC(s->pending) == 0)

static inline int spa_graph_link_trigger(struct spa_graph_link *link)
{
	struct spa_graph_state *state = link->state;

	spa_debug("link %p: state %p: pending %d/%d", link, state,
                        state->pending, state->required);

	if (spa_graph_state_dec(state))
		spa_graph_link_signal(link);

        return state->status;
}
struct spa_graph {
	uint32_t flags;			/* flags */
	struct spa_graph_node *parent;	/* parent node or NULL when driver */
	struct spa_graph_state *state;	/* state of graph */
	struct spa_list nodes;		/* list of nodes of this graph */
};

struct spa_graph_node_callbacks {
#define SPA_VERSION_GRAPH_NODE_CALLBACKS	0
	uint32_t version;

	int (*process) (void *data, struct spa_graph_node *node);
	int (*reuse_buffer) (void *data, struct spa_graph_node *node,
			uint32_t port_id, uint32_t buffer_id);
};

struct spa_graph_node {
	struct spa_list link;		/**< link in graph nodes list */
	struct spa_graph *graph;	/**< owner graph */
	struct spa_list ports[2];	/**< list of input and output ports */
	struct spa_list links;		/**< list of links to next nodes */
	uint32_t flags;			/**< node flags */
	struct spa_graph_state *state;	/**< state of the node */
	struct spa_graph_link graph_link;	/**< link in graph */
	struct spa_graph *subgraph;	/**< subgraph or NULL */
	struct spa_callbacks callbacks;
	struct spa_list sched_link;	/**< link for scheduler */
};

#define spa_graph_node_call(n,method,version,...)			\
({									\
	int __res = 0;							\
	spa_callbacks_call_res(&(n)->callbacks,				\
			struct spa_graph_node_callbacks, __res,		\
			method, (version), ##__VA_ARGS__);		\
	__res;								\
})

#define spa_graph_node_process(n)		spa_graph_node_call((n), process, 0, (n))
#define spa_graph_node_reuse_buffer(n,p,i)	spa_graph_node_call((n), reuse_buffer, 0, (n), (p), (i))

struct spa_graph_port {
	struct spa_list link;		/**< link in node port list */
	struct spa_graph_node *node;	/**< owner node */
	enum spa_direction direction;	/**< port direction */
	uint32_t port_id;		/**< port id */
	uint32_t flags;			/**< port flags */
	struct spa_graph_port *peer;	/**< peer */
};

static inline int spa_graph_node_trigger(struct spa_graph_node *node)
{
	struct spa_graph_link *l;
	spa_debug("node %p trigger", node);
	spa_list_for_each(l, &node->links, link)
		spa_graph_link_trigger(l);
	return 0;
}

static inline int spa_graph_run(struct spa_graph *graph)
{
	struct spa_graph_node *n, *t;
	struct spa_list pending;

	spa_graph_state_reset(graph->state);
	spa_debug("graph %p run with state %p pending %d/%d", graph, graph->state,
			graph->state->pending, graph->state->required);

	spa_list_init(&pending);

	spa_list_for_each(n, &graph->nodes, link) {
		struct spa_graph_state *s = n->state;
		spa_graph_state_reset(s);
		spa_debug("graph %p node %p: state %p pending %d/%d status %d", graph, n,
				s, s->pending, s->required, s->status);
		if (--s->pending == 0)
			spa_list_append(&pending, &n->sched_link);
	}
	spa_list_for_each_safe(n, t, &pending, sched_link)
		spa_graph_node_process(n);

	return 0;
}

static inline int spa_graph_finish(struct spa_graph *graph)
{
	spa_debug("graph %p finish", graph);
	if (graph->parent)
		return spa_graph_node_trigger(graph->parent);
	return 0;
}
static inline int spa_graph_link_signal_node(void *data)
{
	struct spa_graph_node *node = (struct spa_graph_node *)data;
	spa_debug("node %p call process", node);
	return spa_graph_node_process(node);
}

static inline int spa_graph_link_signal_graph(void *data)
{
	struct spa_graph_node *node = (struct spa_graph_node *)data;
	return spa_graph_finish(node->graph);
}

static inline void spa_graph_init(struct spa_graph *graph, struct spa_graph_state *state)
{
	spa_list_init(&graph->nodes);
	graph->flags = 0;
	graph->state = state;
	spa_debug("graph %p init state %p", graph, state);
}

static inline void
spa_graph_link_add(struct spa_graph_node *out,
		   struct spa_graph_state *state,
		   struct spa_graph_link *link)
{
	link->state = state;
	state->required++;
	spa_debug("node %p add link %p to state %p %d", out, link, state, state->required);
	spa_list_append(&out->links, &link->link);
}

static inline void spa_graph_link_remove(struct spa_graph_link *link)
{
	link->state->required--;
	spa_debug("link %p state %p remove %d", link, link->state, link->state->required);
	spa_list_remove(&link->link);
}

static inline void
spa_graph_node_init(struct spa_graph_node *node, struct spa_graph_state *state)
{
	spa_list_init(&node->ports[SPA_DIRECTION_INPUT]);
	spa_list_init(&node->ports[SPA_DIRECTION_OUTPUT]);
	spa_list_init(&node->links);
	node->flags = 0;
	node->subgraph = NULL;
	node->state = state;
	node->state->required = node->state->pending = 0;
	node->state->status = SPA_STATUS_OK;
	node->graph_link.signal = spa_graph_link_signal_graph;
	node->graph_link.signal_data = node;
	spa_debug("node %p init state %p", node, state);
}


static inline int spa_graph_node_impl_sub_process(void *data SPA_UNUSED, struct spa_graph_node *node)
{
	struct spa_graph *graph = node->subgraph;
	spa_debug("node %p: sub process %p", node, graph);
	return spa_graph_run(graph);
}

static const struct spa_graph_node_callbacks spa_graph_node_sub_impl_default = {
	SPA_VERSION_GRAPH_NODE_CALLBACKS,
	.process = spa_graph_node_impl_sub_process,
};

static inline void spa_graph_node_set_subgraph(struct spa_graph_node *node,
		struct spa_graph *subgraph)
{
	node->subgraph = subgraph;
	subgraph->parent = node;
	spa_debug("node %p set subgraph %p", node, subgraph);
}

static inline void
spa_graph_node_set_callbacks(struct spa_graph_node *node,
		const struct spa_graph_node_callbacks *callbacks,
		void *data)
{
	node->callbacks = SPA_CALLBACKS_INIT(callbacks, data);
}

static inline void
spa_graph_node_add(struct spa_graph *graph,
		   struct spa_graph_node *node)
{
	node->graph = graph;
	spa_list_append(&graph->nodes, &node->link);
	node->state->required++;
	spa_debug("node %p add to graph %p, state %p required %d",
			node, graph, node->state, node->state->required);
	spa_graph_link_add(node, graph->state, &node->graph_link);
}

static inline void spa_graph_node_remove(struct spa_graph_node *node)
{
	spa_debug("node %p remove from graph %p, state %p required %d",
			node, node->graph, node->state, node->state->required);
	spa_graph_link_remove(&node->graph_link);
	node->state->required--;
	spa_list_remove(&node->link);
}


static inline void
spa_graph_port_init(struct spa_graph_port *port,
		    enum spa_direction direction,
		    uint32_t port_id,
		    uint32_t flags)
{
	spa_debug("port %p init type %d id %d", port, direction, port_id);
	port->direction = direction;
	port->port_id = port_id;
	port->flags = flags;
}

static inline void
spa_graph_port_add(struct spa_graph_node *node,
		   struct spa_graph_port *port)
{
	spa_debug("port %p add to node %p", port, node);
	port->node = node;
	spa_list_append(&node->ports[port->direction], &port->link);
}

static inline void spa_graph_port_remove(struct spa_graph_port *port)
{
	spa_debug("port %p remove", port);
	spa_list_remove(&port->link);
}

static inline void
spa_graph_port_link(struct spa_graph_port *out, struct spa_graph_port *in)
{
	spa_debug("port %p link to %p %p %p", out, in, in->node, in->node->state);
	out->peer = in;
	in->peer = out;
}

static inline void
spa_graph_port_unlink(struct spa_graph_port *port)
{
	spa_debug("port %p unlink from %p", port, port->peer);
	if (port->peer) {
		port->peer->peer = NULL;
		port->peer = NULL;
	}
}

static inline int spa_graph_node_impl_process(void *data, struct spa_graph_node *node)
{
	struct spa_node *n = (struct spa_node *)data;
	struct spa_graph_state *state = node->state;

	spa_debug("node %p: process state %p: %d, node %p", node, state, state->status, n);
	if ((state->status = spa_node_process(n)) != SPA_STATUS_OK)
		spa_graph_node_trigger(node);

        return state->status;
}

static inline int spa_graph_node_impl_reuse_buffer(void *data, struct spa_graph_node *node SPA_UNUSED,
		uint32_t port_id, uint32_t buffer_id)
{
	struct spa_node *n = (struct spa_node *)data;
	return spa_node_port_reuse_buffer(n, port_id, buffer_id);
}

static const struct spa_graph_node_callbacks spa_graph_node_impl_default = {
	SPA_VERSION_GRAPH_NODE_CALLBACKS,
	.process = spa_graph_node_impl_process,
	.reuse_buffer = spa_graph_node_impl_reuse_buffer,
};

/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SPA_GRAPH_H */
