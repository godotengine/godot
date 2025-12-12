/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2019 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_NODE_UTILS_H
#define SPA_NODE_UTILS_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \addtogroup spa_node
 * \{
 */

#include <spa/pod/builder.h>

#include <spa/node/node.h>

struct spa_result_node_params_data {
	struct spa_pod_builder *builder;
	struct spa_result_node_params data;
};

static inline void spa_result_func_node_params(void *data,
		int seq SPA_UNUSED, int res SPA_UNUSED, uint32_t type SPA_UNUSED, const void *result)
{
	struct spa_result_node_params_data *d =
		(struct spa_result_node_params_data *) data;
	const struct spa_result_node_params *r =
		(const struct spa_result_node_params *) result;
	uint32_t offset = d->builder->state.offset;
	if (spa_pod_builder_raw_padded(d->builder, r->param, SPA_POD_SIZE(r->param)) < 0)
		return;
	d->data.next = r->next;
	d->data.param = spa_pod_builder_deref(d->builder, offset);
}

static inline int spa_node_enum_params_sync(struct spa_node *node,
			uint32_t id, uint32_t *index,
			const struct spa_pod *filter,
			struct spa_pod **param,
			struct spa_pod_builder *builder)
{
	struct spa_result_node_params_data data = { builder, {0}};
	struct spa_hook listener = {{0}, {0}, 0, 0};
	static const struct spa_node_events node_events = {
		.version = SPA_VERSION_NODE_EVENTS,
		.info = NULL,
		.port_info = NULL,
		.result = spa_result_func_node_params,
	};
	int res;

	res = spa_node_add_listener(node, &listener, &node_events, &data);
	if (res >= 0) {
		res = spa_node_enum_params(node, 0, id, *index, 1, filter);
		spa_hook_remove(&listener);
	}

	if (data.data.param == NULL) {
		if (res > 0)
			res = 0;
	} else {
		*index = data.data.next;
		*param = data.data.param;
		res = 1;
	}
	return res;
}

static inline int spa_node_port_enum_params_sync(struct spa_node *node,
			enum spa_direction direction, uint32_t port_id,
			uint32_t id, uint32_t *index,
			const struct spa_pod *filter,
			struct spa_pod **param,
			struct spa_pod_builder *builder)
{
	struct spa_result_node_params_data data = { builder, {0}};
	struct spa_hook listener = {{0}, {0}, 0, 0};
	static const struct spa_node_events node_events = {
		.version = SPA_VERSION_NODE_EVENTS,
		.info = NULL,
		.port_info = NULL,
		.result = spa_result_func_node_params,
	};
	int res;

	res = spa_node_add_listener(node, &listener, &node_events, &data);
	if (res >= 0) {
		res = spa_node_port_enum_params(node, 0, direction, port_id,
				id, *index, 1, filter);
		spa_hook_remove(&listener);
	}

	if (data.data.param == NULL) {
		if (res > 0)
			res = 0;
	} else {
		*index = data.data.next;
		*param = data.data.param;
		res = 1;
	}
	return res;
}

#define spa_node_emit(hooks,method,version,...)					\
		spa_hook_list_call_simple(hooks, struct spa_node_events,	\
				method, version, ##__VA_ARGS__)

#define spa_node_emit_info(hooks,...)		spa_node_emit(hooks,info, 0, __VA_ARGS__)
#define spa_node_emit_port_info(hooks,...)	spa_node_emit(hooks,port_info, 0, __VA_ARGS__)
#define spa_node_emit_result(hooks,...)		spa_node_emit(hooks,result, 0, __VA_ARGS__)
#define spa_node_emit_event(hooks,...)		spa_node_emit(hooks,event, 0, __VA_ARGS__)


#define spa_node_call(callbacks,method,version,...)			\
({									\
	int _res;							\
	spa_callbacks_call_fast_res(callbacks, struct spa_node_callbacks,	\
			_res, method, version, ##__VA_ARGS__);		\
	_res;								\
})

#define spa_node_call_ready(hook,...)		spa_node_call(hook, ready, 0, __VA_ARGS__)
#define spa_node_call_reuse_buffer(hook,...)	spa_node_call(hook, reuse_buffer, 0, __VA_ARGS__)
#define spa_node_call_xrun(hook,...)		spa_node_call(hook, xrun, 0, __VA_ARGS__)

/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SPA_NODE_UTILS_H */
