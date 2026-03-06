/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2019 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_DEVICE_UTILS_H
#define SPA_DEVICE_UTILS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <spa/pod/builder.h>
#include <spa/monitor/device.h>

/**
 * \addtogroup spa_device
 * \{
 */

struct spa_result_device_params_data {
	struct spa_pod_builder *builder;
	struct spa_result_device_params data;
};

static inline void spa_result_func_device_params(void *data, int seq SPA_UNUSED, int res SPA_UNUSED,
		uint32_t type SPA_UNUSED, const void *result)
{
	struct spa_result_device_params_data *d =
		(struct spa_result_device_params_data *)data;
	const struct spa_result_device_params *r =
		(const struct spa_result_device_params *)result;
	uint32_t offset = d->builder->state.offset;
	if (spa_pod_builder_raw_padded(d->builder, r->param, SPA_POD_SIZE(r->param)) < 0)
		return;
	d->data.next = r->next;
	d->data.param = spa_pod_builder_deref(d->builder, offset);
}

static inline int spa_device_enum_params_sync(struct spa_device *device,
			uint32_t id, uint32_t *index,
			const struct spa_pod *filter,
			struct spa_pod **param,
			struct spa_pod_builder *builder)
{
	struct spa_result_device_params_data data = { builder, {0}};
	struct spa_hook listener = {{0}, {0}, 0, 0};
	static const struct spa_device_events device_events = {
		.version = SPA_VERSION_DEVICE_EVENTS,
		.info = NULL,
		.result = spa_result_func_device_params,
	};
	int res;

	spa_device_add_listener(device, &listener, &device_events, &data);
	res = spa_device_enum_params(device, 0, id, *index, 1, filter);
	spa_hook_remove(&listener);

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

#define spa_device_emit(hooks,method,version,...)				\
		spa_hook_list_call_simple(hooks, struct spa_device_events,	\
				method, version, ##__VA_ARGS__)

#define spa_device_emit_info(hooks,i)		spa_device_emit(hooks,info, 0, i)
#define spa_device_emit_result(hooks,s,r,t,res)	spa_device_emit(hooks,result, 0, s, r, t, res)
#define spa_device_emit_event(hooks,e)		spa_device_emit(hooks,event, 0, e)
#define spa_device_emit_object_info(hooks,id,i)	spa_device_emit(hooks,object_info, 0, id, i)

/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SPA_DEVICE_UTILS_H */
