/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_POD_DYNAMIC_H
#define SPA_POD_DYNAMIC_H

#ifdef __cplusplus
extern "C" {
#endif

#include <spa/pod/builder.h>
#include <spa/utils/cleanup.h>

struct spa_pod_dynamic_builder {
	struct spa_pod_builder b;
	void *data;
	uint32_t extend;
	uint32_t _padding;
};

static int spa_pod_dynamic_builder_overflow(void *data, uint32_t size)
{
	struct spa_pod_dynamic_builder *d = (struct spa_pod_dynamic_builder*)data;
	int32_t old_size = d->b.size;
	int32_t new_size = SPA_ROUND_UP_N(size, d->extend);
	void *old_data = d->b.data, *new_data;

	if (old_data == d->data)
		d->b.data = NULL;
	if ((new_data = realloc(d->b.data, new_size)) == NULL)
		return -errno;
	if (old_data == d->data && new_data != old_data && old_size > 0)
		memcpy(new_data, old_data, old_size);
	d->b.data = new_data;
	d->b.size = new_size;
        return 0;
}

static inline void spa_pod_dynamic_builder_init(struct spa_pod_dynamic_builder *builder,
		void *data, uint32_t size, uint32_t extend)
{
	static const struct spa_pod_builder_callbacks spa_pod_dynamic_builder_callbacks = {
		SPA_VERSION_POD_BUILDER_CALLBACKS,
		.overflow = spa_pod_dynamic_builder_overflow
	};
	builder->b = SPA_POD_BUILDER_INIT(data, size);
	spa_pod_builder_set_callbacks(&builder->b, &spa_pod_dynamic_builder_callbacks, builder);
	builder->extend = extend;
	builder->data = data;
}

static inline void spa_pod_dynamic_builder_clean(struct spa_pod_dynamic_builder *builder)
{
	if (builder->data != builder->b.data)
		free(builder->b.data);
}

SPA_DEFINE_AUTO_CLEANUP(spa_pod_dynamic_builder, struct spa_pod_dynamic_builder, {
	spa_pod_dynamic_builder_clean(thing);
})

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SPA_POD_DYNAMIC_H */
