/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2023 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_PARAM_TAG_UTILS_H
#define SPA_PARAM_TAG_UTILS_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \addtogroup spa_param
 * \{
 */

#include <float.h>

#include <spa/utils/dict.h>
#include <spa/pod/builder.h>
#include <spa/pod/parser.h>
#include <spa/param/tag.h>

static inline int
spa_tag_compare(const struct spa_pod *a, const struct spa_pod *b)
{
	return ((a == b) || (a && b && SPA_POD_SIZE(a) == SPA_POD_SIZE(b) &&
	    memcmp(a, b, SPA_POD_SIZE(b)) == 0)) ? 0 : 1;
}

static inline int
spa_tag_parse(const struct spa_pod *tag, struct spa_tag_info *info, void **state)
{
	int res;
	const struct spa_pod_object *obj = (const struct spa_pod_object*)tag;
	const struct spa_pod_prop *first, *start, *cur;

	spa_zero(*info);

	if ((res = spa_pod_parse_object(tag,
			SPA_TYPE_OBJECT_ParamTag, NULL,
			SPA_PARAM_TAG_direction, SPA_POD_Id(&info->direction))) < 0)
		return res;

        first = spa_pod_prop_first(&obj->body);
        start = *state ? spa_pod_prop_next((struct spa_pod_prop*)*state) : first;

	res = 0;
	for (cur = start; spa_pod_prop_is_inside(&obj->body, obj->pod.size, cur);
	     cur = spa_pod_prop_next(cur)) {
		if (cur->key == SPA_PARAM_TAG_info) {
			info->info = &cur->value;
			*state = (void*)cur;
			return 1;
		}
        }
	return 0;
}

static inline int
spa_tag_info_parse(const struct spa_tag_info *info, struct spa_dict *dict, struct spa_dict_item *items)
{
	struct spa_pod_parser prs;
	uint32_t n, n_items;
	const char *key, *value;
	struct spa_pod_frame f[1];

	spa_pod_parser_pod(&prs, info->info);
	if (spa_pod_parser_push_struct(&prs, &f[0]) < 0 ||
	    spa_pod_parser_get_int(&prs, (int32_t*)&n_items) < 0)
		return -EINVAL;

	if (items == NULL) {
		dict->n_items = n_items;
		return 0;
	}
	n_items = SPA_MIN(dict->n_items, n_items);

	for (n = 0; n < n_items; n++) {
		if (spa_pod_parser_get(&prs,
				SPA_POD_String(&key),
				SPA_POD_String(&value),
				NULL) < 0)
			break;
		items[n].key = key;
		items[n].value = value;
	}
	dict->items = items;
	spa_pod_parser_pop(&prs, &f[0]);
	return 0;
}

static inline void
spa_tag_build_start(struct spa_pod_builder *builder, struct spa_pod_frame *f,
		uint32_t id, enum spa_direction direction)
{
	spa_pod_builder_push_object(builder, f, SPA_TYPE_OBJECT_ParamTag, id);
	spa_pod_builder_add(builder,
			SPA_PARAM_TAG_direction, SPA_POD_Id(direction),
			0);
}

static inline void
spa_tag_build_add_info(struct spa_pod_builder *builder, const struct spa_pod *info)
{
	spa_pod_builder_add(builder,
			SPA_PARAM_TAG_info, SPA_POD_Pod(info),
			0);
}

static inline void
spa_tag_build_add_dict(struct spa_pod_builder *builder, const struct spa_dict *dict)
{
	uint32_t i, n_items;
	struct spa_pod_frame f;

	n_items = dict ? dict->n_items : 0;

	spa_pod_builder_prop(builder, SPA_PARAM_TAG_info, SPA_POD_PROP_FLAG_HINT_DICT);
	spa_pod_builder_push_struct(builder, &f);
	spa_pod_builder_int(builder, n_items);
        for (i = 0; i < n_items; i++) {
		spa_pod_builder_string(builder, dict->items[i].key);
		spa_pod_builder_string(builder, dict->items[i].value);
	}
        spa_pod_builder_pop(builder, &f);
}

static inline struct spa_pod *
spa_tag_build_end(struct spa_pod_builder *builder, struct spa_pod_frame *f)
{
	return (struct spa_pod*)spa_pod_builder_pop(builder, f);
}

/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SPA_PARAM_TAG_UTILS_H */
