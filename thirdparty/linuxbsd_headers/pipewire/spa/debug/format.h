/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_DEBUG_FORMAT_H
#define SPA_DEBUG_FORMAT_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \addtogroup spa_debug
 * \{
 */

#include <spa/pod/parser.h>
#include <spa/utils/string.h>
#include <spa/debug/context.h>
#include <spa/debug/types.h>
#include <spa/param/type-info.h>
#include <spa/param/format-utils.h>

static inline int
spa_debug_strbuf_format_value(struct spa_strbuf *buffer, const struct spa_type_info *info,
		uint32_t type, void *body, uint32_t size)
{

	switch (type) {
	case SPA_TYPE_Bool:
		spa_strbuf_append(buffer, "%s", *(int32_t *) body ? "true" : "false");
		break;
	case SPA_TYPE_Id:
	{
		const char *str = spa_debug_type_find_short_name(info, *(int32_t *) body);
		char tmp[64];
		if (str == NULL) {
			snprintf(tmp, sizeof(tmp), "%d", *(int32_t*)body);
			str = tmp;
		}
		spa_strbuf_append(buffer, "%s", str);
		break;
	}
	case SPA_TYPE_Int:
		spa_strbuf_append(buffer, "%d", *(int32_t *) body);
		break;
	case SPA_TYPE_Long:
		spa_strbuf_append(buffer, "%" PRIi64, *(int64_t *) body);
		break;
	case SPA_TYPE_Float:
		spa_strbuf_append(buffer, "%f", *(float *) body);
		break;
	case SPA_TYPE_Double:
		spa_strbuf_append(buffer, "%f", *(double *) body);
		break;
	case SPA_TYPE_String:
		spa_strbuf_append(buffer, "%s", (char *) body);
		break;
	case SPA_TYPE_Rectangle:
	{
		struct spa_rectangle *r = (struct spa_rectangle *)body;
		spa_strbuf_append(buffer, "%" PRIu32 "x%" PRIu32, r->width, r->height);
		break;
	}
	case SPA_TYPE_Fraction:
	{
		struct spa_fraction *f = (struct spa_fraction *)body;
		spa_strbuf_append(buffer, "%" PRIu32 "/%" PRIu32, f->num, f->denom);
		break;
	}
	case SPA_TYPE_Bitmap:
		spa_strbuf_append(buffer, "Bitmap");
		break;
	case SPA_TYPE_Bytes:
		spa_strbuf_append(buffer, "Bytes");
		break;
	case SPA_TYPE_Array:
	{
		void *p;
		struct spa_pod_array_body *b = (struct spa_pod_array_body *)body;
		int i = 0;
		info = info && info->values ? info->values : info;
		spa_strbuf_append(buffer, "< ");
		SPA_POD_ARRAY_BODY_FOREACH(b, size, p) {
			if (i++ > 0)
				spa_strbuf_append(buffer, ", ");
			spa_debug_strbuf_format_value(buffer, info, b->child.type, p, b->child.size);
		}
		spa_strbuf_append(buffer, " >");
		break;
	}
	default:
		spa_strbuf_append(buffer, "INVALID type %d", type);
		break;
	}
	return 0;
}

static inline int
spa_debug_format_value(const struct spa_type_info *info,
		uint32_t type, void *body, uint32_t size)
{
	char buffer[1024];
	struct spa_strbuf buf;
	spa_strbuf_init(&buf, buffer, sizeof(buffer));
	spa_debug_strbuf_format_value(&buf, info, type, body, size);
	spa_debugn("%s", buffer);
	return 0;
}

static inline int spa_debugc_format(struct spa_debug_context *ctx, int indent,
		const struct spa_type_info *info, const struct spa_pod *format)
{
	const char *media_type;
	const char *media_subtype;
	struct spa_pod_prop *prop;
	uint32_t mtype, mstype;

	if (info == NULL)
		info = spa_type_format;

	if (format == NULL || SPA_POD_TYPE(format) != SPA_TYPE_Object)
		return -EINVAL;

	if (spa_format_parse(format, &mtype, &mstype) < 0)
		return -EINVAL;

	media_type = spa_debug_type_find_name(spa_type_media_type, mtype);
	media_subtype = spa_debug_type_find_name(spa_type_media_subtype, mstype);

	spa_debugc(ctx, "%*s %s/%s", indent, "",
		media_type ? spa_debug_type_short_name(media_type) : "unknown",
		media_subtype ? spa_debug_type_short_name(media_subtype) : "unknown");

	SPA_POD_OBJECT_FOREACH((struct spa_pod_object*)format, prop) {
		const char *key;
		const struct spa_type_info *ti;
		uint32_t i, type, size, n_vals, choice;
		const struct spa_pod *val;
		void *vals;
		char buffer[1024];
		struct spa_strbuf buf;

		if (prop->key == SPA_FORMAT_mediaType ||
		    prop->key == SPA_FORMAT_mediaSubtype)
			continue;

		val = spa_pod_get_values(&prop->value, &n_vals, &choice);

		type = val->type;
		size = val->size;
		vals = SPA_POD_BODY(val);

		if (type < SPA_TYPE_None || type >= _SPA_TYPE_LAST)
			continue;

		ti = spa_debug_type_find(info, prop->key);
		key = ti ? ti->name : NULL;

		spa_strbuf_init(&buf, buffer, sizeof(buffer));
		spa_strbuf_append(&buf, "%*s %16s : (%s) ", indent, "",
			key ? spa_debug_type_short_name(key) : "unknown",
			spa_debug_type_short_name(spa_types[type].name));

		if (choice == SPA_CHOICE_None) {
			spa_debug_strbuf_format_value(&buf, ti ? ti->values : NULL, type, vals, size);
		} else {
			const char *ssep, *esep, *sep;

			switch (choice) {
			case SPA_CHOICE_Range:
			case SPA_CHOICE_Step:
				ssep = "[ ";
				sep = ", ";
				esep = " ]";
				break;
			default:
			case SPA_CHOICE_Enum:
			case SPA_CHOICE_Flags:
				ssep = "{ ";
				sep = ", ";
				esep = " }";
				break;
			}

			spa_strbuf_append(&buf, "%s", ssep);

			for (i = 1; i < n_vals; i++) {
				vals = SPA_PTROFF(vals, size, void);
				if (i > 1)
					spa_strbuf_append(&buf, "%s", sep);
				spa_debug_strbuf_format_value(&buf, ti ? ti->values : NULL, type, vals, size);
			}
			spa_strbuf_append(&buf, "%s", esep);
		}
		spa_debugc(ctx, "%s", buffer);
	}
	return 0;
}

static inline int spa_debug_format(int indent,
		const struct spa_type_info *info, const struct spa_pod *format)
{
	return spa_debugc_format(NULL, indent, info, format);
}
/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SPA_DEBUG_FORMAT_H */
