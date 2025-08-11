/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_POD_COMPARE_H
#define SPA_POD_COMPARE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdarg.h>
#include <errno.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include <spa/param/props.h>
#include <spa/pod/iter.h>
#include <spa/pod/builder.h>

/**
 * \addtogroup spa_pod
 * \{
 */

static inline int spa_pod_compare_value(uint32_t type, const void *r1, const void *r2, uint32_t size)
{
	switch (type) {
	case SPA_TYPE_None:
		return 0;
	case SPA_TYPE_Bool:
	case SPA_TYPE_Id:
		return *(uint32_t *) r1 == *(uint32_t *) r2 ? 0 : 1;
	case SPA_TYPE_Int:
		return *(int32_t *) r1 - *(int32_t *) r2;
	case SPA_TYPE_Long:
		return *(int64_t *) r1 - *(int64_t *) r2;
	case SPA_TYPE_Float:
		return *(float *) r1 - *(float *) r2;
	case SPA_TYPE_Double:
		return *(double *) r1 - *(double *) r2;
	case SPA_TYPE_String:
		return strcmp((char *)r1, (char *)r2);
	case SPA_TYPE_Bytes:
		return memcmp((char *)r1, (char *)r2, size);
	case SPA_TYPE_Rectangle:
	{
		const struct spa_rectangle *rec1 = (struct spa_rectangle *) r1,
		    *rec2 = (struct spa_rectangle *) r2;
		if (rec1->width == rec2->width && rec1->height == rec2->height)
			return 0;
		else if (rec1->width < rec2->width || rec1->height < rec2->height)
			return -1;
		else
			return 1;
	}
	case SPA_TYPE_Fraction:
	{
		const struct spa_fraction *f1 = (struct spa_fraction *) r1,
		    *f2 = (struct spa_fraction *) r2;
		int64_t n1, n2;
		n1 = ((int64_t) f1->num) * f2->denom;
		n2 = ((int64_t) f2->num) * f1->denom;
		if (n1 < n2)
			return -1;
		else if (n1 > n2)
			return 1;
		else
			return 0;
	}
	default:
		break;
	}
	return 0;
}

static inline int spa_pod_compare(const struct spa_pod *pod1,
				  const struct spa_pod *pod2)
{
	int res = 0;
	uint32_t n_vals1, n_vals2;
	uint32_t choice1, choice2;

        spa_return_val_if_fail(pod1 != NULL, -EINVAL);
        spa_return_val_if_fail(pod2 != NULL, -EINVAL);

	pod1 = spa_pod_get_values(pod1,  &n_vals1, &choice1);
	pod2 = spa_pod_get_values(pod2,  &n_vals2, &choice2);

	if (n_vals1 != n_vals2)
		return -EINVAL;

	if (SPA_POD_TYPE(pod1) != SPA_POD_TYPE(pod2))
		return -EINVAL;

	switch (SPA_POD_TYPE(pod1)) {
	case SPA_TYPE_Struct:
	{
		const struct spa_pod *p1, *p2;
		size_t p1s, p2s;

		p1 = (const struct spa_pod*)SPA_POD_BODY_CONST(pod1);
		p1s = SPA_POD_BODY_SIZE(pod1);
		p2 = (const struct spa_pod*)SPA_POD_BODY_CONST(pod2);
		p2s = SPA_POD_BODY_SIZE(pod2);

		while (true) {
			if (!spa_pod_is_inside(pod1, p1s, p1) ||
			    !spa_pod_is_inside(pod2, p2s, p2))
				return -EINVAL;

			if ((res = spa_pod_compare(p1, p2)) != 0)
				return res;

			p1 = (const struct spa_pod*)spa_pod_next(p1);
			p2 = (const struct spa_pod*)spa_pod_next(p2);
		}
		break;
	}
	case SPA_TYPE_Object:
	{
		const struct spa_pod_prop *p1, *p2;
		const struct spa_pod_object *o1, *o2;

		o1 = (const struct spa_pod_object*)pod1;
		o2 = (const struct spa_pod_object*)pod2;

		p2 = NULL;
		SPA_POD_OBJECT_FOREACH(o1, p1) {
			if ((p2 = spa_pod_object_find_prop(o2, p2, p1->key)) == NULL)
				return 1;
			if ((res = spa_pod_compare(&p1->value, &p2->value)) != 0)
				return res;
		}
		p1 = NULL;
		SPA_POD_OBJECT_FOREACH(o2, p2) {
			if ((p1 = spa_pod_object_find_prop(o1, p1, p2->key)) == NULL)
				return -1;
		}
		break;
	}
	case SPA_TYPE_Array:
	{
		if (SPA_POD_BODY_SIZE(pod1) != SPA_POD_BODY_SIZE(pod2))
			return -EINVAL;
		res = memcmp(SPA_POD_BODY(pod1), SPA_POD_BODY(pod2), SPA_POD_BODY_SIZE(pod2));
		break;
	}
	default:
		if (SPA_POD_BODY_SIZE(pod1) != SPA_POD_BODY_SIZE(pod2))
			return -EINVAL;
		res = spa_pod_compare_value(SPA_POD_TYPE(pod1),
				SPA_POD_BODY(pod1), SPA_POD_BODY(pod2),
				SPA_POD_BODY_SIZE(pod1));
		break;
	}
	return res;
}

/**
 * \}
 */

#ifdef __cplusplus
}
#endif

#endif
