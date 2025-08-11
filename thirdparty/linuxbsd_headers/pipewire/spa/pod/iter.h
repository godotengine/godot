/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_POD_ITER_H
#define SPA_POD_ITER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <errno.h>
#include <sys/types.h>

#include <spa/pod/pod.h>

/**
 * \addtogroup spa_pod
 * \{
 */

struct spa_pod_frame {
	struct spa_pod pod;
	struct spa_pod_frame *parent;
	uint32_t offset;
	uint32_t flags;
};

static inline bool spa_pod_is_inside(const void *pod, uint32_t size, const void *iter)
{
	return SPA_POD_BODY(iter) <= SPA_PTROFF(pod, size, void) &&
		SPA_PTROFF(iter, SPA_POD_SIZE(iter), void) <= SPA_PTROFF(pod, size, void);
}

static inline void *spa_pod_next(const void *iter)
{
	return SPA_PTROFF(iter, SPA_ROUND_UP_N(SPA_POD_SIZE(iter), 8), void);
}

static inline struct spa_pod_prop *spa_pod_prop_first(const struct spa_pod_object_body *body)
{
	return SPA_PTROFF(body, sizeof(struct spa_pod_object_body), struct spa_pod_prop);
}

static inline bool spa_pod_prop_is_inside(const struct spa_pod_object_body *body,
		uint32_t size, const struct spa_pod_prop *iter)
{
	return SPA_POD_CONTENTS(struct spa_pod_prop, iter) <= SPA_PTROFF(body, size, void) &&
		SPA_PTROFF(iter, SPA_POD_PROP_SIZE(iter), void) <= SPA_PTROFF(body, size, void);
}

static inline struct spa_pod_prop *spa_pod_prop_next(const struct spa_pod_prop *iter)
{
	return SPA_PTROFF(iter, SPA_ROUND_UP_N(SPA_POD_PROP_SIZE(iter), 8), struct spa_pod_prop);
}

static inline struct spa_pod_control *spa_pod_control_first(const struct spa_pod_sequence_body *body)
{
	return SPA_PTROFF(body, sizeof(struct spa_pod_sequence_body), struct spa_pod_control);
}

static inline bool spa_pod_control_is_inside(const struct spa_pod_sequence_body *body,
		uint32_t size, const struct spa_pod_control *iter)
{
	return SPA_POD_CONTENTS(struct spa_pod_control, iter) <= SPA_PTROFF(body, size, void) &&
		SPA_PTROFF(iter, SPA_POD_CONTROL_SIZE(iter), void) <= SPA_PTROFF(body, size, void);
}

static inline struct spa_pod_control *spa_pod_control_next(const struct spa_pod_control *iter)
{
	return SPA_PTROFF(iter, SPA_ROUND_UP_N(SPA_POD_CONTROL_SIZE(iter), 8), struct spa_pod_control);
}

#define SPA_POD_ARRAY_BODY_FOREACH(body, _size, iter)							\
	for ((iter) = (__typeof__(iter))SPA_PTROFF((body), sizeof(struct spa_pod_array_body), void);	\
	     (iter) < (__typeof__(iter))SPA_PTROFF((body), (_size), void);				\
	     (iter) = (__typeof__(iter))SPA_PTROFF((iter), (body)->child.size, void))

#define SPA_POD_ARRAY_FOREACH(obj, iter)							\
	SPA_POD_ARRAY_BODY_FOREACH(&(obj)->body, SPA_POD_BODY_SIZE(obj), iter)

#define SPA_POD_CHOICE_BODY_FOREACH(body, _size, iter)							\
	for ((iter) = (__typeof__(iter))SPA_PTROFF((body), sizeof(struct spa_pod_choice_body), void);	\
	     (iter) < (__typeof__(iter))SPA_PTROFF((body), (_size), void);				\
	     (iter) = (__typeof__(iter))SPA_PTROFF((iter), (body)->child.size, void))

#define SPA_POD_CHOICE_FOREACH(obj, iter)							\
	SPA_POD_CHOICE_BODY_FOREACH(&(obj)->body, SPA_POD_BODY_SIZE(obj), iter)

#define SPA_POD_FOREACH(pod, size, iter)					\
	for ((iter) = (pod);							\
	     spa_pod_is_inside(pod, size, iter);				\
	     (iter) = (__typeof__(iter))spa_pod_next(iter))

#define SPA_POD_STRUCT_FOREACH(obj, iter)							\
	SPA_POD_FOREACH(SPA_POD_BODY(obj), SPA_POD_BODY_SIZE(obj), iter)

#define SPA_POD_OBJECT_BODY_FOREACH(body, size, iter)						\
	for ((iter) = spa_pod_prop_first(body);				\
	     spa_pod_prop_is_inside(body, size, iter);			\
	     (iter) = spa_pod_prop_next(iter))

#define SPA_POD_OBJECT_FOREACH(obj, iter)							\
	SPA_POD_OBJECT_BODY_FOREACH(&(obj)->body, SPA_POD_BODY_SIZE(obj), iter)

#define SPA_POD_SEQUENCE_BODY_FOREACH(body, size, iter)						\
	for ((iter) = spa_pod_control_first(body);						\
	     spa_pod_control_is_inside(body, size, iter);						\
	     (iter) = spa_pod_control_next(iter))

#define SPA_POD_SEQUENCE_FOREACH(seq, iter)							\
	SPA_POD_SEQUENCE_BODY_FOREACH(&(seq)->body, SPA_POD_BODY_SIZE(seq), iter)


static inline void *spa_pod_from_data(void *data, size_t maxsize, off_t offset, size_t size)
{
	void *pod;
	if (size < sizeof(struct spa_pod) || offset + size > maxsize)
		return NULL;
	pod = SPA_PTROFF(data, offset, void);
	if (SPA_POD_SIZE(pod) > size)
		return NULL;
	return pod;
}

static inline int spa_pod_is_none(const struct spa_pod *pod)
{
	return (SPA_POD_TYPE(pod) == SPA_TYPE_None);
}

static inline int spa_pod_is_bool(const struct spa_pod *pod)
{
	return (SPA_POD_TYPE(pod) == SPA_TYPE_Bool && SPA_POD_BODY_SIZE(pod) >= sizeof(int32_t));
}

static inline int spa_pod_get_bool(const struct spa_pod *pod, bool *value)
{
	if (!spa_pod_is_bool(pod))
		return -EINVAL;
	*value = !!SPA_POD_VALUE(struct spa_pod_bool, pod);
	return 0;
}

static inline int spa_pod_is_id(const struct spa_pod *pod)
{
	return (SPA_POD_TYPE(pod) == SPA_TYPE_Id && SPA_POD_BODY_SIZE(pod) >= sizeof(uint32_t));
}

static inline int spa_pod_get_id(const struct spa_pod *pod, uint32_t *value)
{
	if (!spa_pod_is_id(pod))
		return -EINVAL;
	*value = SPA_POD_VALUE(struct spa_pod_id, pod);
	return 0;
}

static inline int spa_pod_is_int(const struct spa_pod *pod)
{
	return (SPA_POD_TYPE(pod) == SPA_TYPE_Int && SPA_POD_BODY_SIZE(pod) >= sizeof(int32_t));
}

static inline int spa_pod_get_int(const struct spa_pod *pod, int32_t *value)
{
	if (!spa_pod_is_int(pod))
		return -EINVAL;
	*value = SPA_POD_VALUE(struct spa_pod_int, pod);
	return 0;
}

static inline int spa_pod_is_long(const struct spa_pod *pod)
{
	return (SPA_POD_TYPE(pod) == SPA_TYPE_Long && SPA_POD_BODY_SIZE(pod) >= sizeof(int64_t));
}

static inline int spa_pod_get_long(const struct spa_pod *pod, int64_t *value)
{
	if (!spa_pod_is_long(pod))
		return -EINVAL;
	*value = SPA_POD_VALUE(struct spa_pod_long, pod);
	return 0;
}

static inline int spa_pod_is_float(const struct spa_pod *pod)
{
	return (SPA_POD_TYPE(pod) == SPA_TYPE_Float && SPA_POD_BODY_SIZE(pod) >= sizeof(float));
}

static inline int spa_pod_get_float(const struct spa_pod *pod, float *value)
{
	if (!spa_pod_is_float(pod))
		return -EINVAL;
	*value = SPA_POD_VALUE(struct spa_pod_float, pod);
	return 0;
}

static inline int spa_pod_is_double(const struct spa_pod *pod)
{
	return (SPA_POD_TYPE(pod) == SPA_TYPE_Double && SPA_POD_BODY_SIZE(pod) >= sizeof(double));
}

static inline int spa_pod_get_double(const struct spa_pod *pod, double *value)
{
	if (!spa_pod_is_double(pod))
		return -EINVAL;
	*value = SPA_POD_VALUE(struct spa_pod_double, pod);
	return 0;
}

static inline int spa_pod_is_string(const struct spa_pod *pod)
{
	const char *s = (const char *)SPA_POD_CONTENTS(struct spa_pod_string, pod);
	return (SPA_POD_TYPE(pod) == SPA_TYPE_String &&
			SPA_POD_BODY_SIZE(pod) > 0 &&
			s[SPA_POD_BODY_SIZE(pod)-1] == '\0');
}

static inline int spa_pod_get_string(const struct spa_pod *pod, const char **value)
{
	if (!spa_pod_is_string(pod))
		return -EINVAL;
	*value = (const char *)SPA_POD_CONTENTS(struct spa_pod_string, pod);
	return 0;
}

static inline int spa_pod_copy_string(const struct spa_pod *pod, size_t maxlen, char *dest)
{
	const char *s = (const char *)SPA_POD_CONTENTS(struct spa_pod_string, pod);
	if (!spa_pod_is_string(pod) || maxlen < 1)
		return -EINVAL;
	strncpy(dest, s, maxlen-1);
	dest[maxlen-1]= '\0';
	return 0;
}

static inline int spa_pod_is_bytes(const struct spa_pod *pod)
{
	return SPA_POD_TYPE(pod) == SPA_TYPE_Bytes;
}

static inline int spa_pod_get_bytes(const struct spa_pod *pod, const void **value, uint32_t *len)
{
	if (!spa_pod_is_bytes(pod))
		return -EINVAL;
	*value = (const void *)SPA_POD_CONTENTS(struct spa_pod_bytes, pod);
	*len = SPA_POD_BODY_SIZE(pod);
	return 0;
}

static inline int spa_pod_is_pointer(const struct spa_pod *pod)
{
	return (SPA_POD_TYPE(pod) == SPA_TYPE_Pointer &&
			SPA_POD_BODY_SIZE(pod) >= sizeof(struct spa_pod_pointer_body));
}

static inline int spa_pod_get_pointer(const struct spa_pod *pod, uint32_t *type, const void **value)
{
	if (!spa_pod_is_pointer(pod))
		return -EINVAL;
	*type = ((struct spa_pod_pointer*)pod)->body.type;
	*value = ((struct spa_pod_pointer*)pod)->body.value;
	return 0;
}

static inline int spa_pod_is_fd(const struct spa_pod *pod)
{
	return (SPA_POD_TYPE(pod) == SPA_TYPE_Fd &&
			SPA_POD_BODY_SIZE(pod) >= sizeof(int64_t));
}

static inline int spa_pod_get_fd(const struct spa_pod *pod, int64_t *value)
{
	if (!spa_pod_is_fd(pod))
		return -EINVAL;
	*value = SPA_POD_VALUE(struct spa_pod_fd, pod);
	return 0;
}

static inline int spa_pod_is_rectangle(const struct spa_pod *pod)
{
	return (SPA_POD_TYPE(pod) == SPA_TYPE_Rectangle &&
			SPA_POD_BODY_SIZE(pod) >= sizeof(struct spa_rectangle));
}

static inline int spa_pod_get_rectangle(const struct spa_pod *pod, struct spa_rectangle *value)
{
	if (!spa_pod_is_rectangle(pod))
		return -EINVAL;
	*value = SPA_POD_VALUE(struct spa_pod_rectangle, pod);
	return 0;
}

static inline int spa_pod_is_fraction(const struct spa_pod *pod)
{
	return (SPA_POD_TYPE(pod) == SPA_TYPE_Fraction &&
			SPA_POD_BODY_SIZE(pod) >= sizeof(struct spa_fraction));
}

static inline int spa_pod_get_fraction(const struct spa_pod *pod, struct spa_fraction *value)
{
	spa_return_val_if_fail(spa_pod_is_fraction(pod), -EINVAL);
	*value = SPA_POD_VALUE(struct spa_pod_fraction, pod);
	return 0;
}

static inline int spa_pod_is_bitmap(const struct spa_pod *pod)
{
	return (SPA_POD_TYPE(pod) == SPA_TYPE_Bitmap &&
			SPA_POD_BODY_SIZE(pod) >= sizeof(uint8_t));
}

static inline int spa_pod_is_array(const struct spa_pod *pod)
{
	return (SPA_POD_TYPE(pod) == SPA_TYPE_Array &&
			SPA_POD_BODY_SIZE(pod) >= sizeof(struct spa_pod_array_body));
}

static inline void *spa_pod_get_array(const struct spa_pod *pod, uint32_t *n_values)
{
	spa_return_val_if_fail(spa_pod_is_array(pod), NULL);
	*n_values = SPA_POD_ARRAY_N_VALUES(pod);
	return SPA_POD_ARRAY_VALUES(pod);
}

static inline uint32_t spa_pod_copy_array(const struct spa_pod *pod, uint32_t type,
		void *values, uint32_t max_values)
{
	uint32_t n_values;
	void *v = spa_pod_get_array(pod, &n_values);
	if (v == NULL || max_values == 0 || SPA_POD_ARRAY_VALUE_TYPE(pod) != type)
		return 0;
	n_values = SPA_MIN(n_values, max_values);
	memcpy(values, v, SPA_POD_ARRAY_VALUE_SIZE(pod) * n_values);
	return n_values;
}

static inline int spa_pod_is_choice(const struct spa_pod *pod)
{
	return (SPA_POD_TYPE(pod) == SPA_TYPE_Choice &&
			SPA_POD_BODY_SIZE(pod) >= sizeof(struct spa_pod_choice_body));
}

static inline struct spa_pod *spa_pod_get_values(const struct spa_pod *pod, uint32_t *n_vals, uint32_t *choice)
{
	if (pod->type == SPA_TYPE_Choice) {
		*n_vals = SPA_POD_CHOICE_N_VALUES(pod);
		if ((*choice = SPA_POD_CHOICE_TYPE(pod)) == SPA_CHOICE_None)
			*n_vals = SPA_MIN(1u, SPA_POD_CHOICE_N_VALUES(pod));
		return (struct spa_pod*)SPA_POD_CHOICE_CHILD(pod);
	} else {
		*n_vals = 1;
		*choice = SPA_CHOICE_None;
		return (struct spa_pod*)pod;
	}
}

static inline int spa_pod_is_struct(const struct spa_pod *pod)
{
	return (SPA_POD_TYPE(pod) == SPA_TYPE_Struct);
}

static inline int spa_pod_is_object(const struct spa_pod *pod)
{
	return (SPA_POD_TYPE(pod) == SPA_TYPE_Object &&
			SPA_POD_BODY_SIZE(pod) >= sizeof(struct spa_pod_object_body));
}

static inline bool spa_pod_is_object_type(const struct spa_pod *pod, uint32_t type)
{
	return (pod && spa_pod_is_object(pod) && SPA_POD_OBJECT_TYPE(pod) == type);
}

static inline bool spa_pod_is_object_id(const struct spa_pod *pod, uint32_t id)
{
	return (pod && spa_pod_is_object(pod) && SPA_POD_OBJECT_ID(pod) == id);
}

static inline int spa_pod_is_sequence(const struct spa_pod *pod)
{
	return (SPA_POD_TYPE(pod) == SPA_TYPE_Sequence &&
			SPA_POD_BODY_SIZE(pod) >= sizeof(struct spa_pod_sequence_body));
}

static inline const struct spa_pod_prop *spa_pod_object_find_prop(const struct spa_pod_object *pod,
		const struct spa_pod_prop *start, uint32_t key)
{
	const struct spa_pod_prop *first, *res;

	first = spa_pod_prop_first(&pod->body);
	start = start ? spa_pod_prop_next(start) : first;

	for (res = start; spa_pod_prop_is_inside(&pod->body, pod->pod.size, res);
	     res = spa_pod_prop_next(res)) {
		if (res->key == key)
			return res;
	}
	for (res = first; res != start; res = spa_pod_prop_next(res)) {
		if (res->key == key)
			return res;
	}
	return NULL;
}

static inline const struct spa_pod_prop *spa_pod_find_prop(const struct spa_pod *pod,
		const struct spa_pod_prop *start, uint32_t key)
{
	if (!spa_pod_is_object(pod))
		return NULL;
	return spa_pod_object_find_prop((const struct spa_pod_object *)pod, start, key);
}

static inline int spa_pod_object_fixate(struct spa_pod_object *pod)
{
	struct spa_pod_prop *res;
	SPA_POD_OBJECT_FOREACH(pod, res) {
		if (res->value.type == SPA_TYPE_Choice &&
		    !SPA_FLAG_IS_SET(res->flags, SPA_POD_PROP_FLAG_DONT_FIXATE))
			((struct spa_pod_choice*)&res->value)->body.type = SPA_CHOICE_None;
	}
	return 0;
}

static inline int spa_pod_fixate(struct spa_pod *pod)
{
	if (!spa_pod_is_object(pod))
		return -EINVAL;
	return spa_pod_object_fixate((struct spa_pod_object *)pod);
}

static inline int spa_pod_object_is_fixated(const struct spa_pod_object *pod)
{
	struct spa_pod_prop *res;
	SPA_POD_OBJECT_FOREACH(pod, res) {
		if (res->value.type == SPA_TYPE_Choice &&
		   ((struct spa_pod_choice*)&res->value)->body.type != SPA_CHOICE_None)
			return 0;
	}
	return 1;
}

static inline int spa_pod_is_fixated(const struct spa_pod *pod)
{
	if (!spa_pod_is_object(pod))
		return -EINVAL;
	return spa_pod_object_is_fixated((const struct spa_pod_object *)pod);
}

/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SPA_POD_H */
