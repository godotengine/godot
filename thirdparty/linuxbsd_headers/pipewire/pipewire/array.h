/* PipeWire */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef PIPEWIRE_ARRAY_H
#define PIPEWIRE_ARRAY_H

#ifdef __cplusplus
extern "C" {
#endif

#include <errno.h>

#include <spa/utils/defs.h>

/** \defgroup pw_array Array
 *
 * \brief An array object
 *
 * The array is a dynamically resizable data structure that can
 * hold items of the same size.
 */

/**
 * \addtogroup pw_array
 * \{
 */
struct pw_array {
	void *data;		/**< pointer to array data */
	size_t size;		/**< length of array in bytes */
	size_t alloc;		/**< number of allocated memory in \a data */
	size_t extend;		/**< number of bytes to extend with */
};

#define PW_ARRAY_INIT(extend) ((struct pw_array) { NULL, 0, 0, (extend) })

#define pw_array_get_len_s(a,s)			((a)->size / (s))
#define pw_array_get_unchecked_s(a,idx,s,t)	SPA_PTROFF((a)->data,(idx)*(s),t)
#define pw_array_check_index_s(a,idx,s)		((idx) < pw_array_get_len_s(a,s))

/** Get the number of items of type \a t in array */
#define pw_array_get_len(a,t)			pw_array_get_len_s(a,sizeof(t))
/** Get the item with index \a idx and type \a t from array */
#define pw_array_get_unchecked(a,idx,t)		pw_array_get_unchecked_s(a,idx,sizeof(t),t)
/** Check if an item with index \a idx and type \a t exist in array */
#define pw_array_check_index(a,idx,t)		pw_array_check_index_s(a,idx,sizeof(t))

#define pw_array_first(a)	((a)->data)
#define pw_array_end(a)		SPA_PTROFF((a)->data, (a)->size, void)
#define pw_array_check(a,p)	(SPA_PTROFF(p,sizeof(*(p)),void) <= pw_array_end(a))

#define pw_array_for_each(pos, array)					\
	for ((pos) = (__typeof__(pos)) pw_array_first(array);		\
	     pw_array_check(array, pos);				\
	     (pos)++)

#define pw_array_consume(pos, array)					\
	for ((pos) = (__typeof__(pos)) pw_array_first(array);		\
	     pw_array_check(array, pos);				\
	     (pos) = (__typeof__(pos)) pw_array_first(array))

#define pw_array_remove(a,p)						\
({									\
	(a)->size -= sizeof(*(p));					\
	memmove(p, SPA_PTROFF((p), sizeof(*(p)), void),			\
                SPA_PTRDIFF(pw_array_end(a),(p)));			\
})

/** Initialize the array with given extend */
static inline void pw_array_init(struct pw_array *arr, size_t extend)
{
	arr->data = NULL;
	arr->size = arr->alloc = 0;
	arr->extend = extend;
}

/** Clear the array */
static inline void pw_array_clear(struct pw_array *arr)
{
	free(arr->data);
	pw_array_init(arr, arr->extend);
}

/** Reset the array */
static inline void pw_array_reset(struct pw_array *arr)
{
	arr->size = 0;
}

/** Make sure \a size bytes can be added to the array */
static inline int pw_array_ensure_size(struct pw_array *arr, size_t size)
{
	size_t alloc, need;

	alloc = arr->alloc;
	need = arr->size + size;

	if (SPA_UNLIKELY(alloc < need)) {
		void *data;
		alloc = SPA_MAX(alloc, arr->extend);
		spa_assert(alloc != 0); /* forgot pw_array_init */
		while (alloc < need)
			alloc *= 2;
		if (SPA_UNLIKELY((data = realloc(arr->data, alloc)) == NULL))
			return -errno;
		arr->data = data;
		arr->alloc = alloc;
	}
	return 0;
}

/** Add \a ref size bytes to \a arr. A pointer to memory that can
 * hold at least \a size bytes is returned */
static inline void *pw_array_add(struct pw_array *arr, size_t size)
{
	void *p;

	if (pw_array_ensure_size(arr, size) < 0)
		return NULL;

	p = SPA_PTROFF(arr->data, arr->size, void);
	arr->size += size;

	return p;
}

/** Add \a ref size bytes to \a arr. When there is not enough memory to
 * hold \a size bytes, NULL is returned */
static inline void *pw_array_add_fixed(struct pw_array *arr, size_t size)
{
	void *p;

	if (SPA_UNLIKELY(arr->alloc < arr->size + size)) {
		errno = ENOSPC;
		return NULL;
	}

	p = SPA_PTROFF(arr->data, arr->size, void);
	arr->size += size;

	return p;
}

/** Add a pointer to array */
#define pw_array_add_ptr(a,p)					\
	*((void**) pw_array_add(a, sizeof(void*))) = (p)

/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* PIPEWIRE_ARRAY_H */
