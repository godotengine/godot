/* PipeWire */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef PIPEWIRE_MAP_H
#define PIPEWIRE_MAP_H

#ifdef __cplusplus
extern "C" {
#endif

#include <string.h>
#include <errno.h>

#include <spa/utils/defs.h>
#include <pipewire/array.h>

/** \defgroup pw_map Map
 *
 * \brief A map that holds pointers to objects indexed by id
 *
 * The map is a sparse version of the \ref pw_array "pw_array" that manages the
 * indices of elements for the caller. Adding items with
 * pw_map_insert_new() returns the assigned index for that item; if items
 * are removed the map re-uses indices to keep the array at the minimum
 * required size.
 *
 * \code{.c}
 * struct pw_map map = PW_MAP_INIT(4);
 *
 * idx1 = pw_map_insert_new(&map, ptr1);
 * idx2 = pw_map_insert_new(&map, ptr2);
 * // the map is now [ptr1, ptr2], size 2
 * pw_map_remove(&map, idx1);
 * // the map is now [<unused>, ptr2], size 2
 * pw_map_insert_new(&map, ptr3);
 * // the map is now [ptr3, ptr2], size 2
 * \endcode
 */

/**
 * \addtogroup pw_map
 * \{
 */

/** \private
 * An entry in the map. This is used internally only. Each element in the
 * backing pw_array is a union pw_map_item. For real items, the data pointer
 * points to the item. If an element has been removed, pw_map->free_list
 * is the index of the most recently removed item. That item contains
 * the index of the next removed item until item->next is SPA_ID_INVALID.
 *
 * The free list is prepended only, the last item to be removed will be the
 * first item to get re-used on the next insert.
 */
union pw_map_item {
	uintptr_t next;	/* next free index */
	void *data;	/* data of this item, must be an even address */
};

/** A map. This struct should be treated as opaque by the caller. */
struct pw_map {
	struct pw_array items;	/* an array with the map items */
	uint32_t free_list;	/* first free index */
};

/** \param extend the amount of bytes to grow the map with when needed */
#define PW_MAP_INIT(extend) ((struct pw_map) { PW_ARRAY_INIT(extend), SPA_ID_INVALID })

/**
 * Get the number of currently allocated elements in the map.
 * \note pw_map_get_size() returns the currently allocated number of
 * elements in the map, not the number of actually set elements.
 * \return the number of available elements before the map needs to grow
 */
#define pw_map_get_size(m)            pw_array_get_len(&(m)->items, union pw_map_item)
#define pw_map_get_item(m,id)         pw_array_get_unchecked(&(m)->items,id,union pw_map_item)
#define pw_map_item_is_free(item)     ((item)->next & 0x1)
#define pw_map_id_is_free(m,id)       (pw_map_item_is_free(pw_map_get_item(m,id)))
/** \return true if the id fits within the current map size */
#define pw_map_check_id(m,id)         ((id) < pw_map_get_size(m))
/** \return true if there is a valid item at \a id  */
#define pw_map_has_item(m,id)         (pw_map_check_id(m,id) && !pw_map_id_is_free(m, id))
#define pw_map_lookup_unchecked(m,id) pw_map_get_item(m,id)->data

/** Convert an id to a pointer that can be inserted into the map */
#define PW_MAP_ID_TO_PTR(id)          (SPA_UINT32_TO_PTR((id)<<1))
/** Convert a pointer to an id that can be retrieved from the map */
#define PW_MAP_PTR_TO_ID(p)           (SPA_PTR_TO_UINT32(p)>>1)

/** Initialize a map
 * \param map the map to initialize
 * \param size the initial size of the map
 * \param extend the amount to bytes to grow the map with when needed
 */
static inline void pw_map_init(struct pw_map *map, size_t size, size_t extend)
{
	pw_array_init(&map->items, extend * sizeof(union pw_map_item));
	pw_array_ensure_size(&map->items, size * sizeof(union pw_map_item));
	map->free_list = SPA_ID_INVALID;
}

/** Clear a map and free the data storage. All previously returned ids
 * must be treated as invalid.
 */
static inline void pw_map_clear(struct pw_map *map)
{
	pw_array_clear(&map->items);
}

/** Reset a map but keep previously allocated storage. All previously
 * returned ids must be treated as invalid.
 */
static inline void pw_map_reset(struct pw_map *map)
{
	pw_array_reset(&map->items);
	map->free_list = SPA_ID_INVALID;
}

/** Insert data in the map. This function causes the map to grow if required.
 * \param map the map to insert into
 * \param data the item to add
 * \return the id where the item was inserted or SPA_ID_INVALID when the
 *	item can not be inserted.
 */
static inline uint32_t pw_map_insert_new(struct pw_map *map, void *data)
{
	union pw_map_item *start, *item;
	uint32_t id;

	if (map->free_list != SPA_ID_INVALID) {
		start = (union pw_map_item *) map->items.data;
		item = &start[map->free_list >> 1]; /* lsb always 1, see pw_map_remove */
		map->free_list = item->next;
	} else {
		item = (union pw_map_item *) pw_array_add(&map->items, sizeof(union pw_map_item));
		if (item == NULL)
			return SPA_ID_INVALID;
		start = (union pw_map_item *) map->items.data;
	}
	item->data = data;
	id = (item - start);
	return id;
}

/** Replace the data in the map at an index.
 *
 * \param map the map to insert into
 * \param id the index to insert at, must be less or equal to pw_map_get_size()
 * \param data the data to insert
 * \return 0 on success, -ENOSPC value when the index is invalid or a negative errno
 */
static inline int pw_map_insert_at(struct pw_map *map, uint32_t id, void *data)
{
	size_t size = pw_map_get_size(map);
	union pw_map_item *item;

	if (id > size)
		return -ENOSPC;
	else if (id == size) {
		item = (union pw_map_item *) pw_array_add(&map->items, sizeof(union pw_map_item));
		if (item == NULL)
			return -errno;
	} else {
		item = pw_map_get_item(map, id);
		if (pw_map_item_is_free(item))
			return -EINVAL;
	}
	item->data = data;
	return 0;
}

/** Remove an item at index. The id may get re-used in the future.
 *
 * \param map the map to remove from
 * \param id the index to remove
 */
static inline void pw_map_remove(struct pw_map *map, uint32_t id)
{
	if (pw_map_id_is_free(map, id))
		return;

	pw_map_get_item(map, id)->next = map->free_list;
	map->free_list = (id << 1) | 1;
}

/** Find an item in the map
 * \param map the map to use
 * \param id the index to look at
 * \return the item at \a id or NULL when no such item exists
 */
static inline void *pw_map_lookup(const struct pw_map *map, uint32_t id)
{
	if (SPA_LIKELY(pw_map_check_id(map, id))) {
		union pw_map_item *item = pw_map_get_item(map, id);
		if (!pw_map_item_is_free(item))
			return item->data;
	}
	return NULL;
}

/** Iterate all map items
 * \param map the map to iterate
 * \param func the function to call for each item, the item data and \a data is
 *		passed to the function. When \a func returns a non-zero result,
 *		iteration ends and the result is returned.
 * \param data data to pass to \a func
 * \return the result of the last call to \a func or 0 when all callbacks returned 0.
 */
static inline int pw_map_for_each(const struct pw_map *map,
				  int (*func) (void *item_data, void *data), void *data)
{
	union pw_map_item *item;
	int res = 0;

	pw_array_for_each(item, &map->items) {
		if (!pw_map_item_is_free(item))
			if ((res = func(item->data, data)) != 0)
				break;
	}
	return res;
}

/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* PIPEWIRE_MAP_H */
