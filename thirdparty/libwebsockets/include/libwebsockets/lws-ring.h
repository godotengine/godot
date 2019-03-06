/*
 * libwebsockets - small server side websockets and web server implementation
 *
 * Copyright (C) 2010-2018 Andy Green <andy@warmcat.com>
 *
 *  This library is free software; you can redistribute it and/or
 *  modify it under the terms of the GNU Lesser General Public
 *  License as published by the Free Software Foundation:
 *  version 2.1 of the License.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *  Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public
 *  License along with this library; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 *  MA  02110-1301  USA
 *
 * included from libwebsockets.h
 */

/** \defgroup lws_ring LWS Ringbuffer APIs
 * ##lws_ring: generic ringbuffer struct
 *
 * Provides an abstract ringbuffer api supporting one head and one or an
 * unlimited number of tails.
 *
 * All of the members are opaque and manipulated by lws_ring_...() apis.
 *
 * The lws_ring and its buffer is allocated at runtime on the heap, using
 *
 *  - lws_ring_create()
 *  - lws_ring_destroy()
 *
 * It may contain any type, the size of the "element" stored in the ring
 * buffer and the number of elements is given at creation time.
 *
 * When you create the ringbuffer, you can optionally provide an element
 * destroy callback that frees any allocations inside the element.  This is then
 * automatically called for elements with no tail behind them, ie, elements
 * which don't have any pending consumer are auto-freed.
 *
 * Whole elements may be inserted into the ringbuffer and removed from it, using
 *
 *  - lws_ring_insert()
 *  - lws_ring_consume()
 *
 * You can find out how many whole elements are free or waiting using
 *
 *  - lws_ring_get_count_free_elements()
 *  - lws_ring_get_count_waiting_elements()
 *
 * In addition there are special purpose optional byte-centric apis
 *
 *  - lws_ring_next_linear_insert_range()
 *  - lws_ring_bump_head()
 *
 *  which let you, eg, read() directly into the ringbuffer without needing
 *  an intermediate bounce buffer.
 *
 *  The accessors understand that the ring wraps, and optimizes insertion and
 *  consumption into one or two memcpy()s depending on if the head or tail
 *  wraps.
 *
 *  lws_ring only supports a single head, but optionally multiple tails with
 *  an API to inform it when the "oldest" tail has moved on.  You can give
 *  NULL where-ever an api asks for a tail pointer, and it will use an internal
 *  single tail pointer for convenience.
 *
 *  The "oldest tail", which is the only tail if you give it NULL instead of
 *  some other tail, is used to track which elements in the ringbuffer are
 *  still unread by anyone.
 *
 *   - lws_ring_update_oldest_tail()
 */
///@{
struct lws_ring;

/**
 * lws_ring_create(): create a new ringbuffer
 *
 * \param element_len: the size in bytes of one element in the ringbuffer
 * \param count: the number of elements the ringbuffer can contain
 * \param destroy_element: NULL, or callback to be called for each element
 *			   that is removed from the ringbuffer due to the
 *			   oldest tail moving beyond it
 *
 * Creates the ringbuffer and allocates the storage.  Returns the new
 * lws_ring *, or NULL if the allocation failed.
 *
 * If non-NULL, destroy_element will get called back for every element that is
 * retired from the ringbuffer after the oldest tail has gone past it, and for
 * any element still left in the ringbuffer when it is destroyed.  It replaces
 * all other element destruction code in your user code.
 */
LWS_VISIBLE LWS_EXTERN struct lws_ring *
lws_ring_create(size_t element_len, size_t count,
		void (*destroy_element)(void *element));

/**
 * lws_ring_destroy():  destroy a previously created ringbuffer
 *
 * \param ring: the struct lws_ring to destroy
 *
 * Destroys the ringbuffer allocation and the struct lws_ring itself.
 */
LWS_VISIBLE LWS_EXTERN void
lws_ring_destroy(struct lws_ring *ring);

/**
 * lws_ring_get_count_free_elements():  return how many elements can fit
 *				      in the free space
 *
 * \param ring: the struct lws_ring to report on
 *
 * Returns how much room is left in the ringbuffer for whole element insertion.
 */
LWS_VISIBLE LWS_EXTERN size_t
lws_ring_get_count_free_elements(struct lws_ring *ring);

/**
 * lws_ring_get_count_waiting_elements():  return how many elements can be consumed
 *
 * \param ring: the struct lws_ring to report on
 * \param tail: a pointer to the tail struct to use, or NULL for single tail
 *
 * Returns how many elements are waiting to be consumed from the perspective
 * of the tail pointer given.
 */
LWS_VISIBLE LWS_EXTERN size_t
lws_ring_get_count_waiting_elements(struct lws_ring *ring, uint32_t *tail);

/**
 * lws_ring_insert():  attempt to insert up to max_count elements from src
 *
 * \param ring: the struct lws_ring to report on
 * \param src: the array of elements to be inserted
 * \param max_count: the number of available elements at src
 *
 * Attempts to insert as many of the elements at src as possible, up to the
 * maximum max_count.  Returns the number of elements actually inserted.
 */
LWS_VISIBLE LWS_EXTERN size_t
lws_ring_insert(struct lws_ring *ring, const void *src, size_t max_count);

/**
 * lws_ring_consume():  attempt to copy out and remove up to max_count elements
 *		        to src
 *
 * \param ring: the struct lws_ring to report on
 * \param tail: a pointer to the tail struct to use, or NULL for single tail
 * \param dest: the array of elements to be inserted. or NULL for no copy
 * \param max_count: the number of available elements at src
 *
 * Attempts to copy out as many waiting elements as possible into dest, from
 * the perspective of the given tail, up to max_count.  If dest is NULL, the
 * copying out is not done but the elements are logically consumed as usual.
 * NULL dest is useful in combination with lws_ring_get_element(), where you
 * can use the element direct from the ringbuffer and then call this with NULL
 * dest to logically consume it.
 *
 * Increments the tail position according to how many elements could be
 * consumed.
 *
 * Returns the number of elements consumed.
 */
LWS_VISIBLE LWS_EXTERN size_t
lws_ring_consume(struct lws_ring *ring, uint32_t *tail, void *dest,
		 size_t max_count);

/**
 * lws_ring_get_element():  get a pointer to the next waiting element for tail
 *
 * \param ring: the struct lws_ring to report on
 * \param tail: a pointer to the tail struct to use, or NULL for single tail
 *
 * Points to the next element that tail would consume, directly in the
 * ringbuffer.  This lets you write() or otherwise use the element without
 * having to copy it out somewhere first.
 *
 * After calling this, you must call lws_ring_consume(ring, &tail, NULL, 1)
 * which will logically consume the element you used up and increment your
 * tail (tail may also be NULL there if you use a single tail).
 *
 * Returns NULL if no waiting element, or a const void * pointing to it.
 */
LWS_VISIBLE LWS_EXTERN const void *
lws_ring_get_element(struct lws_ring *ring, uint32_t *tail);

/**
 * lws_ring_update_oldest_tail():  free up elements older than tail for reuse
 *
 * \param ring: the struct lws_ring to report on
 * \param tail: a pointer to the tail struct to use, or NULL for single tail
 *
 * If you are using multiple tails, you must use this API to inform the
 * lws_ring when none of the tails still need elements in the fifo any more,
 * by updating it when the "oldest" tail has moved on.
 */
LWS_VISIBLE LWS_EXTERN void
lws_ring_update_oldest_tail(struct lws_ring *ring, uint32_t tail);

/**
 * lws_ring_get_oldest_tail():  get current oldest available data index
 *
 * \param ring: the struct lws_ring to report on
 *
 * If you are initializing a new ringbuffer consumer, you can set its tail to
 * this to start it from the oldest ringbuffer entry still available.
 */
LWS_VISIBLE LWS_EXTERN uint32_t
lws_ring_get_oldest_tail(struct lws_ring *ring);

/**
 * lws_ring_next_linear_insert_range():  used to write directly into the ring
 *
 * \param ring: the struct lws_ring to report on
 * \param start: pointer to a void * set to the start of the next ringbuffer area
 * \param bytes: pointer to a size_t set to the max length you may use from *start
 *
 * This provides a low-level, bytewise access directly into the ringbuffer
 * allowing direct insertion of data without having to use a bounce buffer.
 *
 * The api reports the position and length of the next linear range that can
 * be written in the ringbuffer, ie, up to the point it would wrap, and sets
 * *start and *bytes accordingly.  You can then, eg, directly read() into
 * *start for up to *bytes, and use lws_ring_bump_head() to update the lws_ring
 * with what you have done.
 *
 * Returns nonzero if no insertion is currently possible.
 */
LWS_VISIBLE LWS_EXTERN int
lws_ring_next_linear_insert_range(struct lws_ring *ring, void **start,
				  size_t *bytes);

/**
 * lws_ring_bump_head():  used to write directly into the ring
 *
 * \param ring: the struct lws_ring to operate on
 * \param bytes: the number of bytes you inserted at the current head
 */
LWS_VISIBLE LWS_EXTERN void
lws_ring_bump_head(struct lws_ring *ring, size_t bytes);

LWS_VISIBLE LWS_EXTERN void
lws_ring_dump(struct lws_ring *ring, uint32_t *tail);

/*
 * This is a helper that combines the common pattern of needing to consume
 * some ringbuffer elements, move the consumer tail on, and check if that
 * has moved any ringbuffer elements out of scope, because it was the last
 * consumer that had not already consumed them.
 *
 * Elements that go out of scope because the oldest tail is now after them
 * get garbage-collected by calling the destroy_element callback on them
 * defined when the ringbuffer was created.
 */

#define lws_ring_consume_and_update_oldest_tail(\
		___ring,    /* the lws_ring object */ \
		___type,    /* type of objects with tails */ \
		___ptail,   /* ptr to tail of obj with tail doing consuming */ \
		___count,   /* count of payload objects being consumed */ \
		___list_head,	/* head of list of objects with tails */ \
		___mtail,   /* member name of tail in ___type */ \
		___mlist  /* member name of next list member ptr in ___type */ \
	) { \
		int ___n, ___m; \
	\
	___n = lws_ring_get_oldest_tail(___ring) == *(___ptail); \
	lws_ring_consume(___ring, ___ptail, NULL, ___count); \
	if (___n) { \
		uint32_t ___oldest; \
		___n = 0; \
		___oldest = *(___ptail); \
		lws_start_foreach_llp(___type **, ___ppss, ___list_head) { \
			___m = lws_ring_get_count_waiting_elements( \
					___ring, &(*___ppss)->tail); \
			if (___m >= ___n) { \
				___n = ___m; \
				___oldest = (*___ppss)->tail; \
			} \
		} lws_end_foreach_llp(___ppss, ___mlist); \
	\
		lws_ring_update_oldest_tail(___ring, ___oldest); \
	} \
}

/*
 * This does the same as the lws_ring_consume_and_update_oldest_tail()
 * helper, but for the simpler case there is only one consumer, so one
 * tail, and that tail is always the oldest tail.
 */

#define lws_ring_consume_single_tail(\
		___ring,  /* the lws_ring object */ \
		___ptail, /* ptr to tail of obj with tail doing consuming */ \
		___count  /* count of payload objects being consumed */ \
	) { \
	lws_ring_consume(___ring, ___ptail, NULL, ___count); \
	lws_ring_update_oldest_tail(___ring, *(___ptail)); \
}
///@}
