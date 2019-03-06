/*
 * libwebsockets - lws alloc chunk
 *
 * Copyright (C) 2018 Andy Green <andy@warmcat.com>
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

/** \defgroup log lwsac
 *
 * ##Allocated Chunks
 *
 * If you know you will be allocating a large, unknown number of same or
 * differently sized objects, it's certainly possible to do it with libc
 * malloc.  However the allocation cost in time and memory overhead can
 * add up, and deallocation means walking the structure of every object and
 * freeing them in turn.
 *
 * lwsac (LWS Allocated Chunks) allocates chunks intended to be larger
 * than your objects (4000 bytes by default) which you linearly allocate from
 * using lwsac_use().
 *
 * If your next request won't fit in the current chunk, a new chunk is added
 * to the chain of chunks and the allocaton done from there.  If the request
 * is larger than the chunk size, an oversize chunk is created to satisfy it.
 *
 * When you are finished with the allocations, you call lwsac_free() and
 * free all the *chunks*.  So you may have thousands of objects in the chunks,
 * but they are all destroyed with the chunks without having to deallocate them
 * one by one pointlessly.
 */
///@{

struct lwsac;
typedef unsigned char * lwsac_cached_file_t;


#define lws_list_ptr_container(P,T,M) ((T *)((char *)(P) - offsetof(T, M)))

/*
 * linked-list helper that's commonly useful to manage lists of things
 * allocated using lwsac.
 *
 * These lists point to their corresponding "next" member in the target, NOT
 * the original containing struct.  To get the containing struct, you must use
 * lws_list_ptr_container() to convert.
 *
 * It's like that because it means we no longer have to have the next pointer
 * at the start of the struct, and we can have the same struct on multiple
 * linked-lists with everything held in the struct itself.
 */
typedef void * lws_list_ptr;

/*
 * optional sorting callback called by lws_list_ptr_insert() to sort the right
 * things inside the opqaue struct being sorted / inserted on the list.
 */
typedef int (*lws_list_ptr_sort_func_t)(lws_list_ptr a, lws_list_ptr b);

#define lws_list_ptr_advance(_lp) _lp = *((void **)_lp)

/* sort may be NULL if you don't care about order */
LWS_VISIBLE LWS_EXTERN void
lws_list_ptr_insert(lws_list_ptr *phead, lws_list_ptr *add,
		    lws_list_ptr_sort_func_t sort);


/**
 * lwsac_use - allocate / use some memory from a lwsac
 *
 * \param head: pointer to the lwsac list object
 * \param ensure: the number of bytes we want to use
 * \param chunk_size: 0, or the size of the chunk to (over)allocate if
 *			what we want won't fit in the current tail chunk.  If
 *			0, the default value of 4000 is used. If ensure is
 *			larger, it is used instead.
 *
 * This also serves to init the lwsac if *head is NULL.  Basically it does
 * whatever is necessary to return you a pointer to ensure bytes of memory
 * reserved for the caller.
 *
 * Returns NULL if OOM.
 */
LWS_VISIBLE LWS_EXTERN void *
lwsac_use(struct lwsac **head, size_t ensure, size_t chunk_size);

/**
 * lwsac_free - deallocate all chunks in the lwsac and set head NULL
 *
 * \param head: pointer to the lwsac list object
 *
 * This deallocates all chunks in the lwsac, then sets *head to NULL.  All
 * lwsac_use() pointers are invalidated in one hit without individual frees.
 */
LWS_VISIBLE LWS_EXTERN void
lwsac_free(struct lwsac **head);

/*
 * Optional helpers useful for where consumers may need to defer destruction
 * until all consumers are finished with the lwsac
 */

/**
 * lwsac_detach() - destroy an lwsac unless somebody else is referencing it
 *
 * \param head: pointer to the lwsac list object
 *
 * The creator of the lwsac can all this instead of lwsac_free() when it itself
 * has finished with the lwsac, but other code may be consuming it.
 *
 * If there are no other references, the lwsac is destroyed, *head is set to
 * NULL and that's the end; however if something else has called
 * lwsac_reference() on the lwsac, it simply returns.  When lws_unreference()
 * is called and no references are left, it will be destroyed then.
 */
LWS_VISIBLE LWS_EXTERN void
lwsac_detach(struct lwsac **head);

/**
 * lwsac_reference() - increase the lwsac reference count
 *
 * \param head: pointer to the lwsac list object
 *
 * Increment the reference count on the lwsac to defer destruction.
 */
LWS_VISIBLE LWS_EXTERN void
lwsac_reference(struct lwsac *head);

/**
 * lwsac_reference() - increase the lwsac reference count
 *
 * \param head: pointer to the lwsac list object
 *
 * Decrement the reference count on the lwsac... if it reached 0 on a detached
 * lwsac then the lwsac is immediately destroyed and *head set to NULL.
 */
LWS_VISIBLE LWS_EXTERN void
lwsac_unreference(struct lwsac **head);


/* helpers to keep a file cached in memory */

LWS_VISIBLE LWS_EXTERN void
lwsac_use_cached_file_start(lwsac_cached_file_t cache);

LWS_VISIBLE LWS_EXTERN void
lwsac_use_cached_file_end(lwsac_cached_file_t *cache);

LWS_VISIBLE LWS_EXTERN void
lwsac_use_cached_file_detach(lwsac_cached_file_t *cache);

LWS_VISIBLE LWS_EXTERN int
lwsac_cached_file(const char *filepath, lwsac_cached_file_t *cache,
		  size_t *len);

/* more advanced helpers */

LWS_VISIBLE LWS_EXTERN size_t
lwsac_sizeof(void);

LWS_VISIBLE LWS_EXTERN size_t
lwsac_get_tail_pos(struct lwsac *lac);

LWS_VISIBLE LWS_EXTERN struct lwsac *
lwsac_get_next(struct lwsac *lac);

LWS_VISIBLE LWS_EXTERN size_t
lwsac_align(size_t length);

LWS_VISIBLE LWS_EXTERN void
lwsac_info(struct lwsac *head);

LWS_VISIBLE LWS_EXTERN uint64_t
lwsac_total_alloc(struct lwsac *head);

///@}
