/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_BUFFER_ALLOC_H
#define SPA_BUFFER_ALLOC_H

#ifdef __cplusplus
extern "C" {
#endif

#include <spa/buffer/buffer.h>

/**
 * \addtogroup spa_buffer
 * \{
 */

/** information about the buffer layout */
struct spa_buffer_alloc_info {
#define SPA_BUFFER_ALLOC_FLAG_INLINE_META	(1<<0)	/**< add metadata data in the skeleton */
#define SPA_BUFFER_ALLOC_FLAG_INLINE_CHUNK	(1<<1)	/**< add chunk data in the skeleton */
#define SPA_BUFFER_ALLOC_FLAG_INLINE_DATA	(1<<2)	/**< add buffer data to the skeleton */
#define SPA_BUFFER_ALLOC_FLAG_INLINE_ALL	0b111
#define SPA_BUFFER_ALLOC_FLAG_NO_DATA		(1<<3)	/**< don't set data pointers */
	uint32_t flags;
	uint32_t max_align;	/**< max of all alignments */
	uint32_t n_metas;
	uint32_t n_datas;
	struct spa_meta *metas;
	struct spa_data *datas;
	uint32_t *data_aligns;
	size_t skel_size;	/**< size of the struct spa_buffer and inlined meta/chunk/data */
	size_t meta_size;	/**< size of the meta if not inlined */
	size_t chunk_size;	/**< size of the chunk if not inlined */
	size_t data_size;	/**< size of the data if not inlined */
	size_t mem_size;	/**< size of the total memory if not inlined */
};

/**
 * Fill buffer allocation information
 *
 * Fill \a info with allocation information needed to allocate buffers
 * with the given number of metadata and data members.
 *
 * The required size of the skeleton (the struct spa_buffer) information
 * and the memory (for the metadata, chunk and buffer memory) will be
 * calculated.
 *
 * The flags member in \a info should be configured before calling this
 * functions.
 *
 * \param info the information to fill
 * \param n_metas the number of metadatas for the buffer
 * \param metas an array of metadata items
 * \param n_datas the number of datas for the buffer
 * \param datas an array of \a n_datas items
 * \param data_aligns \a n_datas alignments
 * \return 0 on success.
 * */
static inline int spa_buffer_alloc_fill_info(struct spa_buffer_alloc_info *info,
					     uint32_t n_metas, struct spa_meta metas[],
					     uint32_t n_datas, struct spa_data datas[],
					     uint32_t data_aligns[])
{
	size_t size, *target;
	uint32_t i;

	info->n_metas = n_metas;
	info->metas = metas;
	info->n_datas = n_datas;
	info->datas = datas;
	info->data_aligns = data_aligns;
	info->max_align = 16;
	info->mem_size = 0;
	/*
	 * The buffer skeleton is placed in memory like below and can
	 * be accessed as a regular structure.
	 *
	 *      +==============================+
	 *      | struct spa_buffer            |
	 *      |   uint32_t n_metas           | number of metas
	 *      |   uint32_t n_datas           | number of datas
	 *    +-|   struct spa_meta *metas     | pointer to array of metas
	 *   +|-|   struct spa_data *datas     | pointer to array of datas
	 *   || +------------------------------+
	 *   |+>| struct spa_meta              |
	 *   |  |   uint32_t type              | metadata
	 *   |  |   uint32_t size              | size of metadata
	 *  +|--|   void *data                 | pointer to metadata
	 *  ||  | ... <n_metas>                | more spa_meta follow
	 *  ||  +------------------------------+
	 *  |+->| struct spa_data              |
	 *  |   |   uint32_t type              | memory type
	 *  |   |   uint32_t flags             |
	 *  |   |   int fd                     | fd of shared memory block
	 *  |   |   uint32_t mapoffset         | offset in shared memory of data
	 *  |   |   uint32_t maxsize           | size of data block
	 *  | +-|   void *data                 | pointer to data
	 *  |+|-|   struct spa_chunk *chunk    | pointer to chunk
	 *  ||| | ... <n_datas>                | more spa_data follow
	 *  ||| +==============================+
	 *  VVV
	 *
	 * metadata, chunk and memory can either be placed right
	 * after the skeleton (inlined) or in a separate piece of memory.
	 *
	 *  vvv
	 *  ||| +==============================+
	 *  +-->| meta data memory             | metadata memory, 8 byte aligned
	 *   || | ... <n_metas>                |
	 *   || +------------------------------+
	 *   +->| struct spa_chunk             | memory for n_datas chunks
	 *    | |   uint32_t offset            |
	 *    | |   uint32_t size              |
	 *    | |   int32_t stride             |
	 *    | |   int32_t dummy              |
	 *    | | ... <n_datas> chunks         |
	 *    | +------------------------------+
	 *    +>| data                         | memory for n_datas data, aligned
	 *      | ... <n_datas> blocks         | according to alignments
	 *      +==============================+
	 */
	info->skel_size = sizeof(struct spa_buffer);
        info->skel_size += n_metas * sizeof(struct spa_meta);
        info->skel_size += n_datas * sizeof(struct spa_data);

	for (i = 0, size = 0; i < n_metas; i++)
		size += SPA_ROUND_UP_N(metas[i].size, 8);
	info->meta_size = size;

	if (SPA_FLAG_IS_SET(info->flags, SPA_BUFFER_ALLOC_FLAG_INLINE_META))
		target = &info->skel_size;
	else
		target = &info->mem_size;
	*target += info->meta_size;

	info->chunk_size = n_datas * sizeof(struct spa_chunk);
	if (SPA_FLAG_IS_SET(info->flags, SPA_BUFFER_ALLOC_FLAG_INLINE_CHUNK))
		target = &info->skel_size;
	else
	        target = &info->mem_size;
	*target += info->chunk_size;

	for (i = 0, size = 0; i < n_datas; i++) {
		int64_t align = data_aligns[i];
		info->max_align = SPA_MAX(info->max_align, data_aligns[i]);
		size = SPA_ROUND_UP_N(size, align);
		size += datas[i].maxsize;
	}
	info->data_size = size;

	if (!SPA_FLAG_IS_SET(info->flags, SPA_BUFFER_ALLOC_FLAG_NO_DATA) &&
	    SPA_FLAG_IS_SET(info->flags, SPA_BUFFER_ALLOC_FLAG_INLINE_DATA))
		target = &info->skel_size;
	else
		target = &info->mem_size;

	*target = SPA_ROUND_UP_N(*target, n_datas ? data_aligns[0] : 1);
	*target += info->data_size;
	*target = SPA_ROUND_UP_N(*target, info->max_align);

	return 0;
}

/**
 * Fill skeleton and data according to the allocation info
 *
 * Use the allocation info to create a struct \ref spa_buffer into
 * \a skel_mem and \a data_mem.
 *
 * Depending on the flags given when calling \ref
 * spa_buffer_alloc_fill_info(), the buffer meta, chunk and memory
 * will be referenced in either skel_mem or data_mem.
 *
 * \param info an allocation info
 * \param skel_mem memory to hold the struct \ref spa_buffer and the
 *  pointers to meta, chunk and memory.
 * \param data_mem memory to hold the meta, chunk and memory
 * \return a struct \ref spa_buffer in \a skel_mem
 */
static inline struct spa_buffer *
spa_buffer_alloc_layout(struct spa_buffer_alloc_info *info,
			void *skel_mem, void *data_mem)
{
	struct spa_buffer *b = (struct spa_buffer*)skel_mem;
	size_t size;
	uint32_t i;
	void **dp, *skel, *data;
	struct spa_chunk *cp;

	b->n_metas = info->n_metas;
	b->metas = SPA_PTROFF(b, sizeof(struct spa_buffer), struct spa_meta);
	b->n_datas = info->n_datas;
	b->datas = SPA_PTROFF(b->metas, info->n_metas * sizeof(struct spa_meta), struct spa_data);

	skel = SPA_PTROFF(b->datas, info->n_datas * sizeof(struct spa_data), void);
	data = data_mem;

	if (SPA_FLAG_IS_SET(info->flags, SPA_BUFFER_ALLOC_FLAG_INLINE_META))
		dp = &skel;
	else
		dp = &data;

	for (i = 0; i < info->n_metas; i++) {
		struct spa_meta *m = &b->metas[i];
		*m = info->metas[i];
		m->data = *dp;
		*dp = SPA_PTROFF(*dp, SPA_ROUND_UP_N(m->size, 8), void);
	}

	size = info->n_datas * sizeof(struct spa_chunk);
	if (SPA_FLAG_IS_SET(info->flags, SPA_BUFFER_ALLOC_FLAG_INLINE_CHUNK)) {
		cp = (struct spa_chunk*)skel;
		skel = SPA_PTROFF(skel, size, void);
	}
	else {
		cp = (struct spa_chunk*)data;
		data = SPA_PTROFF(data, size, void);
	}

	if (SPA_FLAG_IS_SET(info->flags, SPA_BUFFER_ALLOC_FLAG_INLINE_DATA))
		dp = &skel;
	else
		dp = &data;

	for (i = 0; i < info->n_datas; i++) {
		struct spa_data *d = &b->datas[i];

		*d = info->datas[i];
		d->chunk = &cp[i];
		if (!SPA_FLAG_IS_SET(info->flags, SPA_BUFFER_ALLOC_FLAG_NO_DATA)) {
			*dp = SPA_PTR_ALIGN(*dp, info->data_aligns[i], void);
			d->data = *dp;
			*dp = SPA_PTROFF(*dp, d->maxsize, void);
		}
	}
	return b;
}

/**
 * Layout an array of buffers
 *
 * Use the allocation info to layout the memory of an array of buffers.
 *
 * \a skel_mem should point to at least info->skel_size * \a n_buffers bytes
 * of memory.
 * \a data_mem should point to at least info->mem_size * \a n_buffers bytes
 * of memory.
 *
 * \param info the allocation info for one buffer
 * \param n_buffers the number of buffers to create
 * \param buffers a array with space to hold \a n_buffers pointers to buffers
 * \param skel_mem memory for the struct \ref spa_buffer
 * \param data_mem memory for the meta, chunk, memory of the buffer if not
 *		inlined in the skeleton.
 * \return 0 on success.
 *
 */
static inline int
spa_buffer_alloc_layout_array(struct spa_buffer_alloc_info *info,
			      uint32_t n_buffers, struct spa_buffer *buffers[],
			      void *skel_mem, void *data_mem)
{
	uint32_t i;
	for (i = 0; i < n_buffers; i++) {
		buffers[i] = spa_buffer_alloc_layout(info, skel_mem, data_mem);
		skel_mem = SPA_PTROFF(skel_mem, info->skel_size, void);
		data_mem = SPA_PTROFF(data_mem, info->mem_size, void);
        }
	return 0;
}

/**
 * Allocate an array of buffers
 *
 * Allocate \a n_buffers with the given metadata, memory and alignment
 * information.
 *
 * The buffer array, structures, data and metadata will all be allocated
 * in one block of memory with the proper requested alignment.
 *
 * \param n_buffers the number of buffers to create
 * \param flags extra flags
 * \param n_metas number of metadatas
 * \param metas \a n_metas metadata specification
 * \param n_datas number of datas
 * \param datas \a n_datas memory specification
 * \param data_aligns \a n_datas alignment specifications
 * \returns an array of \a n_buffers pointers to struct \ref spa_buffer
 *     with the given metadata, data and alignment or NULL when
 *     allocation failed.
 *
 */
static inline struct spa_buffer **
spa_buffer_alloc_array(uint32_t n_buffers, uint32_t flags,
		       uint32_t n_metas, struct spa_meta metas[],
		       uint32_t n_datas, struct spa_data datas[],
		       uint32_t data_aligns[])
{

	struct spa_buffer **buffers;
	struct spa_buffer_alloc_info info = { flags | SPA_BUFFER_ALLOC_FLAG_INLINE_ALL,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	void *skel;

	spa_buffer_alloc_fill_info(&info, n_metas, metas, n_datas, datas, data_aligns);

	buffers = (struct spa_buffer **)calloc(1, info.max_align +
			n_buffers * (sizeof(struct spa_buffer *) + info.skel_size));
	if (buffers == NULL)
		return NULL;

	skel = SPA_PTROFF(buffers, sizeof(struct spa_buffer *) * n_buffers, void);
	skel = SPA_PTR_ALIGN(skel, info.max_align, void);

	spa_buffer_alloc_layout_array(&info, n_buffers, buffers, skel, NULL);

	return buffers;
}

/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SPA_BUFFER_ALLOC_H */
