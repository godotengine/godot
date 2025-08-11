/* PipeWire */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef PIPEWIRE_MEM_H
#define PIPEWIRE_MEM_H

#include <pipewire/properties.h>

#ifdef __cplusplus
extern "C" {
#endif

/** \defgroup pw_memblock Memory Blocks
 * Memory allocation and pools.
 */

/**
 * \addtogroup pw_memblock
 * \{
 */

/** Flags passed to \ref pw_mempool_alloc() */
enum pw_memblock_flags {
	PW_MEMBLOCK_FLAG_NONE =		0,
	PW_MEMBLOCK_FLAG_READABLE =	(1 << 0),	/**< memory is readable */
	PW_MEMBLOCK_FLAG_WRITABLE =	(1 << 1),	/**< memory is writable */
	PW_MEMBLOCK_FLAG_SEAL =		(1 << 2),	/**< seal the fd */
	PW_MEMBLOCK_FLAG_MAP =		(1 << 3),	/**< mmap the fd */
	PW_MEMBLOCK_FLAG_DONT_CLOSE =	(1 << 4),	/**< don't close fd */
	PW_MEMBLOCK_FLAG_DONT_NOTIFY =	(1 << 5),	/**< don't notify events */

	PW_MEMBLOCK_FLAG_READWRITE = PW_MEMBLOCK_FLAG_READABLE | PW_MEMBLOCK_FLAG_WRITABLE,
};

enum pw_memmap_flags {
	PW_MEMMAP_FLAG_NONE =		0,
	PW_MEMMAP_FLAG_READ =		(1 << 0),	/**< map in read mode */
	PW_MEMMAP_FLAG_WRITE =		(1 << 1),	/**< map in write mode */
	PW_MEMMAP_FLAG_TWICE =		(1 << 2),	/**< map the same area twice after each other,
							  *  creating a circular ringbuffer */
	PW_MEMMAP_FLAG_PRIVATE =	(1 << 3),	/**< writes will be private */
	PW_MEMMAP_FLAG_LOCKED =		(1 << 4),	/**< lock the memory into RAM */
	PW_MEMMAP_FLAG_READWRITE = PW_MEMMAP_FLAG_READ | PW_MEMMAP_FLAG_WRITE,
};

struct pw_memchunk;

/**
 *
 * A memory pool is a collection of pw_memblocks */
struct pw_mempool {
	struct pw_properties *props;
};

/**
 * Memory block structure */
struct pw_memblock {
	struct pw_mempool *pool;	/**< owner pool */
	uint32_t id;			/**< unique id */
	int ref;			/**< refcount */
	uint32_t flags;			/**< flags for the memory block on of enum pw_memblock_flags */
	uint32_t type;			/**< type of the fd, one of enum spa_data_type */
	int fd;				/**< fd */
	uint32_t size;			/**< size of memory */
	struct pw_memmap *map;		/**< optional map when PW_MEMBLOCK_FLAG_MAP was given */
};

/** a mapped region of a pw_memblock */
struct pw_memmap {
	struct pw_memblock *block;	/**< owner memblock */
	void *ptr;			/**< mapped pointer */
	uint32_t flags;			/**< flags for the mapping on of enum pw_memmap_flags */
	uint32_t offset;		/**< offset in memblock */
	uint32_t size;			/**< size in memblock */
	uint32_t tag[5];		/**< user tag */
};

struct pw_mempool_events {
#define PW_VERSION_MEMPOOL_EVENTS	0
	uint32_t version;

	/** the pool is destroyed */
	void (*destroy) (void *data);

	/** a new memory block is added to the pool */
	void (*added) (void *data, struct pw_memblock *block);

	/** a memory block is removed from the pool */
	void (*removed) (void *data, struct pw_memblock *block);
};

/** Create a new memory pool */
struct pw_mempool *pw_mempool_new(struct pw_properties *props);

/** Listen for events */
void pw_mempool_add_listener(struct pw_mempool *pool,
                            struct spa_hook *listener,
                            const struct pw_mempool_events *events,
                            void *data);

/** Clear a pool */
void pw_mempool_clear(struct pw_mempool *pool);

/** Clear and destroy a pool */
void pw_mempool_destroy(struct pw_mempool *pool);


/** Allocate a memory block from the pool */
struct pw_memblock * pw_mempool_alloc(struct pw_mempool *pool,
		enum pw_memblock_flags flags, uint32_t type, size_t size);

/** Import a block from another pool */
struct pw_memblock * pw_mempool_import_block(struct pw_mempool *pool,
		struct pw_memblock *mem);

/** Import an fd into the pool */
struct pw_memblock * pw_mempool_import(struct pw_mempool *pool,
		enum pw_memblock_flags flags, uint32_t type, int fd);

/** Free a memblock regardless of the refcount and destroy all mappings */
void pw_memblock_free(struct pw_memblock *mem);

/** Unref a memblock */
static inline void pw_memblock_unref(struct pw_memblock *mem)
{
	if (--mem->ref == 0)
		pw_memblock_free(mem);
}

/** Remove a memblock for given \a id */
int pw_mempool_remove_id(struct pw_mempool *pool, uint32_t id);

/** Find memblock for given \a ptr */
struct pw_memblock * pw_mempool_find_ptr(struct pw_mempool *pool, const void *ptr);

/** Find memblock for given \a id */
struct pw_memblock * pw_mempool_find_id(struct pw_mempool *pool, uint32_t id);

/** Find memblock for given \a fd */
struct pw_memblock * pw_mempool_find_fd(struct pw_mempool *pool, int fd);


/** Map a region of a memory block */
struct pw_memmap * pw_memblock_map(struct pw_memblock *block,
		enum pw_memmap_flags flags, uint32_t offset, uint32_t size,
		uint32_t tag[5]);

/** Map a region of a memory block with \a id */
struct pw_memmap * pw_mempool_map_id(struct pw_mempool *pool, uint32_t id,
		enum pw_memmap_flags flags, uint32_t offset, uint32_t size,
		uint32_t tag[5]);

struct pw_memmap * pw_mempool_import_map(struct pw_mempool *pool,
		struct pw_mempool *other, void *data, uint32_t size, uint32_t tag[5]);

/** find a map with the given tag */
struct pw_memmap * pw_mempool_find_tag(struct pw_mempool *pool, uint32_t tag[5], size_t size);

/** Unmap a region */
int pw_memmap_free(struct pw_memmap *map);


/** parameters to map a memory range */
struct pw_map_range {
	uint32_t start;		/** offset in first page with start of data */
	uint32_t offset;	/** page aligned offset to map */
	uint32_t size;		/** size to map */
};

#define PW_MAP_RANGE_INIT (struct pw_map_range){ 0, }

/** Calculate parameters to mmap() memory into \a range so that
 * \a size bytes at \a offset can be mapped with mmap().  */
static inline void pw_map_range_init(struct pw_map_range *range,
				     uint32_t offset, uint32_t size,
				     uint32_t page_size)
{
	range->offset = SPA_ROUND_DOWN_N(offset, page_size);
	range->start = offset - range->offset;
	range->size = SPA_ROUND_UP_N(range->start + size, page_size);
}

/**
 * \}
 */

#ifdef __cplusplus
}
#endif

#endif /* PIPEWIRE_MEM_H */
