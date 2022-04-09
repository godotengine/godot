/*
 *    Stack-less Just-In-Time compiler
 *
 *    Copyright Zoltan Herczeg (hzmester@freemail.hu). All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are
 * permitted provided that the following conditions are met:
 *
 *   1. Redistributions of source code must retain the above copyright notice, this list of
 *      conditions and the following disclaimer.
 *
 *   2. Redistributions in binary form must reproduce the above copyright notice, this list
 *      of conditions and the following disclaimer in the documentation and/or other materials
 *      provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER(S) AND CONTRIBUTORS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
 * SHALL THE COPYRIGHT HOLDER(S) OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
 * TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
   This file contains a simple executable memory allocator

   It is assumed, that executable code blocks are usually medium (or sometimes
   large) memory blocks, and the allocator is not too frequently called (less
   optimized than other allocators). Thus, using it as a generic allocator is
   not suggested.

   How does it work:
     Memory is allocated in continuous memory areas called chunks by alloc_chunk()
     Chunk format:
     [ block ][ block ] ... [ block ][ block terminator ]

   All blocks and the block terminator is started with block_header. The block
   header contains the size of the previous and the next block. These sizes
   can also contain special values.
     Block size:
       0 - The block is a free_block, with a different size member.
       1 - The block is a block terminator.
       n - The block is used at the moment, and the value contains its size.
     Previous block size:
       0 - This is the first block of the memory chunk.
       n - The size of the previous block.

   Using these size values we can go forward or backward on the block chain.
   The unused blocks are stored in a chain list pointed by free_blocks. This
   list is useful if we need to find a suitable memory area when the allocator
   is called.

   When a block is freed, the new free block is connected to its adjacent free
   blocks if possible.

     [ free block ][ used block ][ free block ]
   and "used block" is freed, the three blocks are connected together:
     [           one big free block           ]
*/

/* --------------------------------------------------------------------- */
/*  System (OS) functions                                                */
/* --------------------------------------------------------------------- */

/* 64 KByte. */
#define CHUNK_SIZE	0x10000

/*
   alloc_chunk / free_chunk :
     * allocate executable system memory chunks
     * the size is always divisible by CHUNK_SIZE
   SLJIT_ALLOCATOR_LOCK / SLJIT_ALLOCATOR_UNLOCK :
     * provided as part of sljitUtils
     * only the allocator requires this lock, sljit is fully thread safe
       as it only uses local variables
*/

#ifdef _WIN32
#define SLJIT_UPDATE_WX_FLAGS(from, to, enable_exec)

static SLJIT_INLINE void* alloc_chunk(sljit_uw size)
{
	return VirtualAlloc(NULL, size, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE);
}

static SLJIT_INLINE void free_chunk(void *chunk, sljit_uw size)
{
	SLJIT_UNUSED_ARG(size);
	VirtualFree(chunk, 0, MEM_RELEASE);
}

#else /* POSIX */

#if defined(__APPLE__) && defined(MAP_JIT)
/*
   On macOS systems, returns MAP_JIT if it is defined _and_ we're running on a
   version where it's OK to have more than one JIT block or where MAP_JIT is
   required.
   On non-macOS systems, returns MAP_JIT if it is defined.
*/
#include <TargetConditionals.h>
#if TARGET_OS_OSX
#if defined SLJIT_CONFIG_X86 && SLJIT_CONFIG_X86
#ifdef MAP_ANON
#include <sys/utsname.h>
#include <stdlib.h>

#define SLJIT_MAP_JIT	(get_map_jit_flag())

static SLJIT_INLINE int get_map_jit_flag()
{
	sljit_sw page_size;
	void *ptr;
	struct utsname name;
	static int map_jit_flag = -1;

	if (map_jit_flag < 0) {
		map_jit_flag = 0;
		uname(&name);

		/* Kernel version for 10.14.0 (Mojave) or later */
		if (atoi(name.release) >= 18) {
			page_size = get_page_alignment() + 1;
			/* Only use MAP_JIT if a hardened runtime is used */
			ptr = mmap(NULL, page_size, PROT_WRITE | PROT_EXEC,
					MAP_PRIVATE | MAP_ANON, -1, 0);

			if (ptr != MAP_FAILED)
				munmap(ptr, page_size);
			else
				map_jit_flag = MAP_JIT;
		}
	}
	return map_jit_flag;
}
#endif /* MAP_ANON */
#else /* !SLJIT_CONFIG_X86 */
#if !(defined SLJIT_CONFIG_ARM && SLJIT_CONFIG_ARM)
#error Unsupported architecture
#endif /* SLJIT_CONFIG_ARM */
#include <pthread.h>

#define SLJIT_MAP_JIT	(MAP_JIT)
#define SLJIT_UPDATE_WX_FLAGS(from, to, enable_exec) \
                        apple_update_wx_flags(enable_exec)

static SLJIT_INLINE void apple_update_wx_flags(sljit_s32 enable_exec)
{
	pthread_jit_write_protect_np(enable_exec);
}
#endif /* SLJIT_CONFIG_X86 */
#else /* !TARGET_OS_OSX */
#define SLJIT_MAP_JIT	(MAP_JIT)
#endif /* TARGET_OS_OSX */
#endif /* __APPLE__ && MAP_JIT */
#ifndef SLJIT_UPDATE_WX_FLAGS
#define SLJIT_UPDATE_WX_FLAGS(from, to, enable_exec)
#endif /* !SLJIT_UPDATE_WX_FLAGS */
#ifndef SLJIT_MAP_JIT
#define SLJIT_MAP_JIT	(0)
#endif /* !SLJIT_MAP_JIT */

static SLJIT_INLINE void* alloc_chunk(sljit_uw size)
{
	void *retval;
	int prot = PROT_READ | PROT_WRITE | PROT_EXEC;
	int flags = MAP_PRIVATE;
	int fd = -1;

#ifdef PROT_MAX
	prot |= PROT_MAX(prot);
#endif

#ifdef MAP_ANON
	flags |= MAP_ANON | SLJIT_MAP_JIT;
#else /* !MAP_ANON */
	if (SLJIT_UNLIKELY((dev_zero < 0) && open_dev_zero()))
		return NULL;

	fd = dev_zero;
#endif /* MAP_ANON */

	retval = mmap(NULL, size, prot, flags, fd, 0);
	if (retval == MAP_FAILED)
		return NULL;

	if (mprotect(retval, size, PROT_READ | PROT_WRITE | PROT_EXEC) < 0) {
		munmap(retval, size);
		return NULL;
	}

	SLJIT_UPDATE_WX_FLAGS(retval, (uint8_t *)retval + size, 0);

	return retval;
}

static SLJIT_INLINE void free_chunk(void *chunk, sljit_uw size)
{
	munmap(chunk, size);
}

#endif /* windows */

/* --------------------------------------------------------------------- */
/*  Common functions                                                     */
/* --------------------------------------------------------------------- */

#define CHUNK_MASK	(~(CHUNK_SIZE - 1))

struct block_header {
	sljit_uw size;
	sljit_uw prev_size;
};

struct free_block {
	struct block_header header;
	struct free_block *next;
	struct free_block *prev;
	sljit_uw size;
};

#define AS_BLOCK_HEADER(base, offset) \
	((struct block_header*)(((sljit_u8*)base) + offset))
#define AS_FREE_BLOCK(base, offset) \
	((struct free_block*)(((sljit_u8*)base) + offset))
#define MEM_START(base)		((void*)(((sljit_u8*)base) + sizeof(struct block_header)))
#define ALIGN_SIZE(size)	(((size) + sizeof(struct block_header) + 7) & ~7)

static struct free_block* free_blocks;
static sljit_uw allocated_size;
static sljit_uw total_size;

static SLJIT_INLINE void sljit_insert_free_block(struct free_block *free_block, sljit_uw size)
{
	free_block->header.size = 0;
	free_block->size = size;

	free_block->next = free_blocks;
	free_block->prev = NULL;
	if (free_blocks)
		free_blocks->prev = free_block;
	free_blocks = free_block;
}

static SLJIT_INLINE void sljit_remove_free_block(struct free_block *free_block)
{
	if (free_block->next)
		free_block->next->prev = free_block->prev;

	if (free_block->prev)
		free_block->prev->next = free_block->next;
	else {
		SLJIT_ASSERT(free_blocks == free_block);
		free_blocks = free_block->next;
	}
}

SLJIT_API_FUNC_ATTRIBUTE void* sljit_malloc_exec(sljit_uw size)
{
	struct block_header *header;
	struct block_header *next_header;
	struct free_block *free_block;
	sljit_uw chunk_size;

	SLJIT_ALLOCATOR_LOCK();
	if (size < (64 - sizeof(struct block_header)))
		size = (64 - sizeof(struct block_header));
	size = ALIGN_SIZE(size);

	free_block = free_blocks;
	while (free_block) {
		if (free_block->size >= size) {
			chunk_size = free_block->size;
			SLJIT_UPDATE_WX_FLAGS(NULL, NULL, 0);
			if (chunk_size > size + 64) {
				/* We just cut a block from the end of the free block. */
				chunk_size -= size;
				free_block->size = chunk_size;
				header = AS_BLOCK_HEADER(free_block, chunk_size);
				header->prev_size = chunk_size;
				AS_BLOCK_HEADER(header, size)->prev_size = size;
			}
			else {
				sljit_remove_free_block(free_block);
				header = (struct block_header*)free_block;
				size = chunk_size;
			}
			allocated_size += size;
			header->size = size;
			SLJIT_ALLOCATOR_UNLOCK();
			return MEM_START(header);
		}
		free_block = free_block->next;
	}

	chunk_size = (size + sizeof(struct block_header) + CHUNK_SIZE - 1) & CHUNK_MASK;
	header = (struct block_header*)alloc_chunk(chunk_size);
	if (!header) {
		SLJIT_ALLOCATOR_UNLOCK();
		return NULL;
	}

	chunk_size -= sizeof(struct block_header);
	total_size += chunk_size;

	header->prev_size = 0;
	if (chunk_size > size + 64) {
		/* Cut the allocated space into a free and a used block. */
		allocated_size += size;
		header->size = size;
		chunk_size -= size;

		free_block = AS_FREE_BLOCK(header, size);
		free_block->header.prev_size = size;
		sljit_insert_free_block(free_block, chunk_size);
		next_header = AS_BLOCK_HEADER(free_block, chunk_size);
	}
	else {
		/* All space belongs to this allocation. */
		allocated_size += chunk_size;
		header->size = chunk_size;
		next_header = AS_BLOCK_HEADER(header, chunk_size);
	}
	next_header->size = 1;
	next_header->prev_size = chunk_size;
	SLJIT_ALLOCATOR_UNLOCK();
	return MEM_START(header);
}

SLJIT_API_FUNC_ATTRIBUTE void sljit_free_exec(void* ptr)
{
	struct block_header *header;
	struct free_block* free_block;

	SLJIT_ALLOCATOR_LOCK();
	header = AS_BLOCK_HEADER(ptr, -(sljit_sw)sizeof(struct block_header));
	allocated_size -= header->size;

	/* Connecting free blocks together if possible. */
	SLJIT_UPDATE_WX_FLAGS(NULL, NULL, 0);

	/* If header->prev_size == 0, free_block will equal to header.
	   In this case, free_block->header.size will be > 0. */
	free_block = AS_FREE_BLOCK(header, -(sljit_sw)header->prev_size);
	if (SLJIT_UNLIKELY(!free_block->header.size)) {
		free_block->size += header->size;
		header = AS_BLOCK_HEADER(free_block, free_block->size);
		header->prev_size = free_block->size;
	}
	else {
		free_block = (struct free_block*)header;
		sljit_insert_free_block(free_block, header->size);
	}

	header = AS_BLOCK_HEADER(free_block, free_block->size);
	if (SLJIT_UNLIKELY(!header->size)) {
		free_block->size += ((struct free_block*)header)->size;
		sljit_remove_free_block((struct free_block*)header);
		header = AS_BLOCK_HEADER(free_block, free_block->size);
		header->prev_size = free_block->size;
	}

	/* The whole chunk is free. */
	if (SLJIT_UNLIKELY(!free_block->header.prev_size && header->size == 1)) {
		/* If this block is freed, we still have (allocated_size / 2) free space. */
		if (total_size - free_block->size > (allocated_size * 3 / 2)) {
			total_size -= free_block->size;
			sljit_remove_free_block(free_block);
			free_chunk(free_block, free_block->size + sizeof(struct block_header));
		}
	}

	SLJIT_UPDATE_WX_FLAGS(NULL, NULL, 1);
	SLJIT_ALLOCATOR_UNLOCK();
}

SLJIT_API_FUNC_ATTRIBUTE void sljit_free_unused_memory_exec(void)
{
	struct free_block* free_block;
	struct free_block* next_free_block;

	SLJIT_ALLOCATOR_LOCK();
	SLJIT_UPDATE_WX_FLAGS(NULL, NULL, 0);

	free_block = free_blocks;
	while (free_block) {
		next_free_block = free_block->next;
		if (!free_block->header.prev_size && 
				AS_BLOCK_HEADER(free_block, free_block->size)->size == 1) {
			total_size -= free_block->size;
			sljit_remove_free_block(free_block);
			free_chunk(free_block, free_block->size + sizeof(struct block_header));
		}
		free_block = next_free_block;
	}

	SLJIT_ASSERT((total_size && free_blocks) || (!total_size && !free_blocks));
	SLJIT_UPDATE_WX_FLAGS(NULL, NULL, 1);
	SLJIT_ALLOCATOR_UNLOCK();
}
