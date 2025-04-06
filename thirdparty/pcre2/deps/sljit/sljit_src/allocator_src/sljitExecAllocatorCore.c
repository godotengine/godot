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

/* Expected functions:
     alloc_chunk / free_chunk :
       * allocate executable system memory chunks
       * the size is always divisible by CHUNK_SIZE
     SLJIT_ALLOCATOR_LOCK / SLJIT_ALLOCATOR_UNLOCK :
       * provided as part of sljitUtils
       * only the allocator requires this lock, sljit is fully thread safe
         as it only uses local variables

   Supported defines:
     SLJIT_HAS_CHUNK_HEADER - (optional) sljit_chunk_header is defined
     SLJIT_HAS_EXECUTABLE_OFFSET - (optional) has executable offset data
     SLJIT_UPDATE_WX_FLAGS - (optional) update WX flags
*/

#ifdef SLJIT_HAS_CHUNK_HEADER
#define CHUNK_HEADER_SIZE (sizeof(struct sljit_chunk_header))
#else /* !SLJIT_HAS_CHUNK_HEADER */
#define CHUNK_HEADER_SIZE 0
#endif /* SLJIT_HAS_CHUNK_HEADER */

#ifndef SLJIT_UPDATE_WX_FLAGS
#define SLJIT_UPDATE_WX_FLAGS(from, to, enable_exec)
#endif /* SLJIT_UPDATE_WX_FLAGS */

#ifndef CHUNK_SIZE
/* 64 KByte if not specified. */
#define CHUNK_SIZE	(sljit_uw)0x10000
#endif /* CHUNK_SIZE */

struct block_header {
	sljit_uw size;
	sljit_uw prev_size;
#ifdef SLJIT_HAS_EXECUTABLE_OFFSET
	sljit_sw executable_offset;
#endif /* SLJIT_HAS_EXECUTABLE_OFFSET */
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
#define MEM_START(base)		((void*)((base) + 1))
#define CHUNK_MASK		(~(CHUNK_SIZE - 1))
#define ALIGN_SIZE(size)	(((size) + sizeof(struct block_header) + 7u) & ~(sljit_uw)7)
#define CHUNK_EXTRA_SIZE	(sizeof(struct block_header) + CHUNK_HEADER_SIZE)

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

#ifdef SLJIT_HAS_CHUNK_HEADER
	struct sljit_chunk_header *chunk_header;
#else /* !SLJIT_HAS_CHUNK_HEADER */
	void *chunk_header;
#endif /* SLJIT_HAS_CHUNK_HEADER */

#ifdef SLJIT_HAS_EXECUTABLE_OFFSET
	sljit_sw executable_offset;
#endif /* SLJIT_HAS_EXECUTABLE_OFFSET */

	if (size < (64 - sizeof(struct block_header)))
		size = (64 - sizeof(struct block_header));
	size = ALIGN_SIZE(size);

	SLJIT_ALLOCATOR_LOCK();
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
#ifdef SLJIT_HAS_EXECUTABLE_OFFSET
				header->executable_offset = free_block->header.executable_offset;
#endif /* SLJIT_HAS_EXECUTABLE_OFFSET */
				AS_BLOCK_HEADER(header, size)->prev_size = size;
			} else {
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

	chunk_size = (size + CHUNK_EXTRA_SIZE + CHUNK_SIZE - 1) & CHUNK_MASK;

	chunk_header = alloc_chunk(chunk_size);
	if (!chunk_header) {
		SLJIT_ALLOCATOR_UNLOCK();
		return NULL;
	}

#ifdef SLJIT_HAS_EXECUTABLE_OFFSET
	executable_offset = (sljit_sw)((sljit_u8*)chunk_header->executable - (sljit_u8*)chunk_header);
#endif /* SLJIT_HAS_EXECUTABLE_OFFSET */

	chunk_size -= CHUNK_EXTRA_SIZE;
	total_size += chunk_size;

	header = (struct block_header*)(((sljit_u8*)chunk_header) + CHUNK_HEADER_SIZE);

	header->prev_size = 0;
#ifdef SLJIT_HAS_EXECUTABLE_OFFSET
	header->executable_offset = executable_offset;
#endif /* SLJIT_HAS_EXECUTABLE_OFFSET */

	if (chunk_size > size + 64) {
		/* Cut the allocated space into a free and a used block. */
		allocated_size += size;
		header->size = size;
		chunk_size -= size;

		free_block = AS_FREE_BLOCK(header, size);
		free_block->header.prev_size = size;
#ifdef SLJIT_HAS_EXECUTABLE_OFFSET
		free_block->header.executable_offset = executable_offset;
#endif /* SLJIT_HAS_EXECUTABLE_OFFSET */
		sljit_insert_free_block(free_block, chunk_size);
		next_header = AS_BLOCK_HEADER(free_block, chunk_size);
	} else {
		/* All space belongs to this allocation. */
		allocated_size += chunk_size;
		header->size = chunk_size;
		next_header = AS_BLOCK_HEADER(header, chunk_size);
	}
	next_header->size = 1;
	next_header->prev_size = chunk_size;
#ifdef SLJIT_HAS_EXECUTABLE_OFFSET
	next_header->executable_offset = executable_offset;
#endif /* SLJIT_HAS_EXECUTABLE_OFFSET */
	SLJIT_ALLOCATOR_UNLOCK();
	return MEM_START(header);
}

SLJIT_API_FUNC_ATTRIBUTE void sljit_free_exec(void *ptr)
{
	struct block_header *header;
	struct free_block *free_block;

	SLJIT_ALLOCATOR_LOCK();
	header = AS_BLOCK_HEADER(ptr, -(sljit_sw)sizeof(struct block_header));
#ifdef SLJIT_HAS_EXECUTABLE_OFFSET
	header = AS_BLOCK_HEADER(header, -header->executable_offset);
#endif /* SLJIT_HAS_EXECUTABLE_OFFSET */
	allocated_size -= header->size;

	SLJIT_UPDATE_WX_FLAGS(NULL, NULL, 0);

	/* Connecting free blocks together if possible. */

	/* If header->prev_size == 0, free_block will equal to header.
	   In this case, free_block->header.size will be > 0. */
	free_block = AS_FREE_BLOCK(header, -(sljit_sw)header->prev_size);
	if (SLJIT_UNLIKELY(!free_block->header.size)) {
		free_block->size += header->size;
		header = AS_BLOCK_HEADER(free_block, free_block->size);
		header->prev_size = free_block->size;
	} else {
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
			free_chunk(free_block, free_block->size + CHUNK_EXTRA_SIZE);
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
			free_chunk(free_block, free_block->size + CHUNK_EXTRA_SIZE);
		}
		free_block = next_free_block;
	}

	SLJIT_ASSERT(total_size || (!total_size && !free_blocks));
	SLJIT_UPDATE_WX_FLAGS(NULL, NULL, 1);
	SLJIT_ALLOCATOR_UNLOCK();
}

#ifdef SLJIT_HAS_EXECUTABLE_OFFSET
SLJIT_API_FUNC_ATTRIBUTE sljit_sw sljit_exec_offset(void *code)
{
	return ((struct block_header*)SLJIT_CODE_TO_PTR(code))[-1].executable_offset;
}
#endif /* SLJIT_HAS_EXECUTABLE_OFFSET */
