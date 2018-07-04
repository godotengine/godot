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

struct chunk_header {
	void *executable;
	int fd;
};

/*
   alloc_chunk / free_chunk :
     * allocate executable system memory chunks
     * the size is always divisible by CHUNK_SIZE
   allocator_grab_lock / allocator_release_lock :
     * make the allocator thread safe
     * can be empty if the OS (or the application) does not support threading
     * only the allocator requires this lock, sljit is fully thread safe
       as it only uses local variables
*/

#include <fcntl.h>

#ifndef O_NOATIME
#define O_NOATIME 0
#endif

#ifdef __O_TMPFILE
#ifndef O_TMPFILE
#define O_TMPFILE	(__O_TMPFILE | O_DIRECTORY)
#endif
#endif

int mkostemp(char *template, int flags);
char *secure_getenv(const char *name);

static SLJIT_INLINE int create_tempfile(void)
{
	int fd;

	char tmp_name[256];
	size_t tmp_name_len;
	char *dir;
	size_t len;

#ifdef P_tmpdir
	len = (P_tmpdir != NULL) ? strlen(P_tmpdir) : 0;

	if (len > 0 && len < sizeof(tmp_name)) {
		strcpy(tmp_name, P_tmpdir);
		tmp_name_len = len;
	}
	else {
		strcpy(tmp_name, "/tmp");
		tmp_name_len = 4;
	}
#else
	strcpy(tmp_name, "/tmp");
	tmp_name_len = 4;
#endif

	dir = secure_getenv("TMPDIR");
	if (dir) {
		len = strlen(dir);
		if (len > 0 && len < sizeof(tmp_name)) {
			strcpy(tmp_name, dir);
			tmp_name_len = len;
		}
	}

	SLJIT_ASSERT(tmp_name_len > 0 && tmp_name_len < sizeof(tmp_name));

	while (tmp_name_len > 0 && tmp_name[tmp_name_len - 1] == '/') {
		tmp_name_len--;
		tmp_name[tmp_name_len] = '\0';
	}

#ifdef O_TMPFILE
	fd = open(tmp_name, O_TMPFILE | O_EXCL | O_RDWR | O_NOATIME | O_CLOEXEC, S_IRUSR | S_IWUSR);
	if (fd != -1)
		return fd;
#endif

	if (tmp_name_len + 7 >= sizeof(tmp_name))
	{
		return -1;
	}

	strcpy(tmp_name + tmp_name_len, "/XXXXXX");
	fd = mkostemp(tmp_name, O_CLOEXEC | O_NOATIME);

	if (fd == -1)
		return fd;

	if (unlink(tmp_name)) {
		close(fd);
		return -1;
	}

	return fd;
}

static SLJIT_INLINE struct chunk_header* alloc_chunk(sljit_uw size)
{
	struct chunk_header *retval;
	int fd;

	fd = create_tempfile();
	if (fd == -1)
		return NULL;

	if (ftruncate(fd, size)) {
		close(fd);
		return NULL;
	}

	retval = (struct chunk_header *)mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

	if (retval == MAP_FAILED) {
		close(fd);
		return NULL;
	}

	retval->executable = mmap(NULL, size, PROT_READ | PROT_EXEC, MAP_SHARED, fd, 0);

	if (retval->executable == MAP_FAILED) {
		munmap(retval, size);
		close(fd);
		return NULL;
	}

	retval->fd = fd;
	return retval;
}

static SLJIT_INLINE void free_chunk(void *chunk, sljit_uw size)
{
	struct chunk_header *header = ((struct chunk_header *)chunk) - 1;

	int fd = header->fd;
	munmap(header->executable, size);
	munmap(header, size);
	close(fd);
}

/* --------------------------------------------------------------------- */
/*  Common functions                                                     */
/* --------------------------------------------------------------------- */

#define CHUNK_MASK	(~(CHUNK_SIZE - 1))

struct block_header {
	sljit_uw size;
	sljit_uw prev_size;
	sljit_sw executable_offset;
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
	struct chunk_header *chunk_header;
	struct block_header *header;
	struct block_header *next_header;
	struct free_block *free_block;
	sljit_uw chunk_size;
	sljit_sw executable_offset;

	allocator_grab_lock();
	if (size < (64 - sizeof(struct block_header)))
		size = (64 - sizeof(struct block_header));
	size = ALIGN_SIZE(size);

	free_block = free_blocks;
	while (free_block) {
		if (free_block->size >= size) {
			chunk_size = free_block->size;
			if (chunk_size > size + 64) {
				/* We just cut a block from the end of the free block. */
				chunk_size -= size;
				free_block->size = chunk_size;
				header = AS_BLOCK_HEADER(free_block, chunk_size);
				header->prev_size = chunk_size;
				header->executable_offset = free_block->header.executable_offset;
				AS_BLOCK_HEADER(header, size)->prev_size = size;
			}
			else {
				sljit_remove_free_block(free_block);
				header = (struct block_header*)free_block;
				size = chunk_size;
			}
			allocated_size += size;
			header->size = size;
			allocator_release_lock();
			return MEM_START(header);
		}
		free_block = free_block->next;
	}

	chunk_size = sizeof(struct chunk_header) + sizeof(struct block_header);
	chunk_size = (chunk_size + size + CHUNK_SIZE - 1) & CHUNK_MASK;

	chunk_header = alloc_chunk(chunk_size);
	if (!chunk_header) {
		allocator_release_lock();
		return NULL;
	}

	executable_offset = (sljit_sw)((sljit_u8*)chunk_header->executable - (sljit_u8*)chunk_header);

	chunk_size -= sizeof(struct chunk_header) + sizeof(struct block_header);
	total_size += chunk_size;

	header = (struct block_header *)(chunk_header + 1);

	header->prev_size = 0;
	header->executable_offset = executable_offset;
	if (chunk_size > size + 64) {
		/* Cut the allocated space into a free and a used block. */
		allocated_size += size;
		header->size = size;
		chunk_size -= size;

		free_block = AS_FREE_BLOCK(header, size);
		free_block->header.prev_size = size;
		free_block->header.executable_offset = executable_offset;
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
	next_header->executable_offset = executable_offset;
	allocator_release_lock();
	return MEM_START(header);
}

SLJIT_API_FUNC_ATTRIBUTE void sljit_free_exec(void* ptr)
{
	struct block_header *header;
	struct free_block* free_block;

	allocator_grab_lock();
	header = AS_BLOCK_HEADER(ptr, -(sljit_sw)sizeof(struct block_header));
	header = AS_BLOCK_HEADER(header, -header->executable_offset);
	allocated_size -= header->size;

	/* Connecting free blocks together if possible. */

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

	allocator_release_lock();
}

SLJIT_API_FUNC_ATTRIBUTE void sljit_free_unused_memory_exec(void)
{
	struct free_block* free_block;
	struct free_block* next_free_block;

	allocator_grab_lock();

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
	allocator_release_lock();
}

SLJIT_API_FUNC_ATTRIBUTE sljit_sw sljit_exec_offset(void* ptr)
{
	return ((struct block_header *)(ptr))[-1].executable_offset;
}
