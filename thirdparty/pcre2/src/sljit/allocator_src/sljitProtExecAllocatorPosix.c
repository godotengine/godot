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

#define SLJIT_HAS_CHUNK_HEADER
#define SLJIT_HAS_EXECUTABLE_OFFSET

struct sljit_chunk_header {
	void *executable;
};

#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>

#ifndef O_NOATIME
#define O_NOATIME 0
#endif

/* this is a linux extension available since kernel 3.11 */
#ifndef O_TMPFILE
#define O_TMPFILE 0x404000
#endif

#ifndef _GNU_SOURCE
char *secure_getenv(const char *name);
int mkostemp(char *template, int flags);
#endif

static SLJIT_INLINE int create_tempfile(void)
{
	int fd;
	char tmp_name[256];
	size_t tmp_name_len = 0;
	char *dir;
	struct stat st;
#if defined(SLJIT_SINGLE_THREADED) && SLJIT_SINGLE_THREADED
	mode_t mode;
#endif

#ifdef HAVE_MEMFD_CREATE
	/* this is a GNU extension, make sure to use -D_GNU_SOURCE */
	fd = memfd_create("sljit", MFD_CLOEXEC);
	if (fd != -1) {
		fchmod(fd, 0);
		return fd;
	}
#endif

	dir = secure_getenv("TMPDIR");

	if (dir) {
		size_t len = strlen(dir);
		if (len > 0 && len < sizeof(tmp_name)) {
			if ((stat(dir, &st) == 0) && S_ISDIR(st.st_mode)) {
				memcpy(tmp_name, dir, len + 1);
				tmp_name_len = len;
			}
		}
	}

#ifdef P_tmpdir
	if (!tmp_name_len) {
		tmp_name_len = strlen(P_tmpdir);
		if (tmp_name_len > 0 && tmp_name_len < sizeof(tmp_name))
			strcpy(tmp_name, P_tmpdir);
	}
#endif
	if (!tmp_name_len) {
		strcpy(tmp_name, "/tmp");
		tmp_name_len = 4;
	}

	SLJIT_ASSERT(tmp_name_len > 0 && tmp_name_len < sizeof(tmp_name));

	if (tmp_name_len > 1 && tmp_name[tmp_name_len - 1] == '/')
		tmp_name[--tmp_name_len] = '\0';

	fd = open(tmp_name, O_TMPFILE | O_EXCL | O_RDWR | O_NOATIME | O_CLOEXEC, 0);
	if (fd != -1)
		return fd;

	if (tmp_name_len >= sizeof(tmp_name) - 7)
		return -1;

	strcpy(tmp_name + tmp_name_len, "/XXXXXX");
#if defined(SLJIT_SINGLE_THREADED) && SLJIT_SINGLE_THREADED
	mode = umask(0777);
#endif
	fd = mkostemp(tmp_name, O_CLOEXEC | O_NOATIME);
#if defined(SLJIT_SINGLE_THREADED) && SLJIT_SINGLE_THREADED
	umask(mode);
#else
	fchmod(fd, 0);
#endif

	if (fd == -1)
		return -1;

	if (unlink(tmp_name)) {
		close(fd);
		return -1;
	}

	return fd;
}

static SLJIT_INLINE struct sljit_chunk_header* alloc_chunk(sljit_uw size)
{
	struct sljit_chunk_header *retval;
	int fd;

	fd = create_tempfile();
	if (fd == -1)
		return NULL;

	if (ftruncate(fd, (off_t)size)) {
		close(fd);
		return NULL;
	}

	retval = (struct sljit_chunk_header *)mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

	if (retval == MAP_FAILED) {
		close(fd);
		return NULL;
	}

	retval->executable = mmap(NULL, size, PROT_READ | PROT_EXEC, MAP_SHARED, fd, 0);

	if (retval->executable == MAP_FAILED) {
		munmap((void *)retval, size);
		close(fd);
		return NULL;
	}

	close(fd);
	return retval;
}

static SLJIT_INLINE void free_chunk(void *chunk, sljit_uw size)
{
	struct sljit_chunk_header *header = ((struct sljit_chunk_header *)chunk) - 1;

	munmap(header->executable, size);
	munmap((void *)header, size);
}

#include "sljitExecAllocatorCore.c"
