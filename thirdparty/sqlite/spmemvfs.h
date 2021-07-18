/*
* BSD 2-Clause License
*
* Copyright 2009 Stephen Liu
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
* * Redistributions of source code must retain the above copyright notice, this
*   list of conditions and the following disclaimer.
*
* * Redistributions in binary form must reproduce the above copyright notice,
*   this list of conditions and the following disclaimer in the documentation
*   and/or other materials provided with the distribution.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef __spmemvfs_h__
#define __spmemvfs_h__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include "sqlite3.h"

#define SPMEMVFS_NAME "spmemvfs"

typedef struct spmembuffer_t {
	char *data;
	int64_t used;
	int64_t total;
} spmembuffer_t;

typedef struct spmemvfs_db_t {
	sqlite3 * handle;
	spmembuffer_t * mem;
} spmemvfs_db_t;

int spmemvfs_env_init();

void spmemvfs_env_fini();

int spmemvfs_open_db( spmemvfs_db_t * db, const char * path, spmembuffer_t * mem );

int spmemvfs_close_db( spmemvfs_db_t * db );

#ifdef __cplusplus
}
#endif

#endif

