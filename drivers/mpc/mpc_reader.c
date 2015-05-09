/*
  Copyright (c) 2005-2009, The Musepack Development Team
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  * Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above
  copyright notice, this list of conditions and the following
  disclaimer in the documentation and/or other materials provided
  with the distribution.

  * Neither the name of the The Musepack Development Team nor the
  names of its contributors may be used to endorse or promote
  products derived from this software without specific prior
  written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
/// \file mpc_reader.c
/// Contains implementations for simple file-based mpc_reader
#include <mpc/reader.h>
#include "internal.h"
#include <stdio.h>
#include <string.h> // memset()

#define STDIO_MAGIC 0xF34B963C ///< Just a random safe-check value...
typedef struct mpc_reader_stdio_t {
    FILE       *p_file;
    int         file_size;
    mpc_bool_t  is_seekable;
    mpc_int32_t magic;
} mpc_reader_stdio;

/// mpc_reader callback implementations
static mpc_int32_t
read_stdio(mpc_reader *p_reader, void *ptr, mpc_int32_t size)
{
    mpc_reader_stdio *p_stdio = (mpc_reader_stdio*) p_reader->data;
    if(p_stdio->magic != STDIO_MAGIC) return MPC_STATUS_FILE;
    return (mpc_int32_t) fread(ptr, 1, size, p_stdio->p_file);
}

static mpc_bool_t
seek_stdio(mpc_reader *p_reader, mpc_int32_t offset)
{
    mpc_reader_stdio *p_stdio = (mpc_reader_stdio*) p_reader->data;
    if(p_stdio->magic != STDIO_MAGIC) return MPC_FALSE;
    return p_stdio->is_seekable ? fseek(p_stdio->p_file, offset, SEEK_SET) == 0 : MPC_FALSE;
}

static mpc_int32_t
tell_stdio(mpc_reader *p_reader)
{
    mpc_reader_stdio *p_stdio = (mpc_reader_stdio*) p_reader->data;
    if(p_stdio->magic != STDIO_MAGIC) return MPC_STATUS_FILE;
    return ftell(p_stdio->p_file);
}

static mpc_int32_t
get_size_stdio(mpc_reader *p_reader)
{
    mpc_reader_stdio *p_stdio = (mpc_reader_stdio*) p_reader->data;
    if(p_stdio->magic != STDIO_MAGIC) return MPC_STATUS_FILE;
    return p_stdio->file_size;
}

static mpc_bool_t
canseek_stdio(mpc_reader *p_reader)
{
    mpc_reader_stdio *p_stdio = (mpc_reader_stdio*) p_reader->data;
    if(p_stdio->magic != STDIO_MAGIC) return MPC_FALSE;
    return p_stdio->is_seekable;
}

mpc_status
mpc_reader_init_stdio_stream(mpc_reader * p_reader, FILE * p_file)
{
    mpc_reader tmp_reader; mpc_reader_stdio *p_stdio; int err;

    p_stdio = NULL;
    memset(&tmp_reader, 0, sizeof tmp_reader);
    p_stdio = malloc(sizeof *p_stdio);
    if(!p_stdio) return MPC_STATUS_FILE;
    memset(p_stdio, 0, sizeof *p_stdio);

    p_stdio->magic  = STDIO_MAGIC;
    p_stdio->p_file = p_file;
    p_stdio->is_seekable = MPC_TRUE;
    err = fseek(p_stdio->p_file, 0, SEEK_END);
    if(err < 0) goto clean;
    err = ftell(p_stdio->p_file);
    if(err < 0) goto clean;
    p_stdio->file_size = err;
    err = fseek(p_stdio->p_file, 0, SEEK_SET);
    if(err < 0) goto clean;

    tmp_reader.data     = p_stdio;
    tmp_reader.canseek  = canseek_stdio;
    tmp_reader.get_size = get_size_stdio;
    tmp_reader.read     = read_stdio;
    tmp_reader.seek     = seek_stdio;
    tmp_reader.tell     = tell_stdio;

    *p_reader = tmp_reader;
    return MPC_STATUS_OK;
clean:
    if(p_stdio && p_stdio->p_file)
        fclose(p_stdio->p_file);
    free(p_stdio);
    return MPC_STATUS_FILE;
}

mpc_status
mpc_reader_init_stdio(mpc_reader *p_reader, const char *filename)
{
	FILE * stream = fopen(filename, "rb");
	if (stream == NULL) return MPC_STATUS_FILE;
	return mpc_reader_init_stdio_stream(p_reader,stream);
}

void
mpc_reader_exit_stdio(mpc_reader *p_reader)
{
    mpc_reader_stdio *p_stdio = (mpc_reader_stdio*) p_reader->data;
    if(p_stdio->magic != STDIO_MAGIC) return;
    fclose(p_stdio->p_file);
    free(p_stdio);
    p_reader->data = NULL;
}

