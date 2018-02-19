/*
 * libopenmpt_stream_callbacks_fd.h
 * --------------------------------
 * Purpose: libopenmpt public c interface
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */

#ifndef LIBOPENMPT_STREAM_CALLBACKS_FD_H
#define LIBOPENMPT_STREAM_CALLBACKS_FD_H

#include "libopenmpt.h"

#ifdef _MSC_VER
#include <io.h>
#endif
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#ifndef _MSC_VER
#include <unistd.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* This stuff has to be in a header file because of possibly different MSVC CRTs which cause problems for fd crossing CRT boundaries. */

static size_t openmpt_stream_fd_read_func( void * stream, void * dst, size_t bytes ) {
	int fd = 0;
	#if defined(_MSC_VER)
		size_t retval = 0;
		int to_read = 0;
		int ret_read = 0;
	#else
		ssize_t retval = 0;
	#endif
	fd = (int)(uintptr_t)stream;
	if ( fd < 0 ) {
		return 0;
	}
	#if defined(_MSC_VER)
		retval = 0;
		while ( bytes > 0 ) {
			to_read = 0;
			if ( bytes < (size_t)INT_MAX ) {
				to_read = (int)bytes;
			} else {
				to_read = INT_MAX;
			}
			ret_read = _read( fd, dst, to_read );
			if ( ret_read <= 0 ) {
				return retval;
			}
			bytes -= ret_read;
			retval += ret_read;
		}
	#else
		retval = read( fd, dst, bytes );
	#endif
	if ( retval <= 0 ) {
		return 0;
	}
	return retval;
}

static openmpt_stream_callbacks openmpt_stream_get_fd_callbacks(void) {
	openmpt_stream_callbacks retval;
	memset( &retval, 0, sizeof( openmpt_stream_callbacks ) );
	retval.read = openmpt_stream_fd_read_func;
	return retval;
}

#ifdef __cplusplus
}
#endif

#endif /* LIBOPENMPT_STREAM_CALLBACKS_FD_H */

