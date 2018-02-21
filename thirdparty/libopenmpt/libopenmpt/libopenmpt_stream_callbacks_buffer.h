/*
 * libopenmpt_stream_callbacks_buffer.h
 * ------------------------------------
 * Purpose: libopenmpt public c interface
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */

#ifndef LIBOPENMPT_STREAM_CALLBACKS_BUFFER_H
#define LIBOPENMPT_STREAM_CALLBACKS_BUFFER_H

#include "libopenmpt.h"

/* The use of this header requires:

#include <libopenmpt/libopenmpt.h>
#if defined( LIBOPENMPT_STREAM_CALLBACKS_BUFFER )
#include <libopenmpt/libopenmpt_stream_callbacks_buffer.h>
#else
#error "libopenmpt too old."
#endif

*/

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct openmpt_stream_buffer {
	const void * file_data; /* or prefix data IFF prefix_size < file_size */
	int64_t file_size;
	int64_t file_pos;
	int64_t prefix_size;
	int overflow;
} openmpt_stream_buffer;

static size_t openmpt_stream_buffer_read_func( void * stream, void * dst, size_t bytes ) {
	openmpt_stream_buffer * s = (openmpt_stream_buffer*)stream;
	int64_t offset = 0;
	int64_t begpos = 0;
	int64_t endpos = 0;
	size_t valid_bytes = 0;
	if ( !s ) {
		return 0;
	}
	offset = bytes;
	begpos = s->file_pos;
	endpos = s->file_pos;
	valid_bytes = 0;
	endpos = (uint64_t)endpos + (uint64_t)offset;
	if ( ( offset > 0 ) && !( (uint64_t)endpos > (uint64_t)begpos ) ) {
		/* integer wrapped */
		return 0;
	}
	if ( bytes == 0 ) {
		return 0;
	}
	if ( begpos >= s->file_size ) {
		return 0;
	}
	if ( endpos > s->file_size ) {
		/* clip to eof */
		bytes = bytes - (size_t)( endpos - s->file_size );
		endpos = endpos - ( endpos - s->file_size );
	}
	memset( dst, 0, bytes );
	if ( begpos >= s->prefix_size ) {
		s->overflow = 1;
		valid_bytes = 0;
	} else if ( endpos > s->prefix_size ) {
		s->overflow = 1;
		valid_bytes = bytes - (size_t)( endpos - s->prefix_size );
	} else {
		valid_bytes = bytes;
	}
	memcpy( dst, (const char*)s->file_data + s->file_pos, valid_bytes );
	s->file_pos = s->file_pos + bytes;
	return bytes;
}

static int openmpt_stream_buffer_seek_func( void * stream, int64_t offset, int whence ) {
	openmpt_stream_buffer * s = (openmpt_stream_buffer*)stream;
	int result = -1;
	if ( !s ) {
		return -1;
	}
	switch ( whence ) {
		case OPENMPT_STREAM_SEEK_SET:
			if ( offset < 0 ) {
				return -1;
			}
			if ( offset > s->file_size ) {
				return -1;
			}
			s->file_pos = offset;
			result = 0;
			break;
		case OPENMPT_STREAM_SEEK_CUR:
			do {
				int64_t oldpos = s->file_pos;
				int64_t pos = s->file_pos;
				pos = (uint64_t)pos + (uint64_t)offset;
				if ( ( offset > 0 ) && !( (uint64_t)pos > (uint64_t)oldpos ) ) {
					/* integer wrapped */
					return -1;
				}
				if ( ( offset < 0 ) && !( (uint64_t)pos < (uint64_t)oldpos ) ) {
					/* integer wrapped */
					return -1;
				}
				s->file_pos = pos;
			} while(0);
			result = 0;
			break;
		case OPENMPT_STREAM_SEEK_END:
			if ( offset > 0 ) {
				return -1;
			}
			do {
				int64_t oldpos = s->file_pos;
				int64_t pos = s->file_pos;
				pos = s->file_size;
				pos = (uint64_t)pos + (uint64_t)offset;
				if ( ( offset < 0 ) && !( (uint64_t)pos < (uint64_t)oldpos ) ) {
					/* integer wrapped */
					return -1;
				}
				s->file_pos = pos;
			} while(0);
			result = 0;
			break;
	}
	return result;
}

static int64_t openmpt_stream_buffer_tell_func( void * stream ) {
	openmpt_stream_buffer * s = (openmpt_stream_buffer*)stream;
	if ( !s ) {
		return -1;
	}
	return s->file_pos;
}

static void openmpt_stream_buffer_init( openmpt_stream_buffer * buffer, const void * file_data, int64_t file_size ) {
	memset( buffer, 0, sizeof( openmpt_stream_buffer ) );
	buffer->file_data = file_data;
	buffer->file_size = file_size;
	buffer->file_pos = 0;
	buffer->prefix_size = file_size;
	buffer->overflow = 0;
}

#define openmpt_stream_buffer_init_prefix_only( buffer_, prefix_data_, prefix_size_, file_size_ ) do { \
	openmpt_stream_buffer_init( (buffer_), (prefix_data_), (file_size_) ); \
	(buffer_)->prefix_size = (prefix_size_); \
} while(0)

#define openmpt_stream_buffer_overflowed( buffer_ ) ( (buffer_)->overflow )

static openmpt_stream_callbacks openmpt_stream_get_buffer_callbacks(void) {
	openmpt_stream_callbacks retval;
	memset( &retval, 0, sizeof( openmpt_stream_callbacks ) );
	retval.read = openmpt_stream_buffer_read_func;
	retval.seek = openmpt_stream_buffer_seek_func;
	retval.tell = openmpt_stream_buffer_tell_func;
	return retval;
}

#ifdef __cplusplus
}
#endif

#endif /* LIBOPENMPT_STREAM_CALLBACKS_BUFFER_H */

