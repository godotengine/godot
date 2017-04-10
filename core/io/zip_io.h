/*************************************************************************/
/*  zip_io.h                                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/
#ifndef ZIP_IO_H
#define ZIP_IO_H

#include "io/unzip.h"
#include "io/zip.h"
#include "os/copymem.h"
#include "os/file_access.h"

static void *zipio_open(void *data, const char *p_fname, int mode) {

	FileAccess *&f = *(FileAccess **)data;

	String fname;
	fname.parse_utf8(p_fname);

	if (mode & ZLIB_FILEFUNC_MODE_WRITE) {
		f = FileAccess::open(fname, FileAccess::WRITE);
	} else {

		f = FileAccess::open(fname, FileAccess::READ);
	}

	if (!f)
		return NULL;

	return data;
};

static uLong zipio_read(void *data, void *fdata, void *buf, uLong size) {

	FileAccess *f = *(FileAccess **)data;
	return f->get_buffer((uint8_t *)buf, size);
};

static uLong zipio_write(voidpf opaque, voidpf stream, const void *buf, uLong size) {

	FileAccess *f = *(FileAccess **)opaque;
	f->store_buffer((uint8_t *)buf, size);
	return size;
};

static long zipio_tell(voidpf opaque, voidpf stream) {

	FileAccess *f = *(FileAccess **)opaque;
	return f->get_pos();
};

static long zipio_seek(voidpf opaque, voidpf stream, uLong offset, int origin) {

	FileAccess *f = *(FileAccess **)opaque;

	int pos = offset;
	switch (origin) {

		case ZLIB_FILEFUNC_SEEK_CUR:
			pos = f->get_pos() + offset;
			break;
		case ZLIB_FILEFUNC_SEEK_END:
			pos = f->get_len() + offset;
			break;
		default:
			break;
	};

	f->seek(pos);
	return 0;
};

static int zipio_close(voidpf opaque, voidpf stream) {

	FileAccess *&f = *(FileAccess **)opaque;
	if (f) {
		f->close();
		f = NULL;
	}
	return 0;
};

static int zipio_testerror(voidpf opaque, voidpf stream) {

	FileAccess *f = *(FileAccess **)opaque;
	return (f && f->get_error() != OK) ? 1 : 0;
};

static voidpf zipio_alloc(voidpf opaque, uInt items, uInt size) {

	voidpf ptr = memalloc(items * size);
	zeromem(ptr, items * size);
	return ptr;
}

static void zipio_free(voidpf opaque, voidpf address) {

	memfree(address);
}

static zlib_filefunc_def zipio_create_io_from_file(FileAccess **p_file) {

	zlib_filefunc_def io;
	io.opaque = p_file;
	io.zopen_file = zipio_open;
	io.zread_file = zipio_read;
	io.zwrite_file = zipio_write;
	io.ztell_file = zipio_tell;
	io.zseek_file = zipio_seek;
	io.zclose_file = zipio_close;
	io.zerror_file = zipio_testerror;
	io.alloc_mem = zipio_alloc;
	io.free_mem = zipio_free;
	return io;
}

#endif // ZIP_IO_H
