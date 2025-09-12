/**************************************************************************/
/*  zip_reader.h                                                          */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#pragma once

#include "core/io/file_access.h"
#include "core/object/ref_counted.h"

#include "thirdparty/minizip/unzip.h"

class ZIPReader : public RefCounted {
	GDCLASS(ZIPReader, RefCounted)

	unzFile uzf = nullptr;

	Ref<FileAccess> fa;

	Vector<uint8_t> source;
	uint64_t source_cursor = 0;

	static void *_zipio_mem_open(voidpf p_opaque, const char *p_fname, int p_mode);
	static uLong _zipio_mem_read(voidpf p_opaque, voidpf p_stream, void *p_buf, uLong p_size);
	static uLong _zipio_mem_write(voidpf p_opaque, voidpf p_stream, const void *p_buf, uLong p_size);
	static long _zipio_mem_tell(voidpf p_opaque, voidpf p_stream);
	static long _zipio_mem_seek(voidpf p_opaque, voidpf p_stream, uLong p_offset, int p_origin);
	static int _zipio_mem_close(voidpf p_opaque, voidpf p_stream);
	static int _zipio_mem_testerror(voidpf p_opaque, voidpf p_stream);

protected:
	static void _bind_methods();

public:
	Error open(const String &p_path);
	Error open_buffer(const Vector<uint8_t> &p_buffer);
	Error close();

	// Operations on the ZIP archive.
	PackedStringArray get_files();
	int get_compression_level(const String &p_path, bool p_case_sensitive);

	// Operations on files within the ZIP archive.
	PackedByteArray read_file(const String &p_path, bool p_case_sensitive);
	bool file_exists(const String &p_path, bool p_case_sensitive);

	~ZIPReader();
};
