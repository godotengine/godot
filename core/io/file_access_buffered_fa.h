/*************************************************************************/
/*  file_access_buffered_fa.h                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef FILE_ACCESS_BUFFERED_FA_H
#define FILE_ACCESS_BUFFERED_FA_H

#include "core/io/file_access_buffered.h"

template <class T>
class FileAccessBufferedFA : public FileAccessBuffered {
	T f;

	int read_data_block(int p_offset, int p_size, uint8_t *p_dest = 0) const {
		ERR_FAIL_COND_V_MSG(!f.is_open(), -1, "Can't read data block when file is not opened.");

		((T *)&f)->seek(p_offset);

		if (p_dest) {
			f.get_buffer(p_dest, p_size);
			return p_size;

		} else {
			cache.offset = p_offset;
			cache.buffer.resize(p_size);

			// on Vector
			//uint8_t* write = cache.buffer.ptrw();
			//f.get_buffer(write.ptrw(), p_size);

			// on vector
			f.get_buffer(cache.buffer.ptrw(), p_size);

			return p_size;
		}
	}

	static FileAccess *create() {
		return memnew(FileAccessBufferedFA<T>());
	}

protected:
	virtual void _set_access_type(AccessType p_access) {
		f._set_access_type(p_access);
		FileAccessBuffered::_set_access_type(p_access);
	}

public:
	void flush() {
		f.flush();
	}

	void store_8(uint8_t p_dest) {
		f.store_8(p_dest);
	}

	void store_buffer(const uint8_t *p_src, int p_length) {
		f.store_buffer(p_src, p_length);
	}

	bool file_exists(const String &p_name) {
		return f.file_exists(p_name);
	}

	Error _open(const String &p_path, int p_mode_flags) {
		close();

		Error ret = f._open(p_path, p_mode_flags);
		if (ret != OK)
			return ret;
		//ERR_FAIL_COND_V( ret != OK, ret );

		file.size = f.get_len();
		file.offset = 0;
		file.open = true;
		file.name = p_path;
		file.access_flags = p_mode_flags;

		cache.buffer.resize(0);
		cache.offset = 0;

		return set_error(OK);
	}

	void close() {
		f.close();

		file.offset = 0;
		file.size = 0;
		file.open = false;
		file.name = "";

		cache.buffer.resize(0);
		cache.offset = 0;
		set_error(OK);
	}

	virtual uint64_t _get_modified_time(const String &p_file) {
		return f._get_modified_time(p_file);
	}

	virtual uint32_t _get_unix_permissions(const String &p_file) {
		return f._get_unix_permissions(p_file);
	}

	virtual Error _set_unix_permissions(const String &p_file, uint32_t p_permissions) {
		return f._set_unix_permissions(p_file, p_permissions);
	}

	FileAccessBufferedFA() {}
};

#endif // FILE_ACCESS_BUFFERED_FA_H
