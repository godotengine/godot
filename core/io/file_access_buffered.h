/*************************************************************************/
/*  file_access_buffered.h                                               */
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

#ifndef FILE_ACCESS_BUFFERED_H
#define FILE_ACCESS_BUFFERED_H

#include "core/os/file_access.h"

#include "core/ustring.h"

class FileAccessBuffered : public FileAccess {
public:
	enum {
		DEFAULT_CACHE_SIZE = 128 * 1024,
	};

private:
	int cache_size = DEFAULT_CACHE_SIZE;

	int cache_data_left() const;
	mutable Error last_error;

protected:
	Error set_error(Error p_error) const;

	mutable struct File {
		bool open = false;
		int size = 0;
		int offset = 0;
		String name;
		int access_flags = 0;
	} file;

	mutable struct Cache {
		Vector<uint8_t> buffer;
		int offset = 0;
	} cache;

	virtual int read_data_block(int p_offset, int p_size, uint8_t *p_dest = nullptr) const = 0;

	void set_cache_size(int p_size);
	int get_cache_size();

public:
	virtual size_t get_position() const; ///< get position in the file
	virtual size_t get_len() const; ///< get size of the file

	virtual void seek(size_t p_position); ///< seek to a given position
	virtual void seek_end(int64_t p_position = 0); ///< seek from the end of file

	virtual bool eof_reached() const;

	virtual uint8_t get_8() const;
	virtual int get_buffer(uint8_t *p_dest, int p_length) const; ///< get an array of bytes

	virtual bool is_open() const;

	virtual Error get_error() const;

	FileAccessBuffered() {}
	virtual ~FileAccessBuffered() {}
};

#endif
