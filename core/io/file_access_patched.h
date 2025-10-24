/**************************************************************************/
/*  file_access_patched.h                                                 */
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

#include "file_access.h"
#include "file_access_memory.h"

class FileAccessPatched : public FileAccess {
	GDSOFTCLASS(FileAccessPatched, FileAccess);

	Ref<FileAccess> old_file;
	mutable Vector<uint8_t> patched_file_data;
	mutable Ref<FileAccessMemory> patched_file;
	mutable Error last_error = OK;

	Error _apply_patch() const;
	bool _try_apply_patch() const;

protected:
	virtual BitField<UnixPermissionFlags> _get_unix_permissions(const String &p_file) override { return 0; }
	virtual Error _set_unix_permissions(const String &p_file, BitField<UnixPermissionFlags> p_permissions) override { return FAILED; }

	virtual bool _get_hidden_attribute(const String &p_file) override { return false; }
	virtual Error _set_hidden_attribute(const String &p_file, bool p_hidden) override { return ERR_UNAVAILABLE; }

	virtual bool _get_read_only_attribute(const String &p_file) override { return false; }
	virtual Error _set_read_only_attribute(const String &p_file, bool p_ro) override { return ERR_UNAVAILABLE; }

	virtual uint64_t _get_modified_time(const String &p_file) override { return 0; }
	virtual uint64_t _get_access_time(const String &p_file) override { return 0; }
	virtual int64_t _get_size(const String &p_file) override { return -1; }

	virtual Error open_internal(const String &p_path, int p_mode_flags) override { return ERR_UNAVAILABLE; }

public:
	Error open_custom(const Ref<FileAccess> &p_old_file);

	virtual bool is_open() const override;

	virtual void seek(uint64_t p_position) override;
	virtual void seek_end(int64_t p_position = 0) override;

	virtual uint64_t get_position() const override;
	virtual uint64_t get_length() const override;
	virtual bool eof_reached() const override;
	virtual Error get_error() const override;

	virtual bool store_buffer(const uint8_t *p_src, uint64_t p_length) override;
	virtual uint64_t get_buffer(uint8_t *p_dst, uint64_t p_length) const override;
	virtual Error resize(int64_t p_length) override { return ERR_UNAVAILABLE; }

	virtual void flush() override;
	virtual void close() override;

	virtual bool file_exists(const String &p_name) override;
};
