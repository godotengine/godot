/**************************************************************************/
/*  file_access_android.h                                                 */
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

#ifndef FILE_ACCESS_ANDROID_H
#define FILE_ACCESS_ANDROID_H

#include "core/io/file_access.h"

#include <android/asset_manager.h>
#include <android/log.h>
#include <stdio.h>

class FileAccessAndroid : public FileAccess {
	mutable AAsset *asset = nullptr;
	mutable uint64_t len = 0;
	mutable uint64_t pos = 0;
	mutable bool eof = false;
	String absolute_path;
	String path_src;

	void _close();

public:
	static AAssetManager *asset_manager;

	virtual Error open_internal(const String &p_path, int p_mode_flags) override; // open a file
	virtual bool is_open() const override; // true when file is open

	/// returns the path for the current open file
	virtual String get_path() const override;
	/// returns the absolute path for the current open file
	virtual String get_path_absolute() const override;

	virtual void seek(uint64_t p_position) override; // seek to a given position
	virtual void seek_end(int64_t p_position = 0) override; // seek from the end of file
	virtual uint64_t get_position() const override; // get position in the file
	virtual uint64_t get_length() const override; // get size of the file

	virtual bool eof_reached() const override; // reading passed EOF

	virtual uint8_t get_8() const override; // get a byte
	virtual uint16_t get_16() const override;
	virtual uint32_t get_32() const override;
	virtual uint64_t get_64() const override;
	virtual uint64_t get_buffer(uint8_t *p_dst, uint64_t p_length) const override;

	virtual Error get_error() const override; // get last error

	virtual void flush() override;
	virtual void store_8(uint8_t p_dest) override; // store a byte
	virtual void store_16(uint16_t p_dest) override;
	virtual void store_32(uint32_t p_dest) override;
	virtual void store_64(uint64_t p_dest) override;

	virtual bool file_exists(const String &p_path) override; // return true if a file exists

	virtual uint64_t _get_modified_time(const String &p_file) override { return 0; }
	virtual BitField<FileAccess::UnixPermissionFlags> _get_unix_permissions(const String &p_file) override { return 0; }
	virtual Error _set_unix_permissions(const String &p_file, BitField<FileAccess::UnixPermissionFlags> p_permissions) override { return FAILED; }

	virtual bool _get_hidden_attribute(const String &p_file) override { return false; }
	virtual Error _set_hidden_attribute(const String &p_file, bool p_hidden) override { return ERR_UNAVAILABLE; }
	virtual bool _get_read_only_attribute(const String &p_file) override { return false; }
	virtual Error _set_read_only_attribute(const String &p_file, bool p_ro) override { return ERR_UNAVAILABLE; }

	virtual void close() override;

	~FileAccessAndroid();
};

#endif // FILE_ACCESS_ANDROID_H
