/**************************************************************************/
/*  file_access_openharmony.cpp                                           */
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

#include "file_access_openharmony.h"

#include "dir_access_openharmony.h"
#include "os_openharmony.h"

NativeResourceManager *FileAccessOpenHarmony::resource_manager = nullptr;

void FileAccessOpenHarmony::setup(NativeResourceManager *p_resource_manager) {
	FileAccessOpenHarmony::resource_manager = p_resource_manager;
}

Error FileAccessOpenHarmony::get_rawfile_content(const char *p_path, String &r_content) {
	if (resource_manager == nullptr) {
		return ERR_FILE_NOT_FOUND;
	}
	RawFile64 *rawfile = OH_ResourceManager_OpenRawFile64(resource_manager, p_path);
	if (rawfile == nullptr) {
		return ERR_FILE_NOT_FOUND;
	}
	uint64_t length = OH_ResourceManager_GetRawFileSize64(rawfile);
	uint8_t *buffer = (uint8_t *)memalloc(length);
	uint64_t read = OH_ResourceManager_ReadRawFile64(rawfile, buffer, length);
	OH_ResourceManager_CloseRawFile64(rawfile);
	if (read != length) {
		memfree(buffer);
		return ERR_FILE_CORRUPT;
	}
	r_content = String::utf8((const char *)buffer, length);
	memfree(buffer);
	return OK;
}

bool FileAccessOpenHarmony::is_in_bundle(String p_path) {
	return p_path.begins_with(OS_OpenHarmony::get_singleton()->get_bundle_resource_dir());
}

Error FileAccessOpenHarmony::open_internal(const String &p_path, int p_mode_flags) {
	String file = fix_path(p_path);
	close();
	_cpath = "";
	_is_rawfile = false;
	if (is_in_bundle(file)) {
		if (p_mode_flags != FileAccess::READ) {
			return ERR_FILE_CANT_WRITE;
		}
		String rawfile_path = file.trim_prefix(OS_OpenHarmony::get_singleton()->get_bundle_resource_dir());
		_rawfile = OH_ResourceManager_OpenRawFile64(resource_manager, rawfile_path.utf8().get_data());
		if (_rawfile == nullptr) {
			return ERR_FILE_NOT_FOUND;
		}
		_cpath = file;
		_is_rawfile = true;
		return OK;
	}
	return FileAccessUnix::open_internal(p_path, p_mode_flags);
}

bool FileAccessOpenHarmony::is_open() const {
	if (_is_rawfile) {
		return _rawfile != nullptr;
	}
	return FileAccessUnix::is_open();
}

String FileAccessOpenHarmony::get_path() const {
	if (_is_rawfile) {
		return _cpath;
	}
	return FileAccessUnix::get_path();
}

String FileAccessOpenHarmony::get_path_absolute() const {
	if (_is_rawfile) {
		return _cpath;
	}
	return FileAccessUnix::get_path_absolute();
}

void FileAccessOpenHarmony::seek(uint64_t p_position) {
	if (_is_rawfile) {
		if (_rawfile) {
			OH_ResourceManager_SeekRawFile64(_rawfile, p_position, SEEK_SET);
		}
		return;
	}
	return FileAccessUnix::seek(p_position);
}

void FileAccessOpenHarmony::seek_end(int64_t p_position) {
	if (_is_rawfile) {
		if (_rawfile) {
			OH_ResourceManager_SeekRawFile64(_rawfile, p_position, SEEK_END);
		}
		return;
	}
	return FileAccessUnix::seek_end(p_position);
}

uint64_t FileAccessOpenHarmony::get_position() const {
	if (_is_rawfile) {
		if (_rawfile) {
			return OH_ResourceManager_GetRawFileOffset64(_rawfile);
		}
		return 0;
	}
	return FileAccessUnix::get_position();
}

uint64_t FileAccessOpenHarmony::get_length() const {
	if (_is_rawfile) {
		if (_rawfile) {
			return OH_ResourceManager_GetRawFileSize64(_rawfile);
		}
		return 0;
	}
	return FileAccessUnix::get_length();
}

bool FileAccessOpenHarmony::eof_reached() const {
	if (_is_rawfile) {
		if (_rawfile) {
			return OH_ResourceManager_GetRawFileRemainingLength64(_rawfile) <= 0;
		}
		return true;
	}
	return FileAccessUnix::eof_reached();
}

uint64_t FileAccessOpenHarmony::get_buffer(uint8_t *p_dst, uint64_t p_length) const {
	if (_is_rawfile) {
		if (_rawfile) {
			return OH_ResourceManager_ReadRawFile64(_rawfile, p_dst, p_length);
		}
		return 0;
	}
	return FileAccessUnix::get_buffer(p_dst, p_length);
}

Error FileAccessOpenHarmony::get_error() const {
	if (_is_rawfile) {
		return OK;
	}
	return FileAccessUnix::get_error();
}

Error FileAccessOpenHarmony::resize(int64_t p_length) {
	if (_is_rawfile) {
		return ERR_FILE_NO_PERMISSION;
	}
	return FileAccessUnix::resize(p_length);
}

void FileAccessOpenHarmony::flush() {
	if (_is_rawfile) {
		return;
	}
	FileAccessUnix::flush();
}

bool FileAccessOpenHarmony::store_buffer(const uint8_t *p_src, uint64_t p_length) {
	if (_is_rawfile) {
		return false;
	}
	return FileAccessUnix::store_buffer(p_src, p_length);
}

bool FileAccessOpenHarmony::file_exists(const String &p_path) {
	String file = fix_path(p_path);
	if (is_in_bundle(file)) {
		Ref<DirAccess> dir_access = DirAccessOpenHarmony::create(DirAccess::AccessType::ACCESS_FILESYSTEM);
		return dir_access->file_exists(file);
	}
	return FileAccessUnix::file_exists(p_path);
}

uint64_t FileAccessOpenHarmony::_get_modified_time(const String &p_file) {
	String file = fix_path(p_file);
	if (is_in_bundle(file)) {
		return 0;
	}
	return FileAccessUnix::_get_modified_time(p_file);
}

BitField<FileAccess::UnixPermissionFlags> FileAccessOpenHarmony::_get_unix_permissions(const String &p_file) {
	String file = fix_path(p_file);
	if (is_in_bundle(file)) {
		return UNIX_READ_OWNER;
	}
	return FileAccessUnix::_get_unix_permissions(p_file);
}

Error FileAccessOpenHarmony::_set_unix_permissions(const String &p_file, BitField<FileAccess::UnixPermissionFlags> p_permissions) {
	String file = fix_path(p_file);
	if (is_in_bundle(file)) {
		return ERR_FILE_NO_PERMISSION;
	}
	return FileAccessUnix::_set_unix_permissions(p_file, p_permissions);
}

void FileAccessOpenHarmony::close() {
	if (_is_rawfile) {
		if (_rawfile) {
			OH_ResourceManager_CloseRawFile64(_rawfile);
			_rawfile = nullptr;
		}
		return;
	}
	FileAccessUnix::close();
}

FileAccessOpenHarmony::FileAccessOpenHarmony() {
}

FileAccessOpenHarmony::~FileAccessOpenHarmony() {
	close();
}
