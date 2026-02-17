/**************************************************************************/
/*  zip_packer.cpp                                                        */
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

#include "zip_packer.h"
#include "zip_packer.compat.inc"

#include "core/io/zip_io.h"
#include "core/os/time.h"

Error ZIPPacker::open(const String &p_path, ZipAppend p_append) {
	if (fa.is_valid()) {
		close();
	}

	zlib_filefunc_def io = zipio_create_io(&fa);
	zf = zipOpen2(p_path.utf8().get_data(), p_append, nullptr, &io);
	return zf != nullptr ? OK : FAILED;
}

Error ZIPPacker::close() {
	ERR_FAIL_COND_V_MSG(fa.is_null(), FAILED, "ZIPPacker cannot be closed because it is not open.");

	Error err = zipClose(zf, nullptr) == ZIP_OK ? OK : FAILED;
	if (err == OK) {
		DEV_ASSERT(fa.is_null());
		directories.clear();
		zf = nullptr;
	}

	return err;
}

void ZIPPacker::set_compression_level(int p_compression_level) {
	ERR_FAIL_COND_MSG(p_compression_level < Z_DEFAULT_COMPRESSION || p_compression_level > Z_BEST_COMPRESSION, "Invalid compression level.");
	compression_level = p_compression_level;
}

int ZIPPacker::get_compression_level() const {
	return compression_level;
}

Error ZIPPacker::start_file(const String &p_path, BitField<FileAccess::UnixPermissionFlags> p_permissions, uint64_t p_modified_time) {
	ERR_FAIL_COND_V_MSG(fa.is_null(), FAILED, "ZIPPacker must be opened before use.");

	if (!p_path.get_base_dir().is_empty() && !directories.has(p_path.get_base_dir())) {
		add_directory(p_path.get_base_dir(), 0755, p_modified_time);
	}

	uint64_t time = p_modified_time;
	if (time == 0) {
		time = Time::get_singleton()->get_unix_time_from_system();
	}
	Dictionary tz = Time::get_singleton()->get_time_zone_from_system();
	time += tz["bias"].operator int() * 60;
	Dictionary dt = Time::get_singleton()->get_datetime_dict_from_unix_time(time);

	zip_fileinfo zipfi;
	zipfi.tmz_date.tm_year = dt["year"];
	zipfi.tmz_date.tm_mon = dt["month"].operator int() - 1; // Note: "tm" month range - 0..11, Godot month range - 1..12, https://www.cplusplus.com/reference/ctime/tm/
	zipfi.tmz_date.tm_mday = dt["day"];
	zipfi.tmz_date.tm_hour = dt["hour"];
	zipfi.tmz_date.tm_min = dt["minute"];
	zipfi.tmz_date.tm_sec = dt["second"];
	zipfi.dosDate = 0;

	// 0100000: regular file type
	// 0000644: permissions rw-r--r--
	uint32_t _mode = p_permissions;
	if (_mode == 0) {
		_mode = 0100644;
	} else {
		_mode |= 0100000;
	}
	zipfi.external_fa = (_mode << 16L) | ((_mode & 0200) ? 0 : 1); // UUUUUUUU UUUUUUUU 00000000 00ADVSHR: Unix permissions (U) + DOS read-only flag (R).
	zipfi.internal_fa = 0;

	int err = zipOpenNewFileInZip4(zf,
			p_path.utf8().get_data(),
			&zipfi,
			nullptr,
			0,
			nullptr,
			0,
			nullptr,
			Z_DEFLATED,
			compression_level,
			0,
			-MAX_WBITS,
			DEF_MEM_LEVEL,
			Z_DEFAULT_STRATEGY,
			nullptr,
			0,
			0x0314, // "version made by", 0x03 - Unix, 0x14 - ZIP specification version 2.0, required to store Unix file permissions
			1 << 11); // Bit 11 is the language encoding flag. When set, filename and comment fields must be encoded using UTF-8.
	return err == ZIP_OK ? OK : FAILED;
}

Error ZIPPacker::write_file(const Vector<uint8_t> &p_data) {
	ERR_FAIL_COND_V_MSG(fa.is_null(), FAILED, "ZIPPacker must be opened before use.");

	return zipWriteInFileInZip(zf, p_data.ptr(), p_data.size()) == ZIP_OK ? OK : FAILED;
}

Error ZIPPacker::close_file() {
	ERR_FAIL_COND_V_MSG(fa.is_null(), FAILED, "ZIPPacker must be opened before use.");

	return zipCloseFileInZip(zf) == ZIP_OK ? OK : FAILED;
}

Error ZIPPacker::add_directory(const String &p_path, BitField<FileAccess::UnixPermissionFlags> p_permissions, uint64_t p_modified_time) {
	ERR_FAIL_COND_V_MSG(fa.is_null(), FAILED, "ZIPPacker must be opened before use.");
	ERR_FAIL_COND_V_MSG(directories.has(p_path), ERR_CANT_CREATE, vformat("Directory '%s' already exists.", p_path));

	uint64_t time = p_modified_time;
	if (time == 0) {
		time = Time::get_singleton()->get_unix_time_from_system();
	}
	Dictionary tz = Time::get_singleton()->get_time_zone_from_system();
	time += tz["bias"].operator int() * 60;
	Dictionary dt = Time::get_singleton()->get_datetime_dict_from_unix_time(time);

	zip_fileinfo zipfi;
	zipfi.tmz_date.tm_year = dt["year"];
	zipfi.tmz_date.tm_mon = dt["month"].operator int() - 1; // Note: "tm" month range - 0..11, Godot month range - 1..12, https://www.cplusplus.com/reference/ctime/tm/
	zipfi.tmz_date.tm_mday = dt["day"];
	zipfi.tmz_date.tm_hour = dt["hour"];
	zipfi.tmz_date.tm_min = dt["minute"];
	zipfi.tmz_date.tm_sec = dt["second"];
	zipfi.dosDate = 0;

	// 0040000: directory file type
	// 0000755: permissions rwxr-xr-x
	uint32_t _mode = p_permissions;
	if (_mode == 0) {
		_mode = 0040755;
	} else {
		_mode |= 0040000;
	}
	zipfi.external_fa = (_mode << 16L) | 0x10 | ((_mode & 0200) ? 0 : 1); // UUUUUUUU UUUUUUUU 00000000 00ADVSHR: Unix permissions (U) + DOS directory flag (D) + DOS read-only flag (R).
	zipfi.internal_fa = 0;

	int err = zipOpenNewFileInZip4(zf,
			p_path.utf8().get_data(),
			&zipfi,
			nullptr,
			0,
			nullptr,
			0,
			nullptr,
			Z_DEFLATED,
			compression_level,
			0,
			-MAX_WBITS,
			DEF_MEM_LEVEL,
			Z_DEFAULT_STRATEGY,
			nullptr,
			0,
			0x0314, // "version made by", 0x03 - Unix, 0x14 - ZIP specification version 2.0, required to store Unix file permissions
			1 << 11); // Bit 11 is the language encoding flag. When set, filename and comment fields must be encoded using UTF-8.
	zipCloseFileInZip(zf);
	if (err != ZIP_OK) {
		return FAILED;
	}

	directories.insert(p_path);
	return OK;
}

void ZIPPacker::_bind_methods() {
	ClassDB::bind_method(D_METHOD("open", "path", "append"), &ZIPPacker::open, DEFVAL(Variant(APPEND_CREATE)));
	ClassDB::bind_method(D_METHOD("set_compression_level", "compression_level"), &ZIPPacker::set_compression_level);
	ClassDB::bind_method(D_METHOD("get_compression_level"), &ZIPPacker::get_compression_level);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "compression_level"), "set_compression_level", "get_compression_level");
	ClassDB::bind_method(D_METHOD("add_directory", "path", "permissions", "modified_time"), &ZIPPacker::add_directory, DEFVAL(0755), DEFVAL(0));
	ClassDB::bind_method(D_METHOD("start_file", "path", "permissions", "modified_time"), &ZIPPacker::start_file, DEFVAL(0644), DEFVAL(0));
	ClassDB::bind_method(D_METHOD("write_file", "data"), &ZIPPacker::write_file);
	ClassDB::bind_method(D_METHOD("close_file"), &ZIPPacker::close_file);
	ClassDB::bind_method(D_METHOD("close"), &ZIPPacker::close);

	BIND_ENUM_CONSTANT(APPEND_CREATE);
	BIND_ENUM_CONSTANT(APPEND_CREATEAFTER);
	BIND_ENUM_CONSTANT(APPEND_ADDINZIP);

	BIND_ENUM_CONSTANT(COMPRESSION_DEFAULT);
	BIND_ENUM_CONSTANT(COMPRESSION_NONE);
	BIND_ENUM_CONSTANT(COMPRESSION_FAST);
	BIND_ENUM_CONSTANT(COMPRESSION_BEST);
}

ZIPPacker::ZIPPacker() {
}

ZIPPacker::~ZIPPacker() {
	if (fa.is_valid()) {
		close();
	}
}
