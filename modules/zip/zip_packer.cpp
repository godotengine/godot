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

#include "core/error/error_macros.h"
#include "core/io/dir_access.h"
#include "core/io/file_access.h"
#include "core/io/zip_io.h"
#include "core/object/class_db.h"
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

	if (!p_path.get_base_dir().is_empty() && !directories.has(p_path.get_base_dir() + "/")) {
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
	String path = p_path.ends_with("/") ? p_path : p_path + "/";
	ERR_FAIL_COND_V_MSG(fa.is_null(), FAILED, "ZIPPacker must be opened before use.");
	ERR_FAIL_COND_V_MSG(directories.has(path), ERR_CANT_CREATE, vformat("Directory '%s' already exists.", path));

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
			path.utf8().get_data(),
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

	directories.insert(path);
	return OK;
}

Error ZIPPacker::compress(const PackedStringArray &p_input_paths, const String &p_output_path, int p_compression_level, ZipAppend p_append) {
	int err = UNZ_OK;
	Error gdErr = OK;

	// Open the zip file.
	ZIPPacker zip_packer = ZIPPacker();
	zip_packer.set_compression_level(p_compression_level);
	gdErr = zip_packer.open(p_output_path, p_append);
	if (gdErr != OK) {
		return gdErr;
	}

	// Create a queue of pending input paths to check.
	struct PendingPath {
		String input_path;
		String output_path;
	};
	LocalVector<PendingPath> pending_paths;
	pending_paths.reserve((int32_t)p_input_paths.size());
	for (const String &input_path : p_input_paths) {
		pending_paths.push_back({ input_path,
				input_path.trim_suffix("/").trim_suffix("\\").get_file() });
	}

	// Write each file/directory to the zip file.
	while (!pending_paths.is_empty()) {
		// Get the current file/directory to write to the zip file.
		PendingPath current_path = pending_paths[pending_paths.size() - 1];
		pending_paths.remove_at(pending_paths.size() - 1);

		// The current file/directory is a directory.
		if (DirAccess::dir_exists_absolute(current_path.input_path)) {
			// Open the current directory.
			Ref<DirAccess> current_dir = DirAccess::open(current_path.input_path, &gdErr);
			if (current_dir.is_null()) {
				return gdErr;
			}

			// Start listing the files in the current directory.
			gdErr = current_dir->list_dir_begin();
			if (gdErr != OK) {
				return gdErr;
			}

			// Add each file/directory in the current directory to the queue.
			while (true) {
				String next_in_dir = current_dir->get_next();
				if (next_in_dir.is_empty()) {
					break;
				}
				if (next_in_dir == "." || next_in_dir == "..") {
					continue;
				}
				pending_paths.push_back({ current_path.input_path.path_join(next_in_dir),
						current_path.output_path.path_join(next_in_dir) });
			}

			// Stop listing the files in the current directory.
			current_dir->list_dir_end();

			// Add a slash to the end of the directory path in the zip file.
			if (!current_path.output_path.ends_with("/") && !current_path.output_path.ends_with("\\")) {
				current_path.output_path += '/';
			}

			// Get the last modified time of the current directory.
			uint64_t last_modified_time = 0; // TODO: There is currently no DirAccess::get_modified_time method.

			// Write the current directory to the zip file.
			gdErr = zip_packer.start_file(current_path.output_path, 0644, last_modified_time);
			if (gdErr != OK) {
				return gdErr;
			}

			// Finish writing the current directory to the zip file.
			gdErr = zip_packer.close_file();
			if (gdErr != OK) {
				return gdErr;
			}
		}
		// The current file/directory is a file.
		else {
			// Open the current file.
			Ref<FileAccess> current_file = FileAccess::open(current_path.input_path, FileAccess::ModeFlags::READ, &gdErr);
			if (current_file.is_null()) {
				return gdErr;
			}

			// Get the last modified time of the current file.
			uint64_t last_modified_time = FileAccess::get_modified_time(current_path.input_path);

			// Start writing the current file to the zip file.
			gdErr = zip_packer.start_file(current_path.output_path, 0644, last_modified_time);
			if (gdErr != OK) {
				return gdErr;
			}

			// Write each chunk of the current file to the zip file.
			uint8_t buffer[64 * 1024]; // 64KiB buffer
			while (!current_file->eof_reached()) {
				// Read a chunk of the current file.
				uint64_t bytes_read = current_file->get_buffer(buffer, sizeof(buffer));

				// Reached the end of the current file.
				if (bytes_read == 0) {
					break;
				}

				// Write the current chunk to the output file in the zip file.
				err = zipWriteInFileInZip(zip_packer.zf, buffer, bytes_read);
				ERR_FAIL_COND_V(err != ZIP_OK, ERR_FILE_CANT_WRITE);
			}

			// Finish writing the current file to the zip file.
			gdErr = zip_packer.close_file();
			if (gdErr != OK) {
				return gdErr;
			}
		}
	}

	// The zip file was successfully created.
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

	ClassDB::bind_static_method("ZIPPacker", D_METHOD("compress", "input_paths", "output_path", "compression_level", "append"), &ZIPPacker::compress, DEFVAL(COMPRESSION_DEFAULT), DEFVAL(APPEND_CREATE));

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
