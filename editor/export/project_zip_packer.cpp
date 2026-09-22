/**************************************************************************/
/*  project_zip_packer.cpp                                                */
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

#include "project_zip_packer.h"

#include "core/config/project_settings.h"
#include "core/io/dir_access.h"
#include "core/io/file_access.h"
#include "core/io/zip_io.h"
#include "core/os/os.h"
#include "core/os/time.h"
#include "core/string/ustring.h"

String ProjectZIPPacker::get_project_zip_safe_name() {
	// Name the downloaded ZIP file to contain the project name and download date for easier organization.
	// Replace characters not allowed (or risky) in Windows file names with safe characters.
	// In the project name, all invalid characters become an empty string so that a name
	// like "Platformer 2: Godette's Revenge" becomes "platformer_2-_godette-s_revenge".
	const String project_name = GLOBAL_GET("application/config/name");
	const String project_name_safe = project_name.to_lower().replace_char(' ', '_');
	const String datetime_safe =
			Time::get_singleton()->get_datetime_string_from_system(false, true).replace_char(' ', '_');
	const String output_name = OS::get_singleton()->get_safe_dir_name(vformat("%s_%s.zip", project_name_safe, datetime_safe));
	return output_name;
}

void ProjectZIPPacker::pack_project_zip(const String &p_path) {
	Ref<FileAccess> io_fa;
	zlib_filefunc_def io = zipio_create_io(&io_fa);

	String resource_path = ProjectSettings::get_singleton()->get_resource_path();
	const String base_path = resource_path.substr(0, resource_path.rfind_char('/')) + "/";

	zipFile zip = zipOpen2(p_path.utf8().get_data(), APPEND_STATUS_CREATE, nullptr, &io);
	_zip_recursive(resource_path, base_path, zip);
	zipClose(zip, nullptr);
}

void ProjectZIPPacker::_zip_file(const String &p_path, const String &p_base_path, zipFile p_zip) {
	Ref<FileAccess> f = FileAccess::open(p_path, FileAccess::READ);
	if (f.is_null()) {
		WARN_PRINT("Unable to open file for zipping: " + p_path);
		return;
	}
	Vector<uint8_t> data;
	uint64_t len = f->get_length();
	data.resize(len);
	f->get_buffer(data.ptrw(), len);

	uint64_t time = FileAccess::get_modified_time(p_path);
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
	// 0000755: permissions rwxr-xr-x
	// 0000644: permissions rw-r--r--
	uint32_t _mode = FileAccess::get_unix_permissions(p_path);
	if (_mode == 0) {
		_mode = 0100644;
	} else {
		_mode |= 0100000;
	}
	zipfi.external_fa = (_mode << 16L) | ((_mode & 0200) ? 0 : 1); // UUUUUUUUUUUUUUUU0000000000ADVSHR: Unix permissions (U) + DOS read-only flag (R).
	zipfi.internal_fa = 0;

	String path = p_path.trim_prefix(p_base_path);
	zipOpenNewFileInZip4(p_zip,
			path.utf8().get_data(),
			&zipfi,
			nullptr,
			0,
			nullptr,
			0,
			nullptr,
			Z_DEFLATED,
			Z_DEFAULT_COMPRESSION,
			0,
			-MAX_WBITS,
			DEF_MEM_LEVEL,
			Z_DEFAULT_STRATEGY,
			nullptr,
			0,
			0x0314, // "version made by", 0x03 - Unix, 0x14 - ZIP specification version 2.0, required to store Unix file permissions
			1 << 11); // Bit 11 is the language encoding flag. When set, filename and comment fields must be encoded using UTF-8.
	zipWriteInFileInZip(p_zip, data.ptr(), data.size());
	zipCloseFileInZip(p_zip);
}

void ProjectZIPPacker::_zip_recursive(const String &p_path, const String &p_base_path, zipFile p_zip) {
	Ref<DirAccess> dir = DirAccess::open(p_path);
	if (dir.is_null()) {
		WARN_PRINT("Unable to open directory for zipping: " + p_path);
		return;
	}
	dir->list_dir_begin();
	String cur = dir->get_next();
	String project_data_dir_name = ProjectSettings::get_singleton()->get_project_data_dir_name();
	while (!cur.is_empty()) {
		String cs = p_path.path_join(cur);
		if (cur == "." || cur == ".." || cur == project_data_dir_name) {
			// Skip
		} else if (dir->current_is_dir()) {
			String path = cs.trim_prefix(p_base_path) + "/";
			zipOpenNewFileInZip4(p_zip,
					path.utf8().get_data(),
					nullptr,
					nullptr,
					0,
					nullptr,
					0,
					nullptr,
					Z_DEFLATED,
					Z_DEFAULT_COMPRESSION,
					0,
					-MAX_WBITS,
					DEF_MEM_LEVEL,
					Z_DEFAULT_STRATEGY,
					nullptr,
					0,
					0x0314, // "version made by", 0x03 - Unix, 0x14 - ZIP specification version 2.0, required to store Unix file permissions
					1 << 11); // Bit 11 is the language encoding flag. When set, filename and comment fields must be encoded using UTF-8.
			zipCloseFileInZip(p_zip);
			_zip_recursive(cs, p_base_path, p_zip);
		} else {
			_zip_file(cs, p_base_path, p_zip);
		}
		cur = dir->get_next();
	}
}
