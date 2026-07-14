/**************************************************************************/
/*  zip_reader.cpp                                                        */
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

#include "zip_reader.h"

#include "core/error/error_macros.h"
#include "core/io/dir_access.h"
#include "core/io/file_access.h"
#include "core/io/zip_io.h"
#include "core/object/class_db.h"

Error ZIPReader::open(const String &p_path) {
	if (fa.is_valid()) {
		close();
	}

	zlib_filefunc_def io = zipio_create_io(&fa);
	uzf = unzOpen2(p_path.utf8().get_data(), &io);
	return uzf != nullptr ? OK : FAILED;
}

Error ZIPReader::close() {
	ERR_FAIL_COND_V_MSG(fa.is_null(), FAILED, "ZIPReader cannot be closed because it is not open.");

	Error err = unzClose(uzf) == UNZ_OK ? OK : FAILED;
	if (err == OK) {
		DEV_ASSERT(fa.is_null());
		uzf = nullptr;
	}

	return err;
}

PackedStringArray ZIPReader::get_files() {
	ERR_FAIL_COND_V_MSG(fa.is_null(), PackedStringArray(), "ZIPReader must be opened before use.");

	unz_global_info gi;
	int err = unzGetGlobalInfo(uzf, &gi);
	ERR_FAIL_COND_V(err != UNZ_OK, PackedStringArray());
	if (gi.number_entry == 0) {
		return PackedStringArray();
	}

	err = unzGoToFirstFile(uzf);
	ERR_FAIL_COND_V(err != UNZ_OK, PackedStringArray());

	List<String> s;
	do {
		unz_file_info64 file_info;
		String filepath;

		err = godot_unzip_get_current_file_info(uzf, file_info, filepath);
		if (err == UNZ_OK) {
			s.push_back(filepath);
		}
	} while (unzGoToNextFile(uzf) == UNZ_OK);

	PackedStringArray arr;
	arr.resize(s.size());
	int idx = 0;
	for (const List<String>::Element *E = s.front(); E; E = E->next()) {
		arr.set(idx++, E->get());
	}
	return arr;
}

PackedByteArray ZIPReader::read_file(const String &p_path, bool p_case_sensitive) {
	ERR_FAIL_COND_V_MSG(fa.is_null(), PackedByteArray(), "ZIPReader must be opened before use.");

	int err = UNZ_OK;

	// Locate and open the file.
	err = godot_unzip_locate_file(uzf, p_path, p_case_sensitive);
	ERR_FAIL_COND_V_MSG(err != UNZ_OK, PackedByteArray(), "File does not exist in zip archive: " + p_path);
	err = unzOpenCurrentFile(uzf);
	ERR_FAIL_COND_V_MSG(err != UNZ_OK, PackedByteArray(), "Could not open file within zip archive.");

	// Read the file info.
	unz_file_info info;
	err = unzGetCurrentFileInfo(uzf, &info, nullptr, 0, nullptr, 0, nullptr, 0);
	ERR_FAIL_COND_V_MSG(err != UNZ_OK, PackedByteArray(), "Unable to read file information from zip archive.");
	ERR_FAIL_COND_V_MSG(info.uncompressed_size > INT_MAX, PackedByteArray(), "File contents too large to read from zip archive (>2 GB).");

	// Read the file data.
	PackedByteArray data;
	data.resize(info.uncompressed_size);
	uint8_t *buffer = data.ptrw();
	int to_read = data.size();
	while (to_read > 0) {
		int bytes_read = unzReadCurrentFile(uzf, buffer, to_read);
		ERR_FAIL_COND_V_MSG(bytes_read < 0, PackedByteArray(), "IO/zlib error reading file from zip archive.");
		ERR_FAIL_COND_V_MSG(bytes_read == UNZ_EOF && to_read != 0, PackedByteArray(), "Incomplete file read from zip archive.");
		DEV_ASSERT(bytes_read <= to_read);
		buffer += bytes_read;
		to_read -= bytes_read;
	}

	// Verify the data and return.
	err = unzCloseCurrentFile(uzf);
	ERR_FAIL_COND_V_MSG(err != UNZ_OK, PackedByteArray(), "CRC error reading file from zip archive.");
	return data;
}

bool ZIPReader::file_exists(const String &p_path, bool p_case_sensitive) {
	ERR_FAIL_COND_V_MSG(fa.is_null(), false, "ZIPReader must be opened before use.");

	int cs = p_case_sensitive ? 1 : 2;
	if (unzLocateFile(uzf, p_path.utf8().get_data(), cs) != UNZ_OK) {
		return false;
	}
	if (unzOpenCurrentFile(uzf) != UNZ_OK) {
		return false;
	}

	unzCloseCurrentFile(uzf);
	return true;
}

int ZIPReader::get_compression_level(const String &p_path, bool p_case_sensitive) {
	ERR_FAIL_COND_V_MSG(fa.is_null(), -1, "ZIPReader must be opened before use.");

	int cs = p_case_sensitive ? 1 : 2;
	if (unzLocateFile(uzf, p_path.utf8().get_data(), cs) != UNZ_OK) {
		return -1;
	}

	int method;
	int level;
	if (unzOpenCurrentFile2(uzf, &method, &level, 1) != UNZ_OK) {
		return -1;
	}

	unzCloseCurrentFile(uzf);

	return level;
}

Error ZIPReader::decompress(const String &p_input_path, const String &p_output_path) {
	int err = UNZ_OK;
	Error gdErr = OK;

	// Open the zip file.
	ZIPReader zip_reader = ZIPReader();
	gdErr = zip_reader.open(p_input_path);
	if (gdErr != OK) {
		return gdErr;
	}

	// Create an output directory.
	DirAccess::make_dir_recursive_absolute(p_output_path);
	Ref<DirAccess> output_directory = DirAccess::open(p_output_path, &gdErr);
	if (output_directory.is_null()) {
		return gdErr;
	}

	// Go to the first file in the zip file.
	err = unzGoToFirstFile(zip_reader.uzf);
	ERR_FAIL_COND_V(err != UNZ_OK, ERR_FILE_CORRUPT);

	// Read each file in the zip file.
	while (true) {
		// Open the current file in the zip file.
		err = unzOpenCurrentFile(zip_reader.uzf);
		ERR_FAIL_COND_V_MSG(err != UNZ_OK, ERR_FILE_CORRUPT, "Could not open file within zip archive.");

		// Read the file info for the current file in the zip file.
		unz_file_info64 file_info;
		String file_path;
		err = godot_unzip_get_current_file_info(zip_reader.uzf, file_info, file_path);
		ERR_FAIL_COND_V_MSG(err != UNZ_OK, ERR_FILE_CORRUPT, "Unable to read file information from zip archive.");

		// Replace back-slashes with forward-slashes.
		file_path = file_path.replace_char('\\', '/');

		// Ensure the file path doesn't escape the output directory.
		String simplified_file_path = file_path.simplify_path();
		bool is_malicious_path = simplified_file_path.is_absolute_path() || simplified_file_path == ".." || simplified_file_path.begins_with("../");
		ERR_FAIL_COND_V_MSG(is_malicious_path, ERR_FILE_CORRUPT, "Extracting a ZIP entry would create a file/directory outside the output directory.");

		// The current file is a directory.
		if (file_path.ends_with("/")) {
			// Create the output sub directory.
			gdErr = output_directory->make_dir_recursive(file_path.trim_suffix("/"));
			if (gdErr != OK) {
				return gdErr;
			}
		}
		// The current file is a file.
		else {
			// Create a sub directory for the output file.
			gdErr = output_directory->make_dir_recursive(file_path.get_base_dir());
			if (gdErr != OK) {
				return gdErr;
			}
			// Create an output file.
			Ref<FileAccess> output_file = FileAccess::open(p_output_path.path_join(file_path), FileAccess::WRITE, &gdErr);
			if (output_file.is_null()) {
				return gdErr;
			}

			// Read each chunk of the current file in the zip file.
			uint8_t buffer[64 * 1024]; // 64KiB buffer
			while (true) {
				// Read a chunk of the current file in the zip file.
				int bytes_read = unzReadCurrentFile(zip_reader.uzf, buffer, sizeof(buffer));
				ERR_FAIL_COND_V_MSG(bytes_read < 0, ERR_FILE_CORRUPT, "IO/zlib error reading file from zip archive.");

				// Reached the end of the current file in the zip file.
				if (bytes_read == 0) {
					break;
				}

				// Write the current chunk to the output file.
				bool write_err = output_file->store_buffer(buffer, bytes_read);
				ERR_FAIL_COND_V(!write_err, ERR_FILE_CANT_WRITE);
			}
		}

		// Close the current file in the zip file.
		err = unzCloseCurrentFile(zip_reader.uzf);
		ERR_FAIL_COND_V_MSG(err != UNZ_OK, ERR_FILE_CORRUPT, "CRC error reading file from zip archive.");

		// Go to the next file in the zip file.
		err = unzGoToNextFile(zip_reader.uzf);

		// Reached the end of the list of files in the zip file.
		if (err != UNZ_OK) {
			break;
		}
	}

	// The zip file was successfully extracted.
	return OK;
}

ZIPReader::ZIPReader() {}

ZIPReader::~ZIPReader() {
	if (fa.is_valid()) {
		close();
	}
}

void ZIPReader::_bind_methods() {
	ClassDB::bind_method(D_METHOD("open", "path"), &ZIPReader::open);
	ClassDB::bind_method(D_METHOD("close"), &ZIPReader::close);
	ClassDB::bind_method(D_METHOD("get_files"), &ZIPReader::get_files);
	ClassDB::bind_method(D_METHOD("read_file", "path", "case_sensitive"), &ZIPReader::read_file, DEFVAL(Variant(true)));
	ClassDB::bind_method(D_METHOD("file_exists", "path", "case_sensitive"), &ZIPReader::file_exists, DEFVAL(Variant(true)));
	ClassDB::bind_method(D_METHOD("get_compression_level", "path", "case_sensitive"), &ZIPReader::get_compression_level, DEFVAL(Variant(true)));

	ClassDB::bind_static_method("ZIPReader", D_METHOD("decompress", "input_path", "output_path"), &ZIPReader::decompress);
}
