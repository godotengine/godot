/*************************************************************************/
/*  app_packager.h                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef UWP_APP_PACKAGER_H
#define UWP_APP_PACKAGER_H

#include "core/config/project_settings.h"
#include "core/core_bind.h"
#include "core/crypto/crypto_core.h"
#include "core/io/dir_access.h"
#include "core/io/file_access.h"
#include "core/io/marshalls.h"
#include "core/io/zip_io.h"
#include "core/object/class_db.h"
#include "core/version.h"
#include "editor/editor_export.h"
#include "editor/editor_node.h"

#include "thirdparty/minizip/unzip.h"
#include "thirdparty/minizip/zip.h"

#include <zlib.h>

class AppxPackager {
	enum {
		FILE_HEADER_MAGIC = 0x04034b50,
		DATA_DESCRIPTOR_MAGIC = 0x08074b50,
		CENTRAL_DIR_MAGIC = 0x02014b50,
		END_OF_CENTRAL_DIR_MAGIC = 0x06054b50,
		ZIP64_END_OF_CENTRAL_DIR_MAGIC = 0x06064b50,
		ZIP64_END_DIR_LOCATOR_MAGIC = 0x07064b50,
		P7X_SIGNATURE = 0x58434b50,
		ZIP64_HEADER_ID = 0x0001,
		ZIP_VERSION = 20,
		ZIP_ARCHIVE_VERSION = 45,
		GENERAL_PURPOSE = 0x00,
		BASE_FILE_HEADER_SIZE = 30,
		DATA_DESCRIPTOR_SIZE = 24,
		BASE_CENTRAL_DIR_SIZE = 46,
		EXTRA_FIELD_LENGTH = 28,
		ZIP64_HEADER_SIZE = 24,
		ZIP64_END_OF_CENTRAL_DIR_SIZE = (56 - 12),
		END_OF_CENTRAL_DIR_SIZE = 42,
		BLOCK_SIZE = 65536,
	};

	struct BlockHash {
		String base64_hash;
		size_t compressed_size = 0;
	};

	struct FileMeta {
		String name;
		int lfh_size = 0;
		bool compressed = false;
		size_t compressed_size = 0;
		size_t uncompressed_size = 0;
		Vector<BlockHash> hashes;
		uLong file_crc32 = 0;
		ZPOS64_T zip_offset = 0;
	};

	String progress_task;
	FileAccess *package = nullptr;

	Set<String> mime_types;

	Vector<FileMeta> file_metadata;

	ZPOS64_T central_dir_offset;
	ZPOS64_T end_of_central_dir_offset;
	Vector<uint8_t> central_dir_data;

	String hash_block(const uint8_t *p_block_data, size_t p_block_len);

	void make_block_map(const String &p_path);
	void make_content_types(const String &p_path);

	_FORCE_INLINE_ unsigned int buf_put_int16(uint16_t p_val, uint8_t *p_buf) {
		for (int i = 0; i < 2; i++) {
			*p_buf++ = (p_val >> (i * 8)) & 0xFF;
		}
		return 2;
	}

	_FORCE_INLINE_ unsigned int buf_put_int32(uint32_t p_val, uint8_t *p_buf) {
		for (int i = 0; i < 4; i++) {
			*p_buf++ = (p_val >> (i * 8)) & 0xFF;
		}
		return 4;
	}

	_FORCE_INLINE_ unsigned int buf_put_int64(uint64_t p_val, uint8_t *p_buf) {
		for (int i = 0; i < 8; i++) {
			*p_buf++ = (p_val >> (i * 8)) & 0xFF;
		}
		return 8;
	}

	_FORCE_INLINE_ unsigned int buf_put_string(String p_val, uint8_t *p_buf) {
		for (int i = 0; i < p_val.length(); i++) {
			*p_buf++ = p_val.utf8().get(i);
		}
		return p_val.length();
	}

	Vector<uint8_t> make_file_header(FileMeta p_file_meta);
	void store_central_dir_header(const FileMeta &p_file, bool p_do_hash = true);
	Vector<uint8_t> make_end_of_central_record();

	String content_type(String p_extension);

public:
	void set_progress_task(String p_task) { progress_task = p_task; }
	void init(FileAccess *p_fa);
	Error add_file(String p_file_name, const uint8_t *p_buffer, size_t p_len, int p_file_no, int p_total_files, bool p_compress = false);
	void finish();

	AppxPackager();
	~AppxPackager();
};

#endif
