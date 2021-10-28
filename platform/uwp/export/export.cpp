/*************************************************************************/
/*  export.cpp                                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "export.h"
#include "core/bind/core_bind.h"
#include "core/crypto/crypto_core.h"
#include "core/io/marshalls.h"
#include "core/io/zip_io.h"
#include "core/object.h"
#include "core/os/dir_access.h"
#include "core/os/file_access.h"
#include "core/project_settings.h"
#include "core/version.h"
#include "editor/editor_export.h"
#include "editor/editor_node.h"
#include "platform/uwp/logo.gen.h"

#include "thirdparty/minizip/unzip.h"
#include "thirdparty/minizip/zip.h"

#include <zlib.h>

// Capabilities
static const char *uwp_capabilities[] = {
	"allJoyn",
	"codeGeneration",
	"internetClient",
	"internetClientServer",
	"privateNetworkClientServer",
	nullptr
};
static const char *uwp_uap_capabilities[] = {
	"appointments",
	"blockedChatMessages",
	"chat",
	"contacts",
	"enterpriseAuthentication",
	"musicLibrary",
	"objects3D",
	"picturesLibrary",
	"phoneCall",
	"removableStorage",
	"sharedUserCertificates",
	"userAccountInformation",
	"videosLibrary",
	"voipCall",
	nullptr
};
static const char *uwp_device_capabilities[] = {
	"bluetooth",
	"location",
	"microphone",
	"proximity",
	"webcam",
	nullptr
};

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
		size_t compressed_size;
	};

	struct FileMeta {
		String name;
		int lfh_size;
		bool compressed;
		size_t compressed_size;
		size_t uncompressed_size;
		Vector<BlockHash> hashes;
		uLong file_crc32;
		ZPOS64_T zip_offset;

		FileMeta() :
				lfh_size(0),
				compressed(false),
				compressed_size(0),
				uncompressed_size(0),
				file_crc32(0),
				zip_offset(0) {}
	};

	String progress_task;
	FileAccess *package;

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

///////////////////////////////////////////////////////////////////////////

String AppxPackager::hash_block(const uint8_t *p_block_data, size_t p_block_len) {
	unsigned char hash[32];
	char base64[45];

	CryptoCore::sha256(p_block_data, p_block_len, hash);
	size_t len = 0;
	CryptoCore::b64_encode((unsigned char *)base64, 45, &len, (unsigned char *)hash, 32);
	base64[44] = '\0';

	return String(base64);
}

void AppxPackager::make_block_map(const String &p_path) {
	FileAccess *tmp_file = FileAccess::open(p_path, FileAccess::WRITE);

	tmp_file->store_string("<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>");
	tmp_file->store_string("<BlockMap xmlns=\"http://schemas.microsoft.com/appx/2010/blockmap\" HashMethod=\"http://www.w3.org/2001/04/xmlenc#sha256\">");

	for (int i = 0; i < file_metadata.size(); i++) {
		FileMeta file = file_metadata[i];

		tmp_file->store_string(
				"<File Name=\"" + file.name.replace("/", "\\") + "\" Size=\"" + itos(file.uncompressed_size) + "\" LfhSize=\"" + itos(file.lfh_size) + "\">");

		for (int j = 0; j < file.hashes.size(); j++) {
			tmp_file->store_string("<Block Hash=\"" + file.hashes[j].base64_hash + "\" ");
			if (file.compressed) {
				tmp_file->store_string("Size=\"" + itos(file.hashes[j].compressed_size) + "\" ");
			}
			tmp_file->store_string("/>");
		}

		tmp_file->store_string("</File>");
	}

	tmp_file->store_string("</BlockMap>");

	tmp_file->close();
	memdelete(tmp_file);
}

String AppxPackager::content_type(String p_extension) {
	if (p_extension == "png") {
		return "image/png";
	} else if (p_extension == "jpg") {
		return "image/jpg";
	} else if (p_extension == "xml") {
		return "application/xml";
	} else if (p_extension == "exe" || p_extension == "dll") {
		return "application/x-msdownload";
	} else {
		return "application/octet-stream";
	}
}

void AppxPackager::make_content_types(const String &p_path) {
	FileAccess *tmp_file = FileAccess::open(p_path, FileAccess::WRITE);

	tmp_file->store_string("<?xml version=\"1.0\" encoding=\"UTF-8\"?>");
	tmp_file->store_string("<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">");

	Map<String, String> types;

	for (int i = 0; i < file_metadata.size(); i++) {
		String ext = file_metadata[i].name.get_extension().to_lower();

		if (types.has(ext)) {
			continue;
		}

		types[ext] = content_type(ext);

		tmp_file->store_string("<Default Extension=\"" + ext + "\" ContentType=\"" + types[ext] + "\" />");
	}

	// Appx signature file
	tmp_file->store_string("<Default Extension=\"p7x\" ContentType=\"application/octet-stream\" />");

	// Override for package files
	tmp_file->store_string("<Override PartName=\"/AppxManifest.xml\" ContentType=\"application/vnd.ms-appx.manifest+xml\" />");
	tmp_file->store_string("<Override PartName=\"/AppxBlockMap.xml\" ContentType=\"application/vnd.ms-appx.blockmap+xml\" />");
	tmp_file->store_string("<Override PartName=\"/AppxSignature.p7x\" ContentType=\"application/vnd.ms-appx.signature\" />");
	tmp_file->store_string("<Override PartName=\"/AppxMetadata/CodeIntegrity.cat\" ContentType=\"application/vnd.ms-pkiseccat\" />");

	tmp_file->store_string("</Types>");

	tmp_file->close();
	memdelete(tmp_file);
}

Vector<uint8_t> AppxPackager::make_file_header(FileMeta p_file_meta) {
	Vector<uint8_t> buf;
	buf.resize(BASE_FILE_HEADER_SIZE + p_file_meta.name.length());

	int offs = 0;
	// Write magic
	offs += buf_put_int32(FILE_HEADER_MAGIC, &buf.write[offs]);

	// Version
	offs += buf_put_int16(ZIP_VERSION, &buf.write[offs]);

	// Special flag
	offs += buf_put_int16(GENERAL_PURPOSE, &buf.write[offs]);

	// Compression
	offs += buf_put_int16(p_file_meta.compressed ? Z_DEFLATED : 0, &buf.write[offs]);

	// File date and time
	offs += buf_put_int32(0, &buf.write[offs]);

	// CRC-32
	offs += buf_put_int32(p_file_meta.file_crc32, &buf.write[offs]);

	// Compressed size
	offs += buf_put_int32(p_file_meta.compressed_size, &buf.write[offs]);

	// Uncompressed size
	offs += buf_put_int32(p_file_meta.uncompressed_size, &buf.write[offs]);

	// File name length
	offs += buf_put_int16(p_file_meta.name.length(), &buf.write[offs]);

	// Extra data length
	offs += buf_put_int16(0, &buf.write[offs]);

	// File name
	offs += buf_put_string(p_file_meta.name, &buf.write[offs]);

	// Done!
	return buf;
}

void AppxPackager::store_central_dir_header(const FileMeta &p_file, bool p_do_hash) {
	Vector<uint8_t> &buf = central_dir_data;
	int offs = buf.size();
	buf.resize(buf.size() + BASE_CENTRAL_DIR_SIZE + p_file.name.length());

	// Write magic
	offs += buf_put_int32(CENTRAL_DIR_MAGIC, &buf.write[offs]);

	// ZIP versions
	offs += buf_put_int16(ZIP_ARCHIVE_VERSION, &buf.write[offs]);
	offs += buf_put_int16(ZIP_VERSION, &buf.write[offs]);

	// General purpose flag
	offs += buf_put_int16(GENERAL_PURPOSE, &buf.write[offs]);

	// Compression
	offs += buf_put_int16(p_file.compressed ? Z_DEFLATED : 0, &buf.write[offs]);

	// Modification date/time
	offs += buf_put_int32(0, &buf.write[offs]);

	// Crc-32
	offs += buf_put_int32(p_file.file_crc32, &buf.write[offs]);

	// File sizes
	offs += buf_put_int32(p_file.compressed_size, &buf.write[offs]);
	offs += buf_put_int32(p_file.uncompressed_size, &buf.write[offs]);

	// File name length
	offs += buf_put_int16(p_file.name.length(), &buf.write[offs]);

	// Extra field length
	offs += buf_put_int16(0, &buf.write[offs]);

	// Comment length
	offs += buf_put_int16(0, &buf.write[offs]);

	// Disk number start, internal/external file attributes
	for (int i = 0; i < 8; i++) {
		buf.write[offs++] = 0;
	}

	// Relative offset
	offs += buf_put_int32(p_file.zip_offset, &buf.write[offs]);

	// File name
	offs += buf_put_string(p_file.name, &buf.write[offs]);

	// Done!
}

Vector<uint8_t> AppxPackager::make_end_of_central_record() {
	Vector<uint8_t> buf;
	buf.resize(ZIP64_END_OF_CENTRAL_DIR_SIZE + 12 + END_OF_CENTRAL_DIR_SIZE); // Size plus magic

	int offs = 0;

	// Write magic
	offs += buf_put_int32(ZIP64_END_OF_CENTRAL_DIR_MAGIC, &buf.write[offs]);

	// Size of this record
	offs += buf_put_int64(ZIP64_END_OF_CENTRAL_DIR_SIZE, &buf.write[offs]);

	// Version (yes, twice)
	offs += buf_put_int16(ZIP_ARCHIVE_VERSION, &buf.write[offs]);
	offs += buf_put_int16(ZIP_ARCHIVE_VERSION, &buf.write[offs]);

	// Disk number
	for (int i = 0; i < 8; i++) {
		buf.write[offs++] = 0;
	}

	// Number of entries (total and per disk)
	offs += buf_put_int64(file_metadata.size(), &buf.write[offs]);
	offs += buf_put_int64(file_metadata.size(), &buf.write[offs]);

	// Size of central dir
	offs += buf_put_int64(central_dir_data.size(), &buf.write[offs]);

	// Central dir offset
	offs += buf_put_int64(central_dir_offset, &buf.write[offs]);

	////// ZIP64 locator

	// Write magic for zip64 central dir locator
	offs += buf_put_int32(ZIP64_END_DIR_LOCATOR_MAGIC, &buf.write[offs]);

	// Disk number
	for (int i = 0; i < 4; i++) {
		buf.write[offs++] = 0;
	}

	// Relative offset
	offs += buf_put_int64(end_of_central_dir_offset, &buf.write[offs]);

	// Number of disks
	offs += buf_put_int32(1, &buf.write[offs]);

	/////// End of zip directory

	// Write magic for end central dir
	offs += buf_put_int32(END_OF_CENTRAL_DIR_MAGIC, &buf.write[offs]);

	// Dummy stuff for Zip64
	for (int i = 0; i < 4; i++) {
		buf.write[offs++] = 0x0;
	}
	for (int i = 0; i < 12; i++) {
		buf.write[offs++] = 0xFF;
	}

	// Size of comments
	for (int i = 0; i < 2; i++) {
		buf.write[offs++] = 0;
	}

	// Done!
	return buf;
}

void AppxPackager::init(FileAccess *p_fa) {
	package = p_fa;
	central_dir_offset = 0;
	end_of_central_dir_offset = 0;
}

Error AppxPackager::add_file(String p_file_name, const uint8_t *p_buffer, size_t p_len, int p_file_no, int p_total_files, bool p_compress) {
	if (p_file_no >= 1 && p_total_files >= 1) {
		if (EditorNode::progress_task_step(progress_task, "File: " + p_file_name, (p_file_no * 100) / p_total_files)) {
			return ERR_SKIP;
		}
	}

	FileMeta meta;
	meta.name = p_file_name;
	meta.uncompressed_size = p_len;
	meta.compressed_size = p_len;
	meta.compressed = p_compress;
	meta.zip_offset = package->get_position();

	Vector<uint8_t> file_buffer;

	// Data for compression
	z_stream strm;
	FileAccess *strm_f = nullptr;
	Vector<uint8_t> strm_in;
	strm_in.resize(BLOCK_SIZE);
	Vector<uint8_t> strm_out;

	if (p_compress) {
		strm.zalloc = zipio_alloc;
		strm.zfree = zipio_free;
		strm.opaque = &strm_f;

		strm_out.resize(BLOCK_SIZE + 8);

		deflateInit2(&strm, Z_DEFAULT_COMPRESSION, Z_DEFLATED, -15, 8, Z_DEFAULT_STRATEGY);
	}

	int step = 0;

	while (p_len - step > 0) {
		size_t block_size = (p_len - step) > BLOCK_SIZE ? (size_t)BLOCK_SIZE : (p_len - step);

		for (uint64_t i = 0; i < block_size; i++) {
			strm_in.write[i] = p_buffer[step + i];
		}

		BlockHash bh;
		bh.base64_hash = hash_block(strm_in.ptr(), block_size);

		if (p_compress) {
			strm.avail_in = block_size;
			strm.avail_out = strm_out.size();
			strm.next_in = (uint8_t *)strm_in.ptr();
			strm.next_out = strm_out.ptrw();

			int total_out_before = strm.total_out;

			int err = deflate(&strm, Z_FULL_FLUSH);
			ERR_FAIL_COND_V(err < 0, ERR_BUG); // Negative means bug

			bh.compressed_size = strm.total_out - total_out_before;

			//package->store_buffer(strm_out.ptr(), strm.total_out - total_out_before);
			int start = file_buffer.size();
			file_buffer.resize(file_buffer.size() + bh.compressed_size);
			for (uint64_t i = 0; i < bh.compressed_size; i++) {
				file_buffer.write[start + i] = strm_out[i];
			}
		} else {
			bh.compressed_size = block_size;
			//package->store_buffer(strm_in.ptr(), block_size);
			int start = file_buffer.size();
			file_buffer.resize(file_buffer.size() + block_size);
			for (uint64_t i = 0; i < bh.compressed_size; i++) {
				file_buffer.write[start + i] = strm_in[i];
			}
		}

		meta.hashes.push_back(bh);

		step += block_size;
	}

	if (p_compress) {
		strm.avail_in = 0;
		strm.avail_out = strm_out.size();
		strm.next_in = (uint8_t *)strm_in.ptr();
		strm.next_out = strm_out.ptrw();

		int total_out_before = strm.total_out;

		deflate(&strm, Z_FINISH);

		//package->store_buffer(strm_out.ptr(), strm.total_out - total_out_before);
		int start = file_buffer.size();
		file_buffer.resize(file_buffer.size() + (strm.total_out - total_out_before));
		for (uint64_t i = 0; i < (strm.total_out - total_out_before); i++) {
			file_buffer.write[start + i] = strm_out[i];
		}

		deflateEnd(&strm);
		meta.compressed_size = strm.total_out;

	} else {
		meta.compressed_size = p_len;
	}

	// Calculate file CRC-32
	uLong crc = crc32(0L, Z_NULL, 0);
	crc = crc32(crc, p_buffer, p_len);
	meta.file_crc32 = crc;

	// Create file header
	Vector<uint8_t> file_header = make_file_header(meta);
	meta.lfh_size = file_header.size();

	// Store the header and file;
	package->store_buffer(file_header.ptr(), file_header.size());
	package->store_buffer(file_buffer.ptr(), file_buffer.size());

	file_metadata.push_back(meta);

	return OK;
}

void AppxPackager::finish() {
	// Create and add block map file
	EditorNode::progress_task_step("export", "Creating block map...", 4);

	const String &tmp_blockmap_file_path = EditorSettings::get_singleton()->get_cache_dir().plus_file("tmpblockmap.xml");
	make_block_map(tmp_blockmap_file_path);

	FileAccess *blockmap_file = FileAccess::open(tmp_blockmap_file_path, FileAccess::READ);
	Vector<uint8_t> blockmap_buffer;
	blockmap_buffer.resize(blockmap_file->get_len());

	blockmap_file->get_buffer(blockmap_buffer.ptrw(), blockmap_buffer.size());

	add_file("AppxBlockMap.xml", blockmap_buffer.ptr(), blockmap_buffer.size(), -1, -1, true);

	blockmap_file->close();
	memdelete(blockmap_file);

	// Add content types

	EditorNode::progress_task_step("export", "Setting content types...", 5);

	const String &tmp_content_types_file_path = EditorSettings::get_singleton()->get_cache_dir().plus_file("tmpcontenttypes.xml");
	make_content_types(tmp_content_types_file_path);

	FileAccess *types_file = FileAccess::open(tmp_content_types_file_path, FileAccess::READ);
	Vector<uint8_t> types_buffer;
	types_buffer.resize(types_file->get_len());

	types_file->get_buffer(types_buffer.ptrw(), types_buffer.size());

	add_file("[Content_Types].xml", types_buffer.ptr(), types_buffer.size(), -1, -1, true);

	types_file->close();
	memdelete(types_file);

	// Cleanup generated files.
	DirAccess::remove_file_or_error(tmp_blockmap_file_path);
	DirAccess::remove_file_or_error(tmp_content_types_file_path);

	// Pre-process central directory before signing
	for (int i = 0; i < file_metadata.size(); i++) {
		store_central_dir_header(file_metadata[i]);
	}

	// Write central directory
	EditorNode::progress_task_step("export", "Finishing package...", 6);
	central_dir_offset = package->get_position();
	package->store_buffer(central_dir_data.ptr(), central_dir_data.size());

	// End record
	end_of_central_dir_offset = package->get_position();
	Vector<uint8_t> end_record = make_end_of_central_record();
	package->store_buffer(end_record.ptr(), end_record.size());

	package->close();
	memdelete(package);
	package = nullptr;
}

AppxPackager::AppxPackager() {}

AppxPackager::~AppxPackager() {}

////////////////////////////////////////////////////////////////////

class EditorExportPlatformUWP : public EditorExportPlatform {
	GDCLASS(EditorExportPlatformUWP, EditorExportPlatform);

	Ref<ImageTexture> logo;

	enum Platform {
		ARM,
		X86,
		X64
	};

	bool _valid_resource_name(const String &p_name) const {
		if (p_name.empty()) {
			return false;
		}
		if (p_name.ends_with(".")) {
			return false;
		}

		static const char *invalid_names[] = {
			"CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7",
			"COM8", "COM9", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
			nullptr
		};

		const char **t = invalid_names;
		while (*t) {
			if (p_name == *t) {
				return false;
			}
			t++;
		}

		return true;
	}

	bool _valid_guid(const String &p_guid) const {
		Vector<String> parts = p_guid.split("-");

		if (parts.size() != 5) {
			return false;
		}
		if (parts[0].length() != 8) {
			return false;
		}
		for (int i = 1; i < 4; i++) {
			if (parts[i].length() != 4) {
				return false;
			}
		}
		if (parts[4].length() != 12) {
			return false;
		}

		return true;
	}

	bool _valid_bgcolor(const String &p_color) const {
		if (p_color.empty()) {
			return true;
		}
		if (p_color.begins_with("#") && p_color.is_valid_html_color()) {
			return true;
		}

		// Colors from https://msdn.microsoft.com/en-us/library/windows/apps/dn934817.aspx
		static const char *valid_colors[] = {
			"aliceBlue", "antiqueWhite", "aqua", "aquamarine", "azure", "beige",
			"bisque", "black", "blanchedAlmond", "blue", "blueViolet", "brown",
			"burlyWood", "cadetBlue", "chartreuse", "chocolate", "coral", "cornflowerBlue",
			"cornsilk", "crimson", "cyan", "darkBlue", "darkCyan", "darkGoldenrod",
			"darkGray", "darkGreen", "darkKhaki", "darkMagenta", "darkOliveGreen", "darkOrange",
			"darkOrchid", "darkRed", "darkSalmon", "darkSeaGreen", "darkSlateBlue", "darkSlateGray",
			"darkTurquoise", "darkViolet", "deepPink", "deepSkyBlue", "dimGray", "dodgerBlue",
			"firebrick", "floralWhite", "forestGreen", "fuchsia", "gainsboro", "ghostWhite",
			"gold", "goldenrod", "gray", "green", "greenYellow", "honeydew",
			"hotPink", "indianRed", "indigo", "ivory", "khaki", "lavender",
			"lavenderBlush", "lawnGreen", "lemonChiffon", "lightBlue", "lightCoral", "lightCyan",
			"lightGoldenrodYellow", "lightGreen", "lightGray", "lightPink", "lightSalmon", "lightSeaGreen",
			"lightSkyBlue", "lightSlateGray", "lightSteelBlue", "lightYellow", "lime", "limeGreen",
			"linen", "magenta", "maroon", "mediumAquamarine", "mediumBlue", "mediumOrchid",
			"mediumPurple", "mediumSeaGreen", "mediumSlateBlue", "mediumSpringGreen", "mediumTurquoise", "mediumVioletRed",
			"midnightBlue", "mintCream", "mistyRose", "moccasin", "navajoWhite", "navy",
			"oldLace", "olive", "oliveDrab", "orange", "orangeRed", "orchid",
			"paleGoldenrod", "paleGreen", "paleTurquoise", "paleVioletRed", "papayaWhip", "peachPuff",
			"peru", "pink", "plum", "powderBlue", "purple", "red",
			"rosyBrown", "royalBlue", "saddleBrown", "salmon", "sandyBrown", "seaGreen",
			"seaShell", "sienna", "silver", "skyBlue", "slateBlue", "slateGray",
			"snow", "springGreen", "steelBlue", "tan", "teal", "thistle",
			"tomato", "transparent", "turquoise", "violet", "wheat", "white",
			"whiteSmoke", "yellow", "yellowGreen",
			nullptr
		};

		const char **color = valid_colors;

		while (*color) {
			if (p_color == *color) {
				return true;
			}
			color++;
		}

		return false;
	}

	bool _valid_image(const StreamTexture *p_image, int p_width, int p_height) const {
		if (!p_image) {
			return false;
		}

		// TODO: Add resource creation or image rescaling to enable other scales:
		// 1.25, 1.5, 2.0
		return p_width == p_image->get_width() && p_height == p_image->get_height();
	}

	Vector<uint8_t> _fix_manifest(const Ref<EditorExportPreset> &p_preset, const Vector<uint8_t> &p_template, bool p_give_internet) const {
		String result = String::utf8((const char *)p_template.ptr(), p_template.size());

		result = result.replace("$godot_version$", VERSION_FULL_NAME);

		result = result.replace("$identity_name$", p_preset->get("package/unique_name"));
		result = result.replace("$publisher$", p_preset->get("package/publisher"));

		result = result.replace("$product_guid$", p_preset->get("identity/product_guid"));
		result = result.replace("$publisher_guid$", p_preset->get("identity/publisher_guid"));

		String version = itos(p_preset->get("version/major")) + "." + itos(p_preset->get("version/minor")) + "." + itos(p_preset->get("version/build")) + "." + itos(p_preset->get("version/revision"));
		result = result.replace("$version_string$", version);

		Platform arch = (Platform)(int)p_preset->get("architecture/target");
		String architecture = arch == ARM ? "arm" : (arch == X86 ? "x86" : "x64");
		result = result.replace("$architecture$", architecture);

		result = result.replace("$display_name$", String(p_preset->get("package/display_name")).empty() ? (String)ProjectSettings::get_singleton()->get("application/config/name") : String(p_preset->get("package/display_name")));

		result = result.replace("$publisher_display_name$", p_preset->get("package/publisher_display_name"));
		result = result.replace("$app_description$", p_preset->get("package/description"));
		result = result.replace("$bg_color$", p_preset->get("images/background_color"));
		result = result.replace("$short_name$", p_preset->get("package/short_name"));

		String name_on_tiles = "";
		if ((bool)p_preset->get("tiles/show_name_on_square150x150")) {
			name_on_tiles += "          <uap:ShowOn Tile=\"square150x150Logo\" />\n";
		}
		if ((bool)p_preset->get("tiles/show_name_on_wide310x150")) {
			name_on_tiles += "          <uap:ShowOn Tile=\"wide310x150Logo\" />\n";
		}
		if ((bool)p_preset->get("tiles/show_name_on_square310x310")) {
			name_on_tiles += "          <uap:ShowOn Tile=\"square310x310Logo\" />\n";
		}

		String show_name_on_tiles = "";
		if (!name_on_tiles.empty()) {
			show_name_on_tiles = "<uap:ShowNameOnTiles>\n" + name_on_tiles + "        </uap:ShowNameOnTiles>";
		}

		result = result.replace("$name_on_tiles$", name_on_tiles);

		String rotations = "";
		if ((bool)p_preset->get("orientation/landscape")) {
			rotations += "          <uap:Rotation Preference=\"landscape\" />\n";
		}
		if ((bool)p_preset->get("orientation/portrait")) {
			rotations += "          <uap:Rotation Preference=\"portrait\" />\n";
		}
		if ((bool)p_preset->get("orientation/landscape_flipped")) {
			rotations += "          <uap:Rotation Preference=\"landscapeFlipped\" />\n";
		}
		if ((bool)p_preset->get("orientation/portrait_flipped")) {
			rotations += "          <uap:Rotation Preference=\"portraitFlipped\" />\n";
		}

		String rotation_preference = "";
		if (!rotations.empty()) {
			rotation_preference = "<uap:InitialRotationPreference>\n" + rotations + "        </uap:InitialRotationPreference>";
		}

		result = result.replace("$rotation_preference$", rotation_preference);

		String capabilities_elements = "";
		const char **basic = uwp_capabilities;
		while (*basic) {
			if ((bool)p_preset->get("capabilities/" + String(*basic))) {
				capabilities_elements += "    <Capability Name=\"" + String(*basic) + "\" />\n";
			}
			basic++;
		}
		const char **uap = uwp_uap_capabilities;
		while (*uap) {
			if ((bool)p_preset->get("capabilities/" + String(*uap))) {
				capabilities_elements += "    <uap:Capability Name=\"" + String(*uap) + "\" />\n";
			}
			uap++;
		}
		const char **device = uwp_device_capabilities;
		while (*device) {
			if ((bool)p_preset->get("capabilities/" + String(*device))) {
				capabilities_elements += "    <DeviceCapability Name=\"" + String(*device) + "\" />\n";
			}
			device++;
		}

		if (!((bool)p_preset->get("capabilities/internetClient")) && p_give_internet) {
			capabilities_elements += "    <Capability Name=\"internetClient\" />\n";
		}

		String capabilities_string = "<Capabilities />";
		if (!capabilities_elements.empty()) {
			capabilities_string = "<Capabilities>\n" + capabilities_elements + "  </Capabilities>";
		}

		result = result.replace("$capabilities_place$", capabilities_string);

		Vector<uint8_t> r_ret;
		r_ret.resize(result.length());

		for (int i = 0; i < result.length(); i++) {
			r_ret.write[i] = result.utf8().get(i);
		}

		return r_ret;
	}

	Vector<uint8_t> _get_image_data(const Ref<EditorExportPreset> &p_preset, const String &p_path) {
		Vector<uint8_t> data;
		StreamTexture *image = nullptr;

		if (p_path.find("StoreLogo") != -1) {
			image = p_preset->get("images/store_logo").is_zero() ? nullptr : Object::cast_to<StreamTexture>(((Object *)p_preset->get("images/store_logo")));
		} else if (p_path.find("Square44x44Logo") != -1) {
			image = p_preset->get("images/square44x44_logo").is_zero() ? nullptr : Object::cast_to<StreamTexture>(((Object *)p_preset->get("images/square44x44_logo")));
		} else if (p_path.find("Square71x71Logo") != -1) {
			image = p_preset->get("images/square71x71_logo").is_zero() ? nullptr : Object::cast_to<StreamTexture>(((Object *)p_preset->get("images/square71x71_logo")));
		} else if (p_path.find("Square150x150Logo") != -1) {
			image = p_preset->get("images/square150x150_logo").is_zero() ? nullptr : Object::cast_to<StreamTexture>(((Object *)p_preset->get("images/square150x150_logo")));
		} else if (p_path.find("Square310x310Logo") != -1) {
			image = p_preset->get("images/square310x310_logo").is_zero() ? nullptr : Object::cast_to<StreamTexture>(((Object *)p_preset->get("images/square310x310_logo")));
		} else if (p_path.find("Wide310x150Logo") != -1) {
			image = p_preset->get("images/wide310x150_logo").is_zero() ? nullptr : Object::cast_to<StreamTexture>(((Object *)p_preset->get("images/wide310x150_logo")));
		} else if (p_path.find("SplashScreen") != -1) {
			image = p_preset->get("images/splash_screen").is_zero() ? nullptr : Object::cast_to<StreamTexture>(((Object *)p_preset->get("images/splash_screen")));
		} else {
			ERR_PRINT("Unable to load logo");
		}

		if (!image) {
			return data;
		}

		String tmp_path = EditorSettings::get_singleton()->get_cache_dir().plus_file("uwp_tmp_logo.png");

		Error err = image->get_data()->save_png(tmp_path);

		if (err != OK) {
			String err_string = "Couldn't save temp logo file.";

			EditorNode::add_io_error(err_string);
			ERR_FAIL_V_MSG(data, err_string);
		}

		FileAccess *f = FileAccess::open(tmp_path, FileAccess::READ, &err);

		if (err != OK) {
			String err_string = "Couldn't open temp logo file.";
			// Cleanup generated file.
			DirAccess::remove_file_or_error(tmp_path);
			EditorNode::add_io_error(err_string);
			ERR_FAIL_V_MSG(data, err_string);
		}

		data.resize(f->get_len());
		f->get_buffer(data.ptrw(), data.size());

		f->close();
		memdelete(f);
		DirAccess::remove_file_or_error(tmp_path);

		return data;
	}

	static bool _should_compress_asset(const String &p_path, const Vector<uint8_t> &p_data) {
		/* TODO: This was copied verbatim from Android export. It should be
		 * refactored to the parent class and also be used for .zip export.
		 */

		/*
		 *  By not compressing files with little or not benefit in doing so,
		 *  a performance gain is expected at runtime. Moreover, if the APK is
		 *  zip-aligned, assets stored as they are can be efficiently read by
		 *  Android by memory-mapping them.
		 */

		// -- Unconditional uncompress to mimic AAPT plus some other

		static const char *unconditional_compress_ext[] = {
			// From https://github.com/android/platform_frameworks_base/blob/master/tools/aapt/Package.cpp
			// These formats are already compressed, or don't compress well:
			".jpg", ".jpeg", ".png", ".gif",
			".wav", ".mp2", ".mp3", ".ogg", ".aac",
			".mpg", ".mpeg", ".mid", ".midi", ".smf", ".jet",
			".rtttl", ".imy", ".xmf", ".mp4", ".m4a",
			".m4v", ".3gp", ".3gpp", ".3g2", ".3gpp2",
			".amr", ".awb", ".wma", ".wmv",
			// Godot-specific:
			".webp", // Same reasoning as .png
			".cfb", // Don't let small config files slow-down startup
			".scn", // Binary scenes are usually already compressed
			".stex", // Streamable textures are usually already compressed
			// Trailer for easier processing
			nullptr
		};

		for (const char **ext = unconditional_compress_ext; *ext; ++ext) {
			if (p_path.to_lower().ends_with(String(*ext))) {
				return false;
			}
		}

		// -- Compressed resource?

		if (p_data.size() >= 4 && p_data[0] == 'R' && p_data[1] == 'S' && p_data[2] == 'C' && p_data[3] == 'C') {
			// Already compressed
			return false;
		}

		// --- TODO: Decide on texture resources according to their image compression setting

		return true;
	}

	static Error save_appx_file(void *p_userdata, const String &p_path, const Vector<uint8_t> &p_data, int p_file, int p_total) {
		AppxPackager *packager = (AppxPackager *)p_userdata;
		String dst_path = p_path.replace_first("res://", "game/");

		return packager->add_file(dst_path, p_data.ptr(), p_data.size(), p_file, p_total, _should_compress_asset(p_path, p_data));
	}

public:
	virtual String get_name() const {
		return "UWP";
	}
	virtual String get_os_name() const {
		return "UWP";
	}

	virtual List<String> get_binary_extensions(const Ref<EditorExportPreset> &p_preset) const {
		List<String> list;
		list.push_back("appx");
		return list;
	}

	virtual Ref<Texture> get_logo() const {
		return logo;
	}

	virtual void get_preset_features(const Ref<EditorExportPreset> &p_preset, List<String> *r_features) {
		r_features->push_back("s3tc");
		r_features->push_back("etc");
		switch ((int)p_preset->get("architecture/target")) {
			case EditorExportPlatformUWP::ARM: {
				r_features->push_back("arm");
			} break;
			case EditorExportPlatformUWP::X86: {
				r_features->push_back("32");
			} break;
			case EditorExportPlatformUWP::X64: {
				r_features->push_back("64");
			} break;
		}
	}

	virtual void get_export_options(List<ExportOption> *r_options) {
		r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "custom_template/debug", PROPERTY_HINT_GLOBAL_FILE, "*.zip"), ""));
		r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "custom_template/release", PROPERTY_HINT_GLOBAL_FILE, "*.zip"), ""));

		r_options->push_back(ExportOption(PropertyInfo(Variant::INT, "architecture/target", PROPERTY_HINT_ENUM, "arm,x86,x64"), 1));

		r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "command_line/extra_args"), ""));

		r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "package/display_name", PROPERTY_HINT_PLACEHOLDER_TEXT, "Game Name"), ""));
		r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "package/short_name", PROPERTY_HINT_PLACEHOLDER_TEXT, "Game Name"), ""));
		r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "package/unique_name", PROPERTY_HINT_PLACEHOLDER_TEXT, "Game.Name"), ""));
		r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "package/description"), ""));
		r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "package/publisher", PROPERTY_HINT_PLACEHOLDER_TEXT, "CN=CompanyName"), ""));
		r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "package/publisher_display_name", PROPERTY_HINT_PLACEHOLDER_TEXT, "Company Name"), ""));

		r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "identity/product_guid", PROPERTY_HINT_PLACEHOLDER_TEXT, "00000000-0000-0000-0000-000000000000"), ""));
		r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "identity/publisher_guid", PROPERTY_HINT_PLACEHOLDER_TEXT, "00000000-0000-0000-0000-000000000000"), ""));

		r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "signing/certificate", PROPERTY_HINT_GLOBAL_FILE, "*.pfx"), ""));
		r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "signing/password"), ""));
		r_options->push_back(ExportOption(PropertyInfo(Variant::INT, "signing/algorithm", PROPERTY_HINT_ENUM, "MD5,SHA1,SHA256"), 2));

		r_options->push_back(ExportOption(PropertyInfo(Variant::INT, "version/major"), 1));
		r_options->push_back(ExportOption(PropertyInfo(Variant::INT, "version/minor"), 0));
		r_options->push_back(ExportOption(PropertyInfo(Variant::INT, "version/build"), 0));
		r_options->push_back(ExportOption(PropertyInfo(Variant::INT, "version/revision"), 0));

		r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "orientation/landscape"), true));
		r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "orientation/portrait"), true));
		r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "orientation/landscape_flipped"), true));
		r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "orientation/portrait_flipped"), true));

		r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "images/background_color"), "transparent"));
		r_options->push_back(ExportOption(PropertyInfo(Variant::OBJECT, "images/store_logo", PROPERTY_HINT_RESOURCE_TYPE, "StreamTexture"), Variant()));
		r_options->push_back(ExportOption(PropertyInfo(Variant::OBJECT, "images/square44x44_logo", PROPERTY_HINT_RESOURCE_TYPE, "StreamTexture"), Variant()));
		r_options->push_back(ExportOption(PropertyInfo(Variant::OBJECT, "images/square71x71_logo", PROPERTY_HINT_RESOURCE_TYPE, "StreamTexture"), Variant()));
		r_options->push_back(ExportOption(PropertyInfo(Variant::OBJECT, "images/square150x150_logo", PROPERTY_HINT_RESOURCE_TYPE, "StreamTexture"), Variant()));
		r_options->push_back(ExportOption(PropertyInfo(Variant::OBJECT, "images/square310x310_logo", PROPERTY_HINT_RESOURCE_TYPE, "StreamTexture"), Variant()));
		r_options->push_back(ExportOption(PropertyInfo(Variant::OBJECT, "images/wide310x150_logo", PROPERTY_HINT_RESOURCE_TYPE, "StreamTexture"), Variant()));
		r_options->push_back(ExportOption(PropertyInfo(Variant::OBJECT, "images/splash_screen", PROPERTY_HINT_RESOURCE_TYPE, "StreamTexture"), Variant()));

		r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "tiles/show_name_on_square150x150"), false));
		r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "tiles/show_name_on_wide310x150"), false));
		r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "tiles/show_name_on_square310x310"), false));

		// Capabilities
		const char **basic = uwp_capabilities;
		while (*basic) {
			r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "capabilities/" + String(*basic)), false));
			basic++;
		}

		const char **uap = uwp_uap_capabilities;
		while (*uap) {
			r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "capabilities/" + String(*uap)), false));
			uap++;
		}

		const char **device = uwp_device_capabilities;
		while (*device) {
			r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "capabilities/" + String(*device)), false));
			device++;
		}
	}

	virtual bool can_export(const Ref<EditorExportPreset> &p_preset, String &r_error, bool &r_missing_templates) const {
		String err;
		bool valid = false;

		// Look for export templates (first official, and if defined custom templates).

		Platform arch = (Platform)(int)(p_preset->get("architecture/target"));
		String platform_infix;
		switch (arch) {
			case EditorExportPlatformUWP::ARM: {
				platform_infix = "arm";
			} break;
			case EditorExportPlatformUWP::X86: {
				platform_infix = "x86";
			} break;
			case EditorExportPlatformUWP::X64: {
				platform_infix = "x64";
			} break;
		}

		bool dvalid = exists_export_template("uwp_" + platform_infix + "_debug.zip", &err);
		bool rvalid = exists_export_template("uwp_" + platform_infix + "_release.zip", &err);

		if (p_preset->get("custom_template/debug") != "") {
			dvalid = FileAccess::exists(p_preset->get("custom_template/debug"));
			if (!dvalid) {
				err += TTR("Custom debug template not found.") + "\n";
			}
		}
		if (p_preset->get("custom_template/release") != "") {
			rvalid = FileAccess::exists(p_preset->get("custom_template/release"));
			if (!rvalid) {
				err += TTR("Custom release template not found.") + "\n";
			}
		}

		valid = dvalid || rvalid;
		r_missing_templates = !valid;

		// Validate the rest of the configuration.

		if (!_valid_resource_name(p_preset->get("package/short_name"))) {
			valid = false;
			err += TTR("Invalid package short name.") + "\n";
		}

		if (!_valid_resource_name(p_preset->get("package/unique_name"))) {
			valid = false;
			err += TTR("Invalid package unique name.") + "\n";
		}

		if (!_valid_resource_name(p_preset->get("package/publisher_display_name"))) {
			valid = false;
			err += TTR("Invalid package publisher display name.") + "\n";
		}

		if (!_valid_guid(p_preset->get("identity/product_guid"))) {
			valid = false;
			err += TTR("Invalid product GUID.") + "\n";
		}

		if (!_valid_guid(p_preset->get("identity/publisher_guid"))) {
			valid = false;
			err += TTR("Invalid publisher GUID.") + "\n";
		}

		if (!_valid_bgcolor(p_preset->get("images/background_color"))) {
			valid = false;
			err += TTR("Invalid background color.") + "\n";
		}

		if (!p_preset->get("images/store_logo").is_zero() && !_valid_image((Object::cast_to<StreamTexture>((Object *)p_preset->get("images/store_logo"))), 50, 50)) {
			valid = false;
			err += TTR("Invalid Store Logo image dimensions (should be 50x50).") + "\n";
		}

		if (!p_preset->get("images/square44x44_logo").is_zero() && !_valid_image((Object::cast_to<StreamTexture>((Object *)p_preset->get("images/square44x44_logo"))), 44, 44)) {
			valid = false;
			err += TTR("Invalid square 44x44 logo image dimensions (should be 44x44).") + "\n";
		}

		if (!p_preset->get("images/square71x71_logo").is_zero() && !_valid_image((Object::cast_to<StreamTexture>((Object *)p_preset->get("images/square71x71_logo"))), 71, 71)) {
			valid = false;
			err += TTR("Invalid square 71x71 logo image dimensions (should be 71x71).") + "\n";
		}

		if (!p_preset->get("images/square150x150_logo").is_zero() && !_valid_image((Object::cast_to<StreamTexture>((Object *)p_preset->get("images/square150x150_logo"))), 150, 150)) {
			valid = false;
			err += TTR("Invalid square 150x150 logo image dimensions (should be 150x150).") + "\n";
		}

		if (!p_preset->get("images/square310x310_logo").is_zero() && !_valid_image((Object::cast_to<StreamTexture>((Object *)p_preset->get("images/square310x310_logo"))), 310, 310)) {
			valid = false;
			err += TTR("Invalid square 310x310 logo image dimensions (should be 310x310).") + "\n";
		}

		if (!p_preset->get("images/wide310x150_logo").is_zero() && !_valid_image((Object::cast_to<StreamTexture>((Object *)p_preset->get("images/wide310x150_logo"))), 310, 150)) {
			valid = false;
			err += TTR("Invalid wide 310x150 logo image dimensions (should be 310x150).") + "\n";
		}

		if (!p_preset->get("images/splash_screen").is_zero() && !_valid_image((Object::cast_to<StreamTexture>((Object *)p_preset->get("images/splash_screen"))), 620, 300)) {
			valid = false;
			err += TTR("Invalid splash screen image dimensions (should be 620x300).") + "\n";
		}

		r_error = err;
		return valid;
	}

	virtual Error export_project(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, int p_flags = 0) {
		ExportNotifier notifier(*this, p_preset, p_debug, p_path, p_flags);

		String src_appx;

		EditorProgress ep("export", "Exporting for UWP", 7, true);

		if (p_debug) {
			src_appx = p_preset->get("custom_template/debug");
		} else {
			src_appx = p_preset->get("custom_template/release");
		}

		src_appx = src_appx.strip_edges();

		Platform arch = (Platform)(int)p_preset->get("architecture/target");

		if (src_appx == "") {
			String err, infix;
			switch (arch) {
				case ARM: {
					infix = "_arm_";
				} break;
				case X86: {
					infix = "_x86_";
				} break;
				case X64: {
					infix = "_x64_";
				} break;
			}
			if (p_debug) {
				src_appx = find_export_template("uwp" + infix + "debug.zip", &err);
			} else {
				src_appx = find_export_template("uwp" + infix + "release.zip", &err);
			}
			if (src_appx == "") {
				EditorNode::add_io_error(err);
				return ERR_FILE_NOT_FOUND;
			}
		}

		if (!DirAccess::exists(p_path.get_base_dir())) {
			return ERR_FILE_BAD_PATH;
		}

		Error err = OK;

		FileAccess *fa_pack = FileAccess::open(p_path, FileAccess::WRITE, &err);
		ERR_FAIL_COND_V_MSG(err != OK, ERR_CANT_CREATE, "Cannot create file '" + p_path + "'.");

		AppxPackager packager;
		packager.init(fa_pack);

		FileAccess *src_f = nullptr;
		zlib_filefunc_def io = zipio_create_io_from_file(&src_f);

		if (ep.step("Creating package...", 0)) {
			return ERR_SKIP;
		}

		unzFile pkg = unzOpen2(src_appx.utf8().get_data(), &io);

		if (!pkg) {
			EditorNode::add_io_error("Could not find template appx to export:\n" + src_appx);
			return ERR_FILE_NOT_FOUND;
		}

		int ret = unzGoToFirstFile(pkg);

		if (ep.step("Copying template files...", 1)) {
			return ERR_SKIP;
		}

		EditorNode::progress_add_task("template_files", "Template files", 100);
		packager.set_progress_task("template_files");

		int template_files_amount = 9;
		int template_file_no = 1;

		while (ret == UNZ_OK) {
			// get file name
			unz_file_info info;
			char fname[16834];
			ret = unzGetCurrentFileInfo(pkg, &info, fname, 16834, nullptr, 0, nullptr, 0);

			String path = fname;

			if (path.ends_with("/")) {
				// Ignore directories
				ret = unzGoToNextFile(pkg);
				continue;
			}

			Vector<uint8_t> data;
			bool do_read = true;

			if (path.begins_with("Assets/")) {
				path = path.replace(".scale-100", "");

				data = _get_image_data(p_preset, path);
				if (data.size() > 0) {
					do_read = false;
				}
			}

			//read
			if (do_read) {
				data.resize(info.uncompressed_size);
				unzOpenCurrentFile(pkg);
				unzReadCurrentFile(pkg, data.ptrw(), data.size());
				unzCloseCurrentFile(pkg);
			}

			if (path == "AppxManifest.xml") {
				data = _fix_manifest(p_preset, data, p_flags & (DEBUG_FLAG_DUMB_CLIENT | DEBUG_FLAG_REMOTE_DEBUG));
			}

			print_line("ADDING: " + path);

			err = packager.add_file(path, data.ptr(), data.size(), template_file_no++, template_files_amount, _should_compress_asset(path, data));
			if (err != OK) {
				return err;
			}

			ret = unzGoToNextFile(pkg);
		}

		EditorNode::progress_end_task("template_files");

		if (ep.step("Creating command line...", 2)) {
			return ERR_SKIP;
		}

		Vector<String> cl = ((String)p_preset->get("command_line/extra_args")).strip_edges().split(" ");
		for (int i = 0; i < cl.size(); i++) {
			if (cl[i].strip_edges().length() == 0) {
				cl.remove(i);
				i--;
			}
		}

		if (!(p_flags & DEBUG_FLAG_DUMB_CLIENT)) {
			cl.push_back("--path");
			cl.push_back("game");
		}

		gen_export_flags(cl, p_flags);

		// Command line file
		Vector<uint8_t> clf;

		// Argc
		clf.resize(4);
		encode_uint32(cl.size(), clf.ptrw());

		for (int i = 0; i < cl.size(); i++) {
			CharString txt = cl[i].utf8();
			int base = clf.size();
			clf.resize(base + 4 + txt.length());
			encode_uint32(txt.length(), &clf.write[base]);
			memcpy(&clf.write[base + 4], txt.ptr(), txt.length());
			print_line(itos(i) + " param: " + cl[i]);
		}

		err = packager.add_file("__cl__.cl", clf.ptr(), clf.size(), -1, -1, false);
		if (err != OK) {
			return err;
		}

		if (ep.step("Adding project files...", 3)) {
			return ERR_SKIP;
		}

		EditorNode::progress_add_task("project_files", "Project Files", 100);
		packager.set_progress_task("project_files");

		err = export_project_files(p_preset, save_appx_file, &packager);

		EditorNode::progress_end_task("project_files");

		if (ep.step("Closing package...", 7)) {
			return ERR_SKIP;
		}

		unzClose(pkg);

		packager.finish();

#ifdef WINDOWS_ENABLED
		// Sign with signtool
		String signtool_path = EditorSettings::get_singleton()->get("export/uwp/signtool");
		if (signtool_path == String()) {
			return OK;
		}

		if (!FileAccess::exists(signtool_path)) {
			ERR_PRINT("Could not find signtool executable at " + signtool_path + ", aborting.");
			return ERR_FILE_NOT_FOUND;
		}

		static String algs[] = { "MD5", "SHA1", "SHA256" };

		String cert_path = EditorSettings::get_singleton()->get("export/uwp/debug_certificate");
		String cert_pass = EditorSettings::get_singleton()->get("export/uwp/debug_password");
		int cert_alg = EditorSettings::get_singleton()->get("export/uwp/debug_algorithm");

		if (!p_debug) {
			cert_path = p_preset->get("signing/certificate");
			cert_pass = p_preset->get("signing/password");
			cert_alg = p_preset->get("signing/algorithm");
		}

		if (cert_path == String()) {
			return OK; // Certificate missing, don't try to sign
		}

		if (!FileAccess::exists(cert_path)) {
			ERR_PRINT("Could not find certificate file at " + cert_path + ", aborting.");
			return ERR_FILE_NOT_FOUND;
		}

		if (cert_alg < 0 || cert_alg > 2) {
			ERR_PRINT("Invalid certificate algorithm " + itos(cert_alg) + ", aborting.");
			return ERR_INVALID_DATA;
		}

		List<String> args;
		args.push_back("sign");
		args.push_back("/fd");
		args.push_back(algs[cert_alg]);
		args.push_back("/a");
		args.push_back("/f");
		args.push_back(cert_path);
		args.push_back("/p");
		args.push_back(cert_pass);
		args.push_back(p_path);

		OS::get_singleton()->execute(signtool_path, args, true);
#endif // WINDOWS_ENABLED

		return OK;
	}

	virtual void get_platform_features(List<String> *r_features) {
		r_features->push_back("pc");
		r_features->push_back("UWP");
	}

	virtual void resolve_platform_feature_priorities(const Ref<EditorExportPreset> &p_preset, Set<String> &p_features) {
	}

	EditorExportPlatformUWP() {
		Ref<Image> img = memnew(Image(_uwp_logo));
		logo.instance();
		logo->create_from_image(img);
	}
};

void register_uwp_exporter() {
#ifdef WINDOWS_ENABLED
	EDITOR_DEF("export/uwp/signtool", "");
	EditorSettings::get_singleton()->add_property_hint(PropertyInfo(Variant::STRING, "export/uwp/signtool", PROPERTY_HINT_GLOBAL_FILE, "*.exe"));
	EDITOR_DEF("export/uwp/debug_certificate", "");
	EditorSettings::get_singleton()->add_property_hint(PropertyInfo(Variant::STRING, "export/uwp/debug_certificate", PROPERTY_HINT_GLOBAL_FILE, "*.pfx"));
	EDITOR_DEF("export/uwp/debug_password", "");
	EDITOR_DEF("export/uwp/debug_algorithm", 2); // SHA256 is the default
	EditorSettings::get_singleton()->add_property_hint(PropertyInfo(Variant::INT, "export/uwp/debug_algorithm", PROPERTY_HINT_ENUM, "MD5,SHA1,SHA256"));
#endif // WINDOWS_ENABLED

	Ref<EditorExportPlatformUWP> exporter;
	exporter.instance();
	EditorExport::get_singleton()->add_export_platform(exporter);
}
