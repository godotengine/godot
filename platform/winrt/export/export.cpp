/*************************************************************************/
/*  export.cpp                                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2016 Juan Linietsky, Ariel Manzur.                 */
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
#include "bind/core_bind.h"
#include "editor/editor_import_export.h"
#include "editor/editor_node.h"
#include "globals.h"
#include "io/marshalls.h"
#include "io/zip_io.h"
#include "object.h"
#include "os/file_access.h"
#include "platform/winrt/logo.gen.h"
#include "thirdparty/minizip/unzip.h"
#include "thirdparty/minizip/zip.h"
#include "thirdparty/misc/base64.h"
#include "thirdparty/misc/sha256.h"
#include "version.h"

// Capabilities
static const char *uwp_capabilities[] = {
	"allJoyn",
	"codeGeneration",
	"internetClient",
	"internetClientServer",
	"privateNetworkClientServer",
	NULL
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
	NULL
};
static const char *uwp_device_capabilites[] = {
	"bluetooth",
	"location",
	"microphone",
	"proximity",
	"webcam",
	NULL
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
	};

	String progress_task;
	FileAccess *package;
	String tmp_blockmap_file_path;
	String tmp_content_types_file_path;

	Set<String> mime_types;

	Vector<FileMeta> file_metadata;

	ZPOS64_T central_dir_offset;
	ZPOS64_T end_of_central_dir_offset;
	Vector<uint8_t> central_dir_data;

	String hash_block(uint8_t *p_block_data, size_t p_block_len);

	void make_block_map();
	void make_content_types();

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
	void store_central_dir_header(const FileMeta p_file);
	Vector<uint8_t> make_end_of_central_record();

	String content_type(String p_extension);

public:
	void set_progress_task(String p_task) { progress_task = p_task; }
	void init(FileAccess *p_fa);
	void add_file(String p_file_name, const uint8_t *p_buffer, size_t p_len, int p_file_no, int p_total_files, bool p_compress = false);
	void finish();

	AppxPackager();
	~AppxPackager();
};

class EditorExportPlatformWinrt : public EditorExportPlatform {

	OBJ_TYPE(EditorExportPlatformWinrt, EditorExportPlatform);

	Ref<ImageTexture> logo;

	enum Platform {
		ARM,
		X86,
		X64
	} arch;

	String custom_release_package;
	String custom_debug_package;

	String cmdline;

	String display_name;
	String short_name;
	String unique_name;
	String description;
	String publisher;
	String publisher_display_name;

	String product_guid;
	String publisher_guid;

	int version_major;
	int version_minor;
	int version_build;
	int version_revision;

	bool orientation_landscape;
	bool orientation_portrait;
	bool orientation_landscape_flipped;
	bool orientation_portrait_flipped;

	String background_color;
	Ref<ImageTexture> store_logo;
	Ref<ImageTexture> square44;
	Ref<ImageTexture> square71;
	Ref<ImageTexture> square150;
	Ref<ImageTexture> square310;
	Ref<ImageTexture> wide310;
	Ref<ImageTexture> splash;

	bool name_on_square150;
	bool name_on_square310;
	bool name_on_wide;

	Set<String> capabilities;
	Set<String> uap_capabilities;
	Set<String> device_capabilities;

	bool sign_package;
	String certificate_path;
	String certificate_pass;
	enum CertAlgorithm {
		MD5,
		SHA1,
		SHA256
	} certificate_algo;

	_FORCE_INLINE_ bool array_has(const char **p_array, const char *p_value) const {
		while (*p_array) {
			if (String(*p_array) == String(p_value)) return true;
			p_array++;
		}
		return false;
	}

	bool _valid_resource_name(const String &p_name) const;
	bool _valid_guid(const String &p_guid) const;
	bool _valid_bgcolor(const String &p_color) const;
	bool _valid_image(const Ref<ImageTexture> p_image, int p_width, int p_height) const;

	Vector<uint8_t> _fix_manifest(const Vector<uint8_t> &p_template, bool p_give_internet) const;
	Vector<uint8_t> _get_image_data(const String &p_path);

	static Error save_appx_file(void *p_userdata, const String &p_path, const Vector<uint8_t> &p_data, int p_file, int p_total);
	static bool _should_compress_asset(const String &p_path, const Vector<uint8_t> &p_data);

protected:
	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;

public:
	virtual String get_name() const { return "Windows Universal"; }
	virtual ImageCompression get_image_compression() const { return IMAGE_COMPRESSION_ETC1; }
	virtual Ref<Texture> get_logo() const { return logo; }

	virtual bool can_export(String *r_error = NULL) const;
	virtual String get_binary_extension() const { return "appx"; }

	virtual Error export_project(const String &p_path, bool p_debug, int p_flags = 0);

	EditorExportPlatformWinrt();
	~EditorExportPlatformWinrt();
};

///////////////////////////////////////////////////////////////////////////

String AppxPackager::hash_block(uint8_t *p_block_data, size_t p_block_len) {

	char hash[32];
	char base64[45];

	sha256_context ctx;
	sha256_init(&ctx);
	sha256_hash(&ctx, p_block_data, p_block_len);
	sha256_done(&ctx, (uint8_t *)hash);

	base64_encode(base64, hash, 32);
	base64[44] = '\0';

	return String(base64);
}

void AppxPackager::make_block_map() {

	FileAccess *tmp_file = FileAccess::open(tmp_blockmap_file_path, FileAccess::WRITE);

	tmp_file->store_string("<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>");
	tmp_file->store_string("<BlockMap xmlns=\"http://schemas.microsoft.com/appx/2010/blockmap\" HashMethod=\"http://www.w3.org/2001/04/xmlenc#sha256\">");

	for (int i = 0; i < file_metadata.size(); i++) {

		FileMeta file = file_metadata[i];

		tmp_file->store_string(
				"<File Name=\"" + file.name.replace("/", "\\") + "\" Size=\"" + itos(file.uncompressed_size) + "\" LfhSize=\"" + itos(file.lfh_size) + "\">");

		for (int j = 0; j < file.hashes.size(); j++) {

			tmp_file->store_string("<Block Hash=\"" + file.hashes[j].base64_hash + "\" ");
			if (file.compressed)
				tmp_file->store_string("Size=\"" + itos(file.hashes[j].compressed_size) + "\" ");
			tmp_file->store_string("/>");
		}

		tmp_file->store_string("</File>");
	}

	tmp_file->store_string("</BlockMap>");

	tmp_file->close();
	memdelete(tmp_file);
	tmp_file = NULL;
}

String AppxPackager::content_type(String p_extension) {

	if (p_extension == "png")
		return "image/png";
	else if (p_extension == "jpg")
		return "image/jpg";
	else if (p_extension == "xml")
		return "application/xml";
	else if (p_extension == "exe" || p_extension == "dll")
		return "application/x-msdownload";
	else
		return "application/octet-stream";
}

void AppxPackager::make_content_types() {

	FileAccess *tmp_file = FileAccess::open(tmp_content_types_file_path, FileAccess::WRITE);

	tmp_file->store_string("<?xml version=\"1.0\" encoding=\"UTF-8\"?>");
	tmp_file->store_string("<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">");

	Map<String, String> types;

	for (int i = 0; i < file_metadata.size(); i++) {

		String ext = file_metadata[i].name.extension();

		if (types.has(ext)) continue;

		types[ext] = content_type(ext);

		tmp_file->store_string("<Default Extension=\"" + ext +
							   "\" ContentType=\"" + types[ext] + "\" />");
	}

	// Override for package files
	tmp_file->store_string("<Override PartName=\"/AppxManifest.xml\" ContentType=\"application/vnd.ms-appx.manifest+xml\" />");
	tmp_file->store_string("<Override PartName=\"/AppxBlockMap.xml\" ContentType=\"application/vnd.ms-appx.blockmap+xml\" />");
	tmp_file->store_string("<Override PartName=\"/AppxSignature.p7x\" ContentType=\"application/vnd.ms-appx.signature\" />");
	tmp_file->store_string("<Override PartName=\"/AppxMetadata/CodeIntegrity.cat\" ContentType=\"application/vnd.ms-pkiseccat\" />");

	tmp_file->store_string("</Types>");

	tmp_file->close();
	memdelete(tmp_file);
	tmp_file = NULL;
}

Vector<uint8_t> AppxPackager::make_file_header(FileMeta p_file_meta) {

	Vector<uint8_t> buf;
	buf.resize(BASE_FILE_HEADER_SIZE + p_file_meta.name.replace(" ", "%20").length());

	int offs = 0;
	// Write magic
	offs += buf_put_int32(FILE_HEADER_MAGIC, &buf[offs]);

	// Version
	offs += buf_put_int16(ZIP_VERSION, &buf[offs]);

	// Special flag
	offs += buf_put_int16(GENERAL_PURPOSE, &buf[offs]);

	// Compression
	offs += buf_put_int16(p_file_meta.compressed ? Z_DEFLATED : 0, &buf[offs]);

	// File date and time
	offs += buf_put_int32(0, &buf[offs]);

	// CRC-32
	offs += buf_put_int32(p_file_meta.file_crc32, &buf[offs]);

	// Compressed size
	offs += buf_put_int32(p_file_meta.compressed_size, &buf[offs]);

	// Uncompressed size
	offs += buf_put_int32(p_file_meta.uncompressed_size, &buf[offs]);

	// File name length
	offs += buf_put_int16(p_file_meta.name.replace(" ", "%20").length(), &buf[offs]);

	// Extra data length
	offs += buf_put_int16(0, &buf[offs]);

	// File name
	offs += buf_put_string(p_file_meta.name.replace(" ", "%20"), &buf[offs]);

	// Done!
	return buf;
}

void AppxPackager::store_central_dir_header(const FileMeta p_file) {

	Vector<uint8_t> &buf = central_dir_data;
	int offs = buf.size();
	buf.resize(buf.size() + BASE_CENTRAL_DIR_SIZE + p_file.name.replace(" ", "%20").length());

	// Write magic
	offs += buf_put_int32(CENTRAL_DIR_MAGIC, &buf[offs]);

	// ZIP versions
	offs += buf_put_int16(ZIP_ARCHIVE_VERSION, &buf[offs]);
	offs += buf_put_int16(ZIP_VERSION, &buf[offs]);

	// General purpose flag
	offs += buf_put_int16(GENERAL_PURPOSE, &buf[offs]);

	// Compression
	offs += buf_put_int16(p_file.compressed ? Z_DEFLATED : 0, &buf[offs]);

	// Modification date/time
	offs += buf_put_int32(0, &buf[offs]);

	// Crc-32
	offs += buf_put_int32(p_file.file_crc32, &buf[offs]);

	// File sizes
	offs += buf_put_int32(p_file.compressed_size, &buf[offs]);
	offs += buf_put_int32(p_file.uncompressed_size, &buf[offs]);

	// File name length
	offs += buf_put_int16(p_file.name.replace(" ", "%20").length(), &buf[offs]);

	// Extra field length
	offs += buf_put_int16(0, &buf[offs]);

	// Comment length
	offs += buf_put_int16(0, &buf[offs]);

	// Disk number start, internal/external file attributes
	for (int i = 0; i < 8; i++) {
		buf[offs++] = 0;
	}

	// Relative offset
	offs += buf_put_int32(p_file.zip_offset, &buf[offs]);

	// File name
	offs += buf_put_string(p_file.name.replace(" ", "%20"), &buf[offs]);

	// Done!
}

Vector<uint8_t> AppxPackager::make_end_of_central_record() {

	Vector<uint8_t> buf;
	buf.resize(ZIP64_END_OF_CENTRAL_DIR_SIZE + 12 + END_OF_CENTRAL_DIR_SIZE); // Size plus magic

	int offs = 0;

	// Write magic
	offs += buf_put_int32(ZIP64_END_OF_CENTRAL_DIR_MAGIC, &buf[offs]);

	// Size of this record
	offs += buf_put_int64(ZIP64_END_OF_CENTRAL_DIR_SIZE, &buf[offs]);

	// Version (yes, twice)
	offs += buf_put_int16(ZIP_ARCHIVE_VERSION, &buf[offs]);
	offs += buf_put_int16(ZIP_ARCHIVE_VERSION, &buf[offs]);

	// Disk number
	for (int i = 0; i < 8; i++) {
		buf[offs++] = 0;
	}

	// Number of entries (total and per disk)
	offs += buf_put_int64(file_metadata.size(), &buf[offs]);
	offs += buf_put_int64(file_metadata.size(), &buf[offs]);

	// Size of central dir
	offs += buf_put_int64(central_dir_data.size(), &buf[offs]);

	// Central dir offset
	offs += buf_put_int64(central_dir_offset, &buf[offs]);

	////// ZIP64 locator

	// Write magic for zip64 central dir locator
	offs += buf_put_int32(ZIP64_END_DIR_LOCATOR_MAGIC, &buf[offs]);

	// Disk number
	for (int i = 0; i < 4; i++) {
		buf[offs++] = 0;
	}

	// Relative offset
	offs += buf_put_int64(end_of_central_dir_offset, &buf[offs]);

	// Number of disks
	offs += buf_put_int32(1, &buf[offs]);

	/////// End of zip directory

	// Write magic for end central dir
	offs += buf_put_int32(END_OF_CENTRAL_DIR_MAGIC, &buf[offs]);

	// Dummy stuff for Zip64
	for (int i = 0; i < 4; i++) {
		buf[offs++] = 0x0;
	}
	for (int i = 0; i < 12; i++) {
		buf[offs++] = 0xFF;
	}

	// Size of comments
	for (int i = 0; i < 2; i++) {
		buf[offs++] = 0;
	}

	// Done!
	return buf;
}

void AppxPackager::init(FileAccess *p_fa) {

	package = p_fa;
	central_dir_offset = 0;
	end_of_central_dir_offset = 0;
	tmp_blockmap_file_path = EditorSettings::get_singleton()->get_settings_path() + "/tmp/tmpblockmap.xml";
	tmp_content_types_file_path = EditorSettings::get_singleton()->get_settings_path() + "/tmp/tmpcontenttypes.xml";
}

void AppxPackager::add_file(String p_file_name, const uint8_t *p_buffer, size_t p_len, int p_file_no, int p_total_files, bool p_compress) {

	if (p_file_no >= 1 && p_total_files >= 1) {
		EditorNode::progress_task_step(progress_task, "File: " + p_file_name, (p_file_no * 100) / p_total_files);
	}

	FileMeta meta;
	meta.name = p_file_name;
	meta.uncompressed_size = p_len;
	meta.compressed_size = p_len;
	meta.compressed = p_compress;
	meta.zip_offset = package->get_pos();

	Vector<uint8_t> file_buffer;

	// Data for compression
	z_stream strm;
	FileAccess *strm_f = NULL;
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

		size_t block_size = (p_len - step) > BLOCK_SIZE ? BLOCK_SIZE : (p_len - step);

		for (int i = 0; i < block_size; i++) {
			strm_in[i] = p_buffer[step + i];
		}

		BlockHash bh;
		bh.base64_hash = hash_block(strm_in.ptr(), block_size);

		if (p_compress) {

			strm.avail_in = block_size;
			strm.avail_out = strm_out.size();
			strm.next_in = strm_in.ptr();
			strm.next_out = strm_out.ptr();

			int total_out_before = strm.total_out;

			deflate(&strm, Z_FULL_FLUSH);
			bh.compressed_size = strm.total_out - total_out_before;

			//package->store_buffer(strm_out.ptr(), strm.total_out - total_out_before);
			int start = file_buffer.size();
			file_buffer.resize(file_buffer.size() + bh.compressed_size);
			for (int i = 0; i < bh.compressed_size; i++)
				file_buffer[start + i] = strm_out[i];

		} else {
			bh.compressed_size = block_size;
			//package->store_buffer(strm_in.ptr(), block_size);
			int start = file_buffer.size();
			file_buffer.resize(file_buffer.size() + block_size);
			for (int i = 0; i < bh.compressed_size; i++)
				file_buffer[start + i] = strm_in[i];
		}

		meta.hashes.push_back(bh);

		step += block_size;
	}

	if (p_compress) {

		strm.avail_in = 0;
		strm.avail_out = strm_out.size();
		strm.next_in = strm_in.ptr();
		strm.next_out = strm_out.ptr();

		int total_out_before = strm.total_out;

		deflate(&strm, Z_FINISH);

		//package->store_buffer(strm_out.ptr(), strm.total_out - total_out_before);
		int start = file_buffer.size();
		file_buffer.resize(file_buffer.size() + (strm.total_out - total_out_before));
		for (int i = 0; i < (strm.total_out - total_out_before); i++)
			file_buffer[start + i] = strm_out[i];

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
}

void AppxPackager::finish() {

	// Create and add block map file
	EditorNode::progress_task_step("export", "Creating block map...", 4);

	make_block_map();
	FileAccess *blockmap_file = FileAccess::open(tmp_blockmap_file_path, FileAccess::READ);
	Vector<uint8_t> blockmap_buffer;
	blockmap_buffer.resize(blockmap_file->get_len());

	blockmap_file->get_buffer(blockmap_buffer.ptr(), blockmap_buffer.size());

	add_file("AppxBlockMap.xml", blockmap_buffer.ptr(), blockmap_buffer.size(), -1, -1, true);

	blockmap_file->close();
	memdelete(blockmap_file);
	blockmap_file = NULL;

	// Add content types
	EditorNode::progress_task_step("export", "Setting content types...", 5);
	make_content_types();

	FileAccess *types_file = FileAccess::open(tmp_content_types_file_path, FileAccess::READ);
	Vector<uint8_t> types_buffer;
	types_buffer.resize(types_file->get_len());

	types_file->get_buffer(types_buffer.ptr(), types_buffer.size());

	add_file("[Content_Types].xml", types_buffer.ptr(), types_buffer.size(), -1, -1, true);

	types_file->close();
	memdelete(types_file);
	types_file = NULL;

	// Pre-process central directory before signing
	for (int i = 0; i < file_metadata.size(); i++) {
		store_central_dir_header(file_metadata[i]);
	}

	// Write central directory
	EditorNode::progress_task_step("export", "Finishing package...", 6);
	central_dir_offset = package->get_pos();
	package->store_buffer(central_dir_data.ptr(), central_dir_data.size());

	// End record
	end_of_central_dir_offset = package->get_pos();
	Vector<uint8_t> end_record = make_end_of_central_record();
	package->store_buffer(end_record.ptr(), end_record.size());

	package->close();
	memdelete(package);
	package = NULL;
}

AppxPackager::AppxPackager() {}

AppxPackager::~AppxPackager() {}

bool EditorExportPlatformWinrt::_valid_resource_name(const String &p_name) const {

	if (p_name.empty()) return false;
	if (p_name.ends_with(".")) return false;

	static const char *invalid_names[] = {
		"CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7",
		"COM8", "COM9", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
		NULL
	};

	const char **t = invalid_names;
	while (*t) {
		if (p_name == *t) return false;
		t++;
	}

	return true;
}

bool EditorExportPlatformWinrt::_valid_guid(const String &p_guid) const {

	Vector<String> parts = p_guid.split("-");

	if (parts.size() != 5) return false;
	if (parts[0].length() != 8) return false;
	for (int i = 1; i < 4; i++)
		if (parts[i].length() != 4) return false;
	if (parts[4].length() != 12) return false;

	return true;
}

bool EditorExportPlatformWinrt::_valid_bgcolor(const String &p_color) const {

	if (p_color.empty()) return true;
	if (p_color.begins_with("#") && p_color.is_valid_html_color()) return true;

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
		NULL
	};

	const char **color = valid_colors;

	while (*color) {
		if (p_color == *color) return true;
		color++;
	}

	return false;
}

bool EditorExportPlatformWinrt::_valid_image(const Ref<ImageTexture> p_image, int p_width, int p_height) const {

	if (!p_image.is_valid()) return false;

	// TODO: Add resource creation or image rescaling to enable other scales:
	// 1.25, 1.5, 2.0
	real_t scales[] = { 1.0 };
	bool valid_w = false;
	bool valid_h = false;

	for (int i = 0; i < 1; i++) {

		int w = ceil(p_width * scales[i]);
		int h = ceil(p_height * scales[i]);

		if (w == p_image->get_width())
			valid_w = true;
		if (h == p_image->get_height())
			valid_h = true;
	}

	return valid_w && valid_h;
}

Vector<uint8_t> EditorExportPlatformWinrt::_fix_manifest(const Vector<uint8_t> &p_template, bool p_give_internet) const {

	String result = String::utf8((const char *)p_template.ptr(), p_template.size());

	result = result.replace("$godot_version$", VERSION_FULL_NAME);

	result = result.replace("$identity_name$", unique_name);
	result = result.replace("$publisher$", publisher);

	result = result.replace("$product_guid$", product_guid);
	result = result.replace("$publisher_guid$", publisher_guid);

	String version = itos(version_major) + "." + itos(version_minor) + "." + itos(version_build) + "." + itos(version_revision);
	result = result.replace("$version_string$", version);

	String architecture = arch == ARM ? "ARM" : arch == X86 ? "x86" : "x64";
	result = result.replace("$architecture$", architecture);

	result = result.replace("$display_name$", display_name.empty() ? (String)Globals::get_singleton()->get("application/name") : display_name);
	result = result.replace("$publisher_display_name$", publisher_display_name);
	result = result.replace("$app_description$", description);
	result = result.replace("$bg_color$", background_color);
	result = result.replace("$short_name$", short_name);

	String name_on_tiles = "";
	if (name_on_square150) {
		name_on_tiles += "          <uap:ShowOn Tile=\"square150x150Logo\" />\n";
	}
	if (name_on_wide) {
		name_on_tiles += "          <uap:ShowOn Tile=\"wide310x150Logo\" />\n";
	}
	if (name_on_square310) {
		name_on_tiles += "          <uap:ShowOn Tile=\"square310x310Logo\" />\n";
	}

	String show_name_on_tiles = "";
	if (!name_on_tiles.empty()) {
		show_name_on_tiles = "<uap:ShowNameOnTiles>\n" + name_on_tiles + "        </uap:ShowNameOnTiles>";
	}

	result = result.replace("$name_on_tiles$", name_on_tiles);

	String rotations = "";
	if (orientation_landscape) {
		rotations += "          <uap:Rotation Preference=\"landscape\" />\n";
	}
	if (orientation_portrait) {
		rotations += "          <uap:Rotation Preference=\"portrait\" />\n";
	}
	if (orientation_landscape_flipped) {
		rotations += "          <uap:Rotation Preference=\"landscapeFlipped\" />\n";
	}
	if (orientation_portrait_flipped) {
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
		if (capabilities.has(*basic)) {
			capabilities_elements += "    <Capability Name=\"" + String(*basic) + "\" />\n";
		}
		basic++;
	}
	const char **uap = uwp_uap_capabilities;
	while (*uap) {
		if (uap_capabilities.has(*uap)) {
			capabilities_elements += "    <uap:Capability Name=\"" + String(*uap) + "\" />\n";
		}
		uap++;
	}
	const char **device = uwp_device_capabilites;
	while (*device) {
		if (uap_capabilities.has(*device)) {
			capabilities_elements += "    <DeviceCapability Name=\"" + String(*device) + "\" />\n";
		}
		device++;
	}

	if (!capabilities.has("internetClient") && p_give_internet) {
		capabilities_elements += "    <Capability Name=\"internetClient\" />\n";
	}

	String capabilities_string = "<Capabilities />";
	if (!capabilities_elements.empty()) {
		capabilities_string = "<Capabilities>\n" + capabilities_elements + "  </Capabilities>";
	}

	result = result.replace("$capabilities_place$", capabilities_string);

	Vector<uint8_t> r_ret;
	r_ret.resize(result.length());

	for (int i = 0; i < result.length(); i++)
		r_ret[i] = result.utf8().get(i);

	return r_ret;
}

Vector<uint8_t> EditorExportPlatformWinrt::_get_image_data(const String &p_path) {

	Vector<uint8_t> data;
	Ref<ImageTexture> ref;

	if (p_path.find("StoreLogo") != -1) {
		ref = store_logo;
	} else if (p_path.find("Square44x44Logo") != -1) {
		ref = square44;
	} else if (p_path.find("Square71x71Logo") != -1) {
		ref = square71;
	} else if (p_path.find("Square150x150Logo") != -1) {
		ref = square150;
	} else if (p_path.find("Square310x310Logo") != -1) {
		ref = square310;
	} else if (p_path.find("Wide310x150Logo") != -1) {
		ref = wide310;
	} else if (p_path.find("SplashScreen") != -1) {
		ref = splash;
	}

	if (!ref.is_valid()) return data;

	String tmp_path = EditorSettings::get_singleton()->get_settings_path().plus_file("tmp/uwp_tmp_logo.png");

	Error err = ref->get_data().save_png(tmp_path);

	if (err != OK) {

		String err_string = "Couldn't save temp logo file.";

		EditorNode::add_io_error(err_string);
		ERR_EXPLAIN(err_string);
		ERR_FAIL_V(data);
	}

	FileAccess *f = FileAccess::open(tmp_path, FileAccess::READ, &err);

	if (err != OK) {

		String err_string = "Couldn't open temp logo file.";

		EditorNode::add_io_error(err_string);
		ERR_EXPLAIN(err_string);
		ERR_FAIL_V(data);
	}

	data.resize(f->get_len());
	f->get_buffer(data.ptr(), data.size());

	f->close();
	memdelete(f);

	// Delete temp file
	DirAccess *dir = DirAccess::open(tmp_path.get_base_dir(), &err);

	if (err != OK) {

		String err_string = "Couldn't open temp path to remove temp logo file.";

		EditorNode::add_io_error(err_string);
		ERR_EXPLAIN(err_string);
		ERR_FAIL_V(data);
	}

	err = dir->remove(tmp_path);

	memdelete(dir);

	if (err != OK) {

		String err_string = "Couldn't remove temp logo file.";

		EditorNode::add_io_error(err_string);
		ERR_EXPLAIN(err_string);
		ERR_FAIL_V(data);
	}

	return data;
}

Error EditorExportPlatformWinrt::save_appx_file(void *p_userdata, const String &p_path, const Vector<uint8_t> &p_data, int p_file, int p_total) {

	AppxPackager *packager = (AppxPackager *)p_userdata;
	String dst_path = p_path.replace_first("res://", "game/");

	packager->add_file(dst_path, p_data.ptr(), p_data.size(), p_file, p_total, _should_compress_asset(p_path, p_data));

	return OK;
}

bool EditorExportPlatformWinrt::_should_compress_asset(const String &p_path, const Vector<uint8_t> &p_data) {

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
		// Trailer for easier processing
		NULL
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

bool EditorExportPlatformWinrt::_set(const StringName &p_name, const Variant &p_value) {

	String n = p_name;

	if (n == "architecture/target")
		arch = (Platform)((int)p_value);
	else if (n == "custom_package/debug")
		custom_debug_package = p_value;
	else if (n == "custom_package/release")
		custom_release_package = p_value;
	else if (n == "command_line/extra_args")
		cmdline = p_value;
	else if (n == "package/display_name")
		display_name = p_value;
	else if (n == "package/short_name")
		short_name = p_value;
	else if (n == "package/unique_name")
		unique_name = p_value;
	else if (n == "package/description")
		description = p_value;
	else if (n == "package/publisher")
		publisher = p_value;
	else if (n == "package/publisher_display_name")
		publisher_display_name = p_value;
	else if (n == "identity/product_guid")
		product_guid = p_value;
	else if (n == "identity/publisher_guid")
		publisher_guid = p_value;
	else if (n == "version/major")
		version_major = p_value;
	else if (n == "version/minor")
		version_minor = p_value;
	else if (n == "version/build")
		version_build = p_value;
	else if (n == "version/revision")
		version_revision = p_value;
	else if (n == "orientation/landscape")
		orientation_landscape = p_value;
	else if (n == "orientation/portrait")
		orientation_portrait = p_value;
	else if (n == "orientation/landscape_flipped")
		orientation_landscape_flipped = p_value;
	else if (n == "orientation/portrait_flipped")
		orientation_portrait_flipped = p_value;
	else if (n == "images/background_color")
		background_color = p_value;
	else if (n == "images/store_logo")
		store_logo = p_value;
	else if (n == "images/square44x44_logo")
		square44 = p_value;
	else if (n == "images/square71x71_logo")
		square71 = p_value;
	else if (n == "images/square150x150_logo")
		square150 = p_value;
	else if (n == "images/square310x310_logo")
		square310 = p_value;
	else if (n == "images/wide310x150_logo")
		wide310 = p_value;
	else if (n == "images/splash_screen")
		splash = p_value;
	else if (n == "tiles/show_name_on_square150x150")
		name_on_square150 = p_value;
	else if (n == "tiles/show_name_on_wide310x150")
		name_on_wide = p_value;
	else if (n == "tiles/show_name_on_square310x310")
		name_on_square310 = p_value;

	else if (n == "signing/sign")
		sign_package = p_value;
	else if (n == "signing/certificate_file")
		certificate_path = p_value;
	else if (n == "signing/certificate_password")
		certificate_pass = p_value;
	else if (n == "signing/certificate_algorithm")
		certificate_algo = (CertAlgorithm)(int)p_value;

	else if (n.begins_with("capabilities/")) {

		String what = n.get_slice("/", 1).replace("_", "");
		bool enable = p_value;

		if (array_has(uwp_capabilities, what.utf8().get_data())) {

			if (enable)
				capabilities.insert(what);
			else
				capabilities.erase(what);

		} else if (array_has(uwp_uap_capabilities, what.utf8().get_data())) {

			if (enable)
				uap_capabilities.insert(what);
			else
				uap_capabilities.erase(what);

		} else if (array_has(uwp_device_capabilites, what.utf8().get_data())) {

			if (enable)
				device_capabilities.insert(what);
			else
				device_capabilities.erase(what);
		}
	} else
		return false;

	return true;
}

bool EditorExportPlatformWinrt::_get(const StringName &p_name, Variant &r_ret) const {

	String n = p_name;

	if (n == "architecture/target")
		r_ret = (int)arch;
	else if (n == "custom_package/debug")
		r_ret = custom_debug_package;
	else if (n == "custom_package/release")
		r_ret = custom_release_package;
	else if (n == "command_line/extra_args")
		r_ret = cmdline;
	else if (n == "package/display_name")
		r_ret = display_name;
	else if (n == "package/short_name")
		r_ret = short_name;
	else if (n == "package/unique_name")
		r_ret = unique_name;
	else if (n == "package/description")
		r_ret = description;
	else if (n == "package/publisher")
		r_ret = publisher;
	else if (n == "package/publisher_display_name")
		r_ret = publisher_display_name;
	else if (n == "identity/product_guid")
		r_ret = product_guid;
	else if (n == "identity/publisher_guid")
		r_ret = publisher_guid;
	else if (n == "version/major")
		r_ret = version_major;
	else if (n == "version/minor")
		r_ret = version_minor;
	else if (n == "version/build")
		r_ret = version_build;
	else if (n == "version/revision")
		r_ret = version_revision;
	else if (n == "orientation/landscape")
		r_ret = orientation_landscape;
	else if (n == "orientation/portrait")
		r_ret = orientation_portrait;
	else if (n == "orientation/landscape_flipped")
		r_ret = orientation_landscape_flipped;
	else if (n == "orientation/portrait_flipped")
		r_ret = orientation_portrait_flipped;
	else if (n == "images/background_color")
		r_ret = background_color;
	else if (n == "images/store_logo")
		r_ret = store_logo;
	else if (n == "images/square44x44_logo")
		r_ret = square44;
	else if (n == "images/square71x71_logo")
		r_ret = square71;
	else if (n == "images/square150x150_logo")
		r_ret = square150;
	else if (n == "images/square310x310_logo")
		r_ret = square310;
	else if (n == "images/wide310x150_logo")
		r_ret = wide310;
	else if (n == "images/splash_screen")
		r_ret = splash;
	else if (n == "tiles/show_name_on_square150x150")
		r_ret = name_on_square150;
	else if (n == "tiles/show_name_on_wide310x150")
		r_ret = name_on_wide;
	else if (n == "tiles/show_name_on_square310x310")
		r_ret = name_on_square310;

	else if (n == "signing/sign")
		r_ret = sign_package;
	else if (n == "signing/certificate_file")
		r_ret = certificate_path;
	else if (n == "signing/certificate_password")
		r_ret = certificate_pass;
	else if (n == "signing/certificate_algorithm")
		r_ret = (int)certificate_algo;

	else if (n.begins_with("capabilities/")) {

		String what = n.get_slice("/", 1).replace("_", "");

		if (array_has(uwp_capabilities, what.utf8().get_data())) {

			r_ret = capabilities.has(what);

		} else if (array_has(uwp_uap_capabilities, what.utf8().get_data())) {

			r_ret = uap_capabilities.has(what);

		} else if (array_has(uwp_device_capabilites, what.utf8().get_data())) {

			r_ret = device_capabilities.has(what);
		}
	} else
		return false;

	return true;
}

void EditorExportPlatformWinrt::_get_property_list(List<PropertyInfo> *p_list) const {

	p_list->push_back(PropertyInfo(Variant::STRING, "custom_package/debug", PROPERTY_HINT_GLOBAL_FILE, "zip"));
	p_list->push_back(PropertyInfo(Variant::STRING, "custom_package/release", PROPERTY_HINT_GLOBAL_FILE, "zip"));

	p_list->push_back(PropertyInfo(Variant::INT, "architecture/target", PROPERTY_HINT_ENUM, "ARM,x86,x64"));

	p_list->push_back(PropertyInfo(Variant::STRING, "command_line/extra_args"));

	p_list->push_back(PropertyInfo(Variant::STRING, "package/display_name"));
	p_list->push_back(PropertyInfo(Variant::STRING, "package/short_name"));
	p_list->push_back(PropertyInfo(Variant::STRING, "package/unique_name"));
	p_list->push_back(PropertyInfo(Variant::STRING, "package/description"));
	p_list->push_back(PropertyInfo(Variant::STRING, "package/publisher"));
	p_list->push_back(PropertyInfo(Variant::STRING, "package/publisher_display_name"));

	p_list->push_back(PropertyInfo(Variant::STRING, "identity/product_guid"));
	p_list->push_back(PropertyInfo(Variant::STRING, "identity/publisher_guid"));

	p_list->push_back(PropertyInfo(Variant::INT, "version/major"));
	p_list->push_back(PropertyInfo(Variant::INT, "version/minor"));
	p_list->push_back(PropertyInfo(Variant::INT, "version/build"));
	p_list->push_back(PropertyInfo(Variant::INT, "version/revision"));

	p_list->push_back(PropertyInfo(Variant::BOOL, "orientation/landscape"));
	p_list->push_back(PropertyInfo(Variant::BOOL, "orientation/portrait"));
	p_list->push_back(PropertyInfo(Variant::BOOL, "orientation/landscape_flipped"));
	p_list->push_back(PropertyInfo(Variant::BOOL, "orientation/portrait_flipped"));

	p_list->push_back(PropertyInfo(Variant::STRING, "images/background_color"));
	p_list->push_back(PropertyInfo(Variant::OBJECT, "images/store_logo", PROPERTY_HINT_RESOURCE_TYPE, "ImageTexture"));
	p_list->push_back(PropertyInfo(Variant::OBJECT, "images/square44x44_logo", PROPERTY_HINT_RESOURCE_TYPE, "ImageTexture"));
	p_list->push_back(PropertyInfo(Variant::OBJECT, "images/square71x71_logo", PROPERTY_HINT_RESOURCE_TYPE, "ImageTexture"));
	p_list->push_back(PropertyInfo(Variant::OBJECT, "images/square150x150_logo", PROPERTY_HINT_RESOURCE_TYPE, "ImageTexture"));
	p_list->push_back(PropertyInfo(Variant::OBJECT, "images/square310x310_logo", PROPERTY_HINT_RESOURCE_TYPE, "ImageTexture"));
	p_list->push_back(PropertyInfo(Variant::OBJECT, "images/wide310x150_logo", PROPERTY_HINT_RESOURCE_TYPE, "ImageTexture"));
	p_list->push_back(PropertyInfo(Variant::OBJECT, "images/splash_screen", PROPERTY_HINT_RESOURCE_TYPE, "ImageTexture"));

	p_list->push_back(PropertyInfo(Variant::BOOL, "tiles/show_name_on_square150x150"));
	p_list->push_back(PropertyInfo(Variant::BOOL, "tiles/show_name_on_wide310x150"));
	p_list->push_back(PropertyInfo(Variant::BOOL, "tiles/show_name_on_square310x310"));

	p_list->push_back(PropertyInfo(Variant::BOOL, "signing/sign"));
	p_list->push_back(PropertyInfo(Variant::STRING, "signing/certificate_file", PROPERTY_HINT_GLOBAL_FILE, "pfx"));
	p_list->push_back(PropertyInfo(Variant::STRING, "signing/certificate_password"));
	p_list->push_back(PropertyInfo(Variant::INT, "signing/certificate_algorithm", PROPERTY_HINT_ENUM, "MD5,SHA1,SHA256"));

	// Capabilites
	const char **basic = uwp_capabilities;
	while (*basic) {
		p_list->push_back(PropertyInfo(Variant::BOOL, "capabilities/" + String(*basic).camelcase_to_underscore(false)));
		basic++;
	}

	const char **uap = uwp_uap_capabilities;
	while (*uap) {
		p_list->push_back(PropertyInfo(Variant::BOOL, "capabilities/" + String(*uap).camelcase_to_underscore(false)));
		uap++;
	}

	const char **device = uwp_device_capabilites;
	while (*device) {
		p_list->push_back(PropertyInfo(Variant::BOOL, "capabilities/" + String(*device).camelcase_to_underscore(false)));
		device++;
	}
}

bool EditorExportPlatformWinrt::can_export(String *r_error) const {

	String err;
	bool valid = true;

	if (!exists_export_template("winrt_x86_debug.zip") || !exists_export_template("winrt_x86_release.zip") || !exists_export_template("winrt_arm_debug.zip") || !exists_export_template("winrt_arm_release.zip") || !exists_export_template("winrt_x64_debug.zip") || !exists_export_template("winrt_x64_release.zip")) {
		valid = false;
		err += TTR("No export templates found.\nDownload and install export templates.") + "\n";
	}

	if (custom_debug_package != "" && !FileAccess::exists(custom_debug_package)) {
		valid = false;
		err += TTR("Custom debug package not found.") + "\n";
	}

	if (custom_release_package != "" && !FileAccess::exists(custom_release_package)) {
		valid = false;
		err += TTR("Custom release package not found.") + "\n";
	}

	if (!_valid_resource_name(unique_name)) {
		valid = false;
		err += TTR("Invalid unique name.") + "\n";
	}

	if (!_valid_guid(product_guid)) {
		valid = false;
		err += TTR("Invalid product GUID.") + "\n";
	}

	if (!_valid_guid(publisher_guid)) {
		valid = false;
		err += TTR("Invalid publisher GUID.") + "\n";
	}

	if (!_valid_bgcolor(background_color)) {
		valid = false;
		err += TTR("Invalid background color.") + "\n";
	}

	if (store_logo.is_valid() && !_valid_image(store_logo, 50, 50)) {
		valid = false;
		err += TTR("Invalid Store Logo image dimensions (should be 50x50).") + "\n";
	}

	if (square44.is_valid() && !_valid_image(square44, 44, 44)) {
		valid = false;
		err += TTR("Invalid square 44x44 logo image dimensions (should be 44x44).") + "\n";
	}

	if (square71.is_valid() && !_valid_image(square71, 71, 71)) {
		valid = false;
		err += TTR("Invalid square 71x71 logo image dimensions (should be 71x71).") + "\n";
	}

	if (square150.is_valid() && !_valid_image(square150, 150, 150)) {
		valid = false;
		err += TTR("Invalid square 150x150 logo image dimensions (should be 150x150).") + "\n";
	}

	if (square310.is_valid() && !_valid_image(square310, 310, 310)) {
		valid = false;
		err += TTR("Invalid square 310x310 logo image dimensions (should be 310x310).") + "\n";
	}

	if (wide310.is_valid() && !_valid_image(wide310, 310, 150)) {
		valid = false;
		err += TTR("Invalid wide 310x150 logo image dimensions (should be 310x150).") + "\n";
	}

	if (splash.is_valid() && !_valid_image(splash, 620, 300)) {
		valid = false;
		err += TTR("Invalid splash screen image dimensions (should be 620x300).") + "\n";
	}

#ifdef WINDOWS_ENABLED
	if (sign_package) {
		String signtool_path = EditorSettings::get_singleton()->get("winrt/signtool");
		if (signtool_path == String() || !FileAccess::exists(signtool_path)) {
			valid = false;
			err += TTR("Signtool executable not configured in editor settings.") + "\n";
		}

		String cert = EditorSettings::get_singleton()->get("winrt/debug_certificate");
		if (cert == String() || !FileAccess::exists(cert)) {
			valid = false;
			err += TTR("Debug certificate not configured in editor settings.") + "\n";
		}

		int cert_algo = EditorSettings::get_singleton()->get("winrt/debug_algorithm");
		if (cert == String() || !FileAccess::exists(cert)) {
			valid = false;
			err += TTR("Invalid certificate algorithm in editor settings.") + "\n";
		}
	}
#endif

	if (r_error)
		*r_error = err;

	return valid;
}

Error EditorExportPlatformWinrt::export_project(const String &p_path, bool p_debug, int p_flags) {

	String src_appx;

	EditorProgress ep("export", "Exporting for Windows Universal", 7);

	if (p_debug)
		src_appx = custom_debug_package;
	else
		src_appx = custom_release_package;

	if (src_appx == "") {
		String err;
		if (p_debug) {
			switch (arch) {
				case X86: {
					src_appx = find_export_template("winrt_x86_debug.zip", &err);
					break;
				}
				case X64: {
					src_appx = find_export_template("winrt_x64_debug.zip", &err);
					break;
				}
				case ARM: {
					src_appx = find_export_template("winrt_arm_debug.zip", &err);
					break;
				}
			}
		} else {
			switch (arch) {
				case X86: {
					src_appx = find_export_template("winrt_x86_release.zip", &err);
					break;
				}
				case X64: {
					src_appx = find_export_template("winrt_x64_release.zip", &err);
					break;
				}
				case ARM: {
					src_appx = find_export_template("winrt_arm_release.zip", &err);
					break;
				}
			}
		}
		if (src_appx == "") {
			EditorNode::add_io_error(err);
			return ERR_FILE_NOT_FOUND;
		}
	}

	Error err = OK;

	FileAccess *fa_pack = FileAccess::open(p_path, FileAccess::WRITE, &err);
	ERR_FAIL_COND_V(err != OK, ERR_CANT_CREATE);

	AppxPackager packager;
	packager.init(fa_pack);

	FileAccess *src_f = NULL;
	zlib_filefunc_def io = zipio_create_io_from_file(&src_f);

	ep.step("Creating package...", 0);

	unzFile pkg = unzOpen2(src_appx.utf8().get_data(), &io);

	if (!pkg) {

		EditorNode::add_io_error("Could not find template appx to export:\n" + src_appx);
		return ERR_FILE_NOT_FOUND;
	}

	int ret = unzGoToFirstFile(pkg);

	ep.step("Copying template files...", 1);

	EditorNode::progress_add_task("template_files", "Template files", 100);
	packager.set_progress_task("template_files");

	int template_files_amount = 9;
	int template_file_no = 1;

	while (ret == UNZ_OK) {

		// get file name
		unz_file_info info;
		char fname[16834];
		ret = unzGetCurrentFileInfo(pkg, &info, fname, 16834, NULL, 0, NULL, 0);

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

			data = _get_image_data(path);
			if (data.size() > 0) do_read = false;
		}

		//read
		if (do_read) {
			data.resize(info.uncompressed_size);
			unzOpenCurrentFile(pkg);
			unzReadCurrentFile(pkg, data.ptr(), data.size());
			unzCloseCurrentFile(pkg);
		}

		if (path == "AppxManifest.xml") {

			data = _fix_manifest(data, p_flags & (EXPORT_DUMB_CLIENT | EXPORT_REMOTE_DEBUG));
		}

		print_line("ADDING: " + path);

		packager.add_file(path, data.ptr(), data.size(), template_file_no++, template_files_amount, _should_compress_asset(path, data));

		ret = unzGoToNextFile(pkg);
	}

	EditorNode::progress_end_task("template_files");

	ep.step("Creating command line...", 2);

	Vector<String> cl = cmdline.strip_edges().split(" ");
	for (int i = 0; i < cl.size(); i++) {
		if (cl[i].strip_edges().length() == 0) {
			cl.remove(i);
			i--;
		}
	}

	if (!(p_flags & EXPORT_DUMB_CLIENT)) {
		cl.push_back("-path");
		cl.push_back("game");
	}

	gen_export_flags(cl, p_flags);

	// Command line file
	Vector<uint8_t> clf;

	// Argc
	clf.resize(4);
	encode_uint32(cl.size(), clf.ptr());

	for (int i = 0; i < cl.size(); i++) {

		CharString txt = cl[i].utf8();
		int base = clf.size();
		clf.resize(base + 4 + txt.length());
		encode_uint32(txt.length(), &clf[base]);
		copymem(&clf[base + 4], txt.ptr(), txt.length());
		print_line(itos(i) + " param: " + cl[i]);
	}

	packager.add_file("__cl__.cl", clf.ptr(), clf.size(), -1, -1, false);

	ep.step("Adding project files...", 3);

	EditorNode::progress_add_task("project_files", "Project Files", 100);
	packager.set_progress_task("project_files");

	err = export_project_files(save_appx_file, &packager, false);

	EditorNode::progress_end_task("project_files");

	ep.step("Closing package...", 7);

	unzClose(pkg);

	packager.finish();

#ifdef WINDOWS_ENABLED
	if (sign_package) {
		// Sign with signtool
		String signtool_path = EditorSettings::get_singleton()->get("winrt/signtool");
		if (signtool_path == String()) {
			return OK;
		}

		if (!FileAccess::exists(signtool_path)) {
			ERR_PRINTS("Could not find signtool executable at " + signtool_path + ", aborting.");
			return ERR_FILE_NOT_FOUND;
		}

		static String algs[] = { "MD5", "SHA1", "SHA256" };

		String cert_path = EditorSettings::get_singleton()->get("winrt/debug_certificate");
		String cert_pass = EditorSettings::get_singleton()->get("winrt/debug_password");
		int cert_alg = EditorSettings::get_singleton()->get("winrt/debug_algorithm");

		if (!p_debug) {
			cert_path = certificate_path;
			cert_pass = certificate_pass;
			cert_alg = certificate_algo;
		}

		if (!FileAccess::exists(cert_path)) {
			ERR_PRINTS("Could not find certificate file at " + cert_path + ", aborting.");
			return ERR_FILE_NOT_FOUND;
		}

		if (cert_alg < 0 || cert_alg > 2) {
			ERR_PRINTS("Invalid certificate algorithm " + itos(cert_alg) + ", aborting.");
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
	}
#endif // WINDOWS_ENABLED

	return OK;
}

EditorExportPlatformWinrt::EditorExportPlatformWinrt() {

	Image img(_winrt_logo);
	logo = Ref<ImageTexture>(memnew(ImageTexture));
	logo->create_from_image(img);

	custom_release_package = "";
	custom_debug_package = "";

	arch = X86;

	display_name = "";
	short_name = "Godot";
	unique_name = "Godot.Engine";
	description = "Godot Engine";
	publisher = "CN=GodotEngine";
	publisher_display_name = "Godot Engine";

	product_guid = "00000000-0000-0000-0000-000000000000";
	publisher_guid = "00000000-0000-0000-0000-000000000000";

	version_major = 1;
	version_minor = 0;
	version_build = 0;
	version_revision = 0;

	orientation_landscape = true;
	orientation_portrait = true;
	orientation_landscape_flipped = true;
	orientation_portrait_flipped = true;

	background_color = "transparent";

	name_on_square150 = false;
	name_on_square310 = false;
	name_on_wide = false;

	sign_package = false;
	certificate_path = "";
	certificate_pass = "";
}

EditorExportPlatformWinrt::~EditorExportPlatformWinrt() {}

void register_winrt_exporter() {

#ifdef WINDOWS_ENABLED
	EDITOR_DEF("winrt/signtool", "");
	EditorSettings::get_singleton()->add_property_hint(PropertyInfo(Variant::STRING, "winrt/signtool", PROPERTY_HINT_GLOBAL_FILE, "*.exe"));
	EDITOR_DEF("winrt/debug_certificate", "");
	EditorSettings::get_singleton()->add_property_hint(PropertyInfo(Variant::STRING, "winrt/debug_certificate", PROPERTY_HINT_GLOBAL_FILE, "*.pfx"));
	EDITOR_DEF("winrt/debug_password", "");
	EDITOR_DEF("winrt/debug_algorithm", 2); // SHA256 is the default
	EditorSettings::get_singleton()->add_property_hint(PropertyInfo(Variant::INT, "winrt/debug_algorithm", PROPERTY_HINT_ENUM, "MD5,SHA1,SHA256"));
#endif // WINDOWS_ENABLED

	Ref<EditorExportPlatformWinrt> exporter = Ref<EditorExportPlatformWinrt>(memnew(EditorExportPlatformWinrt));
	EditorImportExport::get_singleton()->add_export_platform(exporter);
}
