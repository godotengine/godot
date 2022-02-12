/*************************************************************************/
/*  app_packager.cpp                                                     */
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

#include "app_packager.h"

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

	const String &tmp_blockmap_file_path = EditorPaths::get_singleton()->get_cache_dir().plus_file("tmpblockmap.xml");
	make_block_map(tmp_blockmap_file_path);

	FileAccess *blockmap_file = FileAccess::open(tmp_blockmap_file_path, FileAccess::READ);
	Vector<uint8_t> blockmap_buffer;
	blockmap_buffer.resize(blockmap_file->get_length());

	blockmap_file->get_buffer(blockmap_buffer.ptrw(), blockmap_buffer.size());

	add_file("AppxBlockMap.xml", blockmap_buffer.ptr(), blockmap_buffer.size(), -1, -1, true);

	blockmap_file->close();
	memdelete(blockmap_file);

	// Add content types

	EditorNode::progress_task_step("export", "Setting content types...", 5);

	const String &tmp_content_types_file_path = EditorPaths::get_singleton()->get_cache_dir().plus_file("tmpcontenttypes.xml");
	make_content_types(tmp_content_types_file_path);

	FileAccess *types_file = FileAccess::open(tmp_content_types_file_path, FileAccess::READ);
	Vector<uint8_t> types_buffer;
	types_buffer.resize(types_file->get_length());

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
