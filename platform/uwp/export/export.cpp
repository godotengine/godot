/*************************************************************************/
/*  export.cpp                                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
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

/*************************************************************************
 * The code for signing the package was ported from fb-util-for-appx
 * available at https://github.com/facebook/fb-util-for-appx
 * and distributed also under the following license:

BSD License

For fb-util-for-appx software

Copyright (c) 2016, Facebook, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

 * Neither the name Facebook nor the names of its contributors may be used to
   endorse or promote products derived from this software without specific
   prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*************************************************************************/

#if 0
#include "version.h"
#include "export.h"
#include "object.h"
#include "tools/editor/editor_import_export.h"
#include "tools/editor/editor_node.h"
#include "platform/uwp/logo.h"
#include "os/file_access.h"
#include "io/zip.h"
#include "io/unzip.h"
#include "io/zip_io.h"
#include "io/sha256.h"
#include "io/base64.h"
#include "bind/core_bind.h"
#include "globals.h"
#include "io/marshalls.h"

#include <zlib.h>

// Capabilities
static const char* uwp_capabilities[] = {
	"allJoyn",
	"codeGeneration",
	"internetClient",
	"internetClientServer",
	"privateNetworkClientServer",
	NULL
};
static const char* uwp_uap_capabilities[] = {
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
static const char* uwp_device_capabilites[] = {
	"bluetooth",
	"location",
	"microphone",
	"proximity",
	"webcam",
	NULL
};

#ifdef OPENSSL_ENABLED
#include <openssl/bio.h>
#include <openssl/asn1.h>
#include <openssl/pkcs7.h>
#include <openssl/pkcs12.h>
#include <openssl/err.h>
#include <openssl/asn1t.h>
#include <openssl/x509.h>
#include <openssl/ossl_typ.h>

namespace asn1 {
	// https://msdn.microsoft.com/en-us/gg463180.aspx

	struct SPCStatementType {
		ASN1_OBJECT *type;
	};
	DECLARE_ASN1_FUNCTIONS(SPCStatementType)

	struct SPCSpOpusInfo {
		ASN1_TYPE *programName;
		ASN1_TYPE *moreInfo;
	};
	DECLARE_ASN1_FUNCTIONS(SPCSpOpusInfo)

	struct DigestInfo {
		X509_ALGOR *digestAlgorithm;
		ASN1_OCTET_STRING *digest;
	};
	DECLARE_ASN1_FUNCTIONS(DigestInfo)

	struct SPCAttributeTypeAndOptionalValue {
		ASN1_OBJECT *type;
		ASN1_TYPE *value;  // SPCInfoValue
	};
	DECLARE_ASN1_FUNCTIONS(SPCAttributeTypeAndOptionalValue)

	// Undocumented.
	struct SPCInfoValue {
		ASN1_INTEGER *i1;
		ASN1_OCTET_STRING *s1;
		ASN1_INTEGER *i2;
		ASN1_INTEGER *i3;
		ASN1_INTEGER *i4;
		ASN1_INTEGER *i5;
		ASN1_INTEGER *i6;
	};
	DECLARE_ASN1_FUNCTIONS(SPCInfoValue)

	struct SPCIndirectDataContent {
		SPCAttributeTypeAndOptionalValue *data;
		DigestInfo *messageDigest;
	};
	DECLARE_ASN1_FUNCTIONS(SPCIndirectDataContent)

	IMPLEMENT_ASN1_FUNCTIONS(SPCIndirectDataContent)
		ASN1_SEQUENCE(SPCIndirectDataContent) = {
		ASN1_SIMPLE(SPCIndirectDataContent, data,
		SPCAttributeTypeAndOptionalValue),
		ASN1_SIMPLE(SPCIndirectDataContent, messageDigest, DigestInfo),
	} ASN1_SEQUENCE_END(SPCIndirectDataContent)

	IMPLEMENT_ASN1_FUNCTIONS(SPCAttributeTypeAndOptionalValue)
		ASN1_SEQUENCE(SPCAttributeTypeAndOptionalValue) = {
		ASN1_SIMPLE(SPCAttributeTypeAndOptionalValue, type,
		ASN1_OBJECT),
		ASN1_OPT(SPCAttributeTypeAndOptionalValue, value, ASN1_ANY),
	} ASN1_SEQUENCE_END(SPCAttributeTypeAndOptionalValue)

	IMPLEMENT_ASN1_FUNCTIONS(SPCInfoValue)
		ASN1_SEQUENCE(SPCInfoValue) = {
		ASN1_SIMPLE(SPCInfoValue, i1, ASN1_INTEGER),
		ASN1_SIMPLE(SPCInfoValue, s1, ASN1_OCTET_STRING),
		ASN1_SIMPLE(SPCInfoValue, i2, ASN1_INTEGER),
		ASN1_SIMPLE(SPCInfoValue, i3, ASN1_INTEGER),
		ASN1_SIMPLE(SPCInfoValue, i4, ASN1_INTEGER),
		ASN1_SIMPLE(SPCInfoValue, i5, ASN1_INTEGER),
		ASN1_SIMPLE(SPCInfoValue, i6, ASN1_INTEGER),
	} ASN1_SEQUENCE_END(SPCInfoValue)

	IMPLEMENT_ASN1_FUNCTIONS(DigestInfo)
		ASN1_SEQUENCE(DigestInfo) = {
		ASN1_SIMPLE(DigestInfo, digestAlgorithm, X509_ALGOR),
		ASN1_SIMPLE(DigestInfo, digest, ASN1_OCTET_STRING),
	} ASN1_SEQUENCE_END(DigestInfo)

	ASN1_SEQUENCE(SPCSpOpusInfo) = {
		ASN1_OPT(SPCSpOpusInfo, programName, ASN1_ANY),
		ASN1_OPT(SPCSpOpusInfo, moreInfo, ASN1_ANY),
	} ASN1_SEQUENCE_END(SPCSpOpusInfo)
	IMPLEMENT_ASN1_FUNCTIONS(SPCSpOpusInfo)

		ASN1_SEQUENCE(SPCStatementType) = {
		ASN1_SIMPLE(SPCStatementType, type, ASN1_OBJECT),
	} ASN1_SEQUENCE_END(SPCStatementType)
	IMPLEMENT_ASN1_FUNCTIONS(SPCStatementType)
}

class EncodedASN1 {

	uint8_t* i_data;
	size_t i_size;

	EncodedASN1(uint8_t** p_data, size_t p_size) {

		i_data = *p_data;
		i_size = p_size;
	}

public:

	template <typename T, int(*TEncode)(T *, uint8_t **)>
	static EncodedASN1 FromItem(T *item) {
		uint8_t *dataRaw = NULL;
		int size = TEncode(item, &dataRaw);

		return EncodedASN1(&dataRaw, size);
	}

	const uint8_t *data() const {
		return i_data;
	}

	size_t size() const {
		return i_size;
	}

	// Assumes the encoded ASN.1 represents a SEQUENCE and puts it into
	// an ASN1_STRING.
	//
	// The returned object holds a copy of this object's data.
	ASN1_STRING* ToSequenceString() {
		ASN1_STRING* string = ASN1_STRING_new();
		if (!string) {
			return NULL;
		}
		if (!ASN1_STRING_set(string, i_data, i_size)) {
			return NULL;
		}
		return string;
	}

	// Assumes the encoded ASN.1 represents a SEQUENCE and puts it into
	// an ASN1_TYPE.
	//
	// The returned object holds a copy of this object's data.
	ASN1_TYPE* ToSequenceType() {
		ASN1_STRING* string = ToSequenceString();
		ASN1_TYPE* type = ASN1_TYPE_new();
		if (!type) {
			return NULL;
		}
		type->type = V_ASN1_SEQUENCE;
		type->value.sequence = string;
		return type;
	}

};

#endif // OPENSSL_ENABLED

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

	String hash_block(uint8_t* p_block_data, size_t p_block_len);

	void make_block_map();
	void make_content_types();


	_FORCE_INLINE_ unsigned int buf_put_int16(uint16_t p_val, uint8_t * p_buf) {
		for (int i = 0; i < 2; i++) {
			*p_buf++ = (p_val >> (i * 8)) & 0xFF;
		}
		return 2;
	}

	_FORCE_INLINE_ unsigned int buf_put_int32(uint32_t p_val, uint8_t * p_buf) {
		for (int i = 0; i < 4; i++) {
			*p_buf++ = (p_val >> (i * 8)) & 0xFF;
		}
		return 4;
	}

	_FORCE_INLINE_ unsigned int buf_put_int64(uint64_t p_val, uint8_t * p_buf) {
		for (int i = 0; i < 8; i++) {
			*p_buf++ = (p_val >> (i * 8)) & 0xFF;
		}
		return 8;
	}

	_FORCE_INLINE_ unsigned int buf_put_string(String p_val, uint8_t * p_buf) {
		for (int i = 0; i < p_val.length(); i++) {
			*p_buf++ = p_val.utf8().get(i);
		}
		return p_val.length();
	}

	Vector<uint8_t> make_file_header(FileMeta p_file_meta);
	void store_central_dir_header(const FileMeta p_file, bool p_do_hash = true);
	Vector<uint8_t> make_end_of_central_record();

	String content_type(String p_extension);

#ifdef OPENSSL_ENABLED

	// Signing methods and structs:

	String certificate_path;
	String certificate_pass;
	bool sign_package;

	struct CertFile {

		EVP_PKEY* private_key;
		X509* certificate;
	};

	SHA256_CTX axpc_context; // SHA256 context for ZIP file entries
	SHA256_CTX axcd_context; // SHA256 context for ZIP directory entries

	struct AppxDigests {

		uint8_t axpc[SHA256_DIGEST_LENGTH]; // ZIP file entries
		uint8_t axcd[SHA256_DIGEST_LENGTH]; // ZIP directory entry
		uint8_t axct[SHA256_DIGEST_LENGTH]; // Content types XML
		uint8_t axbm[SHA256_DIGEST_LENGTH]; // Block map XML
		uint8_t axci[SHA256_DIGEST_LENGTH]; // Code Integrity file (optional)
	};

	CertFile cert_file;
	AppxDigests digests;

	void MakeSPCInfoValue(asn1::SPCInfoValue &info);
	Error MakeIndirectDataContent(asn1::SPCIndirectDataContent &idc);
	Error add_attributes(PKCS7_SIGNER_INFO *signerInfo);
	void make_digests();
	void write_digest(Vector<uint8_t> &p_out_buffer);

	Error openssl_error(unsigned long p_err);
	Error read_cert_file(const String &p_path, const String &p_password, CertFile* p_out_cf);
	Error sign(const CertFile &p_cert, const AppxDigests &digests, PKCS7* p_out_signature);

#endif // OPENSSL_ENABLED

public:

	enum SignOption {

		SIGN,
		DONT_SIGN,
	};

	void set_progress_task(String p_task) { progress_task = p_task; }
	void init(FileAccess* p_fa, SignOption p_sign, String &p_certificate_path, String &p_certificate_password);
	void add_file(String p_file_name, const uint8_t* p_buffer, size_t p_len, int p_file_no, int p_total_files, bool p_compress = false);
	void finish();

	AppxPackager();
	~AppxPackager();
};

class EditorExportPlatformUWP : public EditorExportPlatform {

	GDCLASS(EditorExportPlatformUWP, EditorExportPlatform);

	Ref<ImageTexture> logo;

	enum Platform {
		ARM,
		X86,
		X64
	} arch;

	bool is_debug;

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

	_FORCE_INLINE_ bool array_has(const char** p_array, const char* p_value) const {
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

	static Error save_appx_file(void *p_userdata, const String& p_path, const Vector<uint8_t>& p_data, int p_file, int p_total);
	static bool _should_compress_asset(const String& p_path, const Vector<uint8_t>& p_data);

protected:

	bool _set(const StringName& p_name, const Variant& p_value);
	bool _get(const StringName& p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;

public:

	virtual String get_name() const { return "Windows Universal"; }
	virtual ImageCompression get_image_compression() const { return IMAGE_COMPRESSION_ETC1; }
	virtual Ref<Texture> get_logo() const { return logo; }

	virtual bool can_export(String *r_error = NULL) const;
	virtual String get_binary_extension() const { return "appx"; }

	virtual Error export_project(const String& p_path, bool p_debug, int p_flags = 0);

	EditorExportPlatformUWP();
	~EditorExportPlatformUWP();
};


///////////////////////////////////////////////////////////////////////////

String AppxPackager::hash_block(uint8_t * p_block_data, size_t p_block_len) {

	char hash[32];
	char base64[45];

	sha256_context ctx;
	sha256_init(&ctx);
	sha256_hash(&ctx, p_block_data, p_block_len);
	sha256_done(&ctx, (uint8_t*)hash);

	base64_encode(base64, hash, 32);
	base64[44] = '\0';

	return String(base64);
}

void AppxPackager::make_block_map() {

	FileAccess* tmp_file = FileAccess::open(tmp_blockmap_file_path, FileAccess::WRITE);

	tmp_file->store_string("<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>");
	tmp_file->store_string("<BlockMap xmlns=\"http://schemas.microsoft.com/appx/2010/blockmap\" HashMethod=\"http://www.w3.org/2001/04/xmlenc#sha256\">");

	for (int i = 0; i < file_metadata.size(); i++) {

		FileMeta file = file_metadata[i];

		tmp_file->store_string(
			"<File Name=\"" + file.name.replace("/", "\\")
			+ "\" Size=\"" + itos(file.uncompressed_size)
			+ "\" LfhSize=\"" + itos(file.lfh_size) + "\">");


		for (int j = 0; j < file.hashes.size(); j++) {

			tmp_file->store_string("<Block Hash=\""
				+ file.hashes[j].base64_hash + "\" ");
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

	FileAccess* tmp_file = FileAccess::open(tmp_content_types_file_path, FileAccess::WRITE);

	tmp_file->store_string("<?xml version=\"1.0\" encoding=\"UTF-8\"?>");
	tmp_file->store_string("<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">");

	Map<String, String>	types;

	for (int i = 0; i < file_metadata.size(); i++) {

		String ext = file_metadata[i].name.get_extension();

		if (types.has(ext)) continue;

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
	tmp_file = NULL;
}

Vector<uint8_t> AppxPackager::make_file_header(FileMeta p_file_meta) {

	Vector<uint8_t> buf;
	buf.resize(BASE_FILE_HEADER_SIZE + p_file_meta.name.length());

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
	offs += buf_put_int16(p_file_meta.name.length(), &buf[offs]);

	// Extra data length
	offs += buf_put_int16(0, &buf[offs]);

	// File name
	offs += buf_put_string(p_file_meta.name, &buf[offs]);

	// Done!
	return buf;
}

void AppxPackager::store_central_dir_header(const FileMeta p_file, bool p_do_hash) {

	Vector<uint8_t> &buf = central_dir_data;
	int offs = buf.size();
	buf.resize(buf.size() + BASE_CENTRAL_DIR_SIZE + p_file.name.length());


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
	offs += buf_put_int16(p_file.name.length(), &buf[offs]);

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
	offs += buf_put_string(p_file.name, &buf[offs]);

#ifdef OPENSSL_ENABLED
	// Calculate the hash for signing
	if (p_do_hash)
		SHA256_Update(&axcd_context, buf.ptr(), buf.size());
#endif // OPENSSL_ENABLED

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

void AppxPackager::init(FileAccess * p_fa, SignOption p_sign, String &p_certificate_path, String &p_certificate_password) {

	package = p_fa;
	central_dir_offset = 0;
	end_of_central_dir_offset = 0;
	tmp_blockmap_file_path = EditorSettings::get_singleton()->get_settings_path() + "/tmp/tmpblockmap.xml";
	tmp_content_types_file_path = EditorSettings::get_singleton()->get_settings_path() + "/tmp/tmpcontenttypes.xml";
#ifdef OPENSSL_ENABLED
	certificate_path = p_certificate_path;
	certificate_pass = p_certificate_password;
	sign_package = p_sign == SIGN;
	SHA256_Init(&axpc_context);
	SHA256_Init(&axcd_context);
#endif // OPENSSL_ENABLED
}

void AppxPackager::add_file(String p_file_name, const uint8_t * p_buffer, size_t p_len, int p_file_no, int p_total_files, bool p_compress) {

	if (p_file_no >= 1 && p_total_files >= 1) {
		EditorNode::progress_task_step(progress_task, "File: " + p_file_name, (p_file_no * 100) / p_total_files);
	}

	bool do_hash = p_file_name != "AppxSignature.p7x";

	FileMeta meta;
	meta.name = p_file_name;
	meta.uncompressed_size = p_len;
	meta.compressed_size = p_len;
	meta.compressed = p_compress;
	meta.zip_offset = package->get_pos();

	Vector<uint8_t> file_buffer;

	// Data for compression
	z_stream strm;
	FileAccess* strm_f = NULL;
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
#ifdef OPENSSL_ENABLED
			if (do_hash)
				SHA256_Update(&axpc_context, strm_out.ptr(), strm.total_out - total_out_before);
#endif // OPENSSL_ENABLED

		} else {
			bh.compressed_size = block_size;
			//package->store_buffer(strm_in.ptr(), block_size);
			int start = file_buffer.size();
			file_buffer.resize(file_buffer.size() + block_size);
			for (int i = 0; i < bh.compressed_size; i++)
				file_buffer[start + i] = strm_in[i];
#ifdef OPENSSL_ENABLED
			if (do_hash)
				SHA256_Update(&axpc_context, strm_in.ptr(), block_size);
#endif // OPENSSL_ENABLED
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
#ifdef OPENSSL_ENABLED
		if (do_hash)
			SHA256_Update(&axpc_context, strm_out.ptr(), strm.total_out - total_out_before);
#endif // OPENSSL_ENABLED

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

#ifdef OPENSSL_ENABLED
	// Hash the data for signing
	if (do_hash) {
		SHA256_Update(&axpc_context, file_header.ptr(), file_header.size());
		SHA256_Update(&axpc_context, file_buffer.ptr(), file_buffer.size());
	}
#endif // OPENSSL_ENABLED

	// Store the header and file;
	package->store_buffer(file_header.ptr(), file_header.size());
	package->store_buffer(file_buffer.ptr(), file_buffer.size());

	file_metadata.push_back(meta);
}

void AppxPackager::finish() {

	// Create and add block map file
	EditorNode::progress_task_step("export", "Creating block map...", 4);

	make_block_map();
	FileAccess* blockmap_file = FileAccess::open(tmp_blockmap_file_path, FileAccess::READ);
	Vector<uint8_t> blockmap_buffer;
	blockmap_buffer.resize(blockmap_file->get_len());

	blockmap_file->get_buffer(blockmap_buffer.ptr(), blockmap_buffer.size());

#ifdef OPENSSL_ENABLED
	// Hash the file for signing
	if (sign_package) {
		SHA256_CTX axbm_context;
		SHA256_Init(&axbm_context);
		SHA256_Update(&axbm_context, blockmap_buffer.ptr(), blockmap_buffer.size());
		SHA256_Final(digests.axbm, &axbm_context);
	}
#endif // OPENSSL_ENABLED

	add_file("AppxBlockMap.xml", blockmap_buffer.ptr(), blockmap_buffer.size(), -1, -1, true);

	blockmap_file->close();
	memdelete(blockmap_file);
	blockmap_file = NULL;

	// Add content types
	EditorNode::progress_task_step("export", "Setting content types...", 5);
	make_content_types();

	FileAccess* types_file = FileAccess::open(tmp_content_types_file_path, FileAccess::READ);
	Vector<uint8_t> types_buffer;
	types_buffer.resize(types_file->get_len());

	types_file->get_buffer(types_buffer.ptr(), types_buffer.size());

#ifdef OPENSSL_ENABLED
	if (sign_package) {
		// Hash the file for signing
		SHA256_CTX axct_context;
		SHA256_Init(&axct_context);
		SHA256_Update(&axct_context, types_buffer.ptr(), types_buffer.size());
		SHA256_Final(digests.axct, &axct_context);
	}
#endif // OPENSSL_ENABLED

	add_file("[Content_Types].xml", types_buffer.ptr(), types_buffer.size(), -1, -1, true);

	types_file->close();
	memdelete(types_file);
	types_file = NULL;

	// Pre-process central directory before signing
	for (int i = 0; i < file_metadata.size(); i++) {
		store_central_dir_header(file_metadata[i]);
	}

#ifdef OPENSSL_ENABLED
	// Create the signature file
	if (sign_package) {

		Error err = read_cert_file(certificate_path, certificate_pass, &cert_file);

		if (err != OK) {
			EditorNode::add_io_error(TTR("Couldn't read the certficate file. Are the path and password both correct?"));
			package->close();
			memdelete(package);
			package = NULL;
			return;
		}


		// Make a temp end of the zip for hashing
		central_dir_offset = package->get_pos();
		end_of_central_dir_offset = central_dir_offset + central_dir_data.size();
		Vector<uint8_t> zip_end_dir = make_end_of_central_record();

		// Hash the end directory
		SHA256_Update(&axcd_context, zip_end_dir.ptr(), zip_end_dir.size());

		// Finish the hashes
		make_digests();

		PKCS7* signature = PKCS7_new();
		if (!signature) {
			EditorNode::add_io_error(TTR("Error creating the signature object."));
			package->close();
			memdelete(package);
			package = NULL;
			return;
		}

		err = sign(cert_file, digests, signature);

		if (err != OK) {
			EditorNode::add_io_error(TTR("Error creating the package signature."));
			package->close();
			memdelete(package);
			package = NULL;
			return;
		}

		// Read the signature as bytes
		BIO* bio_out = BIO_new(BIO_s_mem());
		i2d_PKCS7_bio(bio_out, signature);

		BIO_flush(bio_out);

		uint8_t* bio_ptr;
		size_t bio_size = BIO_get_mem_data(bio_out, &bio_ptr);

		// Create the signature buffer with magic number
		Vector<uint8_t> signature_file;
		signature_file.resize(4 + bio_size);
		buf_put_int32(P7X_SIGNATURE, signature_file.ptr());
		for (int i = 0; i < bio_size; i++)
			signature_file[i + 4] = bio_ptr[i];

		// Add the signature to the package
		add_file("AppxSignature.p7x", signature_file.ptr(), signature_file.size(), -1, -1, true);

		// Add central directory entry
		store_central_dir_header(file_metadata[file_metadata.size() - 1], false);
	}
#endif // OPENSSL_ENABLED


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

#ifdef OPENSSL_ENABLED
// https://support.microsoft.com/en-us/kb/287547
const char SPC_INDIRECT_DATA_OBJID[] = "1.3.6.1.4.1.311.2.1.4";
const char SPC_STATEMENT_TYPE_OBJID[] = "1.3.6.1.4.1.311.2.1.11";
const char SPC_SP_OPUS_INFO_OBJID[] = "1.3.6.1.4.1.311.2.1.12";
const char SPC_SIPINFO_OBJID[] = "1.3.6.1.4.1.311.2.1.30";
#endif // OPENSSL_ENABLED

AppxPackager::AppxPackager() {}

AppxPackager::~AppxPackager() {}


////////////////////////////////////////////////////////////////////

#ifdef OPENSSL_ENABLED
Error AppxPackager::openssl_error(unsigned long p_err) {

	ERR_load_crypto_strings();

	char buffer[256];
	ERR_error_string_n(p_err, buffer, sizeof(buffer));

	String err(buffer);

	ERR_EXPLAIN(err);
	ERR_FAIL_V(FAILED);
}

void AppxPackager::MakeSPCInfoValue(asn1::SPCInfoValue &info) {

	// I have no idea what these numbers mean.
	static uint8_t s1Magic[] = {
		0x4B, 0xDF, 0xC5, 0x0A, 0x07, 0xCE, 0xE2, 0x4D,
		0xB7, 0x6E, 0x23, 0xC8, 0x39, 0xA0, 0x9F, 0xD1,
	};
	ASN1_INTEGER_set(info.i1, 0x01010000);
	ASN1_OCTET_STRING_set(info.s1, s1Magic, sizeof(s1Magic));
	ASN1_INTEGER_set(info.i2, 0x00000000);
	ASN1_INTEGER_set(info.i3, 0x00000000);
	ASN1_INTEGER_set(info.i4, 0x00000000);
	ASN1_INTEGER_set(info.i5, 0x00000000);
	ASN1_INTEGER_set(info.i6, 0x00000000);
}

Error AppxPackager::MakeIndirectDataContent(asn1::SPCIndirectDataContent &idc) {

	using namespace asn1;

	ASN1_TYPE* algorithmParameter = ASN1_TYPE_new();
	if (!algorithmParameter) {
		return openssl_error(ERR_peek_last_error());
	}
	algorithmParameter->type = V_ASN1_NULL;

	SPCInfoValue* infoValue = SPCInfoValue_new();
	if (!infoValue) {
		return openssl_error(ERR_peek_last_error());
	}
	MakeSPCInfoValue(*infoValue);

	ASN1_TYPE* value =
			EncodedASN1::FromItem<asn1::SPCInfoValue,
			asn1::i2d_SPCInfoValue>(infoValue)
			.ToSequenceType();

	{
		Vector<uint8_t> digest;
		write_digest(digest);
		if (!ASN1_OCTET_STRING_set(idc.messageDigest->digest,
			digest.ptr(), digest.size())) {

			return openssl_error(ERR_peek_last_error());
		}
	}

	idc.data->type = OBJ_txt2obj(SPC_SIPINFO_OBJID, 1);
	idc.data->value = value;
	idc.messageDigest->digestAlgorithm->algorithm = OBJ_nid2obj(NID_sha256);
	idc.messageDigest->digestAlgorithm->parameter = algorithmParameter;

	return OK;
}

Error AppxPackager::add_attributes(PKCS7_SIGNER_INFO * p_signer_info) {

	// Add opus attribute
	asn1::SPCSpOpusInfo* opus = asn1::SPCSpOpusInfo_new();
	if (!opus) return openssl_error(ERR_peek_last_error());

	ASN1_STRING* opus_value =
			EncodedASN1::FromItem<asn1::SPCSpOpusInfo,
			asn1::i2d_SPCSpOpusInfo>(opus)
			.ToSequenceString();

	if (!PKCS7_add_signed_attribute(
		p_signer_info,
		OBJ_txt2nid(SPC_SP_OPUS_INFO_OBJID),
		V_ASN1_SEQUENCE,
		opus_value
	)) {

		asn1::SPCSpOpusInfo_free(opus);

		ASN1_STRING_free(opus_value);
		return openssl_error(ERR_peek_last_error());
	}

	// Add content type attribute
	if (!PKCS7_add_signed_attribute(
		p_signer_info,
		NID_pkcs9_contentType,
		V_ASN1_OBJECT,
		OBJ_txt2obj(SPC_INDIRECT_DATA_OBJID, 1)
	)) {

		asn1::SPCSpOpusInfo_free(opus);
		ASN1_STRING_free(opus_value);
		return openssl_error(ERR_peek_last_error());
	}

	// Add statement type attribute
	asn1::SPCStatementType* statement_type = asn1::SPCStatementType_new();
	if (!statement_type) return openssl_error(ERR_peek_last_error());

	statement_type->type = OBJ_nid2obj(NID_ms_code_ind);
	ASN1_STRING* statement_type_value =
			EncodedASN1::FromItem<asn1::SPCStatementType,
			asn1::i2d_SPCStatementType>(statement_type)
			.ToSequenceString();

	if (!PKCS7_add_signed_attribute(
		p_signer_info,
		OBJ_txt2nid(SPC_STATEMENT_TYPE_OBJID),
		V_ASN1_SEQUENCE,
		statement_type_value
	)) {

		ASN1_STRING_free(opus_value);
		asn1::SPCStatementType_free(statement_type);
		ASN1_STRING_free(statement_type_value);

		return openssl_error(ERR_peek_last_error());
	}

	return OK;

}

void AppxPackager::make_digests() {

	// AXPC
	SHA256_Final(digests.axpc, &axpc_context);

	// AXCD
	SHA256_Final(digests.axcd, &axcd_context);

	// AXCI
	for (int i = 0; i < SHA256_DIGEST_LENGTH; i++)
		digests.axci[i] = 0;

}

void AppxPackager::write_digest(Vector<uint8_t>& p_out_buffer) {

	// Size of digests plus 6 32-bit magic numbers
	p_out_buffer.resize((SHA256_DIGEST_LENGTH * 5) + (6 * 4));

	int offs = 0;

	// APPX
	uint32_t sig = 0x58505041;
	offs += buf_put_int32(sig, &p_out_buffer[offs]);

	// AXPC
	uint32_t axpc_sig = 0x43505841;
	offs += buf_put_int32(axpc_sig, &p_out_buffer[offs]);
	for (int i = 0; i < SHA256_DIGEST_LENGTH; i++) {
		p_out_buffer[offs++] = digests.axpc[i];
	}

	// AXCD
	uint32_t axcd_sig = 0x44435841;
	offs += buf_put_int32(axcd_sig, &p_out_buffer[offs]);
	for (int i = 0; i < SHA256_DIGEST_LENGTH; i++) {
		p_out_buffer[offs++] = digests.axcd[i];
	}

	// AXCT
	uint32_t axct_sig = 0x54435841;
	offs += buf_put_int32(axct_sig, &p_out_buffer[offs]);
	for (int i = 0; i < SHA256_DIGEST_LENGTH; i++) {
		p_out_buffer[offs++] = digests.axct[i];
	}

	// AXBM
	uint32_t axbm_sig = 0x4D425841;
	offs += buf_put_int32(axbm_sig, &p_out_buffer[offs]);
	for (int i = 0; i < SHA256_DIGEST_LENGTH; i++) {
		p_out_buffer[offs++] = digests.axbm[i];
	}

	// AXCI
	uint32_t axci_sig = 0x49435841;
	offs += buf_put_int32(axci_sig, &p_out_buffer[offs]);
	for (int i = 0; i < SHA256_DIGEST_LENGTH; i++) {
		p_out_buffer[offs++] = digests.axci[i];
	}

	// Done!
}

Error AppxPackager::read_cert_file(const String & p_path, const String &p_password, CertFile* p_out_cf) {

	ERR_FAIL_COND_V(!p_out_cf, ERR_INVALID_PARAMETER);

	BIO* bio = BIO_new_file(p_path.utf8().get_data(), "rb");
	if (!bio) {
		return openssl_error(ERR_peek_last_error());
	}

	PKCS12* data = d2i_PKCS12_bio(bio, NULL);
	if (!data) {
		BIO_free(bio);
		return openssl_error(ERR_peek_last_error());
	}

	/* Fails to link with GCC, need to solve when implement signing
	if (!PKCS12_parse(data, p_password.utf8().get_data(), &p_out_cf->private_key, &p_out_cf->certificate, NULL)) {
		PKCS12_free(data);
		BIO_free(bio);
		return openssl_error(ERR_peek_last_error());
	}*/

	if (!p_out_cf->private_key) {
		PKCS12_free(data);
		BIO_free(bio);
		return openssl_error(ERR_peek_last_error());
	}

	if (!p_out_cf->certificate) {
		PKCS12_free(data);
		BIO_free(bio);
		return openssl_error(ERR_peek_last_error());
	}

	PKCS12_free(data);
	BIO_free(bio);

	return OK;
}

Error AppxPackager::sign(const CertFile & p_cert, const AppxDigests & digests, PKCS7 * p_out_signature) {

	OpenSSL_add_all_algorithms();

	// Register object IDs
	OBJ_create_and_add_object(SPC_INDIRECT_DATA_OBJID, NULL, NULL);
	OBJ_create_and_add_object(SPC_SIPINFO_OBJID, NULL, NULL);
	OBJ_create_and_add_object(SPC_SP_OPUS_INFO_OBJID, NULL, NULL);
	OBJ_create_and_add_object(SPC_STATEMENT_TYPE_OBJID, NULL, NULL);

	if (!PKCS7_set_type(p_out_signature, NID_pkcs7_signed)) {

		return openssl_error(ERR_peek_last_error());
	}

	PKCS7_SIGNER_INFO *signer_info = PKCS7_add_signature(p_out_signature, p_cert.certificate, p_cert.private_key, EVP_sha256());
	if (!signer_info) return openssl_error(ERR_peek_last_error());

	add_attributes(signer_info);

	if (!PKCS7_content_new(p_out_signature, NID_pkcs7_data)) {

		return openssl_error(ERR_peek_last_error());
	}

	if (!PKCS7_add_certificate(p_out_signature, p_cert.certificate)) {

		return openssl_error(ERR_peek_last_error());
	}

	asn1::SPCIndirectDataContent* idc = asn1::SPCIndirectDataContent_new();

	MakeIndirectDataContent(*idc);
	EncodedASN1 idc_encoded =
		EncodedASN1::FromItem<asn1::SPCIndirectDataContent, asn1::i2d_SPCIndirectDataContent>(idc);

	BIO* signed_data = PKCS7_dataInit(p_out_signature, NULL);

	if (idc_encoded.size() < 2) {

		ERR_EXPLAIN("Invalid encoded size");
		ERR_FAIL_V(FAILED);
	}

	if ((idc_encoded.data()[1] & 0x80) == 0x00) {

		ERR_EXPLAIN("Invalid encoded data");
		ERR_FAIL_V(FAILED);
	}

	size_t skip = 4;

	if (BIO_write(signed_data, idc_encoded.data() + skip, idc_encoded.size() - skip)
		!= idc_encoded.size() - skip) {

		return openssl_error(ERR_peek_last_error());
	}
	if (BIO_flush(signed_data) != 1) {

		return openssl_error(ERR_peek_last_error());
	}

	if (!PKCS7_dataFinal(p_out_signature, signed_data)) {

		return openssl_error(ERR_peek_last_error());
	}

	PKCS7* content = PKCS7_new();
	if (!content) {

		return openssl_error(ERR_peek_last_error());
	}

	content->type = OBJ_txt2obj(SPC_INDIRECT_DATA_OBJID, 1);

	ASN1_TYPE* idc_sequence = idc_encoded.ToSequenceType();
	content->d.other = idc_sequence;

	if (!PKCS7_set_content(p_out_signature, content)) {

		return openssl_error(ERR_peek_last_error());
	}

	return OK;
}

#endif // OPENSSL_ENABLED

////////////////////////////////////////////////////////////////////


bool EditorExportPlatformUWP::_valid_resource_name(const String &p_name) const {

	if (p_name.empty()) return false;
	if (p_name.ends_with(".")) return false;

	static const char* invalid_names[] = {
		"CON","PRN","AUX","NUL","COM1","COM2","COM3","COM4","COM5","COM6","COM7",
		"COM8","COM9","LPT1","LPT2","LPT3","LPT4","LPT5","LPT6","LPT7","LPT8","LPT9",
		NULL
	};

	const char** t = invalid_names;
	while (*t) {
		if (p_name == *t) return false;
		t++;
	}

	return true;
}

bool EditorExportPlatformUWP::_valid_guid(const String & p_guid) const {

	Vector<String> parts = p_guid.split("-");

	if (parts.size() != 5) return false;
	if (parts[0].length() != 8) return false;
	for (int i = 1; i < 4; i++)
		if (parts[i].length() != 4) return false;
	if (parts[4].length() != 12) return false;

	return true;
}

bool EditorExportPlatformUWP::_valid_bgcolor(const String & p_color) const {

	if (p_color.empty()) return true;
	if (p_color.begins_with("#") && p_color.is_valid_html_color()) return true;

	// Colors from https://msdn.microsoft.com/en-us/library/windows/apps/dn934817.aspx
	static const char* valid_colors[] = {
		"aliceBlue","antiqueWhite","aqua","aquamarine","azure","beige",
		"bisque","black","blanchedAlmond","blue","blueViolet","brown",
		"burlyWood","cadetBlue","chartreuse","chocolate","coral","cornflowerBlue",
		"cornsilk","crimson","cyan","darkBlue","darkCyan","darkGoldenrod",
		"darkGray","darkGreen","darkKhaki","darkMagenta","darkOliveGreen","darkOrange",
		"darkOrchid","darkRed","darkSalmon","darkSeaGreen","darkSlateBlue","darkSlateGray",
		"darkTurquoise","darkViolet","deepPink","deepSkyBlue","dimGray","dodgerBlue",
		"firebrick","floralWhite","forestGreen","fuchsia","gainsboro","ghostWhite",
		"gold","goldenrod","gray","green","greenYellow","honeydew",
		"hotPink","indianRed","indigo","ivory","khaki","lavender",
		"lavenderBlush","lawnGreen","lemonChiffon","lightBlue","lightCoral","lightCyan",
		"lightGoldenrodYellow","lightGreen","lightGray","lightPink","lightSalmon","lightSeaGreen",
		"lightSkyBlue","lightSlateGray","lightSteelBlue","lightYellow","lime","limeGreen",
		"linen","magenta","maroon","mediumAquamarine","mediumBlue","mediumOrchid",
		"mediumPurple","mediumSeaGreen","mediumSlateBlue","mediumSpringGreen","mediumTurquoise","mediumVioletRed",
		"midnightBlue","mintCream","mistyRose","moccasin","navajoWhite","navy",
		"oldLace","olive","oliveDrab","orange","orangeRed","orchid",
		"paleGoldenrod","paleGreen","paleTurquoise","paleVioletRed","papayaWhip","peachPuff",
		"peru","pink","plum","powderBlue","purple","red",
		"rosyBrown","royalBlue","saddleBrown","salmon","sandyBrown","seaGreen",
		"seaShell","sienna","silver","skyBlue","slateBlue","slateGray",
		"snow","springGreen","steelBlue","tan","teal","thistle",
		"tomato","transparent","turquoise","violet","wheat","white",
		"whiteSmoke","yellow","yellowGreen",
		NULL
	};

	const char** color = valid_colors;

	while(*color) {
		if (p_color == *color) return true;
		color++;
	}

	return false;
}

bool EditorExportPlatformUWP::_valid_image(const Ref<ImageTexture> p_image, int p_width, int p_height) const {

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

Vector<uint8_t> EditorExportPlatformUWP::_fix_manifest(const Vector<uint8_t> &p_template, bool p_give_internet) const {

	String result = String::utf8((const char*)p_template.ptr(), p_template.size());

	result = result.replace("$godot_version$", VERSION_FULL_NAME);

	result = result.replace("$identity_name$", unique_name);
	result = result.replace("$publisher$", publisher);

	result = result.replace("$product_guid$", product_guid);
	result = result.replace("$publisher_guid$", publisher_guid);

	String version = itos(version_major) + "." + itos(version_minor) + "." + itos(version_build) + "." + itos(version_revision);
	result = result.replace("$version_string$", version);

	String architecture = arch == ARM ? "ARM" : arch == X86 ? "x86" : "x64";
	result = result.replace("$architecture$", architecture);

	result = result.replace("$display_name$", display_name.empty() ? (String)GlobalConfig::get_singleton()->get("application/name") : display_name);
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

Vector<uint8_t> EditorExportPlatformUWP::_get_image_data(const String & p_path) {

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

	FileAccess* f = FileAccess::open(tmp_path, FileAccess::READ, &err);

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
	DirAccess* dir = DirAccess::open(tmp_path.get_base_dir(), &err);

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

Error EditorExportPlatformUWP::save_appx_file(void * p_userdata, const String & p_path, const Vector<uint8_t>& p_data, int p_file, int p_total) {

	AppxPackager *packager = (AppxPackager*)p_userdata;
	String dst_path = p_path.replace_first("res://", "game/");

	packager->add_file(dst_path, p_data.ptr(), p_data.size(), p_file, p_total, _should_compress_asset(p_path, p_data));

	return OK;
}

bool EditorExportPlatformUWP::_should_compress_asset(const String & p_path, const Vector<uint8_t>& p_data) {

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

	static const char* unconditional_compress_ext[] = {
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

	for (const char** ext = unconditional_compress_ext; *ext; ++ext) {
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

bool EditorExportPlatformUWP::_set(const StringName& p_name, const Variant& p_value) {

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

#if 0 // Signing disabled
	else if (n == "signing/sign")
		sign_package = p_value;
	else if (n == "signing/certificate_file")
		certificate_path = p_value;
	else if (n == "signing/certificate_password")
		certificate_pass = p_value;
#endif
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
	} else return false;

	return true;
}

bool EditorExportPlatformUWP::_get(const StringName& p_name, Variant &r_ret) const {

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

#if 0 // Signing disabled
	else if (n == "signing/sign")
		r_ret = sign_package;
	else if (n == "signing/certificate_file")
		r_ret = certificate_path;
	else if (n == "signing/certificate_password")
		r_ret = certificate_pass;
#endif
	else if (n.begins_with("capabilities/")) {

		String what = n.get_slice("/", 1).replace("_", "");

		if (array_has(uwp_capabilities, what.utf8().get_data())) {

			r_ret = capabilities.has(what);

		} else if (array_has(uwp_uap_capabilities, what.utf8().get_data())) {

			r_ret = uap_capabilities.has(what);

		} else if (array_has(uwp_device_capabilites, what.utf8().get_data())) {

			r_ret = device_capabilities.has(what);
		}
	} else return false;

	return true;
}

void EditorExportPlatformUWP::_get_property_list(List<PropertyInfo>* p_list) const {

	p_list->push_back(PropertyInfo(Variant::STRING, "custom_package/debug", PROPERTY_HINT_GLOBAL_FILE, "appx"));
	p_list->push_back(PropertyInfo(Variant::STRING, "custom_package/release", PROPERTY_HINT_GLOBAL_FILE, "appx"));

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

#if 0 // Signing does not work :( disabling for now
	p_list->push_back(PropertyInfo(Variant::BOOL, "signing/sign"));
	p_list->push_back(PropertyInfo(Variant::STRING, "signing/certificate_file", PROPERTY_HINT_GLOBAL_FILE, "pfx"));
	p_list->push_back(PropertyInfo(Variant::STRING, "signing/certificate_password"));
#endif

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

bool EditorExportPlatformUWP::can_export(String * r_error) const {

	String err;
	bool valid = true;

	if (!exists_export_template("uwp_x86_debug.zip") || !exists_export_template("uwp_x86_release.zip")
		|| !exists_export_template("uwp_arm_debug.zip") || !exists_export_template("uwp_arm_release.zip")
		|| !exists_export_template("uwp_x64_debug.zip") || !exists_export_template("uwp_x64_release.zip")) {
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

	if (r_error)
		*r_error = err;

	return valid;
}

Error EditorExportPlatformUWP::export_project(const String & p_path, bool p_debug, int p_flags) {

	String src_appx;

	EditorProgress ep("export", "Exporting for Windows Universal", 7);

	if (is_debug)
		src_appx = custom_debug_package;
	else
		src_appx = custom_release_package;

	if (src_appx == "") {
		String err;
		if (p_debug) {
			switch (arch) {
				case X86: {
					src_appx = find_export_template("uwp_x86_debug.zip", &err);
					break;
				}
				case X64: {
					src_appx = find_export_template("uwp_x64_debug.zip", &err);
					break;
				}
				case ARM: {
					src_appx = find_export_template("uwp_arm_debug.zip", &err);
					break;
				}
			}
		} else {
			switch (arch) {
				case X86: {
					src_appx = find_export_template("uwp_x86_release.zip", &err);
					break;
				}
				case X64: {
					src_appx = find_export_template("uwp_x64_release.zip", &err);
					break;
				}
				case ARM: {
					src_appx = find_export_template("uwp_arm_release.zip", &err);
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
	packager.init(fa_pack, sign_package ? AppxPackager::SIGN : AppxPackager::DONT_SIGN, certificate_path, certificate_pass);

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

			data = _fix_manifest(data, p_flags&(EXPORT_DUMB_CLIENT | EXPORT_REMOTE_DEBUG));
		}

		print_line("ADDING: " + path);

		packager.add_file(path, data.ptr(), data.size(), template_file_no++, template_files_amount, _should_compress_asset(path, data));

		ret = unzGoToNextFile(pkg);
	}

	EditorNode::progress_end_task("template_files");

	ep.step("Creating command line...", 2);

	Vector<String> cl = cmdline.strip_edges().split(" ");
	for (int i = 0;i<cl.size();i++) {
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

	return OK;
}

EditorExportPlatformUWP::EditorExportPlatformUWP() {

	Image img(_uwp_logo);
	logo = Ref<ImageTexture>(memnew(ImageTexture));
	logo->create_from_image(img);

	is_debug = true;

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

EditorExportPlatformUWP::~EditorExportPlatformUWP() {}

#endif
void register_uwp_exporter() {
#if 0
	Ref<EditorExportPlatformUWP> exporter = Ref<EditorExportPlatformUWP>(memnew(EditorExportPlatformUWP));
	EditorImportExport::get_singleton()->add_export_platform(exporter);
#endif
}

