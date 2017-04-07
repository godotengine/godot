/*************************************************************************/
/*  file_access.cpp                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "file_access.h"
#include "core/io/file_access_pack.h"
#include "core/io/marshalls.h"
#include "global_config.h"
#include "io/md5.h"
#include "io/sha256.h"
#include "os/os.h"

FileAccess::CreateFunc FileAccess::create_func[ACCESS_MAX] = { 0, 0 };

FileAccess::FileCloseFailNotify FileAccess::close_fail_notify = NULL;

bool FileAccess::backup_save = false;

FileAccess *FileAccess::create(AccessType p_access) {

	ERR_FAIL_COND_V(!create_func, 0);
	ERR_FAIL_INDEX_V(p_access, ACCESS_MAX, 0);

	FileAccess *ret = create_func[p_access]();
	ret->_set_access_type(p_access);
	return ret;
}

bool FileAccess::exists(const String &p_name) {

	if (PackedData::get_singleton()->has_path(p_name))
		return true;

	FileAccess *f = open(p_name, READ);
	if (!f)
		return false;
	memdelete(f);
	return true;
}

void FileAccess::_set_access_type(AccessType p_access) {

	_access_type = p_access;
};

FileAccess *FileAccess::create_for_path(const String &p_path) {

	FileAccess *ret = NULL;
	if (p_path.begins_with("res://")) {

		ret = create(ACCESS_RESOURCES);
	} else if (p_path.begins_with("user://")) {

		ret = create(ACCESS_USERDATA);

	} else {

		ret = create(ACCESS_FILESYSTEM);
	}

	return ret;
}

Error FileAccess::reopen(const String &p_path, int p_mode_flags) {

	return _open(p_path, p_mode_flags);
};

FileAccess *FileAccess::open(const String &p_path, int p_mode_flags, Error *r_error) {

	//try packed data first

	FileAccess *ret = NULL;
	if (!(p_mode_flags & WRITE) && PackedData::get_singleton() && !PackedData::get_singleton()->is_disabled()) {
		ret = PackedData::get_singleton()->try_open_path(p_path);
		if (ret) {
			if (r_error)
				*r_error = OK;
			return ret;
		}
	}

	ret = create_for_path(p_path);
	Error err = ret->_open(p_path, p_mode_flags);

	if (r_error)
		*r_error = err;
	if (err != OK) {

		memdelete(ret);
		ret = NULL;
	}

	return ret;
}

FileAccess::CreateFunc FileAccess::get_create_func(AccessType p_access) {

	return create_func[p_access];
};

String FileAccess::fix_path(const String &p_path) const {
	//helper used by file accesses that use a single filesystem

	String r_path = p_path.replace("\\", "/");

	switch (_access_type) {

		case ACCESS_RESOURCES: {

			if (GlobalConfig::get_singleton()) {
				if (r_path.begins_with("res://")) {

					String resource_path = GlobalConfig::get_singleton()->get_resource_path();
					if (resource_path != "") {

						return r_path.replace("res:/", resource_path);
					};
					return r_path.replace("res://", "");
				}
			}

		} break;
		case ACCESS_USERDATA: {

			if (r_path.begins_with("user://")) {

				String data_dir = OS::get_singleton()->get_data_dir();
				if (data_dir != "") {

					return r_path.replace("user:/", data_dir);
				};
				return r_path.replace("user://", "");
			}

		} break;
		case ACCESS_FILESYSTEM: {

			return r_path;
		} break;
	}

	return r_path;
}

/* these are all implemented for ease of porting, then can later be optimized */

uint16_t FileAccess::get_16() const {

	uint16_t res;
	uint8_t a, b;

	a = get_8();
	b = get_8();

	if (endian_swap) {

		SWAP(a, b);
	}

	res = b;
	res <<= 8;
	res |= a;

	return res;
}
uint32_t FileAccess::get_32() const {

	uint32_t res;
	uint16_t a, b;

	a = get_16();
	b = get_16();

	if (endian_swap) {

		SWAP(a, b);
	}

	res = b;
	res <<= 16;
	res |= a;

	return res;
}
uint64_t FileAccess::get_64() const {

	uint64_t res;
	uint32_t a, b;

	a = get_32();
	b = get_32();

	if (endian_swap) {

		SWAP(a, b);
	}

	res = b;
	res <<= 32;
	res |= a;

	return res;
}

float FileAccess::get_float() const {

	MarshallFloat m;
	m.i = get_32();
	return m.f;
};

real_t FileAccess::get_real() const {

	if (real_is_double)
		return get_double();
	else
		return get_float();
}

double FileAccess::get_double() const {

	MarshallDouble m;
	m.l = get_64();
	return m.d;
};

String FileAccess::get_line() const {

	CharString line;

	CharType c = get_8();

	while (!eof_reached()) {

		if (c == '\n' || c == '\0') {
			line.push_back(0);
			return String::utf8(line.get_data());
		} else if (c != '\r')
			line.push_back(c);

		c = get_8();
	}
	line.push_back(0);
	return String::utf8(line.get_data());
}

Vector<String> FileAccess::get_csv_line(String delim) const {

	ERR_FAIL_COND_V(delim.length() != 1, Vector<String>());

	String l;
	int qc = 0;
	do {
		l += get_line() + "\n";
		qc = 0;
		for (int i = 0; i < l.length(); i++) {

			if (l[i] == '"')
				qc++;
		}

	} while (qc % 2);

	l = l.substr(0, l.length() - 1);

	Vector<String> strings;

	bool in_quote = false;
	String current;
	for (int i = 0; i < l.length(); i++) {

		CharType c = l[i];
		CharType s[2] = { 0, 0 };

		if (!in_quote && c == delim[0]) {
			strings.push_back(current);
			current = String();
		} else if (c == '"') {
			if (l[i + 1] == '"') {
				s[0] = '"';
				current += s;
				i++;
			} else {

				in_quote = !in_quote;
			}
		} else {
			s[0] = c;
			current += s;
		}
	}

	strings.push_back(current);

	return strings;
}

int FileAccess::get_buffer(uint8_t *p_dst, int p_length) const {

	int i = 0;
	for (i = 0; i < p_length && !eof_reached(); i++)
		p_dst[i] = get_8();

	return i;
}

void FileAccess::store_16(uint16_t p_dest) {

	uint8_t a, b;

	a = p_dest & 0xFF;
	b = p_dest >> 8;

	if (endian_swap) {

		SWAP(a, b);
	}

	store_8(a);
	store_8(b);
}
void FileAccess::store_32(uint32_t p_dest) {

	uint16_t a, b;

	a = p_dest & 0xFFFF;
	b = p_dest >> 16;

	if (endian_swap) {

		SWAP(a, b);
	}

	store_16(a);
	store_16(b);
}
void FileAccess::store_64(uint64_t p_dest) {

	uint32_t a, b;

	a = p_dest & 0xFFFFFFFF;
	b = p_dest >> 32;

	if (endian_swap) {

		SWAP(a, b);
	}

	store_32(a);
	store_32(b);
}

void FileAccess::store_real(real_t p_real) {

	if (sizeof(real_t) == 4)
		store_float(p_real);
	else
		store_double(p_real);
}

void FileAccess::store_float(float p_dest) {

	MarshallFloat m;
	m.f = p_dest;
	store_32(m.i);
};

void FileAccess::store_double(double p_dest) {

	MarshallDouble m;
	m.d = p_dest;
	store_64(m.l);
};

uint64_t FileAccess::get_modified_time(const String &p_file) {

	FileAccess *fa = create_for_path(p_file);
	ERR_FAIL_COND_V(!fa, 0);

	uint64_t mt = fa->_get_modified_time(p_file);
	memdelete(fa);
	return mt;
}

void FileAccess::store_string(const String &p_string) {

	if (p_string.length() == 0)
		return;

	CharString cs = p_string.utf8();
	store_buffer((uint8_t *)&cs[0], cs.length());
}

void FileAccess::store_pascal_string(const String &p_string) {

	CharString cs = p_string.utf8();
	store_32(cs.length());
	store_buffer((uint8_t *)&cs[0], cs.length());
};

String FileAccess::get_pascal_string() {

	uint32_t sl = get_32();
	CharString cs;
	cs.resize(sl + 1);
	get_buffer((uint8_t *)cs.ptr(), sl);
	cs[sl] = 0;

	String ret;
	ret.parse_utf8(cs.ptr());

	return ret;
};

void FileAccess::store_line(const String &p_line) {

	store_string(p_line);
	store_8('\n');
}

void FileAccess::store_buffer(const uint8_t *p_src, int p_length) {

	for (int i = 0; i < p_length; i++)
		store_8(p_src[i]);
}

Vector<uint8_t> FileAccess::get_file_as_array(const String &p_file) {

	FileAccess *f = FileAccess::open(p_file, READ);
	ERR_FAIL_COND_V(!f, Vector<uint8_t>());
	Vector<uint8_t> data;
	data.resize(f->get_len());
	f->get_buffer(data.ptr(), data.size());
	memdelete(f);
	return data;
}

String FileAccess::get_md5(const String &p_file) {

	FileAccess *f = FileAccess::open(p_file, READ);
	if (!f)
		return String();

	MD5_CTX md5;
	MD5Init(&md5);

	unsigned char step[32768];

	while (true) {

		int br = f->get_buffer(step, 32768);
		if (br > 0) {

			MD5Update(&md5, step, br);
		}
		if (br < 4096)
			break;
	}

	MD5Final(&md5);

	String ret = String::md5(md5.digest);

	memdelete(f);
	return ret;
}

String FileAccess::get_sha256(const String &p_file) {

	FileAccess *f = FileAccess::open(p_file, READ);
	if (!f)
		return String();

	sha256_context sha256;
	sha256_init(&sha256);

	unsigned char step[32768];

	while (true) {

		int br = f->get_buffer(step, 32768);
		if (br > 0) {

			sha256_hash(&sha256, step, br);
		}
		if (br < 4096)
			break;
	}

	unsigned char hash[32];

	sha256_done(&sha256, hash);

	memdelete(f);
	return String::hex_encode_buffer(hash, 32);
}

FileAccess::FileAccess() {

	endian_swap = false;
	real_is_double = false;
	_access_type = ACCESS_FILESYSTEM;
};
