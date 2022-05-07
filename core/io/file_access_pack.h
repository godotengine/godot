/*************************************************************************/
/*  file_access_pack.h                                                   */
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

#ifndef FILE_ACCESS_PACK_H
#define FILE_ACCESS_PACK_H

#include "core/list.h"
#include "core/map.h"
#include "core/os/dir_access.h"
#include "core/os/file_access.h"
#include "core/print_string.h"
#include "core/set.h"

// Godot's packed file magic header ("GDPC" in ASCII).
#define PACK_HEADER_MAGIC 0x43504447
// The current packed file format version number.
#define PACK_FORMAT_VERSION 1

class PackSource;

class PackedData {
	friend class FileAccessPack;
	friend class DirAccessPack;
	friend class PackSource;

public:
	struct PackedFile {
		String pack;
		uint64_t offset; //if offset is ZERO, the file was ERASED
		uint64_t size;
		uint8_t md5[16];
		PackSource *src;
	};

private:
	struct PackedDir {
		PackedDir *parent;
		String name;
		Map<String, PackedDir *> subdirs;
		Set<String> files;
	};

	struct PathMD5 {
		uint64_t a;
		uint64_t b;
		bool operator<(const PathMD5 &p_md5) const {
			if (p_md5.a == a) {
				return b < p_md5.b;
			} else {
				return a < p_md5.a;
			}
		}

		bool operator==(const PathMD5 &p_md5) const {
			return a == p_md5.a && b == p_md5.b;
		};

		PathMD5() {
			a = b = 0;
		};

		PathMD5(const Vector<uint8_t> p_buf) {
			a = *((uint64_t *)&p_buf[0]);
			b = *((uint64_t *)&p_buf[8]);
		};
	};

	Map<PathMD5, PackedFile> files;

	Vector<PackSource *> sources;

	PackedDir *root;

	static PackedData *singleton;
	bool disabled;

	void _free_packed_dirs(PackedDir *p_dir);

public:
	void add_pack_source(PackSource *p_source);
	void add_path(const String &p_pkg_path, const String &p_path, uint64_t p_ofs, uint64_t p_size, const uint8_t *p_md5, PackSource *p_src, bool p_replace_files); // for PackSource

	void set_disabled(bool p_disabled) { disabled = p_disabled; }
	_FORCE_INLINE_ bool is_disabled() const { return disabled; }

	static PackedData *get_singleton() { return singleton; }
	Error add_pack(const String &p_path, bool p_replace_files, uint64_t p_offset);

	_FORCE_INLINE_ FileAccess *try_open_path(const String &p_path);
	_FORCE_INLINE_ bool has_path(const String &p_path);

	_FORCE_INLINE_ DirAccess *try_open_directory(const String &p_path);
	_FORCE_INLINE_ bool has_directory(const String &p_path);

	PackedData();
	~PackedData();
};

class PackSource {
public:
	virtual bool try_open_pack(const String &p_path, bool p_replace_files, uint64_t p_offset) = 0;
	virtual FileAccess *get_file(const String &p_path, PackedData::PackedFile *p_file) = 0;
	virtual ~PackSource() {}
};

class PackedSourcePCK : public PackSource {
public:
	virtual bool try_open_pack(const String &p_path, bool p_replace_files, uint64_t p_offset);
	virtual FileAccess *get_file(const String &p_path, PackedData::PackedFile *p_file);
};

class FileAccessPack : public FileAccess {
	PackedData::PackedFile pf;

	mutable uint64_t pos;
	mutable bool eof;

	FileAccess *f;
	virtual Error _open(const String &p_path, int p_mode_flags);
	virtual uint64_t _get_modified_time(const String &p_file) { return 0; }
	virtual uint32_t _get_unix_permissions(const String &p_file) { return 0; }
	virtual Error _set_unix_permissions(const String &p_file, uint32_t p_permissions) { return FAILED; }

public:
	virtual void close();
	virtual bool is_open() const;

	virtual void seek(uint64_t p_position);
	virtual void seek_end(int64_t p_position = 0);
	virtual uint64_t get_position() const;
	virtual uint64_t get_len() const;

	virtual bool eof_reached() const;

	virtual uint8_t get_8() const;

	virtual uint64_t get_buffer(uint8_t *p_dst, uint64_t p_length) const;

	virtual void set_endian_swap(bool p_swap);

	virtual Error get_error() const;

	virtual void flush();
	virtual void store_8(uint8_t p_dest);

	virtual void store_buffer(const uint8_t *p_src, uint64_t p_length);

	virtual bool file_exists(const String &p_name);

	FileAccessPack(const String &p_path, const PackedData::PackedFile &p_file);
	~FileAccessPack();
};

FileAccess *PackedData::try_open_path(const String &p_path) {
	PathMD5 pmd5(p_path.md5_buffer());
	Map<PathMD5, PackedFile>::Element *E = files.find(pmd5);
	if (!E) {
		return nullptr; //not found
	}
	if (E->get().offset == 0) {
		return nullptr; //was erased
	}

	return E->get().src->get_file(p_path, &E->get());
}

bool PackedData::has_path(const String &p_path) {
	return files.has(PathMD5(p_path.md5_buffer()));
}

bool PackedData::has_directory(const String &p_path) {
	DirAccess *da = try_open_directory(p_path);
	if (da) {
		memdelete(da);
		return true;
	} else {
		return false;
	}
}

class DirAccessPack : public DirAccess {
	PackedData::PackedDir *current;

	List<String> list_dirs;
	List<String> list_files;
	bool cdir;

	PackedData::PackedDir *_find_dir(String p_dir);

public:
	virtual Error list_dir_begin();
	virtual String get_next();
	virtual bool current_is_dir() const;
	virtual bool current_is_hidden() const;
	virtual void list_dir_end();

	virtual int get_drive_count();
	virtual String get_drive(int p_drive);

	virtual Error change_dir(String p_dir);
	virtual String get_current_dir();

	virtual bool file_exists(String p_file);
	virtual bool dir_exists(String p_dir);

	virtual Error make_dir(String p_dir);

	virtual Error rename(String p_from, String p_to);
	virtual Error remove(String p_name);

	virtual bool is_link(String p_file) { return false; }
	virtual String read_link(String p_file) { return p_file; }
	virtual Error create_link(String p_source, String p_target) { return FAILED; }

	uint64_t get_space_left();

	virtual String get_filesystem_type() const;

	DirAccessPack();
	~DirAccessPack();
};

DirAccess *PackedData::try_open_directory(const String &p_path) {
	DirAccess *da = memnew(DirAccessPack());
	if (da->change_dir(p_path) != OK) {
		memdelete(da);
		da = nullptr;
	}
	return da;
}

#endif // FILE_ACCESS_PACK_H
