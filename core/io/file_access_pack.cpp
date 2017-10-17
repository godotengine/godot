/*************************************************************************/
/*  file_access_pack.cpp                                                 */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
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
#include "file_access_pack.h"
#include "version.h"

#include <stdio.h>

#define PACK_VERSION 1

Error PackedData::add_pack(const String &p_path) {

	for (int i = 0; i < sources.size(); i++) {

		if (sources[i]->try_open_pack(p_path)) {

			return OK;
		};
	};

	return ERR_FILE_UNRECOGNIZED;
};

void PackedData::add_path(const String &pkg_path, const String &path, uint64_t ofs, uint64_t size, const uint8_t *p_md5, PackSource *p_src) {

	PathMD5 pmd5(path.md5_buffer());
	//printf("adding path %ls, %lli, %lli\n", path.c_str(), pmd5.a, pmd5.b);

	bool exists = files.has(pmd5);

	PackedFile pf;
	pf.pack = pkg_path;
	pf.offset = ofs;
	pf.size = size;
	for (int i = 0; i < 16; i++)
		pf.md5[i] = p_md5[i];
	pf.src = p_src;

	files[pmd5] = pf;

	if (!exists) {
		//search for dir
		String p = path.replace_first("res://", "");
		PackedDir *cd = root;

		if (p.find("/") != -1) { //in a subdir

			Vector<String> ds = p.get_base_dir().split("/");

			for (int j = 0; j < ds.size(); j++) {

				if (!cd->subdirs.has(ds[j])) {

					PackedDir *pd = memnew(PackedDir);
					pd->name = ds[j];
					pd->parent = cd;
					cd->subdirs[pd->name] = pd;
					cd = pd;
				} else {
					cd = cd->subdirs[ds[j]];
				}
			}
		}
		cd->files.insert(path.get_file());
	}
}

void PackedData::add_pack_source(PackSource *p_source) {

	if (p_source != NULL) {
		sources.push_back(p_source);
	}
};

PackedData *PackedData::singleton = NULL;

PackedData::PackedData() {

	singleton = this;
	root = memnew(PackedDir);
	root->parent = NULL;
	disabled = false;

	add_pack_source(memnew(PackedSourcePCK));
}

void PackedData::_free_packed_dirs(PackedDir *p_dir) {

	for (Map<String, PackedDir *>::Element *E = p_dir->subdirs.front(); E; E = E->next())
		_free_packed_dirs(E->get());
	memdelete(p_dir);
}

PackedData::~PackedData() {

	for (int i = 0; i < sources.size(); i++) {
		memdelete(sources[i]);
	}
	_free_packed_dirs(root);
}

//////////////////////////////////////////////////////////////////

bool PackedSourcePCK::try_open_pack(const String &p_path) {

	FileAccess *f = FileAccess::open(p_path, FileAccess::READ);
	if (!f)
		return false;

	//printf("try open %ls!\n", p_path.c_str());

	uint32_t magic = f->get_32();

	if (magic != 0x43504447) {
		//maybe at he end.... self contained exe
		f->seek_end();
		f->seek(f->get_position() - 4);
		magic = f->get_32();
		if (magic != 0x43504447) {

			memdelete(f);
			return false;
		}
		f->seek(f->get_position() - 12);

		uint64_t ds = f->get_64();
		f->seek(f->get_position() - ds - 8);

		magic = f->get_32();
		if (magic != 0x43504447) {

			memdelete(f);
			return false;
		}
	}

	uint32_t version = f->get_32();
	uint32_t ver_major = f->get_32();
	uint32_t ver_minor = f->get_32();
	uint32_t ver_rev = f->get_32();

	ERR_EXPLAIN("Pack version unsupported: " + itos(version));
	ERR_FAIL_COND_V(version != PACK_VERSION, false);
	ERR_EXPLAIN("Pack created with a newer version of the engine: " + itos(ver_major) + "." + itos(ver_minor) + "." + itos(ver_rev));
	ERR_FAIL_COND_V(ver_major > VERSION_MAJOR || (ver_major == VERSION_MAJOR && ver_minor > VERSION_MINOR), false);

	for (int i = 0; i < 16; i++) {
		//reserved
		f->get_32();
	}

	int file_count = f->get_32();

	for (int i = 0; i < file_count; i++) {

		uint32_t sl = f->get_32();
		CharString cs;
		cs.resize(sl + 1);
		f->get_buffer((uint8_t *)cs.ptr(), sl);
		cs[sl] = 0;

		String path;
		path.parse_utf8(cs.ptr());

		uint64_t ofs = f->get_64();
		uint64_t size = f->get_64();
		uint8_t md5[16];
		f->get_buffer(md5, 16);
		PackedData::get_singleton()->add_path(p_path, path, ofs, size, md5, this);
	};

	return true;
};

FileAccess *PackedSourcePCK::get_file(const String &p_path, PackedData::PackedFile *p_file) {

	return memnew(FileAccessPack(p_path, *p_file));
};

//////////////////////////////////////////////////////////////////

Error FileAccessPack::_open(const String &p_path, int p_mode_flags) {

	ERR_FAIL_V(ERR_UNAVAILABLE);
	return ERR_UNAVAILABLE;
}

void FileAccessPack::close() {

	f->close();
}

bool FileAccessPack::is_open() const {

	return f->is_open();
}

void FileAccessPack::seek(size_t p_position) {

	if (p_position > pf.size) {
		eof = true;
	} else {
		eof = false;
	}

	f->seek(pf.offset + p_position);
	pos = p_position;
}
void FileAccessPack::seek_end(int64_t p_position) {

	seek(pf.size + p_position);
}
size_t FileAccessPack::get_position() const {

	return pos;
}
size_t FileAccessPack::get_len() const {

	return pf.size;
}

bool FileAccessPack::eof_reached() const {

	return eof;
}

uint8_t FileAccessPack::get_8() const {

	if (pos >= pf.size) {
		eof = true;
		return 0;
	}

	pos++;
	return f->get_8();
}

int FileAccessPack::get_buffer(uint8_t *p_dst, int p_length) const {

	if (eof)
		return 0;

	int64_t to_read = p_length;
	if (to_read + pos > pf.size) {
		eof = true;
		to_read = int64_t(pf.size) - int64_t(pos);
	}

	pos += p_length;

	if (to_read <= 0)
		return 0;
	f->get_buffer(p_dst, to_read);

	return to_read;
}

void FileAccessPack::set_endian_swap(bool p_swap) {
	FileAccess::set_endian_swap(p_swap);
	f->set_endian_swap(p_swap);
}

Error FileAccessPack::get_error() const {

	if (eof)
		return ERR_FILE_EOF;
	return OK;
}

void FileAccessPack::flush() {

	ERR_FAIL();
}

void FileAccessPack::store_8(uint8_t p_dest) {

	ERR_FAIL();
}

void FileAccessPack::store_buffer(const uint8_t *p_src, int p_length) {

	ERR_FAIL();
}

bool FileAccessPack::file_exists(const String &p_name) {

	return false;
}

FileAccessPack::FileAccessPack(const String &p_path, const PackedData::PackedFile &p_file)
	: pf(p_file),
	  f(FileAccess::open(pf.pack, FileAccess::READ)) {
	if (!f) {
		ERR_EXPLAIN("Can't open pack-referenced file: " + String(pf.pack));
		ERR_FAIL_COND(!f);
	}
	f->seek(pf.offset);
	pos = 0;
	eof = false;
}

FileAccessPack::~FileAccessPack() {
	if (f)
		memdelete(f);
}

//////////////////////////////////////////////////////////////////////////////////
// DIR ACCESS
//////////////////////////////////////////////////////////////////////////////////

Error DirAccessPack::list_dir_begin() {

	list_dirs.clear();
	list_files.clear();

	for (Map<String, PackedData::PackedDir *>::Element *E = current->subdirs.front(); E; E = E->next()) {

		list_dirs.push_back(E->key());
	}

	for (Set<String>::Element *E = current->files.front(); E; E = E->next()) {

		list_files.push_back(E->get());
	}

	return OK;
}

String DirAccessPack::get_next() {

	if (list_dirs.size()) {
		cdir = true;
		String d = list_dirs.front()->get();
		list_dirs.pop_front();
		return d;
	} else if (list_files.size()) {
		cdir = false;
		String f = list_files.front()->get();
		list_files.pop_front();
		return f;
	} else {
		return String();
	}
}
bool DirAccessPack::current_is_dir() const {

	return cdir;
}
bool DirAccessPack::current_is_hidden() const {

	return false;
}
void DirAccessPack::list_dir_end() {

	list_dirs.clear();
	list_files.clear();
}

int DirAccessPack::get_drive_count() {

	return 0;
}
String DirAccessPack::get_drive(int p_drive) {

	return "";
}

Error DirAccessPack::change_dir(String p_dir) {

	String nd = p_dir.replace("\\", "/");
	bool absolute = false;
	if (nd.begins_with("res://")) {
		nd = nd.replace_first("res://", "");
		absolute = true;
	}

	nd = nd.simplify_path();

	if (nd == "") nd = ".";

	if (nd.begins_with("/")) {
		nd = nd.replace_first("/", "");
		absolute = true;
	}

	Vector<String> paths = nd.split("/");

	PackedData::PackedDir *pd;

	if (absolute)
		pd = PackedData::get_singleton()->root;
	else
		pd = current;

	for (int i = 0; i < paths.size(); i++) {

		String p = paths[i];
		if (p == ".") {
			continue;
		} else if (p == "..") {
			if (pd->parent) {
				pd = pd->parent;
			}
		} else if (pd->subdirs.has(p)) {

			pd = pd->subdirs[p];

		} else {

			return ERR_INVALID_PARAMETER;
		}
	}

	current = pd;

	return OK;
}

String DirAccessPack::get_current_dir() {

	PackedData::PackedDir *pd = current;
	String p = current->name;

	while (pd->parent) {
		pd = pd->parent;
		p = pd->name + "/" + p;
	}

	return "res://" + p;
}

bool DirAccessPack::file_exists(String p_file) {

	return current->files.has(p_file);
}

bool DirAccessPack::dir_exists(String p_dir) {

	return current->subdirs.has(p_dir);
}

Error DirAccessPack::make_dir(String p_dir) {

	return ERR_UNAVAILABLE;
}

Error DirAccessPack::rename(String p_from, String p_to) {

	return ERR_UNAVAILABLE;
}
Error DirAccessPack::remove(String p_name) {

	return ERR_UNAVAILABLE;
}

size_t DirAccessPack::get_space_left() {

	return 0;
}

DirAccessPack::DirAccessPack() {

	current = PackedData::get_singleton()->root;
	cdir = false;
}

DirAccessPack::~DirAccessPack() {
}
