/**************************************************************************/
/*  resource_uid.cpp                                                      */
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

#include "resource_uid.h"

#include "core/config/project_settings.h"
#include "core/crypto/crypto_core.h"
#include "core/io/dir_access.h"
#include "core/io/file_access.h"

// These constants are off by 1, causing the 'z' and '9' characters never to be used.
// This cannot be fixed without breaking compatibility; see GH-83843.
static constexpr uint32_t char_count = ('z' - 'a');
static constexpr uint32_t base = char_count + ('9' - '0');

String ResourceUID::get_cache_file() {
	return ProjectSettings::get_singleton()->get_project_data_path().path_join("uid_cache.bin");
}

String ResourceUID::id_to_text(ID p_id) const {
	if (p_id < 0) {
		return "uid://<invalid>";
	}
	String txt;

	while (p_id) {
		uint32_t c = p_id % base;
		if (c < char_count) {
			txt = String::chr('a' + c) + txt;
		} else {
			txt = String::chr('0' + (c - char_count)) + txt;
		}
		p_id /= base;
	}

	return "uid://" + txt;
}

ResourceUID::ID ResourceUID::text_to_id(const String &p_text) const {
	if (!p_text.begins_with("uid://") || p_text == "uid://<invalid>") {
		return INVALID_ID;
	}

	uint32_t l = p_text.length();
	uint64_t uid = 0;
	for (uint32_t i = 6; i < l; i++) {
		uid *= base;
		uint32_t c = p_text[i];
		if (is_ascii_lower_case(c)) {
			uid += c - 'a';
		} else if (is_digit(c)) {
			uid += c - '0' + char_count;
		} else {
			return INVALID_ID;
		}
	}
	return ID(uid & 0x7FFFFFFFFFFFFFFF);
}

ResourceUID::ID ResourceUID::create_id() {
	while (true) {
		ID id = INVALID_ID;
		MutexLock lock(mutex);
		Error err = ((CryptoCore::RandomGenerator *)crypto)->get_random_bytes((uint8_t *)&id, sizeof(id));
		ERR_FAIL_COND_V(err != OK, INVALID_ID);
		id &= 0x7FFFFFFFFFFFFFFF;
		bool exists = unique_ids.has(id);
		if (!exists) {
			return id;
		}
	}
}

bool ResourceUID::has_id(ID p_id) const {
	MutexLock l(mutex);
	return unique_ids.has(p_id);
}
void ResourceUID::add_id(ID p_id, const String &p_path) {
	MutexLock l(mutex);
	ERR_FAIL_COND(unique_ids.has(p_id));
	Cache c;
	c.cs = p_path.utf8();
	unique_ids[p_id] = c;
	changed = true;
}

void ResourceUID::set_id(ID p_id, const String &p_path) {
	MutexLock l(mutex);
	ERR_FAIL_COND(!unique_ids.has(p_id));
	CharString cs = p_path.utf8();
	const char *update_ptr = cs.ptr();
	const char *cached_ptr = unique_ids[p_id].cs.ptr();
	if (update_ptr == nullptr && cached_ptr == nullptr) {
		return; // Both are empty strings.
	}
	if ((update_ptr == nullptr) != (cached_ptr == nullptr) || strcmp(update_ptr, cached_ptr) != 0) {
		unique_ids[p_id].cs = cs;
		unique_ids[p_id].saved_to_cache = false; //changed
		changed = true;
	}
}

String ResourceUID::get_id_path(ID p_id) const {
	MutexLock l(mutex);
	ERR_FAIL_COND_V(!unique_ids.has(p_id), String());
	const CharString &cs = unique_ids[p_id].cs;
	return String::utf8(cs.ptr());
}
void ResourceUID::remove_id(ID p_id) {
	MutexLock l(mutex);
	ERR_FAIL_COND(!unique_ids.has(p_id));
	unique_ids.erase(p_id);
}

Error ResourceUID::save_to_cache() {
	String cache_file = get_cache_file();
	if (!FileAccess::exists(cache_file)) {
		Ref<DirAccess> d = DirAccess::create(DirAccess::ACCESS_RESOURCES);
		d->make_dir_recursive(String(cache_file).get_base_dir()); //ensure base dir exists
	}

	Ref<FileAccess> f = FileAccess::open(cache_file, FileAccess::WRITE);
	if (f.is_null()) {
		return ERR_CANT_OPEN;
	}

	MutexLock l(mutex);
	f->store_32(unique_ids.size());

	cache_entries = 0;

	for (KeyValue<ID, Cache> &E : unique_ids) {
		f->store_64(E.key);
		uint32_t s = E.value.cs.length();
		f->store_32(s);
		f->store_buffer((const uint8_t *)E.value.cs.ptr(), s);
		E.value.saved_to_cache = true;
		cache_entries++;
	}

	changed = false;
	return OK;
}

Error ResourceUID::load_from_cache() {
	Ref<FileAccess> f = FileAccess::open(get_cache_file(), FileAccess::READ);
	if (f.is_null()) {
		return ERR_CANT_OPEN;
	}

	MutexLock l(mutex);
	unique_ids.clear();

	uint32_t entry_count = f->get_32();
	for (uint32_t i = 0; i < entry_count; i++) {
		int64_t id = f->get_64();
		int32_t len = f->get_32();
		Cache c;
		c.cs.resize(len + 1);
		ERR_FAIL_COND_V(c.cs.size() != len + 1, ERR_FILE_CORRUPT); // out of memory
		c.cs[len] = 0;
		int32_t rl = f->get_buffer((uint8_t *)c.cs.ptrw(), len);
		ERR_FAIL_COND_V(rl != len, ERR_FILE_CORRUPT);

		c.saved_to_cache = true;
		unique_ids[id] = c;
	}

	cache_entries = entry_count;
	changed = false;
	return OK;
}

Error ResourceUID::update_cache() {
	if (!changed) {
		return OK;
	}

	if (cache_entries == 0) {
		return save_to_cache();
	}
	MutexLock l(mutex);

	Ref<FileAccess> f;
	for (KeyValue<ID, Cache> &E : unique_ids) {
		if (!E.value.saved_to_cache) {
			if (f.is_null()) {
				f = FileAccess::open(get_cache_file(), FileAccess::READ_WRITE); //append
				if (f.is_null()) {
					return ERR_CANT_OPEN;
				}
				f->seek_end();
			}
			f->store_64(E.key);
			uint32_t s = E.value.cs.length();
			f->store_32(s);
			f->store_buffer((const uint8_t *)E.value.cs.ptr(), s);
			E.value.saved_to_cache = true;
			cache_entries++;
		}
	}

	if (f.is_valid()) {
		f->seek(0);
		f->store_32(cache_entries); //update amount of entries
	}

	changed = false;

	return OK;
}

void ResourceUID::clear() {
	cache_entries = 0;
	unique_ids.clear();
	changed = false;
}
void ResourceUID::_bind_methods() {
	ClassDB::bind_method(D_METHOD("id_to_text", "id"), &ResourceUID::id_to_text);
	ClassDB::bind_method(D_METHOD("text_to_id", "text_id"), &ResourceUID::text_to_id);

	ClassDB::bind_method(D_METHOD("create_id"), &ResourceUID::create_id);

	ClassDB::bind_method(D_METHOD("has_id", "id"), &ResourceUID::has_id);
	ClassDB::bind_method(D_METHOD("add_id", "id", "path"), &ResourceUID::add_id);
	ClassDB::bind_method(D_METHOD("set_id", "id", "path"), &ResourceUID::set_id);
	ClassDB::bind_method(D_METHOD("get_id_path", "id"), &ResourceUID::get_id_path);
	ClassDB::bind_method(D_METHOD("remove_id", "id"), &ResourceUID::remove_id);

	BIND_CONSTANT(INVALID_ID)
}
ResourceUID *ResourceUID::singleton = nullptr;
ResourceUID::ResourceUID() {
	ERR_FAIL_COND(singleton != nullptr);
	singleton = this;
	crypto = memnew(CryptoCore::RandomGenerator);
	((CryptoCore::RandomGenerator *)crypto)->init();
}
ResourceUID::~ResourceUID() {
	memdelete((CryptoCore::RandomGenerator *)crypto);
}
