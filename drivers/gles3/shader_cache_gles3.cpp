/**************************************************************************/
/*  shader_cache_gles3.cpp                                                */
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

#include "shader_cache_gles3.h"

#include "core/crypto/crypto_core.h"
#include "core/os/dir_access.h"
#include "core/os/os.h"
#include "core/project_settings.h"
#include "core/sort_array.h"
#include "core/ustring.h"

String ShaderCacheGLES3::hash_program(const char *const *p_strings_platform, const LocalVector<const char *> &p_vertex_strings, const LocalVector<const char *> &p_fragment_strings) {
	CryptoCore::SHA256Context ctx;
	ctx.start();

	// GL may already reject a binary program if hardware/software has changed, but just in case
	for (const char *const *s = p_strings_platform; *s; s++) {
		uint8_t *bytes = reinterpret_cast<uint8_t *>(const_cast<char *>(*s));
		ctx.update(bytes, strlen(*s));
	}
	for (uint32_t i = 0; i < p_vertex_strings.size(); i++) {
		ctx.update((uint8_t *)p_vertex_strings[i], strlen(p_vertex_strings[i]));
	}
	for (uint32_t i = 0; i < p_fragment_strings.size(); i++) {
		ctx.update((uint8_t *)p_fragment_strings[i], strlen(p_fragment_strings[i]));
	}

	uint8_t hash[32];
	ctx.finish(hash);
	return String::hex_encode_buffer(hash, 32);
}

bool ShaderCacheGLES3::retrieve(const String &p_program_hash, uint32_t *r_format, PoolByteArray *r_data) {
	if (!storage_da) {
		return false;
	}

	FileAccessRef fa = FileAccess::open(storage_path.plus_file(p_program_hash), FileAccess::READ_WRITE);
	if (!fa) {
		return false;
	}

	*r_format = fa->get_32();
	uint32_t binary_len = fa->get_32();
	if (binary_len <= 0 || binary_len > 0x10000000) {
		ERR_PRINT("Program binary cache file is corrupted. Ignoring and removing.");
		fa->close();
		storage_da->remove(p_program_hash);
		return false;
	}
	r_data->resize(binary_len);
	PoolByteArray::Write w = r_data->write();
	if (fa->get_buffer(w.ptr(), binary_len) != static_cast<uint64_t>(binary_len)) {
		ERR_PRINT("Program binary cache file is truncated. Ignoring and removing.");
		fa->close();
		storage_da->remove(p_program_hash);
		return false;
	}

	// Force update modification time (for LRU purge)
	fa->seek(0);
	fa->store_32(*r_format);

	return true;
}

void ShaderCacheGLES3::store(const String &p_program_hash, uint32_t p_program_format, const PoolByteArray &p_program_data) {
	if (!storage_da) {
		return;
	}

	FileAccessRef fa = FileAccess::open(storage_path.plus_file(p_program_hash), FileAccess::WRITE);
	ERR_FAIL_COND(!fa);
	fa->store_32(p_program_format);
	fa->store_32(p_program_data.size());
	PoolByteArray::Read r = p_program_data.read();
	fa->store_buffer(r.ptr(), p_program_data.size());
}

void ShaderCacheGLES3::remove(const String &p_program_hash) {
	if (!storage_da) {
		return;
	}

	storage_da->remove(p_program_hash);
}

void ShaderCacheGLES3::_purge_excess() {
	if (!storage_da) {
		return;
	}

	struct Entry {
		String name;
		uint64_t timestamp;
		uint64_t size;

		bool operator<(const Entry &p_rhs) const {
			return timestamp < p_rhs.timestamp;
		}
	};
	LocalVector<Entry> entries;
	uint64_t total_size = 0;

	ERR_FAIL_COND(storage_da->list_dir_begin() != OK);
	while (true) {
		String f = storage_da->get_next();
		if (f == "") {
			break;
		}
		if (storage_da->current_is_dir()) {
			continue;
		}
		String path = storage_da->get_current_dir().plus_file(f);
		FileAccessRef fa = FileAccess::open(path, FileAccess::READ);
		ERR_CONTINUE(!fa);

		Entry entry;
		entry.name = f;
		entry.timestamp = FileAccess::get_modified_time(path);
		entry.size = fa->get_len();
		entries.push_back(entry);
		total_size += entry.size;
	}
	storage_da->list_dir_end();

	print_verbose("Shader cache size: " + itos(total_size / (1024 * 1024)) + " MiB (max. is " + (itos(storage_size / (1024 * 1024))) + " MiB)");
	if (total_size > storage_size) {
		print_verbose("Purging LRU from shader cache.");
		SortArray<Entry>().sort(entries.ptr(), entries.size());
		for (uint32_t i = 0; i < entries.size(); i++) {
			storage_da->remove(entries[i].name);
			total_size -= entries[i].size;
			if (total_size <= storage_size) {
				break;
			}
		}
	}
}

ShaderCacheGLES3::ShaderCacheGLES3() {
	storage_size = (int)GLOBAL_GET("rendering/gles3/shaders/shader_cache_size_mb") * 1024 * 1024;

	storage_da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	storage_path = OS::get_singleton()->get_cache_path().plus_file(OS::get_singleton()->get_godot_dir_name()).plus_file("shaders");

	print_verbose("Shader cache path: " + storage_path);
	if (storage_da->make_dir_recursive(storage_path) != OK) {
		ERR_PRINT("Couldn't create shader cache directory. Shader cache disabled.");
		memdelete(storage_da);
		storage_da = nullptr;
		return;
	}
	if (storage_da->change_dir(storage_path) != OK) {
		ERR_PRINT("Couldn't open shader cache directory. Shader cache disabled.");
		memdelete(storage_da);
		storage_da = nullptr;
		return;
	}

	_purge_excess();
}

ShaderCacheGLES3::~ShaderCacheGLES3() {
	if (storage_da) {
		memdelete(storage_da);
	}
}
