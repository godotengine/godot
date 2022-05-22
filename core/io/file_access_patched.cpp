/**************************************************************************/
/*  file_access_patched.cpp                                               */
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

#include "file_access_patched.h"

#include "file_access_pack.h"

#include "core/io/delta_encoding.h"
#include "core/os/os.h"

Error FileAccessPatched::_apply_patch() const {
	ERR_FAIL_COND_V(!is_open(), FAILED);

	String path = old_file->get_path();
	Vector<PackedData::PackedFile> delta_patches = PackedData::get_singleton()->get_delta_patches(path);
	Vector<uint8_t> old_file_data = old_file->get_buffer(old_file->get_length());

	for (int i = 0; i < delta_patches.size(); ++i) {
		const PackedData::PackedFile &delta_patch = delta_patches[i];
		ERR_FAIL_COND_V(delta_patch.bundle, FAILED);

		Error err = OK;

		uint64_t total_usec_start = OS::get_singleton()->get_ticks_usec();
		uint64_t io_usec_start = OS::get_singleton()->get_ticks_usec();

		Ref<FileAccess> patch_file = FileAccess::open(delta_patch.pack.name, FileAccess::READ, &err);
		ERR_FAIL_COND_V(err != OK, err);

		patch_file->seek(delta_patch.offset);
		ERR_FAIL_COND_V(patch_file->get_error() != OK, patch_file->get_error());

		Vector<uint8_t> patch_data = patch_file->get_buffer(delta_patch.size);
		ERR_FAIL_COND_V(patch_data.is_empty(), ERR_FILE_CANT_READ);

		uint64_t io_usec_end = OS::get_singleton()->get_ticks_usec();
		uint64_t decode_usec_start = OS::get_singleton()->get_ticks_usec();

		Vector<uint8_t> new_file_data;
		err = DeltaEncoding::decode_delta(old_file_data, patch_data, new_file_data);
		ERR_FAIL_COND_V_MSG(err != OK, err, vformat("Failed to apply delta patch (%d of %d) to \"%s\".", i + 1, delta_patches.size(), path));

		uint64_t decode_usec_end = OS::get_singleton()->get_ticks_usec();

		old_file_data = new_file_data;

		uint64_t total_usec_end = OS::get_singleton()->get_ticks_usec();

		print_verbose(vformat(U"Applied delta patch to \"%s\" from \"%s\" in %d μs (%d μs I/O, %d μs decoding).", path, delta_patch.pack.name.get_file(), total_usec_end - total_usec_start, io_usec_end - io_usec_start, decode_usec_end - decode_usec_start));
	}

	patched_file_data = old_file_data;
	patched_file.instantiate();
	return patched_file->open_custom(patched_file_data.ptr(), patched_file_data.size());
}

bool FileAccessPatched::_try_apply_patch() const {
	if (last_error != OK) {
		return false;
	}

	if (patched_file.is_valid()) {
		return true;
	}

	last_error = _apply_patch();
	return last_error == OK;
}

Error FileAccessPatched::open_custom(const Ref<FileAccess> &p_old_file) {
	close();

	if (!p_old_file->is_open()) {
		last_error = ERR_FILE_CANT_OPEN;
		return last_error;
	}

	old_file = p_old_file;

	return OK;
}

bool FileAccessPatched::is_open() const {
	return old_file.is_valid() && old_file->is_open();
}

void FileAccessPatched::seek(uint64_t p_position) {
	if (!_try_apply_patch()) {
		return;
	}

	patched_file->seek(p_position);
}

void FileAccessPatched::seek_end(int64_t p_position) {
	if (!_try_apply_patch()) {
		return;
	}

	patched_file->seek_end(p_position);
}

uint64_t FileAccessPatched::get_position() const {
	if (!_try_apply_patch()) {
		return 0;
	}

	return patched_file->get_position();
}

uint64_t FileAccessPatched::get_length() const {
	if (!_try_apply_patch()) {
		return 0;
	}

	return patched_file->get_length();
}

bool FileAccessPatched::eof_reached() const {
	if (!_try_apply_patch()) {
		return true;
	}

	return patched_file->eof_reached();
}

Error FileAccessPatched::get_error() const {
	if (last_error != OK) {
		return last_error;
	}

	if (patched_file.is_valid()) {
		Error inner_error = patched_file->get_error();
		if (inner_error != OK) {
			return inner_error;
		}
	}

	return last_error;
}

bool FileAccessPatched::store_buffer(const uint8_t *p_src, uint64_t p_length) {
	if (!_try_apply_patch()) {
		return false;
	}

	return patched_file->store_buffer(p_src, p_length);
}

uint64_t FileAccessPatched::get_buffer(uint8_t *p_dst, uint64_t p_length) const {
	if (!_try_apply_patch()) {
		return 0;
	}

	return patched_file->get_buffer(p_dst, p_length);
}

void FileAccessPatched::flush() {
	if (!_try_apply_patch()) {
		return;
	}

	patched_file->flush();
}

void FileAccessPatched::close() {
	old_file = Ref<FileAccess>();
	patched_file = Ref<FileAccessMemory>();
	patched_file_data.clear();
	last_error = OK;
}

bool FileAccessPatched::file_exists(const String &p_name) {
	ERR_FAIL_COND_V(old_file.is_null(), false);
	return old_file->file_exists(p_name);
}
