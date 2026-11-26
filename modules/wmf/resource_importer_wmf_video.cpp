/**************************************************************************/
/*  resource_importer_wmf_video.cpp                                       */
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

#include "resource_importer_wmf_video.h"

#include "core/error/error_list.h"
#include "core/error/error_macros.h"
#include "core/io/file_access.h"
#include "core/io/resource_saver.h"
#include "video_stream_wmf.h"

String ResourceImporterWMFVideo::get_importer_name() const {
	return "WindowsMediaFoundation";
}

String ResourceImporterWMFVideo::get_visible_name() const {
	return "WindowsMediaFoundation";
}

void ResourceImporterWMFVideo::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("mp4");
	p_extensions->push_back("avi");
	p_extensions->push_back("mkv");
	p_extensions->push_back("webm");
}

String ResourceImporterWMFVideo::get_save_extension() const {
	return "wmfvstr";
}

String ResourceImporterWMFVideo::get_resource_type() const {
	return "VideoStreamWMF";
}

bool ResourceImporterWMFVideo::get_option_visibility(const String &p_path, const String &p_option, const HashMap<StringName, Variant> &p_options) const {
	return true;
}

int ResourceImporterWMFVideo::get_preset_count() const {
	return 0;
}

String ResourceImporterWMFVideo::get_preset_name(int p_idx) const {
	return String();
}

void ResourceImporterWMFVideo::get_import_options(const String &p_path, List<ImportOption> *r_options, int p_preset) const {
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "loop"), false));
}

Error ResourceImporterWMFVideo::import(ResourceUID::ID p_source_id, const String &p_source_file, const String &p_save_path, const HashMap<StringName, Variant> &p_options, List<String> *r_platform_variants, List<String> *r_gen_files, Variant *r_metadata) {
	ERR_FAIL_COND_V(p_source_file.is_empty(), ERR_INVALID_PARAMETER);

	// Load the video file data into memory
	Error err;
	Ref<FileAccess> file = FileAccess::open(p_source_file, FileAccess::READ, &err);
	ERR_FAIL_COND_V_MSG(err != OK, err, "Cannot open file '" + p_source_file + "'.");

	Vector<uint8_t> data;
	uint64_t len = file->get_length();
	data.resize(len);
	uint64_t bytes_read = file->get_buffer(data.ptrw(), len);
	ERR_FAIL_COND_V(bytes_read == 0, ERR_FILE_CORRUPT);
	ERR_FAIL_COND_V_MSG(bytes_read != len, ERR_FILE_CORRUPT, "Cannot read file '" + p_source_file + "'.");
	file.unref();

	Ref<VideoStreamWMF> wmfv_stream;
	wmfv_stream.instantiate();
	wmfv_stream->set_file(p_source_file); // Keep file path for fallback
	return ResourceSaver::save(wmfv_stream, p_save_path + ".wmfvstr");
}

ResourceImporterWMFVideo::ResourceImporterWMFVideo() {
}
