/*************************************************************************/
/*  resource_saver.cpp                                                   */
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
#include "resource_saver.h"
#include "global_config.h"
#include "os/file_access.h"
#include "resource_loader.h"
#include "script_language.h"

ResourceFormatSaver *ResourceSaver::saver[MAX_SAVERS];

int ResourceSaver::saver_count = 0;
bool ResourceSaver::timestamp_on_save = false;
ResourceSavedCallback ResourceSaver::save_callback = 0;

Error ResourceSaver::save(const String &p_path, const RES &p_resource, uint32_t p_flags) {

	String extension = p_path.get_extension();
	Error err = ERR_FILE_UNRECOGNIZED;

	for (int i = 0; i < saver_count; i++) {

		if (!saver[i]->recognize(p_resource))
			continue;

		List<String> extensions;
		bool recognized = false;
		saver[i]->get_recognized_extensions(p_resource, &extensions);

		for (List<String>::Element *E = extensions.front(); E; E = E->next()) {

			if (E->get().nocasecmp_to(extension.get_extension()) == 0)
				recognized = true;
		}

		if (!recognized)
			continue;

		String old_path = p_resource->get_path();

		String local_path = GlobalConfig::get_singleton()->localize_path(p_path);

		RES rwcopy = p_resource;
		if (p_flags & FLAG_CHANGE_PATH)
			rwcopy->set_path(local_path);

		err = saver[i]->save(p_path, p_resource, p_flags);

		if (err == OK) {

#ifdef TOOLS_ENABLED

			((Resource *)p_resource.ptr())->set_edited(false);
			if (timestamp_on_save) {
				uint64_t mt = FileAccess::get_modified_time(p_path);

				((Resource *)p_resource.ptr())->set_last_modified_time(mt);
			}
#endif

			if (p_flags & FLAG_CHANGE_PATH)
				rwcopy->set_path(old_path);

			if (save_callback && p_path.begins_with("res://"))
				save_callback(p_path);

			return OK;
		} else {
		}
	}

	return err;
}

void ResourceSaver::set_save_callback(ResourceSavedCallback p_callback) {

	save_callback = p_callback;
}

void ResourceSaver::get_recognized_extensions(const RES &p_resource, List<String> *p_extensions) {

	for (int i = 0; i < saver_count; i++) {

		saver[i]->get_recognized_extensions(p_resource, p_extensions);
	}
}

void ResourceSaver::add_resource_format_saver(ResourceFormatSaver *p_format_saver, bool p_at_front) {

	ERR_FAIL_COND(saver_count >= MAX_SAVERS);

	if (p_at_front) {
		for (int i = saver_count; i > 0; i--) {
			saver[i] = saver[i - 1];
		}
		saver[0] = p_format_saver;
		saver_count++;
	} else {
		saver[saver_count++] = p_format_saver;
	}
}
