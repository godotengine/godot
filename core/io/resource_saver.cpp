/**************************************************************************/
/*  resource_saver.cpp                                                    */
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

#include "resource_saver.h"
#include "core/config/project_settings.h"
#include "core/io/file_access.h"
#include "core/io/resource_loader.h"
#include "core/object/script_language.h"

Ref<ResourceFormatSaver> ResourceSaver::saver[MAX_SAVERS];

int ResourceSaver::saver_count = 0;
bool ResourceSaver::timestamp_on_save = false;
ResourceSavedCallback ResourceSaver::save_callback = nullptr;
ResourceSaverGetResourceIDForPath ResourceSaver::save_get_id_for_path = nullptr;

Error ResourceFormatSaver::save(const Ref<Resource> &p_resource, const String &p_path, uint32_t p_flags) {
	Error err = ERR_METHOD_NOT_FOUND;
	GDVIRTUAL_CALL(_save, p_resource, p_path, p_flags, err);
	return err;
}

Error ResourceFormatSaver::set_uid(const String &p_path, ResourceUID::ID p_uid) {
	Error err = ERR_FILE_UNRECOGNIZED;
	GDVIRTUAL_CALL(_set_uid, p_path, p_uid, err);
	return err;
}

bool ResourceFormatSaver::recognize(const Ref<Resource> &p_resource) const {
	bool success = false;
	GDVIRTUAL_CALL(_recognize, p_resource, success);
	return success;
}

void ResourceFormatSaver::get_recognized_extensions(const Ref<Resource> &p_resource, List<String> *p_extensions) const {
	PackedStringArray exts;
	if (GDVIRTUAL_CALL(_get_recognized_extensions, p_resource, exts)) {
		const String *r = exts.ptr();
		for (int i = 0; i < exts.size(); ++i) {
			p_extensions->push_back(r[i]);
		}
	}
}

bool ResourceFormatSaver::recognize_path(const Ref<Resource> &p_resource, const String &p_path) const {
	bool ret = false;
	if (GDVIRTUAL_CALL(_recognize_path, p_resource, p_path, ret)) {
		return ret;
	}

	String extension = p_path.get_extension();

	List<String> extensions;
	get_recognized_extensions(p_resource, &extensions);

	for (const String &E : extensions) {
		if (E.nocasecmp_to(extension) == 0) {
			return true;
		}
	}

	return false;
}

void ResourceFormatSaver::_bind_methods() {
	GDVIRTUAL_BIND(_save, "resource", "path", "flags");
	GDVIRTUAL_BIND(_set_uid, "path", "uid");
	GDVIRTUAL_BIND(_recognize, "resource");
	GDVIRTUAL_BIND(_get_recognized_extensions, "resource");
	GDVIRTUAL_BIND(_recognize_path, "resource", "path");
}

Error ResourceSaver::save(const Ref<Resource> &p_resource, const String &p_path, uint32_t p_flags) {
	ERR_FAIL_COND_V_MSG(p_resource.is_null(), ERR_INVALID_PARAMETER, vformat("Can't save empty resource to path '%s'.", p_path));
	String path = p_path;
	if (path.is_empty()) {
		path = p_resource->get_path();
	}
	ERR_FAIL_COND_V_MSG(path.is_empty(), ERR_INVALID_PARAMETER, "Can't save resource to empty path. Provide non-empty path or a Resource with non-empty resource_path.");

	String extension = path.get_extension();
	Error err = ERR_FILE_UNRECOGNIZED;

	for (int i = 0; i < saver_count; i++) {
		if (!saver[i]->recognize(p_resource)) {
			continue;
		}

		if (!saver[i]->recognize_path(p_resource, path)) {
			continue;
		}

		String old_path = p_resource->get_path();

		String local_path = ProjectSettings::get_singleton()->localize_path(path);

		if (p_flags & FLAG_CHANGE_PATH) {
			p_resource->set_path(local_path);
		}

		err = saver[i]->save(p_resource, path, p_flags);

		if (err == OK) {
#ifdef TOOLS_ENABLED

			((Resource *)p_resource.ptr())->set_edited(false);
			if (timestamp_on_save) {
				uint64_t mt = FileAccess::get_modified_time(path);

				((Resource *)p_resource.ptr())->set_last_modified_time(mt);
			}
#endif

			if (p_flags & FLAG_CHANGE_PATH) {
				p_resource->set_path(old_path);
			}

			if (save_callback && path.begins_with("res://")) {
				save_callback(p_resource, path);
			}

			return OK;
		}
	}

	return err;
}

Error ResourceSaver::set_uid(const String &p_path, ResourceUID::ID p_uid) {
	String path = p_path;

	ERR_FAIL_COND_V_MSG(path.is_empty(), ERR_INVALID_PARAMETER, "Can't update UID to empty path. Provide non-empty path.");

	Error err = ERR_FILE_UNRECOGNIZED;

	for (int i = 0; i < saver_count; i++) {
		err = saver[i]->set_uid(path, p_uid);
		if (err == OK) {
			break;
		}
	}

	return err;
}

void ResourceSaver::set_save_callback(ResourceSavedCallback p_callback) {
	save_callback = p_callback;
}

void ResourceSaver::get_recognized_extensions(const Ref<Resource> &p_resource, List<String> *p_extensions) {
	ERR_FAIL_COND_MSG(p_resource.is_null(), "It's not a reference to a valid Resource object.");
	for (int i = 0; i < saver_count; i++) {
		saver[i]->get_recognized_extensions(p_resource, p_extensions);
	}
}

void ResourceSaver::add_resource_format_saver(Ref<ResourceFormatSaver> p_format_saver, bool p_at_front) {
	ERR_FAIL_COND_MSG(p_format_saver.is_null(), "It's not a reference to a valid ResourceFormatSaver object.");
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

void ResourceSaver::remove_resource_format_saver(Ref<ResourceFormatSaver> p_format_saver) {
	ERR_FAIL_COND_MSG(p_format_saver.is_null(), "It's not a reference to a valid ResourceFormatSaver object.");

	// Find saver
	int i = 0;
	for (; i < saver_count; ++i) {
		if (saver[i] == p_format_saver) {
			break;
		}
	}

	ERR_FAIL_COND(i >= saver_count); // Not found

	// Shift next savers up
	for (; i < saver_count - 1; ++i) {
		saver[i] = saver[i + 1];
	}
	saver[saver_count - 1].unref();
	--saver_count;
}

Ref<ResourceFormatSaver> ResourceSaver::_find_custom_resource_format_saver(const String &path) {
	for (int i = 0; i < saver_count; ++i) {
		if (saver[i]->get_script_instance() && saver[i]->get_script_instance()->get_script()->get_path() == path) {
			return saver[i];
		}
	}
	return Ref<ResourceFormatSaver>();
}

bool ResourceSaver::add_custom_resource_format_saver(const String &script_path) {
	if (_find_custom_resource_format_saver(script_path).is_valid()) {
		return false;
	}

	Ref<Resource> res = ResourceLoader::load(script_path);
	ERR_FAIL_COND_V(res.is_null(), false);
	ERR_FAIL_COND_V(!res->is_class("Script"), false);

	Ref<Script> s = res;
	StringName ibt = s->get_instance_base_type();
	bool valid_type = ClassDB::is_parent_class(ibt, "ResourceFormatSaver");
	ERR_FAIL_COND_V_MSG(!valid_type, false, vformat("Failed to add a custom resource saver, script '%s' does not inherit 'ResourceFormatSaver'.", script_path));

	Object *obj = ClassDB::instantiate(ibt);
	ERR_FAIL_NULL_V_MSG(obj, false, vformat("Failed to add a custom resource saver, cannot instantiate '%s'.", ibt));

	Ref<ResourceFormatSaver> crl = Object::cast_to<ResourceFormatSaver>(obj);
	crl->set_script(s);
	ResourceSaver::add_resource_format_saver(crl);

	return true;
}

void ResourceSaver::add_custom_savers() {
	// Custom resource savers exploits global class names

	String custom_saver_base_class = ResourceFormatSaver::get_class_static();

	List<StringName> global_classes;
	ScriptServer::get_global_class_list(&global_classes);

	for (const StringName &class_name : global_classes) {
		StringName base_class = ScriptServer::get_global_class_native_base(class_name);

		if (base_class == custom_saver_base_class) {
			String path = ScriptServer::get_global_class_path(class_name);
			add_custom_resource_format_saver(path);
		}
	}
}

void ResourceSaver::remove_custom_savers() {
	Vector<Ref<ResourceFormatSaver>> custom_savers;
	for (int i = 0; i < saver_count; ++i) {
		if (saver[i]->get_script_instance()) {
			custom_savers.push_back(saver[i]);
		}
	}

	for (int i = 0; i < custom_savers.size(); ++i) {
		remove_resource_format_saver(custom_savers[i]);
	}
}

ResourceUID::ID ResourceSaver::get_resource_id_for_path(const String &p_path, bool p_generate) {
	if (save_get_id_for_path) {
		return save_get_id_for_path(p_path, p_generate);
	}
	return ResourceUID::INVALID_ID;
}

void ResourceSaver::set_get_resource_id_for_path(ResourceSaverGetResourceIDForPath p_callback) {
	save_get_id_for_path = p_callback;
}
