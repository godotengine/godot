/*************************************************************************/
/*  resource_saver.cpp                                                   */
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

Error ResourceFormatSaver::save(const String &p_path, const RES &p_resource, uint32_t p_flags) {
	int64_t res;
	if (GDVIRTUAL_CALL(_save, p_path, p_resource, p_flags, res)) {
		return (Error)res;
	}

	return ERR_METHOD_NOT_FOUND;
}

bool ResourceFormatSaver::recognize(const RES &p_resource) const {
	bool success;
	if (GDVIRTUAL_CALL(_recognize, p_resource, success)) {
		return success;
	}

	return false;
}

void ResourceFormatSaver::get_recognized_extensions(const RES &p_resource, List<String> *p_extensions) const {
	PackedStringArray exts;
	if (GDVIRTUAL_CALL(_get_recognized_extensions, p_resource, exts)) {
		const String *r = exts.ptr();
		for (int i = 0; i < exts.size(); ++i) {
			p_extensions->push_back(r[i]);
		}
	}
}

void ResourceFormatSaver::_bind_methods() {
	GDVIRTUAL_BIND(_save, "path", "resource", "flags");
	GDVIRTUAL_BIND(_recognize, "resource");
	GDVIRTUAL_BIND(_get_recognized_extensions, "resource");
}

Error ResourceSaver::save(const String &p_path, const RES &p_resource, uint32_t p_flags) {
	String extension = p_path.get_extension();
	Error err = ERR_FILE_UNRECOGNIZED;

	for (int i = 0; i < saver_count; i++) {
		if (!saver[i]->recognize(p_resource)) {
			continue;
		}

		List<String> extensions;
		bool recognized = false;
		saver[i]->get_recognized_extensions(p_resource, &extensions);

		for (const String &E : extensions) {
			if (E.nocasecmp_to(extension) == 0) {
				recognized = true;
			}
		}

		if (!recognized) {
			continue;
		}

		String old_path = p_resource->get_path();

		String local_path = ProjectSettings::get_singleton()->localize_path(p_path);

		RES rwcopy = p_resource;
		if (p_flags & FLAG_CHANGE_PATH) {
			rwcopy->set_path(local_path);
		}

		err = saver[i]->save(p_path, p_resource, p_flags);

		if (err == OK) {
#ifdef TOOLS_ENABLED

			((Resource *)p_resource.ptr())->set_edited(false);
			if (timestamp_on_save) {
				uint64_t mt = FileAccess::get_modified_time(p_path);

				((Resource *)p_resource.ptr())->set_last_modified_time(mt);
			}
#endif

			if (p_flags & FLAG_CHANGE_PATH) {
				rwcopy->set_path(old_path);
			}

			if (save_callback && p_path.begins_with("res://")) {
				save_callback(p_resource, p_path);
			}

			return OK;
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

Ref<ResourceFormatSaver> ResourceSaver::_find_custom_resource_format_saver(String path) {
	for (int i = 0; i < saver_count; ++i) {
		if (saver[i]->get_script_instance() && saver[i]->get_script_instance()->get_script()->get_path() == path) {
			return saver[i];
		}
	}
	return Ref<ResourceFormatSaver>();
}

bool ResourceSaver::add_custom_resource_format_saver(String script_path) {
	if (_find_custom_resource_format_saver(script_path).is_valid()) {
		return false;
	}

	Ref<Resource> res = ResourceLoader::load(script_path);
	ERR_FAIL_COND_V(res.is_null(), false);
	ERR_FAIL_COND_V(!res->is_class("Script"), false);

	Ref<Script> s = res;
	StringName ibt = s->get_instance_base_type();
	bool valid_type = ClassDB::is_parent_class(ibt, "ResourceFormatSaver");
	ERR_FAIL_COND_V_MSG(!valid_type, false, "Script does not inherit a CustomResourceSaver: " + script_path + ".");

	Object *obj = ClassDB::instantiate(ibt);

	ERR_FAIL_COND_V_MSG(obj == nullptr, false, "Cannot instance script as custom resource saver, expected 'ResourceFormatSaver' inheritance, got: " + String(ibt) + ".");

	Ref<ResourceFormatSaver> crl = Object::cast_to<ResourceFormatSaver>(obj);
	crl->set_script(s);
	ResourceSaver::add_resource_format_saver(crl);

	return true;
}

void ResourceSaver::remove_custom_resource_format_saver(String script_path) {
	Ref<ResourceFormatSaver> custom_saver = _find_custom_resource_format_saver(script_path);
	if (custom_saver.is_valid()) {
		remove_resource_format_saver(custom_saver);
	}
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
