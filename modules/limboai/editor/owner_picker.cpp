/**
 * owner_picker.cpp
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifdef TOOLS_ENABLED

#include "owner_picker.h"

#include "../util/limbo_compat.h"

#ifdef LIMBOAI_MODULE
#include "editor/editor_file_system.h"
#include "editor/editor_interface.h"
#elif LIMBOAI_GDEXTENSION
#include <godot_cpp/classes/editor_file_system.hpp>
#include <godot_cpp/classes/editor_file_system_directory.hpp>
#include <godot_cpp/classes/editor_interface.hpp>
#include <godot_cpp/classes/resource_loader.hpp>
#endif

Vector<String> OwnerPicker::_find_owners(const String &p_path) const {
	Vector<String> owners;

	if (RESOURCE_PATH_IS_BUILT_IN(p_path)) {
		// For built-in resources we use the path to the containing resource.
		String owner_path = p_path.substr(0, p_path.rfind("::"));
		owners.append(owner_path);
		return owners;
	}

	List<EditorFileSystemDirectory *> dirs;
	dirs.push_back(EDITOR_FILE_SYSTEM()->get_filesystem());
	while (dirs.size() > 0) {
		EditorFileSystemDirectory *efd = dirs.front()->get();
		dirs.pop_front();

		for (int i = 0; i < efd->get_file_count(); i++) {
			String file_path = efd->get_file_path(i);

			Vector<String> deps;
#ifdef LIMBOAI_MODULE
			deps = efd->get_file_deps(i);
#elif LIMBOAI_GDEXTENSION
			PackedStringArray res_deps = ResourceLoader::get_singleton()->get_dependencies(file_path);
			for (String dep : res_deps) {
				if (dep.begins_with("uid://")) {
					dep = dep.get_slice("::", 2);
				}
				deps.append(dep);
			}
#endif // LIMBOAI_MODULE

			for (int j = 0; j < deps.size(); j++) {
				if (deps[j] == p_path) {
					owners.append(file_path);
					break;
				}
			}
		}

		for (int k = 0; k < efd->get_subdir_count(); k++) {
			dirs.push_back(efd->get_subdir(k));
		}
	}

	return owners;
}

void OwnerPicker::pick_and_open_owner_of_resource(const String &p_path) {
	if (p_path.is_empty()) {
		return;
	}

	owners_item_list->clear();

	Vector<String> owners = _find_owners(p_path);
	for (int i = 0; i < owners.size(); i++) {
		owners_item_list->add_item(owners[i]);
	}

	if (owners_item_list->get_item_count() > 0) {
		owners_item_list->select(0);
		owners_item_list->ensure_current_is_visible();
	}

	if (owners_item_list->get_item_count() == 1) {
		// Open owner immediately if there is only one owner.
		_selection_confirmed();
	} else if (owners_item_list->get_item_count() == 0) {
		owners_item_list->hide();
		set_title(TTR("Alert!"));
		set_text(TTR("Couldn't find owner. Looks like it's not used by any other resource."));
		reset_size();
		popup_centered();
	} else {
		owners_item_list->show();
		set_title(TTR("Pick owner"));
		set_text("");
		reset_size();
		popup_centered_ratio(0.3);
		owners_item_list->grab_focus();
	}
}

void OwnerPicker::_item_activated(int p_item) {
	hide();
	emit_signal("confirmed");
}

void OwnerPicker::_selection_confirmed() {
	for (int idx : owners_item_list->get_selected_items()) {
		String owner_path = owners_item_list->get_item_text(idx);
		if (RESOURCE_IS_SCENE_FILE(owner_path)) {
			EditorInterface::get_singleton()->open_scene_from_path(owner_path);
		} else {
			EditorInterface::get_singleton()->edit_resource(RESOURCE_LOAD(owner_path, ""));
		}
	}
}

void OwnerPicker::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY: {
			owners_item_list->connect("item_activated", callable_mp(this, &OwnerPicker::_item_activated));
			connect("confirmed", callable_mp(this, &OwnerPicker::_selection_confirmed));
		} break;
	}
}

void OwnerPicker::_bind_methods() {
}

OwnerPicker::OwnerPicker() {
	owners_item_list = memnew(ItemList);
	// Note: In my tests, editor couldn't process open request for multiple packed scenes at once.
	owners_item_list->set_select_mode(ItemList::SELECT_SINGLE);
	add_child(owners_item_list);
}

#endif // TOOLS_ENABLED
