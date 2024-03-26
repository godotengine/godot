/**
 * limbo_task_db.cpp
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#include "limbo_task_db.h"

#include "limbo_compat.h"

#ifdef LIMBOAI_MODULE
#include "core/config/project_settings.h"
#include "core/io/dir_access.h"
#endif // LIMBOAI_MODULE

#ifdef LIMBOAI_GDEXTENSION
#include <godot_cpp/classes/dir_access.hpp>
#include <godot_cpp/classes/project_settings.hpp>
using namespace godot;
#endif // LIMBOAI_GDEXTENSION

HashMap<String, List<String>> LimboTaskDB::core_tasks;
HashMap<String, List<String>> LimboTaskDB::tasks_cache;

_FORCE_INLINE_ void _populate_scripted_tasks_from_dir(String p_path, List<String> *p_task_classes) {
	if (p_path.is_empty()) {
		return;
	}

	Ref<DirAccess> dir = DIR_ACCESS_CREATE();

	if (dir->change_dir(p_path) == OK) {
		dir->list_dir_begin();
		String fn = dir->get_next();
		while (!fn.is_empty()) {
			if (fn.ends_with(".gd") || fn.ends_with(".cs")) {
				String full_path = p_path.path_join(fn);
				p_task_classes->push_back(full_path);
			}
			fn = dir->get_next();
		}
		dir->list_dir_end();
	} else {
		ERR_FAIL_MSG(vformat("Failed to list \"%s\" directory.", p_path));
	}
}

_FORCE_INLINE_ void _populate_from_user_dir(String p_path, HashMap<String, List<String>> *p_categories) {
	if (p_path.is_empty()) {
		return;
	}

	Ref<DirAccess> dir = DIR_ACCESS_CREATE();
	if (dir->change_dir(p_path) == OK) {
		dir->list_dir_begin();
		String fn = dir->get_next();
		while (!fn.is_empty()) {
			if (dir->current_is_dir() && !fn.begins_with(".")) {
				String full_path;
				String category;
				if (fn == ".") {
					full_path = p_path;
					category = LimboTaskDB::get_misc_category();
				} else {
					full_path = p_path.path_join(fn);
					category = fn.capitalize();
				}

				if (!p_categories->has(category)) {
					p_categories->insert(category, List<String>());
				}

				_populate_scripted_tasks_from_dir(full_path, &p_categories->get(category));
			}
			fn = dir->get_next();
		}
		dir->list_dir_end();

		_populate_scripted_tasks_from_dir(p_path, &p_categories->get(LimboTaskDB::get_misc_category()));

	} else {
		ERR_FAIL_MSG(vformat("Failed to list \"%s\" directory.", p_path));
	}
}

void LimboTaskDB::scan_user_tasks() {
	tasks_cache = HashMap<String, List<String>>(core_tasks);

	if (!tasks_cache.has(LimboTaskDB::get_misc_category())) {
		tasks_cache[LimboTaskDB::get_misc_category()] = List<String>();
	}

	for (int i = 1; i < 4; i++) {
		String dir1 = ProjectSettings::get_singleton()->get_setting_with_override("limbo_ai/behavior_tree/user_task_dir_" + itos(i));
		_populate_from_user_dir(dir1, &tasks_cache);
	}

	for (KeyValue<String, List<String>> &E : tasks_cache) {
		E.value.sort_custom<ComparatorByTaskName>();
	}
}

List<String> LimboTaskDB::get_categories() {
	List<String> r_cat;
	for (const KeyValue<String, List<String>> &E : tasks_cache) {
		r_cat.push_back(E.key);
	}
	r_cat.sort();
	return r_cat;
}

List<String> LimboTaskDB::get_tasks_in_category(const String &p_category) {
	return List<String>(tasks_cache[p_category]);
}
