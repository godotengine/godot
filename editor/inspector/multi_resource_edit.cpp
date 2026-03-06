#include "multi_resource_edit.h"

#include "core/io/resource_loader.h"
#include "core/math/math_fieldwise.h"
#include "editor/editor_node.h"
#include "editor/editor_undo_redo_manager.h"

void MultiResourceEdit::_ensure_resources_loaded() const {
	if (loaded_resources.size() == (int)resource_paths.size()) {
		return;
	}
	loaded_resources.clear();
	for (const String &path : resource_paths) {
		Ref<Resource> r = ResourceLoader::load(path);
		if (r.is_valid()) {
			loaded_resources.push_back(r);
		}
	}
}

bool MultiResourceEdit::_set(const StringName &p_name, const Variant &p_value) {
	return _set_impl(p_name, p_value, "");
}

bool MultiResourceEdit::_set_impl(const StringName &p_name, const Variant &p_value, const String &p_field, bool p_undo_redo) {
	String name = p_name;
	if (name == "scripts") {
		name = "script";
	}

	_ensure_resources_loaded();

	for (Ref<Resource> &r : loaded_resources) {
		Variant new_value;
		if (p_field.is_empty()) {
			new_value = p_value;
		} else {
			new_value = fieldwise_assign(r->get(name), p_value, p_field);
		}
		r->set(name, new_value);
	}

	return true;
}

bool MultiResourceEdit::_get(const StringName &p_name, Variant &r_ret) const {
	String name = p_name;
	if (name == "scripts") {
		name = "script";
	}

	_ensure_resources_loaded();

	for (const Ref<Resource> &r : loaded_resources) {
		bool found;
		r_ret = r->get(name, &found);
		if (found) {
			return true;
		}
	}

	return false;
}

void MultiResourceEdit::_get_property_list(List<PropertyInfo> *p_list) const {
	HashMap<String, PLData> usage;
	int rc = 0;
	List<PLData *> data_list;

	_ensure_resources_loaded();

	for (const Ref<Resource> &r : loaded_resources) {
		List<PropertyInfo> plist;
		r->get_property_list(&plist, true);

		for (PropertyInfo F : plist) {
			if (F.name == "script") {
				continue;
			}

			PLData *usage_data = usage.getptr(F.name);
			if (!usage_data) {
				PLData pld;
				pld.uses = 0;
				pld.info = F;
				pld.info.name = F.name;
				HashMap<String, MultiResourceEdit::PLData>::Iterator I = usage.insert(F.name, pld);
				usage_data = &I->value;
				data_list.push_back(usage_data);
			}

			if (usage_data->info == F) {
				usage_data->uses++;
			}
		}

		rc++;
	}

	for (const PLData *E : data_list) {
		if (rc == E->uses) {
			p_list->push_back(E->info);
		}
	}

	p_list->push_back(PropertyInfo(Variant::OBJECT, "scripts", PROPERTY_HINT_RESOURCE_TYPE, "Script"));
}

String MultiResourceEdit::_get_editor_name() const {
	return vformat(TTR("%s (%d Selected)"), get_edited_class_name(), get_resource_count());
}

bool MultiResourceEdit::_property_can_revert(const StringName &p_name) const {
	if (ClassDB::has_property(get_edited_class_name(), p_name)) {
		_ensure_resources_loaded();
		return loaded_resources.size() > 0;
	}
	return false;
}

bool MultiResourceEdit::_property_get_revert(const StringName &p_name, Variant &r_property) const {
	_ensure_resources_loaded();
	for (const Ref<Resource> &r : loaded_resources) {
		r_property = ClassDB::class_get_default_property_value(r->get_class_name(), p_name);
		return true;
	}
	return false;
}

void MultiResourceEdit::_queue_notify_property_list_changed() {
	if (notify_property_list_changed_pending) {
		return;
	}
	notify_property_list_changed_pending = true;
	callable_mp(this, &MultiResourceEdit::_notify_property_list_changed).call_deferred();
}

void MultiResourceEdit::_notify_property_list_changed() {
	notify_property_list_changed_pending = false;
	notify_property_list_changed();
}

void MultiResourceEdit::add_resource(const String &p_path) {
	resource_paths.push_back(p_path);
}

int MultiResourceEdit::get_resource_count() const {
	return resource_paths.size();
}

String MultiResourceEdit::get_resource_path(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, get_resource_count(), String());
	return resource_paths[p_index];
}

StringName MultiResourceEdit::get_edited_class_name() const {
	_ensure_resources_loaded();

	if (loaded_resources.is_empty()) {
		return SNAME("Resource");
	}

	StringName class_name = loaded_resources[0]->get_class_name();

	bool check_again = true;
	while (check_again) {
		check_again = false;

		if (class_name == SNAME("Resource") || class_name == StringName()) {
			return SNAME("Resource");
		}

		for (const Ref<Resource> &r : loaded_resources) {
			const StringName resource_class_name = r->get_class_name();
			if (class_name == resource_class_name || ClassDB::is_parent_class(resource_class_name, class_name)) {
				continue;
			}

			class_name = ClassDB::get_parent_class(class_name);
			check_again = true;
			break;
		}
	}

	return class_name;
}

void MultiResourceEdit::set_property_field(const StringName &p_property, const Variant &p_value, const String &p_field) {
	Variant::Type type = p_value.get_type();
	if (type == Variant::ARRAY || type == Variant::DICTIONARY) {
		_set_impl(p_property, p_value, "");
	} else {
		_set_impl(p_property, p_value, p_field);
	}
}

void MultiResourceEdit::_bind_methods() {
	ClassDB::bind_method("_hide_script_from_inspector", &MultiResourceEdit::_hide_script_from_inspector);
	ClassDB::bind_method("_hide_metadata_from_inspector", &MultiResourceEdit::_hide_metadata_from_inspector);
	ClassDB::bind_method("_get_editor_name", &MultiResourceEdit::_get_editor_name);
}
