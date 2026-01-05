#pragma once

#include "core/object/ref_counted.h"
#include "core/io/resource.h"

class MultiResourceEdit : public RefCounted {
	GDCLASS(MultiResourceEdit, RefCounted);

	LocalVector<String> resource_paths;
	mutable Vector<Ref<Resource>> loaded_resources;

	void _ensure_resources_loaded() const;
	bool notify_property_list_changed_pending = false;
	struct PLData {
		int uses = 0;
		PropertyInfo info;
	};

	bool _set_impl(const StringName &p_name, const Variant &p_value, const String &p_field, bool p_undo_redo = true);
	void _queue_notify_property_list_changed();
	void _notify_property_list_changed();

protected:
	static void _bind_methods();

	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;

public:
	bool _hide_script_from_inspector() { return true; }
	bool _hide_metadata_from_inspector() { return true; }

	bool _property_can_revert(const StringName &p_name) const;
	bool _property_get_revert(const StringName &p_name, Variant &r_property) const;
	String _get_editor_name() const;

	void add_resource(const String &p_path);

	int get_resource_count() const;
	String get_resource_path(int p_index) const;
	StringName get_edited_class_name() const;

	void set_property_field(const StringName &p_property, const Variant &p_value, const String &p_field);

	bool is_same_selection(const MultiResourceEdit *p_other) const {
		if (get_resource_count() != p_other->get_resource_count()) {
			return false;
		}
		HashSet<String> paths_in_selection;
		for (const String &path : p_other->resource_paths) {
			paths_in_selection.insert(path);
		}
		for (const String &path : resource_paths) {
			if (!paths_in_selection.has(path)) {
				return false;
			}
		}
		return true;
	}
};
