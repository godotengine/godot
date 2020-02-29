/*************************************************************************/
/*  editor_sectioned_inspector.h                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef EDITOR_SECTIONED_INSPECTOR_H
#define EDITOR_SECTIONED_INSPECTOR_H

#include "editor/editor_inspector.h"
#include "scene/gui/split_container.h"
#include "scene/gui/tree.h"

class SectionedInspector;

class SectionedInspectorFilter : public Object {

	GDCLASS(SectionedInspectorFilter, Object);

	SectionedInspector *sectioned_inspector;
	Object *edited;
	String section;
	bool allow_sub;

	bool _set(const StringName &p_name, const Variant &p_value) {

		if (!edited)
			return false;

		String name = p_name;
		if (section != "") {
			name = section + "/" + name;
		}

		bool valid;
		edited->set(name, p_value, &valid);
		return valid;
	}

	bool _get(const StringName &p_name, Variant &r_ret) const {

		if (!edited)
			return false;

		String name = p_name;
		if (section != "") {
			name = section + "/" + name;
		}

		bool valid = false;

		r_ret = edited->get(name, &valid);
		return valid;
	}
	void _get_property_list(List<PropertyInfo> *p_list) const {

		if (!edited)
			return;

		List<PropertyInfo> pinfo;
		edited->get_property_list(&pinfo);
		for (List<PropertyInfo>::Element *E = pinfo.front(); E; E = E->next()) {

			PropertyInfo pi = E->get();
			int sp = pi.name.find("/");

			if (pi.name == "resource_path" || pi.name == "resource_name" || pi.name == "resource_local_to_scene" || pi.name.begins_with("script/") || pi.name.begins_with("_global_script")) //skip resource stuff
				continue;

			if (sp == -1) {
				pi.name = "global/" + pi.name;
			}

			if (pi.name.begins_with(section + "/")) {
				pi.name = pi.name.replace_first(section + "/", "");
				if (!allow_sub && pi.name.find("/") != -1)
					continue;
				p_list->push_back(pi);
			}
		}
	}

	bool property_can_revert(const String &p_name) {

		return edited->call("property_can_revert", section + "/" + p_name);
	}

	Variant property_get_revert(const String &p_name) {

		return edited->call("property_get_revert", section + "/" + p_name);
	}

protected:
	static void _bind_methods() {

		ClassDB::bind_method("property_can_revert", &SectionedInspectorFilter::property_can_revert);
		ClassDB::bind_method("property_get_revert", &SectionedInspectorFilter::property_get_revert);
	}

public:
	SectionedInspector *_get_sectioned_inspector() {
		return sectioned_inspector;
	}

	void _set_sectioned_inspector(SectionedInspector *p_sectioned_inspector) {
		sectioned_inspector = p_sectioned_inspector;
	}

	void set_section(const String &p_section, bool p_allow_sub) {

		section = p_section;
		allow_sub = p_allow_sub;
		_change_notify();
	}

	void set_edited(Object *p_edited) {
		edited = p_edited;
		_change_notify();
	}

	SectionedInspectorFilter() {
		edited = NULL;
	}
};

class SectionedInspector : public HSplitContainer {

	GDCLASS(SectionedInspector, HSplitContainer);

	ObjectID obj;

	Tree *sections;
	SectionedInspectorFilter *filter;

	Map<String, TreeItem *> section_map;
	EditorInspector *inspector;
	LineEdit *search_box;

	String selected_category;

	static void _bind_methods();
	void _section_selected();

	void _search_changed(const String &p_what);

public:
	void register_search_box(LineEdit *p_box);
	EditorInspector *get_inspector();
	void edit(Object *p_object);
	String get_full_item_path(const String &p_item);

	void set_current_section(const String &p_section);
	String get_current_section() const;

	void update_category_list();

	SectionedInspector();
	~SectionedInspector();
};
#endif // EDITOR_SECTIONED_INSPECTOR_H
