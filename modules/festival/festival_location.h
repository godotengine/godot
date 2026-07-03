/**************************************************************************/
/*  festival_location.h                                                   */
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

#pragma once

#include "core/io/resource.h"

// A place on Kraed Maas. Locations form a hierarchy (overworld > building >
// sub-location > sub-sub-location) via `parent_id`, plus lateral
// `connected_location_ids` edges. NPC schedules, items and events all refer to
// locations by id. Imported verbatim from Secret Census (see
// SECRET_CENSUS_COMPATIBILITY.md); `census_data` keeps the raw source record.
class FestivalLocation : public Resource {
	GDCLASS(FestivalLocation, Resource);

	StringName id;
	String display_name;
	StringName kind; // "overworld", "building", "sub-location", "sub-sub-location".
	StringName parent_id;
	String description;
	String notes;
	String required_password;
	bool is_residence = false;
	bool is_progression = false;
	bool is_template = false;
	PackedStringArray resident_npc_ids;
	PackedStringArray connected_location_ids;
	PackedStringArray linked_item_ids;
	PackedStringArray linked_plot_hook_ids;
	Dictionary census_data;

protected:
	static void _bind_methods();

public:
	void set_id(const StringName &p_id);
	StringName get_id() const;
	void set_display_name(const String &p_display_name);
	String get_display_name() const;
	void set_kind(const StringName &p_kind);
	StringName get_kind() const;
	void set_parent_id(const StringName &p_parent_id);
	StringName get_parent_id() const;
	void set_description(const String &p_description);
	String get_description() const;
	void set_notes(const String &p_notes);
	String get_notes() const;
	void set_required_password(const String &p_required_password);
	String get_required_password() const;
	void set_is_residence(bool p_is_residence);
	bool get_is_residence() const;
	void set_is_progression(bool p_is_progression);
	bool get_is_progression() const;
	void set_is_template(bool p_is_template);
	bool get_is_template() const;
	void set_resident_npc_ids(const PackedStringArray &p_ids);
	PackedStringArray get_resident_npc_ids() const;
	void set_connected_location_ids(const PackedStringArray &p_ids);
	PackedStringArray get_connected_location_ids() const;
	void set_linked_item_ids(const PackedStringArray &p_ids);
	PackedStringArray get_linked_item_ids() const;
	void set_linked_plot_hook_ids(const PackedStringArray &p_ids);
	PackedStringArray get_linked_plot_hook_ids() const;
	void set_census_data(const Dictionary &p_census_data);
	Dictionary get_census_data() const;
};
