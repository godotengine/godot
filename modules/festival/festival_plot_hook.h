/**************************************************************************/
/*  festival_plot_hook.h                                                  */
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

// A quest / narrative thread. Plot hooks are the hub entity of a Secret Census
// world: they link NPCs, locations, items, rumors, events, outfits and even
// specific NPC secrets together into one storyline. Each entry of
// `linked_npc_secrets` is a Dictionary { "npcId": String, "secretType":
// "secret"|"darkSecret" } (camelCase keys are part of the import contract —
// see SECRET_CENSUS_COMPATIBILITY.md).
class FestivalPlotHook : public Resource {
	GDCLASS(FestivalPlotHook, Resource);

	StringName id;
	String title;
	String description;
	StringName status; // "active", "completed", "failed".
	bool is_progression = false;
	PackedStringArray linked_npc_ids;
	PackedStringArray linked_location_ids;
	PackedStringArray linked_item_ids;
	PackedStringArray linked_rumor_ids;
	PackedStringArray linked_event_ids;
	PackedStringArray linked_outfit_ids;
	Array linked_npc_secrets;
	Dictionary census_data;

protected:
	static void _bind_methods();

public:
	void set_id(const StringName &p_id);
	StringName get_id() const;
	void set_title(const String &p_title);
	String get_title() const;
	void set_description(const String &p_description);
	String get_description() const;
	void set_status(const StringName &p_status);
	StringName get_status() const;
	void set_is_progression(bool p_is_progression);
	bool get_is_progression() const;
	void set_linked_npc_ids(const PackedStringArray &p_ids);
	PackedStringArray get_linked_npc_ids() const;
	void set_linked_location_ids(const PackedStringArray &p_ids);
	PackedStringArray get_linked_location_ids() const;
	void set_linked_item_ids(const PackedStringArray &p_ids);
	PackedStringArray get_linked_item_ids() const;
	void set_linked_rumor_ids(const PackedStringArray &p_ids);
	PackedStringArray get_linked_rumor_ids() const;
	void set_linked_event_ids(const PackedStringArray &p_ids);
	PackedStringArray get_linked_event_ids() const;
	void set_linked_outfit_ids(const PackedStringArray &p_ids);
	PackedStringArray get_linked_outfit_ids() const;
	void set_linked_npc_secrets(const Array &p_secrets);
	Array get_linked_npc_secrets() const;
	void set_census_data(const Dictionary &p_census_data);
	Dictionary get_census_data() const;
};
