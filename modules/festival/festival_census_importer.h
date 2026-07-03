/**************************************************************************/
/*  festival_census_importer.h                                            */
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

#include "core/object/object.h"
#include "core/templates/hash_map.h"

#include "festival_event.h"
#include "festival_item.h"
#include "festival_knowledge.h"
#include "festival_location.h"
#include "festival_npc_profile.h"
#include "festival_outfit.h"
#include "festival_plot_hook.h"

// Imports a Secret Census world package (`*.census.json`, format
// "secret-census-world", version 1) and turns every record into native
// festival resources registered in the FestivalRegistry. NPC secrets, rumors
// and passwords become FestivalKnowledge entries with stable derived ids, and
// dialogue entries become director-ready interaction Dictionaries.
//
// The full mapping — including everything that must never change on either
// side — is specified in SECRET_CENSUS_COMPATIBILITY.md next to this file.
//
//     var report := FestivalCensusImporter.import_file(
//             "res://my_world.census.json", "res://content")
//
// Passing a save directory writes each resource as a .tres file
// (content/npcs/<id>.tres, content/locations/<id>.tres, ...) so the whole
// world can be inspected and edited in the Godot editor afterwards. Every
// resource also carries the raw source record in its `census_data` property,
// so no Secret Census detail is lost even when this importer doesn't map it
// to a first-class property yet.
class FestivalCensusImporter : public Object {
	GDCLASS(FestivalCensusImporter, Object);

	static FestivalCensusImporter *singleton;

	struct ImportContext {
		HashMap<String, Ref<FestivalKnowledge>> knowledge; // id -> resource
		HashMap<String, String> item_ids_by_lower_name;
		Vector<Ref<Resource>> npcs, locations, items, outfits, plot_hooks, events;
	};

	static PackedStringArray _to_string_array(const Variant &p_value);
	static Array _to_array(const Variant &p_value);
	static Dictionary _to_dictionary(const Variant &p_value);
	static String _str(const Dictionary &p_dict, const String &p_key);
	static bool _bool(const Dictionary &p_dict, const String &p_key, bool p_default = false);
	static String _sanitize_filename(const String &p_id);

	Ref<FestivalKnowledge> _ensure_password_knowledge(ImportContext &p_ctx, const String &p_password);
	Ref<FestivalKnowledge> _add_knowledge(ImportContext &p_ctx, const String &p_id, FestivalKnowledge::Category p_category, const String &p_subject, const String &p_title, const String &p_body, bool p_veracity, const Dictionary &p_census_data);
	Dictionary _schedule_dict(const Dictionary &p_locations, const String &p_slot_key, const Dictionary &p_location_options, ImportContext &p_ctx);
	Dictionary _interaction_from_dialogue(ImportContext &p_ctx, const Dictionary &p_entry, const String &p_npc_id);
	void _import_npc(ImportContext &p_ctx, const Dictionary &p_record);
	void _import_location(ImportContext &p_ctx, const Dictionary &p_record);
	void _import_item(ImportContext &p_ctx, const Dictionary &p_record);
	void _import_outfit(ImportContext &p_ctx, const Dictionary &p_record);
	void _import_plot_hook(ImportContext &p_ctx, const Dictionary &p_record);
	void _import_event(ImportContext &p_ctx, const Dictionary &p_record);
	int _save_group(const Vector<Ref<Resource>> &p_resources, const String &p_save_dir, const String &p_group);

protected:
	static void _bind_methods();

public:
	static FestivalCensusImporter *get_singleton();

	// The stable knowledge id derived for a password string. LOCKED — the
	// notebook persists these ids across runs and save files.
	static String password_knowledge_id(const String &p_password);

	Dictionary import_text(const String &p_json, const String &p_save_dir = String());
	Dictionary import_file(const String &p_path, const String &p_save_dir = String());

	FestivalCensusImporter();
	~FestivalCensusImporter();
};
