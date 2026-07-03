/**************************************************************************/
/*  festival_census_importer.cpp                                          */
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

#include "festival_census_importer.h"

#include "core/object/class_db.h"

#include "festival_registry.h"

#include "core/io/dir_access.h"
#include "core/io/file_access.h"
#include "core/io/json.h"
#include "core/io/resource_saver.h"

FestivalCensusImporter *FestivalCensusImporter::singleton = nullptr;

FestivalCensusImporter *FestivalCensusImporter::get_singleton() { return singleton; }

FestivalCensusImporter::FestivalCensusImporter() {
	ERR_FAIL_COND(singleton != nullptr);
	singleton = this;
}

FestivalCensusImporter::~FestivalCensusImporter() {
	if (singleton == this) {
		singleton = nullptr;
	}
}

PackedStringArray FestivalCensusImporter::_to_string_array(const Variant &p_value) {
	PackedStringArray out;
	if (p_value.get_type() == Variant::ARRAY) {
		const Array arr = p_value;
		for (int i = 0; i < arr.size(); i++) {
			const String s = arr[i];
			if (!s.is_empty()) {
				out.push_back(s);
			}
		}
	} else if (p_value.get_type() == Variant::PACKED_STRING_ARRAY) {
		out = p_value;
	}
	return out;
}

Array FestivalCensusImporter::_to_array(const Variant &p_value) {
	return p_value.get_type() == Variant::ARRAY ? (Array)p_value : Array();
}

Dictionary FestivalCensusImporter::_to_dictionary(const Variant &p_value) {
	return p_value.get_type() == Variant::DICTIONARY ? (Dictionary)p_value : Dictionary();
}

String FestivalCensusImporter::_str(const Dictionary &p_dict, const String &p_key) {
	const Variant v = p_dict.get(p_key, Variant());
	if (v.get_type() == Variant::NIL) {
		return String();
	}
	return v;
}

bool FestivalCensusImporter::_bool(const Dictionary &p_dict, const String &p_key, bool p_default) {
	const Variant v = p_dict.get(p_key, p_default);
	if (v.get_type() == Variant::NIL) {
		return p_default;
	}
	return v;
}

String FestivalCensusImporter::_sanitize_filename(const String &p_id) {
	String out;
	for (int i = 0; i < p_id.length(); i++) {
		const char32_t c = p_id[i];
		const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '.' || c == '-' || c == '_';
		out += ok ? c : (char32_t)'_';
	}
	return out.is_empty() ? String("unnamed") : out;
}

// LOCKED derivation (see SECRET_CENSUS_COMPATIBILITY.md): lowercase, collapse
// every run of non-alphanumeric characters into "_", trim edge "_", then
// prefix with "password.". Notebook save files persist these ids.
String FestivalCensusImporter::password_knowledge_id(const String &p_password) {
	const String lower = p_password.to_lower();
	String slug;
	bool pending_sep = false;
	for (int i = 0; i < lower.length(); i++) {
		const char32_t c = lower[i];
		const bool alnum = (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9');
		if (alnum) {
			if (pending_sep && !slug.is_empty()) {
				slug += '_';
			}
			pending_sep = false;
			slug += c;
		} else {
			pending_sep = true;
		}
	}
	return "password." + slug;
}

Ref<FestivalKnowledge> FestivalCensusImporter::_add_knowledge(ImportContext &p_ctx, const String &p_id, FestivalKnowledge::Category p_category, const String &p_subject, const String &p_title, const String &p_body, bool p_veracity, const Dictionary &p_census_data) {
	Ref<FestivalKnowledge> *existing = p_ctx.knowledge.getptr(p_id);
	if (existing) {
		return *existing;
	}
	Ref<FestivalKnowledge> know;
	know.instantiate();
	know->set_id(p_id);
	know->set_category(p_category);
	if (!p_subject.is_empty()) {
		know->set_subject(p_subject);
	}
	know->set_title(p_title);
	know->set_body(p_body);
	know->set_veracity(p_veracity);
	know->set_census_data(p_census_data);
	p_ctx.knowledge[p_id] = know;
	return know;
}

Ref<FestivalKnowledge> FestivalCensusImporter::_ensure_password_knowledge(ImportContext &p_ctx, const String &p_password) {
	if (p_password.strip_edges().is_empty()) {
		return Ref<FestivalKnowledge>();
	}
	const String id = password_knowledge_id(p_password);
	Dictionary census;
	census["password"] = p_password;
	return _add_knowledge(p_ctx, id, FestivalKnowledge::CATEGORY_PASSWORD, String(), "Password: " + p_password, p_password, true, census);
}

// Builds one schedule Dictionary from a Secret Census slot ("morningSun",
// "morningRain", "afternoon", "night"): { "location": <id>, "options": [...] }.
// Option requiredPasswords also become password knowledge.
Dictionary FestivalCensusImporter::_schedule_dict(const Dictionary &p_locations, const String &p_slot_key, const Dictionary &p_location_options, ImportContext &p_ctx) {
	Dictionary out;
	const String location = _str(p_locations, p_slot_key);
	if (!location.is_empty()) {
		out["location"] = location;
	}
	const Array options = _to_array(p_location_options.get(p_slot_key, Variant()));
	if (!options.is_empty()) {
		for (int i = 0; i < options.size(); i++) {
			const Dictionary option = _to_dictionary(options[i]);
			_ensure_password_knowledge(p_ctx, _str(option, "requiredPassword"));
		}
		out["options"] = options;
	}
	return out;
}

// DialogueEntry -> director interaction Dictionary. Key mapping is LOCKED,
// see SECRET_CENSUS_COMPATIBILITY.md §5. The raw entry rides along under
// "census" so nothing authored in Secret Census is lost.
Dictionary FestivalCensusImporter::_interaction_from_dialogue(ImportContext &p_ctx, const Dictionary &p_entry, const String &p_npc_id) {
	Dictionary out;
	out["id"] = StringName(_str(p_entry, "id"));
	out["dialogue"] = _str(p_entry, "text");

	const String game_state = _str(p_entry, "gameState");
	out["game_state"] = game_state;
	const String dialogue_type = _str(p_entry, "type");
	if (!dialogue_type.is_empty()) {
		out["dialogue_type"] = dialogue_type;
	}

	// Map recognizably named game states onto clock phases.
	int phase = -1;
	if (game_state.findn("morning") >= 0) {
		phase = 0;
	} else if (game_state.findn("afternoon") >= 0) {
		phase = 1;
	} else if (game_state.findn("night") >= 0) {
		phase = 2;
	}
	if (phase >= 0) {
		Array phases;
		phases.push_back(phase);
		out["requires_phase"] = phases;
	}

	String outfit = _str(p_entry, "outfitId");
	if (outfit.is_empty()) {
		outfit = _str(p_entry, "outfit"); // Legacy field.
	}
	if (!outfit.is_empty()) {
		out["requires_outfit"] = StringName(outfit);
	}

	Array requires_knowledge;
	const String required_password = _str(p_entry, "requiredPassword");
	if (!required_password.strip_edges().is_empty()) {
		_ensure_password_knowledge(p_ctx, required_password);
		requires_knowledge.push_back(password_knowledge_id(required_password));
	}
	if (!requires_knowledge.is_empty()) {
		out["requires_knowledge"] = requires_knowledge;
	}

	if (_bool(p_entry, "hasSpecialItem")) {
		const String item_name = _str(p_entry, "specialItemName");
		if (!item_name.is_empty()) {
			out["special_item_name"] = item_name;
			const String *item_id = p_ctx.item_ids_by_lower_name.getptr(item_name.to_lower());
			if (item_id) {
				Array requires_items;
				requires_items.push_back(*item_id);
				out["requires_items"] = requires_items;
			}
		}
	}

	Array grants_knowledge;
	const String linked_rumor = _str(p_entry, "linkedRumorId");
	if (!linked_rumor.is_empty()) {
		grants_knowledge.push_back("rumor." + linked_rumor);
	}
	if (dialogue_type == "secret") {
		grants_knowledge.push_back("npc." + p_npc_id + ".secret");
	} else if (dialogue_type == "darkSecret") {
		grants_knowledge.push_back("npc." + p_npc_id + ".dark_secret");
	}
	const String gives_password = _str(p_entry, "givesPassword");
	if (!gives_password.strip_edges().is_empty()) {
		_ensure_password_knowledge(p_ctx, gives_password);
		grants_knowledge.push_back(password_knowledge_id(gives_password));
	}
	if (!grants_knowledge.is_empty()) {
		out["grants_knowledge"] = grants_knowledge;
	}

	const String gives_item = _str(p_entry, "givesItemId");
	if (!gives_item.is_empty()) {
		Array gives_items;
		gives_items.push_back(gives_item);
		out["gives_items"] = gives_items;
	}

	const Array triggers_events = _to_array(p_entry.get("triggersEventIds", Variant()));
	if (!triggers_events.is_empty()) {
		out["triggers_events"] = triggers_events;
	}

	out["census"] = p_entry;
	return out;
}

void FestivalCensusImporter::_import_npc(ImportContext &p_ctx, const Dictionary &p_record) {
	const String id = _str(p_record, "id");
	if (id.is_empty()) {
		WARN_PRINT("FestivalCensusImporter: skipping NPC record without id.");
		return;
	}
	const String name = _str(p_record, "name");

	Ref<FestivalNPCProfile> npc;
	npc.instantiate();
	npc->set_id(id);
	npc->set_display_name(name);
	npc->set_species(_str(p_record, "species"));
	npc->set_surface_personality(_str(p_record, "personality"));
	npc->set_backstory(_str(p_record, "backstory"));
	npc->set_workplace(_str(p_record, "workplace"));
	npc->set_occupation(_str(p_record, "occupation"));
	npc->set_residence(_str(p_record, "residence"));
	npc->set_notes(_str(p_record, "notes"));

	// The four guarded facts become persistent knowledge with LOCKED ids.
	const String secret_text = _str(p_record, "secret");
	if (!secret_text.strip_edges().is_empty()) {
		const String kid = "npc." + id + ".secret";
		Dictionary census;
		census["npcId"] = id;
		census["text"] = secret_text;
		_add_knowledge(p_ctx, kid, FestivalKnowledge::CATEGORY_SECRET, id, name.is_empty() ? String("Secret") : name + "'s secret", secret_text, true, census);
		npc->set_secret(kid);
	}
	const String dark_secret_text = _str(p_record, "darkSecret");
	if (!dark_secret_text.strip_edges().is_empty()) {
		const String kid = "npc." + id + ".dark_secret";
		Dictionary census;
		census["npcId"] = id;
		census["text"] = dark_secret_text;
		_add_knowledge(p_ctx, kid, FestivalKnowledge::CATEGORY_DARK_SECRET, id, name.is_empty() ? String("Dark secret") : name + "'s dark secret", dark_secret_text, true, census);
		npc->set_dark_secret(kid);
	}

	const Array rumors = _to_array(p_record.get("rumors", Variant()));
	String first_true_rumor, first_false_rumor;
	for (int i = 0; i < rumors.size(); i++) {
		const Dictionary rumor = _to_dictionary(rumors[i]);
		const String rumor_id = _str(rumor, "id");
		if (rumor_id.is_empty()) {
			continue;
		}
		const bool is_true = _bool(rumor, "isTrue", true);
		const String kid = "rumor." + rumor_id;
		_add_knowledge(p_ctx, kid, is_true ? FestivalKnowledge::CATEGORY_RUMOR : FestivalKnowledge::CATEGORY_FALSE_RUMOR, id, name.is_empty() ? String("Rumor") : "Rumor about " + name, _str(rumor, "text"), is_true, rumor);
		if (is_true && first_true_rumor.is_empty()) {
			first_true_rumor = kid;
		} else if (!is_true && first_false_rumor.is_empty()) {
			first_false_rumor = kid;
		}
	}
	if (!first_true_rumor.is_empty()) {
		npc->set_rumor(first_true_rumor);
	}
	if (!first_false_rumor.is_empty()) {
		npc->set_false_rumor(first_false_rumor);
	}

	// Schedules: morningSun is the base morning schedule, morningRain merges
	// on top of it (phase-scoped) when FestivalWeather rolls rain.
	const Dictionary locations = _to_dictionary(p_record.get("locations", Variant()));
	const Dictionary location_options = _to_dictionary(p_record.get("locationOptions", Variant()));
	npc->set_schedule_morning(_schedule_dict(locations, "morningSun", location_options, p_ctx));
	npc->set_schedule_afternoon(_schedule_dict(locations, "afternoon", location_options, p_ctx));
	npc->set_schedule_night(_schedule_dict(locations, "night", location_options, p_ctx));
	const Dictionary morning_rain = _schedule_dict(locations, "morningRain", location_options, p_ctx);
	if (!morning_rain.is_empty()) {
		Dictionary rain_variant;
		rain_variant["morning"] = morning_rain;
		npc->set_weather_variant_rain(rain_variant);
	}

	const Array dialogue = _to_array(p_record.get("dialogue", Variant()));
	Array interactions;
	for (int i = 0; i < dialogue.size(); i++) {
		const Dictionary entry = _to_dictionary(dialogue[i]);
		if (entry.is_empty()) {
			continue;
		}
		interactions.push_back(_interaction_from_dialogue(p_ctx, entry, id));
	}
	npc->set_interactions(interactions);

	const PackedStringArray gives_passwords = _to_string_array(p_record.get("givesPasswords", Variant()));
	for (int i = 0; i < gives_passwords.size(); i++) {
		_ensure_password_knowledge(p_ctx, gives_passwords[i]);
	}
	npc->set_gives_passwords(gives_passwords);

	npc->set_relationships(_to_array(p_record.get("relationships", Variant())));
	npc->set_custom_fields(_to_array(p_record.get("customFields", Variant())));
	npc->set_linked_plot_hook_ids(_to_string_array(p_record.get("linkedPlotHookIds", Variant())));
	npc->set_census_data(p_record);

	p_ctx.npcs.push_back(npc);
}

void FestivalCensusImporter::_import_location(ImportContext &p_ctx, const Dictionary &p_record) {
	const String id = _str(p_record, "id");
	if (id.is_empty()) {
		WARN_PRINT("FestivalCensusImporter: skipping Location record without id.");
		return;
	}
	Ref<FestivalLocation> location;
	location.instantiate();
	location->set_id(id);
	location->set_display_name(_str(p_record, "name"));
	location->set_kind(_str(p_record, "type"));
	location->set_parent_id(_str(p_record, "parentId"));
	location->set_description(_str(p_record, "description"));
	location->set_notes(_str(p_record, "notes"));
	const String required_password = _str(p_record, "requiredPassword");
	location->set_required_password(required_password);
	_ensure_password_knowledge(p_ctx, required_password);
	location->set_is_residence(_bool(p_record, "isResidence"));
	location->set_is_progression(_bool(p_record, "isProgression"));
	location->set_is_template(_bool(p_record, "isTemplate"));
	location->set_resident_npc_ids(_to_string_array(p_record.get("residentNpcIds", Variant())));
	location->set_connected_location_ids(_to_string_array(p_record.get("connectedLocationIds", Variant())));
	location->set_linked_item_ids(_to_string_array(p_record.get("linkedItemIds", Variant())));
	location->set_linked_plot_hook_ids(_to_string_array(p_record.get("linkedPlotHookIds", Variant())));
	location->set_census_data(p_record);
	p_ctx.locations.push_back(location);
}

void FestivalCensusImporter::_import_item(ImportContext &p_ctx, const Dictionary &p_record) {
	const String id = _str(p_record, "id");
	if (id.is_empty()) {
		WARN_PRINT("FestivalCensusImporter: skipping Item record without id.");
		return;
	}
	Ref<FestivalItem> item;
	item.instantiate();
	item->set_id(id);
	const String name = _str(p_record, "name");
	item->set_display_name(name);
	item->set_description(_str(p_record, "description"));
	item->set_holder_type(_str(p_record, "locationType"));
	item->set_holder_id(_str(p_record, "locationId"));
	item->set_grants_extra_power(_bool(p_record, "grantsExtraPower"));
	item->set_power_description(_str(p_record, "powerDescription"));
	item->set_power_category(_str(p_record, "powerCategory"));
	const String gives_password = _str(p_record, "givesPassword");
	item->set_gives_password(gives_password);
	_ensure_password_knowledge(p_ctx, gives_password);
	item->set_linked_plot_hook_id(_str(p_record, "linkedPlotHookId"));
	item->set_is_progression(_bool(p_record, "isProgression"));
	item->set_stage_availability(_to_array(p_record.get("stageAvailability", Variant())));
	item->set_census_data(p_record);
	if (!name.is_empty()) {
		p_ctx.item_ids_by_lower_name[name.to_lower()] = id;
	}
	p_ctx.items.push_back(item);
}

void FestivalCensusImporter::_import_outfit(ImportContext &p_ctx, const Dictionary &p_record) {
	const String id = _str(p_record, "id");
	if (id.is_empty()) {
		WARN_PRINT("FestivalCensusImporter: skipping Outfit record without id.");
		return;
	}
	Ref<FestivalOutfit> outfit;
	outfit.instantiate();
	outfit->set_id(id);
	outfit->set_display_name(_str(p_record, "name"));
	outfit->set_description(_str(p_record, "description"));
	outfit->set_grants_extra_power(_bool(p_record, "grantsExtraPower"));
	outfit->set_powers(_str(p_record, "powers"));
	outfit->set_stage_availability(_to_array(p_record.get("stageAvailability", Variant())));
	outfit->set_linked_dialogue_ids(_to_string_array(p_record.get("linkedDialogueIds", Variant())));
	outfit->set_census_data(p_record);
	// `role` and `authority` are Festival-side gameplay tuning with no Secret
	// Census source yet — author them in the Godot editor after import.
	p_ctx.outfits.push_back(outfit);
}

void FestivalCensusImporter::_import_plot_hook(ImportContext &p_ctx, const Dictionary &p_record) {
	const String id = _str(p_record, "id");
	if (id.is_empty()) {
		WARN_PRINT("FestivalCensusImporter: skipping PlotHook record without id.");
		return;
	}
	Ref<FestivalPlotHook> hook;
	hook.instantiate();
	hook->set_id(id);
	hook->set_title(_str(p_record, "title"));
	hook->set_description(_str(p_record, "description"));
	hook->set_status(_str(p_record, "status"));
	hook->set_is_progression(_bool(p_record, "isProgression"));
	hook->set_linked_npc_ids(_to_string_array(p_record.get("linkedNpcIds", Variant())));
	hook->set_linked_location_ids(_to_string_array(p_record.get("linkedLocationIds", Variant())));
	hook->set_linked_item_ids(_to_string_array(p_record.get("linkedItemIds", Variant())));
	hook->set_linked_rumor_ids(_to_string_array(p_record.get("linkedRumorIds", Variant())));
	hook->set_linked_event_ids(_to_string_array(p_record.get("linkedEventIds", Variant())));
	hook->set_linked_outfit_ids(_to_string_array(p_record.get("linkedOutfits", Variant())));
	hook->set_linked_npc_secrets(_to_array(p_record.get("linkedNpcSecrets", Variant())));
	hook->set_census_data(p_record);
	p_ctx.plot_hooks.push_back(hook);
}

void FestivalCensusImporter::_import_event(ImportContext &p_ctx, const Dictionary &p_record) {
	const String id = _str(p_record, "id");
	if (id.is_empty()) {
		WARN_PRINT("FestivalCensusImporter: skipping GameEvent record without id.");
		return;
	}
	Ref<FestivalEvent> event;
	event.instantiate();
	event->set_id(id);
	event->set_display_name(_str(p_record, "name"));
	event->set_description(_str(p_record, "description"));
	event->set_script_text(_str(p_record, "script"));
	event->set_location_id(_str(p_record, "locationId"));
	event->set_trigger_npc_id(_str(p_record, "triggerNpcId"));
	event->set_trigger_dialogue_id(_str(p_record, "triggerDialogueId"));
	event->set_character_ids(_to_string_array(p_record.get("characterIds", Variant())));
	event->set_game_states(_to_string_array(p_record.get("gameStates", Variant())));
	event->set_is_progression(_bool(p_record, "isProgression"));
	event->set_census_data(p_record);
	p_ctx.events.push_back(event);
}

int FestivalCensusImporter::_save_group(const Vector<Ref<Resource>> &p_resources, const String &p_save_dir, const String &p_group) {
	if (p_resources.is_empty()) {
		return 0;
	}
	const String dir = p_save_dir.path_join(p_group);
	const Error mkdir_err = DirAccess::make_dir_recursive_absolute(dir);
	ERR_FAIL_COND_V_MSG(mkdir_err != OK && mkdir_err != ERR_ALREADY_EXISTS, 0, vformat("FestivalCensusImporter: cannot create directory \"%s\".", dir));
	int saved = 0;
	for (const Ref<Resource> &res : p_resources) {
		const String id = res->get("id");
		const String path = dir.path_join(_sanitize_filename(id) + ".tres");
		if (ResourceSaver::save(res, path) == OK) {
			saved++;
		} else {
			WARN_PRINT(vformat("FestivalCensusImporter: failed to save \"%s\".", path));
		}
	}
	return saved;
}

Dictionary FestivalCensusImporter::import_text(const String &p_json, const String &p_save_dir) {
	Dictionary summary;
	summary["ok"] = false;

	const Variant parsed = JSON::parse_string(p_json);
	if (parsed.get_type() != Variant::DICTIONARY) {
		summary["error"] = "Not a valid JSON object.";
		return summary;
	}
	const Dictionary package = parsed;

	const String format = _str(package, "format");
	if (format != "secret-census-world") {
		summary["error"] = vformat("Unknown format \"%s\" (expected \"secret-census-world\").", format);
		return summary;
	}
	const int format_version = (int)(double)package.get("formatVersion", 0);
	if (format_version != 1) {
		summary["error"] = vformat("Unsupported formatVersion %d (this engine understands version 1).", format_version);
		return summary;
	}

	ImportContext ctx;

	// Items first so dialogue special-item names can resolve to item ids.
	const Array items = _to_array(package.get("items", Variant()));
	for (int i = 0; i < items.size(); i++) {
		_import_item(ctx, _to_dictionary(items[i]));
	}
	const Array locations = _to_array(package.get("locations", Variant()));
	for (int i = 0; i < locations.size(); i++) {
		_import_location(ctx, _to_dictionary(locations[i]));
	}
	const Array outfits = _to_array(package.get("outfits", Variant()));
	for (int i = 0; i < outfits.size(); i++) {
		_import_outfit(ctx, _to_dictionary(outfits[i]));
	}
	const Array npcs = _to_array(package.get("npcs", Variant()));
	for (int i = 0; i < npcs.size(); i++) {
		_import_npc(ctx, _to_dictionary(npcs[i]));
	}
	const Array plot_hooks = _to_array(package.get("plotHooks", Variant()));
	for (int i = 0; i < plot_hooks.size(); i++) {
		_import_plot_hook(ctx, _to_dictionary(plot_hooks[i]));
	}
	const Array events = _to_array(package.get("events", Variant()));
	for (int i = 0; i < events.size(); i++) {
		_import_event(ctx, _to_dictionary(events[i]));
	}

	FestivalRegistry *registry = FestivalRegistry::get_singleton();
	if (registry) {
		for (const Ref<Resource> &res : ctx.npcs) {
			registry->register_resource(res);
		}
		for (const Ref<Resource> &res : ctx.locations) {
			registry->register_resource(res);
		}
		for (const Ref<Resource> &res : ctx.items) {
			registry->register_resource(res);
		}
		for (const Ref<Resource> &res : ctx.outfits) {
			registry->register_resource(res);
		}
		for (const Ref<Resource> &res : ctx.plot_hooks) {
			registry->register_resource(res);
		}
		for (const Ref<Resource> &res : ctx.events) {
			registry->register_resource(res);
		}
		for (const KeyValue<String, Ref<FestivalKnowledge>> &E : ctx.knowledge) {
			registry->register_knowledge(E.value);
		}

		Dictionary world_info;
		world_info["format_version"] = format_version;
		world_info["exported_at"] = package.get("exportedAt", Variant());
		world_info["game"] = _to_dictionary(package.get("game", Variant()));
		world_info["settings"] = _to_dictionary(package.get("settings", Variant()));
		registry->set_world_info(world_info);
	}

	int saved = 0;
	if (!p_save_dir.is_empty()) {
		saved += _save_group(ctx.npcs, p_save_dir, "npcs");
		saved += _save_group(ctx.locations, p_save_dir, "locations");
		saved += _save_group(ctx.items, p_save_dir, "items");
		saved += _save_group(ctx.outfits, p_save_dir, "outfits");
		saved += _save_group(ctx.plot_hooks, p_save_dir, "plot_hooks");
		saved += _save_group(ctx.events, p_save_dir, "events");
		Vector<Ref<Resource>> knowledge_list;
		for (const KeyValue<String, Ref<FestivalKnowledge>> &E : ctx.knowledge) {
			knowledge_list.push_back(E.value);
		}
		saved += _save_group(knowledge_list, p_save_dir, "knowledge");
	}

	summary["ok"] = true;
	summary["format_version"] = format_version;
	summary["npcs"] = (int)ctx.npcs.size();
	summary["locations"] = (int)ctx.locations.size();
	summary["items"] = (int)ctx.items.size();
	summary["outfits"] = (int)ctx.outfits.size();
	summary["plot_hooks"] = (int)ctx.plot_hooks.size();
	summary["events"] = (int)ctx.events.size();
	summary["knowledge"] = (int)ctx.knowledge.size();
	summary["saved_files"] = saved;
	summary["game"] = _to_dictionary(package.get("game", Variant()));
	return summary;
}

Dictionary FestivalCensusImporter::import_file(const String &p_path, const String &p_save_dir) {
	Error err = OK;
	const String text = FileAccess::get_file_as_string(p_path, &err);
	if (err != OK) {
		Dictionary summary;
		summary["ok"] = false;
		summary["error"] = vformat("Cannot read \"%s\" (error %d).", p_path, (int)err);
		return summary;
	}
	return import_text(text, p_save_dir);
}

void FestivalCensusImporter::_bind_methods() {
	ClassDB::bind_method(D_METHOD("import_file", "path", "save_dir"), &FestivalCensusImporter::import_file, DEFVAL(String()));
	ClassDB::bind_method(D_METHOD("import_text", "json", "save_dir"), &FestivalCensusImporter::import_text, DEFVAL(String()));
	ClassDB::bind_static_method("FestivalCensusImporter", D_METHOD("password_knowledge_id", "password"), &FestivalCensusImporter::password_knowledge_id);
}
