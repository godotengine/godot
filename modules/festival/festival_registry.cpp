/**************************************************************************/
/*  festival_registry.cpp                                                 */
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

#include "festival_registry.h"

#include "core/object/class_db.h"

#include "core/io/dir_access.h"
#include "core/io/resource_loader.h"

FestivalRegistry *FestivalRegistry::singleton = nullptr;

FestivalRegistry *FestivalRegistry::get_singleton() { return singleton; }

void FestivalRegistry::register_npc(const Ref<FestivalNPCProfile> &p_npc) {
	ERR_FAIL_COND(p_npc.is_null());
	ERR_FAIL_COND_MSG(p_npc->get_id() == StringName(), "FestivalNPCProfile has an empty id and cannot be registered.");
	npcs[p_npc->get_id()] = p_npc;
}

void FestivalRegistry::register_outfit(const Ref<FestivalOutfit> &p_outfit) {
	ERR_FAIL_COND(p_outfit.is_null());
	ERR_FAIL_COND_MSG(p_outfit->get_id() == StringName(), "FestivalOutfit has an empty id and cannot be registered.");
	outfits[p_outfit->get_id()] = p_outfit;
}

void FestivalRegistry::register_item(const Ref<FestivalItem> &p_item) {
	ERR_FAIL_COND(p_item.is_null());
	ERR_FAIL_COND_MSG(p_item->get_id() == StringName(), "FestivalItem has an empty id and cannot be registered.");
	items[p_item->get_id()] = p_item;
}

void FestivalRegistry::register_knowledge(const Ref<FestivalKnowledge> &p_knowledge) {
	ERR_FAIL_COND(p_knowledge.is_null());
	ERR_FAIL_COND_MSG(p_knowledge->get_id() == StringName(), "FestivalKnowledge has an empty id and cannot be registered.");
	knowledge[p_knowledge->get_id()] = p_knowledge;
}

void FestivalRegistry::register_location(const Ref<FestivalLocation> &p_location) {
	ERR_FAIL_COND(p_location.is_null());
	ERR_FAIL_COND_MSG(p_location->get_id() == StringName(), "FestivalLocation has an empty id and cannot be registered.");
	locations[p_location->get_id()] = p_location;
}

void FestivalRegistry::register_plot_hook(const Ref<FestivalPlotHook> &p_plot_hook) {
	ERR_FAIL_COND(p_plot_hook.is_null());
	ERR_FAIL_COND_MSG(p_plot_hook->get_id() == StringName(), "FestivalPlotHook has an empty id and cannot be registered.");
	plot_hooks[p_plot_hook->get_id()] = p_plot_hook;
}

void FestivalRegistry::register_event(const Ref<FestivalEvent> &p_event) {
	ERR_FAIL_COND(p_event.is_null());
	ERR_FAIL_COND_MSG(p_event->get_id() == StringName(), "FestivalEvent has an empty id and cannot be registered.");
	events[p_event->get_id()] = p_event;
}

bool FestivalRegistry::register_resource(const Ref<Resource> &p_resource) {
	if (p_resource.is_null()) {
		return false;
	}
	Ref<FestivalNPCProfile> npc = p_resource;
	if (npc.is_valid()) {
		register_npc(npc);
		return true;
	}
	Ref<FestivalOutfit> outfit = p_resource;
	if (outfit.is_valid()) {
		register_outfit(outfit);
		return true;
	}
	Ref<FestivalItem> item = p_resource;
	if (item.is_valid()) {
		register_item(item);
		return true;
	}
	Ref<FestivalKnowledge> know = p_resource;
	if (know.is_valid()) {
		register_knowledge(know);
		return true;
	}
	Ref<FestivalLocation> location = p_resource;
	if (location.is_valid()) {
		register_location(location);
		return true;
	}
	Ref<FestivalPlotHook> plot_hook = p_resource;
	if (plot_hook.is_valid()) {
		register_plot_hook(plot_hook);
		return true;
	}
	Ref<FestivalEvent> event = p_resource;
	if (event.is_valid()) {
		register_event(event);
		return true;
	}
	return false;
}

Ref<FestivalNPCProfile> FestivalRegistry::get_npc(const StringName &p_id) const {
	const Ref<FestivalNPCProfile> *r = npcs.getptr(p_id);
	return r ? *r : Ref<FestivalNPCProfile>();
}

Ref<FestivalOutfit> FestivalRegistry::get_outfit(const StringName &p_id) const {
	const Ref<FestivalOutfit> *r = outfits.getptr(p_id);
	return r ? *r : Ref<FestivalOutfit>();
}

Ref<FestivalItem> FestivalRegistry::get_item(const StringName &p_id) const {
	const Ref<FestivalItem> *r = items.getptr(p_id);
	return r ? *r : Ref<FestivalItem>();
}

Ref<FestivalKnowledge> FestivalRegistry::get_knowledge(const StringName &p_id) const {
	const Ref<FestivalKnowledge> *r = knowledge.getptr(p_id);
	return r ? *r : Ref<FestivalKnowledge>();
}

Ref<FestivalLocation> FestivalRegistry::get_location(const StringName &p_id) const {
	const Ref<FestivalLocation> *r = locations.getptr(p_id);
	return r ? *r : Ref<FestivalLocation>();
}

Ref<FestivalPlotHook> FestivalRegistry::get_plot_hook(const StringName &p_id) const {
	const Ref<FestivalPlotHook> *r = plot_hooks.getptr(p_id);
	return r ? *r : Ref<FestivalPlotHook>();
}

Ref<FestivalEvent> FestivalRegistry::get_event(const StringName &p_id) const {
	const Ref<FestivalEvent> *r = events.getptr(p_id);
	return r ? *r : Ref<FestivalEvent>();
}

bool FestivalRegistry::has_npc(const StringName &p_id) const { return npcs.has(p_id); }
bool FestivalRegistry::has_outfit(const StringName &p_id) const { return outfits.has(p_id); }
bool FestivalRegistry::has_item(const StringName &p_id) const { return items.has(p_id); }
bool FestivalRegistry::has_knowledge(const StringName &p_id) const { return knowledge.has(p_id); }
bool FestivalRegistry::has_location(const StringName &p_id) const { return locations.has(p_id); }
bool FestivalRegistry::has_plot_hook(const StringName &p_id) const { return plot_hooks.has(p_id); }
bool FestivalRegistry::has_event(const StringName &p_id) const { return events.has(p_id); }

PackedStringArray FestivalRegistry::get_npc_ids() const {
	PackedStringArray out;
	for (const KeyValue<StringName, Ref<FestivalNPCProfile>> &E : npcs) {
		out.push_back(E.key);
	}
	return out;
}

PackedStringArray FestivalRegistry::get_outfit_ids() const {
	PackedStringArray out;
	for (const KeyValue<StringName, Ref<FestivalOutfit>> &E : outfits) {
		out.push_back(E.key);
	}
	return out;
}

PackedStringArray FestivalRegistry::get_item_ids() const {
	PackedStringArray out;
	for (const KeyValue<StringName, Ref<FestivalItem>> &E : items) {
		out.push_back(E.key);
	}
	return out;
}

PackedStringArray FestivalRegistry::get_knowledge_ids() const {
	PackedStringArray out;
	for (const KeyValue<StringName, Ref<FestivalKnowledge>> &E : knowledge) {
		out.push_back(E.key);
	}
	return out;
}

PackedStringArray FestivalRegistry::get_location_ids() const {
	PackedStringArray out;
	for (const KeyValue<StringName, Ref<FestivalLocation>> &E : locations) {
		out.push_back(E.key);
	}
	return out;
}

PackedStringArray FestivalRegistry::get_plot_hook_ids() const {
	PackedStringArray out;
	for (const KeyValue<StringName, Ref<FestivalPlotHook>> &E : plot_hooks) {
		out.push_back(E.key);
	}
	return out;
}

PackedStringArray FestivalRegistry::get_event_ids() const {
	PackedStringArray out;
	for (const KeyValue<StringName, Ref<FestivalEvent>> &E : events) {
		out.push_back(E.key);
	}
	return out;
}

void FestivalRegistry::set_world_info(const Dictionary &p_world_info) { world_info = p_world_info; }
Dictionary FestivalRegistry::get_world_info() const { return world_info; }

int FestivalRegistry::scan_directory(const String &p_path) {
	int count = 0;
	Ref<DirAccess> da = DirAccess::open(p_path);
	if (da.is_null()) {
		return 0;
	}
	da->list_dir_begin();
	String f = da->get_next();
	while (!f.is_empty()) {
		if (f == "." || f == "..") {
			f = da->get_next();
			continue;
		}
		const String full = p_path.path_join(f);
		if (da->current_is_dir()) {
			count += scan_directory(full);
		} else if (f.ends_with(".tres") || f.ends_with(".res")) {
			Ref<Resource> res = ResourceLoader::load(full);
			if (res.is_valid() && register_resource(res)) {
				count++;
			}
		}
		f = da->get_next();
	}
	da->list_dir_end();
	return count;
}

void FestivalRegistry::clear() {
	npcs.clear();
	outfits.clear();
	items.clear();
	knowledge.clear();
	locations.clear();
	plot_hooks.clear();
	events.clear();
	world_info = Dictionary();
}

FestivalRegistry::FestivalRegistry() {
	ERR_FAIL_COND(singleton != nullptr);
	singleton = this;
}

FestivalRegistry::~FestivalRegistry() {
	if (singleton == this) {
		singleton = nullptr;
	}
}

void FestivalRegistry::_bind_methods() {
	ClassDB::bind_method(D_METHOD("register_npc", "npc"), &FestivalRegistry::register_npc);
	ClassDB::bind_method(D_METHOD("register_outfit", "outfit"), &FestivalRegistry::register_outfit);
	ClassDB::bind_method(D_METHOD("register_item", "item"), &FestivalRegistry::register_item);
	ClassDB::bind_method(D_METHOD("register_knowledge", "knowledge"), &FestivalRegistry::register_knowledge);
	ClassDB::bind_method(D_METHOD("register_location", "location"), &FestivalRegistry::register_location);
	ClassDB::bind_method(D_METHOD("register_plot_hook", "plot_hook"), &FestivalRegistry::register_plot_hook);
	ClassDB::bind_method(D_METHOD("register_event", "event"), &FestivalRegistry::register_event);
	ClassDB::bind_method(D_METHOD("register_resource", "resource"), &FestivalRegistry::register_resource);

	ClassDB::bind_method(D_METHOD("get_npc", "id"), &FestivalRegistry::get_npc);
	ClassDB::bind_method(D_METHOD("get_outfit", "id"), &FestivalRegistry::get_outfit);
	ClassDB::bind_method(D_METHOD("get_item", "id"), &FestivalRegistry::get_item);
	ClassDB::bind_method(D_METHOD("get_knowledge", "id"), &FestivalRegistry::get_knowledge);
	ClassDB::bind_method(D_METHOD("get_location", "id"), &FestivalRegistry::get_location);
	ClassDB::bind_method(D_METHOD("get_plot_hook", "id"), &FestivalRegistry::get_plot_hook);
	ClassDB::bind_method(D_METHOD("get_event", "id"), &FestivalRegistry::get_event);

	ClassDB::bind_method(D_METHOD("has_npc", "id"), &FestivalRegistry::has_npc);
	ClassDB::bind_method(D_METHOD("has_outfit", "id"), &FestivalRegistry::has_outfit);
	ClassDB::bind_method(D_METHOD("has_item", "id"), &FestivalRegistry::has_item);
	ClassDB::bind_method(D_METHOD("has_knowledge", "id"), &FestivalRegistry::has_knowledge);
	ClassDB::bind_method(D_METHOD("has_location", "id"), &FestivalRegistry::has_location);
	ClassDB::bind_method(D_METHOD("has_plot_hook", "id"), &FestivalRegistry::has_plot_hook);
	ClassDB::bind_method(D_METHOD("has_event", "id"), &FestivalRegistry::has_event);

	ClassDB::bind_method(D_METHOD("get_npc_ids"), &FestivalRegistry::get_npc_ids);
	ClassDB::bind_method(D_METHOD("get_outfit_ids"), &FestivalRegistry::get_outfit_ids);
	ClassDB::bind_method(D_METHOD("get_item_ids"), &FestivalRegistry::get_item_ids);
	ClassDB::bind_method(D_METHOD("get_knowledge_ids"), &FestivalRegistry::get_knowledge_ids);
	ClassDB::bind_method(D_METHOD("get_location_ids"), &FestivalRegistry::get_location_ids);
	ClassDB::bind_method(D_METHOD("get_plot_hook_ids"), &FestivalRegistry::get_plot_hook_ids);
	ClassDB::bind_method(D_METHOD("get_event_ids"), &FestivalRegistry::get_event_ids);
	ClassDB::bind_method(D_METHOD("set_world_info", "world_info"), &FestivalRegistry::set_world_info);
	ClassDB::bind_method(D_METHOD("get_world_info"), &FestivalRegistry::get_world_info);

	ClassDB::bind_method(D_METHOD("scan_directory", "path"), &FestivalRegistry::scan_directory);
	ClassDB::bind_method(D_METHOD("clear"), &FestivalRegistry::clear);
}
