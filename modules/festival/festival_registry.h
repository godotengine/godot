/**************************************************************************/
/*  festival_registry.h                                                   */
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

#include "festival_item.h"
#include "festival_knowledge.h"
#include "festival_npc_profile.h"
#include "festival_outfit.h"

// A lookup table of all authored content, keyed by each resource's `id`. Systems
// resolve ids to resources through here (e.g. mapping the current outfit id to
// its perceived role and authority). Content can be registered by hand or bulk
// loaded from a directory of .tres/.res files.
class FestivalRegistry : public Object {
	GDCLASS(FestivalRegistry, Object);

	static FestivalRegistry *singleton;

	HashMap<StringName, Ref<FestivalNPCProfile>> npcs;
	HashMap<StringName, Ref<FestivalOutfit>> outfits;
	HashMap<StringName, Ref<FestivalItem>> items;
	HashMap<StringName, Ref<FestivalKnowledge>> knowledge;

protected:
	static void _bind_methods();

public:
	static FestivalRegistry *get_singleton();

	void register_npc(const Ref<FestivalNPCProfile> &p_npc);
	void register_outfit(const Ref<FestivalOutfit> &p_outfit);
	void register_item(const Ref<FestivalItem> &p_item);
	void register_knowledge(const Ref<FestivalKnowledge> &p_knowledge);
	// Dispatch a resource to the correct table by its type. Returns true if it
	// was a recognized Festival resource.
	bool register_resource(const Ref<Resource> &p_resource);

	Ref<FestivalNPCProfile> get_npc(const StringName &p_id) const;
	Ref<FestivalOutfit> get_outfit(const StringName &p_id) const;
	Ref<FestivalItem> get_item(const StringName &p_id) const;
	Ref<FestivalKnowledge> get_knowledge(const StringName &p_id) const;

	bool has_npc(const StringName &p_id) const;
	bool has_outfit(const StringName &p_id) const;
	bool has_item(const StringName &p_id) const;
	bool has_knowledge(const StringName &p_id) const;

	PackedStringArray get_npc_ids() const;
	PackedStringArray get_outfit_ids() const;
	PackedStringArray get_item_ids() const;
	PackedStringArray get_knowledge_ids() const;

	// Recursively load every .tres/.res under a directory and register it.
	// Returns the number of resources registered.
	int scan_directory(const String &p_path);
	void clear();

	FestivalRegistry();
	~FestivalRegistry();
};
