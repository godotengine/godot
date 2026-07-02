/**************************************************************************/
/*  festival_world.h                                                      */
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
#include "core/templates/hash_set.h"

// All volatile per-run state: world flags, inventory, the outfit Alex currently
// wears, and which items are being visibly presented. Every one of these is
// wiped by reset() at the start of a run -- only the FestivalNotebook survives.
// Together with the notebook and clock/weather this is the perception equation:
//   NPC_Reaction = Outfit + Held_Items + World_State_Flags + Knowledge.
class FestivalWorld : public Object {
	GDCLASS(FestivalWorld, Object);

	static FestivalWorld *singleton;

	HashMap<StringName, Variant> flags;
	HashMap<StringName, int> inventory;
	StringName current_outfit;
	HashSet<StringName> presented;

protected:
	static void _bind_methods();

public:
	static FestivalWorld *get_singleton();

	// Flags.
	void set_flag(const StringName &p_flag, const Variant &p_value);
	Variant get_flag(const StringName &p_flag, const Variant &p_default = Variant()) const;
	bool has_flag(const StringName &p_flag) const;
	void erase_flag(const StringName &p_flag);
	Dictionary get_flags() const;

	// Inventory.
	void add_item(const StringName &p_id, int p_count = 1);
	void remove_item(const StringName &p_id, int p_count = 1);
	bool has_item(const StringName &p_id, int p_count = 1) const;
	int get_item_count(const StringName &p_id) const;
	Dictionary get_items() const;

	// Outfit.
	void set_outfit(const StringName &p_outfit);
	StringName get_outfit() const;

	// Presented items (a visible subset of the inventory).
	void present_item(const StringName &p_id);
	void unpresent_item(const StringName &p_id);
	bool is_presented(const StringName &p_id) const;
	PackedStringArray get_presented() const;
	void clear_presented();

	// Wipe everything (called at the start of every run).
	void reset();

	FestivalWorld();
	~FestivalWorld();
};
