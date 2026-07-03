/**************************************************************************/
/*  festival_director.h                                                   */
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

#include "festival_npc_profile.h"

// The brain that ties every subsystem together and evaluates the core rule:
//
//   NPC_Reaction = Outfit + Held_Items + World_State_Flags + Knowledge
//
// It owns the run lifecycle (begin_run/end_run), resolves what an NPC currently
// perceives and offers, and applies the outcomes of a chosen interaction. It is
// registered as the `Festival` singleton, so game code reads like
// `Festival.begin_run()` and `Festival.resolve_reaction(npc)`.
class FestivalDirector : public Object {
	GDCLASS(FestivalDirector, Object);

	static FestivalDirector *singleton;

	bool _matches_any(const StringName &p_current, const Variant &p_accepted) const;
	bool _check_requirements(const Dictionary &p_interaction) const;
	void _apply_outcomes(const Dictionary &p_interaction);
	Dictionary _find_interaction(const Ref<FestivalNPCProfile> &p_npc, const StringName &p_id) const;

protected:
	static void _bind_methods();

public:
	static FestivalDirector *get_singleton();

	// Run lifecycle.
	void begin_run(int64_t p_seed = -1);
	void end_run();

	// Perception + interaction resolution.
	StringName get_perceived_role() const;
	int get_authority() const;
	Dictionary resolve_reaction(const Ref<FestivalNPCProfile> &p_npc);
	Array get_available_interactions(const Ref<FestivalNPCProfile> &p_npc);
	bool can_interact(const Ref<FestivalNPCProfile> &p_npc, const StringName &p_interaction_id) const;
	bool apply_interaction(const Ref<FestivalNPCProfile> &p_npc, const StringName &p_interaction_id);

	// Knowledge convenience (proxies the persistent notebook).
	void learn(const StringName &p_id);
	bool knows(const StringName &p_id) const;
	bool load_notebook(const String &p_path = "user://festival_notebook.cfg");
	bool save_notebook(const String &p_path = "user://festival_notebook.cfg") const;

	// A snapshot of the full perception context, handy for UI and debugging.
	Dictionary get_context() const;

	FestivalDirector();
	~FestivalDirector();
};
