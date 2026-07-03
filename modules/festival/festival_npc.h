/**************************************************************************/
/*  festival_npc.h                                                        */
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

#include "scene/2d/node_2d.h"

#include "festival_npc_profile.h"

// A live islander placed in a scene. It wraps a FestivalNPCProfile, keeps its
// current schedule state in sync with the clock and weather, and exposes a
// one-call interact() that runs the FestivalDirector's perception resolver.
class FestivalNPC : public Node2D {
	GDCLASS(FestivalNPC, Node2D);

	Ref<FestivalNPCProfile> profile;
	Dictionary current_state;

	void _on_phase_changed(int p_from, int p_to);
	void _on_weather_changed(int p_weather);
	void _on_run_started(int p_weather);

protected:
	static void _bind_methods();
	void _notification(int p_what);

public:
	void set_profile(const Ref<FestivalNPCProfile> &p_profile);
	Ref<FestivalNPCProfile> get_profile() const;

	void refresh_state();
	Dictionary get_current_state() const;
	String get_current_location() const;
	String get_current_activity() const;

	// Resolve what this NPC currently perceives and offers to the player.
	Dictionary interact();

	FestivalNPC();
};
