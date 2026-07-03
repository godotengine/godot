/**************************************************************************/
/*  festival_clock.h                                                      */
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
#include "core/templates/hash_set.h"

// The single-day time system. A run always moves Morning -> Afternoon -> Night
// -> Ended and never backwards; content tied to a phase is lost once it passes.
// Phase progression is driven by the game raising milestones and then calling
// advance_phase(), mirroring Majora's Mask style hidden schedule beats.
class FestivalClock : public Object {
	GDCLASS(FestivalClock, Object);

	static FestivalClock *singleton;

public:
	enum Phase {
		PHASE_MORNING,
		PHASE_AFTERNOON,
		PHASE_NIGHT,
		PHASE_ENDED,
	};

private:
	Phase phase = PHASE_MORNING;
	HashSet<StringName> milestones;

protected:
	static void _bind_methods();

public:
	static FestivalClock *get_singleton();

	Phase get_phase() const;
	String get_phase_name() const;
	bool is_ended() const;

	// Advance one phase. Returns false if the run has already ended.
	bool advance_phase();
	// Begin a fresh run at Morning and clear milestones (per-run reset).
	void reset();

	void trigger_milestone(const StringName &p_id);
	bool has_milestone(const StringName &p_id) const;
	PackedStringArray get_milestones() const;

	FestivalClock();
	~FestivalClock();
};

VARIANT_ENUM_CAST(FestivalClock::Phase);
