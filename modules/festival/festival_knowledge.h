/**************************************************************************/
/*  festival_knowledge.h                                                  */
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
#include "core/variant/binder_common.h"

// A single discoverable fact: a secret, rumor, password, schedule, route, etc.
// Knowledge is the game's real progression currency. It is learned during a run
// but recorded in the FestivalNotebook, which *persists across runs*.
class FestivalKnowledge : public Resource {
	GDCLASS(FestivalKnowledge, Resource);

public:
	enum Category {
		CATEGORY_SECRET,
		CATEGORY_DARK_SECRET,
		CATEGORY_RUMOR,
		CATEGORY_FALSE_RUMOR,
		CATEGORY_PASSWORD,
		CATEGORY_SCHEDULE,
		CATEGORY_ROUTE,
		CATEGORY_FACT,
	};

private:
	StringName id;
	Category category = CATEGORY_FACT;
	StringName subject; // Optional NPC id this knowledge concerns.
	String title;
	String body;
	bool veracity = true; // false => the belief is actually untrue (a false rumor).
	// The raw Secret Census record (e.g. a Rumor with its source and
	// verification details) this knowledge was derived from, verbatim.
	Dictionary census_data;

protected:
	static void _bind_methods();

public:
	void set_id(const StringName &p_id);
	StringName get_id() const;

	void set_category(Category p_category);
	Category get_category() const;

	void set_subject(const StringName &p_subject);
	StringName get_subject() const;

	void set_title(const String &p_title);
	String get_title() const;

	void set_body(const String &p_body);
	String get_body() const;

	void set_veracity(bool p_veracity);
	bool get_veracity() const;

	void set_census_data(const Dictionary &p_census_data);
	Dictionary get_census_data() const;
};

VARIANT_ENUM_CAST(FestivalKnowledge::Category);
