/**************************************************************************/
/*  festival_outfit.h                                                     */
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

// A costume/identity Alex can wear. During the Festival of Disguises the
// islanders believe the costume identity is *real*, so an outfit is the
// primary lever the player has over how NPCs perceive them.
class FestivalOutfit : public Resource {
	GDCLASS(FestivalOutfit, Resource);

	StringName id;
	String display_name;
	StringName role; // Perceived social role, e.g. "constable", "merchant", "civilian".
	int authority = 0; // Higher grants more social access (confessions, fear, discounts...).
	PackedStringArray tags;

protected:
	static void _bind_methods();

public:
	void set_id(const StringName &p_id);
	StringName get_id() const;

	void set_display_name(const String &p_display_name);
	String get_display_name() const;

	void set_role(const StringName &p_role);
	StringName get_role() const;

	void set_authority(int p_authority);
	int get_authority() const;

	void set_tags(const PackedStringArray &p_tags);
	PackedStringArray get_tags() const;

	bool has_tag(const String &p_tag) const;
};
