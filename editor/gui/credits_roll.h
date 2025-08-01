/**************************************************************************/
/*  credits_roll.h                                                        */
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

#include "scene/gui/popup.h"

class Label;
class VBoxContainer;
class Font;

class CreditsRoll : public Popup {
	GDCLASS(CreditsRoll, Popup);

	enum class LabelSize {
		NORMAL,
		HEADER,
		BIG_HEADER,
	};

	int font_size_normal = 0;
	int font_size_header = 0;
	int font_size_big_header = 0;
	Ref<Font> bold_font;

	bool mouse_enabled = false;
	VBoxContainer *content = nullptr;
	Label *project_manager = nullptr;

	Label *_create_label(const String &p_with_text, LabelSize p_size = LabelSize::NORMAL);
	void _create_nothing(int p_size = -1);
	String _build_string(const char *const *p_from) const;

protected:
	void _notification(int p_what);

public:
	void roll_credits();

	CreditsRoll();
};
