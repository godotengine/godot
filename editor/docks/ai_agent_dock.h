/**************************************************************************/
/*  ai_agent_dock.h                                                       */
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

#include "core/templates/vector.h"
#include "editor/docks/editor_dock.h"

class Button;
class Label;
class LineEdit;
class PanelContainer;
class StyleBoxFlat;
class TextureRect;
class VBoxContainer;

class AIAgentDock : public EditorDock {
	GDCLASS(AIAgentDock, EditorDock);

	Button *refresh_button = nullptr;
	Button *settings_button = nullptr;
	Button *compose_button = nullptr;
	Button *add_button = nullptr;
	Button *send_button = nullptr;

	LineEdit *prompt = nullptr;
	TextureRect *empty_state_icon = nullptr;
	PanelContainer *prompt_panel = nullptr;

	Ref<StyleBoxFlat> prompt_panel_style;

	Vector<Label *> muted_labels;
	Vector<Label *> access_labels;

	static inline AIAgentDock *singleton = nullptr;

	void _add_task_row(VBoxContainer *p_parent, const String &p_task, const String &p_time);
	Label *_create_label(const String &p_text, bool p_muted = false);
	void _update_theme();

protected:
	void _notification(int p_what);

public:
	static AIAgentDock *get_singleton() { return singleton; }

	AIAgentDock();
	~AIAgentDock();
};
