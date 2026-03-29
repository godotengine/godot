/**************************************************************************/
/*  agent_dock.h                                                          */
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

#include "editor/docks/editor_dock.h"

class Button;
class Label;
class RichTextLabel;
class TextEdit;
class VBoxContainer;
class HBoxContainer;
class OptionButton;

class AgentDock : public EditorDock {
	GDCLASS(AgentDock, EditorDock);

	static inline AgentDock *singleton = nullptr;

	// Chat UI
	VBoxContainer *main_vbox = nullptr;
	RichTextLabel *chat_display = nullptr;
	HBoxContainer *input_hbox = nullptr;
	TextEdit *chat_input = nullptr;
	Button *send_button = nullptr;

	// Status
	Label *status_label = nullptr;

	// Context indicator
	HBoxContainer *context_bar = nullptr;
	Label *context_label = nullptr;

	// Settings bar
	HBoxContainer *settings_bar = nullptr;
	OptionButton *model_selector = nullptr;
	Button *settings_button = nullptr;

	// State
	bool is_processing = false;
	bool is_streaming = false;

	// Methods
	void _send_message();
	void _on_input_gui_input(const Ref<InputEvent> &p_event);
	void _append_message(const String &p_sender, const String &p_text, const Color &p_color = Color(1, 1, 1));
	void _append_system_message(const String &p_text);
	void _clear_chat();
	void _update_context_display();
	void _on_settings_pressed();
	void _set_processing(bool p_processing);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	static AgentDock *get_singleton() { return singleton; }

	void append_streamed_token(const String &p_token);
	void finish_streamed_response();
	void append_system_message(const String &p_text);
	void set_context_text(const String &p_text);
	void set_status(const String &p_status);

	AgentDock();
	~AgentDock();
};
