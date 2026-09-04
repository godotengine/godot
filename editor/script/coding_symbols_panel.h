/**************************************************************************/
/*  coding_symbols_panel.h                                                */
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

#include "scene/gui/box_container.h"
#include "scene/gui/panel_container.h"

class BoxContainer;
class Button;
class CodingSymbolsPanel;
class ConfirmationDialog;
class HFlowContainer;
class LineEdit;
class SpinBox;


class SymbolSettingsRow : public HBoxContainer {
	GDCLASS(SymbolSettingsRow, HBoxContainer);

public:
	int index = -1;
	CodingSymbolsPanel *panel = nullptr;
};

class CodingSymbolsPanel : public PanelContainer {
	GDCLASS(CodingSymbolsPanel, PanelContainer);
	friend class SymbolSettingsRow;

private:
	struct SymbolData {
		String label;
		String code;
		Button *panel_button = nullptr;
	};

	HFlowContainer *flow_container = nullptr;
	Button *expand_button = nullptr;
	Button *settings_button = nullptr;

	ConfirmationDialog *settings_dialog = nullptr;
	VBoxContainer *settings_list_vbox = nullptr;
	LineEdit *custom_label_edit = nullptr;
	LineEdit *custom_code_edit = nullptr;
	SpinBox *visible_count_spin = nullptr;
	Button *add_btn = nullptr;
	Button *default_btn = nullptr;

	Vector<SymbolData> symbols_metadata;
	int visible_limit = 10;
	bool expanded = false;

	void _add_custom_button(const String &p_label, const String &p_code);
	void _on_symbol_pressed(const String &p_symbol);
	void _toggle_expand();
	void _on_settings_pressed();
	void _rebuild_settings_list();
	void _rebuild_toolbar();
	void _move_symbol_relative(int p_idx, int p_delta);
	void _delete_symbol(int p_idx);
	void _on_visibility_toggled(bool v, int i);
	void _on_add_custom_entry();
	void _on_visible_limit_changed(double p_value);
	void _load_defaults();
	void _hardware_keyboard_connected(bool p_connected);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	CodingSymbolsPanel();
};
