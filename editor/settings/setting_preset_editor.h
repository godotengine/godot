/**************************************************************************/
/*  setting_preset_editor.h                                               */
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

class Label;
class OptionButton;
class Tree;

class SettingPresetEditor : public VBoxContainer {
	GDCLASS(SettingPresetEditor, VBoxContainer);

	HBoxContainer *option_button_container = nullptr;
	Label *setting_preset_label = nullptr;
	OptionButton *setting_preset_selection = nullptr;

	Label *setting_preset_description = nullptr;
	Tree *setting_preset_preview = nullptr;

	void _setting_preset_selected(int p_index);

protected:
	void _notification(int p_what);

public:
	enum SettingPreset {
		SETTING_PRESET_NONE,
		SETTING_PRESET_2D_PIXEL_ART,
		SETTING_PRESET_2D_HIGH_DEFINITION,
		SETTING_PRESET_3D_PIXEL_ART,
		SETTING_PRESET_3D_STYLIZED,
		SETTING_PRESET_3D_HIGH_END,
		SETTING_PRESET_MISC_APPLICATION,
		SETTING_PRESET_MAX,
	};

	String get_setting_preset_name(SettingPreset p_preset);
	String get_setting_preset_description(SettingPreset p_preset);
	StringName get_setting_preset_icon_name(SettingPreset p_preset);
	HashMap<String, Variant> get_setting_preset_values(SettingPreset p_preset);

	SettingPreset get_selected_preset() const;

	SettingPresetEditor();
};
