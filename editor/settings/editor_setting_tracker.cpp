/**************************************************************************/
/*  editor_setting_tracker.cpp                                            */
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

#include "editor_setting_tracker.h"

#include "editor/settings/editor_settings.h"

void EditorSettingTracker::setup(const StringName &p_setting) {
	DEV_ASSERT(setting.is_empty());

	setting = p_setting;
	value = EDITOR_GET(setting);
	EditorSettings::get_singleton()->setting_trackers.insert(this);
}

void EditorSettingTracker::setup(const StringName &p_setting, const Variant &p_default) {
	DEV_ASSERT(setting.is_empty());

	setting = p_setting;
	value = EDITOR_DEF(setting, p_default);
	EditorSettings::get_singleton()->setting_trackers.insert(this);
}

void EditorSettingTracker::set(const Variant &p_value) {
	EditorSettings::get_singleton()->set_setting(setting, p_value);
}

EditorSettingTracker::~EditorSettingTracker() {
	EditorSettings::get_singleton()->setting_trackers.erase(this);
}
