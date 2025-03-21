/**************************************************************************/
/*  license_integration_tool.h                                            */
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

#include "scene/gui/dialogs.h"

class LicenseIntegrationDialog : public ConfirmationDialog {
	GDCLASS(LicenseIntegrationDialog, ConfirmationDialog);

	Button *learn_more_button = nullptr;
	static constexpr const char *ACTION_LEARN_MORE = "learn_more";

	static constexpr const char *META_LICENSE_INTEGRATION_TOOL = "license_integration_tool";
	static constexpr const char *META_ACKNOWLEDGED = "acknowledged";

	static constexpr const char *TARGET_DIR = "res://addons/godot_license_notices_dialog_gds/";
	static constexpr const char *AUTOLOAD_FILE = "autoload/license_notices.tscn";
	static constexpr const char *AUTOLOAD_NAME = "GodotLicenseNotices";

	static constexpr const char *KEYBIND_NAME = "ui_toggle_licenses_dialog";
	static constexpr const Key KEYBIND_KEY = Key::L | KeyModifierMask::CMD_OR_CTRL | KeyModifierMask::SHIFT;

	bool files_succeeded = false;

	void _on_custom_action(const String &p_action);
	void _integrate_license();
	void _integrate_license_setup();
	void _acknowledged();

	void _on_sources_changed(bool p_changed);

	AcceptDialog *result_dialog = nullptr;

protected:
	void _notification(int p_what);

public:
	bool should_prompt();
	void popup_on_demand();

	LicenseIntegrationDialog();
};
