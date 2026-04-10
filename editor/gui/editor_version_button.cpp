/**************************************************************************/
/*  editor_version_button.cpp                                             */
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

#include "editor_version_button.h"

#include "core/os/time.h"
#include "core/string/string_builder.h"
#include "core/version.h"
#include "servers/display/display_server.h"

String _get_version_string(EditorVersionButton::VersionFormat p_format) {
	String main;
	String hash = GODOT_VERSION_HASH;
	String branch = GODOT_VERSION_GIT_BRANCH;
	const int BRANCH_MAX_LENGTH = 20;
	StringBuilder git_data;

	if (!hash.is_empty() || !branch.is_empty() || GODOT_VERSION_GIT_DIRTY) {
		git_data.append(" ");

		if (p_format == EditorVersionButton::FORMAT_WITH_BUILD || p_format == EditorVersionButton::FORMAT_WITH_NAME_AND_BUILD) {
			if (!hash.is_empty()) {
				git_data.append(hash.left(9));
			}
		} else if (!hash.is_empty()) {
			hash = "";
		}

		if (!branch.is_empty() && branch == "master") {
			branch = "";
		}

		if (!branch.is_empty()) {
			// The KoBeWi clause.
			if (branch.length() > BRANCH_MAX_LENGTH) {
				branch = vformat("%s…", branch.substr(0, BRANCH_MAX_LENGTH));
			}

			git_data.append(vformat("@%s", branch));
		}

		if (GODOT_VERSION_GIT_DIRTY) {
			if (!hash.is_empty() || !branch.is_empty()) {
				git_data.append(" ");
			}
			git_data.append("(dirty)");
		}
	}

	switch (p_format) {
		case EditorVersionButton::FORMAT_BASIC: {
			return GODOT_VERSION_FULL_CONFIG + git_data.as_string();
		} break;
		case EditorVersionButton::FORMAT_WITH_BUILD: {
			main = "v" GODOT_VERSION_FULL_BUILD;
		} break;
		case EditorVersionButton::FORMAT_WITH_NAME_AND_BUILD: {
			main = GODOT_VERSION_FULL_NAME;
		} break;
		default: {
			ERR_FAIL_V_MSG(GODOT_VERSION_FULL_NAME, "Unexpected format: " + itos(p_format));
		} break;
	}

	return main + git_data.as_string();
}

void EditorVersionButton::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_POSTINITIALIZE: {
			// This can't be done in the constructor because theme cache is not ready yet.
			set_auto_translate_mode(AUTO_TRANSLATE_MODE_DISABLED);
			set_text(_get_version_string(format));
		} break;

		case NOTIFICATION_TRANSLATION_CHANGED: {
			String build_date;
			if (GODOT_VERSION_TIMESTAMP > 0) {
				build_date = Time::get_singleton()->get_datetime_string_from_unix_time(GODOT_VERSION_TIMESTAMP, true) + " UTC";
			} else {
				build_date = TTR("(unknown)");
			}
			set_tooltip_text(vformat(TTR("Git commit date: %s\nClick to copy the version information."), build_date));
		} break;
	}
}

void EditorVersionButton::pressed() {
	DisplayServer::get_singleton()->clipboard_set(_get_version_string(FORMAT_WITH_BUILD));
}

EditorVersionButton::EditorVersionButton(VersionFormat p_format) {
	format = p_format;
	set_underline_mode(LinkButton::UNDERLINE_MODE_ON_HOVER);
}
