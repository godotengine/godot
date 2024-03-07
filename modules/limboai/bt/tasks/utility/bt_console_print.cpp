/**
 * bt_console_print.cpp
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#include "bt_console_print.h"

#include "../../../util/limbo_compat.h"

String BTConsolePrint::_generate_name() {
	String tx = text;
	if (text.length() > 30) {
		tx = text.substr(0, 30) + "...";
	}
	tx = tx.replace("\"", "\\\"");
	tx = tx.replace("\r", "\\r");
	tx = tx.replace("\t", "\\t");
	tx = tx.replace("\n", "\\n");
	if (bb_format_parameters.size() > 0) {
		return vformat("ConsolePrint  text: \"%s\"  params: %s", tx, bb_format_parameters);
	}
	return vformat("ConsolePrint  text: \"%s\"", tx);
}

BT::Status BTConsolePrint::_tick(double p_delta) {
	switch (bb_format_parameters.size()) {
		case 0: {
			PRINT_LINE(text);
		} break;
		case 1: {
			PRINT_LINE(vformat(text, get_blackboard()->get_var(bb_format_parameters[0], "")));
		} break;
		case 2: {
			PRINT_LINE(vformat(text, get_blackboard()->get_var(bb_format_parameters[0], ""),
					get_blackboard()->get_var(bb_format_parameters[1], "")));
		} break;
		case 3: {
			PRINT_LINE(vformat(text, get_blackboard()->get_var(bb_format_parameters[0], ""),
					get_blackboard()->get_var(bb_format_parameters[1], ""),
					get_blackboard()->get_var(bb_format_parameters[2], "")));
		} break;
		case 4: {
			PRINT_LINE(vformat(text, get_blackboard()->get_var(bb_format_parameters[0], ""),
					get_blackboard()->get_var(bb_format_parameters[1], ""),
					get_blackboard()->get_var(bb_format_parameters[2], ""),
					get_blackboard()->get_var(bb_format_parameters[3], "")));
		} break;
		case 5:
		default: {
			PRINT_LINE(vformat(text, get_blackboard()->get_var(bb_format_parameters[0], ""),
					get_blackboard()->get_var(bb_format_parameters[1], ""),
					get_blackboard()->get_var(bb_format_parameters[2], ""),
					get_blackboard()->get_var(bb_format_parameters[3], ""),
					get_blackboard()->get_var(bb_format_parameters[4], "")));
		} break;
	}
	return SUCCESS;
}

PackedStringArray BTConsolePrint::get_configuration_warnings() {
	PackedStringArray warnings = BTAction::get_configuration_warnings();
	if (bb_format_parameters.size() > 5) {
		warnings.append("ConsolePrint supports up to 5 format arguments.");
	}
	return warnings;
}

void BTConsolePrint::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_text", "text"), &BTConsolePrint::set_text);
	ClassDB::bind_method(D_METHOD("get_text"), &BTConsolePrint::get_text);
	ClassDB::bind_method(D_METHOD("set_bb_format_parameters", "params"), &BTConsolePrint::set_bb_format_parameters);
	ClassDB::bind_method(D_METHOD("get_bb_format_parameters"), &BTConsolePrint::get_bb_format_parameters);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "text", PROPERTY_HINT_MULTILINE_TEXT), "set_text", "get_text");
	ADD_PROPERTY(PropertyInfo(Variant::PACKED_STRING_ARRAY, "bb_format_parameters"), "set_bb_format_parameters", "get_bb_format_parameters");
}
