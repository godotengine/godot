/**
 * bt_console_print.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef BT_CONSOLE_PRINT_H
#define BT_CONSOLE_PRINT_H

#include "../bt_action.h"

class BTConsolePrint : public BTAction {
	GDCLASS(BTConsolePrint, BTAction);
	TASK_CATEGORY(Utility);

private:
	String text;
	PackedStringArray bb_format_parameters;

protected:
	static void _bind_methods();

	virtual String _generate_name() override;
	virtual Status _tick(double p_delta) override;

public:
	void set_text(String p_value) {
		text = p_value;
		emit_changed();
	}
	String get_text() const { return text; }

	void set_bb_format_parameters(const PackedStringArray &p_value) {
		bb_format_parameters = p_value;
		emit_changed();
	}
	PackedStringArray get_bb_format_parameters() const { return bb_format_parameters; }

	virtual PackedStringArray get_configuration_warnings() override;
};

#endif // BT_CONSOLE_PRINT_H
