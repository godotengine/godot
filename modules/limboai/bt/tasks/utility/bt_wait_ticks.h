/**
 * bt_wait_ticks.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef BT_WAIT_TICKS_H
#define BT_WAIT_TICKS_H

#include "../bt_action.h"

class BTWaitTicks : public BTAction {
	GDCLASS(BTWaitTicks, BTAction);
	TASK_CATEGORY(Utility);

private:
	int num_ticks = 1;

	int num_passed = 0;

protected:
	static void _bind_methods();

	virtual String _generate_name() override;
	virtual void _enter() override;
	virtual Status _tick(double p_delta) override;

public:
	void set_num_ticks(int p_value) {
		num_ticks = p_value;
		emit_changed();
	}
	int get_num_ticks() const { return num_ticks; }
};

#endif // BT_WAIT_TICKS_H
