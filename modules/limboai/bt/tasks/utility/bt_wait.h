/**
 * bt_wait.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef BT_WAIT_H
#define BT_WAIT_H

#include "../bt_action.h"

class BTWait : public BTAction {
	GDCLASS(BTWait, BTAction);
	TASK_CATEGORY(Utility);

private:
	double duration = 1.0;

protected:
	static void _bind_methods();

	virtual String _generate_name() override;
	virtual Status _tick(double p_delta) override;

public:
	void set_duration(double p_value) {
		duration = p_value;
		emit_changed();
	}
	double get_duration() const { return duration; }
};

#endif // BT_WAIT_H
