/**
 * bt_random_wait.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef BT_RANDOM_WAIT_H
#define BT_RANDOM_WAIT_H

#include "../bt_action.h"

class BTRandomWait : public BTAction {
	GDCLASS(BTRandomWait, BTAction);
	TASK_CATEGORY(Utility);

private:
	double min_duration = 1.0;
	double max_duration = 2.0;

	double duration = 0.0;

protected:
	static void _bind_methods();

	virtual String _generate_name() override;
	virtual void _enter() override;
	virtual Status _tick(double p_delta) override;

public:
	void set_min_duration(double p_max_duration);
	double get_min_duration() const { return min_duration; }

	void set_max_duration(double p_max_duration);
	double get_max_duration() const { return max_duration; }
};

#endif // BT_RANDOM_WAIT_H
