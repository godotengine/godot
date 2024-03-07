/**
 * bt_parallel.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef BT_PARALLEL_H
#define BT_PARALLEL_H

#include "../bt_composite.h"

class BTParallel : public BTComposite {
	GDCLASS(BTParallel, BTComposite);
	TASK_CATEGORY(Composites);

private:
	int num_successes_required = 1;
	int num_failures_required = 1;
	bool repeat = false;

protected:
	static void _bind_methods();

	virtual void _enter() override;
	virtual Status _tick(double p_delta) override;

public:
	int get_num_successes_required() const { return num_successes_required; }
	void set_num_successes_required(int p_value) {
		num_successes_required = p_value;
		emit_changed();
	}
	int get_num_failures_required() const { return num_failures_required; }
	void set_num_failures_required(int p_value) {
		num_failures_required = p_value;
		emit_changed();
	}
	bool get_repeat() const { return repeat; }
	void set_repeat(bool p_value) {
		repeat = p_value;
		emit_changed();
	}
};

#endif // BT_PARALLEL_H
