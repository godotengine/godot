/**
 * bt_repeat.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef BT_REPEAT_H
#define BT_REPEAT_H

#include "../bt_decorator.h"

class BTRepeat : public BTDecorator {
	GDCLASS(BTRepeat, BTDecorator);
	TASK_CATEGORY(Decorators);

private:
	bool forever = false;
	int times = 1;
	bool abort_on_failure = false;
	int cur_iteration = 0;

protected:
	static void _bind_methods();

	void _get_property_list(List<PropertyInfo> *p_list) const;

	virtual String _generate_name() override;
	virtual void _enter() override;
	virtual Status _tick(double p_delta) override;

public:
	void set_forever(bool p_forever);
	bool get_forever() const { return forever; }

	void set_times(int p_value);
	int get_times() const { return times; }

	void set_abort_on_failure(bool p_value);
	bool get_abort_on_failure() const { return abort_on_failure; }

	BTRepeat();
};

#endif // BT_REPEAT_H
