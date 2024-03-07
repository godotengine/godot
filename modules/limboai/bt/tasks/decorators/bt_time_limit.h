/**
 * bt_time_limit.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef BT_TIME_LIMIT_H
#define BT_TIME_LIMIT_H

#include "../bt_decorator.h"

class BTTimeLimit : public BTDecorator {
	GDCLASS(BTTimeLimit, BTDecorator);
	TASK_CATEGORY(Decorators);

private:
	double time_limit = 5.0;

protected:
	static void _bind_methods();

	virtual String _generate_name() override;
	virtual Status _tick(double p_delta) override;

public:
	void set_time_limit(double p_value);
	double get_time_limit() const { return time_limit; }
};

#endif // BT_TIME_LIMIT_H
