/**
 * bt_delay.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef BT_DELAY_H
#define BT_DELAY_H

#include "../bt_decorator.h"

class BTDelay : public BTDecorator {
	GDCLASS(BTDelay, BTDecorator);
	TASK_CATEGORY(Decorators);

private:
	double seconds = 1.0;

protected:
	static void _bind_methods();

	virtual String _generate_name() override;
	virtual Status _tick(double p_delta) override;

public:
	void set_seconds(double p_value);
	double get_seconds() const { return seconds; }
};

#endif // BT_DELAY_H
