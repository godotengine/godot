/**
 * bt_repeat_until_failure.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef BT_REPEAT_UNTIL_FAILURE_H
#define BT_REPEAT_UNTIL_FAILURE_H

#include "../bt_decorator.h"

class BTRepeatUntilFailure : public BTDecorator {
	GDCLASS(BTRepeatUntilFailure, BTDecorator);
	TASK_CATEGORY(Decorators);

protected:
	static void _bind_methods() {}

	virtual Status _tick(double p_delta) override;
};

#endif // BT_REPEAT_UNTIL_FAILURE_H
