/**
 * bt_decorator.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef BT_DECORATOR_H
#define BT_DECORATOR_H

#include "bt_task.h"

class BTDecorator : public BTTask {
	GDCLASS(BTDecorator, BTTask)

protected:
	static void _bind_methods() {}

	virtual Status _tick(double p_delta) override;

public:
	virtual PackedStringArray get_configuration_warnings() override;
};

#endif // BT_DECORATOR_H
