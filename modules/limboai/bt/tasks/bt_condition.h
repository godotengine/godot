/**
 * bt_condition.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef BT_CONDITION_H
#define BT_CONDITION_H

#include "bt_task.h"

class BTCondition : public BTTask {
	GDCLASS(BTCondition, BTTask);

protected:
	static void _bind_methods() {}

public:
	virtual PackedStringArray get_configuration_warnings() override;
};

#endif // BT_CONDITION_H
