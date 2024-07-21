/**
 * bt_action.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef BT_ACTION_H
#define BT_ACTION_H

#include "bt_task.h"

class BTAction : public BTTask {
	GDCLASS(BTAction, BTTask);

protected:
	static void _bind_methods() {}

public:
	virtual PackedStringArray get_configuration_warnings() override;
};

#endif // BT_ACTION_H
