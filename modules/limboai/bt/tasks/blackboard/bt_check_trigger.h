/**
 * bt_check_trigger.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef BT_CHECK_TRIGGER_H
#define BT_CHECK_TRIGGER_H

#include "../bt_condition.h"

class BTCheckTrigger : public BTCondition {
	GDCLASS(BTCheckTrigger, BTCondition);
	TASK_CATEGORY(Blackboard);

private:
	StringName variable;

protected:
	static void _bind_methods();

	virtual String _generate_name() override;
	virtual Status _tick(double p_delta) override;

public:
	void set_variable(const StringName &p_variable);
	StringName get_variable() const { return variable; }

	virtual PackedStringArray get_configuration_warnings() override;
};

#endif // BT_CHECK_TRIGGER
