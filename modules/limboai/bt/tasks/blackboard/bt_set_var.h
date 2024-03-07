/**
 * bt_set_var.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef BT_SET_VAR_H
#define BT_SET_VAR_H

#include "../bt_action.h"

#include "../../../blackboard/bb_param/bb_variant.h"
#include "../../../util/limbo_utility.h"

class BTSetVar : public BTAction {
	GDCLASS(BTSetVar, BTAction);
	TASK_CATEGORY(Blackboard);

private:
	StringName variable;
	Ref<BBVariant> value;
	LimboUtility::Operation operation = LimboUtility::OPERATION_NONE;

protected:
	static void _bind_methods();

	virtual String _generate_name() override;
	virtual Status _tick(double p_delta) override;

public:
	virtual PackedStringArray get_configuration_warnings() override;

	void set_variable(const StringName &p_variable);
	StringName get_variable() const { return variable; }

	void set_value(const Ref<BBVariant> &p_value);
	Ref<BBVariant> get_value() const { return value; }

	void set_operation(LimboUtility::Operation p_operation);
	LimboUtility::Operation get_operation() const { return operation; }
};

#endif // BT_SET_VAR
