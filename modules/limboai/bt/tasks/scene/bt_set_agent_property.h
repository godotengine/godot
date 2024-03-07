/**
 * bt_set_agent_property.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef BT_SET_AGENT_PROPERTY_H
#define BT_SET_AGENT_PROPERTY_H

#include "../bt_action.h"

#include "../../../blackboard/bb_param/bb_variant.h"
#include "../../../util/limbo_utility.h"

class BTSetAgentProperty : public BTAction {
	GDCLASS(BTSetAgentProperty, BTAction);
	TASK_CATEGORY(Scene);

private:
	StringName property;
	Ref<BBVariant> value;
	LimboUtility::Operation operation = LimboUtility::OPERATION_NONE;

protected:
	static void _bind_methods();

	virtual String _generate_name() override;
	virtual Status _tick(double p_delta) override;

public:
	virtual PackedStringArray get_configuration_warnings() override;

	void set_property(StringName p_prop);
	StringName get_property() const { return property; }

	void set_value(Ref<BBVariant> p_value);
	Ref<BBVariant> get_value() const { return value; }

	void set_operation(LimboUtility::Operation p_operation);
	LimboUtility::Operation get_operation() const { return operation; }
};

#endif // BT_SET_AGENT_PROPERTY_H
