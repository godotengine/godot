/**
 * bt_check_agent_property.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef BT_CHECK_AGENT_PROPERTY
#define BT_CHECK_AGENT_PROPERTY

#include "../bt_condition.h"

#include "../../../blackboard/bb_param/bb_variant.h"
#include "../../../util/limbo_utility.h"

class BTCheckAgentProperty : public BTCondition {
	GDCLASS(BTCheckAgentProperty, BTCondition);
	TASK_CATEGORY(Scene);

private:
	StringName property;
	LimboUtility::CheckType check_type = LimboUtility::CheckType::CHECK_EQUAL;
	Ref<BBVariant> value;

protected:
	static void _bind_methods();

	virtual String _generate_name() override;
	virtual Status _tick(double p_delta) override;

public:
	void set_property(StringName p_prop);
	StringName get_property() const { return property; }

	void set_check_type(LimboUtility::CheckType p_check_type);
	LimboUtility::CheckType get_check_type() const { return check_type; }

	void set_value(Ref<BBVariant> p_value);
	Ref<BBVariant> get_value() const { return value; }

	virtual PackedStringArray get_configuration_warnings() override;
};

#endif // BT_CHECK_AGENT_PROPERTY_H
