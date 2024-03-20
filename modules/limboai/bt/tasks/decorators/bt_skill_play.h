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

#ifndef BT_SKILL_PLAY_H
#define BT_SKILL_PLAY_H

#include "../bt_decorator.h"

#include "bt_new_scope.h"


class BTPlaySkill : public BTNewScope {
	GDCLASS(BTPlaySkill, BTNewScope);
	TASK_CATEGORY(Decorators);

private:
	String skillTree;

protected:
	static void _bind_methods();

	virtual String _generate_name() override
    {
        return "技能播放";
    }
	virtual Status _tick(double p_delta) override
    {
        if(get_blackboard()->get_var("skill_play",false))
        {
            return RUNNING;
        }
        return SUCCESS;
    }

public:
	void set_skill(const String &p_value)
    {
        skillTree = p_value;
    }
	String get_skill() const { return skillTree; }

	virtual void initialize(Node *p_agent, const Ref<Blackboard> &p_blackboard) override;
	virtual PackedStringArray get_configuration_warnings() override;
};


#endif // BT_TIME_LIMIT_H
