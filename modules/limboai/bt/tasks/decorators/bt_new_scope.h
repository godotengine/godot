/**
 * bt_new_scope.h
 * =============================================================================
 * Copyright 2021-2023 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef BT_NEW_SCOPE_H
#define BT_NEW_SCOPE_H

#include "../bt_decorator.h"

class BTNewScope : public BTDecorator {
	GDCLASS(BTNewScope, BTDecorator);
	TASK_CATEGORY(Decorators);

private:
	Dictionary blackboard_data;

protected:
	static void _bind_methods();

	void _set_blackboard_data(const Dictionary &p_value) { blackboard_data = p_value; }
	Dictionary _get_blackboard_data() const { return blackboard_data; }

	virtual Status _tick(double p_delta) override;

public:
	virtual void initialize(Node *p_agent, const Ref<Blackboard> &p_blackboard) override;
};

#endif // BT_NEW_SCOPE_H
