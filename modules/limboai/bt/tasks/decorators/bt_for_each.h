/**
 * bt_for_each.h
 * =============================================================================
 * Copyright 2021-2023 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef BT_FOR_EACH_H
#define BT_FOR_EACH_H

#include "../bt_decorator.h"

class BTForEach : public BTDecorator {
	GDCLASS(BTForEach, BTDecorator);
	TASK_CATEGORY(Decorators);

private:
	String array_var;
	String save_var;

	int current_idx;

protected:
	static void _bind_methods();

	virtual String _generate_name() override;
	virtual void _enter() override;
	virtual Status _tick(double p_delta) override;

public:
	void set_array_var(String p_value);
	String get_array_var() const { return array_var; }

	void set_save_var(String p_value);
	String get_save_var() const { return save_var; }
};

#endif // BT_FOR_EACH_H
