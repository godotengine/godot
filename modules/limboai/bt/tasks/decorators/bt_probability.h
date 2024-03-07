/**
 * bt_probability.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef BT_PROBABILITY_H
#define BT_PROBABILITY_H

#include "../bt_decorator.h"

class BTProbability : public BTDecorator {
	GDCLASS(BTProbability, BTDecorator);
	TASK_CATEGORY(Decorators);

private:
	float run_chance = 0.5;

protected:
	static void _bind_methods();

	virtual String _generate_name() override;
	virtual Status _tick(double p_delta) override;

public:
	void set_run_chance(float p_value);
	float get_run_chance() const { return run_chance; }
};

#endif // BT_PROBABILITY_H
