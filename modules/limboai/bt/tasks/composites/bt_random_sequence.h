/**
 * bt_random_sequence.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef BT_RANDOM_SEQUENCE_H
#define BT_RANDOM_SEQUENCE_H

#include "../bt_composite.h"

class BTRandomSequence : public BTComposite {
	GDCLASS(BTRandomSequence, BTComposite);
	TASK_CATEGORY(Composites);

private:
	int last_running_idx = 0;
	Array indicies;

protected:
	static void _bind_methods() {}

	virtual void _enter() override;
	virtual Status _tick(double p_delta) override;
};

#endif // BT_RANDOM_SEQUENCE_H
