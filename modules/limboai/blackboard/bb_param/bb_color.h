/**
 * bb_color.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef BB_COLOR_H
#define BB_COLOR_H

#include "bb_param.h"

class BBColor : public BBParam {
	GDCLASS(BBColor, BBParam);

protected:
	static void _bind_methods() {}

	virtual Variant::Type get_type() const override { return Variant::COLOR; }
};

#endif // BB_COLOR_H
