/**
 * bb_vector2i.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef BB_VECTOR2I_H
#define BB_VECTOR2I_H

#include "bb_param.h"

class BBVector2i : public BBParam {
	GDCLASS(BBVector2i, BBParam);

protected:
	static void _bind_methods() {}

	virtual Variant::Type get_type() const override { return Variant::VECTOR2I; }
};

#endif // BB_VECTOR2I_H
