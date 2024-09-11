/**
 * bb_int32_array.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef BB_INT32_ARRAY_H
#define BB_INT32_ARRAY_H

#include "bb_param.h"

class BBInt32Array : public BBParam {
	GDCLASS(BBInt32Array, BBParam);

protected:
	static void _bind_methods() {}

	virtual Variant::Type get_type() const override { return Variant::PACKED_INT32_ARRAY; }
};

#endif // BB_INT32_ARRAY_H
