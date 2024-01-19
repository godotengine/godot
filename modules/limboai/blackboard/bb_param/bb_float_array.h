/**
 * bb_float_array.h
 * =============================================================================
 * Copyright 2021-2023 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef BB_FLOAT_ARRAY_H
#define BB_FLOAT_ARRAY_H

#include "bb_param.h"

class BBFloatArray : public BBParam {
	GDCLASS(BBFloatArray, BBParam);

protected:
	virtual Variant::Type get_type() const override { return Variant::PACKED_FLOAT64_ARRAY; }
};

#endif // BB_FLOAT_ARRAY_H