/**
 * bb_transform2d.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef BB_TRANSFORM2D_H
#define BB_TRANSFORM2D_H

#include "bb_param.h"

class BBTransform2D : public BBParam {
	GDCLASS(BBTransform2D, BBParam);

protected:
	static void _bind_methods() {}

	virtual Variant::Type get_type() const override { return Variant::TRANSFORM2D; }
};

#endif // BB_TRANSFORM2D_H
