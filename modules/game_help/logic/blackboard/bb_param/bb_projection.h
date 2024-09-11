/**
 * bb_projection.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef BB_PROJECTION_H
#define BB_PROJECTION_H

#include "bb_param.h"

class BBProjection : public BBParam {
	GDCLASS(BBProjection, BBParam);

protected:
	static void _bind_methods() {}

	virtual Variant::Type get_type() const override { return Variant::PROJECTION; }
};

#endif // BB_PROJECTION_H
