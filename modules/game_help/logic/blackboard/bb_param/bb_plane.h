/**
 * bb_plane.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef BB_PLANE_H
#define BB_PLANE_H

#include "bb_param.h"

class BBPlane : public BBParam {
	GDCLASS(BBPlane, BBParam);

protected:
	static void _bind_methods() {}

	virtual Variant::Type get_type() const override { return Variant::PLANE; }
};

#endif // BB_PLANE_H
