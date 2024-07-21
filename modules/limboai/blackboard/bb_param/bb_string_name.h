/**
 * bb_string_name.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef BB_STRING_NAME_H
#define BB_STRING_NAME_H

#include "bb_param.h"

class BBStringName : public BBParam {
	GDCLASS(BBStringName, BBParam);

protected:
	static void _bind_methods() {}

	virtual Variant::Type get_type() const override { return Variant::STRING_NAME; }
};

#endif // BB_STRING_H
