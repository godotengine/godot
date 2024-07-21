/**
 * bb_variant.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef BB_VARIANT_H
#define BB_VARIANT_H

#include "bb_param.h"

class BBVariant : public BBParam {
	GDCLASS(BBVariant, BBParam);

private:
	Variant::Type type = Variant::NIL;

protected:
	static void _bind_methods();

public:
	virtual Variant::Type get_type() const override;
	void set_type(Variant::Type p_type);

	virtual Variant::Type get_variable_expected_type() const override { return Variant::NIL; }

	BBVariant(const Variant &p_value);
	BBVariant();
};

#endif // BB_VARIANT
