/**
 * bb_dictionary.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef BB_DICTIONARY_H
#define BB_DICTIONARY_H

#include "bb_param.h"

class BBDictionary : public BBParam {
	GDCLASS(BBDictionary, BBParam);

protected:
	static void _bind_methods() {}

	virtual Variant::Type get_type() const override { return Variant::DICTIONARY; }
};

#endif // BB_DICTIONARY_H
