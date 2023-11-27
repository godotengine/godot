/**************************************************************************/
/*  gdextension_compat_hashes.h                                           */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#ifndef GDEXTENSION_COMPAT_HASHES_H
#define GDEXTENSION_COMPAT_HASHES_H

#ifndef DISABLE_DEPRECATED

#include "core/string/string_name.h"
#include "core/templates/hash_map.h"
#include "core/templates/local_vector.h"

class GDExtensionCompatHashes {
	struct Mapping {
		StringName method;
		uint32_t legacy_hash;
		uint32_t current_hash;
	};

	static HashMap<StringName, LocalVector<Mapping>> mappings;

public:
	static void initialize();
	static void finalize();
	static bool lookup_current_hash(const StringName &p_class, const StringName &p_method, uint32_t p_legacy_hash, uint32_t *r_current_hash);
	static bool get_legacy_hashes(const StringName &p_class, const StringName &p_method, Array &r_hashes, bool p_check_valid = true);
};

#endif // DISABLE_DEPRECATED

#endif // GDEXTENSION_COMPAT_HASHES_H
