/**************************************************************************/
/*  gltf_item.h                                                           */
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

#pragma once

#include "../gltf_defines.h"

#include "core/io/resource.h"
#include "core/variant/variant.h"

class GLTFItem : public Resource {
	GDCLASS(GLTFItem, Resource);

	Dictionary _additional_data;

protected:
	static void _bind_methods();

#ifndef DISABLE_DEPRECATED
	// Non-const getter for compatibility. Previously located in GLTFState.
	Variant _get_additional_data_bind_compat_113172(const StringName &p_extension_name);
	static void _bind_compatibility_methods();
#endif // DISABLE_DEPRECATED

public:
	Variant get_additional_data(const StringName &p_extension_name) const;
	bool has_additional_data(const StringName &p_extension_name) const;
	void set_additional_data(const StringName &p_extension_name, Variant p_additional_data);

	virtual Error read_from_dictionary(const Dictionary &p_dictionary);
	virtual Dictionary write_to_dictionary() const;
	GDVIRTUAL1R(Error, _read_from_dictionary, const Dictionary &);
	GDVIRTUAL0RC(Dictionary, _write_to_dictionary);
};
