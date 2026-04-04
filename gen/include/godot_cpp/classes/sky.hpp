/**************************************************************************/
/*  sky.hpp                                                               */
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

// THIS FILE IS GENERATED. EDITS WILL BE LOST.

#pragma once

#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/resource.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Material;

class Sky : public Resource {
	GDEXTENSION_CLASS(Sky, Resource)

public:
	enum RadianceSize {
		RADIANCE_SIZE_32 = 0,
		RADIANCE_SIZE_64 = 1,
		RADIANCE_SIZE_128 = 2,
		RADIANCE_SIZE_256 = 3,
		RADIANCE_SIZE_512 = 4,
		RADIANCE_SIZE_1024 = 5,
		RADIANCE_SIZE_2048 = 6,
		RADIANCE_SIZE_MAX = 7,
	};

	enum ProcessMode {
		PROCESS_MODE_AUTOMATIC = 0,
		PROCESS_MODE_QUALITY = 1,
		PROCESS_MODE_INCREMENTAL = 2,
		PROCESS_MODE_REALTIME = 3,
	};

	void set_radiance_size(Sky::RadianceSize p_size);
	Sky::RadianceSize get_radiance_size() const;
	void set_process_mode(Sky::ProcessMode p_mode);
	Sky::ProcessMode get_process_mode() const;
	void set_material(const Ref<Material> &p_material);
	Ref<Material> get_material() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Resource::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(Sky::RadianceSize);
VARIANT_ENUM_CAST(Sky::ProcessMode);

