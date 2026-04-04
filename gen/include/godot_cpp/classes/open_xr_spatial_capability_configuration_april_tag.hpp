/**************************************************************************/
/*  open_xr_spatial_capability_configuration_april_tag.hpp                */
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

#include <godot_cpp/classes/open_xr_spatial_capability_configuration_base_header.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/variant/packed_int64_array.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class OpenXRSpatialCapabilityConfigurationAprilTag : public OpenXRSpatialCapabilityConfigurationBaseHeader {
	GDEXTENSION_CLASS(OpenXRSpatialCapabilityConfigurationAprilTag, OpenXRSpatialCapabilityConfigurationBaseHeader)

public:
	enum AprilTagDict {
		APRIL_TAG_DICT_16H5 = 1,
		APRIL_TAG_DICT_25H9 = 2,
		APRIL_TAG_DICT_36H10 = 3,
		APRIL_TAG_DICT_36H11 = 4,
	};

	PackedInt64Array get_enabled_components() const;
	void set_april_dict(OpenXRSpatialCapabilityConfigurationAprilTag::AprilTagDict p_april_dict);
	OpenXRSpatialCapabilityConfigurationAprilTag::AprilTagDict get_april_dict() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		OpenXRSpatialCapabilityConfigurationBaseHeader::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(OpenXRSpatialCapabilityConfigurationAprilTag::AprilTagDict);

