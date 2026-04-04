/**************************************************************************/
/*  aspect_ratio_container.hpp                                            */
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

#include <godot_cpp/classes/container.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class AspectRatioContainer : public Container {
	GDEXTENSION_CLASS(AspectRatioContainer, Container)

public:
	enum StretchMode {
		STRETCH_WIDTH_CONTROLS_HEIGHT = 0,
		STRETCH_HEIGHT_CONTROLS_WIDTH = 1,
		STRETCH_FIT = 2,
		STRETCH_COVER = 3,
	};

	enum AlignmentMode {
		ALIGNMENT_BEGIN = 0,
		ALIGNMENT_CENTER = 1,
		ALIGNMENT_END = 2,
	};

	void set_ratio(float p_ratio);
	float get_ratio() const;
	void set_stretch_mode(AspectRatioContainer::StretchMode p_stretch_mode);
	AspectRatioContainer::StretchMode get_stretch_mode() const;
	void set_alignment_horizontal(AspectRatioContainer::AlignmentMode p_alignment_horizontal);
	AspectRatioContainer::AlignmentMode get_alignment_horizontal() const;
	void set_alignment_vertical(AspectRatioContainer::AlignmentMode p_alignment_vertical);
	AspectRatioContainer::AlignmentMode get_alignment_vertical() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Container::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(AspectRatioContainer::StretchMode);
VARIANT_ENUM_CAST(AspectRatioContainer::AlignmentMode);

