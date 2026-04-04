/**************************************************************************/
/*  camera_feed.hpp                                                       */
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

#include <godot_cpp/classes/camera_server.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/transform2d.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Dictionary;
class Image;

class CameraFeed : public RefCounted {
	GDEXTENSION_CLASS(CameraFeed, RefCounted)

public:
	enum FeedDataType {
		FEED_NOIMAGE = 0,
		FEED_RGB = 1,
		FEED_YCBCR = 2,
		FEED_YCBCR_SEP = 3,
		FEED_EXTERNAL = 4,
	};

	enum FeedPosition {
		FEED_UNSPECIFIED = 0,
		FEED_FRONT = 1,
		FEED_BACK = 2,
	};

	int32_t get_id() const;
	bool is_active() const;
	void set_active(bool p_active);
	String get_name() const;
	void set_name(const String &p_name);
	CameraFeed::FeedPosition get_position() const;
	void set_position(CameraFeed::FeedPosition p_position);
	Transform2D get_transform() const;
	void set_transform(const Transform2D &p_transform);
	void set_rgb_image(const Ref<Image> &p_rgb_image);
	void set_ycbcr_image(const Ref<Image> &p_ycbcr_image);
	void set_ycbcr_images(const Ref<Image> &p_y_image, const Ref<Image> &p_cbcr_image);
	void set_external(int32_t p_width, int32_t p_height);
	uint64_t get_texture_tex_id(CameraServer::FeedImage p_feed_image_type);
	CameraFeed::FeedDataType get_datatype() const;
	Array get_formats() const;
	bool set_format(int32_t p_index, const Dictionary &p_parameters);
	virtual bool _activate_feed();
	virtual void _deactivate_feed();

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		RefCounted::register_virtuals<T, B>();
		if constexpr (!std::is_same_v<decltype(&B::_activate_feed), decltype(&T::_activate_feed)>) {
			BIND_VIRTUAL_METHOD(T, _activate_feed, 2240911060);
		}
		if constexpr (!std::is_same_v<decltype(&B::_deactivate_feed), decltype(&T::_deactivate_feed)>) {
			BIND_VIRTUAL_METHOD(T, _deactivate_feed, 3218959716);
		}
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(CameraFeed::FeedDataType);
VARIANT_ENUM_CAST(CameraFeed::FeedPosition);

