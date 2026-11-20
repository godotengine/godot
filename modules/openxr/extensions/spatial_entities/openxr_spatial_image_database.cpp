/**************************************************************************/
/*  openxr_spatial_image_database.cpp                                     */
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

#include "openxr_spatial_image_database.h"

#include "openxr_spatial_image_tracking.h"

#include "core/object/class_db.h"

OpenXRSpatialImageDatabase::OpenXRSpatialImageDatabase() {
	// Always start with one empty entry
	ImageEntry empty;
	images.push_back(empty);
}

void OpenXRSpatialImageDatabase::_bind_methods() {
	ClassDB::bind_method(D_METHOD("has_images"), &OpenXRSpatialImageDatabase::has_images);
	ClassDB::bind_method(D_METHOD("get_image_database"), &OpenXRSpatialImageDatabase::get_image_database);
}

bool OpenXRSpatialImageDatabase::has_images() {
	// We always have one empty entry, so check if larger then 1.
	return images.size() > 1;
}

RID OpenXRSpatialImageDatabase::get_image_database() {
	OpenXRSpatialImageTrackingCapability *image_tracking_capability = OpenXRSpatialImageTrackingCapability::get_singleton();
	ERR_FAIL_NULL_V(image_tracking_capability, RID());
	ERR_FAIL_COND_V(!image_tracking_capability->is_supported(), RID());

	RID image_database = image_tracking_capability->image_database_create();

	for (const ImageEntry &entry : images) {
		if (entry.image.is_valid()) {
			String image_name = entry.image->get_name();
			if (image_name.is_empty()) {
				image_name = entry.image->get_path().get_file();
			}
			image_tracking_capability->image_database_add(image_database, image_name, entry.image, entry.physical_width, entry.is_static);
		}
	}

	return image_database;
}

bool OpenXRSpatialImageDatabase::_set(const StringName &p_name, const Variant &p_value) {
	const String prop_name = p_name;
	if (prop_name.begins_with("images/")) {
		const String index = prop_name.get_slicec('/', 1);

		const int id = index.to_int();
		ERR_FAIL_INDEX_V(id, int(images.size()), false);

		const String node_info = prop_name.get_slicec('/', 2);
		if (node_info == "image") {
			Ref<Image> image = p_value;
			int last_image = int(images.size()) - 1;

			// If we empty anything but the last entry, toast it!
			if (image.is_null() && id < last_image) {
				images.remove_at(id);
				notify_property_list_changed();
				return true;
			}

			images[id].image = image;

			// Set the value for the last entry, add a new entry!
			if (image.is_valid() && id == last_image) {
				ImageEntry empty;
				images.push_back(empty);
				notify_property_list_changed();
			}
			return true;
		} else if (node_info == "physical_width") {
			images[id].physical_width = p_value;
			return true;
		} else if (node_info == "is_static") {
			images[id].is_static = p_value;
			return true;
		}
	}

	return false;
}

bool OpenXRSpatialImageDatabase::_get(const StringName &p_name, Variant &r_ret) const {
	const String prop_name = p_name;
	if (prop_name.begins_with("images/")) {
		const String index = prop_name.get_slicec('/', 1);

		const int id = index.to_int();
		ERR_FAIL_INDEX_V(id, int(images.size()), false);

		const String node_info = prop_name.get_slicec('/', 2);
		if (node_info == "image") {
			r_ret = images[id].image;
			return true;
		} else if (node_info == "physical_width") {
			r_ret = images[id].physical_width;
			return true;
		} else if (node_info == "is_static") {
			r_ret = images[id].is_static;
			return true;
		}
	}

	return false;
}

void OpenXRSpatialImageDatabase::_get_property_list(List<PropertyInfo> *p_list) const {
	for (int i = 0; i < int(images.size()); i++) {
		String prop_name = "images/";
		prop_name += itos(i);

		p_list->push_back(PropertyInfo(Variant::OBJECT, prop_name + "/image", PROPERTY_HINT_RESOURCE_TYPE, "Image"));
		p_list->push_back(PropertyInfo(Variant::FLOAT, prop_name + "/physical_width"));
		p_list->push_back(PropertyInfo(Variant::BOOL, prop_name + "/is_static"));
	}
}
