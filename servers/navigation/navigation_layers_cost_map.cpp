/**************************************************************************/
/*  navigation_layers_cost_map.cpp                                        */
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

#include "navigation_layers_cost_map.h"

#include "core/config/project_settings.h"

bool NavigationLayersCostMap::_set(const StringName &p_path, const Variant &p_value) {
	String path = p_path;
	if (path.begins_with("navigation_layers/")) {
		uint32_t which = path.get_slicec('/', 1).to_int();
		if (path.ends_with("cost")) {
			set_navigation_layer_cost(which, p_value);
			return true;
		}
	}
	return false;
}

bool NavigationLayersCostMap::_get(const StringName &p_path, Variant &r_ret) const {
	String path = p_path;
	if (path.begins_with("navigation_layers/")) {
		uint32_t which = path.get_slicec('/', 1).to_int();
		if (path.ends_with("name")) {
			r_ret = GLOBAL_GET(vformat("%s/layer_%d", PNAME("layer_names/3d_navigation"), which));
			return true;
		}
		if (path.ends_with("cost")) {
			r_ret = get_navigation_layer_cost(which);
			return true;
		}
	}
	return false;
}

void NavigationLayersCostMap::set_navigation_layer_cost(uint32_t p_layer_number, float p_cost) {
	ERR_FAIL_COND_MSG(p_layer_number < 1, "Navigation layer number must be between 1 and 32 inclusive.");
	ERR_FAIL_COND_MSG(p_layer_number > 32, "Navigation layer number must be between 1 and 32 inclusive.");
	if (navigation_layers_cost_map[p_layer_number - 1] == p_cost) {
		return;
	}

	navigation_layers_cost_map[p_layer_number - 1] = p_cost;
	emit_changed();
}

float NavigationLayersCostMap::get_navigation_layer_cost(uint32_t p_layer_number) const {
	ERR_FAIL_COND_V_MSG(p_layer_number < 1, 1.0, "Navigation layer number must be between 1 and 32 inclusive.");
	ERR_FAIL_COND_V_MSG(p_layer_number > 32, 1.0, "Navigation layer number must be between 1 and 32 inclusive.");
	return navigation_layers_cost_map[p_layer_number - 1];
}

void NavigationLayersCostMap::_validate_property(PropertyInfo &property) const {
	if (property.name.begins_with("navigation_layers") && property.name.ends_with("name")) {
		property.usage ^= PROPERTY_USAGE_STORAGE;
	}
}

void NavigationLayersCostMap::_get_property_list(List<PropertyInfo> *p_list) const {
	for (uint32_t i = 0; i < 32; i++) {
		const String prep = vformat("navigation_layers/%d/", i + 1);
		p_list->push_back(PropertyInfo(Variant::STRING, prep + "name", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_READ_ONLY));
		p_list->push_back(PropertyInfo(Variant::FLOAT, prep + "cost", PROPERTY_HINT_RANGE, "1.0,1000.0,0.01,or_greater"));
	}
}

void NavigationLayersCostMap::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_navigation_layer_cost", "layer_number", "cost"), &NavigationLayersCostMap::set_navigation_layer_cost);
	ClassDB::bind_method(D_METHOD("get_navigation_layer_cost", "layer_number"), &NavigationLayersCostMap::get_navigation_layer_cost);
}

NavigationLayersCostMap::NavigationLayersCostMap() {
	navigation_layers_cost_map.resize(32);
	for (uint32_t i = 0; i < 32; i++) {
		navigation_layers_cost_map[i] = 1.0;
	}
}

NavigationLayersCostMap::~NavigationLayersCostMap() {
}

//////////////////////////////////////
