/**************************************************************************/
/*  collision_presets_2d.cpp                                              */
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

#include "collision_presets_2d.h"

#include "core/config/project_settings.h"
#include "core/object/class_db.h"

void CollisionPresets2D::_bind_methods() {
	ClassDB::bind_static_method("CollisionPresets2D", D_METHOD("get_preset", "name"), &CollisionPresets2D::get_preset);
	ClassDB::bind_static_method("CollisionPresets2D", D_METHOD("get_preset_name", "preset"), &CollisionPresets2D::get_preset_name);
	ClassDB::bind_static_method("CollisionPresets2D", D_METHOD("get_preset_layer", "preset"), &CollisionPresets2D::get_preset_layer);
	ClassDB::bind_static_method("CollisionPresets2D", D_METHOD("get_preset_mask", "preset"), &CollisionPresets2D::get_preset_mask);
	ClassDB::bind_static_method("CollisionPresets2D", D_METHOD("has_preset_named", "name"), &CollisionPresets2D::has_preset_named);
	ClassDB::bind_static_method("CollisionPresets2D", D_METHOD("has_preset", "preset"), &CollisionPresets2D::has_preset);
	ClassDB::bind_static_method("CollisionPresets2D", D_METHOD("get_preset_is_custom", "preset"), &CollisionPresets2D::get_preset_is_custom);
}

// Returns false if the preset is custom or does not exist. (presets may be also considered custom if they are
// the default preset, and the default is None)
bool CollisionPresets2D::get_preset_dict(int p_preset, Dictionary *r_dict) {
	if (p_preset == -1) {
		return false;
	}

	int check_preset = p_preset;
	if (check_preset == 0) {
		check_preset = GLOBAL_GET("physics/2d/default_preset");
		if (check_preset == 0) {
			// If the default preset is zero, essentially treat it as <custom>.
			return false;
		}
	}

	Array presets = ProjectSettings::get_singleton()->get_setting("physics/2d/presets");
	for (Dictionary preset : presets) {
		if (int(preset["id"]) != check_preset) {
			continue;
		}

		*r_dict = preset;
		return true;
	}

	ERR_FAIL_V_MSG(false, vformat("2D collision preset %d does not exist.", p_preset));
}

int CollisionPresets2D::get_preset(const String &p_name) {
	Array presets = ProjectSettings::get_singleton()->get_setting("physics/2d/presets");

	for (Dictionary preset : presets) {
		if (String(preset["name"]) == p_name) {
			return preset["id"];
		}
	}

	ERR_FAIL_V_MSG(0, vformat("2D collision preset %s does not exist.", p_name));
}

String CollisionPresets2D::get_preset_name(int p_preset) {
	Dictionary preset;
	if (!get_preset_dict(p_preset, &preset)) {
		return "Invalid Preset";
	}

	return preset["name"];
}

uint32_t CollisionPresets2D::get_preset_layer(int p_preset) {
	Dictionary preset;
	if (!get_preset_dict(p_preset, &preset)) {
		return 0;
	}

	return preset["layer"];
}

uint32_t CollisionPresets2D::get_preset_mask(int p_preset) {
	Dictionary preset;
	if (!get_preset_dict(p_preset, &preset)) {
		return 0;
	}

	return preset["mask"];
}

bool CollisionPresets2D::has_preset_named(const String &p_name) {
	Array presets = ProjectSettings::get_singleton()->get_setting("physics/2d/presets");

	for (Dictionary preset : presets) {
		if (String(preset["name"]) == p_name) {
			return true;
		}
	}

	return false;
}

bool CollisionPresets2D::has_preset(int p_preset) {
	if (p_preset == 0) {
		return true;
	}

	Array presets = ProjectSettings::get_singleton()->get_setting("physics/2d/presets");

	for (Dictionary preset : presets) {
		if (int(preset["id"]) == p_preset) {
			return true;
		}
	}

	return false;
}

bool CollisionPresets2D::get_preset_is_custom(int p_preset) {
	if (p_preset == -1) {
		return true;
	} else if (p_preset == 0) {
		p_preset = ProjectSettings::get_singleton()->get_setting("physics/2d/default_preset");
		if (p_preset == 0) {
			return true;
		}
	}

	return false;
}
