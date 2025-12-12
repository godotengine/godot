/**************************************************************************/
/*  editor_build_profile.cpp                                              */
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

#include "editor_build_profile.h"

#include "core/config/project_settings.h"
#include "core/io/json.h"
#include "editor/editor_node.h"
#include "editor/editor_string_names.h"
#include "editor/file_system/editor_file_system.h"
#include "editor/file_system/editor_paths.h"
#include "editor/gui/editor_file_dialog.h"
#include "editor/settings/editor_settings.h"
#include "editor/themes/editor_scale.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/margin_container.h"
#include "scene/gui/separator.h"

#include "modules/modules_enabled.gen.h" // For mono.

const char *EditorBuildProfile::build_option_identifiers[BUILD_OPTION_MAX] = {
	// This maps to SCons build options.
	"disable_3d",
	"disable_navigation_2d",
	"disable_navigation_3d",
	"disable_xr",
	"module_openxr_enabled",
	"wayland",
	"x11",
	"rendering_device", // FIXME: There's no scons option to disable rendering device.
	"forward_plus_renderer",
	"forward_mobile_renderer",
	"vulkan",
	"d3d12",
	"metal",
	"opengl3",
	"disable_physics_2d",
	"module_godot_physics_2d_enabled",
	"disable_physics_3d",
	"module_godot_physics_3d_enabled",
	"module_jolt_physics_enabled",
	"module_text_server_fb_enabled",
	"module_text_server_adv_enabled",
	"module_freetype_enabled",
	"brotli",
	"graphite",
	"module_msdfgen_enabled",
};

const bool EditorBuildProfile::build_option_disabled_by_default[BUILD_OPTION_MAX] = {
	// This maps to SCons build options.
	false, // 3D
	false, // NAVIGATION_2D
	false, // NAVIGATION_3D
	false, // XR
	false, // OPENXR
	false, // WAYLAND
	false, // X11
	false, // RENDERING_DEVICE
	false, // FORWARD_RENDERER
	false, // MOBILE_RENDERER
	false, // VULKAN
	false, // D3D12
	false, // METAL
	false, // OPENGL
	false, // PHYSICS_2D
	false, // PHYSICS_GODOT_2D
	false, // PHYSICS_3D
	false, // PHYSICS_GODOT_3D
	false, // PHYSICS_JOLT
	true, // TEXT_SERVER_FALLBACK
	false, // TEXT_SERVER_ADVANCED
	false, // DYNAMIC_FONTS
	false, // WOFF2_FONTS
	false, // GRAPHITE_FONTS
	false, // MSDFGEN
};

const bool EditorBuildProfile::build_option_disable_values[BUILD_OPTION_MAX] = {
	// This maps to SCons build options.
	true, // 3D
	true, // NAVIGATION_2D
	true, // NAVIGATION_3D
	true, // XR
	false, // OPENXR
	false, // WAYLAND
	false, // X11
	false, // RENDERING_DEVICE
	false, // FORWARD_RENDERER
	false, // MOBILE_RENDERER
	false, // VULKAN
	false, // D3D12
	false, // METAL
	false, // OPENGL
	true, // PHYSICS_2D
	false, // PHYSICS_GODOT_2D
	true, // PHYSICS_3D
	false, // PHYSICS_GODOT_3D
	false, // PHYSICS_JOLT
	false, // TEXT_SERVER_FALLBACK
	false, // TEXT_SERVER_ADVANCED
	false, // DYNAMIC_FONTS
	false, // WOFF2_FONTS
	false, // GRAPHITE_FONTS
	false, // MSDFGEN
};

// Options that require some resource explicitly asking for them when detecting from the project.
const bool EditorBuildProfile::build_option_explicit_use[BUILD_OPTION_MAX] = {
	false, // 3D
	false, // NAVIGATION_2D
	false, // NAVIGATION_3D
	false, // XR
	false, // OPENXR
	false, // WAYLAND
	false, // X11
	false, // RENDERING_DEVICE
	false, // FORWARD_RENDERER
	false, // MOBILE_RENDERER
	false, // VULKAN
	false, // D3D12
	false, // METAL
	false, // OPENGL
	false, // PHYSICS_2D
	false, // PHYSICS_GODOT_2D
	false, // PHYSICS_3D
	false, // PHYSICS_GODOT_3D
	false, // PHYSICS_JOLT
	false, // TEXT_SERVER_FALLBACK
	false, // TEXT_SERVER_ADVANCED
	false, // DYNAMIC_FONTS
	false, // WOFF2_FONTS
	false, // GRAPHITE_FONTS
	true, // MSDFGEN
};

const EditorBuildProfile::BuildOptionCategory EditorBuildProfile::build_option_category[BUILD_OPTION_MAX] = {
	BUILD_OPTION_CATEGORY_GENERAL, // 3D
	BUILD_OPTION_CATEGORY_GENERAL, // NAVIGATION_2D
	BUILD_OPTION_CATEGORY_GENERAL, // NAVIGATION_3D
	BUILD_OPTION_CATEGORY_GENERAL, // XR
	BUILD_OPTION_CATEGORY_GENERAL, // OPENXR
	BUILD_OPTION_CATEGORY_GENERAL, // WAYLAND
	BUILD_OPTION_CATEGORY_GENERAL, // X11
	BUILD_OPTION_CATEGORY_GRAPHICS, // RENDERING_DEVICE
	BUILD_OPTION_CATEGORY_GRAPHICS, // FORWARD_RENDERER
	BUILD_OPTION_CATEGORY_GRAPHICS, // MOBILE_RENDERER
	BUILD_OPTION_CATEGORY_GRAPHICS, // VULKAN
	BUILD_OPTION_CATEGORY_GRAPHICS, // D3D12
	BUILD_OPTION_CATEGORY_GRAPHICS, // METAL
	BUILD_OPTION_CATEGORY_GRAPHICS, // OPENGL
	BUILD_OPTION_CATEGORY_PHYSICS, // PHYSICS_2D
	BUILD_OPTION_CATEGORY_PHYSICS, // PHYSICS_GODOT_2D
	BUILD_OPTION_CATEGORY_PHYSICS, // PHYSICS_3D
	BUILD_OPTION_CATEGORY_PHYSICS, // PHYSICS_GODOT_3D
	BUILD_OPTION_CATEGORY_PHYSICS, // PHYSICS_JOLT
	BUILD_OPTION_CATEGORY_TEXT_SERVER, // TEXT_SERVER_FALLBACK
	BUILD_OPTION_CATEGORY_TEXT_SERVER, // TEXT_SERVER_ADVANCED
	BUILD_OPTION_CATEGORY_TEXT_SERVER, // DYNAMIC_FONTS
	BUILD_OPTION_CATEGORY_TEXT_SERVER, // WOFF2_FONTS
	BUILD_OPTION_CATEGORY_TEXT_SERVER, // GRAPHITE_FONTS
	BUILD_OPTION_CATEGORY_TEXT_SERVER, // MSDFGEN
};

// Can't assign HashMaps to a HashMap at declaration, so do it in the class' constructor.
HashMap<EditorBuildProfile::BuildOption, HashMap<String, LocalVector<Variant>>> EditorBuildProfile::build_option_settings = {};

/* clang-format off */

const HashMap<EditorBuildProfile::BuildOption, LocalVector<EditorBuildProfile::BuildOption>> EditorBuildProfile::build_option_dependencies = {
	{ BUILD_OPTION_OPENXR, {
			BUILD_OPTION_XR,
	} },
	{ BUILD_OPTION_FORWARD_RENDERER, {
			BUILD_OPTION_RENDERING_DEVICE,
	} },
	{ BUILD_OPTION_MOBILE_RENDERER, {
			BUILD_OPTION_RENDERING_DEVICE,
	} },
	{ BUILD_OPTION_VULKAN, {
			BUILD_OPTION_FORWARD_RENDERER,
			BUILD_OPTION_MOBILE_RENDERER,
	} },
	{ BUILD_OPTION_D3D12, {
			BUILD_OPTION_FORWARD_RENDERER,
			BUILD_OPTION_MOBILE_RENDERER,
	} },
	{ BUILD_OPTION_METAL, {
			BUILD_OPTION_FORWARD_RENDERER,
			BUILD_OPTION_MOBILE_RENDERER,
	} },
	{ BUILD_OPTION_PHYSICS_GODOT_2D, {
			BUILD_OPTION_PHYSICS_2D,
	} },
	{ BUILD_OPTION_PHYSICS_GODOT_3D, {
			BUILD_OPTION_PHYSICS_3D,
	} },
	{ BUILD_OPTION_PHYSICS_JOLT, {
			BUILD_OPTION_PHYSICS_3D,
	} },
	{ BUILD_OPTION_DYNAMIC_FONTS, {
			BUILD_OPTION_TEXT_SERVER_ADVANCED,
	} },
	{ BUILD_OPTION_WOFF2_FONTS, {
			BUILD_OPTION_TEXT_SERVER_ADVANCED,
	} },
	{ BUILD_OPTION_GRAPHITE_FONTS, {
			BUILD_OPTION_TEXT_SERVER_ADVANCED,
	} },
};

const HashMap<EditorBuildProfile::BuildOption, LocalVector<String>> EditorBuildProfile::build_option_classes = {
	{ BUILD_OPTION_3D, {
			"Node3D",
	} },
	{ BUILD_OPTION_NAVIGATION_2D, {
			"NavigationAgent2D",
			"NavigationLink2D",
			"NavigationMeshSourceGeometryData2D",
			"NavigationObstacle2D"
			"NavigationPolygon",
			"NavigationRegion2D",
	} },
	{ BUILD_OPTION_NAVIGATION_3D, {
			"NavigationAgent3D",
			"NavigationLink3D",
			"NavigationMeshSourceGeometryData3D",
			"NavigationObstacle3D",
			"NavigationRegion3D",
	} },
	{ BUILD_OPTION_XR, {
			"XRBodyModifier3D",
			"XRBodyTracker",
			"XRControllerTracker",
			"XRFaceModifier3D",
			"XRFaceTracker",
			"XRHandModifier3D",
			"XRHandTracker",
			"XRInterface",
			"XRInterfaceExtension",
			"XRNode3D",
			"XROrigin3D",
			"XRPose",
			"XRPositionalTracker",
			"XRServer",
			"XRTracker",
			"XRVRS",
	} },
	{ BUILD_OPTION_RENDERING_DEVICE, {
			"RenderingDevice",
	} },
	{ BUILD_OPTION_PHYSICS_2D, {
			"CollisionObject2D",
			"CollisionPolygon2D",
			"CollisionShape2D",
			"Joint2D",
			"PhysicsServer2D",
			"PhysicsServer2DManager",
			"ShapeCast2D",
			"RayCast2D",
			"TouchScreenButton",
	} },
	{ BUILD_OPTION_PHYSICS_3D, {
			"CollisionObject3D",
			"CollisionPolygon3D",
			"CollisionShape3D",
			"CSGShape3D",
			"GPUParticlesAttractor3D",
			"GPUParticlesCollision3D",
			"Joint3D",
			"PhysicalBoneSimulator3D",
			"PhysicsServer3D",
			"PhysicsServer3DManager",
			"PhysicsServer3DRenderingServerHandler",
			"RayCast3D",
			"SoftBody3D",
			"SpringArm3D",
			"VehicleWheel3D",
	} },
	{ BUILD_OPTION_TEXT_SERVER_ADVANCED, {
			"CanvasItem",
			"Label3D",
			"TextServerAdvanced",
	} },
};

/* clang-format on */

void EditorBuildProfile::set_disable_class(const StringName &p_class, bool p_disabled) {
	if (p_disabled) {
		disabled_classes.insert(p_class);
	} else {
		disabled_classes.erase(p_class);
	}
}

bool EditorBuildProfile::is_class_disabled(const StringName &p_class) const {
	if (p_class == StringName()) {
		return false;
	}
	return disabled_classes.has(p_class) || is_class_disabled(ClassDB::get_parent_class_nocheck(p_class));
}

void EditorBuildProfile::set_item_collapsed(const StringName &p_class, bool p_collapsed) {
	if (p_collapsed) {
		collapsed_classes.insert(p_class);
	} else {
		collapsed_classes.erase(p_class);
	}
}

bool EditorBuildProfile::is_item_collapsed(const StringName &p_class) const {
	return collapsed_classes.has(p_class);
}

void EditorBuildProfile::set_disable_build_option(BuildOption p_build_option, bool p_disable) {
	ERR_FAIL_INDEX(p_build_option, BUILD_OPTION_MAX);
	build_options_disabled[p_build_option] = p_disable;
}

void EditorBuildProfile::clear_disabled_classes() {
	disabled_classes.clear();
	collapsed_classes.clear();
}

bool EditorBuildProfile::is_build_option_disabled(BuildOption p_build_option) const {
	ERR_FAIL_INDEX_V(p_build_option, BUILD_OPTION_MAX, false);
	return build_options_disabled[p_build_option];
}

bool EditorBuildProfile::get_build_option_disable_value(BuildOption p_build_option) {
	ERR_FAIL_INDEX_V(p_build_option, BUILD_OPTION_MAX, false);
	return build_option_disable_values[p_build_option];
}

bool EditorBuildProfile::get_build_option_explicit_use(BuildOption p_build_option) {
	ERR_FAIL_INDEX_V(p_build_option, BUILD_OPTION_MAX, false);
	return build_option_explicit_use[p_build_option];
}

void EditorBuildProfile::reset_build_options() {
	for (int i = 0; i < EditorBuildProfile::BUILD_OPTION_MAX; i++) {
		build_options_disabled[i] = build_option_disabled_by_default[i];
	}
}

void EditorBuildProfile::set_force_detect_classes(const String &p_classes) {
	force_detect_classes = p_classes;
}

String EditorBuildProfile::get_force_detect_classes() const {
	return force_detect_classes;
}

String EditorBuildProfile::get_build_option_name(BuildOption p_build_option) {
	ERR_FAIL_INDEX_V(p_build_option, BUILD_OPTION_MAX, String());
	const char *build_option_names[BUILD_OPTION_MAX] = {
		TTRC("3D Engine"),
		TTRC("Navigation (2D)"),
		TTRC("Navigation (3D)"),
		TTRC("XR"),
		TTRC("OpenXR"),
		TTRC("Wayland"),
		TTRC("X11"),
		TTRC("RenderingDevice"),
		TTRC("Forward+ Renderer"),
		TTRC("Mobile Renderer"),
		TTRC("Vulkan"),
		TTRC("D3D12"),
		TTRC("Metal"),
		TTRC("OpenGL"),
		TTRC("Physics Server (2D)"),
		TTRC("Godot Physics (2D)"),
		TTRC("Physics Server (3D)"),
		TTRC("Godot Physics (3D)"),
		TTRC("Jolt Physics"),
		TTRC("Text Server: Fallback"),
		TTRC("Text Server: Advanced"),
		TTRC("TTF, OTF, Type 1, WOFF1 Fonts"),
		TTRC("WOFF2 Fonts"),
		TTRC("SIL Graphite Fonts"),
		TTRC("Multi-channel Signed Distance Field Font Rendering"),
	};
	return TTRGET(build_option_names[p_build_option]);
}

String EditorBuildProfile::get_build_option_description(BuildOption p_build_option) {
	ERR_FAIL_INDEX_V(p_build_option, BUILD_OPTION_MAX, String());

	const char *build_option_descriptions[BUILD_OPTION_MAX] = {
		TTRC("3D Nodes as well as RenderingServer access to 3D features."),
		TTRC("Navigation Server and capabilities for 2D."),
		TTRC("Navigation Server and capabilities for 3D."),
		TTRC("XR (AR and VR)."),
		TTRC("OpenXR standard implementation (requires XR to be enabled)."),
		TTRC("Wayland display (Linux only)."),
		TTRC("X11 display (Linux only)."),
		TTRC("RenderingDevice based rendering (if disabled, the OpenGL backend is required)."),
		TTRC("Forward+ renderer for advanced 3D graphics."),
		TTRC("Mobile renderer for less advanced 3D graphics."),
		TTRC("Vulkan backend of RenderingDevice."),
		TTRC("Direct3D 12 backend of RenderingDevice."),
		TTRC("Metal backend of RenderingDevice (Apple arm64 only)."),
		TTRC("OpenGL backend (if disabled, the RenderingDevice backend is required)."),
		TTRC("Physics Server and capabilities for 2D."),
		TTRC("Godot Physics backend (2D)."),
		TTRC("Physics Server and capabilities for 3D."),
		TTRC("Godot Physics backend (3D)."),
		TTRC("Jolt Physics backend (3D only)."),
		TTRC("Fallback implementation of Text Server\nSupports basic text layouts."),
		TTRC("Text Server implementation powered by ICU and HarfBuzz libraries.\nSupports complex text layouts, BiDi, and contextual OpenType font features."),
		TTRC("TrueType, OpenType, Type 1, and WOFF1 font format support using FreeType library (if disabled, WOFF2 support is also disabled)."),
		TTRC("WOFF2 font format support using FreeType and Brotli libraries."),
		TTRC("SIL Graphite smart font technology support (supported by Advanced Text Server only)."),
		TTRC("Multi-channel signed distance field font rendering support using msdfgen library (pre-rendered MSDF fonts can be used even if this option disabled)."),
	};

	return TTRGET(build_option_descriptions[p_build_option]);
}

String EditorBuildProfile::get_build_option_identifier(BuildOption p_build_option) {
	ERR_FAIL_INDEX_V(p_build_option, BUILD_OPTION_MAX, String());
	return build_option_identifiers[p_build_option];
}

EditorBuildProfile::BuildOptionCategory EditorBuildProfile::get_build_option_category(BuildOption p_build_option) {
	ERR_FAIL_INDEX_V(p_build_option, BUILD_OPTION_MAX, BUILD_OPTION_CATEGORY_GENERAL);
	return build_option_category[p_build_option];
}

LocalVector<EditorBuildProfile::BuildOption> EditorBuildProfile::get_build_option_dependencies(BuildOption p_build_option) {
	ERR_FAIL_INDEX_V(p_build_option, BUILD_OPTION_MAX, LocalVector<EditorBuildProfile::BuildOption>());
	if (build_option_dependencies.has(p_build_option)) {
		return LocalVector<EditorBuildProfile::BuildOption>(build_option_dependencies[p_build_option]);
	}
	return LocalVector<EditorBuildProfile::BuildOption>();
}

HashMap<String, LocalVector<Variant>> EditorBuildProfile::get_build_option_settings(BuildOption p_build_option) {
	ERR_FAIL_INDEX_V(p_build_option, BUILD_OPTION_MAX, (HashMap<String, LocalVector<Variant>>()));
	if (build_option_settings.has(p_build_option)) {
		return HashMap<String, LocalVector<Variant>>(build_option_settings[p_build_option]);
	}
	return HashMap<String, LocalVector<Variant>>();
}

LocalVector<String> EditorBuildProfile::get_build_option_classes(BuildOption p_build_option) {
	ERR_FAIL_INDEX_V(p_build_option, BUILD_OPTION_MAX, LocalVector<String>());
	if (build_option_classes.has(p_build_option)) {
		return LocalVector<String>(build_option_classes[p_build_option]);
	}
	return LocalVector<String>();
}

String EditorBuildProfile::get_build_option_category_name(BuildOptionCategory p_build_option_category) {
	ERR_FAIL_INDEX_V(p_build_option_category, BUILD_OPTION_CATEGORY_MAX, String());

	const char *build_option_subcategories[BUILD_OPTION_CATEGORY_MAX]{
		TTRC("General Features:"),
		TTRC("Graphics and Rendering:"),
		TTRC("Physics Systems:"),
		TTRC("Text Rendering and Font Options:"),
	};

	return TTRGET(build_option_subcategories[p_build_option_category]);
}

Error EditorBuildProfile::save_to_file(const String &p_path) {
	Dictionary data;
	data["type"] = "build_profile";
	Array dis_classes;
	for (const StringName &E : disabled_classes) {
		dis_classes.push_back(String(E));
	}
	dis_classes.sort();
	data["disabled_classes"] = dis_classes;

	Dictionary dis_build_options;
	for (int i = 0; i < BUILD_OPTION_MAX; i++) {
		if (build_options_disabled[i] != build_option_disabled_by_default[i]) {
			if (build_options_disabled[i]) {
				dis_build_options[build_option_identifiers[i]] = build_option_disable_values[i];
			} else {
				dis_build_options[build_option_identifiers[i]] = !build_option_disable_values[i];
			}
		}
	}

	data["disabled_build_options"] = dis_build_options;

	if (!force_detect_classes.is_empty()) {
		data["force_detect_classes"] = force_detect_classes;
	}

	Ref<FileAccess> f = FileAccess::open(p_path, FileAccess::WRITE);
	ERR_FAIL_COND_V_MSG(f.is_null(), ERR_CANT_CREATE, "Cannot create file '" + p_path + "'.");

	String text = JSON::stringify(data, "\t");
	f->store_string(text);
	return OK;
}

Error EditorBuildProfile::load_from_file(const String &p_path) {
	Error err;
	String text = FileAccess::get_file_as_string(p_path, &err);
	if (err != OK) {
		return err;
	}

	JSON json;
	err = json.parse(text);
	if (err != OK) {
		ERR_PRINT("Error parsing '" + p_path + "' on line " + itos(json.get_error_line()) + ": " + json.get_error_message());
		return ERR_PARSE_ERROR;
	}

	Dictionary data = json.get_data();

	if (!data.has("type") || String(data["type"]) != "build_profile") {
		ERR_PRINT("Error parsing '" + p_path + "', it's not a build profile.");
		return ERR_PARSE_ERROR;
	}

	disabled_classes.clear();

	if (data.has("disabled_classes")) {
		Array disabled_classes_arr = data["disabled_classes"];
		for (int i = 0; i < disabled_classes_arr.size(); i++) {
			disabled_classes.insert(disabled_classes_arr[i]);
		}
	}

	for (int i = 0; i < BUILD_OPTION_MAX; i++) {
		build_options_disabled[i] = build_option_disabled_by_default[i];
	}

	if (data.has("disabled_build_options")) {
		Dictionary disabled_build_options_arr = data["disabled_build_options"];

		for (const KeyValue<Variant, Variant> &kv : disabled_build_options_arr) {
			String key = kv.key;

			for (int i = 0; i < BUILD_OPTION_MAX; i++) {
				String f = build_option_identifiers[i];
				if (f == key) {
					build_options_disabled[i] = true;
					break;
				}
			}
		}
	}

	if (data.has("force_detect_classes")) {
		force_detect_classes = data["force_detect_classes"];
	}

	return OK;
}

void EditorBuildProfile::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_disable_class", "class_name", "disable"), &EditorBuildProfile::set_disable_class);
	ClassDB::bind_method(D_METHOD("is_class_disabled", "class_name"), &EditorBuildProfile::is_class_disabled);

	ClassDB::bind_method(D_METHOD("set_disable_build_option", "build_option", "disable"), &EditorBuildProfile::set_disable_build_option);
	ClassDB::bind_method(D_METHOD("is_build_option_disabled", "build_option"), &EditorBuildProfile::is_build_option_disabled);

	ClassDB::bind_method(D_METHOD("get_build_option_name", "build_option"), &EditorBuildProfile::_get_build_option_name);

	ClassDB::bind_method(D_METHOD("save_to_file", "path"), &EditorBuildProfile::save_to_file);
	ClassDB::bind_method(D_METHOD("load_from_file", "path"), &EditorBuildProfile::load_from_file);

	BIND_ENUM_CONSTANT(BUILD_OPTION_3D);
	BIND_ENUM_CONSTANT(BUILD_OPTION_NAVIGATION_2D);
	BIND_ENUM_CONSTANT(BUILD_OPTION_NAVIGATION_3D);
	BIND_ENUM_CONSTANT(BUILD_OPTION_XR);
	BIND_ENUM_CONSTANT(BUILD_OPTION_OPENXR);
	BIND_ENUM_CONSTANT(BUILD_OPTION_WAYLAND);
	BIND_ENUM_CONSTANT(BUILD_OPTION_X11);
	BIND_ENUM_CONSTANT(BUILD_OPTION_RENDERING_DEVICE);
	BIND_ENUM_CONSTANT(BUILD_OPTION_FORWARD_RENDERER);
	BIND_ENUM_CONSTANT(BUILD_OPTION_MOBILE_RENDERER);
	BIND_ENUM_CONSTANT(BUILD_OPTION_VULKAN);
	BIND_ENUM_CONSTANT(BUILD_OPTION_D3D12);
	BIND_ENUM_CONSTANT(BUILD_OPTION_METAL);
	BIND_ENUM_CONSTANT(BUILD_OPTION_OPENGL);
	BIND_ENUM_CONSTANT(BUILD_OPTION_PHYSICS_2D);
	BIND_ENUM_CONSTANT(BUILD_OPTION_PHYSICS_GODOT_2D);
	BIND_ENUM_CONSTANT(BUILD_OPTION_PHYSICS_3D);
	BIND_ENUM_CONSTANT(BUILD_OPTION_PHYSICS_GODOT_3D);
	BIND_ENUM_CONSTANT(BUILD_OPTION_PHYSICS_JOLT);
	BIND_ENUM_CONSTANT(BUILD_OPTION_TEXT_SERVER_FALLBACK);
	BIND_ENUM_CONSTANT(BUILD_OPTION_TEXT_SERVER_ADVANCED);
	BIND_ENUM_CONSTANT(BUILD_OPTION_DYNAMIC_FONTS);
	BIND_ENUM_CONSTANT(BUILD_OPTION_WOFF2_FONTS);
	BIND_ENUM_CONSTANT(BUILD_OPTION_GRAPHITE_FONTS);
	BIND_ENUM_CONSTANT(BUILD_OPTION_MSDFGEN);
	BIND_ENUM_CONSTANT(BUILD_OPTION_MAX);

	BIND_ENUM_CONSTANT(BUILD_OPTION_CATEGORY_GENERAL);
	BIND_ENUM_CONSTANT(BUILD_OPTION_CATEGORY_GRAPHICS);
	BIND_ENUM_CONSTANT(BUILD_OPTION_CATEGORY_PHYSICS);
	BIND_ENUM_CONSTANT(BUILD_OPTION_CATEGORY_TEXT_SERVER);
	BIND_ENUM_CONSTANT(BUILD_OPTION_CATEGORY_MAX);
}

EditorBuildProfile::EditorBuildProfile() {
	reset_build_options();

	HashMap<String, LocalVector<Variant>> settings_openxr = {
		{ "xr/openxr/enabled", { true } },
	};
	build_option_settings.insert(BUILD_OPTION_OPENXR, settings_openxr);
	HashMap<String, LocalVector<Variant>> settings_wayland = {
		{ "display/display_server/driver.linuxbsd", { "default", "wayland" } },
	};
	build_option_settings.insert(BUILD_OPTION_OPENXR, settings_wayland);
	HashMap<String, LocalVector<Variant>> settings_x11 = {
		{ "display/display_server/driver.linuxbsd", { "default", "x11" } },
	};
	build_option_settings.insert(BUILD_OPTION_OPENXR, settings_x11);
	HashMap<String, LocalVector<Variant>> settings_rd = {
		{ "rendering/renderer/rendering_method", { "forward_plus", "mobile" } },
		{ "rendering/renderer/rendering_method.mobile", { "forward_plus", "mobile" } },
		{ "rendering/renderer/rendering_method.web", { "forward_plus", "mobile" } },
	};
	build_option_settings.insert(BUILD_OPTION_RENDERING_DEVICE, settings_rd);
	HashMap<String, LocalVector<Variant>> settings_vulkan = {
		{ "rendering/rendering_device/driver", { "vulkan" } },
		{ "rendering/rendering_device/driver.windows", { "vulkan" } },
		{ "rendering/rendering_device/driver.linuxbsd", { "vulkan" } },
		{ "rendering/rendering_device/driver.android", { "vulkan" } },
		{ "rendering/rendering_device/driver.ios", { "vulkan" } },
		{ "rendering/rendering_device/driver.macos", { "vulkan" } },
		{ "rendering/rendering_device/fallback_to_vulkan", { true } },
	};
	build_option_settings.insert(BUILD_OPTION_VULKAN, settings_vulkan);
	HashMap<String, LocalVector<Variant>> settings_d3d12 = {
		{ "rendering/rendering_device/driver", { "d3d12" } },
		{ "rendering/rendering_device/driver.windows", { "d3d12" } },
		{ "rendering/rendering_device/driver.linuxbsd", { "d3d12" } },
		{ "rendering/rendering_device/driver.android", { "d3d12" } },
		{ "rendering/rendering_device/driver.ios", { "d3d12" } },
		{ "rendering/rendering_device/driver.macos", { "d3d12" } },
		{ "rendering/rendering_device/fallback_to_d3d12", { true } },
	};
	build_option_settings.insert(BUILD_OPTION_VULKAN, settings_vulkan);
	HashMap<String, LocalVector<Variant>> settings_metal = {
		{ "rendering/rendering_device/driver", { "metal" } },
		{ "rendering/rendering_device/driver.ios", { "metal" } },
		{ "rendering/rendering_device/driver.macos", { "metal" } },
	};
	build_option_settings.insert(BUILD_OPTION_METAL, settings_metal);
	HashMap<String, LocalVector<Variant>> settings_opengl = {
		{ "rendering/renderer/rendering_method", { "gl_compatibility" } },
		{ "rendering/renderer/rendering_method.mobile", { "gl_compatibility" } },
		{ "rendering/renderer/rendering_method.web", { "gl_compatibility" } },
		{ "rendering/rendering_device/fallback_to_opengl3", { true } },
	};
	build_option_settings.insert(BUILD_OPTION_OPENGL, settings_opengl);
	HashMap<String, LocalVector<Variant>> settings_phy_godot_3d = {
		{ "physics/3d/physics_engine", { "DEFAULT", "GodotPhysics3D" } },
	};
	build_option_settings.insert(BUILD_OPTION_PHYSICS_GODOT_3D, settings_phy_godot_3d);
	HashMap<String, LocalVector<Variant>> settings_jolt = {
		{ "physics/3d/physics_engine", { "Jolt Physics" } },
	};
	build_option_settings.insert(BUILD_OPTION_PHYSICS_JOLT, settings_jolt);
	HashMap<String, LocalVector<Variant>> settings_msdfgen = {
		{ "gui/theme/default_font_multichannel_signed_distance_field", { true } },
	};
	build_option_settings.insert(BUILD_OPTION_MSDFGEN, settings_msdfgen);
}

//////////////////////////

void EditorBuildProfileManager::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY: {
			String last_file = EditorSettings::get_singleton()->get_project_metadata("build_profile", "last_file_path", "");
			if (!last_file.is_empty()) {
				_import_profile(last_file);
			}
			if (edited.is_null()) {
				edited.instantiate();
				_update_edited_profile();
			}

		} break;
	}
}

void EditorBuildProfileManager::_profile_action(int p_action) {
	last_action = Action(p_action);

	switch (p_action) {
		case ACTION_RESET: {
			confirm_dialog->set_text(TTR("Reset the edited profile?"));
			confirm_dialog->popup_centered();
		} break;
		case ACTION_LOAD: {
			import_profile->popup_file_dialog();
		} break;
		case ACTION_SAVE: {
			if (!profile_path->get_text().is_empty()) {
				Error err = edited->save_to_file(profile_path->get_text());
				if (err != OK) {
					EditorNode::get_singleton()->show_warning(TTR("File saving failed."));
				}
				break;
			}
			[[fallthrough]];
		}
		case ACTION_SAVE_AS: {
			export_profile->popup_file_dialog();
			export_profile->set_current_file(profile_path->get_text());
		} break;
		case ACTION_NEW: {
			confirm_dialog->set_text(TTR("Create a new profile?"));
			confirm_dialog->popup_centered();
		} break;
		case ACTION_DETECT: {
			String text = TTR("This will scan all files in the current project to detect used classes.\nNote that the first scan may take a while, specially in larger projects.");
#ifdef MODULE_MONO_ENABLED
			text += "\n\n" + TTR("Warning: Class detection for C# scripts is not currently available, and such files will be ignored.");
#endif // MODULE_MONO_ENABLED
			confirm_dialog->set_text(text);
			confirm_dialog->popup_centered();
		} break;
		case ACTION_MAX: {
		} break;
	}
}

void EditorBuildProfileManager::_find_files(EditorFileSystemDirectory *p_dir, const HashMap<String, DetectedFile> &p_cache, HashMap<String, DetectedFile> &r_detected) {
	if (p_dir == nullptr || p_dir->get_path().get_file().begins_with(".")) {
		return;
	}

	for (int i = 0; i < p_dir->get_file_count(); i++) {
		String p = p_dir->get_file_path(i);

		if (EditorNode::get_singleton()->progress_task_step("detect_classes_from_project", p, 1)) {
			project_scan_canceled = true;
			return;
		}

		String p_check = p;
		// Make so that the import file is the one checked if available,
		// so the cache can be updated when it changes.
		if (ResourceFormatImporter::get_singleton()->exists(p_check)) {
			p_check += ".import";
		}

		uint64_t timestamp = 0;
		String md5;

		if (p_cache.has(p)) {
			const DetectedFile &cache = p_cache[p];
			// Check if timestamp and MD5 match.
			timestamp = FileAccess::get_modified_time(p_check);
			bool cache_valid = true;
			if (cache.timestamp != timestamp) {
				md5 = FileAccess::get_md5(p_check);
				if (md5 != cache.md5) {
					cache_valid = false;
				}
			}

			if (cache_valid) {
				r_detected.insert(p, cache);
				continue;
			}
		}

		// Not cached, or cache invalid.

		DetectedFile cache;

		HashSet<StringName> classes;
		ResourceLoader::get_classes_used(p, &classes);
		for (const StringName &E : classes) {
			cache.classes.push_back(E);
		}

		HashSet<String> build_deps;
		ResourceFormatImporter::get_singleton()->get_build_dependencies(p, &build_deps);
		for (const String &E : build_deps) {
			cache.build_deps.push_back(E);
		}

		if (md5.is_empty()) {
			cache.timestamp = FileAccess::get_modified_time(p_check);
			cache.md5 = FileAccess::get_md5(p_check);
		} else {
			cache.timestamp = timestamp;
			cache.md5 = md5;
		}

		r_detected.insert(p, cache);
	}

	for (int i = 0; i < p_dir->get_subdir_count(); i++) {
		_find_files(p_dir->get_subdir(i), p_cache, r_detected);
	}
}

void EditorBuildProfileManager::_detect_from_project() {
	EditorNode::get_singleton()->progress_add_task("detect_classes_from_project", TTRC("Scanning Project for Used Classes"), 3, true);

	HashMap<String, DetectedFile> previous_file_cache;

	Ref<FileAccess> f = FileAccess::open(EditorPaths::get_singleton()->get_project_settings_dir().path_join("used_class_cache"), FileAccess::READ);
	if (f.is_valid()) {
		while (!f->eof_reached()) {
			String l = f->get_line();
			Vector<String> fields = l.split("::");
			if (fields.size() == 5) {
				const String &path = fields[0];
				DetectedFile df;
				df.timestamp = fields[1].to_int();
				df.md5 = fields[2];
				df.classes = fields[3].split(",", false);
				df.build_deps = fields[4].split(",", false);
				previous_file_cache.insert(path, df);
			}
		}
		f.unref();
	}

	HashMap<String, DetectedFile> updated_file_cache;

	_find_files(EditorFileSystem::get_singleton()->get_filesystem(), previous_file_cache, updated_file_cache);

	if (project_scan_canceled) {
		project_scan_canceled = false;
		EditorNode::get_singleton()->progress_end_task("detect_classes_from_project");

		return;
	}

	EditorNode::get_singleton()->progress_task_step("detect_classes_from_project", TTRC("Processing Classes Found"), 2);

	HashSet<StringName> used_classes;
	LocalVector<String> used_build_deps;

	// Find classes and update the disk cache in the process.
	f = FileAccess::open(EditorPaths::get_singleton()->get_project_settings_dir().path_join("used_class_cache"), FileAccess::WRITE);

	for (const KeyValue<String, DetectedFile> &E : updated_file_cache) {
		String l = E.key + "::" + itos(E.value.timestamp) + "::" + E.value.md5 + "::";
		for (int i = 0; i < E.value.classes.size(); i++) {
			String c = E.value.classes[i];
			if (i > 0) {
				l += ",";
			}
			l += c;
			used_classes.insert(c);
		}
		l += "::";
		for (int i = 0; i < E.value.build_deps.size(); i++) {
			String c = E.value.build_deps[i];
			if (i > 0) {
				l += ",";
			}
			l += c;
			used_build_deps.push_back(c);
		}
		f->store_line(l);
	}

	f.unref();

	// Add classes that are either necessary for the engine to work properly, or there isn't a way to infer their use.

	const LocalVector<String> hardcoded_classes = { "InputEvent", "MainLoop", "StyleBox" };
	for (const String &hc_class : hardcoded_classes) {
		used_classes.insert(hc_class);

		LocalVector<StringName> inheriters;
		ClassDB::get_inheriters_from_class(hc_class, inheriters);
		for (const StringName &inheriter : inheriters) {
			used_classes.insert(inheriter);
		}
	}

	// Add forced classes typed by the user.

	const Vector<String> force_detect = edited->get_force_detect_classes().split(",");
	for (const String &class_name : force_detect) {
		const String class_stripped = class_name.strip_edges();
		if (!class_stripped.is_empty()) {
			used_classes.insert(class_stripped);
		}
	}

	// Filter all classes to discard inherited ones.

	HashSet<StringName> all_used_classes;

	for (const StringName &E : used_classes) {
		StringName c = E;
		if (!ClassDB::class_exists(c)) {
			// Maybe this is an old class that got replaced? Try getting compat class.
			c = ClassDB::get_compatibility_class(c);
			if (!c) {
				// No luck, skip.
				continue;
			}
		}

		List<StringName> dependencies;
		ClassDB::get_class_dependencies(E, &dependencies);
		for (const StringName &dep : dependencies) {
			if (!all_used_classes.has(dep)) {
				// Add classes which this class depends upon.
				all_used_classes.insert(dep);
			}
		}

		while (c) {
			all_used_classes.insert(c);
			c = ClassDB::get_parent_class(c);
		}
	}

	edited->clear_disabled_classes();

	LocalVector<StringName> all_classes;
	ClassDB::get_class_list(all_classes);

	for (const StringName &class_name : all_classes) {
		if (String(class_name).begins_with("Editor") || ClassDB::get_api_type(class_name) != ClassDB::API_CORE || all_used_classes.has(class_name)) {
			// This class is valid or editor-only, do nothing.
			continue;
		}

		StringName p = ClassDB::get_parent_class(class_name);
		if (!p || all_used_classes.has(p)) {
			// If no parent, or if the parent is enabled, then add to disabled classes.
			// This way we avoid disabling redundant classes.
			edited->set_disable_class(class_name, true);
		}
	}

	edited->reset_build_options();

	for (int i = 0; i < EditorBuildProfile::BUILD_OPTION_MAX; i++) {
		// Check if the build option requires other options that are currently disabled.
		LocalVector<EditorBuildProfile::BuildOption> dependencies = EditorBuildProfile::get_build_option_dependencies(EditorBuildProfile::BuildOption(i));
		if (!dependencies.is_empty()) {
			bool disable = true;
			for (EditorBuildProfile::BuildOption dependency : dependencies) {
				if (!edited->is_build_option_disabled(dependency)) {
					disable = false;
					break;
				}
			}

			if (disable) {
				edited->set_disable_build_option(EditorBuildProfile::BuildOption(i), true);
				continue;
			}
		}

		bool skip = false;
		bool ignore = true;

		// Check if the build option has enabled classes using it.
		const LocalVector<String> classes = EditorBuildProfile::get_build_option_classes(EditorBuildProfile::BuildOption(i));
		if (!classes.is_empty()) {
			for (StringName class_name : classes) {
				if (!edited->is_class_disabled(class_name)) {
					skip = true;
					break;
				}
			}

			if (skip) {
				continue;
			}

			ignore = false;
		}

		// Check if there's project settings requiring it.
		const HashMap<String, LocalVector<Variant>> settings_list = EditorBuildProfile::get_build_option_settings(EditorBuildProfile::BuildOption(i));
		if (!settings_list.is_empty()) {
			for (KeyValue<String, LocalVector<Variant>> KV : settings_list) {
				Variant proj_value = GLOBAL_GET(KV.key);
				for (Variant value : KV.value) {
					if (proj_value == value) {
						skip = true;
						break;
					}
				}

				if (skip) {
					break;
				}
			}

			if (skip) {
				continue;
			}

			ignore = false;
		}

		// Check if a resource setting depends on it.
		if (used_build_deps.has(EditorBuildProfile::get_build_option_identifier(EditorBuildProfile::BuildOption(i)))) {
			continue;
		} else if (EditorBuildProfile::get_build_option_explicit_use(EditorBuildProfile::BuildOption(i))) {
			ignore = false;
		}

		if (!skip && !ignore) {
			edited->set_disable_build_option(EditorBuildProfile::BuildOption(i), true);
		}
	}

	if (edited->is_build_option_disabled(EditorBuildProfile::BUILD_OPTION_TEXT_SERVER_ADVANCED)) {
		edited->set_disable_build_option(EditorBuildProfile::BUILD_OPTION_TEXT_SERVER_FALLBACK, false);
	}

	EditorNode::get_singleton()->progress_end_task("detect_classes_from_project");
}

void EditorBuildProfileManager::_action_confirm() {
	switch (last_action) {
		case ACTION_RESET: {
			edited.instantiate();
			_update_edited_profile();
		} break;
		case ACTION_LOAD: {
		} break;
		case ACTION_SAVE: {
		} break;
		case ACTION_SAVE_AS: {
		} break;
		case ACTION_NEW: {
			profile_path->set_text("");
			edited.instantiate();
			_update_edited_profile();
		} break;
		case ACTION_DETECT: {
			_detect_from_project();
			_update_edited_profile();
		} break;
		case ACTION_MAX: {
		} break;
	}
}

void EditorBuildProfileManager::_hide_requested() {
	_cancel_pressed(); // From AcceptDialog.
}

void EditorBuildProfileManager::_fill_classes_from(TreeItem *p_parent, const String &p_class, const String &p_selected) {
	TreeItem *class_item = class_list->create_item(p_parent);
	class_item->set_cell_mode(0, TreeItem::CELL_MODE_CHECK);
	class_item->set_icon(0, EditorNode::get_singleton()->get_class_icon(p_class));
	const String &text = p_class;

	bool disabled = edited->is_class_disabled(p_class);
	if (disabled) {
		class_item->set_custom_color(0, class_list->get_theme_color(SNAME("font_disabled_color"), EditorStringName(Editor)));
	}

	class_item->set_text(0, text);
	class_item->set_editable(0, true);
	class_item->set_selectable(0, true);
	class_item->set_metadata(0, p_class);

	bool collapsed = edited->is_item_collapsed(p_class);
	class_item->set_collapsed(collapsed);

	if (p_class == p_selected) {
		class_item->select(0);
	}
	if (disabled) {
		// Class disabled, do nothing else (do not show further).
		return;
	}

	class_item->set_checked(0, true); // If it's not disabled, its checked.

	List<StringName> child_classes;
	ClassDB::get_direct_inheriters_from_class(p_class, &child_classes);
	child_classes.sort_custom<StringName::AlphCompare>();

	for (const StringName &name : child_classes) {
		if (String(name).begins_with("Editor") || ClassDB::get_api_type(name) != ClassDB::API_CORE) {
			continue;
		}
		_fill_classes_from(class_item, name, p_selected);
	}
}

void EditorBuildProfileManager::_class_list_item_selected() {
	if (updating_build_options) {
		return;
	}

	TreeItem *item = class_list->get_selected();
	if (!item) {
		return;
	}

	Variant md = item->get_metadata(0);
	if (md.is_string()) {
		description_bit->parse_symbol("class|" + md.operator String() + "|");
	} else if (md.get_type() == Variant::INT) {
		String build_option_description = EditorBuildProfile::get_build_option_description(EditorBuildProfile::BuildOption((int)md));
		description_bit->set_custom_text(TTR(item->get_text(0)), String(), TTRGET(build_option_description));
	}
}

void EditorBuildProfileManager::_class_list_item_edited() {
	if (updating_build_options) {
		return;
	}

	TreeItem *item = class_list->get_edited();
	if (!item) {
		return;
	}

	bool checked = item->is_checked(0);

	Variant md = item->get_metadata(0);
	if (md.is_string()) {
		String class_selected = md;
		edited->set_disable_class(class_selected, !checked);
		_update_edited_profile();
	} else if (md.get_type() == Variant::INT) {
		int build_option_selected = md;
		edited->set_disable_build_option(EditorBuildProfile::BuildOption(build_option_selected), !checked);
	}
}

void EditorBuildProfileManager::_class_list_item_collapsed(Object *p_item) {
	if (updating_build_options) {
		return;
	}

	TreeItem *item = Object::cast_to<TreeItem>(p_item);
	if (!item) {
		return;
	}

	Variant md = item->get_metadata(0);
	if (!md.is_string()) {
		return;
	}

	String class_name = md;
	bool collapsed = item->is_collapsed();
	edited->set_item_collapsed(class_name, collapsed);
}

void EditorBuildProfileManager::_update_edited_profile() {
	String class_selected;
	int build_option_selected = -1;

	if (class_list->get_selected()) {
		Variant md = class_list->get_selected()->get_metadata(0);
		if (md.is_string()) {
			class_selected = md;
		} else if (md.get_type() == Variant::INT) {
			build_option_selected = md;
		}
	}

	class_list->clear();

	updating_build_options = true;

	TreeItem *root = class_list->create_item();

	HashMap<EditorBuildProfile::BuildOptionCategory, TreeItem *> subcats;
	for (int i = 0; i < EditorBuildProfile::BUILD_OPTION_CATEGORY_MAX; i++) {
		TreeItem *build_cat;
		build_cat = class_list->create_item(root);

		build_cat->set_text(0, EditorBuildProfile::get_build_option_category_name(EditorBuildProfile::BuildOptionCategory(i)));
		subcats[EditorBuildProfile::BuildOptionCategory(i)] = build_cat;
	}

	for (int i = 0; i < EditorBuildProfile::BUILD_OPTION_MAX; i++) {
		TreeItem *build_option;
		build_option = class_list->create_item(subcats[EditorBuildProfile::get_build_option_category(EditorBuildProfile::BuildOption(i))]);

		build_option->set_cell_mode(0, TreeItem::CELL_MODE_CHECK);
		build_option->set_text(0, EditorBuildProfile::get_build_option_name(EditorBuildProfile::BuildOption(i)));
		build_option->set_selectable(0, true);
		build_option->set_editable(0, true);
		build_option->set_metadata(0, i);
		if (!edited->is_build_option_disabled(EditorBuildProfile::BuildOption(i))) {
			build_option->set_checked(0, true);
		}

		if (i == build_option_selected) {
			build_option->select(0);
		}
	}

	TreeItem *classes = class_list->create_item(root);
	classes->set_text(0, TTR("Nodes and Classes:"));

	_fill_classes_from(classes, "Node", class_selected);
	_fill_classes_from(classes, "Resource", class_selected);

	force_detect_classes->set_text(edited->get_force_detect_classes());

	updating_build_options = false;

	_class_list_item_selected();
}

void EditorBuildProfileManager::_force_detect_classes_changed(const String &p_text) {
	if (updating_build_options) {
		return;
	}
	edited->set_force_detect_classes(force_detect_classes->get_text());
}

void EditorBuildProfileManager::_import_profile(const String &p_path) {
	Ref<EditorBuildProfile> profile;
	profile.instantiate();
	Error err = profile->load_from_file(p_path);
	String basefile = p_path.get_file();
	if (err != OK) {
		EditorNode::get_singleton()->show_warning(vformat(TTR("File '%s' format is invalid, import aborted."), basefile));
		return;
	}

	profile_path->set_text(p_path);
	EditorSettings::get_singleton()->set_project_metadata("build_profile", "last_file_path", p_path);

	edited = profile;
	_update_edited_profile();
}

void EditorBuildProfileManager::_export_profile(const String &p_path) {
	ERR_FAIL_COND(edited.is_null());
	Error err = edited->save_to_file(p_path);
	if (err != OK) {
		EditorNode::get_singleton()->show_warning(vformat(TTR("Error saving profile to path: '%s'."), p_path));
	} else {
		profile_path->set_text(p_path);
		EditorSettings::get_singleton()->set_project_metadata("build_profile", "last_file_path", p_path);
	}
}

Ref<EditorBuildProfile> EditorBuildProfileManager::get_current_profile() {
	return edited;
}

EditorBuildProfileManager *EditorBuildProfileManager::singleton = nullptr;

void EditorBuildProfileManager::_bind_methods() {
	ClassDB::bind_method("_update_selected_profile", &EditorBuildProfileManager::_update_edited_profile);
}

EditorBuildProfileManager::EditorBuildProfileManager() {
	VBoxContainer *main_vbc = memnew(VBoxContainer);
	add_child(main_vbc);

	HBoxContainer *path_hbc = memnew(HBoxContainer);
	profile_path = memnew(LineEdit);
	path_hbc->add_child(profile_path);
	profile_path->set_accessibility_name(TTRC("Profile Path"));
	profile_path->set_editable(true);
	profile_path->set_h_size_flags(Control::SIZE_EXPAND_FILL);

	profile_actions[ACTION_NEW] = memnew(Button(TTR("New")));
	path_hbc->add_child(profile_actions[ACTION_NEW]);
	profile_actions[ACTION_NEW]->connect(SceneStringName(pressed), callable_mp(this, &EditorBuildProfileManager::_profile_action).bind(ACTION_NEW));

	profile_actions[ACTION_LOAD] = memnew(Button(TTR("Load")));
	path_hbc->add_child(profile_actions[ACTION_LOAD]);
	profile_actions[ACTION_LOAD]->connect(SceneStringName(pressed), callable_mp(this, &EditorBuildProfileManager::_profile_action).bind(ACTION_LOAD));

	profile_actions[ACTION_SAVE] = memnew(Button(TTR("Save")));
	path_hbc->add_child(profile_actions[ACTION_SAVE]);
	profile_actions[ACTION_SAVE]->connect(SceneStringName(pressed), callable_mp(this, &EditorBuildProfileManager::_profile_action).bind(ACTION_SAVE));

	profile_actions[ACTION_SAVE_AS] = memnew(Button(TTR("Save As")));
	path_hbc->add_child(profile_actions[ACTION_SAVE_AS]);
	profile_actions[ACTION_SAVE_AS]->connect(SceneStringName(pressed), callable_mp(this, &EditorBuildProfileManager::_profile_action).bind(ACTION_SAVE_AS));

	main_vbc->add_margin_child(TTR("Profile:"), path_hbc);

	main_vbc->add_child(memnew(HSeparator));

	HBoxContainer *profiles_hbc = memnew(HBoxContainer);

	profile_actions[ACTION_RESET] = memnew(Button(TTR("Reset to Defaults")));
	profiles_hbc->add_child(profile_actions[ACTION_RESET]);
	profile_actions[ACTION_RESET]->connect(SceneStringName(pressed), callable_mp(this, &EditorBuildProfileManager::_profile_action).bind(ACTION_RESET));

	profile_actions[ACTION_DETECT] = memnew(Button(TTR("Detect from Project")));
	profiles_hbc->add_child(profile_actions[ACTION_DETECT]);
	profile_actions[ACTION_DETECT]->connect(SceneStringName(pressed), callable_mp(this, &EditorBuildProfileManager::_profile_action).bind(ACTION_DETECT));

	main_vbc->add_margin_child(TTR("Actions:"), profiles_hbc);

	class_list = memnew(Tree);
	class_list->set_auto_translate_mode(AUTO_TRANSLATE_MODE_DISABLED);
	class_list->set_hide_root(true);
	class_list->set_edit_checkbox_cell_only_when_checkbox_is_pressed(true);
	class_list->set_scroll_hint_mode(Tree::SCROLL_HINT_MODE_BOTH);
	class_list->connect("cell_selected", callable_mp(this, &EditorBuildProfileManager::_class_list_item_selected));
	class_list->connect("item_edited", callable_mp(this, &EditorBuildProfileManager::_class_list_item_edited), CONNECT_DEFERRED);
	class_list->connect("item_collapsed", callable_mp(this, &EditorBuildProfileManager::_class_list_item_collapsed));

	// It will be displayed once the user creates or chooses a profile.
	MarginContainer *mc = main_vbc->add_margin_child(TTRC("Configure Engine Compilation Profile:"), class_list, true);
	mc->set_theme_type_variation("NoBorderHorizontalWindow");
	mc->set_v_size_flags(Control::SIZE_EXPAND_FILL);

	description_bit = memnew(EditorHelpBit);
	description_bit->set_content_height_limits(80 * EDSCALE, 80 * EDSCALE);
	description_bit->connect("request_hide", callable_mp(this, &EditorBuildProfileManager::_hide_requested));
	main_vbc->add_margin_child(TTR("Description:"), description_bit, false);

	confirm_dialog = memnew(ConfirmationDialog);
	add_child(confirm_dialog);
	confirm_dialog->set_title(TTR("Please Confirm:"));
	confirm_dialog->connect(SceneStringName(confirmed), callable_mp(this, &EditorBuildProfileManager::_action_confirm));

	import_profile = memnew(EditorFileDialog);
	add_child(import_profile);
	import_profile->set_file_mode(EditorFileDialog::FILE_MODE_OPEN_FILE);
	import_profile->add_filter("*.gdbuild,*.build", TTR("Engine Compilation Profile"));
	import_profile->connect("file_selected", callable_mp(this, &EditorBuildProfileManager::_import_profile));
	import_profile->set_title(TTR("Load Profile"));
	import_profile->set_access(EditorFileDialog::ACCESS_FILESYSTEM);

	export_profile = memnew(EditorFileDialog);
	add_child(export_profile);
	export_profile->set_file_mode(EditorFileDialog::FILE_MODE_SAVE_FILE);
	export_profile->add_filter("*.gdbuild,*.build", TTR("Engine Compilation Profile"));
	export_profile->connect("file_selected", callable_mp(this, &EditorBuildProfileManager::_export_profile));
	export_profile->set_title(TTR("Export Profile"));
	export_profile->set_access(EditorFileDialog::ACCESS_FILESYSTEM);

	force_detect_classes = memnew(LineEdit);
	force_detect_classes->set_accessibility_name(TTRC("Forced Classes on Detect:"));
	main_vbc->add_margin_child(TTR("Forced Classes on Detect:"), force_detect_classes);
	force_detect_classes->connect(SceneStringName(text_changed), callable_mp(this, &EditorBuildProfileManager::_force_detect_classes_changed));

	set_title(TTR("Edit Compilation Configuration Profile"));

	singleton = this;
}
