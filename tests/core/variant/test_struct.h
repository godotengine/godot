/**************************************************************************/
/*  test_struct.h                                                         */
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

#ifndef TEST_STRUCT_H
#define TEST_STRUCT_H

#include "core/variant/array.h"
#include "core/variant/struct.h"
#include "core/variant/struct_generator.h"
#include "core/variant/variant.h"
#include "scene/main/node.h"
#include "tests/test_macros.h"
#include "tests/test_tools.h"

// TODO: methods to structify:
/*
 * ClassDB::class_get_signal_list()
 * ClassDB::class_get_property_list()
 * ClassDB::class_get_method_list()
 * CodeEdit::_filter_code_completion_candidates()
 * CodeEdit::get_code_completion_options()
 * DisplayServer::tts_get_voices()
 * EditorExportPlugin::_get_export_options()
 * EditorImportPlugin::_get_import_options()
 * ScriptLanguage::CodeCompletionOption
 * ResourceImporter::ImportOption
 * EditorVCSInterface::DiffLine
 * EditorVCSInterface::DiffHunk
 * EditorVCSInterface::DiffFile
 * EditorVCSInterface::Commit
 * EditorVCSInterface::StatusFile
 * EditorVCSInterface::_get_modified_files_data()
 * EditorVCSInterface::_get_diff()
 * EditorVCSInterface::_get_previous_commits()
 * EditorVCSInterface::_get_line_diff()
 * EditorVCSInterface::add_diff_hunks_into_diff_file()
 * EditorVCSInterface::add_line_diffs_into_diff_hunk()
 * Engine::get_author_info()
 * Engine::get_copyright_info()
 * Engine::get_donor_info()
 * Engine::get_version_info()
 * Font::get_opentype_features()
 * Font::get_ot_name_strings()
 * Font::get_supported_variation_list()
 * Font::find_variation()
 * Font::get_opentype_feature_overrides()
 * GLTFCamera::to_dictionary()
 * GLTFCamera::from_dictionary()
 * GLTFLight::to_dictionary()
 * GLTFLight::from_dictionary()
 * GLTFPhysicsBody::to_dictionary()
 * GLTFPhysicsBody::from_dictionary()
 * GLTFPhysicsShape::to_dictionary()
 * GLTFPhysicsShape::from_dictionary()
 * Geometry2D::make_atlas()
 * GraphEdit::get_connection_list()
 * Image::compute_image_metrics()
 * Image::data
 * IP::get_local_interfaces()
 * LightmapGIData::probe_data
 * NavigationAgent2D::waypoint_reached()
 * NavigationAgent2D::link_reached()
 * NavigationAgent3D::waypoint_reached()
 * NavigationAgent3D::link_reached()
 * Object::get_property_list()
 * Object::get_method_list()
 * Object::get_signal_list()
 * Object::get_signal_connection_list()
 * Object::get_incoming_connections()
 * OS::get_memory_info()
 * PhysicsDirectSpaceState2D::get_rest_info()
 * PhysicsDirectSpaceState2D::intersect_point()
 * PhysicsDirectSpaceState2D::intersect_ray()
 * PhysicsDirectSpaceState2D::intersect_shape()
 * PhysicsDirectSpaceState3D::get_rest_info()
 * PhysicsDirectSpaceState3D::intersect_point()
 * PhysicsDirectSpaceState3D::intersect_ray()
 * PhysicsDirectSpaceState3D::intersect_shape()
 * PolygonPathFinder::data
 * ProjectSettings::add_property_info()
 * ProjectSettings::get_global_class_list()
 * RenderingServer::mesh_add_surface()
 * RenderingServer::mesh_get_surface()
 * RenderingServer::get_shader_parameter_list()
 * RenderingServer::mesh_create_from_surfaces()
 * RenderingServer::instance_geometry_get_shader_parameter_list()
 * Script::get_script_property_list()
 * Script::get_script_method_list()
 * Script::get_script_signal_list()
 * ScriptLanguage::_get_method_info()
 * ScriptLanguageExtension::_validate()
 * ScriptLanguageExtension::_complete_code()
 * ScriptLanguageExtension::_lookup_code()
 * ScriptLanguageExtension::_debug_get_current_stack_info()
 * ScriptLanguageExtension::_get_public_functions()
 * ScriptLanguageExtension::_get_public_annotations()
 * ScriptLanguageExtension::_debug_get_stack_level_locals()
 * ScriptLanguageExtension::_debug_get_stack_level_members()
 * ScriptLanguageExtension::_debug_get_globals()
 * ScriptLanguageExtension::_get_built_in_templates()
 * ScriptLanguageExtension::_get_public_constants()
 * ScriptLanguageExtension::_get_public_annotations()
 * ScriptLanguageExtension::_get_global_class_name()
 * ScriptExtension::_get_documentation()
 * ScriptExtension::_get_script_signal_list()
 * ScriptExtension::_get_script_method_list()
 * ScriptExtension::_get_script_property_list()
 * TextServer::font_set_variation_coordinates()
 * TextServer::font_get_variation_coordinates()
 * TextServer::font_get_glyph_contours()
 * TextServer::font_set_opentype_feature_overrides()
 * TextServer::font_get_opentype_feature_overrides()
 * TextServer::font_supported_feature_list()
 * TextServer::font_supported_variation_list()
 * TextServer::shaped_text_add_string()
 * TextServer::shaped_set_span_update_font()
 * TextServer::shaped_text_get_carets()
 * TextServer::shaped_text_get_glyphs()
 * TextServer::shaped_text_sort_logical()
 * TextServer::shaped_text_get_ellipsis_glyphs()
 * TextServerExtension::_font_set_variation_coordinates()
 * TextServerExtension::_font_get_variation_coordinates()
 * TextServerExtension::_font_get_glyph_contours()
 * TextServerExtension::_font_set_opentype_feature_overrides()
 * TextServerExtension::_font_get_opentype_feature_overrides()
 * TextServerExtension::_font_supported_feature_list()
 * TextServerExtension::_font_supported_variation_list()
 * TextServerExtension::_shaped_text_add_string()
 * TextServerExtension::_shaped_set_span_update_font()
 * TextServerManager::get_interfaces()
 * Time::get_datetime_dict_from_unix_time()
 * Time::get_date_dict_from_unix_time()
 * Time::get_time_dict_from_unix_time()
 * Time::get_datetime_dict_from_datetime_string()
 * Time::get_datetime_string_from_datetime_dict()
 * Time::get_unix_time_from_datetime_dict()
 * Time::get_datetime_dict_from_system()
 * Time::get_date_dict_from_system()
 * Time::get_time_dict_from_system()
 * Time::get_time_zone_from_system()
 * Tree::get_range_config()
 * VisualShader::get_node_connections()
 * WebRTCMultiplayerPeer::get_peer()
 * WebRTCMultiplayerPeer::get_peers()
 * WebRTCPeerConnection::initialize()
 * WebRTCPeerConnection::create_data_channel()
 * WebRTCPeerConnectionExtension::_initialize()
 * WebRTCPeerConnectionExtension::_create_data_channel()
 * XRInterface::get_system_info()
 * XRInterfaceExtension::_get_system_info()
 * XRServer::get_interfaces()
 * XRServer::get_trackers()
 *
 *
 * */

namespace TestStruct {

//TEST_CASE("[Struct] PropertyInfo") {
//	Node *my_node = memnew(Node);
//
//	List<PropertyInfo> list;
//	my_node->get_property_list(&list);
//	PropertyInfo info = list.get(0);
//
//	TypedArray<Struct<PropertyInfo>> property_list = my_node->call(SNAME("get_property_list_as_structs"));
//	Struct<PropertyInfo> prop = property_list[0];
//
//	SUBCASE("Equality") {
//		CHECK_EQ(info.name, prop.get_member<struct PropertyInfo::name>());
//		CHECK_EQ(info.class_name, prop.get_member<struct PropertyInfo::class_name>());
//		CHECK_EQ(info.type, prop.get_member<struct PropertyInfo::type>());
//		CHECK_EQ(info.hint, prop.get_member<struct PropertyInfo::hint>());
//		CHECK_EQ(info.hint_string, prop.get_member<struct PropertyInfo::hint_string>());
//		CHECK_EQ(info.usage, prop.get_member<struct PropertyInfo::usage>());
//	}
//
//	SUBCASE("Duplication") {
//		Variant var = prop;
//		CHECK_EQ(var.get_type(), Variant::ARRAY);
//		Variant var_dup = prop.duplicate();
//		CHECK_EQ(var_dup.get_type(), Variant::ARRAY);
//		CHECK_EQ(var, var_dup);
//	}
//
//	SUBCASE("Setget Named") {
//		Variant variant_prop = prop;
//		bool valid = false;
//		variant_prop.set_named(SNAME("name"), SNAME("Changed"), valid);
//		CHECK_EQ(valid, true);
//		Variant val = variant_prop.get_named(SNAME("name"), valid);
//		CHECK_EQ(valid, true);
//		CHECK_EQ((StringName)val, SNAME("Changed"));
//
//		val = variant_prop.get_named(SNAME("oops"), valid);
//		CHECK_EQ(valid, false);
//		CHECK_EQ(val, Variant());
//
//		variant_prop.set_named(SNAME("oops"), SNAME("oh no"), valid);
//		CHECK_EQ(valid, false);
//	}
//	memdelete(my_node);
//}

TEST_CASE("[Struct] StructInfo") {
	StructInfo info_as_c_struct = StructInfo::get_struct_info();
	Struct<StructInfo> info_as_godot_struct = info_as_c_struct;

	SUBCASE("Equality") {
		CHECK_EQ(info_as_c_struct.name, info_as_godot_struct.get_member<struct StructInfo::name>());
		CHECK_EQ(info_as_c_struct.count, info_as_godot_struct.get_member<struct StructInfo::count>());
		CHECK_EQ(info_as_c_struct.names, info_as_godot_struct.get_member<struct StructInfo::names>());
		CHECK_EQ(info_as_c_struct.types, info_as_godot_struct.get_member<struct StructInfo::types>());
		CHECK_EQ(info_as_c_struct.type_names, info_as_godot_struct.get_member<struct StructInfo::type_names>());
		CHECK_EQ(info_as_c_struct.default_values, info_as_godot_struct.get_member<struct StructInfo::default_values>());
	}

	SUBCASE("Duplication") {
		Variant var = info_as_godot_struct;
		CHECK_EQ(var.get_type(), Variant::ARRAY);
		Variant var_dup = info_as_godot_struct.duplicate();
		CHECK_EQ(var_dup.get_type(), Variant::ARRAY);
		CHECK_EQ(var, var_dup);
	}

	SUBCASE("Setget Named") {
		Variant var = info_as_godot_struct;
		bool valid = false;
		var.set_named(SNAME("name"), SNAME("Changed"), valid);
		CHECK_EQ(valid, true);
		Variant val = var.get_named(SNAME("name"), valid);
		CHECK_EQ(valid, true);
		CHECK_EQ((StringName)val, SNAME("Changed"));

		val = var.get_named(SNAME("oops"), valid);
		CHECK_EQ(valid, false);
		CHECK_EQ(val, Variant());

		var.set_named(SNAME("oops"), SNAME("oh no"), valid);
		CHECK_EQ(valid, false);
	}
}

TEST_CASE("[Struct] Validation") {
	struct NamedInt {
		STRUCT_DECLARE(NamedInt);
		STRUCT_MEMBER(String, name, String());
		STRUCT_MEMBER(int, value, 0);
		STRUCT_LAYOUT(TestStruct, NamedInt, struct name, struct value);
	};

	Struct<NamedInt> named_int;
	named_int["name"] = "Godot";
	named_int["value"] = 4;
	CHECK_EQ(((Variant)named_int).stringify(), "[name: \"Godot\", value: 4]");

	SUBCASE("Self Equal") {
		CHECK(named_int.is_same_typed(named_int));
		Variant variant_named_int = named_int;
		CHECK(named_int.is_same_typed(variant_named_int));
		Struct<NamedInt> same_named_int = variant_named_int;
		CHECK_EQ(named_int, same_named_int);
	}

	SUBCASE("Assignment") {
		Struct<NamedInt> a_match = named_int;
		CHECK_EQ(named_int, a_match);
		Array not_a_match;

		ERR_PRINT_OFF;
		named_int.set_named("name", 4);
		CHECK_MESSAGE(named_int["name"] == "Godot", "assigned an int to a string member");

		named_int.set_named("value", "Godot");
		CHECK_MESSAGE((int)named_int["value"] == 4, "assigned a string to an int member");

		named_int = not_a_match;
		CHECK_MESSAGE(named_int != not_a_match, "assigned an empty array to a struct");

		not_a_match.resize(2);
		named_int.assign(not_a_match);
		CHECK_MESSAGE(named_int != not_a_match, "assigned an array with the wrong size");

		not_a_match[0] = 4;
		not_a_match[1] = "Godot";
		named_int.assign(not_a_match);
		CHECK_MESSAGE(named_int != not_a_match, "assigned an array with mismatched types");

		Array also_a_match;
		also_a_match.resize(2);
		also_a_match[0] = "Godooot";
		also_a_match[1] = 5;
		named_int = not_a_match;
		CHECK_MESSAGE(named_int != not_a_match, "assigned a non-struct to a struct using '=' operator");
		ERR_PRINT_ON;

		named_int.assign(also_a_match);
		CHECK_MESSAGE(named_int == also_a_match, "failed to assign an array with correct types using 'assign' function");
	}
}

TEST_CASE("[Struct] Nesting") {
	struct BasicStruct {
		STRUCT_DECLARE(BasicStruct);
		STRUCT_MEMBER(int, int_val, 4);
		STRUCT_MEMBER(float, float_val, 5.5f);
		STRUCT_LAYOUT(TestStruct, BasicStruct, struct int_val, struct float_val);
		BasicStruct() {}
	};
	struct BasicStructLookalike {
		STRUCT_DECLARE(BasicStructLookalike);
		STRUCT_MEMBER(int, int_val, 4);
		STRUCT_MEMBER(float, float_val, 5.5f);
		STRUCT_LAYOUT(TestStruct, BasicStructLookalike, struct int_val, struct float_val);
	};
	struct NestedStruct {
		STRUCT_DECLARE(NestedStruct);
		STRUCT_MEMBER_CLASS_POINTER(Node, node, nullptr);
		STRUCT_MEMBER_STRUCT(BasicStruct, value, BasicStruct());
		STRUCT_LAYOUT(TestStruct, NestedStruct, struct node, struct value);
	};

	REQUIRE_EQ(NestedStruct::Layout::struct_member_count, 2);

	Struct<BasicStruct> basic_struct;
	Struct<BasicStructLookalike> basic_struct_lookalike;
	Struct<NestedStruct> nested_struct;

	SUBCASE("Defaults") {
		CHECK_EQ(basic_struct.get_member<struct BasicStruct::int_val>(), 4);
		CHECK_EQ(basic_struct.get_member<struct BasicStruct::float_val>(), 5.5);

		CHECK_EQ(nested_struct.get_member<struct NestedStruct::node>(), nullptr);
		CHECK_EQ(Struct<BasicStruct>(nested_struct.get_member<struct NestedStruct::value>()), basic_struct);
	}

	SUBCASE("Assignment") {
		basic_struct.set_member<struct BasicStruct::int_val>(1);
		basic_struct.set_member<struct BasicStruct::float_val>(3.14);

		basic_struct_lookalike.set_member<struct BasicStructLookalike::int_val>(2);
		basic_struct_lookalike.set_member<struct BasicStructLookalike::float_val>(2.7);

		Node *node = memnew(Node);
		nested_struct.set_member<struct NestedStruct::node>(node);
		nested_struct.set_member<struct NestedStruct::value>(basic_struct);

		CHECK_EQ(nested_struct.get_member<struct NestedStruct::node>(), node);
		Struct<BasicStruct> basic_struct_match = nested_struct.get_member<struct NestedStruct::value>();
		CHECK_EQ(basic_struct_match, basic_struct);
		CHECK_EQ(basic_struct_match.get_member<struct BasicStruct::int_val>(), basic_struct.get_member<struct BasicStruct::int_val>());
		CHECK_EQ(basic_struct_match.get_member<struct BasicStruct::float_val>(), basic_struct.get_member<struct BasicStruct::float_val>());

		ERR_PRINT_OFF;
		nested_struct.set_named("value", basic_struct_lookalike);
		CHECK_EQ(nested_struct["value"], basic_struct);
		ERR_PRINT_ON;

		memdelete(node);
	}

	SUBCASE("Typed Array of Struct") {
		TypedArray<Struct<BasicStruct>> array;
		Struct<BasicStruct> basic_struct_0;
		basic_struct_0["int_val"] = 1;
		basic_struct_0["float_val"] = 3.14;
		Struct<BasicStruct> basic_struct_1;
		basic_struct_1["int_val"] = 2;
		basic_struct_1["float_val"] = 2.7;
		array.push_back(basic_struct_0);
		array.push_back(basic_struct_1);
		CHECK_EQ(array[0], basic_struct_0);
		CHECK_EQ(array[1], basic_struct_1);

		ERR_PRINT_OFF;
		array.push_back(0);
		CHECK_EQ(array.size(), 2);

		basic_struct_lookalike["int_val"] = 3;
		basic_struct_lookalike["float_val"] = 5.4;
		array.push_back(basic_struct_lookalike);
		CHECK_EQ(array.size(), 2);
		ERR_PRINT_ON;

		TypedArray<Struct<BasicStruct>> array_of_defaults;
		array_of_defaults.resize(2);
		CHECK_EQ(array_of_defaults[0], Variant(Struct<BasicStruct>()));
		CHECK_EQ(array_of_defaults[1], Variant(Struct<BasicStruct>()));
	}
}

TEST_CASE("[Struct] ClassDB") {
	const StructInfo *struct_info = ::ClassDB::get_struct_info(SNAME("Object"), SNAME("PropertyInfo"));
	REQUIRE(struct_info);
	CHECK_EQ(struct_info->count, 6);
	CHECK_EQ(struct_info->name, "Object.PropertyInfo");
	CHECK_EQ(struct_info->names[3], "hint");
	CHECK_EQ(struct_info->types[3], Variant::INT);
	CHECK_EQ(struct_info->type_names[3], "");
	CHECK_EQ((int)struct_info->default_values[3], PROPERTY_HINT_NONE);
}

} // namespace TestStruct

#endif // TEST_STRUCT_H
