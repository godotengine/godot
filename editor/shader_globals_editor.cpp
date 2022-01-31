/*************************************************************************/
/*  shader_globals_editor.cpp                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "shader_globals_editor.h"
#include "editor_node.h"
#include "servers/rendering/shader_language.h"

static const char *global_var_type_names[RS::GLOBAL_VAR_TYPE_MAX] = {
	"bool",
	"bvec2",
	"bvec3",
	"bvec4",
	"int",
	"ivec2",
	"ivec3",
	"ivec4",
	"rect2i",
	"uint",
	"uvec2",
	"uvec3",
	"uvec4",
	"float",
	"vec2",
	"vec3",
	"vec4",
	"color",
	"rect2",
	"mat2",
	"mat3",
	"mat4",
	"transform_2d",
	"transform",
	"sampler2D",
	"sampler2DArray",
	"sampler3D",
	"samplerCube",
};

class ShaderGlobalsEditorInterface : public Object {
	GDCLASS(ShaderGlobalsEditorInterface, Object)

	void _var_changed() {
		emit_signal(SNAME("var_changed"));
	}

protected:
	static void _bind_methods() {
		ClassDB::bind_method("_var_changed", &ShaderGlobalsEditorInterface::_var_changed);
		ADD_SIGNAL(MethodInfo("var_changed"));
	}

	bool _set(const StringName &p_name, const Variant &p_value) {
		Variant existing = RS::get_singleton()->global_variable_get(p_name);

		if (existing.get_type() == Variant::NIL) {
			return false;
		}

		UndoRedo *undo_redo = EditorNode::get_singleton()->get_undo_redo();

		undo_redo->create_action(TTR("Set Shader Global Variable"));
		undo_redo->add_do_method(RS::get_singleton(), "global_variable_set", p_name, p_value);
		undo_redo->add_undo_method(RS::get_singleton(), "global_variable_set", p_name, existing);
		RS::GlobalVariableType type = RS::get_singleton()->global_variable_get_type(p_name);
		Dictionary gv;
		gv["type"] = global_var_type_names[type];
		if (type >= RS::GLOBAL_VAR_TYPE_SAMPLER2D) {
			RES res = p_value;
			if (res.is_valid()) {
				gv["value"] = res->get_path();
			} else {
				gv["value"] = "";
			}
		} else {
			gv["value"] = p_value;
		}

		String path = "shader_globals/" + String(p_name);
		undo_redo->add_do_property(ProjectSettings::get_singleton(), path, gv);
		undo_redo->add_undo_property(ProjectSettings::get_singleton(), path, ProjectSettings::get_singleton()->get(path));
		undo_redo->add_do_method(this, "_var_changed");
		undo_redo->add_undo_method(this, "_var_changed");
		block_update = true;
		undo_redo->commit_action();
		block_update = false;

		return true;
	}

	bool _get(const StringName &p_name, Variant &r_ret) const {
		r_ret = RS::get_singleton()->global_variable_get(p_name);
		return r_ret.get_type() != Variant::NIL;
	}
	void _get_property_list(List<PropertyInfo> *p_list) const {
		Vector<StringName> variables;
		variables = RS::get_singleton()->global_variable_get_list();
		for (int i = 0; i < variables.size(); i++) {
			PropertyInfo pinfo;
			pinfo.name = variables[i];

			switch (RS::get_singleton()->global_variable_get_type(variables[i])) {
				case RS::GLOBAL_VAR_TYPE_BOOL: {
					pinfo.type = Variant::BOOL;
				} break;
				case RS::GLOBAL_VAR_TYPE_BVEC2: {
					pinfo.type = Variant::INT;
					pinfo.hint = PROPERTY_HINT_FLAGS;
					pinfo.hint_string = "x,y";
				} break;
				case RS::GLOBAL_VAR_TYPE_BVEC3: {
					pinfo.type = Variant::INT;
					pinfo.hint = PROPERTY_HINT_FLAGS;
					pinfo.hint_string = "x,y,z";
				} break;
				case RS::GLOBAL_VAR_TYPE_BVEC4: {
					pinfo.type = Variant::INT;
					pinfo.hint = PROPERTY_HINT_FLAGS;
					pinfo.hint_string = "x,y,z,w";
				} break;
				case RS::GLOBAL_VAR_TYPE_INT: {
					pinfo.type = Variant::INT;
				} break;
				case RS::GLOBAL_VAR_TYPE_IVEC2: {
					pinfo.type = Variant::VECTOR2I;
				} break;
				case RS::GLOBAL_VAR_TYPE_IVEC3: {
					pinfo.type = Variant::VECTOR3I;
				} break;
				case RS::GLOBAL_VAR_TYPE_IVEC4: {
					pinfo.type = Variant::PACKED_INT32_ARRAY;
				} break;
				case RS::GLOBAL_VAR_TYPE_RECT2I: {
					pinfo.type = Variant::RECT2I;
				} break;
				case RS::GLOBAL_VAR_TYPE_UINT: {
					pinfo.type = Variant::INT;
				} break;
				case RS::GLOBAL_VAR_TYPE_UVEC2: {
					pinfo.type = Variant::VECTOR2I;
				} break;
				case RS::GLOBAL_VAR_TYPE_UVEC3: {
					pinfo.type = Variant::VECTOR3I;
				} break;
				case RS::GLOBAL_VAR_TYPE_UVEC4: {
					pinfo.type = Variant::PACKED_INT32_ARRAY;
				} break;
				case RS::GLOBAL_VAR_TYPE_FLOAT: {
					pinfo.type = Variant::FLOAT;
				} break;
				case RS::GLOBAL_VAR_TYPE_VEC2: {
					pinfo.type = Variant::VECTOR2;
				} break;
				case RS::GLOBAL_VAR_TYPE_VEC3: {
					pinfo.type = Variant::VECTOR3;
				} break;
				case RS::GLOBAL_VAR_TYPE_VEC4: {
					pinfo.type = Variant::PLANE;
				} break;
				case RS::GLOBAL_VAR_TYPE_RECT2: {
					pinfo.type = Variant::RECT2;
				} break;
				case RS::GLOBAL_VAR_TYPE_COLOR: {
					pinfo.type = Variant::COLOR;
				} break;
				case RS::GLOBAL_VAR_TYPE_MAT2: {
					pinfo.type = Variant::PACKED_INT32_ARRAY;
				} break;
				case RS::GLOBAL_VAR_TYPE_MAT3: {
					pinfo.type = Variant::BASIS;
				} break;
				case RS::GLOBAL_VAR_TYPE_TRANSFORM_2D: {
					pinfo.type = Variant::TRANSFORM2D;
				} break;
				case RS::GLOBAL_VAR_TYPE_TRANSFORM: {
					pinfo.type = Variant::TRANSFORM3D;
				} break;
				case RS::GLOBAL_VAR_TYPE_MAT4: {
					pinfo.type = Variant::PACKED_INT32_ARRAY;
				} break;
				case RS::GLOBAL_VAR_TYPE_SAMPLER2D: {
					pinfo.type = Variant::OBJECT;
					pinfo.hint = PROPERTY_HINT_RESOURCE_TYPE;
					pinfo.hint_string = "Texture2D";
				} break;
				case RS::GLOBAL_VAR_TYPE_SAMPLER2DARRAY: {
					pinfo.type = Variant::OBJECT;
					pinfo.hint = PROPERTY_HINT_RESOURCE_TYPE;
					pinfo.hint_string = "Texture2DArray";
				} break;
				case RS::GLOBAL_VAR_TYPE_SAMPLER3D: {
					pinfo.type = Variant::OBJECT;
					pinfo.hint = PROPERTY_HINT_RESOURCE_TYPE;
					pinfo.hint_string = "Texture3D";
				} break;
				case RS::GLOBAL_VAR_TYPE_SAMPLERCUBE: {
					pinfo.type = Variant::OBJECT;
					pinfo.hint = PROPERTY_HINT_RESOURCE_TYPE;
					pinfo.hint_string = "Cubemap";
				} break;
				default: {
				} break;
			}

			p_list->push_back(pinfo);
		}
	}

public:
	bool block_update = false;

	ShaderGlobalsEditorInterface() {
	}
};

static Variant create_var(RS::GlobalVariableType p_type) {
	switch (p_type) {
		case RS::GLOBAL_VAR_TYPE_BOOL: {
			return false;
		}
		case RS::GLOBAL_VAR_TYPE_BVEC2: {
			return 0; //bits
		}
		case RS::GLOBAL_VAR_TYPE_BVEC3: {
			return 0; //bits
		}
		case RS::GLOBAL_VAR_TYPE_BVEC4: {
			return 0; //bits
		}
		case RS::GLOBAL_VAR_TYPE_INT: {
			return 0; //bits
		}
		case RS::GLOBAL_VAR_TYPE_IVEC2: {
			return Vector2i();
		}
		case RS::GLOBAL_VAR_TYPE_IVEC3: {
			return Vector3i();
		}
		case RS::GLOBAL_VAR_TYPE_IVEC4: {
			Vector<int> v4;
			v4.resize(4);
			v4.write[0] = 0;
			v4.write[1] = 0;
			v4.write[2] = 0;
			v4.write[3] = 0;
			return v4;
		}
		case RS::GLOBAL_VAR_TYPE_RECT2I: {
			return Rect2i();
		}
		case RS::GLOBAL_VAR_TYPE_UINT: {
			return 0;
		}
		case RS::GLOBAL_VAR_TYPE_UVEC2: {
			return Vector2i();
		}
		case RS::GLOBAL_VAR_TYPE_UVEC3: {
			return Vector3i();
		}
		case RS::GLOBAL_VAR_TYPE_UVEC4: {
			Vector<int> v4;
			v4.resize(4);
			v4.write[0] = 0;
			v4.write[1] = 0;
			v4.write[2] = 0;
			v4.write[3] = 0;
			return v4;
		}
		case RS::GLOBAL_VAR_TYPE_FLOAT: {
			return 0.0;
		}
		case RS::GLOBAL_VAR_TYPE_VEC2: {
			return Vector2();
		}
		case RS::GLOBAL_VAR_TYPE_VEC3: {
			return Vector3();
		}
		case RS::GLOBAL_VAR_TYPE_VEC4: {
			return Plane();
		}
		case RS::GLOBAL_VAR_TYPE_RECT2: {
			return Rect2();
		}
		case RS::GLOBAL_VAR_TYPE_COLOR: {
			return Color();
		}
		case RS::GLOBAL_VAR_TYPE_MAT2: {
			Vector<real_t> xform;
			xform.resize(4);
			xform.write[0] = 1;
			xform.write[1] = 0;
			xform.write[2] = 0;
			xform.write[3] = 1;
			return xform;
		}
		case RS::GLOBAL_VAR_TYPE_MAT3: {
			return Basis();
		}
		case RS::GLOBAL_VAR_TYPE_TRANSFORM_2D: {
			return Transform2D();
		}
		case RS::GLOBAL_VAR_TYPE_TRANSFORM: {
			return Transform3D();
		}
		case RS::GLOBAL_VAR_TYPE_MAT4: {
			Vector<real_t> xform;
			xform.resize(16);
			xform.write[0] = 1;
			xform.write[1] = 0;
			xform.write[2] = 0;
			xform.write[3] = 0;

			xform.write[4] = 0;
			xform.write[5] = 1;
			xform.write[6] = 0;
			xform.write[7] = 0;

			xform.write[8] = 0;
			xform.write[9] = 0;
			xform.write[10] = 1;
			xform.write[11] = 0;

			xform.write[12] = 0;
			xform.write[13] = 0;
			xform.write[14] = 0;
			xform.write[15] = 1;

			return xform;
		}
		case RS::GLOBAL_VAR_TYPE_SAMPLER2D: {
			return "";
		}
		case RS::GLOBAL_VAR_TYPE_SAMPLER2DARRAY: {
			return "";
		}
		case RS::GLOBAL_VAR_TYPE_SAMPLER3D: {
			return "";
		}
		case RS::GLOBAL_VAR_TYPE_SAMPLERCUBE: {
			return "";
		}
		default: {
			return Variant();
		}
	}
}

void ShaderGlobalsEditor::_variable_added() {
	String var = variable_name->get_text().strip_edges();
	if (var.is_empty() || !var.is_valid_identifier()) {
		EditorNode::get_singleton()->show_warning(TTR("Please specify a valid variable identifier name."));
		return;
	}

	if (RenderingServer::get_singleton()->global_variable_get(var).get_type() != Variant::NIL) {
		EditorNode::get_singleton()->show_warning(vformat(TTR("Global variable '%s' already exists'"), var));
		return;
	}

	List<String> keywords;
	ShaderLanguage::get_keyword_list(&keywords);

	if (keywords.find(var) != nullptr || var == "script") {
		EditorNode::get_singleton()->show_warning(vformat(TTR("Name '%s' is a reserved shader language keyword."), var));
		return;
	}

	UndoRedo *undo_redo = EditorNode::get_singleton()->get_undo_redo();

	Variant value = create_var(RS::GlobalVariableType(variable_type->get_selected()));

	undo_redo->create_action(TTR("Add Shader Global Variable"));
	undo_redo->add_do_method(RS::get_singleton(), "global_variable_add", var, RS::GlobalVariableType(variable_type->get_selected()), value);
	undo_redo->add_undo_method(RS::get_singleton(), "global_variable_remove", var);
	Dictionary gv;
	gv["type"] = global_var_type_names[variable_type->get_selected()];
	gv["value"] = value;

	undo_redo->add_do_property(ProjectSettings::get_singleton(), "shader_globals/" + var, gv);
	undo_redo->add_undo_property(ProjectSettings::get_singleton(), "shader_globals/" + var, Variant());
	undo_redo->add_do_method(this, "_changed");
	undo_redo->add_undo_method(this, "_changed");
	undo_redo->commit_action();
}

void ShaderGlobalsEditor::_variable_deleted(const String &p_variable) {
	UndoRedo *undo_redo = EditorNode::get_singleton()->get_undo_redo();

	undo_redo->create_action(TTR("Add Shader Global Variable"));
	undo_redo->add_do_method(RS::get_singleton(), "global_variable_remove", p_variable);
	undo_redo->add_undo_method(RS::get_singleton(), "global_variable_add", p_variable, RS::get_singleton()->global_variable_get_type(p_variable), RS::get_singleton()->global_variable_get(p_variable));

	undo_redo->add_do_property(ProjectSettings::get_singleton(), "shader_globals/" + p_variable, Variant());
	undo_redo->add_undo_property(ProjectSettings::get_singleton(), "shader_globals/" + p_variable, ProjectSettings::get_singleton()->get("shader_globals/" + p_variable));
	undo_redo->add_do_method(this, "_changed");
	undo_redo->add_undo_method(this, "_changed");
	undo_redo->commit_action();
}

void ShaderGlobalsEditor::_changed() {
	emit_signal(SNAME("globals_changed"));
	if (!interface->block_update) {
		interface->notify_property_list_changed();
	}
}

void ShaderGlobalsEditor::_bind_methods() {
	ClassDB::bind_method("_changed", &ShaderGlobalsEditor::_changed);
	ADD_SIGNAL(MethodInfo("globals_changed"));
}

void ShaderGlobalsEditor::_notification(int p_what) {
	if (p_what == NOTIFICATION_VISIBILITY_CHANGED) {
		if (is_visible_in_tree()) {
			inspector->edit(interface);
		}
	}
	if (p_what == NOTIFICATION_PREDELETE) {
		inspector->edit(nullptr);
	}
}

ShaderGlobalsEditor::ShaderGlobalsEditor() {
	HBoxContainer *add_menu_hb = memnew(HBoxContainer);
	add_child(add_menu_hb);

	add_menu_hb->add_child(memnew(Label(TTR("Name:"))));
	variable_name = memnew(LineEdit);
	variable_name->set_h_size_flags(SIZE_EXPAND_FILL);
	add_menu_hb->add_child(variable_name);

	add_menu_hb->add_child(memnew(Label(TTR("Type:"))));
	variable_type = memnew(OptionButton);
	variable_type->set_h_size_flags(SIZE_EXPAND_FILL);
	add_menu_hb->add_child(variable_type);

	for (int i = 0; i < RS::GLOBAL_VAR_TYPE_MAX; i++) {
		variable_type->add_item(global_var_type_names[i]);
	}

	variable_add = memnew(Button(TTR("Add")));
	add_menu_hb->add_child(variable_add);
	variable_add->connect("pressed", callable_mp(this, &ShaderGlobalsEditor::_variable_added));

	inspector = memnew(EditorInspector);
	inspector->set_v_size_flags(SIZE_EXPAND_FILL);
	add_child(inspector);
	inspector->set_use_wide_editors(true);
	inspector->set_enable_capitalize_paths(false);
	inspector->set_use_deletable_properties(true);
	inspector->connect("property_deleted", callable_mp(this, &ShaderGlobalsEditor::_variable_deleted), varray(), CONNECT_DEFERRED);

	interface = memnew(ShaderGlobalsEditorInterface);
	interface->connect("var_changed", Callable(this, "_changed"));
}

ShaderGlobalsEditor::~ShaderGlobalsEditor() {
	memdelete(interface);
}
