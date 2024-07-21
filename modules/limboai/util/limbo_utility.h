/**
 * limbo_utility.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef LIMBO_UTILITY_H
#define LIMBO_UTILITY_H

#include "limbo_compat.h"

#ifdef LIMBOAI_MODULE
#include "core/object/object.h"

#include "core/input/shortcut.h"
#include "core/object/class_db.h"
#include "core/variant/binder_common.h"
#include "core/variant/variant.h"
#include "scene/resources/texture.h"
#endif // LIMBOAI_MODULE

#ifdef LIMBOAI_GDEXTENSION
#include <godot_cpp/classes/object.hpp>
#include <godot_cpp/classes/shortcut.hpp>
#include <godot_cpp/classes/texture2d.hpp>
#include <godot_cpp/core/binder_common.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/templates/hash_map.hpp>
using namespace godot;
#endif // LIMBOAI_GDEXTENSION

#define LOGICAL_XOR(a, b) (a) ? !(b) : (b)

class LimboUtility : public Object {
	GDCLASS(LimboUtility, Object);

private:
#ifdef TOOLS_ENABLED
	HashMap<String, Ref<Shortcut>> shortcuts;
#endif // TOOLS_ENABLED

public:
	enum CheckType : unsigned int {
		CHECK_EQUAL,
		CHECK_LESS_THAN,
		CHECK_LESS_THAN_OR_EQUAL,
		CHECK_GREATER_THAN,
		CHECK_GREATER_THAN_OR_EQUAL,
		CHECK_NOT_EQUAL
	};

	enum Operation {
		OPERATION_NONE,
		OPERATION_ADDITION,
		OPERATION_SUBTRACTION,
		OPERATION_MULTIPLICATION,
		OPERATION_DIVISION,
		OPERATION_MODULO,
		OPERATION_POWER,
		OPERATION_BIT_SHIFT_LEFT,
		OPERATION_BIT_SHIFT_RIGHT,
		OPERATION_BIT_AND,
		OPERATION_BIT_OR,
		OPERATION_BIT_XOR,
	};

protected:
	static LimboUtility *singleton;
	static void _bind_methods();

public:
	static LimboUtility *get_singleton();

	String decorate_var(String p_variable) const;
	String decorate_output_var(String p_variable) const;
	String get_status_name(int p_status) const;
	Ref<Texture2D> get_task_icon(String p_class_or_script_path) const;

	String get_check_operator_string(CheckType p_check_type) const;
	bool perform_check(CheckType p_check_type, const Variant &left_value, const Variant &right_value);

	String get_operation_string(Operation p_operation) const;
	Variant perform_operation(Operation p_operation, const Variant &left_value, const Variant &right_value);

	String get_property_hint_text(PropertyHint p_hint) const;
	PackedInt32Array get_property_hints_allowed_for_type(Variant::Type p_type) const;

#ifdef TOOLS_ENABLED
	Ref<Shortcut> add_shortcut(const String &p_path, const String &p_name, Key p_keycode = LW_KEY(NONE));
	bool is_shortcut(const String &p_path, const Ref<InputEvent> &p_event) const;
	Ref<Shortcut> get_shortcut(const String &p_path) const;

	void open_doc_introduction();
	void open_doc_online();
	void open_doc_gdextension_limitations();
	void open_doc_custom_tasks();
	void open_doc_class(const String &p_class_name);
#endif // TOOLS_ENABLED

	LimboUtility();
	~LimboUtility();
};

VARIANT_ENUM_CAST(LimboUtility::CheckType);
VARIANT_ENUM_CAST(LimboUtility::Operation);

#define LW_SHORTCUT(m_path, m_name, m_keycode) (LimboUtility::get_singleton()->add_shortcut(m_path, m_name, m_keycode))
#define LW_IS_SHORTCUT(m_path, m_event) (LimboUtility::get_singleton()->is_shortcut(m_path, m_event))
#define LW_GET_SHORTCUT(m_path) (LimboUtility::get_singleton()->get_shortcut(m_path))

#endif // LIMBO_UTILITY_H
