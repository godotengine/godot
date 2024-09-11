/**
 * bb_param.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef BB_PARAM_H
#define BB_PARAM_H

#include "../../blackboard/blackboard.h"
#include "core/io/resource.h"
#include "core/object/script_language.h"



#define ADD_STYLEBOX_OVERRIDE(m_control, m_name, m_stylebox) (m_control->add_theme_style_override(m_name, m_stylebox))
class BBParam : public Resource {
	GDCLASS(BBParam, Resource);

public:
	enum ValueSource : unsigned int {
		SAVED_VALUE,
		BLACKBOARD_VAR
	};

	static String decorate_var(String p_variable);

	static Variant VARIANT_DEFAULT(Variant::Type p_type);
	static PackedInt32Array get_property_hints_allowed_for_type(Variant::Type p_type);
	static String get_property_hint_text(PropertyHint p_hint);
	static Ref<Texture2D> get_task_icon(String p_class_or_script_path) ;
private:
	ValueSource value_source;
	Variant saved_value;
	StringName variable;

	_FORCE_INLINE_ void _update_name() {
		set_name((value_source == SAVED_VALUE) ? String(saved_value) : decorate_var(variable));
	}

protected:
	static void _bind_methods();

	_FORCE_INLINE_ void _assign_default_value() { saved_value = VARIANT_DEFAULT(get_type()); }

	void _get_property_list(List<PropertyInfo> *p_list) const;

public:
	void set_value_source(ValueSource p_value);
	ValueSource get_value_source() const { return value_source; }

	void set_saved_value(Variant p_value);
	Variant get_saved_value();

	void set_variable(const StringName &p_variable);
	StringName get_variable() const { return variable; }

	virtual String to_string() override;

	virtual Variant::Type get_type() const { return Variant::NIL; }
	virtual Variant::Type get_variable_expected_type() const { return get_type(); }
	virtual Variant get_value(Node *p_scene_root, const Ref<Blackboard> &p_blackboard, const Variant &p_default = Variant());

	BBParam();
};

#endif // BB_PARAM_H
