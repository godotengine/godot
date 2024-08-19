/**
 * blackboard.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef BLACKBOARD_H
#define BLACKBOARD_H

#include "bb_variable.h"

#ifdef LIMBOAI_MODULE
#include "core/object/object.h"
#include "core/object/ref_counted.h"
#include "core/variant/variant.h"
#include "scene/main/node.h"
#endif // LIMBOAI_MODULE

#ifdef LIMBOAI_GDEXTENSION
#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/core/object.hpp>
#include <godot_cpp/templates/hash_map.hpp>
using namespace godot;
#endif // LIMBOAI_GDEXTENSION

class Blackboard : public RefCounted {
	GDCLASS(Blackboard, RefCounted);

protected:
	Ref<Blackboard> parent;
	Callable changed_value_callback;

protected:
	static void _bind_methods();

public:
	void set_changed_value_callback(const Callable &p_callback) { changed_value_callback = p_callback; }
	
	void set_parent(const Ref<Blackboard> &p_blackboard) { parent = p_blackboard; }
	Ref<Blackboard> get_parent() const { return parent; }

	Ref<Blackboard> top() const;

	virtual Variant get_var(const StringName &p_name, const Variant &p_default = Variant(), bool p_complain = true) const{return Variant();}
	virtual void set_var(const StringName &p_name, const Variant &p_value){}
	virtual bool has_var(const StringName &p_name) const{return false;}
	virtual BBVariable get_bb_var(const StringName& p_name)
	{
		return BBVariable();
	}
	virtual bool has_local_var(const StringName &p_name) const { return false; }
	virtual void erase_var(const StringName &p_name){}
	virtual void clear() {  }
	virtual TypedArray<StringName> list_vars() const{ return TypedArray<StringName>(); }

	virtual Dictionary get_vars_as_dict() const
	{
		return Dictionary();
	}
	virtual void populate_from_dict(const Dictionary &p_dictionary){}

	virtual void bind_var_to_property(const StringName &p_name, Object *p_object, const StringName &p_property, bool p_create = false){}
	virtual void unbind_var(const StringName &p_name){}

	virtual void assign_var(const StringName &p_name, const BBVariable &p_var){}

	virtual void link_var(const StringName &p_name, const Ref<Blackboard> &p_target_blackboard, const StringName &p_target_var, bool p_create = false){}
};
// 运行时的黑板
class BlackboardRuntime : public Blackboard
{
	GDCLASS(BlackboardRuntime, Blackboard);
	static void _bind_methods()
	{

	}
	/* data */
public:
	virtual Variant get_var(const StringName &p_name, const Variant &p_default = Variant(), bool p_complain = true) const  override;
	virtual void set_var(const StringName &p_name, const Variant &p_value) override;
	virtual BBVariable get_bb_var(const StringName& p_name) override
	{
		return data[p_name];
	}
	virtual bool has_var(const StringName &p_name) const  override;
	virtual bool has_local_var(const StringName &p_name) const  override { return data.has(p_name); }
	virtual void erase_var(const StringName &p_name) override;
	virtual void clear()  override { data.clear(); }
	virtual TypedArray<StringName> list_vars() const override;

	virtual Dictionary get_vars_as_dict() const override;
	virtual void populate_from_dict(const Dictionary &p_dictionary) override;

	virtual void bind_var_to_property(const StringName &p_name, Object *p_object, const StringName &p_property, bool p_create = false) override;
	virtual void unbind_var(const StringName &p_name) override;

	virtual void assign_var(const StringName &p_name, const BBVariable &p_var) override;

	virtual void link_var(const StringName &p_name, const Ref<Blackboard> &p_target_blackboard, const StringName &p_target_var, bool p_create = false) override;
protected:
	HashMap<StringName, BBVariable> data;

};

// 编辑器的虚拟黑板
class BlackboardEditorVirtual : public Blackboard {
	GDCLASS(BlackboardEditorVirtual, Blackboard);
	static void _bind_methods()
	{
		
	}
public:
	void init(class BlackboardPlan *p_plan)
	{
		blackboard_plan = p_plan;
	}
	virtual Variant get_var(const StringName &p_name, const Variant &p_default = Variant(), bool p_complain = true) const override;
	virtual void set_var(const StringName &p_name, const Variant &p_value) override;
	virtual bool has_var(const StringName &p_name) const override;
protected:
	class BlackboardPlan*  blackboard_plan = nullptr;
};



#endif // BLACKBOARD_H
