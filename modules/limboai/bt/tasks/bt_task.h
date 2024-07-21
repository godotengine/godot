/**
 * bt_task.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef BT_TASK_H
#define BT_TASK_H

#include "../../blackboard/blackboard.h"
#include "../../util/limbo_compat.h"
#include "../../util/limbo_string_names.h"
#include "../../util/limbo_task_db.h"

#ifdef LIMBOAI_MODULE
#include "core/config/engine.h"
#include "core/error/error_macros.h"
#include "core/io/resource.h"
#include "core/math/math_funcs.h"
#include "core/object/object.h"
#include "core/object/ref_counted.h"
#include "core/os/memory.h"
#include "core/string/ustring.h"
#include "core/templates/vector.h"
#include "core/typedefs.h"
#include "core/variant/array.h"
#include "core/variant/binder_common.h"
#include "core/variant/dictionary.h"
#include "scene/resources/texture.h"
#endif // LIMBOAI_MODULE

#ifdef LIMBOAI_GDEXTENSION
#include <godot_cpp/classes/engine.hpp>
#include <godot_cpp/classes/resource.hpp>
#include <godot_cpp/core/gdvirtual.gen.inc>
#include <godot_cpp/core/object.hpp>
#include <godot_cpp/templates/vector.hpp>
using namespace godot;
#endif // LIMBOAI_GDEXTENSION

class BehaviorTree;

/**
 * Base class for BTTask.
 * Note: In order to properly return Status in the _tick virtual method (GDVIRTUAL1R...)
 * we must do VARIANT_ENUM_CAST for Status enum before the actual BTTask class declaration.
 */
class BT : public Resource {
	GDCLASS(BT, Resource);

public:
	enum Status {
		FRESH,
		RUNNING,
		FAILURE,
		SUCCESS,
	};

protected:
	static void _bind_methods();
};

VARIANT_ENUM_CAST(BT::Status)

class BTTask : public BT {
	GDCLASS(BTTask, BT);

private:
	friend class BehaviorTree;

	// Avoid namespace pollution in the derived classes.
	struct Data {
		int index = -1;
		String custom_name;
		Node *agent = nullptr;
		Node *scene_root = nullptr;
		Ref<Blackboard> blackboard;
		BTTask *parent = nullptr;
		Vector<Ref<BTTask>> children;
		Status status = FRESH;
		double elapsed = 0.0;
		bool display_collapsed = false;
#ifdef TOOLS_ENABLED
		ObjectID behavior_tree_id;
#endif
	} data;

	Array _get_children() const;
	void _set_children(Array children);

	PackedStringArray _get_configuration_warnings(); // ! Scripts only.

protected:
	static void _bind_methods();

	virtual String _generate_name();
	virtual void _setup() {}
	virtual void _enter() {}
	virtual void _exit() {}
	virtual Status _tick(double p_delta) { return FAILURE; }

	GDVIRTUAL0RC(String, _generate_name);
	GDVIRTUAL0(_setup);
	GDVIRTUAL0(_enter);
	GDVIRTUAL0(_exit);
	GDVIRTUAL1R(Status, _tick, double);
	GDVIRTUAL0RC(PackedStringArray, _get_configuration_warnings);

public:
	// TODO: GDExtension doesn't have this method hmm...

#ifdef LIMBOAI_MODULE
	virtual bool editor_can_reload_from_file() override { return false; }
#endif // LIMBOAI_MODULE

	_FORCE_INLINE_ Node *get_agent() const { return data.agent; }
	void set_agent(Node *p_agent) { data.agent = p_agent; }

	_FORCE_INLINE_ Node *get_scene_root() const { return data.scene_root; }

	void set_display_collapsed(bool p_display_collapsed);
	bool is_displayed_collapsed() const;

	String get_custom_name() const { return data.custom_name; }
	void set_custom_name(const String &p_name);
	String get_task_name();

	Ref<BTTask> get_root() const;

	virtual Ref<BTTask> clone() const;
	virtual void initialize(Node *p_agent, const Ref<Blackboard> &p_blackboard, Node *p_scene_root);
	virtual PackedStringArray get_configuration_warnings(); // ! Native version.

	Status execute(double p_delta);
	void abort();

	_FORCE_INLINE_ Ref<BTTask> get_parent() const { return Ref<BTTask>(data.parent); }
	_FORCE_INLINE_ bool is_root() const { return data.parent == nullptr; }
	_FORCE_INLINE_ Ref<Blackboard> get_blackboard() const { return data.blackboard; }
	_FORCE_INLINE_ Status get_status() const { return data.status; }
	_FORCE_INLINE_ double get_elapsed_time() const { return data.elapsed; };

	_FORCE_INLINE_ Ref<BTTask> get_child(int p_idx) const {
		ERR_FAIL_INDEX_V(p_idx, data.children.size(), nullptr);
		return data.children.get(p_idx);
	}

	_FORCE_INLINE_ int get_child_count() const { return data.children.size(); }
	int get_child_count_excluding_comments() const;

	void add_child(Ref<BTTask> p_child);
	void add_child_at_index(Ref<BTTask> p_child, int p_idx);
	void remove_child(Ref<BTTask> p_child);
	void remove_child_at_index(int p_idx);

	_FORCE_INLINE_ bool has_child(const Ref<BTTask> &p_child) const { return data.children.find(p_child) != -1; }
	_FORCE_INLINE_ int get_index() const { return data.index; }

	bool is_descendant_of(const Ref<BTTask> &p_task) const;
	Ref<BTTask> next_sibling() const;

	void print_tree(int p_initial_tabs = 0);

#ifdef TOOLS_ENABLED
	Ref<BehaviorTree> editor_get_behavior_tree();
	void editor_set_behavior_tree(const Ref<BehaviorTree> &p_bt);
#endif

	BTTask();
	~BTTask();
};

#endif // BT_TASK_H
