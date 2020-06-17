/*************************************************************************/
/*  visual_script_module_nodes.h                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef VISUAL_SCRIPT_MODULE_NODES_H
#define VISUAL_SCRIPT_MODULE_NODES_H

#include "visual_script.h"
#include "visual_script_nodes.h"

class VisualScriptModuleNode : public VisualScriptNode {
	GDCLASS(VisualScriptModuleNode, VisualScriptNode);

	String module_name;

	struct Port {
		String name;
		Variant::Type type;
	};

	Vector<Port> output_ports;
	Vector<Port> input_ports;

private:
	int update_module_data();

protected:
	static void _bind_methods();

public:
	virtual int get_output_sequence_port_count() const override;
	virtual bool has_input_sequence_port() const override;

	virtual String get_output_sequence_port_text(int p_port) const override;

	virtual int get_input_value_port_count() const override;
	virtual int get_output_value_port_count() const override;

	virtual PropertyInfo get_input_value_port_info(int p_idx) const override;
	virtual PropertyInfo get_output_value_port_info(int p_idx) const override;

	virtual String get_caption() const override;
	virtual String get_text() const override;
	virtual String get_category() const override { return "functions"; }

	void set_module(const String &p_name);
	String get_module_name() const;
	void set_module_by_name(const StringName &p_name);

	virtual VisualScriptNodeInstance *instantiate(VisualScriptInstance *p_instance) override;

	VisualScriptModuleNode();
	~VisualScriptModuleNode();
};

class VisualScriptModuleEntryNode : public VisualScriptLists {
	GDCLASS(VisualScriptModuleEntryNode, VisualScriptLists);

	int stack_size;
	bool stack_less;

public:
	virtual bool is_output_port_editable() const override { return true; }
	virtual bool is_output_port_name_editable() const override { return true; }
	virtual bool is_output_port_type_editable() const override { return true; }

	virtual bool is_input_port_editable() const override { return false; }
	virtual bool is_input_port_name_editable() const override { return false; }
	virtual bool is_input_port_type_editable() const override { return false; }

	virtual int get_output_sequence_port_count() const override { return 1; }
	virtual bool has_input_sequence_port() const override { return false; }

	virtual String get_output_sequence_port_text(int p_port) const override { return ""; }

	virtual int get_input_value_port_count() const override { return 0; }
	virtual int get_output_value_port_count() const override;

	virtual PropertyInfo get_input_value_port_info(int p_idx) const override { return PropertyInfo(); }
	virtual PropertyInfo get_output_value_port_info(int p_idx) const override;

	void set_stack_less(bool p_enable);
	bool is_stack_less() const;
	void set_sequenced(bool p_enable);
	bool is_sequenced() const;
	void set_stack_size(int p_size);
	int get_stack_size() const;

	virtual String get_caption() const override { return "Entry"; }
	virtual String get_text() const override { return ""; }
	virtual String get_category() const override { return "module"; }

	virtual VisualScriptNodeInstance *instantiate(VisualScriptInstance *p_instance) override;

	VisualScriptModuleEntryNode();
	~VisualScriptModuleEntryNode();
};

class VisualScriptModuleExitNode : public VisualScriptLists {
	GDCLASS(VisualScriptModuleExitNode, VisualScriptLists);

	bool with_value;

public:
	virtual bool is_output_port_editable() const override { return false; }
	virtual bool is_output_port_name_editable() const override { return false; }
	virtual bool is_output_port_type_editable() const override { return false; }

	virtual bool is_input_port_editable() const override { return true; }
	virtual bool is_input_port_name_editable() const override { return true; }
	virtual bool is_input_port_type_editable() const override { return true; }

	virtual int get_output_sequence_port_count() const override { return 0; }
	virtual bool has_input_sequence_port() const override { return true; }

	virtual String get_output_sequence_port_text(int p_port) const override { return ""; }

	virtual int get_input_value_port_count() const override;
	virtual int get_output_value_port_count() const override { return 0; }

	virtual PropertyInfo get_input_value_port_info(int p_idx) const override;
	virtual PropertyInfo get_output_value_port_info(int p_idx) const override { return PropertyInfo(); }

	virtual String get_caption() const override { return "Exit"; }
	virtual String get_text() const override { return ""; }
	virtual String get_category() const override { return "module"; }

	virtual VisualScriptNodeInstance *instantiate(VisualScriptInstance *p_instance) override;

	VisualScriptModuleExitNode();
	~VisualScriptModuleExitNode();
};

void register_visual_script_module_nodes();

#endif // VISUAL_SCRIPT_SUBMODULE_NODES_H
