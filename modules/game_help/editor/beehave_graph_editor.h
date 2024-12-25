#pragma once

#include "modules/game_help/logic/beehave/beehave_tree.h"
#include "beehave_graph_node.h"
#include "scene/resources/style_box_flat.h"
#include "scene/resources/image_texture.h"
#include "editor/editor_node.h"
#include "editor/editor_properties.h"
#include "editor/editor_properties_array_dict.h"
#include "editor/editor_resource_picker.h"
#include "editor/editor_resource_picker.h"
#include "editor/inspector_dock.h"
#include "editor/gui/editor_bottom_panel.h"
#include "scene/gui/graph_edit_arranger.h"
#include "scene/gui/view_panner.h"
#include "scene/gui/graph_edit.h"
class BeehaveGraphProperty;

struct BeehaveRunTool
{
	float time_scale = 1.0;
	float delta = 0.0;
	float curr_time = 0.0;

	bool is_running = false;
	bool is_paused = false;

	Ref<BeehaveRuncontext> run_context;
	Ref<BeehaveTree> tree;

	void play(const Ref<BeehaveTree>& p_tree);
	void pause()
	{
		if (tree.is_null())
		{
			return;
		}
		is_paused = true;
	}

	void tick(float p_delta)
	{
		if (tree.is_null())
		{
			return;
		}
		if (!is_running)
		{
			return;
		}
		if (is_paused)
		{
			return;
		}
		delta = time_scale * p_delta;
		curr_time += p_delta;
		run_context->delta = delta;
		run_context->time = curr_time;
		if (tree->process(run_context) != 2)
		{
			is_running = false;
		}
	}

	void stop()
	{
		if (tree.is_valid())
		{
			tree->set_debug_break_node(nullptr);
		}
		is_running = false;
		is_paused = false;
		delta = 0.0f;
		run_context.unref();
		tree.unref();
	}
};

class BeehaveGraphEditor : public GraphEdit
{

	GDCLASS(BeehaveGraphEditor, GraphEdit);

public:

	static Color get_select_color()
	{
		return Color(0.7f, 0.65f, 1.0f, 0.5f);
	}

public:
	void set_beehave_tree(Ref<BeehaveGraphFrames> p_frames,Ref<BeehaveTree> p_beehave_tree,BeehaveGraphProperty *p_beehave_graph_property)
	{
		if(beehave_tree == p_beehave_tree)
		{
			return;
		}
		frames = p_frames;
		set_multi_select_active(false);
		beehave_graph_property = p_beehave_graph_property;
		active_nodes.clear();
		beehave_tree = p_beehave_tree;
		_update_graph();
	}
	void _ready() override
	{
		set_minimap_enabled(false);
		set_process(true);

		layout_button = memnew(Button);
		layout_button->set_flat(false);
		layout_button->set_focus_mode(FOCUS_NONE);
		layout_button->connect("pressed", callable_mp(this, &BeehaveGraphEditor::on_layout_changed));

		get_menu_hbox()->add_child(layout_button);
		_update_layout_button();

	}
	void _update_graph()
	{
		if(updating_graph)
		{
			return;
		}
		updating_graph = true;
		clear_connections();
        auto nodes = _get_child_nodes();
        for(int i = 0; i < nodes.size(); i++)
        {
			BeehaveGraphNodes* child = Object::cast_to<BeehaveGraphNodes>(nodes[i]);
			remove_child(child);
			child->queue_free();

        }
		if(beehave_tree.is_null())
		{
			return;
		}
		_add_nodes(beehave_tree->get_root_node());
		_connect_nodes(beehave_tree->get_root_node());
		callable_mp(this, &BeehaveGraphEditor::_arrange_nodes).call_deferred(beehave_tree);
		_arrange_nodes(beehave_tree);
		updating_graph = false;

	}

	void _add_nodes(const Ref<BeehaveNode>& p_beehave_node,const Ref<BeehaveNode>& p_parent_beehave_node = Ref<BeehaveNode>())
	{
		if(p_beehave_node.is_null())
		{
			return;
		}
		BeehaveGraphNodes* nodes = memnew(BeehaveGraphNodes);
		//nodes->set_title(p_beehave_node->get_lable_name());
		nodes->set_name(p_beehave_node->get_id());
		String tile_name = p_beehave_node->get_lable_name();
		if(p_beehave_node->get_name().length() > 0)
		{
			tile_name = "\n (" + p_beehave_node->get_name() + ")";
		}
		nodes->_init(beehave_graph_property,p_parent_beehave_node,p_beehave_node,frames, tile_name, "test", p_beehave_node->get_icon(), horizontal_layout);

		add_child(nodes);
		if(Object::cast_to<BeehaveLeaf>(p_beehave_node.ptr()))
		{
			nodes->set_slots(true,false);
		}
		else if(Object::cast_to<BeehaveComposite>(p_beehave_node.ptr()) || Object::cast_to<BeehaveDecorator>(p_beehave_node.ptr()))
		{
			nodes->set_slots(true,true);
		}
		if(!p_beehave_node->get_editor_collapsed_children())
		{
			for(int i = 0; i < p_beehave_node->get_child_count(); i++) 
			{
				_add_nodes(p_beehave_node->get_child(i),p_beehave_node); 
			}
		}

	}
	void _connect_nodes(const Ref<BeehaveNode>& p_beehave_node)
	{
		for(int i = 0; i < p_beehave_node->get_child_count(); i++)
		{
			connect_node(p_beehave_node->get_id(),0,p_beehave_node->get_child(i)->get_id(),0);
			_connect_nodes(p_beehave_node->get_child(i));
		}
	}
	void _arrange_nodes(Ref<BeehaveTree> p_beehave_tree)
	{
		if(arraging_nodes)
		{
			return;
		}

		Ref<BeehaveGraphTreeNode> tree_node = _create_tree_nodes(p_beehave_tree->get_root_node(),Ref<BeehaveGraphTreeNode>());
		tree_node->update_position(horizontal_layout);
		_place_nodes(tree_node);

		arraging_nodes = false;
	}
	Ref<BeehaveGraphTreeNode> _create_tree_nodes(const Ref<BeehaveNode>& p_beehave_node,const Ref<BeehaveGraphTreeNode>& p_parent)
	{
		Ref<BeehaveGraphTreeNode> tree_node = memnew(BeehaveGraphTreeNode);
		Node* n = get_node(NodePath(p_beehave_node->get_id()));
		tree_node->_init(p_parent.ptr(), Object::cast_to<BeehaveGraphNodes>(n ));
		
		if(!p_beehave_node->get_editor_collapsed_children())
		{
			for(int i = 0; i < p_beehave_node->get_child_count(); i++)
			{
				Ref<BeehaveGraphTreeNode> child = _create_tree_nodes(p_beehave_node->get_child(i),tree_node);
				tree_node->children.push_back(child);
			}
		}
		return tree_node;
	}
	void _place_nodes(Ref<BeehaveGraphTreeNode> p_tree_node)
	{
		p_tree_node->item->set_position_offset(Vector2( p_tree_node->pos_x,p_tree_node->pos_y));

		for(uint32_t i = 0; i < p_tree_node->children.size(); i++)
		{
			_place_nodes(p_tree_node->children[i]);
		}
	}

public:
	void on_layout_changed()
	{
		set_horizontal_layout(!horizontal_layout);
	}
	void set_horizontal_layout(bool p_horizontal_layout)
	{
		horizontal_layout = p_horizontal_layout;
		_update_layout_button();
		_update_graph();
	}
	Control* get_menu_container()
	{
		return get_menu_hbox();
	}

	String get_status(int p_status)
	{
		if (/* condition */ p_status == 0)
		{
			return "Success";
		}
		else if (/* condition */ p_status == 1)
		{
			return "Failure";
		}
		else
		{
			return "Running";
		}
		
	}

	void process_begin()
	{
		
		TypedArray<BeehaveGraphNodes> nodes = _get_child_nodes();
		Ref<BeehaveNode> beehave_node = Object::cast_to<BeehaveNode>( ObjectDB::get_instance(beehave_tree->last_editor_id));
		for(int i = 0; i < nodes.size(); ++i)
		{
			BeehaveGraphNodes* node = Object::cast_to<BeehaveGraphNodes>(nodes[i]);
			node->set_meta("status",-1);
			if(node->beehave_node == beehave_node)
			{
				node->set_title_color(get_select_color());
			}
			else{
				node->set_title_color(Color(1,1,1,0.85f));

			}
			
		}
	}

	void process_tick(BeehaveRunTool& run_tool)
	{
		if (run_tool.run_context.is_null())
		{
			active_nodes.clear();
			return;
		}
		TypedArray<BeehaveGraphNodes> nodes = _get_child_nodes();
		for (int i = 0; i < nodes.size(); ++i)
		{
			BeehaveGraphNodes* node = Object::cast_to<BeehaveGraphNodes>(nodes[i]);
			int p_status = run_tool.run_context->get_run_state(node->beehave_node.ptr());
			node->set_meta("status", p_status);
			node->set_status(p_status);
			node->set_text(get_status(p_status));

			if(p_status == 0 || p_status == 2)
			{
				if(!active_nodes.has(node))
				{
					active_nodes.push_back(node);
				}
			}
		}

	}

	void process_end()
	{
		TypedArray<BeehaveGraphNodes> nodes = _get_child_nodes();
		for(int i = 0; i < nodes.size(); ++i)
		{
			BeehaveGraphNodes* node = Object::cast_to<BeehaveGraphNodes>(nodes[i]);
			int status = node->get_meta("status");
			if(status == 0)
			{
				active_nodes.erase(node);
				node->set_color(SUCCESS_COLOR);
			}
			else if(status == 1)
			{
				active_nodes.erase(node);
				node->set_color(INACTIVE_COLOR);
			}
			else if(status == 2)
			{
				node->set_color(ACTIVE_COLOR);
			}
			else
			{
				node->set_text(" ");
				node->set_color(INACTIVE_COLOR);
				node->set_status(status);
			}
			node->update_text();
		}
	}

	TypedArray<BeehaveGraphNodes> _get_child_nodes()const
	{
		TypedArray<Node> c = get_children();
		TypedArray<BeehaveGraphNodes> rs;
		for(int i = 0; i < c.size(); ++i)
		{
			BeehaveGraphNodes* node = Object::cast_to<BeehaveGraphNodes>(c[i]);
			if(node)
			{
				rs.push_back(node);
			}
		}
		return rs;
	}

	virtual PackedVector2Array get_connection_line(const Vector2 &p_from, const Vector2 &p_to) const override
	{
		return _get_elbow_connection_line(p_from, p_to);
	}

	PackedVector2Array _get_elbow_connection_line(Vector2 p_from, Vector2 p_to) const
	{
		PackedVector2Array points;
		points.push_back(p_from);

		Vector2 mid_position = (p_from + p_to) / 2;
		if(horizontal_layout)
		{
			points.push_back(Vector2(mid_position.x, p_from.y));
			points.push_back(Vector2(mid_position.x, p_to.y));
		}
		else
		{
			points.push_back(Vector2(p_from.x, mid_position.y));
			points.push_back(Vector2(p_to.x, mid_position.y));
		}

		points.push_back(p_to);
		return points;
		
	}

	int progress = 0;
	virtual void process(double delta)override
	{
		if (!active_nodes.is_empty())
		{
			progress += (delta >= 0.05) ? 10 : 1;
			if (progress >= 1000)
			{
				progress = 0;
			}
			queue_redraw();
		}
	}
	void _draw() override;
	void _update_layout_button()
	{
		EditorBottomPanel* p_control = EditorNode::get_bottom_panel();
		if(horizontal_layout)
		{
			layout_button->set_button_icon(p_control->get_editor_theme_icon("horizontal_layout"));
		}
		else
		{
			layout_button->set_button_icon(p_control->get_editor_theme_icon("vertical_layout"));
		}

	}
	

protected:
	Color INACTIVE_COLOR  = Color("#898989");
	Color ACTIVE_COLOR = Color("#c29c06");
	Color SUCCESS_COLOR = Color("#07783a");
	int PROGRESS_SHIFT = 50;

	bool updating_graph = false;
	bool arraging_nodes = false;
	bool horizontal_layout = true;
	int children_count = 0;
	Ref<BeehaveTree> beehave_tree;
	LocalVector<BeehaveGraphNodes*> active_nodes;
	Ref<BeehaveGraphFrames> frames;
	Button* layout_button = nullptr;
	BeehaveGraphProperty* beehave_graph_property = nullptr;
};

// 绿豆蝇行为树子节点列表编辑
class BeehaveNodeChildChildEditor : public EditorPropertyArray {
	GDCLASS(BeehaveNodeChildChildEditor, EditorPropertyArray);


public:
	BeehaveNodeChildChildEditor()
	{
		show_add = false;
	}
	void setup(Ref<BeehaveNode> p_beehave_node,StringName property_name,Variant::Type p_array_type, const String &p_hint_string = "")
	{
		beehave_node = p_beehave_node;
		base_class_type::setup(p_array_type, p_hint_string);
		set_object_and_property(p_beehave_node.ptr(),property_name);
		set_label(property_name);
	}
	void on_reorder_button_gui_input(const Ref<InputEvent> &p_event)
	{
		_reorder_button_gui_input(p_event);
	}
	void on_reorder_button_up()
	{
		_reorder_button_up();
	}
	void on_reorder_button_down(int p_idx)
	{
		_reorder_button_down(p_idx);
	}
	void on_change_type(Object *p_button, int p_slot_index)
	{
		_change_type(p_button,p_slot_index);
	}
	void on_remove_pressed(int p_idx)
	{
		_remove_pressed(p_idx);
	}
	void on_update_state()
	{
	}
	virtual void _on_clear_slots()
	{
	}
	virtual void _create_new_property_slot() override
	{
		EditorBottomPanel* p_control = EditorNode::get_bottom_panel();
		int idx = slots.size();
		HBoxContainer *hbox = memnew(HBoxContainer);

		Button *reorder_button = memnew(Button);
		reorder_button->set_button_icon(p_control->get_editor_theme_icon(SNAME("TripleBar")));
		reorder_button->set_default_cursor_shape(Control::CURSOR_MOVE);
		reorder_button->set_disabled(is_read_only());
		reorder_button->connect(SceneStringName(gui_input), callable_mp(this, &BeehaveNodeChildChildEditor::on_reorder_button_gui_input));
		reorder_button->connect(SNAME("button_up"), callable_mp(this, &BeehaveNodeChildChildEditor::on_reorder_button_up));
		reorder_button->connect(SNAME("button_down"), callable_mp(this, &BeehaveNodeChildChildEditor::on_reorder_button_down).bind(idx));

		hbox->add_child(reorder_button);
		EditorProperty *prop = memnew(EditorPropertyNil);
		hbox->add_child(prop);

		// 增加状态按钮

		bool is_untyped_array = object->get_array().get_type() == Variant::ARRAY && subtype == Variant::NIL;

		if (is_untyped_array) {
			Button *edit_btn = memnew(Button);
			edit_btn->set_button_icon(p_control->get_editor_theme_icon(SNAME("Edit")));
			edit_btn->set_disabled(is_read_only());
			edit_btn->connect(SceneStringName(pressed), callable_mp(this, &BeehaveNodeChildChildEditor::on_change_type).bind(edit_btn, idx));
			hbox->add_child(edit_btn);
		} else {
			Button *remove_btn = memnew(Button);
			remove_btn->set_button_icon(p_control->get_editor_theme_icon(SNAME("Remove")));
			remove_btn->set_disabled(is_read_only());
			remove_btn->connect(SceneStringName(pressed), callable_mp(this, &BeehaveNodeChildChildEditor::on_remove_pressed).bind(idx));
			hbox->add_child(remove_btn);
		}
		property_vbox->add_child(hbox);

		EditorPropertyArray::Slot slot;
		slot.prop = prop;
		slot.object = object;
		slot.container = hbox;
		slot.reorder_button = reorder_button;
		slot.set_index(idx + page_index * page_length);
		slots.push_back(slot);
	}
protected:
	Ref<BeehaveNode> beehave_node;

};



// 绿豆蝇行为树属性编辑
class BeehaveGraphProperty : public VBoxContainer
{
	GDCLASS(BeehaveGraphProperty, VBoxContainer);
public:
	enum PlayState
	{
		PLAY_STATE_PLAY,
		PLAY_STATE_PAUSE,
		PLAY_STATE_STOP
	};
	BeehaveGraphProperty();
	void setup(Ref<BeehaveTree> p_beehave_tree,VBoxContainer * p_select_node_property_vbox);
	void process(double delta)override;
	void set_editor_node(Ref<BeehaveNode> p_beehave_node);
public:
	void _on_section_pressed() {
		if(beehave_editor == nullptr || frames.is_null())
		{
			return;
		}
		set_collapsed(!is_collapsed());
	}
	void on_beehave_node_change()
	{
		beehave_editor->_update_graph();
	}
	void _on_play_pressed() {
		run_tool.play(beehave_tree);
		_update_play_gui_state();
	}
	void _on_pause_pressed() {
		run_tool.pause();
		
		_update_play_gui_state();
	}
	void _on_stop_pressed() {
		run_tool.stop();
		_update_play_gui_state();
	}
	void _update_play_gui_state() {
		if(run_tool.is_running)
		{
			play_button->set_disabled(true);
			play_button->set_focus_mode(FOCUS_NONE);
		}else
		{
			play_button->set_disabled(false);
			play_button->set_focus_mode(FOCUS_CLICK);
		}

		if(run_tool.is_paused)
		{
			pause_button->set_disabled(true);
			pause_button->set_focus_mode(FOCUS_NONE);
		}else
		{
			pause_button->set_disabled(false);
			pause_button->set_focus_mode(FOCUS_CLICK);
		}

		if(run_tool.is_paused || run_tool.is_running)
		{
			stop_button->set_disabled(false);
			stop_button->set_focus_mode(FOCUS_CLICK);
		}else
		{
			stop_button->set_disabled(true);
			stop_button->set_focus_mode(FOCUS_NONE);
		}
	}
	void _on_slider_changed(double p_value) {
		run_tool.time_scale = p_value;
	}
	void on_remove_debug(Ref<BeehaveNode> p_beehave_node)
	{
		if (beehave_tree->get_debug_break_node() == p_beehave_node.ptr())
		{
			beehave_tree->set_debug_break_node(nullptr);
		}
	}
protected:
	void set_collapsed(bool p_collapsed) {
		if(p_collapsed)
		{
			set_custom_minimum_size(Size2(0, 20));
		}else
		{
			set_custom_minimum_size(Size2(0, 600));
		}
		beehave_editor->set_visible(!p_collapsed);
		sub_inspector->set_visible(!p_collapsed);
		child_list->set_visible(!p_collapsed);
	#ifdef TOOLS_ENABLED
		beehave_tree->editor_set_section_unfold("Beehave Tree Condition", !p_collapsed);
	#endif
		section_button->set_button_icon(p_collapsed ? frames->arrow_right_icon : frames->arrow_down_icon);
	}
	bool is_collapsed() const {
		return !beehave_editor->is_visible();
	}



	
	void _sub_inspector_property_keyed(const String &p_property, const Variant &p_value, bool p_advance) {
		// The second parameter could be null, causing the event to fire with less arguments, so use the pointer call which preserves it.
		const Variant args[3] = { String("children") + ":" + p_property, p_value, p_advance };
		const Variant *argp[3] = { &args[0], &args[1], &args[2] };
		emit_signalp(SNAME("property_keyed_with_value"), argp, 3);
	}

	void _sub_inspector_resource_selected(const Ref<RefCounted> &p_resource, const String &p_property) {
		emit_signal(SNAME("resource_selected"), String("children") + ":" + p_property, p_resource);
	}

	void _sub_inspector_object_id_selected(int p_id) {
		emit_signal(SNAME("object_id_selected"), "children", p_id);
	}

	void update_creted_beehave_node_state();

	
protected:
	Ref<BeehaveGraphFrames> frames;
	Ref<BeehaveTree> beehave_tree;
	Ref<BeehaveNode> beehave_node;
	BeehaveGraphEditor* beehave_editor = nullptr;

	HBoxContainer* play_help = nullptr;
	Button*  section_button = nullptr;
	VBoxContainer* select_node_property_vbox = nullptr;
	EditorInspector* sub_inspector = nullptr;
	BeehaveNodeChildChildEditor* child_list = nullptr;
	Button* play_button = nullptr;
	Button* pause_button = nullptr;
	Button* stop_button = nullptr;
	BeehaveRunTool run_tool;

	PlayState play_state = PLAY_STATE_STOP;




};
