#include "SpicyParticleSystemPlugin.h"
#include "scene/3d/spicyparticlesystem/SpicyParticleSystemNode.h"
#include "editor/gui/editor_spin_slider.h"
#include "editor/editor_interface.h"

#include "scene/gui/panel_container.h"
#include "scene/gui/check_button.h"
#include "scene/gui/box_container.h"
#include "scene/resources/style_box_flat.h"

////////////////////////////////////////////////////////////////////////

void SpicyParticleSystemModuleInspectorPlugin::_control_ref_added(Control* red)
{
	Node* parent_control = red->get_parent();
	parent_control = parent_control->get_child(0);
	if (parent_control != nullptr)
	{
		Array arr = parent_control->get_children();
		for (int i = 0; i < arr.size(); i++)
		{
			EditorProperty* prop = Object::cast_to<EditorProperty>(arr[i]);
			if (prop != nullptr)
			{
				//this is a hack to make the property checkable
				//prop->set_checkable(true);
				String name = prop->get_edited_property();
				if (name.ends_with("_x"))
				{
					prop->set_self_modulate(base_ref->get_theme_color("property_color_x", "Editor"));
				}
				if (name.ends_with("_y"))
				{
					prop->set_self_modulate(base_ref->get_theme_color("property_color_y", "Editor"));
				}
				if (name.ends_with("_z"))
				{
					prop->set_self_modulate(base_ref->get_theme_color("property_color_z", "Editor"));
				}
				if (name.ends_with("_w"))
				{
					prop->set_self_modulate(base_ref->get_theme_color("property_color_w", "Editor"));
				}
			}
		}
	}
}

SpicyParticleSystemModuleInspectorPlugin::SpicyParticleSystemModuleInspectorPlugin()
{
}

SpicyParticleSystemModuleInspectorPlugin::~SpicyParticleSystemModuleInspectorPlugin()
{
}

bool SpicyParticleSystemModuleInspectorPlugin::_can_handle(Object* p_object) const
{
	return Object::cast_to<SpicyParticleGenerator>(p_object) != nullptr || Object::cast_to<SpicyParticleUpdater>(p_object);
}

void SpicyParticleSystemModuleInspectorPlugin::_parse_end(Object* object)
{
	Control control_ref = Control();
	control_ref.set_name("CustomControl");
	add_custom_control(&control_ref);
	control_ref.connect("tree_entered", callable_mp(this, &SpicyParticleSystemModuleInspectorPlugin::_control_ref_added).bind(&control_ref));
	control_ref.set_visible(false);
}

////////////////////////////////////////////////////////////////////////


SpicyParticleSystemInspectorPlugin::SpicyParticleSystemInspectorPlugin()
{
}

SpicyParticleSystemInspectorPlugin::~SpicyParticleSystemInspectorPlugin()
{
}

bool SpicyParticleSystemInspectorPlugin::_can_handle(Object* p_object) const
{
	return Object::cast_to<SpicyParticleSystemNode>(p_object);
}

bool SpicyParticleSystemInspectorPlugin::_parse_property(Object* object, Variant::Type type, const String& name, PropertyHint hint_type, const String& hint_string, BitField<PropertyUsageFlags> usage_flags, bool wide)
{
	if (name == "seed")
	{
		EditorPropertyRandomInteger* editor = memnew(EditorPropertyRandomInteger);
		editor->set_base_ref(base_ref);
		editor->setup(0, 9999999, 1, false, false, false, "");

		add_property_editor(name, editor);

		return true;
	}

	//if (name == "simulation_time")
	//{
	//	return true;
	//}
	return false;
}

//void SpicyParticleSystemInspectorPlugin::_parse_group(Object* object, const String& group)
//{
//	//if (group == "Generators")
//	//{
//	//	//Could be used instead of having the empty resource slots
//	//	Button* button = memnew(Button);
//	//	button->set_text("Add Generator");
//	//	button->set_theme_type_variation("InspectorActionButton");
//	//	button->set_h_size_flags(Control::SizeFlags::SIZE_SHRINK_CENTER);
//	//	button->set_icon(base_ref->get_theme_icon("Add", "EditorIcons"));
//	//	add_custom_control(button);
//	//	//button->connect(SNAME("pressed"), callable_mp(this, &EditorInspector::_show_add_meta_dialog));
//	//}
//}


////////////////////////////////////////////////////////////////////////


void EditorPropertyRandomInteger::_randomize_integer()
{
	int64_t seed = Math::rand();

	emit_changed(get_edited_property(), seed);
}

void EditorPropertyRandomInteger::_value_changed(int64_t p_val)
{
	if (setting) {
		return;
	}
	emit_changed(get_edited_property(), p_val);
}

void EditorPropertyRandomInteger::_set_read_only(bool p_read_only)
{
	spin->set_read_only(p_read_only);
}

void EditorPropertyRandomInteger::_bind_methods()
{
}

void EditorPropertyRandomInteger::update_property()
{
	int64_t val = get_edited_object()->get(get_edited_property());
	setting = true;
	spin->set_value(val);
	setting = false;
}

void EditorPropertyRandomInteger::setup(int64_t p_min, int64_t p_max, int64_t p_step, bool p_hide_slider, bool p_allow_greater, bool p_allow_lesser, const String& p_suffix)
{
	spin->set_min(p_min);
	spin->set_max(p_max);
	spin->set_step(p_step);
	spin->set_hide_slider(p_hide_slider);
	spin->set_allow_greater(p_allow_greater);
	spin->set_allow_lesser(p_allow_lesser);
	spin->set_suffix(p_suffix);

	spin->connect("value_changed", callable_mp(this, &EditorPropertyRandomInteger::_value_changed));

	button->set_icon(base_ref->get_theme_icon("RandomNumberGenerator", "EditorIcons"));
	button->connect("pressed", callable_mp(this, &EditorPropertyRandomInteger::_randomize_integer));

}

EditorPropertyRandomInteger::EditorPropertyRandomInteger()
{
	HBoxContainer container = HBoxContainer();
	container.set_mouse_filter(Control::MOUSE_FILTER_IGNORE);
	add_child(&container);

	spin = memnew(EditorSpinSlider);
	spin->set_flat(true);
	spin->set_h_size_flags(Control::SizeFlags::SIZE_EXPAND_FILL);
	container.add_child(spin);
	add_focusable(spin);

	button = memnew(Button);
	button->set_theme_type_variation("InspectorActionButton");
	button->set_h_size_flags(Control::SizeFlags::SIZE_SHRINK_END);
	button->set_icon_alignment(HorizontalAlignment::HORIZONTAL_ALIGNMENT_CENTER);

	container.add_child(button);

	add_focusable(button);

}


////////////////////////////////////////////////////////////////////////

#pragma region EditorPlugin

void SpicyParticleSystemPlugin::set_pause_icon()
{
	play_pause_button->set_icon(pause_icon);
}

void SpicyParticleSystemPlugin::set_play_icon()
{
	play_pause_button->set_icon(play_icon);
}

void SpicyParticleSystemPlugin::_play_pause()
{
	//if (system_node != nullptr) {
	//	if (system_node->get_is_playing())
	//	{
	//		set_play_icon();
	//		system_node->pause(only_selected);
	//	}
	//	else
	//	{
	//		set_pause_icon();
	//		system_node->play(only_selected);
	//	}
	//}

	for (int i = 0; i < selection.size(); i++)
	{
		SpicyParticleSystemNode* node = Object::cast_to<SpicyParticleSystemNode>(selection[i]);
		if (node != nullptr)
		{
			if (node->get_is_playing())
			{
				node->pause();
			}
			else
			{
				node->play();
			}
		}
	}

	if (system_node != nullptr) 
	{
		if (system_node->get_is_playing())
		{
			set_play_icon();
		}
		else
		{
			set_pause_icon();
		}
	}
}

void SpicyParticleSystemPlugin::_restart()
{
	//if (system_node != nullptr)
	//{
	//	set_pause_icon();
	//	system_node->restart(only_selected);
	//}

	for (int i = 0; i < selection.size(); i++)
	{
		SpicyParticleSystemNode* node = Object::cast_to<SpicyParticleSystemNode>(selection[i]);
		if (node != nullptr)
		{
			node->restart();
		}
	}

	set_play_icon();
}

void SpicyParticleSystemPlugin::_stop()
{
	//if (system_node != nullptr)
	//{
	//	set_play_icon();
	//	system_node->stop(only_selected);
	//}

	for (int i = 0; i < selection.size(); i++)
	{
		SpicyParticleSystemNode* node = Object::cast_to<SpicyParticleSystemNode>(selection[i]);
		if (node != nullptr)
		{
			node->stop();
		}
	}

	set_play_icon();
}

void SpicyParticleSystemPlugin::_set_simulation_time(double time)
{
	//if (system_node != nullptr)
	//{
	//	system_node->set_simulation_time(time);
	//	set_play_icon();
	//}

	for (int i = 0; i < selection.size(); i++)
	{
		SpicyParticleSystemNode* node = Object::cast_to<SpicyParticleSystemNode>(selection[i]);
		if (node != nullptr)
		{
			node->seek(time, true);
		}
	}

	set_play_icon();
}

void SpicyParticleSystemPlugin::_move_button_up()
{
	move_button_down = false;
}

void SpicyParticleSystemPlugin::_set_only_selected(bool show)
{
	only_selected = show;

	_update_selection();
}

void SpicyParticleSystemPlugin::_update_selection()
{
	for (int i = 0; i < selection.size(); i++)
	{
		SpicyParticleSystemNode* node = Object::cast_to<SpicyParticleSystemNode>(selection[i]);
		if (node != nullptr)
		{
			node->_set_render(true);
		}
	}

	selection = get_editor_interface()->get_selection()->get_selected_nodes();

	if (!only_selected)
		return;

	for (int i = 0; i < selection.size(); i++)
	{
		SpicyParticleSystemNode* node = Object::cast_to<SpicyParticleSystemNode>(selection[i]);
		if (node != nullptr)
		{
			node->_set_render(false);
		}
	}

	for (int i = 0; i < selection.size(); i++)
	{
		SpicyParticleSystemNode* node = Object::cast_to<SpicyParticleSystemNode>(selection[i]);
		if (node != nullptr)
		{
			node->_set_render(true, false);
		}
	}
}

void SpicyParticleSystemPlugin::_move_button_down()
{
	move_button_down = true;
	mouse_offset = on_screen_editor->get_position() - view_control->get_local_mouse_position();
}

SpicyParticleSystemPlugin::SpicyParticleSystemPlugin()
{
}

SpicyParticleSystemPlugin::~SpicyParticleSystemPlugin()
{

}

void SpicyParticleSystemPlugin::_enter_tree()
{
	module_inspector_plugin.instantiate();
	module_inspector_plugin->set_base_ref(get_editor_interface()->get_base_control());
	add_inspector_plugin(module_inspector_plugin);

	inspector_plugin.instantiate();
	inspector_plugin->set_base_ref(get_editor_interface()->get_base_control());
	add_inspector_plugin(inspector_plugin);

	_create_on_screen_editor();
}

void SpicyParticleSystemPlugin::_exit_tree()
{
	remove_inspector_plugin(module_inspector_plugin);
	remove_inspector_plugin(inspector_plugin);
	on_screen_editor->queue_free();

	//is it necessary?
	memdelete(on_screen_editor);
}

void SpicyParticleSystemPlugin::_process(double delta)
{
	if (system_node != nullptr && on_screen_editor != nullptr && on_screen_editor->is_visible())
	{
		_update_on_screen_editor();

		if (move_button_down)
		{
			Vector2 view_size = view_control->get_size();
			Vector2 mouse_pos = view_control->get_local_mouse_position() + mouse_offset;
			Vector2 control_size = on_screen_editor->get_size();
			Vector2 new_pos = mouse_pos;

			new_pos.x = CLAMP(mouse_pos.x, 5, view_size.x - control_size.x - 5);
			new_pos.y = CLAMP(mouse_pos.y, 5, view_size.y - control_size.y - 5);

			on_screen_editor->set_position(new_pos);
		}
	}
}

void SpicyParticleSystemPlugin::_create_on_screen_editor()
{
	if (on_screen_editor != nullptr)
		return;

	const Control* base_control = get_editor_interface()->get_base_control();
	play_icon = base_control->get_theme_icon("MainPlay", "EditorIcons");
	pause_icon = base_control->get_theme_icon("Pause", "EditorIcons");
	restart_icon = base_control->get_theme_icon("Reload", "EditorIcons");
	stop_icon = base_control->get_theme_icon("Stop", "EditorIcons");

	get_editor_interface()->get_selection()->connect("selection_changed", callable_mp(this, &SpicyParticleSystemPlugin::_update_selection));

	on_screen_editor = memnew(PanelContainer);
	on_screen_editor->set_name("spicy_particle_ose");
	on_screen_editor->set_custom_minimum_size(Vector2(220, 140));
	on_screen_editor->set_v_size_flags(Panel::SIZE_EXPAND);
	on_screen_editor->set_anchors_preset(Control::PRESET_CENTER_RIGHT);
	on_screen_editor->set_position(Vector2(-240, 20));


	VBoxContainer* vbox_parent = memnew(VBoxContainer);
	vbox_parent->set_mouse_filter(Control::MOUSE_FILTER_IGNORE);
	vbox_parent->set_custom_minimum_size(Vector2(220, 50));
	vbox_parent->set_anchors_preset(Control::PRESET_FULL_RECT);
	on_screen_editor->add_child(vbox_parent);

	button_move = memnew(Button);
	button_move->set_keep_pressed_outside(true);
	button_move->set_custom_minimum_size(Vector2(220, 8));
	button_move->set_scale(Vector2(1, 0.8));
	button_move->set_keep_pressed_outside(true);
	button_move->set_modulate(Color(1, 1, 1, 0.5));
	button_move->connect("button_down", callable_mp(this, &SpicyParticleSystemPlugin::_move_button_down));
	button_move->connect("button_up", callable_mp(this, &SpicyParticleSystemPlugin::_move_button_up));
	vbox_parent->add_child(button_move);

	CheckButton* button_only_selected = memnew(CheckButton);
	button_only_selected->set_text("Selected only");
	button_only_selected->set_toggle_mode(true);
	button_only_selected->connect("toggled", callable_mp(this, &SpicyParticleSystemPlugin::_set_only_selected));
	vbox_parent->add_child(button_only_selected);

	HBoxContainer* hbox_top = memnew(HBoxContainer);
	vbox_parent->add_child(hbox_top);

	Label* label_alive = memnew(Label);
	label_alive->set_text("Alive count: ");
	label_alive->set_custom_minimum_size(Vector2(100, 0));
	hbox_top->add_child(label_alive);

	label_count = memnew(Label);
	label_count->set_text(String::num_int64(0));
	label_count->set_custom_minimum_size(Vector2(100, 0));
	hbox_top->add_child(label_count);

	HBoxContainer* hbox_mid = memnew(HBoxContainer);
	vbox_parent->add_child(hbox_mid);

	Label* label_sim = memnew(Label);
	label_sim->set_text("Simulation time: ");
	label_sim->set_custom_minimum_size(Vector2(100, 0));
	hbox_mid->add_child(label_sim);

	spin_box_time = memnew(EditorSpinSlider);
	spin_box_time->set_custom_minimum_size(Vector2(100, 0));
	spin_box_time->set_value_no_signal(0);
	spin_box_time->set_hide_slider(true);
	spin_box_time->set_suffix("s");
	spin_box_time->set_max(999);
	spin_box_time->set_step(0.001);
	spin_box_time->connect("value_changed", callable_mp(this, &SpicyParticleSystemPlugin::_set_simulation_time));
	hbox_mid->add_child(spin_box_time);

	HBoxContainer* hbox_bottom = memnew(HBoxContainer);
	hbox_bottom->set_h_size_flags(Control::SIZE_SHRINK_CENTER);
	vbox_parent->add_child(hbox_bottom);

	Vector2 icon_size = Vector2(50, 5);

	play_pause_button = memnew(Button);
	play_pause_button->set_custom_minimum_size(icon_size);
	play_pause_button->set_icon(play_icon);
	play_pause_button->set_icon_alignment(HorizontalAlignment::HORIZONTAL_ALIGNMENT_CENTER);
	play_pause_button->connect("pressed", callable_mp(this, &SpicyParticleSystemPlugin::_play_pause));
	hbox_bottom->add_child(play_pause_button);

	Button* button_restart = memnew(Button);
	button_restart->set_custom_minimum_size(icon_size);
	button_restart->set_icon(restart_icon);
	button_restart->set_icon_alignment(HorizontalAlignment::HORIZONTAL_ALIGNMENT_CENTER);
	button_restart->connect("pressed", callable_mp(this, &SpicyParticleSystemPlugin::_restart));
	hbox_bottom->add_child(button_restart);

	Button* button_stop = memnew(Button);
	button_stop->set_custom_minimum_size(icon_size);
	button_stop->set_icon(stop_icon);
	button_stop->set_icon_alignment(HorizontalAlignment::HORIZONTAL_ALIGNMENT_CENTER);
	button_stop->connect("pressed", callable_mp(this, &SpicyParticleSystemPlugin::_stop));
	hbox_bottom->add_child(button_stop);

	//theme
	Ref<StyleBoxFlat> style_box = Ref<StyleBoxFlat>(memnew(StyleBoxFlat));
	style_box->set_bg_color(Color(0.152941, 0.152941, 0.152941, 0.501961));
	style_box->set_border_width_all(1);
	style_box->set_border_color(Color(0.411765, 0.411765, 0.411765, 1));
	style_box->set_corner_radius_all(5);
	style_box->set_expand_margin_all(5);
	on_screen_editor->add_theme_style_override("panel", style_box);

	Ref<StyleBoxEmpty> style_box2 = Ref<StyleBoxEmpty>(memnew(StyleBoxEmpty));
	button_move->add_theme_style_override("focus", style_box2);
}

void SpicyParticleSystemPlugin::_update_on_screen_editor()
{
	spin_box_time->set_value_no_signal(system_node->get_simulation_time());
	label_count->set_text(String::num_int64(system_node->get_particle_count()));

	//get all alive count from hierarchy

	//int count = 0;

	//SpicyParticleSystemNode* node = Object::cast_to<SpicyParticleSystemNode>(system_node->get_parent());
	//bool found = false;
	//if (node != nullptr) {
	//	while (!found)
	//	{
	//		SpicyParticleSystemNode* temp = Object::cast_to<SpicyParticleSystemNode>(node->get_parent());

	//		if (temp == nullptr)
	//		{
	//			found = true;
	//		}
	//		node = temp;
	//	}
	//}
	//else
	//{
	//	node = system_node;
	//}



	//TypedArray<SpicyParticleSystemNode> children = node->find_children("*", SpicyParticleSystemNode::get_class_static(), true, false);
	//for (int i = 0; i < children.size(); ++i)
	//{
	//	SpicyParticleSystemNode* child = cast_to<SpicyParticleSystemNode>(children[i]);
	//	count += child->get_particle_count();
	//}


	//label_count->set_text(String::num_int64(count));
}

void SpicyParticleSystemPlugin::_edit(Object* object)
{

	if (Object::cast_to<SpicyParticleSystemNode>(object) == nullptr || system_node != Object::cast_to<SpicyParticleSystemNode>(object))
	{
		if (system_node != nullptr)
		{
			SpicyParticleSystemNode* temp = Object::cast_to<SpicyParticleSystemNode>(system_node);

			if (temp->is_connected("play", callable_mp(this, &SpicyParticleSystemPlugin::set_pause_icon)))
			{
				temp->disconnect("play", callable_mp(this, &SpicyParticleSystemPlugin::set_pause_icon));
			}
			if (temp->is_connected("stop", callable_mp(this, &SpicyParticleSystemPlugin::set_play_icon)))
			{
				temp->disconnect("stop", callable_mp(this, &SpicyParticleSystemPlugin::set_play_icon));
			}
		}
	}

	system_node = Object::cast_to<SpicyParticleSystemNode>(object);

	if (system_node != nullptr) {

		if (!system_node->is_connected("play", callable_mp(this, &SpicyParticleSystemPlugin::set_pause_icon)))
		{
			system_node->connect("play", callable_mp(this, &SpicyParticleSystemPlugin::set_pause_icon));
		}
		if (!system_node->is_connected("stop", callable_mp(this, &SpicyParticleSystemPlugin::set_play_icon)))
		{
			system_node->connect("stop", callable_mp(this, &SpicyParticleSystemPlugin::set_play_icon));
		}


		if (system_node->get_is_playing())
		{
			set_pause_icon();
		}
		else
		{
			set_play_icon();
		}

	}
}

void SpicyParticleSystemPlugin::_make_visible(bool visible)
{
	on_screen_editor->set_visible(visible);
}

bool SpicyParticleSystemPlugin::_handles(Object* object) const
{
	return Object::cast_to<SpicyParticleSystemNode>(object) != nullptr;
}

void SpicyParticleSystemPlugin::_forward_3d_draw_over_viewport(Control* viewport_control)
{
	view_control = viewport_control;

	if (viewport_control->has_node(NodePath("spicy_particle_ose")))
		return;

	if (on_screen_editor->get_parent() == viewport_control)
		return;
	else if (on_screen_editor->get_parent() == nullptr)
	{
		viewport_control->add_child(on_screen_editor);
	}
}

int32_t SpicyParticleSystemPlugin::_forward_3d_gui_input(Camera3D* viewport_camera, const Ref<InputEvent>& event)
{
	if (typeid(event) == typeid(InputEventMouseMotion))
	{
		update_overlays();
	}

	return 0;
}

#pragma endregion
