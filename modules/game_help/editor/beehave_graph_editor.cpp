#include "beehave_graph_editor.h"
#include "modules/game_help/logic/body_main.h"


void BeehaveRunTool::play(const Ref<BeehaveTree>& p_tree)
{
    stop();
    CharacterBodyMain* main = CharacterBodyMain::get_current_editor_player();
    if (main == nullptr || main->get_blackboard_plan().is_null())
    {
        return;
    }
    tree = p_tree;
    if (tree.is_null())
    {
        return;
    }
    run_context.instantiate();
    is_running = true;
    is_paused = false;
    tree->init(run_context);
    run_context->actor = main;
    run_context->delta = delta;
    run_context->blackboard = main->get_blackboard_plan()->get_editor_blackboard();
}
/******************************************************************************************************************/

void BeehaveGraphEditor::_draw()
{
    if(active_nodes.is_empty())
    {
        return;
    }		
    float circle_size = MAX(3,6 * get_zoom());
    float progress_shift = PROGRESS_SHIFT * get_zoom();

    auto _connections = get_connection_list();
    for(auto& i : _connections)
    {
        StringName from = i->from_node;
        StringName to = i->to_node;
        BeehaveGraphNodes* from_node =  Object::cast_to<BeehaveGraphNodes>( get_node(NodePath(from)) );
        BeehaveGraphNodes* to_node =  Object::cast_to<BeehaveGraphNodes>(get_node(NodePath(to)));

        if(!active_nodes.has(from_node) && !active_nodes.has(to_node))
        {
            continue;
        }
        
        if((int)from_node->get_meta("status") < 0 && (int)to_node->get_meta("status") < 0)
        {
            return;
        }

        Vector2 output_port_position;
        Vector2 input_port_position;


        float scale_factor = from_node->get_rect().size.x / from_node->get_size().x;

        Vector2 from_graph_position = (from_node->get_position() + from_node->get_output_port_position(0) * get_zoom());
        Vector2 to_graph_position = (to_node->get_position() + to_node->get_input_port_position(0) * get_zoom());
        auto line = _get_elbow_connection_line(from_graph_position, to_graph_position);

        Curve2D curve;
        for(auto& j : line)
        {
            curve.add_point(j);
        }

        int max_steps = curve.get_baked_length();
        float current_shift = progress % max_steps;
        auto p = curve.sample_baked(current_shift);

        draw_circle(p, circle_size, ACTIVE_COLOR);

        int shift = current_shift - progress_shift;

        while(shift >= 0)
        {
            p = curve.sample_baked(shift);
            draw_circle(p, circle_size, ACTIVE_COLOR);
            shift -= max_steps;
        }

        shift = current_shift + progress_shift;
        while(shift < max_steps)
        {
            p = curve.sample_baked(shift);
            draw_circle(p, circle_size, ACTIVE_COLOR);
            shift += max_steps;
        }

    }
}

/******************************************************************************************************************/
BeehaveGraphProperty::BeehaveGraphProperty()
{
    set_layout_mode(LayoutMode::LAYOUT_MODE_CONTAINER);

    HBoxContainer* section = memnew(HBoxContainer);
    add_child(section);
    section->set_layout_mode(LayoutMode::LAYOUT_MODE_CONTAINER);
    section->set_h_size_flags(SIZE_EXPAND_FILL);


    
    section_button = memnew(Button);
    section_button->set_h_size_flags(SIZE_EXPAND_FILL);
    section_button->set_text(L"绿豆蝇行为树");
    section_button->connect(SceneStringName(pressed), callable_mp(this, &BeehaveGraphProperty::_on_section_pressed));
    section->add_child(section_button);


    play_help = memnew(HBoxContainer);
    section->add_child(play_help);

    HSlider* slider = memnew(HSlider);
    slider->set_min(0);
    slider->set_max(5);
    slider->set_step(0.01);
    slider->set_value(1.0);
    slider->set_custom_minimum_size(Vector2(80, 0));
    play_help->add_child(slider);
    slider->connect(SceneStringName(value_changed), callable_mp(this, &BeehaveGraphProperty::_on_slider_changed));


    play_button = memnew(Button);
    play_button->set_text(L"播放");
    play_button->connect(SceneStringName(pressed), callable_mp(this, &BeehaveGraphProperty::_on_play_pressed));
    play_help->add_child(play_button);

    pause_button = memnew(Button);
    pause_button->set_text(L"暂停");
    pause_button->connect(SceneStringName(pressed), callable_mp(this, &BeehaveGraphProperty::_on_pause_pressed));
    pause_button->set_disabled(true);
    play_help->add_child(pause_button);

    stop_button = memnew(Button);
    stop_button->set_text(L"停止");
    stop_button->connect(SceneStringName(pressed), callable_mp(this, &BeehaveGraphProperty::_on_stop_pressed));
    stop_button->set_disabled(true);
    play_help->add_child(stop_button);

    beehave_editor = memnew(BeehaveGraphEditor);
    add_child(beehave_editor);
    beehave_editor->set_layout_mode(LayoutMode::LAYOUT_MODE_CONTAINER);
    beehave_editor->set_v_size_flags(Control::SIZE_EXPAND_FILL);

    set_process(true);

    //add_child(select_node_property_vbox);

}

void BeehaveGraphProperty::setup(Ref<BeehaveTree> p_beehave_tree,VBoxContainer * p_select_node_property_vbox)
{
    frames.instantiate();
    frames->init();

    select_node_property_vbox = p_select_node_property_vbox;
    select_node_property_vbox->set_layout_mode(LayoutMode::LAYOUT_MODE_CONTAINER);
    beehave_tree = p_beehave_tree;
    Ref<BeehaveNode> node = Object::cast_to<BeehaveNode>(ObjectDB::get_instance(beehave_tree->last_editor_id));
    if(node.is_null())
    {
        node = beehave_tree->get_root_node();
    }
    if (node.is_null())
    {
        return;
    }
    child_list = memnew(BeehaveNodeChildChildEditor);
    child_list->set_layout_mode(LayoutMode::LAYOUT_MODE_CONTAINER);
    child_list->set_visible(true);
    child_list->set_modulate(BeehaveGraphEditor::get_select_color());
    select_node_property_vbox->add_child(child_list);




    sub_inspector = memnew(EditorInspector);
    
    sub_inspector->set_vertical_scroll_mode(ScrollContainer::SCROLL_MODE_DISABLED);
    sub_inspector->set_use_doc_hints(true);

    //sub_inspector->set_sub_inspector(true);
    sub_inspector->set_property_name_style(InspectorDock::get_singleton()->get_property_name_style());

    sub_inspector->connect("property_keyed", callable_mp(this, &BeehaveGraphProperty::_sub_inspector_property_keyed));
    sub_inspector->connect("resource_selected", callable_mp(this, &BeehaveGraphProperty::_sub_inspector_resource_selected));
    sub_inspector->connect("object_id_selected", callable_mp(this, &BeehaveGraphProperty::_sub_inspector_object_id_selected));
    sub_inspector->set_keying(true);
    sub_inspector->set_read_only(false);
    sub_inspector->set_use_folding(true);

    sub_inspector->set_mouse_filter(MOUSE_FILTER_STOP);
    sub_inspector->set_modulate(BeehaveGraphEditor::get_select_color());

    select_node_property_vbox->add_child(sub_inspector);




    beehave_tree = p_beehave_tree;
    set_editor_node( node);
    beehave_editor->set_beehave_tree(frames,p_beehave_tree,this);



    
#ifdef TOOLS_ENABLED
        set_collapsed(!beehave_tree->editor_is_section_unfolded("Beehave Tree Condition"));
#endif
    _update_play_gui_state();
}
void BeehaveGraphProperty::process(double delta)
{
    auto body = CharacterBodyMain::get_current_editor_player();
    if(body != nullptr)
    {
        play_help->set_visible(!body->get_editor_run_ai());
        // 更新播放状态
    }
    else
    {
        play_help->set_visible(false);
    }
    run_tool.tick(delta);
    beehave_editor->process_begin();
    beehave_editor->process_tick(run_tool);
    beehave_editor->process_end();
}
void BeehaveGraphProperty::set_editor_node(Ref<BeehaveNode> p_beehave_node)
{
    beehave_node = p_beehave_node;
    if (beehave_node.is_valid())
    {
        beehave_tree->last_editor_id = beehave_node->get_instance_id();
    }
    sub_inspector->edit(beehave_node.ptr());
    update_creted_beehave_node_state();
}
void BeehaveGraphProperty::update_creted_beehave_node_state()
{
    // 更新按鈕狀態
    bool show = true;
    if (beehave_node->get_supper_child_count() < 0)
    {
        show = true;
    }
    else if(beehave_node->get_supper_child_count() == 0)
    {
        show = false;
    }
    else
    {
        if(beehave_node->get_child_count() >= beehave_node->get_supper_child_count())
        {
            show = false;
        }
    }
    if(Object::cast_to<BeehaveLeaf>(beehave_node.ptr()))
    {
        show = false;
    }
    child_list->setup(beehave_node, "children", Variant::OBJECT, MAKE_RESOURCE_TYPE_HINT("BeehaveNode"));
    child_list->update_property();
    

}


/*****************************************************************************************************************************/
void BeehaveGraphNodes::on_selected(bool p_selected)
{
	if (p_selected) {
		beehave_graph_property->set_editor_node(beehave_node);
	}
}
void BeehaveGraphNodes::on_beehave_node_change()
{
	beehave_graph_property->on_beehave_node_change();
	
	on_selected(true);
}
void BeehaveGraphNodes::_on_debug_break()
{
	beehave_node->set_debug_enabled(! beehave_node->get_debug_enabled());
	debug_break->set_button_icon(beehave_node->get_debug_enabled() ? frames->debug_enable_icon : frames->debug_disable_icon);
	debug_break->set_modulate(beehave_node->get_debug_enabled() ? Color(0.5, 0.5, 0.5, 1) :Color(1, 1, 1, 1) );
	if(!beehave_node->get_debug_enabled())
	{
		beehave_graph_property->on_remove_debug(beehave_node);
	}
}