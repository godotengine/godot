#include "beehave_graph_node.h"
#include "scene/gui/texture_rect.h"
#include "scene/gui/separator.h"
#include "scene/gui/check_box.h"





void BeehaveGraphNodes::_ready()
{

    set_custom_minimum_size(Vector2(50, 50) * frames->scale);
    add_theme_color_override("close_color", Color::hex(0xFFFFFF00));
    add_theme_icon_override("close",frames->icon_close);
    
    
    //# For top port
    Control* top_port = memnew(Control);
    add_child(top_port);

    icon_rect = memnew(TextureRect);
    icon_rect->set_stretch_mode(TextureRect::STRETCH_KEEP_ASPECT_CENTERED);
    icon_rect->set_texture(icon);
    
    titlebar_hbox = get_titlebar_hbox();
    titlebar_hbox->get_child(0)->queue_free();

    debug_break = memnew(Button);
    debug_break->set_button_icon(beehave_node->get_debug_enabled() ? frames->debug_enable_icon : frames->debug_disable_icon);
    debug_break->set_modulate(beehave_node->get_debug_enabled() ? Color(0.5, 0.5, 0.5, 1) :Color(1, 1, 1, 1) );
    titlebar_hbox->add_child(debug_break);
    debug_break->connect("pressed", callable_mp(this, &BeehaveGraphNodes::_on_debug_break));

    titlebar_hbox->set_alignment(BoxContainer::ALIGNMENT_BEGIN);
    titlebar_hbox->add_child(icon_rect);

    {
        Separator* separator = nullptr;
        HBoxContainer* hbox = memnew(HBoxContainer);
        add_child(hbox);
        hbox->set_h_size_flags(SIZE_EXPAND_FILL);
        hbox->set_custom_minimum_size(Vector2(160, 0));

        enable = memnew(CheckBox);
        hbox->add_child(enable);
        enable->set_pressed(beehave_node->get_enable());
        enable->connect("toggled", callable_mp(this, &BeehaveGraphNodes::_on_enable));


        // 增加一个空白区域
        Control* blank = memnew(Control);
        blank->set_h_size_flags(SIZE_EXPAND_FILL);
        hbox->add_child(blank);
        if(parent_beehave_node.is_valid()) 
        {

            move_up = memnew(Button);
            hbox->add_child(move_up);


            move_down = memnew(Button);
            hbox->add_child(move_down);

            
            if(horizontal)
            {
                move_up->set_button_icon(frames->move_up_icon);
                move_down->set_tooltip_text(L"上移");
                move_down->set_button_icon(frames->move_down_icon);
                move_down->set_tooltip_text(L"下移");
            }else
            {
                move_up->set_button_icon(frames->move_left_icon);
                move_up->set_tooltip_text(L"左移");
                move_down->set_button_icon(frames->move_right_icon);
                move_up->set_tooltip_text(L"右移");
            }
            move_up->connect("pressed", callable_mp(this, &BeehaveGraphNodes::_on_move_up));
            move_down->connect("pressed", callable_mp(this, &BeehaveGraphNodes::_on_move_down));

            separator = memnew(Separator);
            separator->set_modulate(Color(1, 1, 1, 0.2));
            hbox->add_child(separator);

        }


        bool is_show_create_child_node = true;
        if (beehave_node->get_supper_child_count() < 0)
        {

        }
        else if(beehave_node->get_supper_child_count() == 0)
        {
            is_show_create_child_node = false;
        }
        else
        {
            if(beehave_node->get_child_count() >= beehave_node->get_supper_child_count())
            {
                is_show_create_child_node = false;
            }
        }
        if(is_show_create_child_node)
        {
            create_child_node = memnew(Button);
            hbox->add_child(create_child_node);
            create_child_node->set_button_icon(frames->add_node_icon);
            create_child_node->connect("pressed", callable_mp(this, &BeehaveGraphNodes::_on_create_child_node));
            if(beehave_node->get_editor_collapsed_children())
            {
                create_child_node->set_disabled(beehave_node->get_editor_collapsed_children());
                create_child_node->set_tooltip_text(L"子节点被折叠,无法新增子节点");
            }
            else{
                
                create_child_node->set_tooltip_text(L"增加子节点");
            }
        }

        if(parent_beehave_node.is_valid()) 
        {
            separator = memnew(Separator);
            separator->set_modulate(Color(1, 1, 1, 0.2));
            hbox->add_child(separator);


            delete_node = memnew(Button);
            hbox->add_child(delete_node);
            delete_node->set_button_icon(frames->remove_node_icon);
            delete_node->connect("pressed", callable_mp(this, &BeehaveGraphNodes::_on_delete_node));
            delete_node->set_tooltip_text(L"删除本节点");
        }


    }
    
    Separator* separator = memnew(Separator);
    separator->set_h_size_flags(SIZE_EXPAND_FILL);
    separator->set_modulate(Color(1, 1, 1, 0.2));
    add_child(separator);


    title_label = memnew(Label);
    title_label->add_theme_color_override("font_color", Color(1, 1, 1, 1));
    title_label->add_theme_font_override("font", frames->font);
    title_label->set_vertical_alignment(VerticalAlignment::VERTICAL_ALIGNMENT_CENTER);
    title_label->set_h_size_flags(SIZE_EXPAND_FILL);
    title_label->set_text(title_text);
    titlebar_hbox->add_child(title_label);


    label = memnew(Label);
    label->set_text(text.is_empty() ? " " : text);
    add_child(label);


    state_label = memnew(Label);
    state_label->set_text(L"");
    add_child(state_label);

    if(beehave_node->get_child_count() > 0)
    {
        child_collapsed = memnew(Button);
        add_child(child_collapsed);
        child_collapsed->set_text(L"(收起/展开)子节点");
        child_collapsed->connect("pressed", callable_mp(this, &BeehaveGraphNodes::_on_child_collapsed));
        child_collapsed->set_button_icon(beehave_node->get_editor_collapsed_children() ? frames->arrow_right_icon : frames->arrow_down_icon);
    }
    else{
        separator = memnew(Separator);
        separator->set_h_size_flags(SIZE_EXPAND_FILL);
        separator->set_modulate(Color(1, 1, 1, 0.2));
        add_child(separator);
    }


    // For bottom port
    Control* bottom_port = memnew(Control);
    add_child(bottom_port);


    connect(SceneStringName(minimum_size_changed), callable_mp(this, &BeehaveGraphNodes::_on_size_changed));
    callable_mp(this, &BeehaveGraphNodes::_on_size_changed).call_deferred();

}
