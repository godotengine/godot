#pragma once
#include "core/object/ref_counted.h"
#include "core/templates/local_vector.h"
#include "scene/gui/graph_node.h"
#include "modules/game_help/logic/beehave/beehave_tree.h"
#include "scene/resources/style_box_flat.h"
#include "scene/resources/image_texture.h"
#include "editor/editor_node.h"
#include "editor/editor_properties.h"
#include "editor/inspector_dock.h"
#include "editor/gui/editor_bottom_panel.h"
#include "scene/gui/graph_edit_arranger.h"
#include "scene/gui/view_panner.h"
#include "scene/gui/graph_edit.h"

#include "editor/themes/editor_scale.h"
class BeehaveGraphProperty;
class BeehaveGraphFrames : public RefCounted
{

	GDCLASS(BeehaveGraphFrames, RefCounted);

public:
	Color SUCCESS_COLOR = Color("#07783a");
	Color NORMAI_COLOR = Color("#15181e");
	Color FAILURE_COLOR = Color("#82010b");
	Color RUNNING_COLOR = Color("#c29ce6");

	Ref<StyleBoxFlat> panal_normal;
	Ref<StyleBoxFlat> panal_success;
	Ref<StyleBoxFlat> panal_failure;
	Ref<StyleBoxFlat> panal_running;

	Ref<StyleBoxFlat> titlebar_normal;
	Ref<StyleBoxFlat> titlebar_success;
	Ref<StyleBoxFlat> titlebar_failure;
	Ref<StyleBoxFlat> titlebar_running;


	Ref<ImageTexture> icon_close;
	
	Ref<ImageTexture> icon_port_top;
	Ref<ImageTexture> icon_port_bottom;
	Ref<ImageTexture> icon_port_left;
	Ref<ImageTexture> icon_port_right;


	Ref<ImageTexture> arrow_down_icon;
	Ref<ImageTexture> arrow_right_icon;
	Ref<Font> font;
	float scale = 1.0f;
	BeehaveGraphFrames()
	{

	}
	void init()
	{
		EditorBottomPanel* p_control = EditorNode::get_bottom_panel();

		titlebar_normal = p_control->get_theme_stylebox(SNAME("titlebar"), SNAME("GraphEdit"))->duplicate();
		titlebar_success = titlebar_normal->duplicate();
		titlebar_failure = titlebar_normal->duplicate();
		titlebar_running = titlebar_normal->duplicate();


		titlebar_success->set_bg_color(SUCCESS_COLOR);
		titlebar_failure->set_bg_color(FAILURE_COLOR);
		titlebar_running->set_bg_color(RUNNING_COLOR);

		titlebar_success->set_border_color(SUCCESS_COLOR);
		titlebar_failure->set_border_color(FAILURE_COLOR);
		titlebar_running->set_border_color(RUNNING_COLOR);


		
		panal_normal = p_control->get_theme_stylebox(SNAME("panel"), SNAME("GraphEdit"))->duplicate();
		panal_success = p_control->get_theme_stylebox(SNAME("panel_selected"), SNAME("GraphEdit"))->duplicate();
		panal_failure = panal_success->duplicate();
		panal_running = panal_success->duplicate();

		panal_success->set_border_color(SUCCESS_COLOR);
		panal_failure->set_border_color(FAILURE_COLOR);
		panal_running->set_border_color(RUNNING_COLOR);

		icon_close = memnew(ImageTexture);
		font = p_control->get_theme_font(SNAME("title_font"))->duplicate();
        Ref<FontVariation> variation = font;
        Ref<FontFile> file = font;
		if(variation.is_valid())
		{
			variation->set_variation_embolden(1);
		}
		else if(file.is_valid())
		{
			file->set_font_weight(700);
		}

		icon_port_top = p_control->get_editor_theme_icon(SNAME("port_top"));
		icon_port_bottom = p_control->get_editor_theme_icon(SNAME("port_bottom"));
		icon_port_left = p_control->get_editor_theme_icon(SNAME("port_left"));
		icon_port_right = p_control->get_editor_theme_icon(SNAME("port_right"));
		
		arrow_down_icon = p_control->get_editor_theme_icon(SNAME("GuiTreeArrowDown"));
		arrow_right_icon = p_control->get_editor_theme_icon(SNAME("GuiTreeArrowRight"));
	}
};

// BeehaveGraphNodes
class BeehaveGraphNodes : public GraphNode
{

	GDCLASS(BeehaveGraphNodes, GraphNode);
public:

	String title_text;
	String text;
	Ref<Texture2D> icon;
	float layout_size = 0;
	TextureRect* icon_rect;
	Label* title_label;
	Label* label;
	HBoxContainer* titlebar_hbox;
	Ref<BeehaveGraphFrames> frames;
	bool horizontal = false;
	Ref<BeehaveNode> beehave_node;
	BeehaveGraphProperty* beehave_graph_property = nullptr;
	BeehaveGraphNodes()
	{
		
	}
	void _init(BeehaveGraphProperty *p_beehave_graph_property,const Ref<BeehaveNode>& p_beehave_node,Ref<BeehaveGraphFrames> p_frames,const String& p_title_text,const String& p_text, const StringName& p_icon, bool p_horizontal)
	{

		beehave_graph_property = p_beehave_graph_property;
		beehave_node = p_beehave_node;
		title_text = p_title_text;
		text = p_text;
		frames = p_frames;
		horizontal = p_horizontal;
		EditorBottomPanel* p_control = EditorNode::get_bottom_panel();
		icon = p_control->get_editor_theme_icon(p_icon);
		set_draggable(false);
	}
	void _ready()
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
		titlebar_hbox->set_alignment(BoxContainer::ALIGNMENT_BEGIN);
		titlebar_hbox->add_child(icon_rect);

		title_label = memnew(Label);
		title_label->add_theme_color_override("font_color", Color(1, 1, 1, 1));
		title_label->add_theme_font_override("font", frames->font);
		title_label->set_vertical_alignment(VerticalAlignment::VERTICAL_ALIGNMENT_CENTER);
		title_label->set_text(title_text);
		titlebar_hbox->add_child(title_label);
		

		label = memnew(Label);
		label->set_text(text.is_empty() ? " " : text);
		add_child(label);

		// For bottom port
		Control* bottom_port = memnew(Control);
		add_child(bottom_port);


		connect(SceneStringName(minimum_size_changed), callable_mp(this, &BeehaveGraphNodes::_on_size_changed));
		callable_mp(this, &BeehaveGraphNodes::_on_size_changed).call_deferred();

	}

	virtual void draw_port(int p_slot_index, Point2i p_pos, bool p_left, const Color &p_color) override
	{
		if(horizontal)
		{
			if(is_slot_enabled_left(1))
			{
				draw_texture(frames->icon_port_left, get_input_port_position(0) + Vector2(-4, 10),p_color);
			}
			if(is_slot_enabled_right(1))
			{
				draw_texture(frames->icon_port_right, get_output_port_position(0) + Vector2(5, -7),p_color);
			}
		}
		else
		{
			if(p_slot_index == 0 && is_slot_enabled_left(0))
			{
				draw_texture(frames->icon_port_top, get_input_port_position(0) + Vector2(-5, 10),p_color);
			}
			else if(p_slot_index == 1 )
			{
				draw_texture(frames->icon_port_bottom, get_output_port_position(0) + Vector2(5, -7),p_color);
			}
		}
	}
	virtual Vector2 get_input_port_position(int p_port_idx) override
	{
		if (horizontal)
		{
			return  Vector2(2, get_size().y / 2) ;
		}
		else
		{
			return Vector2(get_size().x / 2, 2) ;
		}
	}
	virtual Vector2 get_output_port_position(int p_port_idx) override
	{
		if (horizontal)
		{
			return  Vector2(get_size().x - 2, get_size().y / 2);
		}
		else
		{
			return Vector2(get_size().x / 2, get_size().y - 2);
		}
	}
	void set_status(int p_status)
	{
		if(p_status == 0)
		{
			_set_stylebox_override(frames->panal_success,frames->titlebar_success);
		}
		else if(p_status == 1)
		{
			_set_stylebox_override(frames->panal_failure,frames->titlebar_failure);
		}
		else if(p_status == 2)
		{
			_set_stylebox_override(frames->panal_running,frames->titlebar_running);
		}
		else
		{
			_set_stylebox_override(frames->panal_normal,frames->titlebar_normal);
		}
	}

	void set_slots(bool p_left, bool p_right)
	{
		if(horizontal)
		{
			set_slot(1,p_left,-1,Color(1,1,1,1),p_right,-1,Color(1,1,1,1),frames->icon_port_left,frames->icon_port_right);
		}
		else
		{
			set_slot(0,p_left,-1,Color(1,1,1,1),false,-1,Color::hex(0xFFFFFF00),frames->icon_port_top,nullptr);
			set_slot(2,false,-1,Color::hex(0xFFFFFF00),p_right,-1,Color(1,1,1,1),nullptr,frames->icon_port_bottom);
		}
	}
	void set_color(const Color &p_color)
	{
		set_input_color(p_color);
		set_output_color(p_color);
	}

	void set_input_color(const Color &p_color)
	{
		set_slot_color_left(horizontal ? 0 : 1,p_color);
	}

	void set_output_color(const Color &p_color)
	{
		set_slot_color_right(horizontal ? 2 : 1,p_color);
	}

	void _set_stylebox_override(const Ref<StyleBox> &panel_stylebox,const Ref<StyleBox>& titlebar_stylebox)
	{        
		add_theme_style_override("panel", panel_stylebox);
		add_theme_style_override("titlebar", titlebar_stylebox);
	}

	float get_layout_size()
	{
		return horizontal ? get_size().x : get_size().y;
	}

	void _on_size_changed()
	{
		add_theme_constant_override("port_offset", horizontal ? Math::round(get_size().x) : 12 * EDSCALE);
	}
	void set_text(String p_text)
	{
		text = p_text;
		text = String::num_int64(get_position_offset().x) + " " + String::num_int64(get_position_offset().y)
			+ "\n" + String::num_int64(get_position().x) + " " + String::num_int64(get_position().y);
		label->set_text(text);
	}
	void set_title_text(String p_text)
	{
		title_text = p_text;
		title_label->set_text(title_text);
	}
	void update_text()
	{
	
		String tile_name = beehave_node->get_lable_name();
		if(beehave_node->get_name().length() > 0)
		{
			tile_name += "\n (" + beehave_node->get_name() + ")";
		}
		set_title_text(tile_name);
	}
	virtual void on_selected(bool p_selected);
};

class BeehaveGraphTreeNode : public RefCounted
{
    public:
    float SIBLING_DISTANCE = 90.0;
    float LEVEL_DISTANCE = 180.0;
    float pos_x = 0;
    float pos_y = 0;
    float mod = 0;
    Vector2 total_size;
	
    BeehaveGraphTreeNode* parent = nullptr;
    LocalVector<Ref<BeehaveGraphTreeNode>> children;
    BeehaveGraphNodes* item = nullptr;

    bool _init(BeehaveGraphTreeNode* p_parent,BeehaveGraphNodes* p_item)
    {
        item = p_item;
        parent = p_parent;
        return true;
    }

    bool is_leaf()
    {
        return children.is_empty();
    }
    bool is_most_left()
    {
        if(parent == nullptr)
        {
            return true;
        }
        return parent->children[0] == this;
    }
    bool is_most_right()
    {
        if(parent == nullptr)
        {
            return true;
        }
        return parent->children[parent->children.size() - 1] == this;
    }

    Ref<BeehaveGraphTreeNode> get_previous_sibling()
    {
        if(parent == nullptr)
        {
            return Ref<BeehaveGraphTreeNode>();
        }
        for(uint32_t i = 0; i < parent->children.size(); i++)
        {
            if(parent->children[i] == this)
            {
                if(i == 0)
                {
                    return Ref<BeehaveGraphTreeNode>();
                }
                return parent->children[i - 1];
            }
        }
        return Ref<BeehaveGraphTreeNode>();
    }

    Ref<BeehaveGraphTreeNode> get_next_sibling()
    {
        if(parent == nullptr)
        {
            return Ref<BeehaveGraphTreeNode>();
        }
        for(uint32_t i = 0; i < parent->children.size(); i++)
        {
            if(parent->children[i] == this)
            {
                if(i == parent->children.size() - 1)
                {
                    return Ref<BeehaveGraphTreeNode>();
                }
                return parent->children[i + 1];
            }
        }
        return Ref<BeehaveGraphTreeNode>();
    }

    Ref<BeehaveGraphTreeNode> get_most_left_sibling()
    {
        if(parent == nullptr)
        {
            return Ref<BeehaveGraphTreeNode>();
        }
        if(is_most_left())
        {
            return this;
        }
        return parent->children[0];
    }
    Ref<BeehaveGraphTreeNode> get_most_left_child()
    {
        if(children.is_empty())
        {
            return Ref<BeehaveGraphTreeNode>();
        }
        return children[0];
    }
    Ref<BeehaveGraphTreeNode> get_most_right_child()
    {
        if(children.is_empty())
        {
            return Ref<BeehaveGraphTreeNode>();
        }
        return children[children.size() - 1];
    }

	void update_bound_size(bool p_horizontal_layout)
	{
        for(uint32_t i = 0; i < children.size(); i++)
        {
            children[i]->update_bound_size(p_horizontal_layout);
        }
        if(children.size() > 0)
        {
            total_size = Vector2(0,0);
            for(uint32_t i = 0; i < children.size(); i++)
            {
                total_size += children[i]->total_size;
            }
            if(p_horizontal_layout)
            {
                total_size.y += SIBLING_DISTANCE * (children.size() - 1);
            }
            else
            {
                total_size.x += SIBLING_DISTANCE * (children.size() - 1);
            }
        }
        else
        {
            total_size = get_cell_size();
        }
	}
	Vector2 get_cell_size()
	{
		return Vector2(180, 90);
	}

    void compute_position(const Vector2& p_offset,bool p_horizontal_layout)
    {
        
        if(p_horizontal_layout)
        {
            pos_x = p_offset.x;
			pos_y = p_offset.y + (total_size.y / 2);
            Vector2 offset = p_offset;
            offset.x += get_cell_size().x + LEVEL_DISTANCE;
            for(uint32_t i = 0; i < children.size(); i++)
            {
                children[i]->compute_position(offset,p_horizontal_layout);
                offset.y += children[i]->total_size.y + SIBLING_DISTANCE;
            }
        }
        else
        {
			pos_x = p_offset.x + total_size.x / 2;
			pos_y = p_offset.y;
            Vector2 offset = p_offset;
            offset.y += get_cell_size().y + LEVEL_DISTANCE;
            for(uint32_t i = 0; i < children.size(); i++)
            {
                children[i]->compute_position(offset,p_horizontal_layout);
                offset.x += children[i]->total_size.x + SIBLING_DISTANCE;
            }
        }
    }

    void update_position(bool p_horizontal_layout)
    {
        update_bound_size(p_horizontal_layout);
        Vector2 offset = Vector2(0,0);
        compute_position(offset,p_horizontal_layout);

    }


};
