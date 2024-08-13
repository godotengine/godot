#pragma once

#include "modules/game_help/logic/beehave/beehave_tree.h"
#include "scene/resources/style_box_flat.h"
#include "scene/resources/image_texture.h"
#include "editor/editor_node.h"
#include "editor/editor_properties.h"
#include "editor/inspector_dock.h"
#include "scene/gui/graph_edit_arranger.h"
#include "scene/gui/view_panner.h"
#include "scene/gui/graph_edit.h"

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
	Ref<Font> font;
	float scale = 1.0f;
	BeehaveGraphFrames()
	{

	}
	void init(Control* p_control)
	{
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

	void _init(Ref<BeehaveGraphFrames> p_frames,const StringName& p_icon, bool p_horizontal)
	{
		frames = p_frames;
		horizontal = p_horizontal;
		icon = get_editor_theme_icon(p_icon);
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


		connect("_on_size_changed", callable_mp(this, &BeehaveGraphNodes::_on_size_changed));

	}

	virtual void draw_port(int p_slot_index, Point2i p_pos, bool p_left, const Color &p_color) override
	{
		if(horizontal)
		{
			if(is_slot_enabled_left(1))
			{
				draw_texture(frames->icon_port_left,Vector2(0,get_size().y/2) + Vector2(-4,-5),p_color);
			}
			if(is_slot_enabled_right(1))
			{
				draw_texture(frames->icon_port_right,Vector2(get_size().x, get_size().y/2) + Vector2(4,-5),p_color);
			}
		}
		else
		{
			if(p_slot_index == 0 && is_slot_enabled_left(0))
			{
				draw_texture(frames->icon_port_top,Vector2(get_size().x/2,0) + Vector2(-4.5f,-7),p_color);
			}
			else if(p_slot_index == 1 )
			{
				draw_texture(frames->icon_port_bottom,Vector2(get_size().x/2,get_size().y) + Vector2(-4.5f,5),p_color);
			}
		}
	}
	Vector2 get_custom_input_port_position(bool p_horizontal) const
	{
		if(p_horizontal)
		{
			return Vector2(0, get_size().y/2);
		}
		return Vector2(get_size().x/2, 0);
	}

	Vector2 get_custom_output_port_position(bool p_horizontal) const
	{
		if(p_horizontal)
		{
			return Vector2(get_size().x, get_size().y/2);
		}
		return Vector2(get_size().x/2, get_size().y);
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
		add_theme_constant_override("port_offset", horizontal ? Math::round(get_size().x) : 12 * frames->scale);
	}
	void set_text(String p_text)
	{
		text = p_text;
		label->set_text(text);
	}
};
class BeehaveGraphEditor : public GraphEdit
{

	GDCLASS(BeehaveGraphEditor, GraphEdit);

	class TreeNode : public RefCounted
	{
		public:
		float SIBLING_DISTANCE = 20.0;
		float LEVEL_DISTANCE = 40.0;
		float x = 0;
		float y = 0;
		float mod = 0;
		TreeNode* parent = nullptr;
		LocalVector<Ref<TreeNode>> children;
		BeehaveGraphNodes* item = nullptr;

		bool _init(TreeNode* p_parent,BeehaveGraphNodes* p_item)
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

		Ref<TreeNode> get_previous_sibling()
		{
			if(parent == nullptr)
			{
				return Ref<TreeNode>();
			}
			for(uint32_t i = 0; i < parent->children.size(); i++)
			{
				if(parent->children[i] == this)
				{
					if(i == 0)
					{
						return Ref<TreeNode>();
					}
					return parent->children[i - 1];
				}
			}
			return Ref<TreeNode>();
		}

		Ref<TreeNode> get_next_sibling()
		{
			if(parent == nullptr)
			{
				return Ref<TreeNode>();
			}
			for(uint32_t i = 0; i < parent->children.size(); i++)
			{
				if(parent->children[i] == this)
				{
					if(i == parent->children.size() - 1)
					{
						return Ref<TreeNode>();
					}
					return parent->children[i + 1];
				}
			}
			return Ref<TreeNode>();
		}

		Ref<TreeNode> get_most_left_sibling()
		{
			if(parent == nullptr)
			{
				return Ref<TreeNode>();
			}
			if(is_most_left())
			{
				return this;
			}
			return parent->children[0];
		}
		Ref<TreeNode> get_most_left_child()
		{
			if(children.is_empty())
			{
				return Ref<TreeNode>();
			}
			return children[0];
		}
		Ref<TreeNode> get_most_right_child()
		{
			if(children.is_empty())
			{
				return Ref<TreeNode>();
			}
			return children[children.size() - 1];
		}

		void update_position(bool p_horizontal_layout)
		{
			_initialize_nodes(this,0);
			_calculate_initial_x(this);

			_check_all_children_on_screen(this);
			_calculate_final_positions(this,0);

			if(p_horizontal_layout)
			{
				_swap_x_y(this);
				_calculate_x(this,0);
			}
			else
			{
				_calculate_y(this,0);
			}

		}

		void _initialize_nodes(Ref<TreeNode> p_node,int depth)
		{
			p_node->x = -1;
			p_node->y = depth;
			p_node->mod = 0;

			for(uint32_t i = 0; i < p_node->children.size(); i++)
			{
				_initialize_nodes(p_node->children[i],depth + 1);
			}
		}

		void _calculate_initial_x(Ref<TreeNode> p_node)
		{
			for(uint32_t i = 0; i < p_node->children.size(); i++)
			{
				_calculate_initial_x(p_node->children[i]);
			}
			if(p_node->is_leaf())
			{
				if(!p_node->is_most_left())
				{
					Ref<TreeNode> p_node_prev = p_node->get_previous_sibling();
					p_node->x = p_node_prev->x + p_node_prev->item->get_layout_size() + SIBLING_DISTANCE;
				}
				else
				{
					p_node->x = 0;
				}
			}
			else
			{
				float mid = 0;
				if(p_node->children.size() == 1)
				{
					float offset = p_node->children[0]->item->get_layout_size() - p_node->item->get_layout_size();
					offset /= 2;
					mid = p_node->children[0]->x + offset;
				}
				else
				{
					Ref<TreeNode> p_node_most_left = p_node->get_most_left_child();
					Ref<TreeNode> p_node_most_right = p_node->get_most_right_child();
					mid = (p_node_most_left->x + p_node_most_right->x - p_node_most_left->item->get_layout_size()) / 2;
				}

				if(p_node->is_most_left())
				{
					p_node->x = mid;
				}
				else
				{
					p_node->x = p_node->get_previous_sibling()->x + p_node->get_previous_sibling()->item->get_layout_size() + SIBLING_DISTANCE;
				}
				p_node->mod = p_node->x - mid;
			}


			if(!p_node->is_leaf() && !p_node->is_most_left())
			{
				_check_for_conflicts(p_node);
			}
		}

		void _calculate_final_positions(Ref<TreeNode> p_node,float mod_sum)
		{
			p_node->x += mod_sum;
			mod_sum += p_node->mod;
			for(uint32_t i = 0; i < p_node->children.size(); i++)
			{
				_calculate_final_positions(p_node->children[i],mod_sum);
			}
		}

		void _check_all_children_on_screen(Ref<TreeNode> p_node)
		{
			HashMap<uint32_t,float> node_contour;
			_get_left_contour(p_node,0,node_contour);
			float shift_amount = 0;
			for(auto& it : node_contour)
			{
				if(it.value + shift_amount < 0)
				{
					shift_amount = -it.value;
				}
			}
			if(shift_amount > 0)
			{
				p_node->x += shift_amount;
				p_node->mod += shift_amount;
			}
		}

		void _check_for_conflicts(Ref<TreeNode> p_node)
		{
			float min_distance = SIBLING_DISTANCE;
			float shift_value = 0;
			Ref<TreeNode> shift_sibling;


			HashMap<uint32_t,float> node_contour;
			_get_left_contour(p_node,0,node_contour);

			Ref<TreeNode> sibling = p_node->get_most_left_sibling();

			while(sibling.is_valid() && sibling != p_node)
			{
				HashMap<uint32_t,float> sibling_contour;
				_get_right_contour(sibling,0,sibling_contour);

				for(uint32_t level = p_node->y; level < MIN(sibling_contour.size(),node_contour.size()); level++)
				{

					float distance = node_contour[level] - sibling_contour[level];
					if(distance + shift_value < min_distance)
					{
						shift_value = min_distance - distance;
						shift_sibling = sibling;
					}
				}
				sibling = sibling->get_next_sibling();
			}
			if(shift_value > 0)
			{
				p_node->x += shift_value;
				p_node->mod += shift_value;
				if(shift_sibling.is_valid())
				{
					_center_nodes_between(shift_sibling,p_node);
				}
			}

		}
		void _center_nodes_between(Ref<TreeNode> p_left_node,Ref<TreeNode> p_right_node)
		{
			auto left_index = p_left_node->children.find(p_right_node);
			auto right_index = p_right_node->children.find(p_left_node);

			int num_nodes_between = right_index - left_index - 1;
			if(num_nodes_between > 0)
			{
				// The extra distance that needs to be split into num_nodes_between + 1
				// in order to find the new node spacing so that nodes are equally spa
				float distance_to_allocate = p_right_node->x - p_left_node->x - p_left_node->item->get_layout_size();
				// Subtract sizes on nodes in between
				for(int i = left_index + 1; i < right_index; i++)
				{
					distance_to_allocate -= p_left_node->children[i]->item->get_layout_size();
				}
				float distance_between_nodes = distance_to_allocate / (num_nodes_between + 1);

				Ref<TreeNode> prev_node = p_left_node;
				Ref<TreeNode> middle_node = p_left_node->get_next_sibling();

				while(middle_node != p_right_node)
				{
					float desired_x = prev_node->x + prev_node->item->get_layout_size() + distance_between_nodes;
					float offset = desired_x - middle_node->x;
					middle_node->x += offset;
					middle_node->mod += offset;
					prev_node = middle_node;
					middle_node = middle_node->get_next_sibling();	
				}
			}
		}

		void _get_left_contour(Ref<TreeNode> p_node,float mod_sum,HashMap<uint32_t,float>& values)
		{
			float node_left = p_node->x + mod_sum;
			int depth = p_node->y;
			if(!values.has(depth))
			{
				values[depth] = node_left;
			}
			else
			{
				values[depth] = MIN(values[depth],node_left);
			}
			for(uint32_t i = 0; i < p_node->children.size(); i++)
			{
				_get_left_contour(p_node->children[i],mod_sum + p_node->mod,values);
			}
		}
		void _get_right_contour(Ref<TreeNode> p_node,float mod_sum,HashMap<uint32_t,float>& values)
		{
			float node_right = p_node->x + mod_sum + p_node->item->get_layout_size();
			int depth = p_node->y;
			if(!values.has(depth))
			{
				values[depth] = node_right;
			}
			else
			{
				values[depth] = MAX(values[depth],node_right);
			}
			for(uint32_t i = 0; i < p_node->children.size(); i++)
			{
				_get_right_contour(p_node->children[i],mod_sum + p_node->mod,values);
			}
		}

		void _swap_x_y(Ref<TreeNode> p_node)
		{
			for(uint32_t i = 0; i < p_node->children.size(); i++)
			{
				_swap_x_y(p_node->children[i]);
			}

			int temp = p_node->x;
			p_node->x = p_node->y;
			p_node->y = temp;
		}

		void _calculate_x(Ref<TreeNode> p_node,int offset)
		{
			p_node->x = offset;
			Ref<TreeNode> sibling = p_node->get_next_sibling();
			int max_size = p_node->item->get_size().x;
			while(sibling.is_valid())
			{
				max_size = MAX(max_size,sibling->item->get_size().x);
				sibling = sibling->get_next_sibling();
			}

			for(uint32_t i = 0; i < p_node->children.size(); i++)
			{
				_calculate_x(p_node->children[i],offset + max_size);
			}
		}

		void _calculate_y(Ref<TreeNode> p_node,int offset)
		{
			p_node->y = offset;
			Ref<TreeNode> sibling = p_node->get_next_sibling();
			int max_size = p_node->item->get_size().y;
			while(sibling.is_valid())
			{
				max_size = MAX(max_size,sibling->item->get_size().y);
				sibling = sibling->get_next_sibling();
			}

		}

	};

public:
	void set_beehave_tree(Ref<BeehaveTree> p_beehave_tree)
	{
		if(beehave_tree == p_beehave_tree)
		{
			return;
		}
		active_nodes.clear();
		beehave_tree = p_beehave_tree;
		_update_graph();
	}
	void _ready() override
	{
		frames.instantiate();
		frames->init(this);
		set_minimap_enabled(false);

		layout_button = memnew(Button);
		layout_button->set_flat(false);
		layout_button->set_focus_mode(FOCUS_NONE);
		layout_button->connect("pressed", callable_mp(this, &BeehaveGraphEditor::on_layout_changed));

		get_menu_hbox()->add_child(layout_button);

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
		updating_graph = false;

	}

	void _add_nodes(const Ref<BeehaveNode>& p_beehave_node)
	{
		if(p_beehave_node.is_null())
		{
			return;
		}
		BeehaveGraphNodes* nodes = memnew(BeehaveGraphNodes);
		nodes->set_title(p_beehave_node->get_lable_name());
		nodes->set_name(p_beehave_node->get_id());
		nodes->_init(frames,p_beehave_node->get_icon(),horizontal_layout);

		add_child(nodes);
		if(Object::cast_to<BeehaveLeaf>(p_beehave_node.ptr()))
		{
			nodes->set_slots(true,false);
		}
		else if(Object::cast_to<BeehaveComposite>(p_beehave_node.ptr()) || Object::cast_to<BeehaveDecorator>(p_beehave_node.ptr()))
		{
			nodes->set_slots(true,true);
		}

		for(int i = 0; i < p_beehave_node->get_child_count(); i++) 
		{
			_add_nodes(p_beehave_node->get_child(i)); 
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

		Ref<TreeNode> tree_node = p_beehave_tree->get_root_node();
		tree_node->update_position(horizontal_layout);
		_place_nodes(tree_node);

		arraging_nodes = false;
	}
	Ref<TreeNode> _create_tree_nodes(const Ref<BeehaveNode>& p_beehave_node,const Ref<TreeNode>& p_parent)
	{
		Ref<TreeNode> tree_node = memnew(TreeNode);
		Node* n = get_node(NodePath(p_beehave_node->get_id()));
		tree_node->_init(p_parent.ptr(), Object::cast_to<BeehaveGraphNodes>(n ));

		for(int i = 0; i < p_beehave_node->get_child_count(); i++)
		{
			Ref<TreeNode> child = _create_tree_nodes(p_beehave_node->get_child(i),tree_node);
			tree_node->children.push_back(child);
		}
		return tree_node;
	}
	void _place_nodes(Ref<TreeNode> p_tree_node)
	{
		p_tree_node->item->set_position_offset(Vector2( p_tree_node->x,p_tree_node->y));

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

	void process_begin(StringName id_name)
	{
		TypedArray<BeehaveGraphNodes> nodes = _get_child_nodes();
		for(int i = 0; i < nodes.size(); ++i)
		{
			BeehaveGraphNodes* node = Object::cast_to<BeehaveGraphNodes>(nodes[i]);
			if(node->get_name() == id_name)
			{
				node->set_meta("status",-1);
			}
		}
	}

	void process_tick(StringName id_name,int p_status)
	{
		BeehaveGraphNodes* nodes = Object::cast_to<BeehaveGraphNodes>(get_node_or_null(NodePath( id_name)));
		if(nodes != nullptr)
		{
			nodes->set_meta("status",p_status);
			nodes->set_status(p_status);
			nodes->set_text(get_status(p_status));

			if(p_status == 0 || p_status == 2)
			{
				if(!active_nodes.has(nodes))
				{
					active_nodes.push_back(nodes);
				}
			}
		}

	}

	void process_end(StringName id_name)
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
		}
	}

	TypedArray<BeehaveGraphNodes> _get_child_nodes()
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

	PackedVector2Array _get_connection_line(Vector2 p_from, Vector2 p_to)
	{
		TypedArray<BeehaveGraphNodes> nodes = _get_child_nodes();
		for(int i = 0; i < nodes.size(); ++i)
		{
			BeehaveGraphNodes* child = Object::cast_to<BeehaveGraphNodes>(nodes[i]);
			for(int port = 0; port < child->get_input_port_count(); ++port)
			{
				if (! (child->get_position_offset() + child->get_input_port_position(port)).is_equal_approx(p_to))
				{
					continue;
				}
				p_to = child->get_position_offset() + child->get_custom_input_port_position(horizontal_layout);

			}

			for(int port = 0; port < child->get_output_port_count(); ++port)
			{
				if( ! (child->get_position_offset() + child->get_output_port_position(port)).is_equal_approx(p_from))
					continue;
				p_from = child->get_position_offset() + child->get_custom_output_port_position(horizontal_layout);
			}
		}
		return _get_elbow_connection_line(p_from, p_to);
	}

	PackedVector2Array _get_elbow_connection_line(Vector2 p_from, Vector2 p_to)
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
	void _process(float delta)
	{
		if(!active_nodes.is_empty())
		{
			progress += (delta >= 0.05) ? 10 : 1;
			if(progress >= 1000)
			{
				progress = 0;
			}
			queue_redraw();
		}
	}

	void _draw()
	{
		if(!active_nodes.is_empty())
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

			if(!active_nodes.has(from_node) || !active_nodes.has(to_node))
			{
				continue;
			}
			
			if((int)from_node->get_meta("status") < 0 || (int)to_node->get_meta("status") < 0)
			{
				return;
			}

			Vector2 output_port_position;
			Vector2 input_port_position;


			float scale_factor = from_node->get_rect().size.x / from_node->get_size().x;

			auto line = _get_elbow_connection_line(
				from_node->get_position() + from_node->get_custom_output_port_position(horizontal_layout) * scale_factor,
				to_node->get_position() + to_node->get_custom_input_port_position(horizontal_layout) * scale_factor
			);

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

	void _update_layout_button()
	{
		if(horizontal_layout)
		{
			layout_button->set_icon(get_editor_theme_icon("horizontal_layout"));
		}
		else
		{
			layout_button->set_icon(get_editor_theme_icon("vertical_layout"));
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
	Button* layout_button;
};

// 设置编辑的对象
class BeehaveNodeChildChildEditor : public EditorPropertyArray {
	GDCLASS(BeehaveNodeChildChildEditor, EditorPropertyArray);


public:
	void setup(Ref<BeehaveNode> p_beehave_node,StringName property_name,Variant::Type p_array_type, const String &p_hint_string = "")
	{
		beehave_node = p_beehave_node;
		set_object_and_property(p_beehave_node.ptr(),property_name);
		base_class_type::setup(p_array_type,p_hint_string);
		// 关闭内置的增加节点按钮
		button_add_item->set_visible(false);
	}	void on_reorder_button_gui_input(const Ref<InputEvent> &p_event)
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
		int idx = slots.size();
		HBoxContainer *hbox = memnew(HBoxContainer);

		Button *reorder_button = memnew(Button);
		reorder_button->set_icon(get_editor_theme_icon(SNAME("TripleBar")));
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
			edit_btn->set_icon(get_editor_theme_icon(SNAME("Edit")));
			edit_btn->set_disabled(is_read_only());
			edit_btn->connect(SceneStringName(pressed), callable_mp(this, &BeehaveNodeChildChildEditor::on_change_type).bind(edit_btn, idx));
			hbox->add_child(edit_btn);
		} else {
			Button *remove_btn = memnew(Button);
			remove_btn->set_icon(get_editor_theme_icon(SNAME("Remove")));
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
		slot.state_button = nullptr;
		slot.set_index(idx + page_index * page_length);
		slots.push_back(slot);
	}
protected:
	Ref<BeehaveNode> beehave_node;

};



class BeehaveGraphProperty : public VBoxContainer
{
	GDCLASS(BeehaveGraphProperty, VBoxContainer);
public:
	BeehaveGraphProperty()
	{
		set_custom_minimum_size(Vector2(300,200));

		beehave_editor = memnew(BeehaveGraphEditor);
		add_child(beehave_editor);
		beehave_editor->set_layout_mode(LayoutMode::LAYOUT_MODE_CONTAINER);
		beehave_editor->set_v_size_flags(Control::SIZE_EXPAND_FILL);


		sub_inspector = memnew(EditorInspector);
		
		sub_inspector->set_vertical_scroll_mode(ScrollContainer::SCROLL_MODE_DISABLED);
		sub_inspector->set_use_doc_hints(true);

		sub_inspector->set_sub_inspector(true);
		sub_inspector->set_property_name_style(InspectorDock::get_singleton()->get_property_name_style());

		sub_inspector->connect("property_keyed", callable_mp(this, &BeehaveGraphProperty::_sub_inspector_property_keyed));
		sub_inspector->connect("resource_selected", callable_mp(this, &BeehaveGraphProperty::_sub_inspector_resource_selected));
		sub_inspector->connect("object_id_selected", callable_mp(this, &BeehaveGraphProperty::_sub_inspector_object_id_selected));
		sub_inspector->set_keying(true);
		sub_inspector->set_read_only(false);
		sub_inspector->set_use_folding(false);

		sub_inspector->set_mouse_filter(MOUSE_FILTER_STOP);

		add_child(sub_inspector);


		child_list = memnew(BeehaveNodeChildChildEditor);
		beehave_editor->add_child(child_list);
		child_list->setup(beehave_node, "children", Variant::OBJECT,  MAKE_RESOURCE_TYPE_HINT("BeehaveNode"));
		child_list->set_layout_mode(LayoutMode::LAYOUT_MODE_CONTAINER);
		child_list->set_visible(false);
	}
	void setup(Ref<BeehaveTree> p_beehave_tree)
	{
		beehave_tree = p_beehave_tree;
	}

	
	void BeehaveGraphProperty::_sub_inspector_property_keyed(const String &p_property, const Variant &p_value, bool p_advance) {
		// The second parameter could be null, causing the event to fire with less arguments, so use the pointer call which preserves it.
		const Variant args[3] = { String("children") + ":" + p_property, p_value, p_advance };
		const Variant *argp[3] = { &args[0], &args[1], &args[2] };
		emit_signalp(SNAME("property_keyed_with_value"), argp, 3);
	}

	void BeehaveGraphProperty::_sub_inspector_resource_selected(const Ref<RefCounted> &p_resource, const String &p_property) {
		emit_signal(SNAME("resource_selected"), String("children") + ":" + p_property, p_resource);
	}

	void BeehaveGraphProperty::_sub_inspector_object_id_selected(int p_id) {
		emit_signal(SNAME("object_id_selected"), "children", p_id);
	}
	void set_editor_node(Ref<BeehaveTree> p_beehave_tree,Ref<BeehaveNode> p_beehave_node)
	{
		beehave_tree = p_beehave_tree;
		beehave_node = p_beehave_node;
		beehave_tree->last_editor_id = beehave_node->get_instance_id();
		sub_inspector->edit(beehave_node.ptr());
	}

	void create_leaf_pop_sub()
	{
		leaf_pop_sub = memnew(PopupMenu);

		leaf_class_name.clear();
		HashSet<StringName> leaf_class_name_set;
		_add_allowed_type(SNAME("BeehaveLeaf"), &leaf_class_name_set);

		for (const StringName &S : leaf_class_name_set) {
			leaf_class_name.push_back(S);
			leaf_pop_sub->add_item(S);
		}
		leaf_pop_sub->connect(SceneStringName(id_pressed), callable_mp(this, &BeehaveGraphProperty::on_leaf_pop_pressed));

	}

	void create_composite_pop_sub()
	{
		composite_pop_sub = memnew(PopupMenu);
		composite_class_name.clear();
		HashSet<StringName> composite_class_name_set;

		_add_allowed_type(SNAME("BeehaveComposite"), &composite_class_name_set);
		for (const StringName &S : composite_class_name_set) {
			composite_class_name.push_back(S);
			composite_pop_sub->add_item(S);
		}

		composite_pop_sub->connect(SceneStringName(id_pressed), callable_mp(this, &BeehaveGraphProperty::on_composite_pop_pressed));
	}

	void create_decorator_pop_sub()
	{
		decorator_pop_sub = memnew(PopupMenu);
		decorator_class_name.clear();
		HashSet<StringName> decorator_class_name_set;
		_add_allowed_type(SNAME("BeehaveDecorator"), &decorator_class_name_set);
		for (const StringName &S : decorator_class_name_set) {
			decorator_class_name.push_back(S);
			decorator_pop_sub->add_item(S);
		}
		decorator_pop_sub->connect(SceneStringName(id_pressed), callable_mp(this, &BeehaveGraphProperty::on_decorator_pop_pressed));
	}

	void on_leaf_pop_pressed(int p_op)
	{
		Ref<BeehaveLeaf> leaf = create_class_instance(leaf_class_name[p_op]);
		if(leaf.is_valid())
		{
			beehave_node->add_child(leaf);
			beehave_tree->notify_property_list_changed();
		}
	}

	void on_composite_pop_pressed(int p_op)
	{
		Ref<BeehaveComposite> composite = create_class_instance(composite_class_name[p_op]);
		if(composite.is_valid())
		{
			beehave_node->add_child(composite);
			beehave_tree->notify_property_list_changed();
		}
	}
	
	void on_decorator_pop_pressed(int p_op)
	{
		Ref<BeehaveDecorator> decorator = create_class_instance(decorator_class_name[p_op]);
		if(decorator.is_valid())
		{
			beehave_node->add_child(decorator);
			beehave_tree->notify_property_list_changed();
		}
	}

	Ref<RefCounted> create_class_instance(StringName p_class_name)
	{
		Variant obj;

		if (ScriptServer::is_global_class(p_class_name)) {
			obj = EditorNode::get_editor_data().script_class_instance(p_class_name);
		} else {
			obj = ClassDB::instantiate(p_class_name);
		}

		if (!obj) {
			obj = EditorNode::get_editor_data().instantiate_custom_type(p_class_name, "Resource");
		}

		RefCounted *resp = Object::cast_to<RefCounted>(obj);
		

		return Ref<RefCounted>(resp);
	}
	
	static void _add_allowed_type(const StringName &p_type, HashSet<StringName> *p_vector) {
		if (p_vector->has(p_type)) {
			// Already added
			return;
		}

		if (ClassDB::class_exists(p_type)) {
			// Engine class,

			if (!ClassDB::is_virtual(p_type)) {
				p_vector->insert(p_type);
			}

			List<StringName> inheriters;
			ClassDB::get_inheriters_from_class(p_type, &inheriters);
			for (const StringName &S : inheriters) {
				_add_allowed_type(S, p_vector);
			}
		} else {
			// Script class.
			p_vector->insert(p_type);
		}

		List<StringName> inheriters;
		ScriptServer::get_inheriters_list(p_type, &inheriters);
		for (const StringName &S : inheriters) {
			_add_allowed_type(S, p_vector);
		}
	}
	
protected:
	Ref<BeehaveTree> beehave_tree;
	Ref<BeehaveNode> beehave_node;
	BeehaveGraphEditor* beehave_editor = nullptr;
	EditorInspector* sub_inspector = nullptr;
	BeehaveNodeChildChildEditor* child_list = nullptr;
	Button* buton_create_beehave_node = nullptr;
	PopupMenu* cteate_beehave_node_pop = nullptr;

	PopupMenu* leaf_pop_sub = nullptr;
	PopupMenu* composite_pop_sub = nullptr;
	PopupMenu* decorator_pop_sub = nullptr;


	LocalVector<StringName> leaf_class_name;
	LocalVector<StringName> composite_class_name;
	LocalVector<StringName> decorator_class_name;


};
