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
	Ref<ImageTexture> add_template_icon;
	
	Ref<ImageTexture> icon_port_top;
	Ref<ImageTexture> icon_port_bottom;
	Ref<ImageTexture> icon_port_left;
	Ref<ImageTexture> icon_port_right;


	Ref<ImageTexture> arrow_down_icon;
	Ref<ImageTexture> arrow_right_icon;

	Ref<ImageTexture> move_left_icon;
	Ref<ImageTexture> move_right_icon;
	Ref<ImageTexture> move_up_icon;
	Ref<ImageTexture> move_down_icon;

	
	Ref<ImageTexture> add_node_icon;
	Ref<ImageTexture> remove_node_icon;


	Ref<ImageTexture> debug_enable_icon;
	Ref<ImageTexture> debug_disable_icon;

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
		add_template_icon = p_control->get_theme_icon(SNAME("EditKey"));

		icon_port_top = p_control->get_editor_theme_icon(SNAME("port_top"));
		icon_port_bottom = p_control->get_editor_theme_icon(SNAME("port_bottom"));
		icon_port_left = p_control->get_editor_theme_icon(SNAME("port_left"));
		icon_port_right = p_control->get_editor_theme_icon(SNAME("port_right"));
		
		arrow_down_icon = p_control->get_editor_theme_icon(SNAME("GuiTreeArrowDown"));
		arrow_right_icon = p_control->get_editor_theme_icon(SNAME("GuiTreeArrowRight"));

		move_left_icon = p_control->get_editor_theme_icon(SNAME("MoveLeft"));
		move_right_icon = p_control->get_editor_theme_icon(SNAME("MoveRight"));
		move_up_icon = p_control->get_editor_theme_icon(SNAME("MoveUp"));
		move_down_icon = p_control->get_editor_theme_icon(SNAME("MoveDown"));

		add_node_icon = p_control->get_editor_theme_icon(SNAME("InsertBefore"));
		remove_node_icon = p_control->get_editor_theme_icon(SNAME("Remove"));


		debug_enable_icon = p_control->get_editor_theme_icon(SNAME("DebugSkipBreakpointsOff"));
		debug_disable_icon = p_control->get_editor_theme_icon(SNAME("DebugSkipBreakpointsOn"));


	}
};

class BeehaveGraphNodesCreatePopmenu : public RefCounted
{

	GDCLASS(BeehaveGraphNodesCreatePopmenu, RefCounted);
public:
	// 初始化会自动清除按钮的所有子节点
	void init(Button* p_parent, Ref<BeehaveNode> p_beehave_node)
	{
		while(p_parent->get_child_count() > 0)
		{
			p_parent->get_child(p_parent->get_child_count() - 1)->queue_free();
			p_parent->remove_child(p_parent->get_child(p_parent->get_child_count() - 1));
		}
		parent = p_parent;
		beehave_node = p_beehave_node;


		
		cteate_beehave_node_pop = memnew(PopupMenu);
		cteate_beehave_node_pop->set_auto_translate_mode(PopupMenu::AUTO_TRANSLATE_MODE_ALWAYS);
		cteate_beehave_node_pop->set_visible(false);
		parent->add_child(cteate_beehave_node_pop);

		
		create_leaf_pop_sub();
		create_composite_pop_sub();
		create_decorator_pop_sub();
		cteate_beehave_node_pop->add_submenu_node_item(L"叶节点", leaf_pop_sub);
		cteate_beehave_node_pop->add_submenu_node_item(L"组合节点", composite_pop_sub);
		cteate_beehave_node_pop->add_submenu_node_item(L"修饰器", decorator_pop_sub);
		cteate_beehave_node_pop->add_separator();
		cteate_beehave_node_pop->add_item(L"添加模板子节点");
		cteate_beehave_node_pop->add_item(L"添加到模板列表");
	}
	void popup_on_target() {
		cteate_beehave_node_pop->reset_size();
		Rect2i usable_rect =  Rect2i(Point2i(0,0), DisplayServer::get_singleton()->window_get_size_with_decorations());
		Rect2i cp_rect = Rect2i(Point2i(0,0), parent->get_size());

		for(int i = 0; i < 4; i++) {
			if(i > 1)
			{
				cp_rect.position.y = parent->get_global_position().x - parent->get_size().y;
			}
			else
			{
				cp_rect.position.y = parent->get_global_position().y + parent->get_size().y;
			}
			if(i & 1) {
				cp_rect.position.x = parent->get_global_position().x ;
			}
			else
			{
				cp_rect.position.x = parent->get_global_position().x - MAX(0,cp_rect.size.x - parent->get_size().x);
			}
			if(usable_rect.encloses(cp_rect))
			{
				break;
			}
		}
		Point2i main_window_position = DisplayServer::get_singleton()->window_get_position();
		Point2i popup_position = main_window_position + Point2i(cp_rect.position);
		cteate_beehave_node_pop->set_position(popup_position);
		cteate_beehave_node_pop->popup();
		
	}
	Callable on_node_changed;
protected:

	void on_leaf_pop_pressed(int p_op)
	{
		Ref<BeehaveLeaf> leaf = create_class_instance(leaf_class_name[p_op]);
		if(leaf.is_valid())
		{
			beehave_node->add_child(leaf);
			on_node_changed.call();
		}
	}

	void on_composite_pop_pressed(int p_op)
	{
		Ref<BeehaveComposite> composite = create_class_instance(composite_class_name[p_op]);
		if(composite.is_valid())
		{
			beehave_node->add_child(composite);
			on_node_changed.call();
		}
	}
	
	void on_decorator_pop_pressed(int p_op)
	{
		Ref<BeehaveDecorator> decorator = create_class_instance(decorator_class_name[p_op]);
		if(decorator.is_valid())
		{
			beehave_node->add_child(decorator);
			on_node_changed.call();
		}
	}

protected:
	// 增加节点到模板
	void on_add_node_to_template(const String& p_template_group, const String& p_template_name,const String& p_template_dis){

	}
	void create_leaf_pop_sub()
	{
		leaf_pop_sub = memnew(PopupMenu);

		leaf_class_name.clear();
		HashSet<StringName> leaf_class_name_set;
		_add_allowed_type(SNAME("BeehaveLeaf"), &leaf_class_name_set);
		int i = 0;
		for (const StringName &S : leaf_class_name_set) {
			leaf_class_name.push_back(S);
			Ref<BeehaveLeaf> instance = create_class_instance(S);
			leaf_pop_sub->add_item(instance->get_lable_name());
			leaf_pop_sub->set_item_tooltip(i, instance->get_tooltip());
			leaf_class_name_to_instance[S] = instance;
			++i;
		}
		leaf_pop_sub->connect(SceneStringName(id_pressed), callable_mp(this, &BeehaveGraphNodesCreatePopmenu::on_leaf_pop_pressed));

	}
	void create_composite_pop_sub()
	{
		composite_pop_sub = memnew(PopupMenu);
		composite_class_name.clear();
		HashSet<StringName> composite_class_name_set;

		_add_allowed_type(SNAME("BeehaveComposite"), &composite_class_name_set);
		int i = 0;
		for (const StringName &S : composite_class_name_set) {
			composite_class_name.push_back(S);
			Ref<BeehaveComposite> instance = create_class_instance(S);
			composite_pop_sub->add_item(instance->get_lable_name());
			composite_pop_sub->set_item_tooltip(i, instance->get_tooltip());
			composite_class_name_to_instance[S] = instance;
			++i;
		}

		composite_pop_sub->connect(SceneStringName(id_pressed), callable_mp(this, &BeehaveGraphNodesCreatePopmenu::on_composite_pop_pressed));
	}

	void create_decorator_pop_sub()
	{
		decorator_pop_sub = memnew(PopupMenu);
		decorator_class_name.clear();
		HashSet<StringName> decorator_class_name_set;
		_add_allowed_type(SNAME("BeehaveDecorator"), &decorator_class_name_set);
		int i = 0;
		for (const StringName &S : decorator_class_name_set) {
			decorator_class_name.push_back(S);
			Ref<BeehaveDecorator> instance = create_class_instance(S);
			decorator_pop_sub->add_item(instance->get_lable_name());
			decorator_pop_sub->set_item_tooltip(i, instance->get_tooltip());
			decorator_class_name_to_instance[S] = instance;
			++i;
		}
		decorator_pop_sub->connect(SceneStringName(id_pressed), callable_mp(this, &BeehaveGraphNodesCreatePopmenu::on_decorator_pop_pressed));
	}

protected:		
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

			if (!ClassDB::is_virtual(p_type) && !ClassDB::is_abstract(p_type)) {
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
	Control* parent = nullptr;
	Ref<BeehaveNode> beehave_node ;
	PopupMenu* cteate_beehave_node_pop = nullptr;

	PopupMenu* leaf_pop_sub = nullptr;
	PopupMenu* composite_pop_sub = nullptr;
	PopupMenu* decorator_pop_sub = nullptr;

	LocalVector<StringName> leaf_class_name;
	HashMap<StringName,Ref<BeehaveLeaf>> leaf_class_name_to_instance;
	LocalVector<StringName> composite_class_name;
	HashMap<StringName,Ref<BeehaveComposite>> composite_class_name_to_instance;
	LocalVector<StringName> decorator_class_name;
	HashMap<StringName,Ref<BeehaveDecorator>> decorator_class_name_to_instance;
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
	Button* debug_break;
	Label* title_label;
	Label* label;
	Label* state_label;
	HBoxContainer* titlebar_hbox;

	CheckBox* enable = nullptr;
	Button* move_up = nullptr;
	Button* move_down = nullptr;
	Button* create_child_node = nullptr;
	Button* delete_node = nullptr;
	Button* child_collapsed = nullptr;
	AcceptDialog* dialog = nullptr;

	Ref<BeehaveGraphFrames> frames;
	Ref<BeehaveGraphNodesCreatePopmenu> cteate_beehave_node_pop;
	Ref<BeehaveNode> parent_beehave_node;
	Ref<BeehaveNode> beehave_node;
	BeehaveGraphProperty* beehave_graph_property = nullptr;
	bool horizontal = false;
	BeehaveGraphNodes()
	{
		
	}
	void set_title_color(Color p_color)
	{
		title_label->set_modulate(p_color);
	}
	void _init(BeehaveGraphProperty *p_beehave_graph_property,const Ref<BeehaveNode>& p_parent_beehave_node,const Ref<BeehaveNode>& p_beehave_node,Ref<BeehaveGraphFrames> p_frames,const String& p_title_text,const String& p_text, const StringName& p_icon, bool p_horizontal)
	{

		beehave_graph_property = p_beehave_graph_property;
		parent_beehave_node = p_parent_beehave_node;
		beehave_node = p_beehave_node;
		title_text = p_title_text;
		text = p_text;
		frames = p_frames;
		horizontal = p_horizontal;
		EditorBottomPanel* p_control = EditorNode::get_bottom_panel();
		icon = p_control->get_editor_theme_icon(p_icon);
		set_draggable(false);
		if(Object::cast_to<BeehaveNodeTemplateRef>(beehave_node.ptr()))
		{
			set_modulate(Color(0.5, 0.8, 0.9, 1));
		}
	}
	void _ready();

	virtual void draw_port(int p_slot_index, Point2i p_pos, bool p_left, const Color &p_color) override
	{
		if(horizontal)
		{
			if(is_slot_enabled_left(1))
			{
				draw_texture(frames->icon_port_left, get_input_port_position(0) + Vector2(-4, -7),p_color);
			}
			if(is_slot_enabled_right(1) && !beehave_node->get_editor_collapsed_children())
			{
				draw_texture(frames->icon_port_right, get_output_port_position(0) + Vector2(-5, -7),p_color);
			}
		}
		else
		{
			if(p_slot_index == 0 && is_slot_enabled_left(0))
			{
				draw_texture(frames->icon_port_top, get_input_port_position(0) + Vector2(-5, -5),p_color);
			}
			else if(p_slot_index == 1  && !beehave_node->get_editor_collapsed_children())
			{
				draw_texture(frames->icon_port_bottom, get_output_port_position(0) + Vector2(-5, -7),p_color);
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
		set_slot_color_left(horizontal ? 1 : 0,p_color);
	}

	void set_output_color(const Color &p_color)
	{
		set_slot_color_right(horizontal ? 1 : 2,p_color);
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
		//label->set_text(beehave_node->get_annotation());
		label->set_text(String(L"") + String::num_int64(get_size().x) + " " + String::num_int64(get_size().y));
		state_label->set_text(text);
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
	void _on_enable(bool p_pressed)
	{
		beehave_node->set_enable(p_pressed);
		on_beehave_node_change();
	}
	void _on_move_up()
	{
		parent_beehave_node->move_child_up(beehave_node);
		on_beehave_node_change();
		on_selected(true);
	}

	void _on_move_down()
	{
		parent_beehave_node->move_child_down(beehave_node);
		on_beehave_node_change();		
		on_selected(true);
	}
	void _on_create_child_node()
	{
		if (cteate_beehave_node_pop.is_null())
		{
			cteate_beehave_node_pop.instantiate();
			cteate_beehave_node_pop->init(create_child_node,  beehave_node);
			cteate_beehave_node_pop->on_node_changed = callable_mp(this, &BeehaveGraphNodes::on_beehave_node_change);
		}
		cteate_beehave_node_pop->popup_on_target();

	}

	void _on_delete_node()
	{
		if(dialog == nullptr)
		{
			
			dialog = memnew(AcceptDialog);
			dialog->set_title(L"是否删除节点");
			Label* dialog_msg = memnew(Label);
			dialog_msg->set_text(L"删除节点会删除所有子节点，确认删除吗？");
			dialog->add_child(dialog_msg);
			delete_node->add_child(dialog);
			dialog->connect(SceneStringName(confirmed), callable_mp(this, &BeehaveGraphNodes::_delete_node));
		}
		dialog->popup_centered_ratio(0.2);
	}
	void _on_child_collapsed()
	{
		beehave_node->set_editor_collapsed_children(! beehave_node->get_editor_collapsed_children());
		on_beehave_node_change();
	}
	void _delete_node()
	{
		parent_beehave_node->remove_child(beehave_node);
		on_beehave_node_change();
	}
	void _on_debug_break();
	virtual void on_selected(bool p_selected);
	virtual void on_beehave_node_change();
};

class BeehaveGraphTreeNode : public RefCounted
{
    public:
    float SIBLING_DISTANCE = 30.0f;
    float LEVEL_DISTANCE = 30.0f;
    float pos_x = 0;
    float pos_y = 0;
    float mod = 0;
    float total_size = 0;
	
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
            total_size = 0;
            for(uint32_t i = 0; i < children.size(); i++)
            {
                total_size += children[i]->total_size;
            }
            if(p_horizontal_layout)
            {
                total_size += SIBLING_DISTANCE * (children.size() - 1);
            }
            else
            {
                total_size += SIBLING_DISTANCE * (children.size() - 1);
            }
        }
        else
        {
			if (p_horizontal_layout)
			{
				total_size += get_cell_size().y;
			}
			else
			{
				total_size += get_cell_size().x;
			}
        }
	}
	Vector2 get_cell_size()
	{
		return Vector2(320, 280);
	}

    void compute_position(const Vector2& p_offset,bool p_horizontal_layout)
    {
        
        if(p_horizontal_layout)
        {
            pos_x = p_offset.x;
			pos_y = p_offset.y + (total_size / 2);
            Vector2 offset = p_offset;
            offset.x += get_cell_size().x + LEVEL_DISTANCE;
            for(uint32_t i = 0; i < children.size(); i++)
            {
                children[i]->compute_position(offset,p_horizontal_layout);
                offset.y += children[i]->total_size + SIBLING_DISTANCE;
            }
        }
        else
        {
			pos_x = p_offset.x + total_size / 2;
			pos_y = p_offset.y;
            Vector2 offset = p_offset;
            offset.y += get_cell_size().y + LEVEL_DISTANCE;
            for(uint32_t i = 0; i < children.size(); i++)
            {
                children[i]->compute_position(offset,p_horizontal_layout);
                offset.x += children[i]->total_size + SIBLING_DISTANCE;
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
