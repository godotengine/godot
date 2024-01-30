#ifndef GAME_GUI_ARBITRARY_COORD_H
#define GAME_GUI_ARBITRARY_COORD_H
#include "game_gui_compoent.h"
// 在其自身边界内的任意坐标处定位其子节点，类似于精灵。
// 不适用于实际的Sprite2D；请使用GGTextureRect或其他Control类型作为子节点。通常与单个子节点一起使用。

class GUIArbitraryCoord : public GUIComponent
{
    GDCLASS(GUIArbitraryCoord, GUIComponent)
    static void _bind_methods();
public:
    // 定位模式枚举。
    enum PositioningMode
    {
        P_PROPORTIONAL,  // 使用0.0到1.0之间的分数指定子节点位置。
        P_FIXED,         // 子节点位置使用固定像素偏移量。
        P_PARAMETER      // 使用参数作为子节点的相对像素偏移量。
    };

    // 缩放因子枚举。
    enum ScaleFactor
    {
        S_CONSTANT,      // 使用固定的缩放因子进行缩放。
        S_PARAMETER      // 使用子树参数进行缩放。
    };
    bool is_scale_font;
    void set_scale_font(bool value) { is_scale_font = value; request_layout(); }
    bool get_scale_font() { return is_scale_font; }


    float reference_height;
    void set_reference_height(float value) { reference_height = value; request_layout(); }
    float get_reference_height() { return reference_height; }

    float reference_font_size = 0;
    void set_reference_font_size(float value) { reference_font_size = value; request_layout(); }
    float get_reference_font_size() { return reference_font_size; }

    // 子节点定位模式。
    PositioningMode positioning_mode = P_PROPORTIONAL;
    void set_positioning_mode(PositioningMode value)
    {
        if (positioning_mode == value)
            return;
        switch (value)
        {
            case P_PROPORTIONAL:
            {
                if (positioning_mode == P_FIXED)
                {
                    set_child_x( child_x / get_size().x);
                    set_child_y(child_y / get_size().y);
                }
                else
                {
                    set_child_x( 0.5);
                    set_child_y(0.5);
                }
            }
            break;
            case P_FIXED:
            {
                if (positioning_mode == P_PROPORTIONAL)
                {
                    set_child_x( child_x * get_size().x );
                    set_child_y( child_y * get_size().y );
                }
                else
                {
                    set_child_x(get_size().x / 2.0);
                    set_child_y(get_size().y / 2.0);
                }
            }
            break;        
            default:
                break;
        }
        positioning_mode = value;
        request_layout();
    }
    PositioningMode get_positioning_mode() { return positioning_mode; }

    // 此组件内的子节点 'x' 偏移量。在[b]Proportional[/b]定位模式下使用0.0-1.0，在[b]Fixed[/b]定位模式下使用整数值。
    float child_x = 0.5;
    void set_child_x(float value)
    {
        if (child_x == value) return;
        child_x = value;
        request_layout();

    }
    float get_child_x() { return child_x; }

    // 此组件内的子节点 'y' 偏移量。在[b]Proportional[/b]定位模式下使用0.0-1.0，在[b]Fixed[/b]定位模式下使用整数值。
    float child_y = 0.5;
    void set_child_y(float value)
    {
        if (child_y == value) return;
        child_y = value;
        request_layout();

    }
    float get_child_y() { return child_y; }
    
    // 用于子节点 'x' 偏移量的参数名称。
    String child_x_parameter = "";
    void set_child_x_parameter(String value)
    {
        if (child_x_parameter == value)
            return;
        child_x_parameter = value;
        request_layout();
    }
    String get_child_x_parameter() { return child_x_parameter; }

    // 用于子节点 'y' 偏移量的参数名称。
    String child_y_parameter = "" ;
    void set_child_y_parameter(String value)
    {
        if (child_y_parameter == value)
            return;
        child_y_parameter = value;
        request_layout();

    }
    String get_child_y_parameter() { return child_y_parameter; }

    // 水平缩放模式。
    ScaleFactor h_scale_factor = S_CONSTANT;
    void set_h_scale_factor(ScaleFactor value)
    {
        if (h_scale_factor == value)
            return;
        h_scale_factor = value;
        request_layout();
    }
    ScaleFactor get_h_scale_factor() { return h_scale_factor; }

    // 垂直缩放模式。
    ScaleFactor v_scale_factor = S_CONSTANT ;
    void set_v_scale_factor(ScaleFactor value)
    {
        if (v_scale_factor == value)
            return;
        v_scale_factor = value;
        request_layout();
    }
    ScaleFactor get_v_scale_factor() { return v_scale_factor; }

    // 当[member h_scale_factor]为[b]Constant[/b]时使用的水平缩放因子。
    float h_scale_constant = 1.0;
    void set_h_scale_constant(float value)
    {
        if (h_scale_constant == value) return;
        h_scale_constant = value;
        request_layout();
    }
    float get_h_scale_constant() { return h_scale_constant; }

    // 当[member v_scale_factor]为[b]Constant[/b]时使用的垂直缩放因子。
    float v_scale_constant = 1.0;
    void set_v_scale_constant(float value)
    {
        if (v_scale_constant == value) return;
        v_scale_constant = value;
        request_layout();
    }
    float get_v_scale_constant() { return v_scale_constant; }

    // 当[member h_scale_factor]为[b]Parameter[/b]时使用的水平缩放因子。
    String h_scale_parameter = "";
    void set_h_scale_parameter(String value)
    {
        if (h_scale_parameter == value) return;
        h_scale_parameter = value;
        request_layout();
    }
    String get_h_scale_parameter() { return h_scale_parameter; }

    // 当[member v_scale_factor]为[b]Parameter[/b]时使用的垂直缩放因子。
    String v_scale_parameter = "";
    void set_v_scale_parameter(String value)
    {
        if (v_scale_parameter == value) return;
        v_scale_parameter = value;
        request_layout();
    }
    String get_v_scale_parameter() { return v_scale_parameter; }

    Vector2 _get_scale()
    {
        auto sx = 0.0;
        auto sy = 0.0;

        switch(h_scale_factor)
        {
            case S_CONSTANT:
                sx = h_scale_constant;
                break;
            case S_PARAMETER:
                sx = get_parameter( h_scale_parameter );
                break;
        }

        switch(v_scale_factor)
        {
            case S_CONSTANT:
                sy = v_scale_constant;
                break;
            case S_PARAMETER:
                sy = get_parameter( v_scale_parameter );
                break;
        }

        return Vector2(sx,sy);

    }

    virtual void _perform_child_layout( Rect2 available_bounds )override
    {        
        for(int i = 0; i < get_child_count(); ++i)
        {
            auto child = get_child(i);
            auto con = Object::cast_to<Control>(child);
            if (con && con->is_visible())continue;


            int x_pos = 0;
            int y_pos = 0;

            switch (positioning_mode)
            {
                case P_PROPORTIONAL:
                    x_pos = available_bounds.size.x * child_x;
                    y_pos = available_bounds.size.y * child_y;
                case P_FIXED:
                    x_pos = child_x;
                    y_pos = child_y;
                case P_PARAMETER:
                    x_pos = get_parameter( child_x_parameter, child_x );
                    y_pos = get_parameter( child_y_parameter, child_y );
            }

            // 调整 x_pos 和 y_pos 以适应 SIZE_SHRINK_X。
            if (con->get_h_size_flags() & (SIZE_SHRINK_CENTER | SIZE_FILL))
                x_pos -= int(con->get_size().x / 2.0);
            else if (con->get_h_size_flags() & SIZE_SHRINK_END)
                x_pos -= int(con->get_size().x);

            if (con->get_v_size_flags() & (SIZE_SHRINK_CENTER | SIZE_FILL))
                y_pos -= int(con->get_size().y / 2.0);
            else if (con->get_v_size_flags() & SIZE_SHRINK_END)
                y_pos -= int(con->get_size().y);


            _perform_component_layout( con, Rect2(Vector2(x_pos,y_pos),con->get_size()) );            

        }
    }

    virtual void _resolve_child_sizes( Vector2 available_size, bool limited=false ) override
    {
	    auto scale = _get_scale();
        for(int i = 0; i < get_child_count(); ++i)
        {
            auto child = get_child(i);
            auto con = Object::cast_to<Control>(child);
            if (con && con->is_visible())continue;

            // 解析一次以获取子节点的完整大小
            _resolve_child_size( child, available_size, limited );

            // 将缩放因子应用于子节点
            _resolve_child_size( child, (con->get_size() * scale).floor(), limited );

            if(is_scale_font)
            {
				auto cur_scale = Math::floor(reference_node.size.y) / reference_height;

				// Override the size of the font to dynamically size it
				auto cur_size = reference_font_size * cur_scale;
			    con->add_theme_font_size_override( "font_size", cur_size )

            }
        }

    }


};


VARIANT_ENUM_CAST(GUIArbitraryCoord::PositioningMode)
VARIANT_ENUM_CAST(GUIArbitraryCoord::ScaleFactor)

#endif