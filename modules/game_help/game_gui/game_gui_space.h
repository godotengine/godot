#ifndef GAME_GUI_SPACE_H
#define GAME_GUI_SPACE_H
#include "game_gui_compoent.h"

// gui组件空白区域

class GUISpace : public GUIComponent 
{
    GDCLASS(GUISpace, GUIComponent);
    static void _bind_methods();
public:
    enum MarginType
    {
        PROPORTIONAL,  // 指定边距介于 0.0 和 1.0 之间，类似锚点。
        FIXED,         // 边距具有固定的像素大小。
        PARAMETER      // 边距使用参数作为其像素大小。
    };
    // 指定将环绕子内容区域的边距的类型。
    MarginType margin_type = PROPORTIONAL;
    void set_margin_type(MarginType value)
    {
        if (margin_type == value) return;
        if (reference_node)
        {
                
                auto ref_size = reference_node->get_size();
                if (value == PROPORTIONAL)
                {
                    if (margin_type == FIXED && ref_size.x && ref_size.y)
                    {
                        set_left_margin(left_margin / ref_size.x);
                        set_top_margin(top_margin / ref_size.y);
                        set_right_margin(right_margin / ref_size.x); 
                        set_bottom_margin(bottom_margin / ref_size.y);
                    }
                    else
                    {
                        set_left_margin(0.0);
                        set_top_margin(0.0);
                        set_right_margin(0.0);
                        set_bottom_margin(0.0);

                    }

                }
                else if( value == FIXED)
                {
                    if (margin_type == PROPORTIONAL && ref_size.x && ref_size.y)
                    {
                        set_left_margin( left_margin * ref_size.x );
                        set_top_margin( top_margin * ref_size.y );
                        set_right_margin( right_margin * ref_size.x );
                        set_bottom_margin( bottom_margin * ref_size.y );

                    }
                    else
                    {
                        set_left_margin(0.0);
                        set_top_margin(0.0);
                        set_right_margin(0.0);
                        set_bottom_margin(0.0);
                    }
                }
        }
        else
        {
            if (value == PROPORTIONAL)
            {
                auto size = get_size();
                if (margin_type == FIXED && size.x && size.y)
                {
                    set_left_margin(left_margin / size.x);
                    set_top_margin(top_margin / size.y);
                    set_right_margin( 1.0 - (right_margin/size.x));
                    set_bottom_margin( 1.0 - (bottom_margin/size.y));
                }
                else
                {
                    set_left_margin(0.0);
                    set_top_margin(0.0);
                    set_right_margin(0.0);
                    set_bottom_margin(0.0);
                }

            }
            else if( value == FIXED)
            {
                auto size = get_size();
                if (margin_type == FIXED && size.x && size.y)
                {
                    set_left_margin( left_margin * size.x );
                    set_top_margin( top_margin * size.y );
                    set_right_margin( (1.0 - right_margin) * size.x );
                    set_bottom_margin( (1.0 - bottom_margin) * size.y );
                }
                else
                {
                    set_left_margin(0.0);
                    set_top_margin(0.0);
                    set_right_margin(0.0);
                    set_bottom_margin(0.0);

                }

            }

        }

        margin_type = value;
        request_layout();           
        
    }	
    MarginType get_margin_type()
    {
        return margin_type;
    }
    // [b]Proportional[/b] 边距类型，无 [member reference_node][br]  0.0：无边距[br]  0.25：25% 边距，依此类推[br]
    // [b]Proportional[/b] 边距类型，[member reference_node] 设置[br]  0.0：无边距[br]  0.25：边距为参考节点大小的 25%，依此类推[br]
    // [b]Fixed[/b] 边距类型[br]  0：无边距[br]  25：25 像素边距，依此类推[br]
    float left_margin = 0.0;
    void set_left_margin(float value)
    {
        if (left_margin == value) return;
        left_margin = value;
        request_layout();

    }
    float get_left_margin()
    {
        return left_margin;
    }

    // [b]Proportional[/b] 边距类型，无 [member reference_node][br]  0.0：无边距[br]  0.25：25% 边距，依此类推[br]
    // [b]Proportional[/b] 边距类型，[member reference_node] 设置[br]  0.0：无边距[br]  0.25：边距为参考节点大小的 25%，依此类推[br]
    // [b]Fixed[/b] 边距类型[br]  0：无边距[br]  25：25 像素边距，依此类推[br]
    float top_margin = 0.0 ;
    void set_top_margin(float value)
    {
        if (top_margin == value) return;
        top_margin = value;
        request_layout();
    }
    float get_top_margin()
    {
        return top_margin;
    }
    // [b]Proportional[/b] 边距类型，无 [member reference_node][br]  0.0：无边距[br]  0.75：25% 边距，依此类推[br]
    // [b]Proportional[/b] 边距类型，[member reference_node] 设置[br]  0.0：无边距[br]  0.25：边距为参考节点大小的 25%，依此类推[br]
    // [b]Fixed[/b] 边距类型[br]  0：无边距[br]  25：25 像素边距，依此类推[br]
    float right_margin = 1.0;
    void set_right_margin(float value)
    {
        if (right_margin == value) return;
        right_margin = value;
        request_layout();
    }
    float get_right_margin()
    {
        return right_margin;
    }

    // [b]Proportional[/b] 边距类型，无 [member reference_node][br]  0.0：无边距[br]  0.75：25% 边距，依此类推[br]
    // [b]Proportional[/b] 边距类型，[member reference_node] 设置[br]  0.0：无边距[br]  0.25：边距为参考节点大小的 25%，依此类推[br]
    // [b]Fixed[/b] 边距类型[br]  0：无边距[br]  25：25 像素边距，依此类推[br]
    float bottom_margin = 1.0;
    void set_bottom_margin(float value)
    {
        if (bottom_margin == value) return;
        bottom_margin = value;
        request_layout();
    }
    float get_bottom_margin()
    {
        return bottom_margin;
    }
    // [b]Parameter[/b] 边距类型[br]  "": 无左边距[br]  "abc": 左边距为 [code]get_parameter("abc")[/code] 像素，依此类推。
    String left_parameter = "";
    void set_left_parameter(String value)
    {
        if (left_parameter == value) return;
        left_parameter = value;
        request_layout();
    }
    String get_left_parameter()
    {
        return left_parameter;
    }

    // [b]Parameter[/b] 边距类型[br]  "": 无上边距[br]  "abc": 上边距为 [code]get_parameter("abc")[/code] 像素，依此类推。
    String top_parameter = "" ;
    void set_top_parameter(String value)
    {
        if (top_parameter == value) return;
        top_parameter = value;
        request_layout();
    }
    String get_top_parameter()
    {
        return top_parameter;
    }

    // [b]Parameter[/b] 边距类型[br]  "": 无右边距[br]  "abc": 右边距为 [code]get_parameter("abc")[/code] 像素，依此类推。
    String right_parameter = "";
    void set_right_parameter(String value)
    {
        if (right_parameter == value) return;
        right_parameter = value;
        request_layout();
    }
    String get_right_parameter()
    {
        return right_parameter;
    }
    // [b]Parameter[/b] 边距类型[br]  "": 无下边距[br]  "abc": 下边距为 [code]get_parameter("abc")[/code] 像素，依此类推。
    String bottom_parameter = "";
    void set_bottom_parameter(String value)
    {
        if (bottom_parameter == value) return;
        bottom_parameter = value;
        request_layout();
    }
    String get_bottom_parameter()
    {
        return bottom_parameter;
    }

    virtual void _resolve_shrink_to_fit_height( Vector2 available_size )override
    {
        GUIComponent::_resolve_shrink_to_fit_height( available_size );
        Vector2 size = get_size();
        switch (margin_type)
        {
            case PROPORTIONAL:
            {
                if (reference_node)
                {
                    size.y += int( top_margin * reference_node->get_size().y );
                    size.y += int( bottom_margin * reference_node->get_size().y );
                    
                }
                else
                {

                    size.y += int( available_size.y * top_margin );
                    size.y += int( available_size.y * bottom_margin );
                }

            
            }
            break;
            case FIXED:
            {
                size.y += top_margin;
                size.y += bottom_margin;

            }
            break;
            case PARAMETER:
            {
                size.y += (float)get_parameter(top_parameter,0);
                size.y += (float)get_parameter(bottom_parameter,0);

            }
        }
        set_size(size);

    }
    virtual void _resolve_shrink_to_fit_width( Vector2 available_size ) override
    {
        GUIComponent::_resolve_shrink_to_fit_width( available_size );
        Vector2 size = get_size();
        switch (margin_type)
        {
            case PROPORTIONAL:
            {
                if (reference_node)
                {
                    size.x += int( left_margin * reference_node->get_size().x );
                    size.x += int( right_margin * reference_node->get_size().x );

                }
                else
                {
                    size.x += int( available_size.x * left_margin );
                    size.x += int( available_size.x * right_margin );

                }

            }
            break;
            case FIXED:
            {
                size.x += left_margin;
                size.x += right_margin;

            }
            break;
            case PARAMETER:
            {
                size.x += (float)get_parameter(left_parameter,0);
                size.x += (float)get_parameter(right_parameter,0);

            }
            break;

        }
        set_size(size);

    }

    virtual Rect2 _with_margins( Rect2 rect )override
    {
        Vector2 size = get_size();
        switch (margin_type)
        {
            case PROPORTIONAL:
            {
                if (reference_node)
                {
                    auto left = int( left_margin * reference_node->get_size().x );
                    auto right = int( right_margin * reference_node->get_size().x );
                    auto top = int( top_margin * reference_node->get_size().y );
                    auto bottom = int( bottom_margin * reference_node->get_size().y );
                    auto x = rect.position.x + left;
                    auto y = rect.position.y + top;
                    auto x2 = rect.position.x + (rect.size.x - right);
                    auto y2 = rect.position.y + (rect.size.y - bottom);
                    auto w = x2 - x;
                    auto h = y2 - y;
                    if (w < 0) w = 0;
                    if (h < 0) h = 0;
                    return Rect2( x, y, w, h );

                }
                else
                {
                    auto x = rect.position.x + Math::floor( rect.size.x * left_margin );
                    auto y = rect.position.y + Math::floor( rect.size.y * top_margin );
                    auto x2 = rect.position.x + Math::floor( rect.size.x * right_margin );
                    auto y2 = rect.position.y + Math::floor( rect.size.y * bottom_margin );
                    auto w = x2 - x;
                    auto h = y2 - y;
                    if (w < 0) w = 0;
                    if (h < 0) h = 0;
                    return Rect2( x, y, w, h );

                }

            }
            break;
            case FIXED:
            {
                auto x = rect.position.x + left_margin;
                auto y = rect.position.y + top_margin;
                auto x2 = rect.position.x + (rect.size.x - right_margin);
                auto y2 = rect.position.y + (rect.size.y - bottom_margin);
                auto w = x2 - x;
                auto h = y2 - y;
                if (w < 0) w = 0;
                if (h < 0) h = 0;
                return Rect2( x, y, w, h );

            }
            break;
            case PARAMETER:
            {
                auto x = rect.position.x + (float)get_parameter(left_parameter,0);
                auto y = rect.position.y + (float)get_parameter(top_parameter,0);
                auto x2 = rect.position.x + (rect.size.x - (float)get_parameter(right_parameter,0));
                auto y2 = rect.position.y + (rect.size.y - (float)get_parameter(bottom_parameter,0));
                auto w = x2 - x;
                auto h = y2 - y;
                if (w < 0) w = 0;
                if (h < 0) h = 0;
                return Rect2( x, y, w, h );
                
            }
        }
        return rect;

    }



};

VARIANT_ENUM_CAST(GUISpace::MarginType)
#endif