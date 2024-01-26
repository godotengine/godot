#ifndef GAME_GUI_VBOX_H
#define GAME_GUI_VBOX_H
#include "game_gui_compoent.h"

class GUIVBox : public GUIComponent
{
    GDCLASS(GUIVBox, GUIComponent);
    static void _bind_methods();
public:
    enum VerticalContentAlignment
    {
        TOP,     // 将内容顶部对齐。
        CENTER,  // 将内容居中对齐。
        BOTTOM   // 将内容底部对齐。
    };

    // 将内容作为整体垂直对齐。
    VerticalContentAlignment content_alignment = CENTER;
	void set_content_alignment(VerticalContentAlignment value)
    {
		content_alignment = value;
		request_layout();
    }
    VerticalContentAlignment get_content_alignment()
    {
        return content_alignment;
    }

    Vector<int> _min_heights;  // 存储每个子节点的最小高度
    Vector<int> _max_heights;  // 存储每个子节点的最大高度

    virtual void _resolve_child_sizes( Vector2 available_size, bool limited=false )override
    {
        // 解析并收集最小和最大尺寸
	    _max_heights.clear();
        for(int i = 0; i < get_child_count(); ++i)
        {
            auto child = get_child(i);
            auto con = Object::cast_to<Control>(child);
            if (con && con->is_visible())
            {
                _resolve_child_size( child, available_size, true );
                _max_heights.push_back( int(con->get_size().y) );
            }
		    else
			    _max_heights.push_back( 0 );
        }

	    _min_heights.clear();
        for(int i = 0; i < get_child_count(); ++i)
        {
            auto child = get_child(i);
            auto con = Object::cast_to<Control>(child);
            if (con && con->is_visible())
            {
                _resolve_child_size( child, Vector2(available_size.x,0), true );
                _min_heights.push_back( int(con->get_size().y) );

            }
		    else
			    _min_heights.push_back( 0 );
        }

        int expand_count = 0;
        float total_stretch_ratio = 0.0;
        int fixed_height = 0;
        int min_height = 0;

	    // 将其余子节点保持在最小高度，并将其最大尺寸设置为固定值。

        for(int i = 0; i < get_child_count(); ++i)
        {
            auto child = get_child(i);
            auto con = Object::cast_to<Control>(child);
            if (!con && !con->is_visible())continue;

            auto has_mode = Object::cast_to<GUIComponent>(child);
            if (has_mode && (has_mode->horizontal_mode == GUIComponent::S_ASPECT_FIT 
                || has_mode->horizontal_mode == GUIComponent::S_PROPORTIONAL
                || has_mode->horizontal_mode == GUIComponent::S_SHRINK_TO_FIT)
            )
            {
                _resolve_child_size( child, available_size, limited );
                auto h = int(con->get_size().y);
                min_height += h;
                fixed_height += h;
                _min_heights.write[i] = h;
                _max_heights.write[i] = h;
            }

            else
            {
                auto h = _min_heights[i];
                min_height += h;

                if (_min_heights[i] == _max_heights[i])
                {
                    fixed_height += h;
                    _resolve_child_size( child, con->get_size(), limited );  // 最终解析
                }
                else
                {
                    expand_count += 1;
                    total_stretch_ratio += con->get_stretch_ratio();
                }


            }
            if (expand_count == 0 || total_stretch_ratio == 0.0 || min_height >= available_size.y)
                return;

            auto excess_height = int(available_size.y - fixed_height);
            auto remaining_height = excess_height;

            // 找到最小高度大于其比例的子节点。让它们保持最小高度并调整其余。
            auto remaining_total_stretch_ratio = total_stretch_ratio;
            for(int i = 0; i < get_child_count(); ++i)
            {
                auto child = get_child(i);
                auto con = Object::cast_to<Control>(child);
                if (!con && !con->is_visible())continue;

                if (_min_heights[i] == _max_heights[i])
                    continue;

                auto h = 0;
                if (expand_count == 1)
                    h = remaining_height;
                else
                    h = int( excess_height * con->get_stretch_ratio() / total_stretch_ratio );

                if (h < _min_heights[i])
                {
                    h = _min_heights[i];
                    remaining_height -= h;
                    expand_count -= 1;
                    remaining_total_stretch_ratio -= con->get_stretch_ratio();
                    _min_heights.write[i] = _max_heights[i];  // 在下一次遍历中跳过此节点
                    _resolve_child_size( child, con->get_size(), limited );  // 最终解析

                }

                excess_height = remaining_height;
                total_stretch_ratio = remaining_total_stretch_ratio;
            }
            if (expand_count == 0 || abs(total_stretch_ratio) < 0.0001)
                return;

            // 将剩余高度分配给具有最大高度小于其比例的子节点。
            for(int i = 0; i < get_child_count(); ++i)
            {
                auto child = get_child(i);
                auto con = Object::cast_to<Control>(child);
                if (!con && !con->is_visible())continue;
                if (_min_heights.write[i] == _max_heights.write[i])
                    continue;

                auto h = 0;
                if (expand_count == 1)
                    h = remaining_height;
                else
                    h = int( excess_height * con->get_stretch_ratio() / total_stretch_ratio );

                if (h > _max_heights.write[i])
                {
                    h = _max_heights.write[i];
                    _resolve_child_size( child, Vector2(available_size.x,h), limited );
                    remaining_height -= h;
                    expand_count -= 1;
                    remaining_total_stretch_ratio -= con->get_stretch_ratio();
                    _min_heights.write[i] = _max_heights.write[i];  // 在下一次遍历中跳过此节点
                }
            }

            excess_height = remaining_height;
            total_stretch_ratio = remaining_total_stretch_ratio;
            if (expand_count == 0 || abs(total_stretch_ratio) < 0.0001)
                return;

            // 如果这是 shrink-to-fit 高度，则完成；不要将剩余空间添加到子节点。
            if (vertical_mode == GUIComponent::S_SHRINK_TO_FIT)
                return;

            // 分配剩余高度
            for(int i = 0; i < get_child_count(); ++i)
            {
                auto child = get_child(i);
                auto con = Object::cast_to<Control>(child);
                if (!con && !con->is_visible())continue;
                if (_min_heights.write[i] == _max_heights.write[i])
                    continue;

                auto h = 0;
                if (expand_count == 1)
                    h = remaining_height;
                else
                    h = int( excess_height * con->get_stretch_ratio() / total_stretch_ratio );

                _resolve_child_size( child, Vector2(available_size.x,h), limited );
                remaining_height -= h;
                expand_count -= 1;
            }
        }
    }

    virtual void _resolve_shrink_to_fit_height( Vector2 _available_size )override
    {        
        auto size = get_size();
        size.y = _get_sum_of_child_sizes().y;
        set_size( size );
    }

    virtual void  _resolve_shrink_to_fit_width( Vector2 _available_size )
    {
        auto size = get_size();
        size.x = _get_largest_child_size().x;
        set_size( size );
    }

    virtual void _perform_layout( Rect2 available_bounds )
    {
        _place_component( this, available_bounds );
        auto inner_bounds = _with_margins( Rect2(Vector2(0,0),get_size()) );
        auto pos = inner_bounds.position;
        auto sz = inner_bounds.size;
        auto diff = sz.y - _get_sum_of_child_sizes().y;
        
        switch (content_alignment)
        {
            case TOP:
                break;
            case CENTER:
                pos.y += int(diff/2.0);
                break;
            case BOTTOM:
                pos.y += diff;
                break;
        }

        for(int i = 0; i < get_child_count(); ++i)
        {
            auto child = get_child(i);
            auto con = Object::cast_to<Control>(child);
            if (!con || con->is_visible())
                continue;
            _perform_component_layout( child, Rect2(pos,Vector2(sz.x,con->get_size().y)) );
            pos += Vector2( 0, con->get_size().y );
        }

    }

};

VARIANT_ENUM_CAST(GUIVBox::VerticalContentAlignment)
#endif