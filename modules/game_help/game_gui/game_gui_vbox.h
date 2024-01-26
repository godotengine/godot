#ifndef GAME_GUI_VBOX_H
#define GAME_GUI_VBOX_H
#include "game_gui_compoent.h"

class GUIVBox : public GUIComponent
{
    GDCLASS(GUIVBox, GUIComponent)
    static void _bind_methods();

    public:
    enum HorizontalContentAlignment
    {
        LEFT,    // Left-align the content.
        CENTER,  // Center the content.
        RIGHT    // Right-align the content.
    };
    HorizontalContentAlignment content_alignment = CENTER;

    void set_content_alignment(HorizontalContentAlignment p_alignment)
    {
        content_alignment = p_alignment;
		request_layout();
    }
    HorizontalContentAlignment get_content_alignment() { return content_alignment; }
    Vector<int> _min_widths;
    Vector<int> _max_widths;

    virtual void _resolve_child_sizes( Vector2 available_size, bool limited = false ) override
    {
        // Resolve for and collect min and max sizes
        _max_widths.clear();
        for(int i = 0; i < get_child_count(); ++i)
        {
            auto child = get_child(i);
            auto con = Object::cast_to<Control>(child);
            if (con && con->is_visible())
            {
                _resolve_child_size( con, available_size, true );
                _max_widths.push_back( int(con->get_size().x) );
            }
            else
                _max_widths.push_back( 0 );
        }

        _min_widths.clear();
        for(int i = 0; i < get_child_count(); ++i)
        {
            auto child = get_child(i);
            auto con = Object::cast_to<Control>(child);
            if (con && con->is_visible())
            {
                _resolve_child_size( child, Vector2(0,available_size.y), true );
                _min_widths.push_back( int(con->get_size().x) );
            }
            else
                _min_widths.push_back( 0 );

        }

        int expand_count = 0;
        float total_stretch_ratio = 0.0f;
        int fixed_width = 0;
        int min_width = 0;

        // Leaving other children at their minimum width, set aspect-fit, proportional,
        // and shrink-to-fit width nodes to their maximum size.
        //var modes = [GGComponent.ScalingMode.ASPECT_FIT,GGComponent.ScalingMode.PROPORTIONAL,GGComponent.ScalingMode.SHRINK_TO_FIT]
        
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
                auto w = int(con->get_size().x);
                min_width += w;
                fixed_width += w;
                _min_widths.write[i] = w;
                _max_widths.write[i] = w;
            }

            else
            {
                auto w = _min_widths.write[i];
                min_width += w;

                if (_min_widths[i] == _max_widths.write[i])
                {
                    fixed_width += w;
                    _resolve_child_size( child, con->get_size(), limited );  // final resolve

                }
                else
                {
                    expand_count += 1;
                    total_stretch_ratio += con->get_stretch_ratio();

                }

            }

        }

        if (expand_count == 0 || total_stretch_ratio == 0.0 || min_width >= available_size.x) return;

        auto excess_width = int(available_size.x - fixed_width);
        auto remaining_width = excess_width;

        // Find children with a min width larger than their portion. Let them keep their min width and adjust remaining.
        auto remaining_total_stretch_ratio = total_stretch_ratio;
        for(int i = 0; i < get_child_count(); ++i)
        {
            auto child = get_child(i);
            auto con = Object::cast_to<Control>(child);
            if(!con ||  !con->is_visible()) continue;
            if (_min_widths.write[i] == _max_widths.write[i]) continue;

            auto w = 0;
            if (expand_count == 1) 
                w = remaining_width;
            else
                w = int( excess_width * con->get_stretch_ratio() / total_stretch_ratio );

            if (w < _min_widths.write[i])
            {
                w = _min_widths.write[i];
                remaining_width -= w;
                expand_count -= 1;
                remaining_total_stretch_ratio -= con->get_stretch_ratio();
                _min_widths.write[i] = _max_widths.write[i];  // skip this node in the next pass
                _resolve_child_size( con, con->get_size(), limited );  // final resolve

            }

        }

        excess_width = remaining_width;
        total_stretch_ratio = remaining_total_stretch_ratio;
        if (expand_count == 0 || abs(total_stretch_ratio) < 0.0001) return;

	    // Distribute remaining width next to children with a max width smaller than their portion.
        for(int i = 0; i < get_child_count(); ++i)
        {
            auto child = get_child(i);
            auto con = Object::cast_to<Control>(child);
            if (!con || !con->is_visible()) 
                continue;
            if (_min_widths[i] == _max_widths[i])
                continue;

            auto w = 0;
            if (expand_count == 1)
                w = remaining_width;
            else
                w = int( excess_width * con->get_stretch_ratio() / total_stretch_ratio );

            if(w > _max_widths[i])
            {
                w = _max_widths[i];
                _resolve_child_size( con, Vector2(w,available_size.y), limited );
                remaining_width -= w;
                expand_count -= 1;
                remaining_total_stretch_ratio -= con->get_stretch_ratio();
                _min_widths.write[i] = _max_widths[i];  // skip this node in the next pass

            }
        }

        excess_width = remaining_width;
        total_stretch_ratio = remaining_total_stretch_ratio;
        if (expand_count == 0 || abs(total_stretch_ratio) < 0.0001)
            return;

        // If this GGHBox is shrink-to-fit width then we're done; don't add remaining space to
        // the children.
        if (horizontal_mode == GUIComponent::S_SHRINK_TO_FIT)
            return;

	    // Distribute remaining width
        for(int i = 0; i < get_child_count(); ++i)
        {
            auto child = get_child(i);
            auto con = Object::cast_to<Control>(child);
            if (!con || !con->is_visible()) 
                continue;
            if (_min_widths[i] == _max_widths[i])
                continue;

            auto w = 0;
            if (expand_count == 1)
                w = remaining_width;
            else
                w = int( excess_width * con->get_stretch_ratio() / total_stretch_ratio );

            _resolve_child_size( con, Vector2(w,available_size.y), limited );
            remaining_width -= w;
            expand_count -= 1;

        }

    }
    virtual void _resolve_shrink_to_fit_width( Vector2 _available_size ) override
    {
        auto size = get_size();
        size.x = _get_sum_of_child_sizes().x;
        set_size( size );
    }

    virtual void _resolve_shrink_to_fit_height( Vector2 _available_size ) override
    {
        auto size = get_size();
        size.y = _get_largest_child_size().y;
        set_size( size );

    }

    virtual void _perform_layout( Rect2 available_bounds )
    {
        _place_component( this, available_bounds );

        auto inner_bounds = _with_margins( Rect2(Vector2(0,0),get_size()) );
        auto pos = inner_bounds.position;
        auto sz = inner_bounds.size;

        auto diff = sz.y - _get_sum_of_child_sizes().x;
        switch (content_alignment)
        {
            case LEFT:
            {            
            }
            break;
            case CENTER:
            {
                pos.x += int(diff/2.0);
            }
            break;
            case RIGHT:
            {
                pos.x += diff;
            }
            break;
        
            default:
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
#endif