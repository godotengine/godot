#ifndef GAME_GUI_COMPOENT_H
#define GAME_GUI_COMPOENT_H

#include "scene/gui/control.h"
#include "scene/gui/container.h"
#include "scene/main/viewport.h"

class GUIComponent : public Container
{
    GDCLASS(GUIComponent, Container);
    static void _bind_methods();
    public:
    // The possible horizontal and vertical sizing modes for a GameGUI component.
    enum ScalingMode
    {
        S_EXPAND_TO_FILL, // Fill all available space along this dimension.
        S_ASPECT_FIT,     // Dynamically adjusts size to maintain aspect ratio [member layout_size].x:[member layout_size].y, just small enough to entirely fit available space.
        S_ASPECT_FILL,    // Dynamically adjusts size to maintain aspect ratio [member layout_size].x:[member layout_size].y, just large enough to entirely fill available space.
        S_PROPORTIONAL,   // The layout size represents a proportional fraction of 1) the available area or 2) the size of the [member reference_node] if defined.
        S_SHRINK_TO_FIT,  // Make the size just large enough to contain all child nodes in their layout.
        S_FIXED,          // Fixed pixel size along this dimension.
        S_PARAMETER       // One of the subtree [member parameters] is used as the size.
    };

    // The text sizing mode for [GGLabel], [GGRichTextLabel], and [GGButton].
    enum TextSizeMode
    {
        DEFAULT,    // Text size is whatever size you assign in the editor.
        SCALE,      // Text scales with the size of a reference node.
        PARAMETER   // Text size is set to the value of one of the defined [member parameters].
    };

    // The texture fill mode used by [method fill_texture].
    enum FillMode
    {
        STRETCH,  // Stretch or compress each patch to cover the available space.
        TILE,     // Repeatedly tile each patch at its original pixel size to cover the available space.
        TILE_FIT  // Tile each patche, stretching slightly as necessary to ensure a whole number of tiles fit in the available space.
    };
    ScalingMode horizontal_mode = S_EXPAND_TO_FILL;
    ScalingMode vertical_mode = S_EXPAND_TO_FILL;
    Vector2 layout_size = Vector2(0,0);
    Control* reference_node;
    String height_parameter = "";
    Dictionary parameters ;
	void set_horizontal_mode(ScalingMode value)
    {
		if (horizontal_mode == value)
            return;
		horizontal_mode = value;
		if(value == S_ASPECT_FIT || value == S_ASPECT_FILL)
        {
			if(vertical_mode == S_PROPORTIONAL || vertical_mode == S_FIXED || vertical_mode == S_PARAMETER)
            {
                vertical_mode = value;
            }
			if (layout_size.x  < 0.0001) layout_size.x = 1;
			if (layout_size.y  < 0.0001) layout_size.y = 1;

        }
		else if( vertical_mode  ==  S_ASPECT_FIT || vertical_mode  ==  S_ASPECT_FILL){
			if (!(value == S_EXPAND_TO_FILL || value == S_SHRINK_TO_FIT || value == S_PARAMETER))
            {
              vertical_mode = value;  
            } 
            
        }
		if (value == S_PROPORTIONAL)
        {
			if (layout_size.x < 0.0001 || layout_size.x > 1) layout_size.x = 1;
			if (layout_size.y < 0.0001 || layout_size.x > 1) layout_size.y = 1;

        }
		request_layout();

    }
    ScalingMode get_horizontal_mode(){return horizontal_mode;}
    
	void set_vertical_mode(ScalingMode value)
    {
		if (vertical_mode == value)
         return;
		vertical_mode = value;
		if (value == S_ASPECT_FIT || value == S_ASPECT_FILL)
        {
			if (horizontal_mode == S_PROPORTIONAL || horizontal_mode == S_FIXED || horizontal_mode == S_PARAMETER)
            {
                horizontal_mode = value;
            } 
			if (Math::abs(layout_size.x)  < 0.0001) layout_size.x = 1;
			if (Math::abs(layout_size.y)  < 0.0001) layout_size.y = 1;

        }
		else if( horizontal_mode == S_ASPECT_FIT || horizontal_mode == S_ASPECT_FILL)
        {
			if (! (value == S_EXPAND_TO_FILL || value == S_SHRINK_TO_FIT || value == S_PARAMETER))
            {
                horizontal_mode = value;
            } 
        }
		if (value == S_PROPORTIONAL)
        {
			if (layout_size.x < 0.0001 || layout_size.x > 1) layout_size.x = 1;
			if (layout_size.y < 0.0001 || layout_size.x > 1) layout_size.y = 1;

        }
		request_layout();

    }
    ScalingMode get_vertical_mode(){return vertical_mode;}
	void set_layout_size(Vector2 value)
    {
		if (layout_size == value)
         return;
		// The initial Vector2(0,0) may come in as e.g. 0.00000000000208 for x and y
		if (Math::abs(value.x) < 0.00001) value.x = 0;
		if (Math::abs(value.y) < 0.00001) value.y = 0;
		layout_size = value;
		request_layout();
        
    }
    Vector2 get_layout_size(){return layout_size;}
    // An optional node to use as a size reference for [b]Proportional[/b] scaling
    // mode. The reference node must be in a subtree higher in the scene tree than
    // this node. Often the size reference is an invisible root-level square-aspect
    // component; this allows same-size horizontal and vertical proportional spacers.
	void set_reference_node(Control* value)
    {
		if (reference_node != value)
        {
			reference_node = value;
			request_layout();

        }
    }
    Control* get_reference_node(){return reference_node;}
    String width_parameter = "";
	void set_width_parameter(String value)
    {
		if (width_parameter == value)
            return;
		width_parameter = value;
		if (value != "" && has_parameter(value))
			request_layout();

    }
    String get_width_parameter(){return width_parameter;}
    // The name of the parameter to use for the [b]Parameter[/b] vertical_mode scaling mode.
	void set_height_parameter(String value)
    {
		if (height_parameter == value)
            return;
		height_parameter = value;
		if (value != "" || has_parameter(value))
			request_layout();
    }
    String get_height_parameter(){return height_parameter;}
    // Parameter definitions for nodes that use scaling mode PARAMETER. Parameters are stored
    // the root of a GUIComponent subtree. Use [method get_parameter], [method has_parameter], and
    // [method set_parameter] to access parameters from any subtree nodes.
	void set_parameters(Dictionary value)
    {
		if (parameters == value)
            return;
		parameters = value;
		request_layout();

    }
    Dictionary get_parameters(){return parameters;}
    // A top-level GUIComponent is one that has no GUIComponent parent.
    // It oversees the layout of its descendent nodes.
    bool _is_top_level = false;
    int _layout_stage = 0 ;// top-level component use. 0=layout finished, 1=layout requested, 2=performing layout

    //-------------------------------------------------------------------------------
    // EXTERNAL API
    //-------------------------------------------------------------------------------

    // Utility method that draws a texture with any combination of horizontal and vertical fill modes: Stretch, Tile, Tile Fit.
    // Used primarily by [GUIComponent].
    void fill_texture( const Ref<Texture2D>& texture, Rect2 dest_rect, Rect2 src_rect, FillMode horizontal_fill_mode = STRETCH,
            FillMode vertical_fill_mode = STRETCH, Color _modulate=Color(1,1,1,1) )
    {
	    if (dest_rect.size.x <= 0 || dest_rect.size.y <= 0)
            return;

        if (horizontal_fill_mode == TILE  && src_rect.size.x > dest_rect.size.x)
            horizontal_fill_mode = TILE_FIT;

        if (vertical_fill_mode == TILE && src_rect.size.y > dest_rect.size.y)
            vertical_fill_mode = TILE_FIT;

        switch (horizontal_fill_mode)
        {
        case TILE:
        {
                auto tile_size = src_rect.size;
                auto dest_pos  = dest_rect.position;
                auto dest_w = dest_rect.size.x;
                auto dest_h = dest_rect.size.y;
                while (dest_w > 0){
                    if (tile_size.x <= dest_w)
                        {
                            auto _dest_rect = Rect2( dest_pos, Vector2(tile_size.x,dest_h) );
                        fill_texture( texture, _dest_rect, src_rect, STRETCH, vertical_fill_mode, _modulate );
                        }
                    else{
                        auto _dest_rect = Rect2( dest_pos, Vector2(dest_w,dest_h) );
                        auto _src_rect = Rect2( src_rect.position, Vector2(dest_w,src_rect.size.y) );
                        fill_texture( texture, _dest_rect, _src_rect, STRETCH, vertical_fill_mode, _modulate );
                        return;

                    }

                    dest_pos += Vector2( tile_size.x, 0 );
                    dest_w   -= tile_size.x;
                }
                return;

        }
            break;
        
        case TILE_FIT:
            {
                auto n = int( (dest_rect.size.x / src_rect.size.x) + 0.5 );
                if (n == 0)
                    fill_texture( texture, dest_rect, src_rect, STRETCH, vertical_fill_mode, _modulate );
                else
                {
                    auto tile_size = Vector2( dest_rect.size.x / n, src_rect.size.y );
                    auto dest_pos  = dest_rect.position;
                    auto dest_w = dest_rect.size.x;
                    auto dest_h = dest_rect.size.y;
                    while( dest_w > 0){
                        if (tile_size.x <= dest_w)
                        {
                            auto _dest_rect = Rect2( dest_pos, Vector2(tile_size.x,dest_h) );
                            fill_texture( texture, _dest_rect, src_rect, STRETCH, vertical_fill_mode, _modulate );
                        }
                        else
                        {
                            auto _dest_rect = Rect2( dest_pos, Vector2(dest_w,dest_h) );
                            fill_texture( texture, _dest_rect, src_rect, STRETCH, vertical_fill_mode, _modulate );
                            return;
                        }
                        dest_pos += Vector2( tile_size.x, 0 );
                        dest_w   -= tile_size.x;
                    }
                }
                return;
            }
            break;
        default:
                break;
        } 


	    switch( vertical_fill_mode)
        {
		    case TILE:
			{
                auto tile_size = src_rect.size;
                auto dest_pos  = dest_rect.position;
                auto dest_w = dest_rect.size.x;
                auto dest_h = dest_rect.size.y;
                while (dest_h > 0)
                {
                    if (tile_size.y <= dest_h)
                    {
                            auto _dest_rect = Rect2( dest_pos, Vector2(dest_w, tile_size.y) );
                        fill_texture( texture, _dest_rect, src_rect, horizontal_fill_mode, STRETCH, _modulate );
                    }
                    else
                    {

                    auto _dest_rect = Rect2( dest_pos, Vector2(dest_w,dest_h) );
                        auto _src_rect = Rect2( src_rect.position, Vector2(src_rect.size.x,dest_h) );
                        fill_texture( texture, _dest_rect, _src_rect, horizontal_fill_mode, STRETCH, _modulate );
                        return;
                    }
                    dest_pos += Vector2( 0, tile_size.y );
                    dest_h   -= tile_size.y;
                }
                return;
            }
            break;

		case TILE_FIT:
			{
                auto n = int( (dest_rect.size.y / src_rect.size.y) + 0.5 );
                if (n == 0)
                    fill_texture( texture, dest_rect, src_rect, horizontal_fill_mode, STRETCH, _modulate );
                else
                {
                    auto tile_size = Vector2( src_rect.size.x, dest_rect.size.y / n );
                    auto dest_pos  = dest_rect.position;
                    auto dest_w = dest_rect.size.x;
                    auto dest_h = dest_rect.size.y;
                    while (dest_h > 0)
                    {
                        if (tile_size.y <= dest_h)
                        {
                            auto _dest_rect = Rect2( dest_pos, Vector2(dest_w, tile_size.y) );
                            fill_texture( texture, _dest_rect, src_rect, horizontal_fill_mode, STRETCH, _modulate );
                        }   
                        else
                        {   
                            auto _dest_rect = Rect2( dest_pos, Vector2(dest_w,dest_h) );
                            fill_texture( texture, _dest_rect, src_rect, horizontal_fill_mode, STRETCH, _modulate );
                            return;

                        } 

                        dest_pos += Vector2( 0, tile_size.y );
                        dest_h   -= tile_size.y;
                    }
                    return;
                }
            }
            break;
        }

        
	    // Horizontal and vertical fill are both STRETCH
	    draw_texture_rect_region( texture, dest_rect, src_rect, _modulate );

    }
    // Returns the specified parameter's value if it exists in the [member parameters]
    // of this node or a [GUIComponent] ancestor. If it doesn't exist, returns
    // [code]0[/code] or a specified default result.
    Variant get_parameter( String parameter_name, Variant default_result=0 )
    {
        auto top = get_top_level_component();
        if (top && top->parameters.has(parameter_name))
            return top->parameters[parameter_name];
        else
            return default_result;

    }
    // Returns the root of this GUIComponent subtree.
    GUIComponent* get_top_level_component()
    {
        auto cur = this;
        while (cur && !cur->_is_top_level)
            cur = Object::cast_to<GUIComponent>( cur->get_parent());
        return cur;

    }
    // Returns [code]true[/code] if the specified parameter exists in the
    // [member parameters] of this node or one of its ancestors.
    bool has_parameter( String parameter_name )
    {
        auto top = get_top_level_component();
        if (top)
            return top->parameters.has(parameter_name);
        else
            return false;
    }
    // Sets the named parameter's value in the top-level root of this subtree.
    void set_parameter( String parameter_name, Variant value )
    {
        auto top = get_top_level_component();
        if (top) top->parameters[parameter_name] = value;

    }
    // Layout is performed automatically in most cases, but request_layout() can be
    // called for edge cases.
    void request_layout()
    {

        if (_is_top_level)
        {
            if (_layout_stage == 0)
            {
                _layout_stage = 1;
                queue_sort();
            }	
        }
        else
        {
            auto* top = get_top_level_component();
            if (top) 
            {
                top->request_layout();
            }
        }
    }
    public:
    //-------------------------------------------------------------------------------
    // KEY OVERRIDES
    //-------------------------------------------------------------------------------
    virtual void _on_resolve_size( Vector2 available_size )
    {
        // Overrideable.
        // Called just before this component's size is resolved.
        // Override and adjust this component's size if desired.
        //pass

    }
    virtual void _on_update_size()
    {
        // Overrideable.
        // Called at the beginning of layout.
        // Override and adjust this GUIComponent's size if desired.
        //pass

    }

    virtual void _perform_child_layout( Rect2 available_bounds )
    {
        for(int i = 0;i < get_child_count(); ++i)
        {
            _perform_component_layout( get_child(i), available_bounds );

        }
        
    }
    virtual void _resolve_child_sizes( Vector2 available_size, bool limited=false )
    {
        for(int i = 0;i < get_child_count(); ++i)
            _resolve_child_size( get_child(i), available_size, limited );
    }
    virtual void _resolve_shrink_to_fit_height( Vector2 _available_size )
    {
        // Override in extended classes.
        auto s = get_size();
        s.y = _get_largest_child_size().y;
        set_size(s);
    }
    virtual void _resolve_shrink_to_fit_width( Vector2 _available_size )
    {
        // Override in extended classes.
        auto s = get_size();
        s.x = _get_largest_child_size().x;
        set_size(s);
    }
    virtual Rect2 _with_margins( Rect2 rect )
    {
	    return rect;
    }
    protected:
    //-------------------------------------------------------------------------------
    // INTERNAL GAMEGUI API
    //-------------------------------------------------------------------------------

    Vector2 _get_largest_child_size()
    {
        // 'x' and 'y' will possibly come from different children.
        auto max_w = 0.0;
        auto max_h = 0.0;
        for(int i = 0;i < get_child_count(); ++i)
        {
            Control* child = Object::cast_to<Control>( get_child(i));
            if (!child->is_visible())
                continue;
            if (child->is_class("Control"))   // includes GUIComponent
            {
                max_w = MAX( max_w, child->get_size().x );
                max_h = MAX( max_h, child->get_size().y );

            }

        }
        return Vector2(max_w,max_h);

    }
    Vector2 _get_sum_of_child_sizes()
    {
        auto sum = Vector2(0,0);
        for(int i = 0;i < get_child_count(); ++i)
        {
            Control* child = Object::cast_to<Control>( get_child(i));
            if (!child->is_visible())
                continue;
            sum += child->get_size();

        }
        return sum;

    }
    void _perform_layout( Rect2 available_bounds )
    {
        _place_component( this, available_bounds );
        auto bounds = _with_margins( Rect2(Vector2(0,0),get_size()) );
        _perform_child_layout( bounds );

    }
    void _place_component( Control* component, Rect2 available_bounds )
    {
        component->set_position(_rect_position_within_parent_bounds( component, component->get_size(), available_bounds ));

    }
    Vector2 _rect_position_within_parent_bounds( Control* component, Vector2 rect_size, Rect2 available_bounds )
    {
        auto pos = available_bounds.position;

        GUIComponent* com = Object::cast_to<GUIComponent>( component ); // child->cast_to<GUIComponent>();
        if (com)  // includes GUIComponent
        {
            int64_t h_size_flags = com->get_h_size_flags();
            int64_t v_size_flags = com->get_v_size_flags();
            if (h_size_flags & (SIZE_SHRINK_CENTER | SIZE_FILL) )
            {
                pos.x += floor( (available_bounds.size.x - rect_size.x) / 2 );

            }
            else if( h_size_flags & SIZE_SHRINK_END)
            {
                pos.x += available_bounds.size.x - rect_size.x;
            }

            if (h_size_flags & (SIZE_SHRINK_CENTER | SIZE_FILL))
                pos.y += floor( (available_bounds.size.y - rect_size.y) / 2 );
            else if( h_size_flags & SIZE_SHRINK_END)
                pos.y += available_bounds.size.y - rect_size.y;
        }

        return pos;

    }
    void _resolve_child_size( Node * child, Vector2 available_size, bool limited=false )
    {
        if(child == nullptr)
        {
            return;
        }
        
        Control* con = Object::cast_to<Control>( child ); // child->cast_to<Control>();
        GUIComponent* com = Object::cast_to<GUIComponent>( child ); // child->cast_to<GUIComponent>();
        if ( con->is_visible())
            return;
        if (com)
            com->_resolve_size( available_size, limited );
        else
        {
            _resolve_component_size( child, available_size );
            _resolve_shrink_to_fit_size( child, available_size );
        }

    }

    Vector2 _resolve_component_size( Node * component, Vector2 available_size )
    {


        auto component_size = available_size;
        GUIComponent* is_gg = Object::cast_to<GUIComponent>( component ); // child->cast_to<GUIComponent>();
        Control* con = Object::cast_to<Control>( component ); // child->cast_to<Control>();
        if (is_gg )
            is_gg->_on_resolve_size( available_size );

        auto has_mode = (is_gg || component->has_method("request_layout"));
        auto h_mode = S_EXPAND_TO_FILL;
        if (is_gg)
        {
            h_mode = is_gg->horizontal_mode;
        }
        auto v_mode = S_EXPAND_TO_FILL;
        if (is_gg)
        {
            v_mode = is_gg->vertical_mode;
        }

        switch(h_mode)
        {
            case S_EXPAND_TO_FILL:
                //pass # use available width
                break;
            case S_SHRINK_TO_FIT:
                if (!is_gg)
                    component_size.x = is_gg->get_size().x;
                break;
            case S_ASPECT_FIT:
                if(is_gg)
                {
                    if (v_mode == S_ASPECT_FILL)
                        component_size.x = Math::floor( (available_size.y / is_gg->layout_size.y) * is_gg->layout_size.x );
                    else
                    {
                        float fit_x = Math::floor( (available_size.y / is_gg->layout_size.y) * is_gg->layout_size.x );
                        if (fit_x <= available_size.x) component_size.x = fit_x;

                    }

                }
                break;
            case S_ASPECT_FILL:
                if (v_mode != S_ASPECT_FIT && is_gg)
                {
                    float scale_x = (available_size.x / is_gg->layout_size.x);
                    float scale_y = (available_size.y / is_gg->layout_size.y);
                    component_size.x = Math::floor( MAX(scale_x,scale_y) * is_gg->layout_size.x );

                }
                break;
            case S_FIXED:
                if(is_gg)
                {
                    component_size.x = is_gg->layout_size.x;
                }
                break;
            case S_PARAMETER:
                    if(is_gg)
                        component_size.x = get_parameter( is_gg->width_parameter, is_gg->layout_size.x );
                break;
            case S_PROPORTIONAL:
                if(is_gg)
                {
                    if (reference_node)
                        component_size.x = int(is_gg->layout_size.x * reference_node->get_size().x);
                    else
                        component_size.x = int(is_gg->layout_size.x * available_size.x);

                }
                
                break;

        }

        switch (v_mode)
        {
            case S_EXPAND_TO_FILL:
                //pass # use available height
                break;
            case S_SHRINK_TO_FIT:
                if (!is_gg && con)
                    component_size.y = con->get_size().y;
                break;
            case S_ASPECT_FIT:
                if(is_gg)
                {
                    if (h_mode == S_ASPECT_FILL)
                        component_size.y = Math::floor( (available_size.x / is_gg->layout_size.x) * is_gg->layout_size.y );
                    else
                    {
                        float fit_y = Math::floor( (available_size.x / is_gg->layout_size.x) * is_gg->layout_size.y );
                        if (fit_y <= available_size.y) component_size.y = fit_y;

                    }
                }
                break;
            case S_ASPECT_FILL:
                    if(is_gg)
                    {
                        if (h_mode != S_ASPECT_FIT)
                        {
                            float scale_x = (available_size.x / is_gg->layout_size.x);
                            float scale_y = (available_size.y / is_gg->layout_size.y);
                            component_size.y = Math::floor( MAX(scale_x,scale_y) * is_gg->layout_size.y );
                        }
                    }
                    break;
            case S_FIXED:
                if(is_gg)
                    component_size.y = is_gg->layout_size.y;
                break;
            case S_PARAMETER:
                if(is_gg)
                    component_size.y = get_parameter( is_gg->height_parameter, is_gg->layout_size.y );
                break;
            case S_PROPORTIONAL:
                if(is_gg)
                {
                    if (reference_node)
                        component_size.y = int(is_gg->layout_size.y * reference_node->get_size().y);
                    else
                        component_size.y = int(is_gg->layout_size.y * available_size.y);
                }
                break;
        } 

        if (!is_gg && con) 
        {
            con->set_size( component_size);
            
        }

        return component_size;

    }

    void _perform_component_layout( Node * component, Rect2 available_bounds )
    {
        
        GUIComponent* com = Object::cast_to<GUIComponent>( component ); // child->cast_to<GUIComponent>();
        Control* con = Object::cast_to<Control>( component ); // child->cast_to<Control>();
        if (com)
            com->_perform_layout(available_bounds);
        else if(con)
            _place_component( con, available_bounds );
    }
    Vector2 _resolve_shrink_to_fit_size( Node * component, Vector2 available_size )
    {
        GUIComponent* com = Object::cast_to<GUIComponent>( component ); // child->cast_to<GUIComponent>();
        if (!com) return available_size;
        Vector2 component_size = available_size;
        if( com->horizontal_mode == S_SHRINK_TO_FIT)
        {
            if (com)
                _resolve_shrink_to_fit_width( available_size );
            else
                component_size.x = com->get_size().x;
        }

        if( com->vertical_mode == S_SHRINK_TO_FIT)
        {
            if (com)
                _resolve_shrink_to_fit_height( available_size );
            else
                component_size.y = com->get_size().y;
        }
        return component_size;
    }
    void _resolve_size( Vector2 available_size, bool limited =false )
    {
        // limited
        //   Don't recurse to children unless necessary (Shrink to Fit mode). limited==true indicates
        //   that several sizing options are being checked by GGHBox/GGVBox/etc., so we only need
        //   to figure out the size of this node.
        available_size = _resolve_component_size( this, available_size );
        auto inner_size = _with_margins( Rect2(Vector2(0,0), available_size) ).size;
        if (! limited || horizontal_mode == S_SHRINK_TO_FIT || vertical_mode == S_SHRINK_TO_FIT)
        {
            _resolve_child_sizes( inner_size, limited );
            _resolve_shrink_to_fit_size( this, available_size );
        }
    }
    void _update_layout()
    {
        if (! _is_top_level || _layout_stage == 2)
            return;
        _layout_stage = 2;

        _update_safe_area();

        emit_signal("begin_layout");

        _update_size();
        _resolve_size( get_size() );
        _perform_layout( _with_margins(Rect2(Vector2(0,0),get_size())) );

        emit_signal("end_layout");
        _layout_stage = 0;

    }
    void _update_size()
    {
        _on_update_size();

        for(int i = 0;i < get_child_count(); ++i)
        {
            auto child = get_child(i);
            GUIComponent* com = Object::cast_to<GUIComponent>( child ); // child->cast_to<GUIComponent>();
            if (com)
                com->_update_size();
            else if( child->has_method("_on_update_size"))
                child->call("_on_update_size");
        }

    }
    public :
    virtual void _enter_tree() override
    {
        _is_top_level = ! (get_parent()->is_class("GUIComponent"));
        if (_is_top_level)
        {
            connect( "resized", Callable(this, "_on_resized") );
            connect( "sort_children", Callable(this, "_on_sort_children") );
            _update_safe_area();

        }

        connect( "child_entered_tree", Callable(this, "_on_child_entered_tree") );
        connect( "child_exiting_tree", Callable(this, "_on_child_exiting_tree") );
        connect( "child_order_changed", Callable(this, "request_layout") );

        request_layout();

    }
    virtual void _exit_tree() override
    {
        if (_is_top_level)
        {
            disconnect( "resized", Callable(this,"request_layout"));
            disconnect( "sort_children", Callable(this,"_on_sort_children" ));
        }
        disconnect( "child_entered_tree", Callable(this,"_on_child_entered_tree" ));
        disconnect( "child_exiting_tree", Callable(this,"_on_child_exiting_tree" ));
        disconnect( "child_order_changed", Callable(this,"request_layout" ));

        request_layout();

    }
    virtual void _on_child_entered_tree( Node *child )
    {
        Control* con = Object::cast_to<Control>( child ); // child->cast_to<Control>();
        if (con) con->connect("visibility_changed", Callable(this, "request_layout"));
        if (Engine::get_singleton()->is_editor_hint() && con)
        {
            child->connect("minimum_size_changed", Callable(this,"request_layout") );
            child->connect("resized", Callable(this,"request_layout") );
            child->connect("size_flags_changed", Callable(this,"request_layout") );
        }
        request_layout();

    }
    virtual void _on_child_exiting_tree( Node * child )
    {
        Control* con = Object::cast_to<Control>( child ); // child->cast_to<Control>();
        if (con) disconnect("visibility_changed",Callable(this,"request_layout" ));
        if (Engine::get_singleton()->is_editor_hint() && con)
        {
            disconnect( "minimum_size_changed",Callable(this,"request_layout" ));
            disconnect( "resized", Callable(this,"request_layout" ));
            disconnect( "size_flags_changed",Callable(this,"request_layout" ));

        }

        request_layout();

    }
    virtual void _on_child_visibility_changed()
    {
        request_layout();

    }

    virtual void _on_sort_children()
    {
        _update_layout();

    }
    virtual void _update_safe_area()
    {
        auto viewport = get_viewport();
        if (viewport)
        {
            auto display_size = viewport->get_visible_rect().size;
            auto safe_area = Rect2( Vector2(0,0), display_size );

            if( DisplayServer::get_singleton()->window_get_mode() == DisplayServer::WINDOW_MODE_FULLSCREEN 
                || DisplayServer::get_singleton()->window_get_mode() == DisplayServer::WINDOW_MODE_EXCLUSIVE_FULLSCREEN)
                    safe_area = DisplayServer::get_singleton()->get_display_safe_area();

            set_parameter( "safe_area_left_margin", safe_area.position.x );
            set_parameter( "safe_area_top_margin", safe_area.position.y );
            set_parameter( "safe_area_right_margin", display_size.x - safe_area.get_end().x );
            set_parameter( "safe_area_bottom_margin", display_size.y - safe_area.get_end().y );
        }
        
    }









};


VARIANT_ENUM_CAST(GUIComponent::ScalingMode)
VARIANT_ENUM_CAST(GUIComponent::TextSizeMode)
VARIANT_ENUM_CAST(GUIComponent::FillMode)


#endif