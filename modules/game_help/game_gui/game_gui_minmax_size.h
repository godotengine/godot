#ifndef GAME_GUI_ARBITRARY_COORD_H
#define GAME_GUI_ARBITRARY_COORD_H
#include "game_gui_compoent.h"

class GUIMinMaxSize : public GUICompoent
{
    GDCLASS(GUIMinMaxSize, GUICompoent);
    static void _bind_methods();
public:

    Vector2 min_size = Vector2(0,0);
    void set_min_size(Vector2 value)
    {
        if (min_size == value) return;
        min_size = value;
        request_layout();
    }
    Vector2 get_min_size()
    {
        return min_size;
    } 

    Vector2 max_size = Vector2(0,0);
    void set_max_size(Vector2 value)
    {
        if (max_size == value) return;
        max_size = value;
        request_layout();
    }
    Vector2 get_max_size()
    {
        return max_size;
    } 

    
    int _effective_min_height()
    {
        return int(min_size.y);
    }
    int _effective_min_width()
    {
        return int(min_size.x);

    }
    int _effective_max_height()
    {
        return int(max_size.y);

    }
    int _effective_max_width()
    {
        return int(max_size.x);

    }

    virtual void _resolve_size( Vector2 available_size, bool limited=false )
    {
		available_size.x = MAX( available_size.x, _effective_min_width() );
	
		available_size.y = MAX( available_size.y, _effective_min_height() );

		available_size.x = MIN( available_size.x, _effective_max_width() );
	
		available_size.y = MIN( available_size.y, _effective_max_height() );

	    GUICompoent::_resolve_size( available_size, limited );

    }


};

#endif
