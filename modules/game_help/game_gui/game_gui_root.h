#ifndef GAME_GUI_VBOX_H
#define GAME_GUI_VBOX_H
#include "game_gui_compoent.h"

class GUIRoot : public GUIComponent
{
    GDCLASS(GUIRoot, GUIComponent);
    static void _bind_methods()
    {
        ClassDB::bind_method(D_METHOD("set_initial_window_size", "value"), &GUIRoot::set_initial_window_size);
        ClassDB::bind_method(D_METHOD("get_initial_window_size"), &GUIRoot::get_initial_window_size);

        ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "initial_window_size"), "set_initial_window_size", "get_initial_window_size");

    }
public:
    Vector2 initial_window_size = Vector2(1280,720);

	void set_initial_window_size(Vector2 value)
    {
		if (initial_window_size == value) return;
		initial_window_size = value;

		auto viewport = get_viewport();
		if (! viewport) return;  // engine is setting the initial value of this property
		if (! Engine::get_singleton()->is_editor_hint() || viewport != get_parent())
         return;

		if (value != size)
			size = value;
		request_layout();

    }
    Vector2 get_initial_window_size()
    {
        return initial_window_size;
    }
    
    virtual void _ready()override
    {
		if (! Engine::get_singleton()->is_editor_hint() || viewport != get_parent())
        {
            DisplayServerwindow::get_singleton()->window_set_size( initial_window_size );

            auto screen_size = Vector2( DisplayServerwindow::get_singleton()->screen_get_size());
            auto pos = Vector2i( (screen_size-initial_window_size)/2.0 );
            DisplayServerwindow::get_singleton()-> _set_position( pos );

            set_anchors_and_offsets_preset( PRESET_FULL_RECT );

        }
        else
            set_initial_window_size( get_size());

    }
    virtual void _enter_tree()override
    {
        __supper::_enter_tree();
        connect( "size_changed", Callable(this, "_on_resized"));

    }

    virtual void _exit_tree()override
    {
        __supper::_exit_tree();
        disconnect( "size_changed", Callable(this, "_on_resized"));

    }
            
    virtual void _on_resized()
    {
        if (Engine::get_singleton()->is_editor_hint())
            set_initial_window_size( get_size());
    }
};
#endif