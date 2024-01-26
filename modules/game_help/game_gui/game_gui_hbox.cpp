#include "game_gui_hbox.h"

void GUIHBox::_bind_methods()
{   

    ClassDB::bind_method(D_METHOD("set_content_alignment", "mode"), &GUIHBox::set_content_alignment);
    ClassDB::bind_method(D_METHOD("get_content_alignment"), &GUIHBox::get_content_alignment);

    
	ADD_GROUP("HBox", "hbox_");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "hbox_content_alignment", PROPERTY_HINT_ENUM, "LEFT,CENTER,RIGHT"), "set_content_alignment", "get_content_alignment");


}