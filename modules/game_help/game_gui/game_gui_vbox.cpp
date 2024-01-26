#include "game_gui_vbox.h"

void GUIVBox::_bind_methods()
{   

    ClassDB::bind_method(D_METHOD("set_content_alignment", "mode"), &GUIVBox::set_content_alignment);
    ClassDB::bind_method(D_METHOD("get_content_alignment"), &GUIVBox::get_content_alignment);

    
	ADD_GROUP("VBox", "vbox_");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "vbox_content_alignment", PROPERTY_HINT_ENUM, "TOP,CENTER,BOTTOM"), "set_content_alignment", "get_content_alignment");


}