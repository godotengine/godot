#include "game_gui_compoent.h"

void GGComponent::_bind_methods()
{
    ADD_SIGNAL(MethodInfo("begin_layout"));
    ADD_SIGNAL(MethodInfo("end_layout"));

}
