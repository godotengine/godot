#include "spx_input_proxy.h"
#include "spx_engine.h"

#define SPX_CALLBACK SpxEngine::get_singleton()->get_callbacks()
void SpxInputProxy::ready() {
	set_process_input(true);
}

void SpxInputProxy::input(const Ref<InputEvent> &p_event) {
	Ref<InputEventKey> k = p_event;
	if (k.is_valid()) {
		if (k->is_pressed()) {
			SPX_CALLBACK->func_on_key_pressed((GdInt)k->get_keycode());
		} else if (k->is_released()) {
			SPX_CALLBACK->func_on_key_released((GdInt)k->get_keycode());
		}
	}
}
