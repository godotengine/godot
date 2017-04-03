#include "godot_input_event.h"

#include "os/input_event.h"

#ifdef __cplusplus
extern "C" {
#endif

void _input_event_api_anchor() {
}

void GDAPI godot_input_event_new(godot_input_event *p_ie) {
	InputEvent *ie = (InputEvent *)p_ie;
	*ie = InputEvent();
}

godot_bool GDAPI godot_input_event_is_pressed(const godot_input_event *p_ie) {
	const InputEvent *ie = (const InputEvent *)p_ie;
	return ie->is_pressed();
}

godot_bool GDAPI godot_input_event_is_action(const godot_input_event *p_ie, const godot_string *p_action) {
	const InputEvent *ie = (const InputEvent *)p_ie;
	const String *action = (const String *)p_action;
	return ie->is_action(*action);
}

godot_bool GDAPI godot_input_event_is_action_pressed(const godot_input_event *p_ie, const godot_string *p_action) {
	const InputEvent *ie = (const InputEvent *)p_ie;
	const String *action = (const String *)p_action;
	return ie->is_action_pressed(*action);
}

godot_bool GDAPI godot_input_event_is_action_released(const godot_input_event *p_ie, const godot_string *p_action) {
	const InputEvent *ie = (const InputEvent *)p_ie;
	const String *action = (const String *)p_action;
	return ie->is_action_released(*action);
}

godot_bool GDAPI godot_input_event_is_echo(const godot_input_event *p_ie) {
	const InputEvent *ie = (const InputEvent *)p_ie;
	return ie->is_echo();
}

void GDAPI godot_input_event_set_as_action(godot_input_event *p_ie, const godot_string *p_action, const godot_bool p_pressed) {
	InputEvent *ie = (InputEvent *)p_ie;
	const String *action = (const String *)p_action;
	return ie->set_as_action(*action, p_pressed);
}

godot_string GDAPI godot_input_event_as_string(const godot_input_event *p_ie) {
	const InputEvent *ie = (const InputEvent *)p_ie;
	godot_string str;
	String *s = (String *)&str;
	memnew_placement(s, String);
	*s = (String)*ie;
	return str;
}

uint32_t GDAPI *godot_input_event_get_id(godot_input_event *p_ie) {
	InputEvent *ie = (InputEvent *)p_ie;
	return &ie->ID;
}

godot_input_event_type GDAPI *godot_input_event_get_type(godot_input_event *p_ie) {
	InputEvent *ie = (InputEvent *)p_ie;
	return (godot_input_event_type *)&ie->type;
}

godot_int GDAPI *godot_input_event_get_device(godot_input_event *p_ie) {
	InputEvent *ie = (InputEvent *)p_ie;
	return &ie->device;
}

static InputModifierState *_get_mod_for_type(InputEvent *ie) {
	switch (ie->type) {
		case InputEvent::MOUSE_BUTTON:
			return &ie->mouse_button.mod;
		case InputEvent::MOUSE_MOTION:
			return &ie->mouse_motion.mod;
		case InputEvent::KEY:
			return &ie->key.mod;
		default:
			return 0;
	}
}

godot_bool GDAPI *godot_input_event_mod_get_alt(godot_input_event *p_event) {
	InputEvent *ie = (InputEvent *)p_event;
	InputModifierState *mod = _get_mod_for_type(ie);
	return &mod->alt;
}

godot_bool GDAPI *godot_input_event_mod_get_ctrl(godot_input_event *p_event) {
	InputEvent *ie = (InputEvent *)p_event;
	InputModifierState *mod = _get_mod_for_type(ie);
	return &mod->control;
}

godot_bool GDAPI *godot_input_event_mod_get_command(godot_input_event *p_event) {
	InputEvent *ie = (InputEvent *)p_event;
	InputModifierState *mod = _get_mod_for_type(ie);
	return &mod->command;
}

godot_bool GDAPI *godot_input_event_mod_get_shift(godot_input_event *p_event) {
	InputEvent *ie = (InputEvent *)p_event;
	InputModifierState *mod = _get_mod_for_type(ie);
	return &mod->shift;
}

godot_bool GDAPI *godot_input_event_mod_get_meta(godot_input_event *p_event) {
	InputEvent *ie = (InputEvent *)p_event;
	InputModifierState *mod = _get_mod_for_type(ie);
	return &mod->meta;
}

uint32_t GDAPI *godot_input_event_key_get_scancode(godot_input_event *p_event) {
	InputEvent *ie = (InputEvent *)p_event;
	return &ie->key.scancode;
}

uint32_t GDAPI *godot_input_event_key_get_unicode(godot_input_event *p_event) {
	InputEvent *ie = (InputEvent *)p_event;
	return &ie->key.unicode;
}

godot_bool GDAPI *godot_input_event_key_get_pressed(godot_input_event *p_event) {
	InputEvent *ie = (InputEvent *)p_event;
	return &ie->key.pressed;
}

godot_bool GDAPI *godot_input_event_key_get_echo(godot_input_event *p_event) {
	InputEvent *ie = (InputEvent *)p_event;
	return &ie->key.echo;
}

float GDAPI *godot_input_event_mouse_get_x(godot_input_event *p_event) {
	InputEvent *ie = (InputEvent *)p_event;
	return &ie->mouse_button.x;
}

float GDAPI *godot_input_event_mouse_get_y(godot_input_event *p_event) {
	InputEvent *ie = (InputEvent *)p_event;
	return &ie->mouse_button.y;
}

float GDAPI *godot_input_event_mouse_get_global_x(godot_input_event *p_event) {
	InputEvent *ie = (InputEvent *)p_event;
	return &ie->mouse_button.global_x;
}

float GDAPI *godot_input_event_mouse_get_global_y(godot_input_event *p_event) {
	InputEvent *ie = (InputEvent *)p_event;
	return &ie->mouse_button.global_y;
}

godot_int GDAPI *godot_input_event_mouse_get_button_mask(godot_input_event *p_event) {
	InputEvent *ie = (InputEvent *)p_event;
	return &ie->mouse_button.button_mask;
}

godot_int GDAPI *godot_input_event_mouse_button_get_button_index(godot_input_event *p_event) {
	InputEvent *ie = (InputEvent *)p_event;
	return &ie->mouse_button.button_index;
}

godot_bool GDAPI *godot_input_event_mouse_button_get_pressed(godot_input_event *p_event) {
	InputEvent *ie = (InputEvent *)p_event;
	return &ie->mouse_button.pressed;
}

godot_bool GDAPI *godot_input_event_mouse_button_get_doubleclick(godot_input_event *p_event) {
	InputEvent *ie = (InputEvent *)p_event;
	return &ie->mouse_button.doubleclick;
}

float GDAPI *godot_input_event_mouse_motion_get_relative_x(godot_input_event *p_event) {
	InputEvent *ie = (InputEvent *)p_event;
	return &ie->mouse_motion.relative_x;
}

float GDAPI *godot_input_event_mouse_motion_get_relative_y(godot_input_event *p_event) {
	InputEvent *ie = (InputEvent *)p_event;
	return &ie->mouse_motion.relative_y;
}

float GDAPI *godot_input_event_mouse_motion_get_speed_x(godot_input_event *p_event) {
	InputEvent *ie = (InputEvent *)p_event;
	return &ie->mouse_motion.speed_x;
}

float GDAPI *godot_input_event_mouse_motion_get_speed_y(godot_input_event *p_event) {
	InputEvent *ie = (InputEvent *)p_event;
	return &ie->mouse_motion.speed_y;
}

godot_int GDAPI *godot_input_event_joypad_motion_get_axis(godot_input_event *p_event) {
	InputEvent *ie = (InputEvent *)p_event;
	return &ie->joy_motion.axis;
}

float GDAPI *godot_input_event_joypad_motion_get_axis_value(godot_input_event *p_event) {
	InputEvent *ie = (InputEvent *)p_event;
	return &ie->joy_motion.axis_value;
}

godot_int GDAPI *godot_input_event_joypad_button_get_button_index(godot_input_event *p_event) {
	InputEvent *ie = (InputEvent *)p_event;
	return &ie->joy_button.button_index;
}

godot_bool GDAPI *godot_input_event_joypad_button_get_pressed(godot_input_event *p_event) {
	InputEvent *ie = (InputEvent *)p_event;
	return &ie->joy_button.pressed;
}

float GDAPI *godot_input_event_joypad_button_get_pressure(godot_input_event *p_event) {
	InputEvent *ie = (InputEvent *)p_event;
	return &ie->joy_button.pressure;
}

godot_int GDAPI *godot_input_event_screen_touch_get_index(godot_input_event *p_event) {
	InputEvent *ie = (InputEvent *)p_event;
	return &ie->screen_touch.index;
}

float GDAPI *godot_input_event_screen_touch_get_x(godot_input_event *p_event) {
	InputEvent *ie = (InputEvent *)p_event;
	return &ie->screen_touch.x;
}

float GDAPI *godot_input_event_screen_touch_get_y(godot_input_event *p_event) {
	InputEvent *ie = (InputEvent *)p_event;
	return &ie->screen_touch.y;
}

godot_bool GDAPI *godot_input_event_screen_touch_get_pressed(godot_input_event *p_event) {
	InputEvent *ie = (InputEvent *)p_event;
	return &ie->screen_touch.pressed;
}

godot_int GDAPI *godot_input_event_screen_drag_get_index(godot_input_event *p_event) {
	InputEvent *ie = (InputEvent *)p_event;
	return &ie->screen_drag.index;
}

float GDAPI *godot_input_event_screen_drag_get_x(godot_input_event *p_event) {
	InputEvent *ie = (InputEvent *)p_event;
	return &ie->screen_drag.x;
}

float GDAPI *godot_input_event_screen_drag_get_y(godot_input_event *p_event) {
	InputEvent *ie = (InputEvent *)p_event;
	return &ie->screen_drag.y;
}

float GDAPI *godot_input_event_screen_drag_get_relative_x(godot_input_event *p_event) {
	InputEvent *ie = (InputEvent *)p_event;
	return &ie->screen_drag.relative_x;
}

float GDAPI *godot_input_event_screen_drag_get_relative_y(godot_input_event *p_event) {
	InputEvent *ie = (InputEvent *)p_event;
	return &ie->screen_drag.relative_y;
}

float GDAPI *godot_input_event_screen_drag_get_speed_x(godot_input_event *p_event) {
	InputEvent *ie = (InputEvent *)p_event;
	return &ie->screen_drag.speed_x;
}

float GDAPI *godot_input_event_screen_drag_get_speed_y(godot_input_event *p_event) {
	InputEvent *ie = (InputEvent *)p_event;
	return &ie->screen_drag.speed_y;
}

#ifdef __cplusplus
}
#endif
