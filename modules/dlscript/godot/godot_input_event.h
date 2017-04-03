#ifndef GODOT_DLSCRIPT_INPUT_EVENT_H
#define GODOT_DLSCRIPT_INPUT_EVENT_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#ifndef GODOT_CORE_API_GODOT_INPUT_EVENT_TYPE_DEFINED
typedef struct godot_input_event {
	uint8_t _dont_touch_that[56];
} godot_input_event;
#endif

enum godot_input_event_type {
	GODOT_INPUT_EVENT_TYPE_NONE,
	GODOT_INPUT_EVENT_TYPE_KEY,
	GODOT_INPUT_EVENT_TYPE_MOUSE_MOTION,
	GODOT_INPUT_EVENT_TYPE_MOUSE_BUTTON,
	GODOT_INPUT_EVENT_TYPE_JOYPAD_MOTION,
	GODOT_INPUT_EVENT_TYPE_JOYPAD_BUTTON,
	GODOT_INPUT_EVENT_TYPE_SCREEN_TOUCH,
	GODOT_INPUT_EVENT_TYPE_SCREEN_DRAG,
	GODOT_INPUT_EVENT_TYPE_ACTION,
	GODOT_INPUT_EVENT_TYPE_TYPE_MAX
};
typedef enum godot_input_event_type godot_input_event_type;

enum {
	GODOT_BUTTON_LEFT = 1,
	GODOT_BUTTON_RIGHT = 2,
	GODOT_BUTTON_MIDDLE = 3,
	GODOT_BUTTON_WHEEL_UP = 4,
	GODOT_BUTTON_WHEEL_DOWN = 5,
	GODOT_BUTTON_WHEEL_LEFT = 6,
	GODOT_BUTTON_WHEEL_RIGHT = 7,
	GODOT_BUTTON_MASK_LEFT = (1 << (GODOT_BUTTON_LEFT - 1)),
	GODOT_BUTTON_MASK_RIGHT = (1 << (GODOT_BUTTON_RIGHT - 1)),
	GODOT_BUTTON_MASK_MIDDLE = (1 << (GODOT_BUTTON_MIDDLE - 1)),

};

enum {

	GODOT_JOY_BUTTON_0 = 0,
	GODOT_JOY_BUTTON_1 = 1,
	GODOT_JOY_BUTTON_2 = 2,
	GODOT_JOY_BUTTON_3 = 3,
	GODOT_JOY_BUTTON_4 = 4,
	GODOT_JOY_BUTTON_5 = 5,
	GODOT_JOY_BUTTON_6 = 6,
	GODOT_JOY_BUTTON_7 = 7,
	GODOT_JOY_BUTTON_8 = 8,
	GODOT_JOY_BUTTON_9 = 9,
	GODOT_JOY_BUTTON_10 = 10,
	GODOT_JOY_BUTTON_11 = 11,
	GODOT_JOY_BUTTON_12 = 12,
	GODOT_JOY_BUTTON_13 = 13,
	GODOT_JOY_BUTTON_14 = 14,
	GODOT_JOY_BUTTON_15 = 15,
	GODOT_JOY_BUTTON_MAX = 16,

	GODOT_JOY_L = GODOT_JOY_BUTTON_4,
	GODOT_JOY_R = GODOT_JOY_BUTTON_5,
	GODOT_JOY_L2 = GODOT_JOY_BUTTON_6,
	GODOT_JOY_R2 = GODOT_JOY_BUTTON_7,
	GODOT_JOY_L3 = GODOT_JOY_BUTTON_8,
	GODOT_JOY_R3 = GODOT_JOY_BUTTON_9,
	GODOT_JOY_SELECT = GODOT_JOY_BUTTON_10,
	GODOT_JOY_START = GODOT_JOY_BUTTON_11,
	GODOT_JOY_DPAD_UP = GODOT_JOY_BUTTON_12,
	GODOT_JOY_DPAD_DOWN = GODOT_JOY_BUTTON_13,
	GODOT_JOY_DPAD_LEFT = GODOT_JOY_BUTTON_14,
	GODOT_JOY_DPAD_RIGHT = GODOT_JOY_BUTTON_15,

	// a little history about game controllers (who copied who)

	GODOT_JOY_SNES_B = GODOT_JOY_BUTTON_0,
	GODOT_JOY_SNES_A = GODOT_JOY_BUTTON_1,
	GODOT_JOY_SNES_Y = GODOT_JOY_BUTTON_2,
	GODOT_JOY_SNES_X = GODOT_JOY_BUTTON_3,

	GODOT_JOY_SONY_CIRCLE = GODOT_JOY_SNES_A,
	GODOT_JOY_SONY_X = GODOT_JOY_SNES_B,
	GODOT_JOY_SONY_SQUARE = GODOT_JOY_SNES_Y,
	GODOT_JOY_SONY_TRIANGLE = GODOT_JOY_SNES_X,

	GODOT_JOY_SEGA_B = GODOT_JOY_SNES_A,
	GODOT_JOY_SEGA_A = GODOT_JOY_SNES_B,
	GODOT_JOY_SEGA_X = GODOT_JOY_SNES_Y,
	GODOT_JOY_SEGA_Y = GODOT_JOY_SNES_X,

	GODOT_JOY_XBOX_B = GODOT_JOY_SEGA_B,
	GODOT_JOY_XBOX_A = GODOT_JOY_SEGA_A,
	GODOT_JOY_XBOX_X = GODOT_JOY_SEGA_X,
	GODOT_JOY_XBOX_Y = GODOT_JOY_SEGA_Y,

	GODOT_JOY_DS_A = GODOT_JOY_SNES_A,
	GODOT_JOY_DS_B = GODOT_JOY_SNES_B,
	GODOT_JOY_DS_X = GODOT_JOY_SNES_X,
	GODOT_JOY_DS_Y = GODOT_JOY_SNES_Y,

	GODOT_JOY_WII_C = GODOT_JOY_BUTTON_5,
	GODOT_JOY_WII_Z = GODOT_JOY_BUTTON_6,

	GODOT_JOY_WII_MINUS = GODOT_JOY_BUTTON_9,
	GODOT_JOY_WII_PLUS = GODOT_JOY_BUTTON_10,

	// end of history

	GODOT_JOY_AXIS_0 = 0,
	GODOT_JOY_AXIS_1 = 1,
	GODOT_JOY_AXIS_2 = 2,
	GODOT_JOY_AXIS_3 = 3,
	GODOT_JOY_AXIS_4 = 4,
	GODOT_JOY_AXIS_5 = 5,
	GODOT_JOY_AXIS_6 = 6,
	GODOT_JOY_AXIS_7 = 7,
	GODOT_JOY_AXIS_MAX = 8,

	GODOT_JOY_ANALOG_0_X = GODOT_JOY_AXIS_0,
	GODOT_JOY_ANALOG_0_Y = GODOT_JOY_AXIS_1,

	GODOT_JOY_ANALOG_1_X = GODOT_JOY_AXIS_2,
	GODOT_JOY_ANALOG_1_Y = GODOT_JOY_AXIS_3,

	GODOT_JOY_ANALOG_2_X = GODOT_JOY_AXIS_4,
	GODOT_JOY_ANALOG_2_Y = GODOT_JOY_AXIS_5,

	GODOT_JOY_ANALOG_L2 = GODOT_JOY_AXIS_6,
	GODOT_JOY_ANALOG_R2 = GODOT_JOY_AXIS_7,
};

#include "../godot.h"

void GDAPI godot_input_event_new(godot_input_event *p_ie);

godot_bool GDAPI godot_input_event_is_pressed(const godot_input_event *p_ie);
godot_bool GDAPI godot_input_event_is_action(const godot_input_event *p_ie, const godot_string *p_action);
godot_bool GDAPI godot_input_event_is_action_pressed(const godot_input_event *p_ie, const godot_string *p_action);
godot_bool GDAPI godot_input_event_is_action_released(const godot_input_event *p_ie, const godot_string *p_action);
godot_bool GDAPI godot_input_event_is_echo(const godot_input_event *p_ie);
void GDAPI godot_input_event_set_as_action(godot_input_event *p_ie, const godot_string *p_action, const godot_bool p_pressed);

godot_string GDAPI godot_input_event_as_string(const godot_input_event *p_ie);

// Note:
// We're returning pointers to the fields in the unions.
// This is because I'm too lazy to write setter functions

uint32_t GDAPI *godot_input_event_get_id(godot_input_event *p_ie);
godot_input_event_type GDAPI *godot_input_event_get_type(godot_input_event *p_ie);
godot_int GDAPI *godot_input_event_get_device(godot_input_event *p_ie);

godot_bool GDAPI *godot_input_event_mod_get_alt(godot_input_event *p_event);
godot_bool GDAPI *godot_input_event_mod_get_ctrl(godot_input_event *p_event);
godot_bool GDAPI *godot_input_event_mod_get_command(godot_input_event *p_event);
godot_bool GDAPI *godot_input_event_mod_get_shift(godot_input_event *p_event);
godot_bool GDAPI *godot_input_event_mod_get_meta(godot_input_event *p_event);

uint32_t GDAPI *godot_input_event_key_get_scancode(godot_input_event *p_event);
uint32_t GDAPI *godot_input_event_key_get_unicode(godot_input_event *p_event);
godot_bool GDAPI *godot_input_event_key_get_pressed(godot_input_event *p_event);
godot_bool GDAPI *godot_input_event_key_get_echo(godot_input_event *p_event);

float GDAPI *godot_input_event_mouse_get_x(godot_input_event *p_event);
float GDAPI *godot_input_event_mouse_get_y(godot_input_event *p_event);
float GDAPI *godot_input_event_mouse_get_global_x(godot_input_event *p_event);
float GDAPI *godot_input_event_mouse_get_global_y(godot_input_event *p_event);
godot_int GDAPI *godot_input_event_mouse_get_button_mask(godot_input_event *p_event);

godot_int GDAPI *godot_input_event_mouse_button_get_button_index(godot_input_event *p_event);
godot_bool GDAPI *godot_input_event_mouse_button_get_pressed(godot_input_event *p_event);
godot_bool GDAPI *godot_input_event_mouse_button_get_doubleclick(godot_input_event *p_event);

float GDAPI *godot_input_event_mouse_motion_get_relative_x(godot_input_event *p_event);
float GDAPI *godot_input_event_mouse_motion_get_relative_y(godot_input_event *p_event);
float GDAPI *godot_input_event_mouse_motion_get_speed_x(godot_input_event *p_event);
float GDAPI *godot_input_event_mouse_motion_get_speed_y(godot_input_event *p_event);

godot_int GDAPI *godot_input_event_joypad_motion_get_axis(godot_input_event *p_event);
float GDAPI *godot_input_event_joypad_motion_get_axis_value(godot_input_event *p_event);

godot_int GDAPI *godot_input_event_joypad_button_get_button_index(godot_input_event *p_event);
godot_bool GDAPI *godot_input_event_joypad_button_get_pressed(godot_input_event *p_event);
float GDAPI *godot_input_event_joypad_button_get_pressure(godot_input_event *p_event);

godot_int GDAPI *godot_input_event_screen_touch_get_index(godot_input_event *p_event);
float GDAPI *godot_input_event_screen_touch_get_x(godot_input_event *p_event);
float GDAPI *godot_input_event_screen_touch_get_y(godot_input_event *p_event);
godot_bool GDAPI *godot_input_event_screen_touch_get_pressed(godot_input_event *p_event);

godot_int GDAPI *godot_input_event_screen_drag_get_index(godot_input_event *p_event);
float GDAPI *godot_input_event_screen_drag_get_x(godot_input_event *p_event);
float GDAPI *godot_input_event_screen_drag_get_y(godot_input_event *p_event);
float GDAPI *godot_input_event_screen_drag_get_relative_x(godot_input_event *p_event);
float GDAPI *godot_input_event_screen_drag_get_relative_y(godot_input_event *p_event);
float GDAPI *godot_input_event_screen_drag_get_speed_x(godot_input_event *p_event);
float GDAPI *godot_input_event_screen_drag_get_speed_y(godot_input_event *p_event);

#ifdef __cplusplus
}
#endif

#endif // GODOT_DLSCRIPT_INPUT_EVENT_H
