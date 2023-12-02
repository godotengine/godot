#include "joypad_switch.h"

#include <iostream>

//when only both joy-con are use as a single controller (general case)
const std::vector<std::pair<uint, JoyButton>> JoypadSwitch::switch_joy_dual_button_map = {
	{ HidNpadButton_A, JoyButton::A },
	{ HidNpadButton_B, JoyButton::B },
	{ HidNpadButton_X, JoyButton::X },
	{ HidNpadButton_Y, JoyButton::Y },
	{ HidNpadButton_StickL, JoyButton::LEFT_STICK },
	{ HidNpadButton_StickR, JoyButton::RIGHT_STICK },
	{ HidNpadButton_L, JoyButton::LEFT_SHOULDER },
	{ HidNpadButton_R, JoyButton::RIGHT_SHOULDER },
	{ HidNpadButton_ZL, JoyButton::PADDLE1 }, //using axis TRIGGER instead ?
	{ HidNpadButton_ZR, JoyButton::PADDLE2 }, //using axis TRIGGER instead ?
	{ HidNpadButton_Plus, JoyButton::START },
	{ HidNpadButton_Minus, JoyButton::BACK },
	{ HidNpadButton_Left, JoyButton::DPAD_LEFT },
	{ HidNpadButton_Up, JoyButton::DPAD_UP },
	{ HidNpadButton_Right, JoyButton::DPAD_RIGHT },
	{ HidNpadButton_Down, JoyButton::DPAD_DOWN },
};

//when only right joy-con is use as a controller horizontally
const std::vector<std::pair<uint, JoyButton>> JoypadSwitch::switch_joy_right_button_map = {
	{ HidNpadButton_A, JoyButton::B },
	{ HidNpadButton_B, JoyButton::Y },
	{ HidNpadButton_X, JoyButton::A },
	{ HidNpadButton_Y, JoyButton::X },
	{ HidNpadButton_ZR, JoyButton::MISC1 }, //not sure twhat to do with this button
	{ HidNpadButton_Plus, JoyButton::START },
	{ HidNpadButton_StickRLeft, JoyButton::DPAD_UP },
	{ HidNpadButton_StickRUp, JoyButton::DPAD_RIGHT },
	{ HidNpadButton_StickRRight, JoyButton::DPAD_DOWN },
	{ HidNpadButton_StickRDown, JoyButton::DPAD_LEFT },
	{ HidNpadButton_RightSL, JoyButton::LEFT_SHOULDER },
	{ HidNpadButton_RightSR, JoyButton::RIGHT_SHOULDER }
};

//when only left joy-con is use as a controller horizontally
const std::vector<std::pair<uint, JoyButton>> JoypadSwitch::switch_joy_left_button_map = {
	{ HidNpadButton_ZL, JoyButton::MISC1 }, //not sure twhat to do with this button
	{ HidNpadButton_Minus, JoyButton::START },
	{ HidNpadButton_Left, JoyButton::B },
	{ HidNpadButton_Up, JoyButton::Y },
	{ HidNpadButton_Right, JoyButton::X },
	{ HidNpadButton_Down, JoyButton::A },
	{ HidNpadButton_StickLLeft, JoyButton::DPAD_DOWN },
	{ HidNpadButton_StickLUp, JoyButton::DPAD_LEFT },
	{ HidNpadButton_StickLRight, JoyButton::DPAD_RIGHT },
	{ HidNpadButton_StickLDown, JoyButton::DPAD_RIGHT },
	{ HidNpadButton_LeftSL, JoyButton::LEFT_SHOULDER },
	{ HidNpadButton_LeftSR, JoyButton::RIGHT_SHOULDER }
};

void JoypadSwitch::initialize(Input *input) {
	print_line("JoypadSwitch::initialize");

	_input = input;

	//accept up to 8 controllers, all modes
	padConfigureInput(_pads.size(), HidNpadStyleSet_NpadStandard);
	// first controler initialized as is #1 AND handheld
	padInitialize(&_pads[0], HidNpadIdType_No1, HidNpadIdType_Handheld);
	// from 2 -> 8 controller controler initialized as is #N
	for (uint i = 1; i < _pads.size(); i++) {
		_pads[i].id = i;
		padInitialize(&_pads[i], HidNpadIdType(i));
	}
}

void JoypadSwitch::discover_pad(PadStateSwitch &pad) {
	print_line("JoypadSwitch::discover_pad(" + String::num(pad.id) + ")");

	pad.initialized = true;
	bool solo = false;
	String joy_name = "switch-pad-" + String::num(pad.id);
	if (pad.style_set & HidNpadStyleTag_NpadJoyLeft) {
		pad.mapping = switch_joy_left_button_map;
		joy_name += "::solo-left";
		solo = true;
	} else if (pad.style_set & HidNpadStyleTag_NpadJoyRight) {
		pad.mapping = switch_joy_right_button_map;
		joy_name += "::solo-right";
		solo = true;
	} else if (pad.style_set & HidNpadStyleTag_NpadJoyDual) {
		pad.mapping = switch_joy_dual_button_map;
		joy_name += "::dual";
	} else if (pad.style_set & HidNpadStyleTag_NpadFullKey) {
		pad.mapping = switch_joy_dual_button_map;
		joy_name += "::pro";
	} else if (pad.style_set & HidNpadStyleTag_NpadHandheld) {
		pad.mapping = switch_joy_dual_button_map;
		joy_name += "::handheld";
	} else {
		pad.mapping = switch_joy_dual_button_map;
		joy_name += "::other";
	}

	if (solo) {
		HidNpadControllerColor color;
		hidGetNpadControllerColorSingle((HidNpadIdType)pad.id, &color);
		joy_name += "::#" + String::num_int64(color.main, 16);
	} else {
		HidNpadControllerColor color_l, color_r;
		hidGetNpadControllerColorSplit((HidNpadIdType)pad.id, &color_l, &color_r);
		joy_name += "::#" + String::num_int64(color_l.main, 16);
		joy_name += "::#" + String::num_int64(color_r.main, 16);
	}

	_input->joy_connection_changed(pad.id, true, joy_name);
	std::cout << "joy_connection_changed pad(" << pad.id << ") "
			  << "name(" << joy_name.utf8().get_data() << ") "
			  << "read_handheld(" << pad.read_handheld << ") "
			  << "active_handheld(" << pad.active_handheld << ") "
			  << "attributes(" << pad.attributes << ") "
			  << "style_set(" << pad.style_set << ")" << std::endl;
}

void JoypadSwitch::dispatch(PadStateSwitch &pad) {
}

void JoypadSwitch::process() {
	for (uint i = 0; i < _pads.size(); i++) {
		PadStateSwitch &pad = _pads[i];
		padUpdate(&pad);

		u64 kDown = padGetButtonsDown(&pad);
		u64 kUp = padGetButtonsUp(&pad);

		if (!pad.initialized && kDown) {
			discover_pad(pad);
		}

		for (const auto &button : pad.mapping) {
			if (kDown & button.first) {
				_input->joy_button(pad.id, button.second, true);
			}
			if (kUp & button.first) {
				_input->joy_button(pad.id, button.second, false);
			}
		}

		HidAnalogStickState leftStick = pad.sticks[0];
		HidAnalogStickState rightStick = pad.sticks[1];

		if (pad.style_set & HidNpadStyleTag_NpadJoyLeft) {
			// only left stick available and rotated 90 anti-clock wise
			_input->joy_axis(i, JoyAxis::LEFT_Y, (float)(leftStick.x) / float(JOYSTICK_MAX));
			_input->joy_axis(i, JoyAxis::LEFT_X, -(float)(leftStick.y) / float(JOYSTICK_MAX));
		} else if (pad.style_set & HidNpadStyleTag_NpadJoyRight) {
			// only left stick available and rotated 90 clock wise
			_input->joy_axis(i, JoyAxis::LEFT_Y, -(float)(rightStick.x) / float(JOYSTICK_MAX));
			_input->joy_axis(i, JoyAxis::LEFT_X, (float)(rightStick.y) / float(JOYSTICK_MAX));
		} else {
			// both sticks no rotations
			_input->joy_axis(i, JoyAxis::LEFT_X, (float)(leftStick.x) / float(JOYSTICK_MAX));
			_input->joy_axis(i, JoyAxis::LEFT_Y, (float)(leftStick.y) / float(JOYSTICK_MAX));
			_input->joy_axis(i, JoyAxis::RIGHT_X, (float)(rightStick.x) / float(JOYSTICK_MAX));
			_input->joy_axis(i, JoyAxis::RIGHT_Y, (float)(rightStick.y) / float(JOYSTICK_MAX));
		}
	}
}

JoypadSwitch::JoypadSwitch() {
}