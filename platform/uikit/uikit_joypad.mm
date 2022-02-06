/*************************************************************************/
/*  uikit_joypad.mm                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#import "uikit_joypad.h"

#include "core/project_settings.h"
#include "drivers/coreaudio/audio_driver_coreaudio.h"
#include "main/main.h"
#include "uikit_os.h"

@interface UIKitJoypadObserver (JoypadSearch)

- (int)getJoyIdForControllerName:(NSString *)controllerName;

@end

UIKitJoypad::UIKitJoypad() {
	observer = [[UIKitJoypadObserver alloc] init];
	[observer startObserving];
}

UIKitJoypad::~UIKitJoypad() {
	if (observer) {
		[observer finishObserving];
		observer = nil;
	}
}

void UIKitJoypad::start_processing() {
	if (observer) {
		[observer startProcessing];
	}
}

int UIKitJoypad::joy_id_for_name(const String &p_name) {
	if (!observer) {
		return -1;
	}

	@autoreleasepool {
		NSString *controllerName = [[NSString alloc] initWithUTF8String:p_name.utf8().get_data()];

		return [observer getJoyIdForControllerName:controllerName];
	}
}

@interface UIKitJoypadObserver ()

@property(assign, nonatomic) BOOL isObserving;
@property(assign, nonatomic) BOOL isProcessing;
@property(strong, nonatomic) NSMutableDictionary *connectedJoypads;
@property(strong, nonatomic) NSMutableArray *joypadsQueue;

@end

@implementation UIKitJoypadObserver

- (instancetype)init {
	self = [super init];

	if (self) {
		[self uikit_commonInit];
	}

	return self;
}

- (void)uikit_commonInit {
	self.isObserving = NO;
	self.isProcessing = NO;
}

- (void)startProcessing {
	self.isProcessing = YES;

	for (GCController *controller in self.joypadsQueue) {
		[self addUIKitJoypad:controller];
	}

	[self.joypadsQueue removeAllObjects];
}

- (void)startObserving {
	if (self.isObserving) {
		return;
	}

	self.isObserving = YES;

	self.connectedJoypads = [NSMutableDictionary dictionary];
	self.joypadsQueue = [NSMutableArray array];

	// get told when controllers connect, this will be called right away for
	// already connected controllers
	[[NSNotificationCenter defaultCenter]
			addObserver:self
			   selector:@selector(controllerWasConnected:)
				   name:GCControllerDidConnectNotification
				 object:nil];

	// get told when controllers disconnect
	[[NSNotificationCenter defaultCenter]
			addObserver:self
			   selector:@selector(controllerWasDisconnected:)
				   name:GCControllerDidDisconnectNotification
				 object:nil];
}

- (void)finishObserving {
	if (self.isObserving) {
		[[NSNotificationCenter defaultCenter] removeObserver:self];
	}

	self.isObserving = NO;
	self.isProcessing = NO;

	self.connectedJoypads = nil;
	self.joypadsQueue = nil;
}

- (void)dealloc {
	[self finishObserving];
}

- (int)getJoyIdForController:(GCController *)controller {
	NSArray *keys = [self.connectedJoypads allKeysForObject:controller];

	for (NSNumber *key in keys) {
		int joy_id = [key intValue];
		return joy_id;
	};

	return -1;
}

- (int)getJoyIdForControllerName:(NSString *)controllerName {
	NSArray *keys = [self.connectedJoypads allKeys];

	for (NSNumber *key in keys) {
		int joy_id = [key intValue];
		GCController *controller = self.connectedJoypads[key];

		if ([controller.vendorName containsString:controllerName]) {
			return joy_id;
		}
	};

	return -1;
}

- (void)addUIKitJoypad:(GCController *)controller {
	//     get a new id for our controller
	int joy_id = OS_UIKit::get_singleton()->get_unused_joy_id();

	if (joy_id == -1) {
		printf("Couldn't retrieve new joy id\n");
		return;
	}

	// assign our player index
	if (controller.playerIndex == GCControllerPlayerIndexUnset) {
		controller.playerIndex = [self getFreePlayerIndex];
	};

	// tell Godot about our new controller
	OS_UIKit::get_singleton()->joy_connection_changed(joy_id, true, String::utf8([controller.vendorName UTF8String]));

	// add it to our dictionary, this will retain our controllers
	[self.connectedJoypads setObject:controller forKey:[NSNumber numberWithInt:joy_id]];

	// set our input handler
	[self setControllerInputHandler:controller];
}

- (void)controllerWasConnected:(NSNotification *)notification {
	// get our controller
	GCController *controller = (GCController *)notification.object;

	if (!controller) {
		printf("Couldn't retrieve new controller\n");
		return;
	}

	if ([[self.connectedJoypads allKeysForObject:controller] count] > 0) {
		printf("Controller is already registered\n");
	} else if (!self.isProcessing) {
		[self.joypadsQueue addObject:controller];
	} else {
		[self addUIKitJoypad:controller];
	}
}

- (void)controllerWasDisconnected:(NSNotification *)notification {
	// find our joystick, there should be only one in our dictionary
	GCController *controller = (GCController *)notification.object;

	if (!controller) {
		return;
	}

	NSArray *keys = [self.connectedJoypads allKeysForObject:controller];
	for (NSNumber *key in keys) {
		// tell Godot this joystick is no longer there
		int joy_id = [key intValue];
		OS_UIKit::get_singleton()->joy_connection_changed(joy_id, false, "");

		// and remove it from our dictionary
		[self.connectedJoypads removeObjectForKey:key];
	};
};

- (GCControllerPlayerIndex)getFreePlayerIndex {
	bool have_player_1 = false;
	bool have_player_2 = false;
	bool have_player_3 = false;
	bool have_player_4 = false;

	if (self.connectedJoypads == nil) {
		NSArray *keys = [self.connectedJoypads allKeys];
		for (NSNumber *key in keys) {
			GCController *controller = [self.connectedJoypads objectForKey:key];
			if (controller.playerIndex == GCControllerPlayerIndex1) {
				have_player_1 = true;
			} else if (controller.playerIndex == GCControllerPlayerIndex2) {
				have_player_2 = true;
			} else if (controller.playerIndex == GCControllerPlayerIndex3) {
				have_player_3 = true;
			} else if (controller.playerIndex == GCControllerPlayerIndex4) {
				have_player_4 = true;
			};
		};
	};

	if (!have_player_1) {
		return GCControllerPlayerIndex1;
	} else if (!have_player_2) {
		return GCControllerPlayerIndex2;
	} else if (!have_player_3) {
		return GCControllerPlayerIndex3;
	} else if (!have_player_4) {
		return GCControllerPlayerIndex4;
	} else {
		return GCControllerPlayerIndexUnset;
	};
}

- (void)setControllerInputHandler:(GCController *)controller {
	// Hook in the callback handler for the correct gamepad profile.
	// This is a bit of a weird design choice on Apples part.
	// You need to select the most capable gamepad profile for the
	// gamepad attached.
	if (controller.extendedGamepad != nil) {
		// The extended gamepad profile has all the input you could possibly find on
		// a gamepad but will only be active if your gamepad actually has all of
		// these...
		_weakify(self);
		_weakify(controller);

		controller.extendedGamepad.valueChangedHandler = ^(GCExtendedGamepad *gamepad, GCControllerElement *element) {
			_strongify(self);
			_strongify(controller);

			int joy_id = [self getJoyIdForController:controller];

			if (element == gamepad.buttonA) {
				OS_UIKit::get_singleton()->joy_button(joy_id, JOY_BUTTON_0,
						gamepad.buttonA.isPressed);
			} else if (element == gamepad.buttonB) {
				OS_UIKit::get_singleton()->joy_button(joy_id, JOY_BUTTON_1,
						gamepad.buttonB.isPressed);
			} else if (element == gamepad.buttonX) {
				OS_UIKit::get_singleton()->joy_button(joy_id, JOY_BUTTON_2,
						gamepad.buttonX.isPressed);
			} else if (element == gamepad.buttonY) {
				OS_UIKit::get_singleton()->joy_button(joy_id, JOY_BUTTON_3,
						gamepad.buttonY.isPressed);
			} else if (element == gamepad.leftShoulder) {
				OS_UIKit::get_singleton()->joy_button(joy_id, JOY_L,
						gamepad.leftShoulder.isPressed);
			} else if (element == gamepad.rightShoulder) {
				OS_UIKit::get_singleton()->joy_button(joy_id, JOY_R,
						gamepad.rightShoulder.isPressed);
			} else if (element == gamepad.leftTrigger) {
				OS_UIKit::get_singleton()->joy_button(joy_id, JOY_L2,
						gamepad.leftTrigger.isPressed);
			} else if (element == gamepad.rightTrigger) {
				OS_UIKit::get_singleton()->joy_button(joy_id, JOY_R2,
						gamepad.rightTrigger.isPressed);
			} else if (element == gamepad.dpad) {
				OS_UIKit::get_singleton()->joy_button(joy_id, JOY_DPAD_UP,
						gamepad.dpad.up.isPressed);
				OS_UIKit::get_singleton()->joy_button(joy_id, JOY_DPAD_DOWN,
						gamepad.dpad.down.isPressed);
				OS_UIKit::get_singleton()->joy_button(joy_id, JOY_DPAD_LEFT,
						gamepad.dpad.left.isPressed);
				OS_UIKit::get_singleton()->joy_button(joy_id, JOY_DPAD_RIGHT,
						gamepad.dpad.right.isPressed);
			}

			if (element == gamepad.leftThumbstick) {
				float value = gamepad.leftThumbstick.xAxis.value;
				OS_UIKit::get_singleton()->joy_axis(joy_id, JOY_ANALOG_LX, value);
				value = -gamepad.leftThumbstick.yAxis.value;
				OS_UIKit::get_singleton()->joy_axis(joy_id, JOY_ANALOG_LY, value);
			} else if (element == gamepad.rightThumbstick) {
				float value = gamepad.rightThumbstick.xAxis.value;
				OS_UIKit::get_singleton()->joy_axis(joy_id, JOY_ANALOG_RX, value);
				value = -gamepad.rightThumbstick.yAxis.value;
				OS_UIKit::get_singleton()->joy_axis(joy_id, JOY_ANALOG_RY, value);
			} else if (element == gamepad.leftTrigger) {
				float value = gamepad.leftTrigger.value;
				OS_UIKit::get_singleton()->joy_axis(joy_id, JOY_ANALOG_L2, value);
			} else if (element == gamepad.rightTrigger) {
				float value = gamepad.rightTrigger.value;
				OS_UIKit::get_singleton()->joy_axis(joy_id, JOY_ANALOG_R2, value);
			}

			if (@available(iOS 13.0, tvOS 13.0, *)) {
				if (element == gamepad.buttonMenu) {
					OS_UIKit::get_singleton()->joy_button(joy_id, JOY_START,
							gamepad.buttonMenu.isPressed);
				} else if (element == gamepad.buttonOptions) {
					OS_UIKit::get_singleton()->joy_button(joy_id, JOY_SELECT,
							gamepad.buttonOptions.isPressed);
				}
			}
		};
	} else if (controller.microGamepad != nil) {
		// micro gamepads were added in OS 9 and feature just 2 buttons and a d-pad
		_weakify(self);
		_weakify(controller);

		controller.microGamepad.valueChangedHandler = ^(GCMicroGamepad *, GCControllerElement *element) {
			_strongify(self);
			_strongify(controller);

			// Callback gamepad sometimes has different address then
			// the one used by `controller.microGamepad` instance
			// which results in gamepad loosing some button events.

			GCMicroGamepad *gamepad = controller.microGamepad;

			int joy_id = [self getJoyIdForController:controller];

			if (element == gamepad.buttonA) {
				OS_UIKit::get_singleton()->joy_button(joy_id, JOY_BUTTON_0,
						gamepad.buttonA.isPressed);
			} else if (element == gamepad.buttonX) {
				OS_UIKit::get_singleton()->joy_button(joy_id, JOY_BUTTON_2,
						gamepad.buttonX.isPressed);
			} else if (element == gamepad.dpad) {
				float value = gamepad.dpad.xAxis.value;
				OS_UIKit::get_singleton()->joy_axis(joy_id, JOY_AXIS_4, value);

				OS_UIKit::get_singleton()->joy_button(joy_id, JOY_DPAD_LEFT, gamepad.dpad.left.isPressed);
				OS_UIKit::get_singleton()->joy_button(joy_id, JOY_DPAD_RIGHT, gamepad.dpad.right.isPressed);

				value = -gamepad.dpad.yAxis.value;
				OS_UIKit::get_singleton()->joy_axis(joy_id, JOY_AXIS_5, value);

				OS_UIKit::get_singleton()->joy_button(joy_id, JOY_DPAD_UP, gamepad.dpad.up.isPressed);
				OS_UIKit::get_singleton()->joy_button(joy_id, JOY_DPAD_DOWN, gamepad.dpad.down.isPressed);
			}
		};
	}

	///@TODO need to add support for controller.motion which gives us access to
	/// the orientation of the device (if supported)

	///@TODO need to add support for controllerPausedHandler which should be a
	/// toggle
};

@end
