/**************************************************************************/
/*  joypad_ios.mm                                                         */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#import "joypad_ios.h"

#import "godot_view.h"
#import "os_ios.h"

#include "core/config/project_settings.h"
#import "drivers/coreaudio/audio_driver_coreaudio.h"
#include "main/main.h"

@implementation RumbleMotor

- (instancetype)initWithController:(GCController *)controller locality:(GCHapticsLocality)locality {
	self = [super init];
	self.engine = [controller.haptics createEngineWithLocality:locality];
	self.player = nil;
	return self;
}

- (void)execute_pattern:(CHHapticPattern *)pattern {
	NSError *error;
	id<CHHapticPatternPlayer> player = [self.engine createPlayerWithPattern:pattern error:&error];

	// When all players have stopped for an engine, stop the engine.
	[self.engine notifyWhenPlayersFinished:^CHHapticEngineFinishedAction(NSError *_Nullable error) {
		return CHHapticEngineFinishedActionStopEngine;
	}];

	self.player = player;

	// Starts the engine and returns if an error was encountered.
	if (![self.engine startAndReturnError:&error]) {
		print_verbose("Couldn't start controller haptic engine");
		return;
	}
	if (![self.player startAtTime:0 error:&error]) {
		print_verbose("Couldn't execute controller haptic pattern");
	}
}

- (void)stop {
	NSError *error;
	[self.player stopAtTime:0 error:&error];
	self.player = nil;
}

@end

@implementation RumbleContext

- (instancetype)init {
	self = [super init];
	self.weak_motor = nil;
	self.strong_motor = nil;
	return self;
}

- (bool)hasMotors {
	return self.weak_motor != nil && self.strong_motor != nil;
}
- (bool)hasActivePlayers {
	if (![self hasMotors]) {
		return NO;
	}
	return self.weak_motor.player != nil && self.strong_motor.player != nil;
}

@end

@implementation JoypadData

- (instancetype)init {
	self = [super init];
	return self;
}
- (instancetype)init:(GCController *)controller {
	self = [super init];
	self.controller = controller;
	self.l_mode = Input::JOY_ADAPTIVE_TRIGGER_MODE_OFF;
	self.r_mode = Input::JOY_ADAPTIVE_TRIGGER_MODE_OFF;

	if (@available(iOS 14, *)) {
		// Haptics within the controller is only available in macOS 11+
		self.rumble_context = [[RumbleContext alloc] init];

		// Create Weak and Strong motors for controller.
		self.rumble_context.weak_motor = [[RumbleMotor alloc] initWithController:controller locality:GCHapticsLocalityRightHandle];
		self.rumble_context.strong_motor = [[RumbleMotor alloc] initWithController:controller locality:GCHapticsLocalityLeftHandle];

		// If the rumble motors aren't available, disable force feedback.
		if (![self.rumble_context hasMotors]) {
			self.force_feedback = NO;
		} else {
			self.force_feedback = YES;
		}
	} else {
		self.force_feedback = NO;
	}

	self.ff_effect_timestamp = 0;

	return self;
}

@end

JoypadIOS::JoypadIOS() {
	observer = [[JoypadIOSObserver alloc] init];
	[observer startObserving];
}

JoypadIOS::~JoypadIOS() {
	if (observer) {
		[observer finishObserving];
		observer = nil;
	}
}

void JoypadIOS::start_processing() {
	if (observer) {
		[observer startProcessing];
	}
	process_joypads();
}

API_AVAILABLE(ios(14))
CHHapticPattern *get_vibration_pattern(float p_magnitude, float p_duration) {
	// Creates a vibration pattern with an intensity and duration.
	NSDictionary *hapticDict = @{
		CHHapticPatternKeyPattern : @[
			@{
				CHHapticPatternKeyEvent : @{
					CHHapticPatternKeyEventType : CHHapticEventTypeHapticContinuous,
					CHHapticPatternKeyTime : @(CHHapticTimeImmediate),
					CHHapticPatternKeyEventDuration : [NSNumber numberWithFloat:p_duration],

					CHHapticPatternKeyEventParameters : @[
						@{
							CHHapticPatternKeyParameterID : CHHapticEventParameterIDHapticIntensity,
							CHHapticPatternKeyParameterValue : [NSNumber numberWithFloat:p_magnitude]
						},
					],
				},
			},
		],
	};
	NSError *error;
	CHHapticPattern *pattern = [[CHHapticPattern alloc] initWithDictionary:hapticDict error:&error];
	return pattern;
}

void JoypadIOS::joypad_vibration_start(JoypadData *p_joypad, float p_weak_magnitude, float p_strong_magnitude, float p_duration, uint64_t p_timestamp) {
	if (!p_joypad.force_feedback || p_weak_magnitude < 0.f || p_weak_magnitude > 1.f || p_strong_magnitude < 0.f || p_strong_magnitude > 1.f) {
		return;
	}

	// If there is active vibration players, stop them.
	if ([p_joypad.rumble_context hasActivePlayers]) {
		joypad_vibration_stop(p_joypad, p_timestamp);
	}

	// Gets the default vibration pattern and creates a player for each motor.
	CHHapticPattern *weak_pattern = get_vibration_pattern(p_weak_magnitude, p_duration);
	CHHapticPattern *strong_pattern = get_vibration_pattern(p_strong_magnitude, p_duration);

	RumbleMotor *weak_motor = p_joypad.rumble_context.weak_motor;
	RumbleMotor *strong_motor = p_joypad.rumble_context.strong_motor;

	[weak_motor execute_pattern:weak_pattern];
	[strong_motor execute_pattern:strong_pattern];

	p_joypad.ff_effect_timestamp = p_timestamp;
}

void JoypadIOS::joypad_vibration_stop(JoypadData *p_joypad, uint64_t p_timestamp) {
	if (!p_joypad.force_feedback) {
		return;
	}
	// If there is no active vibration players, exit.
	if (![p_joypad.rumble_context hasActivePlayers]) {
		return;
	}

	RumbleMotor *weak_motor = p_joypad.rumble_context.weak_motor;
	RumbleMotor *strong_motor = p_joypad.rumble_context.strong_motor;

	[weak_motor stop];
	[strong_motor stop];

	p_joypad.ff_effect_timestamp = p_timestamp;
}

@interface JoypadIOSObserver ()

@property(assign, nonatomic) BOOL isObserving;
@property(assign, nonatomic) BOOL isProcessing;
@property(strong, nonatomic) NSMutableDictionary<NSNumber *, JoypadData *> *connectedJoypads;
@property(strong, nonatomic) NSMutableArray<JoypadData *> *joypadsQueue;

@end

@implementation JoypadIOSObserver

- (instancetype)init {
	self = [super init];

	if (self) {
		[self godot_commonInit];
	}

	return self;
}

- (void)godot_commonInit {
	self.isObserving = NO;
	self.isProcessing = NO;
}

- (void)startProcessing {
	self.isProcessing = YES;

	for (GCController *controller in self.joypadsQueue) {
		[self addiOSJoypad:controller];
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

	// Get told when controllers connect, this will be called right away for
	// already connected controllers.
	[[NSNotificationCenter defaultCenter]
			addObserver:self
			   selector:@selector(controllerWasConnected:)
				   name:GCControllerDidConnectNotification
				 object:nil];

	// Get told when controllers disconnect.
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

- (NSArray<NSNumber *> *)getAllKeysForController:(GCController *)controller {
	NSArray *keys = [self.connectedJoypads allKeys];
	NSMutableArray *final_keys = [NSMutableArray array];

	for (NSNumber *key in keys) {
		JoypadData *joypad = [self.connectedJoypads objectForKey:key];
		if (joypad.controller == controller) {
			[final_keys addObject:key];
		}
	}

	return final_keys;
}

- (int)getJoyIdForController:(GCController *)controller {
	NSArray *keys = [self getAllKeysForController:controller];

	for (NSNumber *key in keys) {
		int joy_id = [key intValue];
		return joy_id;
	}

	return -1;
}

- (void)addiOSJoypad:(GCController *)controller {
	// Get a new id for our controller.
	int joy_id = Input::get_singleton()->get_unused_joy_id();

	if (joy_id == -1) {
		print_verbose("Couldn't retrieve new joy ID.");
		return;
	}

	// Assign our player index.
	if (controller.playerIndex == GCControllerPlayerIndexUnset) {
		controller.playerIndex = [self getFreePlayerIndex];
	}

	// Read current color and sensors state.
	if (@available(iOS 14, *)) {
		if (controller.motion != nil) {
			Input::get_singleton()->set_joy_sensors_enabled(joy_id, controller.motion.sensorsActive);
		}
	}

	JoypadData *joypad = [[JoypadData alloc] init:controller];
	if (@available(iOS 14, *)) {
		if (controller.light) {
			Color c = Color(controller.light.color.red, controller.light.color.green, controller.light.color.blue);
			joypad.color = c;
			Input::get_singleton()->set_joy_light(joy_id, c);
		}
	}
	// Tell Godot about our new controller.
	Input::get_singleton()->joy_connection_changed(joy_id, true, String::utf8([controller.vendorName UTF8String]));

	// Add it to our dictionary, this will retain our controllers.
	[self.connectedJoypads setObject:joypad forKey:[NSNumber numberWithInt:joy_id]];

	// Set our input handler.
	[self setControllerInputHandler:controller];
}

- (void)controllerWasConnected:(NSNotification *)notification {
	// Get our controller.
	GCController *controller = (GCController *)notification.object;

	if (!controller) {
		print_verbose("Couldn't retrieve new controller.");
		return;
	}

	if ([[self getAllKeysForController:controller] count] > 0) {
		print_verbose("Controller is already registered.");
	} else if (!self.isProcessing) {
		JoypadData *joypad = [[JoypadData alloc] init:controller];
		[self.joypadsQueue addObject:joypad];
	} else {
		[self addiOSJoypad:controller];
	}
}

- (void)controllerWasDisconnected:(NSNotification *)notification {
	// Find our joystick, there should be only one in our dictionary.
	GCController *controller = (GCController *)notification.object;

	if (!controller) {
		return;
	}

	NSArray *keys = [self getAllKeysForController:controller];
	for (NSNumber *key in keys) {
		// Tell Godot this joystick is no longer there.
		int joy_id = [key intValue];
		Input::get_singleton()->joy_connection_changed(joy_id, false, "");

		// And remove it from our dictionary.
		[self.connectedJoypads removeObjectForKey:key];
	}
}

- (GCControllerPlayerIndex)getFreePlayerIndex {
	bool have_player_1 = false;
	bool have_player_2 = false;
	bool have_player_3 = false;
	bool have_player_4 = false;

	if (self.connectedJoypads == nil) {
		NSArray *keys = [self.connectedJoypads allKeys];
		for (NSNumber *key in keys) {
			JoypadData *joypad = [self.connectedJoypads objectForKey:key];
			GCController *controller = joypad.controller;
			if (controller.playerIndex == GCControllerPlayerIndex1) {
				have_player_1 = true;
			} else if (controller.playerIndex == GCControllerPlayerIndex2) {
				have_player_2 = true;
			} else if (controller.playerIndex == GCControllerPlayerIndex3) {
				have_player_3 = true;
			} else if (controller.playerIndex == GCControllerPlayerIndex4) {
				have_player_4 = true;
			}
		}
	}

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
	}
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
				Input::get_singleton()->joy_button(joy_id, JoyButton::A,
						gamepad.buttonA.isPressed);
			} else if (element == gamepad.buttonB) {
				Input::get_singleton()->joy_button(joy_id, JoyButton::B,
						gamepad.buttonB.isPressed);
			} else if (element == gamepad.buttonX) {
				Input::get_singleton()->joy_button(joy_id, JoyButton::X,
						gamepad.buttonX.isPressed);
			} else if (element == gamepad.buttonY) {
				Input::get_singleton()->joy_button(joy_id, JoyButton::Y,
						gamepad.buttonY.isPressed);
			} else if (element == gamepad.leftShoulder) {
				Input::get_singleton()->joy_button(joy_id, JoyButton::LEFT_SHOULDER,
						gamepad.leftShoulder.isPressed);
			} else if (element == gamepad.rightShoulder) {
				Input::get_singleton()->joy_button(joy_id, JoyButton::RIGHT_SHOULDER,
						gamepad.rightShoulder.isPressed);
			} else if (element == gamepad.dpad) {
				Input::get_singleton()->joy_button(joy_id, JoyButton::DPAD_UP,
						gamepad.dpad.up.isPressed);
				Input::get_singleton()->joy_button(joy_id, JoyButton::DPAD_DOWN,
						gamepad.dpad.down.isPressed);
				Input::get_singleton()->joy_button(joy_id, JoyButton::DPAD_LEFT,
						gamepad.dpad.left.isPressed);
				Input::get_singleton()->joy_button(joy_id, JoyButton::DPAD_RIGHT,
						gamepad.dpad.right.isPressed);
			}

			if (element == gamepad.leftThumbstick) {
				float value = gamepad.leftThumbstick.xAxis.value;
				Input::get_singleton()->joy_axis(joy_id, JoyAxis::LEFT_X, value);
				value = -gamepad.leftThumbstick.yAxis.value;
				Input::get_singleton()->joy_axis(joy_id, JoyAxis::LEFT_Y, value);
			} else if (element == gamepad.rightThumbstick) {
				float value = gamepad.rightThumbstick.xAxis.value;
				Input::get_singleton()->joy_axis(joy_id, JoyAxis::RIGHT_X, value);
				value = -gamepad.rightThumbstick.yAxis.value;
				Input::get_singleton()->joy_axis(joy_id, JoyAxis::RIGHT_Y, value);
			} else if (element == gamepad.leftTrigger) {
				float value = gamepad.leftTrigger.value;
				Input::get_singleton()->joy_axis(joy_id, JoyAxis::TRIGGER_LEFT, value);
			} else if (element == gamepad.rightTrigger) {
				float value = gamepad.rightTrigger.value;
				Input::get_singleton()->joy_axis(joy_id, JoyAxis::TRIGGER_RIGHT, value);
			}

			if (@available(iOS 12.1, *)) {
				if (element == gamepad.leftThumbstickButton) {
					Input::get_singleton()->joy_button(joy_id, JoyButton::LEFT_STICK,
							gamepad.leftThumbstickButton.isPressed);
				} else if (element == gamepad.rightThumbstickButton) {
					Input::get_singleton()->joy_button(joy_id, JoyButton::RIGHT_STICK,
							gamepad.rightThumbstickButton.isPressed);
				}
			}

			if (@available(iOS 13, *)) {
				// iOS uses 'buttonOptions' and 'buttonMenu' names for BACK and START joy buttons.
				if (element == gamepad.buttonOptions) {
					Input::get_singleton()->joy_button(joy_id, JoyButton::BACK,
							gamepad.buttonOptions.isPressed);
				} else if (element == gamepad.buttonMenu) {
					Input::get_singleton()->joy_button(joy_id, JoyButton::START,
							gamepad.buttonMenu.isPressed);
				}
			}

			if (@available(iOS 14, *)) {
				// iOS uses 'buttonHome' for the GUIDE joy button.
				if (element == gamepad.buttonHome) {
					Input::get_singleton()->joy_button(joy_id, JoyButton::GUIDE,
							gamepad.buttonHome.isPressed);
				}
				if ([gamepad isKindOfClass:[GCXboxGamepad class]]) {
					GCXboxGamepad *xboxGamepad = (GCXboxGamepad *)gamepad;
					if (element == xboxGamepad.paddleButton1) {
						Input::get_singleton()->joy_button(joy_id, JoyButton::PADDLE1,
								xboxGamepad.paddleButton1.isPressed);
					} else if (element == xboxGamepad.paddleButton2) {
						Input::get_singleton()->joy_button(joy_id, JoyButton::PADDLE2,
								xboxGamepad.paddleButton2.isPressed);
					} else if (element == xboxGamepad.paddleButton3) {
						Input::get_singleton()->joy_button(joy_id, JoyButton::PADDLE3,
								xboxGamepad.paddleButton3.isPressed);
					} else if (element == xboxGamepad.paddleButton4) {
						Input::get_singleton()->joy_button(joy_id, JoyButton::PADDLE4,
								xboxGamepad.paddleButton4.isPressed);
					}
				}
				if ([gamepad isKindOfClass:[GCDualShockGamepad class]]) {
					GCDualShockGamepad *dsGamepad = (GCDualShockGamepad *)gamepad;
					if (element == dsGamepad.touchpadButton) {
						Input::get_singleton()->joy_button(joy_id, JoyButton::TOUCHPAD,
								dsGamepad.touchpadButton.isPressed);
					}
					if (element == dsGamepad.touchpadPrimary) {
						float value = dsGamepad.touchpadPrimary.xAxis.value;
						Input::get_singleton()->joy_axis(joy_id, JoyAxis::PRIMARY_FINGER_X, value);
						value = -dsGamepad.touchpadPrimary.yAxis.value;
						Input::get_singleton()->joy_axis(joy_id, JoyAxis::PRIMARY_FINGER_Y, value);
					} else if (element == dsGamepad.touchpadSecondary) {
						float value = dsGamepad.touchpadSecondary.xAxis.value;
						Input::get_singleton()->joy_axis(joy_id, JoyAxis::SECONDARY_FINGER_X, value);
						value = -dsGamepad.touchpadSecondary.yAxis.value;
						Input::get_singleton()->joy_axis(joy_id, JoyAxis::SECONDARY_FINGER_Y, value);
					}
				}
			}

			if (@available(iOS 14.5, *)) {
				if ([gamepad isKindOfClass:[GCDualSenseGamepad class]]) {
					GCDualSenseGamepad *dsGamepad = (GCDualSenseGamepad *)gamepad;
					if (element == dsGamepad.touchpadButton) {
						Input::get_singleton()->joy_button(joy_id, JoyButton::TOUCHPAD,
								dsGamepad.touchpadButton.isPressed);
					}
					if (element == dsGamepad.touchpadPrimary) {
						float value = dsGamepad.touchpadPrimary.xAxis.value;
						Input::get_singleton()->joy_axis(joy_id, JoyAxis::PRIMARY_FINGER_X, value);
						value = -dsGamepad.touchpadPrimary.yAxis.value;
						Input::get_singleton()->joy_axis(joy_id, JoyAxis::PRIMARY_FINGER_Y, value);
					} else if (element == dsGamepad.touchpadSecondary) {
						float value = dsGamepad.touchpadSecondary.xAxis.value;
						Input::get_singleton()->joy_axis(joy_id, JoyAxis::SECONDARY_FINGER_X, value);
						value = -dsGamepad.touchpadSecondary.yAxis.value;
						Input::get_singleton()->joy_axis(joy_id, JoyAxis::SECONDARY_FINGER_Y, value);
					}
				}
			}

			if (@available(iOS 15, *)) {
				if ([gamepad isKindOfClass:[GCXboxGamepad class]]) {
					GCXboxGamepad *xboxGamepad = (GCXboxGamepad *)gamepad;
					if (element == xboxGamepad.buttonShare) {
						Input::get_singleton()->joy_button(joy_id, JoyButton::MISC1,
								xboxGamepad.buttonShare.isPressed);
					}
				}
			}
		};
	} else if (controller.microGamepad != nil) {
		// Micro gamepads were added in macOS 10.11 and feature just 2 buttons and a d-pad.
		_weakify(self);
		_weakify(controller);

		controller.microGamepad.valueChangedHandler = ^(GCMicroGamepad *gamepad, GCControllerElement *element) {
			_strongify(self);
			_strongify(controller);

			int joy_id = [self getJoyIdForController:controller];

			if (element == gamepad.buttonA) {
				Input::get_singleton()->joy_button(joy_id, JoyButton::A,
						gamepad.buttonA.isPressed);
			} else if (element == gamepad.buttonX) {
				Input::get_singleton()->joy_button(joy_id, JoyButton::X,
						gamepad.buttonX.isPressed);
			} else if (element == gamepad.dpad) {
				Input::get_singleton()->joy_button(joy_id, JoyButton::DPAD_UP,
						gamepad.dpad.up.isPressed);
				Input::get_singleton()->joy_button(joy_id, JoyButton::DPAD_DOWN,
						gamepad.dpad.down.isPressed);
				Input::get_singleton()->joy_button(joy_id, JoyButton::DPAD_LEFT, gamepad.dpad.left.isPressed);
				Input::get_singleton()->joy_button(joy_id, JoyButton::DPAD_RIGHT, gamepad.dpad.right.isPressed);
			}
		};
	}

	// The orientation of the device (if supported).
	if (@available(iOS 14, *)) {
		if (controller.motion != nil) {
			_weakify(self);
			_weakify(controller);

			controller.motion.valueChangedHandler = ^(GCMotion *motion) {
				_strongify(self);
				_strongify(controller);
				int joy_id = [self getJoyIdForController:controller];

				if (motion.hasGravityAndUserAcceleration) {
					Input::get_singleton()->set_joy_gravity(joy_id, Vector3(motion.gravity.x, motion.gravity.y, motion.gravity.z));
				}
				Input::get_singleton()->set_joy_accelerometer(joy_id, Vector3(motion.acceleration.x, motion.acceleration.y, motion.acceleration.z));
				if (motion.hasRotationRate) {
					Input::get_singleton()->set_joy_gyroscope(joy_id, Vector3(motion.rotationRate.x, motion.rotationRate.y, motion.rotationRate.z));
				}
				Input::get_singleton()->set_joy_sensors_enabled(joy_id, motion.sensorsActive);
			};
		}
	}
}

@end

void JoypadIOS::process_joypads() {
	if (@available(iOS 14, *)) {
		NSArray *keys = [observer.connectedJoypads allKeys];

		for (NSNumber *key in keys) {
			int id = key.intValue;
			Input *input = Input::get_singleton();
			JoypadData *joypad = [observer.connectedJoypads objectForKey:key];

			if (joypad.controller.battery != nil) {
				switch (joypad.controller.battery.batteryState) {
					case GCDeviceBatteryStateDischarging: {
						input->set_joy_battery_state(id, Input::JOY_BATTERY_STATE_DISCHARGING);
					} break;
					case GCDeviceBatteryStateCharging: {
						input->set_joy_battery_state(id, Input::JOY_BATTERY_STATE_CHARGING);
					} break;
					case GCDeviceBatteryStateFull: {
						input->set_joy_battery_state(id, Input::JOY_BATTERY_STATE_FULL);
					} break;
					default: {
						input->set_joy_battery_state(id, Input::JOY_BATTERY_STATE_UNKNOWN);
					} break;
				}
				input->set_joy_battery_level(id, joypad.controller.battery.batteryLevel);
			}
			if (joypad.controller.light != nil) {
				Color color = input->get_joy_light(id);
				if (joypad.color != color) {
					joypad.controller.light.color = [[GCColor alloc] initWithRed:color.r green:color.g blue:color.b];
				}
			}

			if (@available(iOS 14.5, *)) {
				if (joypad.controller.extendedGamepad != nil && [joypad.controller.extendedGamepad isKindOfClass:[GCDualSenseGamepad class]]) {
					GCDualSenseGamepad *dsGamepad = (GCDualSenseGamepad *)joypad.controller.extendedGamepad;

					Input::JoyAdaptiveTriggerMode l_mode = input->get_joy_adaptive_trigger_mode(id, JoyAxis::TRIGGER_LEFT);
					Vector2 l_strength = input->get_joy_adaptive_trigger_strength(id, JoyAxis::TRIGGER_LEFT);
					Vector2 l_position = input->get_joy_adaptive_trigger_position(id, JoyAxis::TRIGGER_LEFT);
					if (l_mode != joypad.l_mode || l_strength != joypad.l_strength || l_position != joypad.l_position) {
						switch (l_mode) {
							case Input::JOY_ADAPTIVE_TRIGGER_MODE_OFF: {
								[dsGamepad.leftTrigger setModeOff];
							} break;
							case Input::JOY_ADAPTIVE_TRIGGER_MODE_FEEDBACK: {
								[dsGamepad.leftTrigger setModeFeedbackWithStartPosition:l_position.x resistiveStrength:l_strength.x];
							} break;
							case Input::JOY_ADAPTIVE_TRIGGER_MODE_WEAPON: {
								[dsGamepad.leftTrigger setModeWeaponWithStartPosition:l_position.x endPosition:l_position.y resistiveStrength:l_strength.x];
							} break;
							case Input::JOY_ADAPTIVE_TRIGGER_MODE_VIBRATION: {
								[dsGamepad.leftTrigger setModeVibrationWithStartPosition:l_position.x amplitude:l_strength.x frequency:l_strength.y];
							} break;
							case Input::JOY_ADAPTIVE_TRIGGER_MODE_SLOPE_FEEDBACK: {
								if (@available(iOS 15.4, *)) {
									[dsGamepad.leftTrigger setModeSlopeFeedbackWithStartPosition:l_position.x endPosition:l_position.y startStrength:l_strength.x endStrength:l_strength.y];
								}
							} break;
							default:
								break;
						}
						joypad.l_mode = l_mode;
						joypad.l_strength = l_strength;
						joypad.l_position = l_position;
					}
					Input::JoyAdaptiveTriggerMode r_mode = input->get_joy_adaptive_trigger_mode(id, JoyAxis::TRIGGER_RIGHT);
					Vector2 r_strength = input->get_joy_adaptive_trigger_strength(id, JoyAxis::TRIGGER_RIGHT);
					Vector2 r_position = input->get_joy_adaptive_trigger_position(id, JoyAxis::TRIGGER_RIGHT);
					if (r_mode != joypad.r_mode || r_strength != joypad.r_strength || r_position != joypad.r_position) {
						switch (r_mode) {
							case Input::JOY_ADAPTIVE_TRIGGER_MODE_OFF: {
								[dsGamepad.rightTrigger setModeOff];
							} break;
							case Input::JOY_ADAPTIVE_TRIGGER_MODE_FEEDBACK: {
								[dsGamepad.rightTrigger setModeFeedbackWithStartPosition:r_position.x resistiveStrength:r_strength.x];
							} break;
							case Input::JOY_ADAPTIVE_TRIGGER_MODE_WEAPON: {
								[dsGamepad.rightTrigger setModeWeaponWithStartPosition:r_position.x endPosition:r_position.y resistiveStrength:r_strength.x];
							} break;
							case Input::JOY_ADAPTIVE_TRIGGER_MODE_VIBRATION: {
								[dsGamepad.rightTrigger setModeVibrationWithStartPosition:r_position.x amplitude:r_strength.x frequency:r_strength.y];
							} break;
							case Input::JOY_ADAPTIVE_TRIGGER_MODE_SLOPE_FEEDBACK: {
								if (@available(iOS 15.4, *)) {
									[dsGamepad.rightTrigger setModeSlopeFeedbackWithStartPosition:r_position.x endPosition:r_position.y startStrength:r_strength.x endStrength:r_strength.y];
								}
							} break;
							default:
								break;
						}
						joypad.r_mode = r_mode;
						joypad.r_strength = r_strength;
						joypad.r_position = r_position;
					}
				}
			}

			if (joypad.controller != nil && joypad.controller.motion != nil) {
				bool sensors_enabled = input->get_joy_sensors_enabled(id);
				if (joypad.controller.motion.sensorsActive != sensors_enabled) {
					joypad.controller.motion.sensorsActive = sensors_enabled;
				}
			}

			if (joypad.force_feedback) {
				uint64_t timestamp = input->get_joy_vibration_timestamp(id);

				if (timestamp > (unsigned)joypad.ff_effect_timestamp) {
					Vector2 strength = input->get_joy_vibration_strength(id);
					float duration = input->get_joy_vibration_duration(id);
					if (duration == 0) {
						duration = GCHapticDurationInfinite;
					}

					if (strength.x == 0 && strength.y == 0) {
						joypad_vibration_stop(joypad, timestamp);
					} else {
						joypad_vibration_start(joypad, strength.x, strength.y, duration, timestamp);
					}
				}
			}
		}
	}
}
