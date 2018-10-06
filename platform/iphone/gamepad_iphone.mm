#import "gamepad_iphone.h"
#import "app_delegate.h"
#import "core/project_settings.h"
#import "os_iphone.h"

void _ios_add_joystick(GCController *controller, AppDelegate *delegate) {
	[[GamepadManager sharedManager] addController:controller];
}

@interface GamepadManager ()

@property(nonatomic, strong) NSMutableDictionary *gamepadById;
@property(nonatomic, strong) NSMutableArray *gamepads;
@property(nonatomic, strong) NSMutableArray *pendingGamepads;

@end

@implementation GamepadManager

+ (instancetype)sharedManager {
	static dispatch_once_t onceToken;
	static GamepadManager *gGamepadManager = nil;
	dispatch_once(&onceToken, ^{
		gGamepadManager = [[GamepadManager alloc] init];
	});
	return gGamepadManager;
}

- (instancetype)init {
	self = [super init];
	if (self) {
		self.gamepadById = [NSMutableDictionary dictionary];
		self.gamepads = [NSMutableArray arrayWithCapacity:5];
		self.pendingGamepads = [NSMutableArray arrayWithCapacity:3];

		// this will be called immediately for all
		// connected controllers
		[[NSNotificationCenter defaultCenter]
				addObserver:self
				   selector:@selector(controllerWasConnected:)
					   name:GCControllerDidConnectNotification
					 object:nil];

		[[NSNotificationCenter defaultCenter]
				addObserver:self
				   selector:@selector(controllerWasDisconnected:)
					   name:GCControllerDidDisconnectNotification
					 object:nil];
	}
	return self;
}

- (void)dealloc {
	[[NSNotificationCenter defaultCenter]
			removeObserver:self
					  name:GCControllerDidConnectNotification
					object:nil];

	[[NSNotificationCenter defaultCenter]
			removeObserver:self
					  name:GCControllerDidDisconnectNotification
					object:nil];
	[super dealloc];
}

- (void)setReady:(BOOL)ready {
	_ready = ready;

	if (_ready) {
		for (GCController *controller in self.pendingGamepads) {
			[self addController:controller];
		}
		[self.pendingGamepads removeAllObjects];
	}
}

- (GCControllerPlayerIndex)getFreePlayerIndex {
	if (self.gamepads.count == 0) {
		return GCControllerPlayerIndex1;
	}
	return (GCControllerPlayerIndex)self.gamepads.count - 1;
}

- (int)getJoyIdForController:(GCController *)controller {
	NSNumber *key = [self.gamepadById allKeysForObject:controller].firstObject;
	if (key) {
		return key.intValue;
	}
	return -1;
}

- (void)controllerWasConnected:(NSNotification *)notification {
	GCController *controller = (GCController *)notification.object;
	if (self.isReady) {
		[self addController:controller];
	} else {
		[self.pendingGamepads addObject:controller];
	}
}

- (void)controllerWasDisconnected:(NSNotification *)notification {
	GCController *controller = (GCController *)notification.object;
	if ([self.gamepads containsObject:controller]) {
		// tell Godot this joystick is no longer there
		int joy_id = [self getJoyIdForController:controller];
		OSIPhone::get_singleton()->joy_connection_changed(joy_id, false, "");

		[self removeController:controller];
	}
}

- (void)addController:(GCController *)controller {
	// get a new id for our controller
	int joy_id = OSIPhone::get_singleton()->get_unused_joy_id();
	if (joy_id != -1) {
		// assign our player index
		if (controller.playerIndex == GCControllerPlayerIndexUnset) {
			controller.playerIndex = [self getFreePlayerIndex];
		}

		// tell Godot about our new controller
		OSIPhone::get_singleton()->joy_connection_changed(
				joy_id, true, [controller.vendorName UTF8String]);

		// add it to our dictionary, this will retain our controllers
		self.gamepadById[@(joy_id)] = controller;
		[self.gamepads addObject:controller];

		// set our input handler
		[self setControllerInputHandler:controller];
	} else {
		printf("Couldn't retrieve new joy id\n");
	}
}

- (void)removeController:(GCController *)controller {
	NSNumber *key = [self.gamepadById allKeysForObject:controller].firstObject;
	[self.gamepadById removeObjectForKey:key];

	[self.gamepads removeObject:controller];
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
		controller.extendedGamepad.valueChangedHandler = ^(
				GCExtendedGamepad *gamepad, GCControllerElement *element) {
			int joy_id = [self getJoyIdForController:controller];

			if (element == gamepad.buttonA) {
				OSIPhone::get_singleton()->joy_button(joy_id, JOY_BUTTON_0,
						gamepad.buttonA.isPressed);
			} else if (element == gamepad.buttonB) {
				OSIPhone::get_singleton()->joy_button(joy_id, JOY_BUTTON_1,
						gamepad.buttonB.isPressed);
			} else if (element == gamepad.buttonX) {
				OSIPhone::get_singleton()->joy_button(joy_id, JOY_BUTTON_2,
						gamepad.buttonX.isPressed);
			} else if (element == gamepad.buttonY) {
				OSIPhone::get_singleton()->joy_button(joy_id, JOY_BUTTON_3,
						gamepad.buttonY.isPressed);
			} else if (element == gamepad.leftShoulder) {
				OSIPhone::get_singleton()->joy_button(joy_id, JOY_L,
						gamepad.leftShoulder.isPressed);
			} else if (element == gamepad.rightShoulder) {
				OSIPhone::get_singleton()->joy_button(joy_id, JOY_R,
						gamepad.rightShoulder.isPressed);
			} else if (element == gamepad.leftTrigger) {
				OSIPhone::get_singleton()->joy_button(joy_id, JOY_L2,
						gamepad.leftTrigger.isPressed);
			} else if (element == gamepad.rightTrigger) {
				OSIPhone::get_singleton()->joy_button(joy_id, JOY_R2,
						gamepad.rightTrigger.isPressed);
			} else if (element == gamepad.dpad) {
				OSIPhone::get_singleton()->joy_button(joy_id, JOY_DPAD_UP,
						gamepad.dpad.up.isPressed);
				OSIPhone::get_singleton()->joy_button(joy_id, JOY_DPAD_DOWN,
						gamepad.dpad.down.isPressed);
				OSIPhone::get_singleton()->joy_button(joy_id, JOY_DPAD_LEFT,
						gamepad.dpad.left.isPressed);
				OSIPhone::get_singleton()->joy_button(joy_id, JOY_DPAD_RIGHT,
						gamepad.dpad.right.isPressed);
			}

			InputDefault::JoyAxis jx;
			jx.min = -1;
			if (element == gamepad.leftThumbstick) {
				jx.value = gamepad.leftThumbstick.xAxis.value;
				OSIPhone::get_singleton()->joy_axis(joy_id, JOY_ANALOG_LX, jx);
				jx.value = -gamepad.leftThumbstick.yAxis.value;
				OSIPhone::get_singleton()->joy_axis(joy_id, JOY_ANALOG_LY, jx);
			} else if (element == gamepad.rightThumbstick) {
				jx.value = gamepad.rightThumbstick.xAxis.value;
				OSIPhone::get_singleton()->joy_axis(joy_id, JOY_ANALOG_RX, jx);
				jx.value = -gamepad.rightThumbstick.yAxis.value;
				OSIPhone::get_singleton()->joy_axis(joy_id, JOY_ANALOG_RY, jx);
			} else if (element == gamepad.leftTrigger) {
				jx.value = gamepad.leftTrigger.value;
				OSIPhone::get_singleton()->joy_axis(joy_id, JOY_ANALOG_L2, jx);
			} else if (element == gamepad.rightTrigger) {
				jx.value = gamepad.rightTrigger.value;
				OSIPhone::get_singleton()->joy_axis(joy_id, JOY_ANALOG_R2, jx);
			}
		};
	} else if (controller.gamepad != nil) {
		// gamepad is the standard profile with 4 buttons, shoulder buttons and a
		// D-pad
		controller.gamepad.valueChangedHandler = ^(GCGamepad *gamepad,
				GCControllerElement *element) {
			int joy_id = [self getJoyIdForController:controller];

			if (element == gamepad.buttonA) {
				OSIPhone::get_singleton()->joy_button(joy_id, JOY_BUTTON_0,
						gamepad.buttonA.isPressed);
			} else if (element == gamepad.buttonB) {
				OSIPhone::get_singleton()->joy_button(joy_id, JOY_BUTTON_1,
						gamepad.buttonB.isPressed);
			} else if (element == gamepad.buttonX) {
				OSIPhone::get_singleton()->joy_button(joy_id, JOY_BUTTON_2,
						gamepad.buttonX.isPressed);
			} else if (element == gamepad.buttonY) {
				OSIPhone::get_singleton()->joy_button(joy_id, JOY_BUTTON_3,
						gamepad.buttonY.isPressed);
			} else if (element == gamepad.leftShoulder) {
				OSIPhone::get_singleton()->joy_button(joy_id, JOY_L,
						gamepad.leftShoulder.isPressed);
			} else if (element == gamepad.rightShoulder) {
				OSIPhone::get_singleton()->joy_button(joy_id, JOY_R,
						gamepad.rightShoulder.isPressed);
			} else if (element == gamepad.dpad) {
				OSIPhone::get_singleton()->joy_button(joy_id, JOY_DPAD_UP,
						gamepad.dpad.up.isPressed);
				OSIPhone::get_singleton()->joy_button(joy_id, JOY_DPAD_DOWN,
						gamepad.dpad.down.isPressed);
				OSIPhone::get_singleton()->joy_button(joy_id, JOY_DPAD_LEFT,
						gamepad.dpad.left.isPressed);
				OSIPhone::get_singleton()->joy_button(joy_id, JOY_DPAD_RIGHT,
						gamepad.dpad.right.isPressed);
			}
		};
#ifdef ADD_MICRO_GAMEPAD // disabling this for now, only available on iOS 9+,
		// while we are setting that as the minimum, seems our
		// build environment doesn't like it
	} else if (controller.microGamepad != nil) {
		// micro gamepads were added in OS 9 and feature just 2 buttons and a d-pad
		controller.microGamepad.valueChangedHandler =
				^(GCMicroGamepad *gamepad, GCControllerElement *element) {
					int joy_id = [self getJoyIdForController:controller];

					if (element == gamepad.buttonA) {
						OSIPhone::get_singleton()->joy_button(joy_id, JOY_BUTTON_0,
								gamepad.buttonA.isPressed);
					} else if (element == gamepad.buttonX) {
						OSIPhone::get_singleton()->joy_button(joy_id, JOY_BUTTON_2,
								gamepad.buttonX.isPressed);
					} else if (element == gamepad.dpad) {
						OSIPhone::get_singleton()->joy_button(joy_id, JOY_DPAD_UP,
								gamepad.dpad.up.isPressed);
						OSIPhone::get_singleton()->joy_button(joy_id, JOY_DPAD_DOWN,
								gamepad.dpad.down.isPressed);
						OSIPhone::get_singleton()->joy_button(joy_id, JOY_DPAD_LEFT,
								gamepad.dpad.left.isPressed);
						OSIPhone::get_singleton()->joy_button(joy_id, JOY_DPAD_RIGHT,
								gamepad.dpad.right.isPressed);
					}
				};
#endif
	}

	///@TODO need to add support for controller.motion which gives us access to
	/// the orientation of the device (if supported)

	///@TODO need to add support for controllerPausedHandler which should be a
	/// toggle
}

@end