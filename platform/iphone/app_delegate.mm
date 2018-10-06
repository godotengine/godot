/*************************************************************************/
/*  app_delegate.mm                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

#import "app_delegate.h"
#import "gamepad_iphone.h"
#import "view_controller.h"
#import <CoreMotion/CoreMotion.h>

#import "core/project_settings.h"
#import "drivers/coreaudio/audio_driver_coreaudio.h"
#import "main/main.h"
#import "os_iphone.h"

#import <AudioToolbox/AudioServices.h>

#define kFilteringFactor 0.1
#define kAccelerometerFrequency 100.0 // Hz

extern void _set_keep_screen_on(bool p_enabled);

void _vibrate() {
	AudioServicesPlaySystemSound(kSystemSoundID_Vibrate);
};

@implementation AppDelegate

extern int gargc;
extern char **gargv;
extern int iphone_main(int, int, int, char **, String);
extern void iphone_finish();

CMMotionManager *motionManager;
bool motionInitialised;

OS::VideoMode _get_video_mode() {
	int backingWidth;
	int backingHeight;
	glGetRenderbufferParameterivOES(GL_RENDERBUFFER_OES,
			GL_RENDERBUFFER_WIDTH_OES, &backingWidth);
	glGetRenderbufferParameterivOES(GL_RENDERBUFFER_OES,
			GL_RENDERBUFFER_HEIGHT_OES, &backingHeight);

	OS::VideoMode vm;
	vm.fullscreen = true;
	vm.width = backingWidth;
	vm.height = backingHeight;
	vm.resizable = false;
	return vm;
};

static int frame_count = 0;
- (void)drawView:(GLView *)view {
	switch (frame_count) {
		case 0: {
			OS::get_singleton()->set_video_mode(_get_video_mode());

			if (!OS::get_singleton()) {
				exit(0);
			};
			++frame_count;

			NSString *locale_code = [[NSLocale currentLocale] localeIdentifier];
			OSIPhone::get_singleton()->set_locale(
					String::utf8([locale_code UTF8String]));

			NSString *uuid = [[NSUserDefaults standardUserDefaults]
					objectForKey:@"identiferForVendor"];
			if (!uuid) {
				uuid = [[NSUUID UUID] UUIDString];

				[[NSUserDefaults standardUserDefaults]
						setObject:uuid
						   forKey:@"identifierForVendor"];
			}
			OSIPhone::get_singleton()->set_unique_id(String::utf8([uuid UTF8String]));

		}; break;

		case 1: {

			Main::setup2();
			++frame_count;

			[GamepadManager sharedManager].ready = YES;

			NSDictionary *info = [[NSBundle mainBundle] infoDictionary];
			for (NSString *key in info) {
				String ukey = String::utf8([key UTF8String]);
				ukey += "Info.plist/";

				id value = info[key];
				if ([value isKindOfClass:NSString.class]) {
					String uval = String::utf8([value UTF8String]);
					ProjectSettings::get_singleton()->set(ukey, uval);
				} else if ([value isKindOfClass:NSNumber.class]) {
					NSNumber *n = (NSNumber *)value;
					ProjectSettings::get_singleton()->set(ukey, [n doubleValue]);
				}
			}

		}; break;

		case 2: {

			Main::start();
			++frame_count;

		}; break; // no fallthrough

		default: {
			if (OSIPhone::get_singleton()) {
				// OSIPhone::get_singleton()->update_accelerometer(accel[0], accel[1],
				// accel[2]);
				if (motionInitialised) {
					// Just using polling approach for now, we can set this up so it sends
					// data to us in intervals, might be better. See Apple reference pages
					// for more details:
					// https://developer.apple.com/reference/coremotion/cmmotionmanager?language=objc

					// Apple splits our accelerometer date into a gravity and user movement
					// component. We add them back together
					CMAcceleration gravity = motionManager.deviceMotion.gravity;
					CMAcceleration acceleration =
							motionManager.deviceMotion.userAcceleration;

					///@TODO We don't seem to be getting data here, is my device broken or
					/// is this code incorrect?
					CMMagneticField magnetic =
							motionManager.deviceMotion.magneticField.field;

					///@TODO we can access rotationRate as a CMRotationRate variable
					///(processed date) or CMGyroData (raw data), have to see what works
					/// best
					CMRotationRate rotation = motionManager.deviceMotion.rotationRate;

					// Adjust for screen orientation.
					// [[UIDevice currentDevice] orientation] changes even if we've fixed
					// our orientation which is not a good thing when you're trying to get
					// your user to move the screen in all directions and want consistent
					// output

					///@TODO Using [[UIApplication sharedApplication] statusBarOrientation]
					/// is a bit of a hack. Godot obviously knows the orientation so maybe
					/// we
					// can use that instead? (note that left and right seem swapped)

					switch ([[UIApplication sharedApplication] statusBarOrientation]) {
						case UIDeviceOrientationLandscapeLeft: {
							OSIPhone::get_singleton()->update_gravity(-gravity.y, gravity.x,
									gravity.z);
							OSIPhone::get_singleton()->update_accelerometer(
									-(acceleration.y + gravity.y), (acceleration.x + gravity.x),
									acceleration.z + gravity.z);
							OSIPhone::get_singleton()->update_magnetometer(
									-magnetic.y, magnetic.x, magnetic.z);
							OSIPhone::get_singleton()->update_gyroscope(-rotation.y, rotation.x,
									rotation.z);
						}; break;
						case UIDeviceOrientationLandscapeRight: {
							OSIPhone::get_singleton()->update_gravity(gravity.y, -gravity.x,
									gravity.z);
							OSIPhone::get_singleton()->update_accelerometer(
									(acceleration.y + gravity.y), -(acceleration.x + gravity.x),
									acceleration.z + gravity.z);
							OSIPhone::get_singleton()->update_magnetometer(
									magnetic.y, -magnetic.x, magnetic.z);
							OSIPhone::get_singleton()->update_gyroscope(rotation.y, -rotation.x,
									rotation.z);
						}; break;
						case UIDeviceOrientationPortraitUpsideDown: {
							OSIPhone::get_singleton()->update_gravity(-gravity.x, gravity.y,
									gravity.z);
							OSIPhone::get_singleton()->update_accelerometer(
									-(acceleration.x + gravity.x), (acceleration.y + gravity.y),
									acceleration.z + gravity.z);
							OSIPhone::get_singleton()->update_magnetometer(
									-magnetic.x, magnetic.y, magnetic.z);
							OSIPhone::get_singleton()->update_gyroscope(-rotation.x, rotation.y,
									rotation.z);
						}; break;
						default: { // assume portrait
							OSIPhone::get_singleton()->update_gravity(gravity.x, gravity.y,
									gravity.z);
							OSIPhone::get_singleton()->update_accelerometer(
									acceleration.x + gravity.x, acceleration.y + gravity.y,
									acceleration.z + gravity.z);
							OSIPhone::get_singleton()->update_magnetometer(magnetic.x, magnetic.y,
									magnetic.z);
							OSIPhone::get_singleton()->update_gyroscope(rotation.x, rotation.y,
									rotation.z);
						}; break;
					};
				}

				OSIPhone::get_singleton()->iterate();
			};

		}; break;
	};
};

- (void)applicationDidReceiveMemoryWarning:(UIApplication *)application {
	if (OS::get_singleton()->get_main_loop()) {
		OS::get_singleton()->get_main_loop()->notification(
				MainLoop::NOTIFICATION_OS_MEMORY_WARNING);
	}
};

- (BOOL)application:(UIApplication *)application didFinishLaunchingWithOptions:(NSDictionary *)launchOptions {
	self.focused = NO;

	// Create a full-screen window
	CGRect rect = [[UIScreen mainScreen] bounds];
	self.window = [[UIWindow alloc] initWithFrame:rect];
	// Show the window
	[self.window makeKeyAndVisible];

	OS::VideoMode vm = _get_video_mode();
	NSArray *paths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory,
			NSUserDomainMask, YES);
	NSString *documents = paths.firstObject;
	int err = iphone_main(vm.width, vm.height, gargc, gargv, String::utf8([documents UTF8String]));
	if (err != 0) {
		// bail, things did not go very well for us, should probably output a message on screen with our error code...
		exit(0);
		return NO;
	};

	_set_keep_screen_on(bool(GLOBAL_DEF("display/window/energy_saving/keep_screen_on", true)) ? YES : NO);

	// Create our engine view controller
	GodotGameViewController *rootVC = [[GodotGameViewController alloc] init];
	self.window.rootViewController = rootVC;

	GLView *glView = rootVC.glView;
	glView.delegate = self;

	glView.useCADisplayLink =
			bool(GLOBAL_DEF("display.iOS/use_cadisplaylink", true)) ? YES : NO;
	printf("cadisaplylink: %d", glView.useCADisplayLink);

	[glView startAnimation];

	// Configure and start accelerometer
	if (!motionInitialised) {
		motionManager = [[CMMotionManager alloc] init];
		if (motionManager.deviceMotionAvailable) {
			motionManager.deviceMotionUpdateInterval = 1.0 / 70.0;
			[motionManager startDeviceMotionUpdatesUsingReferenceFrame:
								   CMAttitudeReferenceFrameXMagneticNorthZVertical];
			motionInitialised = YES;
		}
	}

	[[NSNotificationCenter defaultCenter]
			addObserver:self
			   selector:@selector(onAudioInterruption:)
				   name:AVAudioSessionInterruptionNotification
				 object:[AVAudioSession sharedInstance]];

	// prevent to stop music in another background app
	[[AVAudioSession sharedInstance] setCategory:AVAudioSessionCategoryAmbient error:nil];

	return YES;
};

- (void)onAudioInterruption:(NSNotification *)notification {
	NSLog(@"Audio interruption. Details: %@", notification.userInfo);

	if ([notification.userInfo[AVAudioSessionInterruptionTypeKey] isEqualToNumber:@(AVAudioSessionInterruptionTypeBegan)]) {
		[self toggleFocus:NO];
	} else if ([notification.userInfo[AVAudioSessionInterruptionTypeKey] isEqualToNumber:@(AVAudioSessionInterruptionTypeEnded)]) {
		[self toggleFocus:YES];
	}
}

// When application goes to background (e.g. user switches to another app or presses Home),
// then applicationWillResignActive -> applicationDidEnterBackground are called.
// When user opens the inactive app again,
// applicationWillEnterForeground -> applicationDidBecomeActive are called.

// There are cases when applicationWillResignActive -> applicationDidBecomeActive
// sequence is called without the app going to background. For example, that happens
// if you open the app list without switching to another app or open/close the
// notification panel by swiping from the upper part of the screen.

- (void)applicationWillResignActive:(UIApplication *)application {
	[self toggleFocus:NO];
}

- (void)applicationDidBecomeActive:(UIApplication *)application {
	[self toggleFocus:YES];
}

- (void)toggleFocus:(BOOL)isFocused {
	if (_focused != isFocused) {
		_focused = isFocused;

		// Handle notifications to main loop
		if (OS::get_singleton()->get_main_loop()) {
			int notification = self.isFocused ? MainLoop::NOTIFICATION_WM_FOCUS_IN : MainLoop::NOTIFICATION_WM_FOCUS_OUT;
			OS::get_singleton()->get_main_loop()->notification(notification);
		}

		// OpenGL Animation
		GodotGameViewController *viewController = (GodotGameViewController *)self.window.rootViewController;
		if (self.isFocused) {
			[viewController.glView startAnimation];
		} else {
			[viewController.glView stopAnimation];
		}

		// Native Video
		if (OS::get_singleton()->native_video_is_playing()) {
			if (self.isFocused) {
				OSIPhone::get_singleton()->native_video_unpause();
			} else {
				OSIPhone::get_singleton()->native_video_focus_out();
			}
		}

		// Native Audio
		AudioDriverCoreAudio *audio = dynamic_cast<AudioDriverCoreAudio *>(AudioDriverCoreAudio::get_singleton());
		if (audio) {
			self.isFocused ? audio->start() : audio->stop();
		}
	}
}

- (void)dealloc {
	[self.window release];
	[super dealloc];
}

@end
