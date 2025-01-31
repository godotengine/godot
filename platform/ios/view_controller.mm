/**************************************************************************/
/*  view_controller.mm                                                    */
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

#import "view_controller.h"

#import "display_server_ios.h"
#import "godot_view.h"
#import "godot_view_renderer.h"
#import "key_mapping_ios.h"
#import "keyboard_input_view.h"
#import "os_ios.h"

#include "core/config/project_settings.h"

#import <AVFoundation/AVFoundation.h>
#import <GameController/GameController.h>

@interface ViewController () <GodotViewDelegate>

@property(strong, nonatomic) GodotViewRenderer *renderer;
@property(strong, nonatomic) GodotKeyboardInputView *keyboardView;

@property(strong, nonatomic) UIView *godotLoadingOverlay;

@property(strong, nonatomic) NSMutableArray *connectedMice;

@property(nonatomic) Vector2 last_position;

@end

@implementation ViewController

- (GodotView *)godotView {
	return (GodotView *)self.view;
}

- (void)pressesBegan:(NSSet<UIPress *> *)presses withEvent:(UIPressesEvent *)event {
	[super pressesBegan:presses withEvent:event];

	if (!DisplayServerIOS::get_singleton() || DisplayServerIOS::get_singleton()->is_keyboard_active()) {
		return;
	}
	if (@available(iOS 13.4, *)) {
		for (UIPress *press in presses) {
			String u32lbl = String::utf8([press.key.charactersIgnoringModifiers UTF8String]);
			String u32text = String::utf8([press.key.characters UTF8String]);
			Key key = KeyMappingIOS::remap_key(press.key.keyCode);

			if (press.key.keyCode == 0 && u32text.is_empty() && u32lbl.is_empty()) {
				continue;
			}

			char32_t us = 0;
			if (!u32lbl.is_empty() && !u32lbl.begins_with("UIKey")) {
				us = u32lbl[0];
			}

			KeyLocation location = KeyMappingIOS::key_location(press.key.keyCode);

			if (!u32text.is_empty() && !u32text.begins_with("UIKey")) {
				for (int i = 0; i < u32text.length(); i++) {
					const char32_t c = u32text[i];
					DisplayServerIOS::get_singleton()->key(fix_keycode(us, key), c, fix_key_label(us, key), key, press.key.modifierFlags, true, location);
				}
			} else {
				DisplayServerIOS::get_singleton()->key(fix_keycode(us, key), 0, fix_key_label(us, key), key, press.key.modifierFlags, true, location);
			}
		}
	}
}

- (void)pressesEnded:(NSSet<UIPress *> *)presses withEvent:(UIPressesEvent *)event {
	[super pressesEnded:presses withEvent:event];

	if (!DisplayServerIOS::get_singleton() || DisplayServerIOS::get_singleton()->is_keyboard_active()) {
		return;
	}
	if (@available(iOS 13.4, *)) {
		for (UIPress *press in presses) {
			String u32lbl = String::utf8([press.key.charactersIgnoringModifiers UTF8String]);
			Key key = KeyMappingIOS::remap_key(press.key.keyCode);

			if (press.key.keyCode == 0 && u32lbl.is_empty()) {
				continue;
			}

			char32_t us = 0;
			if (!u32lbl.is_empty() && !u32lbl.begins_with("UIKey")) {
				us = u32lbl[0];
			}

			KeyLocation location = KeyMappingIOS::key_location(press.key.keyCode);

			DisplayServerIOS::get_singleton()->key(fix_keycode(us, key), 0, fix_key_label(us, key), key, press.key.modifierFlags, false, location);
		}
	}
}

- (void)loadView {
	GodotView *view = [[GodotView alloc] init];
	GodotViewRenderer *renderer = [[GodotViewRenderer alloc] init];

	self.renderer = renderer;
	self.view = view;

	view.renderer = self.renderer;
	view.delegate = self;
}

- (instancetype)initWithNibName:(NSString *)nibNameOrNil bundle:(NSBundle *)nibBundleOrNil {
	self = [super initWithNibName:nibNameOrNil bundle:nibBundleOrNil];

	if (self) {
		[self godot_commonInit];
	}

	return self;
}

- (instancetype)initWithCoder:(NSCoder *)coder {
	self = [super initWithCoder:coder];

	if (self) {
		[self godot_commonInit];
	}

	return self;
}

- (void)godot_commonInit {
	// Initialize view controller values.

	self.hasMouse = false;
	self.connectedMice = [NSMutableArray array];
	if (@available(iOS 14, *)) {
		[[NSNotificationCenter defaultCenter]
				addObserver:self
				   selector:@selector(mouseWasConnected:)
					   name:GCMouseDidConnectNotification
					 object:nil];

		[[NSNotificationCenter defaultCenter]
				addObserver:self
				   selector:@selector(mouseWasDisconnected:)
					   name:GCMouseDidDisconnectNotification
					 object:nil];
	}
}

- (void)mouseWasConnected:(NSNotification *)notification {
	if (@available(iOS 14, *)) {
		GCMouse *mouse = (GCMouse *)notification.object;

		if (mouse) {
			print_verbose(vformat("Mouse connected: %s", String::utf8([mouse.vendorName UTF8String])));
			[self.connectedMice addObject:mouse];
			[self setMouseInputHandler:mouse];
		}
		self.hasMouse = (self.connectedMice.count != 0);
	}
}

- (void)mouseWasDisconnected:(NSNotification *)notification {
	if (@available(iOS 14, *)) {
		GCMouse *mouse = (GCMouse *)notification.object;

		if (mouse) {
			print_verbose(vformat("Mouse disconnected: %s", String::utf8([mouse.vendorName UTF8String])));
			[self.connectedMice removeObject:mouse];
		}
		self.hasMouse = (self.connectedMice.count != 0);
	}
}

- (BitField<MouseButtonMask>)mouseGetButtonState {
	BitField<MouseButtonMask> last_button_state = 0;
	if (@available(iOS 14, *)) {
		const GCMouseInput *mouse_input = [[GCMouse current] mouseInput];
		if (mouse_input != Nil) {
			if (mouse_input.leftButton.pressed) {
				last_button_state.set_flag(MouseButtonMask::LEFT);
			}
			if (mouse_input.rightButton.pressed) {
				last_button_state.set_flag(MouseButtonMask::RIGHT);
			}
			if (mouse_input.middleButton.pressed) {
				last_button_state.set_flag(MouseButtonMask::MIDDLE);
			}
			for (NSUInteger i = 0; i < [mouse_input.auxiliaryButtons count]; i++) {
				GCControllerButtonInput *button_element = [mouse_input.auxiliaryButtons objectAtIndex:i];
				if (button_element.pressed) {
					last_button_state.set_flag((MouseButtonMask)((NSUInteger)MouseButtonMask::MB_XBUTTON1 << i));
				}
			}
		}
	}
	return last_button_state;
}

- (void)sendMouseButton:(MouseButton)button pressed:(bool)pressed pos:(Vector2)pos {
	Ref<InputEventMouseButton> mb;
	mb.instantiate();
	mb->set_button_index(button);
	mb->set_pressed(pressed);
	mb->set_button_mask([self mouseGetButtonState]);
	mb->set_position(pos);
	mb->set_global_position(pos);
	Input::get_singleton()->parse_input_event(mb);
}

- (void)sendScroll:(Vector2)factor pos:(Vector2)pos {
	if (factor.x != 0) { // Note: X is primary wheel, not horizontal.
		Ref<InputEventMouseButton> sc;
		sc.instantiate();
		sc->set_button_index(factor.x < 0 ? MouseButton::WHEEL_DOWN : MouseButton::WHEEL_UP);
		sc->set_factor(Math::abs(factor.x));
		sc->set_pressed(true);
		BitField<MouseButtonMask> scroll_mask = [self mouseGetButtonState];
		scroll_mask.set_flag(mouse_button_to_mask(sc->get_button_index()));
		sc->set_button_mask(scroll_mask);
		sc->set_position(pos);
		sc->set_global_position(pos);
		Input::get_singleton()->parse_input_event(sc);
		sc = sc->duplicate();
		sc->set_pressed(false);
		scroll_mask.clear_flag(mouse_button_to_mask(sc->get_button_index()));
		sc->set_button_mask(scroll_mask);
		Input::get_singleton()->parse_input_event(sc);
	}
	if (factor.y != 0) { // Note: Y is secondary wheel, not vertical.
		Ref<InputEventMouseButton> sc;
		sc.instantiate();
		sc->set_button_index(factor.y < 0 ? MouseButton::WHEEL_LEFT : MouseButton::WHEEL_RIGHT);
		sc->set_factor(Math::abs(factor.y));
		sc->set_pressed(true);
		BitField<MouseButtonMask> scroll_mask = [self mouseGetButtonState];
		scroll_mask.set_flag(mouse_button_to_mask(sc->get_button_index()));
		sc->set_button_mask(scroll_mask);
		sc->set_position(pos);
		sc->set_global_position(pos);
		Input::get_singleton()->parse_input_event(sc);
		sc = sc->duplicate();
		sc->set_pressed(false);
		scroll_mask.clear_flag(mouse_button_to_mask(sc->get_button_index()));
		sc->set_button_mask(scroll_mask);
		Input::get_singleton()->parse_input_event(sc);
	}
}

- (void)sendMove:(Vector2)pos delta:(Vector2)delta {
	Ref<InputEventMouseMotion> mm;
	mm.instantiate();

	mm->set_button_mask([self mouseGetButtonState]);
	mm->set_position(pos);
	mm->set_global_position(pos);
	mm->set_velocity(Input::get_singleton()->get_last_mouse_velocity());
	mm->set_screen_velocity(mm->get_velocity());
	mm->set_relative(delta);
	mm->set_relative_screen_position(mm->get_relative());
	Input::get_singleton()->parse_input_event(mm);
}

- (void)setMouseInputHandler:(GCMouse *)mouse API_AVAILABLE(ios(14.0)) {
	if (mouse.mouseInput != nil) {
		mouse.mouseInput.mouseMovedHandler = ^(GCMouseInput *mouse_input, float deltaX, float deltaY) {
			if (DisplayServerIOS::get_singleton()) {
				if (DisplayServerIOS::get_singleton()->mouse_get_mode() == DisplayServer::MouseMode::MOUSE_MODE_CAPTURED) {
					[self sendMove:DisplayServerIOS::get_singleton()->window_get_size() / 2 delta:Vector2(deltaX, deltaY)];
				} else {
					[self sendMove:self.last_position delta:Vector2(deltaX, deltaY)];
				}
			}
		};
		mouse.mouseInput.leftButton.pressedChangedHandler = ^(GCControllerButtonInput *_Nonnull button, float value, BOOL pressed) {
			if (DisplayServerIOS::get_singleton()) {
				if (DisplayServerIOS::get_singleton()->mouse_get_mode() == DisplayServer::MouseMode::MOUSE_MODE_CAPTURED) {
					[self sendMouseButton:MouseButton::LEFT pressed:button.isPressed pos:DisplayServerIOS::get_singleton()->window_get_size() / 2];
				} else {
					[self sendMouseButton:MouseButton::LEFT pressed:button.isPressed pos:self.last_position];
				}
			}
		};
		mouse.mouseInput.rightButton.pressedChangedHandler = ^(GCControllerButtonInput *_Nonnull button, float value, BOOL pressed) {
			if (DisplayServerIOS::get_singleton()) {
				if (DisplayServerIOS::get_singleton()->mouse_get_mode() == DisplayServer::MouseMode::MOUSE_MODE_CAPTURED) {
					[self sendMouseButton:MouseButton::RIGHT pressed:button.isPressed pos:DisplayServerIOS::get_singleton()->window_get_size() / 2];
				} else {
					[self sendMouseButton:MouseButton::RIGHT pressed:button.isPressed pos:self.last_position];
				}
			}
		};
		mouse.mouseInput.middleButton.pressedChangedHandler = ^(GCControllerButtonInput *_Nonnull button, float value, BOOL pressed) {
			if (DisplayServerIOS::get_singleton()) {
				if (DisplayServerIOS::get_singleton()->mouse_get_mode() == DisplayServer::MouseMode::MOUSE_MODE_CAPTURED) {
					[self sendMouseButton:MouseButton::MIDDLE pressed:button.isPressed pos:DisplayServerIOS::get_singleton()->window_get_size() / 2];
				} else {
					[self sendMouseButton:MouseButton::MIDDLE pressed:button.isPressed pos:self.last_position];
				}
			}
		};
		for (NSUInteger i = 0; i < [mouse.mouseInput.auxiliaryButtons count]; i++) {
			GCControllerButtonInput *button_element = [mouse.mouseInput.auxiliaryButtons objectAtIndex:i];
			button_element.pressedChangedHandler = ^(GCControllerButtonInput *_Nonnull button, float value, BOOL pressed) {
				if (DisplayServerIOS::get_singleton()) {
					if (DisplayServerIOS::get_singleton()->mouse_get_mode() == DisplayServer::MouseMode::MOUSE_MODE_CAPTURED) {
						[self sendMouseButton:(MouseButton)((NSUInteger)MouseButton::MB_XBUTTON1 + i) pressed:button.isPressed pos:DisplayServerIOS::get_singleton()->window_get_size() / 2];
					} else {
						[self sendMouseButton:(MouseButton)((NSUInteger)MouseButton::MB_XBUTTON1 + i) pressed:button.isPressed pos:self.last_position];
					}
				}
			};
		}
		mouse.mouseInput.scroll.valueChangedHandler = ^(GCControllerDirectionPad *dpad, float xValue, float yValue) {
			if (DisplayServerIOS::get_singleton()) {
				if (DisplayServerIOS::get_singleton()->mouse_get_mode() == DisplayServer::MouseMode::MOUSE_MODE_CAPTURED) {
					[self sendScroll:Vector2(xValue, yValue) pos:DisplayServerIOS::get_singleton()->window_get_size() / 2];
				} else {
					[self sendScroll:Vector2(xValue, yValue) pos:self.last_position];
				}
			}
		};
	}
}

- (void)didReceiveMemoryWarning {
	[super didReceiveMemoryWarning];
	print_verbose("Did receive memory warning!");
}

- (void)viewDidLoad {
	[super viewDidLoad];

	[self observeKeyboard];
	[self displayLoadingOverlay];

	[self setNeedsUpdateOfScreenEdgesDeferringSystemGestures];

	if (@available(iOS 14, *)) {
		UIHoverGestureRecognizer *hoverRecognizer = [[UIHoverGestureRecognizer alloc] initWithTarget:self action:@selector(mouseHover:)];
		[self.view addGestureRecognizer:hoverRecognizer];
	}
}

- (void)mouseHover:(UIHoverGestureRecognizer *)gestureRecognizer API_AVAILABLE(ios(14.0)) {
	if (gestureRecognizer.state == UIGestureRecognizerStateBegan || gestureRecognizer.state == UIGestureRecognizerStateChanged) {
		CGPoint p = [gestureRecognizer locationInView:gestureRecognizer.view];
		self.last_position = Vector2(p.x, p.y) * self.view.contentScaleFactor;
	}
}

- (void)observeKeyboard {
	print_verbose("Setting up keyboard input view.");
	self.keyboardView = [GodotKeyboardInputView new];
	[self.view addSubview:self.keyboardView];

	print_verbose("Adding observer for keyboard show/hide.");
	[[NSNotificationCenter defaultCenter]
			addObserver:self
			   selector:@selector(keyboardOnScreen:)
				   name:UIKeyboardDidShowNotification
				 object:nil];
	[[NSNotificationCenter defaultCenter]
			addObserver:self
			   selector:@selector(keyboardHidden:)
				   name:UIKeyboardDidHideNotification
				 object:nil];
}

- (void)displayLoadingOverlay {
	NSBundle *bundle = [NSBundle mainBundle];
	NSString *storyboardName = @"Launch Screen";

	if ([bundle pathForResource:storyboardName ofType:@"storyboardc"] == nil) {
		return;
	}

	UIStoryboard *launchStoryboard = [UIStoryboard storyboardWithName:storyboardName bundle:bundle];

	UIViewController *controller = [launchStoryboard instantiateInitialViewController];
	self.godotLoadingOverlay = controller.view;
	self.godotLoadingOverlay.frame = self.view.bounds;
	self.godotLoadingOverlay.autoresizingMask = UIViewAutoresizingFlexibleHeight | UIViewAutoresizingFlexibleWidth;

	[self.view addSubview:self.godotLoadingOverlay];
}

- (BOOL)godotViewFinishedSetup:(GodotView *)view {
	[self.godotLoadingOverlay removeFromSuperview];
	self.godotLoadingOverlay = nil;

	return YES;
}

- (void)dealloc {
	self.connectedMice = nil;
	self.keyboardView = nil;

	self.renderer = nil;

	if (self.godotLoadingOverlay) {
		[self.godotLoadingOverlay removeFromSuperview];
		self.godotLoadingOverlay = nil;
	}

	[[NSNotificationCenter defaultCenter] removeObserver:self];
}

// MARK: Orientation

- (UIRectEdge)preferredScreenEdgesDeferringSystemGestures {
	if (GLOBAL_GET("display/window/ios/suppress_ui_gesture")) {
		return UIRectEdgeAll;
	} else {
		return UIRectEdgeNone;
	}
}

- (BOOL)shouldAutorotate {
	if (!DisplayServerIOS::get_singleton()) {
		return NO;
	}

	switch (DisplayServerIOS::get_singleton()->screen_get_orientation(DisplayServer::SCREEN_OF_MAIN_WINDOW)) {
		case DisplayServer::SCREEN_SENSOR:
		case DisplayServer::SCREEN_SENSOR_LANDSCAPE:
		case DisplayServer::SCREEN_SENSOR_PORTRAIT:
			return YES;
		default:
			return NO;
	}
}

- (UIInterfaceOrientationMask)supportedInterfaceOrientations {
	if (!DisplayServerIOS::get_singleton()) {
		return UIInterfaceOrientationMaskAll;
	}

	switch (DisplayServerIOS::get_singleton()->screen_get_orientation(DisplayServer::SCREEN_OF_MAIN_WINDOW)) {
		case DisplayServer::SCREEN_PORTRAIT:
			return UIInterfaceOrientationMaskPortrait;
		case DisplayServer::SCREEN_REVERSE_LANDSCAPE:
			if (UIDevice.currentDevice.userInterfaceIdiom == UIUserInterfaceIdiomPad) {
				return UIInterfaceOrientationMaskLandscapeLeft;
			} else {
				return UIInterfaceOrientationMaskLandscapeRight;
			}
		case DisplayServer::SCREEN_REVERSE_PORTRAIT:
			return UIInterfaceOrientationMaskPortraitUpsideDown;
		case DisplayServer::SCREEN_SENSOR_LANDSCAPE:
			return UIInterfaceOrientationMaskLandscape;
		case DisplayServer::SCREEN_SENSOR_PORTRAIT:
			return UIInterfaceOrientationMaskPortrait | UIInterfaceOrientationMaskPortraitUpsideDown;
		case DisplayServer::SCREEN_SENSOR:
			return UIInterfaceOrientationMaskAll;
		case DisplayServer::SCREEN_LANDSCAPE:
			if (UIDevice.currentDevice.userInterfaceIdiom == UIUserInterfaceIdiomPad) {
				return UIInterfaceOrientationMaskLandscapeRight;
			} else {
				return UIInterfaceOrientationMaskLandscapeLeft;
			}
	}
}

- (BOOL)prefersStatusBarHidden {
	if (GLOBAL_GET("display/window/ios/hide_status_bar")) {
		return YES;
	} else {
		return NO;
	}
}

- (BOOL)prefersHomeIndicatorAutoHidden {
	if (GLOBAL_GET("display/window/ios/hide_home_indicator")) {
		return YES;
	} else {
		return NO;
	}
}

- (BOOL)prefersPointerLocked API_AVAILABLE(ios(14.0)) {
	if (DisplayServerIOS::get_singleton()) {
		DisplayServer::MouseMode mm = DisplayServerIOS::get_singleton()->mouse_get_mode();
		if (mm == DisplayServer::MouseMode::MOUSE_MODE_VISIBLE) {
			return NO;
		} else {
			return YES;
		}
	}
	return NO;
}

// MARK: Keyboard

- (void)keyboardOnScreen:(NSNotification *)notification {
	NSDictionary *info = notification.userInfo;
	NSValue *value = info[UIKeyboardFrameEndUserInfoKey];

	CGRect rawFrame = [value CGRectValue];
	CGRect keyboardFrame = [self.view convertRect:rawFrame fromView:nil];

	if (DisplayServerIOS::get_singleton()) {
		DisplayServerIOS::get_singleton()->virtual_keyboard_set_height(keyboardFrame.size.height);
	}
}

- (void)keyboardHidden:(NSNotification *)notification {
	if (DisplayServerIOS::get_singleton()) {
		DisplayServerIOS::get_singleton()->virtual_keyboard_set_height(0);
	}
}

@end
