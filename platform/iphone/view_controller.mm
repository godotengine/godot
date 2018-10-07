/*************************************************************************/
/*  view_controller.mm                                                   */
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

#import "view_controller.h"
#import "core/os/keyboard.h"
#import "gl_view.h"
#import "os_iphone.h"
#import "video_iphone.h"

static String keyboard_text;
static GodotGameViewController *_instance = nil;

@interface GodotGameViewController ()

@property(nonatomic, strong) NSMutableArray *activeTouches;

@end

@implementation GodotGameViewController

- (id)initWithFrame:(CGRect)frame {
	if (self = [super init]) {
		self.activeTouches = [NSMutableArray arrayWithCapacity:10];
		self.view.frame = frame;

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

		_instance = self;
	}
	return self;
}

- (void)loadView {
	const CGRect frame = [UIApplication sharedApplication].keyWindow.bounds;
	self.view = [[GLView alloc] initWithFrame:frame];

	MediaView *mediaView = [[MediaView alloc] initWithFrame:frame];
	mediaView.hidden = YES;
	[self.view addSubview:mediaView];
}

- (void)viewDidLoad {
	[super viewDidLoad];

	if (@available(iOS 11.0, *)) {
		[self setNeedsUpdateOfScreenEdgesDeferringSystemGestures];
	}
}

- (UIRectEdge)preferredScreenEdgesDeferringSystemGestures {
	return UIRectEdgeAll;
}

- (GLView *)glView {
	return (GLView *)self.view;
}

#pragma mark - UIViewController Overrides
- (BOOL)shouldAutorotate {
	switch (OS::get_singleton()->get_screen_orientation()) {
		case OS::SCREEN_SENSOR:
		case OS::SCREEN_SENSOR_LANDSCAPE:
		case OS::SCREEN_SENSOR_PORTRAIT:
			return YES;
		default:
			return NO;
	}
};

- (UIInterfaceOrientationMask)supportedInterfaceOrientations {
	return UIInterfaceOrientationMaskAll;
};

- (BOOL)prefersStatusBarHidden {
	return YES;
}

#pragma mark - Touch Handling via UIControl
- (void)touchesBegan:(NSSet *)touches withEvent:(UIEvent *)event {
	UITouch *touch = [touches anyObject];

	[self.activeTouches addObject:touch];

	CGPoint touchPoint = [self scaledPoint:[touch locationInView:self.view]];
	OSIPhone::get_singleton()->touch_press([self.activeTouches indexOfObject:touch], touchPoint.x, touchPoint.y, true, touch.tapCount > 1);
}

- (void)touchesMoved:(NSSet *)touches withEvent:(UIEvent *)event {
	UITouch *touch = [touches anyObject];

	CGPoint touchPoint = [self scaledPoint:[touch locationInView:self.view]];
	CGPoint previousPoint = [self scaledPoint:[touch previousLocationInView:self.view]];
	OSIPhone::get_singleton()->touch_drag([self.activeTouches indexOfObject:touch], previousPoint.x, previousPoint.y, touchPoint.x, touchPoint.y);
}

- (void)touchesEnded:(NSSet *)touches withEvent:(UIEvent *)event {
	UITouch *touch = [touches anyObject];

	CGPoint touchPoint = [self scaledPoint:[touch locationInView:self.view]];
	OSIPhone::get_singleton()->touch_press([self.activeTouches indexOfObject:touch], touchPoint.x, touchPoint.y, false, false);

	[self.activeTouches removeObject:touch];
}

- (void)touchesCancelled:(NSSet *)touches withEvent:(UIEvent *)event {
	UITouch *touch = [touches anyObject];

	OSIPhone::get_singleton()->touches_cancelled();

	[self.activeTouches removeObject:touch];
}

- (CGPoint)scaledPoint:(CGPoint)point {
	return CGPointMake(point.x * self.view.contentScaleFactor, point.y * self.view.contentScaleFactor);
}

#pragma mark - Keyboard Input
- (BOOL)canBecomeFirstResponder {
	return YES;
}

- (void)keyboardOnScreen:(NSNotification *)notification {
	NSValue *value = notification.userInfo[UIKeyboardFrameEndUserInfoKey];
	CGRect frame = [value CGRectValue];
	const CGFloat kScaledHeight = frame.size.height * [UIScreen mainScreen].nativeScale;
	OSIPhone::get_singleton()->set_virtual_keyboard_height(kScaledHeight);
}

- (void)keyboardHidden:(NSNotification *)notification {
	OSIPhone::get_singleton()->set_virtual_keyboard_height(0);
}

- (void)deleteBackward {
	if (keyboard_text.length())
		keyboard_text.erase(keyboard_text.length() - 1, 1);
	OSIPhone::get_singleton()->key(KEY_BACKSPACE, true);
}

- (BOOL)hasText {
	return keyboard_text.length() ? YES : NO;
}

- (void)insertText:(NSString *)p_text {
	String character;
	character.parse_utf8([p_text UTF8String]);
	keyboard_text = keyboard_text + character;
	OSIPhone::get_singleton()->key(character[0] == 10 ? KEY_ENTER : character[0], true);
	printf("inserting text with character %i\n", character[0]);
}

#pragma mark - Game Center
- (void)gameCenterViewControllerDidFinish:(GKGameCenterViewController *)gameCenterViewController {
#ifdef GAME_CENTER_ENABLED
	//[gameCenterViewController dismissViewControllerAnimated:YES completion:^{GameCenter::get_singleton()->game_center_closed();}];//version for signaling when overlay is completely gone
	GameCenter::get_singleton()->game_center_closed();
	[gameCenterViewController dismissViewControllerAnimated:YES completion:nil];
#endif
}

@end

#pragma mark - Bridged Text Input
void _show_keyboard(String p_existing) {
	keyboard_text = p_existing;
	NSLog(@"Show keyboard");
	[_instance becomeFirstResponder];
};

void _hide_keyboard() {
	NSLog(@"Hide keyboard and clear text");
	[_instance resignFirstResponder];
	keyboard_text = "";
};

Rect2 _get_ios_window_safe_area(float p_window_width, float p_window_height) {
	UIEdgeInsets insets = UIEdgeInsetsMake(0, 0, 0, 0);
	if ([_instance.view respondsToSelector:@selector(safeAreaInsets)]) {
		insets = [_instance.view safeAreaInsets];
	}
	ERR_FAIL_COND_V(insets.left < 0 || insets.top < 0 || insets.right < 0 || insets.bottom < 0,
			Rect2(0, 0, p_window_width, p_window_height));
	return Rect2(insets.left, insets.top, p_window_width - insets.right - insets.left, p_window_height - insets.bottom - insets.top);
}