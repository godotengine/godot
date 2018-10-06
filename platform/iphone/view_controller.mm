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
#import "gl_view.h"
#import "os_iphone.h"

@interface GodotGameViewController ()

@property(nonatomic, strong) NSMutableArray *activeTouches;

@end

@implementation GodotGameViewController

- (id)initWithFrame:(CGRect)frame {
	if (self = [super init]) {
		self.activeTouches = [NSMutableArray arrayWithCapacity:10];
		self.view.frame = frame;
	}
	return self;
}

- (void)loadView {
	const CGRect frame = [UIApplication sharedApplication].keyWindow.bounds;
	self.view = [[GLView alloc] initWithFrame:frame];
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

#pragma mark - View Geometry
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
	switch (OS::get_singleton()->get_screen_orientation()) {
		case OS::SCREEN_PORTRAIT:
			return UIInterfaceOrientationMaskPortrait;
		case OS::SCREEN_REVERSE_LANDSCAPE:
			return UIInterfaceOrientationMaskLandscapeRight;
		case OS::SCREEN_REVERSE_PORTRAIT:
			return UIInterfaceOrientationMaskPortraitUpsideDown;
		case OS::SCREEN_SENSOR_LANDSCAPE:
			return UIInterfaceOrientationMaskLandscape;
		case OS::SCREEN_SENSOR_PORTRAIT:
			return UIInterfaceOrientationMaskPortrait | UIInterfaceOrientationMaskPortraitUpsideDown;
		case OS::SCREEN_SENSOR:
			return UIInterfaceOrientationMaskAll;
		case OS::SCREEN_LANDSCAPE:
			return UIInterfaceOrientationMaskLandscapeLeft;
	}
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

#ifdef GAME_CENTER_ENABLED
- (void)gameCenterViewControllerDidFinish:(GKGameCenterViewController *)gameCenterViewController {
	//[gameCenterViewController dismissViewControllerAnimated:YES completion:^{GameCenter::get_singleton()->game_center_closed();}];//version for signaling when overlay is completely gone
	GameCenter::get_singleton()->game_center_closed();
	[gameCenterViewController dismissViewControllerAnimated:YES completion:nil];
}
#endif

@end
