/*************************************************************************/
/*  uikit_view_controller.mm                                             */
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

#import "uikit_view_controller.h"

#include "core/project_settings.h"
#import "uikit_view.h"
#import "uikit_view_renderer.h"

@interface UIKitViewController ()

@property(strong, nonatomic) UIView *uikitLoadingOverlay;

@end

@implementation UIKitViewController

- (UIKitView *)uikitView {
	return (UIKitView *)self.view;
}

- (void)loadView {
	UIKitView *view = [[UIKitView alloc] init];
	[view initializeRendering];

	self.view = view;

	view.delegate = self;
}

- (instancetype)initWithNibName:(NSString *)nibNameOrNil bundle:(NSBundle *)nibBundleOrNil {
	self = [super initWithNibName:nibNameOrNil bundle:nibBundleOrNil];

	if (self) {
		[self uikit_commonInit];
	}

	return self;
}

- (instancetype)initWithCoder:(NSCoder *)coder {
	self = [super initWithCoder:coder];

	if (self) {
		[self uikit_commonInit];
	}

	return self;
}

- (void)uikit_commonInit {
	// Initialize view controller values.
}

- (void)didReceiveMemoryWarning {
	[super didReceiveMemoryWarning];
	printf("*********** did receive memory warning!\n");
};

- (void)viewDidLoad {
	[super viewDidLoad];

	[self displayLoadingOverlay];
}

- (void)displayLoadingOverlay {
	NSBundle *bundle = [NSBundle mainBundle];
	NSString *storyboardName = @"Launch Screen";

	if ([bundle pathForResource:storyboardName ofType:@"storyboardc"] == nil) {
		return;
	}

	UIStoryboard *launchStoryboard = [UIStoryboard storyboardWithName:storyboardName bundle:bundle];

	UIViewController *controller = [launchStoryboard instantiateInitialViewController];
	self.uikitLoadingOverlay = controller.view;
	self.uikitLoadingOverlay.frame = self.view.bounds;
	self.uikitLoadingOverlay.autoresizingMask = UIViewAutoresizingFlexibleHeight | UIViewAutoresizingFlexibleWidth;

	[self.view addSubview:self.uikitLoadingOverlay];
}

- (BOOL)uikitViewFinishedSetup:(UIKitView *)view {
	[self.uikitLoadingOverlay removeFromSuperview];
	self.uikitLoadingOverlay = nil;

	return YES;
}

- (void)dealloc {
	if (self.uikitLoadingOverlay) {
		[self.uikitLoadingOverlay removeFromSuperview];
		self.uikitLoadingOverlay = nil;
	}

	[[NSNotificationCenter defaultCenter] removeObserver:self];
}

@end
