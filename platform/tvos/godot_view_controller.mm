/*************************************************************************/
/*  godot_view_controller.mm                                             */
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

#import "godot_view_controller.h"

#include "core/project_settings.h"
#import "godot_view.h"
#import "godot_view_renderer.h"
#import "keyboard_input_view.h"
#include "os_tvos.h"

@interface GodotViewController ()

@property(strong, nonatomic) GodotViewRenderer *renderer;
@property(strong, nonatomic) GodotKeyboardInputView *keyboardView;

@end

@implementation GodotViewController

- (GodotView *)godotView {
	return (GodotView *)self.view;
}

- (void)loadView {
	GodotView *view = [[GodotView alloc] init];
	[view initializeRendering];

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
}

- (void)viewDidLoad {
	[super viewDidLoad];

	[self observeKeyboard];
}

- (void)observeKeyboard {
	printf("******** setting up keyboard input view\n");
	self.keyboardView = [GodotKeyboardInputView new];
	[self.view addSubview:self.keyboardView];
}

- (void)dealloc {
	[self.keyboardView removeFromSuperview];
	self.keyboardView = nil;

	self.renderer = nil;

	[[NSNotificationCenter defaultCenter] removeObserver:self];
}

@end
