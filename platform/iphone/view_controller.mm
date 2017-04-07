/*************************************************************************/
/*  view_controller.mm                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "os_iphone.h"

extern "C" {

int add_path(int, char**);
int add_cmdline(int, char**);

int add_path(int p_argc, char** p_args) {

	NSString* str = [[[NSBundle mainBundle] infoDictionary] objectForKey:@"godot_path"];
	if (!str)
		return p_argc;

	p_args[p_argc++] = "-path";
	[str retain]; // memory leak lol (maybe make it static here and delete it in ViewController destructor? @todo
	p_args[p_argc++] = (char*)[str cString];
	p_args[p_argc] = NULL;

	return p_argc;
};

int add_cmdline(int p_argc, char** p_args) {

	NSArray* arr = [[[NSBundle mainBundle] infoDictionary] objectForKey:@"godot_cmdline"];
	if (!arr)
		return p_argc;

	for (int i=0; i < [arr count]; i++) {

		NSString* str = [arr objectAtIndex:i];
		if (!str)
			continue;
		[str retain]; // @todo delete these at some point
		p_args[p_argc++] = (char*)[str cString];
	};

	p_args[p_argc] = NULL;

	return p_argc;
};

};

@interface ViewController ()

@end

@implementation ViewController

- (void)didReceiveMemoryWarning {

	printf("*********** did receive memory warning!\n");
};

- (BOOL)shouldAutorotateToInterfaceOrientation:(UIInterfaceOrientation)p_orientation {

	if (/*OSIPhone::get_singleton() == NULL*/TRUE) {

		printf("checking on info.plist\n");
		NSArray* arr = [[[NSBundle mainBundle] infoDictionary] objectForKey:@"UISupportedInterfaceOrientations"];
		switch(p_orientation) {

		case UIInterfaceOrientationLandscapeLeft:
			return [arr indexOfObject:@"UIInterfaceOrientationLandscapeLeft"] != NSNotFound ? YES : NO;

		case UIInterfaceOrientationLandscapeRight:
			return [arr indexOfObject:@"UIInterfaceOrientationLandscapeRight"] != NSNotFound ? YES : NO;

		case UIInterfaceOrientationPortrait:
			return [arr indexOfObject:@"UIInterfaceOrientationPortrait"] != NSNotFound ? YES : NO;

		case UIInterfaceOrientationPortraitUpsideDown:
			return [arr indexOfObject:@"UIInterfaceOrientationPortraitUpsideDown"] != NSNotFound ? YES : NO;

		default:
			return NO;
		}
	};

	uint8_t supported = OSIPhone::get_singleton()->get_orientations();
	switch(p_orientation) {

	case UIInterfaceOrientationLandscapeLeft:
		return supported & (1<<OSIPhone::LandscapeLeft) ? YES : NO;

	case UIInterfaceOrientationLandscapeRight:
		return supported & (1<<OSIPhone::LandscapeRight) ? YES : NO;

	case UIInterfaceOrientationPortrait:
		return supported & (1<<OSIPhone::PortraitDown) ? YES : NO;

	case UIInterfaceOrientationPortraitUpsideDown:
		return supported & (1<<OSIPhone::PortraitUp) ? YES : NO;

	default:
		return NO;
	}
};

- (BOOL)prefersStatusBarHidden
{
	return YES;
}

#ifdef GAME_CENTER_ENABLED
- (void) gameCenterViewControllerDidFinish:(GKGameCenterViewController*) gameCenterViewController {
    //[gameCenterViewController dismissViewControllerAnimated:YES completion:^{GameCenter::get_singleton()->game_center_closed();}];//version for signaling when overlay is completely gone
    GameCenter::get_singleton()->game_center_closed();
    [gameCenterViewController dismissViewControllerAnimated:YES completion:nil];
}
#endif

@end
