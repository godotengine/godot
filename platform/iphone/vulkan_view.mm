/*************************************************************************/
/*  gl_view.mm                                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#import "vulkan_view.h"

#include "core/os/keyboard.h"
#include "core/project_settings.h"
#include "os_iphone.h"
#include "servers/audio_server.h"

#import <QuartzCore/QuartzCore.h>

@implementation VulkanView

@synthesize animationInterval;

// Implement this to override the default layer class (which is [CALayer class]).
// We do this so that our view will be backed by a layer that is capable of OpenGL ES rendering.
+ (Class)layerClass {
	return [CAMetalLayer class];
}

//The GL view is stored in the nib file. When it's unarchived it's sent -initWithCoder:
- (id)initWithCoder:(NSCoder *)coder {
	active = FALSE;
	if ((self = [super initWithCoder:coder])) {
		self = [self initVulkan];
	}
	return self;
}

- (id)initVulkan {
	// Get our backing layer
	CAMetalLayer *metalLayer = (CAMetalLayer *)self.layer;

	// FIXME: Add Vulkan support via MoltenVK. Add fallback code back?

	// Create GL ES 2 context
	if (GLOBAL_GET("rendering/quality/driver/driver_name") == "Vulkan") {
		NSLog(@"Setting up an Vulkan context.");
	}

	if (![self createFramebuffer]) {
		NSLog(@"Failed to create frame buffer!");
		return nil;
	}

	// Default the animation interval to 1/60th of a second.
	animationInterval = 1.0 / 60.0;
	return self;
}


@synthesize useCADisplayLink;

// If our view is resized, we'll be asked to layout subviews.
// This is the perfect opportunity to also update the framebuffer so that it is
// the same size as our display area.

- (void)layoutSubviews {
	[self destroyFramebuffer];
	[self createFramebuffer];
	[self drawView];
}

- (BOOL)createFramebuffer {
	// Generate IDs for a framebuffer object and a color renderbuffer
	UIScreen *mainscr = [UIScreen mainScreen];
	printf("******** screen size %i, %i\n", (int)mainscr.currentMode.size.width, (int)mainscr.currentMode.size.height);
	self.contentScaleFactor = mainscr.nativeScale;

	if (OS::get_singleton()) {
		OS::VideoMode vm;
		vm.fullscreen = true;
		vm.width = backingWidth;
		vm.height = backingHeight;
		vm.resizable = false;
		OS::get_singleton()->set_video_mode(vm);
		OSIPhone::get_singleton()->set_base_framebuffer(viewFramebuffer);
	};
	// gl_view_base_fb = viewFramebuffer;

	return YES;
}

// Clean up any buffers we have allocated.
- (void)destroyFramebuffer {

}

- (void)startAnimation {
	if (active)
		return;
	active = TRUE;
	printf("start animation!\n");
	if (useCADisplayLink) {

		// Approximate frame rate
		// assumes device refreshes at 60 fps
		int frameInterval = (int)floor(animationInterval * 60.0f);

		displayLink = [CADisplayLink displayLinkWithTarget:self selector:@selector(drawView)];
		[displayLink setFrameInterval:frameInterval];

		// Setup DisplayLink in main thread
		[displayLink addToRunLoop:[NSRunLoop currentRunLoop] forMode:NSRunLoopCommonModes];
	} else {
		animationTimer = [NSTimer scheduledTimerWithTimeInterval:animationInterval target:self selector:@selector(drawView) userInfo:nil repeats:YES];
	}
}

- (void)stopAnimation {
	if (!active)
		return;
	active = FALSE;
	printf("******** stop animation!\n");

	if (useCADisplayLink) {
		[displayLink invalidate];
		displayLink = nil;
	} else {
		[animationTimer invalidate];
		animationTimer = nil;
	}
}

- (void)setAnimationInterval:(NSTimeInterval)interval {
	animationInterval = interval;
	if ((useCADisplayLink && displayLink) || (!useCADisplayLink && animationTimer)) {
		[self stopAnimation];
		[self startAnimation];
	}
}

// Updates the OpenGL view when the timer fires
- (void)drawView {

	if (!active) {
		printf("draw view not active!\n");
		return;
	};
	if (useCADisplayLink) {
		// Pause the CADisplayLink to avoid recursion
		[displayLink setPaused:YES];

		// Process all input events
		while (CFRunLoopRunInMode(kCFRunLoopDefaultMode, 0.0, TRUE) == kCFRunLoopRunHandledSource)
			;

		// We are good to go, resume the CADisplayLink
		[displayLink setPaused:NO];
	}

	// Make sure that you are drawing to the current context

	// If our drawing delegate needs to have the view setup, then call -setupView: and flag that it won't need to be called again.
	// if (!delegateSetup) {
	// 	[delegate setupView:self];
	// 	delegateSetup = YES;
	// }

	// [delegate drawView:self];
}

// When created via code however, we get initWithFrame
- (id)initWithFrame:(CGRect)frame {
	self = [super initWithFrame:frame];
	// _instance = self;
	printf("after init super %p\n", self);
	if (self != nil) {
		self = [self initVulkan];
		printf("after init vulkan %p\n", self);
	}

	return self;
}

// Stop animating and release resources when they are no longer needed.
- (void)dealloc {
	[self stopAnimation];

	[super dealloc];
}

@end
