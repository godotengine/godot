/*************************************************************************/
/*  display_layer.mm                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#import "display_layer.h"

#include "core/os/keyboard.h"
#include "core/project_settings.h"
#include "main/main.h"
#include "os_iphone.h"
#include "servers/audio_server.h"
#include "thirdparty/metalangle/include/GLES2/gl2.h"
#include "thirdparty/metalangle/ios/xcode/MGLKit/MGLContext.h"

#import <AudioToolbox/AudioServices.h>
#import <GameController/GameController.h>
#import <QuartzCore/QuartzCore.h>
#import <UIKit/UIKit.h>

int gl_view_base_fb;
bool gles3_available = true;

@implementation GodotOpenGLLayer {
	MGLContext *context;
}

- (void)initializeDisplayLayer {
	// Configure it so that it is opaque, does not retain the contents of the backbuffer when displayed, and uses RGBA8888 color.

	self.opaque = YES;
	self.retainedBacking = NO;
	self.drawableColorFormat = MGLDrawableColorFormatRGBA8888;
	bool fallback_gl2 = false;
	// Create a GL ES 3 context based on the gl driver from project settings
	if (GLOBAL_GET("rendering/quality/driver/driver_name") == "GLES3") {
		context = [[MGLContext alloc] initWithAPI:kMGLRenderingAPIOpenGLES3];
		NSLog(@"Setting up an OpenGL ES 3.0 context. Based on Project Settings \"rendering/quality/driver/driver_name\"");
		if (!context && GLOBAL_GET("rendering/quality/driver/fallback_to_gles2")) {
			gles3_available = false;
			fallback_gl2 = true;
			NSLog(@"Failed to create OpenGL ES 3.0 context. Falling back to OpenGL ES 2.0");
		}
	}

	// Create GL ES 2 context
	if (GLOBAL_GET("rendering/quality/driver/driver_name") == "GLES2" || fallback_gl2) {
		context = [[MGLContext alloc] initWithAPI:kMGLRenderingAPIOpenGLES2];
		NSLog(@"Setting up an OpenGL ES 2.0 context.");
		if (!context) {
			NSLog(@"Failed to create OpenGL ES 2.0 context!");
			return;
		}
	}

	if (![MGLContext setCurrentContext:context]) {
		NSLog(@"Failed to set MGLContext!");
		return;
	}

	[self updateVideoMode];
}

- (void)layoutDisplayLayer {
	[MGLContext setCurrentContext:context];
	[self updateVideoMode];
}

- (void)startRenderDisplayLayer {
	[MGLContext setCurrentContext:context forLayer:self];

	[self bindDefaultFrameBuffer];
}

- (void)stopRenderDisplayLayer {
	[context present:self];

#ifdef DEBUG_ENABLED
	GLenum err = glGetError();
	if (err) {
		NSLog(@"DrawView: %x error", err);
	}
#endif
}

- (void)dealloc {
	if ([MGLContext currentContext] == context) {
		[MGLContext setCurrentContext:nil];
	}

	if (context) {
		context = nil;
	}
}

- (void)updateVideoMode {
	if (OS::get_singleton()) {
		GLuint backingWidth, backingHeight;
		backingWidth = self.drawableSize.width;
		backingHeight = self.drawableSize.height;

		OS::VideoMode vm;
		vm.fullscreen = true;
		vm.width = backingWidth;
		vm.height = backingHeight;
		vm.resizable = false;
		OS::get_singleton()->set_video_mode(vm);
		OSIPhone::get_singleton()->set_base_framebuffer(self.defaultOpenGLFrameBufferID);
	}

	gl_view_base_fb = self.defaultOpenGLFrameBufferID;
}

@end
