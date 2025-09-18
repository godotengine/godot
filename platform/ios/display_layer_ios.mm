/**************************************************************************/
/*  display_layer_ios.mm                                                  */
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

#import "display_layer_ios.h"

#import "display_server_ios.h"
#import "os_ios.h"

#include "core/config/project_settings.h"
#include "core/os/keyboard.h"
#include "main/main.h"
#include "servers/audio_server.h"

#import <AudioToolbox/AudioServices.h>
#import <GameController/GameController.h>
#import <OpenGLES/EAGL.h>
#import <OpenGLES/ES1/glext.h>
#import <QuartzCore/QuartzCore.h>
#import <UIKit/UIKit.h>

@implementation GDTMetalLayer

- (void)initializeDisplayLayer {
#if defined(TARGET_OS_SIMULATOR) && TARGET_OS_SIMULATOR
	if (@available(iOS 13, *)) {
		// Simulator supports Metal since iOS 13
	} else {
		NSLog(@"iOS Simulator prior to iOS 13 does not support Metal rendering.");
	}
#endif
}

- (void)layoutDisplayLayer {
}

- (void)startRenderDisplayLayer {
}

- (void)stopRenderDisplayLayer {
}

- (void)setupContext:(GLManager *)context withSurface:(Ref<RenderingNativeSurface> *)surface {
}

@end

@implementation GDTOpenGLLayer {
	GLManager *gl_manager;
	Ref<RenderingNativeSurface> native_surface;
}

- (void)initializeDisplayLayer {
	// Configure it so that it is opaque, does not retain the contents of the backbuffer when displayed, and uses RGBA8888 color.
	self.opaque = YES;
	self.drawableProperties = [NSDictionary
			dictionaryWithObjectsAndKeys:[NSNumber numberWithBool:FALSE],
			kEAGLDrawablePropertyRetainedBacking,
			kEAGLColorFormatRGBA8,
			kEAGLDrawablePropertyColorFormat,
			nil];
}

- (void)layoutDisplayLayer {
	gl_manager->window_resize(DisplayServer::MAIN_WINDOW_ID, 0, 0);
}

- (void)startRenderDisplayLayer {
	gl_manager->window_make_current(DisplayServer::MAIN_WINDOW_ID);
}

- (void)stopRenderDisplayLayer {
	gl_manager->swap_buffers();
}

- (void)dealloc {
	gl_manager->window_destroy(DisplayServer::MAIN_WINDOW_ID);
}

- (void)setupContext:(GLManager *)context withSurface:(Ref<RenderingNativeSurface> *)surface {
	gl_manager = context;
	native_surface = *surface;
	gl_manager->initialize();
	gl_manager->window_create(DisplayServer::MAIN_WINDOW_ID, native_surface, 0, 0);
}

@end
