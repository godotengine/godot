/*************************************************************************/
/*  gl_view.mm                                                           */
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

#import "gl_view.h"

#import "core/project_settings.h"
#import "os_iphone.h"
#import "servers/audio_server.h"

#import <OpenGLES/EAGLDrawable.h>

bool gles3_available = true;
int gl_view_base_fb;

CGFloat _points_to_pixels(CGFloat);

CGFloat _points_to_pixels(CGFloat points) {
	float pixelPerInch;
	if (UI_USER_INTERFACE_IDIOM() == UIUserInterfaceIdiomPad) {
		pixelPerInch = 132;
	} else if (UI_USER_INTERFACE_IDIOM() == UIUserInterfaceIdiomPhone) {
		pixelPerInch = 163;
	} else {
		pixelPerInch = 160;
	}
	CGFloat pointsPerInch = 72.0;
	return (points / pointsPerInch * pixelPerInch);
}

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
}

@interface GLView ()

@property(nonatomic, strong) CADisplayLink *displayLink;
@property(nonatomic, strong) NSTimer *animationTimer;
@property(nonatomic, assign) BOOL useCADisplayLink;

@property(nonatomic, assign, getter=isActive) BOOL active;
@property(nonatomic, assign, getter=isSetUpComplete) BOOL setUpComplete;

@end

@implementation GLView {
	// The pixel dimensions of the backbuffer
	GLint backingWidth;
	GLint backingHeight;

	EAGLContext *context;

	// OpenGL names for the renderbuffer and framebuffers used to render to this view
	GLuint viewRenderbuffer, viewFramebuffer;

	// OpenGL name for the depth buffer that is attached to viewFramebuffer, if it exists (0 if it does not exist)
	GLuint depthRenderbuffer;
}

@synthesize animationInterval = _animationInterval;

// Implement this to override the default layer class (which is [CALayer class]).
// We do this so that our view will be backed by a layer that is capable of OpenGL ES rendering.
+ (Class)layerClass {
	return [CAEAGLLayer class];
}

- (id)initWithFrame:(CGRect)frame {
	if (self = [super initWithFrame:frame]) {
		if (![self setUp]) {
			return nil;
		}
	}
	return self;
}

- (BOOL)setUp {
	_instance = self;

	self.useCADisplayLink = bool(GLOBAL_DEF("display.iOS/use_cadisplaylink", true)) ? YES : NO;
	printf("CADisplayLink is %s. From setting 'display.iOS/use_cadisplaylink'\n", self.useCADisplayLink ? "enabled" : "disabled");

	self.active = NO;
	self.multipleTouchEnabled = YES;
	self.autocorrectionType = UITextAutocorrectionTypeNo;

	// Get our backing layer
	CAEAGLLayer *eaglLayer = (CAEAGLLayer *)self.layer;

	// Configure it so that it is opaque, does not retain the contents of the backbuffer when displayed, and uses RGBA8888 color.
	eaglLayer.opaque = YES;
    eaglLayer.drawableProperties = @{ kEAGLDrawablePropertyRetainedBacking : @(NO),
                                      kEAGLDrawablePropertyColorFormat : kEAGLColorFormatRGBA8 };
    
    bool fallback_gl2 = false;
	// Create a GL ES 3 context based on the gl driver from project settings
	if (GLOBAL_GET("rendering/quality/driver/driver_name") == "GLES3") {
		context = [[EAGLContext alloc] initWithAPI:kEAGLRenderingAPIOpenGLES3];
		NSLog(@"Setting up an OpenGL ES 3.0 context. Based on Project Settings \"rendering/quality/driver/driver_name\"");
		if (!context && GLOBAL_GET("rendering/quality/driver/fallback_to_gles2")) {
			gles3_available = false;
			fallback_gl2 = true;
			NSLog(@"Failed to create OpenGL ES 3.0 context. Falling back to OpenGL ES 2.0");
		}
	}

	// Create GL ES 2 context
	if (GLOBAL_GET("rendering/quality/driver/driver_name") == "GLES2" || fallback_gl2) {
		context = [[EAGLContext alloc] initWithAPI:kEAGLRenderingAPIOpenGLES2];
		NSLog(@"Setting up an OpenGL ES 2.0 context.");
		if (!context) {
			NSLog(@"Failed to create OpenGL ES 2.0 context!");
			return nil;
		}
	}

	if (![EAGLContext setCurrentContext:context]) {
		NSLog(@"Failed to set EAGLContext!");
		return nil;
	}
	if (![self createFramebuffer]) {
		NSLog(@"Failed to create frame buffer!");
		return nil;
	}

	// Default the animation interval to 1/60th of a second.
	animationInterval = 1.0 / 60.0;

	self.multipleTouchEnabled = YES;
	self.autocorrectionType = UITextAutocorrectionTypeNo;

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

- (void)dealloc {
	[self stopAnimation];

	if ([EAGLContext currentContext] == context) {
		[EAGLContext setCurrentContext:nil];
	}

	[context release];
	context = nil;

	[super dealloc];
}

- (void)layoutSubviews {
	[EAGLContext setCurrentContext:context];
	[self destroyFramebuffer];
	[self createFramebuffer];
	[self drawView];
}

- (BOOL)createFramebuffer {
	// Generate IDs for a framebuffer object and a color renderbuffer
	glGenFramebuffersOES(1, &viewFramebuffer);
	glBindFramebufferOES(GL_FRAMEBUFFER_OES, viewFramebuffer);

	// Next the render backing store
	[self createRenderBuffer];

	// For this sample, we also need a depth buffer, so we'll create and attach one via another renderbuffer.
	[self createDepthBuffer];

	if (glCheckFramebufferStatusOES(GL_FRAMEBUFFER_OES) != GL_FRAMEBUFFER_COMPLETE_OES) {
		printf("failed to make complete framebuffer object %x", glCheckFramebufferStatusOES(GL_FRAMEBUFFER_OES));
		return NO;
	}

	if (OS::get_singleton()) {
		OSIPhone::get_singleton()->set_base_framebuffer(viewFramebuffer);

		OS::VideoMode vm;
		vm.fullscreen = true;
		vm.resizable = false;
		glGetRenderbufferParameterivOES(GL_RENDERBUFFER_OES, GL_RENDERBUFFER_WIDTH_OES, &vm.width);
		glGetRenderbufferParameterivOES(GL_RENDERBUFFER_OES, GL_RENDERBUFFER_HEIGHT_OES, &vm.height);
		OS::get_singleton()->set_video_mode(vm);
	}

	// Save the gl reference to the frame buffer
	gl_view_base_fb = viewFramebuffer;

	return YES;
}

- (void)createRenderBuffer {

	// Generate a gl render buffer and make it active
	glGenRenderbuffersOES(1, &viewRenderbuffer);
	glBindRenderbufferOES(GL_RENDERBUFFER_OES, viewRenderbuffer);

	// This call associates the storage for the current render buffer with the EAGLDrawable (our CAEAGLLayer)
	// allowing us to draw into a buffer that will later be rendered to screen wherever 
	// the layer is (which corresponds with our view).
	[context renderbufferStorage:GL_RENDERBUFFER_OES
					fromDrawable:(id<EAGLDrawable>)self.layer];
	glFramebufferRenderbufferOES(GL_FRAMEBUFFER_OES, GL_COLOR_ATTACHMENT0_OES, GL_RENDERBUFFER_OES, viewRenderbuffer);

	glGetRenderbufferParameterivOES(GL_RENDERBUFFER_OES,
			GL_RENDERBUFFER_WIDTH_OES, &backingWidth);
	glGetRenderbufferParameterivOES(GL_RENDERBUFFER_OES,
			GL_RENDERBUFFER_HEIGHT_OES, &backingHeight);
}

- (void)createDepthBuffer {
	glGenRenderbuffersOES(1, &depthRenderbuffer);
	glBindRenderbufferOES(GL_RENDERBUFFER_OES, depthRenderbuffer);
	glRenderbufferStorageOES(GL_RENDERBUFFER_OES, GL_DEPTH_COMPONENT16_OES, backingWidth, backingHeight);
	glFramebufferRenderbufferOES(GL_FRAMEBUFFER_OES, GL_DEPTH_ATTACHMENT_OES, GL_RENDERBUFFER_OES, depthRenderbuffer);
}

- (void)destroyFramebuffer {
	glDeleteFramebuffersOES(1, &viewFramebuffer);
	viewFramebuffer = 0;
	glDeleteRenderbuffersOES(1, &viewRenderbuffer);
	viewRenderbuffer = 0;

	if (depthRenderbuffer) {
		glDeleteRenderbuffersOES(1, &depthRenderbuffer);
		depthRenderbuffer = 0;
	}
}

- (void)startAnimation {
	// Already active?
	if (self.isActive) return;

	self.active = YES;

	printf("start animation!\n");

	if (self.useCADisplayLink) {

		// Create a display link and try and grab the fatest refresh rate
		self.displayLink = [CADisplayLink displayLinkWithTarget:self
			  									       selector:@selector(drawView)];
		if ([self.displayLink respondsToSelector:@selector(preferredFramesPerSecond)]) {
			// TODO: Add support for max fps from project settings here, so we're not
			// rendering too quickly
			self.displayLink.preferredFramesPerSecond = 0;
			// We could potentially check the display link's maximumFPS here.
			// If it supports 120 we could force the preferredFPS to that.
			// Currently that the's only way to drive a game at 120Hz
		} else {
			// DEPRECATED: 
			// Approximate frame rate: assumes device refreshes at 60 fps.
			// Note that newer iOS devices are 120Hz screens
			self.displayLink.frameInterval = 1;
		}

		// Setup DisplayLink in main thread
		[self.displayLink addToRunLoop:[NSRunLoop currentRunLoop]
						       forMode:NSRunLoopCommonModes];
	} else {
		// Fallback to low resolution NSTimer instead of 
		self.animationTimer = [NSTimer scheduledTimerWithTimeInterval:self.animationInterval
		 												       target:self
														     selector:@selector(drawView)
														 	 userInfo:nil
														  	 repeats:YES];
	}
}

- (void)stopAnimation {
	if (!self.isActive) return;

	self.active = NO;
	printf("stop animation!\n");

	if (self.useCADisplayLink) {
		[self.displayLink invalidate];
		self.displayLink = nil;
	} else {
		[self.animationTimer invalidate];
		self.animationTimer = nil;
	}
}

- (void)setAnimationInterval:(NSTimeInterval)interval {
	_animationInterval = interval;
	// Reset the timers the timers if we're currently rendering
	if (self.isActive) {
		[self stopAnimation];
	}
	[self startAnimation];
}

- (NSTimeInterval)animationInterval {
	return _animationInterval;
}

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

	if (!active) {
		printf("draw view not active!\n");
		return;
	}

	// Make sure that you are drawing to the current context
	[EAGLContext setCurrentContext:context];

	// If our drawing delegate needs to have the view setup, then call -setupView: and flag that it won't need to be called again.
	if (!self.isSetUpComplete && [self.delegate respondsToSelector:@selector(setupView:)]) {
		[self.delegate setupView:self];
		self.setUpComplete = YES;
	}

	glBindFramebufferOES(GL_FRAMEBUFFER_OES, viewFramebuffer);
	[self.delegate drawView:self];

	glBindRenderbufferOES(GL_RENDERBUFFER_OES, viewRenderbuffer);
	[context presentRenderbuffer:GL_RENDERBUFFER_OES];

#ifdef DEBUG_ENABLED
	GLenum err = glGetError();
	if (err)
		NSLog(@"%x (gl) error", err);
#endif
}

@end
