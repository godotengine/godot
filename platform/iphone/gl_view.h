/*************************************************************************/
/*  gl_view.h                                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
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

#import <AVFoundation/AVFoundation.h>
#import <MediaPlayer/MediaPlayer.h>
#import <OpenGLES/EAGL.h>
#import <OpenGLES/ES1/gl.h>
#import <OpenGLES/ES1/glext.h>
#import <UIKit/UIKit.h>

@protocol GLViewDelegate;

@interface GLView : UIView <UIKeyInput> {
@private
	// The pixel dimensions of the backbuffer
	GLint backingWidth;
	GLint backingHeight;

	EAGLContext *context;

	// OpenGL names for the renderbuffer and framebuffers used to render to this view
	GLuint viewRenderbuffer, viewFramebuffer;

	// OpenGL name for the depth buffer that is attached to viewFramebuffer, if it exists (0 if it does not exist)
	GLuint depthRenderbuffer;

	BOOL useCADisplayLink;
	// CADisplayLink available on 3.1+ synchronizes the animation timer & drawing with the refresh rate of the display, only supports animation intervals of 1/60 1/30 & 1/15
	CADisplayLink *displayLink;

	// An animation timer that, when animation is started, will periodically call -drawView at the given rate.
	// Only used if CADisplayLink is not
	NSTimer *animationTimer;

	NSTimeInterval animationInterval;

	// Delegate to do our drawing, called by -drawView, which can be called manually or via the animation timer.
	id<GLViewDelegate> delegate;

	// Flag to denote that the -setupView method of a delegate has been called.
	// Resets to NO whenever the delegate changes.
	BOOL delegateSetup;
	BOOL active;
	float screen_scale;
}

@property(nonatomic, assign) id<GLViewDelegate> delegate;

// AVPlayer-related properties
@property(strong, nonatomic) AVAsset *avAsset;
@property(strong, nonatomic) AVPlayerItem *avPlayerItem;
@property(strong, nonatomic) AVPlayer *avPlayer;
@property(strong, nonatomic) AVPlayerLayer *avPlayerLayer;

// Old videoplayer properties
@property(strong, nonatomic) MPMoviePlayerController *moviePlayerController;
@property(strong, nonatomic) UIWindow *backgroundWindow;

- (void)startAnimation;
- (void)stopAnimation;
- (void)drawView;

- (BOOL)canBecomeFirstResponder;

- (void)open_keyboard;
- (void)hide_keyboard;
- (void)deleteBackward;
- (BOOL)hasText;
- (void)insertText:(NSString *)p_text;

- (id)initGLES;
- (BOOL)createFramebuffer;
- (void)destroyFramebuffer;

- (void)audioRouteChangeListenerCallback:(NSNotification *)notification;
- (void)keyboardOnScreen:(NSNotification *)notification;
- (void)keyboardHidden:(NSNotification *)notification;

@property NSTimeInterval animationInterval;
@property(nonatomic, assign) BOOL useCADisplayLink;

@end

@protocol GLViewDelegate <NSObject>

@required

// Draw with OpenGL ES
- (void)drawView:(GLView *)view;

@optional

// Called whenever you need to do some initialization before rendering.
- (void)setupView:(GLView *)view;

@end
