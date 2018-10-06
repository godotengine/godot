/*************************************************************************/
/*  gl_view.h                                                            */
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

#import <Foundation/Foundation.h>
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

	// An animation timer that, when animation is started, will periodically call -drawView at the given rate.
	NSTimer *animationTimer;
	// I'd like to remove the above as an option
}

@property(nonatomic, assign) id<GLViewDelegate> delegate;

@property (strong, nonatomic) UIWindow *backgroundWindow;

@property (nonatomic) UITextAutocorrectionType autocorrectionType;
@property (nonatomic, assign) NSTimeInterval animationInterval;
@property (nonatomic, assign) BOOL useCADisplayLink;
@property(nonatomic, assign, getter=isActive) BOOL active;
@property(nonatomic, assign, getter=isSetUpComplete) BOOL setUpComplete;

- (void)startAnimation;
- (void)stopAnimation;
- (void)drawView;

- (void)deleteBackward;
- (BOOL)hasText;
- (void)insertText:(NSString *)p_text;

- (void)keyboardOnScreen:(NSNotification *)notification;
- (void)keyboardHidden:(NSNotification *)notification;

@end

@protocol GLViewDelegate <NSObject>

@required

// Draw with OpenGL ES
- (void)drawView:(GLView *)view;

@optional

// Called whenever you need to do some initialization before rendering.
- (void)setupView:(GLView *)view;

@end
