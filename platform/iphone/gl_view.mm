/*************************************************************************/
/*  gl_view.mm                                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2016 Juan Linietsky, Ariel Manzur.                 */
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

#import <QuartzCore/QuartzCore.h>
#import <OpenGLES/EAGLDrawable.h>
#include "os_iphone.h"
#include "core/os/keyboard.h"
#include "core/globals.h"
#include "servers/audio_server.h"

#import "gl_view.h"

/*
@interface GLView (private)

- (id)initGLES;
- (BOOL)createFramebuffer;
- (void)destroyFramebuffer;
@end
*/

int gl_view_base_fb;
static String keyboard_text;
static GLView* _instance = NULL;

static bool video_found_error = false;
static bool video_playing = false;
static float video_previous_volume = 0.0f;
static CMTime video_current_time;

void _show_keyboard(String);
void _hide_keyboard();
bool _play_video(String, float, String, String);
bool _is_video_playing();
void _pause_video();
void _focus_out_video();
void _unpause_video();
void _stop_video();

void _show_keyboard(String p_existing) {
	keyboard_text = p_existing;
	printf("instance on show is %p\n", _instance);
	[_instance open_keyboard];
};

void _hide_keyboard() {
	printf("instance on hide is %p\n", _instance);
	[_instance hide_keyboard];
	keyboard_text = "";
};

bool _play_video(String p_path, float p_volume, String p_audio_track, String p_subtitle_track) {
	p_path = Globals::get_singleton()->globalize_path(p_path);

	NSString* file_path = [[[NSString alloc] initWithUTF8String:p_path.utf8().get_data()] autorelease];

	_instance.avAsset = [AVAsset assetWithURL:[NSURL fileURLWithPath:file_path]];

	_instance.avPlayerItem =[[AVPlayerItem alloc]initWithAsset:_instance.avAsset];
	[_instance.avPlayerItem addObserver:_instance forKeyPath:@"status" options:0 context:nil];

	_instance.avPlayer = [[AVPlayer alloc]initWithPlayerItem:_instance.avPlayerItem];
	_instance.avPlayerLayer =[AVPlayerLayer playerLayerWithPlayer:_instance.avPlayer];

	[_instance.avPlayer addObserver:_instance forKeyPath:@"status" options:0 context:nil];
	[[NSNotificationCenter defaultCenter] addObserver:_instance
                                        selector:@selector(playerItemDidReachEnd:)
                                               name:AVPlayerItemDidPlayToEndTimeNotification
                                             object:[_instance.avPlayer currentItem]];

	[_instance.avPlayer addObserver:_instance forKeyPath:@"rate" options:NSKeyValueObservingOptionNew context:0];

	[_instance.avPlayerLayer setFrame:_instance.bounds];
	[_instance.layer addSublayer:_instance.avPlayerLayer];
	[_instance.avPlayer play];

	AVMediaSelectionGroup *audioGroup = [_instance.avAsset mediaSelectionGroupForMediaCharacteristic: AVMediaCharacteristicAudible];

	NSMutableArray *allAudioParams = [NSMutableArray array];
	for (id track in audioGroup.options)
	{
		NSString* language = [[track locale] localeIdentifier];
		NSLog(@"subtitle lang: %@", language);
        
        if ([language isEqualToString:[NSString stringWithUTF8String:p_audio_track.utf8()]])
        {
			AVMutableAudioMixInputParameters *audioInputParams = [AVMutableAudioMixInputParameters audioMixInputParameters];
			[audioInputParams setVolume:p_volume atTime:kCMTimeZero];
			[audioInputParams setTrackID:[track trackID]];
			[allAudioParams addObject:audioInputParams];

			AVMutableAudioMix *audioMix = [AVMutableAudioMix audioMix];
			[audioMix setInputParameters:allAudioParams];

			[_instance.avPlayer.currentItem selectMediaOption:track inMediaSelectionGroup: audioGroup];
			[_instance.avPlayer.currentItem setAudioMix:audioMix];

            break;
        }
	}

	AVMediaSelectionGroup *subtitlesGroup = [_instance.avAsset mediaSelectionGroupForMediaCharacteristic: AVMediaCharacteristicLegible];
	NSArray *useableTracks = [AVMediaSelectionGroup mediaSelectionOptionsFromArray:subtitlesGroup.options withoutMediaCharacteristics:[NSArray arrayWithObject:AVMediaCharacteristicContainsOnlyForcedSubtitles]];

	for (id track in useableTracks)
	{
		NSString* language = [[track locale] localeIdentifier];
		NSLog(@"subtitle lang: %@", language);
        
        if ([language isEqualToString:[NSString stringWithUTF8String:p_subtitle_track.utf8()]])
        {
            [_instance.avPlayer.currentItem selectMediaOption:track inMediaSelectionGroup: subtitlesGroup];
            break;
        }
	}

	video_playing = true;

	return true;
}

bool _is_video_playing() {
	if (_instance.avPlayer.error) {
		printf("Error during playback\n");	
	}
	return (_instance.avPlayer.rate > 0 && !_instance.avPlayer.error);
}

void _pause_video() {
	video_current_time = _instance.avPlayer.currentTime;
	[_instance.avPlayer pause];
	video_playing = false;
}

void _focus_out_video() {
	printf("focus out pausing video\n");
	[_instance.avPlayer pause];
};

void _unpause_video() {

	[_instance.avPlayer play];
	video_playing = true;
};

void _stop_video() {
	[_instance.avPlayer pause];
	[_instance.avPlayerLayer removeFromSuperlayer];
	_instance.avPlayer = nil;
	video_playing = false;
}

@implementation GLView

@synthesize animationInterval;
@synthesize isKeyboardShown=isKeyboardShown_;
@synthesize keyboardShowNotification = keyboardShowNotification_;

static const int max_touches = 8;
static UITouch* touches[max_touches];

static void init_touches() {

	for (int i=0; i<max_touches; i++) {
		touches[i] = NULL;
	};
};

static int get_touch_id(UITouch* p_touch) {

	int first = -1;
	for (int i=0; i<max_touches; i++) {
		if (first == -1 && touches[i] == NULL) {
			first = i;
			continue;
		};
		if (touches[i] == p_touch)
			return i;
	};

	if (first != -1) {
		touches[first] = p_touch;
		return first;
	};

	return -1;
};

static int remove_touch(UITouch* p_touch) {

	int remaining = 0;
	for (int i=0; i<max_touches; i++) {

		if (touches[i] == NULL)
			continue;
		if (touches[i] == p_touch)
			touches[i] = NULL;
		else
			++remaining;
	};
	return remaining;
};

static int get_first_id(UITouch* p_touch) {

	for (int i=0; i<max_touches; i++) {

		if (touches[i] != NULL)
			return i;
	};
	return -1;
};

static void clear_touches() {

	for (int i=0; i<max_touches; i++) {

		touches[i] = NULL;
	};
};

// Implement this to override the default layer class (which is [CALayer class]).
// We do this so that our view will be backed by a layer that is capable of OpenGL ES rendering.
+ (Class) layerClass
{
	return [CAEAGLLayer class];
}

//The GL view is stored in the nib file. When it's unarchived it's sent -initWithCoder:
- (id)initWithCoder:(NSCoder*)coder
{
	active = FALSE;
	if((self = [super initWithCoder:coder]))
	{
		self = [self initGLES];
	}	
	return self;
}

-(id)initGLES
{
	// Get our backing layer
	CAEAGLLayer *eaglLayer = (CAEAGLLayer*) self.layer;
	
	// Configure it so that it is opaque, does not retain the contents of the backbuffer when displayed, and uses RGBA8888 color.
	eaglLayer.opaque = YES;
	eaglLayer.drawableProperties = [NSDictionary dictionaryWithObjectsAndKeys:
										[NSNumber numberWithBool:FALSE], kEAGLDrawablePropertyRetainedBacking,
										kEAGLColorFormatRGBA8, kEAGLDrawablePropertyColorFormat,
										nil];
	
	// Create our EAGLContext, and if successful make it current and create our framebuffer.
	context = [[EAGLContext alloc] initWithAPI:kEAGLRenderingAPIOpenGLES2];

	if(!context || ![EAGLContext setCurrentContext:context] || ![self createFramebuffer])
	{
		[self release];
		return nil;
	}
	
	// Default the animation interval to 1/60th of a second.
	animationInterval = 1.0 / 60.0;
	return self;
}

-(id<GLViewDelegate>)delegate
{
	return delegate;
}

// Update the delegate, and if it needs a -setupView: call, set our internal flag so that it will be called.
-(void)setDelegate:(id<GLViewDelegate>)d
{
	delegate = d;
	delegateSetup = ![delegate respondsToSelector:@selector(setupView:)];
}

@synthesize useCADisplayLink;

// If our view is resized, we'll be asked to layout subviews.
// This is the perfect opportunity to also update the framebuffer so that it is
// the same size as our display area.

-(void)layoutSubviews
{
	//printf("HERE\n");
	[EAGLContext setCurrentContext:context];
	[self destroyFramebuffer];
	[self createFramebuffer];
	[self drawView];
	[self drawView];

}

- (BOOL)createFramebuffer
{
	// Generate IDs for a framebuffer object and a color renderbuffer
	UIScreen* mainscr = [UIScreen mainScreen];
	printf("******** screen size %i, %i\n", (int)mainscr.currentMode.size.width, (int)mainscr.currentMode.size.height);
	float minPointSize = MIN(mainscr.bounds.size.width, mainscr.bounds.size.height);
	float minScreenSize = MIN(mainscr.currentMode.size.width, mainscr.currentMode.size.height);
	self.contentScaleFactor = minScreenSize / minPointSize;

	glGenFramebuffersOES(1, &viewFramebuffer);
	glGenRenderbuffersOES(1, &viewRenderbuffer);
	
	glBindFramebufferOES(GL_FRAMEBUFFER_OES, viewFramebuffer);
	glBindRenderbufferOES(GL_RENDERBUFFER_OES, viewRenderbuffer);
	// This call associates the storage for the current render buffer with the EAGLDrawable (our CAEAGLLayer)
	// allowing us to draw into a buffer that will later be rendered to screen whereever the layer is (which corresponds with our view).
	[context renderbufferStorage:GL_RENDERBUFFER_OES fromDrawable:(id<EAGLDrawable>)self.layer];
	glFramebufferRenderbufferOES(GL_FRAMEBUFFER_OES, GL_COLOR_ATTACHMENT0_OES, GL_RENDERBUFFER_OES, viewRenderbuffer);
	
	glGetRenderbufferParameterivOES(GL_RENDERBUFFER_OES, GL_RENDERBUFFER_WIDTH_OES, &backingWidth);
	glGetRenderbufferParameterivOES(GL_RENDERBUFFER_OES, GL_RENDERBUFFER_HEIGHT_OES, &backingHeight);
	
	// For this sample, we also need a depth buffer, so we'll create and attach one via another renderbuffer.
	glGenRenderbuffersOES(1, &depthRenderbuffer);
	glBindRenderbufferOES(GL_RENDERBUFFER_OES, depthRenderbuffer);
	glRenderbufferStorageOES(GL_RENDERBUFFER_OES, GL_DEPTH_COMPONENT16_OES, backingWidth, backingHeight);
	glFramebufferRenderbufferOES(GL_FRAMEBUFFER_OES, GL_DEPTH_ATTACHMENT_OES, GL_RENDERBUFFER_OES, depthRenderbuffer);

	if(glCheckFramebufferStatusOES(GL_FRAMEBUFFER_OES) != GL_FRAMEBUFFER_COMPLETE_OES)
	{
		NSLog(@"failed to make complete framebuffer object %x", glCheckFramebufferStatusOES(GL_FRAMEBUFFER_OES));
		return NO;
	}

	if (OS::get_singleton()) {
		OS::VideoMode vm;
		vm.fullscreen = true;
		vm.width = backingWidth;
		vm.height = backingHeight;
		vm.resizable = false;
		OS::get_singleton()->set_video_mode(vm);
		OSIPhone::get_singleton()->set_base_framebuffer(viewFramebuffer);
	};
	gl_view_base_fb = viewFramebuffer;

	return YES;
}

// Clean up any buffers we have allocated.
- (void)destroyFramebuffer
{
	glDeleteFramebuffersOES(1, &viewFramebuffer);
	viewFramebuffer = 0;
	glDeleteRenderbuffersOES(1, &viewRenderbuffer);
	viewRenderbuffer = 0;
	
	if(depthRenderbuffer)
	{
		glDeleteRenderbuffersOES(1, &depthRenderbuffer);
		depthRenderbuffer = 0;
	}
}

- (void)startAnimation
{
	if (active)
		return;
	active = TRUE;
	printf("start animation!\n");
	if (useCADisplayLink) {

		// Approximate frame rate
		// assumes device refreshes at 60 fps
		int frameInterval = (int) floor(animationInterval * 60.0f);

		displayLink = [CADisplayLink displayLinkWithTarget:self selector:@selector(drawView)];
		[displayLink setFrameInterval:frameInterval];

		// Setup DisplayLink in main thread
		[displayLink addToRunLoop:[NSRunLoop currentRunLoop] forMode:NSRunLoopCommonModes];
	}
	else {
		animationTimer = [NSTimer scheduledTimerWithTimeInterval:animationInterval target:self selector:@selector(drawView) userInfo:nil repeats:YES];
	}

	if (video_playing)
	{
		_unpause_video();
	}
}

- (void)stopAnimation
{
	if (!active)
		return;
	active = FALSE;
	printf("******** stop animation!\n");

	if (useCADisplayLink) {
		[displayLink invalidate];
		displayLink = nil;
	}
	else {
		[animationTimer invalidate];
		animationTimer = nil;
	}

	clear_touches();

	if (video_playing)
	{
		// save position
	}
}

- (void)setAnimationInterval:(NSTimeInterval)interval
{
	animationInterval = interval;
	if ( (useCADisplayLink && displayLink) || ( !useCADisplayLink && animationTimer ) ) {
		[self stopAnimation];
		[self startAnimation];
	}
}

// Updates the OpenGL view when the timer fires
- (void)drawView
{
	if (useCADisplayLink) {
		// Pause the CADisplayLink to avoid recursion
		[displayLink setPaused: YES];

		// Process all input events
		while(CFRunLoopRunInMode(kCFRunLoopDefaultMode, 0, TRUE) == kCFRunLoopRunHandledSource);

		// We are good to go, resume the CADisplayLink
		[displayLink setPaused: NO];
	}

	if (!active) {
		printf("draw view not active!\n");
		return;
	};

	// Make sure that you are drawing to the current context
	[EAGLContext setCurrentContext:context];
	
	// If our drawing delegate needs to have the view setup, then call -setupView: and flag that it won't need to be called again.
	if(!delegateSetup)
	{
		[delegate setupView:self];
		delegateSetup = YES;
	}
	
	glBindFramebufferOES(GL_FRAMEBUFFER_OES, viewFramebuffer);

	[delegate drawView:self];
	
	glBindRenderbufferOES(GL_RENDERBUFFER_OES, viewRenderbuffer);
	[context presentRenderbuffer:GL_RENDERBUFFER_OES];
	
#ifdef DEBUG_ENABLED
	GLenum err = glGetError();
	if(err)
		NSLog(@"%x error", err);
#endif
}

- (void)touchesBegan:(NSSet *)touches withEvent:(UIEvent *)event
{
	if (isKeyboardShown_) {
		[self handleTouchesAfterKeyboardShow];
		return;
	}

	NSArray* tlist = [[event allTouches] allObjects];
	for (unsigned int i=0; i< [tlist count]; i++) {
		
		if ( [touches containsObject:[tlist objectAtIndex:i]] ) {
			
			UITouch* touch = [tlist objectAtIndex:i];
			if (touch.phase != UITouchPhaseBegan)
				continue;
			int tid = get_touch_id(touch);
			ERR_FAIL_COND(tid == -1);
			CGPoint touchPoint = [touch locationInView:self];
			OSIPhone::get_singleton()->mouse_button(tid, touchPoint.x * self.contentScaleFactor, touchPoint.y * self.contentScaleFactor, true, touch.tapCount > 1, tid == 0);
		};
	};
}

- (void)touchesMoved:(NSSet *)touches withEvent:(UIEvent *)event
{
	if (isKeyboardShown_)
		return;

	NSArray* tlist = [[event allTouches] allObjects];
	for (unsigned int i=0; i< [tlist count]; i++) {
		
		if ( [touches containsObject:[tlist objectAtIndex:i]] ) {
			
			UITouch* touch = [tlist objectAtIndex:i];
			if (touch.phase != UITouchPhaseMoved)
				continue;
			int tid = get_touch_id(touch);
			ERR_FAIL_COND(tid == -1);
			int first = get_first_id(touch);
			CGPoint touchPoint = [touch locationInView:self];
			CGPoint prev_point = [touch previousLocationInView:self];
			OSIPhone::get_singleton()->mouse_move(tid, prev_point.x * self.contentScaleFactor, prev_point.y * self.contentScaleFactor, touchPoint.x * self.contentScaleFactor, touchPoint.y * self.contentScaleFactor, first == tid);
		};
	};

}

- (void)touchesEnded:(NSSet *)touches withEvent:(UIEvent *)event
{
	if (isKeyboardShown_)
		return;

	NSArray* tlist = [[event allTouches] allObjects];
	for (unsigned int i=0; i< [tlist count]; i++) {
		
		if ( [touches containsObject:[tlist objectAtIndex:i]] ) {
			
			UITouch* touch = [tlist objectAtIndex:i];
			if (touch.phase != UITouchPhaseEnded)
				continue;
			int tid = get_touch_id(touch);
			ERR_FAIL_COND(tid == -1);
			int rem = remove_touch(touch);
			CGPoint touchPoint = [touch locationInView:self];
			OSIPhone::get_singleton()->mouse_button(tid, touchPoint.x * self.contentScaleFactor, touchPoint.y * self.contentScaleFactor, false, false, rem == 0);
		};
	};
}

- (void)touchesCancelled:(NSSet *)touches withEvent:(UIEvent *)event {
	
	if (isKeyboardShown_)
		return;

	OSIPhone::get_singleton()->touches_cancelled();
	clear_touches();
};

-(void) handleTouchesAfterKeyboardShow
{
	NSArray *subviews = self.subviews;
	
	for(UIView* view in subviews)
	{
		if([view isKindOfClass:NSClassFromString(@"CustomUITextField")])
		{
			if ([view isFirstResponder])
			{
				[view resignFirstResponder];
				return;
			}
		}
	}
}

- (BOOL)canBecomeFirstResponder {
	if (nil != markedText_) {
		[markedText_ release];
	}
	return YES;
};


- (void)open_keyboard {
	//keyboard_text = p_existing;
	[self becomeFirstResponder];
};

- (void)hide_keyboard {
	//keyboard_text = p_existing;
	[self resignFirstResponder];
};

- (void)deleteBackward {
	if (keyboard_text.length())
		keyboard_text.erase(keyboard_text.length() - 1, 1);
	OSIPhone::get_singleton()->key(KEY_BACKSPACE, true);
};

- (BOOL)hasText {
	return keyboard_text.length() ? YES : NO;
};

- (void)insertText:(NSString *)p_text {
	String character;
	character.parse_utf8([p_text UTF8String]);
	keyboard_text = keyboard_text + character;
	OSIPhone::get_singleton()->key(character[0] == 10 ? KEY_ENTER : character[0] , true);
	printf("inserting text with character %i\n", character[0]);
};

- (void)audioRouteChangeListenerCallback:(NSNotification*)notification
{
	printf("*********** route changed!\n");
	NSDictionary *interuptionDict = notification.userInfo;

	NSInteger routeChangeReason = [[interuptionDict valueForKey:AVAudioSessionRouteChangeReasonKey] integerValue];

	switch (routeChangeReason) {

		case AVAudioSessionRouteChangeReasonNewDeviceAvailable:
			NSLog(@"AVAudioSessionRouteChangeReasonNewDeviceAvailable");
			NSLog(@"Headphone/Line plugged in");
			break;

		case AVAudioSessionRouteChangeReasonOldDeviceUnavailable:
			NSLog(@"AVAudioSessionRouteChangeReasonOldDeviceUnavailable");
			NSLog(@"Headphone/Line was pulled. Resuming video play....");
			if (_is_video_playing()) {

				dispatch_after(dispatch_time(DISPATCH_TIME_NOW, 0.5f * NSEC_PER_SEC), dispatch_get_main_queue(), ^{
							[_instance.avPlayer play]; // NOTE: change this line according your current player implementation
							NSLog(@"resumed play");
				});
			};
			break;

		case AVAudioSessionRouteChangeReasonCategoryChange:
			// called at start - also when other audio wants to play
			NSLog(@"AVAudioSessionRouteChangeReasonCategoryChange");
			break;
	}
}


// When created via code however, we get initWithFrame
-(id)initWithFrame:(CGRect)frame
{
	self = [super initWithFrame:frame];
	_instance = self;
	printf("after init super %p\n", self);
	if(self != nil)
	{
		self = [self initGLES];
		printf("after init gles %p\n", self);
	}
	init_touches();
	self. multipleTouchEnabled = YES;
	markedText_ = nil;
	originalRect_ = self.frame;
	self.keyboardShowNotification = nil;

	printf("******** adding observer for sound routing changes\n");
	[[NSNotificationCenter defaultCenter] addObserver:self selector:@selector(audioRouteChangeListenerCallback:)
												 name:AVAudioSessionRouteChangeNotification
											   object:nil];

	//self.autoresizesSubviews = YES;
	//[self setAutoresizingMask:UIViewAutoresizingFlexibleWidth | UIViewAutoresizingFlexibleWidth];

	return self;
}

// -(BOOL)automaticallyForwardAppearanceAndRotationMethodsToChildViewControllers {
//     return YES;
// }

// - (BOOL)shouldAutorotateToInterfaceOrientation:(UIInterfaceOrientation)interfaceOrientation{
//     return YES;
// }

// Stop animating and release resources when they are no longer needed.
- (void)dealloc
{
	[self stopAnimation];
	
	if([EAGLContext currentContext] == context)
	{
		[EAGLContext setCurrentContext:nil];
	}
	
	[context release];
	context = nil;

	self.keyboardShowNotification = NULL; // implicit release

	[super dealloc];
}

- (void)observeValueForKeyPath:(NSString *)keyPath ofObject:(id)object
                        change:(NSDictionary *)change context:(void *)context {

	if (object == _instance.avPlayerItem && [keyPath isEqualToString:@"status"]) {
        if (_instance.avPlayerItem.status == AVPlayerStatusFailed || _instance.avPlayer.status == AVPlayerStatusFailed) {
        	_stop_video();
            video_found_error = true;
        }

        if(_instance.avPlayer.status == AVPlayerStatusReadyToPlay && 
        	_instance.avPlayerItem.status == AVPlayerItemStatusReadyToPlay && 
        	CMTIME_COMPARE_INLINE(video_current_time, ==, kCMTimeZero)) {

        	//NSLog(@"time: %@", video_current_time);

    		[_instance.avPlayer seekToTime:video_current_time];
    		video_current_time = kCMTimeZero;
		}
    }

	if (object == _instance.avPlayer && [keyPath isEqualToString:@"rate"]) {
		NSLog(@"Player playback rate changed: %.5f", _instance.avPlayer.rate);
		if (_is_video_playing() && _instance.avPlayer.rate == 0.0 && !_instance.avPlayer.error) {
			dispatch_after(dispatch_time(DISPATCH_TIME_NOW, 0.5f * NSEC_PER_SEC), dispatch_get_main_queue(), ^{
						[_instance.avPlayer play]; // NOTE: change this line according your current player implementation
						NSLog(@"resumed play");
			});

			NSLog(@" . . . PAUSED (or just started)");
		}
	}
}

- (void)playerItemDidReachEnd:(NSNotification *)notification {
    _stop_video();
}

/*
- (void)moviePlayBackDidFinish:(NSNotification*)notification {
    

    NSNumber* reason = [[notification userInfo] objectForKey:MPMoviePlayerPlaybackDidFinishReasonUserInfoKey];
    switch ([reason intValue]) {
        case MPMovieFinishReasonPlaybackEnded:
            //NSLog(@"Playback Ended");
            break;
        case MPMovieFinishReasonPlaybackError:
            //NSLog(@"Playback Error");
            video_found_error = true;
            break;
        case MPMovieFinishReasonUserExited:
            //NSLog(@"User Exited");
            video_found_error = true;
            break;
        default:
        	//NSLog(@"Unsupported reason!");
        	break;
    }

    MPMoviePlayerController *player = [notification object];

    [[NSNotificationCenter defaultCenter]
      removeObserver:self
      name:MPMoviePlayerPlaybackDidFinishNotification
      object:player];

    [_instance.moviePlayerController stop];
    [_instance.moviePlayerController.view removeFromSuperview];

	//[[MPMusicPlayerController applicationMusicPlayer] setVolume: video_previous_volume];
	video_playing = false;
}
*/

#pragma mark UITextInputTrait protocol

-(UITextAutocapitalizationType) autocapitalizationType
{
	return UITextAutocapitalizationTypeNone;
}

#pragma mark -
#pragma mark UITextInput protocol

#pragma mark UITextInput - properties

@synthesize beginningOfDocument;
@synthesize endOfDocument;
@synthesize inputDelegate;
@synthesize markedTextRange;
@synthesize markedTextStyle;
// @synthesize selectedTextRange;	   // must implement
@synthesize tokenizer;

/* Text may have a selection, either zero-length (a caret) or ranged.  Editing operations are
 * always performed on the text from this selection.  nil corresponds to no selection. */
- (void)setSelectedTextRange:(UITextRange *)aSelectedTextRange;
{
	//printf("UITextRange:setSelectedTextRange");
}
- (UITextRange *)selectedTextRange;
{
	return [[[UITextRange alloc] init] autorelease];
}

#pragma mark UITextInput - Replacing and Returning Text

- (NSString *)textInRange:(UITextRange *)range;
{
	//printf("textInRange");
	return @"";
}
- (void)replaceRange:(UITextRange *)range withText:(NSString *)theText;
{
	//printf("replaceRange");
}

#pragma mark UITextInput - Working with Marked and Selected Text



/* If text can be selected, it can be marked. Marked text represents provisionally
 * inserted text that has yet to be confirmed by the user.  It requires unique visual
 * treatment in its display.  If there is any marked text, the selection, whether a
 * caret or an extended range, always resides within.
 *
 * Setting marked text either replaces the existing marked text or, if none is present,
 * inserts it from the current selection. */ 

- (void)setMarkedTextRange:(UITextRange *)markedTextRange;
{
	//printf("setMarkedTextRange");
}

- (UITextRange *)markedTextRange;
{
	//printf("markedTextRange");
	return nil; // Nil if no marked text.
}
- (void)setMarkedTextStyle:(NSDictionary *)markedTextStyle;
{
	//printf("setMarkedTextStyle");
	
}
- (NSDictionary *)markedTextStyle;
{
	//printf("markedTextStyle");
	return nil;
}
- (void)setMarkedText:(NSString *)markedText selectedRange:(NSRange)selectedRange;
{
	printf("setMarkedText");
	if (markedText == markedText_) {
		return;
	}
	if (nil != markedText_) {
		[markedText_ release];
	}
	markedText_ = markedText;
	[markedText_ retain];
}
- (void)unmarkText;
{
	printf("unmarkText");
	if (nil == markedText_)
	{
		return;
	}

	String character;
	if (!character.parse_utf8([markedText_ UTF8String])) {
		keyboard_text = keyboard_text + character;
		for (int idx=0;idx<character.length();idx++) {
			OSIPhone::get_singleton()->key(character[idx] == 10 ? KEY_ENTER : character[idx] , true);
			printf("inserting text with character %i\n", character[idx]);
		}
	}
	[markedText_ release];
	markedText_ = nil;
}

#pragma mark Methods for creating ranges and positions.

- (UITextRange *)textRangeFromPosition:(UITextPosition *)fromPosition toPosition:(UITextPosition *)toPosition;
{
	//printf("textRangeFromPosition");
	return nil;
}
- (UITextPosition *)positionFromPosition:(UITextPosition *)position offset:(NSInteger)offset;
{
	//printf("positionFromPosition");
	return nil;
}
- (UITextPosition *)positionFromPosition:(UITextPosition *)position inDirection:(UITextLayoutDirection)direction offset:(NSInteger)offset;
{
	//printf("positionFromPosition");
	return nil;
}

/* Simple evaluation of positions */
- (NSComparisonResult)comparePosition:(UITextPosition *)position toPosition:(UITextPosition *)other;
{
	//printf("comparePosition");
	return (NSComparisonResult)0;
}
- (NSInteger)offsetFromPosition:(UITextPosition *)from toPosition:(UITextPosition *)toPosition;
{
	//printf("offsetFromPosition");
	return 0;
}

- (UITextPosition *)positionWithinRange:(UITextRange *)range farthestInDirection:(UITextLayoutDirection)direction;
{
	//printf("positionWithinRange");
	return nil;
}
- (UITextRange *)characterRangeByExtendingPosition:(UITextPosition *)position inDirection:(UITextLayoutDirection)direction;
{
	//printf("characterRangeByExtendingPosition");
	return nil;
}

#pragma mark Writing direction

- (UITextWritingDirection)baseWritingDirectionForPosition:(UITextPosition *)position inDirection:(UITextStorageDirection)direction;
{
	//printf("baseWritingDirectionForPosition");
	return UITextWritingDirectionNatural;
}
- (void)setBaseWritingDirection:(UITextWritingDirection)writingDirection forRange:(UITextRange *)range;
{
	//printf("setBaseWritingDirection");
}

#pragma mark Geometry

/* Geometry used to provide, for example, a correction rect. */
- (CGRect)firstRectForRange:(UITextRange *)range;
{
	//printf("firstRectForRange");
	return CGRectNull;
}
- (CGRect)caretRectForPosition:(UITextPosition *)position;
{
	printf("caretRectForPosition");
	return caretRect_;
}

#pragma mark Hit testing

/* JS - Find the closest position to a given point */
- (UITextPosition *)closestPositionToPoint:(CGPoint)point;
{
	//printf("closestPositionToPoint");
	return nil;
}
- (UITextPosition *)closestPositionToPoint:(CGPoint)point withinRange:(UITextRange *)range;
{
	//printf("closestPositionToPoint");
	return nil;
}
- (UITextRange *)characterRangeAtPoint:(CGPoint)point;
{
	//printf("characterRangeAtPoint");
	return nil;
}

- (NSArray *)selectionRectsForRange:(UITextRange *)range
{
	//printf("selectionRectsForRange");
	return nil;
}

#pragma mark -

#pragma mark UIKeyboard notification

- (void)onUIKeyboardNotification:(NSNotification *)notif;
{
	NSString * type = notif.name;
	
	NSDictionary* info = [notif userInfo];
	CGRect begin = [self convertRect: 
					[[info objectForKey:UIKeyboardFrameBeginUserInfoKey] CGRectValue]
							fromView:self];
	CGRect end = [self convertRect: 
				  [[info objectForKey:UIKeyboardFrameEndUserInfoKey] CGRectValue]
						  fromView:self];
	double aniDuration = [[info objectForKey:UIKeyboardAnimationDurationUserInfoKey] doubleValue];
	
	CGSize viewSize = self.frame.size;
	CGFloat tmp;
	
	switch ([[UIApplication sharedApplication] statusBarOrientation])
	{
		case UIInterfaceOrientationPortrait:
			begin.origin.y = viewSize.height - begin.origin.y - begin.size.height;
			end.origin.y = viewSize.height - end.origin.y - end.size.height;
			break;
			
		case UIInterfaceOrientationPortraitUpsideDown:
			begin.origin.x = viewSize.width - (begin.origin.x + begin.size.width);
			end.origin.x = viewSize.width - (end.origin.x + end.size.width);
			break;
			
		case UIInterfaceOrientationLandscapeLeft:
			tmp = begin.size.width;
			begin.size.width = begin.size.height;
			begin.size.height = tmp;
			tmp = end.size.width;
			end.size.width = end.size.height;
			end.size.height = tmp;
			tmp = viewSize.width;
			viewSize.width = viewSize.height;
			viewSize.height = tmp;
			
			tmp = begin.origin.x;
			begin.origin.x = begin.origin.y;
			begin.origin.y = viewSize.height - tmp - begin.size.height;
			tmp = end.origin.x;
			end.origin.x = end.origin.y;
			end.origin.y = viewSize.height - tmp - end.size.height;
			break;
			
		case UIInterfaceOrientationLandscapeRight:
			tmp = begin.size.width;
			begin.size.width = begin.size.height;
			begin.size.height = tmp;
			tmp = end.size.width;
			end.size.width = end.size.height;
			end.size.height = tmp;
			tmp = viewSize.width;
			viewSize.width = viewSize.height;
			viewSize.height = tmp;
			
			tmp = begin.origin.x;
			begin.origin.x = begin.origin.y;
			begin.origin.y = tmp;
			tmp = end.origin.x;
			end.origin.x = end.origin.y;
			end.origin.y = tmp;
			break;
			
		default:
			break;
	}
	
	float scaleX = 1;//cocos2d::CCEGLView::sharedOpenGLView()->getScaleX();
	float scaleY = 1;//cocos2d::CCEGLView::sharedOpenGLView()->getScaleY();
	
	
	if (self.contentScaleFactor == 2.0f)
	{
		// Convert to pixel coordinate
		
		begin = CGRectApplyAffineTransform(begin, CGAffineTransformScale(CGAffineTransformIdentity, 2.0f, 2.0f));
		end = CGRectApplyAffineTransform(end, CGAffineTransformScale(CGAffineTransformIdentity, 2.0f, 2.0f));
	}
	
	float offestY = 1;//cocos2d::CCEGLView::sharedOpenGLView()->getViewPortRect().origin.y;
	printf("offestY = %f", offestY);
	if (offestY < 0.0f)
	{
		begin.origin.y += offestY;
		begin.size.height -= offestY;
		end.size.height -= offestY;
	}
	
	// Convert to desigin coordinate
	begin = CGRectApplyAffineTransform(begin, CGAffineTransformScale(CGAffineTransformIdentity, 1.0f/scaleX, 1.0f/scaleY));
	end = CGRectApplyAffineTransform(end, CGAffineTransformScale(CGAffineTransformIdentity, 1.0f/scaleX, 1.0f/scaleY));

	
//	cocos2d::CCIMEKeyboardNotificationInfo notiInfo;
//	notiInfo.begin = cocos2d::CCRect(begin.origin.x,
//									 begin.origin.y,
//									 begin.size.width,
//									 begin.size.height);
//	notiInfo.end = cocos2d::CCRect(end.origin.x,
//								   end.origin.y,
//								   end.size.width,
//								   end.size.height);
//	notiInfo.duration = (float)aniDuration;
	
//	cocos2d::CCIMEDispatcher* dispatcher = cocos2d::CCIMEDispatcher::sharedDispatcher();
	if (UIKeyboardWillShowNotification == type) 
	{
//		self.keyboardShowNotification = notif; // implicit copy
//		dispatcher->dispatchKeyboardWillShow(notiInfo);
	}
	else if (UIKeyboardDidShowNotification == type)
	{
//		//CGSize screenSize = self.window.screen.bounds.size;
//		dispatcher->dispatchKeyboardDidShow(notiInfo);
		caretRect_ = end;
		caretRect_.origin.y = viewSize.height - (caretRect_.origin.y + caretRect_.size.height + [UIFont smallSystemFontSize]);
		caretRect_.size.height = 0;
		isKeyboardShown_ = YES;
	}
	else if (UIKeyboardWillHideNotification == type)
	{
//		dispatcher->dispatchKeyboardWillHide(notiInfo);
	}
	else if (UIKeyboardDidHideNotification == type)
	{
		caretRect_ = CGRectZero;
//		dispatcher->dispatchKeyboardDidHide(notiInfo);
		isKeyboardShown_ = NO;
	}
}

-(void) doAnimationWhenKeyboardMoveWithDuration:(float)duration distance:(float)dis
{
	[UIView beginAnimations:nil context:NULL];
	[UIView setAnimationDelegate:self];
	[UIView setAnimationDuration:duration];
	[UIView setAnimationBeginsFromCurrentState:YES];
	
	//NSLog(@"[animation] dis = %f, scale = %f \n", dis, cocos2d::CCEGLView::sharedOpenGLView()->getScaleY());
	
	if (dis < 0.0f) dis = 0.0f;

	//dis *= cocos2d::CCEGLView::sharedOpenGLView()->getScaleY();
	
	if (self.contentScaleFactor == 2.0f)
	{
		dis /= 2.0f;
	}
	
	switch ([[UIApplication sharedApplication] statusBarOrientation])
	{
		case UIInterfaceOrientationPortrait:
			self.frame = CGRectMake(originalRect_.origin.x, originalRect_.origin.y - dis, originalRect_.size.width, originalRect_.size.height);
			break;
			
		case UIInterfaceOrientationPortraitUpsideDown:
			self.frame = CGRectMake(originalRect_.origin.x, originalRect_.origin.y + dis, originalRect_.size.width, originalRect_.size.height);
			break;
			
		case UIInterfaceOrientationLandscapeLeft:
			self.frame = CGRectMake(originalRect_.origin.x - dis, originalRect_.origin.y , originalRect_.size.width, originalRect_.size.height);
			break;
			
		case UIInterfaceOrientationLandscapeRight:
			self.frame = CGRectMake(originalRect_.origin.x + dis, originalRect_.origin.y , originalRect_.size.width, originalRect_.size.height);
			break;
			
		default:
			break;
	}
	
	[UIView commitAnimations];
}

-(void) doAnimationWhenAnotherEditBeClicked
{
	if (self.keyboardShowNotification != nil)
	{
		[[NSNotificationCenter defaultCenter]postNotification:self.keyboardShowNotification];
	}
}

@end
