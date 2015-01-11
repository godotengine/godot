/*************************************************************************/
/*  gl_view.mm                                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                 */
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

/*
bool _play_video(String p_path, float p_volume) {
	
	float player_volume = p_volume * AudioServer::get_singleton()->get_singleton()->get_stream_global_volume_scale();
	video_previous_volume = [[MPMusicPlayerController applicationMusicPlayer] volume];

	//[[MPMusicPlayerController applicationMusicPlayer] setVolume: player_volume];

	p_path = Globals::get_singleton()->globalize_path(p_path);

	NSString* file_path = [[[NSString alloc] initWithUTF8String:p_path.utf8().get_data()] autorelease];
	NSURL *file_url = [NSURL fileURLWithPath:file_path];
		
	_instance.moviePlayerController = [[MPMoviePlayerController alloc] initWithContentURL:file_url];
	_instance.moviePlayerController.controlStyle = MPMovieControlStyleNone;
	[_instance.moviePlayerController setScalingMode:MPMovieScalingModeAspectFit];
	//[_instance.moviePlayerController setScalingMode:MPMovieScalingModeAspectFill];
	
	[[NSNotificationCenter defaultCenter] addObserver:_instance
                   selector:@selector(moviePlayBackDidFinish:)
                   name:MPMoviePlayerPlaybackDidFinishNotification
                   object:_instance.moviePlayerController];
	
	[_instance.moviePlayerController.view setFrame:_instance.bounds];
	_instance.moviePlayerController.view.userInteractionEnabled = NO;
	[_instance addSubview:_instance.moviePlayerController.view];
	[_instance.moviePlayerController play];

	video_playing = true;

	return true;
}
*/

bool _play_video(String p_path, float p_volume, String p_audio_track, String p_subtitle_track) {
	p_path = Globals::get_singleton()->globalize_path(p_path);

	NSString* file_path = [[[NSString alloc] initWithUTF8String:p_path.utf8().get_data()] autorelease];
	//NSURL *file_url = [NSURL fileURLWithPath:file_path];

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
	//NSInteger playback_state = _instance.moviePlayerController.playbackState;
	//return video_playing || _instance.moviePlayerController.playbackState == MPMoviePlaybackStatePlaying;
	//if (video_found_error)
	//	return false;
	//return (_instance.moviePlayerController.playbackState == MPMoviePlaybackStatePlaying);

	return video_playing || (_instance.avPlayer.rate > 0 && !_instance.avPlayer.error);
}

void _pause_video() {
	//[_instance.moviePlayerController pause];
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

	//video_current_time = kCMTimeZero;
};

void _stop_video() {
	//[_instance.moviePlayerController stop];
	//[_instance.moviePlayerController.view removeFromSuperview];
	//[[MPMusicPlayerController applicationMusicPlayer] setVolume: video_previous_volume];

	[_instance.avPlayer pause];
	[_instance.avPlayerLayer removeFromSuperlayer];
	_instance.avPlayer = nil;
	video_playing = false;
}

@implementation GLView

@synthesize animationInterval;

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

// If our view is resized, we'll be asked to layout subviews.
// This is the perfect opportunity to also update the framebuffer so that it is
// the same size as our display area.

-(void)layoutSubviews
{
	printf("HERE\n");
	[EAGLContext setCurrentContext:context];
	[self destroyFramebuffer];
	[self createFramebuffer];
	[self drawView];
}

- (BOOL)createFramebuffer
{
	// Generate IDs for a framebuffer object and a color renderbuffer
	UIScreen* mainscr = [UIScreen mainScreen];
	printf("******** screen size %i, %i\n", (int)mainscr.currentMode.size.width, (int)mainscr.currentMode.size.height);
	if (mainscr.currentMode.size.width == 640 || mainscr.currentMode.size.width == 960) // modern iphone, can go to 640x960
		self.contentScaleFactor = 2.0;

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
	animationTimer = [NSTimer scheduledTimerWithTimeInterval:animationInterval target:self selector:@selector(drawView) userInfo:nil repeats:YES];

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
	[animationTimer invalidate];
	animationTimer = nil;
	clear_touches();

	if (video_playing)
	{
		// save position
	}
}

- (void)setAnimationInterval:(NSTimeInterval)interval
{
	animationInterval = interval;
	
	if(animationTimer)
	{
		[self stopAnimation];
		[self startAnimation];
	}
}

// Updates the OpenGL view when the timer fires
- (void)drawView
{
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
	
	OSIPhone::get_singleton()->touches_cancelled();
	clear_touches();
};

- (BOOL)canBecomeFirstResponder {
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

@end
