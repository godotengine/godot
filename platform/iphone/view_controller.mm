/*************************************************************************/
/*  view_controller.mm                                                   */
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

#import "view_controller.h"
#include "core/project_settings.h"
#include "display_server_iphone.h"
#import "godot_view.h"
#import "godot_view_renderer.h"
#include "os_iphone.h"

#import <GameController/GameController.h>

@interface ViewController ()

@property(strong, nonatomic) GodotViewRenderer *renderer;

// TODO: separate view to handle video
// AVPlayer-related properties
@property(strong, nonatomic) AVAsset *avAsset;
@property(strong, nonatomic) AVPlayerItem *avPlayerItem;
@property(strong, nonatomic) AVPlayer *avPlayer;
@property(strong, nonatomic) AVPlayerLayer *avPlayerLayer;
@property(assign, nonatomic) CMTime videoCurrentTime;
@property(assign, nonatomic) BOOL isVideoCurrentlyPlaying;
@property(assign, nonatomic) BOOL videoHasFoundError;

@end

@implementation ViewController

- (GodotView *)godotView {
	return (GodotView *)self.view;
}

- (void)loadView {
	GodotView *view = [[GodotView alloc] init];
	GodotViewRenderer *renderer = [[GodotViewRenderer alloc] init];

	self.renderer = renderer;
	self.view = view;

	view.renderer = self.renderer;

	[renderer release];
	[view release];
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
	self.isVideoCurrentlyPlaying = NO;
	self.videoCurrentTime = kCMTimeZero;
	self.videoHasFoundError = false;
}

- (void)didReceiveMemoryWarning {
	[super didReceiveMemoryWarning];
	printf("*********** did receive memory warning!\n");
}

- (void)viewDidLoad {
	[super viewDidLoad];

	[self observeKeyboard];
	[self observeAudio];

	if (@available(iOS 11.0, *)) {
		[self setNeedsUpdateOfScreenEdgesDeferringSystemGestures];
	}
}

- (void)observeKeyboard {
	printf("******** adding observer for keyboard show/hide\n");
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

- (void)observeAudio {
	printf("******** adding observer for sound routing changes\n");
	[[NSNotificationCenter defaultCenter]
			addObserver:self
			   selector:@selector(audioRouteChangeListenerCallback:)
				   name:AVAudioSessionRouteChangeNotification
				 object:nil];
}

- (void)observeValueForKeyPath:(NSString *)keyPath ofObject:(id)object change:(NSDictionary *)change context:(void *)context {
	if (object == self.avPlayerItem && [keyPath isEqualToString:@"status"]) {
		[self handleVideoOrPlayerStatus];
	}

	if (object == self.avPlayer && [keyPath isEqualToString:@"rate"]) {
		[self handleVideoPlayRate];
	}
}

- (void)dealloc {
	[self stopVideo];

	self.renderer = nil;

	[[NSNotificationCenter defaultCenter] removeObserver:self];

	[super dealloc];
}

// MARK: Orientation

- (UIRectEdge)preferredScreenEdgesDeferringSystemGestures {
	return UIRectEdgeAll;
}

- (BOOL)shouldAutorotate {
	if (!DisplayServerIPhone::get_singleton()) {
		return NO;
	}

	switch (DisplayServerIPhone::get_singleton()->screen_get_orientation(DisplayServer::SCREEN_OF_MAIN_WINDOW)) {
		case DisplayServer::SCREEN_SENSOR:
		case DisplayServer::SCREEN_SENSOR_LANDSCAPE:
		case DisplayServer::SCREEN_SENSOR_PORTRAIT:
			return YES;
		default:
			return NO;
	}
}

- (UIInterfaceOrientationMask)supportedInterfaceOrientations {
	if (!DisplayServerIPhone::get_singleton()) {
		return UIInterfaceOrientationMaskAll;
	}

	switch (DisplayServerIPhone::get_singleton()->screen_get_orientation(DisplayServer::SCREEN_OF_MAIN_WINDOW)) {
		case DisplayServer::SCREEN_PORTRAIT:
			return UIInterfaceOrientationMaskPortrait;
		case DisplayServer::SCREEN_REVERSE_LANDSCAPE:
			return UIInterfaceOrientationMaskLandscapeRight;
		case DisplayServer::SCREEN_REVERSE_PORTRAIT:
			return UIInterfaceOrientationMaskPortraitUpsideDown;
		case DisplayServer::SCREEN_SENSOR_LANDSCAPE:
			return UIInterfaceOrientationMaskLandscape;
		case DisplayServer::SCREEN_SENSOR_PORTRAIT:
			return UIInterfaceOrientationMaskPortrait | UIInterfaceOrientationMaskPortraitUpsideDown;
		case DisplayServer::SCREEN_SENSOR:
			return UIInterfaceOrientationMaskAll;
		case DisplayServer::SCREEN_LANDSCAPE:
			return UIInterfaceOrientationMaskLandscapeLeft;
	}
};

- (BOOL)prefersStatusBarHidden {
	return YES;
}

- (BOOL)prefersHomeIndicatorAutoHidden {
	if (GLOBAL_GET("display/window/ios/hide_home_indicator")) {
		return YES;
	} else {
		return NO;
	}
}

// MARK: Keyboard

- (void)keyboardOnScreen:(NSNotification *)notification {
	NSDictionary *info = notification.userInfo;
	NSValue *value = info[UIKeyboardFrameEndUserInfoKey];

	CGRect rawFrame = [value CGRectValue];
	CGRect keyboardFrame = [self.view convertRect:rawFrame fromView:nil];

	if (DisplayServerIPhone::get_singleton()) {
		DisplayServerIPhone::get_singleton()->virtual_keyboard_set_height(keyboardFrame.size.height);
	}
}

- (void)keyboardHidden:(NSNotification *)notification {
	if (DisplayServerIPhone::get_singleton()) {
		DisplayServerIPhone::get_singleton()->virtual_keyboard_set_height(0);
	}
}

// MARK: Audio

- (void)audioRouteChangeListenerCallback:(NSNotification *)notification {
	printf("*********** route changed!\n");
	NSDictionary *interuptionDict = notification.userInfo;

	NSInteger routeChangeReason = [[interuptionDict valueForKey:AVAudioSessionRouteChangeReasonKey] integerValue];

	switch (routeChangeReason) {
		case AVAudioSessionRouteChangeReasonNewDeviceAvailable: {
			NSLog(@"AVAudioSessionRouteChangeReasonNewDeviceAvailable");
			NSLog(@"Headphone/Line plugged in");
		} break;
		case AVAudioSessionRouteChangeReasonOldDeviceUnavailable: {
			NSLog(@"AVAudioSessionRouteChangeReasonOldDeviceUnavailable");
			NSLog(@"Headphone/Line was pulled. Resuming video play....");
			if ([self isVideoPlaying]) {
				dispatch_after(dispatch_time(DISPATCH_TIME_NOW, 0.5f * NSEC_PER_SEC), dispatch_get_main_queue(), ^{
					[self.avPlayer play]; // NOTE: change this line according your current player implementation
					NSLog(@"resumed play");
				});
			}
		} break;
		case AVAudioSessionRouteChangeReasonCategoryChange: {
			// called at start - also when other audio wants to play
			NSLog(@"AVAudioSessionRouteChangeReasonCategoryChange");
		} break;
	}
}

// MARK: Native Video Player

- (void)handleVideoOrPlayerStatus {
	if (self.avPlayerItem.status == AVPlayerItemStatusFailed || self.avPlayer.status == AVPlayerStatusFailed) {
		[self stopVideo];
		self.videoHasFoundError = true;
	}

	if (self.avPlayer.status == AVPlayerStatusReadyToPlay && self.avPlayerItem.status == AVPlayerItemStatusReadyToPlay && CMTimeCompare(self.videoCurrentTime, kCMTimeZero) == 0) {
		//        NSLog(@"time: %@", self.video_current_time);
		[self.avPlayer seekToTime:self.videoCurrentTime];
		self.videoCurrentTime = kCMTimeZero;
	}
}

- (void)handleVideoPlayRate {
	NSLog(@"Player playback rate changed: %.5f", self.avPlayer.rate);
	if ([self isVideoPlaying] && self.avPlayer.rate == 0.0 && !self.avPlayer.error) {
		dispatch_after(dispatch_time(DISPATCH_TIME_NOW, 0.5f * NSEC_PER_SEC), dispatch_get_main_queue(), ^{
			[self.avPlayer play]; // NOTE: change this line according your current player implementation
			NSLog(@"resumed play");
		});

		NSLog(@" . . . PAUSED (or just started)");
	}
}

- (BOOL)playVideoAtPath:(NSString *)filePath volume:(float)videoVolume audio:(NSString *)audioTrack subtitle:(NSString *)subtitleTrack {
	self.avAsset = [AVAsset assetWithURL:[NSURL fileURLWithPath:filePath]];

	self.avPlayerItem = [AVPlayerItem playerItemWithAsset:self.avAsset];
	[self.avPlayerItem addObserver:self forKeyPath:@"status" options:0 context:nil];

	self.avPlayer = [AVPlayer playerWithPlayerItem:self.avPlayerItem];
	self.avPlayerLayer = [AVPlayerLayer playerLayerWithPlayer:self.avPlayer];

	[self.avPlayer addObserver:self forKeyPath:@"status" options:0 context:nil];
	[[NSNotificationCenter defaultCenter]
			addObserver:self
			   selector:@selector(playerItemDidReachEnd:)
				   name:AVPlayerItemDidPlayToEndTimeNotification
				 object:[self.avPlayer currentItem]];

	[self.avPlayer addObserver:self forKeyPath:@"rate" options:NSKeyValueObservingOptionNew context:0];

	[self.avPlayerLayer setFrame:self.view.bounds];
	[self.view.layer addSublayer:self.avPlayerLayer];
	[self.avPlayer play];

	AVMediaSelectionGroup *audioGroup = [self.avAsset mediaSelectionGroupForMediaCharacteristic:AVMediaCharacteristicAudible];

	NSMutableArray *allAudioParams = [NSMutableArray array];
	for (id track in audioGroup.options) {
		NSString *language = [[track locale] localeIdentifier];
		NSLog(@"subtitle lang: %@", language);

		if ([language isEqualToString:audioTrack]) {
			AVMutableAudioMixInputParameters *audioInputParams = [AVMutableAudioMixInputParameters audioMixInputParameters];
			[audioInputParams setVolume:videoVolume atTime:kCMTimeZero];
			[audioInputParams setTrackID:[track trackID]];
			[allAudioParams addObject:audioInputParams];

			AVMutableAudioMix *audioMix = [AVMutableAudioMix audioMix];
			[audioMix setInputParameters:allAudioParams];

			[self.avPlayer.currentItem selectMediaOption:track inMediaSelectionGroup:audioGroup];
			[self.avPlayer.currentItem setAudioMix:audioMix];

			break;
		}
	}

	AVMediaSelectionGroup *subtitlesGroup = [self.avAsset mediaSelectionGroupForMediaCharacteristic:AVMediaCharacteristicLegible];
	NSArray *useableTracks = [AVMediaSelectionGroup mediaSelectionOptionsFromArray:subtitlesGroup.options withoutMediaCharacteristics:[NSArray arrayWithObject:AVMediaCharacteristicContainsOnlyForcedSubtitles]];

	for (id track in useableTracks) {
		NSString *language = [[track locale] localeIdentifier];
		NSLog(@"subtitle lang: %@", language);

		if ([language isEqualToString:subtitleTrack]) {
			[self.avPlayer.currentItem selectMediaOption:track inMediaSelectionGroup:subtitlesGroup];
			break;
		}
	}

	self.isVideoCurrentlyPlaying = YES;

	return true;
}

- (BOOL)isVideoPlaying {
	if (self.avPlayer.error) {
		printf("Error during playback\n");
	}
	return (self.avPlayer.rate > 0 && !self.avPlayer.error);
}

- (void)pauseVideo {
	self.videoCurrentTime = self.avPlayer.currentTime;
	[self.avPlayer pause];
	self.isVideoCurrentlyPlaying = NO;
}

- (void)unpauseVideo {
	[self.avPlayer play];
	self.isVideoCurrentlyPlaying = YES;
}

- (void)playerItemDidReachEnd:(NSNotification *)notification {
	[self stopVideo];
}

- (void)stopVideo {
	[self.avPlayer pause];
	[self.avPlayerLayer removeFromSuperlayer];
	self.avPlayerLayer = nil;

	if (self.avPlayerItem) {
		[self.avPlayerItem removeObserver:self forKeyPath:@"status"];
		self.avPlayerItem = nil;
	}

	if (self.avPlayer) {
		[self.avPlayer removeObserver:self forKeyPath:@"status"];
		self.avPlayer = nil;
	}

	self.avAsset = nil;

	self.isVideoCurrentlyPlaying = NO;
}

// MARK: Delegates

#ifdef GAME_CENTER_ENABLED
- (void)gameCenterViewControllerDidFinish:(GKGameCenterViewController *)gameCenterViewController {
	//[gameCenterViewController dismissViewControllerAnimated:YES completion:^{GameCenter::get_singleton()->game_center_closed();}];//version for signaling when overlay is completely gone
	GameCenter::get_singleton()->game_center_closed();
	[gameCenterViewController dismissViewControllerAnimated:YES completion:nil];
}
#endif

@end
