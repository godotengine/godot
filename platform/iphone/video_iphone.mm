#import "video_iphone.h"
#import "core/project_settings.h"
#import "core/ustring.h"

static MediaView *_instance = nil;

bool _play_video(String p_path, float p_volume, String p_audio_track, String p_subtitle_track) {
	p_path = ProjectSettings::get_singleton()->globalize_path(p_path);

	NSString *file_path = [[[NSString alloc] initWithUTF8String:p_path.utf8().get_data()] autorelease];

	_instance.avAsset = [AVAsset assetWithURL:[NSURL fileURLWithPath:file_path]];
	_instance.avPlayerItem = [[AVPlayerItem alloc] initWithAsset:_instance.avAsset];
	[_instance.avPlayerItem addObserver:_instance forKeyPath:@"status" options:0 context:nil];

	_instance.avPlayer = [[AVPlayer alloc] initWithPlayerItem:_instance.avPlayerItem];
	[_instance.avPlayer addObserver:_instance forKeyPath:@"status" options:0 context:nil];
	[_instance.avPlayer addObserver:_instance forKeyPath:@"rate" options:NSKeyValueObservingOptionNew context:0];
	[[NSNotificationCenter defaultCenter]
			addObserver:_instance
			   selector:@selector(playerItemDidReachEnd:)
				   name:AVPlayerItemDidPlayToEndTimeNotification
				 object:[_instance.avPlayer currentItem]];

	_instance.avPlayerLayer = [AVPlayerLayer playerLayerWithPlayer:_instance.avPlayer];
	_instance.avPlayerLayer.frame = _instance.bounds;
	[_instance.layer addSublayer:_instance.avPlayerLayer];

	[_instance.avPlayer play];

	// Find audio options
	AVMediaSelectionGroup *audioGroup = [_instance.avAsset mediaSelectionGroupForMediaCharacteristic:AVMediaCharacteristicAudible];
	NSMutableArray *allAudioParams = [NSMutableArray array];
	for (id track in audioGroup.options) {
		NSString *language = [[track locale] localeIdentifier];
		NSLog(@"audible track - subtitle lang: %@", language);

		if ([language isEqualToString:[NSString stringWithUTF8String:p_audio_track.utf8()]]) {
			AVMutableAudioMixInputParameters *audioInputParams = [AVMutableAudioMixInputParameters audioMixInputParameters];
			[audioInputParams setVolume:p_volume atTime:kCMTimeZero];
			[audioInputParams setTrackID:[track trackID]];
			[allAudioParams addObject:audioInputParams];

			AVMutableAudioMix *audioMix = [AVMutableAudioMix audioMix];
			[audioMix setInputParameters:allAudioParams];

			[_instance.avPlayer.currentItem selectMediaOption:track inMediaSelectionGroup:audioGroup];
			[_instance.avPlayer.currentItem setAudioMix:audioMix];

			break;
		}
	}

	// Find captioning options
	AVMediaSelectionGroup *subtitlesGroup = [_instance.avAsset mediaSelectionGroupForMediaCharacteristic:AVMediaCharacteristicLegible];
	NSArray *legibleTracks = [AVMediaSelectionGroup mediaSelectionOptionsFromArray:subtitlesGroup.options
													   withoutMediaCharacteristics:@[ AVMediaCharacteristicContainsOnlyForcedSubtitles ]];
	for (id track in legibleTracks) {
		NSString *language = [[track locale] localeIdentifier];
		NSLog(@"legible track - subtitle lang: %@", language);

		if ([language isEqualToString:[NSString stringWithUTF8String:p_subtitle_track.utf8()]]) {
			[_instance.avPlayer.currentItem selectMediaOption:track inMediaSelectionGroup:subtitlesGroup];
			break;
		}
	}

	return true;
}

bool _is_video_playing() {
	return (_instance.avPlayer.rate > 0 && !_instance.avPlayer.error);
}

void _pause_video() {
	[_instance.avPlayer pause];
}

void _focus_out_video() {
	printf("focus out pausing video\n");
	_pause_video();
};

void _unpause_video() {
	[_instance.avPlayer play];
};

void _stop_video() {
	_pause_video();
	[_instance.avPlayerLayer removeFromSuperlayer];
	_instance.avPlayer = nil;
}

@interface MediaView ()

@end

@implementation MediaView

- (id)initWithFrame:(CGRect)frame {
	if (self = [super initWithFrame:frame]) {
		[[NSNotificationCenter defaultCenter]
				addObserver:self
				   selector:@selector(audioRouteChangeListenerCallback:)
					   name:AVAudioSessionRouteChangeNotification
					 object:nil];
	}
	return self;
}

- (void)audioRouteChangeListenerCallback:(NSNotification *)notification {
	NSNumber *reason = notification.userInfo[AVAudioSessionRouteChangeReasonKey];
	switch (reason.integerValue) {
		case AVAudioSessionRouteChangeReasonNewDeviceAvailable: {
			NSLog(@"AVAudioSessionRouteChangeReasonNewDeviceAvailable - Headphone/Line plugged in");
			break;
		}
		case AVAudioSessionRouteChangeReasonOldDeviceUnavailable: {
			NSLog(@"AVAudioSessionRouteChangeReasonOldDeviceUnavailable - Headphone / Line was pulled. Resuming video play in ~500ms....");
			if (_is_video_playing()) {

				dispatch_after(dispatch_time(DISPATCH_TIME_NOW, 0.5f * NSEC_PER_SEC), dispatch_get_main_queue(), ^{
					[self.avPlayer play]; // NOTE: change this line according your current player implementation
					NSLog(@"resumed play");
				});
			}
			break;
		}
		case AVAudioSessionRouteChangeReasonCategoryChange: {
			// called at start - also when other audio wants to play
			NSLog(@"AVAudioSessionRouteChangeReasonCategoryChange");
			break;
		}
	}
}

- (void)observeValueForKeyPath:(NSString *)keyPath ofObject:(id)object change:(NSDictionary *)change context:(void *)context {
	if (object == self.avPlayerItem && [keyPath isEqualToString:@"status"]) {
		if (self.avPlayerItem.status == AVPlayerStatusFailed || self.avPlayer.status == AVPlayerStatusFailed) {
			_stop_video();
		}

		if (self.avPlayerItem.status == AVPlayerItemStatusReadyToPlay) {
			[self.avPlayer seekToTime:kCMTimeZero];
		}
	}
}

- (void)playerItemDidReachEnd:(NSNotification *)notification {
	_stop_video();
}

@end