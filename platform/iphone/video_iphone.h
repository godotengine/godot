#import <AVFoundation/AVFoundation.h>
#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>

@interface MediaView : UIView

@property(strong, nonatomic) AVAsset *avAsset;
@property(strong, nonatomic) AVPlayerItem *avPlayerItem;
@property(strong, nonatomic) AVPlayer *avPlayer;
@property(strong, nonatomic) AVPlayerLayer *avPlayerLayer;

@end