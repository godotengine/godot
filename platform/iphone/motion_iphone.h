#import <CoreMotion/CoreMotion.h>

@interface MotionTracker : NSObject

+ (instancetype)sharedTracker;

- (void)updateMotion;

@end

