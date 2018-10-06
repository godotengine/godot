#import <Foundation/Foundation.h>

@interface GamepadManager : NSObject

@property(nonatomic, getter=isReady) BOOL ready;

+ (instancetype)sharedManager;

@end