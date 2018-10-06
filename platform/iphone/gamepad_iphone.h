#import <GameController/GameController.h>

@class AppDelegate;

void _ios_add_joystick(GCController *controller, AppDelegate *delegate);

@interface GamepadManager : NSObject

@property(nonatomic, getter=isReady) BOOL ready;

+ (instancetype)sharedManager;

@end