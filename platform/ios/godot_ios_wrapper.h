// godot_ios_wrapper.h

#import <Foundation/Foundation.h>

@interface GodotIosWrapper : NSObject

+ (void)ios_finish_wrapper;
+ (int)ios_main_wrapper:(int)argc argv:(char **)argv;

@end
