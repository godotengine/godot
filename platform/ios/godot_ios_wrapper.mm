// godot_ios_wrapper.mm

#import "godot_ios_wrapper.h"

@implementation GodotIosWrapper

extern int ios_main(int, char **);
extern void ios_finish();

+ (void)ios_finish_wrapper {
    ios_finish();
}

+ (int)ios_main_wrapper:(int)argc argv:(char **)argv {
    return ios_main(argc, argv);
}

@end
