#import <UIKit/UIKit.h>
#import "app_delegate.h"
#include <stdio.h>

int gargc;
char** gargv;

int main(int argc, char *argv[])
{
	printf("*********** main.m\n");
	gargc = argc;
	gargv = argv;

	NSAutoreleasePool *pool = [NSAutoreleasePool new];
	AppDelegate* app = [AppDelegate alloc];
	printf("running app main\n");
	UIApplicationMain(argc, argv, nil, @"AppDelegate");
	printf("main done, pool release\n");
	[pool release];
	return 0;
}

