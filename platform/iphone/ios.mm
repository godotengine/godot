#include "ios.h"

#import <UIKit/UIKit.h>

void iOS::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("get_rate_url","app_id"),&iOS::get_rate_url);
};

String iOS::get_rate_url(int p_app_id) const {
	String templ = "itms-apps://ax.itunes.apple.com/WebObjects/MZStore.woa/wa/viewContentsUserReviews?type=Purple+Software&id=APP_ID";
	String templ_iOS7 = "itms-apps://itunes.apple.com/app/idAPP_ID";
	String templ_iOS8 = "itms-apps://itunes.apple.com/WebObjects/MZStore.woa/wa/viewContentsUserReviews?id=APP_ID&onlyLatestVersion=true&pageNumber=0&sortOrdering=1&type=Purple+Software";

	//ios7 before
	String ret = templ;

	// iOS 7 needs a different templateReviewURL @see https://github.com/arashpayan/appirater/issues/131
	if ([[[UIDevice currentDevice] systemVersion] floatValue] >= 7.0 && [[[UIDevice currentDevice] systemVersion] floatValue] < 7.1)
	{
		ret = templ_iOS7;
	}
	// iOS 8 needs a different templateReviewURL also @see https://github.com/arashpayan/appirater/issues/182
	else if ([[[UIDevice currentDevice] systemVersion] floatValue] >= 8.0)
	{
		ret = templ_iOS8;
	}

	// ios7 for everything?
	ret = templ_iOS7.replace("APP_ID", String::num(p_app_id));

	printf("returning rate url %ls\n", ret.c_str());
	return ret;
};

iOS::iOS() {};
