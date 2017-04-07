/*************************************************************************/
/*  ios.mm                                                               */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "ios.h"

#import <UIKit/UIKit.h>

void iOS::_bind_methods() {

	ClassDB::bind_method(D_METHOD("get_rate_url","app_id"),&iOS::get_rate_url);
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
