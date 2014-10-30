/*************************************************************************/
/*  game_center.h                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                 */
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
#include "platform_91.h"

extern "C" {
#import <Foundation/Foundation.h>

#import <NdComPlatform/NdComPlatform.h>
#import <NdComPlatform/NdComPlatformAPIResponse.h>
#import <NdComPlatform/NdCPNotifications.h>
#import <NdComPlatform/NdComPlatformError.h>
};

@interface NdNotification : NSObject
- (NdNotification *) init;
@end

@implementation NdNotification
- (NdNotification *) init {

	[super init];

    [[NSNotificationCenter defaultCenter] addObserver:self selector:@selector(SNSInitResult:) name:(NSString *)kNdCPInitDidFinishNotification object:nil];
	[[NSNotificationCenter defaultCenter] addObserver:self selector:@selector(SNSLoginResult:) name:(NSString *)kNdCPLoginNotification object:nil];
	[[NSNotificationCenter defaultCenter] addObserver:self selector:@selector(SNSSessionInvalid:) name:(NSString *)kNdCPSessionInvalidNotification object:nil];
	[[NSNotificationCenter defaultCenter] addObserver:self selector:@selector(SNSleavePlatform:) name:(NSString *)kNdCPLeavePlatformNotification object:nil];
    [[NSNotificationCenter defaultCenter] addObserver:self selector:@selector(SNSPauseExist:) name:(NSString *)kNdCPPauseDidExitNotification object:nil];
	[[NSNotificationCenter defaultCenter] addObserver:self selector:@selector(SNSBuyResult:) name:(NSString *)kNdCPBuyResultNotification object:nil];
	[[NSNotificationCenter defaultCenter] addObserver:self selector:@selector(SNSLeaveComplatformUI:) name:(NSString *)kNdCPLeavePlatformNotification object:nil];

	return self;
}

- (void)dealloc {

    [[NSNotificationCenter defaultCenter] removeObserver:self name:(NSString *)kNdCPInitDidFinishNotification object:nil];
	[[NSNotificationCenter defaultCenter] removeObserver:self name:(NSString *)kNdCPLoginNotification object:nil];
	[[NSNotificationCenter defaultCenter] removeObserver:self name:(NSString *)kNdCPSessionInvalidNotification object:nil];
	[[NSNotificationCenter defaultCenter] removeObserver:self name:(NSString *)kNdCPLeavePlatformNotification object:nil];
    [[NSNotificationCenter defaultCenter] removeObserver:self name:(NSString *)kNdCPPauseDidExitNotification object:nil];
	[[NSNotificationCenter defaultCenter] removeObserver:self name:(NSString *)kNdCPBuyResultNotification object:nil];
	[[NSNotificationCenter defaultCenter] removeObserver:self name:(NSString *)kNdCPLeavePlatformNotification object:nil];

	[super dealloc];
}


- (void)SNSInitResult:(NSNotification *)notify
{

	Dictionary event;
	event["type"] = "init";
	event["error"] = "ok";
	Platform91::get_singleton()->post_event(event);
}

- (void)SNSLoginResult:(NSNotification *)notify
{
	NSDictionary *dict = [notify userInfo];
	BOOL success = [[dict objectForKey:@"result"] boolValue];
	NdGuestAccountStatus* guestStatus = (NdGuestAccountStatus*)[dict objectForKey:@"NdGuestAccountStatus"];

	Dictionary event;
	event["type"] = "login";
	// login succes
	if([[NdComPlatform defaultPlatform] isLogined] && success) {
		
		// 也可以通过[[NdComPlatform defaultPlatform] getCurrentLoginState]判断是否游客登录状态
		event["error"] = "ok";
		NSString* strUin = [[NdComPlatform defaultPlatform] loginUin];
		event["uin"] = [strUin UTF8String];

		// check if is guest login
		if(guestStatus) {

			// guest login
			event["guest"] = true;

			if([guestStatus isGuestLogined])
				event["registered"] = false;
			else if([guestStatus isGuestRegistered])
				event["registered"] = true;
		} else {

			// normal user login
			event["guest"] = false;

		}
	} else {

		// login failure
		int error = [[dict objectForKey:@"error"] intValue];
		event["error"] = Platform91::error_to_string(error);
	}
	Platform91::get_singleton()->post_event(event);
}

- (void)SNSSessionInvalid:(NSNotification *)notify
{

	Dictionary event;
	event["type"] = "session_invalid";
	event["error"] = "ok";
	Platform91::get_singleton()->post_event(event);
}

- (void)SNSleavePlatform:(NSNotification *)notify
{

	Dictionary event;
	event["type"] = "leave_platform";
	event["error"] = "ok";
	Platform91::get_singleton()->post_event(event);
}

- (void)SNSPauseExist:(NSNotification *)notify
{

	Dictionary event;
	event["type"] = "pause_exit";
	event["error"] = "ok";
	Platform91::get_singleton()->post_event(event);
}

- (void)SNSBuyResult:(NSNotification*)notify
{
	NSDictionary *dic = [notify userInfo];

	Dictionary event;
	event["type"] = "buy_result";
	event["error"] = Platform91::error_to_string([[dic objectForKey:@"error"] intValue]);
	event["result"] = [[dic objectForKey:@"result"] boolValue];

	NdBuyInfo* buyInfo = (NdBuyInfo*)[dic objectForKey:@"buyInfo"];
	event["product_id"] = [buyInfo.productId UTF8String];
	event["product_count"] = buyInfo.productCount;
	// 如果购买失败，可能无法得到cooOrderSerial, productPrice等字段值
	event["order_serial"] = [buyInfo.cooOrderSerial UTF8String];	
	// 如果购买虚拟商品失败，可以从vgErrorInfo获取具体失败详情
	NdVGErrorInfo* vgErrInfo = (NdVGErrorInfo*)[dic objectForKey:@"vgErrorInfo"];
	if(vgErrInfo) {
		event["error_code"] = vgErrInfo.nErrCode;
		event["error_desc"] = [vgErrInfo.strErrDesc UTF8String];
	}
	Platform91::get_singleton()->post_event(event);
}

- (void)SNSLeaveComplatformUI:(NSNotification*)notify
{

	Dictionary event;
	event["type"] = "leave_complatform";
	event["error"] = "ok";
	Platform91::get_singleton()->post_event(event);
}

- (void)checkPaySuccessDidFinish:(int)error
         cooOrderSerial:(NSString*)cooOrderSerial
         bSuccess:(BOOL)bSuccess
{

	Dictionary event;
	event["type"] = "check_pay";
	event["error"] = "ok";
	event["order_serial"] = [cooOrderSerial UTF8String];
	event["success"] = bSuccess;
	Platform91::get_singleton()->post_event(event);
}

@end

static NdNotification *notification = NULL;

static int string_to_orientation(const String& s) {

	if(s == "portrait")
		return UIDeviceOrientationPortrait;
	else if(s == "portrait_upsidedown")
		return UIDeviceOrientationPortraitUpsideDown;
	else if(s == "landscape_left")
		return UIDeviceOrientationLandscapeLeft;
	else if(s == "landscape_right")
		return UIDeviceOrientationLandscapeRight;
	return UIDeviceOrientationPortrait;
}

String Platform91::on_request(const String& p_type, const Dictionary& p_params) {

	int err = ND_COM_PLATFORM_NO_ERROR;
	if(p_type == "init") {

		ERR_FAIL_COND_V(!p_params.has("app_id"), "invalid_param");
		ERR_FAIL_COND_V(!p_params.has("app_key"), "invalid_param");
		int app_id = p_params["app_id"];
		String app_key = p_params["app_key"];

		int orientation = UIDeviceOrientationPortrait;
		if(p_params.has("orientation"))
			orientation = string_to_orientation(p_params["orientation"]);

		NdInitConfigure *cfg = [[[NdInitConfigure alloc] init] autorelease];
		cfg.appid = app_id;
		cfg.appKey = [[[NSString alloc] initWithUTF8String:app_key.utf8().get_data()] autorelease];
		cfg.versionCheckLevel = ND_VERSION_CHECK_LEVEL_STRICT;
		cfg.orientation = orientation;

		err = [[NdComPlatform defaultPlatform] NdInit:cfg];
	}
	else if(p_type == "set_debug_mode") {

		[[NdComPlatform defaultPlatform] NdSetDebugMode:0];
	}
	else if(p_type == "show_toolbar") {

		ERR_FAIL_COND_V(!p_params.has("place"), "invalid_param");
		String s = p_params["place"];
		NdToolBarPlace place = NdToolBarAtTopLeft;

		if(s == "hide") {

			[[NdComPlatform defaultPlatform] NdHideToolBar];
			return "ok";
		}
		else if(s == "top_left")
			place = NdToolBarAtTopLeft;
		else if(s == "top_right")
			place = NdToolBarAtTopRight;
		else if(s == "middle_left")
			place = NdToolBarAtMiddleLeft;
		else if(s == "middle_right")
			place = NdToolBarAtMiddleRight;
		else if(s == "bottom_left")
			place = NdToolBarAtBottomLeft;
		else if(s == "bottom_right")
			place = NdToolBarAtBottomRight;

		[[NdComPlatform defaultPlatform] NdShowToolBar:place];
	}
	else if(p_type == "set_auto_rotate") {

		ERR_FAIL_COND_V(!p_params.has("enabled"), "invalid_param");
		bool enabled = p_params["enabled"];
		[[NdComPlatform defaultPlatform] NdSetAutoRotation:enabled ? YES : NO];
	}
	else if(p_type == "set_screen_orientation") {

		ERR_FAIL_COND_V(!p_params.has("orientation"), "invalid_param");
		int orientation = string_to_orientation(p_params["orientation"]);
		[[NdComPlatform defaultPlatform] NdSetScreenOrientation:orientation];
	}
	else if(p_type == "login") {

		err = [[NdComPlatform defaultPlatform] NdLogin:0];
	}
	else if(p_type == "logout") {

		err = [[NdComPlatform defaultPlatform] NdLogout:1];
	}
	else if(p_type == "guest_login") {

		err = [[NdComPlatform defaultPlatform] NdLoginEx:0];
	}
	else if(p_type == "guest_regist") {

		[[NdComPlatform defaultPlatform] NdGuestRegist:0];
	}
	else if(p_type == "is_login") {

		if([[NdComPlatform defaultPlatform] isLogined])
			return "ok";
		else
			return "not_login";
	}
	else if(p_type == "get_login_state") {

		int state = [[NdComPlatform defaultPlatform] getCurrentLoginState];
		if(state == ND_LOGIN_STATE_NOT_LOGIN)
			return "not_login";
		else if(state == ND_LOGIN_STATE_GUEST_LOGIN)
			return "guest_login";
		else if(state == ND_LOGIN_STATE_NORMAL_LOGIN)
			return "normal_login";
	}
	else if(p_type == "get_uin") {

		NdMyUserInfo *info = [[NdComPlatform defaultPlatform] NdGetMyInfo];
		return [info.baseInfo.uin UTF8String];
	}
	else if(p_type == "get_nickname") {

		NdMyUserInfo *info = [[NdComPlatform defaultPlatform] NdGetMyInfo];
		return [info.baseInfo.nickName UTF8String];
	}
	else if(p_type == "switch_account") {

		[[NdComPlatform defaultPlatform] NdSwitchAccount];
	}
	else if(p_type == "enter_account_manage") {

		[[NdComPlatform defaultPlatform] NdEnterAccountManage];
	}
	else if(p_type == "feedback") {

		err = [[NdComPlatform defaultPlatform] NdUserFeedBack];
	}
	else if(p_type == "pause") {

		err = [[NdComPlatform defaultPlatform] NdPause];
	}
	else if(p_type == "enter_platform") {

		[[NdComPlatform defaultPlatform] NdEnterPlatform:0];
	}
	else if(p_type == "enter_friend_center") {

		[[NdComPlatform defaultPlatform] NdEnterFriendCenter:0];
	}
	else if(p_type == "enter_user_space") {

		ERR_FAIL_COND_V(!p_params.has("uin"), "invalid_param");
		String uin = p_params["uin"];
		NSString *strUin = [[[NSString alloc] initWithUTF8String:uin.utf8().get_data()] autorelease];
		err = [[NdComPlatform defaultPlatform] NdEnterUserSpace:strUin];
	}
	else if(p_type == "invite_friend") {

		ERR_FAIL_COND_V(!p_params.has("content"), "invalid_param");
		String content = p_params["content"];
		NSString *strContent = [[[NSString alloc] initWithUTF8String:content.utf8().get_data()] autorelease];
		err = [[NdComPlatform defaultPlatform] NdInviteFriend:strContent];
	}
	else if(p_type == "enter_app_center") {

		int app_id = 0;
		if(p_params.has(app_id))
			app_id = p_params["app_id"];

		[[NdComPlatform defaultPlatform] NdEnterAppCenter:0 appId:app_id];
	}
	else if(p_type == "enter_app_bbs") {

		err = [[NdComPlatform defaultPlatform] NdEnterAppBBS:0];
	}
	else if(p_type == "enter_user_setting") {

		err = [[NdComPlatform defaultPlatform] NdEnterUserSetting:0];
	}
	else if(p_type == "buy") {

		ERR_FAIL_COND_V(!p_params.has("order_serial"), "invalid_param");
		ERR_FAIL_COND_V(!p_params.has("product_name"), "invalid_param");
		ERR_FAIL_COND_V(!p_params.has("product_price"), "invalid_param");
		ERR_FAIL_COND_V(!p_params.has("pay_description"), "invalid_param");

		bool async = false;
		if(p_params.has("async"))
			async = p_params["async"];

		String order_serial = p_params["order_serial"];
		String product_name = p_params["product_name"];
		real_t product_price = p_params["product_price"];
		String pay_description = p_params["pay_description"];

		NdBuyInfo *buyInfo = [[NdBuyInfo new] autorelease];
		buyInfo.cooOrderSerial = [[[NSString alloc] initWithUTF8String:order_serial.utf8().get_data()] autorelease];
		buyInfo.productName = [[[NSString alloc] initWithUTF8String:product_name.utf8().get_data()] autorelease];
		buyInfo.productPrice = product_price;
		buyInfo.payDescription = [[[NSString alloc] initWithUTF8String:pay_description.utf8().get_data()] autorelease];

		if(async)
			err = [[NdComPlatform defaultPlatform]  NdUniPayAsyn: buyInfo];
		else
			err = [[NdComPlatform defaultPlatform] NdUniPay:buyInfo];
		// TODO: record buyInfo.cooOrderSerial
	}
	else if(p_type == "purchase") {

		ERR_FAIL_COND_V(!p_params.has("order_serial"), "invalid_param");
		ERR_FAIL_COND_V(!p_params.has("pay_coins"), "invalid_param");
		ERR_FAIL_COND_V(!p_params.has("pay_description"), "invalid_param");

		String order_serial = p_params["order_serial"];
		real_t pay_coins = p_params["pay_coins"];
		String pay_description = p_params["pay_description"];

		NSString *uuidString = [[[NSString alloc] initWithUTF8String:order_serial.utf8().get_data()] autorelease];
		NSString *payDescription = [[[NSString alloc] initWithUTF8String:pay_description.utf8().get_data()] autorelease];

		err = [[NdComPlatform defaultPlatform] NdUniPayForCoin:uuidString needPayCoins:pay_coins payDescription: payDescription];
	}
	else if(p_type == "check_order") {

		ERR_FAIL_COND_V(!p_params.has("order_serial"), "invalid_param");
		String order_serial = p_params["order_serial"];

		NSString *order = [[[NSString alloc] initWithUTF8String:order_serial.utf8().get_data()] autorelease];
		err = [[NdComPlatform defaultPlatform] NdCheckPaySuccess:order delegate: ::notification];
	}
	else if(p_type == "share") {

		ERR_FAIL_COND_V(!p_params.has("content"), "content");
		ERR_FAIL_COND_V(!p_params.has("image"), "invalid_param");

		String content = p_params["content"];
		String image = p_params["image"];

		NdImageInfo* imgInfo = nil;
		if(image != "screen_shot") {
			NSString *path = [[[NSString alloc] initWithUTF8String:image.utf8().get_data()] autorelease];
			imgInfo = [NdImageInfo imageInfoWithFile:path]; //@"200px-Rotating_earth_(large).gif"];
		} else
			imgInfo = [NdImageInfo imageInfoWithScreenShot];
		
		NSString *str = [[[NSString alloc] initWithUTF8String:content.utf8().get_data()] autorelease];
		err = [[NdComPlatform defaultPlatform] NdShareToThirdPlatform:str imageInfo:imgInfo];
	}
	return error_to_string(err);
}

Platform91::Platform91() {

	::notification = [[NdNotification alloc] init];
} 

Platform91::~Platform91() {

	[::notification release];
}

