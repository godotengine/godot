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

static void auth_info_2_dict(NdVGAuthInfoBase *info, Dictionary& dict) {

	dict["can_use_in_this_imei"] = info.canUseInThisImei;

	switch (info.vgFeeType) {

		case ND_VG_FEE_TYPE_INVALID:
			dict["type"] = "invalid";
			break;

		case ND_VG_FEE_TYPE_POSSESS:
			dict["type"] = "possess";
			break;
			
		case ND_VG_FEE_TYPE_SUBSCRIBE: {
				dict["type"] = "subscribe";
				NdVGAuthInfoSubscribe* subscribe = (NdVGAuthInfoSubscribe *) info;
				dict["remain_cnt"] = subscribe.nRemainCnt;
				dict["start_time"] = subscribe.strStartTime;
				dict["end_time"] = subscribe.strEndTime;
			}
			break;
			
		case ND_VG_FEE_TYPE_CONSUME: {
				dict["type"] = "consume";
				NdVGAuthInfoConsume* subscribe = (NdVGAuthInfoConsume*) info;
				dict["remain_cnt"] = subscribe.nRemainCnt;
			}
			break;
	}
}

static void user_info_2_dict(NdUserInfo *userInfo, Dictionary& info) {

	info["uin"] = [userInfo.uin UTF8String];
	info["nickName"] = [userInfo.nickName UTF8String];
	info["bornYear"] = userInfo.bornYear;
	info["bornMonth"] = userInfo.bornMonth;
	info["bornDay"] = userInfo.bornDay;
	info["sex"] = userInfo.sex;
	info["province"] = [userInfo.province UTF8String];
	info["city"] = [userInfo.city UTF8String];
	info["trueName"] = [userInfo.trueName UTF8String];
	info["point"] = [userInfo.point UTF8String];
	info["emotion"] = [userInfo.emotion UTF8String];
	info["checkSum"] = [userInfo.checkSum UTF8String];
}

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
	[[NSNotificationCenter defaultCenter] addObserver:self selector:@selector(SNSUserPortraitDidChange:) name:(NSString *)kNdCPUserPortraitDidChange object:nil];
	[[NSNotificationCenter defaultCenter] addObserver:self selector:@selector(SNSUserInfoDidChange:) name:(NSString *)kNdCPUserInfoDidChange object:nil];

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
	[[NSNotificationCenter defaultCenter] removeObserver:self name:(NSString *)kNdCPUserPortraitDidChange object:nil];
	[[NSNotificationCenter defaultCenter] removeObserver:self name:(NSString *)kNdCPUserInfoDidChange object:nil];

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
	event["result"] = success ? true : false;
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
	event["result"] = [[dic objectForKey:@"result"] boolValue] ? true : false;

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

- (void)SNSUserPortraitDidChange:(NSNotification*)notify
{

	Dictionary event;
	event["type"] = "user_portrait_did_change";
	event["error"] = "ok";
	Platform91::get_singleton()->post_event(event);
}

- (void)SNSUserInfoDidChange:(NSNotification*)notify
{	

	Dictionary event;
	event["type"] = "user_info_did_change";
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
	event["success"] = bSuccess ? true : false;
	Platform91::get_singleton()->post_event(event);
}

- (void)getCategoryListDidFinish:(int)error    records:(NSArray*)records
{

	Dictionary event;
	event["type"] = "category_list";
	event["error"] = "ok";
	Array arr;
	for(NdVGCategory *cate in records) {
		Dictionary rec;
		rec["id"] = [cate.strCateId UTF8String];
		rec["name"] = [cate.strCateName UTF8String];
		arr.append(rec);
	}
	event["records"] = arr;
	Platform91::get_singleton()->post_event(event);
}

- (void)getAppPromotionDidFinish:(int)error  promotion:(NSString*)promotion
{

	Dictionary event;
	event["type"] = "app_promotion";
	event["error"] = Platform91::error_to_string(error);
	event["promotion"] = [promotion UTF8String];
	Platform91::get_singleton()->post_event(event);
}

- (void)getCommodityListDidFinish:(int)error  cateId:(NSString*)cateId  feeType:(int)feeType 
						packageId:(NSString*)packageId result:(NdBasePageList*)result
{

	Dictionary event;
	event["type"] = "commodity_list";
	event["error"] = Platform91::error_to_string(error);
	if(error == 0) {
		event["total_count"] = result.totalCount;

		Array arr;
		for(NdVGCommodityInfo *info in result.records) {

			Dictionary rec;
			rec["product_id"] = [info.strProductId UTF8String];
			rec["product_name"] = [info.strProductName UTF8String];
			rec["cate_id"] = [info.strCateId UTF8String];
			rec["origin_price"] = [info.strOriginPrice UTF8String];
			rec["sale_price"] = [info.strSalePrice UTF8String];
			rec["checksum"] = [info.strChecksum UTF8String];
			rec["unit"] = [info.strUnit UTF8String];
			rec["goods_desc"] = [info.strGoodsDesc UTF8String];

			Dictionary fee_info;
			fee_info["bind_2_imei"] = info.vgFeeInfo.bind2Imei ? true : false;

			switch(info.vgFeeInfo.vgFeeType) {

				case ND_VG_FEE_TYPE_INVALID:
					fee_info["type"] = "invalid";
					break;

				case ND_VG_FEE_TYPE_POSSESS:
					fee_info["type"] = "possess";
					break;

				case ND_VG_FEE_TYPE_SUBSCRIBE: {
						fee_info["type"] = "subscribe";
						NdVGFeeInfoSubscribe* vgFeeInfo = (NdVGFeeInfoSubscribe*)info.vgFeeInfo;
						fee_info["auth_cnt_per_goods"] = vgFeeInfo.nAuthCntPerGoods;
						fee_info["auth_days"] = vgFeeInfo.nAuthDays;
					}
					break;

				case ND_VG_FEE_TYPE_CONSUME: {
						fee_info["type"] = "consume";
						NdVGFeeInfoConsume* vgFeeInfo = (NdVGFeeInfoConsume*)info.vgFeeInfo;
						fee_info["is_buy_limit_per_user_infinite"] = vgFeeInfo.isBuyLimitPerUserInfinite ? true : false;
						fee_info["is_stock_count_infinite"] = vgFeeInfo.isStockCountInfinite ? true : false;
						fee_info["time_period"] = vgFeeInfo.strTimePeriod;
				}
				break;

				default:
					break;
			}
			rec["fee_info"] = fee_info;

			arr.append(rec);
		}
		event["records"] = arr;
	}
}

- (void)useHoldingDidFinish:(int)error  useRequest:(NdVGUseRequest*)useRequest
                useResult:(NdVGUseResult*)useResult
{

	Dictionary event;
	event["type"] = "use";
	event["error"] = Platform91::error_to_string(error);

	if(error == 0) {
		Dictionary req;
		req["use_count"] = useRequest.nUseCount;
		req["product_id"] = [useRequest.strProductId UTF8String];
		req["ext_param"] = [useRequest.strExtParam UTF8String];
		event["request"] = req;

		Dictionary res;
		res["can_use"] = useResult.bCanUse ? true : false;
		res["err_code"] = useResult.nErrCode;
		res["err_desc"] = [useResult.strErrDesc UTF8String];

		Dictionary auth_info;
		auth_info_2_dict(useResult.vgAuthInfo, auth_info);
		res["auth_info"] = auth_info;
		event["result"] = res;

		event["can_use"] = useResult.bCanUse ? true : false;		
	}
	Platform91::get_singleton()->post_event(event);
}

- (void)NdProductIsPayedDidFinish:(int)error canUseInThisImei:(BOOL)canUse
                       errCode:(int)errCode  errDesc:(NSString*)errDesc
{

	Dictionary event;
	event["type"] = "is_payed";
	event["error"] = Platform91::error_to_string(error);
	event["can_use"] = canUse ? true : false;
	event["err_code"] = errCode;
	event["err_desc"] = [errDesc UTF8String];
	Platform91::get_singleton()->post_event(event);
}

- (void)NdProductIsExpiredDidFinish:(int)error isExpired:(BOOL)isExpired canUseInThisImei:(BOOL)canUse
                        errCode:(int)errCode  errDesc:(NSString*)errDesc
{

	Dictionary event;
	event["type"] = "is_expired";
	event["error"] = Platform91::error_to_string(error);
	event["is_expired"] = isExpired ? true : false;
	event["can_use"] = canUse ? true : false;
	event["err_code"] = errCode;
	event["err_desc"] = [errDesc UTF8String];
	Platform91::get_singleton()->post_event(event);
}

- (void)NdGetUserProductDidFinish:(int)error   canUse:(BOOL)canUse  errCode:(int)errCode
                       errDesc:(NSString*)errDesc  authInfo:(NdVGAuthInfoBase *)vgAuthInfo
{

	Dictionary event;
	event["type"] = "user_product";
	event["error"] = Platform91::error_to_string(error);
	event["can_use"] = canUse ? true : false;
	event["err_code"] = errCode;
	event["err_desc"] = [errDesc UTF8String];
	if(error == 0) {
		Dictionary auth_info;
		auth_info_2_dict(vgAuthInfo, auth_info);
		event["auth_info"] = auth_info;
	}
	Platform91::get_singleton()->post_event(event);
}

- (void)NdGetUserProductsListDidFinish:(int)error  result:(NdBasePageList*)result
{

	Dictionary event;
	event["type"] = "user_products_list";
	event["error"] = Platform91::error_to_string(error);

	if(error == 0) {
		Array arr;
		for (NdVGHoldingInfo* info in result.records) {

			Dictionary rec;
			rec["product_id"] = [info.strProductId UTF8String];
			rec["product_name"] = [info.strProductName UTF8String];
			Dictionary auth_info;
			auth_info_2_dict(info.vgAuthInfo, auth_info);
			rec["auth_info"] = auth_info;
			arr.append(rec);
		}
		event["records"] = arr;
	}
	Platform91::get_singleton()->post_event(event);
}

- (void)NdGetVirtualBalanceDidFinish:(int)error  balance:(NSString*)balance
{

	Dictionary event;
	event["type"] = "virtual_balance";
	event["error"] = Platform91::error_to_string(error);
	event["balance"] = [balance UTF8String];
	Platform91::get_singleton()->post_event(event);
}

- (void)getAppUserListDidFinish:(int)error resultList:(NdStrangerUserInfoList *)userInfoList
{

	Dictionary event;
	event["type"] = "app_user_list";
	event["error"] = Platform91::error_to_string(error);

	if(error == 0) {
		Dictionary pagination;
		pagination["page_size"] = userInfoList.pagination.pageSize;
		pagination["page_index"] = userInfoList.pagination.pageIndex;
		event["pagination"] = pagination;

		Array arr;
		for (NdStrangerUserInfoList *info in userInfoList.records) {

			Array list;
			for(NdStrangerUserInfo *us in info.records) {

				Dictionary rec;
				rec["province"] = [us.province UTF8String];
				rec["city"] = [us.city UTF8String];
				rec["sex"] = us.sex;
				rec["age"] = us.age;
				rec["onlineStatus"] = us.onlineStatus;

				Dictionary base;
				base["uin"] = [us.baseUserInfo.uin UTF8String];
				base["nickName"] = [us.baseUserInfo.nickName UTF8String];
				base["checkSum"] = [us.baseUserInfo.checkSum UTF8String];
				base["my_friend"] = us.baseUserInfo.bMyFriend ? true : false;

				rec["base"] = base;
				list.append(rec);
			}
			arr.append(list);
		}
		event["records"] = arr;
	}
	Platform91::get_singleton()->post_event(event);
}

- (void)getAppMyFriendListDidFinish:(int)error resultList:(NdFriendUserInfoList *)userInfoList
{

	Dictionary event;
	event["type"] = "app_my_friend_list";
	event["error"] = Platform91::error_to_string(error);

	if(error == 0)
	{
		Dictionary pagination;
		pagination["page_size"] = userInfoList.pagination.pageSize;
		pagination["page_index"] = userInfoList.pagination.pageIndex;
		event["pagination"] = pagination;

		Array arr;
		for (NdFriendUserInfoList *info in userInfoList.records) {

			Array list;
			for(NdFriendUserInfo *us in info.records) {

				Dictionary rec;
				rec["point"] = [us.point UTF8String];
				rec["emotion"] = [us.emotion UTF8String];
				rec["online_status"] = us.onlineStatus;

				Dictionary base;
				base["uin"] = [us.baseUserInfo.uin UTF8String];
				base["nickName"] = [us.baseUserInfo.nickName UTF8String];
				base["checkSum"] = [us.baseUserInfo.checkSum UTF8String];
				base["my_friend"] = us.baseUserInfo.bMyFriend ? true : false;

				rec["base"] = base;
				list.append(rec);
			}
			arr.append(list);
		}
		event["records"] = arr;
	}
	Platform91::get_singleton()->post_event(event);
}

- (void)searchMyFriendDidFinish:(int)error resultList:(NdFriendUserInfoList *)userInfoList
{

	Dictionary event;
	event["type"] = "my_friend_list";
	event["error"] = Platform91::error_to_string(error);

	if(error == 0) {
		Dictionary pagination;
		pagination["page_size"] = userInfoList.pagination.pageSize;
		pagination["page_index"] = userInfoList.pagination.pageIndex;
		event["pagination"] = pagination;

		Array arr;
		for (NdFriendUserInfoList *info in userInfoList.records) {

			Array list;
			for(NdFriendUserInfo *us in info.records) {

				Dictionary rec;
				rec["point"] = [us.point UTF8String];
				rec["emotion"] = [us.emotion UTF8String];
				rec["online_status"] = us.onlineStatus;

				Dictionary base;
				base["uin"] = [us.baseUserInfo.uin UTF8String];
				base["nickName"] = [us.baseUserInfo.nickName UTF8String];
				base["checkSum"] = [us.baseUserInfo.checkSum UTF8String];
				base["my_friend"] = us.baseUserInfo.bMyFriend ? true : false;

				rec["base"] = base;
				list.append(rec);
			}
			arr.append(list);
		}
		event["records"] = arr;
	}
	Platform91::get_singleton()->post_event(event);
}

- (void)getUserInfoDidFinish:(int)error userInfo:(NdUserInfo *)userInfo
{

	Dictionary event;
	event["type"] = "user_info";
	event["error"] = Platform91::error_to_string(error);

	if(error == 0) {
		Dictionary info;
		user_info_2_dict(userInfo, info);
		event["user_info"] = info;
	}
	Platform91::get_singleton()->post_event(event);
}

- (void)sendFriendMsgDidFinish:(int)error msgId:(NSString *)msgId
{

	Dictionary event;
	event["type"] = "chat";
	event["error"] = Platform91::error_to_string(error);
	event["msg_id"] = [msgId UTF8String];

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

		bool guest = false;
		if(p_params.has("guest"))
			guest = p_params["guest"];
		err = guest
			? [[NdComPlatform defaultPlatform] NdLogin:0]
			: [[NdComPlatform defaultPlatform] NdLoginEx:0];
	}
	else if(p_type == "logout") {

		err = [[NdComPlatform defaultPlatform] NdLogout:1];
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
	else if(p_type == "get_detail_info") {

		[[NdComPlatform defaultPlatform] NdGetMyInfoDetail: ::notification];
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
	else if(p_type == "enter_shop") {

		int nFeeType = ND_VG_FEE_TYPE_POSSESS | ND_VG_FEE_TYPE_SUBSCRIBE | ND_VG_FEE_TYPE_CONSUME;
		err = [[NdComPlatform defaultPlatform] NdEnterVirtualShop:nil feeType:nFeeType];
	}
	else if(p_type == "get_category_list") {

    	err = [[NdComPlatform defaultPlatform] NdGetCategoryList: ::notification];
	}
	else if(p_type == "get_commodity_list") {

		ERR_FAIL_COND_V(!p_params.has("page_size"), "invalid_param");
		ERR_FAIL_COND_V(!p_params.has("page_index"), "invalid_param");

		NdPagination* pagination = [[NdPagination new] autorelease];
		pagination.pageSize = p_params["page_size"];
		pagination.pageIndex = p_params["page_index"];

		int nFeeType = ND_VG_FEE_TYPE_POSSESS | ND_VG_FEE_TYPE_SUBSCRIBE | ND_VG_FEE_TYPE_CONSUME;
		err = [[NdComPlatform defaultPlatform] NdGetCommodityList:nil feeType:nFeeType pagination:pagination packageId:nil delegate: ::notification];
	}
	else if(p_type == "get_virtual_balance") {

		[[NdComPlatform defaultPlatform] NdGetVirtualBalance: ::notification];
	}
	else if(p_type == "get_app_promotion") {

		err = [[NdComPlatform defaultPlatform] NdGetAppPromotion: ::notification];
	}
	else if(p_type == "order") {

		ERR_FAIL_COND_V(!p_params.has("product_id"), "invalid_param");
		ERR_FAIL_COND_V(!p_params.has("count"), "invalid_param");
		ERR_FAIL_COND_V(!p_params.has("pay_description"), "invalid_param");

		String product_id = p_params["product_id"];
		real_t count = p_params["count"];
		String pay_description = p_params["pay_description"];

		NdVGOrderRequest *order_request = [NdVGOrderRequest orderRequestWithProductId:
			[[[NSString alloc] initWithUTF8String:product_id.utf8().get_data()] autorelease]
			productCount:count
			payDescription:[[[NSString alloc] initWithUTF8String:pay_description.utf8().get_data()] autorelease]
		];
		err = [[NdComPlatform defaultPlatform] NdBuyCommodity:order_request];
	}
	else if(p_type == "use") {

		ERR_FAIL_COND_V(!p_params.has("product_id"), "invalid_param");
		ERR_FAIL_COND_V(!p_params.has("count"), "invalid_param");

		String product_id = p_params["product_id"];

		NdVGUseRequest* useRqst = [[NdVGUseRequest new] autorelease];
		useRqst.nUseCount = p_params["count"];
		useRqst.strProductId = [[[NSString alloc] initWithUTF8String:product_id.utf8().get_data()] autorelease];

		err = [[NdComPlatform defaultPlatform] NdUseHolding:useRqst delegate: ::notification];
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
	else if(p_type == "is_payed") {

		ERR_FAIL_COND_V(!p_params.has("product_id"), "invalid_param");

		String product_id = p_params["product_id"];
		err = [[NdComPlatform defaultPlatform] NdProductIsPayed:
			[[[NSString alloc] initWithUTF8String:product_id.utf8().get_data()] autorelease]
			delegate: ::notification
		];
	}
	else if(p_type == "is_expired") {

		ERR_FAIL_COND_V(!p_params.has("product_id"), "invalid_param");

		String product_id = p_params["product_id"];
		err = [[NdComPlatform defaultPlatform] NdProductIsExpired:
			[[[NSString alloc] initWithUTF8String:product_id.utf8().get_data()] autorelease]
			delegate: ::notification
		];
	}
	else if(p_type == "get_user_product") {

		ERR_FAIL_COND_V(!p_params.has("product_id"), "invalid_param");

		String product_id = p_params["product_id"];
		err = [[NdComPlatform defaultPlatform] NdGetUserProduct:
			[[[NSString alloc] initWithUTF8String:product_id.utf8().get_data()] autorelease]
			delegate: ::notification
		];
	}
	else if(p_type == "get_user_products_list") {

		ERR_FAIL_COND_V(!p_params.has("page_size"), "invalid_param");
		ERR_FAIL_COND_V(!p_params.has("page_index"), "invalid_param");

		NdPagination* pagination = [[NdPagination new] autorelease];
		pagination.pageSize = p_params["page_size"];
		pagination.pageIndex = p_params["page_index"];
		err = [[NdComPlatform defaultPlatform] NdGetUserProductsList:pagination delegate: ::notification];
	}
	else if(p_type == "get_app_user_list") {

		ERR_FAIL_COND_V(!p_params.has("page_size"), "invalid_param");
		ERR_FAIL_COND_V(!p_params.has("page_index"), "invalid_param");

		NdPagination* pagination = [[NdPagination new] autorelease];
		pagination.pageSize = p_params["page_size"];
		pagination.pageIndex = p_params["page_index"];
		err = [[NdComPlatform defaultPlatform] NdGetAppUserList:pagination delegate: ::notification];

	}
	else if(p_type == "get_app_my_friend_list") {

		ERR_FAIL_COND_V(!p_params.has("page_size"), "invalid_param");
		ERR_FAIL_COND_V(!p_params.has("page_index"), "invalid_param");

		NdPagination* pagination = [[NdPagination new] autorelease];
		pagination.pageSize = p_params["page_size"];
		pagination.pageIndex = p_params["page_index"];
		err = [[NdComPlatform defaultPlatform] NdGetAppMyFriendList:pagination delegate: ::notification];
	}
	else if(p_type == "get_my_friend_list") {

		ERR_FAIL_COND_V(!p_params.has("page_size"), "invalid_param");
		ERR_FAIL_COND_V(!p_params.has("page_index"), "invalid_param");

		NdPagination* pagination = [[NdPagination new] autorelease];
		pagination.pageSize = p_params["page_size"];
		pagination.pageIndex = p_params["page_index"];
		err = [[NdComPlatform defaultPlatform] NdGetMyFriendList:pagination delegate: ::notification];
	}
	else if(p_type == "get_user_info_detail") {

		ERR_FAIL_COND_V(!p_params.has("uin"), "invalid_param");
		ERR_FAIL_COND_V(!p_params.has("type"), "invalid_param");

		String uin = p_params["uin"];
		String s = p_params["type"];

		int flag = 1;
		if(s == "base")
			flag = 1;
		else if(s == "score")
			flag = 2;
		else if(s == "emotion")
			flag = 4;

		[[NdComPlatform defaultPlatform] NdGetUserInfoDetail:
			[[[NSString alloc] initWithUTF8String:uin.utf8().get_data()] autorelease]
			flag:flag
			delegate: ::notification
		];
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
	else if(p_type == "chat") {
		ERR_FAIL_COND_V(!p_params.has("uin"), "content");
		ERR_FAIL_COND_V(!p_params.has("content"), "content");

		String uin = p_params["uin"];
		String content = p_params["content"];

		err = [[NdComPlatform defaultPlatform] NdSendFriendMsg:
			[[[NSString alloc] initWithUTF8String:uin.utf8().get_data()] autorelease]
			msgContent:[[[NSString alloc] initWithUTF8String:content.utf8().get_data()] autorelease]
			delegate: ::notification
		];
	}
	return error_to_string(err);
}

Platform91::Platform91() {

	::notification = [[NdNotification alloc] init];
} 

Platform91::~Platform91() {

	[::notification release];
}

