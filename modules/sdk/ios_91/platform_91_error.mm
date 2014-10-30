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

#import <NdComPlatform/NdComPlatform.h>

const char *Platform91::error_to_string(int p_error) {

	switch(p_error) {
		case ND_COM_PLATFORM_NO_ERROR:
			return "ok";
		case ND_COM_PLATFORM_ERROR_UNKNOWN:
			return "unknown_error";
		case ND_COM_PLATFORM_ERROR_NETWORK_FAIL:
			return "network_fail";
		case ND_COM_PLATFORM_ERROR_PACKAGE_INVALID:
			return "package_invalid";
		case ND_COM_PLATFORM_ERROR_SESSIONID_INVALID:
			return "session_invalid";
	}

// #define ND_COM_PLATFORM_NO_ERROR								0					/**<  */
// #define ND_COM_PLATFORM_ERROR_UNKNOWN							-1					/**< 未知错误 */
// #define ND_COM_PLATFORM_ERROR_NETWORK_FAIL						-2					/**< 网络连接错误 */
// #define ND_COM_PLATFORM_ERROR_PACKAGE_INVALID					-3					/**< 数据包不全、丢失或无效*/
// #define ND_COM_PLATFORM_ERROR_SESSIONID_INVALID					-4					/**< SessionId（用户的会话标识）无效 */
// #define ND_COM_PLATFORM_ERROR_PARAM								-5					/**< 参数值错误或非法，请检查参数值是否有效 */
// #define ND_COM_PLATFORM_ERROR_CLIENT_APP_ID_INVALID				-6					/**< 无效的应用ID接入 */
// #define ND_COM_PLATFORM_ERROR_NETWORK_ERROR						-7					/**< 网络通信发生错误 */
// #define ND_COM_PLATFORM_ERROR_APP_KEY_INVALID					-8					/**< 该用户未授权接入（AppKey无效）*/
// #define ND_COM_PLATFORM_ERROR_NO_SIM							-9					/**< 未检测到SIM卡 */
// #define ND_COM_PLATFORM_ERROR_SERVER_RETURN_ERROR				-10					/**< 服务器处理发生错误，请求无法完成 */
// #define ND_COM_PLATFORM_ERROR_NOT_LOGINED						-11					/**< 未登录 */
// #define ND_COM_PLATFORM_ERROR_USER_CANCEL						-12					/**< 用户取消 */
// #define ND_COM_PLATFORM_ERROR_BUSINESS_SYSTEM_UNCHECKED			-13					/**< 业务系统未通过审核 */
// #define ND_COM_PLATFORM_ERROR_SDK_VERSION_INVALID				-14					/**< SDK版本号无效 */
// #define ND_COM_PLATFORM_ERROR_NOT_PERMITTED						-15					/**< 接口不允许调用（比如，游客权限不足) */

// #define ND_COM_PLATFORM_ERROR_ACCOUNT_INVALID					-100				/**< 账号格式不合法，合法账号为4－70个字符，仅允许小写字母及数字，支持邮箱注册 */
// #define ND_COM_PLATFORM_ERROR_PASSWORD_INVALID					-101				/**< 密码格式不合法，密码不能为空，长度为6－12个字符，由字母和数字组成，大小写敏感 */
// #define ND_COM_PLATFORM_ERROR_LOGIN_FAIL						-102				/**< 登录失败 */
// #define ND_COM_PLATFORM_ERROR_ACCOUNT_NOT_EXIST					-103				/**< 91通行证账号不存在或者停用 */
// #define ND_COM_PLATFORM_ERROR_ACCOUNT_PASSWORD_ERROR			-104				/**< 91通行证账号密码错误 */
// #define ND_COM_PLATFORM_ERROR_TOO_MUCH_ACCOUNT_REGISTED			-105				/**< 该手机注册的账号数目已达到上限，无法再注册 */
// #define ND_COM_PLATFORM_ERROR_REGIST_FAIL						-106				/**< 注册失败 */
// #define ND_COM_PLATFORM_ERROR_ACCOUNT_HAS_EXIST					-107				/**< 该91通行证账号已经被注册 */
// #define ND_COM_PLATFORM_ERROR_VERIFY_ACCOUNT_FAIL				-108				/**< 账号验证失败 */
// #define ND_COM_PLATFORM_ERROR_PARAM_INVALID						-109				/**< 参数无效 */
// #define ND_COM_PLATFORM_ERROR_IGNORE_CONTACT_LIST				-110				/**< 相同的通讯录已经上传，忽略此次上传 */
// #define ND_COM_PLATFORM_ERROR_DEVICE_NEVER_LOGINED				-111				/**< 该设备没有登录过用户 */
// #define ND_COM_PLATFORM_ERROR_DEVICE_CANNOT_AUTO_LOGIN			-112				/**< 该设备不能自动登录 */
// #define ND_COM_PLATFORM_ERROR_ACCOUNT_PRESERVED					-113				/**< 账号无效（可能被保留） */
// #define ND_COM_PLATFORM_ERROR_AUTO_LOGIN_SIGN_INVALID			-114				/**< 自动登录凭据失效，请重新输入密码登录 */

// #define ND_COM_PLATFORM_ERROR_NICKNAME_INVALID					-201				/**< 昵称不合法，合法昵称由1－16个非空字符构成，请勿使用敏感词汇 */

// #define ND_COM_PLATFORM_ERROR_NEW_PASSWORD_INVALID				-301				/**< 新密码格式非法，密码不能为空，长度为6－12个字符，由字母和数字组成，大小写敏感 */
// #define ND_COM_PLATFORM_ERROR_OLD_PASSWORD_INVALID				-302				/**< 旧密码格式非法，不能为空*/
// #define ND_COM_PLATFORM_ERROR_OLD_PASSWORD_ERROR				-303				/**< 原密码错误 */

// #define ND_COM_PLATFORM_ERROR_HAS_SET_PHONE_NUM					-401				/**< 该用户已经设定了手机号码 */
// #define ND_COM_PLATFORM_ERROR_PHONE_NUM_BE_REGISTERED			-402				/**< 该手机号已经被注册 */
// #define ND_COM_PLATFORM_ERROR_PHONE_NUM_REPEAT_CHECK			-403				/**< 指定时间内不能重复发送（手机号注册短信验证） */
// #define ND_COM_PLATFORM_ERROR_PHONE_CHECK_CODE_INVALID			-404				/**< 手机短信验证码无效 */

// #define ND_COM_PLATFORM_ERROR_TRUE_NAME_INVALID					-501				/**< 真实姓名不合法 */

// #define ND_COM_PLATFORM_ERROR_EMOTION_LENGTH_TOO_LONG			-601				/**< 心情长度太长，不能超过140个字符 */
// #define ND_COM_PLATFORM_ERROR_CONTENT_INVALID					-602				/**< 内容不合法 */

// #define ND_COM_PLATFORM_ERROR_PERMISSION_NOT_ENOUGH				-701				/**< 权限不足 */

// #define ND_COM_PLATFORM_ERROR_IMAGE_SIZE_TOO_LARGE				-801				/**< 发送的图片数据超过了服务器允许的大小 */
// #define ND_COM_PLATFORM_ERROR_IMAGE_DATA_INVALID				-802				/**< 发送的图片数据内容不合法 */

// #define ND_COM_PLATFORM_ERROR_PHOTO_NOT_CHANGED					-1001				/**< 头像没有变更 */
// #define ND_COM_PLATFORM_ERROR_NO_CUSTOM_PHOTO					-1002				/**< 该用户没有自定义头像 */

// #define ND_COM_PLATFORM_ERROR_APP_NOT_EXIST						-2001				/**< 该应用不存在 */
// #define ND_COM_PLATFORM_ERROR_ICON_NOT_CHANGED					-2002				/**< 图标没有变更 */
// #define ND_COM_PLATFORM_ERROR_NO_CUSTOM_ICON					-2003				/**< 无自定义图标 */
// #define ND_COM_PLATFORM_ERROR_ICON_NOT_EXIST					-2004				/**< 该图标不存在 */

// #define ND_COM_PLATFORM_ERROR_PAY_PASSWORD_ERROR				-3001				/**< 支付密码错误 */
// #define ND_COM_PLATFORM_ERROR_PAY_ACCOUNT_NOT_ACTIVED			-3002				/**< 该账号未在商城开户 */
// #define ND_COM_PLATFORM_ERROR_PAY_PASSWORD_NOT_SET				-3003				/**< 支付密码未设置 */

// #define ND_COM_PLATFORM_ERROR_PAY_PASSWORD_NOT_VERIFY			-4001				/**< 支付密码未验证 */
// #define ND_COM_PLATFORM_ERROR_BALANCE_NOT_ENOUGH				-4002				/**< 余额不足，无法支付 */
// #define ND_COM_PLATFORM_ERROR_ORDER_SERIAL_DUPLICATE			-4003				/**< 订单号重复 */
// #define ND_COM_PLATFORM_ERROR_ORDER_SERIAL_SUBMITTED			-4004				/**< 订单已提交 */

// #define ND_COM_PLATFORM_ERROR_PAGE_REQUIRED_NOT_VALID			-5001				/**< 页码超过范围 */

// #define ND_COM_PLATFORM_ERROR_RECHARGE_MONEY_INVALID			-6001				/**< 充值面额非法 */
// #define ND_COM_PLATFORM_ERROR_SMS_RECHARGE_ACCOUNT_INVALID		-6002				/**< 短信支付账号无效 */
// #define ND_COM_PLATFORM_ERROR_NO_PHONE_NUM						-6003				/**< 没有手机号码，账号未绑定手机号 */

// #define ND_COM_PLATFORM_ERROR_RECHARGE_CARD_NUMBER_ERROR		-7001				/**< 充值卡卡号无效 */
// #define ND_COM_PLATFORM_ERROR_RECHARGE_CARD_PASSWORD_ERROR		-7002				/**< 充值卡密码无效 */
// #define ND_COM_PLATFORM_ERROR_RECHARGE_CARD_TYPE_NOT_SUPPORT	-7003				/**< 充值卡类型不支持 */

// #define ND_COM_PLATFORM_ERROR_USER_NOT_EXIST					-10011				/**< 该用户不存在 */
// #define ND_COM_PLATFORM_ERROR_FRIEND_NOT_EXIST					-10012				/**< 该好友不存在 */

// #define ND_COM_PLATFORM_ERROR_ALREADY_BE_YOUR_FRIEND			-11002				/**< 用户已经是您的好友 */
// #define ND_COM_PLATFORM_ERROR_NOTE_LENGTH_INVALID				-11003				/**< 备注长度不合法 */
// #define ND_COM_PLATFORM_ERROR_ARRIVE_MAX_FRIEND_NUM				-11004				/**< 到达好友个数上限，需要删除好友 */

// #define ND_COM_PLATFORM_ERROR_APP_ID_INVALID					-13001				/**< 应用Id不合法 */
// #define ND_COM_PLATFORM_ERROR_ACTIVITY_TYPE_INVALID				-13002				/**< 动态类型不合法 */

// #define ND_COM_PLATFORM_ERROR_MSG_NOT_EXIST						-14001				/**< 没有该条消息 */

// #define ND_COM_PLATFORM_ERROR_CONTENT_LENGTH_INVALID			-15001				/**< 内容长度不合法 */
// #define ND_COM_PLATFORM_ERROR_NOT_ALLOWED_TO_SEND_MSG			-15002				/**< 发送者被禁止发消息 */
// #define ND_COM_PLATFORM_ERROR_CAN_NOT_SEND_MSG_TO_SELF			-15003				/**< 不能给自己发送短消息 */

// #define ND_COM_PLATFORM_ERROR_CLIENT_TAG						-16001				/**< 该标签为客户端标签 */
// #define ND_COM_PLATFORM_ERROR_INVALID_COMMAND_TAG				-16002				/**< 无效的标签指令 */
// #define ND_COM_PLATFORM_ERROR_INVALID_CONTENT_TAG				-16003				/**< 无效的标签文本 */
// #define ND_COM_PLATFORM_ERROR_CUSTOM_TAG_ARG_NOT_ENOUGH			-16004				/**< 自定义标签参数不足 */
// #define ND_COM_PLATFORM_ERROR_CUSTOM_TAG_INVALID				-16005				/**< 自定义标签参数无效 */

// #define ND_COM_PLATFORM_ERROR_FEEDBACK_ID_INVALID				-17001				/**< 反馈类型ID无效 */

// #define ND_COM_PLATFORM_ERROR_TEMPLATEID_INVALID				-18001				/**< 模板ID无效 */
// #define ND_COM_PLATFORM_ERROR_TEMPLATE_PARAMLIST_ERROR			-18002				/**< 模板参数错误 */
// #define ND_COM_PLATFORM_ERROR_PAY_FAILED						-18003				/**< 支付失败 */
// #define ND_COM_PLATFORM_ERROR_PAY_CANCELED						-18004				/**< 取消支付 */

// #define ND_COM_PLATFORM_ERROR_LEADERBOARD_NOT_EXIST				-19001				/**< 该排行榜不存在 */
// #define ND_COM_PLATFORM_ERROR_LEADERBOARD_USERLIST_NOT_EXIST	-19002				/**< 用户排行列表不存在 */
// #define ND_COM_PLATFORM_ERROR_FRIENDS_NOBODY_PLAYING			-19003				/**< 该用户没有好友在玩 */
// #define ND_COM_PLATFORM_ERROR_ACHIEVEMENT_NOT_EXIST				-19004				/**< 成就不存在 */

// #define ND_COM_PLATFORM_ERROR_91_HAS_NOT_BIND_3RD				-19030				/**< 用户没有绑定第三方账号 */
// #define ND_COM_PLATFORM_ERROR_3RD_SHARED_CONTENT_REPEAT			-19031				/**< 内容重复 */
// #define ND_COM_PLATFORM_ERROR_PAY_ORDER_NOT_EXIST				-19032				/**< 无此订单 */
// #define ND_COM_PLATFORM_ERROR_NOT_MY_REQUEST_FOR_PAY			-19033				/**< 该代付订单的代付人不是当前用户 */
// #define ND_COM_PLATFORM_ERROR_NOT_MY_FRIEND_ANYMORE				-19034				/**< 不是你的好友 */

// #define ND_COM_PLATFORM_ERROR_3RD_ACCOUNT_HAS_NO_FRIENDS		-19040				/**< 该第三方账号没有好友 */
// #define ND_COM_PLATFORM_ERROR_91_HAS_BIND_3RD					-19041				/**< 已经绑定指定类型的第三方账号 */
// #define ND_COM_PLATFORM_ERROR_3RD_HAS_BIND_OTHER_91				-19042				/**< 第三方账号已经绑定其他91账号 */
// #define ND_COM_PLATFORM_ERROR_3RD_HAS_BIND_91					-19043				/**< 已经绑定91账号 */
// #define ND_COM_PLATFORM_ERROR_3RD_ACCOUNT_INFO_LOST				-19044				/**< 第三方登陆信息丢失 */
// #define ND_COM_PLATFORM_ERROR_CAN_NOT_VERIFY_3RD_ACCOUNT		-19045				/**< 无法验证第三方账号密码 */
// #define ND_COM_PLATFORM_ERROR_91_ACCOUNT_EXCEPTION				-19046				/**< 第三方账号绑定的91账号异常（停用等） */

// #define ND_COM_PLATFORM_ERROR_3RD_SESSION_ID_INVALID			-20001				/**< ThirdSessionId无效 */

// #define ND_COM_PLATFORM_ERROR_VG_CATEGORY_INVALID				-21001				/**< 商品类别无效 */
// #define ND_COM_PLATFORM_ERROR_VG_FEE_TYPE_INVALID				-21002				/**< 商品计费类型无效 */

// #define ND_COM_PLATFORM_ERROR_3RD_INFO_INVALID					-22001				/**< 第三方信息不存在 */
// #define ND_COM_PLATFORM_ERROR_CANNOT_UNBIND_LOGINED_3RD_ACCOUNT	-22002				/**< 不能解除当前登录的第三方帐号绑定 */
// #define ND_COM_PLATFORM_ERROR_VERFIER_INVALID					-22003				/**< 校验码错误 */

// #define ND_COM_PLATFORM_ERROR_REPEAT_SENDING					-23001				/**< 重复发送 */
// #define ND_COM_PLATFORM_ERROR_PAY_REQUEST_TIMEOUT				-23002				/**< 支付超时，请稍候重试 */
// #define ND_COM_PLATFORM_ERROR_VG_PRODUCT_USE_SIGN_INVALID		-23003				/**< 虚拟商品使用标志无效 */
// #define ND_COM_PLATFORM_ERROR_VG_PRODUCT_ID_INVALID				-23004				/**< 虚拟商品ID无效 */

// #define ND_COM_PLATFORM_ERROR_VG_MONEY_TYPE_FAILED				-24001				/**< 查询虚拟商品币种失败 */
// #define ND_COM_PLATFORM_ERROR_VG_ORDER_FAILED					-24002				/**< 获取虚拟商品订单号失败 */
// #define ND_COM_PLATFORM_ERROR_VG_BACK_FROM_RECHARGE				-24003				/**< 退出充值界面（购买游戏币虚拟商品时） */

// #define ND_COM_PLATFORM_ERROR_BD_INVALID_PHONE_NUM				-25001				/**< 手机号码格式无效 */
// #define	ND_COM_PLATFORM_ERROR_BD_ACCOUNT_HAS_BIND_PHONE_NUM		-25002				/**< 重复绑定，账号已经绑定其他手机号 */
// #define	ND_COM_PLATFORM_ERROR_BD_PHONE_NUM_HAS_BIND_ACCOUNT		-25003				/**< 手机号已经绑定其他账号 */
// #define	ND_COM_PLATFORM_ERROR_BD_PHONE_NUM_DIDNOT_BIND			-25004				/**< 账号未绑定任何手机号 */
// #define	ND_COM_PLATFORM_ERROR_BD_WRONG_BIND_PHONE_NUM			-25005				/**< 手机号码与绑定手机号码不一致 */
// #define	ND_COM_PLATFORM_ERROR_BD_WRONG_SMS_VERIFY_CODE			-25006				/**< 短信验证码错误 */
// #define	ND_COM_PLATFORM_ERROR_BD_SMS_VERIFY_CODE_OUT_EXPIRE		-25007				/**< 短信验证码过期 */
// #define	ND_COM_PLATFORM_ERROR_BD_PHONE_VERIFY_FAIL				-25008				/**< 未通过手机号码验证(重置密码时) */
// #define	ND_COM_PLATFORM_ERROR_BD_UNFIT_LOTTERY_CONDITION		-25009				/**< 不符合抽奖条件 */
// #define	ND_COM_PLATFORM_ERROR_BD_HAS_LOTTERY					-25010				/**< 已经抽过奖 */
// #define	ND_COM_PLATFORM_ERROR_BD_OUT_NUM_SMS_SENDED				-25011				/**< 短信发送次数过多 */
// #define ND_COM_PLATFORM_ERROR_BD_VIP_CANNOT_RESTE_ON_PHONE		-25012				/**< VIP用户无法使用手机找回密码 */
// #define ND_COM_PLATFORM_ERROR_BD_DIFFERENT_PHONE_NUM			-25013				/**< 手机号码前后不一致 */

// #define ND_COM_PLATFORM_ERROR_HAS_ASSOCIATE_91                  -26001				/**< 有关联的91账号，不能以游客方式登录 */
// #define ND_COM_PLATFORM_ERROR_NO_NEED_BECOME_REGULAR			-26002				/**< 非游客登录状态 */
// #define ND_COM_PLATFORM_ERROR_UIN_INVALID                       -26003              /**< Uin不合法*/
// #define ND_COM_PLATFORM_ERROR_GUEST_NOT_PERMITTED				-26004				/**< 游客权限不够，需要注册91账号 */

// #define ND_COM_PLATFORM_ERROR_HIGH_FREQUENT_OPERATION			-27001				/**< 操作过于频繁 */
// #define ND_COM_PLATFORM_ERROR_PROMOTED_APP_NOT_TOUCHED			-27002				/**< 该应用在推广墙上未被点击 */

// #define ND_COM_PLATFORM_ERROR_3RD_AUTH_FAILED					-28001				/**< 验证第三方账号授权失败 */
// #define ND_COM_PLATFORM_ERROR_3RD_REAUTH_FAILDED				-28002				/**< 验证第三方绑定信息失败 */
	return "ok";
}


// /*
//  *  NdComPlatformError.h
//  *  NdComPlatform
//  *
//  *  Created by Sie Kensou on 10-8-12.
//  *  Copyright 2010 NetDragon WebSoft Inc.. All rights reserved.
//  *
//  */

// /*!
//  该头文件定义的返回的错误代码编号
//  */



