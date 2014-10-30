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
			return "error_unknown";
		case ND_COM_PLATFORM_ERROR_NETWORK_FAIL:
			return "error_network_fail";
		case ND_COM_PLATFORM_ERROR_PACKAGE_INVALID:
			return "error_package_invalid";
		case ND_COM_PLATFORM_ERROR_SESSIONID_INVALID:
			return "error_session_invalid";
		case ND_COM_PLATFORM_ERROR_PARAM:
			return "error_param";
		case ND_COM_PLATFORM_ERROR_CLIENT_APP_ID_INVALID:
			return "error_client_app_id_invalid";
		case ND_COM_PLATFORM_ERROR_NETWORK_ERROR:
			return "error_network_error";
		case ND_COM_PLATFORM_ERROR_APP_KEY_INVALID:
			return "error_app_key_invalid";
		case ND_COM_PLATFORM_ERROR_NO_SIM:
			return "error_no_sim";
		case ND_COM_PLATFORM_ERROR_SERVER_RETURN_ERROR:
			return "error_server_return_error";
		case ND_COM_PLATFORM_ERROR_NOT_LOGINED:
			return "error_not_logined";
		case ND_COM_PLATFORM_ERROR_USER_CANCEL:
			return "error_user_cancel";
		case ND_COM_PLATFORM_ERROR_BUSINESS_SYSTEM_UNCHECKED:
			return "error_business_system_unchecked";
		case ND_COM_PLATFORM_ERROR_SDK_VERSION_INVALID:
			return "error_sdk_version_invalid";
		case ND_COM_PLATFORM_ERROR_NOT_PERMITTED:
			return "error_not_permitted";

		case ND_COM_PLATFORM_ERROR_ACCOUNT_INVALID:
			return "error_account_invalid";
		case ND_COM_PLATFORM_ERROR_PASSWORD_INVALID:
			return "error_password_invalid";
		case ND_COM_PLATFORM_ERROR_LOGIN_FAIL:
			return "error_login_fail";
		case ND_COM_PLATFORM_ERROR_ACCOUNT_NOT_EXIST:
			return "error_account_not_exist";
		case ND_COM_PLATFORM_ERROR_ACCOUNT_PASSWORD_ERROR:
			return "error_account_password_error";
		case ND_COM_PLATFORM_ERROR_TOO_MUCH_ACCOUNT_REGISTED:
			return "error_too_much_account_registed";
		case ND_COM_PLATFORM_ERROR_REGIST_FAIL:
			return "error_regist_fail";
		case ND_COM_PLATFORM_ERROR_ACCOUNT_HAS_EXIST:
			return "error_account_has_exist";
		case ND_COM_PLATFORM_ERROR_VERIFY_ACCOUNT_FAIL:
			return "error_verify_account_fail";
		case ND_COM_PLATFORM_ERROR_PARAM_INVALID:
			return "error_param_invalid";
		case ND_COM_PLATFORM_ERROR_IGNORE_CONTACT_LIST:
			return "error_ignore_contact_list";
		case ND_COM_PLATFORM_ERROR_DEVICE_NEVER_LOGINED:
			return "error_device_never_logined";
		case ND_COM_PLATFORM_ERROR_DEVICE_CANNOT_AUTO_LOGIN:
			return "error_device_cannot_auto_login";
		case ND_COM_PLATFORM_ERROR_ACCOUNT_PRESERVED:
			return "error_account_preserved";
		case ND_COM_PLATFORM_ERROR_AUTO_LOGIN_SIGN_INVALID:
			return "error_auto_login_sign_invalid";

		case ND_COM_PLATFORM_ERROR_NICKNAME_INVALID:
			return "error_nickname_invalid";

		case ND_COM_PLATFORM_ERROR_NEW_PASSWORD_INVALID:
			return "error_new_password_invalid";
		case ND_COM_PLATFORM_ERROR_OLD_PASSWORD_INVALID:
			return "error_old_password_invalid";
		case ND_COM_PLATFORM_ERROR_OLD_PASSWORD_ERROR:
			return "error_old_password_error";

		case ND_COM_PLATFORM_ERROR_HAS_SET_PHONE_NUM:
			return "error_has_set_phone_num";
		case ND_COM_PLATFORM_ERROR_PHONE_NUM_BE_REGISTERED:
			return "error_phone_num_be_registered";
		case ND_COM_PLATFORM_ERROR_PHONE_NUM_REPEAT_CHECK:
			return "error_phone_num_repeat_check";
		case ND_COM_PLATFORM_ERROR_PHONE_CHECK_CODE_INVALID:
			return "error_phone_check_code_invalid";

		case ND_COM_PLATFORM_ERROR_TRUE_NAME_INVALID:
			return "error_true_name_invalid";

		case ND_COM_PLATFORM_ERROR_EMOTION_LENGTH_TOO_LONG:
			return "error_emotion_length_too_long";
		case ND_COM_PLATFORM_ERROR_CONTENT_INVALID:
			return "error_content_invalid";

		case ND_COM_PLATFORM_ERROR_PERMISSION_NOT_ENOUGH:
			return "error_permission_not_enough";

		case ND_COM_PLATFORM_ERROR_IMAGE_SIZE_TOO_LARGE:
			return "error_image_size_too_large";
		case ND_COM_PLATFORM_ERROR_IMAGE_DATA_INVALID:
			return "error_image_data_invalid";

		case ND_COM_PLATFORM_ERROR_PHOTO_NOT_CHANGED:
			return "error_photo_not_changed";
		case ND_COM_PLATFORM_ERROR_NO_CUSTOM_PHOTO:
			return "error_no_custom_photo";

		case ND_COM_PLATFORM_ERROR_APP_NOT_EXIST:
			return "error_app_not_exist";
		case ND_COM_PLATFORM_ERROR_ICON_NOT_CHANGED:
			return "error_icon_not_changed";
		case ND_COM_PLATFORM_ERROR_NO_CUSTOM_ICON:
			return "error_no_custom_icon";
		case ND_COM_PLATFORM_ERROR_ICON_NOT_EXIST:
			return "error_icon_not_exist";

		case ND_COM_PLATFORM_ERROR_PAY_PASSWORD_ERROR:
			return "error_pay_password_error";
		case ND_COM_PLATFORM_ERROR_PAY_ACCOUNT_NOT_ACTIVED:
			return "error_pay_account_not_actived";
		case ND_COM_PLATFORM_ERROR_PAY_PASSWORD_NOT_SET:
			return "error_pay_password_not_set";

		case ND_COM_PLATFORM_ERROR_PAY_PASSWORD_NOT_VERIFY:
			return "error_pay_password_not_verify";
		case ND_COM_PLATFORM_ERROR_BALANCE_NOT_ENOUGH:
			return "error_balance_not_enough";
		case ND_COM_PLATFORM_ERROR_ORDER_SERIAL_DUPLICATE:
			return "error_order_serial_duplicate";
		case ND_COM_PLATFORM_ERROR_ORDER_SERIAL_SUBMITTED:
			return "error_order_serial_submitted";

		case ND_COM_PLATFORM_ERROR_PAGE_REQUIRED_NOT_VALID:
			return "error_page_required_not_valid";

		case ND_COM_PLATFORM_ERROR_RECHARGE_MONEY_INVALID:
			return "error_recharge_money_invalid";
		case ND_COM_PLATFORM_ERROR_SMS_RECHARGE_ACCOUNT_INVALID:
			return "error_sms_recharge_account_invalid";
		case ND_COM_PLATFORM_ERROR_NO_PHONE_NUM:
			return "error_no_phone_num";

		case ND_COM_PLATFORM_ERROR_RECHARGE_CARD_NUMBER_ERROR:
			return "error_recharge_card_number_error";
		case ND_COM_PLATFORM_ERROR_RECHARGE_CARD_PASSWORD_ERROR:
			return "error_recharge_card_password_error";
		case ND_COM_PLATFORM_ERROR_RECHARGE_CARD_TYPE_NOT_SUPPORT:
			return "error_recharge_card_type_not_support";

		case ND_COM_PLATFORM_ERROR_USER_NOT_EXIST:
			return "error_user_not_exist";
		case ND_COM_PLATFORM_ERROR_FRIEND_NOT_EXIST:
			return "error_friend_not_exist";

		case ND_COM_PLATFORM_ERROR_ALREADY_BE_YOUR_FRIEND:
			return "error_already_be_your_friend";
		case ND_COM_PLATFORM_ERROR_NOTE_LENGTH_INVALID:
			return "error_note_length_invalid";
		case ND_COM_PLATFORM_ERROR_ARRIVE_MAX_FRIEND_NUM:
			return "error_arrive_max_friend_num";

		case ND_COM_PLATFORM_ERROR_APP_ID_INVALID:
			return "error_app_id_invalid";
		case ND_COM_PLATFORM_ERROR_ACTIVITY_TYPE_INVALID:
			return "error_activity_type_invalid";

		case ND_COM_PLATFORM_ERROR_MSG_NOT_EXIST:
			return "error_msg_not_exist";

		case ND_COM_PLATFORM_ERROR_CONTENT_LENGTH_INVALID:
			return "error_content_length_invalid";
		case ND_COM_PLATFORM_ERROR_NOT_ALLOWED_TO_SEND_MSG:
			return "error_not_allowed_to_send_msg";
		case ND_COM_PLATFORM_ERROR_CAN_NOT_SEND_MSG_TO_SELF:
			return "error_can_not_send_msg_to_self";

		case ND_COM_PLATFORM_ERROR_CLIENT_TAG:
			return "error_client_tag";
		case ND_COM_PLATFORM_ERROR_INVALID_COMMAND_TAG:
			return "error_invalid_command_tag";
		case ND_COM_PLATFORM_ERROR_INVALID_CONTENT_TAG:
			return "error_invalid_content_tag";
		case ND_COM_PLATFORM_ERROR_CUSTOM_TAG_ARG_NOT_ENOUGH:
			return "error_custom_tag_arg_not_enough";
		case ND_COM_PLATFORM_ERROR_CUSTOM_TAG_INVALID:
			return "error_custom_tag_invalid";

		case ND_COM_PLATFORM_ERROR_FEEDBACK_ID_INVALID:
			return "error_feedback_id_invalid";

		case ND_COM_PLATFORM_ERROR_TEMPLATEID_INVALID:
			return "error_templateid_invalid";
		case ND_COM_PLATFORM_ERROR_TEMPLATE_PARAMLIST_ERROR:
			return "error_template_paramlist_error";
		case ND_COM_PLATFORM_ERROR_PAY_FAILED:
			return "error_pay_failed";
		case ND_COM_PLATFORM_ERROR_PAY_CANCELED:
			return "error_pay_canceled";

		case ND_COM_PLATFORM_ERROR_LEADERBOARD_NOT_EXIST:
			return "error_leaderboard_not_exist";
		case ND_COM_PLATFORM_ERROR_LEADERBOARD_USERLIST_NOT_EXIST:
			return "error_leaderboard_userlist_not_exist";
		case ND_COM_PLATFORM_ERROR_FRIENDS_NOBODY_PLAYING:
			return "error_friends_nobody_playing";
		case ND_COM_PLATFORM_ERROR_ACHIEVEMENT_NOT_EXIST:
			return "error_achievement_not_exist";

		case ND_COM_PLATFORM_ERROR_91_HAS_NOT_BIND_3RD:
			return "error_91_has_not_bind_3rd";
		case ND_COM_PLATFORM_ERROR_3RD_SHARED_CONTENT_REPEAT:
			return "error_3rd_shared_content_repeat";
		case ND_COM_PLATFORM_ERROR_PAY_ORDER_NOT_EXIST:
			return "error_pay_order_not_exist";
		case ND_COM_PLATFORM_ERROR_NOT_MY_REQUEST_FOR_PAY:
			return "error_not_my_request_for_pay";
		case ND_COM_PLATFORM_ERROR_NOT_MY_FRIEND_ANYMORE:
			return "error_not_my_friend_anymore";

		case ND_COM_PLATFORM_ERROR_3RD_ACCOUNT_HAS_NO_FRIENDS:
			return "error_3rd_account_has_no_friends";
		case ND_COM_PLATFORM_ERROR_91_HAS_BIND_3RD:
			return "error_91_has_bind_3rd";
		case ND_COM_PLATFORM_ERROR_3RD_HAS_BIND_OTHER_91:
			return "error_3rd_has_bind_other_91";
		case ND_COM_PLATFORM_ERROR_3RD_HAS_BIND_91:
			return "error_3rd_has_bind_91";
		case ND_COM_PLATFORM_ERROR_3RD_ACCOUNT_INFO_LOST:
			return "error_3rd_account_info_lost";
		case ND_COM_PLATFORM_ERROR_CAN_NOT_VERIFY_3RD_ACCOUNT:
			return "error_can_not_verify_3rd_account";
		case ND_COM_PLATFORM_ERROR_91_ACCOUNT_EXCEPTION:
			return "error_91_account_exception";

		case ND_COM_PLATFORM_ERROR_3RD_SESSION_ID_INVALID:
			return "error_3rd_session_id_invalid";

		case ND_COM_PLATFORM_ERROR_VG_CATEGORY_INVALID:
			return "error_vg_category_invalid";
		case ND_COM_PLATFORM_ERROR_VG_FEE_TYPE_INVALID:
			return "error_vg_fee_type_invalid";

		case ND_COM_PLATFORM_ERROR_3RD_INFO_INVALID:
			return "error_3rd_info_invalid";
		case ND_COM_PLATFORM_ERROR_CANNOT_UNBIND_LOGINED_3RD_ACCOUNT:
			return "error_cannot_unbind_logined_3rd_account";
		case ND_COM_PLATFORM_ERROR_VERFIER_INVALID:
			return "error_verfier_invalid";

		case ND_COM_PLATFORM_ERROR_REPEAT_SENDING:
			return "error_repeat_sending";
		case ND_COM_PLATFORM_ERROR_PAY_REQUEST_TIMEOUT:
			return "error_pay_request_timeout";
		case ND_COM_PLATFORM_ERROR_VG_PRODUCT_USE_SIGN_INVALID:
			return "error_vg_product_use_sign_invalid";
		case ND_COM_PLATFORM_ERROR_VG_PRODUCT_ID_INVALID:
			return "error_vg_product_id_invalid";

		case ND_COM_PLATFORM_ERROR_VG_MONEY_TYPE_FAILED:
			return "error_vg_money_type_failed";
		case ND_COM_PLATFORM_ERROR_VG_ORDER_FAILED:
			return "error_vg_order_failed";
		case ND_COM_PLATFORM_ERROR_VG_BACK_FROM_RECHARGE:
			return "error_vg_back_from_recharge";

		case ND_COM_PLATFORM_ERROR_BD_INVALID_PHONE_NUM:
			return "error_bd_invalid_phone_num";
		case ND_COM_PLATFORM_ERROR_BD_ACCOUNT_HAS_BIND_PHONE_NUM:
			return "error_bd_account_has_bind_phone_num";
		case ND_COM_PLATFORM_ERROR_BD_PHONE_NUM_HAS_BIND_ACCOUNT:
			return "error_bd_phone_num_has_bind_account";
		case ND_COM_PLATFORM_ERROR_BD_PHONE_NUM_DIDNOT_BIND:
			return "error_bd_phone_num_didnot_bind";
		case ND_COM_PLATFORM_ERROR_BD_WRONG_BIND_PHONE_NUM:
			return "error_bd_wrong_bind_phone_num";
		case ND_COM_PLATFORM_ERROR_BD_WRONG_SMS_VERIFY_CODE:
			return "error_bd_wrong_sms_verify_code";
		case ND_COM_PLATFORM_ERROR_BD_SMS_VERIFY_CODE_OUT_EXPIRE:
			return "error_bd_sms_verify_code_out_expire";
		case ND_COM_PLATFORM_ERROR_BD_PHONE_VERIFY_FAIL:
			return "error_bd_phone_verify_fail";
		case ND_COM_PLATFORM_ERROR_BD_UNFIT_LOTTERY_CONDITION:
			return "error_bd_unfit_lottery_condition";
		case ND_COM_PLATFORM_ERROR_BD_HAS_LOTTERY:
			return "error_bd_has_lottery";
		case ND_COM_PLATFORM_ERROR_BD_OUT_NUM_SMS_SENDED:
			return "error_bd_out_num_sms_sended";
		case ND_COM_PLATFORM_ERROR_BD_VIP_CANNOT_RESTE_ON_PHONE:
			return "error_bd_vip_cannot_reste_on_phone";
		case ND_COM_PLATFORM_ERROR_BD_DIFFERENT_PHONE_NUM:
			return "error_bd_different_phone_num";

		case ND_COM_PLATFORM_ERROR_HAS_ASSOCIATE_91:
			return "error_has_associate_91";
		case ND_COM_PLATFORM_ERROR_NO_NEED_BECOME_REGULAR:
			return "error_no_need_become_regular";
		case ND_COM_PLATFORM_ERROR_UIN_INVALID:
			return "error_uin_invalid";
		case ND_COM_PLATFORM_ERROR_GUEST_NOT_PERMITTED:
			return "error_guest_not_permitted";

		case ND_COM_PLATFORM_ERROR_HIGH_FREQUENT_OPERATION:
			return "error_high_frequent_operation";
		case ND_COM_PLATFORM_ERROR_PROMOTED_APP_NOT_TOUCHED:
			return "error_promoted_app_not_touched";

		case ND_COM_PLATFORM_ERROR_3RD_AUTH_FAILED:
			return "error_3rd_auth_failed";
		case ND_COM_PLATFORM_ERROR_3RD_REAUTH_FAILDED:
			return "error_3rd_reauth_failded";
	}
	return "unknow_error_code";
}
