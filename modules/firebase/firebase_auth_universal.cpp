/*************************************************************************/
/*  firebase_auth.cpp                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "firebase_auth.h"

FirebaseAuth::~FirebaseAuth() {
}

void FirebaseAuth::wait_for_future_user(const firebase::Future<User *> &future, const String &signal) {
	future.OnCompletion([this, signal](const firebase::Future<User *> &result) {
		Dictionary dict;
		dict["auth_error"] = result.error();
		if (result.error() != kAuthErrorNone) {
			dict["error_message"] = result.error_message();
		}
		// make sure signal emit in next frame
		call_deferred("emit_signal", signal, dict);
	});
}

bool FirebaseAuth::has_user() {
	return auth->current_user() != nullptr;
}

Array FirebaseAuth::get_provider_ids() {
	Array providers;
	if (!has_user()) {
		return providers;
	}
	User *user = auth->current_user();
	for (auto it = user->provider_data().begin(); it != user->provider_data().end(); ++it) {
		UserInfoInterface *profile = *it;
		providers.append(String(profile->provider_id().c_str()));
	}
	return providers;
}

void FirebaseAuth::retrieve_token(const bool force_refresh) {
	if (!has_user()) {
		Dictionary dict;
		dict["auth_error"] = kAuthErrorUserNotFound;
		dict["error_message"] = "sign in first";
		call_deferred("emit_signal", "on_retrieve_token_complete", dict);
		return;
	}
	firebase::Future<std::string> result = auth->current_user()->GetToken(force_refresh);
	result.OnCompletion([this](const firebase::Future<std::string> &result) {
		Dictionary dict;
		dict["auth_error"] = result.error();
		if (result.error() == kAuthErrorNone) {
			dict["token"] = result.result()->c_str();
		} else {
			dict["error_message"] = result.error_message();
		}
		call_deferred("emit_signal", "on_retrieve_token_complete", dict);
	});
}

void FirebaseAuth::sign_in_anonymously() {
	wait_for_future_user(auth->SignInAnonymously(), "on_sign_in_complete");
}

void FirebaseAuth::sign_in_with_custom_token(const String &custom_token) {
	wait_for_future_user(auth->SignInWithCustomToken(custom_token.utf8().get_data()), "on_sign_in_complete");
}

void FirebaseAuth::sign_in_with_google_id_token(const String &google_id_token) {
	Credential credential = GoogleAuthProvider::GetCredential(google_id_token.utf8().get_data(), nullptr);
	wait_for_future_user(auth->SignInWithCredential(credential), "on_sign_in_complete");
}

void FirebaseAuth::sign_in_with_facebook_access_token(const String &facebook_access_token) {
	Credential credential = FacebookAuthProvider::GetCredential(facebook_access_token.utf8().get_data());
	wait_for_future_user(auth->SignInWithCredential(credential), "on_sign_in_complete");
}

void FirebaseAuth::sign_out() {
	auth->SignOut();
}

void FirebaseAuth::link_with_google_id_token(const String &google_id_token) {
	if (!has_user()) {
		Dictionary dict;
		dict["auth_error"] = kAuthErrorUserNotFound;
		dict["error_message"] = "sign in first";
		call_deferred("emit_signal", "on_link_complete", dict);
		return;
	}
	Credential credential = GoogleAuthProvider::GetCredential(google_id_token.utf8().get_data(), nullptr);
	User *current_user = auth->current_user();
	wait_for_future_user(current_user->LinkWithCredential(credential), "on_link_complete");
}

void FirebaseAuth::link_with_facebook_access_token(const String &facebook_access_token) {
	if (!has_user()) {
		Dictionary dict;
		dict["auth_error"] = kAuthErrorUserNotFound;
		dict["error_message"] = "sign in first";
		call_deferred("emit_signal", "on_link_complete", dict);
		return;
	}
	Credential credential = FacebookAuthProvider::GetCredential(facebook_access_token.utf8().get_data());
	User *current_user = auth->current_user();
	wait_for_future_user(current_user->LinkWithCredential(credential), "on_link_complete");
}

void FirebaseAuth::StateListner::OnAuthStateChanged(Auth *auth) {
	firebase_auth->emit_signal("on_auth_state_changed", auth->current_user() != nullptr);
}

void FirebaseAuth::TokenListner::OnIdTokenChanged(Auth *auth) {
	firebase_auth->emit_signal("on_id_token_changed");
}

void FirebaseAuth::_bind_methods() {
	ClassDB::bind_method(D_METHOD("initialize", "config"), &FirebaseAuth::initialize, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("has_user"), &FirebaseAuth::has_user);
	ClassDB::bind_method(D_METHOD("get_provider_ids"), &FirebaseAuth::get_provider_ids);
	ClassDB::bind_method(D_METHOD("retrieve_token"), &FirebaseAuth::retrieve_token, DEFVAL(true));
	ClassDB::bind_method(D_METHOD("sign_in_anonymously"), &FirebaseAuth::sign_in_anonymously);
	ClassDB::bind_method(D_METHOD("sign_in_with_custom_token", "custom_token"), &FirebaseAuth::sign_in_with_custom_token, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("sign_in_with_google_id_token", "google_id_token"), &FirebaseAuth::sign_in_with_google_id_token, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("sign_in_with_facebook_access_token", "facebook_access_token"),
			&FirebaseAuth::sign_in_with_facebook_access_token, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("sign_out"), &FirebaseAuth::sign_out);
	ClassDB::bind_method(D_METHOD("request_google_id_token"), &FirebaseAuth::request_google_id_token);
	ClassDB::bind_method(D_METHOD("link_with_google_id_token", "google_id_token"), &FirebaseAuth::link_with_google_id_token, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("request_facebook_access_token"), &FirebaseAuth::request_facebook_access_token);
	ClassDB::bind_method(D_METHOD("link_with_facebook_access_token", "facebook_access_token"), &FirebaseAuth::link_with_facebook_access_token, DEFVAL(""));

	ADD_SIGNAL(MethodInfo("on_retrieve_token_complete", PropertyInfo(Variant::DICTIONARY, "dict")));
	ADD_SIGNAL(MethodInfo("on_sign_in_complete", PropertyInfo(Variant::DICTIONARY, "dict")));
	ADD_SIGNAL(MethodInfo("on_request_google_id_token_complete", PropertyInfo(Variant::DICTIONARY, "dict")));
	ADD_SIGNAL(MethodInfo("on_request_facebook_access_token_complete",
			PropertyInfo(Variant::DICTIONARY, "dict")));
	ADD_SIGNAL(MethodInfo("on_link_complete", PropertyInfo(Variant::DICTIONARY, "dict")));
	ADD_SIGNAL(MethodInfo("on_auth_state_changed", PropertyInfo(Variant::BOOL, "sign_in")));
	ADD_SIGNAL(MethodInfo("on_id_token_changed"));

	BIND_ENUM_CONSTANT(kAuthErrorNone);
	BIND_ENUM_CONSTANT(kAuthErrorUnimplemented);
	BIND_ENUM_CONSTANT(kAuthErrorFailure);
	BIND_ENUM_CONSTANT(kAuthErrorInvalidCustomToken);
	BIND_ENUM_CONSTANT(kAuthErrorCustomTokenMismatch);
	BIND_ENUM_CONSTANT(kAuthErrorInvalidCredential);
	BIND_ENUM_CONSTANT(kAuthErrorUserDisabled);
	BIND_ENUM_CONSTANT(kAuthErrorAccountExistsWithDifferentCredentials);
	BIND_ENUM_CONSTANT(kAuthErrorOperationNotAllowed);
	BIND_ENUM_CONSTANT(kAuthErrorEmailAlreadyInUse);
	BIND_ENUM_CONSTANT(kAuthErrorRequiresRecentLogin);
	BIND_ENUM_CONSTANT(kAuthErrorCredentialAlreadyInUse);
	BIND_ENUM_CONSTANT(kAuthErrorInvalidEmail);
	BIND_ENUM_CONSTANT(kAuthErrorWrongPassword);
	BIND_ENUM_CONSTANT(kAuthErrorTooManyRequests);
	BIND_ENUM_CONSTANT(kAuthErrorUserNotFound);
	BIND_ENUM_CONSTANT(kAuthErrorProviderAlreadyLinked);
	BIND_ENUM_CONSTANT(kAuthErrorNoSuchProvider);
	BIND_ENUM_CONSTANT(kAuthErrorInvalidUserToken);
	BIND_ENUM_CONSTANT(kAuthErrorUserTokenExpired);
	BIND_ENUM_CONSTANT(kAuthErrorNetworkRequestFailed);
	BIND_ENUM_CONSTANT(kAuthErrorInvalidApiKey);
	BIND_ENUM_CONSTANT(kAuthErrorAppNotAuthorized);
	BIND_ENUM_CONSTANT(kAuthErrorUserMismatch);
	BIND_ENUM_CONSTANT(kAuthErrorWeakPassword);
	BIND_ENUM_CONSTANT(kAuthErrorNoSignedInUser);
	BIND_ENUM_CONSTANT(kAuthErrorApiNotAvailable);
	BIND_ENUM_CONSTANT(kAuthErrorExpiredActionCode);
	BIND_ENUM_CONSTANT(kAuthErrorInvalidActionCode);
	BIND_ENUM_CONSTANT(kAuthErrorInvalidMessagePayload);
	BIND_ENUM_CONSTANT(kAuthErrorInvalidPhoneNumber);
	BIND_ENUM_CONSTANT(kAuthErrorMissingPhoneNumber);
	BIND_ENUM_CONSTANT(kAuthErrorInvalidRecipientEmail);
	BIND_ENUM_CONSTANT(kAuthErrorInvalidSender);
	BIND_ENUM_CONSTANT(kAuthErrorInvalidVerificationCode);
	BIND_ENUM_CONSTANT(kAuthErrorInvalidVerificationId);
	BIND_ENUM_CONSTANT(kAuthErrorMissingVerificationCode);
	BIND_ENUM_CONSTANT(kAuthErrorMissingVerificationId);
	BIND_ENUM_CONSTANT(kAuthErrorMissingEmail);
	BIND_ENUM_CONSTANT(kAuthErrorMissingPassword);
	BIND_ENUM_CONSTANT(kAuthErrorQuotaExceeded);
	BIND_ENUM_CONSTANT(kAuthErrorRetryPhoneAuth);
	BIND_ENUM_CONSTANT(kAuthErrorSessionExpired);
	BIND_ENUM_CONSTANT(kAuthErrorAppNotVerified);
	BIND_ENUM_CONSTANT(kAuthErrorAppVerificationFailed);
	BIND_ENUM_CONSTANT(kAuthErrorCaptchaCheckFailed);
	BIND_ENUM_CONSTANT(kAuthErrorInvalidAppCredential);
	BIND_ENUM_CONSTANT(kAuthErrorMissingAppCredential);
	BIND_ENUM_CONSTANT(kAuthErrorInvalidClientId);
	BIND_ENUM_CONSTANT(kAuthErrorInvalidContinueUri);
	BIND_ENUM_CONSTANT(kAuthErrorMissingContinueUri);
	BIND_ENUM_CONSTANT(kAuthErrorKeychainError);
	BIND_ENUM_CONSTANT(kAuthErrorMissingAppToken);
	BIND_ENUM_CONSTANT(kAuthErrorMissingIosBundleId);
	BIND_ENUM_CONSTANT(kAuthErrorNotificationNotForwarded);
	BIND_ENUM_CONSTANT(kAuthErrorUnauthorizedDomain);
	BIND_ENUM_CONSTANT(kAuthErrorWebContextAlreadyPresented);
	BIND_ENUM_CONSTANT(kAuthErrorWebContextCancelled);
	BIND_ENUM_CONSTANT(kAuthErrorDynamicLinkNotActivated);
	BIND_ENUM_CONSTANT(kAuthErrorCancelled);
	BIND_ENUM_CONSTANT(kAuthErrorInvalidProviderId);
	BIND_ENUM_CONSTANT(kAuthErrorWebInternalError);
	BIND_ENUM_CONSTANT(kAuthErrorWebStorateUnsupported);
	BIND_ENUM_CONSTANT(kAuthErrorTenantIdMismatch);
	BIND_ENUM_CONSTANT(kAuthErrorUnsupportedTenantOperation);
	BIND_ENUM_CONSTANT(kAuthErrorInvalidLinkDomain);
	BIND_ENUM_CONSTANT(kAuthErrorRejectedCredential);
}
