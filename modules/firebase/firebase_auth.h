/*************************************************************************/
/*  firebase_auth.h                                                      */
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

#ifndef FIREBASE_AUTH_H
#define FIREBASE_AUTH_H

#include "firebase/auth.h"
#include "scene/main/node.h"

using namespace firebase::auth;

VARIANT_ENUM_CAST(AuthError);

class FirebaseAuth : public Node {

	GDCLASS(FirebaseAuth, Node);

private:
	firebase::App *app;
	firebase::auth::Auth *auth;

	class StateListner : public AuthStateListener {
	private:
		FirebaseAuth *firebase_auth;

	public:
		virtual void OnAuthStateChanged(Auth *auth);
		StateListner(FirebaseAuth *cls) :
				firebase_auth(cls) {}
	} state_listner;

	class TokenListner : public IdTokenListener {
	private:
		FirebaseAuth *firebase_auth;

	public:
		virtual void OnIdTokenChanged(Auth *auth);
		TokenListner(FirebaseAuth *cls) :
				firebase_auth(cls) {}
	} token_listner;

	void wait_for_future_user(const firebase::Future<User *> &future, const String &signal);

protected:
	static void _bind_methods();

public:
	void initialize(const String &config);
	bool has_user();
	Array get_provider_ids();
	void retrieve_token(const bool force_refresh);
	void sign_in_anonymously();
	void sign_in_with_custom_token(const String &custom_token);
	void sign_in_with_google_id_token(const String &google_id_token);
	void sign_in_with_facebook_access_token(const String &facebook_access_token);
	void sign_out();
	void request_google_id_token();
	void link_with_google_id_token(const String &google_id_token);
	void request_facebook_access_token();
	void link_with_facebook_access_token(const String &facebook_access_token);

	FirebaseAuth();
	~FirebaseAuth();
};

#endif // FIREBASE_AUTH_H
