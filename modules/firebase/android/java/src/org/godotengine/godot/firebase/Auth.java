package org.godotengine.godot.firebase;

import android.app.Activity;
import android.content.Intent;
import android.util.Log;
import com.facebook.AccessToken;
import com.facebook.CallbackManager;
import com.facebook.FacebookCallback;
import com.facebook.FacebookException;
import com.facebook.login.LoginManager;
import com.facebook.login.LoginResult;
import com.godot.game.R;
import com.google.android.gms.auth.api.signin.GoogleSignIn;
import com.google.android.gms.auth.api.signin.GoogleSignInAccount;
import com.google.android.gms.auth.api.signin.GoogleSignInOptions;
import com.google.android.gms.common.api.ApiException;
import com.google.android.gms.tasks.Task;
import java.util.Arrays;
import org.godotengine.godot.Dictionary;
import org.godotengine.godot.Godot;
import org.godotengine.godot.GodotLib;

public class Auth extends Godot.SingletonBase {
	protected static final int RC_GOOGLE_ID_TOKEN = 2010;

	private static final int AUTH_ERROR_NONE = 0;
	private static final int AUTH_ERROR_UNIMPLEMENTED = -1;
	private static final int AUTH_ERROR_FAILURE = 1;

	private static final String TAG = "GodotFirebaseAuth";

	private Godot activity;
	private GoogleSignInOptions gso;
	private CallbackManager fcm;
	private int callbackId = 0;

	static public Godot.SingletonBase initialize(Activity p_activity) {
		return new Auth(p_activity);
	}

	public Auth(Activity p_activity) {
		registerClass("GodotFirebaseAuth", new String[] { "initFirebaseApp", "requestGoogleIdToken", "requestFacebookAccessToken" });
		activity = (Godot)p_activity;
	}

	public void requestGoogleIdToken(final int callbackId) {
		if (this.callbackId > 0) {
			Log.e(TAG, "prev operation not finished. ignore this request");
			return;
		}
		this.callbackId = callbackId;
		if (gso == null) {
			gso = new GoogleSignInOptions.Builder(GoogleSignInOptions.DEFAULT_SIGN_IN)
						  .requestIdToken(activity.getString(R.string.default_web_client_id))
						  .build();
		}
		activity.startActivityForResult(GoogleSignIn.getClient(activity, gso).getSignInIntent(), RC_GOOGLE_ID_TOKEN);
	}

	public void requestFacebookAccessToken(final int callbackId) {
		if (this.callbackId > 0) {
			Log.e(TAG, "prev operation not finished. ignore this request");
			return;
		}
		this.callbackId = callbackId;
		AccessToken accessToken = AccessToken.getCurrentAccessToken();
		if (accessToken != null && !accessToken.isExpired()) {
			Dictionary dict = new Dictionary();
			dict.put("auth_error", AUTH_ERROR_NONE);
			dict.put("access_token", accessToken.getToken());
			emitSignalDeferred("on_request_facebook_access_token_complete", dict);
			return;
		}
		if (fcm == null) {
			fcm = CallbackManager.Factory.create();
			LoginManager.getInstance().registerCallback(fcm, new FacebookCallback<LoginResult>() {
				@Override
				public void onSuccess(LoginResult loginResult) {
					Dictionary dict = new Dictionary();
					dict.put("auth_error", AUTH_ERROR_NONE);
					dict.put("access_token", loginResult.getAccessToken().getToken());
					emitSignalDeferred("on_request_facebook_access_token_complete", dict);
				}

				@Override
				public void onCancel() {
					Dictionary dict = new Dictionary();
					dict.put("auth_error", AUTH_ERROR_FAILURE);
					dict.put("error_message", "user canceled");
					emitSignalDeferred("on_request_facebook_access_token_complete", dict);
				}

				@Override
				public void onError(FacebookException error) {
					Dictionary dict = new Dictionary();
					dict.put("auth_error", AUTH_ERROR_FAILURE);
					dict.put("error_message", error.getMessage());
					emitSignalDeferred("on_request_facebook_access_token_complete", dict);
				}
			});
		}
		LoginManager.getInstance().logInWithReadPermissions(activity, Arrays.asList("public_profile"));
	}

	protected void emitSignalDeferred(String signal, Dictionary dict) {
		assert callbackId > 0;
		GodotLib.calldeferred(callbackId, "emit_signal", new Object[] { signal, dict });
		callbackId = 0;
	}

	protected void onMainActivityResult(int requestCode, int resultCode, Intent data) {
		if (fcm != null) {
			fcm.onActivityResult(requestCode, resultCode, data);
		}
		if (requestCode == RC_GOOGLE_ID_TOKEN) {
			Task<GoogleSignInAccount> task = GoogleSignIn.getSignedInAccountFromIntent(data);
			Dictionary dict = new Dictionary();
			try {
				GoogleSignInAccount account = task.getResult(ApiException.class);
				dict.put("auth_error", AUTH_ERROR_NONE);
				dict.put("google_id_token", account.getIdToken());
			} catch (ApiException e) {
				dict.put("auth_error", AUTH_ERROR_FAILURE);
				dict.put("error_message", e.getMessage());
			}
			emitSignalDeferred("on_request_google_id_token_complete", dict);
		}
	}

	protected void onMainRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {}
}
