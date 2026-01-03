/**************************************************************************/
/*  GodotDownloaderService.java                                           */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

package org.godotengine.godot;

import android.content.Context;
import android.content.SharedPreferences;
import android.util.Log;

import com.google.android.vending.expansion.downloader.impl.DownloaderService;

/**
 * This class demonstrates the minimal client implementation of the
 * DownloaderService from the Downloader library.
 */
public class GodotDownloaderService extends DownloaderService {
	// stuff for LVL -- MODIFY FOR YOUR APPLICATION!
	private static final String BASE64_PUBLIC_KEY = "REPLACE THIS WITH YOUR PUBLIC KEY";
	// used by the preference obfuscater
	private static final byte[] SALT = new byte[] {
		1, 43, -12, -1, 54, 98,
		-100, -12, 43, 2, -8, -4, 9, 5, -106, -108, -33, 45, -1, 84
	};

	/**
	 * This public key comes from your Android Market publisher account, and it
	 * used by the LVL to validate responses from Market on your behalf.
	 */
	@Override
	public String getPublicKey() {
		SharedPreferences prefs = getApplicationContext().getSharedPreferences("app_data_keys", Context.MODE_PRIVATE);
		Log.d("GODOT", "getting public key:" + prefs.getString("store_public_key", null));
		return prefs.getString("store_public_key", null);

		//return BASE64_PUBLIC_KEY;
	}

	/**
	 * This is used by the preference obfuscater to make sure that your
	 * obfuscated preferences are different than the ones used by other
	 * applications.
	 */
	@Override
	public byte[] getSALT() {
		return SALT;
	}

	/**
	 * Fill this in with the class name for your alarm receiver. We do this
	 * because receivers must be unique across all of Android (it's a good idea
	 * to make sure that your receiver is in your unique package)
	 */
	@Override
	public String getAlarmReceiverClassName() {
		Log.d("GODOT", "getAlarmReceiverClassName()");
		return GodotDownloaderAlarmReceiver.class.getName();
	}
}
