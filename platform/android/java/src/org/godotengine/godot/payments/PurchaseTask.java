/*************************************************************************/
/*  PurchaseTask.java                                                    */
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
package org.godotengine.godot.payments;

import org.json.JSONException;
import org.json.JSONObject;

import org.godotengine.godot.GodotLib;
import org.godotengine.godot.utils.Crypt;
import com.android.vending.billing.IInAppBillingService;

import android.app.Activity;
import android.app.PendingIntent;
import android.app.ProgressDialog;
import android.content.Context;
import android.content.Intent;
import android.content.IntentSender.SendIntentException;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.RemoteException;
import android.util.Log;

abstract public class PurchaseTask {

	private static String TAG = "PurchaseTask";

	private Activity context;

	private IInAppBillingService mService;
	public PurchaseTask(IInAppBillingService mService, Activity context) {
		this.context = context;
		this.mService = mService;
	}

	private boolean isLooping = false;

	public void purchase(final String sku, final String transactionId) {
		Log.d(TAG, "Starting purchase for: " + sku);
		PaymentsCache pc = new PaymentsCache(context);
		Boolean isBlocked = pc.getConsumableFlag("block", sku);

		final String hash = transactionId;

		Bundle buyIntentBundle;
		try {
			buyIntentBundle = mService.getBuyIntent(3, context.getApplicationContext().getPackageName(), sku, PaymentsManager.ITEM_TYPE_INAPP, hash);
		} catch (RemoteException e) {
			error(e.getMessage());
			return;
		}
		Object rc = buyIntentBundle.get(PaymentsManager.RESPONSE_CODE);
		int responseCode = 0;
		if (rc == null) {
			responseCode = PaymentsManager.BILLING_RESPONSE_RESULT_OK;
		} else if (rc instanceof Integer) {
			responseCode = ((Integer)rc).intValue();
		} else if (rc instanceof Long) {
			responseCode = (int)((Long)rc).longValue();
		}

		if (responseCode == PaymentsManager.BILLING_RESPONSE_RESULT_USER_CANCELED || responseCode == PaymentsManager.BILLING_RESPONSE_RESULT_BILLING_UNAVAILABLE || responseCode == PaymentsManager.BILLING_RESPONSE_RESULT_ITEM_UNAVAILABLE) {
			canceled();
			return;
		}
		if (responseCode == PaymentsManager.BILLING_RESPONSE_RESULT_ITEM_ALREADY_OWNED) {
			alreadyOwned();
			return;
		}

		PendingIntent pendingIntent = buyIntentBundle.getParcelable(PaymentsManager.RESPONSE_BUY_INTENT);
		pc.setConsumableValue("validation_hash", sku, hash);
		try {
			context.startIntentSenderForResult(
					pendingIntent.getIntentSender(),
					PaymentsManager.REQUEST_CODE_FOR_PURCHASE,
					new Intent(),
					Integer.valueOf(0), Integer.valueOf(0),
					Integer.valueOf(0));
		} catch (SendIntentException e) {
			error(e.getMessage());
		}
	}

	abstract protected void error(String message);
	abstract protected void canceled();
	abstract protected void alreadyOwned();
}
