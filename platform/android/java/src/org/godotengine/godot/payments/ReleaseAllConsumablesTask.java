/*************************************************************************/
/*  ReleaseAllConsumablesTask.java                                       */
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

import android.content.Context;
import android.os.AsyncTask;
import android.os.Bundle;
import android.util.Log;
import com.android.vending.billing.IInAppBillingService;
import java.lang.ref.WeakReference;
import java.util.ArrayList;
import org.json.JSONException;
import org.json.JSONObject;

abstract public class ReleaseAllConsumablesTask {

	private Context context;
	private IInAppBillingService mService;

	private static class ReleaseAllConsumablesAsyncTask extends AsyncTask<String, String, String> {

		private WeakReference<ReleaseAllConsumablesTask> mTask;
		private String mSku;
		private String mReceipt;
		private String mSignature;
		private String mToken;

		ReleaseAllConsumablesAsyncTask(ReleaseAllConsumablesTask task, String sku, String receipt, String signature, String token) {
			mTask = new WeakReference<ReleaseAllConsumablesTask>(task);

			mSku = sku;
			mReceipt = receipt;
			mSignature = signature;
			mToken = token;
		}

		@Override
		protected String doInBackground(String... params) {
			ReleaseAllConsumablesTask consume = mTask.get();
			if (consume != null) {
				return consume.doInBackground(mToken);
			}
			return null;
		}

		@Override
		protected void onPostExecute(String param) {
			ReleaseAllConsumablesTask consume = mTask.get();
			if (consume != null) {
				consume.success(mSku, mReceipt, mSignature, mToken);
			}
		}
	}

	public ReleaseAllConsumablesTask(IInAppBillingService mService, Context context) {
		this.context = context;
		this.mService = mService;
	}

	public void consumeItAll() {
		try {
			//Log.d("godot", "consumeItall for " + context.getPackageName());
			Bundle bundle = mService.getPurchases(3, context.getPackageName(), "inapp", null);

			if (bundle.getInt("RESPONSE_CODE") == 0) {

				final ArrayList<String> myPurchases = bundle.getStringArrayList("INAPP_PURCHASE_DATA_LIST");
				final ArrayList<String> mySignatures = bundle.getStringArrayList("INAPP_DATA_SIGNATURE_LIST");

				if (myPurchases == null || myPurchases.size() == 0) {
					//Log.d("godot", "No purchases!");
					notRequired();
					return;
				}

				//Log.d("godot", "# products to be consumed:" + myPurchases.size());
				for (int i = 0; i < myPurchases.size(); i++) {

					try {
						String receipt = myPurchases.get(i);
						JSONObject inappPurchaseData = new JSONObject(receipt);
						String sku = inappPurchaseData.getString("productId");
						String token = inappPurchaseData.getString("purchaseToken");
						String signature = mySignatures.get(i);
						//Log.d("godot", "A punto de consumir un item con token:" + token + "\n" + receipt);
						new ReleaseAllConsumablesAsyncTask(this, sku, receipt, signature, token).execute();
					} catch (JSONException e) {
					}
				}
			}
		} catch (Exception e) {
			Log.d("godot", "Error releasing products:" + e.getClass().getName() + ":" + e.getMessage());
		}
	}

	private String doInBackground(String token) {
		try {
			//Log.d("godot", "Requesting to consume an item with token ." + token);
			int response = mService.consumePurchase(3, context.getPackageName(), token);
			//Log.d("godot", "consumePurchase response: " + response);
			if (response == 0 || response == 8) {
				return null;
			}
		} catch (Exception e) {
			Log.d("godot", "Error " + e.getClass().getName() + ":" + e.getMessage());
		}
		return null;
	}

	abstract protected void success(String sku, String receipt, String signature, String token);
	abstract protected void error(String message);
	abstract protected void notRequired();
}
