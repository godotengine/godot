/*************************************************************************/
/*  GodotPaymentV3.java                                                  */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
package org.godotengine.godot;

import android.app.Activity;
import android.util.Log;

import org.godotengine.godot.payments.PaymentsManager;
import org.json.JSONException;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class GodotPaymentV3 extends Godot.SingletonBase {

	private Godot activity;
	private Integer purchaseCallbackId = 0;
	private String accessToken;
	private String purchaseValidationUrlPrefix;
	private String transactionId;
	private PaymentsManager mPaymentManager;
	private Dictionary mSkuDetails = new Dictionary();

	public void purchase(final String sku, final String transactionId) {
		activity.runOnUiThread(new Runnable() {
			@Override
			public void run() {
				mPaymentManager.requestPurchase(sku, transactionId);
			}
		});
	}

	static public Godot.SingletonBase initialize(Activity p_activity) {

		return new GodotPaymentV3(p_activity);
	}

	public GodotPaymentV3(Activity p_activity) {

		registerClass("GodotPayments", new String[] { "purchase", "setPurchaseCallbackId", "setPurchaseValidationUrlPrefix", "setTransactionId", "getSignature", "consumeUnconsumedPurchases", "requestPurchased", "setAutoConsume", "consume", "querySkuDetails" });
		activity = (Godot)p_activity;
		mPaymentManager = activity.getPaymentsManager();
		mPaymentManager.setBaseSingleton(this);
	}

	public void consumeUnconsumedPurchases() {
		activity.runOnUiThread(new Runnable() {
			@Override
			public void run() {
				mPaymentManager.consumeUnconsumedPurchases();
			}
		});
	}

	private String signature;

	public String getSignature() {
		return this.signature;
	}

	public void callbackSuccess(String ticket, String signature, String sku) {
		GodotLib.calldeferred(purchaseCallbackId, "purchase_success", new Object[] { ticket, signature, sku });
	}

	public void callbackSuccessProductMassConsumed(String ticket, String signature, String sku) {
		Log.d(this.getClass().getName(), "callbackSuccessProductMassConsumed > " + ticket + "," + signature + "," + sku);
		GodotLib.calldeferred(purchaseCallbackId, "consume_success", new Object[] { ticket, signature, sku });
	}

	public void callbackSuccessNoUnconsumedPurchases() {
		GodotLib.calldeferred(purchaseCallbackId, "consume_not_required", new Object[] {});
	}

	public void callbackFailConsume() {
		GodotLib.calldeferred(purchaseCallbackId, "consume_fail", new Object[] {});
	}

	public void callbackFail() {
		GodotLib.calldeferred(purchaseCallbackId, "purchase_fail", new Object[] {});
	}

	public void callbackCancel() {
		GodotLib.calldeferred(purchaseCallbackId, "purchase_cancel", new Object[] {});
	}

	public void callbackAlreadyOwned(String sku) {
		GodotLib.calldeferred(purchaseCallbackId, "purchase_owned", new Object[] { sku });
	}

	public int getPurchaseCallbackId() {
		return purchaseCallbackId;
	}

	public void setPurchaseCallbackId(int purchaseCallbackId) {
		this.purchaseCallbackId = purchaseCallbackId;
	}

	public String getPurchaseValidationUrlPrefix() {
		return this.purchaseValidationUrlPrefix;
	}

	public void setPurchaseValidationUrlPrefix(String url) {
		this.purchaseValidationUrlPrefix = url;
	}

	public String getAccessToken() {
		return accessToken;
	}

	public void setAccessToken(String accessToken) {
		this.accessToken = accessToken;
	}

	public void setTransactionId(String transactionId) {
		this.transactionId = transactionId;
	}

	public String getTransactionId() {
		return this.transactionId;
	}

	// request purchased items are not consumed
	public void requestPurchased() {
		activity.runOnUiThread(new Runnable() {
			@Override
			public void run() {
				mPaymentManager.requestPurchased();
			}
		});
	}

	// callback for requestPurchased()
	public void callbackPurchased(String receipt, String signature, String sku) {
		GodotLib.calldeferred(purchaseCallbackId, "has_purchased", new Object[] { receipt, signature, sku });
	}

	// consume item automatically after purchase. default is true.
	public void setAutoConsume(boolean autoConsume) {
		mPaymentManager.setAutoConsume(autoConsume);
	}

	// consume a specific item
	public void consume(String sku) {
		mPaymentManager.consume(sku);
	}

	// query in app item detail info
	public void querySkuDetails(String[] list) {
		List<String> nKeys = Arrays.asList(list);
		List<String> cKeys = Arrays.asList(mSkuDetails.get_keys());
		ArrayList<String> fKeys = new ArrayList<String>();
		for (String key : nKeys) {
			if (!cKeys.contains(key)) {
				fKeys.add(key);
			}
		}
		if (fKeys.size() > 0) {
			mPaymentManager.querySkuDetails(fKeys.toArray(new String[0]));
		} else {
			completeSkuDetail();
		}
	}

	public void addSkuDetail(String itemJson) {
		JSONObject o = null;
		try {
			o = new JSONObject(itemJson);
			Dictionary item = new Dictionary();
			item.put("type", o.optString("type"));
			item.put("product_id", o.optString("productId"));
			item.put("title", o.optString("title"));
			item.put("description", o.optString("description"));
			item.put("price", o.optString("price"));
			item.put("price_currency_code", o.optString("price_currency_code"));
			item.put("price_amount", 0.000001d * o.optLong("price_amount_micros"));
			mSkuDetails.put(item.get("product_id").toString(), item);
		} catch (JSONException e) {
			e.printStackTrace();
		}
	}

	public void completeSkuDetail() {
		GodotLib.calldeferred(purchaseCallbackId, "sku_details_complete", new Object[] { mSkuDetails });
	}

	public void errorSkuDetail(String errorMessage) {
		GodotLib.calldeferred(purchaseCallbackId, "sku_details_error", new Object[] { errorMessage });
	}
}
