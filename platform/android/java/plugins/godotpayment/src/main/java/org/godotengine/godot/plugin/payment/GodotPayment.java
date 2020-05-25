/*************************************************************************/
/*  GodotPayment.java                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

package org.godotengine.godot.plugin.payment;

import org.godotengine.godot.Dictionary;
import org.godotengine.godot.Godot;
import org.godotengine.godot.plugin.GodotPlugin;
import org.godotengine.godot.plugin.SignalInfo;
import org.godotengine.godot.plugin.payment.utils.GodotPaymentUtils;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.collection.ArraySet;

import com.android.billingclient.api.AcknowledgePurchaseParams;
import com.android.billingclient.api.AcknowledgePurchaseResponseListener;
import com.android.billingclient.api.BillingClient;
import com.android.billingclient.api.BillingClientStateListener;
import com.android.billingclient.api.BillingFlowParams;
import com.android.billingclient.api.BillingResult;
import com.android.billingclient.api.ConsumeParams;
import com.android.billingclient.api.ConsumeResponseListener;
import com.android.billingclient.api.Purchase;
import com.android.billingclient.api.PurchasesUpdatedListener;
import com.android.billingclient.api.SkuDetails;
import com.android.billingclient.api.SkuDetailsParams;
import com.android.billingclient.api.SkuDetailsResponseListener;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Set;

public class GodotPayment extends GodotPlugin implements PurchasesUpdatedListener, BillingClientStateListener {

	private final BillingClient billingClient;
	private final HashMap<String, SkuDetails> skuDetailsCache = new HashMap<>(); // sku → SkuDetails

	public GodotPayment(Godot godot) {
		super(godot);

		billingClient = BillingClient
								.newBuilder(getGodot())
								.enablePendingPurchases()
								.setListener(this)
								.build();
	}

	public void startConnection() {
		billingClient.startConnection(this);
	}

	public void endConnection() {
		billingClient.endConnection();
	}

	public boolean isReady() {
		return this.billingClient.isReady();
	}

	public Dictionary queryPurchases(String type) {
		Purchase.PurchasesResult result = billingClient.queryPurchases(type);

		Dictionary returnValue = new Dictionary();
		if (result.getBillingResult().getResponseCode() == BillingClient.BillingResponseCode.OK) {
			returnValue.put("status", 0); // OK = 0
			returnValue.put("purchases", GodotPaymentUtils.convertPurchaseListToDictionaryObjectArray(result.getPurchasesList()));
		} else {
			returnValue.put("status", 1); // FAILED = 1
			returnValue.put("response_code", result.getBillingResult().getResponseCode());
			returnValue.put("debug_message", result.getBillingResult().getDebugMessage());
		}

		return returnValue;
	}

	public void querySkuDetails(final String[] list, String type) {
		List<String> skuList = Arrays.asList(list);

		SkuDetailsParams.Builder params = SkuDetailsParams.newBuilder()
												  .setSkusList(skuList)
												  .setType(type);

		billingClient.querySkuDetailsAsync(params.build(), new SkuDetailsResponseListener() {
			@Override
			public void onSkuDetailsResponse(BillingResult billingResult,
					List<SkuDetails> skuDetailsList) {
				if (billingResult.getResponseCode() == BillingClient.BillingResponseCode.OK) {
					for (SkuDetails skuDetails : skuDetailsList) {
						skuDetailsCache.put(skuDetails.getSku(), skuDetails);
					}
					emitSignal("sku_details_query_completed", (Object)GodotPaymentUtils.convertSkuDetailsListToDictionaryObjectArray(skuDetailsList));
				} else {
					emitSignal("sku_details_query_error", billingResult.getResponseCode(), billingResult.getDebugMessage(), list);
				}
			}
		});
	}

	public void acknowledgePurchase(final String purchaseToken) {
		AcknowledgePurchaseParams acknowledgePurchaseParams =
				AcknowledgePurchaseParams.newBuilder()
						.setPurchaseToken(purchaseToken)
						.build();
		billingClient.acknowledgePurchase(acknowledgePurchaseParams, new AcknowledgePurchaseResponseListener() {
			@Override
			public void onAcknowledgePurchaseResponse(BillingResult billingResult) {
				if (billingResult.getResponseCode() == BillingClient.BillingResponseCode.OK) {
					emitSignal("purchase_acknowledged", purchaseToken);
				} else {
					emitSignal("purchase_acknowledgement_error", billingResult.getResponseCode(), billingResult.getDebugMessage(), purchaseToken);
				}
			}
		});
	}

	public void consumePurchase(String purchaseToken) {
		ConsumeParams consumeParams = ConsumeParams.newBuilder()
											  .setPurchaseToken(purchaseToken)
											  .build();

		billingClient.consumeAsync(consumeParams, new ConsumeResponseListener() {
			@Override
			public void onConsumeResponse(BillingResult billingResult, String purchaseToken) {
				if (billingResult.getResponseCode() == BillingClient.BillingResponseCode.OK) {
					emitSignal("purchase_consumed", purchaseToken);
				} else {
					emitSignal("purchase_consumption_error", billingResult.getResponseCode(), billingResult.getDebugMessage(), purchaseToken);
				}
			}
		});
	}

	@Override
	public void onBillingSetupFinished(BillingResult billingResult) {
		if (billingResult.getResponseCode() == BillingClient.BillingResponseCode.OK) {
			emitSignal("connected");
		} else {
			emitSignal("connect_error", billingResult.getResponseCode(), billingResult.getDebugMessage());
		}
	}

	@Override
	public void onBillingServiceDisconnected() {
		emitSignal("disconnected");
	}

	public Dictionary purchase(String sku) {
		if (!skuDetailsCache.containsKey(sku)) {
			emitSignal("purchase_error", null, "You must query the sku details and wait for the result before purchasing!");
		}

		SkuDetails skuDetails = skuDetailsCache.get(sku);
		BillingFlowParams purchaseParams = BillingFlowParams.newBuilder()
												   .setSkuDetails(skuDetails)
												   .build();

		BillingResult result = billingClient.launchBillingFlow(getGodot(), purchaseParams);

		Dictionary returnValue = new Dictionary();
		if (result.getResponseCode() == BillingClient.BillingResponseCode.OK) {
			returnValue.put("status", 0); // OK = 0
		} else {
			returnValue.put("status", 1); // FAILED = 1
			returnValue.put("response_code", result.getResponseCode());
			returnValue.put("debug_message", result.getDebugMessage());
		}

		return returnValue;
	}

	@Override
	public void onPurchasesUpdated(final BillingResult billingResult, @Nullable final List<Purchase> list) {
		if (billingResult.getResponseCode() == BillingClient.BillingResponseCode.OK && list != null) {
			emitSignal("purchases_updated", (Object)GodotPaymentUtils.convertPurchaseListToDictionaryObjectArray(list));
		} else {
			emitSignal("purchase_error", billingResult.getResponseCode(), billingResult.getDebugMessage());
		}
	}

	@NonNull
	@Override
	public String getPluginName() {
		return "GodotPayment";
	}

	@NonNull
	@Override
	public List<String> getPluginMethods() {
		return Arrays.asList("startConnection", "endConnection", "purchase", "querySkuDetails", "isReady", "queryPurchases", "acknowledgePurchase");
	}

	@NonNull
	@Override
	public Set<SignalInfo> getPluginSignals() {
		Set<SignalInfo> signals = new ArraySet<>();

		signals.add(new SignalInfo("connected"));
		signals.add(new SignalInfo("disconnected"));
		signals.add(new SignalInfo("connect_error", Integer.class, String.class));
		signals.add(new SignalInfo("purchases_updated", Object[].class));
		signals.add(new SignalInfo("purchase_error", Integer.class, String.class));
		signals.add(new SignalInfo("sku_details_query_completed", Object[].class));
		signals.add(new SignalInfo("sku_details_query_error", Integer.class, String.class, String[].class));
		signals.add(new SignalInfo("purchase_acknowledged", String.class));
		signals.add(new SignalInfo("purchase_acknowledgement_error", Integer.class, String.class, String.class));
		signals.add(new SignalInfo("purchase_consumed", String.class));
		signals.add(new SignalInfo("purchase_consumption_error", Integer.class, String.class, String.class));

		return signals;
	}
}
