package org.godotengine.godot.plugin.payment.utils;

import org.godotengine.godot.Dictionary;

import com.android.billingclient.api.Purchase;
import com.android.billingclient.api.SkuDetails;

import java.util.List;

public class GodotPaymentUtils {
	public static Dictionary convertPurchaseToDictionary(Purchase purchase) {
		Dictionary dictionary = new Dictionary();
		dictionary.put("order_id", purchase.getOrderId());
		dictionary.put("package_name", purchase.getPackageName());
		dictionary.put("purchase_state", Integer.valueOf(purchase.getPurchaseState()));
		dictionary.put("purchase_time", Long.valueOf(purchase.getPurchaseTime()));
		dictionary.put("purchase_token", purchase.getPurchaseToken());
		dictionary.put("signature", purchase.getSignature());
		dictionary.put("sku", purchase.getSku());
		dictionary.put("is_acknowledged", Boolean.valueOf(purchase.isAcknowledged()));
		dictionary.put("is_auto_renewing", Boolean.valueOf(purchase.isAutoRenewing()));
		return dictionary;
	}

	public static Dictionary convertSkuDetailsToDictionary(SkuDetails details) {
		Dictionary dictionary = new Dictionary();
		dictionary.put("sku", details.getSku());
		dictionary.put("title", details.getTitle());
		dictionary.put("description", details.getDescription());
		dictionary.put("price", details.getPrice());
		dictionary.put("price_currency_code", details.getPriceCurrencyCode());
		dictionary.put("price_amount_micros", Long.valueOf(details.getPriceAmountMicros()));
		dictionary.put("free_trial_period", details.getFreeTrialPeriod());
		dictionary.put("icon_url", details.getIconUrl());
		dictionary.put("introductory_price", details.getIntroductoryPrice());
		dictionary.put("introductory_price_amount_micros", Long.valueOf(details.getIntroductoryPriceAmountMicros()));
		dictionary.put("introductory_price_cycles", details.getIntroductoryPriceCycles());
		dictionary.put("introductory_price_period", details.getIntroductoryPricePeriod());
		dictionary.put("original_price", details.getOriginalPrice());
		dictionary.put("original_price_amount_micros", Long.valueOf(details.getOriginalPriceAmountMicros()));
		dictionary.put("subscription_period", details.getSubscriptionPeriod());
		dictionary.put("type", details.getType());
		dictionary.put("is_rewarded", Boolean.valueOf(details.isRewarded()));
		return dictionary;
	}

	public static Object[] convertPurchaseListToDictionaryObjectArray(List<Purchase> purchases) {
		Object[] purchaseDictionaries = new Object[purchases.size()];

		for (int i = 0; i < purchases.size(); i++) {
			purchaseDictionaries[i] = GodotPaymentUtils.convertPurchaseToDictionary(purchases.get(i));
		}

		return purchaseDictionaries;
	}

	public static Object[] convertSkuDetailsListToDictionaryObjectArray(List<SkuDetails> skuDetails) {
		Object[] skuDetailsDictionaries = new Object[skuDetails.size()];

		for (int i = 0; i < skuDetails.size(); i++) {
			skuDetailsDictionaries[i] = GodotPaymentUtils.convertSkuDetailsToDictionary(skuDetails.get(i));
		}

		return skuDetailsDictionaries;
	}
}
