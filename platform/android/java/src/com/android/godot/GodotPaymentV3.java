package com.android.godot;

import com.android.godot.Dictionary;
import android.app.Activity;
import android.util.Log;


public class GodotPaymentV3 extends Godot.SingletonBase {

	private Godot activity;

	private Integer purchaseCallbackId = 0;

	private String accessToken;
	
	private String purchaseValidationUrlPrefix;
	
	private String transactionId;

	public void purchase( String _sku, String _transactionId) {
		final String sku = _sku;
		final String transactionId = _transactionId;
		activity.getPaymentsManager().setBaseSingleton(this);
		activity.runOnUiThread(new Runnable() {
			@Override
			public void run() {
				activity.getPaymentsManager().requestPurchase(sku, transactionId);				
			}
		});
	}
	
/*	public string requestPurchasedTicket(){
	    activity.getPaymentsManager()
	}

*/
    static public Godot.SingletonBase initialize(Activity p_activity) {

        return new GodotPaymentV3(p_activity);
    }

	
	public GodotPaymentV3(Activity p_activity) {

		registerClass("GodotPayments", new String[] {"purchase", "setPurchaseCallbackId", "setPurchaseValidationUrlPrefix", "setTransactionId", "getSignature", "consumeUnconsumedPurchases", "requestPurchased", "setAutoConsume", "consume"});
		activity=(Godot) p_activity;
	}

	public void consumeUnconsumedPurchases(){
		activity.getPaymentsManager().setBaseSingleton(this);
		activity.runOnUiThread(new Runnable() {
			@Override
			public void run() {
				activity.getPaymentsManager().consumeUnconsumedPurchases();				
			}
		});
	}

	private String signature;
	public String getSignature(){
	        return this.signature;
	}
	
	
	public void callbackSuccess(String ticket, String signature, String sku){
//		Log.d(this.getClass().getName(), "PRE-Send callback to purchase success");
		GodotLib.callobject(purchaseCallbackId, "purchase_success", new Object[]{ticket, signature, sku});
//		Log.d(this.getClass().getName(), "POST-Send callback to purchase success");
}

	public void callbackSuccessProductMassConsumed(String ticket, String signature, String sku){
//		Log.d(this.getClass().getName(), "PRE-Send callback to consume success");
		Log.d(this.getClass().getName(), "callbackSuccessProductMassConsumed > "+ticket+","+signature+","+sku);
        	GodotLib.calldeferred(purchaseCallbackId, "consume_success", new Object[]{ticket, signature, sku});
//		Log.d(this.getClass().getName(), "POST-Send callback to consume success");
	}

	public void callbackSuccessNoUnconsumedPurchases(){
		GodotLib.calldeferred(purchaseCallbackId, "no_validation_required", new Object[]{});
	}
	
	public void callbackFail(){
		GodotLib.calldeferred(purchaseCallbackId, "purchase_fail", new Object[]{});
//		GodotLib.callobject(purchaseCallbackId, "purchase_fail", new Object[]{});
	}
	
	public void callbackCancel(){
		GodotLib.calldeferred(purchaseCallbackId, "purchase_cancel", new Object[]{});
//		GodotLib.callobject(purchaseCallbackId, "purchase_cancel", new Object[]{});
	}
	
	public void callbackAlreadyOwned(String sku){
		GodotLib.calldeferred(purchaseCallbackId, "purchase_owned", new Object[]{sku});
	}
	
	public int getPurchaseCallbackId() {
		return purchaseCallbackId;
	}

	public void setPurchaseCallbackId(int purchaseCallbackId) {
		this.purchaseCallbackId = purchaseCallbackId;
	}

	public String getPurchaseValidationUrlPrefix(){
		return this.purchaseValidationUrlPrefix ;
	}

	public void setPurchaseValidationUrlPrefix(String url){
		this.purchaseValidationUrlPrefix = url;
	}

	public String getAccessToken() {
		return accessToken;
	}

	public void setAccessToken(String accessToken) {
		this.accessToken = accessToken;
	}
	
	public void setTransactionId(String transactionId){
		this.transactionId = transactionId;
	}
	
	public String getTransactionId(){
		return this.transactionId;
	}
	
	// request purchased items are not consumed
	public void requestPurchased(){
		activity.getPaymentsManager().setBaseSingleton(this);
		activity.runOnUiThread(new Runnable() {
			@Override
			public void run() {
				activity.getPaymentsManager().requestPurchased();				
			}
		});
	}
	
	// callback for requestPurchased()
	public void callbackPurchased(String receipt, String signature, String sku){
		GodotLib.calldeferred(purchaseCallbackId, "has_purchased", new Object[]{receipt, signature, sku});
	}
	
	// consume item automatically after purchase. default is true.
	public void setAutoConsume(boolean autoConsume){
		activity.getPaymentsManager().setAutoConsume(autoConsume);
	}
	
	// consume a specific item
	public void consume(String sku){
		activity.getPaymentsManager().consume(sku);
	}
}

