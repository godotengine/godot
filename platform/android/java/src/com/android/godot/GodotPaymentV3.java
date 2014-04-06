package com.android.godot;


import android.app.Activity;


public class GodotPaymentV3 extends Godot.SingletonBase {

	private Godot activity;

	private Integer purchaseCallbackId = 0;

	private String accessToken;
	
	private String purchaseValidationUrlPrefix;

	public void purchase( String _sku) {
		final String sku = _sku;
		activity.getPaymentsManager().setBaseSingleton(this);
		activity.runOnUiThread(new Runnable() {
			@Override
			public void run() {
				activity.getPaymentsManager().requestPurchase(sku);				
			}
		});
	};


    static public Godot.SingletonBase initialize(Activity p_activity) {

        return new GodotPaymentV3(p_activity);
    }

	
	public GodotPaymentV3(Activity p_activity) {

		registerClass("GodotPayments", new String[] {"purchase", "setPurchaseCallbackId", "setPurchaseValidationUrlPrefix"});
		activity=(Godot) p_activity;
	}


	
	public void callbackSuccess(){
        GodotLib.callobject(purchaseCallbackId, "purchase_success", new Object[]{});
	}
	
	public void callbackFail(){
        GodotLib.callobject(purchaseCallbackId, "purchase_fail", new Object[]{});
	}
	
	public void callbackCancel(){
		GodotLib.callobject(purchaseCallbackId, "purchase_cancel", new Object[]{});
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
	
}
