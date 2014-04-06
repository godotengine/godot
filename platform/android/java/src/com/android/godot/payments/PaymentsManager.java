package com.android.godot.payments;

import android.app.Activity;
import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.content.ServiceConnection;
import android.os.IBinder;

import com.android.godot.Godot;
import com.android.godot.GodotPaymentV3;
import com.android.vending.billing.IInAppBillingService;

public class PaymentsManager {

	public static final int BILLING_RESPONSE_RESULT_OK = 0;

	
	public static final int REQUEST_CODE_FOR_PURCHASE = 0x1001;
	
	
	private Activity activity;
	IInAppBillingService mService;

	
	public void setActivity(Activity activity){
		this.activity = activity;
	}

	public static PaymentsManager createManager(Activity activity){
		PaymentsManager manager = new PaymentsManager(activity);
		return manager;
	}
	
	private PaymentsManager(Activity activity){
		this.activity = activity;
	}
	
	public PaymentsManager initService(){
		activity.bindService(
				new Intent("com.android.vending.billing.InAppBillingService.BIND"), 
				mServiceConn, 
				Context.BIND_AUTO_CREATE);
		return this;
	}

	public void destroy(){
		if (mService != null) {
	        activity.unbindService(mServiceConn);
	    }  
	}
	
	ServiceConnection mServiceConn = new ServiceConnection() {
	    @Override
	    public void onServiceDisconnected(ComponentName name) {
	    	mService = null;
	    }

	    @Override
	    public void onServiceConnected(ComponentName name, 
	    		IBinder service) {
		mService = IInAppBillingService.Stub.asInterface(service);
	    }
	};
	
	public void requestPurchase(String sku){
		new PurchaseTask(mService, Godot.getInstance()) {
			
			@Override
			protected void error(String message) {
				godotPaymentV3.callbackFail();
				
			}
			
			@Override
			protected void canceled() {
				godotPaymentV3.callbackCancel();
			}
		}.purchase(sku);

	}

	public void processPurchaseResponse(int resultCode, Intent data) {
		new HandlePurchaseTask(activity){

			@Override
			protected void success(String purchaseToken, String sku) {
				validatePurchase(purchaseToken, sku);
			}

			@Override
			protected void error(String message) {
				godotPaymentV3.callbackFail();
				
			}

			@Override
			protected void canceled() {
				godotPaymentV3.callbackCancel();
				
			}}.handlePurchaseRequest(resultCode, data);
	}
	
	public void validatePurchase(String purchaseToken, final String sku){
		
		new ValidateTask(activity, godotPaymentV3){

			@Override
			protected void success() {
				
				new ConsumeTask(mService, activity) {
					
					@Override
					protected void success() {
						godotPaymentV3.callbackSuccess();
						
					}
					
					@Override
					protected void error(String message) {
						godotPaymentV3.callbackFail();
						
					}
				}.consume(sku);
				
			}

			@Override
			protected void error(String message) {
				godotPaymentV3.callbackFail();
				
			}

			@Override
			protected void canceled() {
				godotPaymentV3.callbackCancel();
				
			}
		}.validatePurchase(sku);
	}
	
	private GodotPaymentV3 godotPaymentV3;
	
	public void setBaseSingleton(GodotPaymentV3 godotPaymentV3) {
		this.godotPaymentV3 = godotPaymentV3;
		
	}


}

