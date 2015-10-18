package com.android.godot.payments;

import java.util.ArrayList;
import java.util.List;

import android.os.RemoteException;
import android.app.Activity;
import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.content.ServiceConnection;
import android.os.IBinder;
import android.util.Log;
import android.os.Bundle;

import org.json.JSONException;
import org.json.JSONObject;
import org.json.JSONStringer;

import com.android.godot.Dictionary;
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
		Intent intent = new Intent("com.android.vending.billing.InAppBillingService.BIND");
		intent.setPackage("com.android.vending");
		activity.bindService(
				intent, 
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
	
	public void requestPurchase(String sku, String transactionId){
		new PurchaseTask(mService, Godot.getInstance()) {
			
			@Override
			protected void error(String message) {
				godotPaymentV3.callbackFail();
				
			}
			
			@Override
			protected void canceled() {
				godotPaymentV3.callbackCancel();
			}
		}.purchase(sku, transactionId);

	}

	public void consumeUnconsumedPurchases(){
		new ReleaseAllConsumablesTask(mService, activity) {
			
			@Override
			protected void success(String sku, String receipt, String signature, String token) {
				godotPaymentV3.callbackSuccessProductMassConsumed(receipt, signature, sku);
			}
			
			@Override
			protected void error(String message) {
				godotPaymentV3.callbackFail();
				
			}

			@Override
			protected void notRequired() {
				godotPaymentV3.callbackSuccessNoUnconsumedPurchases();
				
			}
		}.consumeItAll();
	}
	
	public void processPurchaseResponse(int resultCode, Intent data) {
		new HandlePurchaseTask(activity){

			@Override
			protected void success(final String sku, final String signature, final String ticket) {
				godotPaymentV3.callbackSuccess(ticket, signature);
				new ConsumeTask(mService, activity) {
					
					@Override
					protected void success(String ticket) {
//						godotPaymentV3.callbackSuccess("");
					}
					
					@Override
					protected void error(String message) {
						godotPaymentV3.callbackFail();
						
					}
				}.consume(sku);

				
//				godotPaymentV3.callbackSuccess(new PaymentsCache(activity).getConsumableValue("ticket", sku),signature);
//			    godotPaymentV3.callbackSuccess(ticket);
			    //validatePurchase(purchaseToken, sku);
			}

			@Override
			protected void error(String message) {
				godotPaymentV3.callbackFail();
				
			}

			@Override
			protected void canceled() {
				godotPaymentV3.callbackCancel();
				
			}
			}.handlePurchaseRequest(resultCode, data);
	}
	
	public void validatePurchase(String purchaseToken, final String sku){
		
		new ValidateTask(activity, godotPaymentV3){

			@Override
			protected void success() {
				
				new ConsumeTask(mService, activity) {
					
					@Override
					protected void success(String ticket) {
						godotPaymentV3.callbackSuccess(ticket, null);
						
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

