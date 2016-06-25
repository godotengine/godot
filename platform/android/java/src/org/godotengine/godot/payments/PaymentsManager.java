/*************************************************************************/
/*  PaymentsManager.java                                                 */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2016 Juan Linietsky, Ariel Manzur.                 */
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

import org.godotengine.godot.Dictionary;
import org.godotengine.godot.Godot;
import org.godotengine.godot.GodotPaymentV3;
import com.android.vending.billing.IInAppBillingService;

public class PaymentsManager {

	public static final int BILLING_RESPONSE_RESULT_OK = 0;
	public static final int REQUEST_CODE_FOR_PURCHASE = 0x1001;
	private static boolean auto_consume = true;
	
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
	    public void onServiceConnected(ComponentName name, IBinder service) {
			mService = IInAppBillingService.Stub.asInterface(service);
	    }
	};
	
	public void requestPurchase(final String sku, String transactionId){
		new PurchaseTask(mService, Godot.getInstance()) {
			
			@Override
			protected void error(String message) {
				godotPaymentV3.callbackFail();
				
			}
			
			@Override
			protected void canceled() {
				godotPaymentV3.callbackCancel();
			}
			
			@Override
			protected void alreadyOwned() {
				godotPaymentV3.callbackAlreadyOwned(sku);
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

	public void requestPurchased() {
		try {
			PaymentsCache pc = new PaymentsCache(Godot.getInstance());

			String continueToken = null;

			do {
				Bundle bundle = mService.getPurchases(3, activity.getPackageName(), "inapp", continueToken);

				if (bundle.getInt("RESPONSE_CODE") == 0) {

					final ArrayList<String> myPurchases = bundle.getStringArrayList("INAPP_PURCHASE_DATA_LIST");
					final ArrayList<String> mySignatures = bundle.getStringArrayList("INAPP_DATA_SIGNATURE_LIST");

					if (myPurchases == null || myPurchases.size() == 0) {
						godotPaymentV3.callbackPurchased("", "", "");
						return;
					}

					for (int i = 0; i < myPurchases.size(); i++) {

						try {
							String receipt = myPurchases.get(i);
							JSONObject inappPurchaseData = new JSONObject(receipt);
							String sku = inappPurchaseData.getString("productId");
							String token = inappPurchaseData.getString("purchaseToken");
							String signature = mySignatures.get(i);

							pc.setConsumableValue("ticket_signautre", sku, signature);
							pc.setConsumableValue("ticket", sku, receipt);
							pc.setConsumableFlag("block", sku, true);
							pc.setConsumableValue("token", sku, token);

							godotPaymentV3.callbackPurchased(receipt, signature, sku);
						} catch (JSONException e) {
						}
					}
				}
				continueToken = bundle.getString("INAPP_CONTINUATION_TOKEN");
				Log.d("godot", "continue token = " + continueToken);
			} while (!TextUtils.isEmpty(continueToken));
		} catch (Exception e) {
			Log.d("godot", "Error requesting purchased products:" + e.getClass().getName() + ":" + e.getMessage());
		}
	}
	
	public void processPurchaseResponse(int resultCode, Intent data) {
		new HandlePurchaseTask(activity){

			@Override
			protected void success(final String sku, final String signature, final String ticket) {
				godotPaymentV3.callbackSuccess(ticket, signature, sku);

				if (auto_consume){
					new ConsumeTask(mService, activity) {
					
						@Override
						protected void success(String ticket) {
//							godotPaymentV3.callbackSuccess("");
						}
					
						@Override
						protected void error(String message) {
							godotPaymentV3.callbackFail();
						
						}
					}.consume(sku);
				}
				
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
						godotPaymentV3.callbackSuccess(ticket, null, sku);
						
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
	
	public void setAutoConsume(boolean autoConsume){
		auto_consume = autoConsume;
	}
	
	public void consume(final String sku){
		new ConsumeTask(mService, activity) {
			
			@Override
			protected void success(String ticket) {
				godotPaymentV3.callbackSuccessProductMassConsumed(ticket, "", sku);
				
			}
			
			@Override
			protected void error(String message) {
				godotPaymentV3.callbackFail();
				
			}
		}.consume(sku);
	}
	
	private GodotPaymentV3 godotPaymentV3;
	
	public void setBaseSingleton(GodotPaymentV3 godotPaymentV3) {
		this.godotPaymentV3 = godotPaymentV3;
	}

}

