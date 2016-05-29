package org.godotengine.godot.payments;

import java.util.ArrayList;

import org.json.JSONException;
import org.json.JSONObject;

import org.godotengine.godot.Dictionary;
import org.godotengine.godot.Godot;
import com.android.vending.billing.IInAppBillingService;

import android.content.Context;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.RemoteException;
import android.util.Log;

abstract public class ReleaseAllConsumablesTask {

	private Context context;
	private IInAppBillingService mService;
	
	public ReleaseAllConsumablesTask(IInAppBillingService mService, Context context ){
		this.context = context;
		this.mService = mService;
	}
	

	public void consumeItAll(){
		try{
//			Log.d("godot", "consumeItall for " + context.getPackageName());
			Bundle bundle = mService.getPurchases(3, context.getPackageName(), "inapp",null);
			
			for (String key : bundle.keySet()) {
			    Object value = bundle.get(key);
//			    Log.d("godot", String.format("%s %s (%s)", key,  
//			        value.toString(), value.getClass().getName()));
			}
			
			
			if (bundle.getInt("RESPONSE_CODE") == 0){

				final ArrayList<String> myPurchases = bundle.getStringArrayList("INAPP_PURCHASE_DATA_LIST");
				final ArrayList<String> mySignatures = bundle.getStringArrayList("INAPP_DATA_SIGNATURE_LIST");
				

				if (myPurchases == null || myPurchases.size() == 0){
//					Log.d("godot", "No purchases!");
					notRequired();
					return;
				}
		
				
//				Log.d("godot", "# products to be consumed:" + myPurchases.size());
				for (int i=0;i<myPurchases.size();i++)
				{
					
					try{
						String receipt = myPurchases.get(i);
						JSONObject inappPurchaseData = new JSONObject(receipt);
						String sku = inappPurchaseData.getString("productId");
						String token = inappPurchaseData.getString("purchaseToken");
						String signature = mySignatures.get(i);
//						Log.d("godot", "A punto de consumir un item con token:" + token + "\n" + receipt);
						new GenericConsumeTask(context, mService, sku, receipt,signature, token) {
							
							@Override
							public void onSuccess(String sku, String receipt, String signature, String token) {
								ReleaseAllConsumablesTask.this.success(sku, receipt, signature, token);
							}
						}.execute();
						
					} catch (JSONException e) {
					}
				}

			}
		}catch(Exception e){
			Log.d("godot", "Error releasing products:" + e.getClass().getName() + ":" + e.getMessage());
		}
	}
	
	abstract protected void success(String sku, String receipt, String signature, String token);
	abstract protected void error(String message);
	abstract protected void notRequired();
	
}
