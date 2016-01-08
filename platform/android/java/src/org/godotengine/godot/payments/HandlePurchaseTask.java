package org.godotengine.godot.payments;

import org.json.JSONException;
import org.json.JSONObject;

import org.godotengine.godot.GodotLib;
import org.godotengine.godot.utils.Crypt;
import com.android.vending.billing.IInAppBillingService;

import android.app.Activity;
import android.app.PendingIntent;
import android.app.ProgressDialog;
import android.content.Context;
import android.content.Intent;
import android.content.IntentSender.SendIntentException;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.RemoteException;
import android.util.Log;

abstract public class HandlePurchaseTask {

	private Activity context;
	
	public HandlePurchaseTask(Activity context ){
		this.context = context;
	}
	
	
	public void handlePurchaseRequest(int resultCode, Intent data){
//		Log.d("XXX", "Handling purchase response");
//		int responseCode = data.getIntExtra("RESPONSE_CODE", 0);
		PaymentsCache pc = new PaymentsCache(context);
		
		String purchaseData = data.getStringExtra("INAPP_PURCHASE_DATA");
//		Log.d("XXX", "Purchase data:" + purchaseData);
		String dataSignature = data.getStringExtra("INAPP_DATA_SIGNATURE");
		//Log.d("XXX", "Purchase signature:" + dataSignature);
		
		if (resultCode == Activity.RESULT_OK) {
			
			try {
//				Log.d("SARLANGA", purchaseData);
				
				
				JSONObject jo = new JSONObject(purchaseData);
//				String sku = jo.getString("productId");
//				alert("You have bought the " + sku + ". Excellent choice, aventurer!");
//				String orderId = jo.getString("orderId");
//				String packageName = jo.getString("packageName");
				String productId = jo.getString("productId");
//				Long purchaseTime = jo.getLong("purchaseTime");
//				Integer state = jo.getInt("purchaseState");
				String developerPayload = jo.getString("developerPayload");
				String purchaseToken = jo.getString("purchaseToken");
				
				if(! pc.getConsumableValue("validation_hash", productId).equals(developerPayload) ) {
					error("Untrusted callback");
					return;
				}
//				Log.d("XXX", "Este es el product ID:" + productId);
				pc.setConsumableValue("ticket_signautre", productId, dataSignature);
				pc.setConsumableValue("ticket", productId, purchaseData);
				pc.setConsumableFlag("block", productId, true);
				pc.setConsumableValue("token", productId, purchaseToken);
				
				success(productId, dataSignature, purchaseData);
				return;
			}	catch (JSONException e) {
				error(e.getMessage());
			}
		}else if( resultCode == Activity.RESULT_CANCELED){
			canceled();
		}
	}

	abstract protected void success(String sku, String signature, String ticket);
	abstract protected void error(String message);
	abstract protected void canceled();

	
}
