package org.godotengine.godot.payments;

import org.json.JSONException;
import org.json.JSONObject;

import org.godotengine.godot.Godot;
import org.godotengine.godot.GodotLib;
import org.godotengine.godot.GodotPaymentV3;
import org.godotengine.godot.utils.Crypt;
import org.godotengine.godot.utils.HttpRequester;
import org.godotengine.godot.utils.RequestParams;
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

abstract public class ValidateTask {

	private Activity context;
	private GodotPaymentV3 godotPaymentsV3;
	public ValidateTask(Activity context, GodotPaymentV3 godotPaymentsV3){
		this.context = context;
		this.godotPaymentsV3 = godotPaymentsV3;
	}
	
	public void validatePurchase(final String sku){
		new AsyncTask<String, String, String>(){

			
			private ProgressDialog dialog;

			@Override
			protected void onPreExecute(){
				dialog = ProgressDialog.show(context, null, "Please wait...");
			}
			
			@Override
			protected String doInBackground(String... params) {
				PaymentsCache pc = new PaymentsCache(context);
				String url = godotPaymentsV3.getPurchaseValidationUrlPrefix();
				RequestParams param = new RequestParams();
				param.setUrl(url);
				param.put("ticket", pc.getConsumableValue("ticket", sku));
				param.put("purchaseToken", pc.getConsumableValue("token", sku));
				param.put("sku", sku);
//				Log.d("XXX", "Haciendo request a " + url);
//				Log.d("XXX", "ticket: " + pc.getConsumableValue("ticket", sku));
//				Log.d("XXX", "purchaseToken: " + pc.getConsumableValue("token", sku));
//				Log.d("XXX", "sku: " + sku);
				param.put("package", context.getApplicationContext().getPackageName());
				HttpRequester requester = new HttpRequester();
				String jsonResponse = requester.post(param);
//				Log.d("XXX", "Validation response:\n"+jsonResponse);
				return jsonResponse;
			}
			
			@Override
			protected void onPostExecute(String response){
				if(dialog != null){
					dialog.dismiss();
				}
				JSONObject j;
				try {
					j = new JSONObject(response);
					if(j.getString("status").equals("OK")){
						success();
						return;
					}else if(j.getString("status") != null){
						error(j.getString("message"));
					}else{
						error("Connection error");
					}
				} catch (JSONException e) {
					error(e.getMessage());
				}catch (Exception e){
					error(e.getMessage());
				}

				
			}
			
		}.execute();
	}
	abstract protected void success();
	abstract protected void error(String message);
	abstract protected void canceled();

	
}
