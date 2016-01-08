package org.godotengine.godot.payments;

import com.android.vending.billing.IInAppBillingService;

import android.content.Context;
import android.os.AsyncTask;
import android.os.RemoteException;
import android.util.Log;

abstract public class GenericConsumeTask extends AsyncTask<String, String, String>{

	private Context context;
	private IInAppBillingService mService;

	
	
	
	public GenericConsumeTask(Context context, IInAppBillingService mService, String sku, String receipt, String signature, String token){
		this.context = context;
		this.mService = mService;
		this.sku = sku;
		this.receipt = receipt;
		this.signature = signature;
		this.token = token;
	}
	
	private String sku;
	private String receipt;
	private String signature;
	private String token;
	
	@Override
	protected String doInBackground(String... params) {
		try {
//			Log.d("godot", "Requesting to consume an item with token ." + token);
			int response = mService.consumePurchase(3, context.getPackageName(), token);
//			Log.d("godot", "consumePurchase response: " + response);
			if(response == 0 || response == 8){
				return null;
			}
		} catch (Exception e) {
			Log.d("godot", "Error " + e.getClass().getName() + ":" + e.getMessage());
		}
		return null;
	}
	
	protected void onPostExecute(String sarasa){
		onSuccess(sku, receipt, signature, token);
	}
	
	abstract public void onSuccess(String sku, String receipt, String signature, String token);

}
