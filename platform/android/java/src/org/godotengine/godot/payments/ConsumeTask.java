package org.godotengine.godot.payments;

import com.android.vending.billing.IInAppBillingService;

import android.content.Context;
import android.os.AsyncTask;
import android.os.RemoteException;
import android.util.Log;

abstract public class ConsumeTask {

	private Context context;
	
	private IInAppBillingService mService;
	public ConsumeTask(IInAppBillingService mService, Context context ){
		this.context = context;
		this.mService = mService;
	}
	

	public void consume(final String sku){
//		Log.d("XXX", "Consuming product " + sku);
		PaymentsCache pc = new PaymentsCache(context);
		Boolean isBlocked = pc.getConsumableFlag("block", sku);
		String _token = pc.getConsumableValue("token", sku);
//		Log.d("XXX", "token " + _token);		
		if(!isBlocked && _token == null){
//			_token = "inapp:"+context.getPackageName()+":android.test.purchased";
//			Log.d("XXX", "Consuming product " + sku + " with token " + _token);
		}else if(!isBlocked){
//			Log.d("XXX", "It is not blocked ¿?");
			return;
		}else if(_token == null){
//			Log.d("XXX", "No token available");
			this.error("No token for sku:" + sku);
			return;
		}
		final String token = _token;
		new AsyncTask<String, String, String>(){

			@Override
			protected String doInBackground(String... params) {
				try {
//					Log.d("XXX", "Requesting to release item.");
					int response = mService.consumePurchase(3, context.getPackageName(), token);
//					Log.d("XXX", "release response code: " + response);
					if(response == 0 || response == 8){
						return null;
					}
				} catch (RemoteException e) {
					return e.getMessage();
					
				}
				return "Some error";
			}
			
			protected void onPostExecute(String param){
				if(param == null){
					success( new PaymentsCache(context).getConsumableValue("ticket", sku) );
				}else{
					error(param);
				}
			}
			
		}.execute();
	}
	
	abstract protected void success(String ticket);
	abstract protected void error(String message);
	
}
