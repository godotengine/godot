/*************************************************************************/
/*  GenericConsumeTask.java                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
			//Log.d("godot", "Requesting to consume an item with token ." + token);
			int response = mService.consumePurchase(3, context.getPackageName(), token);
			//Log.d("godot", "consumePurchase response: " + response);
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
