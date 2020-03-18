/*************************************************************************/
/*  ValidateTask.java                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

package org.godotengine.godot.plugin.payment;

import android.app.Activity;
import android.app.ProgressDialog;
import android.os.AsyncTask;
import java.lang.ref.WeakReference;
import org.godotengine.godot.utils.HttpRequester;
import org.godotengine.godot.utils.RequestParams;
import org.json.JSONException;
import org.json.JSONObject;

abstract public class ValidateTask {

	private Activity context;
	private GodotPayment godotPayments;
	private ProgressDialog dialog;
	private String mSku;

	private static class ValidateAsyncTask extends AsyncTask<String, String, String> {
		private WeakReference<ValidateTask> mTask;

		ValidateAsyncTask(ValidateTask task) {
			mTask = new WeakReference<>(task);
		}

		@Override
		protected void onPreExecute() {
			ValidateTask task = mTask.get();
			if (task != null) {
				task.onPreExecute();
			}
		}

		@Override
		protected String doInBackground(String... params) {
			ValidateTask task = mTask.get();
			if (task != null) {
				return task.doInBackground(params);
			}
			return null;
		}

		@Override
		protected void onPostExecute(String response) {
			ValidateTask task = mTask.get();
			if (task != null) {
				task.onPostExecute(response);
			}
		}
	}

	public ValidateTask(Activity context, GodotPayment godotPayments) {
		this.context = context;
		this.godotPayments = godotPayments;
	}

	public void validatePurchase(final String sku) {
		mSku = sku;
		new ValidateAsyncTask(this).execute();
	}

	private void onPreExecute() {
		dialog = ProgressDialog.show(context, null, "Please wait...");
	}

	private String doInBackground(String... params) {
		PaymentsCache pc = new PaymentsCache(context);
		String url = godotPayments.getPurchaseValidationUrlPrefix();
		RequestParams param = new RequestParams();
		param.setUrl(url);
		param.put("ticket", pc.getConsumableValue("ticket", mSku));
		param.put("purchaseToken", pc.getConsumableValue("token", mSku));
		param.put("sku", mSku);
		//Log.d("XXX", "Haciendo request a " + url);
		//Log.d("XXX", "ticket: " + pc.getConsumableValue("ticket", sku));
		//Log.d("XXX", "purchaseToken: " + pc.getConsumableValue("token", sku));
		//Log.d("XXX", "sku: " + sku);
		param.put("package", context.getApplicationContext().getPackageName());
		HttpRequester requester = new HttpRequester();
		String jsonResponse = requester.post(param);
		//Log.d("XXX", "Validation response:\n"+jsonResponse);
		return jsonResponse;
	}

	private void onPostExecute(String response) {
		if (dialog != null) {
			dialog.dismiss();
			dialog = null;
		}
		JSONObject j;
		try {
			j = new JSONObject(response);
			if (j.getString("status").equals("OK")) {
				success();
				return;
			} else if (j.getString("status") != null) {
				error(j.getString("message"));
			} else {
				error("Connection error");
			}
		} catch (JSONException e) {
			error(e.getMessage());
		} catch (Exception e) {
			error(e.getMessage());
		}
	}

	abstract protected void success();
	abstract protected void error(String message);
	abstract protected void canceled();
}
