/*************************************************************************/
/*  PaymentsManager.java                                                 */
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
import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.content.ServiceConnection;
import android.os.Bundle;
import android.os.IBinder;
import android.os.RemoteException;
import android.text.TextUtils;
import android.util.Log;
import com.android.vending.billing.IInAppBillingService;
import java.util.ArrayList;
import java.util.Arrays;
import org.json.JSONException;
import org.json.JSONObject;

public class PaymentsManager {

	public static final int BILLING_RESPONSE_RESULT_OK = 0;
	public static final int REQUEST_CODE_FOR_PURCHASE = 0x1001;
	private static boolean auto_consume = true;

	private final Activity activity;
	private final GodotPayment godotPayment;
	IInAppBillingService mService;

	PaymentsManager(Activity activity, GodotPayment godotPayment) {
		this.activity = activity;
		this.godotPayment = godotPayment;
	}

	public PaymentsManager initService() {
		Intent intent = new Intent("com.android.vending.billing.InAppBillingService.BIND");
		intent.setPackage("com.android.vending");
		activity.bindService(
				intent,
				mServiceConn,
				Context.BIND_AUTO_CREATE);
		return this;
	}

	public void destroy() {
		if (mService != null) {
			activity.unbindService(mServiceConn);
		}
	}

	ServiceConnection mServiceConn = new ServiceConnection() {
		@Override
		public void onServiceDisconnected(ComponentName name) {
			mService = null;

			// At this stage, godotPayment might not have been initialized yet.
			if (godotPayment != null) {
				godotPayment.callbackDisconnected();
			}
		}

		@Override
		public void onServiceConnected(ComponentName name, IBinder service) {
			mService = IInAppBillingService.Stub.asInterface(service);

			// At this stage, godotPayment might not have been initialized yet.
			if (godotPayment != null) {
				godotPayment.callbackConnected();
			}
		}
	};

	public void requestPurchase(final String sku, String transactionId) {
		new PurchaseTask(mService, activity) {
			@Override
			protected void error(String message) {
				godotPayment.callbackFail(message);
			}

			@Override
			protected void canceled() {
				godotPayment.callbackCancel();
			}

			@Override
			protected void alreadyOwned() {
				godotPayment.callbackAlreadyOwned(sku);
			}
		}
				.purchase(sku, transactionId);
	}

	public boolean isConnected() {
		return mService != null;
	}

	public void consumeUnconsumedPurchases() {
		new ReleaseAllConsumablesTask(mService, activity) {
			@Override
			protected void success(String sku, String receipt, String signature, String token) {
				godotPayment.callbackSuccessProductMassConsumed(receipt, signature, sku);
			}

			@Override
			protected void error(String message) {
				Log.d("godot", "consumeUnconsumedPurchases :" + message);
				godotPayment.callbackFailConsume(message);
			}

			@Override
			protected void notRequired() {
				Log.d("godot", "callbackSuccessNoUnconsumedPurchases :");
				godotPayment.callbackSuccessNoUnconsumedPurchases();
			}
		}
				.consumeItAll();
	}

	public void requestPurchased() {
		try {
			PaymentsCache pc = new PaymentsCache(activity);

			String continueToken = null;

			do {
				Bundle bundle = mService.getPurchases(3, activity.getPackageName(), "inapp", continueToken);

				if (bundle.getInt("RESPONSE_CODE") == 0) {

					final ArrayList<String> myPurchases = bundle.getStringArrayList("INAPP_PURCHASE_DATA_LIST");
					final ArrayList<String> mySignatures = bundle.getStringArrayList("INAPP_DATA_SIGNATURE_LIST");

					if (myPurchases == null || myPurchases.size() == 0) {
						godotPayment.callbackPurchased("", "", "");
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

							godotPayment.callbackPurchased(receipt, signature, sku);
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
		new HandlePurchaseTask(activity) {
			@Override
			protected void success(final String sku, final String signature, final String ticket) {
				godotPayment.callbackSuccess(ticket, signature, sku);

				if (auto_consume) {
					new ConsumeTask(mService, activity) {
						@Override
						protected void success(String ticket) {
						}

						@Override
						protected void error(String message) {
							godotPayment.callbackFail(message);
						}
					}
							.consume(sku);
				}
			}

			@Override
			protected void error(String message) {
				godotPayment.callbackFail(message);
			}

			@Override
			protected void canceled() {
				godotPayment.callbackCancel();
			}
		}
				.handlePurchaseRequest(resultCode, data);
	}

	public void validatePurchase(String purchaseToken, final String sku) {

		new ValidateTask(activity, godotPayment) {
			@Override
			protected void success() {

				new ConsumeTask(mService, activity) {
					@Override
					protected void success(String ticket) {
						godotPayment.callbackSuccess(ticket, null, sku);
					}

					@Override
					protected void error(String message) {
						godotPayment.callbackFail(message);
					}
				}
						.consume(sku);
			}

			@Override
			protected void error(String message) {
				godotPayment.callbackFail(message);
			}

			@Override
			protected void canceled() {
				godotPayment.callbackCancel();
			}
		}
				.validatePurchase(sku);
	}

	public void setAutoConsume(boolean autoConsume) {
		auto_consume = autoConsume;
	}

	public void consume(final String sku) {
		new ConsumeTask(mService, activity) {
			@Override
			protected void success(String ticket) {
				godotPayment.callbackSuccessProductMassConsumed(ticket, "", sku);
			}

			@Override
			protected void error(String message) {
				godotPayment.callbackFailConsume(message);
			}
		}
				.consume(sku);
	}

	// Workaround to bug where sometimes response codes come as Long instead of Integer
	int getResponseCodeFromBundle(Bundle b) {
		Object o = b.get("RESPONSE_CODE");
		if (o == null) {
			//logDebug("Bundle with null response code, assuming OK (known issue)");
			return BILLING_RESPONSE_RESULT_OK;
		} else if (o instanceof Integer)
			return ((Integer)o).intValue();
		else if (o instanceof Long)
			return (int)((Long)o).longValue();
		else {
			//logError("Unexpected type for bundle response code.");
			//logError(o.getClass().getName());
			throw new RuntimeException("Unexpected type for bundle response code: " + o.getClass().getName());
		}
	}

	/**
	 * Returns a human-readable description for the given response code.
	 *
	 * @param code The response code
	 * @return A human-readable string explaining the result code.
	 * It also includes the result code numerically.
	 */
	public static String getResponseDesc(int code) {
		String[] iab_msgs = ("0:OK/1:User Canceled/2:Unknown/"
							 +
							 "3:Billing Unavailable/4:Item unavailable/"
							 +
							 "5:Developer Error/6:Error/7:Item Already Owned/"
							 +
							 "8:Item not owned")
									.split("/");
		String[] iabhelper_msgs = ("0:OK/-1001:Remote exception during initialization/"
								   +
								   "-1002:Bad response received/"
								   +
								   "-1003:Purchase signature verification failed/"
								   +
								   "-1004:Send intent failed/"
								   +
								   "-1005:User cancelled/"
								   +
								   "-1006:Unknown purchase response/"
								   +
								   "-1007:Missing token/"
								   +
								   "-1008:Unknown error/"
								   +
								   "-1009:Subscriptions not available/"
								   +
								   "-1010:Invalid consumption attempt")
										  .split("/");

		if (code <= -1000) {
			int index = -1000 - code;
			if (index >= 0 && index < iabhelper_msgs.length)
				return iabhelper_msgs[index];
			else
				return String.valueOf(code) + ":Unknown IAB Helper Error";
		} else if (code < 0 || code >= iab_msgs.length)
			return String.valueOf(code) + ":Unknown";
		else
			return iab_msgs[code];
	}

	public void querySkuDetails(final String[] list) {
		(new Thread(new Runnable() {
			@Override
			public void run() {
				ArrayList<String> skuList = new ArrayList<String>(Arrays.asList(list));
				if (skuList.size() == 0) {
					return;
				}
				// Split the sku list in blocks of no more than 20 elements.
				ArrayList<ArrayList<String>> packs = new ArrayList<ArrayList<String>>();
				ArrayList<String> tempList;
				int n = skuList.size() / 20;
				int mod = skuList.size() % 20;
				for (int i = 0; i < n; i++) {
					tempList = new ArrayList<String>();
					for (String s : skuList.subList(i * 20, i * 20 + 20)) {
						tempList.add(s);
					}
					packs.add(tempList);
				}
				if (mod != 0) {
					tempList = new ArrayList<String>();
					for (String s : skuList.subList(n * 20, n * 20 + mod)) {
						tempList.add(s);
					}
					packs.add(tempList);
				}
				for (ArrayList<String> skuPartList : packs) {
					Bundle querySkus = new Bundle();
					querySkus.putStringArrayList("ITEM_ID_LIST", skuPartList);
					Bundle skuDetails = null;
					try {
						skuDetails = mService.getSkuDetails(3, activity.getPackageName(), "inapp", querySkus);
						if (!skuDetails.containsKey("DETAILS_LIST")) {
							int response = getResponseCodeFromBundle(skuDetails);
							if (response != BILLING_RESPONSE_RESULT_OK) {
								godotPayment.errorSkuDetail(getResponseDesc(response));
							} else {
								godotPayment.errorSkuDetail("No error but no detail list.");
							}
							return;
						}

						ArrayList<String> responseList = skuDetails.getStringArrayList("DETAILS_LIST");

						for (String thisResponse : responseList) {
							Log.d("godot", "response = " + thisResponse);
							godotPayment.addSkuDetail(thisResponse);
						}
					} catch (RemoteException e) {
						e.printStackTrace();
						godotPayment.errorSkuDetail("RemoteException error!");
					}
				}
				godotPayment.completeSkuDetail();
			}
		}))
				.start();
	}
}
