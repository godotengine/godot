/*************************************************************************/
/*  PaymentsManager.java                                                 */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

import android.app.Activity;
import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.content.pm.ResolveInfo;
import android.content.ServiceConnection;
import android.os.Bundle;
import android.os.IBinder;
import android.os.RemoteException;
import android.text.TextUtils;
import android.util.Log;

import com.android.vending.billing.IInAppBillingService;

import org.godotengine.godot.Godot;
import org.godotengine.godot.GodotPaymentV3;
import org.json.JSONException;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class PaymentsManager {

	private static String TAG = "PaymentsManager";
	private static boolean auto_consume = true;

	private GodotPaymentV3 godotPaymentV3;

	// Is setup done?
	private boolean mSetupDone = false;

	// Has this object been disposed of? (If so, we should ignore callbacks, etc)
	private boolean mDisposed = false;

	// Connection to the service
	IInAppBillingService mService;
	ServiceConnection mServiceConn;

	private Activity activity;
	private Context context;

	// Billing response codes
	public static final int BILLING_RESPONSE_RESULT_OK = 0;
	public static final int BILLING_RESPONSE_RESULT_USER_CANCELED = 1;
	public static final int BILLING_RESPONSE_RESULT_SERVICE_UNAVAILABLE = 2;
	public static final int BILLING_RESPONSE_RESULT_BILLING_UNAVAILABLE = 3;
	public static final int BILLING_RESPONSE_RESULT_ITEM_UNAVAILABLE = 4;
	public static final int BILLING_RESPONSE_RESULT_DEVELOPER_ERROR = 5;
	public static final int BILLING_RESPONSE_RESULT_ERROR = 6;
	public static final int BILLING_RESPONSE_RESULT_ITEM_ALREADY_OWNED = 7;
	public static final int BILLING_RESPONSE_RESULT_ITEM_NOT_OWNED = 8;

	// IAB Helper error codes
	public static final int IABHELPER_ERROR_BASE = -1000;
	public static final int IABHELPER_REMOTE_EXCEPTION = -1001;
	public static final int IABHELPER_BAD_RESPONSE = -1002;
	public static final int IABHELPER_VERIFICATION_FAILED = -1003;
	public static final int IABHELPER_SEND_INTENT_FAILED = -1004;
	public static final int IABHELPER_USER_CANCELLED = -1005;
	public static final int IABHELPER_UNKNOWN_PURCHASE_RESPONSE = -1006;
	public static final int IABHELPER_MISSING_TOKEN = -1007;
	public static final int IABHELPER_UNKNOWN_ERROR = -1008;
	public static final int IABHELPER_SUBSCRIPTIONS_NOT_AVAILABLE = -1009;
	public static final int IABHELPER_INVALID_CONSUMPTION = -1010;
	public static final int IABHELPER_SUBSCRIPTION_UPDATE_NOT_AVAILABLE = -1011;

	// Keys for the responses from InAppBillingService
	public static final String RESPONSE_CODE = "RESPONSE_CODE";
	public static final String RESPONSE_GET_SKU_DETAILS_LIST = "DETAILS_LIST";
	public static final String RESPONSE_BUY_INTENT = "BUY_INTENT";
	public static final String RESPONSE_INAPP_PURCHASE_DATA = "INAPP_PURCHASE_DATA";
	public static final String RESPONSE_INAPP_SIGNATURE = "INAPP_DATA_SIGNATURE";
	public static final String RESPONSE_INAPP_ITEM_LIST = "INAPP_PURCHASE_ITEM_LIST";
	public static final String RESPONSE_INAPP_PURCHASE_DATA_LIST = "INAPP_PURCHASE_DATA_LIST";
	public static final String RESPONSE_INAPP_SIGNATURE_LIST = "INAPP_DATA_SIGNATURE_LIST";
	public static final String INAPP_CONTINUATION_TOKEN = "INAPP_CONTINUATION_TOKEN";

	// Item types
	public static final String ITEM_TYPE_INAPP = "inapp";
	public static final String ITEM_TYPE_SUBS = "subs";

	public static final int REQUEST_CODE_FOR_PURCHASE = 10001;

	public void setActivity(Activity activity) {
		this.activity = activity;
		this.context = activity.getApplicationContext();
	}

	public static PaymentsManager createManager(Activity activity) {
		PaymentsManager manager = new PaymentsManager(activity);
		return manager;
	}

	private PaymentsManager(Activity activity) {
		this.activity = activity;
		this.context = activity.getApplicationContext();
	}

	public void initService() {
		// Cancel the service creation if it has already been created or it creating
		if (mDisposed || mSetupDone) {
			return;
		}

		mServiceConn = new ServiceConnection() {
			@Override
			public void onServiceDisconnected(ComponentName name) {
				mService = null;

				// At this stage, godotPaymentV3 might not have been initialized yet.
				if (godotPaymentV3 != null) {
					godotPaymentV3.callbackDisconnected();
				}
			}

			@Override
			public void onServiceConnected(ComponentName name, IBinder service) {
				if (mDisposed) return;

				mService = IInAppBillingService.Stub.asInterface(service);

				String packageName = context.getPackageName();

				try {
					// check for in-app billing v3 support
					int response = mService.isBillingSupported(3, packageName, ITEM_TYPE_INAPP);

					if (response != BILLING_RESPONSE_RESULT_OK) {
						Log.i(TAG, "Device does not support billing 3.");

						return;
					}

					mSetupDone = true;
				} catch (RemoteException e) {
					Log.d(TAG, "Error binding ServiceConnection:" + e.getMessage());
					return;
				}

				// At this stage, godotPaymentV3 might not have been initialized yet.
				if (godotPaymentV3 != null) {
					godotPaymentV3.callbackConnected();
				}
			}
		};

		Intent serviceIntent = new Intent("com.android.vending.billing.InAppBillingService.BIND");
		serviceIntent.setPackage("com.android.vending");

		List<ResolveInfo> intentServices = context.getPackageManager().queryIntentServices(serviceIntent, 0);
		if (intentServices != null && !intentServices.isEmpty()) {
			// service available to handle that Intent
			context.bindService(serviceIntent, mServiceConn, Context.BIND_AUTO_CREATE);
		} else {
			Log.i(TAG, "Billing service unavailable on device.");
		}

		return;
	}

	public void destroy() {
		if (mService != null) {
			try {
				activity.unbindService(mServiceConn);
			} catch (IllegalArgumentException e) {
				// Somehow we've already been unbound. This is a non-fatal
				// error.
				Log.e(TAG, "Unable to unbind from payment service (already unbound)");
			}
		}

		mSetupDone = false;
		mDisposed = true;
		mServiceConn = null;
		mService = null;
		activity = null;
		context = null;
	}

	public void requestPurchase(final String sku, String transactionId) {
		if (!isConnected()) return;

		PurchaseTask purchaseTask = new PurchaseTask(mService, Godot.getInstance()) {
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
		};

		purchaseTask.purchase(sku, transactionId);
	}

	public boolean isConnected() {
		return mSetupDone && mService != null;
	}

	public void consumeUnconsumedPurchases() {
		ReleaseAllConsumablesTask releaseAllConsumablesTask = new ReleaseAllConsumablesTask(mService, activity) {
			@Override
			protected void success(String sku, String receipt, String signature, String token) {
				godotPaymentV3.callbackSuccessProductMassConsumed(receipt, signature, sku);
			}

			@Override
			protected void error(String message) {
				Log.d(TAG, "consumeUnconsumedPurchases :" + message);
				godotPaymentV3.callbackFailConsume();
			}

			@Override
			protected void notRequired() {
				Log.d(TAG, "callbackSuccessNoUnconsumedPurchases :");
				godotPaymentV3.callbackSuccessNoUnconsumedPurchases();
			}
		};

		releaseAllConsumablesTask.consumeItAll();
	}

	public void requestPurchased() {
		try {
			PaymentsCache pc = new PaymentsCache(Godot.getInstance());

			String continueToken = null;

			do {
				Bundle bundle = mService.getPurchases(3, activity.getPackageName(), ITEM_TYPE_INAPP, continueToken);

				if (bundle.getInt(RESPONSE_CODE) == 0) {

					final ArrayList<String> myPurchases = bundle.getStringArrayList(RESPONSE_INAPP_PURCHASE_DATA_LIST);
					final ArrayList<String> mySignatures = bundle.getStringArrayList(RESPONSE_INAPP_SIGNATURE_LIST);

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

							pc.setConsumableValue("ticket_signature", sku, signature);
							pc.setConsumableValue("ticket", sku, receipt);
							pc.setConsumableFlag("block", sku, true);
							pc.setConsumableValue("token", sku, token);

							godotPaymentV3.callbackPurchased(receipt, signature, sku);
						} catch (JSONException e) {
						}
					}
				}
				continueToken = bundle.getString(INAPP_CONTINUATION_TOKEN);
				Log.d(TAG, "continue token = " + continueToken);
			} while (!TextUtils.isEmpty(continueToken));
		} catch (Exception e) {
			Log.d(TAG, "Error requesting purchased products:" + e.getClass().getName() + ":" + e.getMessage());
		}
	}

	public void processPurchaseResponse(int resultCode, Intent data) {
		HandlePurchaseTask handlePurchaseTask = new HandlePurchaseTask(activity) {
			@Override
			protected void success(final String sku, final String signature, final String ticket) {
				godotPaymentV3.callbackSuccess(ticket, signature, sku);

				if (auto_consume) {
					new ConsumeTask(mService, activity) {
						@Override
						protected void success(String ticket) {
						}

						@Override
						protected void error(String message) {
							godotPaymentV3.callbackFail();
						}
					}
							.consume(sku);
				}
			}

			@Override
			protected void error(String message) {
				godotPaymentV3.callbackFail();
			}

			@Override
			protected void canceled() {
				godotPaymentV3.callbackCancel();
			}
		};

		handlePurchaseTask.handlePurchaseRequest(resultCode, data);
	}

	public void validatePurchase(String purchaseToken, final String sku) {

		new ValidateTask(activity, godotPaymentV3) {
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
				}
						.consume(sku);
			}

			@Override
			protected void error(String message) {
				godotPaymentV3.callbackFail();
			}

			@Override
			protected void canceled() {
				godotPaymentV3.callbackCancel();
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
				godotPaymentV3.callbackSuccessProductMassConsumed(ticket, "", sku);
			}

			@Override
			protected void error(String message) {
				godotPaymentV3.callbackFailConsume();
			}
		}
				.consume(sku);
	}

	// Workaround to bug where sometimes response codes come as Long instead of Integer
	int getResponseCodeFromBundle(Bundle b) {
		Object o = b.get(RESPONSE_CODE);
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

				if (mService == null) {
					godotPaymentV3.errorSkuDetail("Payment manager is not initialized");

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
						skuDetails = mService.getSkuDetails(3, activity.getPackageName(), ITEM_TYPE_INAPP, querySkus);
						if (!skuDetails.containsKey(RESPONSE_GET_SKU_DETAILS_LIST)) {
							int response = getResponseCodeFromBundle(skuDetails);
							if (response != BILLING_RESPONSE_RESULT_OK) {
								godotPaymentV3.errorSkuDetail(getResponseDesc(response));
							} else {
								godotPaymentV3.errorSkuDetail("No error but no detail list.");
							}
							return;
						}

						ArrayList<String> responseList = skuDetails.getStringArrayList(RESPONSE_GET_SKU_DETAILS_LIST);

						for (String thisResponse : responseList) {
							Log.d(TAG, "response = " + thisResponse);
							godotPaymentV3.addSkuDetail(thisResponse);
						}
					} catch (RemoteException e) {
						e.printStackTrace();
						godotPaymentV3.errorSkuDetail("RemoteException error!");
					}
				}
				godotPaymentV3.completeSkuDetail();
			}
		}))
				.start();
	}

	public void setBaseSingleton(GodotPaymentV3 godotPaymentV3) {
		this.godotPaymentV3 = godotPaymentV3;
	}
}
