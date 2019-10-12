/*
 * Copyright (C) 2012 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.android.vending.billing;

import android.os.Bundle;

/**
 * InAppBillingService is the service that provides in-app billing version 3 and beyond.
 * This service provides the following features:
 * 1. Provides a new API to get details of in-app items published for the app including
 *    price, type, title and description.
 * 2. The purchase flow is synchronous and purchase information is available immediately
 *    after it completes.
 * 3. Purchase information of in-app purchases is maintained within the Google Play system
 *    till the purchase is consumed.
 * 4. An API to consume a purchase of an inapp item. All purchases of one-time
 *    in-app items are consumable and thereafter can be purchased again.
 * 5. An API to get current purchases of the user immediately. This will not contain any
 *    consumed purchases.
 *
 * All calls will give a response code with the following possible values
 * RESULT_OK = 0 - success
 * RESULT_USER_CANCELED = 1 - User pressed back or canceled a dialog
 * RESULT_SERVICE_UNAVAILABLE = 2 - The network connection is down
 * RESULT_BILLING_UNAVAILABLE = 3 - This billing API version is not supported for the type requested
 * RESULT_ITEM_UNAVAILABLE = 4 - Requested SKU is not available for purchase
 * RESULT_DEVELOPER_ERROR = 5 - Invalid arguments provided to the API
 * RESULT_ERROR = 6 - Fatal error during the API action
 * RESULT_ITEM_ALREADY_OWNED = 7 - Failure to purchase since item is already owned
 * RESULT_ITEM_NOT_OWNED = 8 - Failure to consume since item is not owned
 */
interface IInAppBillingService {
    /**
     * Checks support for the requested billing API version, package and in-app type.
     * Minimum API version supported by this interface is 3.
     * @param apiVersion billing API version that the app is using
     * @param packageName the package name of the calling app
     * @param type type of the in-app item being purchased ("inapp" for one-time purchases
     *        and "subs" for subscriptions)
     * @return RESULT_OK(0) on success and appropriate response code on failures.
     */
    int isBillingSupported(int apiVersion, String packageName, String type);

    /**
     * Provides details of a list of SKUs
     * Given a list of SKUs of a valid type in the skusBundle, this returns a bundle
     * with a list JSON strings containing the productId, price, title and description.
     * This API can be called with a maximum of 20 SKUs.
     * @param apiVersion billing API version that the app is using
     * @param packageName the package name of the calling app
     * @param type of the in-app items ("inapp" for one-time purchases
     *        and "subs" for subscriptions)
     * @param skusBundle bundle containing a StringArrayList of SKUs with key "ITEM_ID_LIST"
     * @return Bundle containing the following key-value pairs
     *         "RESPONSE_CODE" with int value, RESULT_OK(0) if success, appropriate response codes
     *                         on failures.
     *         "DETAILS_LIST" with a StringArrayList containing purchase information
     *                        in JSON format similar to:
     *                        '{ "productId" : "exampleSku",
     *                           "type" : "inapp",
     *                           "price" : "$5.00",
     *                           "price_currency": "USD",
     *                           "price_amount_micros": 5000000,
     *                           "title : "Example Title",
     *                           "description" : "This is an example description" }'
     */
    Bundle getSkuDetails(int apiVersion, String packageName, String type, in Bundle skusBundle);

    /**
     * Returns a pending intent to launch the purchase flow for an in-app item by providing a SKU,
     * the type, a unique purchase token and an optional developer payload.
     * @param apiVersion billing API version that the app is using
     * @param packageName package name of the calling app
     * @param sku the SKU of the in-app item as published in the developer console
     * @param type of the in-app item being purchased ("inapp" for one-time purchases
     *        and "subs" for subscriptions)
     * @param developerPayload optional argument to be sent back with the purchase information
     * @return Bundle containing the following key-value pairs
     *         "RESPONSE_CODE" with int value, RESULT_OK(0) if success, appropriate response codes
     *                         on failures.
     *         "BUY_INTENT" - PendingIntent to start the purchase flow
     *
     * The Pending intent should be launched with startIntentSenderForResult. When purchase flow
     * has completed, the onActivityResult() will give a resultCode of OK or CANCELED.
     * If the purchase is successful, the result data will contain the following key-value pairs
     *         "RESPONSE_CODE" with int value, RESULT_OK(0) if success, appropriate response
     *                         codes on failures.
     *         "INAPP_PURCHASE_DATA" - String in JSON format similar to
     *                                 '{"orderId":"12999763169054705758.1371079406387615",
     *                                   "packageName":"com.example.app",
     *                                   "productId":"exampleSku",
     *                                   "purchaseTime":1345678900000,
     *                                   "purchaseToken" : "122333444455555",
     *                                   "developerPayload":"example developer payload" }'
     *         "INAPP_DATA_SIGNATURE" - String containing the signature of the purchase data that
     *                                  was signed with the private key of the developer
     */
    Bundle getBuyIntent(int apiVersion, String packageName, String sku, String type,
        String developerPayload);

    /**
     * Returns the current SKUs owned by the user of the type and package name specified along with
     * purchase information and a signature of the data to be validated.
     * This will return all SKUs that have been purchased in V3 and managed items purchased using
     * V1 and V2 that have not been consumed.
     * @param apiVersion billing API version that the app is using
     * @param packageName package name of the calling app
     * @param type of the in-app items being requested ("inapp" for one-time purchases
     *        and "subs" for subscriptions)
     * @param continuationToken to be set as null for the first call, if the number of owned
     *        skus are too many, a continuationToken is returned in the response bundle.
     *        This method can be called again with the continuation token to get the next set of
     *        owned skus.
     * @return Bundle containing the following key-value pairs
     *         "RESPONSE_CODE" with int value, RESULT_OK(0) if success, appropriate response codes
                               on failures.
     *         "INAPP_PURCHASE_ITEM_LIST" - StringArrayList containing the list of SKUs
     *         "INAPP_PURCHASE_DATA_LIST" - StringArrayList containing the purchase information
     *         "INAPP_DATA_SIGNATURE_LIST"- StringArrayList containing the signatures
     *                                      of the purchase information
     *         "INAPP_CONTINUATION_TOKEN" - String containing a continuation token for the
     *                                      next set of in-app purchases. Only set if the
     *                                      user has more owned skus than the current list.
     */
    Bundle getPurchases(int apiVersion, String packageName, String type, String continuationToken);

    /**
     * Consume the last purchase of the given SKU. This will result in this item being removed
     * from all subsequent responses to getPurchases() and allow re-purchase of this item.
     * @param apiVersion billing API version that the app is using
     * @param packageName package name of the calling app
     * @param purchaseToken token in the purchase information JSON that identifies the purchase
     *        to be consumed
     * @return RESULT_OK(0) if consumption succeeded, appropriate response codes on failures.
     */
    int consumePurchase(int apiVersion, String packageName, String purchaseToken);

    /**
     * This API is currently under development.
     */
    int stub(int apiVersion, String packageName, String type);

    /**
     * Returns a pending intent to launch the purchase flow for upgrading or downgrading a
     * subscription. The existing owned SKU(s) should be provided along with the new SKU that
     * the user is upgrading or downgrading to.
     * @param apiVersion billing API version that the app is using, must be 5 or later
     * @param packageName package name of the calling app
     * @param oldSkus the SKU(s) that the user is upgrading or downgrading from,
     *        if null or empty this method will behave like {@link #getBuyIntent}
     * @param newSku the SKU that the user is upgrading or downgrading to
     * @param type of the item being purchased, currently must be "subs"
     * @param developerPayload optional argument to be sent back with the purchase information
     * @return Bundle containing the following key-value pairs
     *         "RESPONSE_CODE" with int value, RESULT_OK(0) if success, appropriate response codes
     *                         on failures.
     *         "BUY_INTENT" - PendingIntent to start the purchase flow
     *
     * The Pending intent should be launched with startIntentSenderForResult. When purchase flow
     * has completed, the onActivityResult() will give a resultCode of OK or CANCELED.
     * If the purchase is successful, the result data will contain the following key-value pairs
     *         "RESPONSE_CODE" with int value, RESULT_OK(0) if success, appropriate response
     *                         codes on failures.
     *         "INAPP_PURCHASE_DATA" - String in JSON format similar to
     *                                 '{"orderId":"12999763169054705758.1371079406387615",
     *                                   "packageName":"com.example.app",
     *                                   "productId":"exampleSku",
     *                                   "purchaseTime":1345678900000,
     *                                   "purchaseToken" : "122333444455555",
     *                                   "developerPayload":"example developer payload" }'
     *         "INAPP_DATA_SIGNATURE" - String containing the signature of the purchase data that
     *                                  was signed with the private key of the developer
     */
    Bundle getBuyIntentToReplaceSkus(int apiVersion, String packageName,
        in List<String> oldSkus, String newSku, String type, String developerPayload);

    /**
     * Returns a pending intent to launch the purchase flow for an in-app item. This method is
     * a variant of the {@link #getBuyIntent} method and takes an additional {@code extraParams}
     * parameter. This parameter is a Bundle of optional keys and values that affect the
     * operation of the method.
     * @param apiVersion billing API version that the app is using, must be 6 or later
     * @param packageName package name of the calling app
     * @param sku the SKU of the in-app item as published in the developer console
     * @param type of the in-app item being purchased ("inapp" for one-time purchases
     *        and "subs" for subscriptions)
     * @param developerPayload optional argument to be sent back with the purchase information
     * @extraParams a Bundle with the following optional keys:
     *        "skusToReplace" - List<String> - an optional list of SKUs that the user is
     *                          upgrading or downgrading from.
     *                          Pass this field if the purchase is upgrading or downgrading
     *                          existing subscriptions.
     *                          The specified SKUs are replaced with the SKUs that the user is
     *                          purchasing. Google Play replaces the specified SKUs at the start of
     *                          the next billing cycle.
     * "replaceSkusProration" - Boolean - whether the user should be credited for any unused
     *                          subscription time on the SKUs they are upgrading or downgrading.
     *                          If you set this field to true, Google Play swaps out the old SKUs
     *                          and credits the user with the unused value of their subscription
     *                          time on a pro-rated basis.
     *                          Google Play applies this credit to the new subscription, and does
     *                          not begin billing the user for the new subscription until after
     *                          the credit is used up.
     *                          If you set this field to false, the user does not receive credit for
     *                          any unused subscription time and the recurrence date does not
     *                          change.
     *                          Default value is true. Ignored if you do not pass skusToReplace.
     *            "accountId" - String - an optional obfuscated string that is uniquely
     *                          associated with the user's account in your app.
     *                          If you pass this value, Google Play can use it to detect irregular
     *                          activity, such as many devices making purchases on the same
     *                          account in a short period of time.
     *                          Do not use the developer ID or the user's Google ID for this field.
     *                          In addition, this field should not contain the user's ID in
     *                          cleartext.
     *                          We recommend that you use a one-way hash to generate a string from
     *                          the user's ID, and store the hashed string in this field.
     *                   "vr" - Boolean - an optional flag indicating whether the returned intent
     *                          should start a VR purchase flow. The apiVersion must also be 7 or
     *                          later to use this flag.
     */
    Bundle getBuyIntentExtraParams(int apiVersion, String packageName, String sku,
        String type, String developerPayload, in Bundle extraParams);

    /**
     * Returns the most recent purchase made by the user for each SKU, even if that purchase is
     * expired, canceled, or consumed.
     * @param apiVersion billing API version that the app is using, must be 6 or later
     * @param packageName package name of the calling app
     * @param type of the in-app items being requested ("inapp" for one-time purchases
     *        and "subs" for subscriptions)
     * @param continuationToken to be set as null for the first call, if the number of owned
     *        skus is too large, a continuationToken is returned in the response bundle.
     *        This method can be called again with the continuation token to get the next set of
     *        owned skus.
     * @param extraParams a Bundle with extra params that would be appended into http request
     *        query string. Not used at this moment. Reserved for future functionality.
     * @return Bundle containing the following key-value pairs
     *         "RESPONSE_CODE" with int value: RESULT_OK(0) if success,
     *         {@link IabHelper#BILLING_RESPONSE_RESULT_*} response codes on failures.
     *
     *         "INAPP_PURCHASE_ITEM_LIST" - ArrayList<String> containing the list of SKUs
     *         "INAPP_PURCHASE_DATA_LIST" - ArrayList<String> containing the purchase information
     *         "INAPP_DATA_SIGNATURE_LIST"- ArrayList<String> containing the signatures
     *                                      of the purchase information
     *         "INAPP_CONTINUATION_TOKEN" - String containing a continuation token for the
     *                                      next set of in-app purchases. Only set if the
     *                                      user has more owned skus than the current list.
     */
    Bundle getPurchaseHistory(int apiVersion, String packageName, String type,
        String continuationToken, in Bundle extraParams);

    /**
    * This method is a variant of {@link #isBillingSupported}} that takes an additional
    * {@code extraParams} parameter.
    * @param apiVersion billing API version that the app is using, must be 7 or later
    * @param packageName package name of the calling app
    * @param type of the in-app item being purchased ("inapp" for one-time purchases and "subs"
    *        for subscriptions)
    * @param extraParams a Bundle with the following optional keys:
    *        "vr" - Boolean - an optional flag to indicate whether {link #getBuyIntentExtraParams}
    *               supports returning a VR purchase flow.
    * @return RESULT_OK(0) on success and appropriate response code on failures.
    */
    int isBillingSupportedExtraParams(int apiVersion, String packageName, String type,
        in Bundle extraParams);
}
