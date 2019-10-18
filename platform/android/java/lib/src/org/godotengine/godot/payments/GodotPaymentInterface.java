/*************************************************************************/
/*  GodotPaymentInterface.java                                           */
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

public interface GodotPaymentInterface {
	void purchase(String sku, String transactionId);

	void consumeUnconsumedPurchases();

	String getSignature();

	void callbackSuccess(String ticket, String signature, String sku);

	void callbackSuccessProductMassConsumed(String ticket, String signature, String sku);

	void callbackSuccessNoUnconsumedPurchases();

	void callbackFailConsume(String message);

	void callbackFail(String message);

	void callbackCancel();

	void callbackAlreadyOwned(String sku);

	int getPurchaseCallbackId();

	void setPurchaseCallbackId(int purchaseCallbackId);

	String getPurchaseValidationUrlPrefix();

	void setPurchaseValidationUrlPrefix(String url);

	String getAccessToken();

	void setAccessToken(String accessToken);

	void setTransactionId(String transactionId);

	String getTransactionId();

	// request purchased items are not consumed
	void requestPurchased();

	// callback for requestPurchased()
	void callbackPurchased(String receipt, String signature, String sku);

	void callbackDisconnected();

	void callbackConnected();

	// true if connected, false otherwise
	boolean isConnected();

	// consume item automatically after purchase. default is true.
	void setAutoConsume(boolean autoConsume);

	// consume a specific item
	void consume(String sku);

	// query in app item detail info
	void querySkuDetails(String[] list);

	void addSkuDetail(String itemJson);

	void completeSkuDetail();

	void errorSkuDetail(String errorMessage);
}
