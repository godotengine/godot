/*************************************************************************/
/*  in_app_store.mm                                                      */
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

#ifdef STOREKIT_ENABLED

#import "in_app_store.h"

extern "C" {
#import <Foundation/Foundation.h>
#import <StoreKit/StoreKit.h>
};

bool auto_finish_transactions = true;
NSMutableDictionary *pending_transactions = [NSMutableDictionary dictionary];

@interface SKProduct (LocalizedPrice)

@property(nonatomic, readonly) NSString *localizedPrice;

@end

@implementation SKProduct (LocalizedPrice)

- (NSString *)localizedPrice {
	NSNumberFormatter *numberFormatter = [[NSNumberFormatter alloc] init];
	[numberFormatter setFormatterBehavior:NSNumberFormatterBehavior10_4];
	[numberFormatter setNumberStyle:NSNumberFormatterCurrencyStyle];
	[numberFormatter setLocale:self.priceLocale];
	NSString *formattedString = [numberFormatter stringFromNumber:self.price];
	[numberFormatter release];
	return formattedString;
}

@end

InAppStore *InAppStore::instance = NULL;

void InAppStore::_bind_methods() {
	ClassDB::bind_method(D_METHOD("request_product_info"), &InAppStore::request_product_info);
	ClassDB::bind_method(D_METHOD("restore_purchases"), &InAppStore::restore_purchases);
	ClassDB::bind_method(D_METHOD("purchase"), &InAppStore::purchase);

	ClassDB::bind_method(D_METHOD("get_pending_event_count"), &InAppStore::get_pending_event_count);
	ClassDB::bind_method(D_METHOD("pop_pending_event"), &InAppStore::pop_pending_event);
	ClassDB::bind_method(D_METHOD("finish_transaction"), &InAppStore::finish_transaction);
	ClassDB::bind_method(D_METHOD("set_auto_finish_transaction"), &InAppStore::set_auto_finish_transaction);
}

static inline Dictionary TransactionDictionary(SKPaymentTransaction *transaction) {
	Dictionary ret;
	ret["result"] = "ok";
	ret["product_id"] = String::utf8([transaction.payment.productIdentifier UTF8String]);
	if (transaction.transactionState == SKPaymentTransactionStatePurchased) {
		ret["type"] = "purchase";
		ret["transaction_id"] = String::utf8([transaction.transactionIdentifier UTF8String]);
	} else if (transaction.transactionState == SKPaymentTransactionStateFailed) {
		ret["type"] = "purchase";
		ret["error"] = String::utf8([transaction.error.localizedDescription UTF8String]);
	} else if (transaction.transactionState == SKPaymentTransactionStateRestored) {
		ret["type"] = "restore";
	}
	return ret;
}

@interface ProductsDelegate : NSObject <SKProductsRequestDelegate>

@end

@implementation ProductsDelegate

- (void)productsRequest:(SKProductsRequest *)request didReceiveResponse:(SKProductsResponse *)response {
	PoolStringArray titles;
	PoolStringArray descriptions;
	PoolRealArray prices;
	PoolStringArray ids;
	PoolStringArray localized_prices;
	PoolStringArray currency_codes;

	for (SKProduct *product in response.products) {
		const char *title = [product.localizedTitle UTF8String];
		titles.push_back(String::utf8(title != NULL ? title : ""));
		const char *description = [product.localizedDescription UTF8String];
		descriptions.push_back(String::utf8(description != NULL ? description : ""));
		prices.push_back([product.price doubleValue]);
		ids.push_back(String::utf8([product.productIdentifier UTF8String]));
		localized_prices.push_back(String::utf8([product.localizedPrice UTF8String]));
		currency_codes.push_back(String::utf8([[[product priceLocale] objectForKey:NSLocaleCurrencyCode] UTF8String]));
	}

	Dictionary ret;
	ret["type"] = "product_info";
	ret["result"] = "ok";
	ret["titles"] = titles;
	ret["descriptions"] = descriptions;
	ret["prices"] = prices;
	ret["ids"] = ids;
	ret["localized_prices"] = localized_prices;
	ret["currency_codes"] = currency_codes;

	PoolStringArray invalid_ids;

	for (NSString *ipid in response.invalidProductIdentifiers) {
		invalid_ids.push_back(String::utf8([ipid UTF8String]));
	}
	ret["invalid_ids"] = invalid_ids;

	InAppStore::get_singleton()->_post_event(ret);

	[request release];
}

@end

Error InAppStore::request_product_info(Variant p_params) {

	Dictionary params = p_params;
	ERR_FAIL_COND_V(!params.has("product_ids"), ERR_INVALID_PARAMETER);

	PoolStringArray pids = params["product_ids"];
	printf("************ request product info! %i\n", pids.size());

	NSMutableSet *products = [NSMutableSet setWithCapacity:pids.size()];
	for (int i = 0; i < pids.size(); i++) {
		printf("******** adding %ls to product list\n", pids[i].c_str());
		[products addObject:@(pids[i].utf8().get_data())];
	}

	// Create a delegate to respond to this request
	ProductsDelegate *delegate = [[ProductsDelegate alloc] init];
	// Create a products request from the identifiers for StoreKit
	SKProductsRequest *request = [[SKProductsRequest alloc] initWithProductIdentifiers:[products copy]];
	request.delegate = delegate;
	// Start the request
	[request start];

	return OK;
}

Error InAppStore::restore_purchases() {

	printf("restoring purchases!\n");
	[[SKPaymentQueue defaultQueue] restoreCompletedTransactions];

	return OK;
}

@interface TransObserver : NSObject <SKPaymentTransactionObserver>

@end

@implementation TransObserver

- (void)paymentQueue:(SKPaymentQueue *)queue updatedTransactions:(NSArray *)transactions {

	printf("transactions updated!\n");
	for (SKPaymentTransaction *transaction in transactions) {
		// Create a variant with this info
		Dictionary ret = TransactionDictionary(transaction);

		String pid = String::utf8([transaction.payment.productIdentifier UTF8String]);

		switch (transaction.transactionState) {
			case SKPaymentTransactionStatePurchased: {
				printf("status purchased!\n");
				InAppStore::get_singleton()->_record_purchase(pid);

				// Get the transaction receipt file path location in the app bundle.
				NSURL *receiptFileURL = [[NSBundle mainBundle] appStoreReceiptURL];

				// Read in the contents of the transaction file.
				NSData *receipt = [NSData dataWithContentsOfURL:receiptFileURL];

				Dictionary receipt_ret;
				receipt_ret["receipt"] = String::utf8(receipt != nil ? [receipt.description UTF8String] : "");
				receipt_ret["sdk"] = 7;
				ret["receipt"] = receipt_ret;

				if (auto_finish_transactions) {
					[[SKPaymentQueue defaultQueue] finishTransaction:transaction];
				} else {
					[pending_transactions setObject:transaction forKey:transaction.payment.productIdentifier];
				}

			}; break;
			case SKPaymentTransactionStateFailed: {
				printf("status transaction failed!\n");
				[[SKPaymentQueue defaultQueue] finishTransaction:transaction];
			} break;
			case SKPaymentTransactionStateRestored: {
				printf("status transaction restored!\n");
				InAppStore::get_singleton()->_record_purchase(pid);
				[[SKPaymentQueue defaultQueue] finishTransaction:transaction];
			} break;
			default: {
				printf("status default %i!\n", (int)transaction.transactionState);
			}; break;
		}

		// Send the variant back to Godot
		InAppStore::get_singleton()->_post_event(ret);
	}
}

@end

Error InAppStore::purchase(Variant p_params) {

	ERR_FAIL_COND_V(![SKPaymentQueue canMakePayments], ERR_UNAVAILABLE);
	if (![SKPaymentQueue canMakePayments])
		return ERR_UNAVAILABLE;

	printf("attempting to purchase!\n");

	Dictionary params = p_params;
	ERR_FAIL_COND_V(!params.has("product_id"), ERR_INVALID_PARAMETER);

	NSString *pid = [[[NSString alloc] initWithUTF8String:String(params["product_id"]).utf8().get_data()] autorelease];
	SKPayment *payment = [SKPayment paymentWithProductIdentifier:pid];
	SKPaymentQueue *defq = [SKPaymentQueue defaultQueue];
	[defq addPayment:payment];

	printf("purchase queued!\n");

	return OK;
};

int InAppStore::get_pending_event_count() {
	return pending_events.size();
};

Variant InAppStore::pop_pending_event() {

	Variant front = pending_events.front()->get();
	pending_events.pop_front();

	return front;
};

void InAppStore::_post_event(Variant p_event) {

	pending_events.push_back(p_event);
};

void InAppStore::_record_purchase(String product_id) {

	String skey = "purchased/" + product_id;
	NSString *key = [[[NSString alloc] initWithUTF8String:skey.utf8().get_data()] autorelease];
	[[NSUserDefaults standardUserDefaults] setBool:YES forKey:key];
	[[NSUserDefaults standardUserDefaults] synchronize];
};

InAppStore *InAppStore::get_singleton() {

	return instance;
};

InAppStore::InAppStore() {
	ERR_FAIL_COND(instance != NULL);
	instance = this;
	auto_finish_transactions = false;

	TransObserver *observer = [[TransObserver alloc] init];
	[[SKPaymentQueue defaultQueue] addTransactionObserver:observer];
};

void InAppStore::finish_transaction(String product_id) {
	NSString *prod_id = [NSString stringWithCString:product_id.utf8().get_data()
										   encoding:NSUTF8StringEncoding];

	if ([pending_transactions objectForKey:prod_id]) {
		[[SKPaymentQueue defaultQueue] finishTransaction:[pending_transactions objectForKey:prod_id]];
		[pending_transactions removeObjectForKey:prod_id];
	}
};

void InAppStore::set_auto_finish_transaction(bool b) {
	auto_finish_transactions = b;
}

InAppStore::~InAppStore(){};

#endif
