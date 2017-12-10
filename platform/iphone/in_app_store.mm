/*************************************************************************/
/*  in_app_store.mm                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
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
#ifdef STOREKIT_ENABLED

#include "in_app_store.h"

extern "C" {
#import <Foundation/Foundation.h>
#import <StoreKit/StoreKit.h>
};

bool auto_finish_transactions = true;
NSMutableDictionary *pending_transactions = [NSMutableDictionary dictionary];

@interface SKProduct (LocalizedPrice)
@property(nonatomic, readonly) NSString *localizedPrice;
@end

//----------------------------------//
// SKProduct extension
//----------------------------------//
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
	ClassDB::bind_method(D_METHOD("purchase"), &InAppStore::purchase);

	ClassDB::bind_method(D_METHOD("get_pending_event_count"), &InAppStore::get_pending_event_count);
	ClassDB::bind_method(D_METHOD("pop_pending_event"), &InAppStore::pop_pending_event);
	ClassDB::bind_method(D_METHOD("finish_transaction"), &InAppStore::finish_transaction);
	ClassDB::bind_method(D_METHOD("set_auto_finish_transaction"), &InAppStore::set_auto_finish_transaction);
};

@interface ProductsDelegate : NSObject <SKProductsRequestDelegate> {
};

@end

@implementation ProductsDelegate

- (void)productsRequest:(SKProductsRequest *)request didReceiveResponse:(SKProductsResponse *)response {

	NSArray *products = response.products;
	Dictionary ret;
	ret["type"] = "product_info";
	ret["result"] = "ok";
	PoolStringArray titles;
	PoolStringArray descriptions;
	PoolRealArray prices;
	PoolStringArray ids;
	PoolStringArray localized_prices;
	PoolStringArray currency_codes;

	for (int i = 0; i < [products count]; i++) {

		SKProduct *product = [products objectAtIndex:i];

		const char *str = [product.localizedTitle UTF8String];
		titles.push_back(String::utf8(str != NULL ? str : ""));

		str = [product.localizedDescription UTF8String];
		descriptions.push_back(String::utf8(str != NULL ? str : ""));
		prices.push_back([product.price doubleValue]);
		ids.push_back(String::utf8([product.productIdentifier UTF8String]));
		localized_prices.push_back(String::utf8([product.localizedPrice UTF8String]));
		currency_codes.push_back(String::utf8([[[product priceLocale] objectForKey:NSLocaleCurrencyCode] UTF8String]));
	};
	ret["titles"] = titles;
	ret["descriptions"] = descriptions;
	ret["prices"] = prices;
	ret["ids"] = ids;
	ret["localized_prices"] = localized_prices;
	ret["currency_codes"] = currency_codes;

	PoolStringArray invalid_ids;

	for (NSString *ipid in response.invalidProductIdentifiers) {

		invalid_ids.push_back(String::utf8([ipid UTF8String]));
	};
	ret["invalid_ids"] = invalid_ids;

	InAppStore::get_singleton()->_post_event(ret);

	[request release];
};

@end

Error InAppStore::request_product_info(Variant p_params) {

	Dictionary params = p_params;
	ERR_FAIL_COND_V(!params.has("product_ids"), ERR_INVALID_PARAMETER);

	PoolStringArray pids = params["product_ids"];
	printf("************ request product info! %i\n", pids.size());

	NSMutableArray *array = [[[NSMutableArray alloc] initWithCapacity:pids.size()] autorelease];
	for (int i = 0; i < pids.size(); i++) {
		printf("******** adding %ls to product list\n", pids[i].c_str());
		NSString *pid = [[[NSString alloc] initWithUTF8String:pids[i].utf8().get_data()] autorelease];
		[array addObject:pid];
	};

	NSSet *products = [[[NSSet alloc] initWithArray:array] autorelease];
	SKProductsRequest *request = [[SKProductsRequest alloc] initWithProductIdentifiers:products];

	ProductsDelegate *delegate = [[ProductsDelegate alloc] init];

	request.delegate = delegate;
	[request start];

	return OK;
};

@interface TransObserver : NSObject <SKPaymentTransactionObserver> {
};
@end

@implementation TransObserver

- (void)paymentQueue:(SKPaymentQueue *)queue updatedTransactions:(NSArray *)transactions {

	printf("transactions updated!\n");
	for (SKPaymentTransaction *transaction in transactions) {

		switch (transaction.transactionState) {
			case SKPaymentTransactionStatePurchased: {
				printf("status purchased!\n");
				String pid = String::utf8([transaction.payment.productIdentifier UTF8String]);
				String transactionId = String::utf8([transaction.transactionIdentifier UTF8String]);
				InAppStore::get_singleton()->_record_purchase(pid);
				Dictionary ret;
				ret["type"] = "purchase";
				ret["result"] = "ok";
				ret["product_id"] = pid;
				ret["transaction_id"] = transactionId;

				NSData *receipt = nil;
				int sdk_version = 6;

				if ([[[UIDevice currentDevice] systemVersion] floatValue] >= 7.0) {

					NSURL *receiptFileURL = nil;
					NSBundle *bundle = [NSBundle mainBundle];
					if ([bundle respondsToSelector:@selector(appStoreReceiptURL)]) {

						// Get the transaction receipt file path location in the app bundle.
						receiptFileURL = [bundle appStoreReceiptURL];

						// Read in the contents of the transaction file.
						receipt = [NSData dataWithContentsOfURL:receiptFileURL];
						sdk_version = 7;

					} else {
						// Fall back to deprecated transaction receipt,
						// which is still available in iOS 7.

						// Use SKPaymentTransaction's transactionReceipt.
						receipt = transaction.transactionReceipt;
					}

				} else {
					receipt = transaction.transactionReceipt;
				}

				NSString *receipt_to_send = nil;
				if (receipt != nil) {
					receipt_to_send = [receipt description];
				}
				Dictionary receipt_ret;
				receipt_ret["receipt"] = String::utf8(receipt_to_send != nil ? [receipt_to_send UTF8String] : "");
				receipt_ret["sdk"] = sdk_version;
				ret["receipt"] = receipt_ret;

				InAppStore::get_singleton()->_post_event(ret);

				if (auto_finish_transactions) {
					[[SKPaymentQueue defaultQueue] finishTransaction:transaction];
				} else {
					[pending_transactions setObject:transaction forKey:transaction.payment.productIdentifier];
				}

			}; break;
			case SKPaymentTransactionStateFailed: {
				printf("status transaction failed!\n");
				String pid = String::utf8([transaction.payment.productIdentifier UTF8String]);
				Dictionary ret;
				ret["type"] = "purchase";
				ret["result"] = "error";
				ret["product_id"] = pid;
				InAppStore::get_singleton()->_post_event(ret);
				[[SKPaymentQueue defaultQueue] finishTransaction:transaction];
			} break;
			case SKPaymentTransactionStateRestored: {
				printf("status transaction restored!\n");
				String pid = String::utf8([transaction.originalTransaction.payment.productIdentifier UTF8String]);
				InAppStore::get_singleton()->_record_purchase(pid);
				[[SKPaymentQueue defaultQueue] finishTransaction:transaction];
			} break;
			default: {
				printf("status default %i!\n", (int)transaction.transactionState);
			}; break;
		};
	};
};

@end

Error InAppStore::purchase(Variant p_params) {

	ERR_FAIL_COND_V(![SKPaymentQueue canMakePayments], ERR_UNAVAILABLE);
	if (![SKPaymentQueue canMakePayments])
		return ERR_UNAVAILABLE;

	printf("purchasing!\n");
	Dictionary params = p_params;
	ERR_FAIL_COND_V(!params.has("product_id"), ERR_INVALID_PARAMETER);

	NSString *pid = [[[NSString alloc] initWithUTF8String:String(params["product_id"]).utf8().get_data()] autorelease];
	SKPayment *payment = [SKPayment paymentWithProductIdentifier:pid];
	SKPaymentQueue *defq = [SKPaymentQueue defaultQueue];
	[defq addPayment:payment];
	printf("purchase sent!\n");

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
	//pending_transactions = [NSMutableDictionary dictionary];
};

void InAppStore::finish_transaction(String product_id) {
	NSString *prod_id = [NSString stringWithCString:product_id.utf8().get_data() encoding:NSUTF8StringEncoding];

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
