/*************************************************************************/
/*  in_app_store.mm                                                      */
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

#ifdef STOREKIT_ENABLED

#include "in_app_store.h"

#import <Foundation/Foundation.h>
#import <StoreKit/StoreKit.h>

InAppStore *InAppStore::instance = NULL;

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
	return formattedString;
}

@end

@interface GodotProductsDelegate : NSObject <SKProductsRequestDelegate>

@property(nonatomic, strong) NSMutableArray *loadedProducts;
@property(nonatomic, strong) NSMutableArray *pendingRequests;

- (void)performRequestWithProductIDs:(NSSet *)productIDs;
- (Error)purchaseProductWithProductID:(NSString *)productID;
- (void)reset;

@end

@implementation GodotProductsDelegate

- (instancetype)init {
	self = [super init];

	if (self) {
		[self godot_commonInit];
	}

	return self;
}

- (void)godot_commonInit {
	self.loadedProducts = [NSMutableArray new];
	self.pendingRequests = [NSMutableArray new];
}

- (void)performRequestWithProductIDs:(NSSet *)productIDs {
	SKProductsRequest *request = [[SKProductsRequest alloc] initWithProductIdentifiers:productIDs];

	request.delegate = self;
	[self.pendingRequests addObject:request];
	[request start];
}

- (Error)purchaseProductWithProductID:(NSString *)productID {
	SKProduct *product = nil;

	NSLog(@"searching for product!");

	if (self.loadedProducts) {
		for (SKProduct *storedProduct in self.loadedProducts) {
			if ([storedProduct.productIdentifier isEqualToString:productID]) {
				product = storedProduct;
				break;
			}
		}
	}

	if (!product) {
		return ERR_INVALID_PARAMETER;
	}

	NSLog(@"product found!");

	SKPayment *payment = [SKPayment paymentWithProduct:product];
	[[SKPaymentQueue defaultQueue] addPayment:payment];

	NSLog(@"purchase sent!");

	return OK;
}

- (void)reset {
	[self.loadedProducts removeAllObjects];
	[self.pendingRequests removeAllObjects];
}

- (void)request:(SKRequest *)request didFailWithError:(NSError *)error {
	[self.pendingRequests removeObject:request];

	Dictionary ret;
	ret["type"] = "product_info";
	ret["result"] = "error";
	ret["error"] = String::utf8([error.localizedDescription UTF8String]);

	InAppStore::get_singleton()->_post_event(ret);
}

- (void)productsRequest:(SKProductsRequest *)request didReceiveResponse:(SKProductsResponse *)response {
	[self.pendingRequests removeObject:request];

	NSArray *products = response.products;
	[self.loadedProducts addObjectsFromArray:products];

	Dictionary ret;
	ret["type"] = "product_info";
	ret["result"] = "ok";
	PackedStringArray titles;
	PackedStringArray descriptions;
	PackedFloat32Array prices;
	PackedStringArray ids;
	PackedStringArray localized_prices;
	PackedStringArray currency_codes;

	for (NSUInteger i = 0; i < [products count]; i++) {
		SKProduct *product = [products objectAtIndex:i];

		const char *str = [product.localizedTitle UTF8String];
		titles.push_back(String::utf8(str != NULL ? str : ""));

		str = [product.localizedDescription UTF8String];
		descriptions.push_back(String::utf8(str != NULL ? str : ""));
		prices.push_back([product.price doubleValue]);
		ids.push_back(String::utf8([product.productIdentifier UTF8String]));
		localized_prices.push_back(String::utf8([product.localizedPrice UTF8String]));
		currency_codes.push_back(String::utf8([[[product priceLocale] objectForKey:NSLocaleCurrencyCode] UTF8String]));
	}

	ret["titles"] = titles;
	ret["descriptions"] = descriptions;
	ret["prices"] = prices;
	ret["ids"] = ids;
	ret["localized_prices"] = localized_prices;
	ret["currency_codes"] = currency_codes;

	PackedStringArray invalid_ids;

	for (NSString *ipid in response.invalidProductIdentifiers) {
		invalid_ids.push_back(String::utf8([ipid UTF8String]));
	}

	ret["invalid_ids"] = invalid_ids;

	InAppStore::get_singleton()->_post_event(ret);
}

@end

@interface GodotTransactionsObserver : NSObject <SKPaymentTransactionObserver>

@property(nonatomic, assign) BOOL shouldAutoFinishTransactions;
@property(nonatomic, strong) NSMutableDictionary *pendingTransactions;

- (void)finishTransactionWithProductID:(NSString *)productID;
- (void)reset;

@end

@implementation GodotTransactionsObserver

- (instancetype)init {
	self = [super init];

	if (self) {
		[self godot_commonInit];
	}

	return self;
}

- (void)godot_commonInit {
	self.pendingTransactions = [NSMutableDictionary new];
}

- (void)finishTransactionWithProductID:(NSString *)productID {
	SKPaymentTransaction *transaction = self.pendingTransactions[productID];

	if (transaction) {
		[[SKPaymentQueue defaultQueue] finishTransaction:transaction];
	}

	self.pendingTransactions[productID] = nil;
}

- (void)reset {
	[self.pendingTransactions removeAllObjects];
}

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
				int sdk_version = [[[UIDevice currentDevice] systemVersion] intValue];

				NSBundle *bundle = [NSBundle mainBundle];
				// Get the transaction receipt file path location in the app bundle.
				NSURL *receiptFileURL = [bundle appStoreReceiptURL];

				// Read in the contents of the transaction file.
				receipt = [NSData dataWithContentsOfURL:receiptFileURL];

				NSString *receipt_to_send = nil;

				if (receipt != nil) {
					receipt_to_send = [receipt base64EncodedStringWithOptions:0];
				}
				Dictionary receipt_ret;
				receipt_ret["receipt"] = String::utf8(receipt_to_send != nil ? [receipt_to_send UTF8String] : "");
				receipt_ret["sdk"] = sdk_version;
				ret["receipt"] = receipt_ret;

				InAppStore::get_singleton()->_post_event(ret);

				if (self.shouldAutoFinishTransactions) {
					[[SKPaymentQueue defaultQueue] finishTransaction:transaction];
				} else {
					self.pendingTransactions[transaction.payment.productIdentifier] = transaction;
				}

			} break;
			case SKPaymentTransactionStateFailed: {
				printf("status transaction failed!\n");
				String pid = String::utf8([transaction.payment.productIdentifier UTF8String]);
				Dictionary ret;
				ret["type"] = "purchase";
				ret["result"] = "error";
				ret["product_id"] = pid;
				ret["error"] = String::utf8([transaction.error.localizedDescription UTF8String]);
				InAppStore::get_singleton()->_post_event(ret);
				[[SKPaymentQueue defaultQueue] finishTransaction:transaction];
			} break;
			case SKPaymentTransactionStateRestored: {
				printf("status transaction restored!\n");
				String pid = String::utf8([transaction.originalTransaction.payment.productIdentifier UTF8String]);
				InAppStore::get_singleton()->_record_purchase(pid);
				Dictionary ret;
				ret["type"] = "restore";
				ret["result"] = "ok";
				ret["product_id"] = pid;
				InAppStore::get_singleton()->_post_event(ret);
				[[SKPaymentQueue defaultQueue] finishTransaction:transaction];
			} break;
			default: {
				printf("status default %i!\n", (int)transaction.transactionState);
			} break;
		}
	}
}

@end

void InAppStore::_bind_methods() {
	ClassDB::bind_method(D_METHOD("request_product_info"), &InAppStore::request_product_info);
	ClassDB::bind_method(D_METHOD("restore_purchases"), &InAppStore::restore_purchases);
	ClassDB::bind_method(D_METHOD("purchase"), &InAppStore::purchase);

	ClassDB::bind_method(D_METHOD("get_pending_event_count"), &InAppStore::get_pending_event_count);
	ClassDB::bind_method(D_METHOD("pop_pending_event"), &InAppStore::pop_pending_event);
	ClassDB::bind_method(D_METHOD("finish_transaction"), &InAppStore::finish_transaction);
	ClassDB::bind_method(D_METHOD("set_auto_finish_transaction"), &InAppStore::set_auto_finish_transaction);
}

Error InAppStore::request_product_info(Dictionary p_params) {
	ERR_FAIL_COND_V(!p_params.has("product_ids"), ERR_INVALID_PARAMETER);

	PackedStringArray pids = p_params["product_ids"];
	printf("************ request product info! %i\n", pids.size());

	NSMutableArray *array = [[NSMutableArray alloc] initWithCapacity:pids.size()];
	for (int i = 0; i < pids.size(); i++) {
		printf("******** adding %s to product list\n", pids[i].utf8().get_data());
		NSString *pid = [[NSString alloc] initWithUTF8String:pids[i].utf8().get_data()];
		[array addObject:pid];
	};

	NSSet *products = [[NSSet alloc] initWithArray:array];

	[products_request_delegate performRequestWithProductIDs:products];

	return OK;
}

Error InAppStore::restore_purchases() {
	printf("restoring purchases!\n");
	[[SKPaymentQueue defaultQueue] restoreCompletedTransactions];

	return OK;
}

Error InAppStore::purchase(Dictionary p_params) {
	ERR_FAIL_COND_V(![SKPaymentQueue canMakePayments], ERR_UNAVAILABLE);
	if (![SKPaymentQueue canMakePayments]) {
		return ERR_UNAVAILABLE;
	}

	printf("purchasing!\n");
	Dictionary params = p_params;
	ERR_FAIL_COND_V(!params.has("product_id"), ERR_INVALID_PARAMETER);

	NSString *pid = [[NSString alloc] initWithUTF8String:String(params["product_id"]).utf8().get_data()];

	return [products_request_delegate purchaseProductWithProductID:pid];
}

int InAppStore::get_pending_event_count() {
	return pending_events.size();
}

Variant InAppStore::pop_pending_event() {
	Variant front = pending_events.front()->get();
	pending_events.pop_front();

	return front;
}

void InAppStore::_post_event(Variant p_event) {
	pending_events.push_back(p_event);
}

void InAppStore::_record_purchase(String product_id) {
	String skey = "purchased/" + product_id;
	NSString *key = [[NSString alloc] initWithUTF8String:skey.utf8().get_data()];
	[[NSUserDefaults standardUserDefaults] setBool:YES forKey:key];
	[[NSUserDefaults standardUserDefaults] synchronize];
}

InAppStore *InAppStore::get_singleton() {
	return instance;
}

InAppStore::InAppStore() {
	ERR_FAIL_COND(instance != NULL);
	instance = this;

	products_request_delegate = [[GodotProductsDelegate alloc] init];
	transactions_observer = [[GodotTransactionsObserver alloc] init];

	[[SKPaymentQueue defaultQueue] addTransactionObserver:transactions_observer];
}

void InAppStore::finish_transaction(String product_id) {
	NSString *prod_id = [NSString stringWithCString:product_id.utf8().get_data() encoding:NSUTF8StringEncoding];

	[transactions_observer finishTransactionWithProductID:prod_id];
}

void InAppStore::set_auto_finish_transaction(bool b) {
	transactions_observer.shouldAutoFinishTransactions = b;
}

InAppStore::~InAppStore() {
	[products_request_delegate reset];
	[transactions_observer reset];

	products_request_delegate = nil;
	[[SKPaymentQueue defaultQueue] removeTransactionObserver:transactions_observer];
	transactions_observer = nil;
}

#endif
