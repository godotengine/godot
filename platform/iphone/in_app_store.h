/*************************************************************************/
/*  in_app_store.h                                                       */
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

#ifndef IN_APP_STORE_H
#define IN_APP_STORE_H

#include "core/class_db.h"

#ifdef __OBJC__
@class GodotProductsDelegate;
@class GodotTransactionsObserver;

typedef GodotProductsDelegate InAppStoreProductDelegate;
typedef GodotTransactionsObserver InAppStoreTransactionObserver;
#else
typedef void InAppStoreProductDelegate;
typedef void InAppStoreTransactionObserver;
#endif

class InAppStore : public Object {
	GDCLASS(InAppStore, Object);

	static InAppStore *instance;
	static void _bind_methods();

	List<Variant> pending_events;

	InAppStoreProductDelegate *products_request_delegate;
	InAppStoreTransactionObserver *transactions_observer;

public:
	Error request_product_info(Dictionary p_params);
	Error restore_purchases();
	Error purchase(Dictionary p_params);

	int get_pending_event_count();
	Variant pop_pending_event();
	void finish_transaction(String product_id);
	void set_auto_finish_transaction(bool b);

	void _post_event(Variant p_event);
	void _record_purchase(String product_id);

	static InAppStore *get_singleton();

	InAppStore();
	~InAppStore();
};

#endif

#endif
