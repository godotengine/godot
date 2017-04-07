/*************************************************************************/
/*  payment_service.cpp                                                  */
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
#ifdef PAYMENT_SERVICE_ENABLED

#include "payment_service.h"

#include "bbutil.h"
#include <errno.h>
#include <string.h>
#include <unistd.h>
extern char *launch_dir_ptr;

void PaymentService::_bind_methods() {

	ClassDB::bind_method(D_METHOD("request_product_info"), &PaymentService::request_product_info);
	ClassDB::bind_method(D_METHOD("purchase"), &PaymentService::purchase);

	ClassDB::bind_method(D_METHOD("get_pending_event_count"), &PaymentService::get_pending_event_count);
	ClassDB::bind_method(D_METHOD("pop_pending_event"), &PaymentService::pop_pending_event);
};

Error PaymentService::request_product_info(Variant p_params) {

	return ERR_UNAVAILABLE;
};

Error PaymentService::purchase(Variant p_params) {

	Dictionary params = p_params;
	ERR_FAIL_COND_V((!params.has("product_id")) && (!params.has("product_sku")), ERR_INVALID_PARAMETER);

	char *id = NULL;
	char *sku = NULL;

	CharString p_id = params.has("product_id") ? String(params["product_id"]).ascii() : CharString();
	CharString p_sku = params.has("product_sku") ? String(params["product_sku"]).ascii() : CharString();
	unsigned int request_id;
	chdir(launch_dir_ptr);
	int ret = paymentservice_purchase_request(params.has("product_sku") ? NULL : p_id.get_data(),
			params.has("product_sku") ? p_sku.get_data() : NULL,
			NULL, NULL, NULL, NULL, get_window_group_id(), &request_id);
	chdir("app/native");

	if (ret != BPS_SUCCESS) {
		int eret = errno;
		printf("purchase error %i, %x, %i, %x\n", ret, ret, eret, eret);
		ERR_FAIL_V((Error)eret);
		return (Error)eret;
	};
	return OK;
};

bool PaymentService::handle_event(bps_event_t *p_event) {

	if (bps_event_get_domain(p_event) != paymentservice_get_domain()) {
		return false;
	};

	Dictionary dict;
	int res = paymentservice_event_get_response_code(p_event);
	if (res == SUCCESS_RESPONSE) {
		dict["result"] = "ok";

		res = bps_event_get_code(p_event);
		if (res == PURCHASE_RESPONSE) {
			dict["type"] = "purchase";
			const char *pid = paymentservice_event_get_digital_good_id(p_event, 0);
			dict["product_id"] = String(pid ? pid : "");
		};

	} else {
		const char *desc = paymentservice_event_get_error_text(p_event);
		if (strcmp(desc, "alreadyPurchased") == 0) {
			dict["result"] = "ok";
		} else {
			dict["result"] = "error";
			dict["error_description"] = paymentservice_event_get_error_text(p_event);
			dict["error_code"] = paymentservice_event_get_error_id(p_event);
			printf("error code is %i\n", paymentservice_event_get_error_id(p_event));
			printf("error description is %s\n", paymentservice_event_get_error_text(p_event));
		};
		dict["product_id"] = "";
	};

	res = bps_event_get_code(p_event);
	if (res == PURCHASE_RESPONSE) {
		dict["type"] = "purchase";
	};

	printf("********** adding event with result %ls\n", String(dict["result"]).c_str());
	pending_events.push_back(dict);

	return true;
};

int PaymentService::get_pending_event_count() {
	return pending_events.size();
};

Variant PaymentService::pop_pending_event() {

	Variant front = pending_events.front()->get();
	pending_events.pop_front();

	return front;
};

PaymentService::PaymentService() {

	paymentservice_request_events(0);
#ifdef DEBUG_ENABLED
	paymentservice_set_connection_mode(true);
#else
	paymentservice_set_connection_mode(false);
#endif
};

PaymentService::~PaymentService(){

};

#endif
