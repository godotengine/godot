/*************************************************************************/
/*  ip.cpp                                                               */
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
#include "ip.h"
#include "hash_map.h"
#include "os/semaphore.h"
#include "os/thread.h"

VARIANT_ENUM_CAST(IP::ResolverStatus);

/************* RESOLVER ******************/

struct _IP_ResolverPrivate {

	struct QueueItem {

		volatile IP::ResolverStatus status;
		IP_Address response;
		String hostname;
		IP::Type type;

		void clear() {
			status = IP::RESOLVER_STATUS_NONE;
			response = IP_Address();
			type = IP::TYPE_NONE;
			hostname = "";
		};

		QueueItem() {
			clear();
		};
	};

	QueueItem queue[IP::RESOLVER_MAX_QUERIES];

	IP::ResolverID find_empty_id() const {

		for (int i = 0; i < IP::RESOLVER_MAX_QUERIES; i++) {
			if (queue[i].status == IP::RESOLVER_STATUS_NONE)
				return i;
		}
		return IP::RESOLVER_INVALID_ID;
	}

	Semaphore *sem;

	Thread *thread;
	//Semaphore* semaphore;
	bool thread_abort;

	void resolve_queues() {

		for (int i = 0; i < IP::RESOLVER_MAX_QUERIES; i++) {

			if (queue[i].status != IP::RESOLVER_STATUS_WAITING)
				continue;
			queue[i].response = IP::get_singleton()->resolve_hostname(queue[i].hostname, queue[i].type);

			if (!queue[i].response.is_valid())
				queue[i].status = IP::RESOLVER_STATUS_ERROR;
			else
				queue[i].status = IP::RESOLVER_STATUS_DONE;
		}
	}

	static void _thread_function(void *self) {

		_IP_ResolverPrivate *ipr = (_IP_ResolverPrivate *)self;

		while (!ipr->thread_abort) {

			ipr->sem->wait();
			GLOBAL_LOCK_FUNCTION;
			ipr->resolve_queues();
		}
	}

	HashMap<String, IP_Address> cache;

	static String get_cache_key(String p_hostname, IP::Type p_type) {
		return itos(p_type) + p_hostname;
	}
};

IP_Address IP::resolve_hostname(const String &p_hostname, IP::Type p_type) {

	GLOBAL_LOCK_FUNCTION;

	String key = _IP_ResolverPrivate::get_cache_key(p_hostname, p_type);
	if (resolver->cache.has(key))
		return resolver->cache[key];

	IP_Address res = _resolve_hostname(p_hostname, p_type);
	resolver->cache[key] = res;
	return res;
}
IP::ResolverID IP::resolve_hostname_queue_item(const String &p_hostname, IP::Type p_type) {

	GLOBAL_LOCK_FUNCTION;

	ResolverID id = resolver->find_empty_id();

	if (id == RESOLVER_INVALID_ID) {
		WARN_PRINT("Out of resolver queries");
		return id;
	}

	String key = _IP_ResolverPrivate::get_cache_key(p_hostname, p_type);
	resolver->queue[id].hostname = p_hostname;
	resolver->queue[id].type = p_type;
	if (resolver->cache.has(key)) {
		resolver->queue[id].response = resolver->cache[key];
		resolver->queue[id].status = IP::RESOLVER_STATUS_DONE;
	} else {
		resolver->queue[id].response = IP_Address();
		resolver->queue[id].status = IP::RESOLVER_STATUS_WAITING;
		if (resolver->thread)
			resolver->sem->post();
		else
			resolver->resolve_queues();
	}

	return id;
}

IP::ResolverStatus IP::get_resolve_item_status(ResolverID p_id) const {

	ERR_FAIL_INDEX_V(p_id, IP::RESOLVER_MAX_QUERIES, IP::RESOLVER_STATUS_NONE);

	GLOBAL_LOCK_FUNCTION;
	ERR_FAIL_COND_V(resolver->queue[p_id].status == IP::RESOLVER_STATUS_NONE, IP::RESOLVER_STATUS_NONE);

	return resolver->queue[p_id].status;
}
IP_Address IP::get_resolve_item_address(ResolverID p_id) const {

	ERR_FAIL_INDEX_V(p_id, IP::RESOLVER_MAX_QUERIES, IP_Address());

	GLOBAL_LOCK_FUNCTION;

	if (resolver->queue[p_id].status != IP::RESOLVER_STATUS_DONE) {
		ERR_EXPLAIN("Resolve of '" + resolver->queue[p_id].hostname + "'' didn't complete yet.");
		ERR_FAIL_COND_V(resolver->queue[p_id].status != IP::RESOLVER_STATUS_DONE, IP_Address());
	}

	return resolver->queue[p_id].response;
}
void IP::erase_resolve_item(ResolverID p_id) {

	ERR_FAIL_INDEX(p_id, IP::RESOLVER_MAX_QUERIES);

	GLOBAL_LOCK_FUNCTION;

	resolver->queue[p_id].status = IP::RESOLVER_STATUS_NONE;
}

void IP::clear_cache(const String &p_hostname) {

	if (p_hostname.empty()) {
		resolver->cache.clear();
	} else {
		resolver->cache.erase(_IP_ResolverPrivate::get_cache_key(p_hostname, IP::TYPE_NONE));
		resolver->cache.erase(_IP_ResolverPrivate::get_cache_key(p_hostname, IP::TYPE_IPV4));
		resolver->cache.erase(_IP_ResolverPrivate::get_cache_key(p_hostname, IP::TYPE_IPV6));
		resolver->cache.erase(_IP_ResolverPrivate::get_cache_key(p_hostname, IP::TYPE_ANY));
	}
};

Array IP::_get_local_addresses() const {

	Array addresses;
	List<IP_Address> ip_addresses;
	get_local_addresses(&ip_addresses);
	for (List<IP_Address>::Element *E = ip_addresses.front(); E; E = E->next()) {
		addresses.push_back(E->get());
	}

	return addresses;
}

void IP::_bind_methods() {

	ClassDB::bind_method(D_METHOD("resolve_hostname", "host", "ip_type"), &IP::resolve_hostname, DEFVAL(IP::TYPE_ANY));
	ClassDB::bind_method(D_METHOD("resolve_hostname_queue_item", "host", "ip_type"), &IP::resolve_hostname_queue_item, DEFVAL(IP::TYPE_ANY));
	ClassDB::bind_method(D_METHOD("get_resolve_item_status", "id"), &IP::get_resolve_item_status);
	ClassDB::bind_method(D_METHOD("get_resolve_item_address", "id"), &IP::get_resolve_item_address);
	ClassDB::bind_method(D_METHOD("erase_resolve_item", "id"), &IP::erase_resolve_item);
	ClassDB::bind_method(D_METHOD("get_local_addresses"), &IP::_get_local_addresses);
	ClassDB::bind_method(D_METHOD("clear_cache"), &IP::clear_cache, DEFVAL(""));

	BIND_CONSTANT(RESOLVER_STATUS_NONE);
	BIND_CONSTANT(RESOLVER_STATUS_WAITING);
	BIND_CONSTANT(RESOLVER_STATUS_DONE);
	BIND_CONSTANT(RESOLVER_STATUS_ERROR);

	BIND_CONSTANT(RESOLVER_MAX_QUERIES);
	BIND_CONSTANT(RESOLVER_INVALID_ID);

	BIND_CONSTANT(TYPE_NONE);
	BIND_CONSTANT(TYPE_IPV4);
	BIND_CONSTANT(TYPE_IPV6);
	BIND_CONSTANT(TYPE_ANY);
}

IP *IP::singleton = NULL;

IP *IP::get_singleton() {

	return singleton;
}

IP *(*IP::_create)() = NULL;

IP *IP::create() {

	ERR_FAIL_COND_V(singleton, NULL);
	ERR_FAIL_COND_V(!_create, NULL);
	return _create();
}

IP::IP() {

	singleton = this;
	resolver = memnew(_IP_ResolverPrivate);
	resolver->sem = NULL;

#ifndef NO_THREADS

	//resolver->sem = Semaphore::create();

	resolver->sem = NULL;
	if (resolver->sem) {
		resolver->thread_abort = false;

		resolver->thread = Thread::create(_IP_ResolverPrivate::_thread_function, resolver);

		if (!resolver->thread)
			memdelete(resolver->sem); //wtf
	} else {
		resolver->thread = NULL;
	}
#else
	resolver->sem = NULL;
	resolver->thread = NULL;
#endif
}

IP::~IP() {

#ifndef NO_THREADS
	if (resolver->thread) {
		resolver->thread_abort = true;
		resolver->sem->post();
		Thread::wait_to_finish(resolver->thread);
		memdelete(resolver->thread);
		memdelete(resolver->sem);
	}
	memdelete(resolver);

#endif
}
