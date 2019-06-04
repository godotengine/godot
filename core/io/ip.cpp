/*************************************************************************/
/*  ip.cpp                                                               */
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
#include "ip.h"
#include "hash_map.h"
#include "os/semaphore.h"
#include "os/thread.h"

VARIANT_ENUM_CAST(IP::ResolverStatus);

/************* RESOLVER ******************/

struct _IP_ResolverPrivate {

	struct QueueItem {

		volatile IP::ResolverStatus status;
		List<IP_Address> response;
		String hostname;
		IP::Type type;

		void clear() {
			status = IP::RESOLVER_STATUS_NONE;
			response.clear();
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

	Mutex *mutex;
	Semaphore *sem;

	Thread *thread;
	//Semaphore* semaphore;
	bool thread_abort;

	void resolve_queues() {

		for (int i = 0; i < IP::RESOLVER_MAX_QUERIES; i++) {

			if (queue[i].status != IP::RESOLVER_STATUS_WAITING)
				continue;
			IP::get_singleton()->_resolve_hostname(queue[i].response, queue[i].hostname, queue[i].type);
			queue[i].status = queue[i].response.empty() ? IP::RESOLVER_STATUS_ERROR : IP::RESOLVER_STATUS_DONE;
		}
	}

	static void _thread_function(void *self) {

		_IP_ResolverPrivate *ipr = (_IP_ResolverPrivate *)self;

		while (!ipr->thread_abort) {

			ipr->sem->wait();

			ipr->mutex->lock();
			ipr->resolve_queues();
			ipr->mutex->unlock();
		}
	}

	HashMap<String, List<IP_Address> > cache;

	static String get_cache_key(String p_hostname, IP::Type p_type) {
		return itos(p_type) + p_hostname;
	}
};

IP_Address IP::resolve_hostname(const String &p_hostname, IP::Type p_type) {

	resolver->mutex->lock();

	List<IP_Address> res;

	String key = _IP_ResolverPrivate::get_cache_key(p_hostname, p_type);
	if (resolver->cache.has(key)) {
		res = resolver->cache[key];
	} else {
		_resolve_hostname(res, p_hostname, p_type);
		resolver->cache[key] = res;
	}
	resolver->mutex->unlock();

	for (int i = 0; i < res.size(); ++i) {
		if (res[i].is_valid()) {
			return res[i];
		}
	}
	return IP_Address();
}

Array IP::resolve_hostname_addresses(const String &p_hostname, Type p_type) {

	resolver->mutex->lock();

	String key = _IP_ResolverPrivate::get_cache_key(p_hostname, p_type);
	if (!resolver->cache.has(key)) {
		_resolve_hostname(resolver->cache[key], p_hostname, p_type);
	}

	List<IP_Address> res = resolver->cache[key];
	resolver->mutex->unlock();

	Array result;
	for (int i = 0; i < res.size(); ++i) {
		if (res[i].is_valid()) {
			result.push_back(String(res[i]));
		}
	}
	return result;
}

IP::ResolverID IP::resolve_hostname_queue_item(const String &p_hostname, IP::Type p_type) {

	resolver->mutex->lock();

	ResolverID id = resolver->find_empty_id();

	if (id == RESOLVER_INVALID_ID) {
		WARN_PRINT("Out of resolver queries");
		resolver->mutex->unlock();
		return id;
	}

	String key = _IP_ResolverPrivate::get_cache_key(p_hostname, p_type);
	resolver->queue[id].hostname = p_hostname;
	resolver->queue[id].type = p_type;
	if (resolver->cache.has(key)) {
		resolver->queue[id].response = resolver->cache[key];
		resolver->queue[id].status = IP::RESOLVER_STATUS_DONE;
	} else {
		resolver->queue[id].response = List<IP_Address>();
		resolver->queue[id].status = IP::RESOLVER_STATUS_WAITING;
		if (resolver->thread)
			resolver->sem->post();
		else
			resolver->resolve_queues();
	}

	resolver->mutex->unlock();
	return id;
}

IP::ResolverStatus IP::get_resolve_item_status(ResolverID p_id) const {

	ERR_FAIL_INDEX_V(p_id, IP::RESOLVER_MAX_QUERIES, IP::RESOLVER_STATUS_NONE);

	resolver->mutex->lock();
	if (resolver->queue[p_id].status == IP::RESOLVER_STATUS_NONE) {
		ERR_PRINT("Condition status == IP::RESOLVER_STATUS_NONE");
		resolver->mutex->unlock();
		return IP::RESOLVER_STATUS_NONE;
	}
	IP::ResolverStatus res = resolver->queue[p_id].status;

	resolver->mutex->unlock();
	return res;
}

IP_Address IP::get_resolve_item_address(ResolverID p_id) const {

	ERR_FAIL_INDEX_V(p_id, IP::RESOLVER_MAX_QUERIES, IP_Address());

	resolver->mutex->lock();

	if (resolver->queue[p_id].status != IP::RESOLVER_STATUS_DONE) {
		ERR_PRINTS("Resolve of '" + resolver->queue[p_id].hostname + "'' didn't complete yet.");
		resolver->mutex->unlock();
		return IP_Address();
	}

	List<IP_Address> res = resolver->queue[p_id].response;

	resolver->mutex->unlock();

	for (int i = 0; i < res.size(); ++i) {
		if (res[i].is_valid()) {
			return res[i];
		}
	}
	return IP_Address();
}

Array IP::get_resolve_item_addresses(ResolverID p_id) const {

	ERR_FAIL_INDEX_V(p_id, IP::RESOLVER_MAX_QUERIES, Array());

	resolver->mutex->lock();

	if (resolver->queue[p_id].status != IP::RESOLVER_STATUS_DONE) {
		ERR_PRINTS("Resolve of '" + resolver->queue[p_id].hostname + "'' didn't complete yet.");
		resolver->mutex->unlock();
		return Array();
	}

	List<IP_Address> res = resolver->queue[p_id].response;

	resolver->mutex->unlock();

	Array result;
	for (int i = 0; i < res.size(); ++i) {
		if (res[i].is_valid()) {
			result.push_back(String(res[i]));
		}
	}
	return result;
}

void IP::erase_resolve_item(ResolverID p_id) {

	ERR_FAIL_INDEX(p_id, IP::RESOLVER_MAX_QUERIES);

	resolver->mutex->lock();

	resolver->queue[p_id].status = IP::RESOLVER_STATUS_NONE;

	resolver->mutex->unlock();
}

void IP::clear_cache(const String &p_hostname) {

	resolver->mutex->lock();

	if (p_hostname.empty()) {
		resolver->cache.clear();
	} else {
		resolver->cache.erase(_IP_ResolverPrivate::get_cache_key(p_hostname, IP::TYPE_NONE));
		resolver->cache.erase(_IP_ResolverPrivate::get_cache_key(p_hostname, IP::TYPE_IPV4));
		resolver->cache.erase(_IP_ResolverPrivate::get_cache_key(p_hostname, IP::TYPE_IPV6));
		resolver->cache.erase(_IP_ResolverPrivate::get_cache_key(p_hostname, IP::TYPE_ANY));
	}

	resolver->mutex->unlock();
}

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

	ObjectTypeDB::bind_method(_MD("resolve_hostname", "host", "ip_type"), &IP::resolve_hostname, DEFVAL(IP::TYPE_ANY));
	ObjectTypeDB::bind_method(_MD("resolve_hostname_addresses", "host", "ip_type"), &IP::resolve_hostname_addresses);
	ObjectTypeDB::bind_method(_MD("resolve_hostname_queue_item", "host", "ip_type"), &IP::resolve_hostname_queue_item, DEFVAL(IP::TYPE_ANY));
	ObjectTypeDB::bind_method(_MD("get_resolve_item_status", "id"), &IP::get_resolve_item_status);
	ObjectTypeDB::bind_method(_MD("get_resolve_item_address", "id"), &IP::get_resolve_item_address);
	ObjectTypeDB::bind_method(_MD("get_resolve_item_addresses", "id"), &IP::get_resolve_item_addresses);
	ObjectTypeDB::bind_method(_MD("erase_resolve_item", "id"), &IP::erase_resolve_item);
	ObjectTypeDB::bind_method(_MD("get_local_addresses"), &IP::_get_local_addresses);
	ObjectTypeDB::bind_method(_MD("clear_cache"), &IP::clear_cache, DEFVAL(""));

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
	resolver->mutex = Mutex::create();

#ifndef NO_THREADS

	resolver->sem = Semaphore::create();
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

#endif

	memdelete(resolver->mutex);
	memdelete(resolver);
}
