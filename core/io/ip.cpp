/**************************************************************************/
/*  ip.cpp                                                                */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "ip.h"

#include "core/os/semaphore.h"
#include "core/os/thread.h"
#include "core/templates/hash_map.h"
#include "core/variant/typed_array.h"

/************* RESOLVER ******************/

struct _IP_ResolverPrivate {
	struct QueueItem {
		SafeNumeric<IP::ResolverStatus> status;

		List<IPAddress> response;

		String hostname;
		IP::Type type;

		void clear() {
			status.set(IP::RESOLVER_STATUS_NONE);
			response.clear();
			type = IP::TYPE_NONE;
			hostname = "";
		}

		QueueItem() {
			clear();
		}
	};

	QueueItem queue[IP::RESOLVER_MAX_QUERIES];

	IP::ResolverID find_empty_id() const {
		for (int i = 0; i < IP::RESOLVER_MAX_QUERIES; i++) {
			if (queue[i].status.get() == IP::RESOLVER_STATUS_NONE) {
				return i;
			}
		}
		return IP::RESOLVER_INVALID_ID;
	}

	Mutex mutex;
	Semaphore sem;

	Thread thread;
	SafeFlag thread_abort;

	void resolve_queues() {
		for (int i = 0; i < IP::RESOLVER_MAX_QUERIES; i++) {
			if (queue[i].status.get() != IP::RESOLVER_STATUS_WAITING) {
				continue;
			}

			MutexLock lock(mutex);
			List<IPAddress> response;
			String hostname = queue[i].hostname;
			IP::Type type = queue[i].type;
			lock.temp_unlock();

			// We should not lock while resolving the hostname,
			// only when modifying the queue.
			IP::get_singleton()->_resolve_hostname(response, hostname, type);

			lock.temp_relock();
			// Could have been completed by another function, or deleted.
			if (queue[i].status.get() != IP::RESOLVER_STATUS_WAITING) {
				continue;
			}
			// We might be overriding another result, but we don't care as long as the result is valid.
			if (response.size()) {
				String key = get_cache_key(hostname, type);
				cache[key] = response;
			}
			queue[i].response = response;
			queue[i].status.set(response.is_empty() ? IP::RESOLVER_STATUS_ERROR : IP::RESOLVER_STATUS_DONE);
		}
	}

	static void _thread_function(void *self) {
		_IP_ResolverPrivate *ipr = static_cast<_IP_ResolverPrivate *>(self);

		while (!ipr->thread_abort.is_set()) {
			ipr->sem.wait();
			ipr->resolve_queues();
		}
	}

	HashMap<String, List<IPAddress>> cache;

	static String get_cache_key(const String &p_hostname, IP::Type p_type) {
		return itos(p_type) + p_hostname;
	}
};

IPAddress IP::resolve_hostname(const String &p_hostname, IP::Type p_type) {
	const PackedStringArray addresses = resolve_hostname_addresses(p_hostname, p_type);
	return addresses.size() ? (IPAddress)addresses[0] : IPAddress();
}

PackedStringArray IP::resolve_hostname_addresses(const String &p_hostname, Type p_type) {
	List<IPAddress> res;
	String key = _IP_ResolverPrivate::get_cache_key(p_hostname, p_type);

	{
		MutexLock lock(resolver->mutex);
		if (resolver->cache.has(key)) {
			res = resolver->cache[key];
		} else {
			// This should be run unlocked so the resolver thread can keep resolving
			// other requests.
			lock.temp_unlock();
			_resolve_hostname(res, p_hostname, p_type);
			lock.temp_relock();
			// We might be overriding another result, but we don't care as long as the result is valid.
			if (res.size()) {
				resolver->cache[key] = res;
			}
		}
	}

	PackedStringArray result;
	for (const IPAddress &E : res) {
		result.push_back(String(E));
	}
	return result;
}

IP::ResolverID IP::resolve_hostname_queue_item(const String &p_hostname, IP::Type p_type) {
	MutexLock lock(resolver->mutex);

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
		resolver->queue[id].status.set(IP::RESOLVER_STATUS_DONE);
	} else {
		resolver->queue[id].response = List<IPAddress>();
		resolver->queue[id].status.set(IP::RESOLVER_STATUS_WAITING);
		if (resolver->thread.is_started()) {
			resolver->sem.post();
		} else {
			resolver->resolve_queues();
		}
	}

	return id;
}

IP::ResolverStatus IP::get_resolve_item_status(ResolverID p_id) const {
	ERR_FAIL_INDEX_V_MSG(p_id, IP::RESOLVER_MAX_QUERIES, IP::RESOLVER_STATUS_NONE, vformat("Too many concurrent DNS resolver queries (%d, but should be %d at most). Try performing less network requests at once.", p_id, IP::RESOLVER_MAX_QUERIES));

	IP::ResolverStatus res = resolver->queue[p_id].status.get();
	if (res == IP::RESOLVER_STATUS_NONE) {
		ERR_PRINT("Condition status == IP::RESOLVER_STATUS_NONE");
		return IP::RESOLVER_STATUS_NONE;
	}
	return res;
}

IPAddress IP::get_resolve_item_address(ResolverID p_id) const {
	ERR_FAIL_INDEX_V_MSG(p_id, IP::RESOLVER_MAX_QUERIES, IPAddress(), vformat("Too many concurrent DNS resolver queries (%d, but should be %d at most). Try performing less network requests at once.", p_id, IP::RESOLVER_MAX_QUERIES));

	MutexLock lock(resolver->mutex);

	if (resolver->queue[p_id].status.get() != IP::RESOLVER_STATUS_DONE) {
		ERR_PRINT(vformat("Resolve of '%s' didn't complete yet.", resolver->queue[p_id].hostname));
		return IPAddress();
	}

	const List<IPAddress> res = List<IPAddress>(resolver->queue[p_id].response);

	for (const IPAddress &E : res) {
		if (E.is_valid()) {
			return E;
		}
	}
	return IPAddress();
}

Array IP::get_resolve_item_addresses(ResolverID p_id) const {
	ERR_FAIL_INDEX_V_MSG(p_id, IP::RESOLVER_MAX_QUERIES, Array(), vformat("Too many concurrent DNS resolver queries (%d, but should be %d at most). Try performing less network requests at once.", p_id, IP::RESOLVER_MAX_QUERIES));
	MutexLock lock(resolver->mutex);

	if (resolver->queue[p_id].status.get() != IP::RESOLVER_STATUS_DONE) {
		ERR_PRINT(vformat("Resolve of '%s' didn't complete yet.", resolver->queue[p_id].hostname));
		return Array();
	}

	const List<IPAddress> res = List<IPAddress>(resolver->queue[p_id].response);

	Array result;
	for (const IPAddress &E : res) {
		if (E.is_valid()) {
			result.push_back(String(E));
		}
	}
	return result;
}

void IP::erase_resolve_item(ResolverID p_id) {
	ERR_FAIL_INDEX_MSG(p_id, IP::RESOLVER_MAX_QUERIES, vformat("Too many concurrent DNS resolver queries (%d, but should be %d at most). Try performing less network requests at once.", p_id, IP::RESOLVER_MAX_QUERIES));

	resolver->queue[p_id].status.set(IP::RESOLVER_STATUS_NONE);
}

void IP::clear_cache(const String &p_hostname) {
	MutexLock lock(resolver->mutex);

	if (p_hostname.is_empty()) {
		resolver->cache.clear();
	} else {
		resolver->cache.erase(_IP_ResolverPrivate::get_cache_key(p_hostname, IP::TYPE_NONE));
		resolver->cache.erase(_IP_ResolverPrivate::get_cache_key(p_hostname, IP::TYPE_IPV4));
		resolver->cache.erase(_IP_ResolverPrivate::get_cache_key(p_hostname, IP::TYPE_IPV6));
		resolver->cache.erase(_IP_ResolverPrivate::get_cache_key(p_hostname, IP::TYPE_ANY));
	}
}

PackedStringArray IP::_get_local_addresses() const {
	PackedStringArray addresses;
	List<IPAddress> ip_addresses;
	get_local_addresses(&ip_addresses);
	for (const IPAddress &E : ip_addresses) {
		addresses.push_back(String(E));
	}

	return addresses;
}

TypedArray<Dictionary> IP::_get_local_interfaces() const {
	TypedArray<Dictionary> results;
	HashMap<String, Interface_Info> interfaces;
	get_local_interfaces(&interfaces);
	for (KeyValue<String, Interface_Info> &E : interfaces) {
		Interface_Info &c = E.value;
		Dictionary rc;
		rc["name"] = c.name;
		rc["friendly"] = c.name_friendly;
		rc["index"] = c.index;

		Array ips;
		for (const IPAddress &F : c.ip_addresses) {
			ips.push_front(F);
		}
		rc["addresses"] = ips;

		results.push_front(rc);
	}

	return results;
}

void IP::get_local_addresses(List<IPAddress> *r_addresses) const {
	HashMap<String, Interface_Info> interfaces;
	get_local_interfaces(&interfaces);
	for (const KeyValue<String, Interface_Info> &E : interfaces) {
		for (const IPAddress &F : E.value.ip_addresses) {
			r_addresses->push_front(F);
		}
	}
}

void IP::_bind_methods() {
	ClassDB::bind_method(D_METHOD("resolve_hostname", "host", "ip_type"), &IP::resolve_hostname, DEFVAL(IP::TYPE_ANY));
	ClassDB::bind_method(D_METHOD("resolve_hostname_addresses", "host", "ip_type"), &IP::resolve_hostname_addresses, DEFVAL(IP::TYPE_ANY));
	ClassDB::bind_method(D_METHOD("resolve_hostname_queue_item", "host", "ip_type"), &IP::resolve_hostname_queue_item, DEFVAL(IP::TYPE_ANY));
	ClassDB::bind_method(D_METHOD("get_resolve_item_status", "id"), &IP::get_resolve_item_status);
	ClassDB::bind_method(D_METHOD("get_resolve_item_address", "id"), &IP::get_resolve_item_address);
	ClassDB::bind_method(D_METHOD("get_resolve_item_addresses", "id"), &IP::get_resolve_item_addresses);
	ClassDB::bind_method(D_METHOD("erase_resolve_item", "id"), &IP::erase_resolve_item);
	ClassDB::bind_method(D_METHOD("get_local_addresses"), &IP::_get_local_addresses);
	ClassDB::bind_method(D_METHOD("get_local_interfaces"), &IP::_get_local_interfaces);
	ClassDB::bind_method(D_METHOD("clear_cache", "hostname"), &IP::clear_cache, DEFVAL(""));

	BIND_ENUM_CONSTANT(RESOLVER_STATUS_NONE);
	BIND_ENUM_CONSTANT(RESOLVER_STATUS_WAITING);
	BIND_ENUM_CONSTANT(RESOLVER_STATUS_DONE);
	BIND_ENUM_CONSTANT(RESOLVER_STATUS_ERROR);

	BIND_CONSTANT(RESOLVER_MAX_QUERIES);
	BIND_CONSTANT(RESOLVER_INVALID_ID);

	BIND_ENUM_CONSTANT(TYPE_NONE);
	BIND_ENUM_CONSTANT(TYPE_IPV4);
	BIND_ENUM_CONSTANT(TYPE_IPV6);
	BIND_ENUM_CONSTANT(TYPE_ANY);
}

IP *IP::get_singleton() {
	return singleton;
}

IP *(*IP::_create)() = nullptr;

IP *IP::create() {
	ERR_FAIL_COND_V_MSG(singleton, nullptr, "IP singleton already exists.");
	ERR_FAIL_NULL_V(_create, nullptr);
	return _create();
}

IP::IP() {
	singleton = this;
	resolver = memnew(_IP_ResolverPrivate);

	resolver->thread_abort.clear();
	resolver->thread.start(_IP_ResolverPrivate::_thread_function, resolver);
}

IP::~IP() {
	resolver->thread_abort.set();
	resolver->sem.post();
	resolver->thread.wait_to_finish();

	memdelete(resolver);
}
