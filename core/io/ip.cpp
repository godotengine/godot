/*************************************************************************/
/*  ip.cpp                                                               */
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

#include "ip.h"

#include "core/hash_map.h"
#include "core/os/semaphore.h"
#include "core/os/thread.h"

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
		}
	};

	QueueItem queue[IP::RESOLVER_MAX_QUERIES];

	IP::ResolverID find_empty_id() const {
		for (int i = 0; i < IP::RESOLVER_MAX_QUERIES; i++) {
			if (queue[i].status == IP::RESOLVER_STATUS_NONE) {
				return i;
			}
		}
		return IP::RESOLVER_INVALID_ID;
	}

	Mutex mutex;
	Semaphore sem;

	Thread *thread;
	//Semaphore* semaphore;
	bool thread_abort;

	void resolve_queues() {
		for (int i = 0; i < IP::RESOLVER_MAX_QUERIES; i++) {
			if (queue[i].status != IP::RESOLVER_STATUS_WAITING) {
				continue;
			}
			queue[i].response = IP::get_singleton()->resolve_hostname(queue[i].hostname, queue[i].type);

			if (!queue[i].response.is_valid()) {
				queue[i].status = IP::RESOLVER_STATUS_ERROR;
			} else {
				queue[i].status = IP::RESOLVER_STATUS_DONE;
			}
		}
	}

	static void _thread_function(void *self) {
		_IP_ResolverPrivate *ipr = (_IP_ResolverPrivate *)self;

		while (!ipr->thread_abort) {
			ipr->sem.wait();

			MutexLock lock(ipr->mutex);
			ipr->resolve_queues();
		}
	}

	HashMap<String, IP_Address> cache;

	static String get_cache_key(String p_hostname, IP::Type p_type) {
		return itos(p_type) + p_hostname;
	}
};

IP_Address IP::resolve_hostname(const String &p_hostname, IP::Type p_type) {
	MutexLock lock(resolver->mutex);

	String key = _IP_ResolverPrivate::get_cache_key(p_hostname, p_type);
	if (resolver->cache.has(key) && resolver->cache[key].is_valid()) {
		IP_Address res = resolver->cache[key];
		return res;
	}

	IP_Address res = _resolve_hostname(p_hostname, p_type);
	resolver->cache[key] = res;
	return res;
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
	if (resolver->cache.has(key) && resolver->cache[key].is_valid()) {
		resolver->queue[id].response = resolver->cache[key];
		resolver->queue[id].status = IP::RESOLVER_STATUS_DONE;
	} else {
		resolver->queue[id].response = IP_Address();
		resolver->queue[id].status = IP::RESOLVER_STATUS_WAITING;
		if (resolver->thread) {
			resolver->sem.post();
		} else {
			resolver->resolve_queues();
		}
	}

	return id;
}

IP::ResolverStatus IP::get_resolve_item_status(ResolverID p_id) const {
	ERR_FAIL_INDEX_V(p_id, IP::RESOLVER_MAX_QUERIES, IP::RESOLVER_STATUS_NONE);

	MutexLock lock(resolver->mutex);

	if (resolver->queue[p_id].status == IP::RESOLVER_STATUS_NONE) {
		ERR_PRINT("Condition status == IP::RESOLVER_STATUS_NONE");
		resolver->mutex.unlock();
		return IP::RESOLVER_STATUS_NONE;
	}
	return resolver->queue[p_id].status;
}

IP_Address IP::get_resolve_item_address(ResolverID p_id) const {
	ERR_FAIL_INDEX_V(p_id, IP::RESOLVER_MAX_QUERIES, IP_Address());

	MutexLock lock(resolver->mutex);

	if (resolver->queue[p_id].status != IP::RESOLVER_STATUS_DONE) {
		ERR_PRINT("Resolve of '" + resolver->queue[p_id].hostname + "'' didn't complete yet.");
		resolver->mutex.unlock();
		return IP_Address();
	}

	return resolver->queue[p_id].response;
}

void IP::erase_resolve_item(ResolverID p_id) {
	ERR_FAIL_INDEX(p_id, IP::RESOLVER_MAX_QUERIES);

	MutexLock lock(resolver->mutex);

	resolver->queue[p_id].status = IP::RESOLVER_STATUS_NONE;
}

void IP::clear_cache(const String &p_hostname) {
	MutexLock lock(resolver->mutex);

	if (p_hostname.empty()) {
		resolver->cache.clear();
	} else {
		resolver->cache.erase(_IP_ResolverPrivate::get_cache_key(p_hostname, IP::TYPE_NONE));
		resolver->cache.erase(_IP_ResolverPrivate::get_cache_key(p_hostname, IP::TYPE_IPV4));
		resolver->cache.erase(_IP_ResolverPrivate::get_cache_key(p_hostname, IP::TYPE_IPV6));
		resolver->cache.erase(_IP_ResolverPrivate::get_cache_key(p_hostname, IP::TYPE_ANY));
	}
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

Array IP::_get_local_interfaces() const {
	Array results;
	Map<String, Interface_Info> interfaces;
	get_local_interfaces(&interfaces);
	for (Map<String, Interface_Info>::Element *E = interfaces.front(); E; E = E->next()) {
		Interface_Info &c = E->get();
		Dictionary rc;
		rc["name"] = c.name;
		rc["friendly"] = c.name_friendly;
		rc["index"] = c.index;

		Array ips;
		for (const List<IP_Address>::Element *F = c.ip_addresses.front(); F; F = F->next()) {
			ips.push_front(F->get());
		}
		rc["addresses"] = ips;

		results.push_front(rc);
	}

	return results;
}

void IP::get_local_addresses(List<IP_Address> *r_addresses) const {
	Map<String, Interface_Info> interfaces;
	get_local_interfaces(&interfaces);
	for (Map<String, Interface_Info>::Element *E = interfaces.front(); E; E = E->next()) {
		for (const List<IP_Address>::Element *F = E->get().ip_addresses.front(); F; F = F->next()) {
			r_addresses->push_front(F->get());
		}
	}
}

void IP::_bind_methods() {
	ClassDB::bind_method(D_METHOD("resolve_hostname", "host", "ip_type"), &IP::resolve_hostname, DEFVAL(IP::TYPE_ANY));
	ClassDB::bind_method(D_METHOD("resolve_hostname_queue_item", "host", "ip_type"), &IP::resolve_hostname_queue_item, DEFVAL(IP::TYPE_ANY));
	ClassDB::bind_method(D_METHOD("get_resolve_item_status", "id"), &IP::get_resolve_item_status);
	ClassDB::bind_method(D_METHOD("get_resolve_item_address", "id"), &IP::get_resolve_item_address);
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

IP *IP::singleton = nullptr;

IP *IP::get_singleton() {
	return singleton;
}

IP *(*IP::_create)() = nullptr;

IP *IP::create() {
	ERR_FAIL_COND_V_MSG(singleton, nullptr, "IP singleton already exist.");
	ERR_FAIL_COND_V(!_create, nullptr);
	return _create();
}

IP::IP() {
	singleton = this;
	resolver = memnew(_IP_ResolverPrivate);

#ifndef NO_THREADS

	resolver->thread_abort = false;

	resolver->thread = Thread::create(_IP_ResolverPrivate::_thread_function, resolver);
#else
	resolver->thread = nullptr;
#endif
}

IP::~IP() {
#ifndef NO_THREADS
	if (resolver->thread) {
		resolver->thread_abort = true;
		resolver->sem.post();
		Thread::wait_to_finish(resolver->thread);
		memdelete(resolver->thread);
	}

#endif

	memdelete(resolver);
}
