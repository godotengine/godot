/*************************************************************************/
/*  ip.cpp                                                               */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                 */
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
#include "os/thread.h"
#include "os/semaphore.h"
#include "hash_map.h"

VARIANT_ENUM_CAST(IP::ResolverStatus);

/************* RESOLVER ******************/


struct _IP_ResolverPrivate {

	struct QueueItem {

		volatile IP::ResolverStatus status;
		IP_Address response;
		String hostname;

		void clear() {
			status = IP::RESOLVER_STATUS_NONE;
			response = IP_Address();
			hostname="";
		};
		
		QueueItem() {
			clear();
		};
	};

	QueueItem queue[IP::RESOLVER_MAX_QUERIES];

	IP::ResolverID find_empty_id() const {

		for(int i=0;i<IP::RESOLVER_MAX_QUERIES;i++) {
			if (queue[i].status==IP::RESOLVER_STATUS_NONE)
				return i;
		}
		return IP::RESOLVER_INVALID_ID;
	}

	Semaphore *sem;

	Thread* thread;
	//Semaphore* semaphore;
	bool thread_abort;

	void resolve_queues() {

		for(int i=0;i<IP::RESOLVER_MAX_QUERIES;i++) {

			if (queue[i].status!=IP::RESOLVER_STATUS_WAITING)
				continue;
			queue[i].response=IP::get_singleton()->resolve_hostname(queue[i].hostname);

			if (queue[i].response.host==0)
				queue[i].status=IP::RESOLVER_STATUS_ERROR;
			else
				queue[i].status=IP::RESOLVER_STATUS_DONE;

		}
	}


	static void _thread_function(void *self) {

		_IP_ResolverPrivate *ipr=(_IP_ResolverPrivate*)self;

		while(!ipr->thread_abort) {

			ipr->sem->wait();
			GLOBAL_LOCK_FUNCTION;
			ipr->resolve_queues();

		}

	}

	HashMap<String, IP_Address> cache;

};



IP_Address IP::resolve_hostname(const String& p_hostname) {

	GLOBAL_LOCK_FUNCTION

	if (resolver->cache.has(p_hostname))
		return resolver->cache[p_hostname];

	IP_Address res = _resolve_hostname(p_hostname);
	resolver->cache[p_hostname]=res;
	return res;

}
IP::ResolverID IP::resolve_hostname_queue_item(const String& p_hostname) {

	GLOBAL_LOCK_FUNCTION

	ResolverID id = resolver->find_empty_id();

	if (id==RESOLVER_INVALID_ID) {
		WARN_PRINT("Out of resolver queries");
		return id;
	}

	resolver->queue[id].hostname=p_hostname;
	if (resolver->cache.has(p_hostname)) {
		resolver->queue[id].response=resolver->cache[p_hostname];
		resolver->queue[id].status=IP::RESOLVER_STATUS_DONE;
	} else {
		resolver->queue[id].response=IP_Address();
		resolver->queue[id].status=IP::RESOLVER_STATUS_WAITING;
		if (resolver->thread)
			resolver->sem->post();
		else
			resolver->resolve_queues();
	}





	return id;
}

IP::ResolverStatus IP::get_resolve_item_status(ResolverID p_id) const {

	ERR_FAIL_INDEX_V(p_id,IP::RESOLVER_MAX_QUERIES,IP::RESOLVER_STATUS_NONE);

	GLOBAL_LOCK_FUNCTION;
	ERR_FAIL_COND_V(resolver->queue[p_id].status==IP::RESOLVER_STATUS_NONE,IP::RESOLVER_STATUS_NONE);

	return resolver->queue[p_id].status;

}
IP_Address IP::get_resolve_item_address(ResolverID p_id) const {

	ERR_FAIL_INDEX_V(p_id,IP::RESOLVER_MAX_QUERIES,IP_Address());

	GLOBAL_LOCK_FUNCTION;

	if (resolver->queue[p_id].status!=IP::RESOLVER_STATUS_DONE) {
		ERR_EXPLAIN("Resolve of '"+resolver->queue[p_id].hostname+"'' didn't complete yet.");
		ERR_FAIL_COND_V(resolver->queue[p_id].status!=IP::RESOLVER_STATUS_DONE,IP_Address());
	}


	return resolver->queue[p_id].response;

}
void IP::erase_resolve_item(ResolverID p_id) {

	ERR_FAIL_INDEX(p_id,IP::RESOLVER_MAX_QUERIES);

	GLOBAL_LOCK_FUNCTION;

	resolver->queue[p_id].status=IP::RESOLVER_STATUS_DONE;

}


Array IP::_get_local_addresses() const {

	Array addresses;
	List<IP_Address> ip_addresses;
	get_local_addresses(&ip_addresses);
	for(List<IP_Address>::Element *E=ip_addresses.front();E;E=E->next()) {
		addresses.push_back(E->get());
	}

	return addresses;
}

void IP::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("resolve_hostname","host"),&IP::resolve_hostname);
	ObjectTypeDB::bind_method(_MD("resolve_hostname_queue_item","host"),&IP::resolve_hostname_queue_item);
	ObjectTypeDB::bind_method(_MD("get_resolve_item_status","id"),&IP::get_resolve_item_status);
	ObjectTypeDB::bind_method(_MD("get_resolve_item_address","id"),&IP::get_resolve_item_address);
	ObjectTypeDB::bind_method(_MD("erase_resolve_item","id"),&IP::erase_resolve_item);
	ObjectTypeDB::bind_method(_MD("get_local_addresses"),&IP::_get_local_addresses);

	BIND_CONSTANT( RESOLVER_STATUS_NONE );
	BIND_CONSTANT( RESOLVER_STATUS_WAITING );
	BIND_CONSTANT( RESOLVER_STATUS_DONE );
	BIND_CONSTANT( RESOLVER_STATUS_ERROR );

	BIND_CONSTANT( RESOLVER_MAX_QUERIES );
	BIND_CONSTANT( RESOLVER_INVALID_ID );

}


IP*IP::singleton=NULL;

IP* IP::get_singleton() {

	return singleton;
}


IP* (*IP::_create)()=NULL;

IP* IP::create() {

	ERR_FAIL_COND_V(singleton,NULL);
	ERR_FAIL_COND_V(!_create,NULL);
	return _create();
}

IP::IP() {

	singleton=this;
	resolver = memnew( _IP_ResolverPrivate );
	resolver->sem=NULL;

#ifndef NO_THREADS

	//resolver->sem = Semaphore::create();

	resolver->sem=NULL;
	if (resolver->sem) {
		resolver->thread_abort=false;

		resolver->thread = Thread::create( _IP_ResolverPrivate::_thread_function,resolver );

		if (!resolver->thread)
			memdelete(resolver->sem); //wtf
	} else {
		resolver->thread=NULL;
	}
#else
	resolver->sem = NULL;
	resolver->thread=NULL;
#endif


}

IP::~IP() {

#ifndef NO_THREADS
	if (resolver->thread) {
		resolver->thread_abort=true;
		resolver->sem->post();
		Thread::wait_to_finish(resolver->thread);
		memdelete( resolver->thread );
		memdelete( resolver->sem);
	}
	memdelete(resolver);

#endif

}
