/*************************************************************************/
/*  message_queue.cpp                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2016 Juan Linietsky, Ariel Manzur.                 */
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
#include "message_queue.h"
#include "globals.h"

MessageQueue *MessageQueue::singleton=NULL;

MessageQueue *MessageQueue::get_singleton() {

	return singleton;
}

Error MessageQueue::push_call(ObjectID p_id, const StringName& p_method, VARIANT_ARG_DECLARE) {

	_THREAD_SAFE_METHOD_

	uint8_t room_needed=sizeof(Message);
	int args=0;
	if (p_arg5.get_type()!=Variant::NIL)
		args=5;
	else if (p_arg4.get_type()!=Variant::NIL)
		args=4;
	else if (p_arg3.get_type()!=Variant::NIL)
		args=3;
	else if (p_arg2.get_type()!=Variant::NIL)
		args=2;
	else if (p_arg1.get_type()!=Variant::NIL)
		args=1;
	else
		args=0;

	room_needed+=sizeof(Variant)*args;

	if ((buffer_end+room_needed) >= buffer_size) {
		String type;
		if (ObjectDB::get_instance(p_id))
			type=ObjectDB::get_instance(p_id)->get_type();
		print_line("failed method: "+type+":"+p_method+" target ID: "+itos(p_id));
		statistics();

	}
	ERR_FAIL_COND_V( (buffer_end+room_needed) >= buffer_size , ERR_OUT_OF_MEMORY );
	Message * msg = memnew_placement( &buffer[ buffer_end ], Message );
	msg->args=args;
	msg->instance_ID=p_id;
	msg->target=p_method;
	msg->type=TYPE_CALL;
	buffer_end+=sizeof(Message);


	if (args>=1) {

		Variant * v = memnew_placement( &buffer[ buffer_end ], Variant );
		buffer_end+=sizeof(Variant);
		*v=p_arg1;
	}

	if (args>=2) {

		Variant * v = memnew_placement( &buffer[ buffer_end ], Variant );
		buffer_end+=sizeof(Variant);
		*v=p_arg2;
	}

	if (args>=3) {

		Variant * v = memnew_placement( &buffer[ buffer_end ], Variant );
		buffer_end+=sizeof(Variant);
		*v=p_arg3;

	}

	if (args>=4) {

		Variant * v = memnew_placement( &buffer[ buffer_end ], Variant );
		buffer_end+=sizeof(Variant);
		*v=p_arg4;
	}

	if (args>=5) {

		Variant * v = memnew_placement( &buffer[ buffer_end ], Variant );
		buffer_end+=sizeof(Variant);
		*v=p_arg5;
	}


	return OK;
}

Error MessageQueue::push_set(ObjectID p_id, const StringName& p_prop, const Variant& p_value) {

	_THREAD_SAFE_METHOD_

	uint8_t room_needed=sizeof(Message)+sizeof(Variant);

	if ((buffer_end+room_needed) >= buffer_size) {
		String type;
		if (ObjectDB::get_instance(p_id))
			type=ObjectDB::get_instance(p_id)->get_type();
		print_line("failed set: "+type+":"+p_prop+" target ID: "+itos(p_id));
		statistics();

	}

	ERR_FAIL_COND_V( (buffer_end+room_needed) >= buffer_size , ERR_OUT_OF_MEMORY );

	Message * msg = memnew_placement( &buffer[ buffer_end ], Message );
	msg->args=1;
	msg->instance_ID=p_id;
	msg->target=p_prop;
	msg->type=TYPE_SET;

	buffer_end+=sizeof(Message);

	Variant * v = memnew_placement( &buffer[ buffer_end ], Variant );
	buffer_end+=sizeof(Variant);
	*v=p_value;


	return OK;
}

Error MessageQueue::push_notification(ObjectID p_id, int p_notification) {

	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND_V(p_notification<0, ERR_INVALID_PARAMETER );

	uint8_t room_needed=sizeof(Message);

	if ((buffer_end+room_needed) >= buffer_size) {
		String type;
		if (ObjectDB::get_instance(p_id))
			type=ObjectDB::get_instance(p_id)->get_type();
		print_line("failed notification: "+itos(p_notification)+" target ID: "+itos(p_id));
		statistics();

	}




	ERR_FAIL_COND_V( (buffer_end+room_needed) >= buffer_size , ERR_OUT_OF_MEMORY );
	Message * msg = memnew_placement( &buffer[ buffer_end ], Message );

	msg->type=TYPE_NOTIFICATION;
	msg->instance_ID=p_id;
	//msg->target;
	msg->notification=p_notification;

	buffer_end+=sizeof(Message);


	return OK;
}

Error MessageQueue::push_call(Object *p_object, const StringName& p_method, VARIANT_ARG_DECLARE) {

	return push_call(p_object->get_instance_ID(),p_method,VARIANT_ARG_PASS);
}

Error MessageQueue::push_notification(Object *p_object, int p_notification) {

	return push_notification(p_object->get_instance_ID(),p_notification);
}
Error MessageQueue::push_set(Object *p_object, const StringName& p_prop, const Variant& p_value) {

	return push_set(p_object->get_instance_ID(),p_prop,p_value);
}


void MessageQueue::statistics() {

	Map<StringName,int> set_count;
	Map<int,int> notify_count;
	Map<StringName,int> call_count;
	int null_count=0;

	uint32_t read_pos=0;
	while (read_pos < buffer_end ) {
		Message *message = (Message*)&buffer[ read_pos ];

		Object *target = ObjectDB::get_instance(message->instance_ID);

		if (target!=NULL) {


			switch(message->type) {

				case TYPE_CALL: {

					if (!call_count.has(message->target))
						call_count[message->target]=0;

					call_count[message->target]++;

				} break;
				case TYPE_NOTIFICATION: {

					if (!notify_count.has(message->notification))
						notify_count[message->notification]=0;

					notify_count[message->notification]++;

				} break;
				case TYPE_SET: {

					if (!set_count.has(message->target))
						set_count[message->target]=0;

					set_count[message->target]++;

				} break;

			}

			//object was deleted
			//WARN_PRINT("Object was deleted while awaiting a callback")
			//should it print a warning?
		} else {

			null_count++;
		}


		read_pos+=sizeof(Message);
		if (message->type!=TYPE_NOTIFICATION)
			read_pos+=sizeof(Variant)*message->args;
	}


	print_line("TOTAL BYTES: "+itos(buffer_end));
	print_line("NULL count: "+itos(null_count));

	for(Map<StringName,int>::Element *E=set_count.front();E;E=E->next()) {

		print_line("SET "+E->key()+": "+itos(E->get()));
	}

	for(Map<StringName,int>::Element *E=call_count.front();E;E=E->next()) {

		print_line("CALL "+E->key()+": "+itos(E->get()));
	}

	for(Map<int,int>::Element *E=notify_count.front();E;E=E->next()) {

		print_line("NOTIFY "+itos(E->key())+": "+itos(E->get()));
	}

}

bool MessageQueue::print() {
#if 0
	uint32_t read_pos=0;
	while (read_pos < buffer_end ) {
		Message *message = (Message*)&buffer[ read_pos ];

		Object *target = ObjectDB::get_instance(message->instance_ID);
		String cname;
		String cfunc;

		if (target==NULL) {
			//object was deleted
			//WARN_PRINT("Object was deleted while awaiting a callback")
			//should it print a warning?
		} else if (message->notification>=0) {

			// messages don't expect a return value
			cfunc="notification # "+itos(message->notification);
			cname=target->get_type();

		} else if (!message->target.empty()) {

			cfunc="property:  "+message->target;
			cname=target->get_type();


		} else if (message->target) {

			cfunc=String(message->target)+"()";
			cname=target->get_type();
		}


		read_pos+=sizeof(Message);
		if (message->type!=TYPE_NOTIFICATION)
			read_pos+=sizeof(Variant)*message->args;
	}
#endif
	return false;
}

int MessageQueue::get_max_buffer_usage() const {

	return buffer_max_used;
}

void MessageQueue::flush() {


	if (buffer_max_used<buffer_end); {
		buffer_max_used=buffer_end;
		//statistics();
	}

	uint32_t read_pos=0;

	//using reverse locking strategy
	_THREAD_SAFE_LOCK_

	while (read_pos<buffer_end) {

		_THREAD_SAFE_UNLOCK_

		//lock on each interation, so a call can re-add itself to the message queue

		Message *message = (Message*)&buffer[ read_pos ];

		Object *target = ObjectDB::get_instance(message->instance_ID);

		if (target!=NULL) {

			switch(message->type) {
				case TYPE_CALL: {

					Variant *args= (Variant*)(message+1);

					// messages don't expect a return value


					target->call( message->target,
						(message->args>=1) ? args[0] : Variant(),
						(message->args>=2) ? args[1] : Variant(),
						(message->args>=3) ? args[2] : Variant(),
						(message->args>=4) ? args[3] : Variant(),
						(message->args>=5) ? args[4] : Variant() );

					for(int i=0;i<message->args;i++) {
						args[i].~Variant();
					}

				} break;
				case TYPE_NOTIFICATION: {

					// messages don't expect a return value
					target->notification(message->notification);

				} break;
				case TYPE_SET: {

					Variant *arg= (Variant*)(message+1);
					// messages don't expect a return value
					target->set(message->target,*arg);

					arg->~Variant();
				} break;
			}

		}

		uint32_t advance = sizeof(Message);
		if (message->type!=TYPE_NOTIFICATION)
			advance+=sizeof(Variant)*message->args;
		message->~Message();

		_THREAD_SAFE_LOCK_
		read_pos+=advance;

	}


	buffer_end=0; // reset buffer
	_THREAD_SAFE_UNLOCK_

}

MessageQueue::MessageQueue() {

	ERR_FAIL_COND(singleton!=NULL);
	singleton=this;

	buffer_end=0;
	buffer_max_used=0;
	buffer_size=GLOBAL_DEF( "core/message_queue_size_kb", DEFAULT_QUEUE_SIZE_KB );
	buffer_size*=1024;
	buffer = memnew_arr( uint8_t, buffer_size );
}


MessageQueue::~MessageQueue() {

	uint32_t read_pos=0;

	while (read_pos < buffer_end ) {

		Message *message = (Message*)&buffer[ read_pos ];
		Variant *args= (Variant*)(message+1);
		int argc = message->args;
		if (message->type!=TYPE_NOTIFICATION) {
			for (int i=0;i<argc;i++)
				args[i].~Variant();
		}
		message->~Message();

		read_pos+=sizeof(Message);
		if (message->type!=TYPE_NOTIFICATION)
			read_pos+=sizeof(Variant)*message->args;
	}

	singleton=NULL;
	memdelete_arr( buffer );
}
