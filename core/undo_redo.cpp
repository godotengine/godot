/*************************************************************************/
/*  undo_redo.cpp                                                        */
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
#include "undo_redo.h"


void UndoRedo::_discard_redo() {

	if (current_action==actions.size()-1)
		return;

	for(int i=current_action+1;i<actions.size();i++) {

		for (List<Operation>::Element *E=actions[i].do_ops.front();E;E=E->next()) {

			if (E->get().type==Operation::TYPE_REFERENCE) {

				Object *obj = ObjectDB::get_instance(E->get().object);
				if (obj)
					memdelete(obj);
			}
		}
		//ERASE do data
	}

	actions.resize(current_action+1);

}


void UndoRedo::create_action(const String& p_name,bool p_mergeable) {

	if (action_level==0) {

		_discard_redo();
		if (p_mergeable && actions.size() && actions[actions.size()-1].name==p_name) {

			//old will replace new (it's mergeable after all)
			// should check references though!
			current_action=actions.size()-2;
			actions[current_action+1].do_ops.clear();
			//actions[current_action+1].undo_ops.clear(); - no, this is kept
			merging=true;

		} else {
			Action new_action;
			new_action.name=p_name;
			actions.push_back(new_action);
			merging=false;
		}
	}

	action_level++;
}

void UndoRedo::add_do_method(Object *p_object,const String& p_method,VARIANT_ARG_DECLARE) {

	VARIANT_ARGPTRS
	ERR_FAIL_COND(action_level<=0);
	ERR_FAIL_COND((current_action+1)>=actions.size());
	Operation do_op;
	do_op.object=p_object->get_instance_ID();
	if (p_object->cast_to<Resource>())
		do_op.resref=Ref<Resource>(p_object->cast_to<Resource>());

	do_op.type=Operation::TYPE_METHOD;
	do_op.name=p_method;

	for(int i=0;i<VARIANT_ARG_MAX;i++) {
		do_op.args[i]=*argptr[i];
	}
	actions[current_action+1].do_ops.push_back(do_op);
}

void UndoRedo::add_undo_method(Object *p_object,const String& p_method,VARIANT_ARG_DECLARE) {

	VARIANT_ARGPTRS
	ERR_FAIL_COND(action_level<=0);
	ERR_FAIL_COND((current_action+1)>=actions.size());
	if (merging)
		return; //- no undo if merging

	Operation undo_op;
	undo_op.object=p_object->get_instance_ID();
	if (p_object->cast_to<Resource>())
		undo_op.resref=Ref<Resource>(p_object->cast_to<Resource>());

	undo_op.type=Operation::TYPE_METHOD;
	undo_op.name=p_method;

	for(int i=0;i<VARIANT_ARG_MAX;i++) {
		undo_op.args[i]=*argptr[i];
	}
	actions[current_action+1].undo_ops.push_back(undo_op);

}
void UndoRedo::add_do_property(Object *p_object,const String& p_property,const Variant& p_value) {

	ERR_FAIL_COND(action_level<=0);
	ERR_FAIL_COND((current_action+1)>=actions.size());
	Operation do_op;
	do_op.object=p_object->get_instance_ID();
	if (p_object->cast_to<Resource>())
		do_op.resref=Ref<Resource>(p_object->cast_to<Resource>());

	do_op.type=Operation::TYPE_PROPERTY;
	do_op.name=p_property;
	do_op.args[0]=p_value;
	actions[current_action+1].do_ops.push_back(do_op);

}
void UndoRedo::add_undo_property(Object *p_object,const String& p_property,const Variant& p_value) {

	ERR_FAIL_COND(action_level<=0);
	ERR_FAIL_COND((current_action+1)>=actions.size());

	Operation undo_op;
	undo_op.object=p_object->get_instance_ID();
	if (p_object->cast_to<Resource>())
		undo_op.resref=Ref<Resource>(p_object->cast_to<Resource>());

	undo_op.type=Operation::TYPE_PROPERTY;
	undo_op.name=p_property;
	undo_op.args[0]=p_value;
	actions[current_action+1].undo_ops.push_back(undo_op);

}
void UndoRedo::add_do_reference(Object *p_object) {

	ERR_FAIL_COND(action_level<=0);
	ERR_FAIL_COND((current_action+1)>=actions.size());
	Operation do_op;
	do_op.object=p_object->get_instance_ID();
	if (p_object->cast_to<Resource>())
		do_op.resref=Ref<Resource>(p_object->cast_to<Resource>());

	do_op.type=Operation::TYPE_REFERENCE;
	actions[current_action+1].do_ops.push_back(do_op);

}
void UndoRedo::add_undo_reference(Object *p_object) {

	ERR_FAIL_COND(action_level<=0);
	ERR_FAIL_COND((current_action+1)>=actions.size());
	Operation undo_op;
	undo_op.object=p_object->get_instance_ID();
	if (p_object->cast_to<Resource>())
		undo_op.resref=Ref<Resource>(p_object->cast_to<Resource>());

	undo_op.type=Operation::TYPE_REFERENCE;
	actions[current_action+1].undo_ops.push_back(undo_op);

}

void UndoRedo::_pop_history_tail() {

	_discard_redo();

	if (!actions.size())
		return;

	for (List<Operation>::Element *E=actions[0].undo_ops.front();E;E=E->next()) {

		if (E->get().type==Operation::TYPE_REFERENCE) {

			Object *obj = ObjectDB::get_instance(E->get().object);
			if (obj)
				memdelete(obj);
		}
	}

	actions.remove(0);
	current_action--;
}

void UndoRedo::commit_action() {

	ERR_FAIL_COND(action_level<=0);
	action_level--;
	if (action_level>0)
		return; //still nested

	redo(); // perform action

	if (max_steps>0 && actions.size()>max_steps) {
		//clear early steps

		while(actions.size() > max_steps)
			_pop_history_tail();

	}

	if (callback && actions.size()>0) {
		callback(callback_ud,actions[actions.size()-1].name);
	}
}


void UndoRedo::_process_operation_list(List<Operation>::Element *E) {

	for (;E;E=E->next()) {

		Operation &op=E->get();

		Object *obj = ObjectDB::get_instance(op.object);
		if (!obj) {
			//corruption
			clear_history();
			ERR_FAIL_COND(!obj);

		}
		switch(op.type) {

			case Operation::TYPE_METHOD: {

				obj->call(op.name,VARIANT_ARGS_FROM_ARRAY(op.args));
			} break;
			case Operation::TYPE_PROPERTY: {

				obj->set(op.name,op.args[0]);
#ifdef TOOLS_ENABLED
				Resource* res = obj->cast_to<Resource>();
				if (res)
					res->set_edited(true);
#endif
			} break;
			case Operation::TYPE_REFERENCE: {
				//do nothing
			} break;

		}
	}
}

void UndoRedo::redo() {

	ERR_FAIL_COND(action_level>0);

	if ((current_action+1)>=actions.size())
		return; //nothing to redo
	current_action++;

	_process_operation_list(actions[current_action].do_ops.front());
	version++;
}

void UndoRedo::undo() {

	ERR_FAIL_COND(action_level>0);
	if (current_action<0)
		return; //nothing to redo
	_process_operation_list(actions[current_action].undo_ops.front());
	current_action--;
}

void UndoRedo::clear_history() {

	ERR_FAIL_COND(action_level>0);
	_discard_redo();

	while(actions.size())
		_pop_history_tail();

	version++;
}

String UndoRedo::get_current_action_name() const {

	ERR_FAIL_COND_V(action_level>0,"");
	if (current_action<0)
		return ""; //nothing to redo
	return actions[current_action].name;
}

void UndoRedo::set_max_steps(int p_max_steps) {

	max_steps=p_max_steps;
}

int UndoRedo::get_max_steps() const {

	return max_steps;
}

uint64_t UndoRedo::get_version() const {

	return version;
}

void UndoRedo::set_commit_notify_callback(CommitNotifyCallback p_callback,void* p_ud) {

	callback=p_callback;
	callback_ud=p_ud;
}

UndoRedo::UndoRedo() {

	version=0;
	action_level=0;
	current_action=-1;
	max_steps=-1;
	merging=true;
	callback=NULL;
	callback_ud=NULL;
}

UndoRedo::~UndoRedo() {

	clear_history();
}
