/*************************************************************************/
/*  state_machine.h                                                           */
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
#ifndef STATE_MACHINE_H
#define STATE_MACHINE_H


#include "scene/resources/mesh_library.h"
#include "scene/3d/spatial.h"
#include "scene/resources/multimesh.h"

//heh heh, godotsphir!! this shares no code and the design is completely different with previous projects i've done..
//should scale better with hardware that supports instancing


class StateMachine : public Node {


	OBJ_TYPE( StateMachine, Node);
	OBJ_CATEGORY("State Machines");

	bool awaiting_update;

	// void _queue_dirty_map();
	// void _update_dirty_map_callback();

	void resource_changed(const RES& p_res);


protected:

	bool _set(const StringName& p_name, const Variant& p_value);
	bool _get(const StringName& p_name,Variant &r_ret) const;
	void _get_property_list( List<PropertyInfo> *p_list) const;

	void _notification(int p_what);
	static void _bind_methods();

public:

	
	StateMachine();
	~StateMachine();


	enum NodeType {

		NODE_OUTPUT,
		NODE_ANIMATION,
		NODE_ONESHOT,
		NODE_MIX,
		NODE_BLEND2,
		NODE_BLEND3,
		NODE_BLEND4,
		NODE_TIMESCALE,
		NODE_TIMESEEK,
		NODE_TRANSITION,

		NODE_MAX,
	};

	enum ConnectError {

		CONNECT_OK,
		CONNECT_INCOMPLETE,
		CONNECT_CYCLE
	};

private:

	enum {

		DISCONNECTED = -1,
	};
	struct Input {

		StringName node;
		//Input() { node=-1;  }
	};

	struct NodeBase {

		bool cycletest;

		NodeType type;
		Point2 pos;


		Vector<Input> inputs;

		NodeBase() { cycletest = false; };
		virtual ~NodeBase() { cycletest = false; }
	};

	struct NodeOut : public NodeBase {

		NodeOut() { type = NODE_OUTPUT; inputs.resize(1); }
	};
};

#endif // CUBE_STATE_MACHINE_H
