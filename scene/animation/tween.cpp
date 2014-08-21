/*************************************************************************/
/*  tween.cpp                                                            */
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
#include "tween.h"

bool Tween::_set(const StringName& p_name, const Variant& p_value) {

	String name=p_name;

	if (name=="playback/speed" || name=="speed") { //bw compatibility
		set_speed(p_value);

	} else if (name=="playback/active") {
		set_active(p_value);

	} else if (name=="playback/repeat") {
		set_repeat(p_value);

	}
	return true;
}

bool Tween::_get(const StringName& p_name,Variant &r_ret) const {

	String name=p_name;

	if (name=="playback/speed") { //bw compatibility
	
		r_ret=speed_scale;
	} else if (name=="playback/active") {

		r_ret=is_active();
	} else if(name=="playback/repeat") {

		r_ret=is_repeat();
	}

	return true;
}

void Tween::_get_property_list(List<PropertyInfo> *p_list) const {

	p_list->push_back( PropertyInfo( Variant::BOOL, "playback/active", PROPERTY_HINT_NONE,"" ) );
	p_list->push_back( PropertyInfo( Variant::BOOL, "playback/repeat", PROPERTY_HINT_NONE,"" ) );
	p_list->push_back( PropertyInfo( Variant::REAL, "playback/speed", PROPERTY_HINT_RANGE, "-64,64,0.01") );
}

void Tween::_notification(int p_what) {

	switch(p_what) {
	
		case NOTIFICATION_ENTER_SCENE: {

			if (!processing) {
				//make sure that a previous process state was not saved
				//only process if "processing" is set
				set_fixed_process(false);
				set_process(false);
			}
		} break;
		case NOTIFICATION_READY: {

		} break;
		case NOTIFICATION_PROCESS: {
			if (tween_process_mode==TWEEN_PROCESS_FIXED)
				break;

			if (processing)
				_tween_process( get_process_delta_time() );
		} break;
		case NOTIFICATION_FIXED_PROCESS: {
		
			if (tween_process_mode==TWEEN_PROCESS_IDLE)
				break;

			if (processing)
				_tween_process( get_fixed_process_delta_time() );
		} break;
		case NOTIFICATION_EXIT_SCENE: {
		
			stop_all();
		} break;
	}
}

void Tween::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("is_active"),&Tween::is_active );
	ObjectTypeDB::bind_method(_MD("set_active","active"),&Tween::set_active );

	ObjectTypeDB::bind_method(_MD("is_repeat"),&Tween::is_repeat );
	ObjectTypeDB::bind_method(_MD("set_repeat","repeat"),&Tween::set_repeat );

	ObjectTypeDB::bind_method(_MD("set_speed","speed"),&Tween::set_speed);
	ObjectTypeDB::bind_method(_MD("get_speed"),&Tween::get_speed);

	ObjectTypeDB::bind_method(_MD("set_tween_process_mode","mode"),&Tween::set_tween_process_mode);
	ObjectTypeDB::bind_method(_MD("get_tween_process_mode"),&Tween::get_tween_process_mode);

	ObjectTypeDB::bind_method(_MD("start"),&Tween::start );
	ObjectTypeDB::bind_method(_MD("reset","object","key"),&Tween::reset );
	ObjectTypeDB::bind_method(_MD("reset_all"),&Tween::reset_all );
	ObjectTypeDB::bind_method(_MD("stop","object","key"),&Tween::stop );
	ObjectTypeDB::bind_method(_MD("stop_all"),&Tween::stop_all );
	ObjectTypeDB::bind_method(_MD("resume","object","key"),&Tween::resume );
	ObjectTypeDB::bind_method(_MD("resume_all"),&Tween::resume_all );
	ObjectTypeDB::bind_method(_MD("remove","object","key"),&Tween::remove );
	ObjectTypeDB::bind_method(_MD("remove_all"),&Tween::remove_all );
	ObjectTypeDB::bind_method(_MD("seek"),&Tween::seek );
	ObjectTypeDB::bind_method(_MD("get_runtime"),&Tween::get_runtime );

	ObjectTypeDB::bind_method(_MD("interpolate_property","object","property","initial_val","final_val","times_in_sec","trans_type","ease_type","delay"),&Tween::interpolate_property, DEFVAL(0) );
	ObjectTypeDB::bind_method(_MD("interpolate_method","object","method","initial_val","final_val","times_in_sec","trans_type","ease_type","delay"),&Tween::interpolate_method, DEFVAL(0) );

	ADD_SIGNAL( MethodInfo("tween_start", PropertyInfo( Variant::OBJECT,"object"), PropertyInfo( Variant::STRING,"key")) );
	ADD_SIGNAL( MethodInfo("tween_step", PropertyInfo( Variant::OBJECT,"object"), PropertyInfo( Variant::STRING,"key"), PropertyInfo( Variant::REAL,"elapsed"), PropertyInfo( Variant::OBJECT,"value")) );
	ADD_SIGNAL( MethodInfo("tween_complete", PropertyInfo( Variant::OBJECT,"object"), PropertyInfo( Variant::STRING,"key")) );

	ADD_PROPERTY( PropertyInfo( Variant::INT, "playback/process_mode", PROPERTY_HINT_ENUM, "Fixed,Idle"), _SCS("set_tween_process_mode"), _SCS("get_tween_process_mode"));
	//ADD_PROPERTY( PropertyInfo( Variant::BOOL, "activate"), _SCS("set_active"), _SCS("is_active"));

	BIND_CONSTANT(TRANS_LINEAR);
	BIND_CONSTANT(TRANS_SINE);
	BIND_CONSTANT(TRANS_QUINT);
	BIND_CONSTANT(TRANS_QUART);
	BIND_CONSTANT(TRANS_QUAD);
	BIND_CONSTANT(TRANS_EXPO);
	BIND_CONSTANT(TRANS_ELASTIC);
	BIND_CONSTANT(TRANS_CUBIC);
	BIND_CONSTANT(TRANS_CIRC);
	BIND_CONSTANT(TRANS_BOUNCE);
	BIND_CONSTANT(TRANS_BACK);

	BIND_CONSTANT(EASE_IN);
	BIND_CONSTANT(EASE_OUT);
	BIND_CONSTANT(EASE_IN_OUT);
	BIND_CONSTANT(EASE_OUT_IN);
}

Variant Tween::_run_equation(InterpolateData& p_data) {

	Variant& initial_val = p_data.initial_val;
	Variant& delta_val = p_data.delta_val;
	Variant result;

#define APPLY_EQUATION(element)\
	r.element = _run_equation(p_data.trans_type, p_data.ease_type, p_data.elapsed - p_data.delay, i.element, d.element, p_data.times_in_sec);

	switch(initial_val.get_type())
	{
	case Variant::INT:
		result = (int) _run_equation(p_data.trans_type, p_data.ease_type, p_data.elapsed - p_data.delay, (int) initial_val, (int) delta_val, p_data.times_in_sec);
		break;

	case Variant::REAL:
		result = _run_equation(p_data.trans_type, p_data.ease_type, p_data.elapsed - p_data.delay, (real_t) initial_val, (real_t) delta_val, p_data.times_in_sec);
		break;

	case Variant::VECTOR2:
		{
			Vector2 i = initial_val;
			Vector2 d = delta_val;
			Vector2 r;

			APPLY_EQUATION(x);
			APPLY_EQUATION(y);

			result = r;
		}
		break;

	case Variant::VECTOR3:
		{
			Vector3 i = initial_val;
			Vector3 d = delta_val;
			Vector3 r;

			APPLY_EQUATION(x);
			APPLY_EQUATION(y);
			APPLY_EQUATION(z);

			result = r;
		}
		break;

	case Variant::MATRIX3:
		{
			Matrix3 i = initial_val;
			Matrix3 d = delta_val;
			Matrix3 r;

			APPLY_EQUATION(elements[0][0]);
			APPLY_EQUATION(elements[0][1]);
			APPLY_EQUATION(elements[0][2]);
			APPLY_EQUATION(elements[1][0]);
			APPLY_EQUATION(elements[1][1]);
			APPLY_EQUATION(elements[1][2]);
			APPLY_EQUATION(elements[2][0]);
			APPLY_EQUATION(elements[2][1]);
			APPLY_EQUATION(elements[2][2]);

			result = r;
		}
	break;

	case Variant::MATRIX32:
		{
			Matrix3 i = initial_val;
			Matrix3 d = delta_val;
			Matrix3 r;

			APPLY_EQUATION(elements[0][0]);
			APPLY_EQUATION(elements[0][1]);
			APPLY_EQUATION(elements[1][0]);
			APPLY_EQUATION(elements[1][1]);
			APPLY_EQUATION(elements[2][0]);
			APPLY_EQUATION(elements[2][1]);

			result = r;
		}
	break;
	case Variant::QUAT:
		{
			Quat i = initial_val;
			Quat d = delta_val;
			Quat r;

			APPLY_EQUATION(x);
			APPLY_EQUATION(y);
			APPLY_EQUATION(z);
			APPLY_EQUATION(w);

			result = r;
		}
		break;
	case Variant::_AABB:
		{
			AABB i = initial_val;
			AABB d = delta_val;
			AABB r;

			APPLY_EQUATION(pos.x);
			APPLY_EQUATION(pos.y);
			APPLY_EQUATION(pos.z);
			APPLY_EQUATION(size.x);
			APPLY_EQUATION(size.y);
			APPLY_EQUATION(size.z);

			result = r;
		}
		break;
	case Variant::TRANSFORM:
		{
			Transform i = initial_val;
			Transform d = delta_val;
			Transform r;

			APPLY_EQUATION(basis.elements[0][0]);
			APPLY_EQUATION(basis.elements[0][1]);
			APPLY_EQUATION(basis.elements[0][2]);
			APPLY_EQUATION(basis.elements[1][0]);
			APPLY_EQUATION(basis.elements[1][1]);
			APPLY_EQUATION(basis.elements[1][2]);
			APPLY_EQUATION(basis.elements[2][0]);
			APPLY_EQUATION(basis.elements[2][1]);
			APPLY_EQUATION(basis.elements[2][2]);
			APPLY_EQUATION(origin.x);
			APPLY_EQUATION(origin.y);
			APPLY_EQUATION(origin.z);

			result = r;
		}
		break;
	case Variant::COLOR:
		{
			Color i = initial_val;
			Color d = delta_val;
			Color r;

			APPLY_EQUATION(r);
			APPLY_EQUATION(g);
			APPLY_EQUATION(b);
			APPLY_EQUATION(a);

			result = r;
		}
		break;
	};
#undef APPLY_EQUATION

	return result;
}

bool Tween::_apply_tween_value(InterpolateData& p_data, Variant& value) {

	Variant& object = p_data.object;

	if(p_data.is_method) {

		Variant *arg[1] = { &value };

		Variant::CallError error;
		object.call(p_data.key, (const Variant **) arg, 1, error);
		if(error.error == Variant::CallError::CALL_OK)
			return true;

		return false;

	} else {

		bool valid = false;
		object.set(p_data.key,value, &valid);
		return valid;
	}
	return true;
}

void Tween::_tween_process(float p_delta) {

	if (speed_scale == 0)
		return;
	p_delta *= speed_scale;

	for(List<InterpolateData>::Element *E=interpolates.front();E;E=E->next()) {

		InterpolateData& data = E->get();
		if(!data.active)
			continue;

		if(data.finish) {
			if(!repeat)
				continue;
			data.elapsed = 0;
			data.finish = false;
		}

		bool prev_delaying = data.elapsed <= data.delay;
		data.elapsed += p_delta;
		if(data.elapsed < data.delay)
			continue;
		else if(prev_delaying)
			emit_signal("tween_start",data.object,data.key);

		if(data.elapsed > (data.delay + data.times_in_sec)) {

			data.elapsed = data.delay + data.times_in_sec;
			data.finish = true;
		}

		Variant result = _run_equation(data);
		emit_signal("tween_step",data.object,data.key,data.elapsed,result);

		_apply_tween_value(data, result);

		if(data.finish)
			emit_signal("tween_complete",data.object,data.key);
	}
}

void Tween::set_tween_process_mode(TweenProcessMode p_mode) {

	if (tween_process_mode==p_mode)
		return;

	bool pr = processing;
	if (pr)
		_set_process(false);
	tween_process_mode=p_mode;
	if (pr)
		_set_process(true);
}

Tween::TweenProcessMode Tween::get_tween_process_mode() const {

	return tween_process_mode;
}

void Tween::_set_process(bool p_process,bool p_force) {

	if (processing==p_process && !p_force)
		return;

	switch(tween_process_mode) {

		case TWEEN_PROCESS_FIXED: set_fixed_process(p_process && active); break;
		case TWEEN_PROCESS_IDLE: set_process(p_process && active); break;
	}

	processing=p_process;
}

bool Tween::is_active() const {

	return active;
}

void Tween::set_active(bool p_active) {

	if (active==p_active)
		return;

	active=p_active;
	_set_process(processing,true);
}

bool Tween::is_repeat() const {

	return repeat;
}

void Tween::set_repeat(bool p_repeat) {

	repeat = p_repeat;
}

void Tween::set_speed(float p_speed) {

	speed_scale=p_speed;
}

float Tween::get_speed() const {

	return speed_scale;
}

bool Tween::start() {

	set_active(true);
	_set_process(true);
	return true;
}

bool Tween::reset(Variant p_object, String p_key) {

	for(List<InterpolateData>::Element *E=interpolates.front();E;E=E->next()) {

		InterpolateData& data = E->get();
		if(data.object == p_object && data.key == p_key) {

			data.elapsed = 0;
			data.finish = false;
			if(data.delay == 0)
				_apply_tween_value(data, data.initial_val);
		}
	}
	return true;
}

bool Tween::reset_all() {

	for(List<InterpolateData>::Element *E=interpolates.front();E;E=E->next()) {

		InterpolateData& data = E->get();
		data.elapsed = 0;
		data.finish = false;
		if(data.delay == 0)
			_apply_tween_value(data, data.initial_val);
	}
	return true;
}

bool Tween::stop(Variant p_object, String p_key) {

	for(List<InterpolateData>::Element *E=interpolates.front();E;E=E->next()) {

		InterpolateData& data = E->get();
		if(data.object == p_object && data.key == p_key)
			data.active = false;
	}
	return true;
}

bool Tween::stop_all() {

	set_active(false);
	_set_process(false);

	for(List<InterpolateData>::Element *E=interpolates.front();E;E=E->next()) {

		InterpolateData& data = E->get();
		data.active = false;
	}
	return true;
}

bool Tween::resume(Variant p_object, String p_key) {

	set_active(true);
	_set_process(true);

	for(List<InterpolateData>::Element *E=interpolates.front();E;E=E->next()) {

		InterpolateData& data = E->get();
		if(data.object == p_object && data.key == p_key)
			data.active = true;
	}
	return true;
}

bool Tween::resume_all() {

	set_active(true);
	_set_process(true);

	for(List<InterpolateData>::Element *E=interpolates.front();E;E=E->next()) {

		InterpolateData& data = E->get();
		data.active = true;
	}
	return true;
}

bool Tween::remove(Variant p_object, String p_key) {

	for(List<InterpolateData>::Element *E=interpolates.front();E;E=E->next()) {

		InterpolateData& data = E->get();
		if(data.object == p_object && data.key == p_key) {

			interpolates.erase(E);
			return true;
		}
	}
	return true;
}

bool Tween::remove_all() {

	set_active(false);
	_set_process(false);
	interpolates.clear();
	return true;
}

bool Tween::seek(real_t p_time) {

	for(List<InterpolateData>::Element *E=interpolates.front();E;E=E->next()) {

		InterpolateData& data = E->get();

		data.elapsed = p_time;
		if(data.elapsed < data.delay)
			continue;
		else if(data.elapsed > (data.delay + data.times_in_sec))
			data.elapsed = (data.delay + data.times_in_sec);

		Variant result = _run_equation(data);

		_apply_tween_value(data, result);
	}
	return true;
}

real_t Tween::get_runtime() {

	real_t runtime = 0;
	for(List<InterpolateData>::Element *E=interpolates.front();E;E=E->next()) {

		InterpolateData& data = E->get();
		real_t t = data.delay + data.times_in_sec;
		if(t > runtime)
			runtime = t;
	}
	return runtime;
}

bool Tween::_calc_delta_val(InterpolateData& p_data) {

	Variant& initial_val = p_data.initial_val;
	Variant& delta_val = p_data.delta_val;
	Variant& final_val = p_data.final_val;

	switch(initial_val.get_type()) {
		case Variant::INT:
			delta_val = (int) final_val - (int) initial_val;
			break;

		case Variant::REAL:
			delta_val = (real_t) final_val - (real_t) initial_val;
			break;

		case Variant::VECTOR2:
			delta_val = final_val.operator Vector2() - initial_val.operator Vector2();
			break;

		case Variant::VECTOR3:
			delta_val = final_val.operator Vector3() - initial_val.operator Vector3();
			break;

		case Variant::MATRIX3:
			{
				Matrix3 i = initial_val;
				Matrix3 f = final_val;
				delta_val = Matrix3(f.elements[0][0] - i.elements[0][0],
					f.elements[0][1] - i.elements[0][1],
					f.elements[0][2] - i.elements[0][2],
					f.elements[1][0] - i.elements[1][0],
					f.elements[1][1] - i.elements[1][1],
					f.elements[1][2] - i.elements[1][2],
					f.elements[2][0] - i.elements[2][0],
					f.elements[2][1] - i.elements[2][1],
					f.elements[2][2] - i.elements[2][2]
				);
			}
			break;

		case Variant::MATRIX32:
			{
				Matrix32 i = initial_val;
				Matrix32 f = final_val;
				Matrix32 d = Matrix32();
				d[0][0] = f.elements[0][0] - i.elements[0][0];
				d[0][1] = f.elements[0][1] - i.elements[0][1];
				d[1][0] = f.elements[1][0] - i.elements[1][0];
				d[1][1] = f.elements[1][1] - i.elements[1][1];
				d[2][0] = f.elements[2][0] - i.elements[2][0];
				d[2][1] = f.elements[2][1] - i.elements[2][1];
				delta_val = d;
			}
			break;
		case Variant::QUAT:
			delta_val = final_val.operator Quat() - initial_val.operator Quat();
			break;
		case Variant::_AABB:
			{
				AABB i = initial_val;
				AABB f = final_val;
				delta_val = AABB(f.pos - i.pos, f.size - i.size);
			}
			break;
		case Variant::TRANSFORM:
			{
				Transform i = initial_val;
				Transform f = final_val;
				Transform d;
				d.set(f.basis.elements[0][0] - i.basis.elements[0][0],
					f.basis.elements[0][1] - i.basis.elements[0][1],
					f.basis.elements[0][2] - i.basis.elements[0][2],
					f.basis.elements[1][0] - i.basis.elements[1][0],
					f.basis.elements[1][1] - i.basis.elements[1][1],
					f.basis.elements[1][2] - i.basis.elements[1][2],
					f.basis.elements[2][0] - i.basis.elements[2][0],
					f.basis.elements[2][1] - i.basis.elements[2][1],
					f.basis.elements[2][2] - i.basis.elements[2][2],
					f.origin.x - i.origin.x,
					f.origin.y - i.origin.y,
					f.origin.z - i.origin.z
				);

				delta_val = d;
			}
			break;
		case Variant::COLOR:
			{
				Color i = initial_val;
				Color f = final_val;
				delta_val = Color(f.r - i.r, f.g - i.g, f.b - i.b, f.a - i.a);
			}
			break;

		default:
			ERR_PRINT("Invalid param type, except(int/real/vector2/vector/matrix/matrix32/quat/aabb/transform/color)");
			return false;
	};
	return true;
}

bool Tween::interpolate_property(Variant p_object
	, String p_property
	, Variant p_initial_val
	, Variant p_final_val
	, real_t p_times_in_sec
	, TransitionType p_trans_type
	, EaseType p_ease_type
	, real_t p_delay
) {
	// convert INT to REAL is better for interpolaters
	if(p_initial_val.get_type() == Variant::INT) p_initial_val = p_initial_val.operator real_t();
	if(p_final_val.get_type() == Variant::INT) p_final_val = p_final_val.operator real_t();

	ERR_FAIL_COND_V(p_object.get_type() != Variant::OBJECT, false);
	ERR_FAIL_COND_V(p_initial_val.get_type() != p_final_val.get_type(), false);
	ERR_FAIL_COND_V(p_times_in_sec <= 0, false);
	ERR_FAIL_COND_V(p_trans_type < 0 || p_trans_type >= TRANS_COUNT, false);
	ERR_FAIL_COND_V(p_ease_type < 0 || p_ease_type >= EASE_COUNT, false);
	ERR_FAIL_COND_V(p_delay < 0, false);

	bool prop_found = false;
	Object *obj = (Object *) p_object;
	List<PropertyInfo> props;
	obj->get_property_list(&props);
	for(List<PropertyInfo>::Element *E=props.front();E;E=E->next()) {

		PropertyInfo& prop=E->get();
		if(prop.name==p_property)
		{
			prop_found = true;
			break;
		}
	}
	ERR_FAIL_COND_V(!prop_found, false);

	InterpolateData data;
	data.active = true;
	data.is_method = false;
	data.finish = false;
	data.elapsed = 0;

	data.object = p_object;
	data.key = p_property;
	data.initial_val = p_initial_val;
	data.final_val = p_final_val;
	data.times_in_sec = p_times_in_sec;
	data.trans_type = p_trans_type;
	data.ease_type = p_ease_type;
	data.delay = p_delay;

	if(!_calc_delta_val(data))
		return false;

	interpolates.push_back(data);
	return true;
}

bool Tween::interpolate_method(Variant p_object
	, String p_method
	, Variant p_initial_val
	, Variant p_final_val
	, real_t p_times_in_sec
	, TransitionType p_trans_type
	, EaseType p_ease_type
	, real_t p_delay
) {
	// convert INT to REAL is better for interpolaters
	if(p_initial_val.get_type() == Variant::INT) p_initial_val = p_initial_val.operator real_t();
	if(p_final_val.get_type() == Variant::INT) p_final_val = p_final_val.operator real_t();

	ERR_FAIL_COND_V(p_object.get_type() != Variant::OBJECT, false);
	ERR_FAIL_COND_V(p_initial_val.get_type() != p_final_val.get_type(), false);
	ERR_FAIL_COND_V(p_times_in_sec <= 0, false);
	ERR_FAIL_COND_V(p_trans_type < 0 || p_trans_type >= TRANS_COUNT, false);
	ERR_FAIL_COND_V(p_ease_type < 0 || p_ease_type >= EASE_COUNT, false);
	ERR_FAIL_COND_V(p_delay < 0, false);

	Object *obj = (Object *) p_object;
	ERR_FAIL_COND_V(!obj->has_method(p_method), false);

	InterpolateData data;
	data.active = true;
	data.is_method = true;
	data.finish = false;
	data.elapsed = 0;

	data.object = p_object;
	data.key = p_method;
	data.initial_val = p_initial_val;
	data.final_val = p_final_val;
	data.times_in_sec = p_times_in_sec;
	data.trans_type = p_trans_type;
	data.ease_type = p_ease_type;
	data.delay = p_delay;

	if(!_calc_delta_val(data))
		return false;

	interpolates.push_back(data);
	return true;
}

Tween::Tween() {

	//String autoplay;
	tween_process_mode=TWEEN_PROCESS_IDLE;
	processing=false;
	active=false;
	repeat=false;
	speed_scale=1;
}

Tween::~Tween() {

}
