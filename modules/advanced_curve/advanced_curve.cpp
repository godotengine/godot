#include "advanced_curve.h"

VARIANT_ENUM_CAST(AdvancedCurve::ApproxMethod);

AdvancedCurve::AdvancedCurve(){

}

AdvancedCurve::~AdvancedCurve(){

}

void AdvancedCurve::_bind_methods(){
	ClassDB::bind_method(D_METHOD("bake_samples"), &AdvancedCurve::bake_samples);
	ClassDB::bind_method(D_METHOD("get_area", "from", "to"), &AdvancedCurve::get_area);
	ClassDB::bind_method(D_METHOD("get_area_no_bake", "from", "to", "method"), &AdvancedCurve::get_area_no_bake);

	ClassDB::bind_method(D_METHOD("is_samples_polluted"), &AdvancedCurve::is_samples_polluted);
	
	ClassDB::bind_method(D_METHOD("set_bake_method", "new_method"), &AdvancedCurve::set_bake_method);
	ClassDB::bind_method(D_METHOD("get_bake_method"), &AdvancedCurve::get_bake_method);

	ClassDB::bind_method(D_METHOD("set_sample_resolution", "new_resolution"), &AdvancedCurve::set_sample_resolution);
	ClassDB::bind_method(D_METHOD("get_sample_resolution"), &AdvancedCurve::get_sample_resolution);

	ClassDB::bind_method(D_METHOD("set_range"), &AdvancedCurve::set_range);
	ClassDB::bind_method(D_METHOD("get_range"), &AdvancedCurve::get_range);

	BIND_ENUM_CONSTANT(AC_NO_METHOD);
	BIND_ENUM_CONSTANT(AC_TRAPEZOID);
	BIND_ENUM_CONSTANT(AC_SIMPSON);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "bake_method", PROPERTY_HINT_ENUM, "NoMethod,Trapezoid,Simpson"), "set_bake_method", "get_bake_method");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "sample_resolution", PROPERTY_HINT_RANGE, "5,5000000"), "set_sample_resolution", "get_sample_resolution");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "range", PROPERTY_HINT_RANGE, "0.01,1000000.0,0.01"), "set_range", "get_range");
}

void AdvancedCurve::set_bake_method(ApproxMethod new_method){
	if (bake_method == new_method) return;
	bake_method = new_method;
	pollute_samples(); emit_changed();
}

void AdvancedCurve::set_sample_resolution(const int& res){
	if (sample_resolution == res || res < 5) return;
	sample_resolution = res;
	pollute_samples(); emit_changed();
}

void AdvancedCurve::set_range(const float& new_range){
	if (range == new_range || range < 0.01) return;
	range = new_range;
	pollute_samples(); emit_changed();
}

void AdvancedCurve::trapezoid_bake(){
	auto micro_length = range / (2.0 * sample_resolution);
	samples_pool.resize(0);
	samples_pool.resize(sample_resolution + 1);
	// Get sigma
	// samples_pool[0] = get_min_value() * micro_length;
	samples_pool.set(0, interpolate_baked(0.0));
	for (int i = 1; i <= sample_resolution; i++){
		auto modifier = 2;
		auto offset = (float)i / sample_resolution;
		if (i == sample_resolution) modifier = 1;
		auto val = samples_pool[i - 1] + (modifier * interpolate_baked(offset) * micro_length);
		samples_pool.set(i, val);
	}
}
void AdvancedCurve::simpson_bake(){
	auto micro_length = range / (1.0 * sample_resolution);
	PoolRealArray mini_pool;
	mini_pool.resize(sample_resolution + 1);
	mini_pool.set(0, interpolate_baked(0.0));
	for (int i = 1; i <= sample_resolution; i++){
		mini_pool.set(i, micro_length * i);
	}
	samples_pool.resize(0);
	samples_pool.resize(sample_resolution + 1);
	samples_pool.set(0, interpolate_baked(0.0));
	for (int i = 1; i <= (sample_resolution / 2); i++){
		auto value = (interpolate_baked(mini_pool[2 * i - 2])) + (4.0 * interpolate_baked(mini_pool[2 * i - 1])) + (interpolate_baked(mini_pool[2 * i]));
		samples_pool.set(i, samples_pool[i - 1] + ((value * micro_length) / 3.0));
	}
}

void AdvancedCurve::bake_samples(){
	// if (!is_samples_polluted()) return;
	switch (bake_method){
		case ApproxMethod::AC_NO_METHOD:
			samples_pool.resize(0);
			return;
		case ApproxMethod::AC_TRAPEZOID: trapezoid_bake(); break;
		case ApproxMethod::AC_SIMPSON: simpson_bake(); break;
	}
	_is_samples_dirty = false;
}

float AdvancedCurve::get_area(const float& from, const float& to){
	if (from >= to) return 0.0;
	if (is_samples_polluted() || is_cache_dirty()) bake_samples();
	if (is_samples_polluted()) return 0.0;
	auto local_from = from / range;
	auto local_to = to / range;
	auto sum = 0.0;
	if (local_from < 0.0){
		if (local_to <= 0.0){
			return (to - from) * interpolate_baked(0.0);
		}
		sum += (-from) * interpolate_baked(0.0);
		local_from = 0.0;
	} 
	if (local_to > 1.0){
		if (local_from >= 1.0){
			return (to - from) * interpolate_baked(1.0);
		}
		local_to = 1.0;
		sum += to * interpolate_baked(1.0);
	}
	// Data Sampling
	auto baked_from = local_from * sample_resolution;
	auto baked_to = local_to * sample_resolution;
	auto prev = samples_pool[(int)baked_from];
	auto curr = samples_pool[(int)baked_to];
	return (curr - prev) + sum;
}

float AdvancedCurve::get_area_no_bake(const float& from, const float& to, ApproxMethod met){
	if (from >= to) return 0.0;
	float sum = 0.0;
	switch (met){
		case ApproxMethod::AC_TRAPEZOID: {
			auto micro_length = range / (2.0 * sample_resolution);
			// No need to multiply by range I think
			sum += interpolate_baked(0.0) + interpolate_baked(1.0);
			for (int i = 1; i < sample_resolution; i++){
				auto offset = (float)i / sample_resolution;
				sum += 2.0 * interpolate_baked(offset);
			}
			sum *= micro_length;
			return sum;
		}
		case ApproxMethod::AC_SIMPSON: {
			auto micro_length = range / (1.0 * sample_resolution);
			PoolRealArray mini_pool;
			mini_pool.resize(sample_resolution + 1);
			mini_pool.set(0, 0);
			for (int i = 1; i <= sample_resolution; i++){
				mini_pool.set(i, micro_length * i);
			}
			for (int i = 1; i <= (sample_resolution / 2); i++){
				sum += (interpolate_baked(mini_pool[2 * i - 2])) + (4.0 * interpolate_baked(mini_pool[2 * i - 1])) + (interpolate_baked(mini_pool[2 * i]));
			}
			sum = sum * micro_length / 3.0;
			return sum;
		}
		default: return sum;
	}
	return sum;
}
