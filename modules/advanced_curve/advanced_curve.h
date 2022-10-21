#ifndef ADVANCED_CURVE_H
#define ADVANCED_CURVE_H

#include "scene/resources/curve.h"
#include "core/variant.h"

class AdvancedCurve : public Curve {
	GDCLASS(AdvancedCurve, Curve);
public:
	enum ApproxMethod : unsigned int{
		AC_NO_METHOD,
		AC_TRAPEZOID,
		AC_SIMPSON,
	};
protected:
	static void _bind_methods();

	inline void pollute_samples() { _is_samples_dirty = true; }
	void trapezoid_bake();
	void simpson_bake();
	static inline float clampf(const float& val, const float& from, const float& to) { return (val < from ? from : (val > to ? to : val)); }
private:
	bool _is_samples_dirty = true;
	int sample_resolution = 50; 
	float range = 5.0;

	ApproxMethod bake_method = ApproxMethod::AC_NO_METHOD;
	PoolRealArray samples_pool;
public:
	AdvancedCurve();
	~AdvancedCurve();

	void bake_samples();

	inline bool is_samples_polluted() const { return _is_samples_dirty; }

	void set_bake_method(ApproxMethod new_method);
	inline ApproxMethod get_bake_method() const { return bake_method; }

	void set_sample_resolution(const int& res);
	inline int get_sample_resolution() const { return sample_resolution; }

	void set_range(const float& new_range);
	inline float get_range() const { return range; }

	float get_area(const float& from, const float& to);
	float get_area_no_bake(const float& from, const float& to, ApproxMethod met = AC_TRAPEZOID);
};

#endif