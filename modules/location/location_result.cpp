/* location_result.cpp */

#include "location_result.h"

/**
	@author Cagdas Caglak <cagdascaglak@gmail.com>
*/

void LocationResult::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_longitude"), &LocationResult::get_longitude);
	ClassDB::bind_method(D_METHOD("get_latitude"), &LocationResult::get_latitude);
	ClassDB::bind_method(D_METHOD("get_horizontal_accuracy"), &LocationResult::get_horizontal_accuracy);
	ClassDB::bind_method(D_METHOD("get_vertical_accuracy"), &LocationResult::get_vertical_accuracy);
	ClassDB::bind_method(D_METHOD("get_altitude"), &LocationResult::get_altitude);
	ClassDB::bind_method(D_METHOD("get_speed"), &LocationResult::get_speed);
	ClassDB::bind_method(D_METHOD("get_time"), &LocationResult::get_time);

	ADD_PROPERTY(PropertyInfo(Variant::REAL, "longitude"), "", "get_longitude");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "latitude"), "", "get_latitude");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "horizaontal_accuracy"), "", "get_horizontal_accuracy");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "vertical_accuracy"), "", "get_vertical_accuracy");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "altitude"), "", "get_altitude");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "speed"), "", "get_speed");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "time"), "", "get_time");
}

real_t LocationResult::get_longitude() const {
	return longitude;
}

real_t LocationResult::get_latitude() const {
	return latitude;
}

real_t LocationResult::get_horizontal_accuracy() const {
	return horizontal_accuracy;
}

real_t LocationResult::get_vertical_accuracy() const {
	return vertical_accuracy;
}

real_t LocationResult::get_altitude() const {
	return altitude;
}

real_t LocationResult::get_speed() const {
	return speed;
}

uint64_t LocationResult::get_time() const {
	return time;
}

LocationResult::LocationResult() {
}
