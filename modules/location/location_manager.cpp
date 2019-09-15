/* location_manager.cpp */

#include "location_manager.h"

/**
	@author Cagdas Caglak <cagdascaglak@gmail.com>
*/

void LocationManager::_bind_methods() {

	ClassDB::bind_method(D_METHOD("request_location_updates", "location_parameters"), &LocationManager::request_location_updates);
	ClassDB::bind_method(D_METHOD("stop_request_location"), &LocationManager::stop_request_location);

	ADD_SIGNAL(MethodInfo("on_location_result", PropertyInfo(Variant::OBJECT, "location_result", PROPERTY_HINT_RESOURCE_TYPE, "LocationResult", 0)));
}

void LocationManager::request_location_updates(const Ref<LocationParam> &p_location_param) {
	Ref<LocationParam> location_param = p_location_param;
	OS::LocationParameter location_parameter;
	location_parameter.interval = location_param->get_interval();
	location_parameter.max_wait_time = location_param->get_max_wait_time();

	OS::get_singleton()->request_location(location_parameter);
}

void LocationManager::stop_request_location() {
	OS::get_singleton()->stop_request_location();
}

void LocationManager::_send_location_data(OS::Location location) {
	location_result->longitude = location.longitute;
	location_result->latitude = location.latitude;
	location_result->horizontal_accuracy = location.horizontal_accuracy;
	location_result->vertical_accuracy = location.vertical_accuracy;
	location_result->altitude = location.altitude;
	location_result->speed = location.speed;
	location_result->time = location.time;
	emit_signal("on_location_result", location_result);
}

LocationManager *LocationManager::singleton = NULL;

LocationManager *LocationManager::get_singleton() {

	return singleton;
}

LocationManager::LocationManager() {
	singleton = this;

	location_result = memnew(LocationResult);
}

LocationManager::~LocationManager() {
	memdelete(location_result);
}
