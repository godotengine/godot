//
// Created by Harris.Lu on 2024/1/7.
//

#include "Ellipsoid.h"

namespace Cesium {

//const Ellipsoid Ellipsoid::_WGS84(6378137.0, 6378137.0, 6356752.3142451793);

void Ellipsoid::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_color"), &Ellipsoid::get_color);

//	ClassDB::bind_method(D_METHOD("getRadii"), &Ellipsoid::getRadii);
//	BIND_CONSTANT()
//	ClassDB::bind_static_method("Ellipsoid", D_METHOD("WGS84"), &Ellipsoid::WGS84);
//	ClassDB::bind_method(D_METHOD("WGS84"), &Ellipsoid::WGS84);
//	ADD_PROPERTY(PropertyInfo(Variant::INT, "WGS84"), "WGS84", "WGS84");
}

Color Ellipsoid::get_color() {
	return color;
}

Ellipsoid::Ellipsoid() {
}

//Ref<Ellipsoid> Ellipsoid::WGS84() {
//	return memnew(Ellipsoid(6378137.0, 6378137.0, 6356752.3142451793));
////	return Ref<Ellipsoid>(6378137.0, 6378137.0, 6356752.3142451793);
//}

} //namespace Cesium