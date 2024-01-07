//
// Created by Harris.Lu on 2024/1/7.
//

#ifndef GODOT_ELLIPSOID_H
#define GODOT_ELLIPSOID_H

#include "core/io/resource.h"
#include "core/math/vector3.h"
#include "core/math/color.h"

using namespace godot;
namespace Cesium {

class Ellipsoid : public Resource {
	GDCLASS(Ellipsoid, Resource);

protected:
	static void _bind_methods();

private:
	Color color = Color(1.0f, 1.0f, 1.0f);

public:
	Color get_color();
//	static /*constexpr*/ const Ellipsoid _WGS84;

//	static Ref<Ellipsoid> WGS84();
	Ellipsoid();


//	Ellipsoid()
//			: Ellipsoid(1.0, 1.0, 1.0) {}

//	Ellipsoid(double x, double y, double z)
//			: Ellipsoid(Vector3(x, y, z)) {}
//
//	Ellipsoid(const Vector3& radii)
//			: _radii(radii),
//			_radiiSquared(radii.x * radii.x, radii.y * radii.y, radii.z * radii.z),
//			_oneOverRadii(1.0 / radii.x, 1.0 / radii.y, 1.0 / radii.z),
//			_oneOverRadiiSquared(
//					1.0 / (radii.x * radii.x),
//					1.0 / (radii.y * radii.y),
//					1.0 / (radii.z * radii.z)),
//			_centerToleranceSquared(1e-1) {}
//
//	Vector3 getRadii() { return this->_radii; }

//private:
//	Vector3 _radii;
//	Vector3 _radiiSquared;
//	Vector3 _oneOverRadii;
//	Vector3 _oneOverRadiiSquared;
//	double _centerToleranceSquared;
};

} //namespace Cesium

#endif //GODOT_ELLIPSOID_H
