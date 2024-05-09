#ifndef QHPLANE_HPP_
#define QHPLANE_HPP_

#include "Vector3.hpp"

namespace quickhull {

	template<typename T>
	class Plane {
	public:
		Vector3<T> m_N;
		
		// Signed distance (if normal is of length 1) to the plane from origin
		T m_D;
		
		// Normal length squared
		T m_sqrNLength;

		bool isPointOnPositiveSide(const Vector3<T>& Q) const {
			T d = m_N.dotProduct(Q)+m_D;
			if (d>=0) return true;
			return false;
		}

		Plane() = default;

		// Construct a plane using normal N and any point P on the plane
		Plane(const Vector3<T>& N, const Vector3<T>& P) : m_N(N), m_D(-N.dotProduct(P)), m_sqrNLength(m_N.x*m_N.x+m_N.y*m_N.y+m_N.z*m_N.z) {
			
		}
	};

}


#endif /* PLANE_HPP_ */
