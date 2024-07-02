
#ifndef QuickHull_MathUtils_hpp
#define QuickHull_MathUtils_hpp

#include "Structs/Vector3.hpp"
#include "Structs/Ray.hpp"

namespace quickhull {
	
	namespace mathutils {
		
		template <typename T>
		inline T getSquaredDistanceBetweenPointAndRay(const Vector3<T>& p, const Ray<T>& r) {
			const Vector3<T> s = p-r.m_S;
			T t = s.dotProduct(r.m_V);
			return s.getLengthSquared() - t*t*r.m_VInvLengthSquared;
		}
		
		// Note that the unit of distance returned is relative to plane's normal's length (divide by N.getNormalized() if needed to get the "real" distance).
		template <typename T>
		inline T getSignedDistanceToPlane(const Vector3<T>& v, const Plane<T>& p) {
			return p.m_N.dotProduct(v) + p.m_D;
		}
		
		template <typename T>
		inline Vector3<T> getTriangleNormal(const Vector3<T>& a,const Vector3<T>& b,const Vector3<T>& c) {
			// We want to get (a-c).crossProduct(b-c) without constructing temp vectors
			T x = a.x - c.x;
			T y = a.y - c.y;
			T z = a.z - c.z;
			T rhsx = b.x - c.x;
			T rhsy = b.y - c.y;
			T rhsz = b.z - c.z;
			T px = y * rhsz - z * rhsy ;
			T py = z * rhsx - x * rhsz ;
			T pz = x * rhsy - y * rhsx ;
			return Vector3<T>(px,py,pz);
		}
		
		
	}
	
}


#endif
