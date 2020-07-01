#include "core/object.h"
#include "scene/resources/mesh.h"

#if !defined(__aligned)

#if (defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)) && !defined(__CYGWIN__)
#define __aligned(...) __declspec(align(__VA_ARGS__))
#else
#define __aligned(...) __attribute__((aligned(__VA_ARGS__)))
#endif

#endif

class RaytraceEngine : public Object {
	//GDCLASS(RaytraceEngine, Object);
public:
	// compatible with embree3 rays
	struct __aligned(16) Ray {
		const static unsigned int RTC_INVALID_GEOMETRY_ID = ((unsigned int)-1); // from rtcore_common.h

		/*! Default construction does nothing. */
		_FORCE_INLINE_ Ray() :
				geomID(RTC_INVALID_GEOMETRY_ID) {}

		/*! Constructs a ray from origin, direction, and ray segment. Near
		 *  has to be smaller than far. */
		_FORCE_INLINE_ Ray(const Vector3 &org,
				const Vector3 &dir,
				float tnear = 0.0f,
				float tfar = INFINITY
				//float time = embree::zero,
				//int mask = -1,
				//unsigned int geomID = RTC_INVALID_GEOMETRY_ID,
				//unsigned int primID = RTC_INVALID_GEOMETRY_ID,
				//unsigned int instID = RTC_INVALID_GEOMETRY_ID
				) :
				org(org),
				tnear(tnear),
				dir(dir),
				time(0.0f),
				tfar(tfar),
				mask(-1),
				u(0.0),
				v(0.0),
				primID(RTC_INVALID_GEOMETRY_ID),
				geomID(RTC_INVALID_GEOMETRY_ID),
				instID(RTC_INVALID_GEOMETRY_ID) {}

		/*! Tests if we hit something. */
		_FORCE_INLINE_ explicit operator bool() const { return geomID != RTC_INVALID_GEOMETRY_ID; }

	public:
		Vector3 org; //!< Ray origin + tnear
		float tnear; //!< Start of ray segment
		Vector3 dir; //!< Ray direction + tfar
		float time; //!< Time of this ray for motion blur.
		float tfar; //!< End of ray segment
		unsigned int mask; //!< used to mask out objects during traversal
		unsigned int id; //!< ray ID
		unsigned int flags; //!< ray flags

		Vector3 normal; //!< Not normalized geometry normal
		float u; //!< Barycentric u coordinate of hit
		float v; //!< Barycentric v coordinate of hit
		unsigned int primID; //!< primitive ID
		unsigned int geomID; //!< geometry ID
		unsigned int instID; //!< instance ID
	};

	struct Triangle {
	public:
		Triangle() {}
		Triangle(unsigned v0, unsigned v1, unsigned v2) :
				v0(v0),
				v1(v1),
				v2(v2) {}

	public:
		unsigned v0, v1, v2;
	};

	virtual bool intersect(Ray &p_ray) = 0;

	virtual void intersect(Vector<Ray> &r_rays) = 0;

	virtual void init_scene() = 0;
	virtual void add_mesh(const Ref<Mesh> p_mesh, const Transform &p_xform, unsigned int p_id) = 0;
	virtual void set_mesh_alpha_texture(Ref<Image> p_alpha_texture, unsigned int p_id) = 0;
	virtual void commit_scene() = 0;

	virtual void set_mesh_filter(const Set<int> &p_mesh_ids) = 0;
	virtual void clear_mesh_filter() = 0;

	RaytraceEngine();

private:
	static RaytraceEngine *singleton;

public:
	static RaytraceEngine *get_singleton();
};
