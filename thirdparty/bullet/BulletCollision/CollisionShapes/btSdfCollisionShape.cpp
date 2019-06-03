#include "btSdfCollisionShape.h"
#include "btMiniSDF.h"
#include "LinearMath/btAabbUtil2.h"

struct btSdfCollisionShapeInternalData
{
	btVector3 m_localScaling;
	btScalar m_margin;
	btMiniSDF m_sdf;

	btSdfCollisionShapeInternalData()
		: m_localScaling(1, 1, 1),
		  m_margin(0)
	{
	}
};

bool btSdfCollisionShape::initializeSDF(const char* sdfData, int sizeInBytes)
{
	bool valid = m_data->m_sdf.load(sdfData, sizeInBytes);
	return valid;
}
btSdfCollisionShape::btSdfCollisionShape()
{
	m_shapeType = SDF_SHAPE_PROXYTYPE;
	m_data = new btSdfCollisionShapeInternalData();

	//"E:/develop/bullet3/data/toys/ground_hole64_64_8.cdf");//ground_cube.cdf");
	/*unsigned int field_id=0;
	Eigen::Vector3d x (1,10,1);
	Eigen::Vector3d gradient;
	double dist = m_data->m_sdf.interpolate(field_id, x, &gradient);
	printf("dist=%g\n", dist);
	*/
}
btSdfCollisionShape::~btSdfCollisionShape()
{
	delete m_data;
}

void btSdfCollisionShape::getAabb(const btTransform& t, btVector3& aabbMin, btVector3& aabbMax) const
{
	btAssert(m_data->m_sdf.isValid());
	btVector3 localAabbMin = m_data->m_sdf.m_domain.m_min;
	btVector3 localAabbMax = m_data->m_sdf.m_domain.m_max;
	btScalar margin(0);
	btTransformAabb(localAabbMin, localAabbMax, margin, t, aabbMin, aabbMax);
}

void btSdfCollisionShape::setLocalScaling(const btVector3& scaling)
{
	m_data->m_localScaling = scaling;
}
const btVector3& btSdfCollisionShape::getLocalScaling() const
{
	return m_data->m_localScaling;
}
void btSdfCollisionShape::calculateLocalInertia(btScalar mass, btVector3& inertia) const
{
	inertia.setValue(0, 0, 0);
}
const char* btSdfCollisionShape::getName() const
{
	return "btSdfCollisionShape";
}
void btSdfCollisionShape::setMargin(btScalar margin)
{
	m_data->m_margin = margin;
}
btScalar btSdfCollisionShape::getMargin() const
{
	return m_data->m_margin;
}

void btSdfCollisionShape::processAllTriangles(btTriangleCallback* callback, const btVector3& aabbMin, const btVector3& aabbMax) const
{
	//not yet
}

bool btSdfCollisionShape::queryPoint(const btVector3& ptInSDF, btScalar& distOut, btVector3& normal)
{
	int field = 0;
	btVector3 grad;
	double dist;
	bool hasResult = m_data->m_sdf.interpolate(field, dist, ptInSDF, &grad);
	if (hasResult)
	{
		normal.setValue(grad[0], grad[1], grad[2]);
		distOut = dist;
	}
	return hasResult;
}
