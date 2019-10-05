#ifndef MINISDF_H
#define MINISDF_H

#include "LinearMath/btVector3.h"
#include "LinearMath/btAabbUtil2.h"
#include "LinearMath/btAlignedObjectArray.h"

struct btMultiIndex
{
	unsigned int ijk[3];
};

struct btAlignedBox3d
{
	btVector3 m_min;
	btVector3 m_max;

	const btVector3& min() const
	{
		return m_min;
	}

	const btVector3& max() const
	{
		return m_max;
	}

	bool contains(const btVector3& x) const
	{
		return TestPointAgainstAabb2(m_min, m_max, x);
	}

	btAlignedBox3d(const btVector3& mn, const btVector3& mx)
		: m_min(mn),
		  m_max(mx)
	{
	}

	btAlignedBox3d()
	{
	}
};

struct btShapeMatrix
{
	double m_vec[32];

	inline double& operator[](int i)
	{
		return m_vec[i];
	}

	inline const double& operator[](int i) const
	{
		return m_vec[i];
	}
};

struct btShapeGradients
{
	btVector3 m_vec[32];

	void topRowsDivide(int row, double denom)
	{
		for (int i = 0; i < row; i++)
		{
			m_vec[i] /= denom;
		}
	}

	void bottomRowsMul(int row, double val)
	{
		for (int i = 32 - row; i < 32; i++)
		{
			m_vec[i] *= val;
		}
	}

	inline btScalar& operator()(int i, int j)
	{
		return m_vec[i][j];
	}
};

struct btCell32
{
	unsigned int m_cells[32];
};

struct btMiniSDF
{
	btAlignedBox3d m_domain;
	unsigned int m_resolution[3];
	btVector3 m_cell_size;
	btVector3 m_inv_cell_size;
	std::size_t m_n_cells;
	std::size_t m_n_fields;
	bool m_isValid;

	btAlignedObjectArray<btAlignedObjectArray<double> > m_nodes;
	btAlignedObjectArray<btAlignedObjectArray<btCell32> > m_cells;
	btAlignedObjectArray<btAlignedObjectArray<unsigned int> > m_cell_map;

	btMiniSDF()
		: m_isValid(false)
	{
	}
	bool load(const char* data, int size);
	bool isValid() const
	{
		return m_isValid;
	}
	unsigned int multiToSingleIndex(btMultiIndex const& ijk) const;

	btAlignedBox3d subdomain(btMultiIndex const& ijk) const;

	btMultiIndex singleToMultiIndex(unsigned int l) const;

	btAlignedBox3d subdomain(unsigned int l) const;

	btShapeMatrix
	shape_function_(btVector3 const& xi, btShapeGradients* gradient = 0) const;

	bool interpolate(unsigned int field_id, double& dist, btVector3 const& x, btVector3* gradient) const;
};

#endif  //MINISDF_H
