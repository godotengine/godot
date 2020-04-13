// a set of routines that let you do common 3d math
// operations without any vector, matrix, or quaternion
// classes or templates.
//
// a vector (or point) is a 'float *' to 3 floating point numbers.
// a matrix is a 'float *' to an array of 16 floating point numbers representing a 4x4 transformation matrix compatible with D3D or OGL
// a quaternion is a 'float *' to 4 floats representing a quaternion x,y,z,w
//

#ifdef _MSC_VER
#pragma warning(disable:4996)
#endif

namespace FLOAT_MATH
{

void fm_inverseRT(const REAL matrix[16],const REAL pos[3],REAL t[3]) // inverse rotate translate the point.
{

	REAL _x = pos[0] - matrix[3*4+0];
	REAL _y = pos[1] - matrix[3*4+1];
	REAL _z = pos[2] - matrix[3*4+2];

	// Multiply inverse-translated source vector by inverted rotation transform

	t[0] = (matrix[0*4+0] * _x) + (matrix[0*4+1] * _y) + (matrix[0*4+2] * _z);
	t[1] = (matrix[1*4+0] * _x) + (matrix[1*4+1] * _y) + (matrix[1*4+2] * _z);
	t[2] = (matrix[2*4+0] * _x) + (matrix[2*4+1] * _y) + (matrix[2*4+2] * _z);

}

REAL fm_getDeterminant(const REAL matrix[16])
{
  REAL tempv[3];
  REAL p0[3];
  REAL p1[3];
  REAL p2[3];


	p0[0] = matrix[0*4+0];
	p0[1] = matrix[0*4+1];
	p0[2] = matrix[0*4+2];

	p1[0] = matrix[1*4+0];
	p1[1] = matrix[1*4+1];
	p1[2] = matrix[1*4+2];

	p2[0] = matrix[2*4+0];
	p2[1] = matrix[2*4+1];
	p2[2] = matrix[2*4+2];

  fm_cross(tempv,p1,p2);

  return fm_dot(p0,tempv);

}

REAL fm_squared(REAL x) { return x*x; };

void fm_decomposeTransform(const REAL local_transform[16],REAL trans[3],REAL rot[4],REAL scale[3])
{

  trans[0] = local_transform[12];
  trans[1] = local_transform[13];
  trans[2] = local_transform[14];

  scale[0] = (REAL)sqrt(fm_squared(local_transform[0*4+0]) + fm_squared(local_transform[0*4+1]) + fm_squared(local_transform[0*4+2]));
  scale[1] = (REAL)sqrt(fm_squared(local_transform[1*4+0]) + fm_squared(local_transform[1*4+1]) + fm_squared(local_transform[1*4+2]));
  scale[2] = (REAL)sqrt(fm_squared(local_transform[2*4+0]) + fm_squared(local_transform[2*4+1]) + fm_squared(local_transform[2*4+2]));

  REAL m[16];
  memcpy(m,local_transform,sizeof(REAL)*16);

  REAL sx = 1.0f / scale[0];
  REAL sy = 1.0f / scale[1];
  REAL sz = 1.0f / scale[2];

  m[0*4+0]*=sx;
  m[0*4+1]*=sx;
  m[0*4+2]*=sx;

  m[1*4+0]*=sy;
  m[1*4+1]*=sy;
  m[1*4+2]*=sy;

  m[2*4+0]*=sz;
  m[2*4+1]*=sz;
  m[2*4+2]*=sz;

  fm_matrixToQuat(m,rot);

}

void fm_getSubMatrix(int32_t ki,int32_t kj,REAL pDst[16],const REAL matrix[16])
{
	int32_t row, col;
	int32_t dstCol = 0, dstRow = 0;

	for ( col = 0; col < 4; col++ )
	{
		if ( col == kj )
		{
			continue;
		}
		for ( dstRow = 0, row = 0; row < 4; row++ )
		{
			if ( row == ki )
			{
				continue;
			}
			pDst[dstCol*4+dstRow] = matrix[col*4+row];
			dstRow++;
		}
		dstCol++;
	}
}

void  fm_inverseTransform(const REAL matrix[16],REAL inverse_matrix[16])
{
	REAL determinant = fm_getDeterminant(matrix);
	determinant = 1.0f / determinant;
	for (int32_t i = 0; i < 4; i++ )
	{
		for (int32_t j = 0; j < 4; j++ )
		{
			int32_t sign = 1 - ( ( i + j ) % 2 ) * 2;
			REAL subMat[16];
			fm_identity(subMat);
			fm_getSubMatrix( i, j, subMat, matrix );
			REAL subDeterminant = fm_getDeterminant(subMat);
			inverse_matrix[i*4+j] = ( subDeterminant * sign ) * determinant;
		}
	}
}

void fm_identity(REAL matrix[16]) // set 4x4 matrix to identity.
{
	matrix[0*4+0] = 1;
	matrix[1*4+1] = 1;
	matrix[2*4+2] = 1;
	matrix[3*4+3] = 1;

	matrix[1*4+0] = 0;
	matrix[2*4+0] = 0;
	matrix[3*4+0] = 0;

	matrix[0*4+1] = 0;
	matrix[2*4+1] = 0;
	matrix[3*4+1] = 0;

	matrix[0*4+2] = 0;
	matrix[1*4+2] = 0;
	matrix[3*4+2] = 0;

	matrix[0*4+3] = 0;
	matrix[1*4+3] = 0;
	matrix[2*4+3] = 0;

}

void  fm_quatToEuler(const REAL quat[4],REAL &ax,REAL &ay,REAL &az)
{
	REAL x = quat[0];
	REAL y = quat[1];
	REAL z = quat[2];
	REAL w = quat[3];

	REAL sint	     = (2.0f * w * y) - (2.0f * x * z);
	REAL cost_temp = 1.0f - (sint * sint);
	REAL cost	   	 = 0;

	if ( (REAL)fabs(cost_temp) > 0.001f )
	{
		cost = (REAL)sqrt( cost_temp );
	}

	REAL sinv, cosv, sinf, cosf;
	if ( (REAL)fabs(cost) > 0.001f )
	{
	cost = 1.0f / cost;
		sinv = ((2.0f * y * z) + (2.0f * w * x)) * cost;
		cosv = (1.0f - (2.0f * x * x) - (2.0f * y * y)) * cost;
		sinf = ((2.0f * x * y) + (2.0f * w * z)) * cost;
		cosf = (1.0f - (2.0f * y * y) - (2.0f * z * z)) * cost;
	}
	else
	{
		sinv = (2.0f * w * x) - (2.0f * y * z);
		cosv = 1.0f - (2.0f * x * x) - (2.0f * z * z);
		sinf = 0;
		cosf = 1.0f;
	}

	// compute output rotations
	ax	= (REAL)atan2( sinv, cosv );
	ay	= (REAL)atan2( sint, cost );
	az	= (REAL)atan2( sinf, cosf );

}

void fm_eulerToMatrix(REAL ax,REAL ay,REAL az,REAL *matrix) // convert euler (in radians) to a dest 4x4 matrix (translation set to zero)
{
  REAL quat[4];
  fm_eulerToQuat(ax,ay,az,quat);
  fm_quatToMatrix(quat,matrix);
}

void fm_getAABB(uint32_t vcount,const REAL *points,uint32_t pstride,REAL *bmin,REAL *bmax)
{

  const uint8_t *source = (const uint8_t *) points;

	bmin[0] = points[0];
	bmin[1] = points[1];
	bmin[2] = points[2];

	bmax[0] = points[0];
	bmax[1] = points[1];
	bmax[2] = points[2];


  for (uint32_t i=1; i<vcount; i++)
  {
	source+=pstride;
	const REAL *p = (const REAL *) source;

	if ( p[0] < bmin[0] ) bmin[0] = p[0];
	if ( p[1] < bmin[1] ) bmin[1] = p[1];
	if ( p[2] < bmin[2] ) bmin[2] = p[2];

		if ( p[0] > bmax[0] ) bmax[0] = p[0];
		if ( p[1] > bmax[1] ) bmax[1] = p[1];
		if ( p[2] > bmax[2] ) bmax[2] = p[2];

  }
}

void  fm_eulerToQuat(const REAL *euler,REAL *quat) // convert euler angles to quaternion.
{
  fm_eulerToQuat(euler[0],euler[1],euler[2],quat);
}

void fm_eulerToQuat(REAL roll,REAL pitch,REAL yaw,REAL *quat) // convert euler angles to quaternion.
{
	roll  *= 0.5f;
	pitch *= 0.5f;
	yaw   *= 0.5f;

	REAL cr = (REAL)cos(roll);
	REAL cp = (REAL)cos(pitch);
	REAL cy = (REAL)cos(yaw);

	REAL sr = (REAL)sin(roll);
	REAL sp = (REAL)sin(pitch);
	REAL sy = (REAL)sin(yaw);

	REAL cpcy = cp * cy;
	REAL spsy = sp * sy;
	REAL spcy = sp * cy;
	REAL cpsy = cp * sy;

	quat[0]   = ( sr * cpcy - cr * spsy);
	quat[1]   = ( cr * spcy + sr * cpsy);
	quat[2]   = ( cr * cpsy - sr * spcy);
	quat[3]   = cr * cpcy + sr * spsy;
}

void fm_quatToMatrix(const REAL *quat,REAL *matrix) // convert quaterinion rotation to matrix, zeros out the translation component.
{

	REAL xx = quat[0]*quat[0];
	REAL yy = quat[1]*quat[1];
	REAL zz = quat[2]*quat[2];
	REAL xy = quat[0]*quat[1];
	REAL xz = quat[0]*quat[2];
	REAL yz = quat[1]*quat[2];
	REAL wx = quat[3]*quat[0];
	REAL wy = quat[3]*quat[1];
	REAL wz = quat[3]*quat[2];

	matrix[0*4+0] = 1 - 2 * ( yy + zz );
	matrix[1*4+0] =     2 * ( xy - wz );
	matrix[2*4+0] =     2 * ( xz + wy );

	matrix[0*4+1] =     2 * ( xy + wz );
	matrix[1*4+1] = 1 - 2 * ( xx + zz );
	matrix[2*4+1] =     2 * ( yz - wx );

	matrix[0*4+2] =     2 * ( xz - wy );
	matrix[1*4+2] =     2 * ( yz + wx );
	matrix[2*4+2] = 1 - 2 * ( xx + yy );

	matrix[3*4+0] = matrix[3*4+1] = matrix[3*4+2] = (REAL) 0.0f;
	matrix[0*4+3] = matrix[1*4+3] = matrix[2*4+3] = (REAL) 0.0f;
	matrix[3*4+3] =(REAL) 1.0f;

}


void fm_quatRotate(const REAL *quat,const REAL *v,REAL *r) // rotate a vector directly by a quaternion.
{
  REAL left[4];

	left[0] =   quat[3]*v[0] + quat[1]*v[2] - v[1]*quat[2];
	left[1] =   quat[3]*v[1] + quat[2]*v[0] - v[2]*quat[0];
	left[2] =   quat[3]*v[2] + quat[0]*v[1] - v[0]*quat[1];
	left[3] = - quat[0]*v[0] - quat[1]*v[1] - quat[2]*v[2];

	r[0] = (left[3]*-quat[0]) + (quat[3]*left[0]) + (left[1]*-quat[2]) - (-quat[1]*left[2]);
	r[1] = (left[3]*-quat[1]) + (quat[3]*left[1]) + (left[2]*-quat[0]) - (-quat[2]*left[0]);
	r[2] = (left[3]*-quat[2]) + (quat[3]*left[2]) + (left[0]*-quat[1]) - (-quat[0]*left[1]);

}


void fm_getTranslation(const REAL *matrix,REAL *t)
{
	t[0] = matrix[3*4+0];
	t[1] = matrix[3*4+1];
	t[2] = matrix[3*4+2];
}

void fm_matrixToQuat(const REAL *matrix,REAL *quat) // convert the 3x3 portion of a 4x4 matrix into a quaterion as x,y,z,w
{

	REAL tr = matrix[0*4+0] + matrix[1*4+1] + matrix[2*4+2];

	// check the diagonal

	if (tr > 0.0f )
	{
		REAL s = (REAL) sqrt ( (double) (tr + 1.0f) );
		quat[3] = s * 0.5f;
		s = 0.5f / s;
		quat[0] = (matrix[1*4+2] - matrix[2*4+1]) * s;
		quat[1] = (matrix[2*4+0] - matrix[0*4+2]) * s;
		quat[2] = (matrix[0*4+1] - matrix[1*4+0]) * s;

	}
	else
	{
		// diagonal is negative
		int32_t nxt[3] = {1, 2, 0};
		REAL  qa[4];

		int32_t i = 0;

		if (matrix[1*4+1] > matrix[0*4+0]) i = 1;
		if (matrix[2*4+2] > matrix[i*4+i]) i = 2;

		int32_t j = nxt[i];
		int32_t k = nxt[j];

		REAL s = (REAL)sqrt ( ((matrix[i*4+i] - (matrix[j*4+j] + matrix[k*4+k])) + 1.0f) );

		qa[i] = s * 0.5f;

		if (s != 0.0f ) s = 0.5f / s;

		qa[3] = (matrix[j*4+k] - matrix[k*4+j]) * s;
		qa[j] = (matrix[i*4+j] + matrix[j*4+i]) * s;
		qa[k] = (matrix[i*4+k] + matrix[k*4+i]) * s;

		quat[0] = qa[0];
		quat[1] = qa[1];
		quat[2] = qa[2];
		quat[3] = qa[3];
	}
//	fm_normalizeQuat(quat);
}


REAL fm_sphereVolume(REAL radius) // return's the volume of a sphere of this radius (4/3 PI * R cubed )
{
	return (4.0f / 3.0f ) * FM_PI * radius * radius * radius;
}


REAL fm_cylinderVolume(REAL radius,REAL h)
{
	return FM_PI * radius * radius *h;
}

REAL fm_capsuleVolume(REAL radius,REAL h)
{
	REAL volume = fm_sphereVolume(radius); // volume of the sphere portion.
	REAL ch = h-radius*2; // this is the cylinder length
	if ( ch > 0 )
	{
		volume+=fm_cylinderVolume(radius,ch);
	}
	return volume;
}

void  fm_transform(const REAL matrix[16],const REAL v[3],REAL t[3]) // rotate and translate this point
{
  if ( matrix )
  {
	REAL tx = (matrix[0*4+0] * v[0]) +  (matrix[1*4+0] * v[1]) + (matrix[2*4+0] * v[2]) + matrix[3*4+0];
	REAL ty = (matrix[0*4+1] * v[0]) +  (matrix[1*4+1] * v[1]) + (matrix[2*4+1] * v[2]) + matrix[3*4+1];
	REAL tz = (matrix[0*4+2] * v[0]) +  (matrix[1*4+2] * v[1]) + (matrix[2*4+2] * v[2]) + matrix[3*4+2];
	t[0] = tx;
	t[1] = ty;
	t[2] = tz;
  }
  else
  {
	t[0] = v[0];
	t[1] = v[1];
	t[2] = v[2];
  }
}

void  fm_rotate(const REAL matrix[16],const REAL v[3],REAL t[3]) // rotate and translate this point
{
  if ( matrix )
  {
	REAL tx = (matrix[0*4+0] * v[0]) +  (matrix[1*4+0] * v[1]) + (matrix[2*4+0] * v[2]);
	REAL ty = (matrix[0*4+1] * v[0]) +  (matrix[1*4+1] * v[1]) + (matrix[2*4+1] * v[2]);
	REAL tz = (matrix[0*4+2] * v[0]) +  (matrix[1*4+2] * v[1]) + (matrix[2*4+2] * v[2]);
	t[0] = tx;
	t[1] = ty;
	t[2] = tz;
  }
  else
  {
	t[0] = v[0];
	t[1] = v[1];
	t[2] = v[2];
  }
}


REAL fm_distance(const REAL *p1,const REAL *p2)
{
	REAL dx = p1[0] - p2[0];
	REAL dy = p1[1] - p2[1];
	REAL dz = p1[2] - p2[2];

	return (REAL)sqrt( dx*dx + dy*dy + dz *dz );
}

REAL fm_distanceSquared(const REAL *p1,const REAL *p2)
{
	REAL dx = p1[0] - p2[0];
	REAL dy = p1[1] - p2[1];
	REAL dz = p1[2] - p2[2];

	return dx*dx + dy*dy + dz *dz;
}


REAL fm_distanceSquaredXZ(const REAL *p1,const REAL *p2)
{
	REAL dx = p1[0] - p2[0];
	REAL dz = p1[2] - p2[2];

	return dx*dx +  dz *dz;
}


REAL fm_computePlane(const REAL *A,const REAL *B,const REAL *C,REAL *n) // returns D
{
	REAL vx = (B[0] - C[0]);
	REAL vy = (B[1] - C[1]);
	REAL vz = (B[2] - C[2]);

	REAL wx = (A[0] - B[0]);
	REAL wy = (A[1] - B[1]);
	REAL wz = (A[2] - B[2]);

	REAL vw_x = vy * wz - vz * wy;
	REAL vw_y = vz * wx - vx * wz;
	REAL vw_z = vx * wy - vy * wx;

	REAL mag = (REAL)sqrt((vw_x * vw_x) + (vw_y * vw_y) + (vw_z * vw_z));

	if ( mag < 0.000001f )
	{
		mag = 0;
	}
	else
	{
		mag = 1.0f/mag;
	}

	REAL x = vw_x * mag;
	REAL y = vw_y * mag;
	REAL z = vw_z * mag;


	REAL D = 0.0f - ((x*A[0])+(y*A[1])+(z*A[2]));

  n[0] = x;
  n[1] = y;
  n[2] = z;

	return D;
}

REAL fm_distToPlane(const REAL *plane,const REAL *p) // computes the distance of this point from the plane.
{
  return p[0]*plane[0]+p[1]*plane[1]+p[2]*plane[2]+plane[3];
}

REAL fm_dot(const REAL *p1,const REAL *p2)
{
  return p1[0]*p2[0]+p1[1]*p2[1]+p1[2]*p2[2];
}

void fm_cross(REAL *cross,const REAL *a,const REAL *b)
{
	cross[0] = a[1]*b[2] - a[2]*b[1];
	cross[1] = a[2]*b[0] - a[0]*b[2];
	cross[2] = a[0]*b[1] - a[1]*b[0];
}

REAL fm_computeNormalVector(REAL *n,const REAL *p1,const REAL *p2)
{
  n[0] = p2[0] - p1[0];
  n[1] = p2[1] - p1[1];
  n[2] = p2[2] - p1[2];
  return fm_normalize(n);
}

bool  fm_computeWindingOrder(const REAL *p1,const REAL *p2,const REAL *p3) // returns true if the triangle is clockwise.
{
  bool ret = false;

  REAL v1[3];
  REAL v2[3];

  fm_computeNormalVector(v1,p1,p2); // p2-p1 (as vector) and then normalized
  fm_computeNormalVector(v2,p1,p3); // p3-p1 (as vector) and then normalized

  REAL cross[3];

  fm_cross(cross, v1, v2 );
  REAL ref[3] = { 1, 0, 0 };

  REAL d = fm_dot( cross, ref );


  if ( d <= 0 )
	ret = false;
  else
	ret = true;

  return ret;
}

REAL fm_normalize(REAL *n) // normalize this vector
{
  REAL dist = (REAL)sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
  if ( dist > 0.0000001f )
  {
	REAL mag = 1.0f / dist;
	n[0]*=mag;
	n[1]*=mag;
	n[2]*=mag;
  }
  else
  {
	n[0] = 1;
	n[1] = 0;
	n[2] = 0;
  }

  return dist;
}


void  fm_matrixMultiply(const REAL *pA,const REAL *pB,REAL *pM)
{
#if 1

  REAL a = pA[0*4+0] * pB[0*4+0] + pA[0*4+1] * pB[1*4+0] + pA[0*4+2] * pB[2*4+0] + pA[0*4+3] * pB[3*4+0];
  REAL b = pA[0*4+0] * pB[0*4+1] + pA[0*4+1] * pB[1*4+1] + pA[0*4+2] * pB[2*4+1] + pA[0*4+3] * pB[3*4+1];
  REAL c = pA[0*4+0] * pB[0*4+2] + pA[0*4+1] * pB[1*4+2] + pA[0*4+2] * pB[2*4+2] + pA[0*4+3] * pB[3*4+2];
  REAL d = pA[0*4+0] * pB[0*4+3] + pA[0*4+1] * pB[1*4+3] + pA[0*4+2] * pB[2*4+3] + pA[0*4+3] * pB[3*4+3];

  REAL e = pA[1*4+0] * pB[0*4+0] + pA[1*4+1] * pB[1*4+0] + pA[1*4+2] * pB[2*4+0] + pA[1*4+3] * pB[3*4+0];
  REAL f = pA[1*4+0] * pB[0*4+1] + pA[1*4+1] * pB[1*4+1] + pA[1*4+2] * pB[2*4+1] + pA[1*4+3] * pB[3*4+1];
  REAL g = pA[1*4+0] * pB[0*4+2] + pA[1*4+1] * pB[1*4+2] + pA[1*4+2] * pB[2*4+2] + pA[1*4+3] * pB[3*4+2];
  REAL h = pA[1*4+0] * pB[0*4+3] + pA[1*4+1] * pB[1*4+3] + pA[1*4+2] * pB[2*4+3] + pA[1*4+3] * pB[3*4+3];

  REAL i = pA[2*4+0] * pB[0*4+0] + pA[2*4+1] * pB[1*4+0] + pA[2*4+2] * pB[2*4+0] + pA[2*4+3] * pB[3*4+0];
  REAL j = pA[2*4+0] * pB[0*4+1] + pA[2*4+1] * pB[1*4+1] + pA[2*4+2] * pB[2*4+1] + pA[2*4+3] * pB[3*4+1];
  REAL k = pA[2*4+0] * pB[0*4+2] + pA[2*4+1] * pB[1*4+2] + pA[2*4+2] * pB[2*4+2] + pA[2*4+3] * pB[3*4+2];
  REAL l = pA[2*4+0] * pB[0*4+3] + pA[2*4+1] * pB[1*4+3] + pA[2*4+2] * pB[2*4+3] + pA[2*4+3] * pB[3*4+3];

  REAL m = pA[3*4+0] * pB[0*4+0] + pA[3*4+1] * pB[1*4+0] + pA[3*4+2] * pB[2*4+0] + pA[3*4+3] * pB[3*4+0];
  REAL n = pA[3*4+0] * pB[0*4+1] + pA[3*4+1] * pB[1*4+1] + pA[3*4+2] * pB[2*4+1] + pA[3*4+3] * pB[3*4+1];
  REAL o = pA[3*4+0] * pB[0*4+2] + pA[3*4+1] * pB[1*4+2] + pA[3*4+2] * pB[2*4+2] + pA[3*4+3] * pB[3*4+2];
  REAL p = pA[3*4+0] * pB[0*4+3] + pA[3*4+1] * pB[1*4+3] + pA[3*4+2] * pB[2*4+3] + pA[3*4+3] * pB[3*4+3];

  pM[0] = a;
  pM[1] = b;
  pM[2] = c;
  pM[3] = d;

  pM[4] = e;
  pM[5] = f;
  pM[6] = g;
  pM[7] = h;

  pM[8] = i;
  pM[9] = j;
  pM[10] = k;
  pM[11] = l;

  pM[12] = m;
  pM[13] = n;
  pM[14] = o;
  pM[15] = p;


#else
	memset(pM, 0, sizeof(REAL)*16);
	for(int32_t i=0; i<4; i++ )
		for(int32_t j=0; j<4; j++ )
			for(int32_t k=0; k<4; k++ )
				pM[4*i+j] +=  pA[4*i+k] * pB[4*k+j];
#endif
}


void  fm_eulerToQuatDX(REAL x,REAL y,REAL z,REAL *quat) // convert euler angles to quaternion using the fucked up DirectX method
{
  REAL matrix[16];
  fm_eulerToMatrix(x,y,z,matrix);
  fm_matrixToQuat(matrix,quat);
}

// implementation copied from: http://blogs.msdn.com/mikepelton/archive/2004/10/29/249501.aspx
void  fm_eulerToMatrixDX(REAL x,REAL y,REAL z,REAL *matrix) // convert euler angles to quaternion using the fucked up DirectX method.
{
  fm_identity(matrix);
  matrix[0*4+0] = (REAL)(cos(z)*cos(y) + sin(z)*sin(x)*sin(y));
  matrix[0*4+1] = (REAL)(sin(z)*cos(x));
  matrix[0*4+2] = (REAL)(cos(z)*-sin(y) + sin(z)*sin(x)*cos(y));

  matrix[1*4+0] = (REAL)(-sin(z)*cos(y)+cos(z)*sin(x)*sin(y));
  matrix[1*4+1] = (REAL)(cos(z)*cos(x));
  matrix[1*4+2] = (REAL)(sin(z)*sin(y) +cos(z)*sin(x)*cos(y));

  matrix[2*4+0] = (REAL)(cos(x)*sin(y));
  matrix[2*4+1] = (REAL)(-sin(x));
  matrix[2*4+2] = (REAL)(cos(x)*cos(y));
}


void  fm_scale(REAL x,REAL y,REAL z,REAL *fscale) // apply scale to the matrix.
{
  fscale[0*4+0] = x;
  fscale[1*4+1] = y;
  fscale[2*4+2] = z;
}


void  fm_composeTransform(const REAL *position,const REAL *quat,const REAL *scale,REAL *matrix)
{
  fm_identity(matrix);
  fm_quatToMatrix(quat,matrix);

  if ( scale && ( scale[0] != 1 || scale[1] != 1 || scale[2] != 1 ) )
  {
	REAL work[16];
	memcpy(work,matrix,sizeof(REAL)*16);
	REAL mscale[16];
	fm_identity(mscale);
	fm_scale(scale[0],scale[1],scale[2],mscale);
	fm_matrixMultiply(work,mscale,matrix);
  }

  matrix[12] = position[0];
  matrix[13] = position[1];
  matrix[14] = position[2];
}


void  fm_setTranslation(const REAL *translation,REAL *matrix)
{
  matrix[12] = translation[0];
  matrix[13] = translation[1];
  matrix[14] = translation[2];
}

static REAL enorm0_3d ( REAL x0, REAL y0, REAL z0, REAL x1, REAL y1, REAL z1 )

/**********************************************************************/

/*
Purpose:

ENORM0_3D computes the Euclidean norm of (P1-P0) in 3D.

Modified:

18 April 1999

Author:

John Burkardt

Parameters:

Input, REAL X0, Y0, Z0, X1, Y1, Z1, the coordinates of the points 
P0 and P1.

Output, REAL ENORM0_3D, the Euclidean norm of (P1-P0).
*/
{
  REAL value;

  value = (REAL)sqrt (
	( x1 - x0 ) * ( x1 - x0 ) + 
	( y1 - y0 ) * ( y1 - y0 ) + 
	( z1 - z0 ) * ( z1 - z0 ) );

  return value;
}


static REAL triangle_area_3d ( REAL x1, REAL y1, REAL z1, REAL x2,REAL y2, REAL z2, REAL x3, REAL y3, REAL z3 )

						/**********************************************************************/

						/*
						Purpose:

						TRIANGLE_AREA_3D computes the area of a triangle in 3D.

						Modified:

						22 April 1999

						Author:

						John Burkardt

						Parameters:

						Input, REAL X1, Y1, Z1, X2, Y2, Z2, X3, Y3, Z3, the (X,Y,Z)
						coordinates of the corners of the triangle.

						Output, REAL TRIANGLE_AREA_3D, the area of the triangle.
						*/
{
  REAL a;
  REAL alpha;
  REAL area;
  REAL b;
  REAL base;
  REAL c;
  REAL dot;
  REAL height;
  /*
  Find the projection of (P3-P1) onto (P2-P1).
  */
  dot = 
	( x2 - x1 ) * ( x3 - x1 ) +
	( y2 - y1 ) * ( y3 - y1 ) +
	( z2 - z1 ) * ( z3 - z1 );

  base = enorm0_3d ( x1, y1, z1, x2, y2, z2 );
  /*
  The height of the triangle is the length of (P3-P1) after its
  projection onto (P2-P1) has been subtracted.
  */
  if ( base == 0.0 ) {

	height = 0.0;

  }
  else {

	alpha = dot / ( base * base );

	a = x3 - x1 - alpha * ( x2 - x1 );
	b = y3 - y1 - alpha * ( y2 - y1 );
	c = z3 - z1 - alpha * ( z2 - z1 );

	height = (REAL)sqrt ( a * a + b * b + c * c );

  }

  area = 0.5f * base * height;

  return area;
}


REAL fm_computeArea(const REAL *p1,const REAL *p2,const REAL *p3)
{
  REAL ret = 0;

  ret = triangle_area_3d(p1[0],p1[1],p1[2],p2[0],p2[1],p2[2],p3[0],p3[1],p3[2]);

  return ret;
}


void  fm_lerp(const REAL *p1,const REAL *p2,REAL *dest,REAL lerpValue)
{
  dest[0] = ((p2[0] - p1[0])*lerpValue) + p1[0];
  dest[1] = ((p2[1] - p1[1])*lerpValue) + p1[1];
  dest[2] = ((p2[2] - p1[2])*lerpValue) + p1[2];
}

bool fm_pointTestXZ(const REAL *p,const REAL *i,const REAL *j)
{
  bool ret = false;

  if (((( i[2] <= p[2] ) && ( p[2]  < j[2] )) || (( j[2] <= p[2] ) && ( p[2]  < i[2] ))) && ( p[0] < (j[0] - i[0]) * (p[2] - i[2]) / (j[2] - i[2]) + i[0]))
	ret = true;

  return ret;
};


bool  fm_insideTriangleXZ(const REAL *p,const REAL *p1,const REAL *p2,const REAL *p3)
{
  bool ret = false;

  int32_t c = 0;
  if ( fm_pointTestXZ(p,p1,p2) ) c = !c;
  if ( fm_pointTestXZ(p,p2,p3) ) c = !c;
  if ( fm_pointTestXZ(p,p3,p1) ) c = !c;
  if ( c ) ret = true;

  return ret;
}

bool  fm_insideAABB(const REAL *pos,const REAL *bmin,const REAL *bmax)
{
  bool ret = false;

  if ( pos[0] >= bmin[0] && pos[0] <= bmax[0] &&
	   pos[1] >= bmin[1] && pos[1] <= bmax[1] &&
	   pos[2] >= bmin[2] && pos[2] <= bmax[2] )
	ret = true;

  return ret;
}


uint32_t fm_clipTestPoint(const REAL *bmin,const REAL *bmax,const REAL *pos)
{
  uint32_t ret = 0;

  if ( pos[0] < bmin[0] )
	ret|=FMCS_XMIN;
  else if ( pos[0] > bmax[0] )
	ret|=FMCS_XMAX;

  if ( pos[1] < bmin[1] )
	ret|=FMCS_YMIN;
  else if ( pos[1] > bmax[1] )
	ret|=FMCS_YMAX;

  if ( pos[2] < bmin[2] )
	ret|=FMCS_ZMIN;
  else if ( pos[2] > bmax[2] )
	ret|=FMCS_ZMAX;

  return ret;
}

uint32_t fm_clipTestPointXZ(const REAL *bmin,const REAL *bmax,const REAL *pos) // only tests X and Z, not Y
{
  uint32_t ret = 0;

  if ( pos[0] < bmin[0] )
	ret|=FMCS_XMIN;
  else if ( pos[0] > bmax[0] )
	ret|=FMCS_XMAX;

  if ( pos[2] < bmin[2] )
	ret|=FMCS_ZMIN;
  else if ( pos[2] > bmax[2] )
	ret|=FMCS_ZMAX;

  return ret;
}

uint32_t fm_clipTestAABB(const REAL *bmin,const REAL *bmax,const REAL *p1,const REAL *p2,const REAL *p3,uint32_t &andCode)
{
  uint32_t orCode = 0;

  andCode = FMCS_XMIN | FMCS_XMAX | FMCS_YMIN | FMCS_YMAX | FMCS_ZMIN | FMCS_ZMAX;

  uint32_t c = fm_clipTestPoint(bmin,bmax,p1);
  orCode|=c;
  andCode&=c;

  c = fm_clipTestPoint(bmin,bmax,p2);
  orCode|=c;
  andCode&=c;

  c = fm_clipTestPoint(bmin,bmax,p3);
  orCode|=c;
  andCode&=c;

  return orCode;
}

bool intersect(const REAL *si,const REAL *ei,const REAL *bmin,const REAL *bmax,REAL *time)
{
  REAL st,et,fst = 0,fet = 1;

  for (int32_t i = 0; i < 3; i++)
  {
	if (*si < *ei)
	{
	  if (*si > *bmax || *ei < *bmin)
		return false;
	  REAL di = *ei - *si;
	  st = (*si < *bmin)? (*bmin - *si) / di: 0;
	  et = (*ei > *bmax)? (*bmax - *si) / di: 1;
	}
	else
	{
	  if (*ei > *bmax || *si < *bmin)
		return false;
	  REAL di = *ei - *si;
	  st = (*si > *bmax)? (*bmax - *si) / di: 0;
	  et = (*ei < *bmin)? (*bmin - *si) / di: 1;
	}

	if (st > fst) fst = st;
	if (et < fet) fet = et;
	if (fet < fst)
	  return false;
	bmin++; bmax++;
	si++; ei++;
  }

  *time = fst;
  return true;
}



bool fm_lineTestAABB(const REAL *p1,const REAL *p2,const REAL *bmin,const REAL *bmax,REAL &time)
{
  bool sect = intersect(p1,p2,bmin,bmax,&time);
  return sect;
}


bool fm_lineTestAABBXZ(const REAL *p1,const REAL *p2,const REAL *bmin,const REAL *bmax,REAL &time)
{
  REAL _bmin[3];
  REAL _bmax[3];

  _bmin[0] = bmin[0];
  _bmin[1] = -1e9;
  _bmin[2] = bmin[2];

  _bmax[0] = bmax[0];
  _bmax[1] = 1e9;
  _bmax[2] = bmax[2];

  bool sect = intersect(p1,p2,_bmin,_bmax,&time);

  return sect;
}

void  fm_minmax(const REAL *p,REAL *bmin,REAL *bmax) // accmulate to a min-max value
{

  if ( p[0] < bmin[0] ) bmin[0] = p[0];
  if ( p[1] < bmin[1] ) bmin[1] = p[1];
  if ( p[2] < bmin[2] ) bmin[2] = p[2];

  if ( p[0] > bmax[0] ) bmax[0] = p[0];
  if ( p[1] > bmax[1] ) bmax[1] = p[1];
  if ( p[2] > bmax[2] ) bmax[2] = p[2];

}

REAL fm_solveX(const REAL *plane,REAL y,REAL z) // solve for X given this plane equation and the other two components.
{
  REAL x = (y*plane[1]+z*plane[2]+plane[3]) / -plane[0];
  return x;
}

REAL fm_solveY(const REAL *plane,REAL x,REAL z) // solve for Y given this plane equation and the other two components.
{
  REAL y = (x*plane[0]+z*plane[2]+plane[3]) / -plane[1];
  return y;
}


REAL fm_solveZ(const REAL *plane,REAL x,REAL y) // solve for Y given this plane equation and the other two components.
{
  REAL z = (x*plane[0]+y*plane[1]+plane[3]) / -plane[2];
  return z;
}


void  fm_getAABBCenter(const REAL *bmin,const REAL *bmax,REAL *center)
{
  center[0] = (bmax[0]-bmin[0])*0.5f+bmin[0];
  center[1] = (bmax[1]-bmin[1])*0.5f+bmin[1];
  center[2] = (bmax[2]-bmin[2])*0.5f+bmin[2];
}

FM_Axis fm_getDominantAxis(const REAL normal[3])
{
  FM_Axis ret = FM_XAXIS;

  REAL x = (REAL)fabs(normal[0]);
  REAL y = (REAL)fabs(normal[1]);
  REAL z = (REAL)fabs(normal[2]);

  if ( y > x && y > z )
	ret = FM_YAXIS;
  else if ( z > x && z > y )
	ret = FM_ZAXIS;

  return ret;
}


bool fm_lineSphereIntersect(const REAL *center,REAL radius,const REAL *p1,const REAL *p2,REAL *intersect)
{
  bool ret = false;

  REAL dir[3];

  dir[0] = p2[0]-p1[0];
  dir[1] = p2[1]-p1[1];
  dir[2] = p2[2]-p1[2];

  REAL distance = (REAL)sqrt( dir[0]*dir[0]+dir[1]*dir[1]+dir[2]*dir[2]);

  if ( distance > 0 )
  {
	REAL recip = 1.0f / distance;
	dir[0]*=recip;
	dir[1]*=recip;
	dir[2]*=recip;
	ret = fm_raySphereIntersect(center,radius,p1,dir,distance,intersect);
  }
  else
  {
	dir[0] = center[0]-p1[0];
	dir[1] = center[1]-p1[1];
	dir[2] = center[2]-p1[2];
	REAL d2 = dir[0]*dir[0]+dir[1]*dir[1]+dir[2]*dir[2];
	REAL r2 = radius*radius;
	if ( d2 < r2 )
	{
	  ret = true;
	  if ( intersect )
	  {
		intersect[0] = p1[0];
		intersect[1] = p1[1];
		intersect[2] = p1[2];
	  }
	}
  }
  return ret;
}

#define DOT(p1,p2) (p1[0]*p2[0]+p1[1]*p2[1]+p1[2]*p2[2])

bool fm_raySphereIntersect(const REAL *center,REAL radius,const REAL *pos,const REAL *dir,REAL distance,REAL *intersect)
{
  bool ret = false;

  REAL E0[3];

  E0[0] = center[0] - pos[0];
  E0[1] = center[1] - pos[1];
  E0[2] = center[2] - pos[2];

  REAL V[3];

  V[0]  = dir[0];
  V[1]  = dir[1];
  V[2]  = dir[2];


  REAL dist2   = E0[0]*E0[0] + E0[1]*E0[1] + E0[2] * E0[2];
  REAL radius2 = radius*radius; // radius squared..

  // Bug Fix For Gem, if origin is *inside* the sphere, invert the
  // direction vector so that we get a valid intersection location.
  if ( dist2 < radius2 )
  {
	V[0]*=-1;
	V[1]*=-1;
	V[2]*=-1;
  }


	REAL v = DOT(E0,V);

	REAL disc = radius2 - (dist2 - v*v);

	if (disc > 0.0f)
	{
		if ( intersect )
		{
		  REAL d = (REAL)sqrt(disc);
	  REAL diff = v-d;
	  if ( diff < distance )
	  {
		intersect[0] = pos[0]+V[0]*diff;
		intersect[1] = pos[1]+V[1]*diff;
		intersect[2] = pos[2]+V[2]*diff;
		ret = true;
	  }
	}
	}

	return ret;
}


void fm_catmullRom(REAL *out_vector,const REAL *p1,const REAL *p2,const REAL *p3,const REAL *p4, const REAL s)
{
  REAL s_squared = s * s;
  REAL s_cubed = s_squared * s;

  REAL coefficient_p1 = -s_cubed + 2*s_squared - s;
  REAL coefficient_p2 = 3 * s_cubed - 5 * s_squared + 2;
  REAL coefficient_p3 = -3 * s_cubed +4 * s_squared + s;
  REAL coefficient_p4 = s_cubed - s_squared;

  out_vector[0] = (coefficient_p1 * p1[0] + coefficient_p2 * p2[0] + coefficient_p3 * p3[0] + coefficient_p4 * p4[0])*0.5f;
  out_vector[1] = (coefficient_p1 * p1[1] + coefficient_p2 * p2[1] + coefficient_p3 * p3[1] + coefficient_p4 * p4[1])*0.5f;
  out_vector[2] = (coefficient_p1 * p1[2] + coefficient_p2 * p2[2] + coefficient_p3 * p3[2] + coefficient_p4 * p4[2])*0.5f;
}

bool fm_intersectAABB(const REAL *bmin1,const REAL *bmax1,const REAL *bmin2,const REAL *bmax2)
{
  if ((bmin1[0] > bmax2[0]) || (bmin2[0] > bmax1[0])) return false;
  if ((bmin1[1] > bmax2[1]) || (bmin2[1] > bmax1[1])) return false;
  if ((bmin1[2] > bmax2[2]) || (bmin2[2] > bmax1[2])) return false;
  return true;

}

bool  fm_insideAABB(const REAL *obmin,const REAL *obmax,const REAL *tbmin,const REAL *tbmax) // test if bounding box tbmin/tmbax is fully inside obmin/obmax
{
  bool ret = false;

  if ( tbmax[0] <= obmax[0] &&
	   tbmax[1] <= obmax[1] &&
	   tbmax[2] <= obmax[2] &&
	   tbmin[0] >= obmin[0] &&
	   tbmin[1] >= obmin[1] &&
	   tbmin[2] >= obmin[2] ) ret = true;

  return ret;
}


// Reference, from Stan Melax in Game Gems I
//  Quaternion q;
//  vector3 c = CrossProduct(v0,v1);
//  REAL   d = DotProduct(v0,v1);
//  REAL   s = (REAL)sqrt((1+d)*2);
//  q.x = c.x / s;
//  q.y = c.y / s;
//  q.z = c.z / s;
//  q.w = s /2.0f;
//  return q;
void fm_rotationArc(const REAL *v0,const REAL *v1,REAL *quat)
{
  REAL cross[3];

  fm_cross(cross,v0,v1);
  REAL d = fm_dot(v0,v1);

  if( d<= -0.99999f ) // 180 about x axis
  {
	  if ( fabsf((float)v0[0]) < 0.1f )
	  {
		  quat[0] = 0;
		  quat[1] = v0[2];
		  quat[2] = -v0[1];
		  quat[3] = 0;
	  }
	  else
	  {
		  quat[0] = v0[1];
		  quat[1] = -v0[0];
		  quat[2] = 0;
		  quat[3] = 0;
	  }
	  REAL magnitudeSquared = quat[0]*quat[0] + quat[1]*quat[1] + quat[2]*quat[2] + quat[3]*quat[3];
	  REAL magnitude = sqrtf((float)magnitudeSquared);
	  REAL recip = 1.0f / magnitude;
	  quat[0]*=recip;
	  quat[1]*=recip;
	  quat[2]*=recip;
	  quat[3]*=recip;
  }
  else
  {
	  REAL s = (REAL)sqrt((1+d)*2);
	  REAL recip = 1.0f / s;

	  quat[0] = cross[0] * recip;
	  quat[1] = cross[1] * recip;
	  quat[2] = cross[2] * recip;
	  quat[3] = s * 0.5f;
  }
}


REAL fm_distancePointLineSegment(const REAL *Point,const REAL *LineStart,const REAL *LineEnd,REAL *intersection,LineSegmentType &type,REAL epsilon)
{
  REAL ret;

  REAL LineMag = fm_distance( LineEnd, LineStart );

  if ( LineMag > 0 )
  {
	REAL U = ( ( ( Point[0] - LineStart[0] ) * ( LineEnd[0] - LineStart[0] ) ) + ( ( Point[1] - LineStart[1] ) * ( LineEnd[1] - LineStart[1] ) ) + ( ( Point[2] - LineStart[2] ) * ( LineEnd[2] - LineStart[2] ) ) ) / ( LineMag * LineMag );
	if( U < 0.0f || U > 1.0f )
	{
	  REAL d1 = fm_distanceSquared(Point,LineStart);
	  REAL d2 = fm_distanceSquared(Point,LineEnd);
	  if ( d1 <= d2 )
	  {
		ret = (REAL)sqrt(d1);
		intersection[0] = LineStart[0];
		intersection[1] = LineStart[1];
		intersection[2] = LineStart[2];
		type = LS_START;
	  }
	  else
	  {
		ret = (REAL)sqrt(d2);
		intersection[0] = LineEnd[0];
		intersection[1] = LineEnd[1];
		intersection[2] = LineEnd[2];
		type = LS_END;
	  }
	}
	else
	{
	  intersection[0] = LineStart[0] + U * ( LineEnd[0] - LineStart[0] );
	  intersection[1] = LineStart[1] + U * ( LineEnd[1] - LineStart[1] );
	  intersection[2] = LineStart[2] + U * ( LineEnd[2] - LineStart[2] );

	  ret = fm_distance(Point,intersection);

	  REAL d1 = fm_distanceSquared(intersection,LineStart);
	  REAL d2 = fm_distanceSquared(intersection,LineEnd);
	  REAL mag = (epsilon*2)*(epsilon*2);

	  if ( d1 < mag ) // if less than 1/100th the total distance, treat is as the 'start'
	  {
		type = LS_START;
	  }
	  else if ( d2 < mag )
	  {
		type = LS_END;
	  }
	  else
	  {
		type = LS_MIDDLE;
	  }

	}
  }
  else
  {
	ret = LineMag;
	intersection[0] = LineEnd[0];
	intersection[1] = LineEnd[1];
	intersection[2] = LineEnd[2];
	type = LS_END;
  }

  return ret;
}


#ifndef BEST_FIT_PLANE_H

#define BEST_FIT_PLANE_H

template <class Type> class Eigen
{
public:


  void DecrSortEigenStuff(void)
  {
	Tridiagonal(); //diagonalize the matrix.
	QLAlgorithm(); //
	DecreasingSort();
	GuaranteeRotation();
  }

  void Tridiagonal(void)
  {
	Type fM00 = mElement[0][0];
	Type fM01 = mElement[0][1];
	Type fM02 = mElement[0][2];
	Type fM11 = mElement[1][1];
	Type fM12 = mElement[1][2];
	Type fM22 = mElement[2][2];

	m_afDiag[0] = fM00;
	m_afSubd[2] = 0;
	if (fM02 != (Type)0.0)
	{
	  Type fLength = (REAL)sqrt(fM01*fM01+fM02*fM02);
	  Type fInvLength = ((Type)1.0)/fLength;
	  fM01 *= fInvLength;
	  fM02 *= fInvLength;
	  Type fQ = ((Type)2.0)*fM01*fM12+fM02*(fM22-fM11);
	  m_afDiag[1] = fM11+fM02*fQ;
	  m_afDiag[2] = fM22-fM02*fQ;
	  m_afSubd[0] = fLength;
	  m_afSubd[1] = fM12-fM01*fQ;
	  mElement[0][0] = (Type)1.0;
	  mElement[0][1] = (Type)0.0;
	  mElement[0][2] = (Type)0.0;
	  mElement[1][0] = (Type)0.0;
	  mElement[1][1] = fM01;
	  mElement[1][2] = fM02;
	  mElement[2][0] = (Type)0.0;
	  mElement[2][1] = fM02;
	  mElement[2][2] = -fM01;
	  m_bIsRotation = false;
	}
	else
	{
	  m_afDiag[1] = fM11;
	  m_afDiag[2] = fM22;
	  m_afSubd[0] = fM01;
	  m_afSubd[1] = fM12;
	  mElement[0][0] = (Type)1.0;
	  mElement[0][1] = (Type)0.0;
	  mElement[0][2] = (Type)0.0;
	  mElement[1][0] = (Type)0.0;
	  mElement[1][1] = (Type)1.0;
	  mElement[1][2] = (Type)0.0;
	  mElement[2][0] = (Type)0.0;
	  mElement[2][1] = (Type)0.0;
	  mElement[2][2] = (Type)1.0;
	  m_bIsRotation = true;
	}
  }

  bool QLAlgorithm(void)
  {
	const int32_t iMaxIter = 32;

	for (int32_t i0 = 0; i0 <3; i0++)
	{
	  int32_t i1;
	  for (i1 = 0; i1 < iMaxIter; i1++)
	  {
		int32_t i2;
		for (i2 = i0; i2 <= (3-2); i2++)
		{
		  Type fTmp = fabs(m_afDiag[i2]) + fabs(m_afDiag[i2+1]);
		  if ( fabs(m_afSubd[i2]) + fTmp == fTmp )
			break;
		}
		if (i2 == i0)
		{
		  break;
		}

		Type fG = (m_afDiag[i0+1] - m_afDiag[i0])/(((Type)2.0) * m_afSubd[i0]);
		Type fR = (REAL)sqrt(fG*fG+(Type)1.0);
		if (fG < (Type)0.0)
		{
		  fG = m_afDiag[i2]-m_afDiag[i0]+m_afSubd[i0]/(fG-fR);
		}
		else
		{
		  fG = m_afDiag[i2]-m_afDiag[i0]+m_afSubd[i0]/(fG+fR);
		}
		Type fSin = (Type)1.0, fCos = (Type)1.0, fP = (Type)0.0;
		for (int32_t i3 = i2-1; i3 >= i0; i3--)
		{
		  Type fF = fSin*m_afSubd[i3];
		  Type fB = fCos*m_afSubd[i3];
		  if (fabs(fF) >= fabs(fG))
		  {
			fCos = fG/fF;
			fR = (REAL)sqrt(fCos*fCos+(Type)1.0);
			m_afSubd[i3+1] = fF*fR;
			fSin = ((Type)1.0)/fR;
			fCos *= fSin;
		  }
		  else
		  {
			fSin = fF/fG;
			fR = (REAL)sqrt(fSin*fSin+(Type)1.0);
			m_afSubd[i3+1] = fG*fR;
			fCos = ((Type)1.0)/fR;
			fSin *= fCos;
		  }
		  fG = m_afDiag[i3+1]-fP;
		  fR = (m_afDiag[i3]-fG)*fSin+((Type)2.0)*fB*fCos;
		  fP = fSin*fR;
		  m_afDiag[i3+1] = fG+fP;
		  fG = fCos*fR-fB;
		  for (int32_t i4 = 0; i4 < 3; i4++)
		  {
			fF = mElement[i4][i3+1];
			mElement[i4][i3+1] = fSin*mElement[i4][i3]+fCos*fF;
			mElement[i4][i3] = fCos*mElement[i4][i3]-fSin*fF;
		  }
		}
		m_afDiag[i0] -= fP;
		m_afSubd[i0] = fG;
		m_afSubd[i2] = (Type)0.0;
	  }
	  if (i1 == iMaxIter)
	  {
		return false;
	  }
	}
	return true;
  }

  void DecreasingSort(void)
  {
	//sort eigenvalues in decreasing order, e[0] >= ... >= e[iSize-1]
	for (int32_t i0 = 0, i1; i0 <= 3-2; i0++)
	{
	  // locate maximum eigenvalue
	  i1 = i0;
	  Type fMax = m_afDiag[i1];
	  int32_t i2;
	  for (i2 = i0+1; i2 < 3; i2++)
	  {
		if (m_afDiag[i2] > fMax)
		{
		  i1 = i2;
		  fMax = m_afDiag[i1];
		}
	  }

	  if (i1 != i0)
	  {
		// swap eigenvalues
		m_afDiag[i1] = m_afDiag[i0];
		m_afDiag[i0] = fMax;
		// swap eigenvectors
		for (i2 = 0; i2 < 3; i2++)
		{
		  Type fTmp = mElement[i2][i0];
		  mElement[i2][i0] = mElement[i2][i1];
		  mElement[i2][i1] = fTmp;
		  m_bIsRotation = !m_bIsRotation;
		}
	  }
	}
  }


  void GuaranteeRotation(void)
  {
	if (!m_bIsRotation)
	{
	  // change sign on the first column
	  for (int32_t iRow = 0; iRow <3; iRow++)
	  {
		mElement[iRow][0] = -mElement[iRow][0];
	  }
	}
  }

  Type mElement[3][3];
  Type m_afDiag[3];
  Type m_afSubd[3];
  bool m_bIsRotation;
};

#endif

bool fm_computeBestFitPlane(uint32_t vcount,
					 const REAL *points,
					 uint32_t vstride,
					 const REAL *weights,
					 uint32_t wstride,
					 REAL *plane,
					REAL *center)
{
  bool ret = false;

  REAL kOrigin[3] = { 0, 0, 0 };

  REAL wtotal = 0;

  {
	const char *source  = (const char *) points;
	const char *wsource = (const char *) weights;

	for (uint32_t i=0; i<vcount; i++)
	{

	  const REAL *p = (const REAL *) source;

	  REAL w = 1;

	  if ( wsource )
	  {
		const REAL *ws = (const REAL *) wsource;
		w = *ws; //
		wsource+=wstride;
	  }

	  kOrigin[0]+=p[0]*w;
	  kOrigin[1]+=p[1]*w;
	  kOrigin[2]+=p[2]*w;

	  wtotal+=w;

	  source+=vstride;
	}
  }

  REAL recip = 1.0f / wtotal; // reciprocol of total weighting

  kOrigin[0]*=recip;
  kOrigin[1]*=recip;
  kOrigin[2]*=recip;

  center[0] = kOrigin[0];
  center[1] = kOrigin[1];
  center[2] = kOrigin[2];


  REAL fSumXX=0;
  REAL fSumXY=0;
  REAL fSumXZ=0;

  REAL fSumYY=0;
  REAL fSumYZ=0;
  REAL fSumZZ=0;


  {
	const char *source  = (const char *) points;
	const char *wsource = (const char *) weights;

	for (uint32_t i=0; i<vcount; i++)
	{

	  const REAL *p = (const REAL *) source;

	  REAL w = 1;

	  if ( wsource )
	  {
		const REAL *ws = (const REAL *) wsource;
		w = *ws; //
		wsource+=wstride;
	  }

	  REAL kDiff[3];

	  kDiff[0] = w*(p[0] - kOrigin[0]); // apply vertex weighting!
	  kDiff[1] = w*(p[1] - kOrigin[1]);
	  kDiff[2] = w*(p[2] - kOrigin[2]);

	  fSumXX+= kDiff[0] * kDiff[0]; // sume of the squares of the differences.
	  fSumXY+= kDiff[0] * kDiff[1]; // sume of the squares of the differences.
	  fSumXZ+= kDiff[0] * kDiff[2]; // sume of the squares of the differences.

	  fSumYY+= kDiff[1] * kDiff[1];
	  fSumYZ+= kDiff[1] * kDiff[2];
	  fSumZZ+= kDiff[2] * kDiff[2];


	  source+=vstride;
	}
  }

  fSumXX *= recip;
  fSumXY *= recip;
  fSumXZ *= recip;
  fSumYY *= recip;
  fSumYZ *= recip;
  fSumZZ *= recip;

  // setup the eigensolver
  Eigen<REAL> kES;

  kES.mElement[0][0] = fSumXX;
  kES.mElement[0][1] = fSumXY;
  kES.mElement[0][2] = fSumXZ;

  kES.mElement[1][0] = fSumXY;
  kES.mElement[1][1] = fSumYY;
  kES.mElement[1][2] = fSumYZ;

  kES.mElement[2][0] = fSumXZ;
  kES.mElement[2][1] = fSumYZ;
  kES.mElement[2][2] = fSumZZ;

  // compute eigenstuff, smallest eigenvalue is in last position
  kES.DecrSortEigenStuff();

  REAL kNormal[3];

  kNormal[0] = kES.mElement[0][2];
  kNormal[1] = kES.mElement[1][2];
  kNormal[2] = kES.mElement[2][2];

  // the minimum energy
  plane[0] = kNormal[0];
  plane[1] = kNormal[1];
  plane[2] = kNormal[2];

  plane[3] = 0 - fm_dot(kNormal,kOrigin);

  ret = true;

  return ret;
}


bool fm_colinear(const REAL a1[3],const REAL a2[3],const REAL b1[3],const REAL b2[3],REAL epsilon)  // true if these two line segments are co-linear.
{
  bool ret = false;

  REAL dir1[3];
  REAL dir2[3];

  dir1[0] = (a2[0] - a1[0]);
  dir1[1] = (a2[1] - a1[1]);
  dir1[2] = (a2[2] - a1[2]);

  dir2[0] = (b2[0]-a1[0]) - (b1[0]-a1[0]);
  dir2[1] = (b2[1]-a1[1]) - (b1[1]-a1[1]);
  dir2[2] = (b2[2]-a2[2]) - (b1[2]-a2[2]);

  fm_normalize(dir1);
  fm_normalize(dir2);

  REAL dot = fm_dot(dir1,dir2);

  if ( dot >= epsilon )
  {
	ret = true;
  }


  return ret;
}

bool fm_colinear(const REAL *p1,const REAL *p2,const REAL *p3,REAL epsilon)
{
  bool ret = false;

  REAL dir1[3];
  REAL dir2[3];

  dir1[0] = p2[0] - p1[0];
  dir1[1] = p2[1] - p1[1];
  dir1[2] = p2[2] - p1[2];

  dir2[0] = p3[0] - p2[0];
  dir2[1] = p3[1] - p2[1];
  dir2[2] = p3[2] - p2[2];

  fm_normalize(dir1);
  fm_normalize(dir2);

  REAL dot = fm_dot(dir1,dir2);

  if ( dot >= epsilon )
  {
	ret = true;
  }


  return ret;
}

void  fm_initMinMax(const REAL *p,REAL *bmin,REAL *bmax)
{
  bmax[0] = bmin[0] = p[0];
  bmax[1] = bmin[1] = p[1];
  bmax[2] = bmin[2] = p[2];
}

IntersectResult fm_intersectLineSegments2d(const REAL *a1,const REAL *a2,const REAL *b1,const REAL *b2,REAL *intersection)
{
  IntersectResult ret;

  REAL denom  = ((b2[1] - b1[1])*(a2[0] - a1[0])) - ((b2[0] - b1[0])*(a2[1] - a1[1]));
  REAL nume_a = ((b2[0] - b1[0])*(a1[1] - b1[1])) - ((b2[1] - b1[1])*(a1[0] - b1[0]));
  REAL nume_b = ((a2[0] - a1[0])*(a1[1] - b1[1])) - ((a2[1] - a1[1])*(a1[0] - b1[0]));
  if (denom == 0 )
  {
	if(nume_a == 0 && nume_b == 0)
	{
	  ret = IR_COINCIDENT;
	}
	else
	{
	  ret = IR_PARALLEL;
	}
  }
  else
  {

	REAL recip = 1 / denom;
	REAL ua = nume_a * recip;
	REAL ub = nume_b * recip;

	if(ua >= 0 && ua <= 1 && ub >= 0 && ub <= 1 )
	{
	  // Get the intersection point.
	  intersection[0] = a1[0] + ua*(a2[0] - a1[0]);
	  intersection[1] = a1[1] + ua*(a2[1] - a1[1]);
	  ret = IR_DO_INTERSECT;
	}
	else
	{
	  ret = IR_DONT_INTERSECT;
	}
  }
  return ret;
}

IntersectResult fm_intersectLineSegments2dTime(const REAL *a1,const REAL *a2,const REAL *b1,const REAL *b2,REAL &t1,REAL &t2)
{
  IntersectResult ret;

  REAL denom  = ((b2[1] - b1[1])*(a2[0] - a1[0])) - ((b2[0] - b1[0])*(a2[1] - a1[1]));
  REAL nume_a = ((b2[0] - b1[0])*(a1[1] - b1[1])) - ((b2[1] - b1[1])*(a1[0] - b1[0]));
  REAL nume_b = ((a2[0] - a1[0])*(a1[1] - b1[1])) - ((a2[1] - a1[1])*(a1[0] - b1[0]));
  if (denom == 0 )
  {
	if(nume_a == 0 && nume_b == 0)
	{
	  ret = IR_COINCIDENT;
	}
	else
	{
	  ret = IR_PARALLEL;
	}
  }
  else
  {

	REAL recip = 1 / denom;
	REAL ua = nume_a * recip;
	REAL ub = nume_b * recip;

	if(ua >= 0 && ua <= 1 && ub >= 0 && ub <= 1 )
	{
	  t1 = ua;
	  t2 = ub;
	  ret = IR_DO_INTERSECT;
	}
	else
	{
	  ret = IR_DONT_INTERSECT;
	}
  }
  return ret;
}

//**** Plane Triangle Intersection





// assumes that the points are on opposite sides of the plane!
bool fm_intersectPointPlane(const REAL *p1,const REAL *p2,REAL *split,const REAL *plane)
{

  REAL dp1 = fm_distToPlane(plane,p1);
  REAL dp2 = fm_distToPlane(plane, p2);
  if (dp1 <= 0 && dp2 <= 0)
  {
	  return false;
  }
  if (dp1 >= 0 && dp2 >= 0)
  {
	  return false;
  }

  REAL dir[3];

  dir[0] = p2[0] - p1[0];
  dir[1] = p2[1] - p1[1];
  dir[2] = p2[2] - p1[2];

  REAL dot1 = dir[0]*plane[0] + dir[1]*plane[1] + dir[2]*plane[2];
  REAL dot2 = dp1 - plane[3];

  REAL    t = -(plane[3] + dot2 ) / dot1;

  split[0] = (dir[0]*t)+p1[0];
  split[1] = (dir[1]*t)+p1[1];
  split[2] = (dir[2]*t)+p1[2];

  return true;
}

PlaneTriResult fm_getSidePlane(const REAL *p,const REAL *plane,REAL epsilon)
{
  PlaneTriResult ret = PTR_ON_PLANE;

  REAL d = fm_distToPlane(plane,p);

  if ( d < -epsilon || d > epsilon )
  {
	if ( d > 0 )
		ret =  PTR_FRONT; // it is 'in front' within the provided epsilon value.
	else
	  ret = PTR_BACK;
  }

  return ret;
}



#ifndef PLANE_TRIANGLE_INTERSECTION_H

#define PLANE_TRIANGLE_INTERSECTION_H

#define MAXPTS 256

template <class Type> class point
{
public:

  void set(const Type *p)
  {
	x = p[0];
	y = p[1];
	z = p[2];
  }

  Type x;
  Type y;
  Type z;
};

template <class Type> class plane
{
public:
  plane(const Type *p)
  {
	normal.x = p[0];
	normal.y = p[1];
	normal.z = p[2];
	D        = p[3];
  }

  Type Classify_Point(const point<Type> &p)
  {
	return p.x*normal.x + p.y*normal.y + p.z*normal.z + D;
  }

  point<Type> normal;
  Type  D;
};

template <class Type> class polygon
{
public:
  polygon(void)
  {
	mVcount = 0;
  }

  polygon(const Type *p1,const Type *p2,const Type *p3)
  {
	mVcount = 3;
	mVertices[0].set(p1);
	mVertices[1].set(p2);
	mVertices[2].set(p3);
  }


  int32_t NumVertices(void) const { return mVcount; };

  const point<Type>& Vertex(int32_t index)
  {
	if ( index < 0 ) index+=mVcount;
	return mVertices[index];
  };


  void set(const point<Type> *pts,int32_t count)
  {
	for (int32_t i=0; i<count; i++)
	{
	  mVertices[i] = pts[i];
	}
	mVcount = count;
  }


  void Split_Polygon(polygon<Type> *poly,plane<Type> *part, polygon<Type> &front, polygon<Type> &back)
  {
	int32_t   count = poly->NumVertices ();
	int32_t   out_c = 0, in_c = 0;
	point<Type> ptA, ptB,outpts[MAXPTS],inpts[MAXPTS];
	Type sideA, sideB;
	ptA = poly->Vertex (count - 1);
	sideA = part->Classify_Point (ptA);
	for (int32_t i = -1; ++i < count;)
	{
	  ptB = poly->Vertex(i);
	  sideB = part->Classify_Point(ptB);
	  if (sideB > 0)
	  {
		if (sideA < 0)
		{
			  point<Type> v;
		  fm_intersectPointPlane(&ptB.x, &ptA.x, &v.x, &part->normal.x );
		  outpts[out_c++] = inpts[in_c++] = v;
		}
		outpts[out_c++] = ptB;
	  }
	  else if (sideB < 0)
	  {
		if (sideA > 0)
		{
		  point<Type> v;
		  fm_intersectPointPlane(&ptB.x, &ptA.x, &v.x, &part->normal.x );
		  outpts[out_c++] = inpts[in_c++] = v;
		}
		inpts[in_c++] = ptB;
	  }
	  else
		 outpts[out_c++] = inpts[in_c++] = ptB;
	  ptA = ptB;
	  sideA = sideB;
	}

	front.set(&outpts[0], out_c);
	back.set(&inpts[0], in_c);
  }

  int32_t           mVcount;
  point<Type>   mVertices[MAXPTS];
};



#endif

static inline void add(const REAL *p,REAL *dest,uint32_t tstride,uint32_t &pcount)
{
  char *d = (char *) dest;
  d = d + pcount*tstride;
  dest = (REAL *) d;
  dest[0] = p[0];
  dest[1] = p[1];
  dest[2] = p[2];
  pcount++;
	assert( pcount <= 4 );
}


PlaneTriResult fm_planeTriIntersection(const REAL *_plane,    // the plane equation in Ax+By+Cz+D format
									const REAL *triangle, // the source triangle.
									uint32_t tstride,  // stride in bytes of the input and output *vertices*
									REAL        epsilon,  // the co-planar epsilon value.
									REAL       *front,    // the triangle in front of the
									uint32_t &fcount,  // number of vertices in the 'front' triangle
									REAL       *back,     // the triangle in back of the plane
									uint32_t &bcount) // the number of vertices in the 'back' triangle.
{

  fcount = 0;
  bcount = 0;

  const char *tsource = (const char *) triangle;

  // get the three vertices of the triangle.
  const REAL *p1     = (const REAL *) (tsource);
  const REAL *p2     = (const REAL *) (tsource+tstride);
  const REAL *p3     = (const REAL *) (tsource+tstride*2);


  PlaneTriResult r1   = fm_getSidePlane(p1,_plane,epsilon); // compute the side of the plane each vertex is on
  PlaneTriResult r2   = fm_getSidePlane(p2,_plane,epsilon);
  PlaneTriResult r3   = fm_getSidePlane(p3,_plane,epsilon);

  // If any of the points lay right *on* the plane....
  if ( r1 == PTR_ON_PLANE || r2 == PTR_ON_PLANE || r3 == PTR_ON_PLANE )
  {
	// If the triangle is completely co-planar, then just treat it as 'front' and return!
	if ( r1 == PTR_ON_PLANE && r2 == PTR_ON_PLANE && r3 == PTR_ON_PLANE )
	{
	  add(p1,front,tstride,fcount);
	  add(p2,front,tstride,fcount);
	  add(p3,front,tstride,fcount);
	  return PTR_FRONT;
	}
	// Decide to place the co-planar points on the same side as the co-planar point.
	PlaneTriResult r= PTR_ON_PLANE;
	if ( r1 != PTR_ON_PLANE )
	  r = r1;
	else if ( r2 != PTR_ON_PLANE )
	  r = r2;
	else if ( r3 != PTR_ON_PLANE )
	  r = r3;

	if ( r1 == PTR_ON_PLANE ) r1 = r;
	if ( r2 == PTR_ON_PLANE ) r2 = r;
	if ( r3 == PTR_ON_PLANE ) r3 = r;

  }

  if ( r1 == r2 && r1 == r3 ) // if all three vertices are on the same side of the plane.
  {
	if ( r1 == PTR_FRONT ) // if all three are in front of the plane, then copy to the 'front' output triangle.
	{
	  add(p1,front,tstride,fcount);
	  add(p2,front,tstride,fcount);
	  add(p3,front,tstride,fcount);
	}
	else
	{
	  add(p1,back,tstride,bcount); // if all three are in 'back' then copy to the 'back' output triangle.
	  add(p2,back,tstride,bcount);
	  add(p3,back,tstride,bcount);
	}
	return r1; // if all three points are on the same side of the plane return result
  }


  polygon<REAL> pi(p1,p2,p3);
  polygon<REAL>  pfront,pback;

  plane<REAL>    part(_plane);

  pi.Split_Polygon(&pi,&part,pfront,pback);

  for (int32_t i=0; i<pfront.mVcount; i++)
  {
	add( &pfront.mVertices[i].x, front, tstride, fcount );
  }

  for (int32_t i=0; i<pback.mVcount; i++)
  {
	add( &pback.mVertices[i].x, back, tstride, bcount );
  }

  PlaneTriResult ret = PTR_SPLIT;

  if ( fcount < 3 ) fcount = 0;
  if ( bcount < 3 ) bcount = 0;

  if ( fcount == 0 && bcount )
	ret = PTR_BACK;

  if ( bcount == 0 && fcount )
	ret = PTR_FRONT;


  return ret;
}

// computes the OBB for this set of points relative to this transform matrix.
void computeOBB(uint32_t vcount,const REAL *points,uint32_t pstride,REAL *sides,REAL *matrix)
{
  const char *src = (const char *) points;

  REAL bmin[3] = { 1e9, 1e9, 1e9 };
  REAL bmax[3] = { -1e9, -1e9, -1e9 };

  for (uint32_t i=0; i<vcount; i++)
  {
	const REAL *p = (const REAL *) src;
	REAL t[3];

	fm_inverseRT(matrix, p, t ); // inverse rotate translate

	if ( t[0] < bmin[0] ) bmin[0] = t[0];
	if ( t[1] < bmin[1] ) bmin[1] = t[1];
	if ( t[2] < bmin[2] ) bmin[2] = t[2];

	if ( t[0] > bmax[0] ) bmax[0] = t[0];
	if ( t[1] > bmax[1] ) bmax[1] = t[1];
	if ( t[2] > bmax[2] ) bmax[2] = t[2];

	src+=pstride;
  }

  REAL center[3];

  sides[0] = bmax[0]-bmin[0];
  sides[1] = bmax[1]-bmin[1];
  sides[2] = bmax[2]-bmin[2];

  center[0] = sides[0]*0.5f+bmin[0];
  center[1] = sides[1]*0.5f+bmin[1];
  center[2] = sides[2]*0.5f+bmin[2];

  REAL ocenter[3];

  fm_rotate(matrix,center,ocenter);

  matrix[12]+=ocenter[0];
  matrix[13]+=ocenter[1];
  matrix[14]+=ocenter[2];

}

void fm_computeBestFitOBB(uint32_t vcount,const REAL *points,uint32_t pstride,REAL *sides,REAL *matrix,bool bruteForce)
{
  REAL plane[4];
  REAL center[3];
  fm_computeBestFitPlane(vcount,points,pstride,0,0,plane,center);
  fm_planeToMatrix(plane,matrix);
  computeOBB( vcount, points, pstride, sides, matrix );

  REAL refmatrix[16];
  memcpy(refmatrix,matrix,16*sizeof(REAL));

  REAL volume = sides[0]*sides[1]*sides[2];
  if ( bruteForce )
  {
	for (REAL a=10; a<180; a+=10)
	{
	  REAL quat[4];
	  fm_eulerToQuat(0,a*FM_DEG_TO_RAD,0,quat);
	  REAL temp[16];
	  REAL pmatrix[16];
	  fm_quatToMatrix(quat,temp);
	  fm_matrixMultiply(temp,refmatrix,pmatrix);
	  REAL psides[3];
	  computeOBB( vcount, points, pstride, psides, pmatrix );
	  REAL v = psides[0]*psides[1]*psides[2];
	  if ( v < volume )
	  {
		volume = v;
		memcpy(matrix,pmatrix,sizeof(REAL)*16);
		sides[0] = psides[0];
		sides[1] = psides[1];
		sides[2] = psides[2];
	  }
	}
  }
}

void fm_computeBestFitOBB(uint32_t vcount,const REAL *points,uint32_t pstride,REAL *sides,REAL *pos,REAL *quat,bool bruteForce)
{
  REAL matrix[16];
  fm_computeBestFitOBB(vcount,points,pstride,sides,matrix,bruteForce);
  fm_getTranslation(matrix,pos);
  fm_matrixToQuat(matrix,quat);
}

void fm_computeBestFitABB(uint32_t vcount,const REAL *points,uint32_t pstride,REAL *sides,REAL *pos)
{
	REAL bmin[3];
	REAL bmax[3];

  bmin[0] = points[0];
  bmin[1] = points[1];
  bmin[2] = points[2];

  bmax[0] = points[0];
  bmax[1] = points[1];
  bmax[2] = points[2];

	const char *cp = (const char *) points;
	for (uint32_t i=0; i<vcount; i++)
	{
		const REAL *p = (const REAL *) cp;

		if ( p[0] < bmin[0] ) bmin[0] = p[0];
		if ( p[1] < bmin[1] ) bmin[1] = p[1];
		if ( p[2] < bmin[2] ) bmin[2] = p[2];

	if ( p[0] > bmax[0] ) bmax[0] = p[0];
	if ( p[1] > bmax[1] ) bmax[1] = p[1];
	if ( p[2] > bmax[2] ) bmax[2] = p[2];

	cp+=pstride;
	}


	sides[0] = bmax[0] - bmin[0];
	sides[1] = bmax[1] - bmin[1];
	sides[2] = bmax[2] - bmin[2];

	pos[0] = bmin[0]+sides[0]*0.5f;
	pos[1] = bmin[1]+sides[1]*0.5f;
	pos[2] = bmin[2]+sides[2]*0.5f;

}


void fm_planeToMatrix(const REAL *plane,REAL *matrix) // convert a plane equation to a 4x4 rotation matrix
{
  REAL ref[3] = { 0, 1, 0 };
  REAL quat[4];
  fm_rotationArc(ref,plane,quat);
  fm_quatToMatrix(quat,matrix);
  REAL origin[3] = { 0, -plane[3], 0 };
  REAL center[3];
  fm_transform(matrix,origin,center);
  fm_setTranslation(center,matrix);
}

void fm_planeToQuat(const REAL *plane,REAL *quat,REAL *pos) // convert a plane equation to a quaternion and translation
{
  REAL ref[3] = { 0, 1, 0 };
  REAL matrix[16];
  fm_rotationArc(ref,plane,quat);
  fm_quatToMatrix(quat,matrix);
  REAL origin[3] = { 0, plane[3], 0 };
  fm_transform(matrix,origin,pos);
}

void fm_eulerMatrix(REAL ax,REAL ay,REAL az,REAL *matrix) // convert euler (in radians) to a dest 4x4 matrix (translation set to zero)
{
  REAL quat[4];
  fm_eulerToQuat(ax,ay,az,quat);
  fm_quatToMatrix(quat,matrix);
}


//**********************************************************
//**********************************************************
//**** Vertex Welding
//**********************************************************
//**********************************************************

#ifndef VERTEX_INDEX_H

#define VERTEX_INDEX_H

namespace VERTEX_INDEX
{

class KdTreeNode;

typedef std::vector< KdTreeNode * > KdTreeNodeVector;

enum Axes
{
  X_AXIS = 0,
  Y_AXIS = 1,
  Z_AXIS = 2
};

class KdTreeFindNode
{
public:
  KdTreeFindNode(void)
  {
	mNode = 0;
	mDistance = 0;
  }
  KdTreeNode  *mNode;
  double        mDistance;
};

class KdTreeInterface
{
public:
  virtual const double * getPositionDouble(uint32_t index) const = 0;
  virtual const float  * getPositionFloat(uint32_t index) const = 0;
};

class KdTreeNode
{
public:
  KdTreeNode(void)
  {
	mIndex = 0;
	mLeft = 0;
	mRight = 0;
  }

  KdTreeNode(uint32_t index)
  {
	mIndex = index;
	mLeft = 0;
	mRight = 0;
  };

	~KdTreeNode(void)
  {
  }


  void addDouble(KdTreeNode *node,Axes dim,const KdTreeInterface *iface)
  {
	const double *nodePosition = iface->getPositionDouble( node->mIndex );
	const double *position     = iface->getPositionDouble( mIndex );
	switch ( dim )
	{
	  case X_AXIS:
		if ( nodePosition[0] <= position[0] )
		{
		  if ( mLeft )
			mLeft->addDouble(node,Y_AXIS,iface);
		  else
			mLeft = node;
		}
		else
		{
		  if ( mRight )
			mRight->addDouble(node,Y_AXIS,iface);
		  else
			mRight = node;
		}
		break;
	  case Y_AXIS:
		if ( nodePosition[1] <= position[1] )
		{
		  if ( mLeft )
			mLeft->addDouble(node,Z_AXIS,iface);
		  else
			mLeft = node;
		}
		else
		{
		  if ( mRight )
			mRight->addDouble(node,Z_AXIS,iface);
		  else
			mRight = node;
		}
		break;
	  case Z_AXIS:
		if ( nodePosition[2] <= position[2] )
		{
		  if ( mLeft )
			mLeft->addDouble(node,X_AXIS,iface);
		  else
			mLeft = node;
		}
		else
		{
		  if ( mRight )
			mRight->addDouble(node,X_AXIS,iface);
		  else
			mRight = node;
		}
		break;
	}

  }


  void addFloat(KdTreeNode *node,Axes dim,const KdTreeInterface *iface)
  {
	const float *nodePosition = iface->getPositionFloat( node->mIndex );
	const float *position     = iface->getPositionFloat( mIndex );
	switch ( dim )
	{
	  case X_AXIS:
		if ( nodePosition[0] <= position[0] )
		{
		  if ( mLeft )
			mLeft->addFloat(node,Y_AXIS,iface);
		  else
			mLeft = node;
		}
		else
		{
		  if ( mRight )
			mRight->addFloat(node,Y_AXIS,iface);
		  else
			mRight = node;
		}
		break;
	  case Y_AXIS:
		if ( nodePosition[1] <= position[1] )
		{
		  if ( mLeft )
			mLeft->addFloat(node,Z_AXIS,iface);
		  else
			mLeft = node;
		}
		else
		{
		  if ( mRight )
			mRight->addFloat(node,Z_AXIS,iface);
		  else
			mRight = node;
		}
		break;
	  case Z_AXIS:
		if ( nodePosition[2] <= position[2] )
		{
		  if ( mLeft )
			mLeft->addFloat(node,X_AXIS,iface);
		  else
			mLeft = node;
		}
		else
		{
		  if ( mRight )
			mRight->addFloat(node,X_AXIS,iface);
		  else
			mRight = node;
		}
		break;
	}

  }


  uint32_t getIndex(void) const { return mIndex; };

  void search(Axes axis,const double *pos,double radius,uint32_t &count,uint32_t maxObjects,KdTreeFindNode *found,const KdTreeInterface *iface)
  {

	const double *position = iface->getPositionDouble(mIndex);

	double dx = pos[0] - position[0];
	double dy = pos[1] - position[1];
	double dz = pos[2] - position[2];

	KdTreeNode *search1 = 0;
	KdTreeNode *search2 = 0;

	switch ( axis )
	{
	  case X_AXIS:
	   if ( dx <= 0 )     // JWR  if we are to the left
	   {
		search1 = mLeft; // JWR  then search to the left
		if ( -dx < radius )  // JWR  if distance to the right is less than our search radius, continue on the right as well.
		  search2 = mRight;
	   }
	   else
	   {
		 search1 = mRight; // JWR  ok, we go down the left tree
		 if ( dx < radius ) // JWR  if the distance from the right is less than our search radius
				search2 = mLeft;
		}
		axis = Y_AXIS;
		break;
	  case Y_AXIS:
		if ( dy <= 0 )
		{
		  search1 = mLeft;
		  if ( -dy < radius )
					search2 = mRight;
		}
		else
		{
		  search1 = mRight;
		  if ( dy < radius )
					search2 = mLeft;
		}
		axis = Z_AXIS;
		break;
	  case Z_AXIS:
		if ( dz <= 0 )
		{
		  search1 = mLeft;
		  if ( -dz < radius )
					search2 = mRight;
		}
		else
		{
		  search1 = mRight;
		  if ( dz < radius )
					search2 = mLeft;
		}
		axis = X_AXIS;
		break;
	}

	double r2 = radius*radius;
	double m  = dx*dx+dy*dy+dz*dz;

	if ( m < r2 )
	{
	  switch ( count )
	  {
		case 0:
		  found[count].mNode = this;
		  found[count].mDistance = m;
		  break;
		case 1:
		  if ( m < found[0].mDistance )
		  {
			if ( maxObjects == 1 )
			{
			  found[0].mNode = this;
			  found[0].mDistance = m;
			}
			else
			{
			  found[1] = found[0];
			  found[0].mNode = this;
			  found[0].mDistance = m;
			}
		  }
		  else if ( maxObjects > 1)
		  {
			found[1].mNode = this;
			found[1].mDistance = m;
		  }
		  break;
		default:
		  {
			bool inserted = false;

			for (uint32_t i=0; i<count; i++)
			{
			  if ( m < found[i].mDistance ) // if this one is closer than a pre-existing one...
			  {
				// insertion sort...
				uint32_t scan = count;
				if ( scan >= maxObjects ) scan=maxObjects-1;
				for (uint32_t j=scan; j>i; j--)
				{
				  found[j] = found[j-1];
				}
				found[i].mNode = this;
				found[i].mDistance = m;
				inserted = true;
				break;
			  }
			}

			if ( !inserted && count < maxObjects )
			{
			  found[count].mNode = this;
			  found[count].mDistance = m;
			}
		  }
		  break;
	  }
	  count++;
	  if ( count > maxObjects )
	  {
		count = maxObjects;
	  }
	}


	if ( search1 )
		search1->search( axis, pos,radius, count, maxObjects, found, iface);

	if ( search2 )
		search2->search( axis, pos,radius, count, maxObjects, found, iface);

  }

  void search(Axes axis,const float *pos,float radius,uint32_t &count,uint32_t maxObjects,KdTreeFindNode *found,const KdTreeInterface *iface)
  {

	const float *position = iface->getPositionFloat(mIndex);

	float dx = pos[0] - position[0];
	float dy = pos[1] - position[1];
	float dz = pos[2] - position[2];

	KdTreeNode *search1 = 0;
	KdTreeNode *search2 = 0;

	switch ( axis )
	{
	  case X_AXIS:
	   if ( dx <= 0 )     // JWR  if we are to the left
	   {
		search1 = mLeft; // JWR  then search to the left
		if ( -dx < radius )  // JWR  if distance to the right is less than our search radius, continue on the right as well.
		  search2 = mRight;
	   }
	   else
	   {
		 search1 = mRight; // JWR  ok, we go down the left tree
		 if ( dx < radius ) // JWR  if the distance from the right is less than our search radius
				search2 = mLeft;
		}
		axis = Y_AXIS;
		break;
	  case Y_AXIS:
		if ( dy <= 0 )
		{
		  search1 = mLeft;
		  if ( -dy < radius )
					search2 = mRight;
		}
		else
		{
		  search1 = mRight;
		  if ( dy < radius )
					search2 = mLeft;
		}
		axis = Z_AXIS;
		break;
	  case Z_AXIS:
		if ( dz <= 0 )
		{
		  search1 = mLeft;
		  if ( -dz < radius )
					search2 = mRight;
		}
		else
		{
		  search1 = mRight;
		  if ( dz < radius )
					search2 = mLeft;
		}
		axis = X_AXIS;
		break;
	}

	float r2 = radius*radius;
	float m  = dx*dx+dy*dy+dz*dz;

	if ( m < r2 )
	{
	  switch ( count )
	  {
		case 0:
		  found[count].mNode = this;
		  found[count].mDistance = m;
		  break;
		case 1:
		  if ( m < found[0].mDistance )
		  {
			if ( maxObjects == 1 )
			{
			  found[0].mNode = this;
			  found[0].mDistance = m;
			}
			else
			{
			  found[1] = found[0];
			  found[0].mNode = this;
			  found[0].mDistance = m;
			}
		  }
		  else if ( maxObjects > 1)
		  {
			found[1].mNode = this;
			found[1].mDistance = m;
		  }
		  break;
		default:
		  {
			bool inserted = false;

			for (uint32_t i=0; i<count; i++)
			{
			  if ( m < found[i].mDistance ) // if this one is closer than a pre-existing one...
			  {
				// insertion sort...
				uint32_t scan = count;
				if ( scan >= maxObjects ) scan=maxObjects-1;
				for (uint32_t j=scan; j>i; j--)
				{
				  found[j] = found[j-1];
				}
				found[i].mNode = this;
				found[i].mDistance = m;
				inserted = true;
				break;
			  }
			}

			if ( !inserted && count < maxObjects )
			{
			  found[count].mNode = this;
			  found[count].mDistance = m;
			}
		  }
		  break;
	  }
	  count++;
	  if ( count > maxObjects )
	  {
		count = maxObjects;
	  }
	}


	if ( search1 )
		search1->search( axis, pos,radius, count, maxObjects, found, iface);

	if ( search2 )
		search2->search( axis, pos,radius, count, maxObjects, found, iface);

  }

private:

  void setLeft(KdTreeNode *left) { mLeft = left; };
  void setRight(KdTreeNode *right) { mRight = right; };

	KdTreeNode *getLeft(void)         { return mLeft; }
	KdTreeNode *getRight(void)        { return mRight; }

  uint32_t          mIndex;
  KdTreeNode     *mLeft;
  KdTreeNode     *mRight;
};


#define MAX_BUNDLE_SIZE 1024  // 1024 nodes at a time, to minimize memory allocation and guarantee that pointers are persistent.

class KdTreeNodeBundle 
{
public:

  KdTreeNodeBundle(void)
  {
	mNext = 0;
	mIndex = 0;
  }

  bool isFull(void) const
  {
	return (bool)( mIndex == MAX_BUNDLE_SIZE );
  }

  KdTreeNode * getNextNode(void)
  {
	assert(mIndex<MAX_BUNDLE_SIZE);
	KdTreeNode *ret = &mNodes[mIndex];
	mIndex++;
	return ret;
  }

  KdTreeNodeBundle  *mNext;
  uint32_t             mIndex;
  KdTreeNode         mNodes[MAX_BUNDLE_SIZE];
};


typedef std::vector< double > DoubleVector;
typedef std::vector< float >  FloatVector;

class KdTree : public KdTreeInterface
{
public:
  KdTree(void)
  {
	mRoot = 0;
	mBundle = 0;
	mVcount = 0;
	mUseDouble = false;
  }

  virtual ~KdTree(void)
  {
	reset();
  }

  const double * getPositionDouble(uint32_t index) const
  {
	assert( mUseDouble );
	assert ( index < mVcount );
	return  &mVerticesDouble[index*3];
  }

  const float * getPositionFloat(uint32_t index) const
  {
	assert( !mUseDouble );
	assert ( index < mVcount );
	return  &mVerticesFloat[index*3];
  }

  uint32_t search(const double *pos,double radius,uint32_t maxObjects,KdTreeFindNode *found) const
  {
	assert( mUseDouble );
	if ( !mRoot )	return 0;
	uint32_t count = 0;
	mRoot->search(X_AXIS,pos,radius,count,maxObjects,found,this);
	return count;
  }

  uint32_t search(const float *pos,float radius,uint32_t maxObjects,KdTreeFindNode *found) const
  {
	assert( !mUseDouble );
	if ( !mRoot )	return 0;
	uint32_t count = 0;
	mRoot->search(X_AXIS,pos,radius,count,maxObjects,found,this);
	return count;
  }

  void reset(void)
  {
	mRoot = 0;
	mVerticesDouble.clear();
	mVerticesFloat.clear();
	KdTreeNodeBundle *bundle = mBundle;
	while ( bundle )
	{
	  KdTreeNodeBundle *next = bundle->mNext;
	  delete bundle;
	  bundle = next;
	}
	mBundle = 0;
	mVcount = 0;
  }

  uint32_t add(double x,double y,double z)
  {
	assert(mUseDouble);
	uint32_t ret = mVcount;
	mVerticesDouble.push_back(x);
	mVerticesDouble.push_back(y);
	mVerticesDouble.push_back(z);
	mVcount++;
	KdTreeNode *node = getNewNode(ret);
	if ( mRoot )
	{
	  mRoot->addDouble(node,X_AXIS,this);
	}
	else
	{
	  mRoot = node;
	}
	return ret;
  }

  uint32_t add(float x,float y,float z)
  {
	assert(!mUseDouble);
	uint32_t ret = mVcount;
	mVerticesFloat.push_back(x);
	mVerticesFloat.push_back(y);
	mVerticesFloat.push_back(z);
	mVcount++;
	KdTreeNode *node = getNewNode(ret);
	if ( mRoot )
	{
	  mRoot->addFloat(node,X_AXIS,this);
	}
	else
	{
	  mRoot = node;
	}
	return ret;
  }

  KdTreeNode * getNewNode(uint32_t index)
  {
	if ( mBundle == 0 )
	{
	  mBundle = new KdTreeNodeBundle;
	}
	if ( mBundle->isFull() )
	{
	  KdTreeNodeBundle *bundle = new KdTreeNodeBundle;
	  mBundle->mNext = bundle;
	  mBundle = bundle;
	}
	KdTreeNode *node = mBundle->getNextNode();
	new ( node ) KdTreeNode(index);
	return node;
  }

  uint32_t getNearest(const double *pos,double radius,bool &_found) const // returns the nearest possible neighbor's index.
  {
	assert( mUseDouble );
	uint32_t ret = 0;

	_found = false;
	KdTreeFindNode found[1];
	uint32_t count = search(pos,radius,1,found);
	if ( count )
	{
	  KdTreeNode *node = found[0].mNode;
	  ret = node->getIndex();
	  _found = true;
	}
	return ret;
  }

  uint32_t getNearest(const float *pos,float radius,bool &_found) const // returns the nearest possible neighbor's index.
  {
	assert( !mUseDouble );
	uint32_t ret = 0;

	_found = false;
	KdTreeFindNode found[1];
	uint32_t count = search(pos,radius,1,found);
	if ( count )
	{
	  KdTreeNode *node = found[0].mNode;
	  ret = node->getIndex();
	  _found = true;
	}
	return ret;
  }

  const double * getVerticesDouble(void) const
  {
	assert( mUseDouble );
	const double *ret = 0;
	if ( !mVerticesDouble.empty() )
	{
	  ret = &mVerticesDouble[0];
	}
	return ret;
  }

  const float * getVerticesFloat(void) const
  {
	assert( !mUseDouble );
	const float * ret = 0;
	if ( !mVerticesFloat.empty() )
	{
	  ret = &mVerticesFloat[0];
	}
	return ret;
  }

  uint32_t getVcount(void) const { return mVcount; };

  void setUseDouble(bool useDouble)
  {
	mUseDouble = useDouble;
  }

private:
  bool                    mUseDouble;
  KdTreeNode             *mRoot;
  KdTreeNodeBundle       *mBundle;
  uint32_t                  mVcount;
  DoubleVector            mVerticesDouble;
  FloatVector             mVerticesFloat;
};

}; // end of namespace VERTEX_INDEX

class MyVertexIndex : public fm_VertexIndex
{
public:
  MyVertexIndex(double granularity,bool snapToGrid)
  {
	mDoubleGranularity = granularity;
	mFloatGranularity  = (float)granularity;
	mSnapToGrid        = snapToGrid;
	mUseDouble         = true;
	mKdTree.setUseDouble(true);
  }

  MyVertexIndex(float granularity,bool snapToGrid)
  {
	mDoubleGranularity = granularity;
	mFloatGranularity  = (float)granularity;
	mSnapToGrid        = snapToGrid;
	mUseDouble         = false;
	mKdTree.setUseDouble(false);
  }

  virtual ~MyVertexIndex(void)
  {

  }


  double snapToGrid(double p)
  {
	double m = fmod(p,mDoubleGranularity);
	p-=m;
	return p;
  }

  float snapToGrid(float p)
  {
	float m = fmodf(p,mFloatGranularity);
	p-=m;
	return p;
  }

  uint32_t    getIndex(const float *_p,bool &newPos)  // get index for a vector float
  {
	uint32_t ret;

	if ( mUseDouble )
	{
	  double p[3];
	  p[0] = _p[0];
	  p[1] = _p[1];
	  p[2] = _p[2];
	  return getIndex(p,newPos);
	}

	newPos = false;

	float p[3];

	if ( mSnapToGrid )
	{
	  p[0] = snapToGrid(_p[0]);
	  p[1] = snapToGrid(_p[1]);
	  p[2] = snapToGrid(_p[2]);
	}
	else
	{
	  p[0] = _p[0];
	  p[1] = _p[1];
	  p[2] = _p[2];
	}

	bool found;
	ret = mKdTree.getNearest(p,mFloatGranularity,found);
	if ( !found )
	{
	  newPos = true;
	  ret = mKdTree.add(p[0],p[1],p[2]);
	}


	return ret;
  }

  uint32_t    getIndex(const double *_p,bool &newPos)  // get index for a vector double
  {
	uint32_t ret;

	if ( !mUseDouble )
	{
	  float p[3];
	  p[0] = (float)_p[0];
	  p[1] = (float)_p[1];
	  p[2] = (float)_p[2];
	  return getIndex(p,newPos);
	}

	newPos = false;

	double p[3];

	if ( mSnapToGrid )
	{
	  p[0] = snapToGrid(_p[0]);
	  p[1] = snapToGrid(_p[1]);
	  p[2] = snapToGrid(_p[2]);
	}
	else
	{
	  p[0] = _p[0];
	  p[1] = _p[1];
	  p[2] = _p[2];
	}

	bool found;
	ret = mKdTree.getNearest(p,mDoubleGranularity,found);
	if ( !found )
	{
	  newPos = true;
	  ret = mKdTree.add(p[0],p[1],p[2]);
	}


	return ret;
  }

  const float *   getVerticesFloat(void) const
  {
	const float * ret = 0;

	assert( !mUseDouble );

	ret = mKdTree.getVerticesFloat();

	return ret;
  }

  const double *  getVerticesDouble(void) const
  {
	const double * ret = 0;

	assert( mUseDouble );

	ret = mKdTree.getVerticesDouble();

	return ret;
  }

  const float *   getVertexFloat(uint32_t index) const
  {
	const float * ret  = 0;
	assert( !mUseDouble );
#ifdef _DEBUG
	uint32_t vcount = mKdTree.getVcount();
	assert( index < vcount );
#endif
	ret =  mKdTree.getVerticesFloat();
	ret = &ret[index*3];
	return ret;
  }

  const double *   getVertexDouble(uint32_t index) const
  {
	const double * ret = 0;
	assert( mUseDouble );
#ifdef _DEBUG
	uint32_t vcount = mKdTree.getVcount();
	assert( index < vcount );
#endif
	ret =  mKdTree.getVerticesDouble();
	ret = &ret[index*3];

	return ret;
  }

  uint32_t    getVcount(void) const
  {
	return mKdTree.getVcount();
  }

  bool isDouble(void) const
  {
	return mUseDouble;
  }


  bool            saveAsObj(const char *fname,uint32_t tcount,uint32_t *indices)
  {
	bool ret = false;


	FILE *fph = fopen(fname,"wb");
	if ( fph )
	{
	  ret = true;

	  uint32_t vcount    = getVcount();
	  if ( mUseDouble )
	  {
		const double *v  = getVerticesDouble();
		for (uint32_t i=0; i<vcount; i++)
		{
		  fprintf(fph,"v %0.9f %0.9f %0.9f\r\n", (float)v[0], (float)v[1], (float)v[2] );
		  v+=3;
		}
	  }
	  else
	  {
		const float *v  = getVerticesFloat();
		for (uint32_t i=0; i<vcount; i++)
		{
		  fprintf(fph,"v %0.9f %0.9f %0.9f\r\n", v[0], v[1], v[2] );
		  v+=3;
		}
	  }

	  for (uint32_t i=0; i<tcount; i++)
	  {
		uint32_t i1 = *indices++;
		uint32_t i2 = *indices++;
		uint32_t i3 = *indices++;
		fprintf(fph,"f %d %d %d\r\n", i1+1, i2+1, i3+1 );
	  }
	  fclose(fph);
	}

	return ret;
  }

private:
  bool    mUseDouble:1;
  bool    mSnapToGrid:1;
  double  mDoubleGranularity;
  float   mFloatGranularity;
  VERTEX_INDEX::KdTree  mKdTree;
};

fm_VertexIndex * fm_createVertexIndex(double granularity,bool snapToGrid) // create an indexed vertex system for doubles
{
  MyVertexIndex *ret = new MyVertexIndex(granularity,snapToGrid);
  return static_cast< fm_VertexIndex *>(ret);
}

fm_VertexIndex * fm_createVertexIndex(float granularity,bool snapToGrid)  // create an indexed vertext system for floats
{
  MyVertexIndex *ret = new MyVertexIndex(granularity,snapToGrid);
  return static_cast< fm_VertexIndex *>(ret);
}

void          fm_releaseVertexIndex(fm_VertexIndex *vindex)
{
  MyVertexIndex *m = static_cast< MyVertexIndex *>(vindex);
  delete m;
}

#endif   // END OF VERTEX WELDING CODE


REAL fm_computeBestFitAABB(uint32_t vcount,const REAL *points,uint32_t pstride,REAL *bmin,REAL *bmax) // returns the diagonal distance
{

  const uint8_t *source = (const uint8_t *) points;

	bmin[0] = points[0];
	bmin[1] = points[1];
	bmin[2] = points[2];

	bmax[0] = points[0];
	bmax[1] = points[1];
	bmax[2] = points[2];


  for (uint32_t i=1; i<vcount; i++)
  {
	source+=pstride;
	const REAL *p = (const REAL *) source;

	if ( p[0] < bmin[0] ) bmin[0] = p[0];
	if ( p[1] < bmin[1] ) bmin[1] = p[1];
	if ( p[2] < bmin[2] ) bmin[2] = p[2];

		if ( p[0] > bmax[0] ) bmax[0] = p[0];
		if ( p[1] > bmax[1] ) bmax[1] = p[1];
		if ( p[2] > bmax[2] ) bmax[2] = p[2];

  }

  REAL dx = bmax[0] - bmin[0];
  REAL dy = bmax[1] - bmin[1];
  REAL dz = bmax[2] - bmin[2];

	return (REAL) sqrt( dx*dx + dy*dy + dz*dz );

}



/* a = b - c */
#define vector(a,b,c) \
	(a)[0] = (b)[0] - (c)[0];	\
	(a)[1] = (b)[1] - (c)[1];	\
	(a)[2] = (b)[2] - (c)[2];



#define innerProduct(v,q) \
		((v)[0] * (q)[0] + \
		(v)[1] * (q)[1] + \
		(v)[2] * (q)[2])

#define crossProduct(a,b,c) \
	(a)[0] = (b)[1] * (c)[2] - (c)[1] * (b)[2]; \
	(a)[1] = (b)[2] * (c)[0] - (c)[2] * (b)[0]; \
	(a)[2] = (b)[0] * (c)[1] - (c)[0] * (b)[1];


bool fm_lineIntersectsTriangle(const REAL *rayStart,const REAL *rayEnd,const REAL *p1,const REAL *p2,const REAL *p3,REAL *sect)
{
	REAL dir[3];

  dir[0] = rayEnd[0] - rayStart[0];
  dir[1] = rayEnd[1] - rayStart[1];
  dir[2] = rayEnd[2] - rayStart[2];

  REAL d = (REAL)sqrt(dir[0]*dir[0] + dir[1]*dir[1] + dir[2]*dir[2]);
  REAL r = 1.0f / d;

  dir[0]*=r;
  dir[1]*=r;
  dir[2]*=r;


  REAL t;

	bool ret = fm_rayIntersectsTriangle(rayStart, dir, p1, p2, p3, t );

	if ( ret )
	{
		if ( t > d )
		{
			sect[0] = rayStart[0] + dir[0]*t;
			sect[1] = rayStart[1] + dir[1]*t;
			sect[2] = rayStart[2] + dir[2]*t;
		}
		else
		{
			ret = false;
		}
	}

  return ret;
}



bool fm_rayIntersectsTriangle(const REAL *p,const REAL *d,const REAL *v0,const REAL *v1,const REAL *v2,REAL &t)
{
	REAL e1[3],e2[3],h[3],s[3],q[3];
	REAL a,f,u,v;

	vector(e1,v1,v0);
	vector(e2,v2,v0);
	crossProduct(h,d,e2);
	a = innerProduct(e1,h);

	if (a > -0.00001 && a < 0.00001)
		return(false);

	f = 1/a;
	vector(s,p,v0);
	u = f * (innerProduct(s,h));

	if (u < 0.0 || u > 1.0)
		return(false);

	crossProduct(q,s,e1);
	v = f * innerProduct(d,q);
	if (v < 0.0 || u + v > 1.0)
		return(false);
	// at this stage we can compute t to find out where
	// the intersection point is on the line
	t = f * innerProduct(e2,q);
	if (t > 0) // ray intersection
		return(true);
	else // this means that there is a line intersection
		 // but not a ray intersection
		 return (false);
}


inline REAL det(const REAL *p1,const REAL *p2,const REAL *p3)
{
  return  p1[0]*p2[1]*p3[2] + p2[0]*p3[1]*p1[2] + p3[0]*p1[1]*p2[2] -p1[0]*p3[1]*p2[2] - p2[0]*p1[1]*p3[2] - p3[0]*p2[1]*p1[2];
}


REAL  fm_computeMeshVolume(const REAL *vertices,uint32_t tcount,const uint32_t *indices)
{
	REAL volume = 0;

	for (uint32_t i=0; i<tcount; i++,indices+=3)
	{
	const REAL *p1 = &vertices[ indices[0]*3 ];
		const REAL *p2 = &vertices[ indices[1]*3 ];
		const REAL *p3 = &vertices[ indices[2]*3 ];
		volume+=det(p1,p2,p3); // compute the volume of the tetrahedran relative to the origin.
	}

	volume*=(1.0f/6.0f);
	if ( volume < 0 )
		volume*=-1;
	return volume;
}


const REAL * fm_getPoint(const REAL *points,uint32_t pstride,uint32_t index)
{
  const uint8_t *scan = (const uint8_t *)points;
  scan+=(index*pstride);
  return (REAL *)scan;
}


bool fm_insideTriangle(REAL Ax, REAL Ay,
					  REAL Bx, REAL By,
					  REAL Cx, REAL Cy,
					  REAL Px, REAL Py)

{
  REAL ax, ay, bx, by, cx, cy, apx, apy, bpx, bpy, cpx, cpy;
  REAL cCROSSap, bCROSScp, aCROSSbp;

  ax = Cx - Bx;  ay = Cy - By;
  bx = Ax - Cx;  by = Ay - Cy;
  cx = Bx - Ax;  cy = By - Ay;
  apx= Px - Ax;  apy= Py - Ay;
  bpx= Px - Bx;  bpy= Py - By;
  cpx= Px - Cx;  cpy= Py - Cy;

  aCROSSbp = ax*bpy - ay*bpx;
  cCROSSap = cx*apy - cy*apx;
  bCROSScp = bx*cpy - by*cpx;

  return ((aCROSSbp >= 0.0f) && (bCROSScp >= 0.0f) && (cCROSSap >= 0.0f));
}


REAL fm_areaPolygon2d(uint32_t pcount,const REAL *points,uint32_t pstride)
{
  int32_t n = (int32_t)pcount;

  REAL A=0.0f;
  for(int32_t p=n-1,q=0; q<n; p=q++)
  {
	const REAL *p1 = fm_getPoint(points,pstride,p);
	const REAL *p2 = fm_getPoint(points,pstride,q);
	A+= p1[0]*p2[1] - p2[0]*p1[1];
  }
  return A*0.5f;
}


bool  fm_pointInsidePolygon2d(uint32_t pcount,const REAL *points,uint32_t pstride,const REAL *point,uint32_t xindex,uint32_t yindex)
{
  uint32_t j = pcount-1;
  int32_t oddNodes = 0;

  REAL x = point[xindex];
  REAL y = point[yindex];

  for (uint32_t i=0; i<pcount; i++)
  {
	const REAL *p1 = fm_getPoint(points,pstride,i);
	const REAL *p2 = fm_getPoint(points,pstride,j);

	REAL x1 = p1[xindex];
	REAL y1 = p1[yindex];

	REAL x2 = p2[xindex];
	REAL y2 = p2[yindex];

	if ( (y1 < y && y2 >= y) ||  (y2 < y && y1 >= y) )
	{
	  if (x1+(y-y1)/(y2-y1)*(x2-x1)<x)
	  {
		oddNodes = 1-oddNodes;
	  }
	}
	j = i;
  }

  return oddNodes ? true : false;
}


uint32_t fm_consolidatePolygon(uint32_t pcount,const REAL *points,uint32_t pstride,REAL *_dest,REAL epsilon) // collapses co-linear edges.
{
  uint32_t ret = 0;


  if ( pcount >= 3 )
  {
	const REAL *prev = fm_getPoint(points,pstride,pcount-1);
	const REAL *current = points;
	const REAL *next    = fm_getPoint(points,pstride,1);
	REAL *dest = _dest;

	for (uint32_t i=0; i<pcount; i++)
	{

	  next = (i+1)==pcount ? points : next;

	  if ( !fm_colinear(prev,current,next,epsilon) )
	  {
		dest[0] = current[0];
		dest[1] = current[1];
		dest[2] = current[2];

		dest+=3;
		ret++;
	  }

	  prev = current;
	  current+=3;
	  next+=3;

	}
  }

  return ret;
}


#ifndef RECT3D_TEMPLATE

#define RECT3D_TEMPLATE

template <class T> class Rect3d
{
public:
  Rect3d(void) { };

  Rect3d(const T *bmin,const T *bmax)
  {

	mMin[0] = bmin[0];
	mMin[1] = bmin[1];
	mMin[2] = bmin[2];

	mMax[0] = bmax[0];
	mMax[1] = bmax[1];
	mMax[2] = bmax[2];

  }

  void SetMin(const T *bmin)
  {
	mMin[0] = bmin[0];
	mMin[1] = bmin[1];
	mMin[2] = bmin[2];
  }

  void SetMax(const T *bmax)
  {
	mMax[0] = bmax[0];
	mMax[1] = bmax[1];
	mMax[2] = bmax[2];
  }

	void SetMin(T x,T y,T z)
	{
		mMin[0] = x;
		mMin[1] = y;
		mMin[2] = z;
	}

	void SetMax(T x,T y,T z)
	{
		mMax[0] = x;
		mMax[1] = y;
		mMax[2] = z;
	}

  T mMin[3];
  T mMax[3];
};

#endif

void splitRect(uint32_t axis,
						   const Rect3d<REAL> &source,
							 Rect3d<REAL> &b1,
							 Rect3d<REAL> &b2,
							 const REAL *midpoint)
{
	switch ( axis )
	{
		case 0:
			b1.SetMin(source.mMin);
			b1.SetMax( midpoint[0], source.mMax[1], source.mMax[2] );

			b2.SetMin( midpoint[0], source.mMin[1], source.mMin[2] );
			b2.SetMax(source.mMax);

			break;
		case 1:
			b1.SetMin(source.mMin);
			b1.SetMax( source.mMax[0], midpoint[1], source.mMax[2] );

			b2.SetMin( source.mMin[0], midpoint[1], source.mMin[2] );
			b2.SetMax(source.mMax);

			break;
		case 2:
			b1.SetMin(source.mMin);
			b1.SetMax( source.mMax[0], source.mMax[1], midpoint[2] );

			b2.SetMin( source.mMin[0], source.mMin[1], midpoint[2] );
			b2.SetMax(source.mMax);

			break;
	}
}

bool fm_computeSplitPlane(uint32_t vcount,
						  const REAL *vertices,
						  uint32_t /* tcount */,
						  const uint32_t * /* indices */,
						  REAL *plane)
{

  REAL sides[3];
  REAL matrix[16];

  fm_computeBestFitOBB( vcount, vertices, sizeof(REAL)*3, sides, matrix );

  REAL bmax[3];
  REAL bmin[3];

  bmax[0] = sides[0]*0.5f;
  bmax[1] = sides[1]*0.5f;
  bmax[2] = sides[2]*0.5f;

  bmin[0] = -bmax[0];
  bmin[1] = -bmax[1];
  bmin[2] = -bmax[2];


  REAL dx = sides[0];
  REAL dy = sides[1];
  REAL dz = sides[2];


	uint32_t axis = 0;

	if ( dy > dx )
	{
		axis = 1;
	}

	if ( dz > dx && dz > dy )
	{
		axis = 2;
	}

  REAL p1[3];
  REAL p2[3];
  REAL p3[3];

  p3[0] = p2[0] = p1[0] = bmin[0] + dx*0.5f;
  p3[1] = p2[1] = p1[1] = bmin[1] + dy*0.5f;
  p3[2] = p2[2] = p1[2] = bmin[2] + dz*0.5f;

  Rect3d<REAL> b(bmin,bmax);

  Rect3d<REAL> b1,b2;

  splitRect(axis,b,b1,b2,p1);


  switch ( axis )
  {
	case 0:
	  p2[1] = bmin[1];
	  p2[2] = bmin[2];

	  if ( dz > dy )
	  {
		p3[1] = bmax[1];
		p3[2] = bmin[2];
	  }
	  else
	  {
		p3[1] = bmin[1];
		p3[2] = bmax[2];
	  }

	  break;
	case 1:
	  p2[0] = bmin[0];
	  p2[2] = bmin[2];

	  if ( dx > dz )
	  {
		p3[0] = bmax[0];
		p3[2] = bmin[2];
	  }
	  else
	  {
		p3[0] = bmin[0];
		p3[2] = bmax[2];
	  }

	  break;
	case 2:
	  p2[0] = bmin[0];
	  p2[1] = bmin[1];

	  if ( dx > dy )
	  {
		p3[0] = bmax[0];
		p3[1] = bmin[1];
	  }
	  else
	  {
		p3[0] = bmin[0];
		p3[1] = bmax[1];
	  }

	  break;
  }

  REAL tp1[3];
  REAL tp2[3];
  REAL tp3[3];

  fm_transform(matrix,p1,tp1);
  fm_transform(matrix,p2,tp2);
  fm_transform(matrix,p3,tp3);

	plane[3] = fm_computePlane(tp1,tp2,tp3,plane);

  return true;

}

#pragma warning(disable:4100)

void fm_nearestPointInTriangle(const REAL * /*nearestPoint*/,const REAL * /*p1*/,const REAL * /*p2*/,const REAL * /*p3*/,REAL * /*nearest*/)
{

}

static REAL Partial(const REAL *a,const REAL *p) 
{
	return (a[0]*p[1]) - (p[0]*a[1]);
}

REAL  fm_areaTriangle(const REAL *p0,const REAL *p1,const REAL *p2)
{
  REAL A = Partial(p0,p1);
	A+= Partial(p1,p2);
	A+= Partial(p2,p0);
	return A*0.5f;
}

void fm_subtract(const REAL *A,const REAL *B,REAL *diff) // compute A-B and store the result in 'diff'
{
  diff[0] = A[0]-B[0];
  diff[1] = A[1]-B[1];
  diff[2] = A[2]-B[2];
}


void  fm_multiplyTransform(const REAL *pA,const REAL *pB,REAL *pM)
{

  REAL a = pA[0*4+0] * pB[0*4+0] + pA[0*4+1] * pB[1*4+0] + pA[0*4+2] * pB[2*4+0] + pA[0*4+3] * pB[3*4+0];
  REAL b = pA[0*4+0] * pB[0*4+1] + pA[0*4+1] * pB[1*4+1] + pA[0*4+2] * pB[2*4+1] + pA[0*4+3] * pB[3*4+1];
  REAL c = pA[0*4+0] * pB[0*4+2] + pA[0*4+1] * pB[1*4+2] + pA[0*4+2] * pB[2*4+2] + pA[0*4+3] * pB[3*4+2];
  REAL d = pA[0*4+0] * pB[0*4+3] + pA[0*4+1] * pB[1*4+3] + pA[0*4+2] * pB[2*4+3] + pA[0*4+3] * pB[3*4+3];

  REAL e = pA[1*4+0] * pB[0*4+0] + pA[1*4+1] * pB[1*4+0] + pA[1*4+2] * pB[2*4+0] + pA[1*4+3] * pB[3*4+0];
  REAL f = pA[1*4+0] * pB[0*4+1] + pA[1*4+1] * pB[1*4+1] + pA[1*4+2] * pB[2*4+1] + pA[1*4+3] * pB[3*4+1];
  REAL g = pA[1*4+0] * pB[0*4+2] + pA[1*4+1] * pB[1*4+2] + pA[1*4+2] * pB[2*4+2] + pA[1*4+3] * pB[3*4+2];
  REAL h = pA[1*4+0] * pB[0*4+3] + pA[1*4+1] * pB[1*4+3] + pA[1*4+2] * pB[2*4+3] + pA[1*4+3] * pB[3*4+3];

  REAL i = pA[2*4+0] * pB[0*4+0] + pA[2*4+1] * pB[1*4+0] + pA[2*4+2] * pB[2*4+0] + pA[2*4+3] * pB[3*4+0];
  REAL j = pA[2*4+0] * pB[0*4+1] + pA[2*4+1] * pB[1*4+1] + pA[2*4+2] * pB[2*4+1] + pA[2*4+3] * pB[3*4+1];
  REAL k = pA[2*4+0] * pB[0*4+2] + pA[2*4+1] * pB[1*4+2] + pA[2*4+2] * pB[2*4+2] + pA[2*4+3] * pB[3*4+2];
  REAL l = pA[2*4+0] * pB[0*4+3] + pA[2*4+1] * pB[1*4+3] + pA[2*4+2] * pB[2*4+3] + pA[2*4+3] * pB[3*4+3];

  REAL m = pA[3*4+0] * pB[0*4+0] + pA[3*4+1] * pB[1*4+0] + pA[3*4+2] * pB[2*4+0] + pA[3*4+3] * pB[3*4+0];
  REAL n = pA[3*4+0] * pB[0*4+1] + pA[3*4+1] * pB[1*4+1] + pA[3*4+2] * pB[2*4+1] + pA[3*4+3] * pB[3*4+1];
  REAL o = pA[3*4+0] * pB[0*4+2] + pA[3*4+1] * pB[1*4+2] + pA[3*4+2] * pB[2*4+2] + pA[3*4+3] * pB[3*4+2];
  REAL p = pA[3*4+0] * pB[0*4+3] + pA[3*4+1] * pB[1*4+3] + pA[3*4+2] * pB[2*4+3] + pA[3*4+3] * pB[3*4+3];

  pM[0] = a;  pM[1] = b;  pM[2] = c;  pM[3] = d;

  pM[4] = e;  pM[5] = f;  pM[6] = g;  pM[7] = h;

  pM[8] = i;  pM[9] = j;  pM[10] = k;  pM[11] = l;

  pM[12] = m;  pM[13] = n;  pM[14] = o;  pM[15] = p;
}

void fm_multiply(REAL *A,REAL scaler)
{
  A[0]*=scaler;
  A[1]*=scaler;
  A[2]*=scaler;
}

void fm_add(const REAL *A,const REAL *B,REAL *sum)
{
  sum[0] = A[0]+B[0];
  sum[1] = A[1]+B[1];
  sum[2] = A[2]+B[2];
}

void fm_copy3(const REAL *source,REAL *dest)
{
  dest[0] = source[0];
  dest[1] = source[1];
  dest[2] = source[2];
}


uint32_t  fm_copyUniqueVertices(uint32_t vcount,const REAL *input_vertices,REAL *output_vertices,uint32_t tcount,const uint32_t *input_indices,uint32_t *output_indices)
{
  uint32_t ret = 0;

  REAL *vertices = (REAL *)malloc(sizeof(REAL)*vcount*3);
  memcpy(vertices,input_vertices,sizeof(REAL)*vcount*3);
  REAL *dest = output_vertices;

  uint32_t *reindex = (uint32_t *)malloc(sizeof(uint32_t)*vcount);
  memset(reindex,0xFF,sizeof(uint32_t)*vcount);

  uint32_t icount = tcount*3;

  for (uint32_t i=0; i<icount; i++)
  {
	uint32_t index = *input_indices++;

	assert( index < vcount );

	if ( reindex[index] == 0xFFFFFFFF )
	{
	  *output_indices++ = ret;
	  reindex[index] = ret;
	  const REAL *pos = &vertices[index*3];
	  dest[0] = pos[0];
	  dest[1] = pos[1];
	  dest[2] = pos[2];
	  dest+=3;
	  ret++;
	}
	else
	{
	  *output_indices++ = reindex[index];
	}
  }
  free(vertices);
  free(reindex);
  return ret;
}

bool    fm_isMeshCoplanar(uint32_t tcount,const uint32_t *indices,const REAL *vertices,bool doubleSided) // returns true if this collection of indexed triangles are co-planar!
{
  bool ret = true;

  if ( tcount > 0 )
  {
	uint32_t i1 = indices[0];
	uint32_t i2 = indices[1];
	uint32_t i3 = indices[2];
	const REAL *p1 = &vertices[i1*3];
	const REAL *p2 = &vertices[i2*3];
	const REAL *p3 = &vertices[i3*3];
	REAL plane[4];
	plane[3] = fm_computePlane(p1,p2,p3,plane);
	const uint32_t *scan = &indices[3];
	for (uint32_t i=1; i<tcount; i++)
	{
	  i1 = *scan++;
	  i2 = *scan++;
	  i3 = *scan++;
	  p1 = &vertices[i1*3];
	  p2 = &vertices[i2*3];
	  p3 = &vertices[i3*3];
	  REAL _plane[4];
	  _plane[3] = fm_computePlane(p1,p2,p3,_plane);
	  if ( !fm_samePlane(plane,_plane,0.01f,0.001f,doubleSided) )
	  {
		ret = false;
		break;
	  }
	}
  }
  return ret;
}


bool fm_samePlane(const REAL p1[4],const REAL p2[4],REAL normalEpsilon,REAL dEpsilon,bool doubleSided)
{
  bool ret = false;

#if 0
  if (p1[0] == p2[0] &&
	  p1[1] == p2[1] &&
	  p1[2] == p2[2] &&
	  p1[3] == p2[3])
  {
	  ret = true;
  }
#else
  REAL diff = (REAL) fabs(p1[3]-p2[3]);
  if ( diff < dEpsilon ) // if the plane -d  co-efficient is within our epsilon
  {
	REAL dot = fm_dot(p1,p2); // compute the dot-product of the vector normals.
	if ( doubleSided ) dot = (REAL)fabs(dot);
	REAL dmin = 1 - normalEpsilon;
	REAL dmax = 1 + normalEpsilon;
	if ( dot >= dmin && dot <= dmax )
	{
	  ret = true; // then the plane equation is for practical purposes identical.
	}
  }
#endif
  return ret;
}


void  fm_initMinMax(REAL bmin[3],REAL bmax[3])
{
  bmin[0] = FLT_MAX;
  bmin[1] = FLT_MAX;
  bmin[2] = FLT_MAX;

  bmax[0] = -FLT_MAX;
  bmax[1] = -FLT_MAX;
  bmax[2] = -FLT_MAX;
}

void fm_inflateMinMax(REAL bmin[3], REAL bmax[3], REAL ratio)
{
	REAL inflate = fm_distance(bmin, bmax)*0.5f*ratio;

	bmin[0] -= inflate;
	bmin[1] -= inflate;
	bmin[2] -= inflate;

	bmax[0] += inflate;
	bmax[1] += inflate;
	bmax[2] += inflate;
}

#ifndef TESSELATE_H

#define TESSELATE_H

typedef std::vector< uint32_t > UintVector;

class Myfm_Tesselate : public fm_Tesselate
{
public:
  virtual ~Myfm_Tesselate(void)
  {

  }

  const uint32_t * tesselate(fm_VertexIndex *vindex,uint32_t tcount,const uint32_t *indices,float longEdge,uint32_t maxDepth,uint32_t &outcount)
  {
	const uint32_t *ret = 0;

	mMaxDepth = maxDepth;
	mLongEdge  = longEdge*longEdge;
	mLongEdgeD = mLongEdge;
	mVertices = vindex;

	if ( mVertices->isDouble() )
	{
	  uint32_t vcount = mVertices->getVcount();
	  double *vertices = (double *)malloc(sizeof(double)*vcount*3);
	  memcpy(vertices,mVertices->getVerticesDouble(),sizeof(double)*vcount*3);

	  for (uint32_t i=0; i<tcount; i++)
	  {
		uint32_t i1 = *indices++;
		uint32_t i2 = *indices++;
		uint32_t i3 = *indices++;

		const double *p1 = &vertices[i1*3];
		const double *p2 = &vertices[i2*3];
		const double *p3 = &vertices[i3*3];

		tesselate(p1,p2,p3,0);

	  }
	  free(vertices);
	}
	else
	{
	  uint32_t vcount = mVertices->getVcount();
	  float *vertices = (float *)malloc(sizeof(float)*vcount*3);
	  memcpy(vertices,mVertices->getVerticesFloat(),sizeof(float)*vcount*3);


	  for (uint32_t i=0; i<tcount; i++)
	  {
		uint32_t i1 = *indices++;
		uint32_t i2 = *indices++;
		uint32_t i3 = *indices++;

		const float *p1 = &vertices[i1*3];
		const float *p2 = &vertices[i2*3];
		const float *p3 = &vertices[i3*3];

		tesselate(p1,p2,p3,0);

	  }
	  free(vertices);
	}

	outcount = (uint32_t)(mIndices.size()/3);
	ret = &mIndices[0];


	return ret;
  }

  void tesselate(const float *p1,const float *p2,const float *p3,uint32_t recurse)
  {
	bool split = false;
	float l1,l2,l3;

	l1 = l2 = l3 = 0;

	if ( recurse < mMaxDepth )
	{
	  l1 = fm_distanceSquared(p1,p2);
		l2 = fm_distanceSquared(p2,p3);
		l3 = fm_distanceSquared(p3,p1);

	  if (  l1 > mLongEdge || l2 > mLongEdge || l3 > mLongEdge )
		split = true;

	}

	if ( split )
	{
		uint32_t edge;

		if ( l1 >= l2 && l1 >= l3 )
			edge = 0;
		else if ( l2 >= l1 && l2 >= l3 )
			edge = 1;
		else
			edge = 2;

			float splits[3];

		switch ( edge )
		{
			case 0:
				{
			fm_lerp(p1,p2,splits,0.5f);
			tesselate(p1,splits,p3, recurse+1 );
			tesselate(splits,p2,p3, recurse+1 );
				}
				break;
			case 1:
				{
			fm_lerp(p2,p3,splits,0.5f);
			tesselate(p1,p2,splits, recurse+1 );
			tesselate(p1,splits,p3, recurse+1 );
				}
				break;
			case 2:
				{
					fm_lerp(p3,p1,splits,0.5f);
			tesselate(p1,p2,splits, recurse+1 );
			tesselate(splits,p2,p3, recurse+1 );
				}
				break;
		}
	}
	else
	{
	  bool newp;

	  uint32_t i1 = mVertices->getIndex(p1,newp);
	  uint32_t i2 = mVertices->getIndex(p2,newp);
	  uint32_t i3 = mVertices->getIndex(p3,newp);

	  mIndices.push_back(i1);
	  mIndices.push_back(i2);
	  mIndices.push_back(i3);
	}

  }

  void tesselate(const double *p1,const double *p2,const double *p3,uint32_t recurse)
  {
	bool split = false;
	double l1,l2,l3;

	l1 = l2 = l3 = 0;

	if ( recurse < mMaxDepth )
	{
	  l1 = fm_distanceSquared(p1,p2);
		l2 = fm_distanceSquared(p2,p3);
		l3 = fm_distanceSquared(p3,p1);

	  if (  l1 > mLongEdgeD || l2 > mLongEdgeD || l3 > mLongEdgeD )
		split = true;

	}

	if ( split )
	{
		uint32_t edge;

		if ( l1 >= l2 && l1 >= l3 )
			edge = 0;
		else if ( l2 >= l1 && l2 >= l3 )
			edge = 1;
		else
			edge = 2;

			double splits[3];

		switch ( edge )
		{
			case 0:
				{
			fm_lerp(p1,p2,splits,0.5);
			tesselate(p1,splits,p3, recurse+1 );
			tesselate(splits,p2,p3, recurse+1 );
				}
				break;
			case 1:
				{
			fm_lerp(p2,p3,splits,0.5);
			tesselate(p1,p2,splits, recurse+1 );
			tesselate(p1,splits,p3, recurse+1 );
				}
				break;
			case 2:
				{
					fm_lerp(p3,p1,splits,0.5);
			tesselate(p1,p2,splits, recurse+1 );
			tesselate(splits,p2,p3, recurse+1 );
				}
				break;
		}
	}
	else
	{
	  bool newp;

	  uint32_t i1 = mVertices->getIndex(p1,newp);
	  uint32_t i2 = mVertices->getIndex(p2,newp);
	  uint32_t i3 = mVertices->getIndex(p3,newp);

	  mIndices.push_back(i1);
	  mIndices.push_back(i2);
	  mIndices.push_back(i3);
	}

  }

private:
  float           mLongEdge;
  double          mLongEdgeD;
  fm_VertexIndex *mVertices;
  UintVector    mIndices;
  uint32_t          mMaxDepth;
};

fm_Tesselate * fm_createTesselate(void)
{
  Myfm_Tesselate *m = new Myfm_Tesselate;
  return static_cast< fm_Tesselate * >(m);
}

void           fm_releaseTesselate(fm_Tesselate *t)
{
  Myfm_Tesselate *m = static_cast< Myfm_Tesselate *>(t);
  delete m;
}

#endif


#ifndef RAY_ABB_INTERSECT

#define RAY_ABB_INTERSECT

//! Integer representation of a floating-point value.
#define IR(x)	((uint32_t&)x)

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
*	A method to compute a ray-AABB intersection.
*	Original code by Andrew Woo, from "Graphics Gems", Academic Press, 1990
*	Optimized code by Pierre Terdiman, 2000 (~20-30% faster on my Celeron 500)
*	Epsilon value added by Klaus Hartmann. (discarding it saves a few cycles only)
*
*	Hence this version is faster as well as more robust than the original one.
*
*	Should work provided:
*	1) the integer representation of 0.0f is 0x00000000
*	2) the sign bit of the float is the most significant one
*
*	Report bugs: p.terdiman@codercorner.com
*
*	\param		aabb		[in] the axis-aligned bounding box
*	\param		origin		[in] ray origin
*	\param		dir			[in] ray direction
*	\param		coord		[out] impact coordinates
*	\return		true if ray intersects AABB
*/
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define RAYAABB_EPSILON 0.00001f
bool fm_intersectRayAABB(const float MinB[3],const float MaxB[3],const float origin[3],const float dir[3],float coord[3])
{
  bool Inside = true;
  float MaxT[3];
  MaxT[0]=MaxT[1]=MaxT[2]=-1.0f;

  // Find candidate planes.
  for(uint32_t i=0;i<3;i++)
  {
	if(origin[i] < MinB[i])
	{
	  coord[i]	= MinB[i];
	  Inside		= false;

	  // Calculate T distances to candidate planes
	  if(IR(dir[i]))	MaxT[i] = (MinB[i] - origin[i]) / dir[i];
	}
	else if(origin[i] > MaxB[i])
	{
	  coord[i]	= MaxB[i];
	  Inside		= false;

	  // Calculate T distances to candidate planes
	  if(IR(dir[i]))	MaxT[i] = (MaxB[i] - origin[i]) / dir[i];
	}
  }

  // Ray origin inside bounding box
  if(Inside)
  {
	coord[0] = origin[0];
	coord[1] = origin[1];
	coord[2] = origin[2];
	return true;
  }

  // Get largest of the maxT's for final choice of intersection
  uint32_t WhichPlane = 0;
  if(MaxT[1] > MaxT[WhichPlane])	WhichPlane = 1;
  if(MaxT[2] > MaxT[WhichPlane])	WhichPlane = 2;

  // Check final candidate actually inside box
  if(IR(MaxT[WhichPlane])&0x80000000) return false;

  for(uint32_t i=0;i<3;i++)
  {
	if(i!=WhichPlane)
	{
	  coord[i] = origin[i] + MaxT[WhichPlane] * dir[i];
#ifdef RAYAABB_EPSILON
	  if(coord[i] < MinB[i] - RAYAABB_EPSILON || coord[i] > MaxB[i] + RAYAABB_EPSILON)	return false;
#else
	  if(coord[i] < MinB[i] || coord[i] > MaxB[i])	return false;
#endif
	}
  }
  return true;	// ray hits box
}

bool fm_intersectLineSegmentAABB(const float bmin[3],const float bmax[3],const float p1[3],const float p2[3],float intersect[3])
{
  bool ret = false;

  float dir[3];
  dir[0] = p2[0] - p1[0];
  dir[1] = p2[1] - p1[1];
  dir[2] = p2[2] - p1[2];
  float dist = fm_normalize(dir);
  if ( dist > RAYAABB_EPSILON )
  {
	ret = fm_intersectRayAABB(bmin,bmax,p1,dir,intersect);
	if ( ret )
	{
	  float d = fm_distanceSquared(p1,intersect);
	  if ( d  > (dist*dist) )
	  {
		ret = false;
	  }
	}
  }
  return ret;
}

#endif

#ifndef OBB_TO_AABB

#define OBB_TO_AABB

#pragma warning(disable:4100)

void    fm_OBBtoAABB(const float /*obmin*/[3],const float /*obmax*/[3],const float /*matrix*/[16],float /*abmin*/[3],float /*abmax*/[3])
{
  assert(0); // not yet implemented.
}


const REAL * computePos(uint32_t index,const REAL *vertices,uint32_t vstride)
{
  const char *tmp = (const char *)vertices;
  tmp+=(index*vstride);
  return (const REAL*)tmp;
}

void computeNormal(uint32_t index,REAL *normals,uint32_t nstride,const REAL *normal)
{
  char *tmp = (char *)normals;
  tmp+=(index*nstride);
  REAL *dest = (REAL *)tmp;
  dest[0]+=normal[0];
  dest[1]+=normal[1];
  dest[2]+=normal[2];
}

void fm_computeMeanNormals(uint32_t vcount,       // the number of vertices
						   const REAL *vertices,     // the base address of the vertex position data.
						   uint32_t vstride,      // the stride between position data.
						   REAL *normals,            // the base address  of the destination for mean vector normals
						   uint32_t nstride,      // the stride between normals
						   uint32_t tcount,       // the number of triangles
						   const uint32_t *indices)     // the triangle indices
{

  // Step #1 : Zero out the vertex normals
  char *dest = (char *)normals;
  for (uint32_t i=0; i<vcount; i++)
  {
	REAL *n = (REAL *)dest;
	n[0] = 0;
	n[1] = 0;
	n[2] = 0;
	dest+=nstride;
  }

  // Step #2 : Compute the face normals and accumulate them
  const uint32_t *scan = indices;
  for (uint32_t i=0; i<tcount; i++)
  {

	uint32_t i1 = *scan++;
	uint32_t i2 = *scan++;
	uint32_t i3 = *scan++;

	const REAL *p1 = computePos(i1,vertices,vstride);
	const REAL *p2 = computePos(i2,vertices,vstride);
	const REAL *p3 = computePos(i3,vertices,vstride);

	REAL normal[3];
	fm_computePlane(p3,p2,p1,normal);

	computeNormal(i1,normals,nstride,normal);
	computeNormal(i2,normals,nstride,normal);
	computeNormal(i3,normals,nstride,normal);
  }


  // Normalize the accumulated normals
  dest = (char *)normals;
  for (uint32_t i=0; i<vcount; i++)
  {
	REAL *n = (REAL *)dest;
	fm_normalize(n);
	dest+=nstride;
  }

}

#endif


#define BIGNUMBER 100000000.0  		/* hundred million */

static inline void Set(REAL *n,REAL x,REAL y,REAL z)
{
	n[0] = x;
	n[1] = y;
	n[2] = z;
};

static inline void Copy(REAL *dest,const REAL *source)
{
	dest[0] = source[0];
	dest[1] = source[1];
	dest[2] = source[2];
}


REAL  fm_computeBestFitSphere(uint32_t vcount,const REAL *points,uint32_t pstride,REAL *center)
{
	REAL radius;
	REAL radius2;

	REAL xmin[3];
	REAL xmax[3];
	REAL ymin[3];
	REAL ymax[3];
	REAL zmin[3];
	REAL zmax[3];
	REAL dia1[3];
	REAL dia2[3];

	/* FIRST PASS: find 6 minima/maxima points */
	Set(xmin,BIGNUMBER,BIGNUMBER,BIGNUMBER);
	Set(xmax,-BIGNUMBER,-BIGNUMBER,-BIGNUMBER);
	Set(ymin,BIGNUMBER,BIGNUMBER,BIGNUMBER);
	Set(ymax,-BIGNUMBER,-BIGNUMBER,-BIGNUMBER);
	Set(zmin,BIGNUMBER,BIGNUMBER,BIGNUMBER);
	Set(zmax,-BIGNUMBER,-BIGNUMBER,-BIGNUMBER);

	{
		const char *scan = (const char *)points;
		for (uint32_t i=0; i<vcount; i++)
		{
			const REAL *caller_p = (const REAL *)scan;
			if (caller_p[0]<xmin[0])
				Copy(xmin,caller_p); /* New xminimum point */
			if (caller_p[0]>xmax[0])
				Copy(xmax,caller_p);
			if (caller_p[1]<ymin[1])
				Copy(ymin,caller_p);
			if (caller_p[1]>ymax[1])
				Copy(ymax,caller_p);
			if (caller_p[2]<zmin[2])
				Copy(zmin,caller_p);
			if (caller_p[2]>zmax[2])
				Copy(zmax,caller_p);
			scan+=pstride;
		}
	}

	/* Set xspan = distance between the 2 points xmin & xmax (squared) */
	REAL dx = xmax[0] - xmin[0];
	REAL dy = xmax[1] - xmin[1];
	REAL dz = xmax[2] - xmin[2];
	REAL xspan = dx*dx + dy*dy + dz*dz;

/* Same for y & z spans */
	dx = ymax[0] - ymin[0];
	dy = ymax[1] - ymin[1];
	dz = ymax[2] - ymin[2];
	REAL yspan = dx*dx + dy*dy + dz*dz;

	dx = zmax[0] - zmin[0];
	dy = zmax[1] - zmin[1];
	dz = zmax[2] - zmin[2];
	REAL zspan = dx*dx + dy*dy + dz*dz;

	/* Set points dia1 & dia2 to the maximally separated pair */
	Copy(dia1,xmin);
	Copy(dia2,xmax); /* assume xspan biggest */
	REAL maxspan = xspan;

	if (yspan>maxspan)
	{
		maxspan = yspan;
		Copy(dia1,ymin);
		Copy(dia2,ymax);
	}

	if (zspan>maxspan)
	{
		maxspan = zspan;
		Copy(dia1,zmin);
		Copy(dia2,zmax);
	}


	/* dia1,dia2 is a diameter of initial sphere */
	/* calc initial center */
	center[0] = (dia1[0]+dia2[0])*0.5f;
	center[1] = (dia1[1]+dia2[1])*0.5f;
	center[2] = (dia1[2]+dia2[2])*0.5f;

	/* calculate initial radius**2 and radius */

	dx = dia2[0]-center[0]; /* x component of radius vector */
	dy = dia2[1]-center[1]; /* y component of radius vector */
	dz = dia2[2]-center[2]; /* z component of radius vector */

	radius2 = dx*dx + dy*dy + dz*dz;
	radius = REAL(sqrt(radius2));

	/* SECOND PASS: increment current sphere */
	{
		const char *scan = (const char *)points;
		for (uint32_t i=0; i<vcount; i++)
		{
			const REAL *caller_p = (const REAL *)scan;
			dx = caller_p[0]-center[0];
			dy = caller_p[1]-center[1];
			dz = caller_p[2]-center[2];
			REAL old_to_p_sq = dx*dx + dy*dy + dz*dz;
			if (old_to_p_sq > radius2) 	/* do r**2 test first */
			{ 	/* this point is outside of current sphere */
				REAL old_to_p = REAL(sqrt(old_to_p_sq));
				/* calc radius of new sphere */
				radius = (radius + old_to_p) * 0.5f;
				radius2 = radius*radius; 	/* for next r**2 compare */
				REAL old_to_new = old_to_p - radius;
				/* calc center of new sphere */
				REAL recip = 1.0f /old_to_p;
				REAL cx = (radius*center[0] + old_to_new*caller_p[0]) * recip;
				REAL cy = (radius*center[1] + old_to_new*caller_p[1]) * recip;
				REAL cz = (radius*center[2] + old_to_new*caller_p[2]) * recip;
				Set(center,cx,cy,cz);
				scan+=pstride;
			}
		}
	}
	return radius;
}


void fm_computeBestFitCapsule(uint32_t vcount,const REAL *points,uint32_t pstride,REAL &radius,REAL &height,REAL matrix[16],bool bruteForce)
{
  REAL sides[3];
  REAL omatrix[16];
  fm_computeBestFitOBB(vcount,points,pstride,sides,omatrix,bruteForce);

  int32_t axis = 0;
  if ( sides[0] > sides[1] && sides[0] > sides[2] )
	axis = 0;
  else if ( sides[1] > sides[0] && sides[1] > sides[2] )
	axis = 1;
  else 
	axis = 2;

  REAL localTransform[16];

  REAL maxDist = 0;
  REAL maxLen = 0;

  switch ( axis )
  {
	case 0:
	  {
		fm_eulerMatrix(0,0,FM_PI/2,localTransform);
		fm_matrixMultiply(localTransform,omatrix,matrix);

		const uint8_t *scan = (const uint8_t *)points;
		for (uint32_t i=0; i<vcount; i++)
		{
		  const REAL *p = (const REAL *)scan;
		  REAL t[3];
		  fm_inverseRT(omatrix,p,t);
		  REAL dist = t[1]*t[1]+t[2]*t[2];
		  if ( dist > maxDist )
		  {
			maxDist = dist;
		  }
		  REAL l = (REAL) fabs(t[0]);
		  if ( l > maxLen )
		  {
			maxLen = l;
		  }
		  scan+=pstride;
		}
	  }
	  height = sides[0];
	  break;
	case 1:
	  {
		fm_eulerMatrix(0,FM_PI/2,0,localTransform);
		fm_matrixMultiply(localTransform,omatrix,matrix);

		const uint8_t *scan = (const uint8_t *)points;
		for (uint32_t i=0; i<vcount; i++)
		{
		  const REAL *p = (const REAL *)scan;
		  REAL t[3];
		  fm_inverseRT(omatrix,p,t);
		  REAL dist = t[0]*t[0]+t[2]*t[2];
		  if ( dist > maxDist )
		  {
			maxDist = dist;
		  }
		  REAL l = (REAL) fabs(t[1]);
		  if ( l > maxLen )
		  {
			maxLen = l;
		  }
		  scan+=pstride;
		}
	  }
	  height = sides[1];
	  break;
	case 2:
	  {
		fm_eulerMatrix(FM_PI/2,0,0,localTransform);
		fm_matrixMultiply(localTransform,omatrix,matrix);

		const uint8_t *scan = (const uint8_t *)points;
		for (uint32_t i=0; i<vcount; i++)
		{
		  const REAL *p = (const REAL *)scan;
		  REAL t[3];
		  fm_inverseRT(omatrix,p,t);
		  REAL dist = t[0]*t[0]+t[1]*t[1];
		  if ( dist > maxDist )
		  {
			maxDist = dist;
		  }
		  REAL l = (REAL) fabs(t[2]);
		  if ( l > maxLen )
		  {
			maxLen = l;
		  }
		  scan+=pstride;
		}
	  }
	  height = sides[2];
	  break;
  }
  radius = (REAL)sqrt(maxDist);
  height = (maxLen*2)-(radius*2);
}


//************* Triangulation

#ifndef TRIANGULATE_H

#define TRIANGULATE_H

typedef uint32_t TU32;

class TVec
{
public:
	TVec(double _x,double _y,double _z) { x = _x; y = _y; z = _z; };
	TVec(void) { };

  double x;
  double y;
  double z;
};

typedef std::vector< TVec >  TVecVector;
typedef std::vector< TU32 >  TU32Vector;

class CTriangulator
{
public:
	///     Default constructor
	CTriangulator();

	///     Default destructor
	virtual ~CTriangulator();

	///     Triangulates the contour
	void triangulate(TU32Vector &indices);

	///     Returns the given point in the triangulator array
	inline TVec get(const TU32 id) { return mPoints[id]; }

	virtual void reset(void)
	{
		mInputPoints.clear();
		mPoints.clear();
		mIndices.clear();
	}

	virtual void addPoint(double x,double y,double z)
	{
		TVec v(x,y,z);
		// update bounding box...
		if ( mInputPoints.empty() )
		{
			mMin = v;
			mMax = v;
		}
		else
		{
			if ( x < mMin.x ) mMin.x = x;
			if ( y < mMin.y ) mMin.y = y;
			if ( z < mMin.z ) mMin.z = z;

			if ( x > mMax.x ) mMax.x = x;
			if ( y > mMax.y ) mMax.y = y;
			if ( z > mMax.z ) mMax.z = z;
		}
		mInputPoints.push_back(v);
	}

	// Triangulation happens in 2d.  We could inverse transform the polygon around the normal direction, or we just use the two most signficant axes
	// Here we find the two longest axes and use them to triangulate.  Inverse transforming them would introduce more doubleing point error and isn't worth it.
	virtual uint32_t * triangulate(uint32_t &tcount,double epsilon)
	{
		uint32_t *ret = 0;
		tcount = 0;
		mEpsilon = epsilon;

		if ( !mInputPoints.empty() )
		{
			mPoints.clear();

		  double dx = mMax.x - mMin.x; // locate the first, second and third longest edges and store them in i1, i2, i3
		  double dy = mMax.y - mMin.y;
		  double dz = mMax.z - mMin.z;

		  uint32_t i1,i2,i3;

		  if ( dx > dy && dx > dz )
		  {
			  i1 = 0;
			  if ( dy > dz )
			  {
				  i2 = 1;
				  i3 = 2;
			  }
			  else
			  {
				  i2 = 2;
				  i3 = 1;
			  }
		  }
		  else if ( dy > dx && dy > dz )
		  {
			  i1 = 1;
			  if ( dx > dz )
			  {
				  i2 = 0;
				  i3 = 2;
			  }
			  else
			  {
				  i2 = 2;
				  i3 = 0;
			  }
		  }
		  else
		  {
			  i1 = 2;
			  if ( dx > dy )
			  {
				  i2 = 0;
				  i3 = 1;
			  }
			  else
			  {
				  i2 = 1;
				  i3 = 0;
			  }
		  }

		  uint32_t pcount = (uint32_t)mInputPoints.size();
		  const double *points = &mInputPoints[0].x;
		  for (uint32_t i=0; i<pcount; i++)
		  {
			TVec v( points[i1], points[i2], points[i3] );
			mPoints.push_back(v);
			points+=3;
		  }

		  mIndices.clear();
		  triangulate(mIndices);
		  tcount = (uint32_t)mIndices.size()/3;
		  if ( tcount )
		  {
			  ret = &mIndices[0];
		  }
		}
		return ret;
	}

	virtual const double * getPoint(uint32_t index)
	{
		return &mInputPoints[index].x;
	}


private:
	double                  mEpsilon;
	TVec                   mMin;
	TVec                   mMax;
	TVecVector             mInputPoints;
	TVecVector             mPoints;
	TU32Vector             mIndices;

	///     Tests if a point is inside the given triangle
	bool _insideTriangle(const TVec& A, const TVec& B, const TVec& C,const TVec& P);

	///     Returns the area of the contour
	double _area();

	bool _snip(int32_t u, int32_t v, int32_t w, int32_t n, int32_t *V);

	///     Processes the triangulation
	void _process(TU32Vector &indices);

};

///     Default constructor
CTriangulator::CTriangulator(void)
{
}

///     Default destructor
CTriangulator::~CTriangulator()
{
}

///     Triangulates the contour
void CTriangulator::triangulate(TU32Vector &indices)
{
	_process(indices);
}

///     Processes the triangulation
void CTriangulator::_process(TU32Vector &indices)
{
	const int32_t n = (const int32_t)mPoints.size();
	if (n < 3)
		return;
	int32_t *V = (int32_t *)malloc(sizeof(int32_t)*n);

	bool flipped = false;

	if (0.0f < _area())
	{
		for (int32_t v = 0; v < n; v++)
			V[v] = v;
	}
	else
	{
		flipped = true;
		for (int32_t v = 0; v < n; v++)
			V[v] = (n - 1) - v;
	}

	int32_t nv = n;
	int32_t count = 2 * nv;
	for (int32_t m = 0, v = nv - 1; nv > 2;)
	{
		if (0 >= (count--))
			return;

		int32_t u = v;
		if (nv <= u)
			u = 0;
		v = u + 1;
		if (nv <= v)
			v = 0;
		int32_t w = v + 1;
		if (nv <= w)
			w = 0;

		if (_snip(u, v, w, nv, V))
		{
			int32_t a, b, c, s, t;
			a = V[u];
			b = V[v];
			c = V[w];
			if ( flipped )
			{
				indices.push_back(a);
				indices.push_back(b);
				indices.push_back(c);
			}
			else
			{
				indices.push_back(c);
				indices.push_back(b);
				indices.push_back(a);
			}
			m++;
			for (s = v, t = v + 1; t < nv; s++, t++)
				V[s] = V[t];
			nv--;
			count = 2 * nv;
		}
	}

	free(V);
}

///     Returns the area of the contour
double CTriangulator::_area()
{
	int32_t n = (uint32_t)mPoints.size();
	double A = 0.0f;
	for (int32_t p = n - 1, q = 0; q < n; p = q++)
	{
		const TVec &pval = mPoints[p];
		const TVec &qval = mPoints[q];
		A += pval.x * qval.y - qval.x * pval.y;
	}
	A*=0.5f;
	return A;
}

bool CTriangulator::_snip(int32_t u, int32_t v, int32_t w, int32_t n, int32_t *V)
{
	int32_t p;

	const TVec &A = mPoints[ V[u] ];
	const TVec &B = mPoints[ V[v] ];
	const TVec &C = mPoints[ V[w] ];

	if (mEpsilon > (((B.x - A.x) * (C.y - A.y)) - ((B.y - A.y) * (C.x - A.x))) )
		return false;

	for (p = 0; p < n; p++)
	{
		if ((p == u) || (p == v) || (p == w))
			continue;
		const TVec &P = mPoints[ V[p] ];
		if (_insideTriangle(A, B, C, P))
			return false;
	}
	return true;
}

///     Tests if a point is inside the given triangle
bool CTriangulator::_insideTriangle(const TVec& A, const TVec& B, const TVec& C,const TVec& P)
{
	double ax, ay, bx, by, cx, cy, apx, apy, bpx, bpy, cpx, cpy;
	double cCROSSap, bCROSScp, aCROSSbp;

	ax = C.x - B.x;  ay = C.y - B.y;
	bx = A.x - C.x;  by = A.y - C.y;
	cx = B.x - A.x;  cy = B.y - A.y;
	apx = P.x - A.x;  apy = P.y - A.y;
	bpx = P.x - B.x;  bpy = P.y - B.y;
	cpx = P.x - C.x;  cpy = P.y - C.y;

	aCROSSbp = ax * bpy - ay * bpx;
	cCROSSap = cx * apy - cy * apx;
	bCROSScp = bx * cpy - by * cpx;

	return ((aCROSSbp >= 0.0f) && (bCROSScp >= 0.0f) && (cCROSSap >= 0.0f));
}

class Triangulate : public fm_Triangulate
{
public:
  Triangulate(void)
  {
	mPointsFloat = 0;
	mPointsDouble = 0;
  }

  virtual ~Triangulate(void)
  {
	reset();
  }
  void reset(void)
  {
	free(mPointsFloat);
	free(mPointsDouble);
	mPointsFloat = 0;
	mPointsDouble = 0;
  }

  virtual const double *       triangulate3d(uint32_t pcount,
											 const double *_points,
											 uint32_t vstride,
											 uint32_t &tcount,
											 bool consolidate,
											 double epsilon)
  {
	reset();

	double *points = (double *)malloc(sizeof(double)*pcount*3);
	if ( consolidate )
	{
	  pcount = fm_consolidatePolygon(pcount,_points,vstride,points,1-epsilon);
	}
	else
	{
	  double *dest = points;
	  for (uint32_t i=0; i<pcount; i++)
	  {
		const double *src = fm_getPoint(_points,vstride,i);
		dest[0] = src[0];
		dest[1] = src[1];
		dest[2] = src[2];
		dest+=3;
	  }
	  vstride = sizeof(double)*3;
	}

	if ( pcount >= 3 )
	{
	  CTriangulator ct;
	  for (uint32_t i=0; i<pcount; i++)
	  {
		const double *src = fm_getPoint(points,vstride,i);
		ct.addPoint( src[0], src[1], src[2] );
	  }
	  uint32_t _tcount;
	  uint32_t *indices = ct.triangulate(_tcount,epsilon);
	  if ( indices )
	  {
		tcount = _tcount;
		mPointsDouble = (double *)malloc(sizeof(double)*tcount*3*3);
		double *dest = mPointsDouble;
		for (uint32_t i=0; i<tcount; i++)
		{
		  uint32_t i1 = indices[i*3+0];
		  uint32_t i2 = indices[i*3+1];
		  uint32_t i3 = indices[i*3+2];
		  const double *p1 = ct.getPoint(i1);
		  const double *p2 = ct.getPoint(i2);
		  const double *p3 = ct.getPoint(i3);

		  dest[0] = p1[0];
		  dest[1] = p1[1];
		  dest[2] = p1[2];

		  dest[3] = p2[0];
		  dest[4] = p2[1];
		  dest[5] = p2[2];

		  dest[6] = p3[0];
		  dest[7] = p3[1];
		  dest[8] = p3[2];
		  dest+=9;
		}
	  }
	}
	free(points);

	return mPointsDouble;
  }

  virtual const float  *       triangulate3d(uint32_t pcount,
											 const float  *points,
											 uint32_t vstride,
											 uint32_t &tcount,
											 bool consolidate,
											 float epsilon)
  {
	reset();

	double *temp = (double *)malloc(sizeof(double)*pcount*3);
	double *dest = temp;
	for (uint32_t i=0; i<pcount; i++)
	{
	  const float *p = fm_getPoint(points,vstride,i);
	  dest[0] = p[0];
	  dest[1] = p[1];
	  dest[2] = p[2];
	  dest+=3;
	}
	const double *results = triangulate3d(pcount,temp,sizeof(double)*3,tcount,consolidate,epsilon);
	if ( results )
	{
	  uint32_t fcount = tcount*3*3;
	  mPointsFloat = (float *)malloc(sizeof(float)*tcount*3*3);
	  for (uint32_t i=0; i<fcount; i++)
	  {
		mPointsFloat[i] = (float) results[i];
	  }
	  free(mPointsDouble);
	  mPointsDouble = 0;
	}
	free(temp);

	return mPointsFloat;
  }

private:
  float *mPointsFloat;
  double *mPointsDouble;
};

fm_Triangulate * fm_createTriangulate(void)
{
  Triangulate *t = new Triangulate;
  return static_cast< fm_Triangulate *>(t);
}

void             fm_releaseTriangulate(fm_Triangulate *t)
{
  Triangulate *tt = static_cast< Triangulate *>(t);
  delete tt;
}

#endif

bool validDistance(const REAL *p1,const REAL *p2,REAL epsilon)
{
	bool ret = true;

	REAL dx = p1[0] - p2[0];
	REAL dy = p1[1] - p2[1];
	REAL dz = p1[2] - p2[2];
	REAL dist = dx*dx+dy*dy+dz*dz;
	if ( dist < (epsilon*epsilon) )
	{
		ret = false;
	}
	return ret;
}

bool fm_isValidTriangle(const REAL *p1,const REAL *p2,const REAL *p3,REAL epsilon)
{
  bool ret = false;

  if ( validDistance(p1,p2,epsilon) &&
	   validDistance(p1,p3,epsilon) &&
	   validDistance(p2,p3,epsilon) )
  {

	  REAL area = fm_computeArea(p1,p2,p3);
	  if ( area > epsilon )
	  {
		REAL _vertices[3*3],vertices[64*3];

		_vertices[0] = p1[0];
		_vertices[1] = p1[1];
		_vertices[2] = p1[2];

		_vertices[3] = p2[0];
		_vertices[4] = p2[1];
		_vertices[5] = p2[2];

		_vertices[6] = p3[0];
		_vertices[7] = p3[1];
		_vertices[8] = p3[2];

		uint32_t pcount = fm_consolidatePolygon(3,_vertices,sizeof(REAL)*3,vertices,1-epsilon);
		if ( pcount == 3 )
		{
		  ret = true;
		}
	  }
  }
  return ret;
}


void  fm_multiplyQuat(const REAL *left,const REAL *right,REAL *quat)
{
	REAL a,b,c,d;

	a = left[3]*right[3] - left[0]*right[0] - left[1]*right[1] - left[2]*right[2];
	b = left[3]*right[0] + right[3]*left[0] + left[1]*right[2] - right[1]*left[2];
	c = left[3]*right[1] + right[3]*left[1] + left[2]*right[0] - right[2]*left[0];
	d = left[3]*right[2] + right[3]*left[2] + left[0]*right[1] - right[0]*left[1];

	quat[3] = a;
	quat[0] = b;
	quat[1] = c;
	quat[2] = d;
}

bool  fm_computeCentroid(uint32_t vcount,     // number of input data points
						 const REAL *points,     // starting address of points array.
						 REAL *center)

{
	bool ret = false;
	if ( vcount )
	{
		center[0] = 0;
		center[1] = 0;
		center[2] = 0;
		const REAL *p = points;
		for (uint32_t i=0; i<vcount; i++)
		{
			center[0]+=p[0];
			center[1]+=p[1];
			center[2]+=p[2];
			p += 3;
		}
		REAL recip = 1.0f / (REAL)vcount;
		center[0]*=recip;
		center[1]*=recip;
		center[2]*=recip;
		ret = true;
	}
	return ret;
}

bool  fm_computeCentroid(uint32_t vcount,     // number of input data points
	const REAL *points,     // starting address of points array.
	uint32_t triCount,
	const uint32_t *indices,
	REAL *center)

{
	bool ret = false;
	if (vcount)
	{
		center[0] = 0;
		center[1] = 0;
		center[2] = 0;

		REAL numerator[3] = { 0, 0, 0 };
		REAL denomintaor = 0;

		for (uint32_t i = 0; i < triCount; i++)
		{
			uint32_t i1 = indices[i * 3 + 0];
			uint32_t i2 = indices[i * 3 + 1];
			uint32_t i3 = indices[i * 3 + 2];

			const REAL *p1 = &points[i1 * 3];
			const REAL *p2 = &points[i2 * 3];
			const REAL *p3 = &points[i3 * 3];

			// Compute the sum of the three positions
			REAL sum[3];
			sum[0] = p1[0] + p2[0] + p3[0];
			sum[1] = p1[1] + p2[1] + p3[1];
			sum[2] = p1[2] + p2[2] + p3[2];

			// Compute the average of the three positions
			sum[0] = sum[0] / 3;
			sum[1] = sum[1] / 3;
			sum[2] = sum[2] / 3;

			// Compute the area of this triangle
			REAL area = fm_computeArea(p1, p2, p3);

			numerator[0]+= (sum[0] * area);
			numerator[1]+= (sum[1] * area);
			numerator[2]+= (sum[2] * area);

			denomintaor += area;

		}
		REAL recip = 1 / denomintaor;
		center[0] = numerator[0] * recip;
		center[1] = numerator[1] * recip;
		center[2] = numerator[2] * recip;
		ret = true;
	}
	return ret;
}


#ifndef TEMPLATE_VEC3
#define TEMPLATE_VEC3
template <class Type> class Vec3
{
public:
	Vec3(void)
	{

	}
	Vec3(Type _x,Type _y,Type _z)
	{
		x = _x;
		y = _y;
		z = _z;
	}
	Type x;
	Type y;
	Type z;
};
#endif

void fm_transformAABB(const REAL bmin[3],const REAL bmax[3],const REAL matrix[16],REAL tbmin[3],REAL tbmax[3])
{
	Vec3<REAL> box[8];
	box[0] = Vec3< REAL >( bmin[0], bmin[1], bmin[2] );
	box[1] = Vec3< REAL >( bmax[0], bmin[1], bmin[2] );
	box[2] = Vec3< REAL >( bmax[0], bmax[1], bmin[2] );
	box[3] = Vec3< REAL >( bmin[0], bmax[1], bmin[2] );
	box[4] = Vec3< REAL >( bmin[0], bmin[1], bmax[2] );
	box[5] = Vec3< REAL >( bmax[0], bmin[1], bmax[2] );
	box[6] = Vec3< REAL >( bmax[0], bmax[1], bmax[2] );
	box[7] = Vec3< REAL >( bmin[0], bmax[1], bmax[2] );
	// transform all 8 corners of the box and then recompute a new AABB
	for (unsigned int i=0; i<8; i++)
	{
		Vec3< REAL > &p = box[i];
		fm_transform(matrix,&p.x,&p.x);
		if ( i == 0 )
		{
			tbmin[0] = tbmax[0] = p.x;
			tbmin[1] = tbmax[1] = p.y;
			tbmin[2] = tbmax[2] = p.z;
		}
		else
		{
			if ( p.x < tbmin[0] ) tbmin[0] = p.x;
			if ( p.y < tbmin[1] ) tbmin[1] = p.y;
			if ( p.z < tbmin[2] ) tbmin[2] = p.z;
			if ( p.x > tbmax[0] ) tbmax[0] = p.x;
			if ( p.y > tbmax[1] ) tbmax[1] = p.y;
			if ( p.z > tbmax[2] ) tbmax[2] = p.z;
		}
	}
}

REAL  fm_normalizeQuat(REAL n[4]) // normalize this quat
{
	REAL dx = n[0]*n[0];
	REAL dy = n[1]*n[1];
	REAL dz = n[2]*n[2];
	REAL dw = n[3]*n[3];

	REAL dist = dx*dx+dy*dy+dz*dz+dw*dw;

	dist = (REAL)sqrt(dist);

	REAL recip = 1.0f / dist;

	n[0]*=recip;
	n[1]*=recip;
	n[2]*=recip;
	n[3]*=recip;

	return dist;
}


}; // end of namespace
