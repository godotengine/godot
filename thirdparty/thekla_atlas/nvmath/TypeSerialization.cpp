// This code is in the public domain -- Ignacio Castaño <castano@gmail.com>

#include "TypeSerialization.h"

#include "nvcore/Stream.h"

#include "nvmath/Vector.h"
#include "nvmath/Matrix.h"
#include "nvmath/Quaternion.h"
#include "nvmath/Basis.h"
#include "nvmath/Box.h"
#include "nvmath/Plane.inl"

using namespace nv;

Stream & nv::operator<< (Stream & s, Vector2 & v)
{
    return s << v.x << v.y;
}

Stream & nv::operator<< (Stream & s, Vector3 & v)
{
    return s << v.x << v.y << v.z;
}

Stream & nv::operator<< (Stream & s, Vector4 & v)
{
    return s << v.x << v.y << v.z << v.w;
}

Stream & nv::operator<< (Stream & s, Matrix & m)
{
    return s;
}

Stream & nv::operator<< (Stream & s, Quaternion & q)
{
    return s << q.x << q.y << q.z << q.w;
}

Stream & nv::operator<< (Stream & s, Basis & basis)
{
    return s << basis.tangent << basis.bitangent << basis.normal;
}

Stream & nv::operator<< (Stream & s, Box & box)
{
    return s << box.minCorner << box.maxCorner;
}

Stream & nv::operator<< (Stream & s, Plane & plane)
{
    return s << plane.v;
}
