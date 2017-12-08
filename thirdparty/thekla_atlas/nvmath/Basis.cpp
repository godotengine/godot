// This code is in the public domain -- Ignacio Castaño <castano@gmail.com>

#include "Basis.h"

using namespace nv;


/// Normalize basis vectors.
void Basis::normalize(float epsilon /*= NV_EPSILON*/)
{
    normal = ::normalizeSafe(normal, Vector3(0.0f), epsilon);
    tangent = ::normalizeSafe(tangent, Vector3(0.0f), epsilon);
    bitangent = ::normalizeSafe(bitangent, Vector3(0.0f), epsilon);
}


/// Gram-Schmidt orthogonalization.
/// @note Works only if the vectors are close to orthogonal.
void Basis::orthonormalize(float epsilon /*= NV_EPSILON*/)
{
    // N' = |N|
    // T' = |T - (N' dot T) N'|
    // B' = |B - (N' dot B) N' - (T' dot B) T'|

    normal = ::normalize(normal, epsilon);

    tangent -= normal * dot(normal, tangent);
    tangent = ::normalize(tangent, epsilon);

    bitangent -= normal * dot(normal, bitangent);
    bitangent -= tangent * dot(tangent, bitangent);
    bitangent = ::normalize(bitangent, epsilon);
}




/// Robust orthonormalization. 
/// Returns an orthonormal basis even when the original is degenerate.
void Basis::robustOrthonormalize(float epsilon /*= NV_EPSILON*/)
{
    // Normalize all vectors.
    normalize(epsilon);

    if (lengthSquared(normal) < epsilon*epsilon)
    {
        // Build normal from tangent and bitangent.
        normal = cross(tangent, bitangent);

        if (lengthSquared(normal) < epsilon*epsilon)
        {
            // Arbitrary basis.
            tangent   = Vector3(1, 0, 0);
            bitangent = Vector3(0, 1, 0);
            normal    = Vector3(0, 0, 1);
            return;
        }

        normal = nv::normalize(normal, epsilon);
    }

    // Project tangents to normal plane.
    tangent -= normal * dot(normal, tangent);
    bitangent -= normal * dot(normal, bitangent);

    if (lengthSquared(tangent) < epsilon*epsilon)
    {
        if (lengthSquared(bitangent) < epsilon*epsilon)
        {
            // Arbitrary basis.
            buildFrameForDirection(normal);
        }
        else
        {
            // Build tangent from bitangent.
            bitangent = nv::normalize(bitangent, epsilon);

            tangent = cross(bitangent, normal);
            nvDebugCheck(isNormalized(tangent, epsilon));
        }
    }
    else
    {
        tangent = nv::normalize(tangent, epsilon);
#if 0
        bitangent -= tangent * dot(tangent, bitangent);

        if (lengthSquared(bitangent) < epsilon*epsilon)
        {
            bitangent = cross(tangent, normal);
            nvDebugCheck(isNormalized(bitangent, epsilon));
        }
        else
        {
            bitangent = nv::normalize(bitangent, epsilon);
        }
#else
        if (lengthSquared(bitangent) < epsilon*epsilon)
        {
            // Build bitangent from tangent.
            bitangent = cross(tangent, normal);
            nvDebugCheck(isNormalized(bitangent, epsilon));
        }
        else
        {
            bitangent = nv::normalize(bitangent, epsilon);

            // At this point tangent and bitangent are orthogonal to normal, but we don't know whether their orientation.
            
            Vector3 bisector;
            if (lengthSquared(tangent + bitangent) < epsilon*epsilon)
            {
                bisector = tangent;
            }
            else
            {
                bisector = nv::normalize(tangent + bitangent);
            }
            Vector3 axis = nv::normalize(cross(bisector, normal));

            //nvDebugCheck(isNormalized(axis, epsilon));
            nvDebugCheck(equal(dot(axis, tangent), -dot(axis, bitangent), epsilon));

            if (dot(axis, tangent) > 0)
            {
                tangent = bisector + axis;
                bitangent = bisector - axis;
            }
            else
            {
                tangent = bisector - axis;
                bitangent = bisector + axis;
            }

            // Make sure the resulting tangents are still perpendicular to the normal.
            tangent -= normal * dot(normal, tangent);
            bitangent -= normal * dot(normal, bitangent);

            // Double check.
            nvDebugCheck(equal(dot(normal, tangent), 0.0f, epsilon));
            nvDebugCheck(equal(dot(normal, bitangent), 0.0f, epsilon));

            // Normalize.
            tangent = nv::normalize(tangent);
            bitangent = nv::normalize(bitangent);

            // If tangent and bitangent are not orthogonal, then derive bitangent from tangent, just in case...
            if (!equal(dot(tangent, bitangent), 0.0f, epsilon)) {
                bitangent = cross(tangent, normal);
                bitangent = nv::normalize(bitangent);
            }
        }
#endif
    }

    /*// Check vector lengths.
    if (!isNormalized(normal, epsilon))
    {
    nvDebug("%f %f %f\n", normal.x, normal.y, normal.z);
    nvDebug("%f %f %f\n", tangent.x, tangent.y, tangent.z);
    nvDebug("%f %f %f\n", bitangent.x, bitangent.y, bitangent.z);
    }*/

    nvDebugCheck(isNormalized(normal, epsilon));
    nvDebugCheck(isNormalized(tangent, epsilon));
    nvDebugCheck(isNormalized(bitangent, epsilon));

    // Check vector angles.
    nvDebugCheck(equal(dot(normal, tangent), 0.0f, epsilon));
    nvDebugCheck(equal(dot(normal, bitangent), 0.0f, epsilon));
    nvDebugCheck(equal(dot(tangent, bitangent), 0.0f, epsilon));

    // Check vector orientation.
    const float det = dot(cross(normal, tangent), bitangent);
    nvDebugCheck(equal(det, 1.0f, epsilon) || equal(det, -1.0f, epsilon));
}


/// Build an arbitrary frame for the given direction.
void Basis::buildFrameForDirection(Vector3::Arg d, float angle/*= 0*/)
{
    nvCheck(isNormalized(d));
    normal = d;

    // Choose minimum axis.
    if (fabsf(normal.x) < fabsf(normal.y) && fabsf(normal.x) < fabsf(normal.z))
    {
        tangent = Vector3(1, 0, 0);
    }
    else if (fabsf(normal.y) < fabsf(normal.z))
    {
        tangent = Vector3(0, 1, 0);
    }
    else
    {
        tangent = Vector3(0, 0, 1);
    }

    // Ortogonalize
    tangent -= normal * dot(normal, tangent);
    tangent = ::normalize(tangent);

    bitangent = cross(normal, tangent);

    // Rotate frame around normal according to angle.
    if (angle != 0.0f) {
        float c = cosf(angle);
        float s = sinf(angle);
        Vector3 tmp = c * tangent - s * bitangent;
        bitangent = s * tangent + c * bitangent;
        tangent = tmp;
    }
}

bool Basis::isValid() const
{
    if (equal(normal, Vector3(0.0f))) return false;
    if (equal(tangent, Vector3(0.0f))) return false;
    if (equal(bitangent, Vector3(0.0f))) return false;

    if (equal(determinant(), 0.0f)) return false;

    return true;
}


/// Transform by this basis. (From this basis to object space).
Vector3 Basis::transform(Vector3::Arg v) const
{
    Vector3 o = tangent * v.x;
    o += bitangent * v.y;
    o += normal * v.z;
    return o;
}

/// Transform by the transpose. (From object space to this basis).
Vector3 Basis::transformT(Vector3::Arg v)
{
    return Vector3(dot(tangent, v), dot(bitangent, v), dot(normal, v));
}

/// Transform by the inverse. (From object space to this basis).
/// @note Uses Cramer's rule so the inverse is not accurate if the basis is ill-conditioned.
Vector3 Basis::transformI(Vector3::Arg v) const
{
    const float det = determinant();
    nvDebugCheck(!equal(det, 0.0f, 0.0f));

    const float idet = 1.0f / det;

    // Rows of the inverse matrix.
    Vector3 r0(
        (bitangent.y * normal.z - bitangent.z * normal.y),
        -(bitangent.x * normal.z - bitangent.z * normal.x),
        (bitangent.x * normal.y - bitangent.y * normal.x));

    Vector3 r1(
        -(tangent.y * normal.z - tangent.z * normal.y),
        (tangent.x * normal.z - tangent.z * normal.x),
        -(tangent.x * normal.y - tangent.y * normal.x));

    Vector3 r2(
        (tangent.y * bitangent.z - tangent.z * bitangent.y),
        -(tangent.x * bitangent.z - tangent.z * bitangent.x),
        (tangent.x * bitangent.y - tangent.y * bitangent.x));

    return Vector3(dot(v, r0), dot(v, r1), dot(v, r2)) * idet;
}


