using System;
using System.Runtime.InteropServices;

namespace Godot
{
    /// <summary>
    /// Plane represents a normalized plane equation.
    /// "Over" or "Above" the plane is considered the side of
    /// the plane towards where the normal is pointing.
    /// </summary>
    [Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct Plane : IEquatable<Plane>
    {
        private Vector3 _normal;
        private real_t _d;

        /// <summary>
        /// The normal of the plane, which must be a unit vector.
        /// In the scalar equation of the plane <c>ax + by + cz = d</c>, this is
        /// the vector <c>(a, b, c)</c>, where <c>d</c> is the <see cref="D"/> property.
        /// </summary>
        /// <value>Equivalent to <see cref="X"/>, <see cref="Y"/>, and <see cref="Z"/>.</value>
        public Vector3 Normal
        {
            readonly get { return _normal; }
            set { _normal = value; }
        }

        /// <summary>
        /// The distance from the origin to the plane (in the direction of
        /// <see cref="Normal"/>). This value is typically non-negative.
        /// In the scalar equation of the plane <c>ax + by + cz = d</c>,
        /// this is <c>d</c>, while the <c>(a, b, c)</c> coordinates are represented
        /// by the <see cref="Normal"/> property.
        /// </summary>
        /// <value>The plane's distance from the origin.</value>
        public real_t D
        {
            readonly get { return _d; }
            set { _d = value; }
        }

        /// <summary>
        /// The X component of the plane's normal vector.
        /// </summary>
        /// <value>Equivalent to <see cref="Normal"/>'s X value.</value>
        public real_t X
        {
            readonly get
            {
                return _normal.X;
            }
            set
            {
                _normal.X = value;
            }
        }

        /// <summary>
        /// The Y component of the plane's normal vector.
        /// </summary>
        /// <value>Equivalent to <see cref="Normal"/>'s Y value.</value>
        public real_t Y
        {
            readonly get
            {
                return _normal.Y;
            }
            set
            {
                _normal.Y = value;
            }
        }

        /// <summary>
        /// The Z component of the plane's normal vector.
        /// </summary>
        /// <value>Equivalent to <see cref="Normal"/>'s Z value.</value>
        public real_t Z
        {
            readonly get
            {
                return _normal.Z;
            }
            set
            {
                _normal.Z = value;
            }
        }

        /// <summary>
        /// Returns the shortest distance from this plane to the position <paramref name="point"/>.
        /// </summary>
        /// <param name="point">The position to use for the calculation.</param>
        /// <returns>The shortest distance.</returns>
        public readonly real_t DistanceTo(Vector3 point)
        {
            return _normal.Dot(point) - _d;
        }

        /// <summary>
        /// Returns the center of the plane, the point on the plane closest to the origin.
        /// The point where the normal line going through the origin intersects the plane.
        /// </summary>
        /// <value>Equivalent to <see cref="Normal"/> multiplied by <see cref="D"/>.</value>
        public readonly Vector3 GetCenter()
        {
            return _normal * _d;
        }

        /// <summary>
        /// Returns <see langword="true"/> if point is inside the plane.
        /// Comparison uses a custom minimum tolerance threshold.
        /// </summary>
        /// <param name="point">The point to check.</param>
        /// <param name="tolerance">The tolerance threshold.</param>
        /// <returns>A <see langword="bool"/> for whether or not the plane has the point.</returns>
        public readonly bool HasPoint(Vector3 point, real_t tolerance = Mathf.Epsilon)
        {
            real_t dist = _normal.Dot(point) - _d;
            return Mathf.Abs(dist) <= tolerance;
        }

        /// <summary>
        /// Returns the intersection point of the three planes: <paramref name="b"/>, <paramref name="c"/>,
        /// and this plane. If no intersection is found, <see langword="null"/> is returned.
        /// </summary>
        /// <param name="b">One of the three planes to use in the calculation.</param>
        /// <param name="c">One of the three planes to use in the calculation.</param>
        /// <returns>The intersection, or <see langword="null"/> if none is found.</returns>
        public readonly Vector3? Intersect3(Plane b, Plane c)
        {
            real_t denom = _normal.Cross(b._normal).Dot(c._normal);

            if (Mathf.IsZeroApprox(denom))
            {
                return null;
            }

            Vector3 result = (b._normal.Cross(c._normal) * _d) +
                                (c._normal.Cross(_normal) * b._d) +
                                (_normal.Cross(b._normal) * c._d);

            return result / denom;
        }

        /// <summary>
        /// Returns the intersection point of a ray consisting of the position <paramref name="from"/>
        /// and the direction normal <paramref name="dir"/> with this plane.
        /// If no intersection is found, <see langword="null"/> is returned.
        /// </summary>
        /// <param name="from">The start of the ray.</param>
        /// <param name="dir">The direction of the ray, normalized.</param>
        /// <returns>The intersection, or <see langword="null"/> if none is found.</returns>
        public readonly Vector3? IntersectsRay(Vector3 from, Vector3 dir)
        {
            real_t den = _normal.Dot(dir);

            if (Mathf.IsZeroApprox(den))
            {
                return null;
            }

            real_t dist = (_normal.Dot(from) - _d) / den;

            // This is a ray, before the emitting pos (from) does not exist
            if (dist > Mathf.Epsilon)
            {
                return null;
            }

            return from - (dir * dist);
        }

        /// <summary>
        /// Returns the intersection point of a line segment from
        /// position <paramref name="begin"/> to position <paramref name="end"/> with this plane.
        /// If no intersection is found, <see langword="null"/> is returned.
        /// </summary>
        /// <param name="begin">The start of the line segment.</param>
        /// <param name="end">The end of the line segment.</param>
        /// <returns>The intersection, or <see langword="null"/> if none is found.</returns>
        public readonly Vector3? IntersectsSegment(Vector3 begin, Vector3 end)
        {
            Vector3 segment = begin - end;
            real_t den = _normal.Dot(segment);

            if (Mathf.IsZeroApprox(den))
            {
                return null;
            }

            real_t dist = (_normal.Dot(begin) - _d) / den;

            // Only allow dist to be in the range of 0 to 1, with tolerance.
            if (dist < -Mathf.Epsilon || dist > 1.0f + Mathf.Epsilon)
            {
                return null;
            }

            return begin - (segment * dist);
        }

        /// <summary>
        /// Returns <see langword="true"/> if this plane is finite, by calling
        /// <see cref="Mathf.IsFinite(real_t)"/> on each component.
        /// </summary>
        /// <returns>Whether this vector is finite or not.</returns>
        public readonly bool IsFinite()
        {
            return _normal.IsFinite() && Mathf.IsFinite(D);
        }

        /// <summary>
        /// Returns <see langword="true"/> if <paramref name="point"/> is located above the plane.
        /// </summary>
        /// <param name="point">The point to check.</param>
        /// <returns>A <see langword="bool"/> for whether or not the point is above the plane.</returns>
        public readonly bool IsPointOver(Vector3 point)
        {
            return _normal.Dot(point) > _d;
        }

        /// <summary>
        /// Returns the plane scaled to unit length.
        /// </summary>
        /// <returns>A normalized version of the plane.</returns>
        public readonly Plane Normalized()
        {
            real_t len = _normal.Length();

            if (len == 0)
            {
                return new Plane(0, 0, 0, 0);
            }

            return new Plane(_normal / len, _d / len);
        }

        /// <summary>
        /// Returns the orthogonal projection of <paramref name="point"/> into the plane.
        /// </summary>
        /// <param name="point">The point to project.</param>
        /// <returns>The projected point.</returns>
        public readonly Vector3 Project(Vector3 point)
        {
            return point - (_normal * DistanceTo(point));
        }

        // Constants
        private static readonly Plane _planeYZ = new Plane(1, 0, 0, 0);
        private static readonly Plane _planeXZ = new Plane(0, 1, 0, 0);
        private static readonly Plane _planeXY = new Plane(0, 0, 1, 0);

        /// <summary>
        /// A <see cref="Plane"/> that extends in the Y and Z axes (normal vector points +X).
        /// </summary>
        /// <value>Equivalent to <c>new Plane(1, 0, 0, 0)</c>.</value>
        public static Plane PlaneYZ { get { return _planeYZ; } }

        /// <summary>
        /// A <see cref="Plane"/> that extends in the X and Z axes (normal vector points +Y).
        /// </summary>
        /// <value>Equivalent to <c>new Plane(0, 1, 0, 0)</c>.</value>
        public static Plane PlaneXZ { get { return _planeXZ; } }

        /// <summary>
        /// A <see cref="Plane"/> that extends in the X and Y axes (normal vector points +Z).
        /// </summary>
        /// <value>Equivalent to <c>new Plane(0, 0, 1, 0)</c>.</value>
        public static Plane PlaneXY { get { return _planeXY; } }

        /// <summary>
        /// Constructs a <see cref="Plane"/> from four values.
        /// <paramref name="a"/>, <paramref name="b"/> and <paramref name="c"/> become the
        /// components of the resulting plane's <see cref="Normal"/> vector.
        /// <paramref name="d"/> becomes the plane's distance from the origin.
        /// </summary>
        /// <param name="a">The X component of the plane's normal vector.</param>
        /// <param name="b">The Y component of the plane's normal vector.</param>
        /// <param name="c">The Z component of the plane's normal vector.</param>
        /// <param name="d">The plane's distance from the origin. This value is typically non-negative.</param>
        public Plane(real_t a, real_t b, real_t c, real_t d)
        {
            _normal = new Vector3(a, b, c);
            _d = d;
        }

        /// <summary>
        /// Constructs a <see cref="Plane"/> from a <paramref name="normal"/> vector.
        /// The plane will intersect the origin.
        /// </summary>
        /// <param name="normal">The normal of the plane, must be a unit vector.</param>
        public Plane(Vector3 normal)
        {
            _normal = normal;
            _d = 0;
        }

        /// <summary>
        /// Constructs a <see cref="Plane"/> from a <paramref name="normal"/> vector and
        /// the plane's distance to the origin <paramref name="d"/>.
        /// </summary>
        /// <param name="normal">The normal of the plane, must be a unit vector.</param>
        /// <param name="d">The plane's distance from the origin. This value is typically non-negative.</param>
        public Plane(Vector3 normal, real_t d)
        {
            _normal = normal;
            _d = d;
        }

        /// <summary>
        /// Constructs a <see cref="Plane"/> from a <paramref name="normal"/> vector and
        /// a <paramref name="point"/> on the plane.
        /// </summary>
        /// <param name="normal">The normal of the plane, must be a unit vector.</param>
        /// <param name="point">The point on the plane.</param>
        public Plane(Vector3 normal, Vector3 point)
        {
            _normal = normal;
            _d = _normal.Dot(point);
        }

        /// <summary>
        /// Constructs a <see cref="Plane"/> from the three points, given in clockwise order.
        /// </summary>
        /// <param name="v1">The first point.</param>
        /// <param name="v2">The second point.</param>
        /// <param name="v3">The third point.</param>
        public Plane(Vector3 v1, Vector3 v2, Vector3 v3)
        {
            _normal = (v1 - v3).Cross(v1 - v2);
            _normal.Normalize();
            _d = _normal.Dot(v1);
        }

        /// <summary>
        /// Returns the negative value of the <see cref="Plane"/>.
        /// This is the same as writing <c>new Plane(-p.Normal, -p.D)</c>.
        /// This operation flips the direction of the normal vector and
        /// also flips the distance value, resulting in a Plane that is
        /// in the same place, but facing the opposite direction.
        /// </summary>
        /// <param name="plane">The plane to negate/flip.</param>
        /// <returns>The negated/flipped plane.</returns>
        public static Plane operator -(Plane plane)
        {
            return new Plane(-plane._normal, -plane._d);
        }

        /// <summary>
        /// Returns <see langword="true"/> if the
        /// <see cref="Plane"/>s are exactly equal.
        /// Note: Due to floating-point precision errors, consider using
        /// <see cref="IsEqualApprox"/> instead, which is more reliable.
        /// </summary>
        /// <param name="left">The left rect.</param>
        /// <param name="right">The right rect.</param>
        /// <returns>Whether or not the planes are exactly equal.</returns>
        public static bool operator ==(Plane left, Plane right)
        {
            return left.Equals(right);
        }

        /// <summary>
        /// Returns <see langword="true"/> if the
        /// <see cref="Plane"/>s are not equal.
        /// Note: Due to floating-point precision errors, consider using
        /// <see cref="IsEqualApprox"/> instead, which is more reliable.
        /// </summary>
        /// <param name="left">The left rect.</param>
        /// <param name="right">The right rect.</param>
        /// <returns>Whether or not the planes are not equal.</returns>
        public static bool operator !=(Plane left, Plane right)
        {
            return !left.Equals(right);
        }

        /// <summary>
        /// Returns <see langword="true"/> if this plane and <paramref name="obj"/> are equal.
        /// </summary>
        /// <param name="obj">The other object to compare.</param>
        /// <returns>Whether or not the plane and the other object are exactly equal.</returns>
        public override readonly bool Equals(object obj)
        {
            return obj is Plane other && Equals(other);
        }

        /// <summary>
        /// Returns <see langword="true"/> if this plane and <paramref name="other"/> are equal.
        /// </summary>
        /// <param name="other">The other plane to compare.</param>
        /// <returns>Whether or not the planes are exactly equal.</returns>
        public readonly bool Equals(Plane other)
        {
            return _normal == other._normal && _d == other._d;
        }

        /// <summary>
        /// Returns <see langword="true"/> if this plane and <paramref name="other"/> are
        /// approximately equal, by running <see cref="Mathf.IsEqualApprox(real_t, real_t)"/> on each component.
        /// </summary>
        /// <param name="other">The other plane to compare.</param>
        /// <returns>Whether or not the planes are approximately equal.</returns>
        public readonly bool IsEqualApprox(Plane other)
        {
            return _normal.IsEqualApprox(other._normal) && Mathf.IsEqualApprox(_d, other._d);
        }

        /// <summary>
        /// Serves as the hash function for <see cref="Plane"/>.
        /// </summary>
        /// <returns>A hash code for this plane.</returns>
        public override readonly int GetHashCode()
        {
            return HashCode.Combine(_normal, _d);
        }

        /// <summary>
        /// Converts this <see cref="Plane"/> to a string.
        /// </summary>
        /// <returns>A string representation of this plane.</returns>
        public override readonly string ToString()
        {
            return $"{_normal}, {_d}";
        }

        /// <summary>
        /// Converts this <see cref="Plane"/> to a string with the given <paramref name="format"/>.
        /// </summary>
        /// <returns>A string representation of this plane.</returns>
        public readonly string ToString(string format)
        {
            return $"{_normal.ToString(format)}, {_d.ToString(format)}";
        }
    }
}
