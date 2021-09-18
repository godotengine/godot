#if REAL_T_IS_DOUBLE
using real_t = System.Double;
#else
using real_t = System.Single;
#endif
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

        /// <summary>
        /// The normal of the plane, which must be normalized.
        /// In the scalar equation of the plane <c>ax + by + cz = d</c>, this is
        /// the vector <c>(a, b, c)</c>, where <c>d</c> is the <see cref="D"/> property.
        /// </summary>
        /// <value>Equivalent to <see cref="x"/>, <see cref="y"/>, and <see cref="z"/>.</value>
        public Vector3 Normal
        {
            get { return _normal; }
            set { _normal = value; }
        }

        /// <summary>
        /// The X component of the plane's normal vector.
        /// </summary>
        /// <value>Equivalent to <see cref="Normal"/>'s X value.</value>
        public real_t x
        {
            get
            {
                return _normal.x;
            }
            set
            {
                _normal.x = value;
            }
        }

        /// <summary>
        /// The Y component of the plane's normal vector.
        /// </summary>
        /// <value>Equivalent to <see cref="Normal"/>'s Y value.</value>
        public real_t y
        {
            get
            {
                return _normal.y;
            }
            set
            {
                _normal.y = value;
            }
        }

        /// <summary>
        /// The Z component of the plane's normal vector.
        /// </summary>
        /// <value>Equivalent to <see cref="Normal"/>'s Z value.</value>
        public real_t z
        {
            get
            {
                return _normal.z;
            }
            set
            {
                _normal.z = value;
            }
        }

        /// <summary>
        /// The distance from the origin to the plane (in the direction of
        /// <see cref="Normal"/>). This value is typically non-negative.
        /// In the scalar equation of the plane <c>ax + by + cz = d</c>,
        /// this is <c>d</c>, while the <c>(a, b, c)</c> coordinates are represented
        /// by the <see cref="Normal"/> property.
        /// </summary>
        /// <value>The plane's distance from the origin.</value>
        public real_t D { get; set; }

        /// <summary>
        /// The center of the plane, the point where the normal line intersects the plane.
        /// </summary>
        /// <value>Equivalent to <see cref="Normal"/> multiplied by <see cref="D"/>.</value>
        public Vector3 Center
        {
            get
            {
                return _normal * D;
            }
            set
            {
                _normal = value.Normalized();
                D = value.Length();
            }
        }

        /// <summary>
        /// Returns the shortest distance from this plane to the position <paramref name="point"/>.
        /// </summary>
        /// <param name="point">The position to use for the calculation.</param>
        /// <returns>The shortest distance.</returns>
        public real_t DistanceTo(Vector3 point)
        {
            return _normal.Dot(point) - D;
        }

        /// <summary>
        /// The center of the plane, the point where the normal line intersects the plane.
        /// Deprecated, use the Center property instead.
        /// </summary>
        /// <returns>Equivalent to <see cref="Normal"/> multiplied by `D`.</returns>
        [Obsolete("GetAnyPoint is deprecated. Use the Center property instead.")]
        public Vector3 GetAnyPoint()
        {
            return _normal * D;
        }

        /// <summary>
        /// Returns <see langword="true"/> if point is inside the plane.
        /// Comparison uses a custom minimum epsilon threshold.
        /// </summary>
        /// <param name="point">The point to check.</param>
        /// <param name="epsilon">The tolerance threshold.</param>
        /// <returns>A <see langword="bool"/> for whether or not the plane has the point.</returns>
        public bool HasPoint(Vector3 point, real_t epsilon = Mathf.Epsilon)
        {
            real_t dist = _normal.Dot(point) - D;
            return Mathf.Abs(dist) <= epsilon;
        }

        /// <summary>
        /// Returns the intersection point of the three planes: <paramref name="b"/>, <paramref name="c"/>,
        /// and this plane. If no intersection is found, <see langword="null"/> is returned.
        /// </summary>
        /// <param name="b">One of the three planes to use in the calculation.</param>
        /// <param name="c">One of the three planes to use in the calculation.</param>
        /// <returns>The intersection, or <see langword="null"/> if none is found.</returns>
        public Vector3? Intersect3(Plane b, Plane c)
        {
            real_t denom = _normal.Cross(b._normal).Dot(c._normal);

            if (Mathf.IsZeroApprox(denom))
            {
                return null;
            }

            Vector3 result = (b._normal.Cross(c._normal) * D) +
                                (c._normal.Cross(_normal) * b.D) +
                                (_normal.Cross(b._normal) * c.D);

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
        public Vector3? IntersectRay(Vector3 from, Vector3 dir)
        {
            real_t den = _normal.Dot(dir);

            if (Mathf.IsZeroApprox(den))
            {
                return null;
            }

            real_t dist = (_normal.Dot(from) - D) / den;

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
        public Vector3? IntersectSegment(Vector3 begin, Vector3 end)
        {
            Vector3 segment = begin - end;
            real_t den = _normal.Dot(segment);

            if (Mathf.IsZeroApprox(den))
            {
                return null;
            }

            real_t dist = (_normal.Dot(begin) - D) / den;

            // Only allow dist to be in the range of 0 to 1, with tolerance.
            if (dist < -Mathf.Epsilon || dist > 1.0f + Mathf.Epsilon)
            {
                return null;
            }

            return begin - (segment * dist);
        }

        /// <summary>
        /// Returns <see langword="true"/> if <paramref name="point"/> is located above the plane.
        /// </summary>
        /// <param name="point">The point to check.</param>
        /// <returns>A <see langword="bool"/> for whether or not the point is above the plane.</returns>
        public bool IsPointOver(Vector3 point)
        {
            return _normal.Dot(point) > D;
        }

        /// <summary>
        /// Returns the plane scaled to unit length.
        /// </summary>
        /// <returns>A normalized version of the plane.</returns>
        public Plane Normalized()
        {
            real_t len = _normal.Length();

            if (len == 0)
            {
                return new Plane(0, 0, 0, 0);
            }

            return new Plane(_normal / len, D / len);
        }

        /// <summary>
        /// Returns the orthogonal projection of <paramref name="point"/> into the plane.
        /// </summary>
        /// <param name="point">The point to project.</param>
        /// <returns>The projected point.</returns>
        public Vector3 Project(Vector3 point)
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
            D = d;
        }

        /// <summary>
        /// Constructs a <see cref="Plane"/> from a <paramref name="normal"/> vector and
        /// the plane's distance to the origin <paramref name="d"/>.
        /// </summary>
        /// <param name="normal">The normal of the plane, must be normalized.</param>
        /// <param name="d">The plane's distance from the origin. This value is typically non-negative.</param>
        public Plane(Vector3 normal, real_t d)
        {
            _normal = normal;
            D = d;
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
            D = _normal.Dot(v1);
        }

        public static Plane operator -(Plane plane)
        {
            return new Plane(-plane._normal, -plane.D);
        }

        public static bool operator ==(Plane left, Plane right)
        {
            return left.Equals(right);
        }

        public static bool operator !=(Plane left, Plane right)
        {
            return !left.Equals(right);
        }

        /// <summary>
        /// Returns <see langword="true"/> if this plane and <paramref name="obj"/> are equal.
        /// </summary>
        /// <param name="obj">The other object to compare.</param>
        /// <returns>Whether or not the plane and the other object are equal.</returns>
        public override bool Equals(object obj)
        {
            if (obj is Plane)
            {
                return Equals((Plane)obj);
            }

            return false;
        }

        /// <summary>
        /// Returns <see langword="true"/> if this plane and <paramref name="other"/> are equal.
        /// </summary>
        /// <param name="other">The other plane to compare.</param>
        /// <returns>Whether or not the planes are equal.</returns>
        public bool Equals(Plane other)
        {
            return _normal == other._normal && D == other.D;
        }

        /// <summary>
        /// Returns <see langword="true"/> if this plane and <paramref name="other"/> are
        /// approximately equal, by running <see cref="Mathf.IsEqualApprox(real_t, real_t)"/> on each component.
        /// </summary>
        /// <param name="other">The other plane to compare.</param>
        /// <returns>Whether or not the planes are approximately equal.</returns>
        public bool IsEqualApprox(Plane other)
        {
            return _normal.IsEqualApprox(other._normal) && Mathf.IsEqualApprox(D, other.D);
        }

        /// <summary>
        /// Serves as the hash function for <see cref="Plane"/>.
        /// </summary>
        /// <returns>A hash code for this plane.</returns>
        public override int GetHashCode()
        {
            return _normal.GetHashCode() ^ D.GetHashCode();
        }

        /// <summary>
        /// Converts this <see cref="Plane"/> to a string.
        /// </summary>
        /// <returns>A string representation of this plane.</returns>
        public override string ToString()
        {
            return String.Format("({0}, {1})", new object[]
            {
                _normal.ToString(),
                D.ToString()
            });
        }

        /// <summary>
        /// Converts this <see cref="Plane"/> to a string with the given <paramref name="format"/>.
        /// </summary>
        /// <returns>A string representation of this plane.</returns>
        public string ToString(string format)
        {
            return String.Format("({0}, {1})", new object[]
            {
                _normal.ToString(format),
                D.ToString(format)
            });
        }
    }
}
