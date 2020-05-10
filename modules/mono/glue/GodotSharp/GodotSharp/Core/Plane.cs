using System;
using System.Runtime.InteropServices;
#if REAL_T_IS_DOUBLE
using real_t = System.Double;
#else
using real_t = System.Single;
#endif

namespace Godot
{
    [Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct Plane : IEquatable<Plane>
    {
        private Vector3 _normal;

        public Vector3 Normal
        {
            get { return _normal; }
            set { _normal = value; }
        }

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

        public real_t Distance { get; set; }

        public Vector3 Center
        {
            get
            {
                return _normal * Distance;
            }
        }

        public real_t DistanceTo(Vector3 point)
        {
            return _normal.Dot(point) - Distance;
        }

        public Vector3 GetAnyPoint()
        {
            return _normal * Distance;
        }

        public bool HasPoint(Vector3 point, real_t epsilon = Mathf.Epsilon)
        {
            real_t dist = _normal.Dot(point) - Distance;
            return Mathf.Abs(dist) <= epsilon;
        }

        public Vector3? Intersect3(Plane b, Plane c)
        {
            real_t denom = _normal.Cross(b._normal).Dot(c._normal);

            if (Mathf.IsZeroApprox(denom))
                return null;

            Vector3 result = b._normal.Cross(c._normal) * Distance +
                                c._normal.Cross(_normal) * b.Distance +
                                _normal.Cross(b._normal) * c.Distance;

            return result / denom;
        }

        public Vector3? IntersectRay(Vector3 from, Vector3 dir)
        {
            real_t den = _normal.Dot(dir);

            if (Mathf.IsZeroApprox(den))
                return null;

            real_t dist = (_normal.Dot(from) - Distance) / den;

            // This is a ray, before the emitting pos (from) does not exist
            if (dist > Mathf.Epsilon)
                return null;

            return from + dir * -dist;
        }

        public Vector3? IntersectSegment(Vector3 begin, Vector3 end)
        {
            Vector3 segment = begin - end;
            real_t den = _normal.Dot(segment);

            if (Mathf.IsZeroApprox(den))
                return null;

            real_t dist = (_normal.Dot(begin) - Distance) / den;

            // Only allow dist to be in the range of 0 to 1, with tolerance.
            if (dist < -Mathf.Epsilon || dist > 1.0f + Mathf.Epsilon)
                return null;

            return begin + segment * -dist;
        }

        public bool IsPointOver(Vector3 point)
        {
            return _normal.Dot(point) > Distance;
        }

        public Plane Normalized()
        {
            real_t len = _normal.Length();

            if (len == 0)
                return new Plane(0, 0, 0, 0);

            return new Plane(_normal / len, Distance / len);
        }

        public Vector3 Project(Vector3 point)
        {
            return point - _normal * DistanceTo(point);
        }

        // Constants
        private static readonly Plane _planeYZ = new Plane(1, 0, 0, 0);
        private static readonly Plane _planeXZ = new Plane(0, 1, 0, 0);
        private static readonly Plane _planeXY = new Plane(0, 0, 1, 0);

        public static Plane PlaneYZ { get { return _planeYZ; } }
        public static Plane PlaneXZ { get { return _planeXZ; } }
        public static Plane PlaneXY { get { return _planeXY; } }

        // Constructors
        public Plane(real_t a, real_t b, real_t c, real_t distance)
        {
            _normal = new Vector3(a, b, c);
            this.Distance = distance;
        }
        public Plane(Vector3 normal, real_t distance)
        {
            this._normal = normal;
            this.Distance = distance;
        }

        public Plane(Vector3 v1, Vector3 v2, Vector3 v3)
        {
            _normal = (v1 - v3).Cross(v1 - v2);
            _normal.Normalize();
            Distance = _normal.Dot(v1);
        }

        public static Plane operator -(Plane plane)
        {
            return new Plane(-plane._normal, -plane.Distance);
        }

        public static bool operator ==(Plane left, Plane right)
        {
            return left.Equals(right);
        }

        public static bool operator !=(Plane left, Plane right)
        {
            return !left.Equals(right);
        }

        public override bool Equals(object obj)
        {
            if (obj is Plane)
            {
                return Equals((Plane)obj);
            }

            return false;
        }

        public bool Equals(Plane other)
        {
            return _normal == other._normal && Distance == other.Distance;
        }

        public bool IsEqualApprox(Plane other)
        {
            return _normal.IsEqualApprox(other._normal) && Mathf.IsEqualApprox(Distance, other.Distance);
        }

        public override int GetHashCode()
        {
            return _normal.GetHashCode() ^ Distance.GetHashCode();
        }

        public override string ToString()
        {
            return String.Format("({0}, {1})", new object[]
            {
                _normal.ToString(),
                Distance.ToString()
            });
        }

        public string ToString(string format)
        {
            return String.Format("({0}, {1})", new object[]
            {
                _normal.ToString(format),
                Distance.ToString(format)
            });
        }
    }
}
