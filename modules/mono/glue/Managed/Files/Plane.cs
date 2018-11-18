using System;
#if REAL_T_IS_DOUBLE
using real_t = System.Double;
#else
using real_t = System.Single;
#endif

namespace Godot
{
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

        public real_t D { get; set; }

        public Vector3 Center
        {
            get
            {
                return _normal * D;
            }
        }

        public real_t DistanceTo(Vector3 point)
        {
            return _normal.Dot(point) - D;
        }

        public Vector3 GetAnyPoint()
        {
            return _normal * D;
        }

        public bool HasPoint(Vector3 point, real_t epsilon = Mathf.Epsilon)
        {
            real_t dist = _normal.Dot(point) - D;
            return Mathf.Abs(dist) <= epsilon;
        }

        public Vector3 Intersect3(Plane b, Plane c)
        {
            real_t denom = _normal.Cross(b._normal).Dot(c._normal);

            if (Mathf.Abs(denom) <= Mathf.Epsilon)
                return new Vector3();

            Vector3 result = b._normal.Cross(c._normal) * D +
                                c._normal.Cross(_normal) * b.D +
                                _normal.Cross(b._normal) * c.D;

            return result / denom;
        }

        public Vector3 IntersectRay(Vector3 from, Vector3 dir)
        {
            real_t den = _normal.Dot(dir);

            if (Mathf.Abs(den) <= Mathf.Epsilon)
                return new Vector3();

            real_t dist = (_normal.Dot(from) - D) / den;

            // This is a ray, before the emitting pos (from) does not exist
            if (dist > Mathf.Epsilon)
                return new Vector3();

            return from + dir * -dist;
        }

        public Vector3 IntersectSegment(Vector3 begin, Vector3 end)
        {
            Vector3 segment = begin - end;
            real_t den = _normal.Dot(segment);

            if (Mathf.Abs(den) <= Mathf.Epsilon)
                return new Vector3();

            real_t dist = (_normal.Dot(begin) - D) / den;

            if (dist < -Mathf.Epsilon || dist > 1.0f + Mathf.Epsilon)
                return new Vector3();

            return begin + segment * -dist;
        }

        public bool IsPointOver(Vector3 point)
        {
            return _normal.Dot(point) > D;
        }

        public Plane Normalized()
        {
            real_t len = _normal.Length();

            if (len == 0)
                return new Plane(0, 0, 0, 0);

            return new Plane(_normal / len, D / len);
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
        public Plane(real_t a, real_t b, real_t c, real_t d)
        {
            _normal = new Vector3(a, b, c);
            this.D = d;
        }
        public Plane(Vector3 normal, real_t d)
        {
            this._normal = normal;
            this.D = d;
        }

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
            return _normal == other._normal && D == other.D;
        }

        public override int GetHashCode()
        {
            return _normal.GetHashCode() ^ D.GetHashCode();
        }

        public override string ToString()
        {
            return String.Format("({0}, {1})", new object[]
            {
                _normal.ToString(),
                D.ToString()
            });
        }

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
