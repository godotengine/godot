using System;

namespace Godot
{
    public struct Plane : IEquatable<Plane>
    {
        Vector3 normal;

        public float x
        {
            get
            {
                return normal.x;
            }
            set
            {
                normal.x = value;
            }
        }

        public float y
        {
            get
            {
                return normal.y;
            }
            set
            {
                normal.y = value;
            }
        }

        public float z
        {
            get
            {
                return normal.z;
            }
            set
            {
                normal.z = value;
            }
        }

        float d;

        public Vector3 Center
        {
            get
            {
                return normal * d;
            }
        }

        public float DistanceTo(Vector3 point)
        {
            return normal.Dot(point) - d;
        }

        public Vector3 GetAnyPoint()
        {
            return normal * d;
        }

        public bool HasPoint(Vector3 point, float epsilon = Mathf.Epsilon)
        {
            float dist = normal.Dot(point) - d;
            return Mathf.Abs(dist) <= epsilon;
        }

        public Vector3 Intersect3(Plane b, Plane c)
        {
            float denom = normal.Cross(b.normal).Dot(c.normal);

            if (Mathf.Abs(denom) <= Mathf.Epsilon)
                return new Vector3();

            Vector3 result = (b.normal.Cross(c.normal) * this.d) +
                                (c.normal.Cross(normal) * b.d) +
                                (normal.Cross(b.normal) * c.d);

            return result / denom;
        }

        public Vector3 IntersectRay(Vector3 from, Vector3 dir)
        {
            float den = normal.Dot(dir);

            if (Mathf.Abs(den) <= Mathf.Epsilon)
                return new Vector3();

            float dist = (normal.Dot(from) - d) / den;

            // This is a ray, before the emiting pos (from) does not exist
            if (dist > Mathf.Epsilon)
                return new Vector3();

            return from + dir * -dist;
        }

        public Vector3 IntersectSegment(Vector3 begin, Vector3 end)
        {
            Vector3 segment = begin - end;
            float den = normal.Dot(segment);

            if (Mathf.Abs(den) <= Mathf.Epsilon)
                return new Vector3();

            float dist = (normal.Dot(begin) - d) / den;

            if (dist < -Mathf.Epsilon || dist > (1.0f + Mathf.Epsilon))
                return new Vector3();

            return begin + segment * -dist;
        }

        public bool IsPointOver(Vector3 point)
        {
            return normal.Dot(point) > d;
        }

        public Plane Normalized()
        {
            float len = normal.Length();

            if (len == 0)
                return new Plane(0, 0, 0, 0);

            return new Plane(normal / len, d / len);
        }

        public Vector3 Project(Vector3 point)
        {
            return point - normal * DistanceTo(point);
        }

        public Plane(float a, float b, float c, float d)
        {
            normal = new Vector3(a, b, c);
            this.d = d;
        }

        public Plane(Vector3 normal, float d)
        {
            this.normal = normal;
            this.d = d;
        }

        public Plane(Vector3 v1, Vector3 v2, Vector3 v3)
        {
            normal = (v1 - v3).Cross(v1 - v2);
            normal.Normalize();
            d = normal.Dot(v1);
        }

        public static Plane operator -(Plane plane)
        {
            return new Plane(-plane.normal, -plane.d);
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
            return normal == other.normal && d == other.d;
        }

        public override int GetHashCode()
        {
            return normal.GetHashCode() ^ d.GetHashCode();
        }

        public override string ToString()
        {
            return String.Format("({0}, {1})", new object[]
            {
                this.normal.ToString(),
                this.d.ToString()
            });
        }

        public string ToString(string format)
        {
            return String.Format("({0}, {1})", new object[]
            {
                this.normal.ToString(format),
                this.d.ToString(format)
            });
        }
    }
}
