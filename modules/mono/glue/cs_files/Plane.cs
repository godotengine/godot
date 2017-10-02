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

        public float distance_to(Vector3 point)
        {
            return normal.dot(point) - d;
        }

        public Vector3 get_any_point()
        {
            return normal * d;
        }

        public bool has_point(Vector3 point, float epsilon = Mathf.Epsilon)
        {
            float dist = normal.dot(point) - d;
            return Mathf.abs(dist) <= epsilon;
        }

        public Vector3 intersect_3(Plane b, Plane c)
        {
            float denom = normal.cross(b.normal).dot(c.normal);

            if (Mathf.abs(denom) <= Mathf.Epsilon)
                return new Vector3();

            Vector3 result = (b.normal.cross(c.normal) * this.d) +
                                (c.normal.cross(normal) * b.d) +
                                (normal.cross(b.normal) * c.d);

            return result / denom;
        }

        public Vector3 intersect_ray(Vector3 from, Vector3 dir)
        {
            float den = normal.dot(dir);

            if (Mathf.abs(den) <= Mathf.Epsilon)
                return new Vector3();

            float dist = (normal.dot(from) - d) / den;

            // This is a ray, before the emiting pos (from) does not exist
            if (dist > Mathf.Epsilon)
                return new Vector3();

            return from + dir * -dist;
        }

        public Vector3 intersect_segment(Vector3 begin, Vector3 end)
        {
            Vector3 segment = begin - end;
            float den = normal.dot(segment);

            if (Mathf.abs(den) <= Mathf.Epsilon)
                return new Vector3();

            float dist = (normal.dot(begin) - d) / den;

            if (dist < -Mathf.Epsilon || dist > (1.0f + Mathf.Epsilon))
                return new Vector3();

            return begin + segment * -dist;
        }

        public bool is_point_over(Vector3 point)
        {
            return normal.dot(point) > d;
        }

        public Plane normalized()
        {
            float len = normal.length();

            if (len == 0)
                return new Plane(0, 0, 0, 0);

            return new Plane(normal / len, d / len);
        }

        public Vector3 project(Vector3 point)
        {
            return point - normal * distance_to(point);
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
            normal = (v1 - v3).cross(v1 - v2);
            normal.normalize();
            d = normal.dot(v1);
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
