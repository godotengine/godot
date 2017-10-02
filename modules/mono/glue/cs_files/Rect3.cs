using System;

// file: core/math/rect3.h
// commit: 7ad14e7a3e6f87ddc450f7e34621eb5200808451
// file: core/math/rect3.cpp
// commit: bd282ff43f23fe845f29a3e25c8efc01bd65ffb0
// file: core/variant_call.cpp
// commit: 5ad9be4c24e9d7dc5672fdc42cea896622fe5685

namespace Godot
{
    public struct Rect3 : IEquatable<Rect3>
    {
        private Vector3 position;
        private Vector3 size;

        public Vector3 Position
        {
            get
            {
                return position;
            }
        }

        public Vector3 Size
        {
            get
            {
                return size;
            }
        }

        public Vector3 End
        {
            get
            {
                return position + size;
            }
        }

        public bool encloses(Rect3 with)
        {
            Vector3 src_min = position;
            Vector3 src_max = position + size;
            Vector3 dst_min = with.position;
            Vector3 dst_max = with.position + with.size;

            return ((src_min.x <= dst_min.x) &&
                    (src_max.x > dst_max.x) &&
                    (src_min.y <= dst_min.y) &&
                    (src_max.y > dst_max.y) &&
                    (src_min.z <= dst_min.z) &&
                    (src_max.z > dst_max.z));
        }

        public Rect3 expand(Vector3 to_point)
        {
            Vector3 begin = position;
            Vector3 end = position + size;

            if (to_point.x < begin.x)
                begin.x = to_point.x;
            if (to_point.y < begin.y)
                begin.y = to_point.y;
            if (to_point.z < begin.z)
                begin.z = to_point.z;

            if (to_point.x > end.x)
                end.x = to_point.x;
            if (to_point.y > end.y)
                end.y = to_point.y;
            if (to_point.z > end.z)
                end.z = to_point.z;

            return new Rect3(begin, end - begin);
        }

        public float get_area()
        {
            return size.x * size.y * size.z;
        }

        public Vector3 get_endpoint(int idx)
        {
            switch (idx)
            {
                case 0:
                    return new Vector3(position.x, position.y, position.z);
                case 1:
                    return new Vector3(position.x, position.y, position.z + size.z);
                case 2:
                    return new Vector3(position.x, position.y + size.y, position.z);
                case 3:
                    return new Vector3(position.x, position.y + size.y, position.z + size.z);
                case 4:
                    return new Vector3(position.x + size.x, position.y, position.z);
                case 5:
                    return new Vector3(position.x + size.x, position.y, position.z + size.z);
                case 6:
                    return new Vector3(position.x + size.x, position.y + size.y, position.z);
                case 7:
                    return new Vector3(position.x + size.x, position.y + size.y, position.z + size.z);
                default:
                    throw new ArgumentOutOfRangeException(nameof(idx), String.Format("Index is {0}, but a value from 0 to 7 is expected.", idx));
            }
        }

        public Vector3 get_longest_axis()
        {
            Vector3 axis = new Vector3(1f, 0f, 0f);
            float max_size = size.x;

            if (size.y > max_size)
            {
                axis = new Vector3(0f, 1f, 0f);
                max_size = size.y;
            }

            if (size.z > max_size)
            {
                axis = new Vector3(0f, 0f, 1f);
                max_size = size.z;
            }

            return axis;
        }

        public Vector3.Axis get_longest_axis_index()
        {
            Vector3.Axis axis = Vector3.Axis.X;
            float max_size = size.x;

            if (size.y > max_size)
            {
                axis = Vector3.Axis.Y;
                max_size = size.y;
            }

            if (size.z > max_size)
            {
                axis = Vector3.Axis.Z;
                max_size = size.z;
            }

            return axis;
        }

        public float get_longest_axis_size()
        {
            float max_size = size.x;

            if (size.y > max_size)
                max_size = size.y;

            if (size.z > max_size)
                max_size = size.z;

            return max_size;
        }

        public Vector3 get_shortest_axis()
        {
            Vector3 axis = new Vector3(1f, 0f, 0f);
            float max_size = size.x;

            if (size.y < max_size)
            {
                axis = new Vector3(0f, 1f, 0f);
                max_size = size.y;
            }

            if (size.z < max_size)
            {
                axis = new Vector3(0f, 0f, 1f);
                max_size = size.z;
            }

            return axis;
        }

        public Vector3.Axis get_shortest_axis_index()
        {
            Vector3.Axis axis = Vector3.Axis.X;
            float max_size = size.x;

            if (size.y < max_size)
            {
                axis = Vector3.Axis.Y;
                max_size = size.y;
            }

            if (size.z < max_size)
            {
                axis = Vector3.Axis.Z;
                max_size = size.z;
            }

            return axis;
        }

        public float get_shortest_axis_size()
        {
            float max_size = size.x;

            if (size.y < max_size)
                max_size = size.y;

            if (size.z < max_size)
                max_size = size.z;

            return max_size;
        }

        public Vector3 get_support(Vector3 dir)
        {
            Vector3 half_extents = size * 0.5f;
            Vector3 ofs = position + half_extents;

            return ofs + new Vector3(
                (dir.x > 0f) ? -half_extents.x : half_extents.x,
                (dir.y > 0f) ? -half_extents.y : half_extents.y,
                (dir.z > 0f) ? -half_extents.z : half_extents.z);
        }

        public Rect3 grow(float by)
        {
            Rect3 res = this;

            res.position.x -= by;
            res.position.y -= by;
            res.position.z -= by;
            res.size.x += 2.0f * by;
            res.size.y += 2.0f * by;
            res.size.z += 2.0f * by;

            return res;
        }

        public bool has_no_area()
        {
            return size.x <= 0f || size.y <= 0f || size.z <= 0f;
        }

        public bool has_no_surface()
        {
            return size.x <= 0f && size.y <= 0f && size.z <= 0f;
        }

        public bool has_point(Vector3 point)
        {
            if (point.x < position.x)
                return false;
            if (point.y < position.y)
                return false;
            if (point.z < position.z)
                return false;
            if (point.x > position.x + size.x)
                return false;
            if (point.y > position.y + size.y)
                return false;
            if (point.z > position.z + size.z)
                return false;

            return true;
        }

        public Rect3 intersection(Rect3 with)
        {
            Vector3 src_min = position;
            Vector3 src_max = position + size;
            Vector3 dst_min = with.position;
            Vector3 dst_max = with.position + with.size;

            Vector3 min, max;

            if (src_min.x > dst_max.x || src_max.x < dst_min.x)
            {
                return new Rect3();
            }
            else
            {
                min.x = (src_min.x > dst_min.x) ? src_min.x : dst_min.x;
                max.x = (src_max.x < dst_max.x) ? src_max.x : dst_max.x;
            }

            if (src_min.y > dst_max.y || src_max.y < dst_min.y)
            {
                return new Rect3();
            }
            else
            {
                min.y = (src_min.y > dst_min.y) ? src_min.y : dst_min.y;
                max.y = (src_max.y < dst_max.y) ? src_max.y : dst_max.y;
            }

            if (src_min.z > dst_max.z || src_max.z < dst_min.z)
            {
                return new Rect3();
            }
            else
            {
                min.z = (src_min.z > dst_min.z) ? src_min.z : dst_min.z;
                max.z = (src_max.z < dst_max.z) ? src_max.z : dst_max.z;
            }

            return new Rect3(min, max - min);
        }

        public bool intersects(Rect3 with)
        {
            if (position.x >= (with.position.x + with.size.x))
                return false;
            if ((position.x + size.x) <= with.position.x)
                return false;
            if (position.y >= (with.position.y + with.size.y))
                return false;
            if ((position.y + size.y) <= with.position.y)
                return false;
            if (position.z >= (with.position.z + with.size.z))
                return false;
            if ((position.z + size.z) <= with.position.z)
                return false;

            return true;
        }

        public bool intersects_plane(Plane plane)
        {
            Vector3[] points =
            {
                new Vector3(position.x, position.y, position.z),
                new Vector3(position.x, position.y, position.z + size.z),
                new Vector3(position.x, position.y + size.y, position.z),
                new Vector3(position.x, position.y + size.y, position.z + size.z),
                new Vector3(position.x + size.x, position.y, position.z),
                new Vector3(position.x + size.x, position.y, position.z + size.z),
                new Vector3(position.x + size.x, position.y + size.y, position.z),
                new Vector3(position.x + size.x, position.y + size.y, position.z + size.z),
            };

            bool over = false;
            bool under = false;

            for (int i = 0; i < 8; i++)
            {
                if (plane.distance_to(points[i]) > 0)
                    over = true;
                else
                    under = true;
            }

            return under && over;
        }

        public bool intersects_segment(Vector3 from, Vector3 to)
        {
            float min = 0f;
            float max = 1f;

            for (int i = 0; i < 3; i++)
            {
                float seg_from = from[i];
                float seg_to = to[i];
                float box_begin = position[i];
                float box_end = box_begin + size[i];
                float cmin, cmax;

                if (seg_from < seg_to)
                {
                    if (seg_from > box_end || seg_to < box_begin)
                        return false;

                    float length = seg_to - seg_from;
                    cmin = seg_from < box_begin ? (box_begin - seg_from) / length : 0f;
                    cmax = seg_to > box_end ? (box_end - seg_from) / length : 1f;
                }
                else
                {
                    if (seg_to > box_end || seg_from < box_begin)
                        return false;

                    float length = seg_to - seg_from;
                    cmin = seg_from > box_end ? (box_end - seg_from) / length : 0f;
                    cmax = seg_to < box_begin ? (box_begin - seg_from) / length : 1f;
                }

                if (cmin > min)
                {
                    min = cmin;
                }

                if (cmax < max)
                    max = cmax;
                if (max < min)
                    return false;
            }

            return true;
        }

        public Rect3 merge(Rect3 with)
        {
            Vector3 beg_1 = position;
            Vector3 beg_2 = with.position;
            Vector3 end_1 = new Vector3(size.x, size.y, size.z) + beg_1;
            Vector3 end_2 = new Vector3(with.size.x, with.size.y, with.size.z) + beg_2;

            Vector3 min = new Vector3(
                              (beg_1.x < beg_2.x) ? beg_1.x : beg_2.x,
                              (beg_1.y < beg_2.y) ? beg_1.y : beg_2.y,
                              (beg_1.z < beg_2.z) ? beg_1.z : beg_2.z
                          );

            Vector3 max = new Vector3(
                              (end_1.x > end_2.x) ? end_1.x : end_2.x,
                              (end_1.y > end_2.y) ? end_1.y : end_2.y,
                              (end_1.z > end_2.z) ? end_1.z : end_2.z
                          );

            return new Rect3(min, max - min);
        }

        public Rect3(Vector3 position, Vector3 size)
        {
            this.position = position;
            this.size = size;
        }

        public static bool operator ==(Rect3 left, Rect3 right)
        {
            return left.Equals(right);
        }

        public static bool operator !=(Rect3 left, Rect3 right)
        {
            return !left.Equals(right);
        }

        public override bool Equals(object obj)
        {
            if (obj is Rect3)
            {
                return Equals((Rect3)obj);
            }

            return false;
        }

        public bool Equals(Rect3 other)
        {
            return position == other.position && size == other.size;
        }

        public override int GetHashCode()
        {
            return position.GetHashCode() ^ size.GetHashCode();
        }

        public override string ToString()
        {
            return String.Format("{0} - {1}", new object[]
                {
                    this.position.ToString(),
                    this.size.ToString()
                });
        }

        public string ToString(string format)
        {
            return String.Format("{0} - {1}", new object[]
                {
                    this.position.ToString(format),
                    this.size.ToString(format)
                });
        }
    }
}
