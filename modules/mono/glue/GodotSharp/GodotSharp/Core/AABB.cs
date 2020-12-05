// file: core/math/aabb.h
// commit: 7ad14e7a3e6f87ddc450f7e34621eb5200808451
// file: core/math/aabb.cpp
// commit: bd282ff43f23fe845f29a3e25c8efc01bd65ffb0
// file: core/variant_call.cpp
// commit: 5ad9be4c24e9d7dc5672fdc42cea896622fe5685
using System;
using System.Runtime.InteropServices;
#if REAL_T_IS_DOUBLE
using real_t = System.Double;
#else
using real_t = System.Single;
#endif

namespace Godot
{
    /// <summary>
    /// Axis-Aligned Bounding Box. AABB consists of a position, a size, and
    /// several utility functions. It is typically used for fast overlap tests.
    /// </summary>
    [Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct AABB : IEquatable<AABB>
    {
        private Vector3 _position;
        private Vector3 _size;

        /// <summary>
        /// Beginning corner. Typically has values lower than End.
        /// </summary>
        /// <value>Directly uses a private field.</value>
        public Vector3 Position
        {
            get { return _position; }
            set { _position = value; }
        }

        /// <summary>
        /// Size from Position to End. Typically all components are positive.
        /// If the size is negative, you can use <see cref="Abs"/> to fix it.
        /// </summary>
        /// <value>Directly uses a private field.</value>
        public Vector3 Size
        {
            get { return _size; }
            set { _size = value; }
        }

        /// <summary>
        /// Ending corner. This is calculated as <see cref="Position"/> plus
        /// <see cref="Size"/>. Setting this value will change the size.
        /// </summary>
        /// <value>Getting is equivalent to `value = Position + Size`, setting is equivalent to `Size = value - Position`.</value>
        public Vector3 End
        {
            get { return _position + _size; }
            set { _size = value - _position; }
        }

        /// <summary>
        /// Returns an AABB with equivalent position and size, modified so that
        /// the most-negative corner is the origin and the size is positive.
        /// </summary>
        /// <returns>The modified AABB.</returns>
        public AABB Abs()
        {
            Vector3 end = End;
            Vector3 topLeft = new Vector3(Mathf.Min(_position.x, end.x), Mathf.Min(_position.y, end.y), Mathf.Min(_position.z, end.z));
            return new AABB(topLeft, _size.Abs());
        }

        /// <summary>
        /// Returns true if this AABB completely encloses another one.
        /// </summary>
        /// <param name="with">The other AABB that may be enclosed.</param>
        /// <returns>A bool for whether or not this AABB encloses `b`.</returns>
        public bool Encloses(AABB with)
        {
            Vector3 src_min = _position;
            Vector3 src_max = _position + _size;
            Vector3 dst_min = with._position;
            Vector3 dst_max = with._position + with._size;

            return src_min.x <= dst_min.x &&
                   src_max.x > dst_max.x &&
                   src_min.y <= dst_min.y &&
                   src_max.y > dst_max.y &&
                   src_min.z <= dst_min.z &&
                   src_max.z > dst_max.z;
        }

        /// <summary>
        /// Returns this AABB expanded to include a given point.
        /// </summary>
        /// <param name="point">The point to include.</param>
        /// <returns>The expanded AABB.</returns>
        public AABB Expand(Vector3 point)
        {
            Vector3 begin = _position;
            Vector3 end = _position + _size;

            if (point.x < begin.x)
            {
                begin.x = point.x;
            }
            if (point.y < begin.y)
            {
                begin.y = point.y;
            }
            if (point.z < begin.z)
            {
                begin.z = point.z;
            }

            if (point.x > end.x)
            {
                end.x = point.x;
            }
            if (point.y > end.y)
            {
                end.y = point.y;
            }
            if (point.z > end.z)
            {
                end.z = point.z;
            }

            return new AABB(begin, end - begin);
        }

        /// <summary>
        /// Returns the area of the AABB.
        /// </summary>
        /// <returns>The area.</returns>
        public real_t GetArea()
        {
            return _size.x * _size.y * _size.z;
        }

        /// <summary>
        /// Gets the position of one of the 8 endpoints of the AABB.
        /// </summary>
        /// <param name="idx">Which endpoint to get.</param>
        /// <returns>An endpoint of the AABB.</returns>
        public Vector3 GetEndpoint(int idx)
        {
            switch (idx)
            {
                case 0:
                    return new Vector3(_position.x, _position.y, _position.z);
                case 1:
                    return new Vector3(_position.x, _position.y, _position.z + _size.z);
                case 2:
                    return new Vector3(_position.x, _position.y + _size.y, _position.z);
                case 3:
                    return new Vector3(_position.x, _position.y + _size.y, _position.z + _size.z);
                case 4:
                    return new Vector3(_position.x + _size.x, _position.y, _position.z);
                case 5:
                    return new Vector3(_position.x + _size.x, _position.y, _position.z + _size.z);
                case 6:
                    return new Vector3(_position.x + _size.x, _position.y + _size.y, _position.z);
                case 7:
                    return new Vector3(_position.x + _size.x, _position.y + _size.y, _position.z + _size.z);
                default:
                    throw new ArgumentOutOfRangeException(nameof(idx), String.Format("Index is {0}, but a value from 0 to 7 is expected.", idx));
            }
        }

        /// <summary>
        /// Returns the normalized longest axis of the AABB.
        /// </summary>
        /// <returns>A vector representing the normalized longest axis of the AABB.</returns>
        public Vector3 GetLongestAxis()
        {
            var axis = new Vector3(1f, 0f, 0f);
            real_t max_size = _size.x;

            if (_size.y > max_size)
            {
                axis = new Vector3(0f, 1f, 0f);
                max_size = _size.y;
            }

            if (_size.z > max_size)
            {
                axis = new Vector3(0f, 0f, 1f);
            }

            return axis;
        }

        /// <summary>
        /// Returns the <see cref="Vector3.Axis"/> index of the longest axis of the AABB.
        /// </summary>
        /// <returns>A <see cref="Vector3.Axis"/> index for which axis is longest.</returns>
        public Vector3.Axis GetLongestAxisIndex()
        {
            var axis = Vector3.Axis.X;
            real_t max_size = _size.x;

            if (_size.y > max_size)
            {
                axis = Vector3.Axis.Y;
                max_size = _size.y;
            }

            if (_size.z > max_size)
            {
                axis = Vector3.Axis.Z;
            }

            return axis;
        }

        /// <summary>
        /// Returns the scalar length of the longest axis of the AABB.
        /// </summary>
        /// <returns>The scalar length of the longest axis of the AABB.</returns>
        public real_t GetLongestAxisSize()
        {
            real_t max_size = _size.x;

            if (_size.y > max_size)
                max_size = _size.y;

            if (_size.z > max_size)
                max_size = _size.z;

            return max_size;
        }

        /// <summary>
        /// Returns the normalized shortest axis of the AABB.
        /// </summary>
        /// <returns>A vector representing the normalized shortest axis of the AABB.</returns>
        public Vector3 GetShortestAxis()
        {
            var axis = new Vector3(1f, 0f, 0f);
            real_t max_size = _size.x;

            if (_size.y < max_size)
            {
                axis = new Vector3(0f, 1f, 0f);
                max_size = _size.y;
            }

            if (_size.z < max_size)
            {
                axis = new Vector3(0f, 0f, 1f);
            }

            return axis;
        }

        /// <summary>
        /// Returns the <see cref="Vector3.Axis"/> index of the shortest axis of the AABB.
        /// </summary>
        /// <returns>A <see cref="Vector3.Axis"/> index for which axis is shortest.</returns>
        public Vector3.Axis GetShortestAxisIndex()
        {
            var axis = Vector3.Axis.X;
            real_t max_size = _size.x;

            if (_size.y < max_size)
            {
                axis = Vector3.Axis.Y;
                max_size = _size.y;
            }

            if (_size.z < max_size)
            {
                axis = Vector3.Axis.Z;
            }

            return axis;
        }

        /// <summary>
        /// Returns the scalar length of the shortest axis of the AABB.
        /// </summary>
        /// <returns>The scalar length of the shortest axis of the AABB.</returns>
        public real_t GetShortestAxisSize()
        {
            real_t max_size = _size.x;

            if (_size.y < max_size)
                max_size = _size.y;

            if (_size.z < max_size)
                max_size = _size.z;

            return max_size;
        }

        /// <summary>
        /// Returns the support point in a given direction.
        /// This is useful for collision detection algorithms.
        /// </summary>
        /// <param name="dir">The direction to find support for.</param>
        /// <returns>A vector representing the support.</returns>
        public Vector3 GetSupport(Vector3 dir)
        {
            Vector3 half_extents = _size * 0.5f;
            Vector3 ofs = _position + half_extents;

            return ofs + new Vector3(
                dir.x > 0f ? -half_extents.x : half_extents.x,
                dir.y > 0f ? -half_extents.y : half_extents.y,
                dir.z > 0f ? -half_extents.z : half_extents.z);
        }

        /// <summary>
        /// Returns a copy of the AABB grown a given amount of units towards all the sides.
        /// </summary>
        /// <param name="by">The amount to grow by.</param>
        /// <returns>The grown AABB.</returns>
        public AABB Grow(real_t by)
        {
            var res = this;

            res._position.x -= by;
            res._position.y -= by;
            res._position.z -= by;
            res._size.x += 2.0f * by;
            res._size.y += 2.0f * by;
            res._size.z += 2.0f * by;

            return res;
        }

        /// <summary>
        /// Returns true if the AABB is flat or empty, or false otherwise.
        /// </summary>
        /// <returns>A bool for whether or not the AABB has area.</returns>
        public bool HasNoArea()
        {
            return _size.x <= 0f || _size.y <= 0f || _size.z <= 0f;
        }

        /// <summary>
        /// Returns true if the AABB has no surface (no size), or false otherwise.
        /// </summary>
        /// <returns>A bool for whether or not the AABB has area.</returns>
        public bool HasNoSurface()
        {
            return _size.x <= 0f && _size.y <= 0f && _size.z <= 0f;
        }

        /// <summary>
        /// Returns true if the AABB contains a point, or false otherwise.
        /// </summary>
        /// <param name="point">The point to check.</param>
        /// <returns>A bool for whether or not the AABB contains `point`.</returns>
        public bool HasPoint(Vector3 point)
        {
            if (point.x < _position.x)
                return false;
            if (point.y < _position.y)
                return false;
            if (point.z < _position.z)
                return false;
            if (point.x > _position.x + _size.x)
                return false;
            if (point.y > _position.y + _size.y)
                return false;
            if (point.z > _position.z + _size.z)
                return false;

            return true;
        }

        /// <summary>
        /// Returns the intersection of this AABB and `b`.
        /// </summary>
        /// <param name="with">The other AABB.</param>
        /// <returns>The clipped AABB.</returns>
        public AABB Intersection(AABB with)
        {
            Vector3 src_min = _position;
            Vector3 src_max = _position + _size;
            Vector3 dst_min = with._position;
            Vector3 dst_max = with._position + with._size;

            Vector3 min, max;

            if (src_min.x > dst_max.x || src_max.x < dst_min.x)
            {
                return new AABB();
            }

            min.x = src_min.x > dst_min.x ? src_min.x : dst_min.x;
            max.x = src_max.x < dst_max.x ? src_max.x : dst_max.x;

            if (src_min.y > dst_max.y || src_max.y < dst_min.y)
            {
                return new AABB();
            }

            min.y = src_min.y > dst_min.y ? src_min.y : dst_min.y;
            max.y = src_max.y < dst_max.y ? src_max.y : dst_max.y;

            if (src_min.z > dst_max.z || src_max.z < dst_min.z)
            {
                return new AABB();
            }

            min.z = src_min.z > dst_min.z ? src_min.z : dst_min.z;
            max.z = src_max.z < dst_max.z ? src_max.z : dst_max.z;

            return new AABB(min, max - min);
        }

        /// <summary>
        /// Returns true if the AABB overlaps with `b`
        /// (i.e. they have at least one point in common).
        ///
        /// If `includeBorders` is true, they will also be considered overlapping
        /// if their borders touch, even without intersection.
        /// </summary>
        /// <param name="with">The other AABB to check for intersections with.</param>
        /// <param name="includeBorders">Whether or not to consider borders.</param>
        /// <returns>A bool for whether or not they are intersecting.</returns>
        public bool Intersects(AABB with, bool includeBorders = false)
        {
            if (includeBorders)
            {
                if (_position.x > with._position.x + with._size.x)
                    return false;
                if (_position.x + _size.x < with._position.x)
                    return false;
                if (_position.y > with._position.y + with._size.y)
                    return false;
                if (_position.y + _size.y < with._position.y)
                    return false;
                if (_position.z > with._position.z + with._size.z)
                    return false;
                if (_position.z + _size.z < with._position.z)
                    return false;
            }
            else
            {
                if (_position.x >= with._position.x + with._size.x)
                    return false;
                if (_position.x + _size.x <= with._position.x)
                    return false;
                if (_position.y >= with._position.y + with._size.y)
                    return false;
                if (_position.y + _size.y <= with._position.y)
                    return false;
                if (_position.z >= with._position.z + with._size.z)
                    return false;
                if (_position.z + _size.z <= with._position.z)
                    return false;
            }

            return true;
        }

        /// <summary>
        /// Returns true if the AABB is on both sides of `plane`.
        /// </summary>
        /// <param name="plane">The plane to check for intersection.</param>
        /// <returns>A bool for whether or not the AABB intersects the plane.</returns>
        public bool IntersectsPlane(Plane plane)
        {
            Vector3[] points =
            {
                new Vector3(_position.x, _position.y, _position.z),
                new Vector3(_position.x, _position.y, _position.z + _size.z),
                new Vector3(_position.x, _position.y + _size.y, _position.z),
                new Vector3(_position.x, _position.y + _size.y, _position.z + _size.z),
                new Vector3(_position.x + _size.x, _position.y, _position.z),
                new Vector3(_position.x + _size.x, _position.y, _position.z + _size.z),
                new Vector3(_position.x + _size.x, _position.y + _size.y, _position.z),
                new Vector3(_position.x + _size.x, _position.y + _size.y, _position.z + _size.z)
            };

            bool over = false;
            bool under = false;

            for (int i = 0; i < 8; i++)
            {
                if (plane.DistanceTo(points[i]) > 0)
                {
                    over = true;
                }
                else
                {
                    under = true;
                }
            }

            return under && over;
        }

        /// <summary>
        /// Returns true if the AABB intersects the line segment between `from` and `to`.
        /// </summary>
        /// <param name="from">The start of the line segment.</param>
        /// <param name="to">The end of the line segment.</param>
        /// <returns>A bool for whether or not the AABB intersects the line segment.</returns>
        public bool IntersectsSegment(Vector3 from, Vector3 to)
        {
            real_t min = 0f;
            real_t max = 1f;

            for (int i = 0; i < 3; i++)
            {
                real_t segFrom = from[i];
                real_t segTo = to[i];
                real_t boxBegin = _position[i];
                real_t boxEnd = boxBegin + _size[i];
                real_t cmin, cmax;

                if (segFrom < segTo)
                {
                    if (segFrom > boxEnd || segTo < boxBegin)
                    {
                        return false;
                    }

                    real_t length = segTo - segFrom;
                    cmin = segFrom < boxBegin ? (boxBegin - segFrom) / length : 0f;
                    cmax = segTo > boxEnd ? (boxEnd - segFrom) / length : 1f;
                }
                else
                {
                    if (segTo > boxEnd || segFrom < boxBegin)
                    {
                        return false;
                    }

                    real_t length = segTo - segFrom;
                    cmin = segFrom > boxEnd ? (boxEnd - segFrom) / length : 0f;
                    cmax = segTo < boxBegin ? (boxBegin - segFrom) / length : 1f;
                }

                if (cmin > min)
                {
                    min = cmin;
                }

                if (cmax < max)
                {
                    max = cmax;
                }
                if (max < min)
                {
                    return false;
                }
            }

            return true;
        }

        /// <summary>
        /// Returns a larger AABB that contains this AABB and `b`.
        /// </summary>
        /// <param name="with">The other AABB.</param>
        /// <returns>The merged AABB.</returns>
        public AABB Merge(AABB with)
        {
            Vector3 beg1 = _position;
            Vector3 beg2 = with._position;
            var end1 = new Vector3(_size.x, _size.y, _size.z) + beg1;
            var end2 = new Vector3(with._size.x, with._size.y, with._size.z) + beg2;

            var min = new Vector3(
                              beg1.x < beg2.x ? beg1.x : beg2.x,
                              beg1.y < beg2.y ? beg1.y : beg2.y,
                              beg1.z < beg2.z ? beg1.z : beg2.z
                          );

            var max = new Vector3(
                              end1.x > end2.x ? end1.x : end2.x,
                              end1.y > end2.y ? end1.y : end2.y,
                              end1.z > end2.z ? end1.z : end2.z
                          );

            return new AABB(min, max - min);
        }

        /// <summary>
        /// Constructs an AABB from a position and size.
        /// </summary>
        /// <param name="position">The position.</param>
        /// <param name="size">The size, typically positive.</param>
        public AABB(Vector3 position, Vector3 size)
        {
            _position = position;
            _size = size;
        }

        /// <summary>
        /// Constructs an AABB from a position, width, height, and depth.
        /// </summary>
        /// <param name="position">The position.</param>
        /// <param name="width">The width, typically positive.</param>
        /// <param name="height">The height, typically positive.</param>
        /// <param name="depth">The depth, typically positive.</param>
        public AABB(Vector3 position, real_t width, real_t height, real_t depth)
        {
            _position = position;
            _size = new Vector3(width, height, depth);
        }

        /// <summary>
        /// Constructs an AABB from x, y, z, and size.
        /// </summary>
        /// <param name="x">The position's X coordinate.</param>
        /// <param name="y">The position's Y coordinate.</param>
        /// <param name="z">The position's Z coordinate.</param>
        /// <param name="size">The size, typically positive.</param>
        public AABB(real_t x, real_t y, real_t z, Vector3 size)
        {
            _position = new Vector3(x, y, z);
            _size = size;
        }

        /// <summary>
        /// Constructs an AABB from x, y, z, width, height, and depth.
        /// </summary>
        /// <param name="x">The position's X coordinate.</param>
        /// <param name="y">The position's Y coordinate.</param>
        /// <param name="z">The position's Z coordinate.</param>
        /// <param name="width">The width, typically positive.</param>
        /// <param name="height">The height, typically positive.</param>
        /// <param name="depth">The depth, typically positive.</param>
        public AABB(real_t x, real_t y, real_t z, real_t width, real_t height, real_t depth)
        {
            _position = new Vector3(x, y, z);
            _size = new Vector3(width, height, depth);
        }

        public static bool operator ==(AABB left, AABB right)
        {
            return left.Equals(right);
        }

        public static bool operator !=(AABB left, AABB right)
        {
            return !left.Equals(right);
        }

        public override bool Equals(object obj)
        {
            if (obj is AABB)
            {
                return Equals((AABB)obj);
            }

            return false;
        }

        public bool Equals(AABB other)
        {
            return _position == other._position && _size == other._size;
        }

        /// <summary>
        /// Returns true if this AABB and `other` are approximately equal, by running
        /// <see cref="Vector3.IsEqualApprox(Vector3)"/> on each component.
        /// </summary>
        /// <param name="other">The other AABB to compare.</param>
        /// <returns>Whether or not the AABBs are approximately equal.</returns>
        public bool IsEqualApprox(AABB other)
        {
            return _position.IsEqualApprox(other._position) && _size.IsEqualApprox(other._size);
        }

        public override int GetHashCode()
        {
            return _position.GetHashCode() ^ _size.GetHashCode();
        }

        public override string ToString()
        {
            return String.Format("{0} - {1}", new object[]
                {
                    _position.ToString(),
                    _size.ToString()
                });
        }

        public string ToString(string format)
        {
            return String.Format("{0} - {1}", new object[]
                {
                    _position.ToString(format),
                    _size.ToString(format)
                });
        }
    }
}
