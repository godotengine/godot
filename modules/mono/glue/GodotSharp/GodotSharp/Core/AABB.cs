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
        /// Beginning corner. Typically has values lower than <see cref="End"/>.
        /// </summary>
        /// <value>Directly uses a private field.</value>
        public Vector3 Position
        {
            get { return _position; }
            set { _position = value; }
        }

        /// <summary>
        /// Size from <see cref="Position"/> to <see cref="End"/>. Typically all components are positive.
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
        /// <value>
        /// Getting is equivalent to <paramref name="value"/> = <see cref="Position"/> + <see cref="Size"/>,
        /// setting is equivalent to <see cref="Size"/> = <paramref name="value"/> - <see cref="Position"/>
        /// </value>
        public Vector3 End
        {
            get { return _position + _size; }
            set { _size = value - _position; }
        }

        /// <summary>
        /// Returns an <see cref="AABB"/> with equivalent position and size, modified so that
        /// the most-negative corner is the origin and the size is positive.
        /// </summary>
        /// <returns>The modified <see cref="AABB"/>.</returns>
        public AABB Abs()
        {
            Vector3 end = End;
            Vector3 topLeft = new Vector3(Mathf.Min(_position.x, end.x), Mathf.Min(_position.y, end.y), Mathf.Min(_position.z, end.z));
            return new AABB(topLeft, _size.Abs());
        }

        /// <summary>
        /// Returns the center of the <see cref="AABB"/>, which is equal
        /// to <see cref="Position"/> + (<see cref="Size"/> / 2).
        /// </summary>
        /// <returns>The center.</returns>
        public Vector3 GetCenter()
        {
            return _position + (_size * 0.5f);
        }

        /// <summary>
        /// Returns <see langword="true"/> if this <see cref="AABB"/> completely encloses another one.
        /// </summary>
        /// <param name="with">The other <see cref="AABB"/> that may be enclosed.</param>
        /// <returns>
        /// A <see langword="bool"/> for whether or not this <see cref="AABB"/> encloses <paramref name="with"/>.
        /// </returns>
        public bool Encloses(AABB with)
        {
            Vector3 srcMin = _position;
            Vector3 srcMax = _position + _size;
            Vector3 dstMin = with._position;
            Vector3 dstMax = with._position + with._size;

            return srcMin.x <= dstMin.x &&
                   srcMax.x > dstMax.x &&
                   srcMin.y <= dstMin.y &&
                   srcMax.y > dstMax.y &&
                   srcMin.z <= dstMin.z &&
                   srcMax.z > dstMax.z;
        }

        /// <summary>
        /// Returns this <see cref="AABB"/> expanded to include a given point.
        /// </summary>
        /// <param name="point">The point to include.</param>
        /// <returns>The expanded <see cref="AABB"/>.</returns>
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
        /// Returns the area of the <see cref="AABB"/>.
        /// </summary>
        /// <returns>The area.</returns>
        public real_t GetArea()
        {
            return _size.x * _size.y * _size.z;
        }

        /// <summary>
        /// Gets the position of one of the 8 endpoints of the <see cref="AABB"/>.
        /// </summary>
        /// <param name="idx">Which endpoint to get.</param>
        /// <returns>An endpoint of the <see cref="AABB"/>.</returns>
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
                {
                    throw new ArgumentOutOfRangeException(nameof(idx),
                        $"Index is {idx}, but a value from 0 to 7 is expected.");
                }
            }
        }

        /// <summary>
        /// Returns the normalized longest axis of the <see cref="AABB"/>.
        /// </summary>
        /// <returns>A vector representing the normalized longest axis of the <see cref="AABB"/>.</returns>
        public Vector3 GetLongestAxis()
        {
            var axis = new Vector3(1f, 0f, 0f);
            real_t maxSize = _size.x;

            if (_size.y > maxSize)
            {
                axis = new Vector3(0f, 1f, 0f);
                maxSize = _size.y;
            }

            if (_size.z > maxSize)
            {
                axis = new Vector3(0f, 0f, 1f);
            }

            return axis;
        }

        /// <summary>
        /// Returns the <see cref="Vector3.Axis"/> index of the longest axis of the <see cref="AABB"/>.
        /// </summary>
        /// <returns>A <see cref="Vector3.Axis"/> index for which axis is longest.</returns>
        public Vector3.Axis GetLongestAxisIndex()
        {
            var axis = Vector3.Axis.X;
            real_t maxSize = _size.x;

            if (_size.y > maxSize)
            {
                axis = Vector3.Axis.Y;
                maxSize = _size.y;
            }

            if (_size.z > maxSize)
            {
                axis = Vector3.Axis.Z;
            }

            return axis;
        }

        /// <summary>
        /// Returns the scalar length of the longest axis of the <see cref="AABB"/>.
        /// </summary>
        /// <returns>The scalar length of the longest axis of the <see cref="AABB"/>.</returns>
        public real_t GetLongestAxisSize()
        {
            real_t maxSize = _size.x;

            if (_size.y > maxSize)
                maxSize = _size.y;

            if (_size.z > maxSize)
                maxSize = _size.z;

            return maxSize;
        }

        /// <summary>
        /// Returns the normalized shortest axis of the <see cref="AABB"/>.
        /// </summary>
        /// <returns>A vector representing the normalized shortest axis of the <see cref="AABB"/>.</returns>
        public Vector3 GetShortestAxis()
        {
            var axis = new Vector3(1f, 0f, 0f);
            real_t maxSize = _size.x;

            if (_size.y < maxSize)
            {
                axis = new Vector3(0f, 1f, 0f);
                maxSize = _size.y;
            }

            if (_size.z < maxSize)
            {
                axis = new Vector3(0f, 0f, 1f);
            }

            return axis;
        }

        /// <summary>
        /// Returns the <see cref="Vector3.Axis"/> index of the shortest axis of the <see cref="AABB"/>.
        /// </summary>
        /// <returns>A <see cref="Vector3.Axis"/> index for which axis is shortest.</returns>
        public Vector3.Axis GetShortestAxisIndex()
        {
            var axis = Vector3.Axis.X;
            real_t maxSize = _size.x;

            if (_size.y < maxSize)
            {
                axis = Vector3.Axis.Y;
                maxSize = _size.y;
            }

            if (_size.z < maxSize)
            {
                axis = Vector3.Axis.Z;
            }

            return axis;
        }

        /// <summary>
        /// Returns the scalar length of the shortest axis of the <see cref="AABB"/>.
        /// </summary>
        /// <returns>The scalar length of the shortest axis of the <see cref="AABB"/>.</returns>
        public real_t GetShortestAxisSize()
        {
            real_t maxSize = _size.x;

            if (_size.y < maxSize)
                maxSize = _size.y;

            if (_size.z < maxSize)
                maxSize = _size.z;

            return maxSize;
        }

        /// <summary>
        /// Returns the support point in a given direction.
        /// This is useful for collision detection algorithms.
        /// </summary>
        /// <param name="dir">The direction to find support for.</param>
        /// <returns>A vector representing the support.</returns>
        public Vector3 GetSupport(Vector3 dir)
        {
            Vector3 halfExtents = _size * 0.5f;
            Vector3 ofs = _position + halfExtents;

            return ofs + new Vector3(
                dir.x > 0f ? -halfExtents.x : halfExtents.x,
                dir.y > 0f ? -halfExtents.y : halfExtents.y,
                dir.z > 0f ? -halfExtents.z : halfExtents.z);
        }

        /// <summary>
        /// Returns a copy of the <see cref="AABB"/> grown a given amount of units towards all the sides.
        /// </summary>
        /// <param name="by">The amount to grow by.</param>
        /// <returns>The grown <see cref="AABB"/>.</returns>
        public AABB Grow(real_t by)
        {
            AABB res = this;

            res._position.x -= by;
            res._position.y -= by;
            res._position.z -= by;
            res._size.x += 2.0f * by;
            res._size.y += 2.0f * by;
            res._size.z += 2.0f * by;

            return res;
        }

        /// <summary>
        /// Returns <see langword="true"/> if the <see cref="AABB"/> is flat or empty,
        /// or <see langword="false"/> otherwise.
        /// </summary>
        /// <returns>
        /// A <see langword="bool"/> for whether or not the <see cref="AABB"/> has area.
        /// </returns>
        public bool HasNoArea()
        {
            return _size.x <= 0f || _size.y <= 0f || _size.z <= 0f;
        }

        /// <summary>
        /// Returns <see langword="true"/> if the <see cref="AABB"/> has no surface (no size),
        /// or <see langword="false"/> otherwise.
        /// </summary>
        /// <returns>
        /// A <see langword="bool"/> for whether or not the <see cref="AABB"/> has area.
        /// </returns>
        public bool HasNoSurface()
        {
            return _size.x <= 0f && _size.y <= 0f && _size.z <= 0f;
        }

        /// <summary>
        /// Returns <see langword="true"/> if the <see cref="AABB"/> contains a point,
        /// or <see langword="false"/> otherwise.
        /// </summary>
        /// <param name="point">The point to check.</param>
        /// <returns>
        /// A <see langword="bool"/> for whether or not the <see cref="AABB"/> contains <paramref name="point"/>.
        /// </returns>
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
        /// Returns the intersection of this <see cref="AABB"/> and <paramref name="with"/>.
        /// </summary>
        /// <param name="with">The other <see cref="AABB"/>.</param>
        /// <returns>The clipped <see cref="AABB"/>.</returns>
        public AABB Intersection(AABB with)
        {
            Vector3 srcMin = _position;
            Vector3 srcMax = _position + _size;
            Vector3 dstMin = with._position;
            Vector3 dstMax = with._position + with._size;

            Vector3 min, max;

            if (srcMin.x > dstMax.x || srcMax.x < dstMin.x)
            {
                return new AABB();
            }

            min.x = srcMin.x > dstMin.x ? srcMin.x : dstMin.x;
            max.x = srcMax.x < dstMax.x ? srcMax.x : dstMax.x;

            if (srcMin.y > dstMax.y || srcMax.y < dstMin.y)
            {
                return new AABB();
            }

            min.y = srcMin.y > dstMin.y ? srcMin.y : dstMin.y;
            max.y = srcMax.y < dstMax.y ? srcMax.y : dstMax.y;

            if (srcMin.z > dstMax.z || srcMax.z < dstMin.z)
            {
                return new AABB();
            }

            min.z = srcMin.z > dstMin.z ? srcMin.z : dstMin.z;
            max.z = srcMax.z < dstMax.z ? srcMax.z : dstMax.z;

            return new AABB(min, max - min);
        }

        /// <summary>
        /// Returns <see langword="true"/> if the <see cref="AABB"/> overlaps with <paramref name="with"/>
        /// (i.e. they have at least one point in common).
        ///
        /// If <paramref name="includeBorders"/> is <see langword="true"/>,
        /// they will also be considered overlapping if their borders touch,
        /// even without intersection.
        /// </summary>
        /// <param name="with">The other <see cref="AABB"/> to check for intersections with.</param>
        /// <param name="includeBorders">Whether or not to consider borders.</param>
        /// <returns>
        /// A <see langword="bool"/> for whether or not they are intersecting.
        /// </returns>
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
        /// Returns <see langword="true"/> if the <see cref="AABB"/> is on both sides of <paramref name="plane"/>.
        /// </summary>
        /// <param name="plane">The <see cref="Plane"/> to check for intersection.</param>
        /// <returns>
        /// A <see langword="bool"/> for whether or not the <see cref="AABB"/> intersects the <see cref="Plane"/>.
        /// </returns>
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
        /// Returns <see langword="true"/> if the <see cref="AABB"/> intersects
        /// the line segment between <paramref name="from"/> and <paramref name="to"/>.
        /// </summary>
        /// <param name="from">The start of the line segment.</param>
        /// <param name="to">The end of the line segment.</param>
        /// <returns>
        /// A <see langword="bool"/> for whether or not the <see cref="AABB"/> intersects the line segment.
        /// </returns>
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
        /// Returns a larger <see cref="AABB"/> that contains this <see cref="AABB"/> and <paramref name="with"/>.
        /// </summary>
        /// <param name="with">The other <see cref="AABB"/>.</param>
        /// <returns>The merged <see cref="AABB"/>.</returns>
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
        /// Constructs an <see cref="AABB"/> from a position and size.
        /// </summary>
        /// <param name="position">The position.</param>
        /// <param name="size">The size, typically positive.</param>
        public AABB(Vector3 position, Vector3 size)
        {
            _position = position;
            _size = size;
        }

        /// <summary>
        /// Constructs an <see cref="AABB"/> from a <paramref name="position"/>,
        /// <paramref name="width"/>, <paramref name="height"/>, and <paramref name="depth"/>.
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
        /// Constructs an <see cref="AABB"/> from <paramref name="x"/>,
        /// <paramref name="y"/>, <paramref name="z"/>, and <paramref name="size"/>.
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
        /// Constructs an <see cref="AABB"/> from <paramref name="x"/>,
        /// <paramref name="y"/>, <paramref name="z"/>, <paramref name="width"/>,
        /// <paramref name="height"/>, and <paramref name="depth"/>.
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

        /// <summary>
        /// Returns <see langword="true"/> if the AABBs are exactly equal.
        /// Note: Due to floating-point precision errors, consider using
        /// <see cref="IsEqualApprox"/> instead, which is more reliable.
        /// </summary>
        /// <param name="left">The left AABB.</param>
        /// <param name="right">The right AABB.</param>
        /// <returns>Whether or not the AABBs are exactly equal.</returns>
        public static bool operator ==(AABB left, AABB right)
        {
            return left.Equals(right);
        }

        /// <summary>
        /// Returns <see langword="true"/> if the AABBs are not equal.
        /// Note: Due to floating-point precision errors, consider using
        /// <see cref="IsEqualApprox"/> instead, which is more reliable.
        /// </summary>
        /// <param name="left">The left AABB.</param>
        /// <param name="right">The right AABB.</param>
        /// <returns>Whether or not the AABBs are not equal.</returns>
        public static bool operator !=(AABB left, AABB right)
        {
            return !left.Equals(right);
        }

        /// <summary>
        /// Returns <see langword="true"/> if the AABB is exactly equal
        /// to the given object (<see paramref="obj"/>).
        /// Note: Due to floating-point precision errors, consider using
        /// <see cref="IsEqualApprox"/> instead, which is more reliable.
        /// </summary>
        /// <param name="obj">The object to compare with.</param>
        /// <returns>Whether or not the AABB and the object are equal.</returns>
        public override bool Equals(object obj)
        {
            if (obj is AABB)
            {
                return Equals((AABB)obj);
            }

            return false;
        }

        /// <summary>
        /// Returns <see langword="true"/> if the AABBs are exactly equal.
        /// Note: Due to floating-point precision errors, consider using
        /// <see cref="IsEqualApprox"/> instead, which is more reliable.
        /// </summary>
        /// <param name="other">The other AABB.</param>
        /// <returns>Whether or not the AABBs are exactly equal.</returns>
        public bool Equals(AABB other)
        {
            return _position == other._position && _size == other._size;
        }

        /// <summary>
        /// Returns <see langword="true"/> if this AABB and <paramref name="other"/> are approximately equal,
        /// by running <see cref="Vector3.IsEqualApprox(Vector3)"/> on each component.
        /// </summary>
        /// <param name="other">The other AABB to compare.</param>
        /// <returns>Whether or not the AABBs structures are approximately equal.</returns>
        public bool IsEqualApprox(AABB other)
        {
            return _position.IsEqualApprox(other._position) && _size.IsEqualApprox(other._size);
        }

        /// <summary>
        /// Serves as the hash function for <see cref="AABB"/>.
        /// </summary>
        /// <returns>A hash code for this AABB.</returns>
        public override int GetHashCode()
        {
            return _position.GetHashCode() ^ _size.GetHashCode();
        }

        /// <summary>
        /// Converts this <see cref="AABB"/> to a string.
        /// </summary>
        /// <returns>A string representation of this AABB.</returns>
        public override string ToString()
        {
            return $"{_position}, {_size}";
        }

        /// <summary>
        /// Converts this <see cref="AABB"/> to a string with the given <paramref name="format"/>.
        /// </summary>
        /// <returns>A string representation of this AABB.</returns>
        public string ToString(string format)
        {
            return $"{_position.ToString(format)}, {_size.ToString(format)}";
        }
    }
}
