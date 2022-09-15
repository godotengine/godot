using System;

namespace Godot
{
    public static partial class Mathf
    {
        /// <include file="Math.xml" path='doc/members/member[@name="Abs"]/*' />
        public static int Abs(int s)
        {
            return Math.Abs(s);
        }

        /// <include file="Math.xml" path='doc/members/member[@name="Clamp"]/*' />
        public static int Clamp(int value, int min, int max)
        {
            return value < min ? min : value > max ? max : value;
        }

        /// <include file="Math.xml" path='doc/members/member[@name="Max"]/*' />
        public static int Max(int a, int b)
        {
            return a > b ? a : b;
        }

        /// <include file="Math.xml" path='doc/members/member[@name="Min"]/*' />
        public static int Min(int a, int b)
        {
            return a < b ? a : b;
        }

        /// <include file="Math.xml" path='doc/members/member[@name="NearestPo2"]/*' />
        public static int NearestPo2(int value)
        {
            value--;
            value |= value >> 1;
            value |= value >> 2;
            value |= value >> 4;
            value |= value >> 8;
            value |= value >> 16;
            value++;
            return value;
        }

        /// <include file="Math.xml" path='doc/members/member[@name="PosMod"]/*' />
        public static int PosMod(int a, int b)
        {
            int c = a % b;
            if ((c < 0 && b > 0) || (c > 0 && b < 0))
            {
                c += b;
            }
            return c;
        }

        /// <include file="Math.xml" path='doc/members/member[@name="Sign"]/*' />
        public static int Sign(int s)
        {
            if (s == 0)
                return 0;
            return s < 0 ? -1 : 1;
        }

        /// <include file="Math.xml" path='doc/members/member[@name="Wrap"]/*' />
        public static int Wrap(int value, int min, int max)
        {
            int range = max - min;
            if (range == 0)
                return min;

            return min + ((((value - min) % range) + range) % range);
        }
    }
}
