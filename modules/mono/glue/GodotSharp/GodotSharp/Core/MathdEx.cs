using System;

namespace Godot
{
#if REAL_IS_DOUBLE
    public static partial class Mathf
#else
    public static partial class Mathd
#endif
    {
        /// <include file="Math.xml" path='doc/members/member[@name="E"]/*' />
        public const double E = (double)2.7182818284590452353602874714M;

        /// <include file="Math.xml" path='doc/members/member[@name="Sqrt2"]/*' />
        public const double Sqrt2 = (double)1.4142135623730950488016887242M;

        /// <include file="Math.xml" path='doc/members/member[@name="Epsilon"]/*' />
        public const double Epsilon = 1e-14;

        /// <include file="Math.xml" path='doc/members/member[@name="DecimalCount"]/*' />
        public static int DecimalCount(double s)
        {
            return DecimalCount((decimal)s);
        }

        /// <include file="Math.xml" path='doc/members/member[@name="DecimalCount"]/*' />
        public static int DecimalCount(decimal s)
        {
            return BitConverter.GetBytes(decimal.GetBits(s)[3])[2];
        }

        /// <include file="Math.xml" path='doc/members/member[@name="CeilToInt"]/*' />
        public static int CeilToInt(double s)
        {
            return (int)Math.Ceiling(s);
        }

        /// <include file="Math.xml" path='doc/members/member[@name="FloorToInt"]/*' />
        public static int FloorToInt(double s)
        {
            return (int)Math.Floor(s);
        }

        /// <include file="Math.xml" path='doc/members/member[@name="RoundToInt"]/*' />
        public static int RoundToInt(double s)
        {
            return (int)Math.Round(s);
        }

        /// <include file="Math.xml" path='doc/members/member[@name="IsEqualApprox(3)"]/*' />
        public static bool IsEqualApprox(double a, double b, double tolerance)
        {
            // Check for exact equality first, required to handle "infinity" values.
            if (a == b)
            {
                return true;
            }
            // Then check for approximate equality.
            return Abs(a - b) < tolerance;
        }
    }
}
