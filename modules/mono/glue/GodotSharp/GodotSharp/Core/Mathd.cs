using System;

namespace Godot
{
    /// <summary>
    /// Provides constants and static methods for common mathematical functions with single precision.
    /// </summary>
#if REAL_T_IS_DOUBLE
    public static partial class Mathf
#else
    public static partial class Mathd
#endif
    {
        /// <include file="Math.xml" path='doc/members/member[@name="Tau"]/*' />
        public const double Tau = (double)6.2831853071795864769252867666M;


        /// <include file="Math.xml" path='doc/members/member[@name="Pi"]/*' />
        public const double Pi = (double)3.1415926535897932384626433833M;

        /// <include file="Math.xml" path='doc/members/member[@name="Inf"]/*' />
        public const double Inf = double.PositiveInfinity;

        /// <include file="Math.xml" path='doc/members/member[@name="NaN"]/*' />
        public const double NaN = double.NaN;

        // 0.0174532924f and 0.0174532925199433
        private const double _degToRadConst = (double)0.0174532925199432957692369077M;
        // 57.29578f and 57.2957795130823
        private const double _radToDegConst = (double)57.295779513082320876798154814M;

        /// <include file="Math.xml" path='doc/members/member[@name="Abs"]/*' />
        public static double Abs(double s)
        {
            return Math.Abs(s);
        }

        /// <include file="Math.xml" path='doc/members/member[@name="Acos"]/*' />
        public static double Acos(double s)
        {
            return (double)Math.Acos(s);
        }

        /// <include file="Math.xml" path='doc/members/member[@name="Asin"]/*' />
        public static double Asin(double s)
        {
            return (double)Math.Asin(s);
        }

        /// <include file="Math.xml" path='doc/members/member[@name="Atan"]/*' />
        public static double Atan(double s)
        {
            return (double)Math.Atan(s);
        }

        /// <include file="Math.xml" path='doc/members/member[@name="Atan2"]/*' />
        public static double Atan2(double y, double x)
        {
            return (double)Math.Atan2(y, x);
        }

        /// <include file="Math.xml" path='doc/members/member[@name="Ceil"]/*' />
        public static double Ceil(double s)
        {
            return (double)Math.Ceiling(s);
        }

        /// <include file="Math.xml" path='doc/members/member[@name="Clamp"]/*' />
        public static double Clamp(double value, double min, double max)
        {
            return value < min ? min : value > max ? max : value;
        }

        /// <include file="Math.xml" path='doc/members/member[@name="Cos"]/*' />
        public static double Cos(double s)
        {
            return (double)Math.Cos(s);
        }

        /// <include file="Math.xml" path='doc/members/member[@name="Cosh"]/*' />
        public static double Cosh(double s)
        {
            return (double)Math.Cosh(s);
        }

        /// <include file="Math.xml" path='doc/members/member[@name="CubicInterpolate"]/*' />
        public static double CubicInterpolate(double from, double to, double pre, double post, double weight)
        {
            return 0.5f *
                    ((from * 2.0f) +
                            (-pre + to) * weight +
                            (2.0f * pre - 5.0f * from + 4.0f * to - post) * (weight * weight) +
                            (-pre + 3.0f * from - 3.0f * to + post) * (weight * weight * weight));
        }

        /// <include file="Math.xml" path='doc/members/member[@name="CubicInterpolateAngle"]/*' />
        public static double CubicInterpolateAngle(double from, double to, double pre, double post, double weight)
        {
            double fromRot = from % Mathf.Tau;

            double preDiff = (pre - fromRot) % Mathf.Tau;
            double preRot = fromRot + (2.0f * preDiff) % Mathf.Tau - preDiff;

            double toDiff = (to - fromRot) % Mathf.Tau;
            double toRot = fromRot + (2.0f * toDiff) % Mathf.Tau - toDiff;

            double postDiff = (post - toRot) % Mathf.Tau;
            double postRot = toRot + (2.0f * postDiff) % Mathf.Tau - postDiff;

            return CubicInterpolate(fromRot, toRot, preRot, postRot, weight);
        }

        /// <include file="Math.xml" path='doc/members/member[@name="CubicInterpolateInTime"]/*' />
        public static double CubicInterpolateInTime(double from, double to, double pre, double post, double weight, double toT, double preT, double postT)
        {
            /* Barry-Goldman method */
            double t = Lerp(0.0f, toT, weight);
            double a1 = Lerp(pre, from, preT == 0 ? 0.0f : (t - preT) / -preT);
            double a2 = Lerp(from, to, toT == 0 ? 0.5f : t / toT);
            double a3 = Lerp(to, post, postT - toT == 0 ? 1.0f : (t - toT) / (postT - toT));
            double b1 = Lerp(a1, a2, toT - preT == 0 ? 0.0f : (t - preT) / (toT - preT));
            double b2 = Lerp(a2, a3, postT == 0 ? 1.0f : t / postT);
            return Lerp(b1, b2, toT == 0 ? 0.5f : t / toT);
        }

        /// <include file="Math.xml" path='doc/members/member[@name="CubicInterpolateAngleInTime"]/*' />
        public static double CubicInterpolateAngleInTime(double from, double to, double pre, double post, double weight,
                    double toT, double preT, double postT)
        {
            double fromRot = from % Mathf.Tau;

            double preDiff = (pre - fromRot) % Mathf.Tau;
            double preRot = fromRot + (2.0f * preDiff) % Mathf.Tau - preDiff;

            double toDiff = (to - fromRot) % Mathf.Tau;
            double toRot = fromRot + (2.0f * toDiff) % Mathf.Tau - toDiff;

            double postDiff = (post - toRot) % Mathf.Tau;
            double postRot = toRot + (2.0f * postDiff) % Mathf.Tau - postDiff;

            return CubicInterpolateInTime(fromRot, toRot, preRot, postRot, weight, toT, preT, postT);
        }

        /// <include file="Math.xml" path='doc/members/member[@name="BezierInterpolate"]/*' />
        public static double BezierInterpolate(double start, double control1, double control2, double end, double t)
        {
            // Formula from Wikipedia article on Bezier curves
            double omt = 1 - t;
            double omt2 = omt * omt;
            double omt3 = omt2 * omt;
            double t2 = t * t;
            double t3 = t2 * t;

            return start * omt3 + control1 * omt2 * t * 3 + control2 * omt * t2 * 3 + end * t3;
        }

        /// <include file="Math.xml" path='doc/members/member[@name="DegToRad"]/*' />
        public static double DegToRad(double deg)
        {
            return deg * _degToRadConst;
        }

        /// <include file="Math.xml" path='doc/members/member[@name="Ease"]/*' />
        public static double Ease(double s, double curve)
        {
            if (s < 0f)
            {
                s = 0f;
            }
            else if (s > 1.0f)
            {
                s = 1.0f;
            }

            if (curve > 0f)
            {
                if (curve < 1.0f)
                {
                    return 1.0f - Pow(1.0f - s, 1.0f / curve);
                }

                return Pow(s, curve);
            }

            if (curve < 0f)
            {
                if (s < 0.5f)
                {
                    return Pow(s * 2.0f, -curve) * 0.5f;
                }

                return ((1.0f - Pow(1.0f - ((s - 0.5f) * 2.0f), -curve)) * 0.5f) + 0.5f;
            }

            return 0f;
        }

        /// <include file="Math.xml" path='doc/members/member[@name="Exp"]/*' />
        public static double Exp(double s)
        {
            return (double)Math.Exp(s);
        }

        /// <include file="Math.xml" path='doc/members/member[@name="Floor"]/*' />
        public static double Floor(double s)
        {
            return (double)Math.Floor(s);
        }

        /// <include file="Math.xml" path='doc/members/member[@name="InverseLerp"]/*' />
        public static double InverseLerp(double from, double to, double weight)
        {
            return (weight - from) / (to - from);
        }

        /// <include file="Math.xml" path='doc/members/member[@name="IsEqualApprox"]/*' />
        public static bool IsEqualApprox(double a, double b)
        {
            // Check for exact equality first, required to handle "infinity" values.
            if (a == b)
            {
                return true;
            }
            // Then check for approximate equality.
            double tolerance = Epsilon * Abs(a);
            if (tolerance < Epsilon)
            {
                tolerance = Epsilon;
            }
            return Abs(a - b) < tolerance;
        }

        /// <include file="Math.xml" path='doc/members/member[@name="IsInf"]/*' />
        public static bool IsInf(double s)
        {
            return double.IsInfinity(s);
        }

        /// <include file="Math.xml" path='doc/members/member[@name="IsNaN"]/*' />
        public static bool IsNaN(double s)
        {
            return double.IsNaN(s);
        }

        /// <include file="Math.xml" path='doc/members/member[@name="IsZeroApprox"]/*' />
        public static bool IsZeroApprox(double s)
        {
            return Abs(s) < Epsilon;
        }

        /// <include file="Math.xml" path='doc/members/member[@name="Lerp"]/*' />
        public static double Lerp(double from, double to, double weight)
        {
            return from + ((to - from) * weight);
        }

        /// <include file="Math.xml" path='doc/members/member[@name="LerpAngle"]/*' />
        public static double LerpAngle(double from, double to, double weight)
        {
            double difference = (to - from) % Mathf.Tau;
            double distance = ((2 * difference) % Mathf.Tau) - difference;
            return from + (distance * weight);
        }

        /// <include file="Math.xml" path='doc/members/member[@name="Log"]/*' />
        public static double Log(double s)
        {
            return (double)Math.Log(s);
        }

        /// <include file="Math.xml" path='doc/members/member[@name="Max"]/*' />
        public static double Max(double a, double b)
        {
            return a > b ? a : b;
        }

        /// <include file="Math.xml" path='doc/members/member[@name="Min"]/*' />
        public static double Min(double a, double b)
        {
            return a < b ? a : b;
        }

        /// <include file="Math.xml" path='doc/members/member[@name="MoveToward"]/*' />
        public static double MoveToward(double from, double to, double delta)
        {
            if (Abs(to - from) <= delta)
                return to;

            return from + (Sign(to - from) * delta);
        }

        /// <include file="Math.xml" path='doc/members/member[@name="PosMod"]/*' />
        public static double PosMod(double a, double b)
        {
            double c = a % b;
            if ((c < 0 && b > 0) || (c > 0 && b < 0))
            {
                c += b;
            }
            return c;
        }

        /// <include file="Math.xml" path='doc/members/member[@name="Pow"]/*' />
        public static double Pow(double x, double y)
        {
            return (double)Math.Pow(x, y);
        }

        /// <include file="Math.xml" path='doc/members/member[@name="RadToDeg"]/*' />
        public static double RadToDeg(double rad)
        {
            return rad * _radToDegConst;
        }

        /// <include file="Math.xml" path='doc/members/member[@name="Remap"]/*' />
        public static double Remap(double value, double inFrom, double inTo, double outFrom, double outTo)
        {
            return Lerp(outFrom, outTo, InverseLerp(inFrom, inTo, value));
        }

        /// <include file="Math.xml" path='doc/members/member[@name="Round"]/*' />
        public static double Round(double s)
        {
            return (double)Math.Round(s);
        }

        /// <include file="Math.xml" path='doc/members/member[@name="Sign"]/*' />
        public static int Sign(double s)
        {
            if (s == 0)
                return 0;
            return s < 0 ? -1 : 1;
        }

        /// <include file="Math.xml" path='doc/members/member[@name="Sin"]/*' />
        public static double Sin(double s)
        {
            return (double)Math.Sin(s);
        }

        /// <include file="Math.xml" path='doc/members/member[@name="Sinh"]/*' />
        public static double Sinh(double s)
        {
            return (double)Math.Sinh(s);
        }

        /// <include file="Math.xml" path='doc/members/member[@name="SmoothStep"]/*' />
        public static double SmoothStep(double from, double to, double weight)
        {
            if (IsEqualApprox(from, to))
            {
                return from;
            }
            double x = Clamp((weight - from) / (to - from), (double)0.0, (double)1.0);
            return x * x * (3 - (2 * x));
        }

        /// <include file="Math.xml" path='doc/members/member[@name="Sqrt"]/*' />
        public static double Sqrt(double s)
        {
            return (double)Math.Sqrt(s);
        }

        /// <include file="Math.xml" path='doc/members/member[@name="StepDecimals"]/*' />
        public static int StepDecimals(double step)
        {
            double[] sd = new double[]
            {
                0.9999,
                0.09999,
                0.009999,
                0.0009999,
                0.00009999,
                0.000009999,
                0.0000009999,
                0.00000009999,
                0.000000009999,
            };
            double abs = Abs(step);
            double decs = abs - (int)abs; // Strip away integer part
            for (int i = 0; i < sd.Length; i++)
            {
                if (decs >= sd[i])
                {
                    return i;
                }
            }
            return 0;
        }

        /// <include file="Math.xml" path='doc/members/member[@name="Snapped"]/*' />
        public static double Snapped(double s, double step)
        {
            if (step != 0f)
            {
                return Floor((s / step) + 0.5f) * step;
            }

            return s;
        }

        /// <include file="Math.xml" path='doc/members/member[@name="Tan"]/*' />
        public static double Tan(double s)
        {
            return (double)Math.Tan(s);
        }

        /// <include file="Math.xml" path='doc/members/member[@name="Tanh"]/*' />
        public static double Tanh(double s)
        {
            return (double)Math.Tanh(s);
        }

        /// <include file="Math.xml" path='doc/members/member[@name="Wrap"]/*' />
        public static double Wrap(double value, double min, double max)
        {
            double range = max - min;
            if (IsZeroApprox(range))
            {
                return min;
            }
            return min + ((((value - min) % range) + range) % range);
        }

        private static double Fract(double value)
        {
            return value - (double)Math.Floor(value);
        }

        /// <include file="Math.xml" path='doc/members/member[@name="PingPong"]/*' />
        public static double PingPong(double value, double length)
        {
            return (length != (double)0.0) ? Abs(Fract((value - length) / (length * (double)2.0)) * length * (double)2.0 - length) : (double)0.0;
        }
    }
}
