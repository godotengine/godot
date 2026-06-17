using System;

public static partial class Mathf
{

	public const real_t Tau = (real_t)6.2831853071795864769252867666M;

	public const real_t Pi = (real_t)3.1415926535897932384626433833M;

	public const real_t Inf = real_t.PositiveInfinity;

	public const real_t NaN = real_t.NaN;

	private const float DegToRadConstF = (float)0.0174532925199432957692369077M;
	private const double DegToRadConstD = (double)0.0174532925199432957692369077M;
	// 57.29578f and 57.2957795130823
	private const float RadToDegConstF = (float)57.295779513082320876798154814M;
	private const double RadToDegConstD = (double)57.295779513082320876798154814M;

	public static int Abs(int s)
	{
		return Math.Abs(s);
	}

	public static float Abs(float s)
	{
		return Math.Abs(s);
	}

	public static double Abs(double s)
	{
		return Math.Abs(s);
	}

	public static float Acos(float s)
	{
		return MathF.Acos(s);
	}

	public static double Acos(double s)
	{
		return Math.Acos(s);
	}

	public static float Acosh(float s)
	{
		return MathF.Acosh(s);
	}

	public static double Acosh(double s)
	{
		return Math.Acosh(s);
	}

	public static float AngleDifference(float from, float to)
	{
		float difference = (to - from) % MathF.Tau;
		return ((2.0f * difference) % MathF.Tau) - difference;
	}

	public static double AngleDifference(double from, double to)
	{
		double difference = (to - from) % Math.Tau;
		return ((2.0 * difference) % Math.Tau) - difference;
	}

	public static float Asin(float s)
	{
		return MathF.Asin(s);
	}

	public static double Asin(double s)
	{
		return Math.Asin(s);
	}

	public static float Asinh(float s)
	{
		return MathF.Asinh(s);
	}

	public static double Asinh(double s)
	{
		return Math.Asinh(s);
	}

	public static float Atan(float s)
	{
		return MathF.Atan(s);
	}

	public static double Atan(double s)
	{
		return Math.Atan(s);
	}

	public static float Atan2(float y, float x)
	{
		return MathF.Atan2(y, x);
	}

	public static double Atan2(double y, double x)
	{
		return Math.Atan2(y, x);
	}

	public static float Atanh(float s)
	{
		return MathF.Atanh(s);
	}

	public static double Atanh(double s)
	{
		return Math.Atanh(s);
	}

	public static float Ceil(float s)
	{
		return MathF.Ceiling(s);
	}

	public static double Ceil(double s)
	{
		return Math.Ceiling(s);
	}

	public static int Clamp(int value, int min, int max)
	{
		return Math.Clamp(value, min, max);
	}

	public static float Clamp(float value, float min, float max)
	{
		return Math.Clamp(value, min, max);
	}

	public static double Clamp(double value, double min, double max)
	{
		return Math.Clamp(value, min, max);
	}

	public static float Cos(float s)
	{
		return MathF.Cos(s);
	}

	public static double Cos(double s)
	{
		return Math.Cos(s);
	}

	public static float Cosh(float s)
	{
		return MathF.Cosh(s);
	}

	public static double Cosh(double s)
	{
		return Math.Cosh(s);
	}

	public static float CubicInterpolate(float from, float to, float pre, float post, float weight)
	{
		return 0.5f *
				((from * 2.0f) +
						(-pre + to) * weight +
						(2.0f * pre - 5.0f * from + 4.0f * to - post) * (weight * weight) +
						(-pre + 3.0f * from - 3.0f * to + post) * (weight * weight * weight));
	}

	public static double CubicInterpolate(double from, double to, double pre, double post, double weight)
	{
		return 0.5 *
				((from * 2.0) +
						(-pre + to) * weight +
						(2.0 * pre - 5.0 * from + 4.0 * to - post) * (weight * weight) +
						(-pre + 3.0 * from - 3.0 * to + post) * (weight * weight * weight));
	}

	public static float CubicInterpolateAngle(float from, float to, float pre, float post, float weight)
	{
		float fromRot = from % MathF.Tau;

		float preDiff = (pre - fromRot) % MathF.Tau;
		float preRot = fromRot + (2.0f * preDiff) % MathF.Tau - preDiff;

		float toDiff = (to - fromRot) % MathF.Tau;
		float toRot = fromRot + (2.0f * toDiff) % MathF.Tau - toDiff;

		float postDiff = (post - toRot) % MathF.Tau;
		float postRot = toRot + (2.0f * postDiff) % MathF.Tau - postDiff;

		return CubicInterpolate(fromRot, toRot, preRot, postRot, weight);
	}

	public static double CubicInterpolateAngle(double from, double to, double pre, double post, double weight)
	{
		double fromRot = from % Math.Tau;

		double preDiff = (pre - fromRot) % Math.Tau;
		double preRot = fromRot + (2.0 * preDiff) % Math.Tau - preDiff;

		double toDiff = (to - fromRot) % Math.Tau;
		double toRot = fromRot + (2.0 * toDiff) % Math.Tau - toDiff;

		double postDiff = (post - toRot) % Math.Tau;
		double postRot = toRot + (2.0 * postDiff) % Math.Tau - postDiff;

		return CubicInterpolate(fromRot, toRot, preRot, postRot, weight);
	}

	public static float CubicInterpolateInTime(float from, float to, float pre, float post, float weight, float toT, float preT, float postT)
	{
		/* Barry-Goldman method */
		float t = Lerp(0.0f, toT, weight);
		float a1 = Lerp(pre, from, preT == 0 ? 0.0f : (t - preT) / -preT);
		float a2 = Lerp(from, to, toT == 0 ? 0.5f : t / toT);
		float a3 = Lerp(to, post, postT - toT == 0 ? 1.0f : (t - toT) / (postT - toT));
		float b1 = Lerp(a1, a2, toT - preT == 0 ? 0.0f : (t - preT) / (toT - preT));
		float b2 = Lerp(a2, a3, postT == 0 ? 1.0f : t / postT);
		return Lerp(b1, b2, toT == 0 ? 0.5f : t / toT);
	}

	public static double CubicInterpolateInTime(double from, double to, double pre, double post, double weight, double toT, double preT, double postT)
	{
		/* Barry-Goldman method */
		double t = Lerp(0.0, toT, weight);
		double a1 = Lerp(pre, from, preT == 0 ? 0.0 : (t - preT) / -preT);
		double a2 = Lerp(from, to, toT == 0 ? 0.5 : t / toT);
		double a3 = Lerp(to, post, postT - toT == 0 ? 1.0 : (t - toT) / (postT - toT));
		double b1 = Lerp(a1, a2, toT - preT == 0 ? 0.0 : (t - preT) / (toT - preT));
		double b2 = Lerp(a2, a3, postT == 0 ? 1.0 : t / postT);
		return Lerp(b1, b2, toT == 0 ? 0.5 : t / toT);
	}

	public static float CubicInterpolateAngleInTime(float from, float to, float pre, float post, float weight, float toT, float preT, float postT)
	{
		float fromRot = from % MathF.Tau;

		float preDiff = (pre - fromRot) % MathF.Tau;
		float preRot = fromRot + (2.0f * preDiff) % MathF.Tau - preDiff;

		float toDiff = (to - fromRot) % MathF.Tau;
		float toRot = fromRot + (2.0f * toDiff) % MathF.Tau - toDiff;

		float postDiff = (post - toRot) % MathF.Tau;
		float postRot = toRot + (2.0f * postDiff) % MathF.Tau - postDiff;

		return CubicInterpolateInTime(fromRot, toRot, preRot, postRot, weight, toT, preT, postT);
	}

	public static double CubicInterpolateAngleInTime(double from, double to, double pre, double post, double weight, double toT, double preT, double postT)
	{
		double fromRot = from % Math.Tau;

		double preDiff = (pre - fromRot) % Math.Tau;
		double preRot = fromRot + (2.0 * preDiff) % Math.Tau - preDiff;

		double toDiff = (to - fromRot) % Math.Tau;
		double toRot = fromRot + (2.0 * toDiff) % Math.Tau - toDiff;

		double postDiff = (post - toRot) % Math.Tau;
		double postRot = toRot + (2.0 * postDiff) % Math.Tau - postDiff;

		return CubicInterpolateInTime(fromRot, toRot, preRot, postRot, weight, toT, preT, postT);
	}

	public static float BezierInterpolate(float start, float control1, float control2, float end, float t)
	{
		// Formula from Wikipedia article on Bezier curves
		float omt = 1.0f - t;
		float omt2 = omt * omt;
		float omt3 = omt2 * omt;
		float t2 = t * t;
		float t3 = t2 * t;

		return start * omt3 + control1 * omt2 * t * 3.0f + control2 * omt * t2 * 3.0f + end * t3;
	}

	public static double BezierInterpolate(double start, double control1, double control2, double end, double t)
	{
		// Formula from Wikipedia article on Bezier curves
		double omt = 1.0 - t;
		double omt2 = omt * omt;
		double omt3 = omt2 * omt;
		double t2 = t * t;
		double t3 = t2 * t;

		return start * omt3 + control1 * omt2 * t * 3.0 + control2 * omt * t2 * 3.0 + end * t3;
	}

	public static float BezierDerivative(float start, float control1, float control2, float end, float t)
	{
		// Formula from Wikipedia article on Bezier curves
		float omt = 1.0f - t;
		float omt2 = omt * omt;
		float t2 = t * t;

		float d = (control1 - start) * 3.0f * omt2 + (control2 - control1) * 6.0f * omt * t + (end - control2) * 3.0f * t2;
		return d;
	}


	public static double BezierDerivative(double start, double control1, double control2, double end, double t)
	{
		// Formula from Wikipedia article on Bezier curves
		double omt = 1.0 - t;
		double omt2 = omt * omt;
		double t2 = t * t;

		double d = (control1 - start) * 3.0 * omt2 + (control2 - control1) * 6.0 * omt * t + (end - control2) * 3.0 * t2;
		return d;
	}

	public static float DbToLinear(float db)
	{
		return MathF.Exp(db * 0.11512925464970228420089957273422f);
	}

	public static double DbToLinear(double db)
	{
		return Math.Exp(db * 0.11512925464970228420089957273422);
	}

	public static float DegToRad(float deg)
	{
		return deg * DegToRadConstF;
	}

	public static double DegToRad(double deg)
	{
		return deg * DegToRadConstD;
	}

	public static float Ease(float s, float curve)
	{
		if (s < 0.0f)
		{
			s = 0.0f;
		}
		else if (s > 1.0f)
		{
			s = 1.0f;
		}

		if (curve > 0.0f)
		{
			if (curve < 1.0f)
			{
				return 1.0f - MathF.Pow(1.0f - s, 1.0f / curve);
			}

			return MathF.Pow(s, curve);
		}

		if (curve < 0.0f)
		{
			if (s < 0.5f)
			{
				return MathF.Pow(s * 2.0f, -curve) * 0.5f;
			}

			return ((1.0f - MathF.Pow(1.0f - ((s - 0.5f) * 2.0f), -curve)) * 0.5f) + 0.5f;
		}

		return 0.0f;
	}

	public static double Ease(double s, double curve)
	{
		if (s < 0.0)
		{
			s = 0.0;
		}
		else if (s > 1.0)
		{
			s = 1.0;
		}

		if (curve > 0)
		{
			if (curve < 1.0)
			{
				return 1.0 - Math.Pow(1.0 - s, 1.0 / curve);
			}

			return Math.Pow(s, curve);
		}

		if (curve < 0.0)
		{
			if (s < 0.5)
			{
				return Math.Pow(s * 2.0, -curve) * 0.5;
			}

			return ((1.0 - Math.Pow(1.0 - ((s - 0.5) * 2.0), -curve)) * 0.5) + 0.5;
		}

		return 0.0;
	}

	public static float Exp(float s)
	{
		return MathF.Exp(s);
	}

	public static double Exp(double s)
	{
		return Math.Exp(s);
	}

	public static float Floor(float s)
	{
		return MathF.Floor(s);
	}

	public static double Floor(double s)
	{
		return Math.Floor(s);
	}

	public static float InverseLerp(float from, float to, float weight)
	{
		return (weight - from) / (to - from);
	}

	public static double InverseLerp(double from, double to, double weight)
	{
		return (weight - from) / (to - from);
	}

	public static bool IsEqualApprox(float a, float b)
	{
		// Check for exact equality first, required to handle "infinity" values.
		if (a == b)
		{
			return true;
		}
		// Then check for approximate equality.
		float tolerance = EpsilonF * Math.Abs(a);
		if (tolerance < EpsilonF)
		{
			tolerance = EpsilonF;
		}
		return Math.Abs(a - b) < tolerance;
	}

	public static bool IsEqualApprox(double a, double b)
	{
		// Check for exact equality first, required to handle "infinity" values.
		if (a == b)
		{
			return true;
		}
		// Then check for approximate equality.
		double tolerance = EpsilonD * Math.Abs(a);
		if (tolerance < EpsilonD)
		{
			tolerance = EpsilonD;
		}
		return Math.Abs(a - b) < tolerance;
	}

	public static bool IsFinite(float s)
	{
		return float.IsFinite(s);
	}

	public static bool IsFinite(double s)
	{
		return double.IsFinite(s);
	}

	public static bool IsInf(float s)
	{
		return float.IsInfinity(s);
	}

	public static bool IsInf(double s)
	{
		return double.IsInfinity(s);
	}

	public static bool IsNaN(float s)
	{
		return float.IsNaN(s);
	}

	public static bool IsNaN(double s)
	{
		return double.IsNaN(s);
	}

	public static bool IsZeroApprox(float s)
	{
		return Math.Abs(s) < EpsilonF;
	}

	public static bool IsZeroApprox(double s)
	{
		return Math.Abs(s) < EpsilonD;
	}

	public static float Lerp(float from, float to, float weight)
	{
		return from + ((to - from) * weight);
	}

	public static double Lerp(double from, double to, double weight)
	{
		return from + ((to - from) * weight);
	}

	public static float LerpAngle(float from, float to, float weight)
	{
		return from + AngleDifference(from, to) * weight;
	}

	public static double LerpAngle(double from, double to, double weight)
	{
		return from + AngleDifference(from, to) * weight;
	}

	public static float LinearToDb(float linear)
	{
		return MathF.Log(linear) * 8.6858896380650365530225783783321f;
	}

	public static double LinearToDb(double linear)
	{
		return Math.Log(linear) * 8.6858896380650365530225783783321;
	}

	public static float Log(float s)
	{
		return MathF.Log(s);
	}

	public static double Log(double s)
	{
		return Math.Log(s);
	}

	public static int Max(int a, int b)
	{
		return Math.Max(a, b);
	}

	public static float Max(float a, float b)
	{
		return Math.Max(a, b);
	}

	public static double Max(double a, double b)
	{
		return Math.Max(a, b);
	}

	public static int Min(int a, int b)
	{
		return Math.Min(a, b);
	}

	public static float Min(float a, float b)
	{
		return Math.Min(a, b);
	}

	public static double Min(double a, double b)
	{
		return Math.Min(a, b);
	}

	public static float MoveToward(float from, float to, float delta)
	{
		if (Math.Abs(to - from) <= delta)
			return to;

		return from + (Math.Sign(to - from) * delta);
	}

	public static double MoveToward(double from, double to, double delta)
	{
		if (Math.Abs(to - from) <= delta)
			return to;

		return from + (Math.Sign(to - from) * delta);
	}

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

	public static int PosMod(int a, int b)
	{
		int c = a % b;
		if ((c < 0 && b > 0) || (c > 0 && b < 0))
		{
			c += b;
		}
		return c;
	}

	public static float PosMod(float a, float b)
	{
		float c = a % b;
		if ((c < 0 && b > 0) || (c > 0 && b < 0))
		{
			c += b;
		}
		return c;
	}

	public static double PosMod(double a, double b)
	{
		double c = a % b;
		if ((c < 0 && b > 0) || (c > 0 && b < 0))
		{
			c += b;
		}
		return c;
	}

	public static float Pow(float x, float y)
	{
		return MathF.Pow(x, y);
	}

	public static double Pow(double x, double y)
	{
		return Math.Pow(x, y);
	}

	public static float RadToDeg(float rad)
	{
		return rad * RadToDegConstF;
	}

	public static double RadToDeg(double rad)
	{
		return rad * RadToDegConstD;
	}

	public static float Remap(float value, float inFrom, float inTo, float outFrom, float outTo)
	{
		return Lerp(outFrom, outTo, InverseLerp(inFrom, inTo, value));
	}

	public static double Remap(double value, double inFrom, double inTo, double outFrom, double outTo)
	{
		return Lerp(outFrom, outTo, InverseLerp(inFrom, inTo, value));
	}

	public static float RotateToward(float from, float to, float delta)
	{
		float difference = AngleDifference(from, to);
		float absDifference = Math.Abs(difference);
		return from + Math.Clamp(delta, absDifference - MathF.PI, absDifference) * (difference >= 0.0f ? 1.0f : -1.0f);
	}


	public static double RotateToward(double from, double to, double delta)
	{
		double difference = AngleDifference(from, to);
		double absDifference = Math.Abs(difference);
		return from + Math.Clamp(delta, absDifference - Math.PI, absDifference) * (difference >= 0.0 ? 1.0 : -1.0);
	}

	public static float Round(float s)
	{
		return MathF.Round(s);
	}

	public static double Round(double s)
	{
		return Math.Round(s);
	}

	public static int Sign(int s)
	{
		return Math.Sign(s);
	}

	public static int Sign(float s)
	{
		return Math.Sign(s);
	}

	public static int Sign(double s)
	{
		return Math.Sign(s);
	}

	public static float Sin(float s)
	{
		return MathF.Sin(s);
	}

	public static double Sin(double s)
	{
		return Math.Sin(s);
	}

	public static float Sinh(float s)
	{
		return MathF.Sinh(s);
	}

	public static double Sinh(double s)
	{
		return Math.Sinh(s);
	}

	public static float SmoothStep(float from, float to, float weight)
	{
		if (IsEqualApprox(from, to))
		{
			return from;
		}
		float x = Math.Clamp((weight - from) / (to - from), 0.0f, 1.0f);
		return x * x * (3 - (2 * x));
	}

	public static double SmoothStep(double from, double to, double weight)
	{
		if (IsEqualApprox(from, to))
		{
			return from;
		}
		double x = Math.Clamp((weight - from) / (to - from), 0.0, 1.0);
		return x * x * (3 - (2 * x));
	}

	public static float Sqrt(float s)
	{
		return MathF.Sqrt(s);
	}

	public static double Sqrt(double s)
	{
		return Math.Sqrt(s);
	}

	public static int StepDecimals(double step)
	{
		ReadOnlySpan<double> sd =
		[
			0.9999,
			0.09999,
			0.009999,
			0.0009999,
			0.00009999,
			0.000009999,
			0.0000009999,
			0.00000009999,
			0.000000009999,
		];
		double abs = Math.Abs(step);
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

	public static float Snapped(float s, float step)
	{
		if (step != 0f)
		{
			return MathF.Floor((s / step) + 0.5f) * step;
		}

		return s;
	}

	public static double Snapped(double s, double step)
	{
		if (step != 0f)
		{
			return Math.Floor((s / step) + 0.5f) * step;
		}

		return s;
	}

	public static float Tan(float s)
	{
		return MathF.Tan(s);
	}

	public static double Tan(double s)
	{
		return Math.Tan(s);
	}

	public static float Tanh(float s)
	{
		return MathF.Tanh(s);
	}

	public static double Tanh(double s)
	{
		return Math.Tanh(s);
	}

	public static int Wrap(int value, int min, int max)
	{
		int range = max - min;
		if (range == 0)
			return min;

		return min + ((((value - min) % range) + range) % range);
	}

	public static float Wrap(float value, float min, float max)
	{
		float range = max - min;
		if (IsZeroApprox(range))
		{
			return min;
		}
		return min + ((((value - min) % range) + range) % range);
	}

	public static double Wrap(double value, double min, double max)
	{
		double range = max - min;
		if (IsZeroApprox(range))
		{
			return min;
		}
		return min + ((((value - min) % range) + range) % range);
	}

	public static float PingPong(float value, float length)
	{
		return (length != 0.0f) ? Math.Abs(Fract((value - length) / (length * 2.0f)) * length * 2.0f - length) : 0.0f;

		static float Fract(float value)
		{
			return value - MathF.Floor(value);
		}
	}

	public static double PingPong(double value, double length)
	{
		return (length != 0.0) ? Math.Abs(Fract((value - length) / (length * 2.0)) * length * 2.0 - length) : 0.0;

		static double Fract(double value)
		{
			return value - Math.Floor(value);
		}
	}










}
