using System;

public static partial class Mathf
{

    public const real_t E = (real_t)2.7182818284590452353602874714M;

    public const real_t Sqrt2 = (real_t)1.4142135623730950488016887242M; // 1.4142136f and 1.414213562373095

    private const float EpsilonF = 1e-06f;
    private const double EpsilonD = 1e-14;

#if REAL_T_IS_DOUBLE
    public const real_t Epsilon = EpsilonD;
#else
    public const real_t Epsilon = EpsilonF;
#endif

    public static int DecimalCount(double s)
    {
        return DecimalCount((decimal)s);
    }
    
    public static int DecimalCount(decimal s)
    {
        return BitConverter.GetBytes(decimal.GetBits(s)[3])[2];
    }
    
    public static int CeilToInt(float s)
    {
        return (int)MathF.Ceiling(s);
    }
    
    public static int CeilToInt(double s)
    {
        return (int)Math.Ceiling(s);
    }
    
    public static int FloorToInt(float s)
    {
        return (int)MathF.Floor(s);
    }
    
    public static int FloorToInt(double s)
    {
        return (int)Math.Floor(s);
    }
    
    public static int RoundToInt(float s)
    {
        return (int)MathF.Round(s);
    }
    
    public static int RoundToInt(double s)
    {
        return (int)Math.Round(s);
    }
    
    public static (float Sin, float Cos) SinCos(float s)
    {
        return MathF.SinCos(s);
    }
    
    public static (double Sin, double Cos) SinCos(double s)
    {
        return Math.SinCos(s);
    }
    
    
    public static bool IsEqualApprox(float a, float b, float tolerance)
    {
        // Check for exact equality first, required to handle "infinity" values.
        if (a == b)
        {
            return true;
        }
        // Then check for approximate equality.
        return Math.Abs(a - b) < tolerance;
    }
    
    public static bool IsEqualApprox(double a, double b, double tolerance)
    {
        // Check for exact equality first, required to handle "infinity" values.
        if (a == b)
        {
            return true;
        }
        // Then check for approximate equality.
        return Math.Abs(a - b) < tolerance;
    }
    
}