using System;

namespace Godot
{
    public static partial class Random
    {
        public static System.Random random = new System.Random();
        public static int RandInt(int minValue, int maxValue)
        {
            return random.Next(minValue, maxValue);
        }
        public static long RandLong(long minValue, long maxValue)
        {
            return random.NextInt64(maxValue) - minValue;
        }
        public static float RandFloat(float minValue, float maxValue)
        {
            return random.NextSingle() * maxValue - minValue;
        }
        public static double RandDouble(double minValue, double maxValue)
        {
            return random.NextDouble() * maxValue - minValue;
        }
        public static short RandByte(short minValue, short maxValue)
        {
            return (short)RandInt(minValue, maxValue);
        }
        public static byte RandByte(byte minValue, byte maxValue)
        {
            return (byte)RandInt(minValue, maxValue);
        }
        public static bool RandBool()
        {
            return RandInt(0, 2) == 0;
        }
        public static char RandCharFromString(string str)
        {
            return str[random.Next(str.Length)];
        }
    }
}
