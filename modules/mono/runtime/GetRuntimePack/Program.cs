using System.Diagnostics;

internal class Program
{
    internal static void Main()
    {
        throw new UnreachableException();
    }
}
