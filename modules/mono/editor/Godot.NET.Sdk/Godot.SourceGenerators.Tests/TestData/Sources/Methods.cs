using Godot;

public partial class Methods : GodotObject
{
    private void MethodWithOverload()
    {
    }

    private void MethodWithOverload(int a)
    {
    }

    private void MethodWithOverload(int a, int b)
    {
    }

    // Should be ignored. The previous one is picked.
    private void MethodWithOverload(float a, float b)
    {
    }

    // Generic methods should be ignored.
    private void GenericMethod<T>(T t)
    {
    }
}
