using Godot;

public class MustBeVariantGD0302
{
    public void MethodOk<[MustBeVariant] T>()
    {
        // T is guaranteed to be a Variant-compatible type because it's annotated with the [MustBeVariant] attribute, so it can be used here.
        new ExampleClass<T>();
        Method<T>();
    }

    public void MethodFail<T>()
    {
        // T is not valid here because it may not a Variant-compatible type and method call and class require it 
        new ExampleClass<T>();
        Method<T>();
    }

    public void Method<[MustBeVariant] T>()
    {
    }
}

public class ExampleClass<[MustBeVariant] T>
{

}
