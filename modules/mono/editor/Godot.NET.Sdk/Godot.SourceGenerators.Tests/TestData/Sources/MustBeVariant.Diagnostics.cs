using System;
using Godot;

[AttributeUsage(AttributeTargets.Class | AttributeTargets.Method, AllowMultiple = true)]
public class GenericTypeAttribute<[MustBeVariant] T> : Attribute
{
}

public class NotVariantClass {

}

public class MustBeVariantMethodsWithDiagnosticClass
{
    public MustBeVariantMethodsWithDiagnosticClass()
    {
        Method<NotVariantClass>();
    }

    public void Method<[MustBeVariant] T>()
    {
    }
}

public class MustBeVariantAnnotatedMethodsWithDiagnosticClass
{
    [GenericType<NotVariantClass>()]
    public void MethodWithNotVariantAttribute()
    {
    }
}

[GenericType<NotVariantClass>()]
public class ClassNotVariantAnnotated
{
}
