using System;

namespace Godot
{
    /// <summary>
    /// Attribute that restricts generic type parameters to be only types
    /// that can be marshaled from/to a <see cref="Variant"/>.
    /// </summary>
    [AttributeUsage(AttributeTargets.GenericParameter)]
    public sealed class MustBeVariantAttribute : Attribute { }
}
