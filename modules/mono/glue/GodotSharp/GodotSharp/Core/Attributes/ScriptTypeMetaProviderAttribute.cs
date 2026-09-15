using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Reflection;
using Godot.Bridge;
using JetBrains.Annotations;

namespace Godot;

#nullable enable

/// <summary>
/// Interface for types that provide metadata for Godot script types.
/// </summary>
public interface IScriptTypeMetaProvider
{
    /// <summary>
    /// Gets the metadata for the Godot script type.
    /// </summary>
    /// <returns>The metadata for the Godot script type.</returns>
    public static abstract ScriptTypeMeta GetGodotClassScriptMeta();
}

/// <summary>
/// Base attribute for attributes that specify a type that provides metadata for a Godot script type.
/// </summary>
public abstract class ScriptTypeMetaProviderBaseAttribute : Attribute
{
    /// <summary>
    /// Gets the metadata for the Godot script type provided by the type specified in the attribute type parameter,
    /// using the provided script type if necessary (e.g. for generic script types with a nested provider generic type definition).
    /// </summary>
    /// <returns>
    /// The metadata for the Godot script type.
    /// </returns>
    /// <param name="scriptType">The script type for which to get the metadata. This parameter is
    /// provided for potential use in derived classes that need to use the script type to get the
    /// metadata, e.g. for generic script types with a nested provider generic type definition.</param>
    public abstract ScriptTypeMeta GetGodotClassScriptMeta(Type scriptType);
}

/// <summary>
/// Attribute used to specify a type that provides metadata for a Godot script type.
/// </summary>
/// <typeparam name="T">
/// The type that provides metadata for a Godot script type.
/// Must implement <see cref="IScriptTypeMetaProvider"/>.
/// </typeparam>
[AttributeUsage(AttributeTargets.Class, Inherited = false)]
[PublicAPI]
public sealed class ScriptTypeMetaProviderAttribute
    <[DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.PublicMethods)] T>
    : ScriptTypeMetaProviderBaseAttribute
    where T : IScriptTypeMetaProvider
{
    /// <summary>
    /// Gets the metadata for the Godot script type provided by the type specified in the attribute type parameter,
    /// using the provided script type if necessary (e.g. for generic script types with a nested provider generic type definition).
    /// </summary>
    /// <returns>
    /// The metadata for the Godot script type.
    /// </returns>
    /// <param name="scriptType">The script type for which to get the metadata. This parameter is not used in this implementation,
    /// but it is provided for consistency with the base method and for potential use in derived classes.</param>
    public override ScriptTypeMeta GetGodotClassScriptMeta(Type scriptType)
    {
        return T.GetGodotClassScriptMeta();
    }
}

/// <summary>
/// Attribute used to specify a type that provides metadata for a generic Godot script type.
/// </summary>
[AttributeUsage(AttributeTargets.Class, Inherited = false)]
[PublicAPI]
public sealed class GenericScriptTypeMetaProviderAttribute : ScriptTypeMetaProviderBaseAttribute
{
    [DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.PublicMethods)]
    private readonly string _nestedProviderGenericTypeDefinitionName;

    /// <summary>
    /// Initializes a new instance of the <see cref="GenericScriptTypeMetaProviderAttribute"/> class with
    /// the specified generic type definition name of the nested provider type.
    /// </summary>
    /// <param name="nestedProviderGenericTypeDefinitionName">
    /// The assembly-qualified name (as expected by <see cref="System.Type.GetType(string)"/>) of the
    /// generic type definition of the provider type. This type must implement <see cref="IScriptTypeMetaProvider"/>
    /// to provides metadata for the Godot script type. The type must be nested within the script type
    /// that this attribute is applied to. Neither this nested provider type nor any possible containing
    /// types between it and the script type generic type definition should have any generic parameters.
    /// </param>
    /// <remarks>
    /// <para><paramref name="nestedProviderGenericTypeDefinitionName"/> must the assembly-qualified name
    /// (as expected by <see cref="System.Type.GetType(string)"/>) of the generic type definition of the
    /// provider type. This type must be nested within the script type that this attribute is applied to.
    /// Neither this nested provider type nor any possible containing types between it and the
    /// script type generic type definition should have any additional generic parameters.</para>
    /// <para>The type specified by <paramref name="nestedProviderGenericTypeDefinitionName"/>
    /// must implement the <see cref="IScriptTypeMetaProvider"/> interface.</para>
    /// </remarks>
    public GenericScriptTypeMetaProviderAttribute(
        [DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.PublicMethods)]
        string nestedProviderGenericTypeDefinitionName)
    {
        _nestedProviderGenericTypeDefinitionName = nestedProviderGenericTypeDefinitionName;
    }

    private const string TrimWarnSuppressJustification =
        "The provided generic type definition name is expected to be a nested type within the " +
        "script type generic type definition, and the generic type definition of the script type " +
        "is verified by this method to be the same as one of the declaring types of the provided " +
        "nested generic type definition. The provided nested generic type definition is annotated " +
        "with DynamicallyAccessedMembers to preserve it and its methods (this applies to all its " +
        "constructed types), so it is safe to use reflection to get the same nested type within " +
        "scriptType and invoke its GetGodotClassScriptMeta method.";

    /// <summary>
    /// Gets the metadata for the Godot script type provided by the type specified in the attribute type parameter,
    /// using the provided script type if necessary (e.g. for generic script types with a nested provider generic type definition).
    /// </summary>
    /// <returns>
    /// The metadata for the Godot script type.
    /// </returns>
    /// <param name="scriptType">The script type for which to get the metadata. This parameter
    /// is used in this implementation to get the nested provider type from the script type,
    /// as the provided generic type definition is expected to be a nested type within the script
    /// type generic type definition, and to invoke the GetGodotClassScriptMeta method on it.</param>
    [UnconditionalSuppressMessage("ReflectionAnalysis", "IL2070:UnrecognizedReflectionPattern",
        Justification = TrimWarnSuppressJustification)]
    [UnconditionalSuppressMessage("ReflectionAnalysis", "IL2075:UnrecognizedReflectionPattern",
        Justification = TrimWarnSuppressJustification)]
    public override ScriptTypeMeta GetGodotClassScriptMeta(Type scriptType)
    {
        if (!scriptType.IsGenericType)
            throw new InvalidOperationException(
                $"The script type '{scriptType.FullName}' is expected to be a generic type in this context.");

        if (!scriptType.IsConstructedGenericType)
            throw new InvalidOperationException(
                $"The script type '{scriptType.FullName}' is expected to be a constructed generic type in this context.");

        var scriptTypeGenericTypeDef = scriptType.GetGenericTypeDefinition();

        var nestedProviderGenericTypeDefinition = Type.GetType(_nestedProviderGenericTypeDefinitionName);

        if (nestedProviderGenericTypeDefinition is null)
            throw new InvalidOperationException(
                $"The provided generic type definition name '{_nestedProviderGenericTypeDefinitionName}' provided to the attribute could not be resolved to a type.");

        if (nestedProviderGenericTypeDefinition == scriptType)
            throw new InvalidOperationException(
                $"The provided generic type definition '{nestedProviderGenericTypeDefinition.FullName}' provided to the attribute is expected to be a nested type of the script type '{scriptType.FullName}', but it is the same type.");

        if (nestedProviderGenericTypeDefinition.DeclaringType is not { } declaringType)
            throw new InvalidOperationException(
                $"The provided generic type definition '{nestedProviderGenericTypeDefinition.FullName}' provided to the attribute is expected to be a nested type within the script type '{scriptType.FullName}', but it is not.");

        if (declaringType != scriptTypeGenericTypeDef)
        {
            var currentDeclaringType = declaringType;
            do
            {
                currentDeclaringType = currentDeclaringType.DeclaringType;
            } while (currentDeclaringType != null && currentDeclaringType != scriptTypeGenericTypeDef);

            if (currentDeclaringType == null)
                throw new InvalidOperationException(
                    $"The provided generic type definition '{nestedProviderGenericTypeDefinition.FullName}' provided to the attribute is nested within the type '{declaringType.FullName}', which is not the expected generic type definition for the script type '{scriptType.FullName}'.");
        }

        // All checks passed, we can now safely use reflection to get the nested type within scriptType,
        // as its generic type definition is the same as one of the declaring types of the provided nested
        // generic type definition, which is annotated with DynamicallyAccessedMembers to preserve the nested
        // type and its methods (this applies to all its constructed types).

        Type nestedType;

        if (declaringType == scriptTypeGenericTypeDef)
        {
            // Fast path for the common case where the provided nested generic type definition
            // is directly nested within the script type generic type definition.

            nestedType = scriptType.GetNestedType(nestedProviderGenericTypeDefinition.Name,
                BindingFlags.Public | BindingFlags.NonPublic)!;
        }
        else
        {
            // Slow path for the less common case where the provided nested generic type definition is nested
            // within the script type generic type definition, but not directly (i.e. there are additional
            // containing types between it and the script type generic type definition).
            // Our source generators don't use this pattern.

            List<string> path = new();
            Type current = nestedProviderGenericTypeDefinition;
            while (current != scriptTypeGenericTypeDef)
            {
                path.Insert(0, current.Name);
                current = current.DeclaringType!;
            }

            nestedType = scriptType;
            foreach (string name in path)
                nestedType = nestedType.GetNestedType(name, BindingFlags.Public | BindingFlags.NonPublic)!;
        }

        if (!nestedType.IsConstructedGenericType)
            throw new InvalidOperationException(
                $"The nested provider type '{nestedType.FullName}' is not a constructed generic type. " +
                $"This is likely because the nested provider type has additional generic parameters " +
                $"other than those of its containing script type '{scriptType.FullName}'.");

        if (!nestedType.IsAssignableTo(typeof(IScriptTypeMetaProvider)))
            throw new InvalidOperationException(
                $"The nested provider type '{nestedType.FullName}' does not implement the expected interface '{typeof(IScriptTypeMetaProvider).FullName}'.");

        var method = nestedType.GetMethod("GetGodotClassScriptMeta",
            BindingFlags.Public | BindingFlags.Static,
            null, Type.EmptyTypes, null)!;
        return (ScriptTypeMeta)method.Invoke(null, null)!;
    }
}
