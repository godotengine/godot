using System;
using JetBrains.Annotations;

#nullable enable

namespace Godot;

/// <summary>
/// Indicates that the class should not be associated with a script file in Godot.
/// </summary>
/// <remarks>
/// <para>This attribute is the opposite of <see cref="Godot.ScriptPathAttribute"/>.
/// It is used to prevent the association of a class with a script file in Godot.</para>
/// <para>The attribute by itself does nothing, but it can be used by source generators
/// to determine whether to associate a class with a script file in Godot.</para>
/// </remarks>
[AttributeUsage(AttributeTargets.Class, Inherited = false)]
[PublicAPI]
public class NoScriptFileAssociationAttribute : Attribute;
