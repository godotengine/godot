using System;

namespace Godot;

/// <summary>
/// Attribute that automatically defers initialization of a member until _Ready() is called.
/// </summary>
[AttributeUsage(AttributeTargets.Method | AttributeTargets.Property)]
public class OnReadyAttribute : Attribute
{
    /// <summary>
    /// Relative or absolute path in a scene tree like <see cref="Godot.NodePath"/>
    /// </summary>
    public string NodePath { get; }

    /// <summary>
    /// Constructs a new OnReadyAttribute instance.
    /// </summary>
    /// <param name="nodePath">Relative or absolute path in a scene tree like <see cref="Godot.NodePath"/></param>
    public OnReadyAttribute(string nodePath)
    {
        NodePath = nodePath;
    }
}
