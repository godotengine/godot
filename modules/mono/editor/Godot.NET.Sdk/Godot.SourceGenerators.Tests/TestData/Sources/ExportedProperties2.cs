using Godot;
using System;

[GlobalClass]
public partial class ExportedProperties2(int health, Resource subResource, string[] strings) : Resource
{
    [Export]
    public int Health { get; set; } = health;
    [Export]
    public Resource SubResource { get; set; } = subResource;
    [Export]
    public string[] Strings { get; set; } = strings ?? System.Array.Empty<string>();
}
