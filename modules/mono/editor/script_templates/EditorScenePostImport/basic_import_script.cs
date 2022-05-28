// meta-description: Basic import script template

#if TOOLS
using _BINDINGS_NAMESPACE_;
using System;

[Tool]
public partial class _CLASS_ : _BASE_
{
    public override Object _PostImport(Node scene)
    {
        // Modify the contents of the scene upon import. For example, setting up LODs:
//      scene.GetNode<MeshInstance3D>("HighPolyMesh").DrawDistanceEnd = 5.0
//      scene.GetNode<MeshInstance3D>("LowPolyMesh").DrawDistanceBegin = 5.0
        return scene // Return the modified root node when you're done.
    }
}
#endif
