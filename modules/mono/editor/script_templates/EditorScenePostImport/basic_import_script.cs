// meta-description: Basic import script template

#if TOOLS
using _BINDINGS_NAMESPACE_;
using System;

[Tool]
public partial class _CLASS_ : _BASE_
{
    public override GodotObject _PostImport(Node scene)
    {
        // Modify the contents of the scene upon import.
        return scene; // Return the modified root node when you're done.
    }
}
#endif
