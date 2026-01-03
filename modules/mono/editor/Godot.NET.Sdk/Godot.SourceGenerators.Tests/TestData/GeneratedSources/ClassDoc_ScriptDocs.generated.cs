partial class ClassDoc
{
#pragma warning disable CS0109 // Disable warning about redundant 'new' keyword
#if TOOLS
    [global::System.ComponentModel.EditorBrowsable(global::System.ComponentModel.EditorBrowsableState.Never)]
    internal new static global::Godot.Collections.Dictionary GetGodotClassDocs()
    {
        var docs = new global::Godot.Collections.Dictionary();
        docs.Add("name","\"ClassDoc.cs\"");
        docs.Add("brief_description",@"This is the class documentation.");
        docs.Add("description",@"This is the class documentation.");

        var propertyDocs = new global::Godot.Collections.Array();
        propertyDocs.Add(new global::Godot.Collections.Dictionary { { "name", PropertyName.MyProperty }, { @"type", @"int" }, { "description", @"There is currently no description for this property." }});
        docs.Add("properties", propertyDocs);

        docs.Add("is_script_doc", true);

        docs.Add("script_path", "ClassDoc.cs");

        return docs;
    }

#endif // TOOLS
#pragma warning restore CS0109
}
