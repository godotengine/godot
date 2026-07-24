partial class ClassAllDoc
{
#pragma warning disable CS0109 // Disable warning about redundant 'new' keyword
#if TOOLS
    [global::System.ComponentModel.EditorBrowsable(global::System.ComponentModel.EditorBrowsableState.Never)]
    internal new static global::Godot.Collections.Dictionary GetGodotClassDocs()
    {
        var docs = new global::Godot.Collections.Dictionary();
        docs.Add("name","\"ClassAllDoc.cs\"");
        docs.Add("brief_description",@"class description test");
        docs.Add("description",@"class description test");

        var propertyDocs = new global::Godot.Collections.Array();
        propertyDocs.Add(new global::Godot.Collections.Dictionary { { "name", PropertyName.PropertyDocTest }, { @"type", @"int" }, { "description", @"property description test [code]ClassAllDoc[/code]" }});
        propertyDocs.Add(new global::Godot.Collections.Dictionary { { "name", PropertyName._fieldDocTest }, { @"type", @"int" }, { "description", @"field description [code]true[/code] test [code]ClassAllDoc[/code]" }});
        docs.Add("properties", propertyDocs);

        var signalDocs  = new global::Godot.Collections.Array();
        signalDocs.Add(new global::Godot.Collections.Dictionary { { "name", SignalName.SignalDocTest }, { "description", @"signal description ~!@#$%^*()_+{}| test [code]ClassAllDoc[/code][br][br][b]Parameters:[/b][br] • [b]num[/b]:" }});
        docs.Add("signals", signalDocs);

        docs.Add("is_script_doc", true);

        docs.Add("script_path", "ClassAllDoc.cs");

        return docs;
    }

#endif // TOOLS
#pragma warning restore CS0109
}
