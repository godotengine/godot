using Godot;
using Godot.Collections;

public partial class ExportDiagnostics_GD0107_OK : Node
{
    [Export]
    public Node NodeField;

    [Export]
    public Node[] SystemArrayOfNodesField;

    [Export]
    public Array<Node> GodotArrayOfNodesField;

    [Export]
    public Dictionary<Node, string> GodotDictionaryWithNodeAsKeyField;

    [Export]
    public Dictionary<string, Node> GodotDictionaryWithNodeAsValueField;

    [Export]
    public Node NodeProperty { get; set; }

    [Export]
    public Node[] SystemArrayOfNodesProperty { get; set; }

    [Export]
    public Array<Node> GodotArrayOfNodesProperty { get; set; }

    [Export]
    public Dictionary<Node, string> GodotDictionaryWithNodeAsKeyProperty { get; set; }

    [Export]
    public Dictionary<string, Node> GodotDictionaryWithNodeAsValueProperty { get; set; }
}

public partial class ExportDiagnostics_GD0107_KO : Resource
{
    [Export]
    public Node {|GD0107:NodeField|};

    [Export]
    public Node[] {|GD0107:SystemArrayOfNodesField|};

    [Export]
    public Array<Node> {|GD0107:GodotArrayOfNodesField|};

    [Export]
    public Dictionary<Node, string> {|GD0107:GodotDictionaryWithNodeAsKeyField|};

    [Export]
    public Dictionary<string, Node> {|GD0107:GodotDictionaryWithNodeAsValueField|};

    [Export]
    public Node {|GD0107:NodeProperty|} { get; set; }

    [Export]
    public Node[] {|GD0107:SystemArrayOfNodesProperty|} { get; set; }

    [Export]
    public Array<Node> {|GD0107:GodotArrayOfNodesProperty|} { get; set; }

    [Export]
    public Dictionary<Node, string> {|GD0107:GodotDictionaryWithNodeAsKeyProperty|} { get; set; }

    [Export]
    public Dictionary<string, Node> {|GD0107:GodotDictionaryWithNodeAsValueProperty|} { get; set; }
}
