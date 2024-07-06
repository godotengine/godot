using Godot;

public partial class ExportedComplexStrings : Node
{
    [Export]
    private string _fieldInterpolated1 = $"The quick brown fox jumps over ({Engine.GetVersionInfo()})";

    [Export]
    private string _fieldInterpolated2 = $"The quick brown fox jumps over ({Engine.GetVersionInfo()["major"],0:G}) the lazy dog.";

    [Export]
    private string _fieldInterpolated3 = $"{((int)Engine.GetVersionInfo()["major"]) * -1 * -1:G} the lazy dog.";

    [Export]
    private string _fieldInterpolated4 = $"{":::fff,,}<,<}},,}]"}";

    [Export]
    public string PropertyInterpolated1
    {
        get;
        private set;
    } = $"The quick brown fox jumps over {GD.VarToStr($"the lazy {Engine.GetVersionInfo()} do")}g.";
}
