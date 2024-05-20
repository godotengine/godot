namespace Godot.SourceGenerators.Sample
{
    public interface IHealth
    {
        [Signal]
        public delegate void OnHealthChangedEventHandler(int health);
    }

    public interface IHealth2
    {
        [Signal]
        public delegate void OnHealthChangedEventHandler(int health);
    }

    public interface INested : IHealth
    {
    }

    public interface ISpell
    {
        [Signal]
        public delegate void OnSpellCastEventHandler(int spellId);
    }

    public partial class EventSignalsFromInterfaces : Node, IHealth, ISpell
    {
        public override void _Ready()
        {
            OnHealthChanged += (int health) =>
                GD.Print($"{nameof(OnHealthChanged)} {health}");

            OnSpellCast += (int spellId) => { GD.Print($"{nameof(OnSpellCast)} {spellId}"); };


            var hasSignalName = !string.IsNullOrWhiteSpace(SignalName.OnHealthChanged);
        }
    }

    public partial class EventSignalsFromInterfacesSecond : Node, IHealth
    {
        public override void _Ready()
        {
            OnHealthChanged += (int health) => { GD.Print($"{nameof(OnHealthChanged)} {health}"); };
        }
    }

    public partial class EventSignalsFromInterfacesThird : Node, INested
    {
        public override void _Ready()
        {
            OnHealthChanged += (int health) => { GD.Print($"{nameof(OnHealthChanged)} {health}"); };
        }
    }
}
