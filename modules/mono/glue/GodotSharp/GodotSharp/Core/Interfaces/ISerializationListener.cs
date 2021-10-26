namespace Godot
{
    public interface ISerializationListener
    {
        void OnBeforeSerialize();
        void OnAfterDeserialize();
    }
}
