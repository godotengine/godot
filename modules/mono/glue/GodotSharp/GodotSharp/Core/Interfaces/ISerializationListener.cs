namespace Godot
{
    /// <summary>
    /// An interface that requires methods for before and after serialization.
    /// </summary>
    public interface ISerializationListener
    {
        void OnBeforeSerialize();
        void OnAfterDeserialize();
    }
}
