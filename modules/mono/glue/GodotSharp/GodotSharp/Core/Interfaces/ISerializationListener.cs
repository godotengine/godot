namespace Godot
{
    /// <summary>
    /// An interface that requires methods for before and after serialization.
    /// </summary>
    public interface ISerializationListener
    {
        /// <summary>
        /// Receives a callback before Godot serializes the object.
        /// </summary>
        void OnBeforeSerialize();
        /// <summary>
        /// Receives a callback after Godot deserialized the object.
        /// </summary>
        void OnAfterDeserialize();
    }
}
