namespace Godot
{
    public static partial class ResourceLoader
    {
        public static T Load<T>(string path) where T : class
        {
            return (T)(object)Load(path);
        }
    }
}
