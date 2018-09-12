namespace Godot
{
    public static partial class ResourceLoader
    {
        public static T Load<T>(string path) where T : Godot.Resource
        {
            return (T) Load(path);
        }
    }
}
