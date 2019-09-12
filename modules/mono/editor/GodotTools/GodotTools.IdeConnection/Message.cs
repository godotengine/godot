using System.Linq;

namespace GodotTools.IdeConnection
{
    public struct Message
    {
        public string Id { get; set; }
        public string[] Arguments { get; set; }

        public Message(string id, params string[] arguments)
        {
            Id = id;
            Arguments = arguments;
        }

        public override string ToString()
        {
            return $"(Id: '{Id}', Arguments: '{string.Join(",", Arguments)}')";
        }
    }
}
