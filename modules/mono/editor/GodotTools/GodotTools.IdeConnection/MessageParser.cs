using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace GodotTools.IdeConnection
{
    public static class MessageParser
    {
        public static bool TryParse(string messageLine, out Message message)
        {
            var arguments = new List<string>();
            var stringBuilder = new StringBuilder();

            bool expectingArgument = true;

            for (int i = 0; i < messageLine.Length; i++)
            {
                char @char = messageLine[i];

                if (@char == ',')
                {
                    if (expectingArgument)
                        arguments.Add(string.Empty);

                    expectingArgument = true;
                    continue;
                }

                bool quoted = false;

                if (messageLine[i] == '"')
                {
                    quoted = true;
                    i++;
                }

                while (i < messageLine.Length)
                {
                    @char = messageLine[i];
                    
                    if (quoted && @char == '"')
                    {
                        i++;
                        break;
                    }

                    if (@char == '\\')
                    {
                        i++;
                        if (i < messageLine.Length)
                            break;

                        stringBuilder.Append(messageLine[i]);
                    }
                    else if (!quoted && @char == ',')
                    {
                        break; // We don't increment the counter to allow the colon to be parsed after this
                    }
                    else
                    {
                        stringBuilder.Append(@char);
                    }
                    
                    i++;
                }
                
                arguments.Add(stringBuilder.ToString());
                stringBuilder.Clear();

                expectingArgument = false;
            }

            if (arguments.Count == 0)
            {
                message = new Message();
                return false;
            }

            message = new Message
            {
                Id = arguments[0],
                Arguments = arguments.Skip(1).ToArray()
            };

            return true;
        }
    }
}
