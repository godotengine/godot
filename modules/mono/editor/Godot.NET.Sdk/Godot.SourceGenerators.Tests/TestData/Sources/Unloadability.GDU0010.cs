using System.Threading.Tasks;
using Godot;

// Positive: [Tool] class calling Task.Run triggers GDU0010
[Tool]
public class ToolClassWithTaskRun
{
    public void RunTask()
    {
        {|GDU0010:Task.Run(() => { })|};
    }
}

// Negative: non-Tool class should NOT trigger
public class NonToolClassWithTaskRun
{
    public void RunTask()
    {
        Task.Run(() => { });
    }
}
