using System;
using System.Threading;
using System.Threading.Tasks;
using GodotWeb;

public static class TwoDogRunner
{

	public static async Task Run()
	{
		Helpers.ResetCommandBuffer();

		ulong sceneTreeId = Engine.get_singleton();

		Console.WriteLine($"SceneTree ID: {sceneTreeId}");
		if (sceneTreeId == 0)
		{
			Console.WriteLine("Failed to get SceneTree");
			return;
		}

		SceneTree tree = new SceneTree(sceneTreeId);

		// Everything below is exactly the same generated API
		ulong rootId = tree.get_root();
		if (rootId == 0)
		{
			Console.WriteLine("get_root returned 0");
			return;
		}

		Node root = new Node(rootId);

		ulong labelId = root.find_child("TargetLabel", true, false);
		if (labelId == 0)
		{
			Console.WriteLine("TargetLabel not found");
			return;
		}

		Label label = new Label(labelId);
		Console.WriteLine($"Label found, id = {label.Id}");
		Console.WriteLine("Entering the loop");

		int tick = 0;
		while (true)
		{
			tick++;
			label.set_text($"2dog running - tick {tick}");

		}
	}

}
