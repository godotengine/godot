# MaterialX Graph Editor 

The MaterialX Graph Editor is an example application for visualizing, creating, and editing MaterialX graphs.  It utilizes the ImGui framework as well as additional ImGui extensions such as the Node Editor.

### Example Images

**Figure 1:** MaterialX Graph Editor with procedural marble example
<img src="/documents/Images/MaterialXGraphEditor_Marble.png" />

## Building The MaterialX Graph Editor
Select the `MATERIALX_BUILD_GRAPH_EDITOR` option in CMake to build the MaterialX Graph Editor.  Installation will copy the **MaterialXGraphEditor** executable to a `/bin` directory within the selected install folder.

### Summary of Graph Editor Features

1.  **Load Material**: Load a material document in the MTLX format.
2.  **Save Material**: Save out a graph as a mterial document in MTLX format.
3.  **New Material**: Clear all information to set up for the creation of a new material
4.  **Node Property Editor**: View or edit properties of the selected node.
5.  **Render View**: View the rendered material.

### Buttons

To display a new material and graph, click the `Load Material` button and and navigate to the [Example Materials](../../resources/Materials/Examples) folder, which contains a selection of materials in the MTLX format, and select a document to load.  The Graph Editor will display the graph hierarchy of the selected document for visualization and editing.

To save out changes to the graphs as MTLX files click the `Save Material` button.  This will save the position of the nodes in the graph for future use as well. 

### Editor Window

The MaterialX document is displayed as nodes in the Editor window.  When a file is intially loaded the material node, surface shader node, and any enclosing nodegraphs will be displayed.  Double-clicking on a nodegraph, or any node defined as a subgraph, will display the contents of that graph.

The current graph hierarchy is displayed above the editor.  To back up and out of a subgraph click the `<` button to the left of the graph names.

Each node and nodegraph displays its name and pins for all of its inputs and outputs.  Nodes can be connected by clicking on the output pin of one node and connecting it to the input pin of another node, thus creating a link.  Links can only be created if the input and output pins have the same type, as designated by the color of the pin.  When a new link is created the material in the render view will automatically be updated. 

Using the tab key on the editor allows the user to add a certain node by bringing up the `Add Node` pop-up.  The nodes are organized in the pop-up window based on their node group but a specfic node can also be searched for by name.  To create a nodegraph select `Node Graph` in the `Add Node` popup and dive into the node in order to add nodes inside of it and populate it. 

In order to connect to the nodegraph to a shader add output nodes inside the nodegraph then travel back up outside the nodegraph and connect the corresponding output pin to the surface shader.  By default, the nodegraph does not contain any output nodes or pins. 

Another type of node present in the `Add Node` pop-up is the group, or background node.  This background node can be used to group specific nodes and label by them by dragging them on to the background node.

To search the editor window for a specific node use `CTRL` + `F` to bring up the search bar. 

### Node Property Editor
When a node is selected in the graph, its information is displayed on the left-hand column in the `Node Property Editor`.  This editor displays the name of the node, its category, its inputs, the input name, types and values.  Inputs that are connected to other nodes will not display a value.

This is where a node's properties such as its name and input values can be adjusted.  When an input value is changed the material is automatically updated to reflect that change.  The node info button displays the `doc` string for the selected node and its inputs if they exist. This `doc` string is currently read only.

The show All Inputs checkbox displays all possible inputs for a node. With the box unchecked only inputs that have a connection or have had a value set will be shown. Only these inputs will be saved out when the graph is saved. 

### Render View
Above the `Node Property Editor`, the `Render View` displays the current material on the Arnold Shader Ball.  If inside a subgraph it will display the material associated with that subgraph; otherwise it will display the output of the selected node.  It automatically updates when any changes are made to the graph.

To adjust the relative sizes of the Node Property Editor and Render View windows, drag the separator between these windows in the application. The render view window camera can be changed using the left or right mouse buttons to manipulate the shader ball. 

### Keyboard Shortcuts

- `TAB`: Add Node Popup
- `Right Click`: pan along the editor
- `Double Click on Node`: Dive into node's subgraph if it has one
- `U`: Go up and out of a subgraph
- `F`: Frame selected node(s)
- `Ctrl + F` to search for a node in the editor by name
- `Ctrl/Cmd + C` for Copying Nodes
- `Ctrl/Cmd+X` for Cutting Nodes
- `Ctrl/Cmd+V` for Pasting Nodes
- `+` : Zoom in with the camera when mouse is over the Render View Window.
- `-` : Zoom out with the camera when mouse is over the Render View Window.

### Command-Line Options

The following are common command-line options for MaterialXGraphEditor, and a complete list can be displayed with the `--help` option.
- `--material [FILENAME]` : Specify the filename of the MTLX document to be displayed in the graph editor
- `--mesh [FILENAME]` : Specify the filename of the OBJ or glTF mesh to be displayed in the graph editor
- `--path [FILEPATH]` : Specify an additional data search path location (e.g. '/projects/MaterialX').  This absolute path will be queried when locating data libraries, XInclude references, and referenced images.
- `--library [FILEPATH]` : Specify an additional data library folder (e.g. 'vendorlib', 'studiolib').  This relative path will be appended to each location in the data search path when loading data libraries.
- `--captureFilename [FILENAME]` : Specify the filename to which the first rendered frame should be written

### Known Limitations

- Creating new connections using the `channels` attribute of an input is not yet supported, though existing `channels` connections will be displayed in graphs.
- Assigning a new `colorspace` attribute to an input is not yet supported, though existing `colorspace` attributes on inputs will be respected by the render view.
