# GUT Documentation
Documentation is hosted at https://gut.readthedocs.io.

The wiki is generated from three types of files:
* RST:  This wiki site index (`index.rst`) is the only RST file that manually created.  All other RST files are generated from code comments.
* Markdown:  The standalone pages of the wiki are all made in Markdown.
* Documentation Comments:  A modified version of Godot's code is used to generate RST files for GUT scripts and `class_names`.


## Code Comment Features
All features listed in [Godot's Documentation Comments](https://docs.godotengine.org/en/stable/tutorials/scripting/gdscript/gdscript_documentation_comments.html) are supported as well as some additional bbcode tags and annotations.  These additional tags will only work on the site and will appear unaltered when viewing documention through the Editor.

* `[wiki][/wiki]` Creates a link to a wiki page.  Use page title between tags.  For example `[wiki]Creating-Tests[/wiki]`.  To link to a GUT class, prefix the `class_name` with `class_`:  `[wiki]class_GutTest[/wiki]`
* `@ignore-uncommented`:  Place in the class description (not short description) to exclude all of the following that do not have document comments above them:
    * Methods
    * Properties
    * Constants
    * Signals
* `@ignore` Can be used in doc comments for Methods, Properties, and Signals to exclude them from the documentation.
* `@internal` Can be used in doc comments for Methods to mark them for "internal use only".  They will be included in the generated documentation but will be marked.

Any links to non GUT classes created in doc comments (like `[Node2D]`) will link to Godot's documentation.  Due to the way the documentation is generated, all non-GUT links are assumed to exist on Godot's site.


## Markdown
Linking to wiki pages in Markdown requires the title be used.  As with doc comments, you must prefix GUT class links with `class_`.
* `[Creating Tests](Creating-Tests)` links to the Creating Tests wiki page.
* `[GutTest][class_GutTest]` links to the class reference page for `GutTest`.
* You must use `<a>` tags to link to anchors in class ref pages:  `<a href="class_ref/class_guttest.html#class-guttest-method-assert-called">assert_called</a>`




# Documentation Files Structure
`documentation/docs/conf.py`<br>
This is the Sphinx configuration file.

`documentation/docs/index.rst`<br>
The Home page and also responsible for generating the Table of Contents for the site.  If you add a new page, it must be added to one of the `.. toctree::` entries.

`documentation/docs`  <br>
The directory for all the wiki pages.  All wiki pages are markdown.

`documentation/_static/css`<br>
The CSS goes in here.

`documentation/_static/images` <br>
Put any wiki related images in here.




# Local Documentation Generation
You can generate the documentation locally to see what it will look like on readthedocs.

### Create the docker image
To create the docker container, run the following from the `documentation` directory.  You only have to do this if the container does not exist.
```
docker-compose -f docker/compose.yml build
```

### Generate the documentation
Each time you want to regenerate the documentation run the following from the `documentation` directory.
```
docker-compose -f docker/compose.yml up
```

### View Generated Documentation
You can view the generated documentation here:
```
documentation/docs/_build/html/index.html
```




# Class Reference Generation
GUT has a set of tools to generate rst files from comments in code.  It is adapted from Godot's scripts.  See `documentation/class_ref/godot_make_rst.py` for changes from Godot's version and additional features.  Most of the scripts in `documentation/class_ref` are from splitting up Godot's script.

The way it works is that XML is generated from scripts.  The XML is then used to generate .rst files.  These .rst files are used by Sphinx to make HTML files.  For the documentation generation, the generated .rst files are under version control, the HTML and XML is not.


## Setup
The class reference documentation generation toolkit wrapper script requires:
* `zsh`
* `python3`
* Docker (per above requirements)

Before generating class reference:
* The project must have been opened in the editor or you have run an import (`godot --import`).  No xml files will be generated if not.
* You must have created the docker image already, per the directions above.
* The environment variable `GODOT` must be set to the godot executable.


## Execution
From the root of this project run:
* `zsh documentation/generate_rst.sh`

Output will be located in the following directories:
* XML:  `documentation/class_ref_xml`
* RST:  `documentation/docs/class_ref`
* HTML:  `documentation/docs/_build/html/class_ref`

Class Reference will be on the index page the the bottom of the TOC.
