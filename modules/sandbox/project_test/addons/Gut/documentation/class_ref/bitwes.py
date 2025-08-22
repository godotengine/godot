import logger as lgr

godot_classes = []
def make_type_link(link_type):
    if(not link_type in godot_classes):
        godot_classes.append(link_type)
        lgr.print_style("bold", f'Linking "{link_type}" as a Godot class.')

    return f"`{link_type} <https://docs.godotengine.org/en/stable/classes/class_{link_type.lower()}.html>`_"
