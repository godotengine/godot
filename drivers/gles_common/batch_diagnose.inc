String get_command_type_string(const RasterizerCanvas::Item::Command &p_command) const {
	String sz = "";

	switch (p_command.type) {
		default:
			break;
		case RasterizerCanvas::Item::Command::TYPE_LINE: {
			sz = "l";
		} break;
		case RasterizerCanvas::Item::Command::TYPE_POLYLINE: {
			sz = "PL";
		} break;
		case RasterizerCanvas::Item::Command::TYPE_RECT: {
			sz = "r";
		} break;
		case RasterizerCanvas::Item::Command::TYPE_NINEPATCH: {
			sz = "n";
		} break;
		case RasterizerCanvas::Item::Command::TYPE_PRIMITIVE: {
			sz = "PR";
		} break;
		case RasterizerCanvas::Item::Command::TYPE_POLYGON: {
			sz = "p";
		} break;
		case RasterizerCanvas::Item::Command::TYPE_MESH: {
			sz = "m";
		} break;
		case RasterizerCanvas::Item::Command::TYPE_MULTIMESH: {
			sz = "MM";
		} break;
		case RasterizerCanvas::Item::Command::TYPE_PARTICLES: {
			sz = "PA";
		} break;
		case RasterizerCanvas::Item::Command::TYPE_CIRCLE: {
			sz = "c";
		} break;
		case RasterizerCanvas::Item::Command::TYPE_TRANSFORM: {
			sz = "t";

			// add a bit more info in debug build
			const RasterizerCanvas::Item::CommandTransform *transform = static_cast<const RasterizerCanvas::Item::CommandTransform *>(&p_command);
			const Transform2D &mat = transform->xform;

			sz += " ";
			sz += String(Variant(mat.elements[2]));
			sz += " ";
		} break;
		case RasterizerCanvas::Item::Command::TYPE_CLIP_IGNORE: {
			sz = "CI";
		} break;
	} // switch

	return sz;
}

void diagnose_batches(RasterizerCanvas::Item::Command *const *p_commands) {
	int num_batches = bdata.batches.size();

	BatchColor curr_color;
	curr_color.set(Color(-1, -1, -1, -1));
	bool first_color_change = true;

	for (int batch_num = 0; batch_num < num_batches; batch_num++) {
		const Batch &batch = bdata.batches[batch_num];
		bdata.frame_string += "\t\t\tbatch ";

		switch (batch.type) {

			case RasterizerStorageCommon::BT_POLY: {
				bdata.frame_string += "P ";
				bdata.frame_string += itos(batch.first_command) + "-";
				bdata.frame_string += itos(batch.num_commands);

				bdata.frame_string += " " + batch.color.to_string();

				if (batch.num_commands > 1) {
					bdata.frame_string += " MULTI";
				}
				if (curr_color != batch.color) {
					curr_color = batch.color;
					if (!first_color_change) {
						bdata.frame_string += " color";
					} else {
						first_color_change = false;
					}
				}
				bdata.frame_string += "\n";
			} break;
			case RasterizerStorageCommon::BT_LINE:
			case RasterizerStorageCommon::BT_LINE_AA: {
				bdata.frame_string += "L ";
				bdata.frame_string += itos(batch.first_command) + "-";
				bdata.frame_string += itos(batch.num_commands);

				bdata.frame_string += " " + batch.color.to_string();

				if (batch.num_commands > 1) {
					bdata.frame_string += " MULTI";
				}
				if (curr_color != batch.color) {
					curr_color = batch.color;
					if (!first_color_change) {
						bdata.frame_string += " color";
					} else {
						first_color_change = false;
					}
				}
				bdata.frame_string += "\n";
			} break;
			case RasterizerStorageCommon::BT_RECT: {
				bdata.frame_string += "R ";
				bdata.frame_string += itos(batch.first_command) + "-";
				bdata.frame_string += itos(batch.num_commands);

				int tex_id = (int)bdata.batch_textures[batch.batch_texture_id].RID_texture.get_id();
				bdata.frame_string += " [" + itos(batch.batch_texture_id) + " - " + itos(tex_id) + "]";

				bdata.frame_string += " " + batch.color.to_string();

				if (batch.num_commands > 1) {
					bdata.frame_string += " MULTI";
				}
				if (curr_color != batch.color) {
					curr_color = batch.color;
					if (!first_color_change) {
						bdata.frame_string += " color";
					} else {
						first_color_change = false;
					}
				}
				bdata.frame_string += "\n";
			} break;
			default: {
				bdata.frame_string += "D ";
				bdata.frame_string += itos(batch.first_command) + "-";
				bdata.frame_string += itos(batch.num_commands) + " ";

				int num_show = MIN(batch.num_commands, 16);
				for (int n = 0; n < num_show; n++) {
					const RasterizerCanvas::Item::Command &comm = *p_commands[batch.first_command + n];
					bdata.frame_string += get_command_type_string(comm) + " ";
				}

				bdata.frame_string += "\n";
			} break;
		}
	}
}
