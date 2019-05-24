

static const spv_ext_inst_desc_t spv_amd_shader_ballot_entries[] = {
  {"SwizzleInvocationsAMD", 1, 0, nullptr, {SPV_OPERAND_TYPE_ID, SPV_OPERAND_TYPE_ID, SPV_OPERAND_TYPE_NONE}},
  {"SwizzleInvocationsMaskedAMD", 2, 0, nullptr, {SPV_OPERAND_TYPE_ID, SPV_OPERAND_TYPE_ID, SPV_OPERAND_TYPE_NONE}},
  {"WriteInvocationAMD", 3, 0, nullptr, {SPV_OPERAND_TYPE_ID, SPV_OPERAND_TYPE_ID, SPV_OPERAND_TYPE_ID, SPV_OPERAND_TYPE_NONE}},
  {"MbcntAMD", 4, 0, nullptr, {SPV_OPERAND_TYPE_ID, SPV_OPERAND_TYPE_NONE}}
};
