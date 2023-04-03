ir_function_signature *
udivmod64(void *mem_ctx, builtin_available_predicate avail)
{
   ir_function_signature *const sig =
      new(mem_ctx) ir_function_signature(glsl_type::uvec4_type, avail);
   ir_factory body(&sig->body, mem_ctx);
   sig->is_defined = true;

   exec_list sig_parameters;

   ir_variable *const r000C = new(mem_ctx) ir_variable(glsl_type::uvec2_type, "n", ir_var_function_in);
   sig_parameters.push_tail(r000C);
   ir_variable *const r000D = new(mem_ctx) ir_variable(glsl_type::uvec2_type, "d", ir_var_function_in);
   sig_parameters.push_tail(r000D);
   ir_variable *const r000E = new(mem_ctx) ir_variable(glsl_type::int_type, "i", ir_var_auto);
   body.emit(r000E);
   ir_variable *const r000F = new(mem_ctx) ir_variable(glsl_type::uint64_t_type, "n64", ir_var_auto);
   body.emit(r000F);
   ir_variable *const r0010 = new(mem_ctx) ir_variable(glsl_type::int_type, "log2_denom", ir_var_auto);
   body.emit(r0010);
   ir_variable *const r0011 = new(mem_ctx) ir_variable(glsl_type::uvec2_type, "quot", ir_var_auto);
   body.emit(r0011);
   body.emit(assign(r0011, ir_constant::zero(mem_ctx, glsl_type::uvec2_type), 0x03));

   ir_expression *const r0012 = expr(ir_unop_find_msb, swizzle_y(r000D));
   body.emit(assign(r0010, add(r0012, body.constant(int(32))), 0x01));

   /* IF CONDITION */
   ir_expression *const r0014 = equal(swizzle_y(r000D), body.constant(0u));
   ir_expression *const r0015 = gequal(swizzle_y(r000C), swizzle_x(r000D));
   ir_expression *const r0016 = logic_and(r0014, r0015);
   ir_if *f0013 = new(mem_ctx) ir_if(operand(r0016).val);
   exec_list *const f0013_parent_instructions = body.instructions;

      /* THEN INSTRUCTIONS */
      body.instructions = &f0013->then_instructions;

      ir_variable *const r0017 = new(mem_ctx) ir_variable(glsl_type::int_type, "i", ir_var_auto);
      body.emit(r0017);
      ir_variable *const r0018 = body.make_temp(glsl_type::int_type, "findMSB_retval");
      body.emit(assign(r0018, expr(ir_unop_find_msb, swizzle_x(r000D)), 0x01));

      body.emit(assign(r0010, r0018, 0x01));

      body.emit(assign(r0017, body.constant(int(31)), 0x01));

      /* LOOP BEGIN */
      ir_loop *f0019 = new(mem_ctx) ir_loop();
      exec_list *const f0019_parent_instructions = body.instructions;

         body.instructions = &f0019->body_instructions;

         /* IF CONDITION */
         ir_expression *const r001B = less(r0017, body.constant(int(1)));
         ir_if *f001A = new(mem_ctx) ir_if(operand(r001B).val);
         exec_list *const f001A_parent_instructions = body.instructions;

            /* THEN INSTRUCTIONS */
            body.instructions = &f001A->then_instructions;

            body.emit(new(mem_ctx) ir_loop_jump(ir_loop_jump::jump_break));


         body.instructions = f001A_parent_instructions;
         body.emit(f001A);

         /* END IF */

         /* IF CONDITION */
         ir_expression *const r001D = sub(body.constant(int(31)), r0017);
         ir_expression *const r001E = lequal(r0018, r001D);
         ir_expression *const r001F = lshift(swizzle_x(r000D), r0017);
         ir_expression *const r0020 = lequal(r001F, swizzle_y(r000C));
         ir_expression *const r0021 = logic_and(r001E, r0020);
         ir_if *f001C = new(mem_ctx) ir_if(operand(r0021).val);
         exec_list *const f001C_parent_instructions = body.instructions;

            /* THEN INSTRUCTIONS */
            body.instructions = &f001C->then_instructions;

            ir_expression *const r0022 = lshift(swizzle_x(r000D), r0017);
            body.emit(assign(r000C, sub(swizzle_y(r000C), r0022), 0x02));

            ir_expression *const r0023 = lshift(body.constant(1u), r0017);
            body.emit(assign(r0011, bit_or(swizzle_y(r0011), r0023), 0x02));


         body.instructions = f001C_parent_instructions;
         body.emit(f001C);

         /* END IF */

         body.emit(assign(r0017, add(r0017, body.constant(int(-1))), 0x01));

      /* LOOP END */

      body.instructions = f0019_parent_instructions;
      body.emit(f0019);

      /* IF CONDITION */
      ir_expression *const r0025 = lequal(swizzle_x(r000D), swizzle_y(r000C));
      ir_if *f0024 = new(mem_ctx) ir_if(operand(r0025).val);
      exec_list *const f0024_parent_instructions = body.instructions;

         /* THEN INSTRUCTIONS */
         body.instructions = &f0024->then_instructions;

         body.emit(assign(r000C, sub(swizzle_y(r000C), swizzle_x(r000D)), 0x02));

         body.emit(assign(r0011, bit_or(swizzle_y(r0011), body.constant(1u)), 0x02));


      body.instructions = f0024_parent_instructions;
      body.emit(f0024);

      /* END IF */


   body.instructions = f0013_parent_instructions;
   body.emit(f0013);

   /* END IF */

   ir_variable *const r0026 = body.make_temp(glsl_type::uint64_t_type, "packUint2x32_retval");
   body.emit(assign(r0026, expr(ir_unop_pack_uint_2x32, r000D), 0x01));

   body.emit(assign(r000F, expr(ir_unop_pack_uint_2x32, r000C), 0x01));

   body.emit(assign(r000E, body.constant(int(31)), 0x01));

   /* LOOP BEGIN */
   ir_loop *f0027 = new(mem_ctx) ir_loop();
   exec_list *const f0027_parent_instructions = body.instructions;

      body.instructions = &f0027->body_instructions;

      /* IF CONDITION */
      ir_expression *const r0029 = less(r000E, body.constant(int(1)));
      ir_if *f0028 = new(mem_ctx) ir_if(operand(r0029).val);
      exec_list *const f0028_parent_instructions = body.instructions;

         /* THEN INSTRUCTIONS */
         body.instructions = &f0028->then_instructions;

         body.emit(new(mem_ctx) ir_loop_jump(ir_loop_jump::jump_break));


      body.instructions = f0028_parent_instructions;
      body.emit(f0028);

      /* END IF */

      /* IF CONDITION */
      ir_expression *const r002B = sub(body.constant(int(63)), r000E);
      ir_expression *const r002C = lequal(r0010, r002B);
      ir_expression *const r002D = lshift(r0026, r000E);
      ir_expression *const r002E = lequal(r002D, r000F);
      ir_expression *const r002F = logic_and(r002C, r002E);
      ir_if *f002A = new(mem_ctx) ir_if(operand(r002F).val);
      exec_list *const f002A_parent_instructions = body.instructions;

         /* THEN INSTRUCTIONS */
         body.instructions = &f002A->then_instructions;

         ir_expression *const r0030 = lshift(r0026, r000E);
         body.emit(assign(r000F, sub(r000F, r0030), 0x01));

         ir_expression *const r0031 = lshift(body.constant(1u), r000E);
         body.emit(assign(r0011, bit_or(swizzle_x(r0011), r0031), 0x01));


      body.instructions = f002A_parent_instructions;
      body.emit(f002A);

      /* END IF */

      body.emit(assign(r000E, add(r000E, body.constant(int(-1))), 0x01));

   /* LOOP END */

   body.instructions = f0027_parent_instructions;
   body.emit(f0027);

   /* IF CONDITION */
   ir_expression *const r0033 = lequal(r0026, r000F);
   ir_if *f0032 = new(mem_ctx) ir_if(operand(r0033).val);
   exec_list *const f0032_parent_instructions = body.instructions;

      /* THEN INSTRUCTIONS */
      body.instructions = &f0032->then_instructions;

      body.emit(assign(r000F, sub(r000F, r0026), 0x01));

      body.emit(assign(r0011, bit_or(swizzle_x(r0011), body.constant(1u)), 0x01));


   body.instructions = f0032_parent_instructions;
   body.emit(f0032);

   /* END IF */

   ir_variable *const r0034 = body.make_temp(glsl_type::uvec4_type, "vec_ctor");
   body.emit(assign(r0034, r0011, 0x03));

   body.emit(assign(r0034, expr(ir_unop_unpack_uint_2x32, r000F), 0x0c));

   body.emit(ret(r0034));

   sig->replace_parameters(&sig_parameters);
   return sig;
}
ir_function_signature *
udiv64(void *mem_ctx, builtin_available_predicate avail)
{
   ir_function_signature *const sig =
      new(mem_ctx) ir_function_signature(glsl_type::uvec2_type, avail);
   ir_factory body(&sig->body, mem_ctx);
   sig->is_defined = true;

   exec_list sig_parameters;

   ir_variable *const r0035 = new(mem_ctx) ir_variable(glsl_type::uvec2_type, "n", ir_var_function_in);
   sig_parameters.push_tail(r0035);
   ir_variable *const r0036 = new(mem_ctx) ir_variable(glsl_type::uvec2_type, "d", ir_var_function_in);
   sig_parameters.push_tail(r0036);
   ir_variable *const r0037 = body.make_temp(glsl_type::uvec2_type, "n");
   body.emit(assign(r0037, r0035, 0x03));

   ir_variable *const r0038 = new(mem_ctx) ir_variable(glsl_type::int_type, "i", ir_var_auto);
   body.emit(r0038);
   ir_variable *const r0039 = new(mem_ctx) ir_variable(glsl_type::uint64_t_type, "n64", ir_var_auto);
   body.emit(r0039);
   ir_variable *const r003A = new(mem_ctx) ir_variable(glsl_type::int_type, "log2_denom", ir_var_auto);
   body.emit(r003A);
   ir_variable *const r003B = new(mem_ctx) ir_variable(glsl_type::uvec2_type, "quot", ir_var_auto);
   body.emit(r003B);
   body.emit(assign(r003B, ir_constant::zero(mem_ctx, glsl_type::uvec2_type), 0x03));

   ir_expression *const r003C = expr(ir_unop_find_msb, swizzle_y(r0036));
   body.emit(assign(r003A, add(r003C, body.constant(int(32))), 0x01));

   /* IF CONDITION */
   ir_expression *const r003E = equal(swizzle_y(r0036), body.constant(0u));
   ir_expression *const r003F = gequal(swizzle_y(r0035), swizzle_x(r0036));
   ir_expression *const r0040 = logic_and(r003E, r003F);
   ir_if *f003D = new(mem_ctx) ir_if(operand(r0040).val);
   exec_list *const f003D_parent_instructions = body.instructions;

      /* THEN INSTRUCTIONS */
      body.instructions = &f003D->then_instructions;

      ir_variable *const r0041 = new(mem_ctx) ir_variable(glsl_type::int_type, "i", ir_var_auto);
      body.emit(r0041);
      ir_variable *const r0042 = body.make_temp(glsl_type::int_type, "findMSB_retval");
      body.emit(assign(r0042, expr(ir_unop_find_msb, swizzle_x(r0036)), 0x01));

      body.emit(assign(r003A, r0042, 0x01));

      body.emit(assign(r0041, body.constant(int(31)), 0x01));

      /* LOOP BEGIN */
      ir_loop *f0043 = new(mem_ctx) ir_loop();
      exec_list *const f0043_parent_instructions = body.instructions;

         body.instructions = &f0043->body_instructions;

         /* IF CONDITION */
         ir_expression *const r0045 = less(r0041, body.constant(int(1)));
         ir_if *f0044 = new(mem_ctx) ir_if(operand(r0045).val);
         exec_list *const f0044_parent_instructions = body.instructions;

            /* THEN INSTRUCTIONS */
            body.instructions = &f0044->then_instructions;

            body.emit(new(mem_ctx) ir_loop_jump(ir_loop_jump::jump_break));


         body.instructions = f0044_parent_instructions;
         body.emit(f0044);

         /* END IF */

         /* IF CONDITION */
         ir_expression *const r0047 = sub(body.constant(int(31)), r0041);
         ir_expression *const r0048 = lequal(r0042, r0047);
         ir_expression *const r0049 = lshift(swizzle_x(r0036), r0041);
         ir_expression *const r004A = lequal(r0049, swizzle_y(r0037));
         ir_expression *const r004B = logic_and(r0048, r004A);
         ir_if *f0046 = new(mem_ctx) ir_if(operand(r004B).val);
         exec_list *const f0046_parent_instructions = body.instructions;

            /* THEN INSTRUCTIONS */
            body.instructions = &f0046->then_instructions;

            ir_expression *const r004C = lshift(swizzle_x(r0036), r0041);
            body.emit(assign(r0037, sub(swizzle_y(r0037), r004C), 0x02));

            ir_expression *const r004D = lshift(body.constant(1u), r0041);
            body.emit(assign(r003B, bit_or(swizzle_y(r003B), r004D), 0x02));


         body.instructions = f0046_parent_instructions;
         body.emit(f0046);

         /* END IF */

         body.emit(assign(r0041, add(r0041, body.constant(int(-1))), 0x01));

      /* LOOP END */

      body.instructions = f0043_parent_instructions;
      body.emit(f0043);

      /* IF CONDITION */
      ir_expression *const r004F = lequal(swizzle_x(r0036), swizzle_y(r0037));
      ir_if *f004E = new(mem_ctx) ir_if(operand(r004F).val);
      exec_list *const f004E_parent_instructions = body.instructions;

         /* THEN INSTRUCTIONS */
         body.instructions = &f004E->then_instructions;

         body.emit(assign(r0037, sub(swizzle_y(r0037), swizzle_x(r0036)), 0x02));

         body.emit(assign(r003B, bit_or(swizzle_y(r003B), body.constant(1u)), 0x02));


      body.instructions = f004E_parent_instructions;
      body.emit(f004E);

      /* END IF */


   body.instructions = f003D_parent_instructions;
   body.emit(f003D);

   /* END IF */

   ir_variable *const r0050 = body.make_temp(glsl_type::uint64_t_type, "packUint2x32_retval");
   body.emit(assign(r0050, expr(ir_unop_pack_uint_2x32, r0036), 0x01));

   body.emit(assign(r0039, expr(ir_unop_pack_uint_2x32, r0037), 0x01));

   body.emit(assign(r0038, body.constant(int(31)), 0x01));

   /* LOOP BEGIN */
   ir_loop *f0051 = new(mem_ctx) ir_loop();
   exec_list *const f0051_parent_instructions = body.instructions;

      body.instructions = &f0051->body_instructions;

      /* IF CONDITION */
      ir_expression *const r0053 = less(r0038, body.constant(int(1)));
      ir_if *f0052 = new(mem_ctx) ir_if(operand(r0053).val);
      exec_list *const f0052_parent_instructions = body.instructions;

         /* THEN INSTRUCTIONS */
         body.instructions = &f0052->then_instructions;

         body.emit(new(mem_ctx) ir_loop_jump(ir_loop_jump::jump_break));


      body.instructions = f0052_parent_instructions;
      body.emit(f0052);

      /* END IF */

      /* IF CONDITION */
      ir_expression *const r0055 = sub(body.constant(int(63)), r0038);
      ir_expression *const r0056 = lequal(r003A, r0055);
      ir_expression *const r0057 = lshift(r0050, r0038);
      ir_expression *const r0058 = lequal(r0057, r0039);
      ir_expression *const r0059 = logic_and(r0056, r0058);
      ir_if *f0054 = new(mem_ctx) ir_if(operand(r0059).val);
      exec_list *const f0054_parent_instructions = body.instructions;

         /* THEN INSTRUCTIONS */
         body.instructions = &f0054->then_instructions;

         ir_expression *const r005A = lshift(r0050, r0038);
         body.emit(assign(r0039, sub(r0039, r005A), 0x01));

         ir_expression *const r005B = lshift(body.constant(1u), r0038);
         body.emit(assign(r003B, bit_or(swizzle_x(r003B), r005B), 0x01));


      body.instructions = f0054_parent_instructions;
      body.emit(f0054);

      /* END IF */

      body.emit(assign(r0038, add(r0038, body.constant(int(-1))), 0x01));

   /* LOOP END */

   body.instructions = f0051_parent_instructions;
   body.emit(f0051);

   /* IF CONDITION */
   ir_expression *const r005D = lequal(r0050, r0039);
   ir_if *f005C = new(mem_ctx) ir_if(operand(r005D).val);
   exec_list *const f005C_parent_instructions = body.instructions;

      /* THEN INSTRUCTIONS */
      body.instructions = &f005C->then_instructions;

      body.emit(assign(r0039, sub(r0039, r0050), 0x01));

      body.emit(assign(r003B, bit_or(swizzle_x(r003B), body.constant(1u)), 0x01));


   body.instructions = f005C_parent_instructions;
   body.emit(f005C);

   /* END IF */

   body.emit(ret(r003B));

   sig->replace_parameters(&sig_parameters);
   return sig;
}
ir_function_signature *
idiv64(void *mem_ctx, builtin_available_predicate avail)
{
   ir_function_signature *const sig =
      new(mem_ctx) ir_function_signature(glsl_type::ivec2_type, avail);
   ir_factory body(&sig->body, mem_ctx);
   sig->is_defined = true;

   exec_list sig_parameters;

   ir_variable *const r005E = new(mem_ctx) ir_variable(glsl_type::ivec2_type, "_n", ir_var_function_in);
   sig_parameters.push_tail(r005E);
   ir_variable *const r005F = new(mem_ctx) ir_variable(glsl_type::ivec2_type, "_d", ir_var_function_in);
   sig_parameters.push_tail(r005F);
   ir_variable *const r0060 = new(mem_ctx) ir_variable(glsl_type::bool_type, "negate", ir_var_auto);
   body.emit(r0060);
   ir_expression *const r0061 = less(swizzle_y(r005E), body.constant(int(0)));
   ir_expression *const r0062 = less(swizzle_y(r005F), body.constant(int(0)));
   body.emit(assign(r0060, nequal(r0061, r0062), 0x01));

   ir_variable *const r0063 = body.make_temp(glsl_type::uvec2_type, "n");
   ir_expression *const r0064 = expr(ir_unop_pack_int_2x32, r005E);
   ir_expression *const r0065 = expr(ir_unop_abs, r0064);
   ir_expression *const r0066 = expr(ir_unop_i642u64, r0065);
   body.emit(assign(r0063, expr(ir_unop_unpack_uint_2x32, r0066), 0x03));

   ir_variable *const r0067 = body.make_temp(glsl_type::uvec2_type, "d");
   ir_expression *const r0068 = expr(ir_unop_pack_int_2x32, r005F);
   ir_expression *const r0069 = expr(ir_unop_abs, r0068);
   ir_expression *const r006A = expr(ir_unop_i642u64, r0069);
   body.emit(assign(r0067, expr(ir_unop_unpack_uint_2x32, r006A), 0x03));

   ir_variable *const r006B = new(mem_ctx) ir_variable(glsl_type::int_type, "i", ir_var_auto);
   body.emit(r006B);
   ir_variable *const r006C = new(mem_ctx) ir_variable(glsl_type::uint64_t_type, "n64", ir_var_auto);
   body.emit(r006C);
   ir_variable *const r006D = new(mem_ctx) ir_variable(glsl_type::int_type, "log2_denom", ir_var_auto);
   body.emit(r006D);
   ir_variable *const r006E = new(mem_ctx) ir_variable(glsl_type::uvec2_type, "quot", ir_var_auto);
   body.emit(r006E);
   body.emit(assign(r006E, ir_constant::zero(mem_ctx, glsl_type::uvec2_type), 0x03));

   ir_expression *const r006F = expr(ir_unop_find_msb, swizzle_y(r0067));
   body.emit(assign(r006D, add(r006F, body.constant(int(32))), 0x01));

   /* IF CONDITION */
   ir_expression *const r0071 = equal(swizzle_y(r0067), body.constant(0u));
   ir_expression *const r0072 = gequal(swizzle_y(r0063), swizzle_x(r0067));
   ir_expression *const r0073 = logic_and(r0071, r0072);
   ir_if *f0070 = new(mem_ctx) ir_if(operand(r0073).val);
   exec_list *const f0070_parent_instructions = body.instructions;

      /* THEN INSTRUCTIONS */
      body.instructions = &f0070->then_instructions;

      ir_variable *const r0074 = new(mem_ctx) ir_variable(glsl_type::int_type, "i", ir_var_auto);
      body.emit(r0074);
      ir_variable *const r0075 = body.make_temp(glsl_type::int_type, "findMSB_retval");
      body.emit(assign(r0075, expr(ir_unop_find_msb, swizzle_x(r0067)), 0x01));

      body.emit(assign(r006D, r0075, 0x01));

      body.emit(assign(r0074, body.constant(int(31)), 0x01));

      /* LOOP BEGIN */
      ir_loop *f0076 = new(mem_ctx) ir_loop();
      exec_list *const f0076_parent_instructions = body.instructions;

         body.instructions = &f0076->body_instructions;

         /* IF CONDITION */
         ir_expression *const r0078 = less(r0074, body.constant(int(1)));
         ir_if *f0077 = new(mem_ctx) ir_if(operand(r0078).val);
         exec_list *const f0077_parent_instructions = body.instructions;

            /* THEN INSTRUCTIONS */
            body.instructions = &f0077->then_instructions;

            body.emit(new(mem_ctx) ir_loop_jump(ir_loop_jump::jump_break));


         body.instructions = f0077_parent_instructions;
         body.emit(f0077);

         /* END IF */

         /* IF CONDITION */
         ir_expression *const r007A = sub(body.constant(int(31)), r0074);
         ir_expression *const r007B = lequal(r0075, r007A);
         ir_expression *const r007C = lshift(swizzle_x(r0067), r0074);
         ir_expression *const r007D = lequal(r007C, swizzle_y(r0063));
         ir_expression *const r007E = logic_and(r007B, r007D);
         ir_if *f0079 = new(mem_ctx) ir_if(operand(r007E).val);
         exec_list *const f0079_parent_instructions = body.instructions;

            /* THEN INSTRUCTIONS */
            body.instructions = &f0079->then_instructions;

            ir_expression *const r007F = lshift(swizzle_x(r0067), r0074);
            body.emit(assign(r0063, sub(swizzle_y(r0063), r007F), 0x02));

            ir_expression *const r0080 = lshift(body.constant(1u), r0074);
            body.emit(assign(r006E, bit_or(swizzle_y(r006E), r0080), 0x02));


         body.instructions = f0079_parent_instructions;
         body.emit(f0079);

         /* END IF */

         body.emit(assign(r0074, add(r0074, body.constant(int(-1))), 0x01));

      /* LOOP END */

      body.instructions = f0076_parent_instructions;
      body.emit(f0076);

      /* IF CONDITION */
      ir_expression *const r0082 = lequal(swizzle_x(r0067), swizzle_y(r0063));
      ir_if *f0081 = new(mem_ctx) ir_if(operand(r0082).val);
      exec_list *const f0081_parent_instructions = body.instructions;

         /* THEN INSTRUCTIONS */
         body.instructions = &f0081->then_instructions;

         body.emit(assign(r0063, sub(swizzle_y(r0063), swizzle_x(r0067)), 0x02));

         body.emit(assign(r006E, bit_or(swizzle_y(r006E), body.constant(1u)), 0x02));


      body.instructions = f0081_parent_instructions;
      body.emit(f0081);

      /* END IF */


   body.instructions = f0070_parent_instructions;
   body.emit(f0070);

   /* END IF */

   ir_variable *const r0083 = body.make_temp(glsl_type::uint64_t_type, "packUint2x32_retval");
   body.emit(assign(r0083, expr(ir_unop_pack_uint_2x32, r0067), 0x01));

   body.emit(assign(r006C, expr(ir_unop_pack_uint_2x32, r0063), 0x01));

   body.emit(assign(r006B, body.constant(int(31)), 0x01));

   /* LOOP BEGIN */
   ir_loop *f0084 = new(mem_ctx) ir_loop();
   exec_list *const f0084_parent_instructions = body.instructions;

      body.instructions = &f0084->body_instructions;

      /* IF CONDITION */
      ir_expression *const r0086 = less(r006B, body.constant(int(1)));
      ir_if *f0085 = new(mem_ctx) ir_if(operand(r0086).val);
      exec_list *const f0085_parent_instructions = body.instructions;

         /* THEN INSTRUCTIONS */
         body.instructions = &f0085->then_instructions;

         body.emit(new(mem_ctx) ir_loop_jump(ir_loop_jump::jump_break));


      body.instructions = f0085_parent_instructions;
      body.emit(f0085);

      /* END IF */

      /* IF CONDITION */
      ir_expression *const r0088 = sub(body.constant(int(63)), r006B);
      ir_expression *const r0089 = lequal(r006D, r0088);
      ir_expression *const r008A = lshift(r0083, r006B);
      ir_expression *const r008B = lequal(r008A, r006C);
      ir_expression *const r008C = logic_and(r0089, r008B);
      ir_if *f0087 = new(mem_ctx) ir_if(operand(r008C).val);
      exec_list *const f0087_parent_instructions = body.instructions;

         /* THEN INSTRUCTIONS */
         body.instructions = &f0087->then_instructions;

         ir_expression *const r008D = lshift(r0083, r006B);
         body.emit(assign(r006C, sub(r006C, r008D), 0x01));

         ir_expression *const r008E = lshift(body.constant(1u), r006B);
         body.emit(assign(r006E, bit_or(swizzle_x(r006E), r008E), 0x01));


      body.instructions = f0087_parent_instructions;
      body.emit(f0087);

      /* END IF */

      body.emit(assign(r006B, add(r006B, body.constant(int(-1))), 0x01));

   /* LOOP END */

   body.instructions = f0084_parent_instructions;
   body.emit(f0084);

   /* IF CONDITION */
   ir_expression *const r0090 = lequal(r0083, r006C);
   ir_if *f008F = new(mem_ctx) ir_if(operand(r0090).val);
   exec_list *const f008F_parent_instructions = body.instructions;

      /* THEN INSTRUCTIONS */
      body.instructions = &f008F->then_instructions;

      body.emit(assign(r006C, sub(r006C, r0083), 0x01));

      body.emit(assign(r006E, bit_or(swizzle_x(r006E), body.constant(1u)), 0x01));


   body.instructions = f008F_parent_instructions;
   body.emit(f008F);

   /* END IF */

   ir_variable *const r0091 = body.make_temp(glsl_type::ivec2_type, "conditional_tmp");
   /* IF CONDITION */
   ir_if *f0092 = new(mem_ctx) ir_if(operand(r0060).val);
   exec_list *const f0092_parent_instructions = body.instructions;

      /* THEN INSTRUCTIONS */
      body.instructions = &f0092->then_instructions;

      ir_expression *const r0093 = expr(ir_unop_pack_uint_2x32, r006E);
      ir_expression *const r0094 = expr(ir_unop_u642i64, r0093);
      ir_expression *const r0095 = neg(r0094);
      body.emit(assign(r0091, expr(ir_unop_unpack_int_2x32, r0095), 0x03));


      /* ELSE INSTRUCTIONS */
      body.instructions = &f0092->else_instructions;

      body.emit(assign(r0091, expr(ir_unop_u2i, r006E), 0x03));


   body.instructions = f0092_parent_instructions;
   body.emit(f0092);

   /* END IF */

   body.emit(ret(r0091));

   sig->replace_parameters(&sig_parameters);
   return sig;
}
ir_function_signature *
umod64(void *mem_ctx, builtin_available_predicate avail)
{
   ir_function_signature *const sig =
      new(mem_ctx) ir_function_signature(glsl_type::uvec2_type, avail);
   ir_factory body(&sig->body, mem_ctx);
   sig->is_defined = true;

   exec_list sig_parameters;

   ir_variable *const r0096 = new(mem_ctx) ir_variable(glsl_type::uvec2_type, "n", ir_var_function_in);
   sig_parameters.push_tail(r0096);
   ir_variable *const r0097 = new(mem_ctx) ir_variable(glsl_type::uvec2_type, "d", ir_var_function_in);
   sig_parameters.push_tail(r0097);
   ir_variable *const r0098 = body.make_temp(glsl_type::uvec2_type, "n");
   body.emit(assign(r0098, r0096, 0x03));

   ir_variable *const r0099 = new(mem_ctx) ir_variable(glsl_type::int_type, "i", ir_var_auto);
   body.emit(r0099);
   ir_variable *const r009A = new(mem_ctx) ir_variable(glsl_type::uint64_t_type, "n64", ir_var_auto);
   body.emit(r009A);
   ir_variable *const r009B = new(mem_ctx) ir_variable(glsl_type::int_type, "log2_denom", ir_var_auto);
   body.emit(r009B);
   ir_variable *const r009C = new(mem_ctx) ir_variable(glsl_type::uvec2_type, "quot", ir_var_auto);
   body.emit(r009C);
   body.emit(assign(r009C, ir_constant::zero(mem_ctx, glsl_type::uvec2_type), 0x03));

   ir_expression *const r009D = expr(ir_unop_find_msb, swizzle_y(r0097));
   body.emit(assign(r009B, add(r009D, body.constant(int(32))), 0x01));

   /* IF CONDITION */
   ir_expression *const r009F = equal(swizzle_y(r0097), body.constant(0u));
   ir_expression *const r00A0 = gequal(swizzle_y(r0096), swizzle_x(r0097));
   ir_expression *const r00A1 = logic_and(r009F, r00A0);
   ir_if *f009E = new(mem_ctx) ir_if(operand(r00A1).val);
   exec_list *const f009E_parent_instructions = body.instructions;

      /* THEN INSTRUCTIONS */
      body.instructions = &f009E->then_instructions;

      ir_variable *const r00A2 = new(mem_ctx) ir_variable(glsl_type::int_type, "i", ir_var_auto);
      body.emit(r00A2);
      ir_variable *const r00A3 = body.make_temp(glsl_type::int_type, "findMSB_retval");
      body.emit(assign(r00A3, expr(ir_unop_find_msb, swizzle_x(r0097)), 0x01));

      body.emit(assign(r009B, r00A3, 0x01));

      body.emit(assign(r00A2, body.constant(int(31)), 0x01));

      /* LOOP BEGIN */
      ir_loop *f00A4 = new(mem_ctx) ir_loop();
      exec_list *const f00A4_parent_instructions = body.instructions;

         body.instructions = &f00A4->body_instructions;

         /* IF CONDITION */
         ir_expression *const r00A6 = less(r00A2, body.constant(int(1)));
         ir_if *f00A5 = new(mem_ctx) ir_if(operand(r00A6).val);
         exec_list *const f00A5_parent_instructions = body.instructions;

            /* THEN INSTRUCTIONS */
            body.instructions = &f00A5->then_instructions;

            body.emit(new(mem_ctx) ir_loop_jump(ir_loop_jump::jump_break));


         body.instructions = f00A5_parent_instructions;
         body.emit(f00A5);

         /* END IF */

         /* IF CONDITION */
         ir_expression *const r00A8 = sub(body.constant(int(31)), r00A2);
         ir_expression *const r00A9 = lequal(r00A3, r00A8);
         ir_expression *const r00AA = lshift(swizzle_x(r0097), r00A2);
         ir_expression *const r00AB = lequal(r00AA, swizzle_y(r0098));
         ir_expression *const r00AC = logic_and(r00A9, r00AB);
         ir_if *f00A7 = new(mem_ctx) ir_if(operand(r00AC).val);
         exec_list *const f00A7_parent_instructions = body.instructions;

            /* THEN INSTRUCTIONS */
            body.instructions = &f00A7->then_instructions;

            ir_expression *const r00AD = lshift(swizzle_x(r0097), r00A2);
            body.emit(assign(r0098, sub(swizzle_y(r0098), r00AD), 0x02));

            ir_expression *const r00AE = lshift(body.constant(1u), r00A2);
            body.emit(assign(r009C, bit_or(swizzle_y(r009C), r00AE), 0x02));


         body.instructions = f00A7_parent_instructions;
         body.emit(f00A7);

         /* END IF */

         body.emit(assign(r00A2, add(r00A2, body.constant(int(-1))), 0x01));

      /* LOOP END */

      body.instructions = f00A4_parent_instructions;
      body.emit(f00A4);

      /* IF CONDITION */
      ir_expression *const r00B0 = lequal(swizzle_x(r0097), swizzle_y(r0098));
      ir_if *f00AF = new(mem_ctx) ir_if(operand(r00B0).val);
      exec_list *const f00AF_parent_instructions = body.instructions;

         /* THEN INSTRUCTIONS */
         body.instructions = &f00AF->then_instructions;

         body.emit(assign(r0098, sub(swizzle_y(r0098), swizzle_x(r0097)), 0x02));

         body.emit(assign(r009C, bit_or(swizzle_y(r009C), body.constant(1u)), 0x02));


      body.instructions = f00AF_parent_instructions;
      body.emit(f00AF);

      /* END IF */


   body.instructions = f009E_parent_instructions;
   body.emit(f009E);

   /* END IF */

   ir_variable *const r00B1 = body.make_temp(glsl_type::uint64_t_type, "packUint2x32_retval");
   body.emit(assign(r00B1, expr(ir_unop_pack_uint_2x32, r0097), 0x01));

   body.emit(assign(r009A, expr(ir_unop_pack_uint_2x32, r0098), 0x01));

   body.emit(assign(r0099, body.constant(int(31)), 0x01));

   /* LOOP BEGIN */
   ir_loop *f00B2 = new(mem_ctx) ir_loop();
   exec_list *const f00B2_parent_instructions = body.instructions;

      body.instructions = &f00B2->body_instructions;

      /* IF CONDITION */
      ir_expression *const r00B4 = less(r0099, body.constant(int(1)));
      ir_if *f00B3 = new(mem_ctx) ir_if(operand(r00B4).val);
      exec_list *const f00B3_parent_instructions = body.instructions;

         /* THEN INSTRUCTIONS */
         body.instructions = &f00B3->then_instructions;

         body.emit(new(mem_ctx) ir_loop_jump(ir_loop_jump::jump_break));


      body.instructions = f00B3_parent_instructions;
      body.emit(f00B3);

      /* END IF */

      /* IF CONDITION */
      ir_expression *const r00B6 = sub(body.constant(int(63)), r0099);
      ir_expression *const r00B7 = lequal(r009B, r00B6);
      ir_expression *const r00B8 = lshift(r00B1, r0099);
      ir_expression *const r00B9 = lequal(r00B8, r009A);
      ir_expression *const r00BA = logic_and(r00B7, r00B9);
      ir_if *f00B5 = new(mem_ctx) ir_if(operand(r00BA).val);
      exec_list *const f00B5_parent_instructions = body.instructions;

         /* THEN INSTRUCTIONS */
         body.instructions = &f00B5->then_instructions;

         ir_expression *const r00BB = lshift(r00B1, r0099);
         body.emit(assign(r009A, sub(r009A, r00BB), 0x01));

         ir_expression *const r00BC = lshift(body.constant(1u), r0099);
         body.emit(assign(r009C, bit_or(swizzle_x(r009C), r00BC), 0x01));


      body.instructions = f00B5_parent_instructions;
      body.emit(f00B5);

      /* END IF */

      body.emit(assign(r0099, add(r0099, body.constant(int(-1))), 0x01));

   /* LOOP END */

   body.instructions = f00B2_parent_instructions;
   body.emit(f00B2);

   /* IF CONDITION */
   ir_expression *const r00BE = lequal(r00B1, r009A);
   ir_if *f00BD = new(mem_ctx) ir_if(operand(r00BE).val);
   exec_list *const f00BD_parent_instructions = body.instructions;

      /* THEN INSTRUCTIONS */
      body.instructions = &f00BD->then_instructions;

      body.emit(assign(r009A, sub(r009A, r00B1), 0x01));

      body.emit(assign(r009C, bit_or(swizzle_x(r009C), body.constant(1u)), 0x01));


   body.instructions = f00BD_parent_instructions;
   body.emit(f00BD);

   /* END IF */

   ir_variable *const r00BF = body.make_temp(glsl_type::uvec4_type, "vec_ctor");
   body.emit(assign(r00BF, r009C, 0x03));

   body.emit(assign(r00BF, expr(ir_unop_unpack_uint_2x32, r009A), 0x0c));

   ir_swizzle *const r00C0 = swizzle(r00BF, MAKE_SWIZZLE4(SWIZZLE_Z, SWIZZLE_W, SWIZZLE_X, SWIZZLE_X), 2);
   body.emit(ret(r00C0));

   sig->replace_parameters(&sig_parameters);
   return sig;
}
ir_function_signature *
imod64(void *mem_ctx, builtin_available_predicate avail)
{
   ir_function_signature *const sig =
      new(mem_ctx) ir_function_signature(glsl_type::ivec2_type, avail);
   ir_factory body(&sig->body, mem_ctx);
   sig->is_defined = true;

   exec_list sig_parameters;

   ir_variable *const r00C1 = new(mem_ctx) ir_variable(glsl_type::ivec2_type, "_n", ir_var_function_in);
   sig_parameters.push_tail(r00C1);
   ir_variable *const r00C2 = new(mem_ctx) ir_variable(glsl_type::ivec2_type, "_d", ir_var_function_in);
   sig_parameters.push_tail(r00C2);
   ir_variable *const r00C3 = new(mem_ctx) ir_variable(glsl_type::bool_type, "negate", ir_var_auto);
   body.emit(r00C3);
   ir_expression *const r00C4 = less(swizzle_y(r00C1), body.constant(int(0)));
   ir_expression *const r00C5 = less(swizzle_y(r00C2), body.constant(int(0)));
   body.emit(assign(r00C3, nequal(r00C4, r00C5), 0x01));

   ir_variable *const r00C6 = body.make_temp(glsl_type::uvec2_type, "n");
   ir_expression *const r00C7 = expr(ir_unop_pack_int_2x32, r00C1);
   ir_expression *const r00C8 = expr(ir_unop_abs, r00C7);
   ir_expression *const r00C9 = expr(ir_unop_i642u64, r00C8);
   body.emit(assign(r00C6, expr(ir_unop_unpack_uint_2x32, r00C9), 0x03));

   ir_variable *const r00CA = body.make_temp(glsl_type::uvec2_type, "d");
   ir_expression *const r00CB = expr(ir_unop_pack_int_2x32, r00C2);
   ir_expression *const r00CC = expr(ir_unop_abs, r00CB);
   ir_expression *const r00CD = expr(ir_unop_i642u64, r00CC);
   body.emit(assign(r00CA, expr(ir_unop_unpack_uint_2x32, r00CD), 0x03));

   ir_variable *const r00CE = new(mem_ctx) ir_variable(glsl_type::int_type, "i", ir_var_auto);
   body.emit(r00CE);
   ir_variable *const r00CF = new(mem_ctx) ir_variable(glsl_type::uint64_t_type, "n64", ir_var_auto);
   body.emit(r00CF);
   ir_variable *const r00D0 = new(mem_ctx) ir_variable(glsl_type::int_type, "log2_denom", ir_var_auto);
   body.emit(r00D0);
   ir_variable *const r00D1 = new(mem_ctx) ir_variable(glsl_type::uvec2_type, "quot", ir_var_auto);
   body.emit(r00D1);
   body.emit(assign(r00D1, ir_constant::zero(mem_ctx, glsl_type::uvec2_type), 0x03));

   ir_expression *const r00D2 = expr(ir_unop_find_msb, swizzle_y(r00CA));
   body.emit(assign(r00D0, add(r00D2, body.constant(int(32))), 0x01));

   /* IF CONDITION */
   ir_expression *const r00D4 = equal(swizzle_y(r00CA), body.constant(0u));
   ir_expression *const r00D5 = gequal(swizzle_y(r00C6), swizzle_x(r00CA));
   ir_expression *const r00D6 = logic_and(r00D4, r00D5);
   ir_if *f00D3 = new(mem_ctx) ir_if(operand(r00D6).val);
   exec_list *const f00D3_parent_instructions = body.instructions;

      /* THEN INSTRUCTIONS */
      body.instructions = &f00D3->then_instructions;

      ir_variable *const r00D7 = new(mem_ctx) ir_variable(glsl_type::int_type, "i", ir_var_auto);
      body.emit(r00D7);
      ir_variable *const r00D8 = body.make_temp(glsl_type::int_type, "findMSB_retval");
      body.emit(assign(r00D8, expr(ir_unop_find_msb, swizzle_x(r00CA)), 0x01));

      body.emit(assign(r00D0, r00D8, 0x01));

      body.emit(assign(r00D7, body.constant(int(31)), 0x01));

      /* LOOP BEGIN */
      ir_loop *f00D9 = new(mem_ctx) ir_loop();
      exec_list *const f00D9_parent_instructions = body.instructions;

         body.instructions = &f00D9->body_instructions;

         /* IF CONDITION */
         ir_expression *const r00DB = less(r00D7, body.constant(int(1)));
         ir_if *f00DA = new(mem_ctx) ir_if(operand(r00DB).val);
         exec_list *const f00DA_parent_instructions = body.instructions;

            /* THEN INSTRUCTIONS */
            body.instructions = &f00DA->then_instructions;

            body.emit(new(mem_ctx) ir_loop_jump(ir_loop_jump::jump_break));


         body.instructions = f00DA_parent_instructions;
         body.emit(f00DA);

         /* END IF */

         /* IF CONDITION */
         ir_expression *const r00DD = sub(body.constant(int(31)), r00D7);
         ir_expression *const r00DE = lequal(r00D8, r00DD);
         ir_expression *const r00DF = lshift(swizzle_x(r00CA), r00D7);
         ir_expression *const r00E0 = lequal(r00DF, swizzle_y(r00C6));
         ir_expression *const r00E1 = logic_and(r00DE, r00E0);
         ir_if *f00DC = new(mem_ctx) ir_if(operand(r00E1).val);
         exec_list *const f00DC_parent_instructions = body.instructions;

            /* THEN INSTRUCTIONS */
            body.instructions = &f00DC->then_instructions;

            ir_expression *const r00E2 = lshift(swizzle_x(r00CA), r00D7);
            body.emit(assign(r00C6, sub(swizzle_y(r00C6), r00E2), 0x02));

            ir_expression *const r00E3 = lshift(body.constant(1u), r00D7);
            body.emit(assign(r00D1, bit_or(swizzle_y(r00D1), r00E3), 0x02));


         body.instructions = f00DC_parent_instructions;
         body.emit(f00DC);

         /* END IF */

         body.emit(assign(r00D7, add(r00D7, body.constant(int(-1))), 0x01));

      /* LOOP END */

      body.instructions = f00D9_parent_instructions;
      body.emit(f00D9);

      /* IF CONDITION */
      ir_expression *const r00E5 = lequal(swizzle_x(r00CA), swizzle_y(r00C6));
      ir_if *f00E4 = new(mem_ctx) ir_if(operand(r00E5).val);
      exec_list *const f00E4_parent_instructions = body.instructions;

         /* THEN INSTRUCTIONS */
         body.instructions = &f00E4->then_instructions;

         body.emit(assign(r00C6, sub(swizzle_y(r00C6), swizzle_x(r00CA)), 0x02));

         body.emit(assign(r00D1, bit_or(swizzle_y(r00D1), body.constant(1u)), 0x02));


      body.instructions = f00E4_parent_instructions;
      body.emit(f00E4);

      /* END IF */


   body.instructions = f00D3_parent_instructions;
   body.emit(f00D3);

   /* END IF */

   ir_variable *const r00E6 = body.make_temp(glsl_type::uint64_t_type, "packUint2x32_retval");
   body.emit(assign(r00E6, expr(ir_unop_pack_uint_2x32, r00CA), 0x01));

   body.emit(assign(r00CF, expr(ir_unop_pack_uint_2x32, r00C6), 0x01));

   body.emit(assign(r00CE, body.constant(int(31)), 0x01));

   /* LOOP BEGIN */
   ir_loop *f00E7 = new(mem_ctx) ir_loop();
   exec_list *const f00E7_parent_instructions = body.instructions;

      body.instructions = &f00E7->body_instructions;

      /* IF CONDITION */
      ir_expression *const r00E9 = less(r00CE, body.constant(int(1)));
      ir_if *f00E8 = new(mem_ctx) ir_if(operand(r00E9).val);
      exec_list *const f00E8_parent_instructions = body.instructions;

         /* THEN INSTRUCTIONS */
         body.instructions = &f00E8->then_instructions;

         body.emit(new(mem_ctx) ir_loop_jump(ir_loop_jump::jump_break));


      body.instructions = f00E8_parent_instructions;
      body.emit(f00E8);

      /* END IF */

      /* IF CONDITION */
      ir_expression *const r00EB = sub(body.constant(int(63)), r00CE);
      ir_expression *const r00EC = lequal(r00D0, r00EB);
      ir_expression *const r00ED = lshift(r00E6, r00CE);
      ir_expression *const r00EE = lequal(r00ED, r00CF);
      ir_expression *const r00EF = logic_and(r00EC, r00EE);
      ir_if *f00EA = new(mem_ctx) ir_if(operand(r00EF).val);
      exec_list *const f00EA_parent_instructions = body.instructions;

         /* THEN INSTRUCTIONS */
         body.instructions = &f00EA->then_instructions;

         ir_expression *const r00F0 = lshift(r00E6, r00CE);
         body.emit(assign(r00CF, sub(r00CF, r00F0), 0x01));

         ir_expression *const r00F1 = lshift(body.constant(1u), r00CE);
         body.emit(assign(r00D1, bit_or(swizzle_x(r00D1), r00F1), 0x01));


      body.instructions = f00EA_parent_instructions;
      body.emit(f00EA);

      /* END IF */

      body.emit(assign(r00CE, add(r00CE, body.constant(int(-1))), 0x01));

   /* LOOP END */

   body.instructions = f00E7_parent_instructions;
   body.emit(f00E7);

   /* IF CONDITION */
   ir_expression *const r00F3 = lequal(r00E6, r00CF);
   ir_if *f00F2 = new(mem_ctx) ir_if(operand(r00F3).val);
   exec_list *const f00F2_parent_instructions = body.instructions;

      /* THEN INSTRUCTIONS */
      body.instructions = &f00F2->then_instructions;

      body.emit(assign(r00CF, sub(r00CF, r00E6), 0x01));

      body.emit(assign(r00D1, bit_or(swizzle_x(r00D1), body.constant(1u)), 0x01));


   body.instructions = f00F2_parent_instructions;
   body.emit(f00F2);

   /* END IF */

   ir_variable *const r00F4 = body.make_temp(glsl_type::uvec4_type, "vec_ctor");
   body.emit(assign(r00F4, r00D1, 0x03));

   body.emit(assign(r00F4, expr(ir_unop_unpack_uint_2x32, r00CF), 0x0c));

   ir_variable *const r00F5 = body.make_temp(glsl_type::ivec2_type, "conditional_tmp");
   /* IF CONDITION */
   ir_if *f00F6 = new(mem_ctx) ir_if(operand(r00C3).val);
   exec_list *const f00F6_parent_instructions = body.instructions;

      /* THEN INSTRUCTIONS */
      body.instructions = &f00F6->then_instructions;

      ir_swizzle *const r00F7 = swizzle(r00F4, MAKE_SWIZZLE4(SWIZZLE_Z, SWIZZLE_W, SWIZZLE_X, SWIZZLE_X), 2);
      ir_expression *const r00F8 = expr(ir_unop_pack_uint_2x32, r00F7);
      ir_expression *const r00F9 = expr(ir_unop_u642i64, r00F8);
      ir_expression *const r00FA = neg(r00F9);
      body.emit(assign(r00F5, expr(ir_unop_unpack_int_2x32, r00FA), 0x03));


      /* ELSE INSTRUCTIONS */
      body.instructions = &f00F6->else_instructions;

      ir_swizzle *const r00FB = swizzle(r00F4, MAKE_SWIZZLE4(SWIZZLE_Z, SWIZZLE_W, SWIZZLE_X, SWIZZLE_X), 2);
      body.emit(assign(r00F5, expr(ir_unop_u2i, r00FB), 0x03));


   body.instructions = f00F6_parent_instructions;
   body.emit(f00F6);

   /* END IF */

   body.emit(ret(r00F5));

   sig->replace_parameters(&sig_parameters);
   return sig;
}
