def chname(prefix, name):
    RENAME_LAYERS = [
        ('encoder.layers', 'layers'),
        ('input_embedding', 'token_embedding',),
        ('ffn.layernorm_before_ffn', 'ln_ff'),
        ('encoder.output_layernorm', 'output_layernorm'),
        ('self_att.self_attention', 'attn'),
        ('self_att.layernorm_before_attention', 'ln_attn'),
        ('attention_out', 'attn_out'),
        ('ffn.ffn', 'ff'),
        ('w_in.w_1', 'w_gated'),
        ('w_in.w_0', 'w_in'),
        ('position_bias.relative_attention_bias', 'position_bias.weight')
        ]
    for (src, dst) in RENAME_LAYERS:
        name = name.replace(src, dst)
    return prefix + "." + name