#ifndef ${n.twn_header_guard}
#define ${n.twn_header_guard}

#include "params.h"
% for layer in n.twn_layers:
<%
l = layer.layer_info
%>
u8 ${l.weights_varname}[${l.weight_buf_size}] __attribute__((aligned (16)));
u8 ${l.beta_varname}[${l.n_out_blk} * OUT_CHAN_BLOCK_SIZE];
u32 ${l.gamma_varname}[${l.n_out_blk} * OUT_CHAN_BLOCK_SIZE];

twn_layer_t ${l.layer_name} = {
    .relu = ${l.relu},
    .resid = false, // sequential nets only for now
    .bn = true,
    .beta_buf = ${l.beta_varname},
    .gamma_buf = ${l.gamma_varname},
    .n_in_blk = ${l.n_in_blk},
    .n_out_blk = ${l.n_out_blk},
    .line_width = ${layer.line_width},
    .n_lines = ${layer.n_lines},
    .dyn_bn_bufs = false,
    .dyn_layer = false,
    .K = ${l.K_text},
    .weight_buf = ${l.weights_varname},
    .linebuf_order = ${l.linebuf_order},
    .pool_type = ${l.pool_type},
    .exec_cycles = 0,
    .comp_cycles = 0
    };

% endfor

#endif