// This file has been autogenerated by the TWN Accelerator export script - don't edit it!
#include "sequential_net.h"
#include "${n.get_net_header}"

% for l in n.layers:
#include "${l.get_layer_header}"
% endfor

##u8 buf_a[${n.max_buf_size}], buf_b[${n.max_buf_size}];
<%
  layers_name = n.name+"_layers"
%>
twn_layer_t * ${layers_name}[${n.n_layers}];

sequential_net_t ${n.name} = {
    .layers = ${layers_name},
    .n_layers = ${n.n_layers},
    .buf_size = 0,
    .buf_a = NULL,
    .buf_b = NULL
    };

sequential_net_t * ${n.get_net_fn}(void) {
% for i, l in enumerate(n.layers):
    ${layers_name}[${i}] = ${l.get_layer_fn}();
% endfor
% if n.init_dim is not None:
    // set the dimensions for each layer
    set_dimensions(&${n.name}, ${n.init_dim[0]}, ${n.init_dim[1]});
% endif
    return &${n.name};
}
