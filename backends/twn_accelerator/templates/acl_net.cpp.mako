<%! import os %>
#include "${os.path.basename(n.header_fn)}"
#include "${os.path.basename(n.twn_header_fn)}"
#include "sd_card_drv.h"
#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "twn_conv_function.h"
#include <iostream>

${n.name}Net::${n.name}Net(${"twn_config_t * twnc, dma_config_t * dmac" if len(n.twn_layers) > 0 else ""})
: ACLLinearNetBase(twnc, dmac)
{}

void ${n.name}Net::setup(void) {
    // boilerplate memory management setup
    setup_memory();
    // create layers
% for l in n.layers:
    ${l.name} = std::make_unique<${l.qualified_type}>(${l.constructor_args});
##    layers.push_back(${l.name});
##    ${l.name} = std::make_unique<${l.qualified_type}>(${l.constructor_args});
% endfor
##    // create tensors
##    % for t in n.all_tensors:
##    ${t.name} = std::make_shared<Tensor>();
##    % endfor
    // setup tensor shapes
% for t in n.all_tensors:
    const TensorShape ${t.name}_shape(${t.shape_str});
% endfor
    // initialize tensors
% for t in n.all_tensors:
    ${t.name}.allocator()->init(TensorInfo(${t.name}_shape, 1, ${t.acl_datatype}, layout), ${t.alignment});
%   if t.is_fixedp:
    ${t.name}.info()->set_quantization_info(QuantizationInfo(${t.step_size}, 0.0));
%   endif
    %endfor
    // read parameters from SD
    read_params();
    // Configure layers
% for i, l in  enumerate(n.layers):
  % if l.configure_args:
##    layers[${i}]->configure(${l.configure_args});
    ${l.name}->configure(${l.configure_args});
  % endif
% endfor
<%
   mem_grp = 0
%>
% for t in n.managed_tensors:
    memory_group${mem_grp}->manage(&${t.name});
    ${t.name}.allocator()->allocate();
<%
  mem_grp = (mem_grp + 1) % 2
%>
% endfor
    // populate memory managers - no idea what this does.
    mm_layers->populate(allocator, 1);
    mm_transitions->populate(allocator, 2);
}

void ${n.name}Net::read_params(void) {
    // fill data arrays
% for t in n.parameters:
    sd_file_to_buf((char *)"${t.out_file}", (void *)${t.name}_arr,
                   sizeof(${t.c_type}) * ${t.tot_size});
% endfor
    // import the data arrays into the parameter tensors
% for t in n.parameters:
    ${t.name}.allocator()->import_memory((void *)${t.name}_arr);
% endfor
    // read TWN layer weight data
% for l in n.twn_layers:
    sd_file_to_buf((char *)"${l.weights_filename}", (void *)${l.layer_info.layer_name}.weight_buf, ${l.layer_info.weight_buf_size});
    sd_file_to_buf((char *)"${l.gamma_filename}", (void *)${l.layer_info.layer_name}.gamma_buf, ${l.layer_info.n_out_ch * 4});
    sd_file_to_buf((char *)"${l.beta_filename}", (void *)${l.layer_info.layer_name}.beta_buf, ${l.layer_info.n_out_ch});
% endfor
    Xil_DCacheInvalidate();
}


void * ${n.name}Net::get_dst_buf(void) {
	ARM_COMPUTE_ERROR_ON(!params_done);
	ARM_COMPUTE_ERROR_ON(!memory_done);
	ARM_COMPUTE_ERROR_ON(!memory_acquired);
	return dst.buffer();
}

void ${n.name}Net::set_input(void * input_data) {
  src.allocator()->import_memory(input_data);
}

void ${n.name}Net::run(void) {
  ARM_COMPUTE_ERROR_ON(!params_done);
  ARM_COMPUTE_ERROR_ON(!memory_done);
  ARM_COMPUTE_ERROR_ON(!memory_acquired);
% for l in n.layers:
    % if l.__class__.__name__ == "ACLDequantLayer":
    // Need to invalidate buffer memory from cache because this data was written by the accelerator and needs to be read by core
    Xil_DCacheInvalidateRange((INTPTR)${l.inputs[0].name}.buffer(), ${l.inputs[0].tot_size}*sizeof(${l.inputs[0].c_type}));
    % endif
    ${l.name}->run();
    % if l.__class__.__name__ in ["ACLQuantLayer", "ACLCastLayer"]:
    // Need to flush buffer memory region from cache because this data was written by core and needs to be read by accelerator
    Xil_DCacheFlushRange((INTPTR)${l.outputs[0].name}.buffer(), ${l.outputs[0].tot_size}*sizeof(${l.outputs[0].c_type}));
    % endif
% endfor
}
