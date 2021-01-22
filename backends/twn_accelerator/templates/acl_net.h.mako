#ifndef ${n.header_guard}
#define ${n.header_guard}

#include "acl_net_base.h"
#include "twn_conv_function.h"

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/Allocator.h"
#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "arm_compute/runtime/Tensor.h"
#include <memory>


using namespace arm_compute;

class ${n.name}Net
: public ACLLinearNetBase
{
public:
    ${n.name}Net(twn_config_t * twnc, dma_config_t * dmac);
    void setup(void);
    void read_params(void);
    void * get_dst_buf(void);
    void set_input(void * input_data);
    void run(void);
private:
// layers
% for l in n.layers:
    ##std::unique_ptr<${l.cpp_namespace}::${l.acl_type}> ${l.name};
    std::unique_ptr<${l.qualified_type}> ${l.name};
% endfor
// parameter tensors and arrays
% for t in n.parameters:
    Tensor ${t.name};
    ${t.c_type} ${t.name}_arr[${t.tot_size}];
% endfor
// intermediate tensors
% for t in n.data_tensors:
   Tensor ${t.name};
% endfor
    DataLayout layout = DataLayout::NHWC;
};

#endif
