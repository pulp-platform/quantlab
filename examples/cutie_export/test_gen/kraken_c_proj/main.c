/* =====================================================================
 * Title:        main.c
 * Description:
 *
 * $Date:        12.04.2021
 *
 * ===================================================================== */
/*
 * Copyright (C) 2020 ETH Zurich and University of Bologna.
 *
 * Authors: Tim Fischer, ETH Zurich
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "pulp.h"
#include "stdio.h"
#include <stdbool.h>

#include "cutie_defines.h"
#include "cutie_driver.h"
#include "layer_params_intf.h"
#include "activations_intf.h"
#include "responses_intf.h"
#include "thresholds_intf.h"
#include "weights_intf.h"
#if CONFIG_IO_UART == 1
#include "kraken_padframe.h"
#endif

#define USE_L2_ACTS

int main() {


#if CONFIG_IO_UART == 1
#if __PLATFORM__ == ARCHI_PLATFORM_BOARD
#define UART_PAD (8)
#else
#define UART_PAD (0)
#endif
  kraken_padframe_aon_pad_gpiob_mux_set(UART_PAD, KRAKEN_PADFRAME_AON_PAD_GPIOB_group_UART0_port_TX);
#endif
    int32_t tot_faults = 0;

    // configure interrupts
    config_interrupt_CUTIE(CUTIE_INTERRUPT_ID, CUTIE_INTERRUPT_LANE);

    // Turn CUTIE off
    turn_off_CUTIE();

    // Config layers
    config_layers_CUTIE(cutieLayerParamsLen/4, cutieLayerParams, cutieThreshsLen, cutieThreshs);

    // Write weights
    write_weights_CUTIE(cutieWeightsLen, cutieWeights);

    printf("initialization finished\n");

    // Process multiple input frames
    for (int i = 0; i < cutieNumExecs; i++) {

        // Clear done flag
        clf_done_CUTIE();

        //Write activations
        write_acts_CUTIE(cutieActsLen, cutieActs + i * (cutieActsLen << 2));
        // Start CUTIE
        turn_on_CUTIE();

        // Wait for CUTIE to finish
        wait_for_evt_id(CUTIE_INTERRUPT_ID);

        // Disable CUTIE
        turn_off_CUTIE();

        printf("checking responses for exec %d\n", i);

        int32_t num_faults = 0;
        if(cutieUseFPoutput) {
            num_faults = check_fp_resp_CUTIE(cutieResponsesLen, cutieResponses + i * cutieResponsesLen);
        }
        else {
            num_faults = check_resp_CUTIE(cutieResponsesLen, (cutieLayerParamsLen/4)%2, cutieResponses);
        }

        tot_faults += num_faults;

        printf("%d Errors in exec %d\n", num_faults, i);
    }

    return -tot_faults;
}
