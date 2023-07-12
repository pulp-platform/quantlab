/* =====================================================================
 * Title:        cutie_driver.c
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

#include "cutie_driver.h"
#include <stdio.h>

void config_interrupt_CUTIE(uint32_t id, uint32_t lane) {
    soc_eu_fcEventMask_setEvent(id);
    // enable interrupts from SoC event unit in APB interrupt controller
    printf("Setting SoC event interrupt (IRQ %d, %d)\n", lane, id);
    hal_itc_enable_set(1<<lane);
}

void wait_for_evt_id(uint32_t id) {
    hal_itc_wait_for_interrupt();
    uint32_t trigger_event = hal_itc_fifo_pop();
    while(trigger_event != id) {
        printf("Interrupt received from unexpected event: %d\n", trigger_event);
        hal_itc_wait_for_interrupt();
        trigger_event = hal_itc_fifo_pop();
    }
    printf("Interrupt received from CUTIE: %d\n", trigger_event);
}

inline void turn_off_CUTIE() {
    pulp_write32(LAYER_PARAMS_CTRL2(0), 0x1);
}

inline void turn_on_CUTIE() {
    pulp_write32(LAYER_PARAMS_CTRL2(0), 0x0);
}

inline void clf_done_CUTIE() {
    pulp_write32(LAYER_PARAMS_CTRL3(0), 0x0);
}

inline void wait_cycles(uint32_t n_cycles) {
    for (int i = 0; i < n_cycles; i++) {
        asm volatile ("nop");
    }
}

void config_layers_CUTIE(uint8_t n_layers, uint32_t* layer_params, uint32_t* n_thresholds, int16_t* thresholds) {

    uint8_t current_layer;
    uint32_t thresh_writes = 0;
    int16_t thresh[2];

    for (int i = 0; i < n_layers<<2; i+=4) {

        current_layer = i>>2;

        // Layer config
        pulp_write32(LAYER_PARAMS_FEATURE_MAP(0), layer_params[i+0]);
        pulp_write32(LAYER_PARAMS_TCN(0)        , layer_params[i+1]);
        pulp_write32(LAYER_PARAMS_KERNEL(0)     , layer_params[i+2]);
        pulp_write32(LAYER_PARAMS_POOLING(0)    , layer_params[i+3]);

        // Set valid
        pulp_write32(LAYER_PARAMS_CTRL1(0), 0x1);
        wait_cycles(8);

        // Write thresholds
        for (int ii = 0; ii < 2*n_thresholds[current_layer]; ii+=2) {
            thresh[1] = thresholds[thresh_writes+ii];
            thresh[0] = thresholds[thresh_writes+ii+1];
            pulp_write32(LAYER_PARAMS_THRESHOLDS(0), *(uint32_t*) thresh);
            wait_cycles(8);
        }
        thresh_writes += 2*n_thresholds[current_layer];
        pulp_write32(LAYER_PARAMS_CTRL1(0), 0x0);
    }
}

void write_weights_CUTIE(uint32_t n_weights, uint32_t* weights) {

    uint32_t addr;
    for (int i=0; i<(n_weights); i+=4){
        addr = weights[i];
        for (int ii = 1; ii <=3; ii++) {
                pulp_write32(addr, weights[i+ii]);
        }
    }
}

void write_acts_CUTIE(uint32_t n_acts, int32_t* acts) {

    uint32_t addr;
    for (int i = 0; i < (n_acts); i++){
        addr = acts[i<<2];
        for (int ii = 1; ii <= 3; ii++) {
                pulp_write32(addr, acts[(i<<2)+ii]);
        }
    }
}

uint32_t check_fp_resp_CUTIE(uint32_t n_responses, int32_t* responses) {

    uint32_t num_faults = 0;
    uint32_t addr;
    int32_t data;

    for (int i = 0; i < n_responses; i+=2) {
        addr = LAYER_PARAMS_FP_OUT_0(0) + (i<<1);
        data = pulp_read32(addr);
        if ((data & LAYER_PARAMS_FP_OUT_0_FP_OUT_0_MASK) !=
            (responses[i] & LAYER_PARAMS_FP_OUT_0_FP_OUT_0_MASK)) {
            num_faults++;
            printf("Mismatch in response %d - expected %d, got %d\n", i, responses[i] & LAYER_PARAMS_FP_OUT_0_FP_OUT_0_MASK, data & LAYER_PARAMS_FP_OUT_0_FP_OUT_0_MASK);
        }
        if (i != n_responses-1) {
          if ((data >> LAYER_PARAMS_FP_OUT_0_FP_OUT_1_OFFSET) !=
              (responses[i+1] & LAYER_PARAMS_FP_OUT_0_FP_OUT_1_MASK)) {
            printf("Mismatch in response %d - expected %d, got %d\n", i, responses[i+1] & LAYER_PARAMS_FP_OUT_0_FP_OUT_1_MASK, data >> LAYER_PARAMS_FP_OUT_0_FP_OUT_1_OFFSET);
            num_faults++;
          }
        }
    }

    return num_faults;
}

uint32_t check_resp_CUTIE(uint32_t n_responses, uint8_t bank, int32_t* responses) {

    uint32_t num_faults = 0;
    uint32_t addr;

    for (int i=0; i<n_responses; i++){
        addr = responses[i<<2];
        addr += (bank<<15); // Choose right bank
        for (int blk=1; blk<=3; blk++) {
          uint32_t resp_act = pulp_read32(addr);
          if(resp_act != responses[(i<<2)+blk]){
            num_faults++;
            printf("el %d/block %d: expected 0x%x, got 0x%x\n", i, blk-1, responses[(i<<2)+blk], resp_act);
          }
        }
    }

    return num_faults;
}
