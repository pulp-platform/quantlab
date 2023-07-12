/* =====================================================================
 * Title:        cutie_driver.h
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

#ifndef __CUTIE_DRIVER_INCLUDE_GUARD
#define __CUTIE_DRIVER_INCLUDE_GUARD

#include "cutie_defines.h"
#include "reg_file.h"
#include "pulp.h"

void config_interrupt_CUTIE(uint32_t id, uint32_t lane);
void wait_for_evt_id(uint32_t id);

void turn_off_CUTIE();
void turn_on_CUTIE();
void clf_done_CUTIE();

void wait_cycles(uint32_t n_cycles);

void config_layers_CUTIE(uint8_t n_layers, uint32_t* layer_params, uint32_t* n_thresholds, int16_t* thresholds);
void write_weights_CUTIE(uint32_t n_weights, uint32_t* weights);
void write_acts_CUTIE(uint32_t n_acts, int32_t* acts);
uint32_t check_fp_resp_CUTIE(uint32_t n_responses, int32_t* responses);
uint32_t check_resp_CUTIE(uint32_t n_responses, uint8_t bank, int32_t* responses);

#endif
