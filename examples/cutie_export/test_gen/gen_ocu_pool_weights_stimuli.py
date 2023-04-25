# ----------------------------------------------------------------------
#
# File: gen_ocu_pool_weights_stimuli.py
#
# Last edited: 24.07.2020        
# 
# Copyright (C) 2020, ETH Zurich and University of Bologna.
#
# Author: Moritz Scherer, ETH Zurich
#
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


#TODO: Write documentation 

import sys
import random
import numpy as np
import copy as copy
from collections import namedtuple
import collections
from io import StringIO
import argparse
from tqdm import tqdm
import json

filename = "ocu_pool_weights"

#dep = open('global_parameters.py', 'r').read()
#exec(dep)
from .global_parameters import *

# This file is part of the golden model representation for the tnn-accel project
# This file is used to generate stimuli for the testbench for the Output Channel Compute Unit: ocu_pool_weigths_staggered_load.sv
# Moritz Scherer, ETHZ 2019

### GLOBAL CONFIG ###

weight_lifetime = 1000
previous_save_enable = 0
pooling_fifowidth = int(k*k*ni)
threshold_fifowidth = 2 * int(np.ceil(np.log2(k*k*ni))+1)
threshold_fifousagewidth = int(np.ceil(np.log2(threshold_fifodepth)))

input_number = 0

### GLOBAL CONFIG ###

### LOCAL CONFIG ###

thresh_bitwidth = int(np.ceil(np.log2(ni*k*k)+1))
if(pooling_fifodepth < 2):
    pseudo_pooling_fifodepth = 2;
else:
    pseudo_pooling_fifodepth = pooling_fifodepth;
    pooling_fifousagewidth = int(np.ceil(np.log2(pseudo_pooling_fifodepth)))

### END LOCAL CONFIG ### 

### INTERFACE CONFIG ###

_output = namedtuple("_output", "out")
_input = namedtuple("_input", "acts weights thresh_pos thresh_neg pooling_fifo_flush pooling_fifo_testmode pooling_store_to_fifo threshold_fifo_flush threshold_fifo_testmode threshold_store_to_fifo threshold_pop alu_operand_sel multiplexer alu_op compute_enable weights_read_bank weights_save_bank weights_save_enable weights_test_enable weights_flush")
 
outputtypes = _output('ternary')
inputtypes = _input('ternary', 'ternary','signed','signed','unsigned','unsigned','unsigned','unsigned','unsigned','unsigned','unsigned','unsigned','unsigned','unsigned','unsigned','unsigned','unsigned','unsigned','unsigned','unsigned')

outputwidths = _output((2,(1)))
inputwidths = _input((2,(k,k,ni)), (2,(1,(ni/weight_stagger))), (thresh_bitwidth,(1)) , (thresh_bitwidth,(1)), (1,(1)),(1,(1)), (1,(1)),(1,(1)), (1,(1)),(1,(1)), (1,(1)),(2,(1)), (1,(1)), (1,(1)), (1,(1)), (1,(1)),(1,(1)), (1 , (weight_stagger,k,k)), (1 , (weight_stagger,k,k)), (1,weight_stagger)) 


### END INTERFACE CONFIG ###

### MODULE STATE ###

cycle_state = 0

pooling_fifo = collections.deque([], int(pooling_fifodepth))
threshold_fifo = collections.deque([], int(threshold_fifodepth))
last_value = 0;
current_sum = 0;
pooling_fifo_usage = 0;
pooling_fifo_usage_d = 0;

threshold_fifo_usage = 0
threshold_fifo_usage_d = 0

last_value_d = 0;
current_sum_d = 0;

thresh_pos = 0
thresh_neg = 0

weights_d = np.zeros((2,k,k,ni))
weights = np.zeros((2,k,k,ni))

### END MODULE STATE ###

### CONFIG MODULE STATE FUNCTIONS ###

def config_module_state():
    global pooling_fifo
    global outputwidths
    global inputwidths
    global thresh_bitwidth
    global pooling_fifousagewidth
    global weights
    global weights_d
    global pooling_fifowidth 
    global weight_stagger

    thresh_bitwidth = int(np.ceil(np.log2(ni*k*k)+1))

    pooling_fifowidth = k*k*ni

    if(fifodepth < 2):
        pseudo_pooling_fifodepth = 2;
    else:
        pseudo_pooling_fifodepth = pooling_fifodepth;

    threshold_fifowidth = 2 * int(np.ceil(np.log2(k*k*ni))+1)
    threshold_fifousagewidth = int(np.ceil(np.log2(threshold_fifodepth)))

    pooling_fifousagewidth = int(np.ceil(np.log2(pseudo_pooling_fifodepth)))
    #pooling_fifo = collections.deque([], pooling_fifodepth)

    weights_d = np.zeros(((2,k,k,ni)))
    weights = np.zeros(((2,k,k,ni)))

    outputwidths = _output((2,(1)))
    inputwidths = _input((2,(k,k,ni)), (2,(1,(ni/weight_stagger))), (thresh_bitwidth,(1)) , (thresh_bitwidth,(1)), (1,(1)),(1,(1)), (1,(1)),(1,(1)), (1,(1)),(1,(1)), (1,(1)),(2,(1)), (1,(1)), (1,(1)), (1,(1)), (1,(1)),(1,(1)), (1 , (weight_stagger,k,k)), (1 , (weight_stagger,k,k)), (1,weight_stagger)) 
    
### END CONFIG MODULE STATE FUNCTIONS ###

### LOCAL COMPUTATION FUNCTIONS ###            

def alu(pooling_store_to_fifo, alu_operand_sel, alu_op, compute_enable):
    
    global last_value_d
    global last_value
    global pooling_fifo_usage_d
    global pooling_fifo_usage
    global pooling_fifo
    global alu_operand_1

    pooling_fifo_usage_d = pooling_fifo_usage
    
    if(compute_enable == 1):
        if (alu_operand_sel == 1):
            alu_operand_1 = pooling_fifo.popleft()
            pooling_fifo_usage_d = pooling_fifo_usage_d - 1
        elif (alu_operand_sel == 2):
            alu_operand_1 = last_value;
        elif (alu_operand_sel == 0):
            alu_operand_1 = 0;
        else:
            alu_operand_1 = int(-2**thresh_bitwidth);
            
        alu_operand_2 = current_sum;
    else:
        alu_operand_1 = 0
        alu_operand_2 = 0
    
    if(alu_op == 1):
        alu_output = alu_operand_1 + alu_operand_2
    else:
        if(alu_operand_1 > alu_operand_2):
            alu_output = alu_operand_1
        else:
            alu_output = alu_operand_2
            
    if(pooling_store_to_fifo == 1):
        if(compute_enable):
            pooling_fifo.append(alu_output);
            pooling_fifo_usage_d = pooling_fifo_usage_d + 1

    last_value_d = alu_output;
    
    return alu_output;

def threshold_input(alu_output, multiplexer):

    global current_sum

    if(multiplexer == 0):
        return current_sum
    elif (multiplexer == 1):
        return alu_output

def threshold_decide(num, thresh_pos, thresh_neg):
    if(num>thresh_pos):
        return 1
    elif(num<thresh_neg):
        return -1
    else:
        return 0

def tern_mult(acts, _weights):
    total = np.multiply(acts,_weights)
    sum_tot = np.sum(total)
    return sum_tot

### END LOCAL COMPUTATION FUNCTIONS ###

### GLOBAL COMPUTATION FUNCTION ###

def tick(inputs):
    global pooling_fifo
    global threshold_fifo
    global current_sum
    global current_sum_d
    global last_value 
    global last_value_d
    global pooling_fifo_usage
    global threshold_fifo_usage
    global weights
    global weights_d
    global previous_save_enable

    global thresh_pos
    global thresh_neg
    
    # APPLICATION

    # COMPUTATION

    if(len(threshold_fifo) == 0):
        thresh_pos = 0
        thresh_neg = 0

    vprint("Thresholds:" + str(thresh_pos) + str(thresh_neg))
    vprint("FIFO length:" + str(len(threshold_fifo)))
    vprint("FIFO:" + str(threshold_fifo))
    
    sum_weights = 0
    if(inputs.weights_read_bank == 0):
        sum_weights = weights[0]
    else:
        sum_weights = weights[1]

    vprint(sum_weights)
        
    current_sum_d = tern_mult(inputs.acts, sum_weights)
    alu_out = alu(inputs.pooling_store_to_fifo, inputs.alu_operand_sel, inputs.alu_op, inputs.compute_enable)
    thresh_in = threshold_input(alu_out, inputs.multiplexer)
    
    out = threshold_decide(thresh_in, thresh_pos, thresh_neg)
    
    weights_d = weights

    if(pooling_fifo_usage == 0 and fifodepth == 0):
        pooling_fifo_empty = 1
        pooling_fifo_full = 1
    elif(pooling_fifo_usage == 0):
        pooling_fifo_empty = 1
        pooling_fifo_full = 0
    elif(pooling_fifo_usage == fifodepth):
        pooling_fifo_full = 1
        pooling_fifo_empty = 0
    else:
        pooling_fifo_empty = 0
        pooling_fifo_full = 0
        
    threshold_fifo_usage = len(threshold_fifo)
        
    if(threshold_fifo_usage == 0):
        threshold_fifo_empty = 1
        threshold_fifo_full = 0
    elif(threshold_fifo_usage == fifodepth):
        threshold_fifo_full = 1
        threshold_fifo_empty = 0
    else:
        threshold_fifo_empty = 0
        threshold_fifo_full = 0
        
    #ACQUISITION

    vprint("current_sum: "+str(current_sum))

    outputs = _output(out)

    #CLOCKEDGE

    for i in range(weight_stagger):
        for j in range(k):
            for m in range(k):
                if(int(inputs.weights_save_enable[i][j][m]) == 1):
                    for p in range(int((ni/weight_stagger))):
                        for q in range(0,2,1):
                            if(inputs.weights_save_bank == q):
                                weights_d[q][j][m][int(p+i*ni/weight_stagger)] = inputs.weights[p]

    for i in range(weight_stagger):
        if(inputs.weights_flush[i] == 1):
            for j in range(k):
                for m in range(k):
                    for p in range((int(ni/weight_stagger))):
                        for q in range(0,2,1):
                            if(inputs.weights_save_bank == q):
                                weights_d[q][j][m][int(p+i*ni/weight_stagger)] = 0
                            
    weights = weights_d

    if(inputs.threshold_store_to_fifo):
        if(len(threshold_fifo) == 0):
            thresh_pos = inputs.thresh_pos
            thresh_neg = inputs.thresh_neg
        threshold_fifo.append((inputs.thresh_pos, inputs.thresh_neg))

    if(inputs.threshold_pop):
        _tuple = threshold_fifo.popleft()
        if(len(threshold_fifo)>0):
            _tuple = threshold_fifo.popleft()
            thresh_pos = _tuple[0]
            thresh_neg = _tuple[1]
            threshold_fifo.appendleft(_tuple)
        else:
            thresh_pos = 0
            thresh_neg = 0

    if(inputs.compute_enable == 1):
        current_sum = current_sum_d
        last_value = last_value_d
        pooling_fifo_usage = pooling_fifo_usage_d

    if(inputs.threshold_fifo_flush == 1):
        threshold_fifo_usage = 0
        threshold_fifo.clear()
        
    if(inputs.pooling_fifo_flush == 1):
        pooling_fifo_usage = 0
        pooling_fifo.clear()

    return outputs;

### END GLOBAL COMPUTATION FUNCTION ###

### TEST CASE STIMULI GENERATION ###

def realistic_test_case():

    global pooling_fifo_usage
    global threshold_fifo
    global input_number

    acts = np.random.randint(-1,2,((k,k,ni)))

    #if(input_number%weight_lifetime < weight_stagger): # Don't excessively change weights
    weights = np.random.randint(-1,2,((int(ni/weight_stagger))))
    #else:
     #   weights = np.zeros(((ni/weight_stagger)))

    thresholds = np.random.randint(-(ni*k*k), (ni*k*k), 2)
    #thresholds = np.random.randint(0, 1, 2)
    thresh_pos = np.zeros(1)
    thresh_neg = np.zeros(1)

    if (thresholds[0] > thresholds[1]):
        thresh_pos = thresholds[0]
        thresh_neg = thresholds[1]
    else:
        thresh_pos = thresholds[1]
        thresh_neg = thresholds[0]

    pooling_fifo_flush = 0
    pooling_fifo_testmode = 0

    if (pooling_fifo_usage > 0):
        alu_operand_sel = np.random.randint(0, 4)    
    else: 
        alu_operand_sel = np.random.randint(2, 4)    

    if (pooling_fifo_usage < fifodepth):
        pooling_store_to_fifo = np.random.randint(0, 2)
    else:
        pooling_store_to_fifo = 0;
    
    threshold_fifo_flush = 0
    threshold_fifo_testmode = 0
    threshold_store_to_fifo = 0

    multiplexer = np.random.randint(0, 2)
       
    alu_op = np.random.randint(0, 2)

    alu_operand_sel = ((alu_op+1)%2)*alu_operand_sel

    threshold_pop = 0
    
    weights_save_enable = np.zeros((k*k*weight_stagger),dtype=int)

    if(input_number%weight_lifetime < k*k*weight_stagger): # Don't excessively change weights
        compute_enable = 0
        weights_save_bank = int(input_number/weight_lifetime) % 2
        if(input_number%weight_lifetime == 0):
            threshold_store_to_fifo = 1
            vprint("Threshold length:" + str(len(threshold_fifo)))
            if(len(threshold_fifo) > 0):
                threshold_pop = 1
                
                
        weights_save_enable[input_number%weight_lifetime] = 1
        
    else:
        weights_save_bank = (int(input_number/weight_lifetime)+1) % 2
        compute_enable = 1
        
    weights_save_enable = np.reshape(weights_save_enable, (weight_stagger,k,k))
    weights_read_bank = (weights_save_bank+1)%2
    weights_flush = np.random.randint(0,1,weight_stagger)
    weights_test_enable = np.zeros((weight_stagger,k,k),dtype=int)
    input_number = input_number+1

    weights_flush = np.random.randint(0,1,weight_stagger)
    
    retinput = _input(acts, weights, thresh_pos, thresh_neg, pooling_fifo_flush, pooling_fifo_testmode, pooling_store_to_fifo, threshold_fifo_flush, threshold_fifo_testmode, threshold_store_to_fifo, threshold_pop, alu_operand_sel,  multiplexer, alu_op, compute_enable, weights_read_bank, weights_save_bank, weights_save_enable, weights_test_enable, weights_flush)
    return retinput

def rand_test_case():

    global pooling_fifo_usage
    global threshold_fifo_usage

    acts = np.random.randint(-1,2,((k,k,ni)))
    weights = np.random.randint(-1,2,(((ni/weight_stagger))))

    thresholds = np.random.randint(-(ni*k*k), (ni*k*k), 2)
    thresh_pos = 0
    thresh_neg = 0

    
    threshold_fifo_flush = 0
    threshold_fifo_testmode = 0
    threshold_store_to_fifo = 0


    if (thresholds[0] > thresholds[1]):
        thresh_pos = thresholds[0]
        thresh_neg = thresholds[1]
    else:
        thresh_pos = thresholds[1]
        thresh_neg = thresholds[0]

    pooling_fifo_flush = 0
    pooling_fifo_testmode = 0
    
    if (pooling_fifo_usage > 0):
        alu_operand_sel = np.random.randint(0, 3)    
    else: 
        alu_operand_sel = np.random.randint(2, 3)    

    if (pooling_fifo_usage < fifodepth):
        pooling_store_to_fifo = np.random.randint(0, 2)
    else:
        pooling_store_to_fifo = 0;


    if (threshold_fifo_usage < threshold_fifodepth):
        threshold_store_to_fifo = np.random.randint(0, 2)
    else:
        threshold_store_to_fifo = 0;

        
    multiplexer = np.random.randint(0, 2)
    alu_op = np.random.randint(0, 2)

    # BUG: Switching Read banks in running mode might cause wrong results (Only in threshold decision, somehow?)
    # Weird. Realistic test case works fine
    
    weights_read_bank = np.random.randint(0, 2)
    weights_save_bank = (weights_read_bank+1)%2

    weights_save_enable = np.random.randint(0, 2, (weight_stagger,k,k))
    weights_flush = np.random.randint(0, 2)
    weights_test_enable = np.zeros((weight_stagger,k,k),dtype=int)

    threshold_save_enable = np.random.randint(0,2)
    
    compute_enable = np.random.randint(0,2)

    previous_read = weights_read_bank
    
    weights_flush = np.random.randint(0,2,weight_stagger)

    retinput = _input(acts, weights, thresh_pos, thresh_neg, pooling_fifo_flush, pooling_fifo_testmode, pooling_store_to_fifo, threshold_fifo_flush, threshold_fifo_testmode, threshold_store_to_fifo, threshold_pop, alu_operand_sel,  multiplexer, alu_op, compute_enable, weights_read_bank, weights_save_bank, weights_save_enable, weights_test_enable, weights_flush)
    return retinput

def no_fifo_test_case():

    acts = np.random.randint(-1,2,((k,k,ni)))
    weights = np.random.randint(-1,2,((k,k,(ni/weight_stagger))))

    thresholds = np.random.randint(-(imw*ni*k*k), (imw*ni*k*k), 2)

    thresh_pos = 0
    thresh_neg = 0

    if (thresholds[0] > thresholds[1]):
        thresh_pos = thresholds[0]
        thresh_neg = thresholds[1]
    else:
        thresh_pos = thresholds[1]
        thresh_neg = thresholds[0]
    
    threshold_fifo_flush = 0
    threshold_fifo_testmode = 0
    threshold_store_to_fifo = 0

    pooling_fifo_flush = 0
    pooling_fifo_testmode = 0
    
    alu_operand_sel = 0

    pooling_store_to_fifo = 0

    multiplexer = np.random.randint(0, 2)
    alu_op = np.random.randint(0, 2)

    weights_read_bank = np.random.randint(0, 2)
    weights_save_bank = np.random.randint(0, 2)

    weights_save_enable = np.random.randint(0, 2, (weight_stagger))
    weights_test_enable = np.zeros((weight_stagger),dtype=int)

    compute_enable = np.random.randint(0,2)

    weights_flush = np.random.randint(0,2,weight_stagger)

    threshold_save_enable = np.random.randint(0,2)

    retinput = _input(acts, weights, thresh_pos, thresh_neg, pooling_fifo_flush, pooling_fifo_testmode, pooling_store_to_fifo, threshold_fifo_flush, threshold_fifo_testmode, threshold_store_to_fifo, threshold_pop, alu_operand_sel,  multiplexer, alu_op, compute_enable, weights_read_bank, weights_save_bank, weights_save_enable, weights_test_enable, weights_flush)
    return retinput

def base_test_case():

    global cycle_state
    
    acts = np.random.randint(-1,2,((k,k,ni)))
    weights = np.random.randint(-1,2,((k,k,(ni/weight_stagger))))

    thresholds = np.random.randint(-(imw*ni*k*k), (imw*ni*k*k), 2)
    thresh_pos = 0
    thresh_neg = 0

    if (thresholds[0] > thresholds[1]):
        thresh_pos = thresholds[0]
        thresh_neg = thresholds[1]
    else:
        thresh_pos = thresholds[1]
        thresh_neg = thresholds[0]

    pooling_fifo_flush = 0
    pooling_fifo_testmode = 0

    pooling_store_to_fifo = 0


    threshold_fifo_flush = 0
    threshold_fifo_testmode = 0

    threshold_store_to_fifo = 0

    
    alu_operand_sel = 0
    
    multiplexer = 0
    alu_op = 0

    weights_read_bank = np.random.randint(0, 1, 1)
    weights_save_bank = np.random.randint(0, 1, 1)

    weights_test_enable = 0
    weights_flush = [0]*weight_stagger #np.random.randint(0,2,weight_stagger)
    
    compute_enable = 1

    weights_save_enable = [1]*weight_stagger
    threshold_save_enable = 1

    retinput = _input(acts, weights, thresh_pos, thresh_neg, pooling_fifo_flush, pooling_fifo_testmode, pooling_store_to_fifo, threshold_fifo_flush, threshold_fifo_testmode, threshold_store_to_fifo, threshold_pop, alu_operand_sel,  multiplexer, alu_op, compute_enable, weights_read_bank, weights_save_bank, weights_save_enable, weights_test_enable, weights_flush)
    
    return retinput

### END TEST CASE STIMULI GENERATION ###

### STIMULI GENERATION FUNCTION ###

def gen_stimuli(name_stimuli, name_exp, num_vectors):

    outputs = []
    inputs = []

    f = open(name_stimuli, 'w+')
    g = open(name_exp, 'w+')

    for i in tqdm(range(num_vectors)):
        curr_input = realistic_test_case()
        curr_output = tick(curr_input)

        inputs.append(curr_input)
        outputs.append(curr_output)

        vprint(curr_input)
        vprint(format_input(curr_input))

        jprint(j_input, curr_input)
        
        vprint(curr_output)
        vprint(format_output(curr_output))

        jprint(j_output, curr_output)

        f.write("%s \n" % format_input(curr_input))
        g.write("%s \n" % format_output(curr_output))

    f.close()
    g.close()

def parse_stimuli(name_stimuli, name_exp, num_vectors):

    outputs = []
    inputs = []

    tilebuffer = jload("tilebuffer_json_out.txt")
    weightmemorybank = jload("weightmemorybank_json_out.txt")

    weightmemoffset = 1
    tilebufferoffset = 2

    f = open(name_stimuli, 'w+')
    g = open(name_exp, 'w+')

    for i in tqdm(range(num_vectors)):
        steering = base_test_case()
        steering = steering._replace(acts=np.asarray((tilebuffer[i+tilebufferoffset])['acts_out'],dtype=int))
        steering = steering._replace(weights=np.asarray((weightmemorybank[i+weightmemoffset])['weights'],dtype=int))
        #steering = steering._replace(weights_save_enable=np.asarray(weightmemorybank[i+weightmemoffset]['ready'],dtype=int))

        curr_input = steering

        curr_output = tick(curr_input)
        
        vprint(curr_input)
        vprint(format_input(curr_input))
        
        jprint(j_input, curr_input)

        vprint(curr_output)
        vprint(format_output(curr_output))
        
        jprint(j_output, curr_output)
        
        f.write("%s \n" % format_input(curr_input))
        g.write("%s \n" % format_output(curr_output))

    f.close()
    g.close()

### END STIMULI GENERATION FUNCTION ###

### PROGRAM ENTRY POINT ###

if __name__=='__main__':

    parser = argparse.ArgumentParser(description="Stimuli generator for ocu_pool_weigths_staggered_load.sv")
    parser.add_argument('-v', '--verbose', metavar='verbosity', dest='verbosity',  type=str2bool, const=True, default=False, nargs='?', help='Enable command line output')
    parser.add_argument('-j', '--json', metavar='jsonOutputEnable', dest='json',  type=str2bool, const=True, default=True, nargs='?', help='Enable JSON output')
    parser.add_argument('-jIn', '--jsonInputFile', metavar='jsonInputFile', dest='jIn',  default=str(filename)+'_json_in.txt', help='Choose your own JSON input destination file')
    parser.add_argument('-jOut', '--jsonOutputFile', metavar='jsonOutputFile', dest='jOut',  default=str(filename)+'_json_out.txt', help='Choose your own JSON output destination file')
    parser.add_argument('-i', '--input', metavar='inputGeneration', dest='input',  type=str2bool, const=True, default=False, nargs='?', help='Set whether to read inputs from stimuli file or generate randomly. Default is false, meaning stimuli are generated, not read. Overrides -j Option')
    parser.add_argument('-s', '--stimuli', metavar='StimuliFile', dest='stimulifile',  default=str(filename)+'_stimuli.txt', help='Choose your own stimuli output file, default is ocu_pool_weights_stimuli.txt')
    parser.add_argument('-o', '--output', metavar='OutputFile', dest='outputfile', default=str(filename)+'_exp_responses.txt', help='Choose your own expected responses output file destination, default is ocu_pool_weights_exp_responses.txt')

    parser.add_argument('-ni', metavar='MaxInputChannels', dest='ni', type=int, default=ni, help='Set the N_I variable for generation\n')
    parser.add_argument('-no', metavar='MaxOutputChannels', dest='no', type=int, default=no, help='Set the N_O variable for generation\n')
    parser.add_argument('-imw', metavar='MaxImageWidth', dest='imw', type=int, default=imw, help='Set the IMW variable for generation\n')
    parser.add_argument('-imh', metavar='MaxImageHeight', dest='imh', type=int, default=imh, help='Set the IMH variable for generation\n')
    parser.add_argument('-k', metavar='MaxKernelSize', dest='k', type=int, default=k, help='Set the K variable for generation\n')
    parser.add_argument('-ws', metavar='WeightStagger', dest='ws', type=int, default=weight_stagger, help='Set the number of load rounds for saving weights\n')
    parser.add_argument('-fi', '--fifodepth', metavar='FIFODepth', dest='fifodepth', type=int, default=pooling_fifodepth, help='Set the POOLING_FIFODEPTH variable for generation\n')
    parser.add_argument('-al', '--averagelifetime', metavar='AvLifetime', dest='al', type=int, default=1000, help='Set the average lifetime of a layer for the realistic test case\n')
    parser.add_argument('-num', '--number-of-vectors', metavar='NumberOfVectors', dest='numvec', type=int, default=number_of_stimuli, help='Choose the number of generated stimuli\n')
    
    args=parser.parse_args()

    ni = args.ni
    no = args.no
    imw = args.imw
    imh = args.imh
    k = args.k
    fifodepth = args.fifodepth
    numvec = args.numvec
    weight_stagger = args.ws
    weight_lifetime = args.al

    jsonIn = args.jIn
    jsonOut = args.jOut

    config_module_state()

    if(args.json == True):
        j_input = open(jsonIn, 'w+')
        j_output = open(jsonOut, 'w+')
    else:
        j_input = open(jsonIn, 'r')
        j_output = open(jsonOut, 'r')

    if(args.input == False):
        gen_stimuli(args.stimulifile,args.outputfile,numvec)
    else:
        #parse_stimuli(args.stimulifile,args.outputfile,numvec)
        parse_stimuli(args.stimulifile,args.outputfile,numvec)

    j_input.close()
    j_output.close()
    
### END PROGRAM ENTRY POINT ###
