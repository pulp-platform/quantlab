# ----------------------------------------------------------------------
#
# File: gen_LUCA_stimuli.py
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
from collections import namedtuple
import collections
from io import StringIO
import argparse
from tqdm import tqdm
import json
 
filename = "LUCA" # Local Uppermost Control Arbiter

#dep = open('global_parameters.py', 'r').read()
#exec(dep)
from .global_parameters import *

# This file is part of the golden model representation for the tnn-accel project
# This file is used to generate stimuli for the testbench for ...
#
# Moritz Scherer, ETHZ 2019

### INTERFACE CONFIG ###

# TB: testmode latch_new_layer k stride_width stride_height padding_type imagewidth imageheight ni readbank
# Weights: latch_new_layer (DIFFERENT ONE) k  (DIFFERENT ONE) ni  (DIFFERENT ONE) no  (DIFFERENT ONE) soft_reset toggle_banks
# OCU: latch_new_layer no
# WB: latch_new_layer no writebank

kbitwidth = int(np.ceil(np.log2(k)))

nibitwidth = int(np.maximum(np.ceil(np.log2(ni)),1))+1
nobitwidth = int(np.maximum(np.ceil(np.log2(no)),1))+1

imagewidthbitwidth = int(np.maximum(np.ceil(np.log2(imagewidth)),1))+1
imageheightbitwidth = int(np.maximum(np.ceil(np.log2(imageheight)),1))+1

numactmemsetsbitwidth = int(np.maximum(np.ceil(np.log2(numactmemsets)),1))

threshbitwidth = int(np.ceil(np.log2(ni*k*k)+1))

_layer = namedtuple("layer", "imagewidth imageheight k ni no stride_width stride_height padding_type pooling_enable pooling_pooling_type pooling_kernel pooling_padding_type skip_in skip_out")

_output = namedtuple("_outputs", "testmode compute_latch_new_layer compute_imagewidth compute_imageheight compute_k compute_ni compute_no stride_width stride_height padding_type pooling_enable pooling_pooling_type pooling_kernel pooling_padding_type skip_in skip_out readbank writebank weights_latch_new_layer weights_k weights_ni weights_no weights_soft_reset weights_toggle_banks fifo_pop compute_done")
_input = namedtuple("_inputs", "store_to_fifo testmode imagewidth imageheight k ni no stride_width stride_height padding_type pooling_enable pooling_pooling_type pooling_kernel pooling_padding_type skip_in skip_out compute_disable tilebuffer_done weightload_done")

# each output type is either ternary, signed or unsigned
outputtypes = _output("unsigned","unsigned","unsigned","unsigned","unsigned","unsigned","unsigned","unsigned","unsigned","unsigned","unsigned","unsigned","unsigned","unsigned","unsigned","unsigned","unsigned","unsigned","unsigned","unsigned","unsigned","unsigned","unsigned","unsigned","unsigned","unsigned")
inputtypes = _input("unsigned","unsigned","unsigned","unsigned","unsigned","unsigned","unsigned","unsigned","unsigned","unsigned","unsigned","unsigned","unsigned","unsigned","unsigned","unsigned","unsigned","unsigned","unsigned")

# each width is a tuple of size per element and a shape tuple 
outputwidths = _output((1,1),(1,1),(imagewidthbitwidth,1),(imageheightbitwidth,1),(kbitwidth,1),(nibitwidth,1),(nobitwidth,1),(kbitwidth,1),(kbitwidth,1),(1,1),(1,1),(1,1),(kbitwidth,1),(1,1),(1,1),(1,1),(numactmemsetsbitwidth,1),(numactmemsetsbitwidth,1),(1,pipelinedepth),(kbitwidth,1),(nibitwidth,1),(nobitwidth,1),(1,1),(1,pipelinedepth),(1,1),(1,1))
inputwidths = _input((1,1),(1,1),(imagewidthbitwidth,1),(imageheightbitwidth,1),(kbitwidth+1,1),(nibitwidth,1),(nobitwidth,1),(kbitwidth,1),(kbitwidth,1),(1,1),(1,1),(1,1),(kbitwidth,1),(1,1),(1,1),(1,1),(1,1),(1,1),(1,pipelinedepth))

number_of_stimuli = 100

### END INTERFACE CONFIG ###

### LOCAL STATE ###

current_layer_done_q = 0
readbank_q = 1
timer_started_q = 0
timer_q = 0
first_layer_run_q = 0
layer_running_q = 0

fifo_popped_q = 0
weightload_ready_q = 1

layernum = 0
cyclenum = 0

ocudelay = 1
computedelay = pipelinedepth - 1 + ocudelay
writebackdelay = computedelay + 1

layer_fifo = collections.deque([], layer_fifodepth)
current_layer = _layer(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
next_layer = _layer(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)


### END LOCAL STATE ###

### CONFIG MODULE STATE FUNCTIONS ###

def config_module_state():
    pass

### END CONFIG MODULE STATE FUNCTIONS ###

### GLOBAL COMPUTATION FUNCTION ###

def tick(inputs):

    #APPLICATION

    global current_layer_done_q
    global timer_started_q
    global timer_q
    global readbank_q
    global layer_fifo
    global current_layer
    global first_layer_run_q
    global layer_running_q
    global next_layer

    global fifo_popped_q
    global cyclenum

    global weightload_ready_q
    
    input_layer = _layer(inputs.imagewidth, inputs.imageheight, inputs.k, inputs.ni, inputs.no, inputs.stride_width, inputs.stride_height, inputs.padding_type, inputs.pooling_enable, inputs.pooling_pooling_type, inputs.pooling_kernel, inputs.pooling_padding_type, inputs.skip_in, inputs.skip_out)
    
    testmode = 0
    compute_latch_new_layer = 0
    compute_k = current_layer.k
    compute_ni = current_layer.ni
    compute_no = current_layer.no
    compute_imagewidth = current_layer.imagewidth
    compute_imageheight = current_layer.imageheight

    pooling_enable = current_layer.pooling_enable
    pooling_kernel = current_layer.pooling_kernel
    pooling_padding_type = current_layer.pooling_padding_type
    pooling_pooling_type = current_layer.pooling_pooling_type

    skip_in = current_layer.skip_in
    skip_out = current_layer.skip_out
    
    stride_width = current_layer.stride_width
    stride_height = current_layer.stride_height
    padding_type = current_layer.padding_type
    readbank_d = readbank_q
    writebank_d = (readbank_q+1)%2

    # TODO: Change these to peek from FIFO (Latency hiding, functionally equivalent)
    weights_latch_new_layer = np.zeros(pipelinedepth,dtype=int)
    weights_k = next_layer.k
    weights_no = next_layer.no
    weights_ni = next_layer.ni
    weights_soft_reset = 0
    weights_toggle_banks = np.zeros(pipelinedepth,dtype=int)

    if(len(layer_fifo) > 0):
        fifo_empty = 0
    else:
        fifo_empty = 1
        
    fifo_pop = 0
    compute_done = 0

    timer_started_d = timer_started_q
    timer_d = timer_q

    layer_running_d = layer_running_q
    weightload_ready_d = weightload_ready_q
    
    #Output computation
    # Next state calculation


    if (inputs.tilebuffer_done == 1 and inputs.weightload_done[0] == 1 and timer_started_q == 0 and layer_running_q == 1):
        timer_started_d = 1
    elif(timer_started_q == 1):
        timer_d = (timer_q + 1)%(writebackdelay)
        if(timer_d == 0):
            timer_started_d = 0
    
    if(inputs.compute_disable == 0):
        if(fifo_empty == 0 and current_layer_done_q == 1 and fifo_popped_q == 0):
            # Initial case
            fifo_pop = 1
        if(fifo_popped_q == 1):
            compute_latch_new_layer = 1
            layer_running_d = 1
            readbank_d = (readbank_q+1)%2
            for i in range(int(weights_no/(no/pipelinedepth))):
                weights_toggle_banks[i] = 1
            
        if(inputs.weightload_done[0] == 1 and weightload_ready_q == 1 and fifo_empty == 0):
            for i in range(int(weights_no/(no/pipelinedepth))):
                weights_latch_new_layer[i] = 1
            weightload_ready_d = 0
            
    if (fifo_empty == 1 and current_layer_done_q == 1 and layer_running_q == 1 and compute_latch_new_layer == 0):
        compute_done = 1
        layer_running_d = 0
        weightload_ready_d = 1
        
    if(fifo_pop == 1):
        weightload_ready_d = 1
            
    #ACQUISITION

    outputs = _output(testmode, compute_latch_new_layer, compute_imagewidth, compute_imageheight, compute_k, compute_ni, compute_no, stride_width, stride_height, padding_type, pooling_enable, pooling_pooling_type, pooling_kernel, pooling_padding_type, skip_in, skip_out, readbank_d, (readbank_d+1)%2, weights_latch_new_layer, weights_k, weights_no, weights_ni, weights_soft_reset, weights_toggle_banks, fifo_pop, compute_done)
    
    # CLOCKEDGE

    if(weightload_ready_q == 0 and inputs.tilebuffer_done == 1):
        current_layer_done_q = 1 
    
    layer_running_q = layer_running_d
    fifo_popped_q = fifo_pop
    weightload_ready_q = weightload_ready_d
    
    if(fifo_pop == 1 and fifo_empty == 0):
        current_layer = next_layer
        next_layer = layer_fifo.popleft()
        
    if(inputs.store_to_fifo == 1):
        if(fifo_empty == 1):
            next_layer = input_layer
        layer_fifo.append(input_layer)

    if(len(layer_fifo)==0):
        next_layer = _layer(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        
    _timer_started_q = timer_started_q
    
    if(timer_started_q == 1 and compute_latch_new_layer == 0):
        timer_q = timer_d
        if(timer_q == 0):
            timer_started_q = 0
            current_layer_done_q = 1
    elif(compute_latch_new_layer == 1):
        first_layer_run_q = 1
        current_layer_done_q = 0
        timer_started_q = 0
        timer_q = 0
    elif(_timer_started_q == 0):
        timer_started_q = timer_started_d
        timer_q = timer_d
        
    readbank_q = readbank_d

    #vprint("q: "+str(q))

    vprint("cyclenum: "+str(cyclenum))
    vprint("INPUTS")
    vprint("inputs.tilebuffer_done: "+str(inputs.tilebuffer_done))
    vprint("inputs.weightload_done: "+str(inputs.weightload_done))
    vprint("inputs.store_to_fifo: "+str(inputs.store_to_fifo))
    vprint("STATE")
    vprint("weightload_ready_q: "+str(weightload_ready_q))
    vprint("fifo_pop: "+str(fifo_pop))
    vprint("fifo_empty: "+str(fifo_empty))
    vprint("fifo_popped_q: "+str(fifo_popped_q))
    vprint("layer_running_q: "+str(layer_running_q))
    vprint("layer_running_d: "+str(layer_running_d))
    vprint("current_layer_done_q: "+str(current_layer_done_q))
    vprint("compute_latch_new_layer: "+str(compute_latch_new_layer))
    vprint("weights_latch_new_layer[0]: "+str(weights_latch_new_layer[0]))
    vprint('')
    vprint("timer_started_q: "+str(timer_started_q))
    vprint("timer_q: "+str(timer_q))
    vprint('')
    vprint("current_layer: "+str(current_layer))
    vprint("next_layer: "+str(next_layer))
    vprint('')
    
    return outputs;

### END GLOBAL COMPUTATION FUNCTION ###

### TEST CASE STIMULI GENERATION ###

def realistic_test_case():

    global cyclenum
    global layernum
    global init_q
    
    testmode = 0

    weightbufferload_done = np.ones(pipelinedepth,dtype=int)
    
    if(cyclenum%50<3):
        tilebuffer_done = 1
    else:
        tilebuffer_done = 0
        
    pooling_enable = 0
    pooling_pooling_type = 0
    pooling_padding_type = 0
    pooling_kernel = k
        
    stride_width = 1
    stride_height = 1
    padding_type = 0
    compute_disable = 0

    imagewidth = 32
    imageheight = 32

    thresholds = np.random.randint(-(k*k*ni),(k*k*ni),(2,no))
    thresh_pos = np.zeros(no,dtype=int)
    thresh_neg = np.zeros(no,dtype=int)
    
    for i in range(no):
        if(thresholds[0][i] > thresholds[1][i]):
            thresh_pos[i] = thresholds[0][i]
            thresh_neg[i] = thresholds[1][i]
        else:
            thresh_neg[i] = thresholds[0][i]
            thresh_pos[i] = thresholds[1][i]
            
    if(layernum < 3):
        store_to_fifo = 1
        layernum = layernum + 1
    else:
        store_to_fifo = 0

    layer_skip_in = 0
    layer_skip_out = 0
    
    cyclenum = cyclenum + 1

    retinput = _input(store_to_fifo, testmode, imagewidth, imageheight, k, ni, no, stride_width, stride_height, padding_type, pooling_enable, pooling_pooling_type, pooling_kernel, pooling_padding_type, layer_skip_in, layer_skip_out, compute_disable, tilebuffer_done, weightbufferload_done)
    
    return retinput

def rand_test_case():
    pass

### END TEST CASE STIMULI GENERATION ###

### STIMULI GENERATION FUNCTION ###

def gen_stimuli(name_stimuli, name_exp, num_vectors):

    outputs = []
    inputs = []

    f = open(name_stimuli, 'w+')
    g = open(name_exp, 'w+')
    j_input = open(jsonIn, 'w+')
    j_output = open(jsonOut, 'w+')


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
    j_input.close()
    j_output.close()

def parse_stimuli(name_stimuli, name_exp, num_vectors):

    outputs = []
    inputs = []

    f = open(name_stimuli, 'r')
    g = open(name_exp, 'w+')

    for i in tqdm(range(num_vectors)):
        curr_input = parse_input(f)
        curr_output = tick(curr_input)
        
        vprint(curr_input)
        vprint(format_input(curr_input))
        
        jprint(j_input, curr_input)

        vprint(curr_output)
        vprint(format_output(curr_output))
        
        jprint(j_output, curr_output)
        
        g.write("%s \n" % format_output(curr_output))

    f.close()
    g.close()

### END STIMULI GENERATION FUNCTION ###

### PROGRAM ENTRY POINT ###

if __name__=='__main__':

    parser = argparse.ArgumentParser(description="Stimuli generator for")
    parser.add_argument('-v', '--verbose', metavar='verbosity', dest='verbosity',  type=str2bool, const=True, default=False, nargs='?', help='Enable command line output')
    parser.add_argument('-j', '--json', metavar='jsonOutputEnable', dest='json',  type=str2bool, const=True, default=True, nargs='?', help='Enable JSON output')
    parser.add_argument('-jIn', '--jsonInputFile', metavar='jsonInputFile', dest='jIn',  default=str(filename)+'_json_in.txt', help='Choose your own JSON input destination file')
    parser.add_argument('-jOut', '--jsonOutputFile', metavar='jsonOutputFile', dest='jOut',  default=str(filename)+'_json_out.txt', help='Choose your own JSON output destination file')
    parser.add_argument('-i', '--input', metavar='inputGeneration', dest='input',  type=str2bool, const=True, default=False, nargs='?', help='Set whether to read inputs from stimuli file or generate randomly. Default is false, meaning stimuli are generated, not read')
    parser.add_argument('-s', '--stimuli', metavar='StimuliFile', dest='stimulifile',  default=str(filename)+'_stimuli.txt', help='Choose your own stimuli output file, default is ocu_pool_weights_stimuli.txt')
    parser.add_argument('-o', '--output', metavar='OutputFile', dest='outputfile', default=str(filename)+'_exp_responses.txt', help='Choose your own expected responses output file destination, default is ocu_pool_weights_exp_responses.txt')

    parser.add_argument('-ni', metavar='MaxInputChannels', dest='ni', type=int, default=ni, help='Set the N_I variable for generation\n')
    parser.add_argument('-no', metavar='MaxOutputChannels', dest='no', type=int, default=no, help='Set the N_O variable for generation\n')
    parser.add_argument('-imw', metavar='MaxImageWidth', dest='imw', type=int, default=imw, help='Set the IMW variable for generation\n')
    parser.add_argument('-imh', metavar='MaxImageHeight', dest='imh', type=int, default=imh, help='Set the IMH variable for generation\n')
    parser.add_argument('-k', metavar='MaxKernelSize', dest='k', type=int, default=k, help='Set the K variable for generation\n')
    parser.add_argument('-num', '--number-of-vectors', metavar='NumberOfVectors', dest='numvec', type=int, default=number_of_stimuli, help='Choose the number of generated stimuli, default is '+str(number_of_stimuli))

    args = parser.parse_args()

    ni = args.ni
    no = args.no
    imw = args.imw
    imh = args.imh
    k = args.k
    numvec = args.numvec

    jsonIn = args.jIn
    jsonOut = args.jOut

    if(args.json == True):
        j_input = open(jsonIn, 'w+')
        j_output = open(jsonOut, 'w+')
    else:
        j_input = open(jsonIn, 'r')
        j_output = open(jsonOut, 'r')

    config_module_state()

    if(args.input == False):
        gen_stimuli(args.stimulifile,args.outputfile,numvec)
    else:
        parse_stimuli(args.stimulifile,args.outputfile,numvec)

    j_input.close()
    j_output.close()


### END PROGRAM ENTRY POINT ###
