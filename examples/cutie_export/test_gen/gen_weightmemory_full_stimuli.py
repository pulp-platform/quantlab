# ----------------------------------------------------------------------
#
# File: gen_weightmemory_full_stimuli.py
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
from io import StringIO
import argparse
from tqdm import tqdm
import json

#dep = open('global_parameters.py', 'r').read()
#exec(dep)
from .global_parameters import *

filename = "weightmemory_full"

# This file is part of the golden model representation for the tnn-accel project
# This file is used to generate stimuli for the testbench for ...
#
# Moritz Scherer, ETHZ 2019

### GLOBAL CONFIG ###

codebook = {}

### END GLOBAL CONFIG ###

### LOCAL CONFIG ###

numbanks = 1

totnumtrits = imagewidth*imageheight*ni
tritsperbank = int(np.ceil(totnumtrits/numbanks))

effectivetritsperword = int(ni/weight_stagger)
physicaltritsperword = int(np.ceil(effectivetritsperword/5))*5
physicalbitsperword = int(physicaltritsperword / 5 * 8)
excessbits = (physicaltritsperword - effectivetritsperword)*2
effectivewordwidth = physicalbitsperword - excessbits
numdecoders = int(physicalbitsperword / 8)

bankdepth = weightmemorybankdepth

fulladdresswidth = int(np.ceil(np.log2(bankdepth)))

bankaddressdepth = int(np.ceil(np.log2(bankdepth)))

leftshiftbitwidth = int(np.ceil(np.log2(numbanks)))
splitbitwidth = int(np.ceil(np.log2(weight_stagger)))+1

weightmem = np.empty((bankdepth), dtype='object')

prev_addr = np.zeros(numbanks)
prev_trits = 2*np.ones((numbanks, int(ni/weight_stagger)),dtype=int)
prev_ready = np.zeros(numbanks, dtype=int)
prev_read_bank = numactmemsets

prev_left_shift = 0
prev_scatter_coefficient = 0

prev_command_source = 0 # 0: external 1: internal
prev_external_we = 1

### END LOCAL CONFIG ###

### INTERFACE CONFIG ###

_output = namedtuple("_outputs", "ready rw_collision weights external_weight external_valid")
_input = namedtuple("_inputs", "external_we external_req external_addr external_wdata read_enable read_addr write_enable write_addr wdata")

# each output type is either ternary, signed or unsigned
outputtypes = _output("unsigned","unsigned","ternary","unsigned","unsigned")
inputtypes= _input("unsigned","unsigned","unsigned","unsigned","unsigned","unsigned","unsigned","unsigned","unsigned")

# each width is a tuple of size per element and a shape tuple 
outputwidths = _output((1,numbanks), (1,numbanks), ((2,(numbanks,int(ni/weight_stagger)) )),(1,physicalbitsperword),(1,1))
inputwidths = _input((1,1), (1,1), (fulladdresswidth,1), (1,(physicalbitsperword)),(1,numbanks), (bankaddressdepth,numbanks), (1,numbanks), (bankaddressdepth,numbanks),  (1,(numbanks,numdecoders,8))) 

number_of_stimuli = 100

### END INTERFACE CONFIG ###

### CONFIG MODULE STATE FUNCTIONS ###

def config_module_state():

    global bankaddressdepth
    global effectivetritsperword
    global physicaltritsperword
    global physicalbitsperword
    global excessbits
    global effectivewordwidth
    global weightmem
    global inputwidths
    global outputwidths
    global numdecoders
    global prev_addr
    global totnumtrits
    global tritsperbank 
    global bankdepth
    global bankaddressdepth
    global leftshiftbitwidth
    global numbanks
    global prev_left_shift
    global prev_scatter_coefficient

    effectivetritsperword = int(ni/weight_stagger)
    physicaltritsperword = int(np.ceil(effectivetritsperword/5))*5
    physicalbitsperword = int(physicaltritsperword / 5 * 8)
    excessbits = (physicaltritsperword - effectivetritsperword)*2
    effectivewordwidth = physicalbitsperword - excessbits
    numdecoders = int(physicalbitsperword / 8)

    numbanks = 1

    totnumtrits = imagewidth*imageheight*ni
    tritsperbank = int(np.ceil(totnumtrits/numbanks))
    bankdepth = 1024

    bankaddressdepth = int(np.ceil(np.log2(bankdepth)))

    leftshiftbitwidth = int(np.ceil(np.log2(numbanks)))
    splitbitwidth = int(np.ceil(np.log2(weight_stagger)))+1

#     vprint('effectivetritsperword: ' + str(effectivetritsperword))
#     vprint('physicaltritsperword: ' + str(physicaltritsperword))
#     vprint('physicalbitsperword: ' + str(physicalbitsperword))
#     vprint('excessbits: ' + str(excessbits))
#     vprint('effectivewordwidth: ' + str(effectivewordwidth))
#     vprint('numdecoders: ' + str(numdecoders))
#     vprint('numbanks: ' + str(numbanks))
#     vprint('totnumtrits: ' + str(totnumtrits))
#     vprint('tritsperbank: ' + str(tritsperbank))
#     vprint('bankdepth: ' + str(bankdepth))
#     vprint('bankaddressdepth: ' + str(bankaddressdepth))
#     vprint('leftshiftbitwidth: ' + str(leftshiftbitwidth))

    weightmem = np.empty((bankdepth), dtype='object')

    prev_addr = np.zeros(numbanks)
    prev_trits = 2*np.ones((numbanks, int(ni/weight_stagger)),dtype=int)
    prev_ready = np.zeros(numbanks, dtype=int)

    outputwidths = _output((1,numbanks), (1,numbanks), ((2,(numbanks,int(ni/weight_stagger)) )),(1,physicalbitsperword),(1,1))
    inputwidths = _input((1,1), (1,1), (fulladdresswidth,1), (1,(physicalbitsperword)),(1,numbanks), (bankaddressdepth,numbanks), (1,numbanks), (bankaddressdepth,numbanks),  (1,(numbanks,numdecoders,8))) 

### END CONFIG MODULE STATE FUNCTIONS ###

def tick(inputs):  
    global prev_addr
    global prev_trits
    global prev_ready
    global prev_read_bank
    
    global prev_left_shift
    global prev_scatter_coefficient

    global prev_command_source
    global prev_external_we

    #APPLICATION
    
    # Output computation

    _ready = []
    _collisions = []
    _strings = []

    _read_enable = []


    if(inputs.external_req == 1):

        command_source = 1
                
        bank = inputs.external_addr%numbanks
        addr = int(inputs.external_addr/numbanks)
        
        read_enable = np.zeros(numbanks, dtype=int)
        read_enable[bank] = (inputs.external_we+1)%2
        write_enable = np.zeros(numbanks, dtype=int)
        write_enable[bank] = inputs.external_we

        read_addr = np.zeros(numbanks, dtype=int)
        write_addr = np.zeros(numbanks, dtype=int)
        read_addr[bank] = addr
        write_addr[bank] = addr

        wdata = np.zeros((numbanks,numdecoders,8),dtype=int)
        wdata[bank] = inputs.external_wdata

    else:
        command_source = 0
        
        read_enable = inputs.read_enable
        write_enable = inputs.write_enable

        read_addr = inputs.read_addr
        write_addr = inputs.write_addr

        wdata = inputs.wdata

    
    ret_weights_encoded = '0'*physicalbitsperword

    for n in range(numbanks):

        codes = []
        content = []
            
        if(read_enable == 1 and write_enable == 1):
            rw_collision_ = 1;           
            read_enable_ = 0;
            write_enable_ = 1;
        else:
            read_enable_ = read_enable;
            write_enable_ = write_enable;
            rw_collision_ = 0;
            
        if (prev_ready == 1):
            weights_encoded = weightmem[prev_addr]
            ret_weights_encoded = str(*weights_encoded)
            _weights_encoded = str(*weights_encoded)
            string = ''

            for i in range(0,len(_weights_encoded),8):
                string = _weights_encoded[i:i+8]
                codes.append(string)
                #print("Enc:")
                #print(string)
                string =''

                
            for i in codes:
                content.append(decode(i))
                
            string = ''
            for i in content:
                string += i
                
            string = string[:effectivetritsperword*2]

        else:
            string = ''
            for i in range(effectivetritsperword):
                string += 'XX'

        if(write_enable_ == 1):
            st = ''
            for j in range(numdecoders):
                for i in range(8):
                    st += str(wdata[n][j][i])
                    
            weightmem[write_addr[n]] = st
        
        ready_ = ~rw_collision_ & read_enable_

        _read_enable.append(read_enable_)
        _ready = ready_
        _collisions.append(rw_collision_)
        _strings.append(string)

    # Next state calculation
    
    ___strings = []
    __strings = (np.asarray(_strings)).reshape(numbanks,-1)
    for i in range(__strings.shape[0]):
        ___strings.append(__strings[(i+numbanks)%numbanks])

    array = []
    for string in ___strings:
        for x in range(0,len(*string),2):
            array.append(parse_ternary(string[0][x:x+2]))
    trits = np.asarray(array)
    trits = trits.reshape(-1)

    prev_addr = read_addr

    #ACQUISITION
    
    encoded = np.empty(physicalbitsperword,dtype=int)
    
    for i in range(len(str(ret_weights_encoded)[:physicalbitsperword])):
        if (str(ret_weights_encoded)[:physicalbitsperword][i] == '1'):
            encoded[i] = 1
        else:
            encoded[i] = 0        
    
    external_weights = encoded
    
    if(prev_command_source == 1):
        external_valid = (prev_external_we+1)%2
        rw_collisions = np.ones(numbanks,dtype=int)
        valid = np.zeros(numbanks,dtype=int)
    else:
        external_valid = 0
        rw_collisions = np.asarray(_collisions)
        valid = np.asarray(prev_ready)

            
    outputs = _output(valid, rw_collisions, trits, external_weights, external_valid)
    #outputs = _output(np.asarray(prev_ready), np.asarray(_collisions), trits)

    prev_ready = _ready
    prev_trits = trits
    prev_command_source = command_source
    prev_external_we = inputs.external_we

    # CLOCKEDGE

    return outputs;

### END GLOBAL COMPUTATION FUNCTION ###

### TEST CASE STIMULI GENERATION ###

def realistic_test_case(): # Realistic in terms of average activity - N_O trits written, K*N_I trits read per cycle

    read_enable = np.ones(numbanks,dtype=int)

    write_enable = np.zeros(int(numbanks),dtype=int) # At most numbanks/k banks need to be written to
    
    for i in range(int(numbanks/k)):
        write_enable[i] = 1

    wdata = np.random.randint(0,2,(numbanks,numdecoders,8)) # Random data, maybe not so realistic

    write_addr = np.random.randint(0,bankdepth,numbanks) # Doesnt matter so much for anything, SRAM is random access
    read_addr = np.random.randint(0,bankdepth) * np.ones(numbanks,dtype=int) # Data will probably be read contiguously

    retinput = _input(read_enable,  read_addr, write_enable,  write_addr,  wdata)
    return retinput
    

def rand_test_case():

    external_we = 0
    external_req = 0
    external_addr = 0
    external_wdata = 0
        
    read_enable = np.random.randint(0,2,numbanks)

    write_enable = np.random.randint(0,2,numbanks)
    wdata = np.random.randint(0,2,(numbanks,numdecoders,8))
    write_addr = np.random.randint(0,bankdepth,numbanks)
    read_addr = np.random.randint(0,bankdepth,numbanks)
    
    retinput = _input(external_we, external_req, external_addr, external_wdata, read_enable, read_addr, write_enable, write_addr,  wdata)    
    return retinput


def write_mem_rand(addr,bank,membank):

    external_we = 0
    external_req = 0
    external_addr = 0
    external_wdata = 0
    
    read_enable = np.zeros(numbanks, dtype=int)

    write_enable = np.zeros(numbanks, dtype=int)
    write_enable[bank] = 1

    wdata = np.random.randint(0,2,(numbanks,numdecoders,8))
    write_addr = np.zeros(numbanks, dtype=int)
    write_addr[bank] = addr
    read_addr = 0*write_addr

    leftshift = 0
    scatter_coefficient = 0

    retinput = _input(external_we, external_req, external_addr, external_wdata, read_enable, read_addr, write_enable, write_addr,  wdata)    
    return retinput

def write_mem_data(_addr,data):

    membank = int(_addr/(numbanks*bankdepth))
    __addr = _addr - (membank*numbanks*bankdepth)
    
    addr = int(__addr/(k*weight_stagger))
    bank = __addr%(k*weight_stagger)
            
    read_enable = np.zeros(numbanks, dtype=int)
    read_enable_bank_set = (membank+1)%numactmemsets

    write_enable = np.zeros(numbanks, dtype=int)
    write_enable[bank] = 1

    write_enable_bank_set = membank
    wdata = data
    write_addr = np.zeros(numbanks, dtype=int)
    write_addr[bank] = addr
    read_addr = 0*write_addr

    leftshift = 0
    scatter_coefficient = 0

    retinput = _input(read_enable,  read_addr, write_enable,  write_addr,  wdata)
    return retinput

def write_mem_ones(addr,bank,membank):

    read_enable = np.zeros(numbanks, dtype=int)
    read_enable_bank_set = (membank+1)%numactmemsets

    write_enable = np.zeros(numbanks, dtype=int)
    write_enable[bank] = 1

    write_enable_bank_set = membank
    wdata = np.random.randint(1,2,(numbanks,numdecoders,8))
    write_addr = np.zeros(numbanks, dtype=int)
    write_addr[bank] = addr
    read_addr = 0*write_addr

    leftshift = 0
    scatter_coefficient = 0

    retinput = _input(read_enable,  read_addr, write_enable,  write_addr,  wdata)
    return retinput

def write_mem_zero(addr,membank):

    read_enable = np.zeros(numbanks, dtype=int)
    read_enable_bank_set = membank
    write_enable = np.ones(numbanks, dtype=int)
    write_enable_bank_set = membank
    wdata = np.random.randint(0,1,(numbanks,numdecoders,8))
    write_addr = addr
    read_addr = 0*write_addr

    leftshift = 0
    scatter_coefficient = 0

    retinput = _input(read_enable,  read_addr, write_enable,  write_addr,  wdata)
    return retinput

### END TEST CASE STIMULI GENERATION ###

### STIMULI GENERATION FUNCTION ###

def zero_initialize():
    for m in range(numactmemsets):
        for n in range(numbanks):
            for i in range(bankdepth):
                curr_input = write_mem_rand(i, n, m)
                curr_output = tick(curr_input)


def gen_stimuli(name_stimuli, name_exp, num_vectors):

    outputs = []
    inputs = []

    f = open(name_stimuli, 'w+')
    g = open(name_exp, 'w+')
    if(args.json == True):
        j_input = open(jsonIn, 'w+')
        j_output = open(jsonOut, 'w+')
    else:
        j_input = open(jsonIn, 'r')
        j_output = open(jsonOut, 'r')

    for n in range(numbanks):
        for i in range(bankdepth):
            curr_input = write_mem_rand(i, n, 0)
            curr_output = tick(curr_input)
            
            inputs.append(curr_input)
            outputs.append(curr_output)
            
            vprint(curr_input)
            vprint(format_input(curr_input))
        
            #jprint(j_input, curr_input)
            
            vprint(curr_output)
            vprint(format_output(curr_output))
            
            f.write("%s \n" % format_input(curr_input))
            g.write("%s \n" % format_output(curr_output))
                
    for i in tqdm(range(num_vectors)):
        curr_input = rand_test_case()
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

def entry_code():
    
    global args
    
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
    parser.add_argument('-ws', metavar='WeightStagger', dest='ws', type=int, default=weight_stagger, help='Set the weight stagger coefficient in the bank\n')
    parser.add_argument('-bd', metavar='BankDepth', dest='bd', type=int, default=bankdepth, help='Set the number of words in the bank\n')
    parser.add_argument('-num', '--number-of-vectors', metavar='NumberOfVectors', dest='numvec', type=int, default=number_of_stimuli, help='Choose the number of generated stimuli, default is '+str(number_of_stimuli))

    args = parser.parse_args()
    
    

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
    parser.add_argument('-ws', metavar='WeightStagger', dest='ws', type=int, default=weight_stagger, help='Set the weight stagger coefficient in the bank\n')
    parser.add_argument('-bd', metavar='BankDepth', dest='bd', type=int, default=bankdepth, help='Set the number of words in the bank\n')
    parser.add_argument('-num', '--number-of-vectors', metavar='NumberOfVectors', dest='numvec', type=int, default=number_of_stimuli, help='Choose the number of generated stimuli, default is '+str(number_of_stimuli))

    args = parser.parse_args()

    ni = args.ni
    no = args.no
    imw = args.imw
    imh = args.imh
    k = args.k
    numvec = args.numvec
    bankdepth = args.bd
    weight_stagger = args.ws

    jsonIn = args.jIn
    jsonOut = args.jOut

    config_module_state()
    gen_codebook()

    if(args.json == True):
        j_input = open(jsonIn, 'w+')
        j_output = open(jsonOut, 'w+')
    else:
        j_input = open(jsonIn, 'r')
        j_output = open(jsonOut, 'r')

    if(args.input == False):
        gen_stimuli(args.stimulifile,args.outputfile,numvec)
    else:
        parse_stimuli(args.stimulifile,args.outputfile,numvec)

    j_input.close()
    j_output.close()


### END PROGRAM ENTRY POINT ###
