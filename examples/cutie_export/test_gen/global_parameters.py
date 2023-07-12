#TODO: Write documentation 

from pathlib import Path
import argparse
import json
import numpy as np 

ni = 96 # Maximum number of input channels
no = 96 # Maximum number of output channels
no_classes = 48 # Maximum number of output classes

imagewidth = 64 # Image width for SRAM memory in activationmemory
imageheight = 64 # Image height for SRAM memory in activationmemory
tcn_width = 24
# imagewidth = 32 # Image width for SRAM memory in activationmemory
# imageheight = 32 # Image height for SRAM memory in activationmemory

k = 3 # QUADRATIC Kernel size
layer_fifodepth = 9

imw = 3*k # Image width for tilebuffer 
imh = k # Image heigth for tilebuffer
pooling_fifodepth = imagewidth/2 # Depth of pooling fifo in OCU Pool
threshold_fifodepth = layer_fifodepth

weight_stagger = 2 # Number of cycles to load one full wight buffer in OCU Pool
pipelinedepth = 2 # Number of pipelinestages
#pipelinedepth = 4

numactmemsets = 2
actmemsetsbitwidth = np.maximum(int(np.ceil(np.log2(numactmemsets))),1)
#weightmemorybankdepth = 256 # Number of words per weight memory bank
weightmemorybankdepth = layer_fifodepth*weight_stagger*k*k
number_of_stimuli = 100 # Default number of stimuli
ACTMEM_START_ADDR = int("0x1EC00000",0)
WEIGHTMEM_START_ADDR = int("0x1EC40000",0)
### ARGPARSE INTERFACE ###

#args = argparse.Namespace()

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def vprint(_input):
    if(args.verbosity == True):
        print(_input)

### END ARGPARSE INTERFACE ###

def format_signals(_signals, signaltypes, signalwidths):
    string = ""

    for j in _signals._fields:
        for i in np.nditer(getattr(_signals, j)):
            string = string + _format(i, getattr(signaltypes, j), (getattr(signalwidths, j))[0])

    return string

def format_input(_input):
    
    string = ""

    for j in _input._fields:
        for i in np.nditer(getattr(_input, j)):
            string = string + _format(i, getattr(inputtypes, j), (getattr(inputwidths, j))[0])
        
    return string

def format_output(_output):
    
    string = ""

    for j in _output._fields:
        for i in np.nditer(getattr(_output, j)):
            string = string + _format(i, getattr(outputtypes, j), (getattr(outputwidths, j))[0])
        
    return string

### JSON SERIALIZATION FUNCTIONS ###

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.int64):
            return int(obj)
        return json.JSONEncoder.default(self, obj)
        
def recursive_to_json(obj):
    _json = {}

    if (isinstance(obj, tuple)):
        datas = obj._asdict()
        for data in datas:
            if isinstance(datas[data], tuple):
                _json[data] = (recursive_to_json(datas[data]))
            else:
                _json[data] = (datas[data])

    return _json

def jprint(f, _input):
    if(args.json == True):
        f.write(str(json.dumps(recursive_to_json(_input), cls=NumpyEncoder))+" \n")

def jload(_file):
    
    objs = []
    with open(_file,"r") as f:
        for x in f:
            objs.append(json.loads(x))

    return objs

### JSON SERIALIZATION FUNCTIONS ###

### SERIALIZATION FUNCTIONS ###

def _format(num, _type, bitwidth=1):
    if(_type == 'ternary'):
        return format_ternary(num)
    else:
        return format_binary(num, bitwidth)
        
def format_ternary(num):
    if(num == 1):
        return '01';
    elif(num == 0):
        return '00';
    elif(num == -1):
        return '11';
    else:
        return 'XX';

def format_binary(num, bitwidth):
    max_val = int(2**(bitwidth))
    
    if(num < 0):
        neg_flag = 1
    else:
        neg_flag = 0

    if(neg_flag == 1):
        _num = max_val + num
    else:
        _num = num

    string = bin(int(_num))[2:]

    string = string.zfill(bitwidth)

    return str(string[-bitwidth:])


### END SERIALIZATION FUNCTIONS ###
### DESERIALIZATION FUNCTIONS ###

def parse_input(f):
    line = f.readline()

    inputlist = []

    for field in (inputwidths._fields):
        if(getattr(inputwidths,field)[1] == 1):
            currentelement = 0;
        else:
            currentelement = np.empty(np.prod(getattr(inputwidths,field)[1]))

        for width in range(np.prod(getattr(inputwidths,field)[1])):

            subline = str(line[0:getattr(inputwidths,field)[0]])

            if(getattr(inputwidths,field)[1] == 1):
                currentelement = _parse(subline, getattr(inputtypes,field), getattr(inputwidths,field)[0]) 
            else:
                currentelement[width] = _parse(subline, getattr(inputtypes,field), getattr(inputwidths,field)[0]) 
            
            line = line[getattr(inputwidths,field)[0]:]

        if(getattr(inputwidths,field)[1] == 1):
            inputlist.append(currentelement)
        else:
            inputlist.append(np.reshape(currentelement,getattr(inputwidths,field)[1]))

    retinput = _input(*inputlist)

    #vprint(retinput)

    return (retinput)


def _parse(line, _type):
    if (_type == 'ternary'):
        return parse_ternary(line)
    else:
        return parse_binary(line, _type, bitwidth)

def parse_binary(num, signedness='signed', bitwidth=1):
    max_val = int(2**bitwidth)
    ret = 0

    if(signedness == 'signed'):
        ret = int(num,2) - max_val
        return ret
    elif(signedness == 'unsigned'):
        ret = int(num,2)
        return ret
    else:
        return ret

def parse_ternary(num):
    if(num == '01'):
        return 1
    elif(num == '11'):
        return -1
    elif(num == '00'):
        return 0
    else:
        return 2

def _parse(num, _type, bitwidth=1):
    if(_type == 'ternary'):
        return parse_ternary(num)
    elif(_type=='unsigned'):
        return parse_binary(num, 'unsigned', bitwidth)
    elif(_type=='signed'):
        return parse_binary(num, 'signed', bitwidth)

### END DESERIALIZATION FUNCTIONS ###

def gen_codebook(stimulifile=None, exp_responsesfile=None):

    global codebook

    codebook = {}
    orig_codebook = {}
    stimuli = []
    responses = []

    if stimulifile is None:
        stimulifile = Path(__file__).resolve().parent.joinpath('decoder_stimuli.txt')
    if exp_responsesfile is None:
        exp_responsesfile = Path(__file__).resolve().parent.joinpath('decoder_exp_responses.txt')

    with open(stimulifile) as f:
        for x in f:
            stimuli.append(x.replace('\n','',1))

    with open(exp_responsesfile) as f:
        for x in f:
            responses.append(x.replace('\n','',1))

    codes = []
    orig_codes = []
    curr_stimuli = ''

    for i in range(len(stimuli)):
        curr_stimuli = stimuli[i]
        orig_codes.append(curr_stimuli)
        if('X' in curr_stimuli):
            while('X' in curr_stimuli):
                codes.append(curr_stimuli.replace('X','1',1))
                codes.append(curr_stimuli.replace('X','0',1))
                curr_stimuli = curr_stimuli.replace('X','0',1)
        else:
            codes.append(curr_stimuli)

        for j in range(len(codes)):
            codebook.update({str(codes[j]):str(responses[i])})
        for j in range(len(orig_codes)):
            orig_codebook.update({str(orig_codes[j]):str(responses[i])})
        codes = []
        orig_codes = []
    return codebook, orig_codebook

def decode(string):
    return codebook[string]
