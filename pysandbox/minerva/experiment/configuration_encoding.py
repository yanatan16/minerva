'''
Created on Mar 28, 2012

@author: jon
'''
import numpy
import minerva.features.generators as gens
import json
from minerva.data import DataNormalizer, OutputFunction
from minerva import regression as regs

## Basic Decoding

_basic_coders = dict({
                  '__numpy__': numpy.__dict__,
                  '__generator__': gens.__dict__,
                  '__regressor__': regs.__dict__
                  })

## Advanced Encode/Decoders
# Encode Normalizers
def decodeNormalizer(dct):
    return DataNormalizer(dct['ror_row'], dct['vol_row'])
def encodeNormalizer(obj):
    return dict({
                 '__normalizer__': True,
                 'ror_row':obj.ror_row,
                 'vol_row':obj.vol_row,
                 })
    
# Encode Output Functions
def decodeOutputFunctions(dct):
    return OutputFunction(dct['func'], dct['row'])
def encodeOutputFunction(obj):
    return dict({
                 '__output_func__': True,
                 'func': obj.func,
                 'row': obj.row
                 })

_advanced_decoders = dict({
                           '__normalizer__' : decodeNormalizer,
                           '__output_func__': decodeOutputFunctions
                           })
_advanced_encoders = dict({
                           type(DataNormalizer()): encodeNormalizer,
                           type(OutputFunction(numpy.mean,1)): encodeOutputFunction
                           })
    
def decodeExperimentConfiguration(dct):
    for name, fns in _basic_coders.iteritems():
        if name in dct:
            return fns[dct[name]]
    for name, fn in _advanced_decoders.iteritems():
        if name in dct:
            return fn(dct) 
    return dct
        
class ExperimentConfigurationEncoder(json.JSONEncoder):
    def default(self, obj):
        for group_name, fns in _basic_coders.iteritems():
            for fn_name, fn in fns.iteritems():
                if fn == obj:
                    return dict({group_name:fn_name})
        if type(obj) in _advanced_encoders:
            return _advanced_encoders[type(obj)](obj)
        return super(ExperimentConfigurationEncoder, self).default(obj)
    
def deserializeConfiguration(filename):
    f = open(filename, 'r')
    config = json.load(f, object_hook=decodeExperimentConfiguration)
    f.close()
    assert config is not None, 'Unable to deserialize configuration'
    return (config['static'], config['variable'])

def serializeConfiguration(filename, static_config, variable_config = dict()):
    f = open(filename, 'w')
    config = dict({'static':static_config,'variable':variable_config})
    json.dump(config, f, indent=2, cls=ExperimentConfigurationEncoder)
    f.close()
    
if __name__=="__main__":
    out = json.dumps(DataNormalizer(),cls=ExperimentConfigurationEncoder)
    i = json.loads(out, object_hook=decodeExperimentConfiguration)
    print out
    print i
    