import os

import sys
#import traceback
import numpy as np
import mat73

# TODO: remove this when releasingChange to $PYTHONPATH installation
sys.path.insert(0, 'engine/set/imagestar/')

from imagestar import ImageStar

SRCLOADER_ERRMSG_INVALID_ARGS_NUM = "Invalid number of input arguments, (should be 4)"
SRCLOADER_ERRMSG_PATH_NOT_STRING = "The given path is not a string"
SRCLOADER_ERRMSG_PATH_DOESNT_EXIST = lambda path : "The folder at the given path %s does not exist" % path
SRCLOADER_ERRMSG_INVALID_SOURCE_TYPE = lambda source_type : "The given source type \'%s\' is not supported" % source_type
SRCLOADER_ERRMSG_INVALID_STORAGE_TYPE = lambda storage_type : "The given storage type \'%s\' is not supported" % storage_type
SRCLOADER_ERRMSG_INVALID_LOAD_METHOD = lambda load_method : "Loading from a \'%s\' file is not supported" % load_method
SRCLOADER_ERRMSG_INVALID_FILE_EXTENSION = lambda file_extension : "Loading from a \'%s\' file is not supported" % file_extension

class SourceLoader:
    
    _available_extensions = ['mat', 'py']
    _available_source_types = ['matlab']
    _available_storage_types = ['folder']
    _available_load_methods = ['standard', 'bounds', 'image']
    
    @staticmethod
    def load_image_star(*args):
        
        path = None
        source_type = None
        storage_type = None
        load_method = None
        
        if len(args) == 4:
            path = args[0]
            source_type = args[1]
            storage_type = args[2]
            load_method = args[3]
        else:
            raise Exception(SRCLOADER_ERRMSG_INVALID_ARGS_NUM)
        
        assert isinstance(path, str), 'error: %s' % SRCLOADER_ERRMSG_PATH_NOT_STRING
        assert os.path.exists(path), 'error: %s' % SRCLOADER_ERRMSG_PATH_DOESNT_EXIST(path)
        assert source_type in SourceLoader._available_source_types, 'error: %s' % SRCLOADER_ERRMSG_INVALID_SOURCE_TYPE(source_type)
        assert storage_type in SourceLoader._available_storage_types, 'error: %s' % SRCLOADER_ERRMSG_INVALID_STORAGE_TYPE(storage_type)
        assert load_method in SourceLoader._available_load_methods, 'error: %s' % SRCLOADER_ERRMSG_INVALID_LOAD_METHOD(load_method)
        
        if storage_type == 'folder':
            return SourceLoader._load_folder(path, source_type, load_method)
        
    @staticmethod
    def load_ndim_array(path):
        assert isinstance(path, str), 'error: %s' % SRCLOADER_ERRMSG_PATH_NOT_STRING
        assert os.path.exists(path), 'error: %s' % SRCLOADER_ERRMSG_PATH_DOESNT_EXIST
        
        file_extension = path[path.rfind('.') + 1 : len(path)]
        
        assert file_extension in SourceLoader._available_extensions, 'error: %s' % SRCLOADER_ERRMSG_INVALID_FILE_EXTENSION(file_extension)
        
        if file_extension == 'mat':
            return np.array(list(mat73.loadmat(path).values())[0])
        
    @staticmethod
    def _load_folder(path, source_type, load_method):
        if source_type == 'matlab':
            return SourceLoader._load_matlab_folder(path, load_method)
            
    @staticmethod
    def _load_matlab_folder(path, load_method):
        if load_method == 'standard':
            assert os.path.exists(path + '/V.mat'), 'error: %s' % SRCLOADER_ERRMSG_PATH_DOESNT_EXIST(path + '/V.mat')
            assert os.path.exists(path + '/C.mat'), 'error: %s' % SRCLOADER_ERRMSG_PATH_DOESNT_EXIST(path + '/C.mat')
            assert os.path.exists(path + '/d.mat'), 'error: %s' % SRCLOADER_ERRMSG_PATH_DOESNT_EXIST(path + '/d.mat')
            assert os.path.exists(path + '/predicate_lb.mat'), 'error: %s' % SRCLOADER_ERRMSG_PATH_DOESNT_EXIST(path + '/pred_lb.mat')
            assert os.path.exists(path + '/predicate_ub.mat'), 'error: %s' % SRCLOADER_ERRMSG_PATH_DOESNT_EXIST(path + '/pred_ub.mat')
            
            V = SourceLoader._read_csv_data(path + '/V.mat')
            if(len(V.shape) == 3):
                V = np.reshape(V, (V.shape[0], V.shape[1], 1, V.shape[2]))
            elif(len(V.shape) == 2):
                V = np.reshape(V, (1, 1, V.shape[0], V.shape[1]))
            
            C = SourceLoader._read_csv_data(path + '/C.mat')
            if(len(C.shape) == 1):
                C = np.reshape(C, (1, C.shape[0]))
                
            d = SourceLoader._read_csv_data(path + '/d.mat')
            
            predicate_lb = SourceLoader._read_csv_data(path + '/predicate_lb.mat')
            # if(len(predicate_lb.shape) == 1):
            #     predicate_lb = np.reshape(predicate_lb, (1, predicate_lb.shape[0]))
            predicate_ub = SourceLoader._read_csv_data(path + '/predicate_ub.mat')
            # if(len(predicate_ub.shape) == 1):
            #     predicate_ub = np.reshape(predicate_ub, (1, predicate_ub.shape[0]))

            return ImageStar(
                    V, C, d, predicate_lb, predicate_ub
                )
        elif load_method == 'bounds':
            assert os.path.exists(path + '/im_lb.mat'), 'error: %s' % SRCLOADER_ERRMSG_PATH_DOESNT_EXIST
            assert os.path.exists(path + '/im_ub.mat'), 'error: %s' % SRCLOADER_ERRMSG_PATH_DOESNT_EXIST
            
            im_lb = SourceLoader._read_csv_data(path + '/im_lb.mat')
            im_ub = SourceLoader._read_csv_data(path + '/im_ub.mat')
    
            return ImageStar(im_lb, im_ub)
        elif load_method == 'image':
            assert os.path.exists(path + '/im.mat'), 'error: %s' % SRCLOADER_ERRMSG_PATH_DOESNT_EXIST
            assert os.path.exists(path + '/im_lb.mat'), 'error: %s' % SRCLOADER_ERRMSG_PATH_DOESNT_EXIST
            assert os.path.exists(path + '/im_ub.mat'), 'error: %s' % SRCLOADER_ERRMSG_PATH_DOESNT_EXIST
            
            im = SourceLoader._read_csv_data(path + '/im.mat')
            im_lb = SourceLoader._read_csv_data(path + '/im_lb.mat')
            im_ub = SourceLoader._read_csv_data(path + '/im_ub.mat')
    
            return ImageStar(im, im_lb, im_ub)
    
    @staticmethod
    def _read_csv_data(path):        
        return np.array(list(mat73.loadmat(path).values())[0])

    # @staticmethod
    # def process_exception(ex): 
    #     ex_type, ex_value, ex_traceback = sys.exc_info()
    #     trace_back = traceback.extract_tb(ex_traceback)
            
    #     stack_trace = ""
            
    #     for trace in trace_back:
    #         stack_trace = stack_trace + "File : %s ,\n Line : %d,\n Func.Name : %s,\n Message : %s\n" % (trace[0], trace[1], trace[2], trace[3])
                    
    #     print("Exception type : %s " % ex_type.__name__)
    #     print("Exception message : %s" %ex_value)
    #     print("Stack trace : %s" %stack_trace)
