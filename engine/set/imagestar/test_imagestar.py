import unittest

from test_inputs.sources import *
from imagestar import ImageStar, os

# TODO: remove this when releasing. Change to $PYTHONPATH installation
sys.path.insert(0, "engine/utils_beta/")
from source_loader import SourceLoader

class TestImageStar(unittest.TestCase):
    """
        Tests the ImageStar class
    """

    @classmethod
    def setResult(cls, amount, errors, failures, skipped):
        cls.amount, cls.errors, cls.failures, cls.skipped = \
            amount, errors, failures, skipped

    def tearDown(self):
        amount = self.currentResult.testsRun
        errors = self.currentResult.errors
        failures = self.currentResult.failures
        skipped = self.currentResult.skipped
        self.setResult(amount, errors, failures, skipped)

    @classmethod
    def tearDownClass(cls):
        print("\n\ntests run: " + str(cls.amount))
        print("errors: " + str(len(cls.errors)))
        print("failures: " + str(len(cls.failures)))
        print("success: " + str(cls.amount - len(cls.errors) - len(cls.failures)))
        print("skipped: " + str(len(cls.skipped)))

    def run(self, result=None):
        self.currentResult = result # remember result for use in tearDown
        unittest.TestCase.run(self, result) # call superclass run method   

    #@unittest.skip("skip it")
    def test_constructor_predicate_init(self):
        """
            Tests the initialization with:
            V -> Basis matrix
            C -> Predicate matrix
            d -> Predicate vector
            predicate_lb -> predicate lower bound
            predicate_ub -> predicate upper bound
        """
        print("\n\nStarting Constructor Test With Basic Attributes.............")
        current_path = os.getcwd() + "/engine/set/imagestar/test_inputs/fmnist_img"

        try:
            print("Loading ImageStar from %s" % current_path)      
            test_star = SourceLoader.load_image_star(os.getcwd() + "/engine/set/imagestar/test_inputs/fmnist_img", 'matlab', 'folder', 'standard')
            print("ImageStar initialized successfully.............")  
            print("V: " + str(test_star.get_V().shape))
            print("C: " + str(test_star.get_C().shape))
        except Exception as ex:
            print("ImageStar initialization failed.............") 
            print("Exception handled => " + str(ex))

        self.assertEqual(isinstance(test_star, ImageStar), True)

    #@unittest.skip("skip it")
    def test_constructor_image_init(self):
        """
            Tests the initialization with:
            IM -> ImageStar
            LB -> Lower image
            UB -> Upper image
        """

        print("\n\nStarting Constructor Test With Image Attributes.............") 
        print("Constructing images.............")        
        test_IM = np.zeros((4, 4, 3))
        test_LB = np.zeros((4, 4, 3))
        test_UB = np.zeros((4, 4, 3))

        test_IM[:,:,0] = np.array([[1, 1, 0, 1], [0, 0, 1, 1], [1, 0, 1, 0], [1, 1, 1, 1]])
        test_IM[:,:,1] = np.array([[0, 1, 0, 0], [1, 0, 0, 1], [0, 1, 1, 0], [0, 0, 0, 1]])
        test_IM[:,:,2] = np.array([[1, 1, 1, 1], [1, 1, 0, 1], [0, 1, 1, 0], [1, 0, 1, 0]])

        test_LB[:,:,0] = np.array([[-0.1, -0.2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]) # attack on pixel (1,,1,) and (1,,2)
        test_LB[:,:,1] = np.array([[-0.1, -0.15, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        test_LB[:,:,2] = test_LB[:,:,1]

        test_UB[:,:,0] = np.array([[0.1, 0.2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        test_UB[:,:,1] = np.array([[0.1, 0.15, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        test_UB[:,:,2] = test_UB[:,:,1]
        print("Images successfully constructed.............")        

        try:
            print("Initializing ImageStar.............") 
            test_star = ImageStar(
                    test_IM, test_LB, test_UB
                )
            print("ImageStar initialized successfully.............")
            print("V: " + str(test_star.get_V().shape))
            print("C: " + str(test_star.get_C().shape))
            print("IM: " + str(test_star.get_IM().shape))
            print("LB: " + str(test_star.get_LB().shape))  
            print("UB: " + str(test_star.get_UB().shape))  
        except Exception as ex:
            print("ImageStar initialization failed.............") 
            print("Exception handled => " + str(ex))

        self.assertEqual(isinstance(test_star, ImageStar), True)

    #@unittest.skip("skip it")
    def test_constructor_bounds_init(self):
        """
            Tests the initialization with:
            lb -> lower bound
            ub -> upper bound
        """

        print("\n\nStarting Constructor Test With Image Bounds Attributes.............") 
        current_path = os.getcwd() + "/engine/set/imagestar/test_inputs/fmnist_img"

        try:
            print("Loading ImageStar from %s" % current_path)     
            test_star = SourceLoader.load_image_star(os.getcwd() + "/engine/set/imagestar/test_inputs/fmnist_img", 'matlab', 'folder', 'bounds')
            print("ImageStar initialized successfully.............")  
            print("V: " + str(test_star.get_V().shape))
            print("C: " + str(test_star.get_C().shape))
            print("im_lb: " + str(test_star.get_im_lb().shape))
            print("im_ub: " + str(test_star.get_im_ub().shape))  
        except Exception as ex:
            print("ImageStar initialization failed.............") 
            print("Exception handled => " + str(ex))

        self.assertEqual(isinstance(test_star, ImageStar), True)

    #@unittest.skip("skip it")
    def test_add_constraints_basic(self):
        """
            Tests the ImageStar's method that compares two points of the ImageStar 
    
            V -> Basis matrix
            C -> Predicate matrix
            d -> Predicate vector
            predicate_lb -> predicate lower bound
            predicate_ub -> predicate upper bound
    
            input -> new constraints
        """

        print("\n\nStarting <add_constraints> Test.............")
        current_path = os.getcwd() + "/engine/set/imagestar/test_inputs/fmnist_img"

        try:
            print("Loading ImageStar from %s" % current_path)
            print("Loading ImageStar.............")         
            test_star = SourceLoader.load_image_star(current_path, 'matlab', 'folder', 'standard')
            print("ImageStar initialized successfully.............")  
            print("V: " + str(test_star.get_V().shape))
            print("C: " + str(test_star.get_C().shape))
        except Exception as ex:
            print("ImageStar initialization failed.............") 
            print("Exception handled => " + str(ex))

        current_C = test_star.get_C()
        current_d = test_star.get_d()
        try:
            print("Adding Constraints............")
            test_C, test_d = ImageStar.add_constraints(test_star, [4, 4, 0], [1, 1, 0])
            print("Constraints added successfully............")
        except Exception as ex:
            print("Constraints addition failed............")
            print("Exception handled => " + str(ex))

        self.assertEqual((test_C.shape[0] == current_C.shape[0] + 1) and \
                         (test_d.shape[0] == current_d.shape[0] + 1), True)

    #@unittest.skip("skip it")
    def test_affine_mapping(self):
        """
            Test affine mapping -> ImageStar .* scale + offset
        
            scale : float -> affine map scale
            offset : np.array([*float]) -> affine map offset
            
            V -> Basis matrix
            C -> Predicate matrix
            d -> Predicate vector
            predicate_lb -> predicate lower bound
            predicate_ub -> predicate upper bound
        """

        print("\n\nStarting <affine_mapping> Test.............")
        current_path = os.getcwd() + "/engine/set/imagestar/test_inputs/imgstar_input"
        am_output_path = os.getcwd() + "/engine/set/imagestar/test_inputs/affine_mapping/affineMap_output.mat"
        scale_path = os.getcwd() + "/engine/set/imagestar/test_inputs/affine_mapping/affineMap_scale.mat"
        offset_path = os.getcwd() + "/engine/set/imagestar/test_inputs/affine_mapping/affineMap_offset.mat"

        try:
            print("Loading ImageStar from %s" % current_path)
            print("Loading ImageStar.............")         
            test_star = SourceLoader.load_image_star(current_path, 'matlab', 'folder', 'standard')
            print("ImageStar initialized successfully.............")  
            print("V: " + str(test_star.get_V().shape))
            print("C: " + str(test_star.get_C().shape))
        except Exception as ex:
            print("ImageStar initialization failed.............") 
            print("Exception handled => " + str(ex))

        print("Loading valid AM output from %s" % am_output_path)
        test_am_output = SourceLoader.load_ndim_array(am_output_path)
        print("Loading scale from %s" % scale_path)
        test_scale = SourceLoader.load_ndim_array(scale_path)
        print("Loading offset from %s" % offset_path)
        test_offset = SourceLoader.load_ndim_array(offset_path)

        try:
            print("Performing Affine Mapping.............")
            am_result = test_star.affine_map(test_scale, test_offset)
            print("Affine Mapping completed.............")
        except Exception as ex:
            print("Affine Mapping failed.............") 
            print("Exception handled => " + str(ex))
            
        self.assertEqual(am_result.get_V().all(), test_am_output.all())    

    #@unittest.skip("skip it")
    def test_affine_mapping_invalid_input(self):
        """
            Test affine mapping -> ImageStar .* scale + offset
        
            scale : float -> affine map scale
            offset : np.array([*float]) -> affine map offset
            
            V -> Basis matrix
            C -> Predicate matrix
            d -> Predicate vector
            predicate_lb -> predicate lower bound
            predicate_ub -> predicate upper bound
        """

        print("\n\nStarting <affine_mapping> Test with Invalid Input.............")
        current_path = os.getcwd() + "/engine/set/imagestar/test_inputs/imgstar_input"

        try:
            print("Loading ImageStar from %s" % current_path)
            print("Loading ImageStar.............")         
            test_star = SourceLoader.load_image_star(current_path, 'matlab', 'folder', 'standard')
            print("ImageStar initialized successfully.............")  
            print("V: " + str(test_star.get_V().shape))
            print("C: " + str(test_star.get_C().shape))
        except Exception as ex:
            print("ImageStar initialization failed.............") 
            
            print("Exception handled => " + str(ex))

        exception_handled = False
        
        try:
            print("Performing Affine Mapping.............")
            am_result = test_star.affine_map("test_scale", "test_offset")
            print("Affine Mapping completed.............")
        except Exception as ex:
            print("Affine Mapping failed.............") 
            print("Exception handled => " + str(ex))
            
            if str(ex) == "error: Input should be a numpy array":
                exception_handled = True
            
        self.assertEqual(exception_handled, True)

    #@unittest.skip("skip it")
    def test_affine_mapping_invalid_dims(self):
        """
            Test affine mapping -> ImageStar .* scale + offset
        
            scale : float -> affine map scale
            offset : np.array([*float]) -> affine map offset
            
            V -> Basis matrix
            C -> Predicate matrix
            d -> Predicate vector
            predicate_lb -> predicate lower bound
            predicate_ub -> predicate upper bound
        """

        print("\n\nStarting <affine_mapping> Test with Invalid Input.............")
        current_path = os.getcwd() + "/engine/set/imagestar/test_inputs/imgstar_input"

        try:
            print("Loading ImageStar from %s" % current_path)
            print("Loading ImageStar.............")         
            test_star = SourceLoader.load_image_star(current_path, 'matlab', 'folder', 'standard')
            print("ImageStar initialized successfully.............")  
            print("V: " + str(test_star.get_V().shape))
            print("C: " + str(test_star.get_C().shape))
        except Exception as ex:
            print("ImageStar initialization failed.............") 
            print("Exception handled => " + str(ex))

        exception_handled = False
        
        try:
            print("Performing Affine Mapping.............")
            am_result = test_star.affine_map(np.ones((5,5)), np.ones((5,5)))
            print("Affine Mapping completed.............")
        except Exception as ex:
            print("Affine Mapping failed.............") 
            print("Exception handled => " + str(ex))
            
            if str(ex) == "error: Inconsistent number of channels between scale array and the ImageStar":
                exception_handled = True
            
        self.assertEqual(exception_handled, True)

    #@unittest.skip("skip it")
    def test_containts_false(self):
        """
            Checks if the initialized ImageStar contains the given image
                       
            V -> Basis matrix
            C -> Predicate matrix
            d -> Predicate vector
            predicate_lb -> predicate lower bound
            predicate_ub -> predicate upper bound
            
            test_input -> the input image
        """

        print("\n\nStarting <contains> False Test.............")
        current_path = os.getcwd() + "/engine/set/imagestar/test_inputs/fmnist_img"
        false_input_path = os.getcwd() + "/engine/set/imagestar/test_inputs/contains/false_input.mat"

        try:
            print("Loading ImageStar from %s" % current_path)
            print("Loading ImageStar.............")         
            test_star = SourceLoader.load_image_star(current_path, 'matlab', 'folder', 'standard')
            print("ImageStar initialized successfully.............")  
            print("V: " + str(test_star.get_V().shape))
            print("C: " + str(test_star.get_C().shape))
        except Exception as ex:
            print("ImageStar initialization failed.............") 
            print("Exception handled => " + str(ex))

        print("Loading the input from %s" % false_input_path)
        test_scale = SourceLoader.load_ndim_array(false_input_path)

        try:
            print("Checking if the ImageStar contains the input.............")
            test_result = test_star.contains(test_scale)
            print("Contains operation completed successfully.............")
        except Exception as ex:
            print("Contains operation failed.............")
            print("Exception handled => " + str(ex))

        self.assertEqual(test_result, False)

    #@unittest.skip("skip it")
    def test_containts_true(self):
        """
            Checks if the initialized ImageStar contains the given image
                       
            V -> Basis matrix
            C -> Predicate matrix
            d -> Predicate vector
            predicate_lb -> predicate lower bound
            predicate_ub -> predicate upper bound
            
            test_input -> the input image
        """

        print("\n\nStarting <contains> True Test.............")
        current_path = os.getcwd() + "/engine/set/imagestar/test_inputs/fmnist_img"
        false_input_path = os.getcwd() + "/engine/set/imagestar/test_inputs/contains/true_input.mat"

        try:
            print("Loading ImageStar from %s" % current_path)
            print("Loading ImageStar.............")         
            test_star = SourceLoader.load_image_star(current_path, 'matlab', 'folder', 'standard')
            print("ImageStar initialized successfully.............")  
            print("V: " + str(test_star.get_V().shape))
            print("C: " + str(test_star.get_C().shape))
        except Exception as ex:
            print("ImageStar initialization failed.............") 
            print("Exception handled => " + str(ex))

        print("Loading the input from %s" % false_input_path)
        test_scale = SourceLoader.load_ndim_array(false_input_path)

        try:
            print("Checking if the ImageStar contains the input.............")
            test_result = test_star.contains(test_scale)
            print("Contains operation completed successfully.............")
        except Exception as ex:
            print("Contains operation failed.............")
            print("Exception handled => " + str(ex))

        self.assertEqual(test_result, True)

    #@unittest.skip("skip it")
    def test_estimate_range_valid_input(self):
        """
            Tests the ImageStar's range estimation method
            
            V -> Basis matrix
            C -> Predicate matrix
            d -> Predicate vector
            predicate_lb -> predicate lower bound
            predicate_ub -> predicate upper bound

            range_input -> valid input range
            range_output -> valid output range
        """
        
        print("\n\nStarting <estimate_range> Test With Valid Input.............")
        current_path = os.getcwd() + "/engine/set/imagestar/test_inputs/fmnist_img"
        range_input_path = os.getcwd() + "/engine/set/imagestar/test_inputs/estimate_range/estimate_range_input.mat"
        range_output_path = os.getcwd() + "/engine/set/imagestar/test_inputs/estimate_range/estimate_range_output.mat"

        try:
            print("Loading ImageStar from %s" % current_path)
            print("Loading ImageStar.............")         
            test_star = SourceLoader.load_image_star(current_path, 'matlab', 'folder', 'standard')
            print("ImageStar initialized successfully.............")  
            print("V: " + str(test_star.get_V().shape))
            print("C: " + str(test_star.get_C().shape))
        except Exception as ex:
            print("ImageStar initialization failed.............") 
            print("Exception handled => " + str(ex))
        
        print("Loading the input from %s" % range_input_path)
        range_input = SourceLoader.load_ndim_array(range_input_path)
        
        print("Loading the input from %s" % range_output_path)
        range_output = SourceLoader.load_ndim_array(range_output_path)
        
        try:
            print("Estimating the range.............")
            result = test_star.estimate_range(range_input[0], range_input[1], range_input[2])
            result = np.array([result[0][0], result[1][0]])
            print("Estimation completed.............")
        except(Exception) as ex:
            print("Estimation failed.............")
            print("Exception handled => " + str(ex))
        
        self.assertEqual(result.all(), range_output.all())

    #@unittest.skip("skip it")
    def test_estimate_range_empty_imgstar(self):
        """
            Tests the ImageStar's range estimation method
            
            V -> Basis matrix
            C -> Predicate matrix
            d -> Predicate vector
            predicate_lb -> predicate lower bound
            predicate_ub -> predicate upper bound

            range_input -> valid input range
            range_output -> valid output range
        """
        
        print("\n\nStarting <estimate_range> Test With an Empty ImageStar.............")
        range_input_path = os.getcwd() + "/engine/set/imagestar/test_inputs/estimate_range/estimate_range_input.mat"

        try:
            print("Initializing ImageStar")        
            test_star = ImageStar()
            print("ImageStar initialized successfully.............")  
            print("V: " + str(test_star.get_V().shape))
            print("C: " + str(test_star.get_C().shape))
        except Exception as ex:
            print("ImageStar initialization failed.............") 
            print("Exception handled => " + str(ex))
        
        print("Loading the input from %s" % range_input_path)
        range_input = SourceLoader.load_ndim_array(range_input_path)        
        exception_handled = False
        try:
            print("Estimating the range.............")
            result = test_star.estimate_range(range_input[0], range_input[1], range_input[2])
            result = np.array([result[0][0], result[1][0]])
            print("Estimation completed.............")
        except(Exception) as ex:
            print("Estimation failed.............")
            print("Exception handled => " + str(ex))
            
            if str(ex) == "error: ImageStar is empty":
                exception_handled = True
        
        self.assertEqual(exception_handled, True)

    #@unittest.skip("skip it")
    def test_estimate_range_invalid_input(self):
        """
            Tests the ImageStar's range estimation method
            
            V -> Basis matrix
            C -> Predicate matrix
            d -> Predicate vector
            predicate_lb -> predicate lower bound
            predicate_ub -> predicate upper bound

            range_input -> valid input range
            range_output -> valid output range
        """
        
        print("\n\nStarting <estimate_range> Test With Invalid Input.............")
        current_path = os.getcwd() + "/engine/set/imagestar/test_inputs/fmnist_img"
        range_input_path = os.getcwd() + "/engine/set/imagestar/test_inputs/estimate_range/estimate_range_input.mat"

        try:
            print("Loading ImageStar from %s" % current_path)
            print("Loading ImageStar.............")         
            test_star = SourceLoader.load_image_star(current_path, 'matlab', 'folder', 'standard')
            print("ImageStar initialized successfully.............")  
            print("V: " + str(test_star.get_V().shape))
            print("C: " + str(test_star.get_C().shape))
        except Exception as ex:
            print("ImageStar initialization failed.............") 
            print("Exception handled => " + str(ex))
        
        print("Loading the input from %s" % range_input_path)
        range_input = SourceLoader.load_ndim_array(range_input_path)
        
        exceptions_handled = [False, False, False, False, False, False]
        test_values = [-1, -1, -1, 29, 29, 5]
        
        for i in range(len(exceptions_handled)):
            try:
                print("Estimating the range.............")
                current_range = np.copy(range_input)
                current_range[i % 3] = test_values[i]
                result = test_star.estimate_range(current_range[0], current_range[1], current_range[2])
                print("Estimation completed.............")
            except(Exception) as ex:
                print("Estimation failed.............")
                print("Exception handled => " + str(ex))
                
                if str(ex) == "error: Invalid index value":
                    exceptions_handled[i] = True
        
        self.assertEqual(all(item == True for item in exceptions_handled), True)


    # def test_evaluation_valid_input(self):
    #     """
    #         Tests evaluation using predicate initialization
        
    #         eval_input : int -> number of images
            
    #         V -> Basis matrix
    #         C -> Predicate matrix
    #         d -> Predicate vector
    #         predicate_lb -> predicate lower bound
    #         predicate_ub -> predicate upper bound
    #     """
                
    #     test_eval_input = read_csv_data(sources[EVALUATION_INIT][EVAL_INPUT_ID])
    #     test_eval_output = read_csv_data(sources[EVALUATION_INIT][EVAL_OUTPUT_ID])
                
    #     test_V = np.reshape(read_csv_data(sources[CONSTRUCTOR_PREDICATE_BOUNDARIES_INIT][V_ID]), (28,28,1,785))
    #     test_C = np.reshape(read_csv_data(sources[CONSTRUCTOR_PREDICATE_BOUNDARIES_INIT][C_ID]), (1, 784))
    #     test_d = read_csv_data(sources[CONSTRUCTOR_PREDICATE_BOUNDARIES_INIT][D_ID])
    #     test_predicate_lb = read_csv_data(sources[CONSTRUCTOR_PREDICATE_BOUNDARIES_INIT][PREDICATE_LB_ID])
    #     test_predicate_ub = read_csv_data(sources[CONSTRUCTOR_PREDICATE_BOUNDARIES_INIT][PREDICATE_UB_ID])
        
    #     test_star = ImageStar(
    #             test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
    #         )
                
    #     try:
    #         test_result = test_star.evaluate(test_eval_input)
    #     except Exception as ex:
    #         completion_flag = False
    #         process_exception(ex)

    # def test_get_local_bound_default(self):
    #     """
    #         Tests the ImageStar's method that calculates the local bounds for the given point and pool size
            
    #         V -> Basis matrix
    #         C -> Predicate matrix
    #         d -> Predicate vector
    #         predicate_lb -> predicate lower bound
    #         predicate_ub -> predicate upper bound

    #         input -> valid input
    #         bounds_output -> valid output bounds
    #     """
        
    #     test_V = np.reshape(read_csv_data(sources[GET_LOCAL_BOUND_INIT][V_ID]), (28, 28, 1, 785))
    #     test_C = np.reshape(read_csv_data(sources[GET_LOCAL_BOUND_INIT][C_ID]), (1, 784))
    #     test_d = read_csv_data(sources[GET_LOCAL_BOUND_INIT][D_ID])
    #     test_predicate_lb = read_csv_data(sources[GET_LOCAL_BOUND_INIT][PREDICATE_LB_ID])
    #     test_predicate_ub = read_csv_data(sources[GET_LOCAL_BOUND_INIT][PREDICATE_UB_ID])
        
    #     test_star = ImageStar(
    #             test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
    #         )
               
    #     test_bounds_input = np.array([int(item) for item in read_csv_data(sources[GET_LOCAL_BOUND_INIT][INPUT_ID])])
    #     test_bounds_output = read_csv_data(sources[GET_LOCAL_BOUND_INIT][OUTPUT_ID]).tolist()
                                
    #     self.assertEqual(test_star.get_local_bound(test_bounds_input[0:2], test_bounds_input[2:4], test_bounds_input[4]), test_bounds_output)

                
    # def test_get_local_bound_glpk(self):
    #     """
    #         Tests the ImageStar's method that calculates the local bounds for the given point and pool size
            
    #         V -> Basis matrix
    #         C -> Predicate matrix
    #         d -> Predicate vector
    #         predicate_lb -> predicate lower bound
    #         predicate_ub -> predicate upper bound

    #         input -> valid input
    #         bounds_output -> valid output bounds
    #     """
        
    #     test_V = np.reshape(read_csv_data(sources[GET_LOCAL_BOUND_INIT][V_ID]), (28, 28, 1, 785))
    #     test_C = np.reshape(read_csv_data(sources[GET_LOCAL_BOUND_INIT][C_ID]), (1, 784))
    #     test_d = read_csv_data(sources[GET_LOCAL_BOUND_INIT][D_ID])
    #     test_predicate_lb = read_csv_data(sources[GET_LOCAL_BOUND_INIT][PREDICATE_LB_ID])
    #     test_predicate_ub = read_csv_data(sources[GET_LOCAL_BOUND_INIT][PREDICATE_UB_ID])
        
    #     test_star = ImageStar(
    #             test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
    #         )
               
    #     test_bounds_input = np.array([int(item) for item in read_csv_data(sources[GET_LOCAL_BOUND_INIT][INPUT_ID])])
    #     test_bounds_output = read_csv_data(sources[GET_LOCAL_BOUND_INIT][OUTPUT_ID]).tolist()
                                
    #     self.assertEqual(test_star.get_local_bound(test_bounds_input[0:2], test_bounds_input[2:4], test_bounds_input[4], 'glpk'), test_bounds_output)

    # def test_get_local_bound_gurobi(self):
    #     """
    #         Tests the ImageStar's method that calculates the local bounds for the given point and pool size
            
    #         V -> Basis matrix
    #         C -> Predicate matrix
    #         d -> Predicate vector
    #         predicate_lb -> predicate lower bound
    #         predicate_ub -> predicate upper bound

    #         input -> valid input
    #         bounds_output -> valid output bounds
    #     """
        
    #     test_V = np.reshape(read_csv_data(sources[GET_LOCAL_BOUND_INIT][V_ID]), (28, 28, 1, 785))
    #     test_C = np.reshape(read_csv_data(sources[GET_LOCAL_BOUND_INIT][C_ID]), (1, 784))
    #     test_d = read_csv_data(sources[GET_LOCAL_BOUND_INIT][D_ID])
    #     test_predicate_lb = read_csv_data(sources[GET_LOCAL_BOUND_INIT][PREDICATE_LB_ID])
    #     test_predicate_ub = read_csv_data(sources[GET_LOCAL_BOUND_INIT][PREDICATE_UB_ID])
        
    #     test_star = ImageStar(
    #             test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
    #         )
               
    #     test_bounds_input = np.array([int(item) for item in read_csv_data(sources[GET_LOCAL_BOUND_INIT][INPUT_ID])])
    #     test_bounds_output = read_csv_data(sources[GET_LOCAL_BOUND_INIT][OUTPUT_ID]).tolist()
                                
    #     self.assertEqual(test_star.get_local_bound(test_bounds_input[0:2], test_bounds_input[2:4], test_bounds_input[4], 'linprog'), test_bounds_output)

    # def test_get_local_bound_glpk_disp(self):
    #     """
    #         Tests the ImageStar's method that calculates the local bounds for the given point and pool size
            
    #         V -> Basis matrix
    #         C -> Predicate matrix
    #         d -> Predicate vector
    #         predicate_lb -> predicate lower bound
    #         predicate_ub -> predicate upper bound

    #         input -> valid input
    #         bounds_output -> valid output bounds
    #     """
        
    #     test_V = np.reshape(read_csv_data(sources[GET_LOCAL_BOUND_INIT][V_ID]), (28, 28, 1, 785))
    #     test_C = np.reshape(read_csv_data(sources[GET_LOCAL_BOUND_INIT][C_ID]), (1, 784))
    #     test_d = read_csv_data(sources[GET_LOCAL_BOUND_INIT][D_ID])
    #     test_predicate_lb = read_csv_data(sources[GET_LOCAL_BOUND_INIT][PREDICATE_LB_ID])
    #     test_predicate_ub = read_csv_data(sources[GET_LOCAL_BOUND_INIT][PREDICATE_UB_ID])
        
    #     test_star = ImageStar(
    #             test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
    #         )
               
    #     test_bounds_input = np.array([int(item) for item in read_csv_data(sources[GET_LOCAL_BOUND_INIT][INPUT_ID])])
    #     test_bounds_output = read_csv_data(sources[GET_LOCAL_BOUND_INIT][OUTPUT_ID]).tolist()
                                
    #     self.assertEqual(test_star.get_local_bound(test_bounds_input[0:2], test_bounds_input[2:4], test_bounds_input[4], 'glpk', ['disp']), test_bounds_output)

    # def test_get_local_bound_gurobi_disp(self):
    #     """
    #         Tests the ImageStar's method that calculates the local bounds for the given point and pool size
            
    #         V -> Basis matrix
    #         C -> Predicate matrix
    #         d -> Predicate vector
    #         predicate_lb -> predicate lower bound
    #         predicate_ub -> predicate upper bound

    #         input -> valid input
    #         bounds_output -> valid output bounds
    #     """
        
    #     test_V = np.reshape(read_csv_data(sources[GET_LOCAL_BOUND_INIT][V_ID]), (28, 28, 1, 785))
    #     test_C = np.reshape(read_csv_data(sources[GET_LOCAL_BOUND_INIT][C_ID]), (1, 784))
    #     test_d = read_csv_data(sources[GET_LOCAL_BOUND_INIT][D_ID])
    #     test_predicate_lb = read_csv_data(sources[GET_LOCAL_BOUND_INIT][PREDICATE_LB_ID])
    #     test_predicate_ub = read_csv_data(sources[GET_LOCAL_BOUND_INIT][PREDICATE_UB_ID])
        
    #     test_star = ImageStar(
    #             test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
    #         )
               
    #     test_bounds_input = np.array([int(item) for item in read_csv_data(sources[GET_LOCAL_BOUND_INIT][INPUT_ID])])
    #     test_bounds_output = read_csv_data(sources[GET_LOCAL_BOUND_INIT][OUTPUT_ID]).tolist()
                                
    #     self.assertEqual(test_star.get_local_bound(test_bounds_input[0:2], test_bounds_input[2:4], test_bounds_input[4], 'linprog', ['disp']), test_bounds_output)


    # def test_get_local_max_index_empty_candidates(self):
    #     """
    #         Tests the ImageStar's method that calculates the local maximum point of the local image.
    #         Candidates set will be empty 
    
    #         V -> Basis matrix
    #         C -> Predicate matrix
    #         d -> Predicate vector
    #         predicate_lb -> predicate lower bound
    #         predicate_ub -> predicate upper bound
    
    #         input -> valid input
    #         local_index -> valid output bounds
    #     """
    
    #     test_V = np.reshape(read_csv_data(sources[GET_LOCAL_MAX_INDEX_INIT][V_ID]), (28, 28, 1, 785))
    #     test_C = np.reshape(read_csv_data(sources[GET_LOCAL_MAX_INDEX_INIT][C_ID]), (1, 784))
    #     test_d = read_csv_data(sources[GET_LOCAL_MAX_INDEX_INIT][D_ID])
    #     test_predicate_lb = read_csv_data(sources[GET_LOCAL_MAX_INDEX_INIT][PREDICATE_LB_ID])
    #     test_predicate_ub = read_csv_data(sources[GET_LOCAL_MAX_INDEX_INIT][PREDICATE_UB_ID])
    
    #     test_star = ImageStar(
    #             test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
    #         )
    
    #     test_local_index_input = np.array([int(item) for item in read_csv_data(sources[GET_LOCAL_MAX_INDEX_INIT][INPUT_ID])])
    #     test_local_index_output = (read_csv_data(sources[GET_LOCAL_MAX_INDEX_INIT][OUTPUT_ID])  - 1).astype('int').tolist()
    
    #     test_result = test_star.get_localMax_index(test_local_index_input[0:2] - 1, test_local_index_input[2:4], test_local_index_input[4] - 1, 'linprog')
    
    
    #     self.assertEqual((test_result == test_local_index_output).all(), True)

    # def test_get_local_max_index_candidates(self):
    #     """
    #         Tests the ImageStar's method that calculates the local maximum point of the local image. 
            
    #         V -> Basis matrix
    #         C -> Predicate matrix
    #         d -> Predicate vector
    #         predicate_lb -> predicate lower bound
    #         predicate_ub -> predicate upper bound

    #         input -> valid input
    #         local_index -> valid output bounds
    #     """
        
    #     test_V = np.reshape(read_csv_data(sources[GET_LOCAL_MAX_INDEX_INIT][V_CANDIDATES_ID]), (24, 24, 3, 785))
    #     test_C = np.reshape(read_csv_data(sources[GET_LOCAL_MAX_INDEX_INIT][C_CANDIDATES_ID]), (1, 784))
    #     test_d = read_csv_data(sources[GET_LOCAL_MAX_INDEX_INIT][D_CANDIDATES_ID])
    #     test_predicate_lb = read_csv_data(sources[GET_LOCAL_MAX_INDEX_INIT][PREDICATE_LB_CANDIDATES_ID])
    #     test_predicate_ub = read_csv_data(sources[GET_LOCAL_MAX_INDEX_INIT][PREDICATE_UB_CANDIDATES_ID])
    #     test_im_lb = read_csv_data(sources[GET_LOCAL_MAX_INDEX_INIT][IM_LB_CANDIDATES_ID])
    #     test_im_ub = read_csv_data(sources[GET_LOCAL_MAX_INDEX_INIT][IM_UB_CANDIDATES_ID])
        
    #     test_star = ImageStar(
    #             test_V, test_C, test_d, test_predicate_lb, test_predicate_ub, test_im_lb, test_im_ub
    #         )
               
    #     test_local_index_input = np.array([int(item) for item in read_csv_data(sources[GET_LOCAL_MAX_INDEX_INIT][INPUT_CANDIDATES_ID])])
    #     test_local_index_output = (read_csv_data(sources[GET_LOCAL_MAX_INDEX_INIT][OUTPUT_CANDIDATES_ID])).tolist()
                        
    #     test_result = test_star.get_localMax_index(test_local_index_input[0:2], test_local_index_input[2:4], test_local_index_input[4])[0].astype('int')
                                
    #     self.assertEqual((test_result== test_local_index_output).all(), True)
        
    # def test_get_local_points(self):
    #     """
    #         Tests the ImageStar's method that calculates the local points for the given point and pool size

            
    #         V -> Basis matrix
    #         C -> Predicate matrix
    #         d -> Predicate vector
    #         predicate_lb -> predicate lower bound
    #         predicate_ub -> predicate upper bound

    #         input -> valid input
    #         bounds_output -> valid output bounds
    #     """
        
    #     test_V = np.reshape(read_csv_data(sources[GET_LOCAL_POINTS_INIT][V_ID]), (28, 28, 1, 785))
    #     test_C = np.reshape(read_csv_data(sources[GET_LOCAL_POINTS_INIT][C_ID]), (1, 784))
    #     test_d = read_csv_data(sources[GET_LOCAL_POINTS_INIT][D_ID])
    #     test_predicate_lb = read_csv_data(sources[GET_LOCAL_POINTS_INIT][PREDICATE_LB_ID])
    #     test_predicate_ub = read_csv_data(sources[GET_LOCAL_POINTS_INIT][PREDICATE_UB_ID])
        
    #     test_star = ImageStar(
    #             test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
    #         )
               
    #     test_points_input = np.array([int(item) for item in read_csv_data(sources[GET_LOCAL_POINTS_INIT][INPUT_ID])])
    #     test_points_output = np.array(read_csv_data(sources[GET_LOCAL_POINTS_INIT][OUTPUT_ID])) - 1
                                
    #     test_result = test_star.get_local_points(test_points_input[0:2], test_points_input[2:4])
                                
    #     self.assertEqual(test_result.all(), test_points_output.all())

    # def test_get_local_max_index2_candidates(self):
    #     """
    #         Tests the ImageStar's method that calculates the local maximum point of the local image. 
            
    #         V -> Basis matrix
    #         C -> Predicate matrix
    #         d -> Predicate vector
    #         predicate_lb -> predicate lower bound
    #         predicate_ub -> predicate upper bound

    #         input -> valid input
    #         local_index -> valid output bounds
    #     """
        
    #     test_V = np.reshape(read_csv_data(sources[GET_LOCAL_MAX_INDEX2_INIT][V_ID]), (24, 24, 3, 785))
    #     test_C = np.reshape(read_csv_data(sources[GET_LOCAL_MAX_INDEX2_INIT][C_ID]), (1, 784))
    #     test_d = read_csv_data(sources[GET_LOCAL_MAX_INDEX2_INIT][D_ID])
    #     test_predicate_lb = read_csv_data(sources[GET_LOCAL_MAX_INDEX2_INIT][PREDICATE_LB_ID])
    #     test_predicate_ub = read_csv_data(sources[GET_LOCAL_MAX_INDEX2_INIT][PREDICATE_UB_ID])
    #     test_im_lb = read_csv_data(sources[GET_LOCAL_MAX_INDEX2_INIT][IM_LB_ID])
    #     test_im_ub = read_csv_data(sources[GET_LOCAL_MAX_INDEX2_INIT][IM_UB_ID])
        
    #     test_star = ImageStar(
    #             test_V, test_C, test_d, test_predicate_lb, test_predicate_ub, test_im_lb, test_im_ub
    #         )
               
    #     test_local_index_input = np.array([int(item) for item in read_csv_data(sources[GET_LOCAL_MAX_INDEX2_INIT][INPUT_LOCAL_MAX_INDEX2_ID])])
    #     test_local_index_output = (read_csv_data(sources[GET_LOCAL_MAX_INDEX2_INIT][OUTPUT_LOCAL_MAX_INDEX2_ID])).tolist()
                         
    #     test_result = test_star.get_localMax_index2(test_local_index_input[0:2], test_local_index_input[2:4], test_local_index_input[4])
                                
    #     self.assertEqual(test_result, test_local_index_output)
        
    # def test_num_attacked_pixels(self):
    #     """
    #         Tests the ImageStar's method that calculates the number of attacked pixels
            
    #         V -> Basis matrix
    #         C -> Predicate matrix
    #         d -> Predicate vector
    #         predicate_lb -> predicate lower bound
    #         predicate_ub -> predicate upper bound

    #         range_output -> valid output range
    #     """
        
    #     test_V = np.reshape(read_csv_data(sources[GET_NUM_ATTACK_PIXELS_INIT][V_ID]), (28, 28, 1, 785))
    #     test_C = np.reshape(read_csv_data(sources[GET_NUM_ATTACK_PIXELS_INIT][C_ID]), (1, 784))
    #     test_d = read_csv_data(sources[GET_NUM_ATTACK_PIXELS_INIT][D_ID])
    #     test_predicate_lb = read_csv_data(sources[GET_NUM_ATTACK_PIXELS_INIT][PREDICATE_LB_ID])
    #     test_predicate_ub = read_csv_data(sources[GET_NUM_ATTACK_PIXELS_INIT][PREDICATE_UB_ID])
        
    #     test_star = ImageStar(
    #             test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
    #         )
               
    #     attacked_pixels_num_output = read_csv_data(sources[GET_NUM_ATTACK_PIXELS_INIT][ATTACKPIXNUM_OUTPUT_ID])[0]
                                
    #     test_result = test_star.get_num_attacked_pixels()
                                
    #     self.assertEqual(test_result, attacked_pixels_num_output)

    # def test_contains_true(self):
    #     """
    #         Tests the ImageStar's range method
            
    #         V -> Basis matrix
    #         C -> Predicate matrix
    #         d -> Predicate vector
    #         predicate_lb -> predicate lower bound
    #         predicate_ub -> predicate upper bound
            
    #         range_input -> input
    #         range_output -> valid output range
    #     """
        
    #     test_V = np.reshape(read_csv_data(sources[GETRANGE_INIT][V_ID]), (28,28,1,785))
    #     test_C = np.reshape(read_csv_data(sources[GETRANGE_INIT][C_ID]), (1, 784))
    #     test_d = read_csv_data(sources[GETRANGE_INIT][D_ID])
    #     test_predicate_lb = read_csv_data(sources[GETRANGE_INIT][PREDICATE_LB_ID])
    #     test_predicate_ub = read_csv_data(sources[GETRANGE_INIT][PREDICATE_UB_ID])
        
    #     test_star = ImageStar(
    #             test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
    #         )
        
    #     range_input = np.array([int(item) for item in read_csv_data(sources[GETRANGE_INIT][INPUT_ID])])
    #     range_output = np.array([read_csv_data(sources[GETRANGE_INIT][OUTPUT_ID])])
        
    #     test_result = test_star.get_range(*range_input)
        
    #     self.assertEqual(test_result.all(), range_output.all())

    # def test_get_ranges(self):
    #     """
    #         Tests the ImageStar's ranges calculation method
            
    #         V -> Basis matrix
    #         C -> Predicate matrix
    #         d -> Predicate vector
    #         predicate_lb -> predicate lower bound
    #         predicate_ub -> predicate upper bound

    #         range_output -> valid output range
    #     """
        
    #     test_V = np.reshape(read_csv_data(sources[GET_RANGES_INIT][V_ID]), (28, 28, 1, 785))
    #     test_C = np.reshape(read_csv_data(sources[GET_RANGES_INIT][C_ID]), (1, 784))
    #     test_d = read_csv_data(sources[GET_RANGES_INIT][D_ID])
    #     test_predicate_lb = read_csv_data(sources[GET_RANGES_INIT][PREDICATE_LB_ID])
    #     test_predicate_ub = read_csv_data(sources[GET_RANGES_INIT][PREDICATE_UB_ID])
        
    #     test_star = ImageStar(
    #             test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
    #         )
        
    #     ranges_output = np.array([read_csv_data(sources[GET_RANGES_INIT][GETRANGES_OUTPUT_ID])])
        
    #     test_result = test_star.get_ranges('linprog', [])
        
    #     self.assertEqual(test_result.all(), ranges_output.all())

    # def test_is_empty_false(self):
    #     """
    #         Checks if the initialized ImageStar is empty
            
    #         V -> Basis matrix
    #         C -> Predicate matrix
    #         d -> Predicate vector
    #         predicate_lb -> predicate lower bound
    #         predicate_ub -> predicate upper bound
    #     """
        
    #     test_V = np.reshape(read_csv_data(sources[IS_EMPTY_INIT][V_ID]), (28,28,1,785))
    #     test_C = np.reshape(read_csv_data(sources[IS_EMPTY_INIT][C_ID]), (1, 784))
    #     test_d = np.array([read_csv_data(sources[IS_EMPTY_INIT][D_ID])])
    #     test_predicate_lb = read_csv_data(sources[IS_EMPTY_INIT][PREDICATE_LB_ID])
    #     test_predicate_ub = read_csv_data(sources[IS_EMPTY_INIT][PREDICATE_UB_ID])
        
    #     test_star = ImageStar(
    #             test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
    #         )
        
    #     completion_flag = True
        
    #     try:
    #         test_result = test_star.is_empty_set()
    #     except Exception as ex:
    #         completion_flag = False
    #         process_exception(ex)
        
    #     self.assertEqual(completion_flag, False)

    # def test_is_empty_true(self):
    #     """
    #         Checks if the empty ImageStar is empty
    #     """
        
    #     test_star = ImageStar()
        
    #     completion_flag = True
        
    #     try:
    #         test_result = test_star.is_empty_set()
    #     except Exception as ex:
    #         completion_flag = False
    #         process_exception(ex)
        
    #     self.assertEqual(completion_flag, True)
        
    # def test_contains_true(self):
    #     """
    #         Tests the ImageStar's projection on the given plain formulated by two points
            
    #         V -> Basis matrix
    #         C -> Predicate matrix
    #         d -> Predicate vector
    #         predicate_lb -> predicate lower bound
    #         predicate_ub -> predicate upper bound
            
    #         test_point1 -> the first point
    #         test_point2 -> the second point
    #     """
        
    #     test_V = np.reshape(read_csv_data(sources[PROJECT2D_INIT][V_ID]), (28,28,1,785))
    #     test_C = np.reshape(read_csv_data(sources[PROJECT2D_INIT][C_ID]), (1, 784))
    #     test_d = np.array([read_csv_data(sources[PROJECT2D_INIT][D_ID])])
    #     test_predicate_lb = read_csv_data(sources[PROJECT2D_INIT][PREDICATE_LB_ID])
    #     test_predicate_ub = read_csv_data(sources[PROJECT2D_INIT][PREDICATE_UB_ID])
        
    #     test_star = ImageStar(
    #             test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
    #         )
        
    #     test_point1 = np.array([4,4,0])#read_csv_data(sources[PROJECT2D_INIT][POINT1_ID])
    #     test_point2 = np.array([3,6,0])#read_csv_data(sources[PROJECT2D_INIT][POINT2_ID])
        
    #     completion_flag = True
        
    #     try:
    #         test_result = test_star.project2D(test_point1, test_point2)
    #     except Exception as ex:
    #         process_exception(ex)
    #         completion_flag = False
            
    #     self.assertEqual(completion_flag, True)
        
    # def test_reshape(self):
    #     """
    #         Tests the ImageStar's method that compares two points of the ImageStar 
    
    #         V -> Basis matrix
    #         C -> Predicate matrix
    #         d -> Predicate vector
    #         predicate_lb -> predicate lower bound
    #         predicate_ub -> predicate upper bound
    
    #         input -> new shape
    #     """
    
    #     test_V = np.reshape(read_csv_data(sources[IS_P1_LARGER_P2_INIT][V_ID]), (28, 28, 1, 785))
    #     test_C = np.reshape(read_csv_data(sources[IS_P1_LARGER_P2_INIT][C_ID]), (1, 784))
    #     test_d = read_csv_data(sources[IS_P1_LARGER_P2_INIT][D_ID])
    #     test_predicate_lb = read_csv_data(sources[IS_P1_LARGER_P2_INIT][PREDICATE_LB_ID])
    #     test_predicate_ub = read_csv_data(sources[IS_P1_LARGER_P2_INIT][PREDICATE_UB_ID])
    
    #     test_star = ImageStar(
    #             test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
    #         )
    
    #     test_result = ImageStar.reshape(test_star, [28, 14, 2])

    # def test_sampling(self):
    #     """
    #         Tests sampling using predicate initialization
        
    #         N : int -> number of images
            
    #         V -> Basis matrix
    #         C -> Predicate matrix
    #         d -> Predicate vector
    #         predicate_lb -> predicate lower bound
    #         predicate_ub -> predicate upper bound
    #     """
        
    #     test_N = 2
        
    #     test_V = np.reshape(read_csv_data(sources[CONSTRUCTOR_PREDICATE_BOUNDARIES_INIT][V_ID]), (28,28,1,785))
    #     test_C = np.reshape(read_csv_data(sources[CONSTRUCTOR_PREDICATE_BOUNDARIES_INIT][C_ID]), (1, 784))
    #     test_d = np.array([read_csv_data(sources[CONSTRUCTOR_PREDICATE_BOUNDARIES_INIT][D_ID])])
    #     test_predicate_lb = read_csv_data(sources[CONSTRUCTOR_PREDICATE_BOUNDARIES_INIT][PREDICATE_LB_ID])
    #     test_predicate_ub = read_csv_data(sources[CONSTRUCTOR_PREDICATE_BOUNDARIES_INIT][PREDICATE_UB_ID])
        
    #     test_star = ImageStar(
    #             test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
    #         )
    #     try:
    #         images = test_star.sample(test_N)
            
    #     except Exception as ex:
    #         completion_flag = False
    #         process_exception(ex)

    #     self.assertEqual(completion_flag, True)
        
    # def test_basic_to_star(self):
    #     """
    #         Tests the initialization with:
    #         lb -> lower bound
    #         ub -> upper bound
    #     """
        
    #     test_lb = read_csv_data(sources[TO_STAR_INIT][TEST_LB_ID])
    #     test_ub = read_csv_data(sources[TO_STAR_INIT][TEST_UB_ID])
    
    #     completion_flag = True
    
    #     try:
    #         test_star = ImageStar(
    #                 test_lb, test_ub
    #             )
            
    #         converted = test_star.to_star()
    #     except Exception as ex:
    #         completion_flag = False
    #         process_exception(ex)

            
    #     self.assertEqual(completion_flag, True)

    # def test_update_ranges(self):
    #     """
    #         Tests the ImageStar's ranges calculation method
            
    #         V -> Basis matrix
    #         C -> Predicate matrix
    #         d -> Predicate vector
    #         predicate_lb -> predicate lower bound
    #         predicate_ub -> predicate upper bound

    #         range_output -> valid output range
    #     """
        
    #     test_V = np.reshape(read_csv_data(sources[UPDATE_RANGES_INIT][V_ID]), (28, 28, 1, 785))
    #     test_C = np.reshape(read_csv_data(sources[UPDATE_RANGES_INIT][C_ID]), (1, 784))
    #     test_d = read_csv_data(sources[UPDATE_RANGES_INIT][D_ID])
    #     test_predicate_lb = read_csv_data(sources[UPDATE_RANGES_INIT][PREDICATE_LB_ID])
    #     test_predicate_ub = read_csv_data(sources[UPDATE_RANGES_INIT][PREDICATE_UB_ID])
        
    #     test_star = ImageStar(
    #             test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
    #         )
        
    #     ranges_input = [np.array([0, 0, 0])]
        
    #     ranges_output = np.array([read_csv_data(sources[UPDATE_RANGES_INIT][UPDATERANGES_OUTPUT_ID])])
                
    #     ranges = test_star.update_ranges(ranges_input)
                
    #     res_flag = True
        
    #     for i in range(len(ranges)):
    #         if ranges[i].all() != ranges_output[i].all():
    #             res_flag = False
    #             break
                
    #     self.assertEqual(res_flag, True)
        
    # def test_is_max(self):
    #     """
    #         Tests the ImageStar's method that compares two points of the ImageStar 
    
    #         V -> Basis matrix
    #         C -> Predicate matrix
    #         d -> Predicate vector
    #         predicate_lb -> predicate lower bound
    #         predicate_ub -> predicate upper bound
    
    #         input -> valid input
    #         local_index -> valid output bounds
    #     """
    #     raise NotImplementedError
    
    #     test_V = np.reshape(read_csv_data(sources[IS_P1_LARGER_P2_INIT][V_ID]), (28, 28, 1, 785))
    #     test_C = np.reshape(read_csv_data(sources[IS_P1_LARGER_P2_INIT][C_ID]), (1, 784))
    #     test_d = read_csv_data(sources[IS_P1_LARGER_P2_INIT][D_ID])
    #     test_predicate_lb = read_csv_data(sources[IS_P1_LARGER_P2_INIT][PREDICATE_LB_ID])
    #     test_predicate_ub = read_csv_data(sources[IS_P1_LARGER_P2_INIT][PREDICATE_UB_ID])
    
    #     test_star = ImageStar(
    #             test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
    #         )
    
    #     test_points_input = np.array([int(item) for item in read_csv_data(sources[IS_P1_LARGER_P2_INIT][INPUT_ID])])
    #     test_points_output = (read_csv_data(sources[IS_P1_LARGER_P2_INIT][OUTPUT_ID])  - 1).tolist()
    
    #     completion_flag = True
        
    #     try:
    #         test_result = test_star.is_p1_larger_p2(test_points_input[0:3], test_points_input[3:6])
    
    #         self.assertEqual(test_result, test_points_output)
    #     except Exception as ex:
    #         completion_flag = False
    #         process_exception(ex)
            
    #     self.assertEqual(completion_flag, True)
        
    # def test_is_p1_larger_p2(self):
    #     """
    #         Tests the ImageStar's method that compares two points of the ImageStar 
    
    #         V -> Basis matrix
    #         C -> Predicate matrix
    #         d -> Predicate vector
    #         predicate_lb -> predicate lower bound
    #         predicate_ub -> predicate upper bound
    
    #         input -> valid input
    #         local_index -> valid output bounds
    #     """
    
    #     #raise NotImplementedError
    
    #     test_V = np.reshape(read_csv_data(sources[IS_P1_LARGER_P2_INIT][V_ID]), (28, 28, 1, 785))
    #     test_C = np.reshape(read_csv_data(sources[IS_P1_LARGER_P2_INIT][C_ID]), (1, 784))
    #     test_d = read_csv_data(sources[IS_P1_LARGER_P2_INIT][D_ID])
    #     test_predicate_lb = read_csv_data(sources[IS_P1_LARGER_P2_INIT][PREDICATE_LB_ID])
    #     test_predicate_ub = read_csv_data(sources[IS_P1_LARGER_P2_INIT][PREDICATE_UB_ID])
    
    #     test_star = ImageStar(
    #             test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
    #         )
            
    #     test_points_input = np.array([int(item) for item in read_csv_data(sources[IS_P1_LARGER_P2_INIT][INPUT_ID])])
    #     test_result = test_star.is_p1_larger_p2(test_points_input[0:3], test_points_input[3:6], 'linprog', [])

    #     self.assertEqual(test_result, True)


if __name__ == '__main__':
    unittest.main()
