import sys, os
import mat73
import scipy

os.chdir("../../../../")
print(os.getcwd())

sys.path.insert(0, "engine/nn/layers/conv")
from conv2dlayer import *

sys.path.insert(0, "engine/nn/layers/fullyconnected")
from fullyconnectedlayer import *

sys.path.insert(0, "engine/nn/layers/relulayer")
from relulayer import *

sys.path.insert(0, "engine/nn/layers/softmaxlayer")
from softmaxlayer import *

sys.path.insert(0, "engine/nn/layers/imageinput")
from imageinputlayer import *

sys.path.insert(0, "engine/nn/layers/maxpooling")
from maxpooling2dlayer import *

sys.path.insert(0, "engine/nn/cnn")
from cnn import *

nn_path = '/home/crescendo/Workspace/StarV/data/vgg16_matlab'

# nn_loader = NN_Loader('layer_from_folder')

# nn_loader.load(nn_path)



def read_csv_data(path):        
    return np.array(list(mat73.loadmat(path).values())[0])

files_list = os.listdir(nn_path)

layers = []

for i in range(len(files_list)):
    layers.append(-1)

for i in range(len(files_list)):
    [current_num, current_name] = files_list[i].split('_')
    
    current_layer = None
    is_added = False
    
    if current_name == 'Convolution2DLayer':
        name = list(mat73.loadmat(nn_path + '/' + files_list[i] + '/' + 'name.mat').values())[0]
        
        if name == None:
            name = 'conv2d' + str(int(current_num) - 1)
        
        weights = read_csv_data(nn_path + '/' + files_list[i] + '/' + 'weights.mat')
            
        bias = read_csv_data(nn_path + '/' + files_list[i] + '/' + 'bias.mat')
        if len(bias.shape) == 1:
            bias = np.reshape(bias, (1, 1, bias.shape[0]))
    
        filter_size = read_csv_data(nn_path + '/' + files_list[i] + '/' + 'filter_size.mat')
        num_filters = read_csv_data(nn_path + '/' + files_list[i] + '/' + 'num_filters.mat')
        num_channel = read_csv_data(nn_path + '/' + files_list[i] + '/' + 'num_channel.mat')
        
        padding_size = read_csv_data(nn_path + '/' + files_list[i] + '/' + 'padding_size.mat')
        stride = read_csv_data(nn_path + '/' + files_list[i] + '/' + 'stride.mat')
        dilation = read_csv_data(nn_path + '/' + files_list[i] + '/' + 'dilation_factor.mat')
        
        current_layer = Conv2DLayer(name, weights, bias, padding_size, stride, dilation)
        is_added = True
    elif current_name == 'FullyConnectedLayer':
        name = list(mat73.loadmat(nn_path + '/' + files_list[i] + '/' + 'name.mat').values())[0]
        
        weights = read_csv_data(nn_path + '/' + files_list[i] + '/' + 'weights.mat')
        bias = read_csv_data(nn_path + '/' + files_list[i] + '/' + 'bias.mat')

        current_layer = FullyConnectedLayer(name, weights, bias)
        is_added = True
    elif current_name == 'MaxPooling2DLayer':
        name = list(mat73.loadmat(nn_path + '/' + files_list[i] + '/' + 'name.mat').values())[0]
        
        pool_size = read_csv_data(nn_path + '/' + files_list[i] + '/' + 'pool_size.mat')
        stride = read_csv_data(nn_path + '/' + files_list[i] + '/' + 'stride.mat')
        padding_size = read_csv_data(nn_path + '/' + files_list[i] + '/' + 'padding_size.mat')

        if len(pool_size.shape) == 1:
            pool_size = np.reshape(pool_size, (1, pool_size.shape[0]))
            
        if len(stride.shape) == 1:
            stride = np.reshape(stride, (1, stride.shape[0]))
            
        if len(padding_size.shape) == 1:
            padding_size = np.reshape(padding_size, (1, padding_size.shape[0]))
            
        current_layer = MaxPooling2DLayer(name, pool_size, stride, padding_size)
        is_added = True
    elif current_name == 'ReLULayer':
        current_layer = ReLULayer()
        is_added = True
    elif current_name == 'SoftmaxLayer':
        current_layer = SoftmaxLayer()
        is_added = True
    elif current_name == 'ImageInputLayer':
        name = list(mat73.loadmat(nn_path + '/' + files_list[i] + '/' + 'name.mat').values())[0]
        
        input_size = read_csv_data(nn_path + '/' + files_list[i] + '/' + 'input_size.mat')
        input_size = np.append(input_size[2], input_size[0:2]).astype('int')
        mean = torch.permute(torch.FloatTensor(read_csv_data(nn_path + '/' + files_list[i] + '/' + 'mean.mat')), (2,0,1)).cpu().detach().numpy()
        
        current_layer = ImageInputLayer(name, input_size, mean)
        is_added = True
        
    if(is_added):
        layers[int(current_num) - 1] = current_layer
        is_added = False
    
layers = [i for i in layers if i != -1]
            
cnn = CNN(layers) 
        
images = scipy.io.loadmat("/home/crescendo/Workspace/StarV/data/examples/CNN/VGG16/DEEPFOOL_Attack/pepper_ori_images.mat")

img = torch.permute(torch.FloatTensor(images['ori_images_1']), (2,0,1)).cpu().detach().numpy()

cnn.evaluate(img)
