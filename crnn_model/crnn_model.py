
from typing import Tuple
import tensorflow as tf
from tensorflow.contrib import rnn
import tensorflow.contrib.slim as slim
from crnn_model import cnn_basenet

filter_size_10x10 =[10,10]
filter_size_3x3 = [3,3]
filter_size_1x1 = [1,1]


class ShadowNet(cnn_basenet.CNNBaseModel):
    """
        Implement the crnn model for sequence recognition
    """
    def __init__(self, phase: str, hidden_nums: int, layers_nums: int, num_classes: int):
        """

        :param phase: 'Train' or 'Test'
        :param hidden_nums: Number of hidden units in each LSTM cell (block)
        :param layers_nums: Number of LSTM cells (blocks)
        :param num_classes: Number of classes (different symbols) to detect
        """
        super(ShadowNet, self).__init__()
        self.__phase = phase
        self.__hidden_nums = hidden_nums
        self.__layers_nums = layers_nums
        self.__num_classes = num_classes

    @property
    def phase(self):
        """

        :return:
        """
        return self.__phase

    @phase.setter
    def phase(self, value: str):
        """

        :param value:
        :return:
        """
        if not isinstance(value, str) or value.lower() not in ['test', 'train']:
            raise ValueError('value should be a str \'Test\' or \'Train\'')
        self.__phase = value.lower()
 
    def __conv_stage(self, inputdata: tf.Tensor, out_dims: int, name: str=None): #-> tf.Tensor:
        """ Standard VGG convolutional stage: 2d conv, relu, and max_pool2d

        :param inputdata: 4D tensor batch x width x height x channels
        :param out_dims: number of output channels / filters
        :return: the max_pool2ded output of the stage
        """
        conv = self.conv2d(inputdata=inputdata, out_channel=out_dims, kernel_size=3, stride=2, use_bias=False, name=name)
        bn = slim.batch_norm(conv, scope = "entry_block1_batch_norm1")
        relu = tf.nn.relu(bn, name = "entry_block1_relu1")
        #max_pool = self.maxpooling(inputdata=relu, kernel_size=2, stride=2)
        return relu
#------------------------------ modified by U. S. Saidrasul -------------------------------
    def __feature_sequence_extraction(self, inputdata: tf.Tensor): # -> Xception network
        """ Implements section 2.1 of the paper: "Feature Sequence Extraction"

        :param inputdata: eg. batch*64*128*1 
        :return:
        """
        conv1 = self.__conv_stage(inputdata=inputdata, out_dims=64, name='conv1')
        residual = slim.conv2d(conv1, 128, filter_size_1x1, stride = 2, scope = 'entry_block1_res_conv')
        residual = slim.batch_norm(residual, scope = 'entry_block1_res_batch_norm')
        
        #block 2
        layer = tf.nn.relu(conv1, name= 'entry_block2_relu1')
        layer = slim.separable_conv2d(layer, 128, filter_size_3x3, depth_multiplier=1, scope = 'entry_block2_conv1')
        layer = slim.batch_norm(layer, scope='entry_block2_batch_norm1')
        

        layer = tf.nn.relu(layer, name = 'entry_block2_relu2')
        layer = slim.separable_conv2d(layer, 128, filter_size_3x3,depth_multiplier=1, scope='entry_block2_conv2')
        layer = slim.batch_norm(layer, scope = 'entry_block2_batch_norm2')
        
        layer = slim.max_pool2d(layer, filter_size_3x3, stride = 2, padding = 'same', scope = 'entry_block2_max_pool2d')
        
        layer = tf.math.add(layer, residual, name='entry_block2_add')
     
        residual = slim.conv2d(layer,256, filter_size_1x1, stride = 2, scope = 'entry_block2_res_conv')
        residual = slim.batch_norm(residual, scope='entry_block2_res_batch_norm')
        print("residual",residual.shape)
        #block 3
        layer = tf.nn.relu(layer, name = "entry_block3_relu")
        layer = slim.separable_conv2d(layer, 256, filter_size_3x3, depth_multiplier=1, scope = 'entry_block3_conv1')
        layer = slim.batch_norm(layer, scope='entry_block3_batch_norm1')
        
        layer = tf.nn.relu(layer, name = 'entry_block3_relu1')
        layer = slim.separable_conv2d(layer, 256, filter_size_3x3, depth_multiplier=1, scope ='entry_block3_conv2')
        layer = slim.batch_norm(layer, scope= 'entry_block3_batch_norm2')
        
        layer = slim.max_pool2d(layer, filter_size_3x3, stride = 2, padding = 'same', scope = 'entry_block3_max_pool2d')
        print ("layer_", layer.shape)
        
        layer = tf.math.add(layer, residual, name = "entry_block3_add")
        
        residual = slim.conv2d(layer, 728, filter_size_1x1, stride=[2,1], scope='entry_block3_res_conv')
        residual = slim.batch_norm(residual, scope = 'entry_block3_res_batch_norm')

        #block 4
        layer = tf.nn.relu(layer, name = "entry_block4_relu")
        layer = slim.separable_conv2d(layer, 728, filter_size_3x3, depth_multiplier=1, scope = 'entry_block4_conv1')
        layer = slim.batch_norm(layer, scope='entry_block4_batch_norm1')
        
        layer = tf.nn.relu(layer, name = 'entry_block4_relu1')
        layer = slim.separable_conv2d(layer, 728, filter_size_3x3, depth_multiplier=1, scope ='entry_block4_conv2')
        layer = slim.batch_norm(layer, scope= 'entry_block4_batch_norm2')
        
        layer = slim.max_pool2d(layer, filter_size_3x3, stride = [2,1], padding = 'same', scope = 'entry_block4_max_pool2d')
        
        layer = tf.math.add(layer, residual, name = "entry_block4_add")
        #end of Entry Flow

        #__________Middle Flow start from here:____________
        #in Middle flow we create several identical blocks and they all have same layers
        #first we initialize number of blocks and then using for loop we create identical blocks 

        blocks_num = 8
        for blocks in range(blocks_num):
            name_prefix = 'middle_block%s_'%(str(blocks+5))
            residual = layer
            layer = tf.nn.relu(layer, name=name_prefix+'relu1')
            layer = slim.separable_conv2d(layer, 728,filter_size_3x3, depth_multiplier=1, scope = name_prefix+'conv1')
            layer = slim.batch_norm(layer, scope=name_prefix+'batch_norm1')
           
            layer = tf.nn.relu(layer, name=name_prefix+'relu2')
            layer = slim.separable_conv2d(layer, 728, filter_size_3x3, depth_multiplier=1, scope=name_prefix+'conv2')
            layer = slim.batch_norm(layer, scope=name_prefix+'batch_norm2')
            
            layer = tf.nn.relu(layer, name=name_prefix+'relu3')
            layer = slim.separable_conv2d(layer, 728, filter_size_3x3, depth_multiplier=1, scope=name_prefix+'conv3')
            layer = slim.batch_norm(layer, scope=name_prefix+'batch_norm3')

            layer = tf.math.add(layer, residual, name=name_prefix+'add')
        #end of Middle Flow
        #__________Exit Flow start from here:____________
        #in Exit Flow blocks are smaller and in each block we increment the number of features 
        #block 1
        residual = slim.conv2d(layer, 1024, filter_size_1x1, stride=[2,1], scope = 'exit_block1_conv1')
        residual = slim.batch_norm(residual, scope='exit_block1_res_batch_norm')

        layer = tf.nn.relu(layer, name ='exit_block1_relu')
        layer = slim.separable_conv2d(layer, 728, filter_size_3x3, depth_multiplier=1, scope='exit_block1_conv2')
        layer = slim.batch_norm(layer, scope = 'exit_block1_batch_norm')

        #block 2
        layer = tf.nn.relu(layer, name ='exit_block2_relu')
        layer = slim.separable_conv2d(layer, 1024, filter_size_3x3, depth_multiplier=1, scope='exit_block2_conv')
        layer = slim.batch_norm(layer, scope = 'exit_block2_batch_norm')
        
        layer = slim.max_pool2d(layer, filter_size_3x3, stride = [2,1], padding = 'same', scope = 'exit_block2_max_pool2d')
        layer = tf.math.add(layer, residual, name = 'exit_block2_add')
        #block 3
        layer = slim.separable_conv2d(layer, 1536, filter_size_3x3, depth_multiplier=1, scope = 'exit_block3_conv') 
        layer = slim.batch_norm(layer, scope = 'exit_block3_batch_norm')
        #block 4
        layer = tf.nn.relu(layer, name = "exit_block4_relu")
        layer = slim.separable_conv2d(layer, 2048, filter_size_3x3, depth_multiplier=1, scope='exit_block4_conv')
        layer = slim.batch_norm(layer, scope = 'exit_block4_batch_norm')
        layer = tf.nn.relu(layer, name='exit_block5_relu')
        #layer = slim.avg_pool2d(layer, filter_size_10x10, stride=[2,1],scope = 'exit_block5_avg_pool')
        layer = slim.conv2d(layer, 2048, [2,2], stride= [2,1], scope = 'exit_block5_conv1')
        relu7 = tf.nn.relu(layer, name='last_relu')  # batch*1*25*512
        
        print("--------------------------relu----------------------------------------------------",relu7.shape)
        return relu7
#------------------------------ modified by U. S. Saidrasul -------------------------------

    def __map_to_sequence(self, inputdata: tf.Tensor): # -> tf.Tensor:
        """ Implements the map to sequence part of the network.

        This is used to convert the CNN feature map to the sequence used in the stacked LSTM layers later on.
        Note that this determines the lenght of the sequences that the LSTM expects
        :param inputdata:
        :return:
        """
        shape = inputdata.get_shape().as_list()
        assert shape[1] == 1  # H of the feature map must equal to 1
        return self.squeeze(inputdata=inputdata, axis=1)

    def __sequence_label(self, inputdata: tf.Tensor):# -> Tuple[tf.Tensor, tf.Tensor]:
        """ Implements the sequence label part of the network
        
        :param inputdata:
        :return:
        """
        with tf.variable_scope('LSTMLayers'):
            # construct stack lstm rcnn layer
            # forward lstm cell
            fw_cell_list = [rnn.BasicLSTMCell(nh, forget_bias=1.0) for nh in [self.__hidden_nums]*self.__layers_nums]
            # Backward direction cells
            bw_cell_list = [rnn.BasicLSTMCell(nh, forget_bias=1.0) for nh in [self.__hidden_nums]*self.__layers_nums]

            stack_lstm_layer, _, _ = rnn.stack_bidirectional_dynamic_rnn(fw_cell_list, bw_cell_list, inputdata,
                                                                         dtype=tf.float32)

            if self.phase.lower() == 'train':
                stack_lstm_layer = self.dropout(inputdata=stack_lstm_layer, keep_prob=0.5)

            [batch_s, _, hidden_nums] = inputdata.get_shape().as_list()  # [batch, width, 2*n_hidden]
            rnn_reshaped = tf.reshape(stack_lstm_layer, [-1, hidden_nums])  # [batch x width, 2*n_hidden]

            w = tf.Variable(tf.truncated_normal([hidden_nums, self.__num_classes], stddev=0.1), name="w")
            # Doing the affine projection

            logits = tf.matmul(rnn_reshaped, w)

            logits = tf.reshape(logits, [batch_s, -1, self.__num_classes])

            raw_pred = tf.argmax(tf.nn.softmax(logits), axis=2, name='raw_prediction')

            # Swap batch and batch axis
            rnn_out = tf.transpose(logits, (1, 0, 2), name='transpose_time_major')  # [width, batch, n_classes]

        return rnn_out, raw_pred

    def build_shadownet(self, inputdata: tf.Tensor): #-> tf.Tensor:
        """ Main routine to construct the network

        :param inputdata:
        :return:
        """
        # first apply the cnn feature extraction stage
        cnn_out = self.__feature_sequence_extraction(inputdata=inputdata)

        # second apply the map to sequence stage
        sequence = self.__map_to_sequence(inputdata=cnn_out)

        # third apply the sequence label stage
        net_out, raw_pred = self.__sequence_label(inputdata=sequence)

        return net_out
