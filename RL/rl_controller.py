'''
    This file is for the RL controller.
'''

# %%

import logging
import numpy as np
import tensorflow as tf
import sys

from Pruning.pattern_pruning_CPU import build_whole_pattern
from Pruning.precompression_extract_joint_training import train_prune
from RL.rl_input import controller_params, pruning_number_list, block_size
import termplotlib as tpl
import copy
import random
import torch

from utils.runs_number_reward import times_reward
from utils.sparsity_ratio import whole_sparsity_ratio, sparsity_ratio

from RL.rl_input import prune_ratios


# logger = logging.getLogger(__name__)


def ema(values):
    """
    Helper function for keeping track of an exponential moving average of a list of values.
    For this module, we use it to maintain an exponential moving average of rewards
    """
    weights = np.exp(np.linspace(-1., 0., len(values)))
    weights /= weights.sum()
    a = np.convolve(values, weights, mode="full")[:len(values)]# Convolution function library
    return a[-1]


class Controller(object):
    def __init__(self):
        self.graph = tf.Graph()

        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(config=config, graph=self.graph)

        self.model = controller_params['model']
        self.epochs = controller_params['epochs']
        self.hidden_units = controller_params['hidden_units']
        self.timing_constraint = controller_params['timing_constraint']

        self.nn1_search_space = controller_params['sw_space']
        self.level_search_space = controller_params['level_space']

        self.nn1_num_para = len(self.nn1_search_space)
        self.level_num_para = len(self.level_search_space)

        self.num_para = self.nn1_num_para + self.level_num_para# Total number of parameters

        self.nn1_beg, self.nn1_end = 0, self.nn1_num_para# the starting position of nn
        self.level_beg, self.level_end = self.nn1_end, self.nn1_end + self.level_num_para

        self.para_2_val = {}
        idx = 0
        for hp in self.nn1_search_space:
            self.para_2_val[idx] = hp#{idx:sw_space}
            idx += 1
        for hp in self.level_search_space:
            self.para_2_val[idx] = hp
            idx += 1
        # print("---para_2_val:",self.para_2_val)


        self.RNN_classifier = {}
        self.RNN_pred_prob = {}
        with self.graph.as_default():
            self.build_controller()

        self.reward_history = []
        self.architecture_history = []
        self.trained_network = {}

        self.explored_info = {}

    def build_controller(self):
        # logger.info('Building RNN Network')
        # Build inputs and placeholders
        with tf.name_scope('controller_inputs'):#Define a workspace named controller_input and work in it
            # Input to the NASCell    placeholder
            self.child_network_paras = tf.compat.v1.placeholder(tf.int64, [None, self.num_para], name='controller_input')
            # Discounted rewards
            self.discounted_rewards = tf.compat.v1.placeholder(tf.float32, (None,), name='discounted_rewards')
            # WW 12-18: input: the batch_size variable will be used to determine the RNN batch
            self.batch_size = tf.compat.v1.placeholder(tf.int32, [], name='batch_size')

        with tf.name_scope('embedding'):
            self.embedding_weights = []
            # share embedding weights for each type of parameters
            embedding_id = 0
            para_2_emb_id = {}
            for i in range(len(self.para_2_val.keys())):
                additional_para_size = len(self.para_2_val[i])
                additional_para_weights = tf.compat.v1.get_variable('state_embeddings_%d' % (embedding_id),
                                                          shape=[additional_para_size, self.hidden_units],
                                                          initializer=tf.initializers.random_uniform(-1., 1.))
                self.embedding_weights.append(additional_para_weights)
                para_2_emb_id[i] = embedding_id
                embedding_id += 1

            self.embedded_input_list = []
            for i in range(self.num_para):
                self.embedded_input_list.append(
                    tf.nn.embedding_lookup(self.embedding_weights[para_2_emb_id[i]], self.child_network_paras[:, i]))# Select the element corresponding to the index in a tensor
            self.embedded_input = tf.stack(self.embedded_input_list, axis=-1)# Matrix stack
            self.embedded_input = tf.transpose(self.embedded_input, perm=[0, 2, 1])# Transpose and rearrange the output dimensions

        # logger.info('Building Controller')
        with tf.name_scope('controller'):
            with tf.compat.v1.variable_scope('RNN'):
                # A tuple of state_size, zero_state and output state used to store the LSTM unit
                nas = tf.contrib.rnn.NASCell(self.hidden_units)
                tmp_state = nas.zero_state(batch_size=self.batch_size, dtype=tf.float32)
                init_state = tf.compat.v1.nn.rnn_cell.LSTMStateTuple(tmp_state[0], tmp_state[1])

                output, final_state = tf.nn.dynamic_rnn(nas, self.embedded_input, initial_state=init_state,
                                                        dtype=tf.float32)
                tmp_list = []
                for para_idx in range(self.num_para):
                    o = output[:, para_idx, :]
                    para_len = len(self.para_2_val[para_idx])# Choose one in a kind of para
                    classifier = tf.layers.dense(o, units=para_len, name='classifier_%d' % (para_idx), reuse=False)#Fully connected layer
                    self.RNN_classifier[para_idx] = classifier
                    prob_pred = tf.nn.softmax(classifier)# Activation function
                    self.RNN_pred_prob[para_idx] = prob_pred# Classification probability
                    child_para = tf.argmax(prob_pred, axis=-1)# Maximum element index value in each row
                    tmp_list.append(child_para)
                self.pred_val = tf.stack(tmp_list, axis=1)


        # logger.info('Building Optimization')
        # Global Optimization composes all RNNs in one, like NAS, where arch_idx = 0
        with tf.name_scope('Optimizer'):
            self.global_step = tf.Variable(0, trainable=False)
            self.learning_rate = tf.compat.v1.train.exponential_decay(0.99, self.global_step, 50, 0.5, staircase=True)#Exponential decay of lr
            self.optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=self.learning_rate)# Optimizer using RMSProp algorithm

        with tf.name_scope('Loss'):
            # We seperately compute loss of each predict parameter since the dim of predicting parameters may not be same
            for para_idx in range(self.num_para):
                if para_idx == 0:# Output layer classification and loss value calculation of input label
                    self.policy_gradient_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=self.RNN_classifier[para_idx], labels=self.child_network_paras[:, para_idx])
                else:
                    self.policy_gradient_loss = tf.add(self.policy_gradient_loss,
                                                       tf.nn.sparse_softmax_cross_entropy_with_logits(
                                                           logits=self.RNN_classifier[para_idx],
                                                           labels=self.child_network_paras[:, para_idx]))
                # get mean of loss
            self.policy_gradient_loss /= self.num_para
            self.total_loss = self.policy_gradient_loss
            self.gradients = self.optimizer.compute_gradients(self.total_loss)# Calculate the gradient

            # Gradients calculated using REINFORCE
            for i, (grad, var) in enumerate(self.gradients):
                if grad is not None:
                    self.gradients[i] = (grad * self.discounted_rewards, var)# global reward affects gradient
                    # Correct gradient

        with tf.name_scope('Train_RNN'):
            # The main training operation. This applies REINFORCE on the weights of the Controller
            self.train_operation = self.optimizer.apply_gradients(self.gradients)# Apply gradients for computational training
            self.update_global_step = tf.compat.v1.assign(self.global_step, self.global_step + 1, name='update_global_step')# Update value

        # logger.info('Successfully built controller')

    def child_network_translate(self, child_network):# translate the selected parameters
        dnn_out = [[None] * len(child_network[0])]
        for para_idx in range(self.num_para):
            # print("---childnetwork[0]", child_network,para_idx,self.num_para,dnn_out)
            dnn_out[0][para_idx] = (self.para_2_val[para_idx][child_network[0][para_idx]])
        return dnn_out

    def generate_child_network(self, child_network_architecture):# obtain the selected child network and the corresponding parameter
        with self.graph.as_default():
            feed_dict = {
                self.child_network_paras: child_network_architecture,
                self.batch_size: 1
            }
            rnn_out = self.sess.run(self.RNN_pred_prob, feed_dict=feed_dict)
            predict_child = np.array([[0] * self.num_para])
            for para_idx, prob in rnn_out.items():
                predict_child[0][para_idx] = np.random.choice(range(len(self.para_2_val[para_idx])), p=prob[0])#choose child network based on probablity
            hyperparameters = self.child_network_translate(predict_child)
            return predict_child, hyperparameters


    def plot_history(self, history, ylim=(-1, 1), title="reward"):
        x = list(range(len(history)))
        y = history
        fig = tpl.figure()
        fig.plot(x, y, ylim=ylim, width=60, height=20, title=title)
        fig.show()


    def para2interface_NN(self, Para_NN1, Para_level,model,epochs,timing_constraint):# Train the pruning network through the selected sub-network parameters
        #split parameters according to layer name
        idx = 0
        every_layer_dict = {}
        for name in prune_ratios:
            if prune_ratios[name] == 0.0:
                continue
            else:
                energy_level = len(pruning_number_list)
                every_layer_dict[name] = Para_NN1[idx:idx+energy_level*4]
                idx += energy_level * 4
        #choose parameters according to pruning number
        level_para = Para_level[0]
        mask_dict_set = []
        for tag in level_para:
            pruning_rate_dict = {}
            for name in every_layer_dict:
                pruning_rate_dict[name] = every_layer_dict[name][tag*4:(tag+1)*4]
            mask_dict_set.append(pruning_rate_dict)

        every_mask_whole_pattern = []
        for dict in mask_dict_set:
            whole_weight_pattern = build_whole_pattern(model, dict, block_size, device='cuda')
            every_mask_whole_pattern.append(whole_weight_pattern)

        before_sparsity_ratio = whole_sparsity_ratio(model, mask_dict_set, every_mask_whole_pattern, device='cuda')

        #compute every latency
        three_latency,runs_reward = times_reward(before_sparsity_ratio)
        for sub_latency in three_latency:
            if sub_latency > timing_constraint:
                accuracy_reward = -1
                reward = accuracy_reward + runs_reward
                return accuracy_reward, runs_reward,reward,mask_dict_set,level_para
        weighted_accuracy, three_sub_accuracy = train_prune(mask_dict_set, model, epochs,every_mask_whole_pattern)

        accuracy_reward = (weighted_accuracy - 0.85) / (0.9745 - 0.85)
        if three_sub_accuracy[0] > three_sub_accuracy[1] > three_sub_accuracy[2]:
            reward = accuracy_reward + runs_reward
            return accuracy_reward, runs_reward, reward,mask_dict_set,level_para
        else:
            reward = accuracy_reward + runs_reward - 0.5
            return accuracy_reward, runs_reward,reward,mask_dict_set,level_para


    def global_train(self):
        with self.graph.as_default():
            self.sess.run(tf.compat.v1.global_variables_initializer())
        step = 0
        total_rewards = 0
        child_network = np.array([[0] * self.num_para], dtype=np.int64)
        model_replica = copy.deepcopy(self.model)
        spartio = sparsity_ratio(self.model,print_enable=True)


        for episode in range(controller_params['max_episodes']):
            assign_model = copy.deepcopy(model_replica)
            # logger.info(
            #     '=-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-=>Episode {}<=-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-='.format(episode))
            print('=-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-=>Episode {}<=-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-='.format(episode))
            step += 1
            episode_reward_buffer = []
            arachitecture_batch = []

            if episode % 50 == 0 and episode != 0:
                print("************Process:**********", str(float(episode) / controller_params['max_episodes'] * 100) + "%")

            for sub_child in range(controller_params["num_children_per_episode"]):
                # Generate a child network architecture
                child_network, hyperparameters = self.generate_child_network(child_network)

                DNA_NN1 = child_network[0][self.nn1_beg:self.nn1_end]
                DNA_level = child_network[0][self.level_beg:self.level_end]

                Para_NN1 = hyperparameters[0][self.nn1_beg:self.nn1_end]
                Para_level = hyperparameters[0][self.level_beg:self.level_end]

                str_NN1 = " ".join(str(x) for x in Para_NN1)
                str_NN2 = " ".join(str(x) for x in Para_level)
                str_NNs = str_NN1 + " " + str_NN2


                # logger.info("--------->NN: {}".format(str_NNs))
                # print("--------->NN: {}".format(str_NNs))

                # logger.info('=====>Step {}/{} in episode {}: HyperParameters: {} <====='.format(sub_child,
                #                                                                                 controller_params["num_children_per_episode"],
                #                                                                                 episode,
                #                                                                                 hyperparameters))


                if str_NNs in self.explored_info.keys():
                    accuracy_reward = self.explored_info[str_NNs][0]
                    runs_reward = self.explored_info[str_NNs][1]
                    reward = self.explored_info[str_NNs][2]
                    mask_dict_set = self.explored_info[str_NNs][3]
                    level_para = self.explored_info[str_NNs][4]
                else:
                    accuracy_reward,runs_reward,reward,mask_dict_set,level_para = self.para2interface_NN(Para_NN1,Para_level,self.model,self.epochs,self.timing_constraint)
                    self.explored_info[str_NNs] = {}
                    self.explored_info[str_NNs][0] = accuracy_reward
                    self.explored_info[str_NNs][1] = runs_reward
                    self.explored_info[str_NNs][2] = reward
                    self.explored_info[str_NNs][3] = mask_dict_set
                    self.explored_info[str_NNs][4] = level_para


                # logger.info("====================Results=======================")
                # logger.info("--------->Accuracy: {},time reward:{}".format(accuracy, runs_reward))
                # logger.info("--------->Reward: {}".format(reward))
                # logger.info("=" * 90)
                print("====================Results=======================")
                print('--------->Episode: {},sparsity_level:{}'.format(episode,level_para))
                print("--------->Accuracy reward: {},time reward:{}".format(accuracy_reward, runs_reward))
                print("--------->Reward: {}".format(reward))
                print("=" * 90)
                torch.set_printoptions(threshold=15000)
                if episode == controller_params['max_episodes'] - 1:
                    print('--------->mask_dict_set:{}'.format(mask_dict_set))


                for name, weight in assign_model.named_parameters():
                    for original_name, original_weight in self.model.named_parameters():
                        if name == original_name:
                            original_weight.data = weight.data

                episode_reward_buffer.append(reward)# The reward of each sub-network
                identified_arch = np.array(
                    list(DNA_NN1) + list(DNA_level))
                arachitecture_batch.append(identified_arch)# parameter list

            current_reward = np.array(episode_reward_buffer)

            mean_reward = np.mean(current_reward)# Average
            self.reward_history.append(mean_reward)
            self.architecture_history.append(child_network)
            total_rewards += mean_reward

            baseline = ema(self.reward_history)# exponential moving average of rewards
            last_reward = self.reward_history[-1]
            rewards = [last_reward - baseline]#global rewards

            feed_dict = {
                self.child_network_paras: arachitecture_batch,
                self.batch_size: len(arachitecture_batch),
                self.discounted_rewards: rewards
            }

            with self.graph.as_default():# return a context manager, the upper and lower managers use this graph as the default graph
                _, _, loss, lr, gs = self.sess.run(
                    [self.train_operation, self.update_global_step, self.total_loss, self.learning_rate,
                     self.global_step], feed_dict=feed_dict)

            # logger.info('=-=-=-=-=-=>Episode: {} | Loss: {} | LR: {} | Mean R: {} | Reward: {}<=-=-=-=-='.format(
            #     episode, loss, (lr, gs), mean_reward, rewards))
            print('=-=-=-=-=-=>Episode: {} | Loss: {} | LR: {} | Mean R: {} | Reward: {}<=-=-=-=-='.format(
                episode, loss, (lr, gs), mean_reward, rewards))

        print("reward history:",self.reward_history)
        # self.plot_history(self.reward_history, ylim=(min(self.reward_history)-0.01, max(self.reward_history)-0.01))


# %%

seed = 0
torch.manual_seed(seed)
random.seed(seed)
logging.basicConfig(stream=sys.stdout,
                    level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')

# print("Begin")
controller = Controller()
controller.global_train()

# %%