'''
    This is the pytorch version of reinforcement learning controller
'''
import argparse

from Gate_predictor.gate_number import Power_MLP
from NAS.NAS_Binary_Model.Binary_MLP import MLP2, MLP1, MLP3

import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F

from NAS.NAS_Binary_Model.Binary_Model_train import parse_args, RL_reward
from NAS.NAS_Binary_Model.Binary_rl_input import controller_parameter


def ema(values):
    """
    Helper function for keeping track of an exponential moving average of a list of values.
    For this module, we use it to maintain an exponential moving average of rewards
    """
    weights = np.exp(np.linspace(-1., 0., len(values)))
    weights /= weights.sum()
    a = np.convolve(values, weights, mode="full")[:len(values)]# convolution function library
    return a[-1]


class Controller(object):
    def __init__(self):
        self.hidden_units = controller_parameter['hidden_units'] # the hidden size of RNN controller
        self.train_args = parse_args()

        # types of parameters search space
        self.layer_search_space = controller_parameter['layer']
        self.arch_search_space = controller_parameter['Arch']

        # the number for each type of parameters
        self.layer_num_para = len(self.layer_search_space)
        self.arch_num_para = len(self.arch_search_space)

        # the begin and end position for each type of parameter
        self.num_para = self.layer_num_para + self.arch_num_para  # total parameter number
        self.layer_beg, self.layer_end = 0, self.layer_num_para # # begin of arch parameter
        self.arch_beg, self.arch_end = self.layer_end, self.layer_end + self.arch_num_para

        # the directory of search space
        self.para_2_val = {}
        idx = 0
        for hp in self.layer_search_space:
            self.para_2_val[idx] = hp # {idx:sw_space}
            idx += 1
        for hp in self.arch_search_space:
            self.para_2_val[idx] = hp
            idx += 1

        self.RNN_classifier = {}
        self.RNN_pred_prob = {}
        self.explored_info = {}

        self.reward_history = []
        self.architecture_history = []

    def init_embedding(self):
        self.embedding_weights = []
        # share embedding weights for each type of parameters
        embedding_id = 0
        self.para_2_emb_id = {}
        for i in range(len(self.para_2_val.keys())):
            additional_para_size = len(self.para_2_val[i])
            additional_para_weights = nn.Embedding(additional_para_size, self.hidden_units)
            additional_para_weights.weight.data.uniform_(-1., 1.)
            # print('---embedding type:', type(additional_para_weights))
            self.embedding_weights.append(additional_para_weights)
            self.para_2_emb_id[i] = embedding_id
            embedding_id += 1
        # print('**embedding 1')

    def embedding(self, child_network_paras):
        # build the embedding of inputs
        self.embedded_input_list = []
        child_network_paras = torch.from_numpy(child_network_paras).to(torch.int64)
        for i in range(self.num_para):
            # print('child_network_paras:', child_network_paras[:, i])
            self.embedded_input_list.append(self.embedding_weights[self.para_2_emb_id[i]].weight.data.index_select(0, child_network_paras[:, i]))
        self.embedded_input = torch.stack(self.embedded_input_list, dim=-1)
        self.embedded_input = self.embedded_input.permute([2, 0, 1]) # transpose input to [seq_len, batch_size, embedding_size]
        # print('**embedding2')

    def rnn_model(self):
        self.rnn = nn.LSTM(self.hidden_units, self.hidden_units) # the default initial state is zero
        # print('**rnn1')

    def rnn_forward(self, model, embedded_input):
        output, final_state = model(embedded_input) # output: (seq_len, batch, num_directions * hidden_size)
        # print('---rnn output:', output, output.shape)
        tmp_list = []
        for para_idx in range(self.num_para):
            o = (output[para_idx, :, :]) # transpose to [batch size, in_feature]
            para_len = len(self.para_2_val[para_idx])
            classifier = nn.Linear(o.shape[-1], para_len)(o) # output shape: [batch_size, para_len]
            self.RNN_classifier[para_idx] = classifier
            prob_pred = F.softmax(classifier, dim=1) # classify probability
            # print('---rnn softmax:', prob_pred, prob_pred.shape)
            self.RNN_pred_prob[para_idx] = prob_pred
            child_para = prob_pred.argmax(dim=-1) # find the index of maximum for each line
            tmp_list.append(child_para)
            # print('argmax tmp_list:', tmp_list)
        self.pred_val = torch.stack(tmp_list, dim=1)
        # print('--pred_val:', self.pred_val, self.pred_val.shape)
        # print('**rnn2')


    def rnn_train(self, child_network_paras, discounted_rewards, criterion):
        for para_idx in range(self.num_para):
            if para_idx == 0:
                # print(type(self.RNN_pred_prob[para_idx]), type(torch.tensor(child_network_paras[:, para_idx])))
                self.policy_gradient_loss = criterion(self.RNN_classifier[para_idx], torch.tensor(child_network_paras[:, para_idx]).to(torch.long))
            else:
                self.policy_gradient_loss = torch.add(self.policy_gradient_loss, criterion(self.RNN_classifier[para_idx], torch.tensor(child_network_paras[:, para_idx]).to(torch.long)))
        # get mean of loss
        self.policy_gradient_loss /= self.num_para
        self.total_loss = self.policy_gradient_loss
        # print('**loss')
        # optimizer
        self.optimizer.zero_grad()
        # print('**optimizer')
        self.total_loss.backward()
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad = p.grad * discounted_rewards # Gradients calculated using REINFORCE
        # print('**gradient')
        self.optimizer.step()
        self.scheduler.step()
        self.global_step += 1
        # print('**train rnn')

    def child_network_translate(self, child_network):# translate the selected parameters
        dnn_out = [[None] * len(child_network[0])]
        for para_idx in range(self.num_para):
            # print("---childnetwork[0]", child_network,para_idx,self.num_para,dnn_out)
            dnn_out[0][para_idx] = (self.para_2_val[para_idx][child_network[0][para_idx]])
        return dnn_out

    def generate_child_network(self, child_network_architecture):
        self.embedding(child_network_architecture)
        self.rnn_forward(self.rnn, self.embedded_input)
        rnn_out = self.RNN_pred_prob
        # print(rnn_out)
        predict_child = np.array([[0] * self.num_para])
        for para_idx, prob in rnn_out.items():
            predict_child[0][para_idx] = np.random.choice(range(len(self.para_2_val[para_idx])), p=prob[0].detach().numpy())  # choose child network based on probablity
        hyperparameters = self.child_network_translate(predict_child)
        return predict_child, hyperparameters

    def para2interface_NN(self, Para_layer, Para_arch):
        reward = 0
        return reward


    def global_train(self):
        step = 0
        total_rewards = 0
        child_network = np.array([[0] * self.num_para], dtype=np.int64)
        self.init_embedding()
        self.rnn_model()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.RMSprop(self.rnn.parameters(), lr=0.99)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.5)
        self.global_step = 0

        for episode in range(controller_parameter['max_episodes']):
            print('=-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-=>Episode {}<=-=-==-=-==-=-==-=-==-=-==-=-==-=-==-=-='.format(episode))
            step += 1
            episode_reward_buffer = []
            arachitecture_batch = []

            if episode % 50 == 0 and episode != 0:
                print("************Process:**********", str(float(episode) / controller_parameter['max_episodes'] * 100) + "%")

            for sub_child in range(controller_parameter["num_children_per_episode"]):
                # Generate a child network architecture
                child_network, hyperparameters = self.generate_child_network(child_network)

                DNA_layer = child_network[0][self.layer_beg: self.layer_end]
                DNA_arch = child_network[0][self.arch_beg:self.arch_end]

                # translate the parameters of controller
                Para_layer = hyperparameters[0][self.layer_beg: self.layer_end]
                Para_arch = hyperparameters[0][self.arch_beg:self.arch_end]
                print(Para_layer, Para_arch)

                # generate the recode string
                str_layer = " ".join(str(x) for x in Para_layer)
                str_arch = " ".join(str(x) for x in Para_arch)
                str_NNs = str_layer + " " + str_arch

                if str_NNs in self.explored_info.keys():
                    _layer = self.explored_info[str_NNs][0]
                    _architecture = self.explored_info[str_NNs][1]
                    reward = self.explored_info[str_NNs][2]
                    reward_acc = self.explored_info[str_NNs][3]
                    reward_power = self.explored_info[str_NNs][4]
                else:
                    _layer, _architecture, reward, reward_acc, reward_power = self.para2interface_NN(Para_layer, Para_arch)
                    self.explored_info[str_NNs] = {}
                    self.explored_info[str_NNs][0] = _layer
                    self.explored_info[str_NNs][1] = _architecture
                    self.explored_info[str_NNs][2] = reward
                    self.explored_info[str_NNs][3] = reward_acc
                    self.explored_info[str_NNs][4] = reward_power

                print("====================Results=======================")
                print('--------->Episode[{}]:layer:({}), architecture:({})'.format(episode, _layer, _architecture))
                print("--------->Reward: {}, accuracy reward: {}, power reward: {}".format(reward, reward_acc, reward_power))
                print("=" * 90)

                episode_reward_buffer.append(reward)  # 每个子network的reward
                identified_arch = np.array(list(DNA_layer) + list(DNA_arch))
                arachitecture_batch.append(identified_arch)  # 参数列表

            current_reward = np.array(episode_reward_buffer)
            mean_reward = np.mean(current_reward)  # 求平均
            self.reward_history.append(mean_reward)
            self.architecture_history.append(child_network)
            total_rewards += mean_reward

            baseline = ema(self.reward_history)  # reward指数平均移动
            last_reward = self.reward_history[-1]
            rewards = last_reward - baseline  # global rewards

            arachitecture_batch = np.array(arachitecture_batch)
            self.embedding(arachitecture_batch)
            self.rnn_forward(self.rnn, self.embedded_input)
            self.rnn_train(arachitecture_batch, rewards, self.criterion)
            loss, lr, gs = self.total_loss, self.scheduler.get_last_lr(), self.global_step
            print('=-=-=-=-=-=>Episode: {} | Loss: {} | LR: {} | Mean R: {} | Reward: {}<=-=-=-=-='.format(
                episode, loss, (lr, gs), mean_reward, rewards))
        print("reward history:", self.reward_history)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
controller = Controller()
controller.global_train()