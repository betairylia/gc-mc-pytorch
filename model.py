from __future__ import print_function

import torch.nn as nn
import torch.optim as optim

from layers import *
from metrics import softmax_accuracy, expected_rmse, softmax_cross_entropy



class RecommenderGAE(nn.Module):
    def __init__(self, u_feature, v_features, u_features_nonzero, v_features_nonzero, class_values, dropout, 
                 input_dim, num_classes, num_support,
                 learning_rate, num_basis_functions, hidden, num_users, num_items, accum,
                 self_connections=False, **kwargs):
        super(RecommenderGAE, self).__init__()

        self.inputs = (placeholders['u_features'], placeholders['v_features'])
        self.u_features_nonzero = placeholders['u_features_nonzero']
        self.v_features_nonzero = placeholders['v_features_nonzero']
        self.support = placeholders['support']
        self.support_t = placeholders['support_t']
        self.dropout = placeholders['dropout']
        self.labels = placeholders['labels']
        self.u_indices = placeholders['user_indices']
        self.v_indices = placeholders['item_indices']
        self.class_values = placeholders['class_values']

        self.hidden = hidden
        self.num_basis_functions = num_basis_functions
        self.num_classes = num_classes
        self.num_support = num_support
        self.input_dim = input_dim
        self.self_connections = self_connections
        self.num_users = num_users
        self.num_items = num_items
        self.accum = accum
        self.learning_rate = learning_rate

        # standard settings: beta1=0.9, beta2=0.999, epsilon=1.e-8
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1.e-8)

        self.build()

        moving_average_decay = 0.995
        self.variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay, self.global_step)
        self.variables_averages_op = self.variable_averages.apply(tf.trainable_variables())

        with tf.control_dependencies([self.opt_op]):
            self.training_op = tf.group(self.variables_averages_op)

        self.embeddings = self.activations[2]

        self._rmse()

    def _loss(self):
        self.loss += softmax_cross_entropy(self.outputs, self.labels)

        tf.summary.scalar('loss', self.loss)

    def _accuracy(self):
        self.accuracy = softmax_accuracy(self.outputs, self.labels)

    def _rmse(self):
        self.rmse = expected_rmse(self.outputs, self.labels, self.class_values)

        tf.summary.scalar('rmse_score', self.rmse)

    def _build(self):
        if self.accum == 'sum':
            self.layers.append(OrdinalMixtureGCN(input_dim=self.input_dim,
                                                 output_dim=self.hidden[0],
                                                 support=self.support,
                                                 support_t=self.support_t,
                                                 num_support=self.num_support,
                                                 u_features_nonzero=self.u_features_nonzero,
                                                 v_features_nonzero=self.v_features_nonzero,
                                                 sparse_inputs=True,
                                                 act=tf.nn.relu,
                                                 bias=False,
                                                 dropout=self.dropout,
                                                 share_user_item_weights=True,
                                                 self_connections=False))

        elif self.accum == 'stack':
            self.layers.append(StackGCN(input_dim=self.input_dim,
                                        output_dim=self.hidden[0],
                                        support=self.support,
                                        support_t=self.support_t,
                                        num_support=self.num_support,
                                        u_features_nonzero=self.u_features_nonzero,
                                        v_features_nonzero=self.v_features_nonzero,
                                        sparse_inputs=True,
                                        act=tf.nn.relu,
                                        dropout=self.dropout,
                                        share_user_item_weights=True))
        else:
            raise ValueError('accumulation function option invalid, can only be stack or sum.')

        self.layers.append(Dense(input_dim=self.hidden[0],
                                 output_dim=self.hidden[1],
                                 act=lambda x: x,
                                 dropout=self.dropout,
                                 share_user_item_weights=True))

        self.layers.append(BilinearMixture(num_classes=self.num_classes,
                                           u_indices=self.u_indices,
                                           v_indices=self.v_indices,
                                           input_dim=self.hidden[1],
                                           num_users=self.num_users,
                                           num_items=self.num_items,
                                           user_item_bias=False,
                                           dropout=0.,
                                           act=lambda x: x,
                                           num_weights=self.num_basis_functions,
                                           diagonal=False))


class RecommenderSideInfoGAE(nn.Module):
    def __init__(self,  
                 u_features, v_features, u_features_nonzero, v_features_nonzero, class_values, dropout,
                 input_dim, feat_hidden_dim, num_classes, num_support,
                 learning_rate, num_basis_functions, hidden, num_users, num_items, accum,
                 num_side_features, self_connections=False, **kwargs):
        super(RecommenderSideInfoGAE, self).__init__()

        u_features = self.csr_to_tensor(u_features)
        v_features = self.csr_to_tensor(v_features)
        u_features.requires_grad=False; v_features.requires_grad=False
        self.inputs = (u_features, v_features)

        self.u_features_nonzero = u_features_nonzero
        self.v_features_nonzero = v_features_nonzero
        self.dropout = dropout
        self.class_values = class_values

        #self.support = placeholders['support']
        #self.support_t = placeholders['support_t']
        #self.labels = placeholders['labels']
        #self.u_indices = placeholders['user_indices']
        #self.v_indices = placeholders['item_indices']

        self.num_side_features = num_side_features
        self.feat_hidden_dim = feat_hidden_dim
        #if num_side_features > 0:
        #    self.u_features_side = placeholders['u_features_side']
        #    self.v_features_side = placeholders['v_features_side']
        #else:
        #    self.u_features_side = None
        #    self.v_features_side = None

        self.hidden = hidden
        self.num_basis_functions = num_basis_functions
        self.num_classes = num_classes
        self.num_support = num_support
        self.input_dim = input_dim
        self.self_connections = self_connections
        self.num_users = num_users
        self.num_items = num_items
        self.accum = accum
        self.learning_rate = learning_rate

        self.layers = []
        self.activations = []
        self._build()

        # standard settings: beta1=0.9, beta2=0.999, epsilon=1.e-8
        #self.optimizer = optim.Adam([p for l in self.layers for p in l.parameters()], 
        #                            lr=self.learning_rate, betas=(0.9, 0.999), eps=1.e-8)

        moving_average_decay = 0.995


    def _loss(self, outputs, labels):
        return softmax_cross_entropy(outputs, labels)

    def _accuracy(self, outputs, labels):
        return softmax_accuracy(outputs, labels)

    def _rmse(self, outputs, labels):
        return expected_rmse(outputs, labels, self.class_values)

    def _build(self):
        if self.accum == 'sum':
            self.layers.append(OrdinalMixtureGCN(input_dim=self.input_dim,
                                                 output_dim=self.hidden[0],
                                                 num_support=self.num_support,
                                                 u_features_nonzero=self.u_features_nonzero,
                                                 v_features_nonzero=self.v_features_nonzero,
                                                 sparse_inputs=True,
                                                 act=F.relu,
                                                 bias=False,
                                                 dropout=self.dropout,
                                                 share_user_item_weights=True,
                                                 self_connections=self.self_connections))

        elif self.accum == 'stack':
            self.layers.append(StackGCN(input_dim=self.input_dim,
                                        output_dim=self.hidden[0],
                                        num_support=self.num_support,
                                        u_features_nonzero=self.u_features_nonzero,
                                        v_features_nonzero=self.v_features_nonzero,
                                        sparse_inputs=True,
                                        act=F.relu,
                                        dropout=self.dropout,
                                        share_user_item_weights=True))

        else:
            raise ValueError('accumulation function option invalid, can only be stack or sum.')

        self.layers.append(Dense(input_dim=self.num_side_features,
                                 output_dim=self.feat_hidden_dim,
                                 act=F.relu,
                                 dropout=0.,
                                 bias=True,
                                 share_user_item_weights=False))

        self.layers.append(Dense(input_dim=self.hidden[0]+self.feat_hidden_dim,
                                 output_dim=self.hidden[1],
                                 act=lambda x: x,
                                 dropout=self.dropout,
                                 share_user_item_weights=False))

        self.layers.append(BilinearMixture(num_classes=self.num_classes,
                                           input_dim=self.hidden[1],
                                           num_users=self.num_users,
                                           num_items=self.num_items,
                                           user_item_bias=False,
                                           dropout=0.,
                                           act=lambda x: x,
                                           num_weights=self.num_basis_functions,
                                           diagonal=False))

    def csr_to_tensor(self, data):
        samples = data.shape[0]
        features = data.shape[1]
        values = data.data
        coo_data = data.tocoo()
        indices = torch.LongTensor([coo_data.row, coo_data.col])
        t = torch.sparse.FloatTensor(indices, torch.from_numpy(values).float(), [samples,features])
        return t

    def forward(self, support, support_t, labels, user_indices, item_indices, u_features_side, v_features_side):

        # gcn layer
        layer = self.layers[0]
        gcn_hidden = layer(self.inputs, support, support_t)

        # dense layer for features
        layer = self.layers[1]
        feat_hidden = layer([u_features_side, v_features_side])

        # concat dense layer
        layer = self.layers[2]

        gcn_u = gcn_hidden[0]
        gcn_v = gcn_hidden[1]
        feat_u = feat_hidden[0]
        feat_v = feat_hidden[1]

        input_u = torch.cat(tensors=[gcn_u, feat_u], dim=1)
        input_v = torch.cat(tensors=[gcn_v, feat_v], dim=1)

        concat_hidden = layer([input_u, input_v])

        self.activations.append(concat_hidden)

        # Build sequential layer model
        #for layer in self.layers[3::]:
        #    hidden = layer(self.activations[-1])
        #    self.activations.append(hidden)
        #self.outputs = self.activations[-1]
        layer = self.layers[-1]
        outputs = layer(self.activations[-1], user_indices, item_indices)

        # Build metrics
        loss = self._loss(outputs, labels)
        accuracy = self._accuracy(outputs, labels)
        rmse = self._rmse(outputs, labels)

        #self.optimizer.minimize(self.loss, global_step=self.global_step)

        return accuracy, loss, rmse

