from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}

def orthogonal(shape, scale=1.1, name=None):
	flat_shape = (shape[0], np.prod(shape[1:]))
	a = np.random.normal(0.0, 1.0, flat_shape)
	u, _, v = np.linalg.svd(a, full_matrices=False)

	q = u if u.shape == flat_shape else v
	q = q.reshape(shape)
	return Parameter(torch.from_numpy(scale * q[:shape[0], :shape[1]])).float()

def dot(x, y, sparse=False):
	"""Wrapper for tf.matmul (sparse vs dense)."""
	if sparse:
		res = torch.sparse.mm(x, y)
	else:
		res = torch.mm(x, y)
	return res


def get_layer_uid(layer_name=''):
	"""Helper function, assigns unique layer IDs
	"""
	if layer_name not in _LAYER_UIDS:
		_LAYER_UIDS[layer_name] = 1
		return 1
	else:
		_LAYER_UIDS[layer_name] += 1
		return _LAYER_UIDS[layer_name]


def dropout_sparse(x, keep_prob, num_nonzero_elems):
	"""Dropout for sparse tensors. Currently fails for very large sparse tensors (>1M elements)
	"""
	noise_shape = [num_nonzero_elems]
	random_tensor = keep_prob
	random_tensor += Parameter(torch.randn(noise_shape))#tf.random_uniform(noise_shape)
	dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
	pre_out = tf.sparse_retain(x, dropout_mask)

	return pre_out * tf.div(1., keep_prob)


class Dense(nn.Module):
	"""Dense layer for two types of nodes in a bipartite graph. """

	def __init__(self, input_dim, output_dim, dropout=0., act=F.relu, share_user_item_weights=False,
				 bias=False, **kwargs):
		super(Dense, self).__init__(**kwargs)

		if not share_user_item_weights:
			self.weights_u = Parameter(torch.randn(input_dim, output_dim))
			self.weights_v = Parameter(torch.randn(input_dim, output_dim))
			if bias:
				self.user_bias = Parameter(torch.randn(output_dim))
				self.item_bias = Parameter(torch.randn(output_dim))

		else:
			self.weights_u = Parameter(torch.randn(input_dim, output_dim))
			self.weights_v = self.weights_u
			if bias:
				self.user_bias = Parameter(torch.randn(output_dim))
				self.item_bias = self.user_bias

		self.bias = bias

		self.dropout = dropout
		self.act = act

		for p in [self.weights_u, self.weights_v]:
			nn.init.xavier_normal_(p)

	def forward(self, inputs):
		x_u = inputs[0]
		x_u = F.dropout(x_u, self.dropout)
		x_u = torch.mm(x_u, self.weights_u)

		x_v = inputs[1]
		x_v = F.dropout(x_v, self.dropout)
		x_v = torch.mm(x_v, self.weights_v)

		u_outputs = self.act(x_u)
		v_outputs = self.act(x_v)

		if self.bias:
			u_outputs += self.user_bias
			v_outputs += self.item_bias

		return u_outputs, v_outputs


class StackGCN(nn.Module):
	"""Graph convolution layer for bipartite graphs and sparse inputs."""

	def __init__(self, input_dim, output_dim, num_support, u_features_nonzero=None,
				 v_features_nonzero=None, sparse_inputs=False, dropout=0.,
				 act=F.relu, share_user_item_weights=True, **kwargs):
		super(StackGCN, self).__init__(**kwargs)

		assert output_dim % num_support == 0, 'output_dim must be multiple of num_support for stackGC layer'

		self.weights_u = Parameter(torch.randn(num_support, input_dim, output_dim//num_support))
		if not share_user_item_weights:
			self.weights_v = Parameter(torch.randn(num_support, input_dim, output_dim//num_support))
		else:
			self.weights_v = self.weights_u

		self.dropout = dropout

		self.sparse_inputs = sparse_inputs
		self.u_features_nonzero = u_features_nonzero
		self.v_features_nonzero = v_features_nonzero
		if sparse_inputs:
			assert u_features_nonzero is not None and v_features_nonzero is not None, \
				'u_features_nonzero and v_features_nonzero can not be None when sparse_inputs is True'

		#self.support = tf.sparse_split(axis=1, num_split=num_support, sp_input=support)
		#self.support_transpose = tf.sparse_split(axis=1, num_split=num_support, sp_input=support_t)

		self.act = act

		for p in [self.weights_u, self.weights_v]:
			nn.init.xavier_normal_(p)

	def forward(self, inputs, support, support_transpose):
		x_u = inputs[0]
		x_v = inputs[1]
		
		'''
		if self.sparse_inputs:
			x_u = dropout_sparse(x_u, 1 - self.dropout, self.u_features_nonzero)
			x_v = dropout_sparse(x_v, 1 - self.dropout, self.v_features_nonzero)
		else:
			x_u = F.dropout(x_u, self.dropout)
			x_v = F.dropout(x_v, self.dropout)
		'''

		supports_u = []
		supports_v = []

		for i in range(len(support)):
			tmp_u = dot(x_u, self.weights_u[i], sparse=self.sparse_inputs)
			tmp_v = dot(x_v, self.weights_v[i], sparse=self.sparse_inputs)

			support_ = support[i]
			support_transpose_ = support_transpose[i]

			supports_u.append(torch.sparse.mm(support_, tmp_v))
			supports_v.append(torch.sparse.mm(support_transpose_, tmp_u))

		z_u = torch.cat(dim=1, tensors=supports_u)
		z_v = torch.cat(dim=1, tensors=supports_v)

		u_outputs = self.act(z_u)
		v_outputs = self.act(z_v)

		return u_outputs, v_outputs


class OrdinalMixtureGCN(nn.Module):

	"""Graph convolution layer for bipartite graphs and sparse inputs."""

	def __init__(self, input_dim, output_dim, num_support, u_features_nonzero=None,
				 v_features_nonzero=None, sparse_inputs=False, dropout=0.,
				 act=F.relu, bias=False, share_user_item_weights=False, self_connections=False, **kwargs):
		super(OrdinalMixtureGCN, self).__init__(**kwargs)

		self.weights_u = Parameter(torch.randn(num_support, input_dim, output_dim//num_support))
		if not share_user_item_weights:
			self.weights_v = Parameter(torch.randn(num_support, input_dim, output_dim//num_support))
		else:
			self.weights_v = self.weights_u

		self.dropout = dropout

		self.sparse_inputs = sparse_inputs
		self.u_features_nonzero = u_features_nonzero
		self.v_features_nonzero = v_features_nonzero
		if sparse_inputs:
			assert u_features_nonzero is not None and v_features_nonzero is not None, \
				'u_features_nonzero and v_features_nonzero can not be None when sparse_inputs is True'

		self.self_connections = self_connections

		self.bias = bias
		#support = tf.sparse_split(axis=1, num_split=num_support, sp_input=support)
		#support_t = tf.sparse_split(axis=1, num_split=num_support, sp_input=support_t)

		'''
		if self_connections:
			self.support = support[:-1]
			self.support_transpose = support_t[:-1]
			self.u_self_connections = support[-1]
			self.v_self_connections = support_t[-1]
			self.weights_u = self.weights_u[:-1]
			self.weights_v = self.weights_v[:-1]
			self.weights_u_self_conn = self.weights_u[-1]
			self.weights_v_self_conn = self.weights_v[-1]

		else:
			self.support = support
			self.support_transpose = support_t
			self.u_self_connections = None
			self.v_self_connections = None
			self.weights_u_self_conn = None
			self.weights_v_self_conn = None

		self.support_nnz = []
		self.support_transpose_nnz = []
		for i in range(self.support)):
			nnz = tf.reduce_sum(tf.shape(self.support[i].values))
			self.support_nnz.append(nnz)
			self.support_transpose_nnz.append(nnz)
		'''

		self.act = act

	def forward(self, inputs, support, support_t):

		'''
		if self.sparse_inputs:
			x_u = dropout_sparse(inputs[0], 1 - self.dropout, self.u_features_nonzero)
			x_v = dropout_sparse(inputs[1], 1 - self.dropout, self.v_features_nonzero)
		else:
			x_u = F.dropout(inputs[0], self.dropout)
			x_v = F.dropout(inputs[1], self.dropout)
		'''

		supports_u = []
		supports_v = []

		# self-connections with identity matrix as support
		if self.self_connections:
			uw = dot(x_u, self.weights_u_self_conn, sparse=self.sparse_inputs)
			supports_u.append(torch.sparse.mm(self.u_self_connections, uw))

			vw = dot(x_v, self.weights_v_self_conn, sparse=self.sparse_inputs)
			supports_v.append(torch.sparse.mm(self.v_self_connections, vw))

		wu = 0.
		wv = 0.
		for i in range(len(self.support)):
			wu += self.weights_u[i]
			wv += self.weights_v[i]

			# multiply feature matrices with weights
			tmp_u = dot(x_u, wu, sparse=self.sparse_inputs)

			tmp_v = dot(x_v, wv, sparse=self.sparse_inputs)

			support = self.support[i]
			support_transpose = self.support_transpose[i]

			# then multiply with rating matrices
			supports_u.append(torch.sparse.mm(support, tmp_v))
			supports_v.append(torch.sparse.mm(support_transpose, tmp_u))

		#z_u = tf.add_n(supports_u)
		#z_v = tf.add_n(supports_v)

		if self.bias:
			z_u += self.bias_u
			z_v += self.bias_v

		u_outputs = self.act(z_u)
		v_outputs = self.act(z_v)

		return u_outputs, v_outputs


class BilinearMixture(nn.Module):
	"""
	Decoder model layer for link-prediction with ratings
	To use in combination with bipartite layers.
	"""

	def __init__(self, num_classes, input_dim, num_users, num_items, user_item_bias=False,
				 dropout=0., act=F.softmax, num_weights=3,
				 diagonal=True, **kwargs):
		super(BilinearMixture, self).__init__(**kwargs)

		weights = []
		for i in range(num_weights):
			if diagonal:
				#  Diagonal weight matrices for each class stored as vectors
				weights.append(weight_variable_random_uniform(1, input_dim, name='weights_%d' % i))

			else:
				weights.append(orthogonal([input_dim, input_dim]))
		self.weights = torch.stack(weights, 0)

		self.weights_scalars = Parameter(torch.randn(num_weights, num_classes))

		if user_item_bias:
			self.user_bias = Parameter(torch.randn(num_users, num_classes))
			self.item_bias = Parameter(torch.randn(num_items, num_classes))

		self.user_item_bias = user_item_bias

		if diagonal:
			self._multiply_inputs_weights = torch.multiply
		else:
			self._multiply_inputs_weights = torch.matmul

		self.num_classes = num_classes
		self.num_weights = num_weights

		self.dropout = dropout
		self.act = act

		for p in [self.weights, self.weights_scalars]:#, self.user_bias, self.item_bias]:
			nn.init.xavier_normal_(p)

	def forward(self, inputs, u_indices, v_indices):

		u_inputs = F.dropout(inputs[0], self.dropout)
		v_inputs = F.dropout(inputs[1], self.dropout)

		u_inputs = torch.index_select(u_inputs, 0, u_indices)
		v_inputs = torch.index_select(v_inputs, 0, v_indices)

		if self.user_item_bias:
			u_bias = torch.index_select(self.user_bias, 0, u_indices)
			v_bias = torch.index_select(self.item_bias, 0, v_indices)
		else:
			u_bias = None
			v_bias = None

		basis_outputs = []
		for i in range(self.num_weights):

			u_w = self._multiply_inputs_weights(u_inputs, self.weights[i])
			x = torch.sum(torch.mul(u_w, v_inputs), dim=1)

			basis_outputs.append(x)

		# Store outputs in (Nu x Nv) x num_classes tensor and apply activation function
		basis_outputs = torch.stack(basis_outputs, dim=1)

		outputs = torch.mm(basis_outputs,  self.weights_scalars)

		if self.user_item_bias:
			outputs += u_bias
			outputs += v_bias

		outputs = self.act(outputs)

		return outputs

