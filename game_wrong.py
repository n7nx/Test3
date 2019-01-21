# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from Speaker import Speaker
from Listener import Listener


class Game:
    def __init__(self):
        speaker = Speaker()
		listener = Listener()
		self.target = speaker.data # target
		self.dense_rep_target = speaker.data_encoder(self.target) # dense representation of target
		self.message = speaker.dense2message(self.dense_rep_target) # message
		self.all_candidate = listener.data # all candidates
		self.dense_rep_candidates = listener.data_encoder(self.all_candidate) # dense representation of all candidate
		self.encoded_message = listener.message2dense(self.message) # encoded message
        # sample from Gibbs
		self.likelihoods = tf.nn.softmax(tf.matmul(z, tf.transpose(self.dense_rep_candidate))) # Gibbs distribution of dot product of z and u in U
		self.dist = tf.distributions.Categorical(probs=self.likelihoods)
		self.prediction = self.dist.sample(1) # sample from Gibbs distribution to get target picked
		self.loss = (self.target == self.prediction)
		
            