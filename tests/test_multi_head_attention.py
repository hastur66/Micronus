from numpy import random
import pytest
from micronus.multihead_attention import MultiHeadAttention, DotProductAttention

input_seq_length = 5  # Maximum length of the input sequence
h = 8  # Number of self-attention heads
d_k = 64  # Dimensionality of the linearly projected queries and keys
d_v = 64  # Dimensionality of the linearly projected values
d_model = 512  # Dimensionality of the model sub-layers' outputs
batch_size = 64  # Batch size from the training process
 
queries = random.random((batch_size, input_seq_length, d_k))
keys = random.random((batch_size, input_seq_length, d_k))
values = random.random((batch_size, input_seq_length, d_v))

attention = DotProductAttention()
output_att = attention(queries, keys, values, d_k)

multihead_attention = MultiHeadAttention(h, d_k, d_v, d_model)
output_multi = multihead_attention(queries, keys, values)

class TestDotProductAttention():
    def test_call(self):
        assert output_att.shape == (64, 5, 64)

class TestMultiHeadAttention():
    
    def test_call(self):
        assert output_multi.shape == (64, 5, 512)
