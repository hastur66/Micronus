from tensorflow.keras.layers import Layer, Dropout, Input
from tensorflow.keras.models import Model
from micronus.multihead_attention import MultiHeadAttention
from micronus.positional_encoding import PositionEmbeddingFixedWeights
from micronus.encoder import AddNormalization, FeedForward
 

class DecoderLayer(Layer):
    """ Define Decoder layer """
    def __init__(self, sequence_length, h, d_k, d_v, d_model, d_ff, rate, **kwargs):
        super(DecoderLayer, self).__init__(**kwargs)
        self.build(input_shape=[None, sequence_length, d_model])
        self.d_model = d_model
        self.sequence_length = sequence_length
        self.multihead_attention1 = MultiHeadAttention(h, d_k, d_v, d_model)
        self.dropout1 = Dropout(rate)
        self.add_norm1 = AddNormalization()
        self.multihead_attention2 = MultiHeadAttention(h, d_k, d_v, d_model)
        self.dropout2 = Dropout(rate)
        self.add_norm2 = AddNormalization()
        self.feed_forward = FeedForward(d_ff, d_model)
        self.dropout3 = Dropout(rate)
        self.add_norm3 = AddNormalization()
 
    def build_graph(self):
        """ Build Decoder layer graph """
        input_layer = Input(shape=(self.sequence_length, self.d_model))
        return Model(inputs=[input_layer], outputs=self.call(input_layer, None, True))
    
    def call(self, x, encoder_output, lookahead_mask, padding_mask, training):
        """

        Parameters
        ----------
        x : int
        Number of self-attention heads.

        encoder_output :
            Output from the encoder.
    
        lookahead_mask :
            Masking condition for the self-attention layer.

        padding_mask :
            Padding condition for the encoder output.

        training :
            Flag indicating whether to apply Dropout layers during training.

        Returns
        -------
        output:
            Decoder layer output.

        """
        multihead_output1 = self.multihead_attention1(x, x, x, lookahead_mask)
        multihead_output1 = self.dropout1(multihead_output1, training=training)
        addnorm_output1 = self.add_norm1(x, multihead_output1)

        multihead_output2 = self.multihead_attention2(addnorm_output1, encoder_output, encoder_output, padding_mask)
        multihead_output2 = self.dropout2(multihead_output2, training=training)
        addnorm_output2 = self.add_norm1(addnorm_output1, multihead_output2)
 
        feedforward_output = self.feed_forward(addnorm_output2)
        feedforward_output = self.dropout3(feedforward_output, training=training)
 
        return self.add_norm3(addnorm_output2, feedforward_output)
 

class Decoder(Layer):
    """ Define Decoder block """

    def __init__(self, vocab_size, sequence_length, h, d_k, d_v, d_model, d_ff, n, rate, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.pos_encoding = PositionEmbeddingFixedWeights(sequence_length, vocab_size, d_model)
        self.dropout = Dropout(rate)
        self.decoder_layer = [DecoderLayer(sequence_length, h, d_k, d_v, d_model, d_ff, rate) for _ in range(n)]
 
    def call(self, output_target, encoder_output, lookahead_mask, padding_mask, training):
        """

        Parameters
        ----------
        output_target :
            target output
            
        encoder_output :
            encoder output
            
        lookahead_mask :
            masking condition

        padding_mask :
            padding condition
            
        training :
            apply the Dropout layers
            

        Returns
        -------
        output :
            decoder block output

        """
        pos_encoding_output = self.pos_encoding(output_target)
        x = self.dropout(pos_encoding_output, training=training)
 
        for i, layer in enumerate(self.decoder_layer):
            x = layer(x, encoder_output, lookahead_mask, padding_mask, training)
 
        return x
