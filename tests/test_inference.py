import pytest
import numpy as np
from micronus.transformer_model import TransformerModel
from micronus.inference import Infer

inferencing_model = TransformerModel()

# Sentence to translate
sentence = ['im thirsty']

# # Load the trained model's weights at the specified epoch
# inferencing_model.load_weights('weights/wghts16.ckpt')
 

# class TestInfer():
#     # def test_init(self):
#     #     # Create a new instance of the 'Infer' class
#     #     translator = Infer(inferencing_model)
#     #     assert isinstance(translator, Infer)

    
#     # def test_call(self):
#     #     # Create a new instance of the 'Infer' class
#     #     translator = Infer(inferencing_model)
        
#     #     # Translate the input sentence
#     #     print(translator(sentence))

#     def test_load_tokenizer(self):
#         # Create a new instance of the 'Infer' class
#         translator = Infer(inferencing_model)
#         translator.load_tokenizer()
#         assert translator.tokenizer is not None

class TestInfer():

    def test_call(self):
        # Create a new instance of the 'Infer' class
        # translator = Infer(inferencing_model)

        # Translate the input sentence
        # print(translator(sentence))
        print(sentence)