import pytest
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from micronus.transformer_model import TransformerModel
from micronus.inference import Infer

# inferencing_model = TransformerModel()

# # Sentence to translate
# # sentence = ['im thirsty']

# # # Load the trained model's weights at the specified epoch
# # inferencing_model.load_weights('weights/wghts16.ckpt')
 

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



# @pytest.fixture
# def infer_instance():
#     # Create a mock inferencing_model
#     mock_inferencing_model = MagicMock()

#     # Initialize an instance of the Infer class with the mock inferencing_model
#     return Infer(mock_inferencing_model)


# def test_load_tokenizer(infer_instance):
#     # TODO: Implement this test
#     pass


# def test_call(infer_instance):
#     # Mock the tokenizer methods and necessary attributes
#     mock_enc_tokenizer = MagicMock()
#     mock_dec_tokenizer = MagicMock()

#     # Replace the load_tokenizer method with MagicMock that returns the mock_enc_tokenizer
#     infer_instance.load_tokenizer = MagicMock(return_value=mock_enc_tokenizer)

#     # Mock tokenizer methods
#     mock_enc_tokenizer.texts_to_sequences = MagicMock(return_value=[[1, 2, 3]])
#     mock_dec_tokenizer.texts_to_sequences = MagicMock(side_effect=[[[4]], [[5]]])

#     # Mock the decoder_output object (TensorArray)
#     mock_decoder_output = MagicMock()
#     mock_decoder_output.stack = MagicMock(return_value=np.array([[4], [5]]))
#     infer_instance.TensorArray = MagicMock(return_value=mock_decoder_output)

#     # Set the enc_seq_length and dec_seq_length values
#     infer_instance.enc_seq_length = 3
#     infer_instance.dec_seq_length = 2

#     # Call the __call__ method with a mock input sentence
#     sentence = ["Test sentence"]
#     output_str = infer_instance(sentence)

#     # Assert that the tokenizer methods were called with the correct arguments
#     mock_enc_tokenizer.texts_to_sequences.assert_called_once_with(["<START> Test sentence <EOS>"])
#     mock_dec_tokenizer.texts_to_sequences.assert_has_calls([MagicMock(["<START>"]), MagicMock(["<EOS>"])])

#     # Assert that the correct values were written to the decoder_output
#     mock_decoder_output.write.assert_has_calls([MagicMock(0, [[4]]), MagicMock(1, [[5]])])

#     # Assert that the mock_inferencing_model.predict method was called correctly
#     infer_instance.transformer.assert_called_once_with(mock_enc_tokenizer, mock_decoder_output.stack(), training=False)

#     # TODO: Add more assertions to verify the correctness of the output_str

#     # Clean up
#     mock_enc_tokenizer.reset_mock()
#     mock_dec_tokenizer.reset_mock()
#     infer_instance.transformer.reset_mock()
#     mock_decoder_output.reset_mock()

class TestInfer(unittest.TestCase):
    def setUp(self):
        # Create a mock inferencing_model
        self.mock_inferencing_model = MagicMock()

        # Initialize an instance of the Infer class with the mock inferencing_model
        self.infer_instance = Infer(self.mock_inferencing_model)

    def test_load_tokenizer(self):
        # TODO: Implement this test
        pass

    def test_call(self):
        # Mock the tokenizer methods and necessary attributes
        mock_enc_tokenizer = MagicMock()
        mock_dec_tokenizer = MagicMock()

        # Replace the load_tokenizer method with MagicMock that returns the mock_enc_tokenizer
        self.infer_instance.load_tokenizer = MagicMock(return_value=mock_enc_tokenizer)

        # Mock tokenizer methods
        mock_enc_tokenizer.texts_to_sequences = MagicMock(return_value=[[1, 2, 3]])
        mock_dec_tokenizer.texts_to_sequences = MagicMock(side_effect=[[[4]], [[5]]])

        # Mock the decoder_output object (TensorArray)
        mock_decoder_output = MagicMock()
        mock_decoder_output.stack = MagicMock(return_value=np.array([[4], [5]]))
        self.infer_instance.TensorArray = MagicMock(return_value=mock_decoder_output)

        # Set the enc_seq_length and dec_seq_length values
        self.infer_instance.enc_seq_length = 3
        self.infer_instance.dec_seq_length = 2

        # Call the __call__ method with a mock input sentence
        sentence = ["Test sentence"]
        output_str = self.infer_instance(sentence)

        # Assert that the tokenizer methods were called with the correct arguments
        mock_enc_tokenizer.texts_to_sequences.assert_called_once_with(["<START> Test sentence <EOS>"])
        mock_dec_tokenizer.texts_to_sequences.assert_has_calls([MagicMock(["<START>"]), MagicMock(["<EOS>"])])

        # Assert that the correct values were written to the decoder_output
        mock_decoder_output.write.assert_has_calls([MagicMock(0, [[4]]), MagicMock(1, [[5]])])

        # Assert that the mock_inferencing_model.predict method was called correctly
        self.mock_inferencing_model.assert_called_once_with(mock_enc_tokenizer, mock_decoder_output.stack(), training=False)

        # TODO: Add more assertions to verify the correctness of the output_str

        # Clean up
        mock_enc_tokenizer.reset_mock()
        mock_dec_tokenizer.reset_mock()
        self.mock_inferencing_model.reset_mock()
        mock_decoder_output.reset_mock()
