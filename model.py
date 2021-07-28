import torch
import torch.nn as nn
import statistics
import torchvision.models as models ## used to load the pytorch models for vision


class EncoderCNN(nn.Module): ## encoder class used for the CNN part
    def __init__(self, embed_size, train_CNN=False):
        super(EncoderCNN, self).__init__()
        self.train_CNN = train_CNN  ## we use just a pre-trained model.
        self.inception = models.inception_v3(pretrained=True, aux_logits=False) ## we use the inception model.
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embed_size) ## fully connected, access last linear layer and replace it with linear and map it to embed size.
        self.relu = nn.ReLU()
        self.times = []
        self.dropout = nn.Dropout(0.5)

    def forward(self, images): ## take input image and compute features with inception of images.
        features = self.inception(images)
        return self.dropout(self.relu(features))


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size) # we need embedding here to map our word to get better representation of word.
                                                          # It will take an index and map into some embed size.
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers) ## LSTM model is build here.
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions): ## features and target caption in dataset.
        embeddings = self.dropout(self.embed(captions))
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0) # concatinate the features with the embedding and on dimension 0..
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs


class CNNtoRNN(nn.Module): # cnn to rnn is hooked here.
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CNNtoRNN, self).__init__()
        self.encoderCNN = EncoderCNN(embed_size)
        self.decoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        features = self.encoderCNN(images)
        outputs = self.decoderRNN(features, captions)
        return outputs

    def caption_image(self, image, vocabulary, max_length=50):
        result_caption = []

        with torch.no_grad():
            x = self.encoderCNN(image).unsqueeze(0)
            states = None

            for _ in range(max_length): ## upto the max_length words prediction, 50 here.
                hiddens, states = self.decoderRNN.lstm(x, states)  ## at beginning it will be initialized as 0.
                output = self.decoderRNN.linear(hiddens.squeeze(0))
                predicted = output.argmax(1) # so we taking word with higgest probablity.
                result_caption.append(predicted.item())
                x = self.decoderRNN.embed(predicted).unsqueeze(0) # taking the predicted word.

                if vocabulary.itos[predicted.item()] == "<EOS>": # check if vocab is equal to end of sentence then break.
                    break

        return [vocabulary.itos[idx] for idx in result_caption]