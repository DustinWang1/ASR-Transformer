import torch
import torchaudio
import torch.nn as nn

char_map_str = """
 ' 0
 <SPACE> 1
 a 2
 b 3
 c 4
 d 5
 e 6
 f 7
 g 8
 h 9
 i 10
 j 11
 k 12
 l 13
 m 14
 n 15
 o 16
 p 17
 q 18
 r 19
 s 20
 t 21
 u 22
 v 23
 w 24
 x 25
 y 26
 z 27
 <EOS> 28
 <SOS> 29
 <PAD> 30
 """


class TextTransform:
    def __init__(self):
        self.char_map_str = char_map_str
        self.char_map = {}
        self.index_map = {}
        for line in char_map_str.strip().split('\n'):
            ch, idx = line.split()
            self.char_map[ch] = int(idx)
            self.index_map[int(idx)] = ch
        self.index_map[1] = ' '

    def text_to_int(self, text):
        translation = []
        for c in text:
            if c == ' ':
                ch = self.char_map["<SPACE>"]
            else:
                ch = self.char_map[c]
            translation.append(int(ch))
        return translation

    def int_to_text(self, labels):
        string = []
        for i in labels:
            string.append(self.index_map[i])
        # return joins all values in the array with no separation. Then it
        return ''.join(string)


train_audio_transforms = nn.Sequential(
    torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128),
    torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
    torchaudio.transforms.TimeMasking(time_mask_param=35)
)

valid_audio_transforms = torchaudio.transforms.MelSpectrogram()

text_transform = TextTransform()


def GreedyDecoder(output, labels, label_lengths):
    # output (batch, seq, vocab_size)
    arg_maxes = torch.argmax(output, dim=2)  # (batch, seq)
    decodes = []
    targets = []
    for i, args in enumerate(arg_maxes):
        decode = []
        targets.append(text_transform.int_to_text(labels[i][:label_lengths[i]].tolist()))
        for j, index in enumerate(args):
            decode.append(index.item())
        decodes.append(text_transform.int_to_text(decode))
    return decodes, targets


def data_processing(data, data_type="train"):
    spectrograms = []
    labels = []
    decoder_inputs = []
    label_lengths = []
    for (waveform, _, utterance, _, _, _) in data:
        if data_type == 'train':
            spec = train_audio_transforms(waveform).squeeze(0).transpose(0, 1)
        else:
            spec = valid_audio_transforms(waveform).squeeze(0).transpose(0, 1)
        spectrograms.append(spec)

        label_ = torch.Tensor(text_transform.text_to_int(utterance.lower()))
        label = torch.cat([label_, torch.Tensor([28])])
        decoder_input = torch.cat([torch.Tensor([29]), label_])

        labels.append(label)
        decoder_inputs.append(decoder_input)
        label_lengths.append(len(label))

    # label and decoder outputs should be padded with padding character
    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, True).unsqueeze(1).transpose(2, 3)
    labels = nn.utils.rnn.pad_sequence(labels, padding_value=30, batch_first=True)
    decoder_inputs = nn.utils.rnn.pad_sequence(decoder_inputs, padding_value=30, batch_first=True)

    return spectrograms, decoder_inputs.to(torch.int64), labels.to(torch.int64), label_lengths


class IterMeter(object):
    """keeps track of total iterations"""
    def __init__(self):
        self.val = 0

    def step(self):
        self.val += 1

    def get(self):
        return self.val