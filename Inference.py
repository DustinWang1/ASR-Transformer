import torch
import torchaudio
import Model
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model_save = torch.load("./model_save/model_saves", map_location=torch.device('cuda'))

test_dataset = torchaudio.datasets.LIBRISPEECH("./data", url="test-clean", download=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=1,
                                          shuffle=False,
                                          collate_fn=lambda x: utils.data_processing(x, 'valid'))

asr_model = Model.build_transformer().to(device)

# asr_model.load_state_dict(model_save)

with torch.no_grad():
    for x, batch in enumerate(test_loader):
        spectrograms, decoder_inputs, labels, label_lengths = batch
        spectrograms, decoder_inputs, labels = spectrograms.to(device), decoder_inputs.to(device), labels.to(device)

        encoder_output = asr_model.encode(spectrograms)

        # start with "<EOS>" and pass into thing until done
        eos = torch.Tensor([28]).to(torch.int64).unsqueeze(0)
        output = []
        decoder_out = asr_model.decode(encoder_output, eos)
        decoder_ch, _ = utils.GreedyDecoder(decoder_out, labels, label_lengths)
        decoder_out = torch.argmax(decoder_out, dim=2)
        output.append(decoder_ch)

        while decoder_out is not eos:
            decoder_out = asr_model.decode(encoder_output, decoder_out)
            decoder_ch, _ = utils.GreedyDecoder(decoder_out, labels, label_lengths)
            decoder_out = torch.argmax(decoder_out, dim=2)
            output.append(decoder_ch)
        break



