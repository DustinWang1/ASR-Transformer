import torch
import torch.nn as nn
import os
import sys
import torchaudio
from comet_ml import Experiment

import TrainTest
import Model
import utils

def main(learning_rate=5e-4,
         batch_size=5,
         epochs=10,
         train_url="train-clean-100",
         test_url="test-clean",
         experiment=Experiment(
             api_key="RDMwDBjNyTYsfyWcwNEzZQulc",
             project_name="general",
             workspace="dustin-wang"
         )
         ):

    hparams = {
        "n_class": 29,
        "n_feats": 128,
        "d_model": 512,
        "stride": 1,
        "dropout": 0.1,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs
    }

    log_file = open('./logs', 'w')

    # Redirect sys.stdout to the log file
    sys.stdout = log_file

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(7)
    device = torch.device("cuda" if use_cuda else "cpu")

    if not os.path.isdir("./data"):
        os.makedirs("./data")

    train_dataset = torchaudio.datasets.LIBRISPEECH("./data", url=train_url, download=True)
    test_dataset = torchaudio.datasets.LIBRISPEECH("./data", url=test_url, download=True)

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=hparams['batch_size'],
                                               shuffle=True,
                                               collate_fn=lambda x: utils.data_processing(x, 'train'),
                                               **kwargs) # For each batch of samples run this function on them
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=hparams['batch_size'],
                                              shuffle=False,
                                              collate_fn=lambda x: utils.data_processing(x, 'valid'),
                                              **kwargs)
    # Data Loader -> (spectrograms, labels, input_lengths, output_lengths)
    # spectrograms -> (batch, channel, n_mels, seq_len)
    # labels -> (batch, seq_len)

    # with torch.no_grad():
    #     specs, labels, input_lengths, output_lengths = next(iter(train_loader))
    #     model = Model.build_transformer()
    #     output = model(specs, labels)
    #     print(output)

    model = Model.build_transformer()
    optimizer = torch.optim.AdamW(model.parameters(), hparams['learning_rate'])
    criterion = nn.CTCLoss(blank=28).to(device)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=hparams['learning_rate'],
                                                    steps_per_epoch=int(len(train_loader)),
                                                    epochs=hparams['epochs'],
                                                    anneal_strategy='linear')

    iter_meter = utils.IterMeter()
    for epoch in range(1, epochs + 1):
        TrainTest.train(model, device, train_loader, criterion, optimizer, scheduler, epoch, iter_meter, experiment)
        TrainTest.test(model, device, test_loader, criterion, epoch, iter_meter, experiment)
        experiment.end()
    torch.save(model.state_dict(), "./model_saves")


if __name__ == '__main__':
    main()
