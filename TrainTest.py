import torch
import torchaudio
import torch.nn as nn
from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model
from torchmetrics.text import CharErrorRate
from tqdm import tqdm
import utils


def train(model, device, train_loader, criterion, optimizer, scheduler, epoch, iter_meter, experiment):
    model.train()
    data_len = len(train_loader.dataset)
    with experiment.train():
        batch_iterator = tqdm(train_loader, desc=f"Processing")
        for batch_idx, _data in enumerate(batch_iterator):
            spectrograms, labels, input_lengths, label_lengths = _data
            spectrograms, labels = spectrograms.to(device), labels.to(device)

            optimizer.zero_grad()

            output = model(spectrograms, labels)  # OUT (batch, time, n_class)
            output = nn.functional.log_softmax(output, dim=2)

            loss = criterion(output.view(-1, 28), labels.view(-1).to(torch.long))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
            loss.backward()

            experiment.log_metric('loss', loss.item(), step=iter_meter.get())
            experiment.log_metric('learning_rate', scheduler.get_last_lr(), step=iter_meter.get())

            optimizer.step()
            scheduler.step()
            iter_meter.step()
            if batch_idx % 100 == 0 or batch_idx == data_len:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(spectrograms), data_len,
                    100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader, criterion, epoch, iter_meter, experiment):
    print('\nevaluating…')
    model.eval()
    test_loss = 0
    test_cer = []
    with experiment.test():
        with torch.no_grad():
            cer = CharErrorRate()
            for I, _data in enumerate(test_loader):
                spectrograms, labels, input_lengths, label_lengths = _data
                spectrograms, labels = spectrograms.to(device), labels.to(device)

                output = model(spectrograms, labels)  # (batch, time, n_class)
                output = nn.functional.log_softmax(output, dim=2)
                output = output.transpose(0, 1) # (time, batch, n_class)

                loss = criterion(output.reshape(-1, 28), labels.view(-1).to(torch.long))
                test_loss += loss.item() / len(test_loader)

                decoded_preds, decoded_targets = utils.GreedyDecoder(output.transpose(0, 1), labels, label_lengths)
                for j in range(len(decoded_preds)):
                    test_cer.append(cer(decoded_preds[j], decoded_targets[j]))

    avg_cer = sum(test_cer)/len(test_cer)
    experiment.log_metric('test_loss', test_loss, step=iter_meter.get())
    experiment.log_metric('cer', avg_cer, step=iter_meter.get())

    print('Test set: Average loss: {:.4f}, Average CER: {:.4f}\n'.format(test_loss,  avg_cer))