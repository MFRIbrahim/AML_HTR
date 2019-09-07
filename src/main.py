import torch
from torchvision import transforms

from beam_search import ctcBeamSearch
from dataset import get_data_loaders
from model import Net
from school import TrainingEnvironment, Trainer, evaluate_model
from transformations import GrayScale, Rescale, ToTensor
from util import WordDeEnCoder, TimeMeasure
from word_prediction import BeamDecoder, BestPathDecoder


def get_available_device():
    if torch.cuda.is_available():
        device = 'cuda'
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        device = 'cpu'

    return device


def dynamic_learning_rate(epoch):
    if epoch < 10:
        return 0.01
    elif epoch >= 10:
        return 0.001
    else:
        return 0.00005


def main():
    with TimeMeasure(enter_msg="Setup everything", exit_msg="Setup finished after {} ms."):
        torch.manual_seed(0)
        device = get_available_device()
        print("Active device:", device)

        # Here we use '|' as a symbol the CTC-blank
        char_list = list("| !\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
        de_en_coder = WordDeEnCoder(char_list)
        word_predictor = BeamDecoder(char_list)
        word_predictor_debug = BestPathDecoder(char_list)

        height = 32
        width = 128
        max_word_length = 32
        transformation = transforms.Compose([GrayScale(),
                                             Rescale(height, width, max_word_length),
                                             ToTensor(char_to_int=de_en_coder.char_to_idx)
                                             ])
        meta_path = "../dataset/words.txt"
        images_path = "../dataset/images"
        relative_train_size = 0.6
        batch_size = 50

        train_loader, test_loader = get_data_loaders(meta_path,
                                                     images_path,
                                                     transformation,
                                                     relative_train_size,
                                                     batch_size)

        retrain_model = True
        model = Net(dropout=0.2).to(device)
        environment = TrainingEnvironment(max_epochs=10, warm_start=False, optimizer_args={"weight_decay": 0})

        trainer = Trainer("net",
                          word_predictor_debug,
                          dynamic_learning_rate=dynamic_learning_rate,
                          environment=environment
                          )

    with TimeMeasure(enter_msg="Get trained model.", exit_msg="Obtained trained model after {} ms."):
        if retrain_model:
            with TimeMeasure(enter_msg="Begin Training.", exit_msg="Finished complete training after {} ms."):
                trainer.train(model, train_loader, device=device)
        else:
            with TimeMeasure(enter_msg="Load pre-trained model.", exit_msg="Finished loading after {} ms."):
                trainer.load_latest_model_state_into(model)

    with TimeMeasure(enter_msg="Evaluate model:", exit_msg="Evaluation finished after {} ms."):
        evaluate_model(msg="test accuracy: {:7.4f}",
                       word_prediction=word_predictor,
                       model=model,
                       data_loader=test_loader,
                       device=device
                       )

        evaluate_model(msg="train accuracy: {:7.4f}",
                       word_prediction=word_predictor,
                       model=model,
                       data_loader=train_loader,
                       device=device
                       )


if __name__ == "__main__":
    main()
