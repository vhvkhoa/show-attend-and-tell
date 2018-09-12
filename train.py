from core.solver import CaptioningSolver
from core.model import CaptionGenerator
from core.utils import load_coco_data


def main():
    # load train dataset
    data = load_coco_data(data_path='./data', split='train')
    word_to_idx = data['word_to_idx']
    # load val dataset to print out scores every epoch
    val_data = load_coco_data(data_path='./data', split='val')

    model = CaptionGenerator(word_to_idx, dim_feature=[196, 1024], dim_embed=512,
                                       dim_hidden=1536, n_time_step=31, prev2out=True,
                                                 ctx2out=True, alpha_c=1.0, selector=True, dropout=True)

    solver = CaptioningSolver(model, data, val_data, n_epochs=20, batch_size=64, update_rule='rmsprop',
                                    learning_rate=0.001, print_every=10, eval_every=200, save_every=200, image_path='./image/',
                                    pretrained_model=None, start_from=None, model_path='model/lstm/', 
                                    test_model='model/lstm/model-10',
                                    print_score=True, log_path='log/log_0/')

    solver.train(beam_size=3)

if __name__ == "__main__":
    main()
