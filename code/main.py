import argparse
import time

from openhgnn.experiment import Experiment

if __name__ == '__main__':
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='TransE', type=str, help='name of models')
    parser.add_argument('--task', '-t', default='link_prediction', type=str, help='name of task')
    # link_prediction / node_classification
    parser.add_argument('--dataset', '-d', default='NeoDTI-Net', type=str, help='name of datasets')
    parser.add_argument('--gpu', '-g', default='0', type=int, help='-1 means cpu')
    parser.add_argument('--use_best_config', action='store_true', help='will load utils.best_config')
    parser.add_argument('--load_from_pretrained', action='store_true', help='load model from the checkpoint')
    args = parser.parse_args()

    experiment = Experiment(model=args.model, dataset=args.dataset, task=args.task, gpu=args.gpu,
                            use_best_config=args.use_best_config, load_from_pretrained=args.load_from_pretrained)

    experiment.run()
    end = time.time()
    print("time: ", (end-start), 's')