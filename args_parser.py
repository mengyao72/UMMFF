import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-root', type=str, default='datas')
    parser.add_argument('-dataset', type=str, default='Washington',
                        choices=['PaviaU', 'Botswana', 'Washington','salinas_corrected'])
    parser.add_argument('--scale_ratio', type=float, default=8)
    parser.add_argument('--n_bands', type=int, default=0)
    parser.add_argument('--n_select_bands', type=int, default=5)

    parser.add_argument('--model_path', type=str,
                        default='./checkpoints/dataset_arch.pkl',
                        help='path for trained encoder')
    parser.add_argument('--train_dir', type=str, default='./model/dataset/train',
                        help='directory for resized images')
    parser.add_argument('--val_dir', type=str, default='./model/dataset/val',
                        help='directory for resized images')

    # learning settingl
    parser.add_argument('--n_epochs', type=int, default=10000,
                        help='end epoch for training')
    # rsicd: 3e-4, ucm: 1e-4,
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--image_size', type=int, default=128)

    args = parser.parse_args()
    return args
