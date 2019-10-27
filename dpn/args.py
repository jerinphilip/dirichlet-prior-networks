
from dpn.models import ConvModel


def add_args(parser):
    # Hparams
    parser.add_argument('--alpha', type=float, required=True)

    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--log_interval', type=int, default=20)
    parser.add_argument('--device', type=str, required=True)
    parser.add_argument('--batch_size', type=int, required=True)

    parser.add_argument('--momentum', type=float, required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--weight_decay', type=float, required=True)

    parser.add_argument('--work_dir', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)

    parser.add_argument('--radius', type=float, default=1.0)
    parser.add_argument('--sigma', type=float, default=1.0)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--num_train_samples', type=int, default=int(1e4))
    parser.add_argument('--num_test_samples', type=int, default=int(1e2))



    # ConvModel.add_args(parser)
