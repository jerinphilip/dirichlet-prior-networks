


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
