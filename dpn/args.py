


def add_args(parser):
    # Hparams
    parser.add_argument('--alpha', type=float, required=True)

    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--device', type=int, required=True)

    parser.add_argument('--momentum', type=float, required=True)
    parser.add_argument('--lr', type=float, required=True)
