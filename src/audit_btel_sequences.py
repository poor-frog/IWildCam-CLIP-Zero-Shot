import sys

from src.config import parse_arguments
from src.datasets.iwildcam import IWildCam, IWildCamVal
from src.models.btel_artifacts import audit_sequences, print_sequence_audit


def main() -> None:
    args = parse_arguments()
    train_data = IWildCam(
        None,
        location=args.data_location,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )
    val_data = IWildCamVal(
        None,
        location=args.data_location,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )
    num_classes = len(train_data.classnames)
    print_sequence_audit(
        audit_sequences(
            train_data.train_dataset,
            split_name="train",
            num_classes=num_classes,
            classnames=train_data.classnames,
        )
    )
    print_sequence_audit(
        audit_sequences(
            val_data.test_dataset,
            split_name="IWildCamVal",
            num_classes=num_classes,
            classnames=train_data.classnames,
        )
    )


if __name__ == "__main__":
    main()
