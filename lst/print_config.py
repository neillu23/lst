# Copyright (c) Meta Platforms, Inc. and affiliates.
from lst.args import TrainArgs
from lst.config_parser import parse_args_to_pydantic_model


def main():
    train_args = parse_args_to_pydantic_model(TrainArgs)
    print(train_args.model_dump_json(indent=4))


if __name__ == "__main__":
    main()
