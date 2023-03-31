import time
from pathlib import Path

import typer

from bionic.train import Trainer
from bionic.utils.common import create_time_taken_string

app = typer.Typer()


@app.command("bionic")
def main(config_path: Path):
    """Integrates networks using BIONIC.

    All relevant parameters for the model should be specified in a `.json` config file.

    See https://github.com/bowang-lab/BIONIC/blob/master/README.md for details on writing
    the config file, as well as usage tips.
    """
    time_start = time.time()
    trainer = Trainer(config_path)
    trainer.train()
    trainer.forward()
    time_end = time.time()
    typer.echo(create_time_taken_string(time_start, time_end))


if __name__ == "__main__":
    app()
